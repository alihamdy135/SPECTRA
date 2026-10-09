import logging
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate import evaluate_method, simplex_grid
from results import ResultStore


class FakeModel:
    def generate(self, prompts):
        return list(prompts)


class FailingModel:
    def generate(self, prompts):
        raise RuntimeError("injected failure")


class EvaluationTests(unittest.TestCase):
    def test_simplex_grid_matches_any_adapter_count(self):
        two_adapter = simplex_grid(adapter_count=2, point_count=3)
        four_adapter = simplex_grid(adapter_count=4, point_count=3)
        self.assertEqual(two_adapter, [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        self.assertEqual(len(four_adapter), 10)
        self.assertTrue(all(len(point) == 4 for point in four_adapter))
        self.assertTrue(all(abs(sum(point) - 1.0) < 1e-12 for point in four_adapter))
        self.assertEqual(simplex_grid(adapter_count=1, point_count=3), [[1.0]])

    def test_evaluate_method_rejects_empty_seed_lists(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            runtime = SimpleNamespace(
                config=SimpleNamespace(evaluation_seeds=(), n_bootstrap=50, seed=11),
                logger=logging.getLogger("test-empty-seeds"),
                model=FakeModel(),
                result_store=ResultStore(Path(temporary_dir)),
            )
            with self.assertRaises(ValueError):
                evaluate_method(
                    runtime,
                    name="test_method",
                    domain="math",
                    condition="test",
                    setup=lambda: None,
                    examples=[("prompt", "reference")],
                )
            self.assertEqual(runtime.result_store.rows, [])

    def test_evaluate_method_uses_notebook_bootstrap_defaults_and_row_columns(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            runtime = SimpleNamespace(
                config=SimpleNamespace(evaluation_seeds=(11, 19), n_bootstrap=50, seed=11),
                logger=logging.getLogger("test-evaluation"),
                model=FakeModel(),
                result_store=ResultStore(Path(temporary_dir)),
            )
            setup_calls = []
            with patch("evaluate.bootstrap_ci", return_value=(1.0, 0.8, 1.0)) as bootstrap:
                output = io.StringIO()
                with redirect_stdout(output):
                    row = evaluate_method(
                        runtime,
                        name="test_method",
                        domain="math",
                        condition="test",
                        setup=lambda: setup_calls.append(True),
                        examples=[("answer one", "answer one"), ("answer two", "answer two")],
                    )
        bootstrap.assert_called_once()
        self.assertEqual(len(bootstrap.call_args.args[0]), 4)
        self.assertTrue(all(abs(value - 1.0) < 1e-8 for value in bootstrap.call_args.args[0]))
        self.assertFalse(bootstrap.call_args.kwargs)
        self.assertEqual(row["n_seeds"], 2)
        self.assertNotIn("n_adapt", row)
        self.assertIn("comp_ci_lo", row)
        self.assertIn("comp_ci_hi", row)
        self.assertNotIn("extra", row)
        self.assertEqual(row["em"], 1.0)
        self.assertEqual(len(setup_calls), 1)
        self.assertEqual(len(runtime.result_store.scores["test/math"]["test_method"]), 4)
        self.assertIn("test_method/math", output.getvalue())

    def test_adaptation_metadata_is_persisted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            runtime = SimpleNamespace(
                config=SimpleNamespace(evaluation_seeds=(11,), n_bootstrap=50, seed=11),
                logger=logging.getLogger("test-adaptation-metadata"),
                model=FakeModel(),
                result_store=ResultStore(Path(temporary_dir)),
            )
            row = evaluate_method(
                runtime,
                name="spectra",
                domain="math",
                condition="test",
                setup=lambda: None,
                examples=[("prompt", "prompt")],
                adapt_time=2.5,
                extra={"n_prime": 20},
            )
            self.assertEqual(row["adapt_time_s"], 2.5)
            self.assertEqual(row["n_prime"], 20)

    def test_cma_adaptation_timer_starts_after_strategy_initialization(self):
        import sys
        from types import ModuleType

        from evaluate import adapt_spectra
        events = []

        class FakeStrategy:
            def __init__(self, initial, sigma, options):
                events.append("strategy_init")
                self.result = SimpleNamespace(xbest=[0.0, 0.0], fbest=0.25)
                self.generations = 0

            def stop(self):
                return self.generations > 0

            def ask(self):
                return [[0.0, 0.0]]

            def tell(self, solutions, losses):
                self.generations += 1

        fake_cma = ModuleType("cma")
        fake_cma.CMAEvolutionStrategy = FakeStrategy
        config = SimpleNamespace(seed=42, router_prior_weight=0.1, cma_sigma0=0.3, cma_pop=8, cma_iters=40)
        model = SimpleNamespace(nll_answer=lambda pairs: 0.25)
        with patch.dict(sys.modules, {"cma": fake_cma}), patch("evaluate.time.time", side_effect=[10.0, 13.5]) as clock:
            coefficients, fitness, history, duration = adapt_spectra(
                [("p", "a")], model, ["math", "code"], lambda values: None, config,
                router_weights=None, verbose=False,
            )
        self.assertEqual(events, ["strategy_init"])
        self.assertEqual(clock.call_count, 2)
        self.assertEqual(duration, 3.5)
        self.assertEqual(fitness, 0.25)
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(float(coefficients.sum()), 1.0)

    def test_generation_failure_is_reported_and_persisted_as_empty_predictions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = ResultStore(Path(temporary_dir))
            runtime = SimpleNamespace(
                config=SimpleNamespace(evaluation_seeds=(11,), n_bootstrap=50, seed=11),
                logger=logging.getLogger("test-evaluation-failure"),
                model=FailingModel(),
                result_store=store,
            )
            with patch("evaluate.bootstrap_ci", return_value=(0.0, 0.0, 0.0)):
                evaluate_method(
                    runtime,
                    name="failed_method",
                    domain="math",
                    condition="test",
                    setup=lambda: None,
                    examples=[("prompt", "reference")],
                )
            self.assertEqual(len(store.rows), 1)
            self.assertEqual(store.rows[0]["em"], 0.0)
            self.assertEqual(store.scores["test/math"]["failed_method"], [0.0])

    def test_summary_rankings_follow_notebook_method_order_for_ties(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = ResultStore(Path(temporary_dir))
            for method in ("spectra", "scalar_fusion", "ties_merge", "equal_merge", "no_adaptation"):
                store.append(
                    {"method": method, "domain": "math", "condition": "exp1_main", "em": 1.0,
                     "f1": 1.0, "rouge_l": 1.0, "composite": 1.0},
                )
            runtime = SimpleNamespace(
                config=SimpleNamespace(output_path=Path(temporary_dir)),
                lora_names=["math"],
                result_store=store,
                logger=logging.getLogger("test-ranking-order"),
            )
            from evaluate import summarize_main_comparison
            output = io.StringIO()
            with redirect_stdout(output):
                summarize_main_comparison(runtime)
        ranking_lines = [line for line in output.getvalue().splitlines() if line.lstrip().startswith(("1.", "2."))]
        first_methods = [line.split(".", 1)[1].strip().split()[0] for line in ranking_lines[:2]]
        self.assertEqual(first_methods, ["no_adaptation", "equal_merge"])
