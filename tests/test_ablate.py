import tempfile
import unittest
import csv
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ablate import (
    _get_or_fit_coefficients,
    effective_sample_counts,
    matching_main_adaptation_time,
    run_cross_domain,
    run_efficiency_analysis,
    run_sample_efficiency,
)
from metrics import compute_metrics
from results import ResultStore


class SampleEfficiencyTests(unittest.TestCase):
    def test_saved_coefficients_without_exp1_result_do_not_replace_notebook_refit(self):
        runtime = SimpleNamespace(
            saved_coefficients={"hybrid": np.array([0.2, 0.3, 0.5])},
            result_store=SimpleNamespace(rows=[]),
            get_adapt_examples=lambda domain, count, seed_offset: [("p", "a")],
            router=SimpleNamespace(fit=lambda prompts: np.array([0.2, 0.3, 0.5])),
            model=object(),
            lora_names=["math", "code", "hybrid"],
            apply_coefficients=lambda values: None,
            config=SimpleNamespace(seed=42, n_adapt=15, router_prior_weight=0.1, cma_sigma0=0.3, cma_pop=8, cma_iters=40),
            save_coefficients=lambda domain, values: None,
        )
        with patch("ablate.adapt_spectra", return_value=(np.array([0.1, 0.2, 0.7]), 1.0, [], 3.0)) as adaptation:
            coefficients, duration = _get_or_fit_coefficients(runtime, "hybrid", seed_offset=5000)
        self.assertEqual(len(adaptation.call_args.args[0]), 1)
        self.assertEqual(duration, 3.0)
        self.assertEqual(coefficients.tolist(), [0.1, 0.2, 0.7])

    def test_counts_preserve_requested_n20_when_available_examples_stop_at_15(self):
        self.assertEqual(
            effective_sample_counts((5, 10, 15, 20), available=15),
            ((5, 5), (10, 10), (15, 15), (20, 15)),
        )

    def test_sample_efficiency_records_requested_n20_as_its_own_condition(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = ResultStore(Path(temporary_dir))
            adapted = []
            calls = []
            runtime = SimpleNamespace(
                config=SimpleNamespace(exp3_adapt_counts=(5, 10, 15, 20)),
                lora_names=["math", "code", "hybrid"],
                result_store=store,
                logger=__import__("logging").getLogger("test-exp3-requested-size"),
                get_test_examples=lambda domain: [("q", "a")],
                get_adapt_examples=lambda domain, count, seed_offset=0: adapted.append((domain, count, seed_offset)) or [(f"p{i}", "a") for i in range(15)],
                router=SimpleNamespace(fit=lambda prompts: [0.2, 0.3, 0.5]),
                model=object(),
                apply_coefficients=lambda values: None,
                reset_base=lambda: None,
                apply_ties=lambda: None,
                apply_equal=lambda: None,
            )

            def capture(runtime, name, domain, condition, setup, examples, adapt_time=None, extra=None, seeds=None):
                calls.append((name, condition, len(examples), adapt_time, extra))
                return {}

            with patch("ablate.evaluate_method", side_effect=capture), patch(
                "ablate.adapt_spectra", return_value=(np.array([0.2, 0.3, 0.5]), 1.0, [], 2.0)
            ) as adaptation:
                counts = run_sample_efficiency(runtime)

        self.assertEqual(counts, ((5, 5), (10, 10), (15, 15), (20, 15)))
        self.assertEqual(adapted, [("hybrid", 20, 7)])
        n20 = [call for call in calls if call[1] == "exp3_n20"]
        self.assertEqual([call[0] for call in n20], ["ties_merge", "equal_merge", "spectra"])
        self.assertTrue(all(call[4]["n_prime"] == 20 for call in n20))
        self.assertEqual(n20[-1][2], 1)
        self.assertEqual([len(call.args[0]) for call in adaptation.call_args_list], [5, 10, 15, 15])

    def test_adaptation_time_matches_domain_condition_and_coefficients(self):
        rows = [
            {"method": "spectra", "domain": "hybrid", "condition": "exp3_n15", "adapt_time_s": 900, "coeffs": [0.2, 0.3, 0.5]},
            {"method": "spectra", "domain": "math", "condition": "exp1_main", "adapt_time_s": 800, "coeffs": [0.2, 0.3, 0.5]},
            {"method": "spectra", "domain": "hybrid", "condition": "exp1_main", "adapt_time_s": 12.5, "coeffs": "[0.2, 0.3, 0.5]"},
        ]
        self.assertEqual(matching_main_adaptation_time(rows, "hybrid", [0.2, 0.3, 0.5]), 12.5)
        self.assertIsNone(matching_main_adaptation_time(rows, "hybrid", [0.5, 0.3, 0.2]))

    def test_invalid_sample_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            effective_sample_counts((5, 0), available=15)

    def test_exp5_cost_sums_all_spectra_results_and_keeps_notebook_columns(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            store = ResultStore(root)
            for domain, duration in (("math", 10), ("code", 20), ("hybrid", 30)):
                store.append({
                    "method": "spectra", "domain": domain, "condition": "exp1_main",
                    "adapt_time_s": duration, "coeffs": [0.2, 0.3, 0.5],
                })
            store.append({"method": "spectra", "domain": "hybrid", "condition": "exp2_ablation", "adapt_time_s": 40})
            runtime = SimpleNamespace(
                config=SimpleNamespace(output_path=root, work_dir=str(root), exp5_test_count=2, n_adapt=15),
                saved_coefficients={"hybrid": np.array([0.2, 0.3, 0.5])},
                result_store=store,
                lora_names=["math", "code", "hybrid"],
                logger=logging.getLogger("test-exp5-cost"),
                model=SimpleNamespace(generate=lambda prompts: list(prompts)),
                get_test_examples=lambda domain, count=None: [("p1", "a1"), ("p2", "a2")],
                reset_base=lambda: None,
                apply_ties=lambda: None,
                apply_coefficients=lambda values: None,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                path = run_efficiency_analysis(runtime)
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                columns = reader.fieldnames
        self.assertEqual(columns, ["method", "eval_per_ex", "peak_vram_mb"])
        self.assertEqual([row["method"] for row in rows], ["no_adaptation", "ties_merge", "spectra"])
        self.assertIn("Total CMA adapt cost: 100.0s", output.getvalue())

    def test_exp6_writes_summary_only_and_uses_default_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            store = ResultStore(root)
            runtime = SimpleNamespace(
                config=SimpleNamespace(seed=42, exp6_test_per_domain=2, exp6_adapt_per_domain=1, output_path=root),
                lora_names=["math", "code", "hybrid"],
                logger=logging.getLogger("test-exp6-summary"),
                result_store=store,
                get_test_examples=lambda domain, count: [(domain + " prompt", domain + " prompt") for _ in range(count)],
                get_adapt_examples=lambda domain, count, seed_offset=0: [(domain + " prompt", "answer") for _ in range(count)],
                model=SimpleNamespace(generate=lambda prompts: list(prompts)),
                router=SimpleNamespace(fit=lambda prompts, verbose=False: np.array([0.2, 0.3, 0.5])),
                apply_coefficients=lambda values: None,
                reset_base=lambda: None,
                apply_equal=lambda: None,
                apply_ties=lambda: None,
            )
            with patch("ablate.adapt_spectra", return_value=(np.array([0.2, 0.3, 0.5]), 0.1, [], 4.5)), patch(
                "ablate.bootstrap_ci", return_value=(0.5, 0.4, 0.6)
            ) as bootstrap, patch("ablate.compute_metrics", wraps=compute_metrics) as metric_calls:
                output = io.StringIO()
                with redirect_stdout(output):
                    path = run_cross_domain(runtime)
            with (root / "spectra_results.csv").open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                columns = reader.fieldnames
        self.assertIsNone(path)
        self.assertFalse((root / "artifacts" / "exp6_significance.csv").exists())
        self.assertEqual(columns, ["method", "domain", "condition", "adapt_time_s", "coeffs", "spectra_comp", "equal_comp", "delta_eq", "timestamp"])
        self.assertEqual([row["method"] for row in rows], ["exp6_summary"])
        self.assertEqual(rows[0]["condition"], "exp6")
        self.assertEqual(float(rows[0]["adapt_time_s"]), 4.5)
        self.assertEqual(store.scores, {})
        self.assertEqual(bootstrap.call_count, 4)
        self.assertTrue(all(not call.kwargs for call in bootstrap.call_args_list))
        self.assertEqual(metric_calls.call_count, 4)
        self.assertTrue(all(not call.kwargs for call in metric_calls.call_args_list))
        self.assertIn("Best cross-domain:", output.getvalue())
