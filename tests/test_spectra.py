import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from models.spectra import softmax, spectra_adapt


class SpectraTests(unittest.TestCase):
    def test_softmax_is_stable_for_large_logits(self):
        weights = softmax(np.array([1000.0, 1001.0, 999.0]))
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[1], weights[0])

    def test_adaptation_returns_cma_result_even_when_initial_fit_is_better(self):
        cma_result = np.array([-1.0, 1.0])
        calls = []
        applied = []

        class Strategy:
            result = SimpleNamespace(xbest=cma_result, fbest=4.0)

            def stop(self):
                return True

        class CMA:
            @staticmethod
            def CMAEvolutionStrategy(initial, sigma, options):
                calls.append((np.asarray(initial).copy(), sigma, options))
                return Strategy()

        class Model:
            def nll_answer(self, pairs):
                return 1.0

        config = SimpleNamespace(
            seed=42,
            router_prior_weight=0.1,
            cma_sigma0=0.3,
            cma_pop=8,
            cma_iters=40,
        )
        module = types.ModuleType("cma")
        module.CMAEvolutionStrategy = CMA.CMAEvolutionStrategy
        with patch.dict("sys.modules", {"cma": module}):
            coefficients, fitness, history, duration = spectra_adapt(
                [("prompt", "response")],
                Model(),
                ["math", "code"],
                applied.append,
                config,
                verbose=False,
            )

        np.testing.assert_allclose(coefficients, softmax(cma_result))
        self.assertEqual(fitness, 4.0)
        self.assertEqual(history, [])
        np.testing.assert_allclose(applied[-1], coefficients)
        np.testing.assert_array_equal(calls[0][0], [2.0, 0.0])
        self.assertEqual(calls[0][1], 0.3)
        self.assertEqual(
            calls[0][2],
            {"popsize": 8, "maxiter": 40, "seed": 42, "verbose": -9, "tolx": 1e-6, "tolfun": 1e-6},
        )

    def test_softmax_is_invariant_to_constant_shift(self):
        values = np.array([-2.0, 0.0, 1.0])
        np.testing.assert_allclose(softmax(values), softmax(values + 50.0))
