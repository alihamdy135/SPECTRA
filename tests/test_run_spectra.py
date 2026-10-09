import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_spectra


class FullRunTests(unittest.TestCase):
    def test_full_run_reuses_model_splits_and_runtime_across_experiments(self):
        config = SimpleNamespace(seed=42)
        model = object()
        train_sets = {"math": ["train"]}
        adapt_sets = {"math": ["adapt"]}
        test_sets = {"math": ["test"]}
        runtime = object()
        ablation_result = {"exp6": "summary"}
        with (
            patch.object(run_spectra, "load_config", return_value=config) as load_config,
            patch.object(run_spectra, "configure_run", return_value="logger") as configure_run,
            patch.object(run_spectra, "set_seed") as set_seed,
            patch.object(
                run_spectra,
                "build_training_runtime",
                return_value=(model, train_sets, adapt_sets, test_sets),
            ) as build_training,
            patch.object(run_spectra, "train_adapters") as train_adapters,
            patch.object(run_spectra, "build_evaluation_runtime", return_value=runtime) as build_evaluation,
            patch.object(run_spectra, "run_main_comparison") as run_main,
            patch.object(run_spectra, "run_ablations", return_value=ablation_result) as run_ablation,
        ):
            result = run_spectra.run_spectra("experiment.yaml")
        self.assertEqual(result, ablation_result)
        load_config.assert_called_once_with("experiment.yaml")
        configure_run.assert_called_once_with(config, "run")
        set_seed.assert_called_once_with(42)
        build_training.assert_called_once_with(config, "logger")
        train_adapters.assert_called_once_with(model, train_sets, config, "logger")
        build_evaluation.assert_called_once_with(
            config,
            "logger",
            model=model,
            split_data=(train_sets, adapt_sets, test_sets),
        )
        run_main.assert_called_once_with(runtime)
        run_ablation.assert_called_once_with(runtime=runtime)


if __name__ == "__main__":
    unittest.main()
