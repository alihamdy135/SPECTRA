import tempfile
import unittest
from pathlib import Path

from config import load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_match_notebook_runtime_and_experiment_settings(self):
        from config import ExperimentConfig

        config = ExperimentConfig()
        self.assertEqual(config.day12_dir, "/kaggle/input/spectra-day1-2-artifacts-v1")
        self.assertEqual(config.dataset_path, "/kaggle/input/spectra-day1-2-artifacts-v1/data/unified_dataset")
        self.assertEqual(config.artifact_root, config.day12_dir)
        self.assertEqual(config.model_path, "")
        self.assertEqual(config.work_dir, "/kaggle/working")
        self.assertEqual(config.output_path, Path("/kaggle/working"))
        self.assertEqual(config.lora_names, ("math", "code", "hybrid"))
        self.assertEqual(config.evaluation_seeds, (42, 123, 256, 512, 1024))
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.lora_r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertEqual(config.lora_targets, ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.assertEqual(config.train_epochs, 2)
        self.assertEqual(config.train_batch, 2)
        self.assertEqual(config.grad_accum, 4)
        self.assertEqual(config.learning_rate, 0.0002)
        self.assertEqual(config.max_seq_len, 256)
        self.assertEqual(config.gen_max_new, 128)
        self.assertEqual(config.cma_pop, 8)
        self.assertEqual(config.cma_iters, 40)
        self.assertEqual(config.cma_sigma0, 0.3)
        self.assertEqual(config.n_adapt, 15)
        self.assertEqual(config.n_test, 50)
        self.assertEqual(config.n_bootstrap, 1000)
        self.assertEqual(config.domain_map, {"math": "math", "code": "coding", "hybrid": "hybrid", "coding": "coding"})
        custom_root = ExperimentConfig(day12_dir="/tmp/custom-day12")
        self.assertEqual(custom_root.dataset_path, "/tmp/custom-day12/data/unified_dataset")
        self.assertEqual(custom_root.artifact_root, custom_root.day12_dir)

        loaded = load_config(Path(__file__).parents[1] / "configs" / "experiment.yaml")
        self.assertEqual(loaded.day12_dir, config.day12_dir)
        self.assertEqual(loaded.model_path, "")
        self.assertEqual(loaded.work_dir, "/kaggle/working")
        self.assertFalse(loaded.trust_remote_code)

    def test_load_config_uses_explicit_values_and_tuple_fields(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "experiment.yaml"
            path.write_text(
                "seed: 7\nn_adapt: 11\nlora_names: [math, code, hybrid]\nevaluation_seeds: [7, 13]\ntrust_remote_code: true\n",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.n_adapt, 11)
        self.assertEqual(config.lora_names, ("math", "code", "hybrid"))
        self.assertEqual(config.evaluation_seeds, (7, 13))
        self.assertTrue(config.trust_remote_code)

    def test_rejects_unsafe_path_components_and_defaults_remote_code_to_false(self):
        from config import ExperimentConfig

        self.assertFalse(ExperimentConfig().trust_remote_code)
        for values in (
            {"run_id": "../outside"},
            {"run_id": "/tmp/outside"},
            {"domain_to_split": (("math", "../outside"),)},
            {"lora_names": ("../outside",)},
            {"run_id": None},
            {"lora_names": (None,)},
            {"domain_to_split": (("math", None),)},
            {"evaluation_seeds": ()},
            {"grid_points": 1},
            {"grid_evaluations": 0},
            {"n_bootstrap": 0},
            {"n_bootstrap": -1},
            {"train_epochs": 0},
            {"cma_pop": 1},
            {"cma_iters": 0},
            {"n_bootstrap": 1.5},
            {"n_test": 2.5},
            {"n_adapt": 3.25},
            {"evaluation_seeds": (42, 10.5)},
            {"exp3_adapt_counts": (5, 12.5)},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ExperimentConfig(**values)

    def test_load_config_rejects_unknown_settings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "experiment.yaml"
            path.write_text("unknown_setting: true\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
