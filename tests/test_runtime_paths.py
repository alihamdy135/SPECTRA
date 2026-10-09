import tempfile
import os
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from runtime import _split_manifest_path, build_training_runtime, load_coefficient_artifact
from config import ExperimentConfig
from utils import Timer, configure_run


class SplitManifestPathTests(unittest.TestCase):
    def test_manifest_path_is_contained_and_rejects_unsafe_names(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "splits"
            self.assertEqual(_split_manifest_path(root, "math"), root / "math.json")
            for name in ("../outside", "/tmp/outside", "math/coding"):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        _split_manifest_path(root, name)

    def test_manifest_path_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "splits"
            outside = Path(temporary_dir) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "math.json").symlink_to(outside / "manifest.json")
            with self.assertRaises(ValueError):
                _split_manifest_path(root, "math")


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_timer_prints_notebook_format(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with patch("utils.time.time", side_effect=(2.0, 3.5)), redirect_stdout(output):
            with Timer("Experiment 1"):
                pass
        self.assertEqual(output.getvalue(), "[TIME] Experiment 1: 1.50s\n")

    def test_runtime_creates_notebook_directories_without_extra_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = ExperimentConfig(work_dir=temporary_dir)
            configure_run(config, "train")
            expected = {"figures", "artifacts", "lora_math", "lora_code", "lora_hybrid", "hf_cache", "hf_home"}
            self.assertEqual({path.name for path in Path(temporary_dir).iterdir()}, expected)
            self.assertFalse((Path(temporary_dir) / "config.json").exists())
            self.assertFalse((Path(temporary_dir) / "logs").exists())
            self.assertFalse((Path(temporary_dir) / "splits").exists())

    def test_runtime_configures_tf32_datasets_caching_and_notebook_cache_paths(self):
        matmul = SimpleNamespace(allow_tf32=False)
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)))
        disable_caching = Mock()
        datasets = SimpleNamespace(disable_caching=disable_caching)

        with tempfile.TemporaryDirectory() as temporary_dir:
            config = ExperimentConfig(work_dir=temporary_dir, run_id="runtime-test")
            with patch.dict(sys.modules, {"torch": torch, "datasets": datasets}):
                with patch.dict(os.environ, {}):
                    logger = configure_run(config, "train")
                    self.assertTrue(matmul.allow_tf32)
                    disable_caching.assert_called_once_with()
                    self.assertEqual(os.environ["HF_DATASETS_CACHE"], str(Path(temporary_dir) / "hf_cache"))
                    self.assertEqual(os.environ["HF_HOME"], str(Path(temporary_dir) / "hf_home"))
                    self.assertTrue(Path(os.environ["HF_DATASETS_CACHE"]).is_dir())
                    self.assertTrue(Path(os.environ["HF_HOME"]).is_dir())
                    for handler in logger.handlers:
                        handler.close()
                    logger.handlers.clear()


class CoefficientArtifactTests(unittest.TestCase):
    def test_loads_only_finite_numeric_vectors_of_expected_shape(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "coeffs.npy"
            np.save(path, np.array([0.25, 0.75]), allow_pickle=False)
            np.testing.assert_allclose(load_coefficient_artifact(path, 2), [0.25, 0.75])

    def test_rejects_object_arrays_without_pickle_loading(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "coeffs.npy"
            np.save(path, np.array(["unsafe"], dtype=object), allow_pickle=True)
            with self.assertRaises(ValueError):
                load_coefficient_artifact(path, 1)

    def test_rejects_invalid_shape_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "coeffs.npy"
            np.save(path, np.array([0.1]), allow_pickle=False)
            with self.assertRaises(ValueError):
                load_coefficient_artifact(path, 2)
            np.save(path, np.array([0.5, np.nan]), allow_pickle=False)
            with self.assertRaises(ValueError):
                load_coefficient_artifact(path, 2)
