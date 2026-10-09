import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import models.lora as lora


class LoraPathParityTests(unittest.TestCase):
    def test_save_path_and_discovery_match_notebook_order_with_config_only(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            day12 = root / "day12"
            explicit = root / "explicit"
            work_dir = root / "work"
            config = SimpleNamespace(
                day12_dir=day12,
                artifact_root=root / "different-artifacts",
                output_path=root / "different-output",
                work_dir=work_dir,
            )
            expected = (
                day12 / "lora_math",
                explicit / "lora_math",
                work_dir / "lora_math",
            )
            self.assertEqual(lora.adapter_output_path(config, "math"), expected[-1])
            self.assertEqual(
                lora._NOTEBOOK_ARTIFACT_ROOT,
                Path("/kaggle/input/datasets/ali320230101/spectra-day1-2-artifacts-v1"),
            )
            for candidate in expected:
                candidate.mkdir(parents=True)
                (candidate / "adapter_config.json").write_text("{}", encoding="utf-8")
            with patch.object(lora, "_NOTEBOOK_ARTIFACT_ROOT", explicit, create=True):
                for finder in (lora.existing_training_adapter_path, lora.existing_adapter_path):
                    self.assertEqual(finder(config, "math"), expected[0])
                    (expected[0] / "adapter_config.json").unlink()
                    self.assertEqual(finder(config, "math"), expected[1])
                    (expected[1] / "adapter_config.json").unlink()
                    self.assertEqual(finder(config, "math"), expected[2])
                    (expected[0] / "adapter_config.json").write_text("{}", encoding="utf-8")
                    (expected[1] / "adapter_config.json").write_text("{}", encoding="utf-8")
