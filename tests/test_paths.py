import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import models.lora as lora
from models.lora import adapter_output_path, existing_adapter_path


class ArtifactPathTests(unittest.TestCase):
    def test_adapter_path_uses_work_dir_and_rejects_unsafe_domain_names(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = SimpleNamespace(work_dir=Path(temporary_dir) / "work")
            expected = config.work_dir / "lora_math"
            self.assertEqual(adapter_output_path(config, "math"), expected)
            for domain in ("../outside", "/tmp/outside", "code/hybrid"):
                with self.subTest(domain=domain):
                    with self.assertRaises(ValueError):
                        adapter_output_path(config, domain)

    def test_eval_lookup_uses_notebook_order_and_accepts_config_only(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            explicit = root / "explicit"
            config = SimpleNamespace(
                day12_dir=root / "day12",
                artifact_root=root / "unused-artifacts",
                output_path=root / "unused-output",
                work_dir=root / "work",
            )
            candidates = (
                config.day12_dir / "lora_math",
                explicit / "lora_math",
                config.work_dir / "lora_math",
            )
            for candidate in candidates:
                candidate.mkdir(parents=True)
                (candidate / "adapter_config.json").write_text("{}", encoding="utf-8")
            with patch.object(lora, "_NOTEBOOK_ARTIFACT_ROOT", explicit):
                self.assertEqual(existing_adapter_path(config, "math"), candidates[0])
                (candidates[0] / "adapter_config.json").unlink()
                self.assertEqual(existing_adapter_path(config, "math"), candidates[1])
                (candidates[1] / "adapter_config.json").unlink()
                self.assertEqual(existing_adapter_path(config, "math"), candidates[2])

    def test_missing_adapter_returns_none(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = SimpleNamespace(day12_dir=root / "day12", work_dir=root / "work")
            with patch.object(lora, "_NOTEBOOK_ARTIFACT_ROOT", root / "explicit"):
                self.assertIsNone(existing_adapter_path(config, "math"))
