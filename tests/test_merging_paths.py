import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from models.merging import load_adapter_deltas


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def __rmul__(self, factor):
        return self

    def __matmul__(self, other):
        return self.value, other.value


class AdapterDeltaTests(unittest.TestCase):
    def test_loads_adapter_matrices_from_the_first_safetensors_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8")
            weight_file = root / "adapter_model.safetensors"
            weight_file.write_bytes(b"weights")
            state = {
                str(weight_file): {
                    "base_model.model.layer.lora_A.weight": FakeTensor("A"),
                    "base_model.model.layer.lora_B.weight": FakeTensor("B"),
                }
            }
            package = types.ModuleType("safetensors")
            package.__path__ = []
            backend = types.ModuleType("safetensors.torch")
            backend.load_file = lambda path, device: state[path]
            with patch.dict(sys.modules, {"safetensors": package, "safetensors.torch": backend}):
                deltas = load_adapter_deltas(root, fallback_rank=1, fallback_alpha=1)
            self.assertEqual(deltas, {"layer.weight": ("B", "A")})

    def test_missing_adapter_weight_files_return_no_deltas(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8")
            self.assertEqual(load_adapter_deltas(root, fallback_rank=1, fallback_alpha=1), {})
