import json
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from models.merging import ModelMerger, apply_ties, compute_ties, load_adapter_deltas


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def float(self):
        return Tensor(self.value.astype(float))

    @property
    def data(self):
        return self

    def to(self, dtype):
        return self

    def copy_(self, other):
        self.value = other.value.copy()

    def abs(self):
        return Tensor(np.abs(self.value))

    def flatten(self):
        return Tensor(self.value.flatten())

    def numel(self):
        return self.value.size

    def __float__(self):
        return float(self.value)

    def kthvalue(self, k):
        return types.SimpleNamespace(values=Tensor(np.partition(self.value, k - 1)[k - 1]))

    def reshape_as(self, other):
        return Tensor(self.value.reshape(other.value.shape))

    def sign(self):
        return Tensor(np.sign(self.value))

    def __ge__(self, other):
        return Tensor(self.value >= (other.value if isinstance(other, Tensor) else other))

    def __eq__(self, other):
        return Tensor(self.value == (other.value if isinstance(other, Tensor) else other))

    def __mul__(self, other):
        return Tensor(self.value * (other.value if isinstance(other, Tensor) else other))

    def __add__(self, other):
        return Tensor(self.value + (other.value if isinstance(other, Tensor) else other))

    def __setitem__(self, item, value):
        self.value[item] = value

    def sum(self, axis=None):
        return Tensor(self.value.sum(axis=axis))

    def mean(self, axis=None):
        return Tensor(self.value.mean(axis=axis))


class Torch:
    bool = bool

    @staticmethod
    def argsort(tensor, descending=False, stable=False):
        indices = np.argsort(tensor.value, kind="stable" if stable else "quicksort")
        return indices[::-1] if descending else indices

    @staticmethod
    def zeros_like(tensor, dtype=None):
        return Tensor(np.zeros_like(tensor.value, dtype=dtype))

    @staticmethod
    def where(condition, left, right):
        return Tensor(np.where(condition.value, left.value, right.value))

    @staticmethod
    def stack(tensors):
        return Tensor(np.stack([tensor.value for tensor in tensors]))


class AdapterTensor:
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def __rmul__(self, factor):
        return self

    def __matmul__(self, other):
        return self.value, other.value


class MergingParityTests(unittest.TestCase):
    def test_safetensors_has_priority_and_unpaired_matrices_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8")
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            binary = root / "adapter.bin"
            for weight_file in (first, second, binary):
                weight_file.write_bytes(b"serialized")
            package = types.ModuleType("safetensors")
            package.__path__ = []
            backend = types.ModuleType("safetensors.torch")
            calls = []
            backend.load_file = lambda path, device: calls.append(path) or {
                "base_model.model.layer.lora_A.weight": AdapterTensor("A")
            }
            fake_torch = types.ModuleType("torch")
            fake_torch.load = lambda *args, **kwargs: self.fail("binary weights must not override safetensors")
            with patch.dict(
                "sys.modules",
                {"safetensors": package, "safetensors.torch": backend, "torch": fake_torch},
            ), patch("models.merging.os.listdir", return_value=["adapter.bin", "second.safetensors", "first.safetensors"]):
                deltas = load_adapter_deltas(root, fallback_rank=1, fallback_alpha=1)

            self.assertEqual(calls, [str(second)])
            self.assertEqual(deltas, {})

    def test_weight_helpers_apply_notebook_coefficients(self):
        class Model:
            def __init__(self):
                self.calls = []

            def set_coefficients(self, coefficients, deltas, names):
                self.calls.append((coefficients, deltas, names))

        model = Model()
        merger = object.__new__(ModelMerger)
        merger.model = model
        merger.all_deltas = {"math": {}, "code": {}}
        merger.lora_names = ["math", "code"]

        merger.reset_base()
        merger.apply_single("code")
        merger.apply_equal()

        self.assertEqual(
            [call[0] for call in model.calls],
            [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        )
        self.assertTrue(all(call[1] is merger.all_deltas for call in model.calls))
        self.assertTrue(all(call[2] is merger.lora_names for call in model.calls))

    def test_apply_ties_adds_the_cached_delta_to_base_weights(self):
        class Model:
            def __init__(self):
                self.parameter = Tensor([0.0, 0.0])
                self.model = self
                self.current_key = (0.0,)
                self.config = types.SimpleNamespace(dtype="float32")
                self.base_weights = {"layer.weight": Tensor([10.0, 10.0])}

            def named_parameters(self):
                return [("layer.weight", self.parameter)]

        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.no_grad = nullcontext
        model = Model()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            apply_ties(model, {"layer.weight": Tensor([1.0, -2.0])})

        np.testing.assert_array_equal(model.parameter.value, [11.0, 8.0])
        self.assertEqual(model.current_key, ("ties",))

    def test_ties_uses_kth_threshold_and_retains_every_equal_magnitude(self):
        values = Tensor([1.0, 2.0, -2.0, 4.0])
        fake_torch = types.ModuleType("torch")
        for name in ("argsort", "zeros_like", "where", "stack"):
            setattr(fake_torch, name, getattr(Torch, name))
        fake_torch.bool = bool
        with patch.dict("sys.modules", {"torch": fake_torch}):
            merged = compute_ties({"math": {"layer.weight": values}}, ["math"], density=0.5)
        np.testing.assert_array_equal(merged["layer.weight"].value, [0.0, 2.0, -2.0, 4.0])

    def test_empty_safetensors_state_falls_back_to_first_bin(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8")
            safe_file = root / "empty.safetensors"
            bin_file = root / "fallback.bin"
            safe_file.write_bytes(b"safe")
            bin_file.write_bytes(b"serialized")
            package = types.ModuleType("safetensors")
            package.__path__ = []
            backend = types.ModuleType("safetensors.torch")
            safe_calls = []
            backend.load_file = lambda path, device: safe_calls.append(path) or {}
            binary_state = {
                "base_model.model.layer.lora_A.weight": AdapterTensor("A"),
                "base_model.model.layer.lora_B.weight": AdapterTensor("B"),
            }
            bin_calls = []
            fake_torch = types.ModuleType("torch")
            fake_torch.load = lambda path, **options: bin_calls.append((path, options)) or binary_state
            with patch.dict(
                "sys.modules",
                {"safetensors": package, "safetensors.torch": backend, "torch": fake_torch},
            ), patch("models.merging.os.listdir", return_value=["empty.safetensors", "fallback.bin"]):
                deltas = load_adapter_deltas(root, fallback_rank=1, fallback_alpha=1)

            self.assertEqual(safe_calls, [str(safe_file)])
            self.assertEqual(bin_calls, [(str(bin_file), {"map_location": "cpu", "weights_only": True})])
            self.assertEqual(deltas, {"layer.weight": ("B", "A")})

    def test_bin_adapter_uses_weights_only_loader(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8")
            weight_file = root / "adapter_model.bin"
            weight_file.write_bytes(b"serialized")
            state = {
                "base_model.model.layer.lora_A.weight": AdapterTensor("A"),
                "base_model.model.layer.lora_B.weight": AdapterTensor("B"),
            }
            calls = []
            fake_torch = types.ModuleType("torch")

            def load(path, **options):
                calls.append((path, options))
                return state

            fake_torch.load = load
            with patch.dict("sys.modules", {"torch": fake_torch}):
                deltas = load_adapter_deltas(root, fallback_rank=1, fallback_alpha=1)

            self.assertEqual(deltas, {"layer.weight": ("B", "A")})
            self.assertEqual(calls, [(str(weight_file), {"map_location": "cpu", "weights_only": True})])
