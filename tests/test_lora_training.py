import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import models.lora as lora
from models.lora import accumulation_window, adapter_output_path, format_example, generation_batch_size_for, train_domain_adapter, warmup_steps_for


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def clone(self):
        return FakeTensor(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.values[key]
        return self.values[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self.values))
            for index in range(start, stop, step):
                self.values[index] = value
        else:
            self.values[key] = value


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, **kwargs):
        if text.startswith("### Q:") and "response" not in text:
            return {"input_ids": [1, 2, 3, 4]}
        if "return_tensors" in kwargs:
            return {"input_ids": [FakeTensor([1, 2, 3, 4])], "attention_mask": [FakeTensor([1, 1, 1, 1])]}
        return {"input_ids": [1, 2, 3, 4]}


class FakeValue:
    def to(self, device):
        return self


class FakeLoss:
    def __init__(self, value, backward_values):
        self.value = value
        self.backward_values = backward_values

    def __truediv__(self, divisor):
        return FakeLoss(self.value / divisor, self.backward_values)

    def backward(self):
        self.backward_values.append(self.value)

    def item(self):
        return self.value


class FakeOptimizer:
    def __init__(self, records):
        self.records = records

    def zero_grad(self):
        self.records["zero_grad"] += 1

    def step(self):
        self.records["optimizer_steps"] += 1


class FakeScheduler:
    def __init__(self, records):
        self.records = records

    def step(self):
        self.records["scheduler_steps"] += 1


class FakeTrainableParameter:
    requires_grad = True


class FakeFrozenParameter:
    def __init__(self):
        self.requires_grad = True


class FakeTrainingModel:
    def __init__(self, records):
        self.records = records
        self.parameter = FakeTrainableParameter()

    def parameters(self):
        return [self.parameter]

    def train(self):
        self.records["train_called"] = True

    def __call__(self, **batch):
        return SimpleNamespace(loss=FakeLoss(8.0, self.records["backward_values"]))

    def save_pretrained(self, path, **kwargs):
        self.records["saved_adapter"] = Path(path)


class FakeBaseModel:
    def __init__(self):
        self.parameter = FakeFrozenParameter()
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def parameters(self):
        return [self.parameter]


class FakeModelWrapper:
    def __init__(self):
        self.model = object()
        self.tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
        self.device = "cpu"
        self.current_key = ("adapter",)

    def replace_model(self, replacement):
        self.model = replacement
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.current_key = None


class LoraTrainingTests(unittest.TestCase):
    def _fake_dependencies(self, batch_count, records):
        torch = types.ModuleType("torch")
        torch.__path__ = []
        torch.float32 = object()
        torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        torch.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda parameters, limit: None))

        class DataLoader:
            def __init__(self, *args, **kwargs):
                self.batches = [{"input_ids": FakeValue()} for _ in range(batch_count)]

            def __len__(self):
                return len(self.batches)

            def __iter__(self):
                return iter(self.batches)

        data_module = types.ModuleType("torch.utils.data")
        data_module.DataLoader = DataLoader
        utils_module = types.ModuleType("torch.utils")
        utils_module.__path__ = []
        utils_module.data = data_module
        torch.utils = utils_module

        def make_optimizer(parameters, **kwargs):
            return FakeOptimizer(records)

        torch.optim = SimpleNamespace(AdamW=make_optimizer)

        peft = types.ModuleType("peft")
        peft.LoraConfig = lambda **kwargs: kwargs
        peft.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
        training_model = FakeTrainingModel(records)
        peft.get_peft_model = lambda model, config: training_model

        transformers = types.ModuleType("transformers")
        replacement = FakeBaseModel()

        def reload_base_model(path, **options):
            records["reload_path"] = path
            records["reload_options"] = options
            return replacement

        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=reload_base_model)

        def make_scheduler(optimizer, warmup_steps, total_steps):
            records["warmup_steps"] = warmup_steps
            records["total_steps"] = total_steps
            return FakeScheduler(records)

        transformers.get_cosine_schedule_with_warmup = make_scheduler
        return {
            "torch": torch,
            "torch.utils": utils_module,
            "torch.utils.data": data_module,
            "peft": peft,
            "transformers": transformers,
        }, training_model, replacement

    def _config(self, root, **overrides):
        values = {
            "output_path": root / "run",
            "work_dir": root / "work",
            "artifact_root": root / "artifacts",
            "day12_dir": root / "day12",
            "seed": 42,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_targets": ("q_proj",),
            "max_seq_len": 32,
            "train_batch": 1,
            "learning_rate": 0.0002,
            "grad_accum": 4,
            "train_epochs": 2,
            "model_path": str(root / "base-model"),
            "dtype": "float32",
            "trust_remote_code": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _train_with_fakes(self, config, root, batch_count=5, records=None):
        records = records if records is not None else {
            "backward_values": [],
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "zero_grad": 0,
        }
        modules, training_model, replacement = self._fake_dependencies(batch_count, records)
        model = FakeModelWrapper()
        with patch.dict("sys.modules", modules), patch.object(lora, "set_training_seed", lambda seed: None):
            result = train_domain_adapter(model, [], "math", config, SimpleNamespace(info=lambda *args: None))
        return result, records, model, training_model, replacement

    def test_incomplete_accumulation_window_is_dropped(self):
        windows = [accumulation_window(step, total_batches=5, grad_accum=4) for step in range(5)]
        self.assertEqual([window[1] for window in windows], [4, 4, 4, 4, 4])
        self.assertEqual([window[2] for window in windows], [False, False, False, True, False])

    def test_one_step_training_uses_notebook_step_count_loss_scale_and_model_reset(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._config(root)
            result, records, model, training_model, replacement = self._train_with_fakes(config, root)
            self.assertEqual(result, adapter_output_path(config, "math"))
            self.assertEqual(records["total_steps"], 2)
            self.assertEqual(records["warmup_steps"], 1)
            self.assertEqual(records["optimizer_steps"], 2)
            self.assertEqual(records["scheduler_steps"], 2)
            self.assertEqual(records["backward_values"], [2.0] * 10)
            self.assertIs(model.model, replacement)
            self.assertTrue(replacement.eval_called)
            self.assertFalse(replacement.parameter.requires_grad)
            self.assertIsNone(model.current_key)
            self.assertIs(training_model.parameter.requires_grad, True)

    def test_base_model_reload_selects_safe_loader_for_bin_only_weights(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._config(root)
            model_path = Path(config.model_path)
            model_path.mkdir(parents=True)
            (model_path / "pytorch_model.bin").write_bytes(b"weights")
            _, records, _, _, _ = self._train_with_fakes(config, root)
            self.assertEqual(records["reload_path"], str(model_path))
            self.assertFalse(records["reload_options"]["use_safetensors"])
            self.assertTrue(records["reload_options"]["weights_only"])

    def test_base_model_reload_prefers_safetensors_when_present(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._config(root)
            model_path = Path(config.model_path)
            model_path.mkdir(parents=True)
            (model_path / "model.safetensors").write_bytes(b"weights")
            _, records, _, _, _ = self._train_with_fakes(config, root)
            self.assertTrue(records["reload_options"]["use_safetensors"])
            self.assertNotIn("weights_only", records["reload_options"])

    def test_training_reuses_notebook_first_candidate_even_without_weight_validation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._config(root)
            explicit_root = root / "explicit-artifacts"
            candidates = (
                config.day12_dir / "lora_math",
                explicit_root / "lora_math",
                adapter_output_path(config, "math"),
            )
            with patch.object(lora, "_NOTEBOOK_ARTIFACT_ROOT", explicit_root):
                for candidate in candidates:
                    candidate.mkdir(parents=True)
                    (candidate / "adapter_config.json").write_text("{}", encoding="utf-8")
                result, _, _, _, _ = self._train_with_fakes(config, root)
                self.assertEqual(result, candidates[0])
                (candidates[0] / "adapter_config.json").unlink()
                result, _, _, _, _ = self._train_with_fakes(config, root)
                self.assertEqual(result, candidates[1])
                (candidates[1] / "adapter_config.json").unlink()
                result, _, _, _, _ = self._train_with_fakes(config, root)
                self.assertEqual(result, candidates[2])

    def test_training_with_less_than_one_full_window_keeps_one_scheduler_step(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._config(root)
            _, records, _, _, _ = self._train_with_fakes(config, root, batch_count=3)
            self.assertEqual(records["total_steps"], 1)
            self.assertEqual(records["warmup_steps"], 1)
            self.assertEqual(records["optimizer_steps"], 0)
            self.assertEqual(records["scheduler_steps"], 0)
            self.assertEqual(records["backward_values"], [2.0] * 6)

    def test_existing_save_directory_without_adapter_config_does_not_block_training(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._config(root)
            saved_path = adapter_output_path(config, "math")
            saved_path.mkdir(parents=True)
            result, records, _, _, _ = self._train_with_fakes(config, root)
            self.assertEqual(result, saved_path)
            self.assertEqual(records["optimizer_steps"], 2)

    def test_format_example_preserves_fully_masked_answer_after_truncation(self):
        with patch.dict("sys.modules", {"torch": types.ModuleType("torch")}):
            example = format_example("question", "response", FakeTokenizer(), max_length=4)
        self.assertEqual(example["labels"].values, [-100, -100, -100, -100])

    def test_generation_batch_size_must_be_positive(self):
        self.assertEqual(generation_batch_size_for(None, 4), 4)
        self.assertEqual(generation_batch_size_for(2, 4), 2)
        for value in (0, -1):
            with self.assertRaises(ValueError):
                generation_batch_size_for(value, 4)

    def test_warmup_matches_notebook_minimum(self):
        self.assertEqual(warmup_steps_for(1), 1)
        self.assertEqual(warmup_steps_for(9), 1)
        self.assertEqual(warmup_steps_for(10), 1)
        self.assertEqual(warmup_steps_for(20), 2)


if __name__ == "__main__":
    unittest.main()
