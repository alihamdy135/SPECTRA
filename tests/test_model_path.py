import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from config import ExperimentConfig
from models.base import MergeableCausalLM, _MODEL_CANDIDATES, find_model_path, validate_model_path


class ModelPathTests(unittest.TestCase):
    def make_model_directory(self, path, weight_name="model.safetensors"):
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (path / weight_name).write_bytes(b"weights")
        return path

    def test_validates_local_model_configuration_and_weight_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            self.assertEqual(validate_model_path(root), root)

    def test_accepts_bin_weights_for_weights_only_loading(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "pytorch_model.bin").write_bytes(b"weights")
            self.assertEqual(validate_model_path(root), root)

    def test_candidate_priority_matches_notebook_paths(self):
        self.assertEqual(
            _MODEL_CANDIDATES,
            (
                "/kaggle/input/qwen2.5-1.5b-instruct_reasoner/pytorch/default/1",
                "/kaggle/input/qwen2-5-1-5b-instruct/pytorch/default/1",
                "/kaggle/input/qwen2.5-1.5b-instruct/pytorch/default/1",
            ),
        )

    def test_finds_priority_candidate_before_a_higher_scored_walk_candidate(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            priority = self.make_model_directory(root / "priority")
            stronger = self.make_model_directory(root / "qwen_1.5b_instruct_reasoner")
            candidates = (str(priority), str(root / "missing"), str(root / "missing2"))
            with patch("models.base._MODEL_CANDIDATES", candidates), patch(
                "models.base.os.walk", return_value=[(str(stronger), [], [])]
            ):
                self.assertEqual(find_model_path(), str(priority))

    def test_walk_uses_exact_keyword_score_for_model_choice(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            penalized = self.make_model_directory(root / "qwen_1.5b_instruct_reasoner_lora_adapter")
            preferred = self.make_model_directory(root / "qwen_1.5b_instruct_reasoner")
            candidates = (str(root / "missing1"), str(root / "missing2"), str(root / "missing3"))
            walked = [(str(penalized), [], []), (str(preferred), [], [])]
            with patch("models.base._MODEL_CANDIDATES", candidates), patch(
                "models.base.os.walk", return_value=walked
            ):
                self.assertEqual(find_model_path(), str(preferred))

    def test_empty_model_path_uses_original_discovery_and_safe_bin_loading(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path = self.make_model_directory(Path(temporary_dir) / "model", "pytorch_model.bin")
            load_calls = []

            class Model:
                def eval(self):
                    return self

                def parameters(self):
                    return []

                def named_parameters(self):
                    return []

            class Tokenizer:
                pad_token = None
                eos_token = "eos"
                eos_token_id = 0
                vocab_size = 4

            class ModelLoader:
                @staticmethod
                def from_pretrained(path, **options):
                    load_calls.append((path, options))
                    return Model()

            class TokenizerLoader:
                @staticmethod
                def from_pretrained(path, **options):
                    return Tokenizer()

            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            fake_torch.float16 = object()
            transformers = types.ModuleType("transformers")
            transformers.AutoModelForCausalLM = ModelLoader
            transformers.AutoTokenizer = TokenizerLoader
            config = ExperimentConfig()
            with patch.dict("sys.modules", {"torch": fake_torch, "transformers": transformers}), patch(
                "models.base.find_model_path", return_value=str(model_path)
            ) as discover:
                loaded = MergeableCausalLM.from_pretrained(config)

            discover.assert_called_once_with()
            self.assertEqual(load_calls[0][0], str(model_path))
            self.assertFalse(load_calls[0][1]["use_safetensors"])
            self.assertTrue(load_calls[0][1]["weights_only"])
            self.assertEqual(config.model_path, str(model_path))
            self.assertIsInstance(loaded, MergeableCausalLM)

    def test_rejects_malformed_model_type(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "config.json").write_text(json.dumps({"model_type": None}), encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            with self.assertRaises(ValueError):
                validate_model_path(root)

    def test_rejects_non_qwen_model_type(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "config.json").write_text(json.dumps({"model_type": "mamba"}), encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            with self.assertRaises(ValueError):
                validate_model_path(root)
