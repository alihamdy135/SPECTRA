import json
import logging
import os
from pathlib import Path
from typing import Protocol, Sequence


_MODEL_CANDIDATES = (
    "/kaggle/input/qwen2.5-1.5b-instruct_reasoner/pytorch/default/1",
    "/kaggle/input/qwen2-5-1-5b-instruct/pytorch/default/1",
    "/kaggle/input/qwen2.5-1.5b-instruct/pytorch/default/1",
)
_ARTIFACT_KEYWORDS = ("baseline_full_ft", "lora_", "day1", "day2", "day3", "spectra-day", "checkpoint")
_BASE_KEYWORDS = ("qwen", "1.5b", "instruct", "reasoner")


def _has_weights(path):
    try:
        return any(name.endswith((".safetensors", ".bin")) for name in os.listdir(path))
    except Exception:
        return False


def _valid_model(path):
    if not os.path.isdir(path):
        return False
    if not all(os.path.isfile(f"{path}/{name}") for name in ("config.json", "tokenizer_config.json")):
        return False
    if not _has_weights(path):
        return False
    try:
        with open(f"{path}/config.json", encoding="utf-8") as stream:
            model_config = json.load(stream)
        return model_config.get("model_type", "").lower() in ("qwen2", "qwen2_moe", "qwen")
    except Exception:
        return False


def find_model_path():
    for path in _MODEL_CANDIDATES:
        if _valid_model(path):
            return path
    found = []
    input_root = "/kaggle/input"
    for root, dirs, _ in os.walk(input_root):
        if root[len(input_root) :].count(os.sep) > 6:
            dirs.clear()
            continue
        if _valid_model(root):
            score = sum(keyword in root.lower() for keyword in _BASE_KEYWORDS) - sum(
                keyword in root.lower() for keyword in _ARTIFACT_KEYWORDS
            )
            found.append((score, root))
    if found:
        found.sort(key=lambda item: -item[0])
        print(f"[MODEL] {found[0][1]}")
        return found[0][1]
    raise FileNotFoundError("No valid model found")


class ModelInterface(Protocol):
    def generate(self, prompts: Sequence[str], batch_size: int | None = None, max_new: int | None = None) -> list[str]: ...

    def nll_answer(self, prompt_response_pairs, max_len: int = 128) -> float: ...

    def domain_nll(self, prompts, max_len: int = 64) -> float: ...

    def set_coefficients(self, coefficients, all_deltas, lora_names) -> None: ...


def validate_model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory was not found: {model_path}")
    required = ("config.json", "tokenizer_config.json")
    if not all((model_path / name).is_file() for name in required):
        raise FileNotFoundError(f"Model directory is missing configuration files: {model_path}")
    if not any(item.is_file() and item.name.endswith((".safetensors", ".bin")) for item in model_path.iterdir()):
        raise FileNotFoundError(f"Model weights were not found: {model_path}")
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    model_type = model_config.get("model_type")
    if not isinstance(model_type, str) or model_type.lower() not in {"qwen2", "qwen2_moe", "qwen"}:
        raise ValueError(f"Unsupported model type: {model_type!r}")
    return model_path


class MergeableCausalLM:
    def __init__(self, model, tokenizer, config, device, logger=None):
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.base_weights = {
            name: parameter.data.float().cpu()
            for name, parameter in self.model.named_parameters()
        }
        self.current_key = None
        from utils import vram_mb

        print(f"  ✓ {len(self.base_weights)} params  VRAM={vram_mb():.0f}MB")

    @classmethod
    def from_pretrained(cls, config, logger=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from utils import print_sep

        print_sep("Model Path")
        if config.model_path:
            model_path = validate_model_path(config.model_path)
        else:
            model_path = Path(find_model_path())
            config.model_path = str(model_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, config.dtype)
        print_sep("Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=config.trust_remote_code,
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        print(f"  Vocab={tokenizer.vocab_size}")
        has_safetensors = any(item.is_file() and item.name.endswith(".safetensors") for item in model_path.iterdir())
        model_options = {
            "dtype": dtype,
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": True,
            "use_safetensors": has_safetensors,
            "device_map": {"": device},
        }
        if not has_safetensors:
            model_options["weights_only"] = True
        print_sep("Loading Model")
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_options)
        return cls(model, tokenizer, config, device, logger)

    def replace_model(self, model):
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.current_key = None

    def set_coefficients(self, coefficients, all_deltas, lora_names):
        key = tuple(round(value, 6) for value in coefficients)
        if key == self.current_key:
            return
        with self.torch.no_grad():
            parameters = dict(self.model.named_parameters())
            for name, base_weight in self.base_weights.items():
                has_delta = any(name in all_deltas[domain] for domain in lora_names)
                if not has_delta:
                    if self.current_key is None and name in parameters:
                        parameters[name].data.copy_(base_weight.to(getattr(self.torch, self.config.dtype)))
                    continue
                weight = base_weight.clone()
                for index, domain in enumerate(lora_names):
                    coefficient = coefficients[index]
                    if abs(coefficient) < 1e-9:
                        continue
                    if name in all_deltas[domain]:
                        weight = weight + coefficient * all_deltas[domain][name]
                parameter = parameters.get(name)
                if parameter is not None:
                    parameter.data.copy_(weight.to(getattr(self.torch, self.config.dtype)))
        self.current_key = key

    def generate(self, prompts: Sequence[str], batch_size: int | None = None, max_new: int | None = None) -> list[str]:
        from models.lora import generation_batch_size_for

        if not prompts:
            return []
        batch_size = generation_batch_size_for(batch_size, self.config.generation_batch_size)
        max_new = max_new if max_new is not None else self.config.gen_max_new
        results = []
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len,
            ).to(self.device)
            try:
                with self.torch.no_grad():
                    output = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                input_length = encoded["input_ids"].shape[1]
                results.extend(
                    self.tokenizer.decode(sequence[input_length:], skip_special_tokens=True).strip()
                    for sequence in output
                )
            except Exception as error:
                print(f"  [WARN] gen: {error}")
                results.extend([""] * len(batch))
        return results

    def nll_answer(self, prompt_response_pairs, max_len: int = 128) -> float:
        total = 0.0
        count = 0
        with self.torch.no_grad():
            for prompt, response in prompt_response_pairs:
                full = f"### Q:\n{prompt}\n\n### A:\n{response}"
                prefix = f"### Q:\n{prompt}\n\n### A:\n"
                encoded = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
                prefix_encoded = self.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=max_len)
                input_ids = encoded["input_ids"]
                labels = input_ids.clone()
                prefix_length = min(prefix_encoded["input_ids"].shape[1], input_ids.shape[1])
                labels[:, :prefix_length] = -100
                if int((labels != -100).sum()) == 0:
                    continue
                try:
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=encoded["attention_mask"],
                        labels=labels,
                    ).loss.item()
                    total += loss
                    count += 1
                except Exception:
                    pass
        return total / max(1, count)

    def domain_nll(self, prompts, max_len: int = 64) -> float:
        total = 0.0
        count = 0
        with self.torch.no_grad():
            for prompt in prompts:
                encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
                input_ids = encoded["input_ids"]
                labels = input_ids.clone()
                labels[:, :1] = -100
                try:
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=encoded.get("attention_mask"),
                        labels=labels,
                    ).loss.item()
                    total += loss
                    count += 1
                except Exception:
                    pass
        return total / max(1, count)
