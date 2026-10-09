import json
import os
import re
from pathlib import Path


_SHARD_PATTERN = re.compile(r"^adapter_model-(\d+)-of-(\d+)\.safetensors$")


def safetensors_weight_files(source: str | Path) -> list[Path]:
    path = Path(source)
    index_path = path / "adapter_model.safetensors.index.json"
    if index_path.exists():
        if not index_path.is_file():
            raise ValueError(f"Invalid adapter shard index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid adapter shard index: {index_path}")
        names = set(weight_map.values())
        if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") for name in names):
            raise ValueError(f"Invalid shard names in adapter index: {index_path}")
        files = sorted(path / name for name in names)
        missing = [item for item in files if not item.is_file() or item.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Adapter shard files are missing or empty: {missing}")
        return files
    files = sorted(item for item in path.glob("*.safetensors") if item.is_file() and item.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(f"Non-empty safetensors weights were not found in {path}")
    shard_parts = [_SHARD_PATTERN.fullmatch(item.name) for item in files]
    if any(shard_parts):
        if not all(shard_parts):
            raise ValueError(f"Mixed numbered and unnumbered adapter shards in {path}")
        totals = {int(match.group(2)) for match in shard_parts}
        numbers = {int(match.group(1)) for match in shard_parts}
        if len(totals) != 1:
            raise ValueError(f"Inconsistent adapter shard totals in {path}")
        total = next(iter(totals))
        if numbers != set(range(1, total + 1)):
            raise FileNotFoundError(f"Numbered adapter shards are incomplete in {path}")
    return files


def load_adapter_deltas(adapter_path: str | Path, fallback_rank: int, fallback_alpha: int):
    source = Path(adapter_path)
    config_path = source / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LoRA configuration was not found: {config_path}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = adapter_config.get("r", fallback_rank)
    alpha = adapter_config.get("lora_alpha", fallback_alpha)
    scale = alpha / rank
    files = os.listdir(source)
    safe_file = next((name for name in files if name.endswith(".safetensors")), None)
    if safe_file is not None:
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("Install safetensors to load LoRA weights") from error
        state = load_file(str(source / safe_file), device="cpu")
    else:
        state = {}
    if not state:
        bin_file = next((name for name in files if name.endswith(".bin")), None)
        if bin_file is not None:
            import torch

            try:
                state = torch.load(str(source / bin_file), map_location="cpu", weights_only=True)
            except TypeError as error:
                raise RuntimeError("Safe .bin loading requires torch.load(weights_only=True)") from error
    matrices_a = {}
    matrices_b = {}
    for key, tensor in state.items():
        clean = key
        for prefix in ("base_model.model.", "base_model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        if "lora_A" in clean:
            matrices_a[clean.replace(".lora_A.weight", ".weight")] = tensor.float()
        elif "lora_B" in clean:
            matrices_b[clean.replace(".lora_B.weight", ".weight")] = tensor.float()
    return {name: scale * matrices_b[name] @ matrices_a[name] for name in matrices_a if name in matrices_b}


def compute_ties(all_deltas, lora_names, density: float = 0.5):
    import torch

    parameter_names = set()
    for deltas in all_deltas.values():
        parameter_names.update(deltas)
    merged = {}
    for parameter_name in parameter_names:
        updates = [all_deltas[domain][parameter_name].float() for domain in lora_names if parameter_name in all_deltas[domain]]
        if not updates:
            continue
        pruned = []
        for update in updates:
            flattened = update.abs().flatten()
            threshold = float(flattened.kthvalue(max(1, int((1 - density) * flattened.numel()))).values)
            pruned.append(torch.where(update.abs() >= threshold, update, torch.zeros_like(update)))
        majority = torch.stack([update.sign() for update in pruned]).sum(0).sign()
        aligned = [update * (update.sign() == majority).float() for update in pruned]
        merged[parameter_name] = torch.stack(aligned).mean(0)
    return merged


def apply_ties(model, ties_deltas):
    import torch

    cache_key = ("ties",)
    if model.current_key == cache_key:
        return
    with torch.no_grad():
        parameters = dict(model.model.named_parameters())
        for name, base_weight in model.base_weights.items():
            delta = ties_deltas.get(name)
            parameter = parameters.get(name)
            if delta is not None and parameter is not None:
                parameter.data.copy_((base_weight.float() + delta).to(getattr(torch, model.config.dtype)))
    model.current_key = cache_key


class ModelMerger:
    def __init__(self, model, all_deltas, lora_names, ties_density=0.5):
        self.model = model
        self.all_deltas = all_deltas
        self.lora_names = list(lora_names)
        self.ties_deltas = compute_ties(all_deltas, self.lora_names, density=ties_density)

    def apply_coefficients(self, coefficients):
        self.model.set_coefficients(coefficients, self.all_deltas, self.lora_names)

    def reset_base(self):
        self.apply_coefficients([0.0] * len(self.lora_names))

    def apply_single(self, domain):
        self.apply_coefficients([1.0 if name == domain else 0.0 for name in self.lora_names])

    def apply_equal(self):
        self.apply_coefficients([1.0 / len(self.lora_names)] * len(self.lora_names))

    def apply_ties(self):
        apply_ties(self.model, self.ties_deltas)
