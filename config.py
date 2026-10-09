from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import os
import re

import yaml


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def _is_safe_component(value: Any) -> bool:
    return isinstance(value, str) and _SAFE_COMPONENT.fullmatch(value) is not None


@dataclass
class ExperimentConfig:
    day12_dir: str = "/kaggle/input/spectra-day1-2-artifacts-v1"
    dataset_path: str = ""
    artifact_root: str = ""
    model_path: str = ""
    work_dir: str = "/kaggle/working"
    run_id: str = ""
    lora_names: tuple[str, ...] = ("math", "code", "hybrid")
    domain_to_split: tuple[tuple[str, str], ...] = (("math", "math"), ("code", "coding"), ("hybrid", "hybrid"), ("coding", "coding"))
    seed: int = 42
    evaluation_seeds: tuple[int, ...] = (42, 123, 256, 512, 1024)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    train_epochs: int = 2
    train_batch: int = 2
    grad_accum: int = 4
    learning_rate: float = 0.0002
    max_seq_len: int = 256
    gen_max_new: int = 128
    generation_batch_size: int = 4
    cma_pop: int = 8
    cma_iters: int = 40
    cma_sigma0: float = 0.3
    n_adapt: int = 15
    n_test: int = 50
    n_bootstrap: int = 1000
    ties_density: float = 0.5
    router_prior_weight: float = 0.1
    grid_points: int = 25
    grid_evaluations: int = 35
    exp3_adapt_counts: tuple[int, ...] = (5, 10, 15, 20)
    exp5_test_count: int = 20
    exp6_test_per_domain: int = 15
    exp6_adapt_per_domain: int = 5
    dtype: str = "float16"
    trust_remote_code: bool = False

    def __post_init__(self):
        if not self.dataset_path:
            object.__setattr__(self, "dataset_path", str(Path(self.day12_dir) / "data" / "unified_dataset"))
        if not self.artifact_root:
            object.__setattr__(self, "artifact_root", self.day12_dir)
        integer_fields = (
            "seed",
            "lora_r",
            "lora_alpha",
            "train_epochs",
            "train_batch",
            "grad_accum",
            "max_seq_len",
            "gen_max_new",
            "generation_batch_size",
            "cma_pop",
            "cma_iters",
            "n_adapt",
            "n_test",
            "n_bootstrap",
            "grid_points",
            "grid_evaluations",
            "exp5_test_count",
            "exp6_test_per_domain",
            "exp6_adapt_per_domain",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ValueError("Count, seed, and dimension settings must be integers")
        if any(type(seed) is not int for seed in self.evaluation_seeds):
            raise ValueError("evaluation_seeds must contain integers")
        if any(type(count) is not int for count in self.exp3_adapt_counts):
            raise ValueError("exp3_adapt_counts must contain integers")
        if not self.lora_names:
            raise ValueError("lora_names cannot be empty")
        if not isinstance(self.run_id, str) or (self.run_id and not _is_safe_component(self.run_id)):
            raise ValueError("run_id must be empty or a safe single path component")
        if any(not _is_safe_component(name) for name in self.lora_names):
            raise ValueError("lora_names must contain safe single path components")
        for pair in self.domain_to_split:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(not _is_safe_component(value) for value in pair):
                raise ValueError("domain_to_split must contain safe single path components")
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        if not self.evaluation_seeds:
            raise ValueError("evaluation_seeds cannot be empty")
        if self.grid_points < 2 or self.grid_evaluations < 1:
            raise ValueError("Grid settings require at least two points and one evaluation")
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be positive")
        if self.train_epochs < 1 or self.cma_pop < 2 or self.cma_iters < 1:
            raise ValueError("Training epochs and CMA settings must be positive")
        if self.seed < 0 or any(seed < 0 for seed in self.evaluation_seeds):
            raise ValueError("Seeds must be non-negative")
        if self.n_adapt < 0 or self.n_test < 0:
            raise ValueError("Split sizes must be non-negative")
        if self.grad_accum < 1 or self.train_batch < 1:
            raise ValueError("Batch sizes must be positive")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")

    @property
    def domain_map(self) -> dict[str, str]:
        return dict(self.domain_to_split)

    @property
    def output_path(self) -> Path:
        return Path(self.work_dir) / self.run_id if self.run_id else Path(self.work_dir)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    loaded: Any = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Configuration must be a YAML mapping")
    allowed = {item.name for item in fields(ExperimentConfig)}
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration settings: {sorted(unknown)}")
    values = dict(loaded)
    tuple_fields = {"lora_names", "evaluation_seeds", "lora_targets", "exp3_adapt_counts"}
    for name in tuple_fields:
        if name in values:
            values[name] = tuple(values[name])
    if "domain_to_split" in values:
        values["domain_to_split"] = tuple(tuple(pair) for pair in values["domain_to_split"])
    environment = {
        "SPECTRA_DAY12_DIR": "day12_dir",
        "SPECTRA_DATASET_PATH": "dataset_path",
        "SPECTRA_ARTIFACT_ROOT": "artifact_root",
        "SPECTRA_MODEL_PATH": "model_path",
        "SPECTRA_WORK_DIR": "work_dir",
        "SPECTRA_RUN_ID": "run_id",
    }
    for variable, key in environment.items():
        if variable in os.environ:
            values[key] = os.environ[variable]
    return ExperimentConfig(**values)
