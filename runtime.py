import random
import re
from dataclasses import dataclass
from pathlib import Path

from data.loader import load_unified_dataset
from data.splits import split_dataset
from models.base import MergeableCausalLM
from models.gates import DomainRouter
from models.lora import existing_adapter_path
from models.merging import ModelMerger, load_adapter_deltas
from results import ResultStore
from utils import atomic_save_npy, print_sep, set_seed


_SAFE_SPLIT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def _validate_coefficients(coefficients, adapter_count: int):
    import numpy as np

    values = np.asarray(coefficients)
    if values.shape != (adapter_count,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Coefficient artifacts must be finite numeric vectors matching the adapter count")
    return values.astype(float, copy=False)


def load_coefficient_artifact(path: str | Path, adapter_count: int):
    import numpy as np

    values = np.load(path, allow_pickle=False)
    return _validate_coefficients(values, adapter_count)


def _split_manifest_path(root: str | Path, split_name: str) -> Path:
    if not isinstance(split_name, str) or not _SAFE_SPLIT_NAME.fullmatch(split_name):
        raise ValueError("Split name must be a safe single path component")
    root_path = Path(root)
    manifest = root_path / f"{split_name}.json"
    if not manifest.resolve().is_relative_to(root_path.resolve()):
        raise ValueError("Split manifest path escapes the configured split directory")
    return manifest


@dataclass
class ExperimentRuntime:
    config: object
    logger: object
    model: MergeableCausalLM
    train_sets: dict
    adapt_sets: dict
    test_sets: dict
    adapter_paths: dict
    all_deltas: dict
    merger: ModelMerger
    router: DomainRouter
    result_store: ResultStore
    saved_coefficients: dict

    @property
    def lora_names(self):
        return list(self.config.lora_names)

    def apply_coefficients(self, coefficients):
        self.merger.apply_coefficients(coefficients)

    def reset_base(self):
        self.merger.reset_base()

    def apply_single(self, domain):
        self.merger.apply_single(domain)

    def apply_equal(self):
        self.merger.apply_equal()

    def apply_ties(self):
        self.merger.apply_ties()

    def get_test_examples(self, domain, count=None):
        dataset = self.test_sets.get(self.config.domain_map.get(domain, domain))
        if dataset is None:
            return []
        size = len(dataset) if count is None else min(count, len(dataset))
        return [(dataset[index]["prompt"], dataset[index]["response"]) for index in range(size)]

    def get_adapt_examples(self, domain, count=None, seed_offset=0):
        dataset = self.adapt_sets.get(self.config.domain_map.get(domain, domain))
        if dataset is None:
            return []
        count = self.config.n_adapt if count is None else count
        set_seed(self.config.seed + seed_offset)
        selected = random.sample(range(len(dataset)), min(count, len(dataset)))
        return [(dataset[index]["prompt"], dataset[index]["response"]) for index in selected]

    def get_adapt_prompts(self, domain, count=None, seed_offset=0):
        return [prompt for prompt, _ in self.get_adapt_examples(domain, count, seed_offset)]

    def save_coefficients(self, domain, coefficients):
        path = Path(self.config.work_dir) / "artifacts" / f"coeffs_{domain}.npy"
        values = _validate_coefficients(coefficients, len(self.lora_names))
        atomic_save_npy(path, values)
        self.saved_coefficients[domain] = values


def load_data_splits(config, logger):
    print_sep("Dataset")
    day12_dir = getattr(config, "day12_dir", None)
    dataset_path = getattr(config, "dataset_path", None)
    if day12_dir is None:
        day12_dir = dataset_path or "/kaggle/input/spectra-day1-2-artifacts-v1"
        if Path(day12_dir).name == "unified_dataset":
            day12_dir = Path(day12_dir).parents[1]
        dataset_path_override = None
    else:
        default_dataset_path = Path(day12_dir) / "data" / "unified_dataset"
        dataset_path_override = dataset_path if dataset_path and Path(dataset_path) != default_dataset_path else None
    dataset_dict = load_unified_dataset(
        day12_dir,
        lora_names=config.lora_names,
        domain_to_split=config.domain_map,
        dataset_path_override=dataset_path_override,
    )
    for split, dataset in dataset_dict.items():
        print(f"  {split}: {len(dataset)}")
    train_sets = {}
    adapt_sets = {}
    test_sets = {}
    for domain in config.lora_names:
        split_name = config.domain_map.get(domain, domain)
        splits = split_dataset(
            dataset_dict[split_name],
            n_test=config.n_test,
            n_adapt=config.n_adapt,
            seed=config.seed,
        )
        train_sets[split_name] = splits.train
        adapt_sets[split_name] = splits.adapt
        test_sets[split_name] = splits.test
        print(
            f"  {split_name}: train={len(splits.train)}, "
            f"adapt={len(splits.adapt)}, test={len(splits.test)}"
        )
    return train_sets, adapt_sets, test_sets


def build_training_runtime(config, logger):
    model = MergeableCausalLM.from_pretrained(config, logger)
    train_sets, adapt_sets, test_sets = load_data_splits(config, logger)
    return model, train_sets, adapt_sets, test_sets


def build_evaluation_runtime(config, logger, model=None, split_data=None):
    if split_data is None:
        train_sets, adapt_sets, test_sets = load_data_splits(config, logger)
    else:
        train_sets, adapt_sets, test_sets = split_data
    if model is None:
        model = MergeableCausalLM.from_pretrained(config, logger)
    print_sep("LoRA Delta Extraction")
    adapter_paths = {}
    all_deltas = {}
    for domain in config.lora_names:
        path = existing_adapter_path(config, domain)
        if path is None:
            raise FileNotFoundError(f"LoRA adapter is missing for {domain}; run train.py first")
        adapter_paths[domain] = path
        all_deltas[domain] = load_adapter_deltas(path, config.lora_r, config.lora_alpha)
        logger.info("Loaded %d adapter deltas for %s from %s", len(all_deltas[domain]), domain, path)
    merger = ModelMerger(model, all_deltas, config.lora_names, config.ties_density)
    router = DomainRouter(model, config.lora_names, merger.apply_single)
    print_sep("SPECTRA v4: Domain-Routing + TTA")
    result_store = ResultStore(config.work_dir)
    return ExperimentRuntime(
        config=config,
        logger=logger,
        model=model,
        train_sets=train_sets,
        adapt_sets=adapt_sets,
        test_sets=test_sets,
        adapter_paths=adapter_paths,
        all_deltas=all_deltas,
        merger=merger,
        router=router,
        result_store=result_store,
        saved_coefficients={},
    )
