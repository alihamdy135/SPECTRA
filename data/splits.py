import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import atomic_write_text


_SOURCE_ROW = "__spectra_source_row__"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetSplits:
    train: Any
    adapt: Any
    test: Any
    indices: dict[str, list[int]]


def _dataset_fingerprint(dataset: Any) -> str:
    columns = set(getattr(dataset, "column_names", []))
    missing = {"prompt", "response"} - columns
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    digest = hashlib.sha256()
    for row in dataset:
        payload = json.dumps(
            [row["prompt"], row["response"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_indices(indices: dict[str, list[int]], total: int) -> None:
    names = ("train", "adapt", "test")
    if set(indices) != set(names):
        raise ValueError("Split manifest must define train, adapt, and test indices")
    flattened = [index for name in names for index in indices[name]]
    if any(not isinstance(index, int) or index < 0 or index >= total for index in flattened):
        raise ValueError("Split manifest contains an out-of-range index")
    if len(flattened) != len(set(flattened)) or set(flattened) != set(range(total)):
        raise ValueError("Split manifest indices must be disjoint and exhaustive")


def _select(dataset: Any, indices: list[int]) -> Any:
    return dataset.select(indices, keep_in_memory=True)


def _from_indices(dataset: Any, indices: dict[str, list[int]]) -> DatasetSplits:
    return DatasetSplits(
        train=_select(dataset, indices["train"]),
        adapt=_select(dataset, indices["adapt"]),
        test=_select(dataset, indices["test"]),
        indices=indices,
    )


def split_dataset(dataset: Any, n_test: int, n_adapt: int, seed: int) -> DatasetSplits:
    if n_test < 0 or n_adapt < 0:
        raise ValueError("Split sizes must be non-negative")
    materialize = getattr(type(dataset), "from_dict", None)
    if callable(materialize):
        dataset = materialize(dataset[:])
    total = len(dataset)
    source_row = _SOURCE_ROW
    columns = set(getattr(dataset, "column_names", []))
    while source_row in columns:
        source_row += "_"
    indexed = dataset.add_column(source_row, list(range(total)))
    shuffled = indexed.shuffle(seed=seed, keep_in_memory=True)
    order = [int(value) for value in shuffled[source_row]]
    test_end = min(n_test, total)
    adapt_end = min(n_test + n_adapt, total)
    ranges = {
        "test": (0, test_end),
        "adapt": (test_end, adapt_end),
        "train": (adapt_end, total),
    }
    indices = {name: order[start:end] for name, (start, end) in ranges.items()}
    _validate_indices(indices, total)
    return _from_indices(dataset, indices)


def load_or_create_splits(
    dataset: Any,
    manifest_path: str | Path,
    n_test: int,
    n_adapt: int,
    seed: int,
) -> DatasetSplits:
    splits = split_dataset(dataset, n_test=n_test, n_adapt=n_adapt, seed=seed)
    try:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "source_sha256": _dataset_fingerprint(dataset),
            "dataset_rows": len(dataset),
            "seed": seed,
            "n_test": n_test,
            "n_adapt": n_adapt,
            "indices": splits.indices,
        }
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass
    return splits
