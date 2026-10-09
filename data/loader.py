from pathlib import Path
from typing import Any, Callable, Mapping


_DEFAULT_FALLBACK_ROOTS = (
    "/kaggle/input/spectra-day1-2-artifacts-v1",
    "/kaggle/input/datasets/ali320230101/spectra-day1-2-artifacts-v1",
)


def _synthetic_dataset_dict(dataset_factory=None, dataset_dict_factory=None):
    if dataset_factory is None or dataset_dict_factory is None:
        try:
            from datasets import Dataset, DatasetDict
        except ImportError as error:
            raise RuntimeError("Install the datasets package to load or create datasets") from error
        dataset_factory = dataset_factory or Dataset
        dataset_dict_factory = dataset_dict_factory or DatasetDict
    return dataset_dict_factory(
        {
            "math": dataset_factory.from_list(
                [
                    {"prompt": f"Solve: x+{index}={index + 10}", "response": "x=10 → \\boxed{10}"}
                    for index in range(200)
                ]
            ),
            "coding": dataset_factory.from_list(
                [
                    {
                        "prompt": f"Write function #{index} returning {index}",
                        "response": f"```python\ndef f(): return {index}\n```",
                    }
                    for index in range(200)
                ]
            ),
            "hybrid": dataset_factory.from_list(
                [
                    {
                        "prompt": f"Compute {index}^2 in code",
                        "response": f"\\boxed{{{index**2}}}\n```python\nprint({index}**2)\n```",
                    }
                    for index in range(200)
                ]
            ),
        }
    )


def load_unified_dataset(
    path: str | Path,
    lora_names,
    domain_to_split: Mapping[str, str],
    load_from_disk_fn: Callable[..., Any] | None = None,
    fallback_roots=None,
    dataset_path_override: str | Path | None = None,
    dataset_factory=None,
    dataset_dict_factory=None,
):
    if load_from_disk_fn is None:
        try:
            from datasets import load_from_disk
        except ImportError as error:
            raise RuntimeError("Install the datasets package to load the unified dataset") from error
        load_from_disk_fn = load_from_disk
    roots = (path, *(_DEFAULT_FALLBACK_ROOTS if fallback_roots is None else fallback_roots))
    candidates = []
    if dataset_path_override is not None:
        candidates.append(Path(dataset_path_override))
    candidates.extend(Path(root) / "data" / "unified_dataset" for root in roots)
    for source in candidates:
        if not source.is_dir():
            continue
        try:
            loaded = load_from_disk_fn(str(source), keep_in_memory=True)
        except Exception as error:
            print(f"[WARN] {error}")
            continue
        print(f"[DATA] {source}")
        return loaded
    return _synthetic_dataset_dict(dataset_factory, dataset_dict_factory)
