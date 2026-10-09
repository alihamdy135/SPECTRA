import tempfile
import unittest
from pathlib import Path

from data.loader import _DEFAULT_FALLBACK_ROOTS, load_unified_dataset


class DatasetStub:
    column_names = ["prompt", "response"]

    def __getitem__(self, name):
        return [f"{name}-{index}" for index in range(len(self))]

    def __len__(self):
        return 4


class ValuesDataset:
    column_names = ["prompt", "response"]

    def __init__(self, prompt, response="answer"):
        self.values = {"prompt": [prompt], "response": [response]}

    def __len__(self):
        return len(self.values["prompt"])

    def __getitem__(self, name):
        return self.values[name]


class FallbackDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0]) if self.rows else []

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, name):
        if isinstance(name, str):
            return [row[name] for row in self.rows]
        return self.rows[name]


class FallbackDatasetDict(dict):
    pass


class LoaderTests(unittest.TestCase):
    def test_fallback_roots_match_notebook_search_order(self):
        self.assertEqual(
            _DEFAULT_FALLBACK_ROOTS,
            (
                "/kaggle/input/spectra-day1-2-artifacts-v1",
                "/kaggle/input/datasets/ali320230101/spectra-day1-2-artifacts-v1",
            ),
        )

    def test_loader_validates_required_domains_and_columns(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "data" / "unified_dataset"
            source.mkdir(parents=True)
            captured = {}

            def loader(path, keep_in_memory):
                captured["path"] = path
                captured["keep_in_memory"] = keep_in_memory
                return {"math": DatasetStub(), "coding": DatasetStub(), "hybrid": DatasetStub()}

            result = load_unified_dataset(
                root,
                lora_names=("math", "code", "hybrid"),
                domain_to_split={"math": "math", "code": "coding", "hybrid": "hybrid"},
                load_from_disk_fn=loader,
            )
        self.assertEqual(set(result), {"math", "coding", "hybrid"})
        self.assertEqual(captured["path"], str(source))
        self.assertTrue(captured["keep_in_memory"])

    def test_loader_uses_notebook_search_order_after_load_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            roots = [Path(temporary_dir) / name for name in ("configured", "published", "dataset")]
            paths = [root / "data" / "unified_dataset" for root in roots]
            for path in paths:
                path.mkdir(parents=True)
            calls = []

            def loader(path, keep_in_memory):
                calls.append((path, keep_in_memory))
                if path == str(paths[0]):
                    raise OSError("unreadable source")
                return {"math": DatasetStub(), "coding": DatasetStub(), "hybrid": DatasetStub()}

            result = load_unified_dataset(
                roots[0],
                lora_names=("math", "code", "hybrid"),
                domain_to_split={"math": "math", "code": "coding", "hybrid": "hybrid"},
                load_from_disk_fn=loader,
                fallback_roots=roots[1:],
            )

        self.assertEqual(calls, [(str(paths[0]), True), (str(paths[1]), True)])
        self.assertEqual(set(result), {"math", "coding", "hybrid"})

    def test_loader_creates_notebook_synthetic_dataset_dict_when_sources_fail(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            roots = [Path(temporary_dir) / name for name in ("configured", "published", "dataset")]
            paths = [root / "data" / "unified_dataset" for root in roots]
            for path in paths:
                path.mkdir(parents=True)
            calls = []

            def loader(path, keep_in_memory):
                calls.append((path, keep_in_memory))
                raise OSError("unreadable source")

            result = load_unified_dataset(
                roots[0],
                lora_names=("math", "code", "hybrid"),
                domain_to_split={"math": "math", "code": "coding", "hybrid": "hybrid"},
                load_from_disk_fn=loader,
                fallback_roots=roots[1:],
                dataset_factory=FallbackDataset,
                dataset_dict_factory=FallbackDatasetDict,
            )

        self.assertEqual(calls, [(str(path), True) for path in paths])
        self.assertIsInstance(result, FallbackDatasetDict)
        self.assertEqual(set(result), {"math", "coding", "hybrid"})
        self.assertEqual([len(result[name]) for name in ("math", "coding", "hybrid")], [200, 200, 200])
        self.assertEqual(result["math"][0], {"prompt": "Solve: x+0=10", "response": "x=10 → \\boxed{10}"})
        self.assertEqual(result["coding"][0], {"prompt": "Write function #0 returning 0", "response": "```python\ndef f(): return 0\n```"})
        self.assertEqual(result["hybrid"][-1], {"prompt": "Compute 199^2 in code", "response": "\\boxed{39601}\n```python\nprint(199**2)\n```"})

    def test_loader_returns_loaded_examples_without_extra_validation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "data" / "unified_dataset").mkdir(parents=True)
            for prompt in (None, "", "   ", 17):
                with self.subTest(prompt=prompt):
                    dataset = {"math": ValuesDataset(prompt)}
                    result = load_unified_dataset(
                        root,
                        lora_names=("math",),
                        domain_to_split={"math": "math"},
                        load_from_disk_fn=lambda *args, **kwargs: dataset,
                    )
                    self.assertIs(result, dataset)
            for response in (None, "", "   ", 17):
                with self.subTest(response=response):
                    dataset = {"math": ValuesDataset("prompt", response)}
                    result = load_unified_dataset(
                        root,
                        lora_names=("math",),
                        domain_to_split={"math": "math"},
                        load_from_disk_fn=lambda *args, **kwargs: dataset,
                    )
                    self.assertIs(result, dataset)

    def test_loader_returns_missing_or_malformed_domain_data(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "data" / "unified_dataset").mkdir(parents=True)
            missing = {"math": DatasetStub()}
            result = load_unified_dataset(
                root,
                lora_names=("math", "code"),
                domain_to_split={"math": "math", "code": "coding"},
                load_from_disk_fn=lambda *args, **kwargs: missing,
            )
            self.assertIs(result, missing)
            malformed = {"math": object()}
            result = load_unified_dataset(
                root,
                lora_names=("math",),
                domain_to_split={"math": "math"},
                load_from_disk_fn=lambda *args, **kwargs: malformed,
            )
            self.assertIs(result, malformed)
