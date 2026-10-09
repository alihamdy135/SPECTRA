import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from config import ExperimentConfig
from data.splits import load_or_create_splits, split_dataset
from runtime import ExperimentRuntime, load_data_splits


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    @property
    def column_names(self):
        return list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    @classmethod
    def from_dict(cls, columns):
        names = list(columns)
        rows = [
            {name: columns[name][index] for name in names}
            for index in range(len(columns[names[0]]))
        ] if names else []
        return cls(rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        if isinstance(key, slice):
            selected = self.rows[key]
            return {name: [row[name] for row in selected] for name in self.column_names}
        return self.rows[key]

    def add_column(self, name, values):
        return FakeDataset([dict(row, **{name: value}) for row, value in zip(self.rows, values)])

    def shuffle(self, seed, keep_in_memory=True):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return FakeDataset(rows)

    def select(self, indices, keep_in_memory=True):
        return FakeDataset([self.rows[index] for index in indices])

    def remove_columns(self, names):
        return FakeDataset([{key: value for key, value in row.items() if key not in names} for row in self.rows])


class SplitDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = FakeDataset([{"prompt": f"p{i}", "response": f"r{i}"} for i in range(20)])

    def test_split_is_deterministic_disjoint_and_exhaustive(self):
        first = split_dataset(self.dataset, n_test=3, n_adapt=4, seed=42)
        second = split_dataset(self.dataset, n_test=3, n_adapt=4, seed=42)
        first_rows = [first.train, first.adapt, first.test]
        second_rows = [second.train, second.adapt, second.test]
        first_prompts = [[row["prompt"] for row in part] for part in first_rows]
        second_prompts = [[row["prompt"] for row in part] for part in second_rows]
        self.assertEqual(first_prompts, second_prompts)
        self.assertEqual([len(part) for part in first_rows], [13, 4, 3])
        flattened = [prompt for part in first_prompts for prompt in part]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {f"p{i}" for i in range(20)})
        self.assertEqual(first.indices, second.indices)

    def test_split_rows_keep_notebook_shuffle_and_partition_order(self):
        shuffled = self.dataset.shuffle(seed=42, keep_in_memory=True)
        expected_test = shuffled.select(range(3), keep_in_memory=True)
        expected_adapt = shuffled.select(range(3, 7), keep_in_memory=True)
        expected_train = shuffled.select(range(7, len(shuffled)), keep_in_memory=True)
        result = split_dataset(self.dataset, n_test=3, n_adapt=4, seed=42)

        self.assertEqual([row["prompt"] for row in result.test], [row["prompt"] for row in expected_test])
        self.assertEqual([row["prompt"] for row in result.adapt], [row["prompt"] for row in expected_adapt])
        self.assertEqual([row["prompt"] for row in result.train], [row["prompt"] for row in expected_train])

    def test_split_preserves_a_preexisting_source_row_column(self):
        rows = [
            {"prompt": f"p{i}", "response": f"r{i}", "__spectra_source_row__": f"original-{i}"}
            for i in range(20)
        ]
        dataset = FakeDataset(rows)
        result = split_dataset(dataset, n_test=3, n_adapt=4, seed=42)
        shuffled = dataset.shuffle(seed=42, keep_in_memory=True)
        expected = [shuffled[index] for index in range(3)]
        self.assertEqual([row["prompt"] for row in result.test], [row["prompt"] for row in expected])
        self.assertEqual([row["__spectra_source_row__"] for row in result.test], [row["__spectra_source_row__"] for row in expected])


class RuntimeDomainAliasTests(unittest.TestCase):
    def test_runtime_resolves_code_adapter_to_coding_split(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = ExperimentConfig(work_dir=temporary_dir, n_test=3, n_adapt=4)
            datasets = {
                name: FakeDataset([{"prompt": f"{name}-p{i}", "response": f"{name}-r{i}"} for i in range(20)])
                for name in ("math", "coding", "hybrid")
            }
            with patch("runtime.load_unified_dataset", return_value=datasets) as loader:
                train, adapt, test = load_data_splits(config, Mock())

        loader.assert_called_once_with(
            config.day12_dir,
            lora_names=config.lora_names,
            domain_to_split=config.domain_map,
            dataset_path_override=None,
        )
        self.assertEqual(list(train), ["math", "coding", "hybrid"])
        self.assertTrue(all(row["prompt"].startswith("coding-") for row in train["coding"]))
        self.assertTrue(all(row["prompt"].startswith("coding-") for row in adapt["coding"]))
        self.assertTrue(all(row["prompt"].startswith("coding-") for row in test["coding"]))

    def test_runtime_getters_resolve_code_adapter_to_coding_split(self):
        config = ExperimentConfig(n_adapt=2)
        coding = FakeDataset([
            {"prompt": f"coding-p{i}", "response": f"coding-r{i}"}
            for i in range(5)
        ])
        runtime = ExperimentRuntime(
            config=config,
            logger=Mock(),
            model=None,
            train_sets={},
            adapt_sets={"coding": coding},
            test_sets={"coding": coding},
            adapter_paths={},
            all_deltas={},
            merger=None,
            router=None,
            result_store=None,
            saved_coefficients={},
        )

        self.assertEqual(runtime.get_test_examples("code"), [
            ("coding-p0", "coding-r0"),
            ("coding-p1", "coding-r1"),
            ("coding-p2", "coding-r2"),
            ("coding-p3", "coding-r3"),
            ("coding-p4", "coding-r4"),
        ])
        self.assertEqual(len(runtime.get_adapt_examples("code")), 2)

    def test_runtime_keeps_empty_partitions_when_notebook_split_would(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = ExperimentConfig(work_dir=temporary_dir, n_test=50, n_adapt=15)
            datasets = {
                name: FakeDataset([{"prompt": f"{name}-p{i}", "response": f"{name}-r{i}"} for i in range(10)])
                for name in ("math", "coding", "hybrid")
            }
            with patch("runtime.load_unified_dataset", return_value=datasets):
                train, adapt, test = load_data_splits(config, Mock())

        self.assertEqual(len(test["coding"]), 10)
        self.assertEqual(len(adapt["coding"]), 0)
        self.assertEqual(len(train["coding"]), 0)


class SplitManifestTests(unittest.TestCase):
    def setUp(self):
        self.dataset = FakeDataset([{"prompt": f"p{i}", "response": f"r{i}"} for i in range(20)])

    def test_split_counts_are_capped_by_dataset_size(self):
        result = split_dataset(self.dataset, n_test=50, n_adapt=15, seed=42)
        self.assertEqual(len(result.test), 20)
        self.assertEqual(len(result.adapt), 0)
        self.assertEqual(len(result.train), 0)

    def test_manifest_does_not_change_rows_or_fail_on_stale_or_corrupt_contents(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = Path(temporary_dir) / "math.json"
            expected = split_dataset(self.dataset, n_test=3, n_adapt=4, seed=42)
            original = load_or_create_splits(self.dataset, manifest, n_test=3, n_adapt=4, seed=42)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            saved["indices"]["adapt"].append(saved["indices"]["train"].pop())
            manifest.write_text(json.dumps(saved), encoding="utf-8")
            second = load_or_create_splits(self.dataset, manifest, n_test=3, n_adapt=4, seed=42)
            self.assertEqual(original.indices, expected.indices)
            self.assertEqual(expected.indices, second.indices)
            self.assertEqual(
                [[row["prompt"] for row in part] for part in (expected.train, expected.adapt, expected.test)],
                [[row["prompt"] for row in part] for part in (second.train, second.adapt, second.test)],
            )

            manifest.write_text("not json", encoding="utf-8")
            third = load_or_create_splits(self.dataset, manifest, n_test=3, n_adapt=4, seed=42)
            self.assertEqual(second.indices, third.indices)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["seed"], 42)

    def test_manifest_settings_and_source_changes_do_not_block_notebook_split(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = Path(temporary_dir) / "math.json"
            first = load_or_create_splits(self.dataset, manifest, n_test=3, n_adapt=4, seed=42)
            changed_data = FakeDataset([{"prompt": "changed", "response": "r0"}] + self.dataset.rows[1:])
            recomputed = split_dataset(changed_data, n_test=3, n_adapt=4, seed=43)
            result = load_or_create_splits(changed_data, manifest, n_test=3, n_adapt=4, seed=43)
            self.assertNotEqual(first.indices, result.indices)
            self.assertEqual(result.indices, recomputed.indices)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 43)

    def test_manifest_write_failure_does_not_block_notebook_split(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            blocker = Path(temporary_dir) / "not-a-directory"
            blocker.write_text("occupied", encoding="utf-8")
            expected = split_dataset(self.dataset, n_test=3, n_adapt=4, seed=42)
            result = load_or_create_splits(
                self.dataset,
                blocker / "math.json",
                n_test=3,
                n_adapt=4,
                seed=42,
            )

        self.assertEqual(result.indices, expected.indices)
