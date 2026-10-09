import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils import atomic_save_npy, atomic_write_csv, atomic_write_text


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_write_replaces_destination_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            target = root / "outside.txt"
            destination = root / "result.json"
            target.write_text("original", encoding="utf-8")
            destination.symlink_to(target)
            atomic_write_text(destination, "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "original")
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_text(encoding="utf-8"), "new")

    def test_csv_and_numpy_artifact_writes_replace_symlink_destinations(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            target = root / "outside.bin"
            target.write_bytes(b"original")
            csv_path = root / "artifact.csv"
            csv_path.symlink_to(target)
            atomic_write_csv(csv_path, [{"value": 3}], ("value",))
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(csv_path.read_text(encoding="utf-8").splitlines(), ["value", "3"])
            npy_path = root / "coefficients.npy"
            npy_path.symlink_to(target)
            atomic_save_npy(npy_path, np.asarray([0.2, 0.8]))
            self.assertEqual(target.read_bytes(), b"original")
            np.testing.assert_allclose(np.load(npy_path, allow_pickle=False), [0.2, 0.8])
