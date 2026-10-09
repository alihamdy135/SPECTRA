import hashlib
import unittest
from pathlib import Path


NOTEBOOK_SHA256 = "541c602263815c727733f0096238510bb2f0cfa9bba172c16d95747d83b17c8a"


class NotebookPreservationTests(unittest.TestCase):
    def test_source_notebook_is_preserved_byte_for_byte(self):
        notebook_path = Path(__file__).resolve().parents[1] / "SPECTRA.ipynb"
        digest = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
        self.assertEqual(digest, NOTEBOOK_SHA256)


if __name__ == "__main__":
    unittest.main()
