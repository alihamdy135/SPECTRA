import csv
import io
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from results import ResultStore, archive_results, print_final_results


class ResultStoreTests(unittest.TestCase):
    def test_new_store_ignores_previous_rows_and_overwrites_csv_on_first_append(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            path = root / "spectra_results.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("method", "domain", "condition", "em"))
                writer.writeheader()
                writer.writerow({"method": "old", "domain": "math", "condition": "exp1_main", "em": 0.2})
            store = ResultStore(root)
            self.assertEqual(store.rows, [])
            self.assertEqual(store.scores, {})
            store.append({"method": "spectra", "domain": "math", "condition": "exp1_main", "em": 0.9})
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["method"] for row in rows], ["spectra"])
            self.assertEqual(float(rows[0]["em"]), 0.9)

    def test_rows_append_and_per_example_scores_remain_in_memory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            store = ResultStore(root)
            row = {"method": "spectra", "domain": "math", "condition": "exp1_main"}
            store.append(row | {"em": 0.5}, "exp1_main/math", "spectra", [0.0])
            store.append(row | {"em": 1.0}, "exp1_main/math", "spectra", [1.0])
            reloaded = ResultStore(root)
            self.assertEqual([item["em"] for item in store.rows], [0.5, 1.0])
            self.assertEqual(store.scores["exp1_main/math"]["spectra"], [1.0])
            self.assertEqual(reloaded.rows, [])
            self.assertEqual(reloaded.scores, {})
            self.assertFalse((root / "result_store.json").exists())
            self.assertFalse((root / "per_example_scores.json").exists())
            self.assertFalse((root / "results.csv").exists())

    def test_notebook_column_order_tracks_first_seen_fields_in_current_run(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            store = ResultStore(root)
            store.append({"method": "scalar_fusion", "domain": "math", "condition": "exp1_main", "sf_weights": "[1]"})
            store.append({"method": "exp6_summary", "domain": "mixed", "condition": "exp6", "spectra_comp": 0.5})
            expected = ["method", "domain", "condition", "sf_weights", "spectra_comp"]
            with (root / "spectra_results.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(csv.DictReader(stream).fieldnames, expected)
            self.assertEqual(store.columns, expected)

    def test_experiment6_summary_row_is_persisted_without_baseline_result_rows(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            store = ResultStore(root)
            for method in ("no_adaptation", "equal_merge", "ties_merge", "spectra"):
                store.save_scores("exp6/mixed", method, [0.1, 0.2])
            store.append(
                {
                    "method": "exp6_summary",
                    "domain": "mixed",
                    "condition": "exp6",
                    "adapt_time_s": 4.5,
                    "coeffs": "[0.2, 0.3, 0.5]",
                    "spectra_comp": 0.25,
                    "equal_comp": 0.15,
                    "delta_eq": 0.1,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )
            with (root / "spectra_results.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["method"] for row in rows], ["exp6_summary"])
            self.assertEqual(rows[0]["delta_eq"], "0.1")
            self.assertEqual(set(store.scores["exp6/mixed"]), {"no_adaptation", "equal_merge", "ties_merge", "spectra"})

    def test_archive_uses_notebook_spectra_results_name(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            store = ResultStore(root)
            store.append({"method": "spectra", "domain": "math", "condition": "exp1_main"})
            archive = archive_results(root)
            self.assertEqual(archive.name, "spectra_results.zip")
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn("spectra_results.csv", zipped.namelist())

    def test_final_report_prints_notebook_table_and_method_order_rankings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = ResultStore(Path(temporary_dir))
            for method in ("equal_merge", "no_adaptation", "spectra"):
                store.append({
                    "method": method, "domain": "math", "condition": "exp1_main",
                    "em": 1.0, "f1": 1.0, "rouge_l": 1.0, "composite": 1.0,
                    "comp_ci_lo": 0.9, "comp_ci_hi": 1.0,
                })
            output = io.StringIO()
            with redirect_stdout(output):
                print_final_results(store)
        report = output.getvalue()
        self.assertIn("95% CI composite", report)
        self.assertIn("[0.9000,1.0000]", report)
        ranking = report.split("SPECTRA FINAL RANKING:", 1)[1].split("SPECTRA wins", 1)[0]
        first = [line.strip().split("#1=", 1)[1].split()[0] for line in ranking.splitlines() if "#1=" in line]
        self.assertEqual(first, ["no_adaptation", "no_adaptation", "no_adaptation", "no_adaptation"])


if __name__ == "__main__":
    unittest.main()
