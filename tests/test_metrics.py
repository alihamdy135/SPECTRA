import math
import unittest

from metrics import bootstrap_ci, cohens_d, compute_metrics, em_score, extract_final_answer, normalize_answer, paired_ttest, token_f1
from scipy import stats


class MetricTests(unittest.TestCase):
    def test_notebook_keyword_argument_names_are_preserved(self):
        self.assertEqual(normalize_answer(s="42"), "42")
        self.assertEqual(extract_final_answer(text="Answer: 42"), "42")
        self.assertEqual(em_score(pred="42", gold="42"), 1.0)
        self.assertEqual(token_f1(pred="42", gold="42"), 1.0)
        self.assertEqual(compute_metrics(preds=["42"], golds=["42"], do_bootstrap=False)["em"], 1.0)

    def test_extract_final_answer_uses_original_flat_boxed_pattern(self):
        self.assertEqual(extract_final_answer("Work. \\boxed{3}, then \\boxed{4}"), "4")
        self.assertEqual(extract_final_answer(r"Work. \boxed{\frac{1}{2}}"), r"\frac{1")

    def test_original_boxed_fraction_normalization(self):
        self.assertEqual(em_score(r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{3}}"), 1.0)

    def test_metrics_truncate_unequal_prediction_reference_lists(self):
        result = compute_metrics(["correct", "ignored"], ["correct"], do_bootstrap=False)
        self.assertEqual(result["em"], 1.0)
        self.assertEqual(len(result["comp_list"]), 1)

    def test_original_decimal_answer_extraction_stops_at_period(self):
        self.assertEqual(em_score("The answer is 3.14", "3"), 1.0)
        self.assertEqual(em_score("The answer is 3.14", "3.14"), 0.0)

    def test_original_normalization_removes_negative_and_decimal_punctuation(self):
        self.assertEqual(normalize_answer("-3.14"), normalize_answer("3.14"))
        self.assertEqual(em_score("-3.14", "3.14"), 1.0)
        self.assertEqual(em_score(r"\boxed{-\frac{1}{2}}", r"\boxed{\frac{1}{2}}"), 1.0)

    def test_exact_match_normalizes_simple_boxed_answer(self):
        self.assertEqual(em_score("The answer is \\boxed{42}", "42"), 1.0)

    def test_token_f1_counts_repeated_tokens(self):
        self.assertAlmostEqual(token_f1("red red blue", "red blue blue"), 2 / 3)

    def test_bootstrap_empty_input_returns_zero_interval(self):
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0, 0.0))

    def test_bootstrap_interval_is_seed_deterministic(self):
        first = bootstrap_ci([0.1, 0.5, 0.9], n_boot=100, seed=17)
        second = bootstrap_ci([0.1, 0.5, 0.9], n_boot=100, seed=17)
        self.assertEqual(first, second)

    def test_paired_test_truncates_to_shorter_input(self):
        actual = paired_ttest([1.0, 2.0, 50.0], [0.0, 0.0])
        expected = stats.ttest_rel([1.0, 2.0], [0.0, 0.0])
        self.assertEqual(actual, (float(expected.statistic), float(expected.pvalue)))

    def test_original_paired_test_returns_neutral_result_for_one_pair(self):
        self.assertEqual(paired_ttest([1.0], [0.0]), (0.0, 1.0))

    def test_original_effect_size_accepts_one_pair(self):
        expected = 1.0 / math.sqrt(1e-9)
        self.assertAlmostEqual(cohens_d([1.0], [0.0]), expected)

    def test_paired_test_preserves_scipy_constant_difference_behavior(self):
        statistic, p_value = paired_ttest([1.0, 1.0], [1.0, 1.0])
        self.assertTrue(math.isnan(statistic))
        self.assertTrue(math.isnan(p_value))

    def test_paired_test_uses_paired_observations(self):
        statistic, p_value = paired_ttest([1.0, 4.0, 9.0, 16.0, 25.0], [0.0, 1.0, 3.0, 6.0, 10.0])
        self.assertGreater(statistic, 0)
        self.assertLess(p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
