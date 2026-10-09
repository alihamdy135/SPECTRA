import logging
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from models.base import MergeableCausalLM


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    @property
    def shape(self):
        return self.value.shape

    @property
    def data(self):
        return self

    def clone(self):
        return Tensor(self.value.copy())

    def float(self):
        return self

    def cpu(self):
        return self

    def to(self, dtype):
        return self

    def copy_(self, other):
        self.value = other.value.copy()

    def __setitem__(self, item, value):
        self.value[item] = value

    def __ne__(self, other):
        return self.value != other

    def sum(self):
        return self.value.sum()

    def __getitem__(self, item):
        return self.value[item]

    def __add__(self, other):
        return Tensor(self.value + (other.value if isinstance(other, Tensor) else other))

    def __mul__(self, other):
        return Tensor(self.value * (other.value if isinstance(other, Tensor) else other))

    def __rmul__(self, other):
        return self * other


class Parameter:
    def __init__(self, value):
        self.data = Tensor(value)
        self.requires_grad = True


class ParameterModel:
    def __init__(self, value):
        self.parameter = Parameter(value)

    def named_parameters(self):
        return [("weight", self.parameter)]


class FakeTorch:
    float32 = object()

    @staticmethod
    def no_grad():
        return nullcontext()


class Encoding(dict):
    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 0
    pad_token = None
    eos_token = "eos"

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        if isinstance(text, list):
            length = 3
            rows = len(text)
        else:
            length = 3 if text.endswith("### A:\n") else 6
            if text.startswith("### Q:\ntruncated"):
                length = 3
            length = min(length, max_length) if max_length is not None else length
            rows = 1
        values = np.ones((rows, length))
        return Encoding(input_ids=Tensor(values), attention_mask=Tensor(values))

    def decode(self, sequence, skip_special_tokens=True):
        return f"prediction-{sequence[-1]}" if len(sequence) else ""


class ScoringModel:
    def __init__(self, fail_first_generation=False, fail_nll_calls=()):
        self.generation_calls = 0
        self.fail_first_generation = fail_first_generation
        self.fail_nll_calls = set(fail_nll_calls)
        self.nll_calls = 0

    def generate(self, **encoded):
        self.generation_calls += 1
        if self.fail_first_generation and self.generation_calls == 1:
            raise RuntimeError("generation error")
        if encoded.get("max_new_tokens", 1) < 1:
            raise RuntimeError("invalid generation length")
        return [[1, 1, 1, 99] for _ in range(len(encoded["input_ids"].value))]

    def __call__(self, input_ids, attention_mask=None, labels=None):
        self.nll_calls += 1
        if self.nll_calls in self.fail_nll_calls:
            raise RuntimeError("loss error")
        return SimpleNamespace(loss=SimpleNamespace(item=lambda: float(self.nll_calls)))


def make_scorer(model, tokenizer=None, batch_size=2):
    scorer = object.__new__(MergeableCausalLM)
    scorer.torch = FakeTorch
    scorer.model = model
    scorer.tokenizer = tokenizer or Tokenizer()
    scorer.config = SimpleNamespace(
        dtype="float32",
        generation_batch_size=batch_size,
        gen_max_new=8,
        max_seq_len=16,
    )
    scorer.device = "cpu"
    scorer.logger = logging.getLogger("test-scorer")
    return scorer


class ScoringFailureTests(unittest.TestCase):
    def test_generation_returns_blank_predictions_for_failed_batch(self):
        scorer = make_scorer(ScoringModel(fail_first_generation=True), batch_size=2)

        predictions = scorer.generate(["one", "two", "three"])

        self.assertEqual(predictions, ["", "", "prediction-99"])

    def test_generation_returns_blank_for_a_generation_error(self):
        scorer = make_scorer(ScoringModel())

        self.assertEqual(scorer.generate(["one"], max_new=0), [""])

    def test_answer_nll_skips_truncated_and_failed_examples(self):
        scorer = make_scorer(ScoringModel(fail_nll_calls=(1,)))

        value = scorer.nll_answer(
            [("truncated", "answer"), ("failed", "answer"), ("good", "answer"), ("good2", "answer")],
            max_len=6,
        )

        self.assertEqual(value, 2.5)

    def test_domain_nll_skips_failed_prompts_and_averages_successes(self):
        scorer = make_scorer(ScoringModel(fail_nll_calls=(1,)))

        value = scorer.domain_nll(["failed", "good", "good2"])

        self.assertEqual(value, 2.5)

    def test_coefficients_calculate_with_original_values_and_only_round_cache_key(self):
        model = object.__new__(MergeableCausalLM)
        model.torch = FakeTorch
        model.model = ParameterModel([0.0])
        model.config = SimpleNamespace(dtype="float32")
        model.base_weights = {"weight": Tensor([0.0])}
        model.current_key = None
        delta = Tensor([1.0])

        model.set_coefficients([0.1234567], {"math": {"weight": delta}}, ["math"])

        self.assertEqual(model.current_key, (0.123457,))
        self.assertAlmostEqual(model.model.parameter.data.value[0], 0.1234567)
        model.set_coefficients([0.1234568], {"math": {"weight": delta}}, ["math"])
        self.assertAlmostEqual(model.model.parameter.data.value[0], 0.1234567)

    def test_nll_returns_zero_when_no_examples_score(self):
        scorer = make_scorer(ScoringModel(fail_nll_calls=(1,)))

        self.assertEqual(scorer.nll_answer([("failed", "answer")]), 0.0)
