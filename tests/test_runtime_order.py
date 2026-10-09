import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runtime import build_training_runtime


class RuntimeOrderTests(unittest.TestCase):
    def test_training_runtime_loads_model_before_dataset(self):
        sequence = []
        model = object()
        splits = ({}, {}, {})

        def load_model(config, logger):
            sequence.append("model")
            return model

        def load_data(config, logger):
            sequence.append("data")
            return splits

        with patch("runtime.MergeableCausalLM.from_pretrained", side_effect=load_model), patch(
            "runtime.load_data_splits", side_effect=load_data
        ):
            result = build_training_runtime(SimpleNamespace(), object())

        self.assertEqual(sequence, ["model", "data"])
        self.assertEqual(result, (model, *splits))


if __name__ == "__main__":
    unittest.main()
