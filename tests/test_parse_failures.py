import pytest

pytest.importorskip("torch")

from eval_hub import _algorithmic_results_to_dict
from utils import EvalResult


def test_algorithmic_results_include_parse_failures():
    result = EvalResult(
        condition="iid",
        task="addition",
        difficulty=0.5,
        accuracy=0.5,
        mean_token_accuracy=0.5,
        mean_distance=0.5,
        mean_prefix_accuracy=0.5,
        mae=None,
        numeric_count=0,
        parse_failures=2,
        n=4,
        avg_gen_len=0.0,
        tokens_per_sec=0.0,
        examples=[],
        sampled_examples=[],
        correct=2,
        tokens_generated=0,
        elapsed=0.0,
    )

    payload = _algorithmic_results_to_dict([result])
    assert payload["conditions"][0]["parse_failures"] == 2
    assert payload["conditions"][0]["parse_failure_rate"] == 0.5
    assert payload["overall_parse_failures"] == 2
    assert payload["overall_parse_failure_rate"] == 0.5
