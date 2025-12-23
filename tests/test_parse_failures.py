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
        target_len_tokens=[],
        pred_len_tokens=[],
        length_ratio=[],
        abs_len_error=[],
        stop_reason_counts={"eos": 0, "max_new_tokens": 0, "other": 0},
        eos_emitted=0,
        repeat_1gram_rate=[],
        repeat_2gram_rate=[],
        unique_token_fraction=[],
        max_run_length=[],
        parse_successes=0,
        parse_failure_counts={},
        numeric_abs_errors=[],
        numeric_rel_errors=[],
        empty_predictions=0,
        first_token_is_eos_count=0,
        invalid_label_count=0,
        label_confusion_counts={},
        format_stats={
            "colon": {"count": 0, "correct": 0, "empty": 0, "eos_first": 0},
            "answer": {"count": 0, "correct": 0, "empty": 0, "eos_first": 0},
            "arrow": {"count": 0, "correct": 0, "empty": 0, "eos_first": 0},
            "fatarrow": {"count": 0, "correct": 0, "empty": 0, "eos_first": 0},
            "equals": {"count": 0, "correct": 0, "empty": 0, "eos_first": 0},
        },
        numeric_length_mismatch_count=0,
        numeric_length_total=0,
    )

    payload = _algorithmic_results_to_dict([result])
    assert payload["conditions"][0]["parse_failures"] == 2
    assert payload["conditions"][0]["parse_failure_rate"] == 0.5
    assert payload["overall_parse_failures"] == 2
    assert payload["overall_parse_failure_rate"] == 0.5
