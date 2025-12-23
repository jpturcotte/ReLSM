import pytest

pytest.importorskip("torch")

from utils import (
    LABEL_TASKS,
    _is_classification_task,
    _is_numeric_task,
    normalize_label_prediction,
    normalize_prediction,
    normalize_target,
)


def test_normalize_prediction_strips_whitespace_and_punctuation():
    text = '  "hello!"  '\
        "\nignored trailing lines"

    assert normalize_prediction("copy", text) == "hello"


def test_normalize_prediction_standardizes_dyck_yes_no():
    yes_variants = ["YES", "Yes.", " yes\nrest"]
    no_variants = ["NO", "no!", " No \nrest"]

    for variant in yes_variants:
        assert normalize_prediction("dyck", variant) == "yes"

    for variant in no_variants:
        assert normalize_prediction("dyck", variant) == "no"


def test_normalize_prediction_extracts_numeric_token():
    assert normalize_prediction("addition", "the answer is 42.0!") == "42"
    assert normalize_prediction("addition", "Answer: 7") == "7"
    assert normalize_prediction("addition", "=> 7") == "7"
    assert normalize_prediction("addition", "= 7") == "7"
    assert normalize_prediction("addition", "7.") == "7"
    assert normalize_prediction("addition", "  7  ") == "7"
    assert normalize_prediction("addition", "The answer is 7") == "7"
    assert normalize_prediction("chain", "Result: -3 apples") == "-3"


def test_normalize_prediction_uses_last_number():
    assert normalize_prediction("addition", "3+4=7") == "7"
    assert normalize_prediction("addition", "I think 3 + 4 = 7") == "7"


def test_normalize_prediction_and_target_share_base_normalization():
    text = "  42.  "
    assert normalize_prediction("addition", text) == normalize_target("addition", text)


def test_normalize_prediction_preserves_operators_and_symbols():
    assert normalize_prediction("copy", "  ++value++  ") == "++value++"
    assert normalize_prediction("copy", " {brackets}!") == "{brackets}"


def test_normalize_prediction_dyck_label_scan():
    assert normalize_prediction("dyck", "Answer: yes") == "yes"
    assert normalize_prediction("dyck", "yes.") == "yes"
    assert normalize_prediction("dyck", "No") == "no"
    assert normalize_prediction("dyck", "nope") == ""


def test_normalize_label_prediction_scan():
    assert normalize_label_prediction("dyck", "Answer: yes") == "yes"
    assert normalize_label_prediction("parity", "Output = 1") == "1"
    assert normalize_label_prediction("compare", " => > ") == ">"


def test_task_type_consistency():
    assert _is_classification_task("parity")
    assert not _is_numeric_task("parity")
    assert "parity" in LABEL_TASKS


# Summary: normalize_prediction now scans for first integer/label tokens, parity is classification,
# and redundant unique_token_ratio metrics were removed to avoid confusion.
