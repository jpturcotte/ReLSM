import pytest

pytest.importorskip("torch")

from utils import normalize_prediction, normalize_target


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
    assert normalize_prediction("addition", "the answer is 42.0!") == "42.0"
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
