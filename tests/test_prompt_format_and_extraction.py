import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import AlgorithmicGenerator
from utils import extract_prediction_from_generate


test_rng = random.Random(0)


class DummyTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, list):
            token_list = tokens
        else:
            token_list = tokens.tolist()
        cleaned = []
        for token in token_list:
            if skip_special_tokens and token in (self.pad_token_id, self.eos_token_id):
                continue
            cleaned.append(str(token))
        return " ".join(cleaned).strip()


def test_algorithmic_prompts_use_answer_prefix():
    for gen in (AlgorithmicGenerator.copy_sequence, AlgorithmicGenerator.reverse_sequence):
        example = gen(rng=test_rng, length=4)
        assert example["input"].endswith("Answer: ")
        assert example["text"] == f"{example['input']}{example['target']}"
        assert "Answer:" not in example["target"]


def test_extract_prediction_from_generate_handles_eos():
    tokenizer = DummyTokenizer()
    prompt_ids = [11, 12]
    answer_ids = [21, 22]
    generated_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]

    prediction, continuation, stop_reason, first_token_is_eos = extract_prediction_from_generate(
        prompt_ids,
        generated_ids,
        tokenizer.eos_token_id,
        tokenizer,
    )

    assert prediction == "21 22"
    assert continuation == answer_ids
    assert stop_reason == "eos"
    assert first_token_is_eos is False

    eos_only_ids = prompt_ids + [tokenizer.eos_token_id]
    prediction, continuation, stop_reason, first_token_is_eos = extract_prediction_from_generate(
        prompt_ids,
        eos_only_ids,
        tokenizer.eos_token_id,
        tokenizer,
    )

    assert prediction == ""
    assert continuation == []
    assert stop_reason == "eos"
    assert first_token_is_eos is True
