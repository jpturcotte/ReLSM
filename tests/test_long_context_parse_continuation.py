"""Long-context answers are parsed from the continuation only.

Rewritten 2026-10-28 (#166): the original test imported eval_hub._prediction_from_output_ids,
which commit 17a72e8 removed, so the whole suite stopped at collection. The same properties
are now checked on the helpers the eval path uses today: utils.extract_prediction_from_generate
(continuation only, cut at EOS) and utils.extract_integer_token (number after "Answer:").
Whether long-context scoring applies the numeric parser is #163's question.
"""
import pytest

torch = pytest.importorskip("torch")

from utils import extract_integer_token, extract_prediction_from_generate  # noqa: E402


class DummyTokenizer:
    eos_token_id = None

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for tok in text.split():
            if tok not in self.token_to_id:
                idx = len(self.token_to_id) + 2
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
            ids.append(self.token_to_id[tok])
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        return " ".join(self.id_to_token[i] for i in list(ids))


def test_long_context_parsing_only_uses_continuation():
    tokenizer = DummyTokenizer()
    prompt = (
        "The passage includes a misleading Answer: marker that should be ignored. "
        "Pay attention to the final question."
    )
    prompt_ids = tokenizer.encode(prompt)
    output_ids = prompt_ids + tokenizer.encode("needle")

    prediction, continuation, stop_reason, _ = extract_prediction_from_generate(
        prompt_ids, output_ids, eos_token_id=None, tokenizer=tokenizer
    )
    assert prediction == "needle"
    assert continuation == tokenizer.encode("needle")
    assert stop_reason == "max_new_tokens"

    # Parsing the full decoded text instead would pick up the prompt's "Answer:" marker.
    full_tail = tokenizer.decode(output_ids).split("Answer:")[-1].strip().split()[0]
    assert full_tail != prediction


def test_long_context_parsing_prefers_numbers_in_tail():
    tokenizer = DummyTokenizer()
    prompt_ids = tokenizer.encode("Find the code.")
    output_ids = prompt_ids + tokenizer.encode("Answer: The code is 1234")

    prediction, _, _, _ = extract_prediction_from_generate(
        prompt_ids, output_ids, eos_token_id=None, tokenizer=tokenizer
    )
    assert prediction == "Answer: The code is 1234"
    assert extract_integer_token(prediction) == "1234"
