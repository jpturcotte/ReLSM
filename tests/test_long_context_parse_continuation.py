import pytest

torch = pytest.importorskip("torch")

from eval_hub import _prediction_from_output_ids


class DummyTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def _tokenize(self, text: str):
        return text.split()

    def _get_id(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def encode(self, text: str) -> list[int]:
        return [self._get_id(tok) for tok in self._tokenize(text)]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        if isinstance(ids, tuple):
            ids = list(ids)
        return " ".join(self.id_to_token[i] for i in ids)


def test_long_context_parsing_only_uses_continuation():
    tokenizer = DummyTokenizer()
    prompt = (
        "The passage includes a misleading Answer: marker that should be ignored. "
        "Pay attention to the final question."
    )
    continuation = "needle"

    prompt_ids = tokenizer.encode(prompt)
    continuation_ids = tokenizer.encode(continuation)
    output_ids = prompt_ids + continuation_ids
    prompt_len = len(prompt_ids)

    parsed = _prediction_from_output_ids(tokenizer, output_ids, prompt_len)
    assert parsed == "needle"

    full_decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    if "Answer:" in full_decoded:
        old_tail = full_decoded.split("Answer:")[-1].strip().split()
    else:
        old_tail = full_decoded.split()
    old_pred = old_tail[0] if old_tail else ""
    assert old_pred != parsed
