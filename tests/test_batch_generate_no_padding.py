import pytest

torch = pytest.importorskip("torch")

from utils import batch_generate


class SimpleTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, truncation=True, return_tensors="pt"):
        # Encode each word as a monotonically increasing token ID so prompt
        # lengths differ without ever emitting the pad token ID.
        tokens = [idx + 5 for idx, _ in enumerate(text.split())]
        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return " ".join(str(t) for t in tokens if t != self.pad_token_id).strip()


class RecordingModel:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, **_: torch.Tensor):
        # Record the raw inputs to ensure no padding tokens are provided and
        # emit the final prompt token as the completion.
        self.calls.append(input_ids)
        completion = input_ids[:, -1:]
        return torch.cat([input_ids, completion], dim=1)


def test_batch_generate_skips_padding_tokens():
    tokenizer = SimpleTokenizer()
    model = RecordingModel()
    prompts = ["short", "a much longer prompt"]

    outputs = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=torch.device("cpu"),
        max_new_tokens=1,
    )

    # Completion is the final prompt token; padding would change this for the
    # shorter prompt if pad tokens were appended.
    assert outputs == ["5", "6"]

    for call in model.calls:
        assert torch.all(call != tokenizer.pad_token_id)
