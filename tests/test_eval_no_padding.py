import torch

from utils import generate_batch


class DeterministicTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, truncation=True, return_tensors="pt"):
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 5
            tokens.append(self.vocab[word])
        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        cleaned = []
        for token in tokens:
            if skip_special_tokens and token in (self.pad_token_id, self.eos_token_id):
                continue
            cleaned.append(str(token))
        return " ".join(cleaned).strip()


class DummyModel:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def generate(self, input_ids, max_new_tokens=2, **_: torch.Tensor):
        last_token = input_ids[:, -1:]
        eos_column = torch.full_like(last_token, self.eos_token_id)
        continuation = torch.cat([last_token, eos_column], dim=1)
        continuation = continuation[:, :max_new_tokens]
        return torch.cat([input_ids, continuation], dim=1)


def test_generate_batch_avoids_padding_and_tracks_lengths():
    tokenizer = DeterministicTokenizer()
    model = DummyModel(eos_token_id=tokenizer.eos_token_id)
    device = torch.device("cpu")

    prompts = ["short", "a much longer prompt"]
    pair_preds, pair_lengths, _ = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=2,
        device=device,
        use_autocast=False,
        task="dummy",
    )

    solo_preds, solo_lengths, _ = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompts[0]],
        max_new_tokens=2,
        device=device,
        use_autocast=False,
        task="dummy",
    )

    assert pair_preds[0] == solo_preds[0]
    assert pair_lengths == [2, 2]
    assert solo_lengths == [2]

    total_generated_tokens = sum(pair_lengths)
    avg_gen_len = total_generated_tokens / len(prompts)
    assert total_generated_tokens == 4
    assert avg_gen_len == 2
