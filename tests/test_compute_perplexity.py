import math
import torch

from utils import compute_perplexity


class ToyConfig:
    def __init__(self, vocab_size: int = 5, max_seq_len: int = 16):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ToyConfig()

    def forward(self, input_ids, labels=None, **_):
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size
        logits = input_ids.new_zeros((batch_size, seq_len, vocab_size), dtype=torch.float)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, None


class ToyTokenizer:
    pad_token_id = 0

    def __call__(self, text, max_length=None, truncation=True, return_tensors=None):
        _ = (max_length, truncation, return_tensors)
        # Encode words starting at token ID 1 to keep pad_token_id unused.
        tokens = [idx + 1 for idx, _ in enumerate(text.split())]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_compute_perplexity_returns_finite_value():
    model = ToyModel()
    tokenizer = ToyTokenizer()

    ppl = compute_perplexity(model, tokenizer, ["hello world"], torch.device("cpu"))

    assert isinstance(ppl, float)
    assert math.isfinite(ppl)
