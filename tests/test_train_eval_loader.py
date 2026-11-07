"""The training-split eval loader for --fixed_data (issue #136).

The fixed set is map-style (one example dict per index). With batch_size=None the collator got one
dict and iterated its string keys: "TypeError: string indices must be integers". The procedural set
yields whole batches, so it keeps batch_size=None.
"""
from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import Subset

from data import AlgorithmicDataset, FixedAlgorithmicDataset, make_collate_fn
from train import _iter_eval_examples, build_train_eval_loader, evaluate_training_split


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(ch) % 200) for ch in text]


class TinyModel(torch.nn.Module):
    def __init__(self, vocab=256):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, vocab)

    def forward(self, input_ids, labels=None):
        assert not torch.is_grad_enabled(), "evaluate_training_split must run without autograd"
        logits = self.emb(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return logits, loss, None


def _fixed_set(n):
    return FixedAlgorithmicDataset(
        tokenizer=CharTokenizer(), num_examples=n, max_seq_len=64, tasks=["addition", "copy"],
        seed=0, include_lengths=True,
    )


def test_fixed_train_eval_loader_batches_map_style_examples():
    tok = CharTokenizer()
    dataset = _fixed_set(10)
    capped = Subset(dataset, range(7))  # the --train_eval_samples cap
    loader = build_train_eval_loader(
        capped, "fixed", batch_size=4, num_workers=0, collate_fn=make_collate_fn(tok)
    )

    batches = list(loader)
    assert [b["input_ids"].shape[0] for b in batches] == [4, 3]

    examples = list(_iter_eval_examples(loader, max_samples=None))
    assert [ex["task"] for ex in examples] == [dataset[i]["task"] for i in range(7)]  # in order
    for i, ex in enumerate(examples):
        n = dataset[i]["input_ids"].numel()
        assert torch.equal(ex["input_ids"][:n], dataset[i]["input_ids"])
        assert torch.equal(ex["labels"][:n], dataset[i]["labels"])
        assert (ex["labels"][n:] == -100).all()

    metrics = evaluate_training_split(TinyModel(), loader, torch.device("cpu"), nullcontext())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0


def test_procedural_train_eval_loader_keeps_dataset_batches():
    tok = CharTokenizer()

    def fresh():  # a new copy each time: tokens_seen carries across passes and moves difficulty
        return AlgorithmicDataset(
            tokenizer=tok, num_examples=12, max_seq_len=64, base_batch_size=4,
            tasks=["addition", "copy"], seed=0, include_lengths=True,
        )

    loader = build_train_eval_loader(
        fresh(), "procedural", batch_size=128, num_workers=0, collate_fn=make_collate_fn(tok)
    )
    assert loader.batch_size is None
    sizes = [b["input_ids"].shape[0] for b in loader]
    assert sizes == [len(batch) for batch in fresh()]
    assert sum(sizes) == 12
