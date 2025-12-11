import itertools

import pytest

try:  # noqa: SIM105 - require torch for these tests
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch not installed in lean envs
    pytest.skip("Skipping curriculum sampler tests because torch is not installed", allow_module_level=True)

from data import CurriculumSampler


class TinySeqDataset(torch.utils.data.Dataset):
    def __init__(self, values, seq_len: int = 2):
        self.values = list(values)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        value = torch.tensor(self.values[idx], dtype=torch.long)
        tokens = torch.full((self.seq_len,), value)
        return {"input_ids": tokens.clone(), "labels": tokens.clone()}


def _loader_from_values(values, *, batch_size: int = 1):
    dataset = TinySeqDataset(values)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _first_label_value(batch):
    return batch["labels"][0, 0].item()


def test_next_batch_recycles_iterators_without_stopiteration():
    alg_loader = _loader_from_values([1, 2, 3])
    lang_loader = _loader_from_values([101, 102])

    sampler = CurriculumSampler(
        alg_loader=alg_loader,
        lang_loader=lang_loader,
        total_tokens=100,
        alg_tokens=50,
        mix_band_tokens=0,
    )

    batches = [_first_label_value(sampler.next_batch()) for _ in range(5)]

    assert batches == [1, 2, 3, 1, 2]
    assert sampler.tokens_seen == 5 * 2  # two tokens per example


def test_next_batch_respects_batch_size_when_recycling():
    alg_loader = _loader_from_values([1, 2, 3, 4], batch_size=2)
    lang_loader = _loader_from_values([101], batch_size=2)

    sampler = CurriculumSampler(
        alg_loader=alg_loader,
        lang_loader=lang_loader,
        total_tokens=100,
        alg_tokens=100,
        mix_band_tokens=0,
    )

    sampler._sample_source = lambda: "alg"

    # Two batches from a batch_size=2, seq_len=2 dataset should each count 4 tokens.
    first_batch = sampler.next_batch()
    second_batch = sampler.next_batch()
    third_batch = sampler.next_batch()

    assert first_batch["labels"].shape == (2, 2)
    assert second_batch["labels"].shape == (2, 2)
    assert third_batch["labels"].shape == (2, 2)
    assert sampler.tokens_seen == 3 * 2 * 2
    assert _first_label_value(first_batch) == 1
    assert _first_label_value(second_batch) == 3
    # After recycling the iterator, we loop back to the start of the dataset.
    assert _first_label_value(third_batch) == 1


def test_reset_restores_loader_positions_and_token_counter():
    alg_loader = _loader_from_values([10, 20])
    lang_loader = _loader_from_values([30, 40])
    lex_loader = _loader_from_values([50])

    sampler = CurriculumSampler(
        alg_loader=alg_loader,
        lang_loader=lang_loader,
        total_tokens=100,
        alg_tokens=60,
        mix_band_tokens=0,
        lexical_frac_phase1=1.0,
        lex_loader=lex_loader,
    )

    first_cycle_sources = itertools.cycle(["alg", "lang", "lex"])
    sampler._sample_source = lambda: next(first_cycle_sources)

    first_batches = [_first_label_value(sampler.next_batch()) for _ in range(3)]
    assert sampler.tokens_seen == 3 * 2

    sampler.reset()
    assert sampler.tokens_seen == 0

    second_cycle_sources = itertools.cycle(["alg", "lang", "lex"])
    sampler._sample_source = lambda: next(second_cycle_sources)

    second_batches = [_first_label_value(sampler.next_batch()) for _ in range(3)]

    assert first_batches == [10, 30, 50]
    assert second_batches == [10, 30, 50]


def test_falls_back_to_language_when_lexical_loader_absent():
    alg_loader = _loader_from_values([5])
    lang_loader = _loader_from_values([101, 102, 103])

    sampler = CurriculumSampler(
        alg_loader=alg_loader,
        lang_loader=lang_loader,
        total_tokens=100,
        alg_tokens=0,
        mix_band_tokens=0,
        lex_loader=None,
    )

    # Force lexical choice even though no lexical loader is provided.
    sampler._sample_source = lambda: "lex"

    batches = [_first_label_value(sampler.next_batch()) for _ in range(4)]

    assert batches == [101, 102, 103, 101]
    # Each batch has batch_size=1 and seq_len=2, so tokens_seen is 8.
    assert sampler.tokens_seen == 4 * 2
