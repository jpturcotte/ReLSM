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


def _loader_from_values(values):
    dataset = TinySeqDataset(values)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


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
