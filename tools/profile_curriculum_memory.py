"""Ad-hoc memory profiling helper for the curriculum sampler.

Run this script directly to sanity check that repeated calls to
``CurriculumSampler.next_batch`` do not accumulate memory. This uses the
standard-library ``tracemalloc`` module to avoid extra dependencies.
"""

import tracemalloc

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional helper
    raise SystemExit("torch is required to profile curriculum sampler memory usage")

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


def _loader_from_values(values, *, batch_size: int = 4):
    dataset = TinySeqDataset(values)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    alg_loader = _loader_from_values(range(16), batch_size=4)
    lang_loader = _loader_from_values(range(100, 110), batch_size=4)

    sampler = CurriculumSampler(
        alg_loader=alg_loader,
        lang_loader=lang_loader,
        total_tokens=10_000,
        alg_tokens=5_000,
        mix_band_tokens=0,
    )

    tracemalloc.start()
    start_current, start_peak = tracemalloc.get_traced_memory()

    for _ in range(1_000):
        sampler.next_batch()

    end_current, end_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    delta_current = end_current - start_current
    delta_peak = end_peak - start_peak

    print(f"Start current/peak: {start_current / 1_048_576:.3f} / {start_peak / 1_048_576:.3f} MiB")
    print(f"End current/peak:   {end_current / 1_048_576:.3f} / {end_peak / 1_048_576:.3f} MiB")
    print(f"Delta current:      {delta_current / 1_048_576:.3f} MiB")
    print(f"Delta peak:         {delta_peak / 1_048_576:.3f} MiB")

    # A generous threshold: growth beyond ~2 MiB would suggest accumulation.
    if delta_current > 2 * 1_048_576:
        raise SystemExit(
            "Observed more than ~2 MiB of retained growth; investigate potential caching."
        )


if __name__ == "__main__":
    main()
