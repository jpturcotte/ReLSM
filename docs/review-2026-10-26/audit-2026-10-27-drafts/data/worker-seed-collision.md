---
slug: worker-seed-collision
title: Worker RNG seeds collide across runs with adjacent --seed values
issue: 224
severity: P3
labels: [bug, P3-minor, area: data, reproducibility, audit-2026-10-27]
nf: [NF-40]
found_by: [code-data]
status: Plausible
related_issues: [#140, #154, #141]
---

## Summary

`AlgorithmicDataset.__iter__` seeds each worker with `random.Random(self.seed + worker_id)`. A run with `--seed 43` gives its worker 0 the stream that `--seed 42` gave its worker 1. With the default `--num_workers 4`, seeds 42 and 43 share three of their four worker streams; seeds 42 and 45 share one. Two "different seed" runs are not independent. Not executed by the audit (memory limit); the arithmetic is read from the code.

## What happens

- [data.py:1069-1074](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1069-L1074): `seed = self.seed + (worker_info.id if worker_info is not None else 0)`; the unseeded branch adds `worker_info.id` to a `SystemRandom` draw. [data.py:1076](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1076): `rng = random.Random(seed)`.
- [data.py:1088](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1088): each worker loops `range(worker_id, num_examples, num_workers)`, the same number of iterations per worker; task and difficulty draws come from `rng` when there is no `difficulty_fn`. Two workers with the same `rng` seed therefore produce the same example sequence (the per-worker progress is pinned near 0 with workers, #141).
- `train.py` passes `seed=args.seed` ([train.py:2207](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2207)); the default `--num_workers` is 4 ([train.py:3888](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3888)).

## Why it matters for the experiments

- A seed sweep (42, 43, 44, …) with workers shares most of its data between neighbours: for 4 workers, seeds s and s+1 share 3 streams, s and s+2 share 2, s and s+3 share 1. Variance across such seeds under-states the true seed variance.
- DAG-v12 ran `num_workers 0` and is not affected. The Jan-11 command used `--num_workers 4`, and 4 is the default.
- Under the task curriculum the jitter and easy-mix draws use the global `random` (gh-prs-b A-6), so the shared part is the task and operand stream, not the jitter.

## How it got here

- PR [#5](https://github.com/jpturcotte/ReLSM/pull/5) (2025-12-10, "Cache algorithmic generator map for iterable dataset", commit `aac7b60`) introduced `seed + worker_info.id` while fixing the per-iterator replay described in its prompt (#140 quotes it). No later PR changed the derivation.

## How we verified it, step by step

1. code-data read `__iter__` and the per-worker loop (D12). Not run: the coordinator's memory limit ruled out a multi-worker DataLoader on the audit machine.
2. The arithmetic: `43 + 0 == 42 + 1`, `43 + 1 == 42 + 2`, `43 + 2 == 42 + 3`. The code-data report wrote "a quarter of their data"; with 4 workers it is three of four streams for adjacent seeds, one of four for seeds three apart.
3. A probe that settles it: construct `AlgorithmicDataset(seed=42)` and `AlgorithmicDataset(seed=43)`, iterate each under a patched `torch.utils.data.get_worker_info` returning `id=1, num_workers=4` and `id=0, num_workers=4` respectively, and hash the first 200 examples' texts. The expected result at HEAD is identical lists.
4. Disagreements: none.

## Suggested fix

- Derive the worker seed from a hash of `(seed, worker_id)`, for example `int.from_bytes(hashlib.sha256(f"{seed}:{worker_id}".encode()).digest()[:8], "little")`, or use `seed * 1000 + worker_id`.
- Do this inside whatever fix #140 chooses for per-epoch seeds (the epoch index belongs in the same hash), and save it with the RNG state for #154.

## Acceptance test

CPU, no model:

```python
import random
import torch.utils.data as tud
from unittest import mock
from data import AlgorithmicDataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")  # HF_HUB_OFFLINE=1
def first_texts(seed, wid):
    ds = AlgorithmicDataset(tokenizer=tok, num_examples=400, max_seq_len=48, base_batch_size=8, seed=seed)
    info = mock.Mock(id=wid, num_workers=4, seed=0)
    with mock.patch.object(tud, "get_worker_info", return_value=info):
        out = []
        for batch in ds:
            out += [tuple(ex["input_ids"].tolist()) for ex in batch]
            if len(out) >= 50: break
    return out[:50]
assert first_texts(42, 1) != first_texts(43, 0)
```

At HEAD the two lists are equal.

## Related

- #140: per-iterator reseeding; the fix should hash seed, worker id and epoch together.
- #154: resume drops the RNG state.
- #141: with workers the per-worker progress stays near 0, which is why the shared streams stay identical.

## Sources

- Audit report code-data, finding D12 and section 5 (not run)
- Audit report gh-prs-b, A-6 (global `random` in the jitter path)
- Merged findings NF-40

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `data/worker-seed-collision`.*
