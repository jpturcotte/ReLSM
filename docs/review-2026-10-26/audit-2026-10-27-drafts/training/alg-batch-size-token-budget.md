---
slug: alg-batch-size-token-budget
title: "--alg_batch_size is a token budget: 106-151 examples per micro-step in DAG-v12"
issue: 213
severity: P2
labels: [bug, P2-important, area: training, area: data, reproducibility, telemetry, baseline-affected, audit-2026-10-27]
nf: [NF-50]
found_by: [artifacts, cli-commands, code-data]
status: Demonstrated
related_issues: [#139, #174, #187, #136, #159, #141]
---

## Summary

In the procedural path `--alg_batch_size` is not a batch size. `AlgorithmicDataset` yields a length bucket when it holds `base_batch_size × 4` examples or when `count × max_len_in_bucket >= base_batch_size × max_seq_len` tokens. In DAG-v12 (96 × 48 = 4,608 tokens) that gave 151 examples per micro-step early and 106 late: 9,162,981 examples over 77,500 micro-steps, 1.23× the nominal 7,440,000; per optimizer update 1,810 examples early and 1,270 late, with 10k–17k supervised tokens per update, drifting with the curriculum. At the defaults (64 × 128) a micro-step is 196–241 examples. Under `--fixed_data` the same flag is a literal `DataLoader(batch_size=64)`. "Batch 96 × accumulation 12" describes no update that happened.

## What happens

- [data.py:1028-1029](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1028-L1029): `_target_batch_tokens = base_batch_size * max_seq_len`; `_max_batch_size = base_batch_size * max_batch_multiplier` (default 4). [data.py:1031-1034](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1031-L1034): three length buckets at thirds of `max_seq_len` (16 and 32 tokens at 48).
- [data.py:1145-1158](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1145-L1158): a bucket is yielded when `current_batch_size >= _max_batch_size` or `current_batch_size * buffer_max_len >= _target_batch_tokens`.
- [train.py:2205](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2205): `base_batch_size=args.alg_batch_size` for the procedural dataset; [train.py:2194](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2194): `DataLoader(alg_dataset, batch_size=args.alg_batch_size, ...)` under `--fixed_data`. The flag has no help text ([train.py:3885](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3885)).
- DAG-v12 metrics (`train.task_count_window.*`, artifacts AR-4): examples per micro-step by 5k-step bin 151, 148, 148, 117, 120, 116, 113, 111, 110, 110, 108, 107, 107, 106, 106, 106; supervised tokens per micro-step 985 early, 858–861 at 5–15k, then 1,269–1,439. Per optimizer update (12 micro-steps): 1,810 → 1,270 examples; 11.8k, 10.3k, then 16.2k–17.3k supervised tokens. Total examples 9,162,981 (1.23× nominal); the 100,000-example iterator restarted about 92 times.
- Audit probe `code-data/probe_batching.py` at 64 × 48 (4,000 examples, CPU, real tokenizer): batches of 1–192 examples (median 96), 18–1,065 supervised tokens, 20 % padding; at 64 × 128: 5–256 examples (median 196), 39 % padding. The long bucket at 48 tokens held only copy, reverse, successor and chain (plus 7 parity of 897); cli-commands' `flag_semantics` run at the defaults: 196–241 examples per micro-batch.

## Why it matters for the experiments

- The number of examples per update is a first-class variable in a grokking study. In DAG-v12 it drifted by 30 % and the supervised tokens per update by 70 % as the curriculum moved; neither is in the run record (#187) or the run name.
- Each task's share of an update depends on which bucket its lengths fall in: short tasks arrive in batches of up to 288, 10–12-item copy/reverse in batches of 96–135. This is the upstream cause of #174's unequal micro-batches and an unintended task reweighting.
- A `--fixed_data` run and a procedural run "at the same batch size" differ 3–4× in examples per step (64 vs 196–241 at the defaults), so they are not a controlled pair.
- Raising `--alg_seq_len` from 48 to 128 doubles the examples per micro-step (#139 comment 2): memory, effective batch and the number of updates per token budget all change at once.

## How it got here

- PR [#120](https://github.com/jpturcotte/ReLSM/pull/120) (2025-12-29, "Token-based dynamic batching for AlgorithmicDataset and DataLoader adjustments"): yield when the buffer reaches `batch_size × max_seq_len` tokens, cap 4× (`max_batch_multiplier`); DataLoaders switched to `batch_size=None`. Testing: "Not run".
- PR [#121](https://github.com/jpturcotte/ReLSM/pull/121) (2025-12-29, "Add length-based bucketing"): three length buckets, after the owner pasted "tokens/sec 2820 → 388, VRAM 5.75 → 7.39 GB". Testing: "Not run".
- PR [#122](https://github.com/jpturcotte/ReLSM/pull/122) (2025-12-29): the flush rule changed to padded size (`buffer_max_len`). DAG-v12 started the same evening on `188439f`, the #122 merge.
- The flag name and the `--fixed_data` `DataLoader(batch_size=...)` date from the first commit and were not revisited.

## How we verified it, step by step

1. artifacts read every row of DAG-v12's `metrics.0.json` with `artifacts/p2_timeline.py` and summed `train.task_count_window.*` per 5k-step bin (`p2_out.txt` A1: examples per micro-step, tokens per micro-step, totals, the 1.23 ratio and the ~92 iterator restarts).
2. code-data iterated `AlgorithmicDataset` on CPU with the real tokenizer for 4,000 examples at three configurations (`code-data/probe_batching.py`): examples per batch, supervised tokens, padding fraction and the per-bucket task mix.
3. cli-commands ran `cli-commands/flag_semantics.py` at the defaults (64 × 128): 196–241 examples per micro-batch (F04), and read `train.py:2194` for the fixed path (C6).
4. The three measurements agree with the code's yield rule (`96 × 48 = 4,608` tokens; `64 × 48 = 3,072`; cap 4×).
5. Disagreements: none. artifacts rates it P2; cli-commands rated its C6 row P3 before the artifact measurement; the merged table takes P2 because the drift is measured on the only surviving run.

## Suggested fix

- Log `len(batch["task"])`, the supervised token count and the bucket index per micro-step, and examples and tokens per optimizer update at each log interval (free; the counters exist).
- Separate the three uses of `alg_seq_len`: truncation limit, batch token budget (`--alg_batch_tokens`) and bucket boundaries (#139 comment 2), and offer a fixed-example mode (`--alg_batch_examples N` with padding) so updates are comparable across difficulty.
- Give the `--fixed_data` loader the same semantics as the procedural path, or name the flag so the two meanings cannot be confused. This is part of the `--fixed_data` contract draft #215 (NF-12) and should land with it.
- Record the chosen mode and the realised examples per update in the run record (#187).

## Acceptance test

CPU, no model:

```python
from data import AlgorithmicDataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")  # HF_HUB_OFFLINE=1
ds = AlgorithmicDataset(tokenizer=tok, num_examples=4000, max_seq_len=48, base_batch_size=64, seed=42)  # plus the new fixed-example option
sizes = [len(b) for b in ds]
assert len(set(sizes[:-1])) == 1 and sizes[0] == 64, (min(sizes), max(sizes))
```

At HEAD `sizes` ranges from 1 to 192 with median 96. For the token-budget mode, the check is instead that each logged micro-step row contains `train/batch/examples` and `train/batch/sup_tokens`.

## Related

- #139: `alg_seq_len` is also the batch token budget (comment 2 proposes the split).
- #174: loss weighting across unequal micro-batches; this is the upstream cause.
- #187: run arguments and realised batch should be in the checkpoint record.
- #136: the `--fixed_data` path; same flag, different meaning.
- #159: the language loader counts every token, the algorithmic loader counts answer tokens; the token budget mixes units.
- #141: the per-worker budget uses the same `tokens_seen`.

## Sources

- Audit report artifacts, finding AR-4; `p2_timeline.py`, `p2_out.txt` A1
- Audit report cli-commands, finding C6 and F04 (`flag_semantics.py`)
- Audit report code-data, section 2 "Batching"; `probe_batching.py`, `probe_batching.out.txt`
- Audit report codex-b, PR #120–#122 paragraph; gh-prs-b PR #120–#122 rows
- Merged findings NF-50; extension rows for #139 and #174

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `training/alg-batch-size-token-budget`.*
