---
slug: token-budgets-count-supervised-tokens-only
title: "Token budgets, LR progress and Tok/s count supervised answer tokens only"
issue: 217
severity: P2
labels: [bug, P2-important, area: training, reproducibility, baseline-affected, audit-2026-10-27]
nf: [NF-09]
found_by: [cli-commands, codex-b, gemini-c, artifacts, code-train-b, claude-c1]
status: Demonstrated
related_issues: [#169, #187, #157, #174, #159, #149, #139, #189]
---

## Summary

`tokens_seen` advances by `(labels[..., 1:] != -100).sum()`. Since PR #80 that excludes padding and since PR #110 it also excludes the prompt, so the counter is answer tokens only. Everything downstream reads it: `--alg_tokens`, `--total_tokens`, `--mix_band_tokens`, `--warmup_tokens`, the LR progress (warmup and the WSD hold), the difficulty schedule, the phase switch, and the printed "Progress" and "Tok/s". The CLI help says "Tokens for Phase 1" and "Total token budget"; the README writes "100M tokens" with no qualifier. At the defaults a micro-batch holds about 8,200 tensor tokens and about 1,030 counted ones (12.5%). DAG-v12's "100M" budget was about 317M full-sequence tokens. Runs before and after 2025-12-25 at the same `--alg_tokens` differ 3–4× in compute and schedule length, and no issue states the unit.

## What happens

- [data.py:1134-1135](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1134-L1135) (per worker) and [data.py:1773-1774](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1773-L1774) (per batch): `n_tokens = (labels[..., 1:] != -100).sum()`; prompt positions are `-100` from [data.py:1130-1132](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1130-L1132).
- Consumers: [train.py:2630](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2630) (`while curriculum.tokens_seen < args.total_tokens`), [train.py:2645](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2645) and [train.py:2669](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2669) (`get_lr`), [train.py:309](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L309) (`progress = tokens_seen / total_tokens`), [train.py:1850](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1850) (difficulty), [train.py:2756](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2756) (Tok/s), [train.py:2763](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2763) (Progress).
- Help text: [train.py:3785-3788](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3785-L3788). README: `--alg_tokens 10000000 --total_tokens 20000000` in the quick start and "100M tokens algorithmic, then 400M tokens language" ([README.md:124](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/README.md#L124)).

Measured:

| source | setting | result |
|---|---|---|
| cli-commands F04 (`cli-commands/flag_semantics.py`) | defaults 64 × 128, Llemma | 216–241 examples per micro-batch, 8,194–8,232 tensor tokens, 891–1,088 counted; 4.6 counted tokens per example; 12.5% |
| codex-b P1 (`codex-b/probe_codex_b.py`) | `alg_seq_len 48`, 1,500 examples | counted/processed 0.23 (d 0.2), 0.28 (d 0.5), 0.23 (d 1.0); padded tokens per counted token 4.5–5.8 |
| gemini-c GC-1 (post-Jan-11 log) | 72 × 48, step 11,000 | `pct masked 0.774`; 11,000 × 72 × 48 × 0.226 = 8.6M = 8.6% of 100M, matching the printed "Progress 8.5%"; raw counting would read 38% |
| artifacts §2b (`artifacts/p2_timeline.py`, DAG-v12) | 96 × 48, accum 12 | `tokens_seen` 100.0M vs Σ(count × standalone target length) 108.8M (+EOS 118.0M); full-sequence tokens processed ≈ 317M, 3.2× the budget |
| code-train-b | DAG-v12 `pct_masked_tokens` 0.63 | ≈ 2.7× (same fact at lower precision) |

Task mix changes the unit: compare, parity and dyck answers are 1–2 tokens; copy and reverse 9–21. Equal `--alg_tokens` is not equal compute when the mix differs.

## Why it matters for the experiments

- DAG-v12 and the Dec 17–24 runs are not on the same budget: before PR #110 the counter included the prompt (and before PR #80, padding). A "100M" run after Dec 25 processes 3–4× the positions of a "100M" run before it, with a 3–4× longer schedule.
- The WSD hold (40% of `total_tokens`), the 2% warmup and the difficulty schedule are all stretched by the masking ratio, which moves with difficulty (longer prompts, fewer counted tokens per position).
- README "~10 min" and "100M tokens" are in an unstated unit; the quick start's 10M is about 80M processed tokens. `summary.avg_tokens_per_sec` (10,259 for DAG-v12) is answer tokens per second.
- No issue states this. #169 says "answer tokens/s" in passing, #159 notes it for the language phase only, #174 is about loss weighting.

## How it got here

- PR [#80](https://github.com/jpturcotte/ReLSM/pull/80) (2025-12-20, "Use HF-style labels, fix token counting") made the count `labels[..., 1:] != -100`, excluding padding. Testing: pytest 1 skipped (no torch).
- PR [#110](https://github.com/jpturcotte/ReLSM/pull/110) (2025-12-25, "Mask prompt tokens in algorithmic datasets"): owner: "the model is forced to predict random prompt tokens (irreducible loss)"; `labels[:prompt_len] = -100`. From this merge the same counter is answer-only. Testing: "not run". Neither PR changed the help text, the README or the step estimate (see #248).
- The project's own "100M supervised tokens" phrasing (`project-history-chatgpt.md`) shows the ambiguity was felt but never written into the code.

## How we verified it, step by step

1. cli-commands (C1) ran `flag_semantics.py` with the real tokenizer at the defaults and at 4 × 64, printing examples, tensor tokens and counted tokens per batch (F04); grepped the help text and README.
2. codex-b (N1) measured counted/processed ratios at three difficulties with `alg_seq_len 48`.
3. gemini-c (GC-1) reproduced the post-Jan-11 log's "Progress 8.5%" from `pct masked 0.774` with answer-only counting and showed raw counting would give 38%.
4. artifacts (§2b) cross-checked DAG-v12's `tokens_seen` against the per-window example counts and target lengths in `metrics.0.json` (108.8M / 118.0M vs 100.0M) and derived ≈ 317M full-sequence tokens; code-train-b got ≈ 2.7× from `pct_masked_tokens`.
5. The coordinator re-read `data.py:1134-1135` and `1773-1774` and confirmed.

## Suggested fix

- Name the quantity in the help text and README ("supervised answer tokens"), or rename the flags (`--alg_supervised_tokens`).
- Count both: add a raw (non-pad) token counter next to `tokens_seen`, log `train/throughput/tokens_seen_raw` and `tokens_per_sec_raw`, and store both in the run record (#187) and in `summary.json`.
- Express budgets, intervals and cooldowns in optimizer steps as #157 already proposes; log examples per batch (#139).
- Keep the schedule's unit explicit in the checkpoint so pre- and post-Dec-25 runs can be compared. Separate from #248, which is the stale constant on top of this unit.

## Acceptance test

CPU, Llemma tokenizer, one default micro-batch from `AlgorithmicDataset` through `CurriculumSampler.next_batch`: the sampler exposes `tokens_seen` and `tokens_seen_raw`; `tokens_seen` equals `(labels[..., 1:] != -100).sum()` and `tokens_seen_raw` equals `(input_ids != pad_id).sum()`, and the help string for `--alg_tokens` contains the word "supervised".

## Related

- #169: README claims; the unit belongs there.
- #187: run record should carry both counters.
- #157: budgets in optimizer steps.
- #174: loss weighting by supervised tokens per micro-batch.
- #159: the language phase counts every token, so one budget mixes two units.
- #149: the x-axis of every telemetry series is this counter.
- #139: log realised examples and tokens per batch.
- #189: the run ledger needs the unit per run.
- #248: the 40-token constant, a separate defect that uses this counter.

## Sources

- Audit report cli-commands, finding C1; probe `cli-commands/flag_semantics.py` (F04)
- Audit report codex-b, finding N1; probe `codex-b/probe_codex_b.py` (P1)
- Audit report gemini-c, finding GC-1 (log G:f5bb lines 12, 31)
- Audit report artifacts, §2b; probe `artifacts/p2_timeline.py`
- Audit report code-train-b ("Checks the brief asked for": `tokens_seen` and padding); claude-c1 (#189 note)
- VERIFIED_BY_COORDINATOR.md (`tokens_seen` row)
