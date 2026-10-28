---
slug: generation-microbatch-five
title: "Eval generates about 5 prompts per call: chunks are re-bucketed by exact length"
issue: 229
severity: P3
labels: [enhancement, P3-minor, area: evaluation, tech-debt, audit-2026-10-27]
nf: [NF-42]
found_by: [code-eval]
status: Demonstrated
related_issues: [#172, #144, #142]
---

## Summary

`evaluate_condition` slices a condition's examples into chunks of `batch_size`, and `generate_batch` then splits each chunk by exact prompt token length and again by `batch_size`. Each task has four or five prompt templates with different lengths, so a chunk of 32 becomes 4–10 micro-batches. In the oracle run at `batch_size=32`, 4,100 prompts took 851 `generate` calls (mean 4.8 prompts per call; 178 calls of size 1). The in-training eval hard-codes `batch_size=8`, so its calls are smaller still. Per-call overhead dominates eval time at this size.

## What happens

- [utils.py:1172-1173](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1172-L1173): `for start_idx in range(0, n, batch_size): batch = dataset[start_idx : start_idx + batch_size]`.
- [utils.py:1056-1061](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1056-L1061): inside `generate_batch`, "Bucket by exact sequence length so we can batch prompts without padding"; [utils.py:1074-1077](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1074-L1077): each bucket is chunked by `batch_size` again and sent to `model.generate`.
- [train.py:627](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L627): the training-time path calls `run_algorithmic_suite(..., batch_size=8, ...)`.
- Audit probe `code-eval/oracle_probe.py` at `batch_size=32`: generate-call size histogram `{1: 178, 2: 136, 3: 101, 4: 91, 5: 64, 6: 48, 7: 55, 8: 44, 9: 26, 10: 26, 11: 29, …, 24: 1}`; 851 calls for 4,100 prompts.

## Why it matters for the experiments

- Eval time is paid per micro-batch: DAG-v12 spent 1.30 h of 4.83 h in 77 evals (35–74 s each, 27.6 % of train+eval time) at `eval_samples 80`; the owner's "eval takes 20 min!" at `eval_samples 256` is the same cost. Fewer, larger calls would cut it several times without changing any number.
- Smaller eval budgets were chosen to keep eval time down (80 examples per task in DAG-v12), which is where the ±5–10-point sampling noise on IID accuracy comes from (#161 comment on parity at 16 samples).
- Results do not change: exact-length buckets are padding-free, so grouping across chunks is numerically identical.

## How it got here

- PR [#19](https://github.com/jpturcotte/ReLSM/pull/19) (2025-12-12, "Add unified evaluation hub for modular tasks", commit `d8240df`) wrote the `range(0, n, batch_size)` chunking in `evaluate_condition`.
- PR [#23](https://github.com/jpturcotte/ReLSM/pull/23) (2025-12-13, "Avoid padding in batch generation"; task: "No padding in eval context; bucket by exact prompt length; regression test") added the per-chunk exact-length buckets inside `generate_batch` (commit `737f7b1`). Testing: "pytest -q (fails: No module named torch)".
- PR [#83](https://github.com/jpturcotte/ReLSM/pull/83) (2025-12-20) rewrote the decode loop and kept both levels of chunking.
- The Dec-19 ChatGPT chat's "Consider left-padding for batch generation" (claude-b1, Fix 3) was dropped.

## How we verified it, step by step

1. code-eval read `evaluate_condition` and `generate_batch` (E3).
2. Audit probe `code-eval/oracle_probe.py` wrapped `model.generate` with a counter and ran the full grid (4,100 prompts) through the real path on CPU at `batch_size=32`: 851 calls, histogram above (`oracle_out.txt` line 47).
3. artifacts probe `artifacts/p2_timeline.py` summed DAG-v12's `eval_duration`: 1.30 h over 77 evals, mean 60.8 s.
4. Disagreements: none.

## Suggested fix

- In `evaluate_condition`, tokenize and sort all `n` prompts by length once, then chunk by `batch_size` within each length group (or pass the whole condition to `generate_batch` and let it bucket once). Expected 5–8× fewer calls at `n = 200`, no numerical change.
- Alternatively left-pad with an attention mask; that path has #172 (cached decoding drops the padding mask), so fix #172 first if this route is taken.
- Use the same `batch_size` in the training-time path as in `eval_hub` (see #227).

## Acceptance test

CPU, tiny model: build a 2-layer `d_model=64` baseline, wrap `model.generate` with a call counter, run `evaluate_condition` on one condition with `n=200`, `batch_size=32`. Assert `calls <= number_of_distinct_prompt_lengths + 200 // 32` (about 12 for a 5-template task). At HEAD the count is about 40–50.

## Related

- #172: padding mask dropped for past tokens in cached decoding (blocks the left-padding variant).
- #144: oracle regression test; the oracle probe is where the call count was measured.
- #142: per-condition `max_new_tokens`; the same loop.
- #227: the hard-coded `batch_size=8` in the training path.

## Sources

- Audit report code-eval, finding E3; `oracle_probe.py`, `oracle_out.txt`
- Audit report artifacts, AR-7 and `p2_out.txt` A2 (eval time share)
- Audit report claude-b1, avenue 2 ("eval takes 20 min!")
- Audit report codex-a, task table row 23
- Merged findings NF-42

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `eval/generation-microbatch-five`.*
