---
slug: attn-entropy-over-padding
title: eval/attn/entropy is computed over padded probe positions, causal mask only
issue: 234
severity: P3
labels: [bug, P3-minor, area: tooling, telemetry, audit-2026-10-27]
nf: [NF-29]
found_by: [code-train-a, claude-b2, gh-prs-b, review-files]
status: Demonstrated
related_issues: [#184, #172, #149]
---

## Summary

The attention-entropy diagnostic recomputes the last block's attention on a right-padded probe batch with a causal mask only. 39 to 55 % of the probe positions are padding, pad queries are sampled like real ones, and pad and EOS keys are in every softmax, so the number is not the attention the model used (the model masks pad keys). On the same examples a tiny model gives 2.864 padded and 1.615 without padding. code-train-a rated this P2 (the only interpretability number in the log is mis-specified) and claude-b2 P3 (diagnostic only); this draft keeps P3 because no decision or reported result read the key and the fix is a few lines.

## What happens

- Probe batch: [train.py:2308-2329](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2308-L2329) builds a `FixedAlgorithmicDataset(num_examples=alg_batch_size, difficulty=0.5)` and takes one collated batch; the collate right-pads to the longest example ([data.py:1607-1622](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1607-L1622)).
- [train.py:792-796](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L792-L796): `compute_attention_probs(attn_module, attn_input, position_ids=...)` with no `mask`. [utils.py:396-398](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L396-L398): when `mask is None` only a causal mask is built, although the function accepts a `mask`.
- [train.py:803-811](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L803-L811): 32 query positions per head are drawn uniformly from `B * T`, so pad rows are sampled in proportion to the pad fraction; [train.py:813-817](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L813-L817) averages their entropies into `eval/attn/entropy/last/mean` and `/std`.
- The model itself masks pad keys ([model.py:842](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L842), `_build_padding_mask`), so the probe's attention differs from the forward pass that produced the hooked input.
- Audit probe `code-train-a/p7_diagnostics_probe_batch.py`: at `alg_seq_len 48` the 96 x 48 probe batch is 38.5 % trailing padding (median example 29 tokens); at 128 the batch pads to 66 and is 54.6 % padding. On a 2.2M-parameter random model, `eval/attn/entropy/last/mean` is 2.864 for the padded batch and 1.615 when the same batch is cut to its first 12 tokens (no padding).
- DAG-v12 logged `attn_entropy_last_mean` between 0.015 and 1.90 over the run; the Jan-11 run printed 0.184 at step 11,000. The trend may be real; the level cannot be read.

## Why it matters for the experiments

- Attention sharpening is one of the reported grokking precursors, and this is the only attention-level number in `metrics.json`. As logged it mixes real queries with pad queries that attend over the whole real prefix plus earlier pads, in a ratio that moves with `alg_seq_len` and the task mix. A rise or fall can come from the pad fraction, not the model.
- None of #132 to #198 and none of the project's decisions used the key, which is why this is P3 and not P2.

## How it got here

PR [#73](https://github.com/jpturcotte/ReLSM/pull/73) (2025-12-20) added the probe with the causal-only mask (gh-prs-b: "probe batch is a padded FixedAlgorithmicDataset batch, so entropies include pad keys"). PR [#112](https://github.com/jpturcotte/ReLSM/pull/112) moved padding into the collate and PR [#126](https://github.com/jpturcotte/ReLSM/pull/126) ([task](https://chatgpt.com/remote/task_e_6955827d86ac832ea0c8f051653eb80b)) gave the model its padding mask; the probe was not updated. The external report of the Oct-26 review listed "diagnostic attention entropy reconstruction may not represent actual masked/padded attention" as open; no agent probed it then (review-files H11).

## How we verified it, step by step

1. code-train-a (T3) read `run_eval_diagnostics` and `compute_attention_probs`, measured the pad fraction of the real probe batch at 48 and 128 tokens, and ran the diagnostic on the same examples padded and unpadded (2.864 vs 1.615). It grepped DAG-v12's `metrics.0.json` for the key's range.
2. claude-b2 (B2-3) reached the same mechanism by code reading, independently, from the Dec-31 masking chat.
3. gh-prs-b (PR #73 row) traced the probe to its PR.
4. review-files (H11) recorded that the Oct-26 external report had flagged it and nobody had checked.
5. Coordinator: not re-checked. Disagreement: severity P2 (code-train-a) vs P3 (claude-b2); P3 chosen, see Summary.

## Suggested fix

In `run_eval_diagnostics`, build `padding_mask = model._build_padding_mask(input_ids, dtype)` and pass `mask = causal + padding_mask` to `compute_attention_probs` (it already takes `mask`); draw the 32 query positions only from positions where `input_ids != pad_token_id` (or `labels != -100`, or the per-example lengths that `include_lengths=True` already provides). Independent of other fixes; the pad == EOS question (NF-03) changes which keys count as padding but not this change.

## Acceptance test

Tiny CPU model, probe batch with at least 30 % trailing padding: `eval/attn/entropy/last/mean` differs by less than 1e-6 between the padded batch and the same examples run one at a time with no padding. Add it to the #184 invariant suite.

## Related

- #184: the CPU invariant suite this check belongs in.
- #172: the other place where a padding mask is dropped (cached decoding).
- #149: the rest of the telemetry whose definitions do not match their names.
- #238, #245: the two other PR #73 diagnostics with the same problem.

## Sources

- Audit report code-train-a, finding T3; probe `p7_diagnostics_probe_batch.py`
- Audit report claude-b2, finding B2-3 and working notes on chat 3 (Dec 31)
- Audit report gh-prs-b, per-PR table row #73
- Audit report review-files, lead H11
- Audit report gemini-c, section 2.6 (Jan-11 log value)
