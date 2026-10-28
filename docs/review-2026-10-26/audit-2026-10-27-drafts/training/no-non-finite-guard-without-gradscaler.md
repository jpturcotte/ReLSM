---
slug: no-non-finite-guard-without-gradscaler
title: "No non-finite guard when GradScaler is off: inf gradient becomes NaN weights"
issue: 247
severity: P3
labels: [bug, P3-minor, area: training, correctness, audit-2026-10-27]
nf: [NF-24]
found_by: [code-train-b]
status: Plausible
related_issues: [#137, #151, #148, #185, #184]
---

## Summary

`GradScaler` is enabled only for fp16. The training dtype is bf16 whenever the GPU supports it, and on CPU the scaler disables itself, so on every realistic path `scaler.step` is a plain `optimizer.step()` with no inf/NaN skip. The loop never checks `torch.isfinite` on the loss or the gradient norm. A single inf gradient passes `clip_grad_norm_` as a total norm of inf, which scales every gradient to NaN or 0, and AdamW then writes NaN weights. Nothing is logged and the run continues. #137 records this chain for the Grokfast overflow and #151 the zero-label NaN loss (which leaves zero gradients, so weights survive); the general case has no issue. The clip step is Demonstrated by a micro-probe; the end-to-end path through `train.py` was not run, so this draft is Plausible.

## What happens

- [train.py:2356-2358](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2356-L2358): `dtype = bfloat16 if is_bf16_supported() else float16`; `scaler = GradScaler(enabled=(dtype == torch.float16))`.
- [train.py:2715](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2715): `scaler.scale(loss).backward()`; [train.py:2730-2731](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2730-L2731): `unscale_` and `clip_grad_norm_`; [train.py:2744-2746](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2744-L2746): `scaler.step`, `update`, `zero_grad`. No branch on `grad_norm`.
- `grep -n isfinite train.py` returns nothing (re-run at `774513d`).

Measured (code-train-b micro-probe): `clip_grad_norm_` on gradients `[inf, 1, 2]` returns total norm `inf` and leaves gradients `[nan, 0, 0]` (clip coefficient `max_norm / inf = 0`, and `inf × 0 = nan`); an AdamW step on that gradient writes NaN into the parameter. Every CPU probe log shows the scaler's "disabled" warning.

## Why it matters for the experiments

- On the project GPU (RTX 4060, bf16) there is no safety net at all. A Grokfast overflow (#137), a bf16 overflow in a long-context forward, or any inf activation turns the model to NaN in one update; the loss then reads NaN and the only trace is the logged loss, after the fact. `grad_clipped` and `grad_norm` are logged per window, so a single bad update can hide inside a 50-step mean.
- #148 asks for an "AMP skips" counter; without a guard there is nothing to count on the bf16 path.
- DAG-v12 shows no sign of it (median grad norm 0.24, `grad_clipped` 10 of 1,550 windows), so no surviving result is affected.

## How it got here

- `GradScaler(enabled=(dtype == torch.float16))` and the bf16 selection appear as unchanged context in PR [#13](https://github.com/jpturcotte/ReLSM/pull/13)'s diff (2025-12-11), so they predate it; PR [#38](https://github.com/jpturcotte/ReLSM/pull/38) (autocast) and PR [#58](https://github.com/jpturcotte/ReLSM/pull/58) (`torch.amp.GradScaler` API) touched the lines without adding a guard. Not traced to a specific introducing PR by this audit; no task asked for a non-finite check.

## How we verified it, step by step

1. code-train-b (N6) read the scaler setup and the step block, grepped for `isfinite` (nothing), and ran the `clip_grad_norm_` micro-probe on `[inf, 1, 2]` followed by an AdamW step.
2. code-train-b confirmed the contrast with #151: an all-`-100` cross-entropy gives NaN loss and exactly zero gradients, so that case does not poison weights; the inf case does.
3. Not run: the path through `train.main()` with an injected inf gradient on CPU, and any bf16 GPU behaviour. That CPU run (a forward hook that sets one gradient element to inf at micro-step 3, then check `final_model.pt` for NaN) would settle it.

## Suggested fix

After `clip_grad_norm_`, if `not torch.isfinite(grad_norm)`: `optimizer.zero_grad()`, skip `scaler.step`, increment a `train/optim/skipped_steps` counter, log a warning with the step, and abort after N consecutive skips. This also provides the "AMP skips" counter #148 asks for. Keep the fp16 scaler path as is. Independent of #137's filter placement; both are needed before a Grokfast run means anything.

## Acceptance test

CPU, tiny model, 5 micro-steps with a hook that sets one element of `tok_emb.weight.grad` to `inf` on micro-step 3: the run exits 0, `train/optim/skipped_steps` is 1, and every tensor in `final_model.pt` is finite. Without the fix the same run writes NaN weights.

## Related

- #137: Grokfast overflow is the known trigger.
- #151: zero-label NaN loss (safe for weights; this case is not).
- #148: the skipped-step counter.
- #185: a NaN run today ends as a "finished" run with NaN weights, not a crash.
- #184: CPU suite.
- #216: the other Grokfast blocker.

## Sources

- Audit report code-train-b, finding N6 and "Checks the brief asked for" (GradScaler and bf16); micro-probe described in that report
- Merged findings §3, #151 row (code-train-b contrast)
