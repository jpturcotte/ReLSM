---
slug: ff-sparsity-measures-down-projection
title: eval/act/sparsity/ff counts near-zero FF output entries, not activation sparsity
issue: 238
severity: P3
labels: [bug, P3-minor, area: tooling, telemetry, audit-2026-10-27]
nf: [NF-30]
found_by: [code-train-a]
status: Demonstrated
related_issues: [#149, #180]
---

## Summary

The "FF activation sparsity" hook captures the output of `block.ff`, which is the SwiGLU down-projection `w2(silu(w1 x) * w3 x)` added to the residual stream, and reports the fraction of entries with `|x| < 1e-4`. That tensor is dense by construction; the gated hidden units, where sparsity would mean something, are never seen. In DAG-v12 the three keys sat between 4.2e-7 and 4.3e-4 for all 77 evals.

## What happens

- [train.py:749-752](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L749-L752): `_make_ff_hook` stores the hooked module's `output`; [train.py:768-770](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L768-L770) registers it on `block.ff` for the early, mid and late blocks.
- [model.py:348-349](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L348-L349): `FeedForward.forward` returns `dropout(w2(silu(w1(x)) * w3(x)))`, so the hook sees the `d_model`-wide down-projection output (dropout is a no-op in eval mode).
- [train.py:781-786](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L781-L786): `(ff_out.abs() < 1e-4).float().mean()` is logged as `eval/act/sparsity/ff/{early,mid,late}`.
- Audit probe `code-train-a/p7_diagnostics_probe_batch.py`, random 2.2M-parameter model: mean `|ff out|` 3.2e-3, fraction below 1e-4 is 0.114, fraction below 1e-2 is 0.978; the logged keys were 0.019 to 0.022 on the probe batch. The threshold is absolute and fp32; on GPU the hooked tensor is bf16.
- DAG-v12 `metrics.0.json`: `act_sparsity.ff.{early,mid,late}` ranged 4.2e-7 to 4.3e-4 across 77 evals.

## Why it matters for the experiments

- Activation sparsity in the FF hidden layer is a known grokking correlate; this key cannot show it because it never looks at the hidden layer. For DAG-v12 it logged noise at the 1e-4 to 1e-7 level, indistinguishable from zero, for the whole run.
- A reader may conclude "no sparsification happened"; nothing in the log supports or refutes that.

## How it got here

PR [#73](https://github.com/jpturcotte/ReLSM/pull/73) (2025-12-20, "Add lightweight training diagnostics and eval-probe metrics") added the hook on `block.ff`. The FF module has no sub-module boundary between the gate and the down-projection, so a module-level hook sees only the final output. Not discussed in the PR body.

## How we verified it, step by step

1. code-train-a (T4) read `_make_ff_hook`, the registration site and `FeedForward.forward`, ran the diagnostic on a random tiny model and measured the magnitude distribution of the hooked tensor (probe `p7`).
2. code-train-a grepped DAG-v12's `metrics.0.json` for `act_sparsity.ff.*` and recorded the range over 77 evals.
3. Coordinator: not re-checked. Single finder.

## Suggested fix

Hook `block.ff.w2` with `register_forward_hook` and read its *input* (the gated hidden `silu(w1 x) * w3 x`), or add the hidden tensor to `FeedForward` as an optional return. Report the fraction below a scale-relative threshold (for example `1e-3 * hidden.abs().mean()`) in fp32, over real (non-pad) positions only, as `eval/act/sparsity/ff_hidden/*`. Or delete the key. Pairs with #234 (same probe batch, same padding issue).

## Acceptance test

Pure PyTorch, tiny `FeedForward`: compute `h = silu(w1 x) * w3 x` directly and the new logged fraction from the hook on the same `x`; they are equal to 1e-7. The old `eval/act/sparsity/ff/*` keys are gone.

## Related

- #149: telemetry whose names do not match their definitions.
- #180: delete-or-wire decisions for code that misleads readers.
- #234, #245: the two other PR #73 diagnostics.

## Sources

- Audit report code-train-a, finding T4; probe `p7_diagnostics_probe_batch.py`; grep of DAG-v12 `metrics.0.json`
- Audit report gh-prs-b, per-PR table row #73 (provenance)
