---
slug: hidden-adamw-betas-and-cudnn-flags
title: "AdamW betas and eps are hard-coded and unrecorded; cuDNN flags forced on CUDA"
issue: 246
severity: P3
labels: [enhancement, P3-minor, area: training, reproducibility, audit-2026-10-27]
nf: [NF-25]
found_by: [codex-a, gh-prs-a, code-train-a]
status: Plausible
related_issues: [#187, #186, #137, #189, #143]
---

## Summary

`build_optimizer` returns `AdamW(param_groups, lr=lr, betas=(0.9, 0.95))` with the default eps 1e-8. There is no flag, and the run record stores only `vars(args)`, so a reader of `metrics.json` cannot tell which betas a run used. The grokking recipes the project emulates (Power et al. 2022, Grokfast's modular-arithmetic setups) use (0.9, 0.98). Separately, `torch.backends.cudnn.deterministic = True` and `benchmark = False` are set on every CUDA run with no flag; they slow convolution and SDPA kernel selection and do not give determinism anyway (no `use_deterministic_algorithms`; DataLoader workers change the stream). Missing knobs and a missing record, not a bug; the GPU cost was not measured, so Plausible.

## What happens

- [train.py:1844](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1844): `return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))`. No `--adam_beta*`/`--adam_eps` in `main()`.
- [train.py:1946](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1946): `logger.set_hyperparameters(vars(args))` is the whole run record; betas and eps are not in it (DAG-v12's hyperparameter row has no such key; code-train-a `p3_optimizer_groups.out` reads betas (0.9, 0.95), eps 1e-8 from the built optimizer).
- [train.py:1935-1936](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1935-L1936): `cudnn.deterministic = True; cudnn.benchmark = False` under `if torch.cuda.is_available()`.

## Why it matters for the experiments

- A direct replication of a grokking recipe needs β₂ 0.98; today it needs a code edit, and nothing in the run record would show the change. The lexical report notes a preprint linking β₂ 0.95 to rare-token logit drift, which touches #218; untested here.
- Every GPU run pays the deterministic-kernel cost (relevant to the SSM depthwise convolution and SDPA) for a determinism the run does not have: with `num_workers > 0` the data stream changes (code-train-b: 0 of 11 batch hashes shared between workers 0 and 2), and #143's reseeding at every eval moves the global RNGs anyway. CPU runs are already bit-identical seed to seed.
- #187 lists what the run record lacks; betas and eps belong on that list.

## How it got here

- PR [#13](https://github.com/jpturcotte/ReLSM/pull/13) (2025-12-11, "Refine optimizer parameter grouping") wrote the `AdamW(..., betas=(0.9, 0.95))` return; the task never specified betas and the earlier form already carried the same pair. Not run.
- PR [#9](https://github.com/jpturcotte/ReLSM/pull/9) (2025-12-10, "Add global seed configuration") added the cuDNN flags for "reproducibility" alongside `--seed`; never revisited. Not run.

## How we verified it, step by step

1. codex-a (F4) and gh-prs-a (A3) read the two sites and the PR history; both flagged the absent CLI path and record.
2. code-train-a built the optimizer with the repo's `build_optimizer` and read betas (0.9, 0.95), eps 1e-8 from `param_groups` (`code-train-a/p3_optimizer_groups.py`).
3. code-train-b showed CPU seed-to-seed determinism (identical weight hashes) and that the worker count changes the stream, which bounds what the cuDNN flags can buy.
4. Not measured: the throughput cost of the cuDNN flags on the 4060 (GPU busy per brief). A 200-step nano run with and without the two flags would settle it.

## Suggested fix

- Add `--adam_beta1`, `--adam_beta2`, `--adam_eps` (defaults equal to today's values), pass them to `build_optimizer`, and record them in the hyperparameter row and the checkpoint (#187).
- Put the cuDNN flags behind `--deterministic` (default off); when on, also call `torch.use_deterministic_algorithms(True, warn_only=True)` so the flag means what it says.
- Record the choice in the ADR (#189). No conflict with #186's group switch; both touch `build_optimizer`.

## Acceptance test

Pure Python, CPU: `main()`'s parser accepts `--adam_beta2 0.98 --adam_eps 1e-6`; the built optimizer's `param_groups[0]["betas"] == (0.9, 0.98)` and `["eps"] == 1e-6`; the hyperparameter row written by `MetricsLogger` contains `adam_beta2: 0.98`; without `--deterministic`, `torch.backends.cudnn.deterministic` is unchanged after setup.

## Related

- #187: the run record that should carry these values.
- #186: the other `build_optimizer` switch.
- #137: Grokfast ablations need the same optimizer settings on both arms.
- #143: eval reseeds the global RNGs, so GPU determinism is already lost.
- #189: ADR.

## Sources

- Audit report codex-a, finding F4; gh-prs-a, finding A3; code-train-a, #186 row and avenue "Weight decay as the grokking literature uses it"; probe `code-train-a/p3_optimizer_groups.py`
- Audit report code-train-b ("Checks the brief asked for": `--seed`)
- Audit report lexical, §9 (β₂ and rare-token drift)
