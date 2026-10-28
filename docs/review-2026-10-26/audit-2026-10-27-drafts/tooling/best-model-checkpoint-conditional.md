---
slug: best-model-checkpoint-conditional
title: best_model.pt is not written when IID exact match stays 0.0; README eval fails
issue: 235
severity: P3
labels: [bug, P3-minor, area: tooling, reproducibility, audit-2026-10-27]
nf: [NF-22]
found_by: [code-train-b, cli-commands]
status: Demonstrated
related_issues: [#169, #185, #136, #162, #189]
---

## Summary

`best_model.pt` is saved only when an IID eval beats `best_alg_acc`, which starts at 0.0 under a strict `>`. A run whose IID exact match never leaves 0.0, a run with `--no-eval_algorithmic`, and a run stopped before its first eval have no `best_model.pt`. All three README evaluation commands and the compare loop pass `--checkpoint .../best_model.pt` literally and die with `FileNotFoundError`. `summary.json` still reports `best_alg_accuracy: 0.0` with no file behind it.

## What happens

- [train.py:2405](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2405): `best_alg_acc = 0.0`.
- [train.py:3288-3292](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3288-L3292): `if overall_acc > best_alg_acc:` ... `torch.save(best_payload, output_dir / "best_model.pt")`. The block is inside the IID eval branch, so `--no-eval_algorithmic` skips it.
- [train.py:3749](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3749): `summary["best_alg_accuracy"] = best_alg_acc` is written whether or not the file exists. Nothing records the step it came from.
- [train.py:2530](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2530): a resume restores `best_alg_acc` but not the file, so a resumed run whose best eval was before the resume point never writes one in the new directory.
- [README.md:76](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/README.md#L76), [:79](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/README.md#L79), [:82](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/README.md#L82) and [:188](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/README.md#L188): `--checkpoint ./runs/nano_baseline/best_model.pt` and `./runs/nano/$exp/best_model.pt`.
- The file that always exists is `final_model.pt` ([train.py:3742](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3742)).

## Why it matters for the experiments

- The README's only evaluation recipe fails on a short, failed or early-stopped run, which is what most of the project's runs were ("all of my grokking experiments failed", #169). Every probe run of two agents ended without the file.
- "Best" is IID exact match on a grid that is not the training distribution (#162). For a grokking run the checkpoint to keep is the one after the held-out jump, which this criterion does not see. That is a design point for #189; this issue is only about the missing file and the broken commands.

## How it got here

The `overall_acc > best_alg_acc` save was already in `train.py` when PR [#57](https://github.com/jpturcotte/ReLSM/pull/57) ([task](https://chatgpt.com/remote/task_e_6944b5513f4c832e9968d658bb5bb96e)) rewrote the checkpoint block to add rotating eval checkpoints; the diff re-indents the `best_model.pt` line without changing the condition. The README commands that name the file came with the eval hub (PR [#19](https://github.com/jpturcotte/ReLSM/pull/19), [#20](https://github.com/jpturcotte/ReLSM/pull/20)). No task asked what happens when no eval scores above 0.0.

## How we verified it, step by step

1. code-train-b (N8) read `train.py:3288-3292` and listed its probe run directories (`base_w0`, `interrupt`, `resume_wof`): none has `best_model.pt`; each `summary.json` says `best_alg_accuracy: 0.0`.
2. cli-commands (C9) ran a default-flag tiny CPU run (R1) and listed the output: `metrics.json`, `summary.json`, `final_model.pt`, `checkpoint_step_10.pt`, `checkpoint_step_20.pt`, four PNGs, no `best_model.pt`. R2 to R19 likewise.
3. cli-commands E6 ran README line 76 verbatim against that directory (audit probe `cli-commands/e6_best_model.log`): exit 1, `FileNotFoundError: ... best_model.pt`.
4. Coordinator: not re-checked. No disagreement between the two agents.

## Suggested fix

Either:

- write `best_model.pt` at the first IID eval unconditionally (`>=` on the first eval, `>` afterwards), or
- point the README commands at `final_model.pt` and add `best_step` (`null` if none) next to `best_alg_accuracy` in `summary.json`.

Touches the same FINISH block as #185 but does not depend on it. Add the checkpoint name to the run record proposed in #189.

## Acceptance test

Run `train.py` on CPU with a tiny model and `--alg_tokens 2000 --eval_interval 10 --eval_samples 2`, so IID accuracy is 0.0 at every eval. Then either the output directory contains `best_model.pt` (option 1), or `summary.json` has `"best_step": null` and the README command, with `final_model.pt`, exits 0 (option 2).

## Related

- #169: README claims; the `best_model.pt` commands belong in its list.
- #185: `summary.json` is only written on normal completion.
- #136: `--fixed_data` runs crash before any eval, so they never produce the file either.
- #162: the IID grid that defines "best".
- #189: the run record should name the checkpoint that was evaluated.
- #244: the "New best! Saved checkpoint." line is also silent by default.

## Sources

- Audit report code-train-b, finding N8
- Audit report cli-commands, finding C9 and section 3c (README command table); log `e6_best_model.log`
- `VERIFIED_BY_COORDINATOR.md`: not re-checked
