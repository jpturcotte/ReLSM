---
slug: train-eval-samples-double-role
title: --train_eval_samples is set size and generation cap; 0 freezes the curriculum
issue: 243
severity: P3
labels: [bug, P3-minor, area: tooling, audit-2026-10-27]
nf: [NF-33]
found_by: [code-train-b, cli-commands]
status: Plausible
related_issues: [#157, #141, #136, #169]
---

## Summary

`--train_eval_samples` is both the size of the procedural train-eval dataset and the `max_samples` cap of the generation pass over it. When it is 0 or `None`, the dataset size silently falls back to `--eval_samples` (documented as the IID per-condition count), but the generation pass still receives 0 and `_iter_eval_examples` returns before the first example, so no per-task accuracy is produced and the task curriculum, which is driven by that pass, never updates. `--train_eval_max_batches` caps only the teacher-forced pass. The help text says none of this.

## What happens

- [train.py:2281](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2281): `num_examples=args.train_eval_samples or args.eval_samples`; 0 is treated as unset.
- [train.py:3467](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3467): `evaluate_task_accuracy(..., max_samples=args.train_eval_samples, ...)`; 0 is passed through.
- [train.py:1091-1097](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1091-L1097): `_iter_eval_examples` returns as soon as `count >= max_samples`, which is true at 0 before the first example.
- [train.py:3456](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3456): `max_batches=args.train_eval_max_batches` applies to `evaluate_training_split` (teacher-forced) only.
- Help: [train.py:3930-3946](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3930-L3946) ("Number of samples to use for training-set accuracy checks", default 2000; "Optional cap on batches when evaluating training accuracy") and [train.py:4024](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L4024) `--eval_samples` ("per interval"). Neither mentions the other.
- Both passes read the same `seed + 123` set, so "samples" also sets the size of the teacher-forced pass unless `--train_eval_max_batches` is given.

## Why it matters for the experiments

- The generation pass is the only input to the task-curriculum EMA and the DAG gates (its per-task accuracies feed `update_metrics`). A user who sets `--train_eval_samples 0` to save eval time gets a run whose curriculum never moves, with no warning.
- Cost: DAG-v12 ran with `train_eval_samples 800` and spent 61 s per eval, 27.6 % of wall time (artifacts AR-7); the generation pass over 800 examples is most of it. The knob that controls that cost is overloaded with the knob that controls the set size.

## How it got here

PR [#57](https://github.com/jpturcotte/ReLSM/pull/57) ([task](https://chatgpt.com/remote/task_e_6944b5513f4c832e9968d658bb5bb96e)) wired `max_samples=args.train_eval_samples` into the generation pass for the fixed set, where set size and sample count were the same thing. PR [#87](https://github.com/jpturcotte/ReLSM/pull/87) switched curriculum runs to the procedural stream; the `or args.eval_samples` fallback already existed when PR [#120](https://github.com/jpturcotte/ReLSM/pull/120) rewrote the loaders (it appears as context in that diff) and was not traced to its PR by this audit. The two roles were never separated.

## How we verified it, step by step

1. code-train-b (N9) read `train.py:2281` and `3467` and traced `max_samples` into `_iter_eval_examples`.
2. cli-commands (C14) read the same lines and `1091-1097`, and documented the overload in its flag table rows for `--train_eval_samples`, `--train_eval_max_batches` and `--eval_samples`. It did not run `--train_eval_samples 0` (dropped under the memory constraint).
3. Coordinator: not re-checked.

Status is Plausible: no run with `--train_eval_samples 0` was executed. The probe that settles it: a tiny CPU run with `--use_task_curriculum --train_eval_samples 0 --eval_interval 10`, checking that `eval/curriculum/ema/*` is unchanged after two evals and that no `train_eval/acc/*` key is logged.

## Suggested fix

Separate the flags: `--train_eval_set_size` (dataset) and `--train_eval_gen_samples` (generation cap), or make `--train_eval_max_batches` apply to both passes. Use a positive-integer `type` so 0 is rejected at parse time, and say in the help that the procedural set does not fall back to `--eval_samples`. Independent of #157's proposal to express intervals in optimizer steps.

## Acceptance test

`python train.py --train_eval_samples 0 ...` exits with an argparse error. With `--train_eval_samples 4 --eval_samples 100 --use_task_curriculum`, the procedural train-eval set has 4 examples (not 100) and the generation pass scores 4; after two evals the curriculum EMA has changed.

## Related

- #157: the curriculum EMA and gates that depend on this pass.
- #141: difficulty schedule and workers; the same eval block.
- #136: the fixed-set train-eval path where the flag was first defined.
- #169: README and help text accuracy.
- #242, #237: the same train-eval block.

## Sources

- Audit report code-train-b, finding N9
- Audit report cli-commands, finding C14 and flag table rows `--train_eval_samples`, `--train_eval_max_batches`, `--eval_samples`
- Audit report artifacts, finding AR-7 (eval cost)
