---
slug: train-eval-set-sampled-by-base-weights
title: Train-eval set ignores the DAG weights; gates are decided on 35 to 83 examples
issue: 206
severity: P2
labels: [bug, P2-important, area: curriculum, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-47]
found_by: [artifacts, code-data]
status: Demonstrated
related_issues: [#157, #162, #167, #189]
---

## Summary

The procedural train-eval dataset is built with `difficulty_fn` but without `weighting_fn` or `weighting_adjust_fn`, so it samples tasks by the static `TASK_BASE_WEIGHTS` table, not by the DAG gates or the curriculum weights. In DAG-v12 the six locked tasks were 68 % of the 800-example train-eval set and 1.95 % of the training examples. Two things follow. The series reported as "memorization tracking" (`train_accuracy` 0.52 to 0.82, `train_eval_loss` 1.37 to 0.49, against a training loss of 0.034) measure a distribution mismatch, not a memorisation gap. And the DAG gates read this set: compare's gate for successor was decided on 35 examples per eval, copy's on 64. #157 states the 68 % figure in one sentence; this draft supplies the mechanism and the loss consequence, and corrects the reading of those two series.

## What happens

- [train.py:2279-2294](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2279-L2294): the train-eval `AlgorithmicDataset` receives `seed=args.seed + 123`, `difficulty_value=None`, `difficulty_fn=eval_difficulty_fn`, `task_weighting=args.task_weighting`, and no `weighting_fn` / `weighting_adjust_fn`. The training dataset at [train.py:2201-2218](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2201-L2218) receives all three.
- [data.py:1172-1181](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1172-L1181): with `weighting_fn` `None`, `_sample_task` falls to the `task_weighting == "adaptive"` branch and `get_task_weight` ([data.py:162-182](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L162-L182)): multiplication 2.5, addition 2.0, reverse 1.5, copy, parity, successor and mod_add 1.0, chain 0.8, dyck and compare 0.7, the last three halved after 50 % progress.
- Measured, DAG-v12 `metrics.0.json`, `train_eval.task_count.*` at the last eval (audit probe `artifacts/p2_timeline.py`, B4): copy 64, reverse 83, compare 35, successor 68, parity 74, chain 58, addition 126, mod_add 80, multiplication 148, dyck 64 (sum 800). Locked six: 544 of 800 = 68 %. Predicted from the table: multiplication 22.5 %, addition 18.0 %; observed 18.5 %, 15.8 %. Training shares over the run (`train.task_count_window.*`): copy 31.6 %, reverse 33.5 %, parity 17.7 %, compare 15.3 %, the six locked tasks 1.95 % together.
- The set is seeded `seed + 123`, so it is fresh samples at the current difficulty, never training data (code-train-a T8); the labels "Training-eval loader (for grokking memorization tracking)" at [train.py:2245](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2245) and "Training accuracy (procedural)" at [train.py:3563](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3563) say otherwise.
- Gate arithmetic: compare must hold an EMA of at least 0.97 (0.98 minus the 0.01 margin) on 35 examples; one error is 0.971, two are 0.943. Copy must hold at least 0.98 on 64: at most one error. With decay 0.5 (#204) the EMA is a two-eval average of these counts.

## Why it matters for the experiments

- `train_accuracy` and `train_eval_loss` in DAG-v12 were read (counter-review, #157's "memorization tracking") as evidence about memorisation. The 14x gap between `train_eval_loss` 0.49 and training loss 0.034 is 68 % of the eval set being tasks the model saw 0.13-0.41 % of the time. The series say nothing about whether the model memorised its training stream.
- The gates that decide the DAG's whole trajectory were decided on 35-83 examples per eval, which is why one or two errors on compare blocked successor for all 77 evals (artifacts §2b: compare's EMA sat at 0.83-0.89 while its difficulty climbed, so successor's prerequisite set was never met).
- Successor's six dependants (#157) thus hinge on the sample size of a set that was never designed to be a gate input.

## How it got here

- PR [#83](https://github.com/jpturcotte/ReLSM/pull/83) (Dec 20) added the procedural train-eval loader as "memorization tracking". PR [#88](https://github.com/jpturcotte/ReLSM/pull/88) (Dec 21) added `weighting_fn` to the training dataset only; PR [#108](https://github.com/jpturcotte/ReLSM/pull/108) (Dec 25) added the DAG adjuster, again to the training dataset only, and made the train-eval accuracies the gate input. Nothing in the three PR bodies mentions the train-eval mix. Not traced to a prompt asking for the omission.

## How we verified it, step by step

1. artifacts ran `artifacts/p2_timeline.py` over DAG-v12's `metrics.0.json`: per-task `task_count` at every eval (B4), the `tf` (teacher-forced) and `teloss` columns against training loss (B1), and the gate traces (B2). Numbers above.
2. code-data's `code-data/probe_batching.py` drew 20,000 tasks from the table: multiplication 20.4 %, addition 16.2 %, reverse 12.4 %, compare 5.7 %, dyck 5.8 %, matching DAG-v12's 148 / 126 / 79 / 31 / 64 of 800 at the eval it checked.
3. The coordinator re-read `train.py:2279-2293` and confirmed the omitted `weighting_fn` (`VERIFIED_BY_COORDINATOR.md`).

## Suggested fix

- Give the train-eval dataset the same `weighting_fn` and `weighting_adjust_fn` as the training dataset (or snapshot the training stream), and log `train_eval/loss/ce` per task, so the series mean what their names say.
- For the gates, stop reading this set: draw a fixed per-task sample (for example 100 per root task at a fixed length) each eval, independent of the mix, and feed only that to `update_metrics`. This is #157's "fresh, task-balanced train-eval sample" item, with the reason it is needed.
- Rename the keys and messages (`heldout_eval` or `curriculum_eval`) so nobody reads them as memorisation again (code-train-a T8).
- Depends on the #157 fix design; conflicts with nothing.

## Acceptance test

Pure Python: build the train-eval `AlgorithmicDataset` exactly as `train.py` does, with a gate snapshot in which six tasks are locked at the 2 % floor, and draw 800 examples. Either the locked tasks are at most 4 % of the draw (same weighting as training) or every task has exactly the configured per-task count (fixed-count design). Today the locked share is about 68 %.

## Related

- #157: states "68 % locked tasks" without the mechanism; the "memorization tracking" reading should be withdrawn there.
- #162: the IID grid is a third distribution; this set is a second one.
- #167: the `train_accuracy` key this draft re-reads.
- #189: the counter-review's memorisation statements should be annotated in the run log.
- #222: the table that decides the mix.
- #242: the "memorization tracking" labels on this set.
- #204: the EMA that averages the 35-example counts.
- #205: locked tasks are also tracked by this set.

## Sources

- Audit report artifacts, finding AR-1 and §2b; audit probe `artifacts/p2_timeline.py` (B1, B2, B4)
- Audit report code-data, #157 row (a); audit probe `code-data/probe_batching.py`
- Audit report code-train-a, finding T8 (naming)
- `VERIFIED_BY_COORDINATOR.md`, "Train-eval dataset omits `weighting_fn`"
