---
slug: locked-tasks-demoted-on-unlock
title: Locked DAG tasks keep a chance-level EMA and are demoted as soon as they unlock
issue: 205
severity: P2
labels: [bug, P2-important, area: curriculum, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-48]
found_by: [artifacts, code-train-b]
status: Demonstrated
related_issues: [#157, #141]
---

## Summary

`task_curriculum.update_metrics` runs for every task in the train-eval set, locked or not, so a locked binary task carries an EMA near 0.5 and a `step_count` that grows every eval. When its gate opens, the controller waits only `unlock_warmup_evals` (which defaults to `curriculum_min_task_evals`, 2 in DAG-v12), sees `step_count >= min_task_evals` and an EMA below 0.55, and subtracts 0.05. In DAG-v12 the only task that ever unlocked, parity, went 0.20 to 0.15 to 0.10 within 1,000 micro-steps of unlocking, spent 26,000 micro-steps below its starting difficulty and reached the final-phase range only at step 55,000. Every future unlock of any task whose chance accuracy is below 0.55 (all but compare and dyck) starts the same way. code-train-b rated this P3; it is P2 here because it cost the one unlock in the one surviving run a third of the run, and it will repeat on every unlock.

## What happens

- [train.py:3477-3480](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3477-L3480): `update_metrics(task, acc, loss_proxy, global_step)` for every task in `per_task_accuracy`, gated or not. The train-eval set samples all tasks (#206).
- [train.py:3518-3545](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3518-L3545): `unlock_warmup_evals` falls back to `args.curriculum_min_task_evals`; `step_curriculum` is skipped only while `evals_since_unlock < unlock_warmup_evals`. Nothing resets `ema_acc`, `step_count` or `last_update_step` at unlock; `DagUnlockState.update_from_ema` only records `unlocked_eval_index` ([curriculum.py:364-365](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L364-L365)).
- [curriculum.py:110-139](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L110-L139): `step_count >= min_task_evals` passes (the count grew while locked), `ema_acc < 0.55` is true at chance, so `difficulty -= 0.05`.
- DAG-v12, parity (audit probe `artifacts/p2_timeline.py`, B2 and B5): gate 0 to 1/3 at step 10,000, 2/3 at 11,000, 1.0 at 12,000. EMA while locked 0.47-0.58 (raw 0.43-0.64; chance for a 0/1 label). Difficulty 0.20 from 1,000 to 11,000, 0.15 at 12,000, 0.10 at 13,000, 0.15 at 19,000, 0.20 at 38,000, then 0.25 at 39,000 and up to 0.90 at 77,000; the 0.60 range of the IID grid was reached at 55,000. Its IID score rose 0.55 to 0.94 as its difficulty entered the grid's range, the "staircase" of the counter-review.
- code-train-b's tiny DAG probe (`dag_w0`, untrained model, EMA 0): the roots themselves went 0.20, 0.15, 0.10, 0.05, 0.00 in four evals, the same rule with `min_difficulty 0.0` (#201).

## Why it matters for the experiments

- The DAG admits a task and then makes it easier than the 0.2 floor it was already learning at. In DAG-v12 that is 26,000 of 77,500 micro-steps for the only task the gates ever opened.
- Any run with more unlocks pays this at each one: successor, chain, addition, mod_add and multiplication all have chance accuracy near 0 and would be demoted on their first eligible eval, in a run where `--curriculum_min_task_evals` is small (the DAG-v12 value, 2, is in the only documented DAG command).
- The parity IID curve, read in #157 and the counter-review as "learning", is to a first approximation the controller walking the task back up the difficulty ladder it had pushed it down.

## How it got here

- PR [#109](https://github.com/jpturcotte/ReLSM/pull/109) (Dec 25, "Add DAG locked-task sampling floor and post-unlock curriculum warmup"): `step_curriculum` only for unlocked tasks after `--dag_unlock_warmup_evals`, defaulting to `--curriculum_min_task_evals`; gh-prs-b's read of the diff: "EMA still updated for all tasks". The PR was a response to the first DAG run driving locked tasks to 0.05; it stopped the stepping, not the tracking. Tests: "Not run".
- DAG-v12 set `curriculum_min_task_evals 2` (flag row in `metrics.0.json`), so the warm-up was two evals.

## How we verified it, step by step

1. artifacts ran `artifacts/p2_timeline.py` over DAG-v12 `metrics.0.json`: gate transitions (B2), per-task EMA and raw accuracy while locked (B1, B4), and the parity difficulty path (B5).
2. code-train-b ran the tiny DAG probe `code-train-b/run_variants.sh` (`dag_w0` log, lines 170-1397) and read `train.py:3518-3545`, `curriculum.py:118-143`.
3. Code read at HEAD of the three sites above. No coordinator re-check of this row.

## Suggested fix

- On unlock, reset the task's `ema_acc`, `step_count` and `last_update_step` (a `TaskCurriculumState.reset_task` called from `train.py` for `newly_unlocked` after `dag_state.update_from_ema`), or skip `update_metrics` while `gate == 0`.
- Start the controller only after `unlock_warmup_evals` real evals of the unlocked task, and never lower a task below `init_difficulty` (or its unlock difficulty) in its first N evals.
- Depends on #201 (the floor) and sits in #157's fix list; conflicts with nothing.

## Acceptance test

Pure Python (extend `tests/test_dag_unlock_state.py` with a `TaskCurriculumState`): feed a locked task accuracy 0.5 for 10 evals, open its gate, feed 0.5 for `unlock_warmup_evals + 1` more evals and call `step_curriculum` each time; the difficulty after the warm-up must be at least `init_difficulty`. Today it is 0.15.

## Related

- #157: add this as the "state carried across the unlock" item.
- #141: a different stuck-at-easy mechanism on the schedule path.
- #201: the rule that does the demoting.
- #206: why locked tasks are measured at all.
- #202: the same controller at the other end of the band.

## Sources

- Audit report artifacts, finding AR-2 and §2b; audit probe `artifacts/p2_timeline.py` (B1, B2, B4, B5)
- Audit report code-train-b, finding N7; probe run `code-train-b/dag_w0`
- Audit report gh-prs-b, PR #109 row
