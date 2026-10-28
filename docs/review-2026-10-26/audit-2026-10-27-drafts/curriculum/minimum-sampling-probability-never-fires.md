---
slug: minimum-sampling-probability-never-fires
title: The 0.05 minimum task-sampling floor never fires by default; dead under DAG
issue: 221
severity: P3
labels: [bug, P3-minor, area: curriculum, tech-debt, audit-2026-10-27]
nf: [NF-18]
found_by: [code-train-a, gh-prs-b, codex-b]
status: Demonstrated
related_issues: [#157, #180]
---

## Summary

`CurriculumWeightingAdjuster` applies a hard-coded 0.05 minimum sampling probability per task, but only while the mean curriculum difficulty is below 0.2. The initial difficulty is 0.2, so the floor is off from the first upward step onwards and only returns if tasks regress below their starting value. Under `--task_curriculum_strategy dag` the adjuster is replaced by `DagWeightingAdjuster`, so the floor never applied in any DAG run. There is no flag and no documentation for either behaviour.

## What happens

- [train.py:252-264](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L252-L264): `__call__` computes the mean of the difficulty snapshot and `if mean_difficulty >= 0.2: return weights`; otherwise `_apply_minimum_probability(weights, self._min_task_prob)`.
- [train.py:2156](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2156): `min_task_prob = 0.05`, no CLI flag. [train.py:2158-2163](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2158-L2163) installs the adjuster; [train.py:2166-2171](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2166-L2171) overwrites `weighting_adjust_fn` with `DagWeightingAdjuster` whenever `dag_state` exists.
- `init_difficulty` is 0.2 ([curriculum.py:18](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L18)) and `min_difficulty` 0.0 (line 19).
- Audit probe `code-train-a/p6_dag_weighting_misc.py`: at mean difficulty 0.1 and 0.19 the floor applies (nine tasks at 0.05); at mean difficulty 0.5 a task with weight 0 gets probability 0. At the exact initial snapshot the probe shows the floor applying too: ten values of 0.2 average to 0.19999... in floating point, which is below 0.2; but at that moment every task weight is 1.0 (EMA 0), so the floor changes nothing. One upward step later the mean is above 0.2 and the floor is off.
- gh-prs-b: in the first DAG run (Dec 25) difficulties did fall to 0.05, where the floor would have applied, but the DAG adjuster had already replaced it.

## Why it matters for the experiments

- The feature PR #102 asked for, "a minimum task sampling floor when curriculum difficulty is low", exists only in the window where tasks are regressing below their start, which is the regime #201 describes. In the normal regime a task with EMA 1.0 gets weight `max(0.05, 1 - ema)` = 0.05 from `get_sampling_weight` ([curriculum.py:149-158](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L149-L158)), so a different 0.05 does the work and this one is dead.
- Nobody reading a run can tell which floor, if any, was in force. Under the DAG the only floor is the 2 % locked floor of #157.

## How it got here

- PR [#102](https://github.com/jpturcotte/ReLSM/pull/102) (Dec 22, "Add minimum task sampling floor when curriculum difficulty is low"): the owner asked for a floor when the mean difficulty is below 0.2; Codex implemented the gate literally; tests "Not run".
- PR [#108](https://github.com/jpturcotte/ReLSM/pull/108) (Dec 25) added `DagWeightingAdjuster` and bypassed it.

## How we verified it, step by step

1. code-train-a ran `code-train-a/p6_dag_weighting_misc.py` (adjuster output at mean difficulty 0.1, 0.19, 0.2, 0.5; DAG weighting floors).
2. gh-prs-b (A-5) and codex-b (PR #102 note) read `train.py:252-265`, `868-893` and `2156-2169` and reached the same arithmetic from the defaults.

## Suggested fix

- Either make it a flag with stated semantics (`--curriculum_min_task_prob`, applied whenever the adaptive weighting is on, including under the DAG for unlocked tasks), or delete the adjuster and its gate and document that `get_sampling_weight`'s 0.05 is the floor.
- No dependency; touches the same lines as the #157 locked-floor discussion.

## Acceptance test

Pure Python: construct `CurriculumWeightingAdjuster` with a snapshot whose mean difficulty is 0.5 and call it with weights `[0.0, 1.0, ...]`; the zero-weight task's renormalised probability must be at least the floor (today 0.0), or the class no longer exists and `--help` names the one floor that applies.

## Related

- #157: the 2 % locked floor is the only sampling floor in DAG runs.
- #180: if the adjuster is deleted, add it to the dead-code list.
- #201: the only regime where this floor engages is the ratchet.

## Sources

- Audit report code-train-a, finding T7; audit probe `code-train-a/p6_dag_weighting_misc.py`
- Audit report gh-prs-b, finding A-5
- Audit report codex-b, PR #102 note
