---
slug: task-weighting-adaptive-is-static
title: "'adaptive' task weighting is a static table, dead under the task curriculum"
issue: 222
severity: P3
labels: [bug, P3-minor, area: curriculum, telemetry, audit-2026-10-27]
nf: [NF-17]
found_by: [cli-commands]
status: Demonstrated
related_issues: [#180, #157, #189]
---

## Summary

`--task_weighting adaptive` (the default) adapts to nothing. `get_task_weight` returns a hand-written table (multiplication 2.5, addition 2.0, reverse 1.5, copy, parity, successor and mod_add 1.0, chain 0.8, dyck and compare 0.7) and halves dyck, compare and chain after 50 % of the token budget; no measurement of the model enters it. Under `--use_task_curriculum` the string is dead for sampling, because `_sample_task` uses `weighting_fn`, yet `train.py` still logs `_compute_task_sampling(..., args.task_weighting)` as `eval/curriculum/prob/*`, so the logged probabilities can differ from the ones used. cli-commands rated this P2; it is P3 here because the one place where the table silently shapes a result, the train-eval mix, is its own P2 (#206); what remains is a misleading name and a misleading log key.

## What happens

- [data.py:176-182](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L176-L182) (and an identical second definition at [data.py:266-273](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L266-L273), which wins): `base = TASK_BASE_WEIGHTS.get(task, 1.0)`; halve for `dyck`, `compare`, `chain` when `progress > 0.5`.
- [data.py:1179-1181](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1179-L1181): used only when `weighting_fn` is `None`. With the task curriculum on, [data.py:1172-1178](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1172-L1178) samples from `weighting_fn` (`1 - ema_acc`, floor 0.05) and the adjusters.
- Audit probe `cli-commands/flag_semantics.py` F08: in 20,000 draws the table gives multiplication 20.4 %, addition 16.2 %, reverse 12.4 %, compare 5.7 %, dyck 5.8 %; both definitions are identical in effect.
- [train.py:3054-3056](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3054-L3056): when `dag_state` is `None`, the eval block logs `eval/curriculum/prob/*` from `args.task_weighting`, not from the weights the loader used. The help text at [train.py:3808-3814](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3808-L3814) says only "Task sampling weights".

## Why it matters for the experiments

- Runs without the task curriculum (every `--fixed_data` run, the Dec 19-20 runs) sampled tasks by a hidden prior that no document mentions: multiplication was drawn 3.6 times as often as compare. "Task X is hard" in those logs partly means "task X was rare".
- agent-reports found the Dec 24 "Étagé" design report reading these hand-written weights (in `metrics.4.json`) as empirical output of a Nash-MTL or bandit allocator that never ran. The name `adaptive` invited that reading.
- Under the curriculum, a reader of `eval/curriculum/prob/*` sees the static table while the loader sampled by competence.

## How it got here

- PR [#66](https://github.com/jpturcotte/ReLSM/pull/66) (Dec 20) added `TASK_BASE_WEIGHTS`, `get_task_weight` and `--task_weighting adaptive`.
- PR [#88](https://github.com/jpturcotte/ReLSM/pull/88) (Dec 21, "Add curriculum-driven task sampling and restore static task weights") added `weighting_fn` and pasted a second copy of the table; its body says the helper "was omitted ... causing a NameError", which was false (gh-prs-b). Tests: "Not run".

## How we verified it, step by step

1. cli-commands ran `cli-commands/flag_semantics.py` F08: both definitions located (lines 176 and 266), table contents, 20,000-draw shares.
2. Code read of `data.py:1172-1181` and `train.py:3054-3056` for the dead-for-sampling and logged-anyway points. artifacts' DAG-v12 train-eval counts (148/126/79/31/64 of 800) match the table's shares, which is the #206 mechanism.

## Suggested fix

- Rename the choice to `static_weighted` (keep `adaptive` as a deprecated alias for one release) and print the table in `--help`.
- Under the task curriculum, log the probabilities actually used (the `weights_snapshot` after adjustment) under `eval/curriculum/prob/*`, and log the realised per-task share from `train.task_count_window.*`.
- Delete the duplicate definition (#180).
- No dependency; the train-eval half is #206.

## Acceptance test

Pure Python: `AlgorithmicDataset._sample_task` with `weighting_fn=None` and `task_weighting="adaptive"` over 20,000 draws at progress 0.0 reproduces the table's shares within 1 percentage point whatever accuracies are fed to a `TaskCurriculumState`; and `python train.py --help` names the table under the renamed choice.

## Related

- #180: duplicate `TASK_BASE_WEIGHTS` / `get_task_weight`.
- #157: the logged `eval/curriculum/prob/*` is one of the series read for the DAG analysis.
- #189: the design-document reading of these weights as an allocator's output.
- #206: the same table decides the train-eval mix and the gate inputs.

## Sources

- Audit report cli-commands, finding C5 and the `--task_weighting` flag row; audit probe `cli-commands/flag_semantics.py` (F08)
- Audit report gh-prs-b, PR #88 row
- Audit report agent-reports, §2 (the Étagé report's reading of the weights)
