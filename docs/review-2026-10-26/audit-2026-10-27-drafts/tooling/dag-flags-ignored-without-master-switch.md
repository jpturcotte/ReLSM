---
slug: dag-flags-ignored-without-master-switch
title: DAG and curriculum flags are silently ignored without --use_task_curriculum
issue: 236
severity: P3
labels: [bug, P3-minor, area: tooling, reproducibility, audit-2026-10-27]
nf: [NF-21]
found_by: [cli-commands]
status: Demonstrated
related_issues: [#157, #187, #189]
---

## Summary

`--task_curriculum_strategy dag`, the eleven `--dag_*` flags and `--curriculum_cooldown`, `--curriculum_jitter` and `--curriculum_min_task_evals` do nothing unless `--use_task_curriculum` is also given. There is no error and no warning. The eleven `--dag_*` arguments have no help text. Two of the ignored flags are still written into the checkpoint's `curriculum_config`, so a saved run can claim settings that never applied.

## What happens

- `task_curriculum`, the samplers and `dag_state` are created only inside `if args.use_task_curriculum:` at [train.py:2046](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2046) and [train.py:2063](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2063); the DAG branch is [train.py:2090](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2090) inside it.
- Probe run R9 (tiny model, CPU) passed `--task_curriculum_strategy dag --dag_patience_evals 1 --dag_replay_ratio 0.9 --dag_locked_floor 0.5 --curriculum_cooldown 1 --curriculum_jitter 0.9` without the switch: exit code 0, banner `Task curriculum: False` (`r9_dag_noflag.log` line 28), zero `eval/curriculum/gate` keys in `metrics.json`, no warning.
- [train.py:3850-3858](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3850-L3858): nine `--dag_*` arguments with no `help=`; `--dag_unlock_warmup_evals` ([train.py:3859-3864](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3859-L3864)) is the only one with help. The `--task_curriculum_strategy` help ([train.py:3825-3831](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3825-L3831)) does not mention the switch.
- [train.py:2438-2439](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2438-L2439): `curriculum_cooldown` and `curriculum_jitter` are written into `curriculum_config` in every checkpoint, switch or not (R9's `final_model.pt` carries `curriculum_config`).
- The README lists `--task_curriculum_strategy` among its documented flags (probe F12), so a user following it can type the DAG flags and get a plain run.

## Why it matters for the experiments

- A "DAG run" started without the switch is a baseline run with a DAG-looking command line. The Jan 11 command in the Gemini chat does include `--use_task_curriculum`; a shortened copy would not fail.
- #187 and #189 ask for run arguments to be saved with the checkpoint. Saving ignored flags without `use_task_curriculum` next to them makes that record wrong.

## How it got here

- PR [#108](https://github.com/jpturcotte/ReLSM/pull/108) (Dec 25) added the `--dag_*` arguments without help text; PR [#109](https://github.com/jpturcotte/ReLSM/pull/109) added `--dag_unlock_warmup_evals` with help. No startup validation of flag combinations exists anywhere in `train.py` (cli-commands §4 avenue 3). Not traced to a prompt by this audit.

## How we verified it, step by step

1. cli-commands ran R9 end to end with `tiny_train.py` (a d 32 model substituted for nano; `train.main()` otherwise unchanged) and inspected the banner, `metrics.json` keys and the checkpoint payload.
2. Code read of `train.py:2046`, `2063`, `2090`, `2438-2439`, `3850-3858`. No coordinator re-check of this row.

## Suggested fix

- A `validate_args()` after `parse_args`: `parser.error` when any `--dag_*` or `--curriculum_*` flag differs from its default, or `--task_curriculum_strategy dag` is given, without `--use_task_curriculum` (or auto-enable the switch with a printed warning).
- Help text for the nine flags.
- Write `use_task_curriculum` and `task_curriculum_strategy` into `curriculum_config` next to the two values already there (#187).
- No dependency.

## Acceptance test

`python train.py --task_curriculum_strategy dag --alg_tokens 1000 ...` without `--use_task_curriculum` exits with code 2 and names the missing switch; `python train.py --help` shows a help line for every `--dag_*` argument (a pure-Python check over `parser._actions`).

## Related

- #157: the flags in question.
- #187: run arguments are not saved with the checkpoint.
- #189: run records; the Jan 11 command line.

## Sources

- Audit report cli-commands, finding C8 and the flag table rows for `--task_curriculum_strategy`, `--curriculum_cooldown`, `--dag_*`; run R9 (`r9_dag_noflag.log`); audit probe `cli-commands/flag_semantics.py` (F12)
