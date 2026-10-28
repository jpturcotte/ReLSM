---
slug: difficulty-placeholders-in-eval-logs
title: "Eval difficulty keys are placeholders: train_eval logs 0.0, IID a hard-coded 0.5"
issue: 237
severity: P3
labels: [bug, P3-minor, area: tooling, telemetry, audit-2026-10-27]
nf: [NF-32]
found_by: [code-eval, artifacts, code-train-a, gh-prs-a]
status: Demonstrated
related_issues: [#162, #167, #141, #161, #187]
---

## Summary

Three "difficulty" numbers appear around every training-time eval and two of them are constants. `train_eval/curriculum/difficulty/<task>` is 0.0 at every eval because the dataset item never carries a `difficulty` key; `eval.difficulty.by_task.<task>` and the console `[eval] <task>/iid difficulty=0.500` are the literal default of `cond.params.get("difficulty", 0.5)`, which no grid condition sets; and the curriculum mean `eval_difficulty` is written twice per eval. #162 (g) already notes the 0.5 constant as context; this draft owns the fix for both placeholders and the duplicate.

## What happens

Train-eval difficulty is 0.0:

- [train.py:1300-1302](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1300-L1302): `difficulty = example.get("difficulty")`, summed only when not `None`.
- [data.py:1137-1144](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1137-L1144): the yielded item has `input_ids`, `labels`, `task` and, with `include_lengths`, the two lengths. No `difficulty`.
- [train.py:1466-1475](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1466-L1475): with no sums, every task gets 0.0; [train.py:1061](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1061) flattens it to `{prefix}/curriculum/difficulty/{task}`.
- DAG-v12 `metrics.0.json`: `train_eval.difficulty.by_task.*` has exactly one distinct value, 0.0, across all 77 evals; the logged `train_eval_samples[*].difficulty` is `None`; at the same steps `curriculum.difficulty.*` ranges 0.2 to 1.0. Audit probe `code-train-a/p5_train_eval_oracle.py` on the real loader: `difficulty_by_task` all 0.0.

IID difficulty is 0.5:

- [eval_hub.py:591](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L591): `difficulty = cond.params.get("difficulty", 0.5)`; gh-prs-a's probe: 0 of 31 grid conditions define the key. The value goes into `EvalResult.difficulty`, the `[eval] .../iid difficulty=0.500` line and `eval.difficulty.by_task.*` (0.5 in all 77 DAG-v12 evals; every sample line of audit run R2 ends `difficulty=0.5`).

Duplicate:

- [train.py:3198](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3198) and [train.py:3609](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3609) both log `difficulty_logged` (the curriculum mean) in the IID record and the train-eval record of the same eval; artifacts counted 154 `eval_difficulty` values for 77 evals.

## Why it matters for the experiments

- The Jan-11 log prints `Eval difficulty by task: all 0.200` (curriculum) directly above `[eval] .../iid difficulty=0.500`; a reader has to know which of the three numbers is real. Anyone reading the `train_eval` difficulty series sees "every task at 0.0" for a run whose curriculum went to 1.0.
- The train-eval stream samples `rng.uniform(min_difficulty, 0.5 + 0.5 * progress)` per example ([data.py:1197-1200](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1197-L1200), no shared `Value` at [train.py:2286](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2286)), so the realised difficulty of the held-out set is the number that would show whether the curriculum reached the data (#141). It is exactly the number that is lost.

## How it got here

PR [#67](https://github.com/jpturcotte/ReLSM/pull/67) (2025-12-20, "Add eval difficulty and sampled prompt logging to evaluation and training flows") added both `cond.params.get("difficulty", 0.5)` and the train-eval `example.get("difficulty")`, against a grid (PR [#66](https://github.com/jpturcotte/ReLSM/pull/66)) that never set the key and a dataset that never carried it. The flattened `by_task` keys came with PR [#128](https://github.com/jpturcotte/ReLSM/pull/128). No task checked the logged values.

## How we verified it, step by step

1. code-eval (E2) read `train.py:1298-1303` and `data.py:1137-1146`, then counted distinct values of `train_eval.difficulty.by_task.*` in DAG-v12's `metrics.0.json`: one (0.0) across 77 evals; `eval.difficulty.by_task.*` always 0.5.
2. artifacts (AR-8) found the same from the samples (`difficulty: None`) and counted the duplicate `eval_difficulty` (154 for 77 evals).
3. code-train-a saw the zero column in its oracle probe (`p5`) and noted it under #167.
4. gh-prs-a (A1) read `eval_hub.py:591` and probed the grid: 0 of 31 conditions carry `difficulty`. Coordinator confirmed `eval_hub.py:591` by reading and the 0.5 constant in the #162 row.
5. Citation fixed in this draft: the merged table cites `train.py:1480-1486` for the 0.0 fallback; those lines are `_median`. The fallback is at 1466-1475.

## Suggested fix

- Carry the sampled difficulty in the item payload (`payload["difficulty"] = torch.tensor(d)`), collate it, and let `evaluate_task_accuracy` read it; this also gives the logging that the difficulty-schedule issues (#141) need.
- In `eval_hub.py`, derive the field from the condition's parameters (for example `digits / train_max`) or drop it and the `difficulty=` token from the console line.
- Log the curriculum mean once, under one key.

## Acceptance test

Tiny CPU run with `--use_task_curriculum --alg_tokens 2000 --eval_interval 10`: for each task, `train_eval/curriculum/difficulty/<task>` equals the mean of that task's sampled difficulties (non-zero, differs between tasks); the IID payload has no `difficulty` field or one computed from `params`; each eval record contains `eval_difficulty` exactly once. Pure check: `AlgorithmicDataset(...).__iter__()` yields items with a `difficulty` key.

## Related

- #162: the IID grid's nominal 0.5 is already listed there as context (item g); the fix lives here.
- #167: the same flattened train-eval record carries the duplicate token-accuracy key.
- #141: the difficulty schedule whose realised value this key should have shown.
- #161, #187: eval JSON and run-record completeness.
- #242: the same train-eval block.

## Sources

- Audit report code-eval, finding E2
- Audit report artifacts, finding AR-8
- Audit report code-train-a, probe `p5_train_eval_oracle.py` and the #167 note
- Audit report gh-prs-a, finding A1 and per-PR table row #67
- `VERIFIED_BY_COORDINATOR.md`: `eval_hub.py:591` confirmed
- Audit report gemini-c, section 2.6 (Jan-11 log lines)
