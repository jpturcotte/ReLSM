---
slug: jitter-and-easy-mix-use-global-random
title: Easy-mix and jitter draws use the global random module, not the dataset RNG
issue: 219
severity: P3
labels: [bug, P3-minor, area: curriculum, reproducibility, audit-2026-10-27]
nf: [NF-19]
found_by: [code-data, gh-prs-b, gemini-b, claude-b1]
status: Demonstrated
related_issues: [#143, #140, #160]
---

## Summary

Under the task curriculum the per-example difficulty is decided by `CurriculumDifficultySampler.__call__`, which draws the 20 % easy mix and the 10 % jitter from the global `random` module; `TaskCurriculumState.get_difficulty` does the same. The dataset's own seeded `random.Random(seed + worker_id)` is used for the task and the operands but not for the difficulty. Under `--seed` the task and operand stream is reproducible and the difficulty stream is not. With `num_workers 0` the stream is shared with everything else in the main process and is reset by the eval reseed (#143); with workers it is torch's per-worker seed. Claude flagged this twice on Dec 21; no issue asks to fix it.

## What happens

- [train.py:216-222](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L216-L222): `random.random() < easy_mix_frac` then `random.uniform(0.0, 0.3)`; `random.random() < curriculum_jitter` then `random.uniform(0.0, difficulty)`.
- [curriculum.py:145-146](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L145-L146): the same two calls in `get_difficulty`.
- The loader passes its seeded `rng` to `generate_example` but calls `self.difficulty_fn(task)` with no rng at [data.py:1092](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1092).
- Audit probe `code-data/probe_schedules.py` (20,000 draws, easy mix 0.2, jitter 0.1): for a snapshot difficulty of 0.65 the realised difficulty is exactly 0.65 in 71.6 % of draws and lands in the easy bucket (below 0.3) in 24.1 %; for a snapshot of 1.0, 72.3 % at 1.0 and 633 of 20,000 in the medium bucket. So 28 % of every task's examples are at a difficulty that no seed controls.
- Consequences: two runs with the same `--seed` differ in which examples were easy; after every eval the global stream restarts (#143), so the same easy-mix and jitter pattern replays each interval; the #140 replay between epochs is exact for operands and inexact for difficulty, which is why #140 and #143 both mention "the curriculum's jitter" as the reason passes differ.

## Why it matters for the experiments

- A grokking study on a fixed or replayed stream wants the whole stream reproducible; today the 28 % easy-or-jittered share is not, and it is reset by the eval cadence.
- Any A/B of two seeds confounds the seed with a different easy-mix pattern of about a fifth of the data.

## How it got here

- PR [#85](https://github.com/jpturcotte/ReLSM/pull/85) (Dec 21) wrote `get_difficulty` with `random.random()`; the review chat the same day (`2025-12-21_3a6ca91d_Task_curriculum_quality_and_integration.md`, lines 45 and 217) said it "uses global `random` module ... should accept `rng` parameter" twice (claude-b1). gemini-b traces the same line to the diff reviewed in chat `86665c16` (L349-351).
- PR [#93](https://github.com/jpturcotte/ReLSM/pull/93) (Dec 21) moved the per-example draw into the picklable `CurriculumDifficultySampler` and copied the global calls. PR [#105](https://github.com/jpturcotte/ReLSM/pull/105) (Dec 23) reverted a floor on the jitter but kept the RNG.

## How we verified it, step by step

1. code-data ran `code-data/probe_schedules.py` (the sampler-style draw at snapshots 0.2, 0.5, 0.65 and 1.0; bucket shares and means).
2. gh-prs-b (A-6), gemini-b (G-B1) and claude-b1 read the two sites; gemini-b states what would settle the reproducibility claim end to end: two 50-step `train.py --seed 42 --num_workers 0 --use_task_curriculum` runs with the per-example difficulty logged, which the current logging does not emit. The bucket numbers above are measured; the cross-run divergence is read from the code.

## Suggested fix

- Change the sampler signature to `difficulty_fn(task, rng)` and pass the dataset's `rng` from `data.py:1092`; do the same for `get_difficulty`. Or give each sampler a private `random.Random(seed + worker_id + k)`.
- Then `--seed` reproduces the whole stream and the #140 replay becomes exact. Depends on #143 (the eval reseed) for full-run reproducibility; small enough to land with #220 in one change.

## Acceptance test

Pure Python, `num_workers 0`: build two identical `AlgorithmicDataset`s with a `CurriculumDifficultySampler`, call `random.seed(1)` before the first pass and `random.seed(2)` before the second, and compare the per-example difficulty sequences of 2,000 examples. They must be identical (today they differ in about 28 % of positions).

## Related

- #143: the eval reseed restarts this stream; cite `train.py:216-222` and `curriculum.py:145-146` there.
- #140: replay between epochs is exact except for these draws.
- #160: another seeding defect (salted `hash()`), on the eval side.
- #220: what the jitter draws.
- #203: the easy mix on the non-curriculum paths.

## Sources

- Audit report code-data, finding D9; audit probe `code-data/probe_schedules.py`
- Audit report gh-prs-b, finding A-6 and PR #85 row
- Audit report gemini-b, finding G-B1 (chat `86665c16`)
- Audit report claude-b1, #143 row (chat `2025-12-21_3a6ca91d_Task_curriculum_quality_and_integration.md`)
