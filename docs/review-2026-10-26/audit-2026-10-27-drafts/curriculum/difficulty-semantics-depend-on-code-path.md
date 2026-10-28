---
slug: difficulty-semantics-depend-on-code-path
title: Difficulty depends on code path; easy mix is a no-op for linear, phased, fixed
issue: 203
severity: P2
labels: [bug, P2-important, area: curriculum, correctness, reproducibility, audit-2026-10-27]
nf: [NF-11]
found_by: [code-train-a, gh-prs-a, gemini-b]
status: Demonstrated
related_issues: [#141, #162, #167, #169]
---

## Summary

The difficulty a run trains at is not defined by `--difficulty_schedule`; it is defined by which of three code paths the flag combination happens to select. Under `linear`, `phased` and `fixed` the training loader reads a shared `multiprocessing.Value` and gets one exact number per step, so `--easy_mix_frac` (default 0.2) produces no easy example at all. `data.py` has its own `linear` branch with a different formula, reachable only by the train-eval dataset. The logged `train/curriculum/difficulty` is a third definition: the noise-free centre of the `smooth` schedule, not the mean of what was sampled. Under `--use_task_curriculum`, `phased` and `fixed` are silently remapped to `linear`, which `generate_example` ignores. None of this is in #141, which covers the worker-local progress counter.

## What happens

Three definitions of "difficulty" at HEAD:

1. Shared-`Value` path. [train.py:2046-2051](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2046-L2051) creates `Value("d", 0.0)` only when the schedule is `linear`, `phased` or `fixed`, the task curriculum is off and `--fixed_data` is off. The training loader then returns the value unchanged at [data.py:1186-1193](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1186-L1193). The main loop writes `difficulty_schedule(...)` into it, which for `linear` is `0.5 + 0.5 * progress` ([train.py:1852-1853](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1852-L1853)). Audit probe `code-train-a/p2_difficulty.py`: `linear` at progress 0 / 0.5 / 1.0 samples exactly 0.500 / 0.750 / 1.000, no spread, no easy mix. The `--easy_mix_frac` help at [train.py:3796-3801](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3796-L3801) says 20 % of samples "should always be easy".
2. `data.py`'s own schedules. When no `Value` is present, [data.py:1198-1200](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1198-L1200) defines `linear` as `uniform(min_difficulty, 0.5 + 0.5 * progress)`, mean 0.25 to 0.50. The only dataset that reaches this branch is the procedural train-eval set built at [train.py:2279-2294](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2279-L2294), which receives `difficulty_value=None` and, when the task curriculum is off, `difficulty_fn=None`. So a `linear` run is scored on a "training" distribution the training loader never produced.
3. The logged value. `difficulty_schedule` at [train.py:1847-1877](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1847-L1877) feeds `train/curriculum/difficulty`. For the default `smooth` it is the deterministic sigmoid centre: 1.000 at progress 0.8 and above. The sampler ([data.py:184-220](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L184-L220)) adds Gaussian noise 0.05 and the easy mix; its measured mean at progress 0.8 is 0.813 with the default easy mix and 0.980 without (probe above).

The remap: with `--use_task_curriculum`, [data.py:1093-1095](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1093-L1095) sets `effective_schedule = "linear"` for `phased` and `fixed`, and `generate_example` ([data.py:934-942](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L934-L942)) only acts on `phased` and `fixed`. The flag is accepted, printed in the config, and ignored for training examples.

## Why it matters for the experiments

- The difficulty a run trained at cannot be read from the flag or from the log. Two runs that differ only in `linear` vs `smooth` also differ in whether 20 % of their data was easy.
- The `--fixed_data` grokking commands, the Dec 19 to Dec 24 runs and the DAG runs each sit on a different one of these paths, so "difficulty 0.5" in one log and "difficulty 0.5" in another are different distributions.
- #141's acceptance test ("num_workers 0 and 2 give the same difficulty histogram") cannot be written while three definitions exist; this issue has to be fixed in the same change.

## How it got here

- PR [#66](https://github.com/jpturcotte/ReLSM/pull/66) (Dec 20) added `phased` and the shared-`Value` plumbing.
- PR [#69](https://github.com/jpturcotte/ReLSM/pull/69) (Dec 20) added `smooth` (made the default), `warmup_ramp` and `--easy_mix_frac`, and limited the `Value` to `linear`/`phased`/`fixed`. gh-prs-a's read of the diff: "`difficulty_schedule()` in train.py:1847 recomputes 'smooth' for the log without the easy-mix noise, so the logged difficulty is not what the workers sample."
- PR [#85](https://github.com/jpturcotte/ReLSM/pull/85) (Dec 21) added `difficulty_fn` and the `phased`/`fixed` to `linear` remap; gemini-b traced the remap to the diff reviewed in chat `86665c16` (lines 606-610). Tests: "Not run".

## How we verified it, step by step

1. code-train-a ran audit probe `code-train-a/p2_difficulty.py`: `train.py:difficulty_schedule` against `data.py:_sample_difficulty` for the five schedules, with and without easy mix, with and without the shared `Value`. Numbers above (0.500/0.750/1.000 deterministic; `smooth` mean 0.813 vs logged 1.000).
2. The coordinator re-read `train.py:1853`, `data.py:1198-1200`, `train.py:2047-2051` and `train.py:2286` and confirmed the three definitions (`VERIFIED_BY_COORDINATOR.md`).
3. gh-prs-a (A8) read PR #69's diff for the logged-centre point; gemini-b (G-B2) read the remap at `data.py:1093-1095`. Both are code reads; the probe covers the numbers.

## Suggested fix

- One difficulty sampler, called from one place, driven by the shared progress counter (the #141 fix). Delete the duplicate `linear`/`phased`/`fixed` branches in `data.py` or make them the only definition.
- Put the sampled `difficulty` into the example payload and log the mean of what was generated per window; `train/curriculum/difficulty` becomes a measured value. This also fills the always-zero `difficulty_by_task` column noted under #167.
- Validate at startup: `phased`/`fixed` with `--use_task_curriculum`, and `--easy_mix_frac > 0` with `linear`/`phased`/`fixed`, either error or warn once.
- Depends on #141 (same code); does not conflict with #157.

## Acceptance test

Pure Python, `num_workers=0`: for each schedule, build the training `AlgorithmicDataset` as `train.py` does, draw 2,000 examples at progress 0.5, and check (a) the mean payload difficulty is within 0.02 of the value logged for that schedule, and (b) the share of examples with difficulty below 0.3 is at least `easy_mix_frac - 0.02`. Today (a) fails for `smooth` and (b) fails for `linear`, `phased` and `fixed`.

## Related

- #141: worker-local progress; add that the train-eval dataset never receives the `Value`.
- #162: the IID grid is a fourth distribution.
- #167: `difficulty_by_task` is 0.0 because no payload carries a difficulty.
- #169: README describes the schedules as if one definition existed.
- #215: the same flags under `--fixed_data`.
- #219: the easy-mix draw's RNG under the task curriculum.

## Sources

- Audit report code-train-a, finding T1; audit probe `code-train-a/p2_difficulty.py`
- Audit report gh-prs-a, finding A8 (PR #69 row)
- Audit report gemini-b, finding G-B2 (chat `86665c16`)
- `VERIFIED_BY_COORDINATOR.md`, "Difficulty definitions"
