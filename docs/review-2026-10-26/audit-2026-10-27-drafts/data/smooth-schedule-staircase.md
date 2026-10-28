---
slug: smooth-schedule-staircase
title: sample_difficulty_smooth is a four-plateau staircase, not a smooth ramp
issue: 223
severity: P3
labels: [bug, P3-minor, area: data, area: curriculum, correctness, audit-2026-10-27]
nf: [NF-39]
found_by: [code-data]
status: Demonstrated
related_issues: [#141, #169, #140]
---

## Summary

The default difficulty schedule blends four plateaus (0.25, 0.45, 0.70, 1.0) with a sigmoid whose width is `transition_width / 4` = 0.0125. The deterministic value moves 0.28 → 0.42 between progress 0.13 and 0.17, is 0.95 at 0.62 and 1.00 from 0.8. With noise 0.05 and clipping, 40 % of draws at progress >= 0.7 are exactly 1.0. The last 38 % of the algorithmic budget trains at the final phase plus the easy mix; the middle phases of the 5–7-phase tables occur 0–2 % of the time. This is a design mismatch, not a crash, but "curriculum" in a non-task-curriculum run means four jumps.

## What happens

- [data.py:184-218](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L184-L218): boundaries `[0.15, 0.35, 0.60]`, targets `[0.25, 0.45, 0.70, 1.0]`; at [data.py:211-212](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L211-L212) `transition_width = 0.05` and `blend = sigmoid((progress - boundary) / (transition_width / 4))`, then Gaussian noise 0.05 and clipping to [0, 1].
- Audit probe `code-data/probe_schedules.py`, deterministic part: 0.250 at progress 0–0.10, 0.284 at 0.13, 0.416 at 0.17, 0.446 at 0.20, 0.492 at 0.33, 0.658 at 0.37, 0.750 at 0.58, 0.950 at 0.62, 1.000 at 0.80.
- With noise and easy mix, 20,000 draws at progress 1.0: 3,964 easy bucket, 16,036 hard bucket, 0 medium; 40.3 % of draws exactly 1.0.
- Realised phase mix at progress 1.0: copy `(10, 12)` 79.8 %, the rest easy-mix phases; copy `(4, 8)` and `(6, 10)` 0 %; addition `(1, 5)` 1.9 %; successor `(16, 26)` 0.1 %.
- Plateau-to-phase map: copy (3,6) → (4,8) → (8,12) → (10,12); addition (1,3) → (1,4) → (1,5) → (3,6); successor (4,12) → (8,16) → (16,26) → (20,32).

## Why it matters for the experiments

- The 5–7-phase tables are traversed in the first 15–35 % of a run; the remaining 62–85 % is the final phase plus 20 % easy mix. A run's "curriculum" reads as four steps, with a 1.0 plateau from progress 0.62.
- The Dec 19–24 non-DAG runs (`smooth`, `easy_mix 0.2`) are the ones affected. With workers the schedule never leaves plateau one (#141): the Dec-21 run logged difficulty 0.250001 at step 50 and 0.250026 at step 3,000.
- Anyone reading the "difficulty" log line of those runs as a continuous ramp over-reads it.

## How it got here

- PR [#69](https://github.com/jpturcotte/ReLSM/pull/69) (2025-12-20, "Improve LR schedules, add smooth difficulty curriculum, and harden numeric parsing", commit `ea3b820` in the branch). The task specified "boundaries 0.15/0.35/0.60, targets 0.25/0.45/0.70/1.0, width 0.05, gauss noise 0.05"; the code divides the width by 4. Testing: "Not run (not requested)". The default schedule became `smooth` in the same PR.

## How we verified it, step by step

1. code-data read `sample_difficulty_smooth` and `get_difficulty_params` (D7).
2. Audit probe `code-data/probe_schedules.py` evaluated the deterministic value at 16 progress points, drew 20,000 samples at each of 15 progress points with the easy mix, mapped each plateau to each task's phase table, and tabulated the realised phase mix at progress 1.0 (numbers above, `probe_schedules.out.txt`).
3. Disagreements: none. The code-data report rates it a design mismatch rather than a bug; the severity stays P3.

## Suggested fix

- Either document the four plateaus and the 1.0 plateau from progress 0.62, or remove the `/ 4` so the blend width is the stated 0.05 (transitions then span about 0.2 of progress), or interpolate linearly between plateaus.
- Add the realised phase mix per task to the startup data report, so a run record states which phases were trained and for what share.
- Independent of other fixes; #209 changes the easy-mix draw inside the same function.

## Acceptance test

Pure Python, no torch:

```python
import random
from data import sample_difficulty_smooth, get_difficulty_params
rng = random.Random(0)
phases = {}
for i in range(20000):
    p = i / 20000
    d = sample_difficulty_smooth(p, easy_mix_frac=0.0, rng=rng)
    key = tuple(sorted(get_difficulty_params("copy", d).items()))
    phases[key] = phases.get(key, 0) + 1
assert min(phases.values()) / 20000 >= 0.03, phases
```

At HEAD copy's `(4, 8)` and `(6, 10)` phases are below 3 % over the whole run.

## Related

- #141: with workers the schedule stays at plateau one (0.25).
- #169: README describes a smooth curriculum.
- #140: the replay interacts with the fixed 1.0 plateau (the same final-phase stream, replayed).
- #209: the 20 % easy draw in the same sampler.

## Sources

- Audit report code-data, finding D7; `probe_schedules.py`, `probe_schedules.out.txt`
- Audit report codex-b, PR #69 paragraph
- Merged findings NF-39; extension row for #141

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `data/smooth-schedule-staircase`.*
