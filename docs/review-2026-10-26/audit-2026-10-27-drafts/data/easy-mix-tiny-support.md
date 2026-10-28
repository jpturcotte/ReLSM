---
slug: easy-mix-tiny-support
title: Easy phases have a tiny support; the 20% easy mix is memorisable by construction
issue: 209
severity: P2
labels: [bug, P2-important, area: data, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-38]
found_by: [code-data]
status: Demonstrated
related_issues: [#140, #169, #141, #158, #157]
---

## Summary

The easiest phases of the task tables have very few distinct examples. In 2,000 generated examples at difficulty 0.05, 88 % of parity texts are duplicates (178 distinct operand/target cores; there are only 60 bit strings of length 2–5), dyck 70 %, addition 58 %, chain 57 %, copy and reverse 50 %. Every difficulty sampler keeps drawing from these phases for the whole run: `sample_difficulty_smooth`, `sample_difficulty_warmup_ramp` and `CurriculumDifficultySampler` return `uniform(0, 0.3)` with probability `easy_mix_frac` (default 0.2). One fifth of the default stream is a few hundred strings per task, seen thousands of times. The DAG runs set `easy_mix_frac 0.0` but pinned six locked tasks at difficulty 0.2 for the whole run, which is the same small support by another path.

## What happens

- [data.py:97-106](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L97-L106): parity phase 1 is `length_range (2, 5)`; phase 2 `(4, 8)`.
- [data.py:202-203](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L202-L203) and [data.py:235-236](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L235-L236): `if rng.random() < easy_mix_frac: return rng.uniform(0.0, 0.3)` in both schedule samplers, at every progress.
- [train.py:216-217](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L216-L217): the task-curriculum sampler does the same with the global `random`.
- Audit probe `code-data/probe_generators.py`, duplicate rate of the exact text in 2,000 examples at d = 0.05: parity 0.882 (0.511 at d = 0.2), dyck 0.701 (0.637 at 0.2), addition 0.584 (0.341 at 0.2), chain 0.575, copy 0.501, reverse 0.501, successor 0.397, multiplication 0.385 (0.383–0.400 for d <= 0.35), compare 0.351. Ignoring the prompt template, parity has 178 distinct cores in 2,000.
- Audit probe `code-data/probe_batching.py`, one 4,000-example pass under the default `smooth` schedule: 3.4 % of examples repeat an earlier example of the same pass (multiplication 8.1 %, addition 5.0 %, parity 5.0 %).
- DAG-v12 restarted its 100,000-example iterator about 92 times (#140); the Dec 19–24 runs used `smooth` with `easy_mix_frac 0.2`.

## Why it matters for the experiments

- A grokking study has to know how much of the stream is memorisable. Here one fifth is, by construction, before any replay. The "infinite procedural data" framing (#140, #169) is wrong twice: the stream replays (#140), and its easy fifth has a support of a few hundred strings per task.
- #140's acceptance test ("epochs share no examples") cannot pass for parity, dyck or compare at the easy phases without changing the tables.
- DAG runs: with `easy_mix_frac 0.0` the mix is gone, but the six locked tasks sat at difficulty 0.2 for 77,500 micro-steps in DAG-v12. At d = 0.2 the duplicate rate per 2,000 is parity 51 %, dyck 64 %, addition 34 %, multiplication 40 %: the locked tasks were trained, at a 2 % share, on a small repeated set.
- With workers the default schedule never leaves the first plateau (#141), so a worker run trains on the small-support phases only.

## How it got here

- PR [#69](https://github.com/jpturcotte/ReLSM/pull/69) (2025-12-20, "Improve LR schedules, add smooth difficulty curriculum, and harden numeric parsing", commit `f9514a3`) added `sample_difficulty_smooth` and `--easy_mix_frac 0.2` as the default ("guaranteed easy mixing" in the docstring). Testing: "Not run (not requested)".
- PR [#101](https://github.com/jpturcotte/ReLSM/pull/101) (2025-12-22, "Respect easy mix in curriculum difficulty") copied the same draw into the task-curriculum sampler; the stated intent was "20% Easy, 10% Replay, 70% Frontier". Testing: "not run".
- The phase tables with `(2, 5)`-bit parity and depth-`(1, 2)` dyck come from PR [#66](https://github.com/jpturcotte/ReLSM/pull/66) and PR [#113](https://github.com/jpturcotte/ReLSM/pull/113). No task asked how many distinct examples a phase has.

## How we verified it, step by step

1. code-data read the three samplers and the phase tables (D4).
2. Audit probe `code-data/probe_generators.py` generated 2,000 examples per task at each of eight difficulties and counted distinct texts and distinct cores (numbers above; `probe_generators.json` has `distinct_texts`, `distinct_cores`, `dup_rate_text`, `dup_rate_core` per task and difficulty).
3. Audit probe `code-data/probe_batching.py` iterated `AlgorithmicDataset` for one 4,000-example pass at the defaults and counted exact repeats within the pass: 3.4 % overall, per task above.
4. Arithmetic: bit strings of length 2–5 number 4 + 8 + 16 + 32 = 60 (62 counting length 1); the code-data report says 62.
5. Disagreements: none.

## Suggested fix

- Print the distinct-example support per task and phase at startup (the generator can enumerate or sample it in seconds on CPU) and write it to the run record (#187).
- Draw the easy mix from a phase with a large support (parity length >= 8, compare 3 digits, dyck depth >= 3) instead of `uniform(0, 0.3)`, or dedupe within a pass by text hash.
- For the DAG, raise the locked-task difficulty floor from 0.2 to a phase with a large support, or dedupe there too (#157).
- Depends on #140's fix for the replay half; the two together decide how many distinct examples a run sees.

## Acceptance test

Pure Python, no torch:

```python
import random
from data import AlgorithmicGenerator, sample_difficulty_smooth
rng = random.Random(0)
gens = AlgorithmicGenerator._get_generators()
for task in ("parity", "dyck", "addition", "compare"):
    texts = set()
    for _ in range(2000):
        d = sample_difficulty_smooth(1.0, easy_mix_frac=1.0, rng=rng)  # the easy-mix draw only
        texts.add(gens[task](rng=rng, difficulty=d)["text"])
    assert len(texts) / 2000 >= 0.9, (task, len(texts))
```

At HEAD parity gives about 0.3 and dyck about 0.4.

## Related

- #140: the stream replays; this is the other half of the repetition.
- #169: README's "infinite data, no overfitting".
- #141: with workers the schedule stays at the small-support plateau.
- #158: `lang_loader` aliasing replays the same stream again.
- #157: the DAG locked-task floor at difficulty 0.2 has the same support problem.
- #223: the schedule that carries the easy mix.
- #208: the easy addition phase is also carry-free.

## Sources

- Audit report code-data, finding D4; `probe_generators.py`, `probe_generators.json`, `probe_batching.py`
- Audit report codex-b, PR #69 and #101 paragraphs; gh-prs-b PR #101 row
- Merged findings NF-38; extension row for #140 / #158

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `data/easy-mix-tiny-support`.*
