---
slug: addition-carry-deterministic
title: Addition carries are deterministic outside the medium difficulty bucket
issue: 208
severity: P2
labels: [bug, P2-important, area: data, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-35]
found_by: [code-data]
status: Demonstrated
related_issues: [#155, #162, #141, #140]
---

## Summary

`AlgorithmicGenerator.addition` chooses the digit bounds of both operands from the difficulty bucket: `(0, 4)` below 0.3, `(0, 9)` from 0.3 to 0.7, `(5, 9)` at 0.7 and above. With `(5, 9)` every column sum is at least 10, so every column carries. With `(0, 4)` no column can carry. Only the medium bucket contains columns whose carry-out depends on the carry-in. The final training phase (3–6 digits, labelled "high carry pressure") never asks the model to decide whether to carry, and the easy mix never produces a carry at all. The IID grid runs at difficulty 0.5, the one bucket where carry is a real decision.

## What happens

- [data.py:443-452](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L443-L452): `_difficulty_bucket(difficulty)` selects `default_digit_bounds` `(0, 4)` ("low carry pressure"), `(0, 9)` or `(5, 9)` ("high carry pressure").
- [data.py:457-463](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L457-L463): both operands are drawn with `_sample_int_with_digit_bounds(rng, digits, *active_bounds)`, so every digit of both numbers lies inside the bounds. The phase tables pass `digit_range` only; `digit_bounds` stays at the bucket default.
- With bounds `(5, 9)` the smallest column sum is 5 + 5 = 10: every column carries. With `(0, 4)` the largest is 4 + 4 = 8: no column carries.
- Audit probe `code-data/probe_generators.py`, 2,000 examples per difficulty: `all_columns_carry` 2,000/2,000 at d = 0.8, 0.95, 1.0; `any_carry` 0/2,000 at d = 0.05 and 0.2. Columns whose carry-out depends on the carry-in (digit sum exactly 9) appear in 0 examples outside the medium bucket, and in 348/2,000 (17 %), 431/2,000 (22 %) and 443/2,000 (22 %) at d = 0.35, 0.5, 0.65.
- The shortcut rate `no_carry_digitwise_sum` (answer equals the digit-wise sum with no carry) is 1.0 at d = 0.05 and 0.2 and 0.0 at d >= 0.8.

## Why it matters for the experiments

- At the final phase the algorithm that scores 100 % is "add the two digits in each column, add 1 to every column but the last, prepend 1". No carry decision is needed. Addition accuracy at difficulty >= 0.7 does not measure carry handling.
- The IID addition condition (difficulty 0.5, bounds `(0, 9)`) needs a different algorithm from the final training phase. The addition IID/training gap in #155 and #162 is not only a digit-count gap.
- Under the DAG, addition sat locked at difficulty 0.2 for the whole of DAG-v12 (77,500 micro-steps). Every addition example it saw was carry-free (easy bucket, bounds `(0, 4)`); its train-eval raw accuracy was 0.52 at the end and its IID accuracy 0.0.
- The 20 % easy mix (`uniform(0, 0.3)`) is carry-free in every run that uses it.
- #155's overlap between `ood_6digit` and the final phase is in length only: the phase's 6-digit sums carry in every column, the OOD condition's are mixed.

## How it got here

- PR [#17](https://github.com/jpturcotte/ReLSM/pull/17) (2025-12-12, merged as `b2aba79`, title "Bias Dyck corruption toward early violations"). The Codex task asked, among five changes, to "rework addition (carry pressure)". The diff added the 1–2 / 2–3 / 4-digit buckets with carry bounds and `_sample_int_with_digit_bounds`. The PR title and body describe only the Dyck bullet. Test line: `pytest tests/test_curriculum_sampler.py` (a module-level skip without torch).
- PR [#18](https://github.com/jpturcotte/ReLSM/pull/18) had an explicit carry axis in the eval grid ("OOD-carry: digits=4, digit_bounds=(7,9)"); PR [#66](https://github.com/jpturcotte/ReLSM/pull/66) rebuilt the grid from `TASK_DIFFICULTY_CONFIG` and dropped it.
- PR [#100](https://github.com/jpturcotte/ReLSM/pull/100) widened the final phase to `(3, 6)` digits and left the bucket bounds as they were.

## How we verified it, step by step

1. code-data read `addition` end to end and listed the three bucket bounds (D1).
2. Audit probe `code-data/probe_generators.py` generated 2,000 examples at each of eight difficulties with the real `EleutherAI/llemma_7b` tokenizer, recomputed every sum (0 wrong), and counted per example: any carry, all columns carry, number of carry-decision columns. Numbers above; raw histograms in `probe_generators.json`.
3. The coordinator re-read `data.py:443-461` and confirmed: hard bucket `(5, 9)` for every digit of both operands, easy bucket `(0, 4)`, both operands share `digits`.
4. Disagreements: none.

## Suggested fix

- Draw digits from `(0, 9)` at every difficulty and make carry pressure an explicit, sampled quantity: choose the number of carry columns (and of digit-sum-9 columns) per example, the way `chain` samples its trailing-9 run. Report the realised carry rate per phase in the startup data report.
- State the bounds in the IID condition (#162) so the eval distribution is written down.
- After this change the final phase and `ood_6digit` will match in carry structure as well as length, which makes #155's overlap complete; fix #155 together with this.
- Independent of #210, but the same generator lines are touched.

## Acceptance test

Pure Python, no torch:

```python
import random
from data import AlgorithmicGenerator
rng = random.Random(0)
for d in (0.05, 0.5, 1.0):
    decision_cols = 0; cols = 0
    for _ in range(2000):
        ex = AlgorithmicGenerator.addition(rng=rng, difficulty=d)
        a, b = [t for t in ex["input"].replace("+", " ").replace("?", " ").replace(".", " ").split() if t.isdigit()][:2]
        for x, y in zip(a[::-1], b[::-1]):
            cols += 1
            decision_cols += (int(x) + int(y) == 9)
    assert decision_cols / cols >= 0.05, (d, decision_cols / cols)
```

At HEAD the ratio is 0.0 at d = 0.05 and d = 1.0.

## Related

- #155: 6-digit "OOD" overlaps the final phase; the overlap is length-only today.
- #162: IID grid is not the training distribution; IID addition has carry decisions, the final phase has none.
- #141: with workers the schedule stays at the easiest phase, i.e. carry-free addition.
- #140: the easy fifth of the stream is carry-free and small.
- #210: the same lines also force equal operand lengths.
- #209: the easy phases this generator falls into have a small support.

## Sources

- Audit report code-data, finding D1 and `probe_generators.py`
- Coordinator re-checks (addition generator bounds)
- Audit report codex-a, task table rows 17 and 18 (carry pressure request; dropped OOD-carry axis)
- Audit report gh-prs-a, PR #17 row
- Merged findings NF-35; extension rows for #155 and #162

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `data/addition-carry-deterministic`.*
