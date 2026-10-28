---
slug: operands-equal-digit-count
title: Operands always share a digit count in addition, multiplication and compare
issue: 210
severity: P2
labels: [bug, P2-important, area: data, area: evaluation, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-36]
found_by: [code-data]
status: Demonstrated
related_issues: [#176, #156, #155, #162]
---

## Summary

In `addition`, `multiplication` and `comparison` one `digits` value is drawn and both operands get exactly that many digits. In 2,000 examples per difficulty at all eight difficulties, the two operands had the same length 2,000/2,000 times for all three tasks. Compare can never be decided by length and never tests it; its first differing digit is the leading digit in 89 % of examples. Addition never shows ragged operands, so alignment is never learned. `utils.build_dataset` passes the same parameters, so the IID and OOD grids inherit the restriction.

## What happens

- [data.py:455-463](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L455-L463): `digits = rng.randint(...)` once; `a` and `b` are both sampled with that `digits`.
- [data.py:498-507](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L498-L507) (multiplication) and [data.py:809-818](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L809-L818) (compare): `low = 10**(digits-1)`, `high = 10**digits - 1`, then `a = rng.randint(low, high)`, `b = rng.randint(low, high)`.
- [utils.py:970-974](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L970-L974): `build_dataset` forwards the condition's `params` (`digits` / `digit_range`) unchanged, so every IID and OOD condition for these tasks is also equal-length.
- Audit probe `code-data/probe_generators.py` (2,000 per difficulty): `same_len` true 2,000/2,000 for addition, multiplication and compare at d = 0.05 … 1.0. Compare shortcuts: `by_length` 0.0 (length never decides), `first_digit_decides` 0.877–0.902.

## Why it matters for the experiments

- Compare: a rule that reads one digit per operand scores 89 % at every difficulty. The DAG-v12 last-eval compare accuracy of 0.912 should be read against that ceiling, not against 33 % chance. Compare is a DAG root, so this shortcut also gates successor.
- Addition: the model never sees `1234 + 7`. Right-alignment of operands of different lengths, the part of addition that needs position handling, is in neither the training nor the eval distribution.
- Multiplication: the four phases never mix a 1-digit with a 3-digit operand.
- The OOD grid (`ood_6digit`, multiplication 4/5 digits) tests longer equal-length operands only, so a "length generalisation" result on these tasks says nothing about mixed lengths.

## How it got here

- PR [#16](https://github.com/jpturcotte/ReLSM/pull/16) (2025-12-12, "Add difficulty-aware algorithmic data templates", commit `ea7f1d6`) introduced the single `digits` draw followed by `a = rng.randint(low, high)` and `b = rng.randint(low, high)` for compare, multiplication and successor.
- PR [#17](https://github.com/jpturcotte/ReLSM/pull/17) (2025-12-12) gave addition the same shape with `_sample_int_with_digit_bounds` called twice on one `digits`.
- No Codex task asked for operands of different lengths; no later PR changed this.

## How we verified it, step by step

1. code-data read the three generators and the `build_dataset` path (D2).
2. Audit probe `code-data/probe_generators.py` counted `digits_a`, `digits_b` and `same_len` over 2,000 examples at each of eight difficulties for the three tasks: `same_len` 2,000/2,000 everywhere; the per-digit histograms of `a` and `b` are identical (addition at d = 1.0: 3 digits 479, 4 digits 539, 5 digits 511, 6 digits 471 for both operands).
3. The same probe scored the compare shortcuts: the first digit decides in 1,754–1,804 of 2,000 examples per difficulty.
4. The coordinator confirmed "both operands share `digits`" while re-reading `data.py:443-461` for #208.
5. Disagreements: none.

## Suggested fix

- Draw the two operand lengths independently inside the phase's `digit_range` (keep an option for equal lengths if a condition wants it).
- Add `length_mismatch_frac` per task and phase to the startup data report.
- Add one mixed-length condition per task to `eval/ood_grid.py`; it costs nothing and is an OOD axis the grid does not have.
- Touches the same lines as #208; do both in one change. Add the compare half to #176.

## Acceptance test

Pure Python, no torch:

```python
import random
from data import AlgorithmicGenerator
rng = random.Random(0)
for gen in (AlgorithmicGenerator.addition, AlgorithmicGenerator.multiplication, AlgorithmicGenerator.comparison):
    mismatched = 0
    for _ in range(2000):
        ex = gen(rng=rng, difficulty=1.0)
        nums = [t for t in ex["input"].replace("?", " ").replace(",", " ").replace(".", " ").split() if t.isdigit()]
        mismatched += len(nums[0]) != len(nums[1])
    assert mismatched / 2000 >= 0.3, (gen.__name__, mismatched)
```

At HEAD `mismatched` is 0 for all three.

## Related

- #176: compare rarely emits `=`; add the equal-length restriction and the 89 % first-digit ceiling.
- #156: compare `=` is never supervised; together with this, compare is a one-digit lookup in practice.
- #155 and #162: the IID and OOD addition conditions are equal-length too.
- #208: same generator, carry structure.

## Sources

- Audit report code-data, finding D2 and `probe_generators.py` / `probe_generators.json`
- Coordinator re-checks (addition generator)
- Merged findings NF-36; extension rows for #155 and #176

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `data/operands-equal-digit-count`.*
