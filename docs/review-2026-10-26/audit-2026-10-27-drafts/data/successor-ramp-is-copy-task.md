---
slug: successor-ramp-is-copy-task
title: Successor at 20-32 digits is a copy task; long-range carry occurs in 0.05 %
issue: 211
severity: P2
labels: [bug, P2-important, area: data, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-37]
found_by: [code-data]
status: Demonstrated
related_issues: [#139, #162, #142, #164]
---

## Summary

`successor` draws `n = rng.randint(0, 10**digits - 1)`, so the run of trailing 9s has length k with probability 10^-k, whatever `digits` is. At every difficulty 89–91 % of examples have no trailing 9, 99 % have at most one, and 0.00–0.25 % have three or more. The comment at `data.py:58` says the ramp to 32 digits exists "to force long-range carry propagation"; the generator never produces it. What the ramp produces is a 26-token answer that copies the prompt digits and increments the last one, which is exactly what overflows `alg_seq_len = 48`.

## What happens

- [data.py:57-70](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L57-L70): seven successor phases from `(1, 4)` to `(20, 32)` digits, `train_max 32`, `ood_start 33`, with the comment "INCREASED DIFFICULTY: Ramps up to 32 digits to force long-range carry propagation."
- [data.py:850-857](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L850-L857): `digits` is drawn from the phase range, `max_n = min(max_val, 10**digits - 1)`, `n = rng.randint(0, max_n)`. Nothing controls the trailing digits.
- Audit probe `code-data/probe_generators.py`, 2,000 examples per difficulty, trailing-9 run length: 0 in 1,781–1,817 (89.1–90.9 %); at most 1 in 1,976–1,986 (98.8–99.3 %); 3 or more in 0–5 (0.00–0.25 %). The shortcut "copy the prefix and increment the last digit" scores 0.891–0.909 at every difficulty.
- Length at the final phase (d = 0.95): total tokens p50 61, max 75; answer p50 26 tokens; 1,944/2,000 examples longer than 48 tokens (1,955 at d = 1.0). At d = 0.65 it is 223/2,000, at 0.8 1,261/2,000.
- `chain` is the task that samples its carry run explicitly (its base ends in a chosen run of 9s).

## Why it matters for the experiments

- The successor curriculum from phase 3 on measures copying longer digit strings inside a 48-token window, not carrying. Its `ood_33digit`/`ood_35digit`/`ood_37digit` conditions test copying length as well.
- #139 counts successor as the fourth truncated task (48 % of its examples in a default pass at 48 tokens). Raising `alg_seq_len` for successor buys longer copies, not carry practice.
- The Dec-23 `5steps`/`20steps` files score successor 1.0 at the IID point (8–16 digits): that is copy-plus-increment, consistent with this structure. DAG-v12 left successor locked at difficulty 0.2 (2–8 digits) and 0.0 IID.
- A grokking study that wants a carry circuit needs examples where the carry chain length varies; this generator supplies 1 % with a chain of length 2 and almost none longer.

## How it got here

- PR [#113](https://github.com/jpturcotte/ReLSM/pull/113) (2025-12-26, "Ramp successor to 32 digits and add granular rungs for sequence tasks", commit `a74b31d`) added the seven-phase ramp and the comment. codex-b: the owner's config was pasted verbatim and matches `data.py:35-150`; Testing: "Not run". The run that day used `--alg_seq_len 32`.
- PR [#114](https://github.com/jpturcotte/ReLSM/pull/114) (2025-12-27, "Align Chain task to Carry Propagation") gave `chain` an explicit carry run and left successor as it was.
- The sampling line itself dates from PR [#16](https://github.com/jpturcotte/ReLSM/pull/16) (2025-12-12), when successor was capped at 6 digits.

## How we verified it, step by step

1. code-data read the successor generator and the phase table (D3).
2. Audit probe `code-data/probe_generators.py` generated 2,000 examples at each of eight difficulties, verified every answer (0 wrong), and built the trailing-9 histogram per difficulty (raw counts in `probe_generators.json`; for example d = 1.0: {0: 1,794, 1: 192, 2: 13, 3: 1}).
3. The same probe measured token lengths with the real tokenizer: 1,944/2,000 over 48 tokens at d = 0.95. This matches #139's second comment.
4. The audit probe `code-data/probe_batching.py` whole-pass view: successor contributed 170 of 623 truncated examples in a 4,000-example pass at 48 tokens.
5. Disagreements: none.

## Suggested fix

- Sample the trailing-9 run length explicitly (for example uniform in `[0, digits - 1]` or a phase parameter `carry_run_range`), as `chain` does, and keep the digit count at what the carry structure needs (8–12 digits fit 48 tokens).
- Rewrite the comment at `data.py:58` to say what the ramp does.
- Make the OOD axis for successor the carry-run length, not the digit count (#162, #142: the 33/35/37-digit conditions are impossible under 32 new tokens anyway).
- Independent of other fixes; it reduces #139's truncation count for successor to zero at 48 tokens.

## Acceptance test

Pure Python, no torch:

```python
import random, re
from data import AlgorithmicGenerator
rng = random.Random(0)
runs = []
for _ in range(2000):
    ex = AlgorithmicGenerator.successor(rng=rng, difficulty=1.0)
    n = re.findall(r"\d+", ex["input"])[0]
    runs.append(len(n) - len(n.rstrip("9")))
assert sum(r >= 2 for r in runs) / 2000 >= 0.2, sum(r >= 2 for r in runs)
```

At HEAD the fraction with a run of 2 or more is about 0.007.

## Related

- #139: `alg_seq_len = 48` truncation; successor is the fourth truncated task and gains nothing from a longer window.
- #162: successor OOD conditions sample 9–10 % in-range operands; the OOD axis should be the carry run.
- #142: 33/35/37-digit conditions cannot be answered in 32 new tokens.
- #164: numeric distance is 1.0 for 16+ digits, so the 20–32-digit phase also breaks the soft metric.
- #208: the other carry task has the opposite defect (carry is forced, not decided).

## Sources

- Audit report code-data, finding D3; `probe_generators.py`, `probe_generators.json`, `probe_batching.py`
- Audit report codex-b, PR #113 and #114 paragraphs; gh-prs-b PR #113/#114 rows
- Merged findings NF-37; extension row for #139

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `data/successor-ramp-is-copy-task`.*
