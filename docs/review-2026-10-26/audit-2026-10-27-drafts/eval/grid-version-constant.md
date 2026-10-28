---
slug: grid-version-constant
title: OOD_GRID_VERSION is a constant; the eval grid silently follows data.py edits
issue: 230
severity: P3
labels: [bug, P3-minor, area: evaluation, reproducibility, audit-2026-10-27]
nf: [NF-41]
found_by: [code-eval, gh-issues, codex-a, gh-prs-a]
status: Demonstrated
related_issues: [#194, #142, #162, #161]
---

## Summary

`eval/ood_grid.py` sets `OOD_GRID_VERSION = "2024-09-05"`, a date before the repository existed. The string was written once on 2025-12-12 and never changed, while the grid changed in PR #66 and PR #82 and moved again when PR #113 and PR #114 edited the phase tables, because `eval/ood_grid.py` imports `TASK_DIFFICULTY_CONFIG` from `data.py`. Every result file carries the same `grid_version` (14 occurrences in the seven `eval_results` files), so the one field meant to tell grids apart cannot.

## What happens

- [eval/ood_grid.py:7](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval/ood_grid.py#L7): `OOD_GRID_VERSION = "2024-09-05"`. `git log -S OOD_GRID_VERSION -- eval/ood_grid.py` returns one commit, `7cfedae` (2025-12-12).
- [eval/ood_grid.py:5](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval/ood_grid.py#L5): `from data import TASK_DIFFICULTY_CONFIG`; the IID conditions take their parameters from the training phase tables at import time.
- [eval_hub.py:516](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L516) and [eval_hub.py:783](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L783): the constant is written into the payload and the metadata of every result file.
- File history: `e5905f0` (PR #66, 2025-12-19) moved the IID points to the final-phase parameters; `37851cb` (PR #82, 2025-12-20) split IID and OOD. PR #113 (2025-12-26) ramped successor to 20–32 digits and PR #114 (2025-12-27) rewrote chain's ranges; both changed the IID conditions without touching `eval/ood_grid.py`.
- `grep '"grid_version"' eval_results/*.json`: 14 hits (7 files × meta + payload), all `"2024-09-05"`.

## Why it matters for the experiments

- The seven Dec-23 files were produced at `8f01ecad` / `66ba4bec`, before PR #113 and #114. A rerun of the same checkpoints at HEAD evaluates successor IID at 20–32 digits instead of the earlier ramp and chain at 24–32 digits, under the same `grid_version`. Nothing in the file says the grid differs.
- #194 lists `grid_version` among the metadata written; #161 proposes adding it to per-condition records. Both would record a meaningless value.

## How it got here

- PR [#20](https://github.com/jpturcotte/ReLSM/pull/20) (2025-12-12, "Refactor evaluation into single deterministic hub"). The owner's task asked: "Add OOD_GRID_VERSION = "YYYY-MM-DD" in ood_grid.py … include it in the output JSON". Codex filled in `"2024-09-05"`. Testing: "compileall".
- PR [#66](https://github.com/jpturcotte/ReLSM/pull/66) (2025-12-20) made the grid import `data.py` and rebuilt it from `TASK_DIFFICULTY_CONFIG`; PR [#82](https://github.com/jpturcotte/ReLSM/pull/82), [#113](https://github.com/jpturcotte/ReLSM/pull/113) and [#114](https://github.com/jpturcotte/ReLSM/pull/114) changed the grid afterwards. No task mentioned the version string again.

## How we verified it, step by step

1. code-eval ran `git log -S OOD_GRID_VERSION -- eval/ood_grid.py` (one hit, `7cfedae`) and `git log -- eval/ood_grid.py` (`2a20678`, `7cfedae`, `e5905f0`, `37851cb`), then grepped the seven result files (14 hits, one value) (E1).
2. gh-issues, codex-a (F2) and gh-prs-a (A9) found the same by reading; gh-prs-a traced the `data.py` import to PR #66 and the silent grid moves to #113/#114.
3. The drafter re-ran the `git log -S` and the grep at HEAD: same results.
4. Disagreements: none. Four reports, one conclusion.

## Suggested fix

- Compute the version at import from the grid itself: `hashlib.sha256(json.dumps([(c.task, c.name, c.params, c.n, c.max_new_tokens) for c in build_grid(None, True, True)], sort_keys=True).encode()).hexdigest()[:12]`, and keep a human-edited date next to it.
- Add a test that stores the current hash and fails when the grid changes without the date being bumped.
- Write the hash into #194's metadata and #142's per-condition budget record.
- Separately decide (#162) whether the IID conditions should be a snapshot in `eval/ood_grid.py` rather than a live import of the training tables.

## Acceptance test

CPU, pure Python:

```python
import hashlib, json
from eval import ood_grid
conds = ood_grid.build_grid(None, include_iid=True, include_ood=True)
h = hashlib.sha256(json.dumps([(c.task, c.name, c.params, c.n, c.max_new_tokens) for c in conds], sort_keys=True, default=str).encode()).hexdigest()[:12]
assert ood_grid.OOD_GRID_VERSION != "2024-09-05"
assert h in ood_grid.OOD_GRID_VERSION or ood_grid.OOD_GRID_HASH == h
```

## Related

- #194: eval metadata records the wrong commit and no dirty flag; `grid_version` is untrustworthy by construction.
- #142: save the resolved budget per condition; add the grid hash there.
- #162: the IID grid is not the training distribution; the live import is why it moves.
- #161: proposes adding `grid_version` per condition.

## Sources

- Audit report code-eval, finding E1
- Audit report gh-issues, finding G1
- Audit report codex-a, finding F2 and task table row 20
- Audit report gh-prs-a, finding A9 and PR #66 row
- Merged findings NF-41

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `eval/grid-version-constant`.*
