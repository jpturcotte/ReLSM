---
slug: degenerate-eval-diagnostics
title: "Degenerate eval diagnostics: format cohorts, repetition metrics, unconsumed keys"
issue: 226
severity: P3
labels: [bug, P3-minor, area: evaluation, telemetry, tech-debt, audit-2026-10-27]
nf: [NF-43]
found_by: [code-eval, gh-prs-b]
status: Demonstrated
related_issues: [#180, #161, #198, #167]
---

## Summary

Three diagnostics computed at every eval say nothing. The prompt-format cohorts (`colon`, `answer`, `arrow`, `fatarrow`, `equals`) are true for 100 % of prompts (`colon`, `answer`) or mark a task rather than a format (`arrow` = every chain prompt, `equals` = the compare prompts with the "(>, <, or =)" hint); every prompt has ended in `Answer:` since PR #103/#116. The repetition metrics score a perfect copy, reverse or successor answer as repetitive (`unique_token_fraction` 0.32, `repeat_2gram_rate` 0.42), because they count SentencePiece `▁` tokens and repeated digits, and an empty continuation scores 0.0 (maximal repetition). `soft_hard_gap` and `numeric_length_mismatch` are written to the logs and read by nothing.

## What happens

- [utils.py:147-156](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L147-L156): `get_prompt_format_flags`. [utils.py:1218-1225](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1218-L1225): per-cohort counts accumulated for every example. [train.py:3152](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3152) and [train.py:3347](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3347): printed only under `--debug_eval_format_stats`.
- Measured on the 4,100 oracle prompts: `colon` 4,100, `answer` 4,100, `arrow` 200 (all chain prompts; `->` is the chain operator), `fatarrow` 0, `equals` 112 (56 % of the 200 compare prompts).
- [utils.py:187-231](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L187-L231): `compute_repetition_metrics` over continuation token ids; an empty continuation returns `unique_token_fraction` 0.0 ([utils.py:190-196](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L190-L196)).
- Oracle run (every answer correct), `diagnostics.repetition.by_task`: `unique_token_fraction` copy 0.324, reverse 0.329, successor 0.327, chain 0.343, parity/dyck/compare 1.0; `repeat_2gram_rate` copy 0.425, reverse 0.415. A 12-item copy is 23 tokens drawn from 11 symbols.
- `soft_hard_gap` is flattened to metrics at [train.py:1049-1056](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1049-L1056) and built in `eval_hub.py`; `numeric_length_mismatch_count` is counted at [utils.py:1290](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1290). No decision, warning or test reads either.

## Why it matters for the experiments

- A reader of `metrics.json` or an eval JSON sees `unique_token_fraction 0.32` on copy and `repeat_2gram_rate 0.42` and reads degeneration into a perfect answer; the metric has no per-task healthy baseline.
- The cohort keys (`gen/format/*`) and the two unconsumed keys add noise to every eval record (#198, #167 are about the same record).
- The "crash detector" these cohorts were built to feed compares consecutive evals on a changing curriculum and never fired on the one regression it was meant for (PR #116's `Answer:` change).

## How it got here

- PR [#106](https://github.com/jpturcotte/ReLSM/pull/106) (2025-12-23, "Refine prediction/label normalization, treat parity as classification, and remove redundant unique_token_ratio", 7 turns). A ChatGPT prompt asked for "prompt-format cohorts (colon/answer/arrow/fatarrow/equals), 'crash detector' warning, `--debug_eval_format_stats`", soft-hard gaps and the numeric length mismatch. Testing: "Not run" ×7.
- PR [#103](https://github.com/jpturcotte/ReLSM/pull/103) (2025-12-23) and PR [#116](https://github.com/jpturcotte/ReLSM/pull/116) (2025-12-28) made every prompt end in `Answer:`, which collapsed the cohorts.
- PR [#83](https://github.com/jpturcotte/ReLSM/pull/83) (2025-12-20, "Add eval diagnostics (length, parse, repetition)") added the repetition metrics on token ids.

## How we verified it, step by step

1. code-eval ran the full grid through the real path with an oracle answerer (audit probe `code-eval/oracle_probe.py`) and read `prompt_format` counts and `diagnostics.repetition.by_task` from the output (numbers above; `oracle_out.txt` line 12, `oracle_out/run1_suite_all/results_algorithmic_all.json`) (E4, E7).
2. code-eval's `metrics_probe.py` checked the edge case: empty continuation → `unique_token_fraction` 0.0.
3. gh-prs-b traced the cohorts to #106 and their collapse to #103/#116 by reading the PR diffs (A-9, Plausible on its own; code-eval's measurement makes the row Demonstrated).
4. The drafter grepped `soft_hard_gap` and `numeric_length_mismatch` in `train.py`, `utils.py`, `eval_hub.py` and `tests/`: they are computed, flattened and written; no reader.
5. Disagreements: none.

## Suggested fix

- Delete `get_prompt_format_flags`, the `format_stats` accumulation, `_print_prompt_format_stats` and `--debug_eval_format_stats` (add them to #180's list).
- Compute repetition on the decoded item sequence (digits or list items, not token ids), or report each metric relative to the target's own value so a perfect answer scores 1.0.
- Drop `soft_hard_gap` and `numeric_length_mismatch`, or give them a consumer (a test or a warning) and a definition in the docs.
- Store `exact_match` per example (#161) so cohort questions can be answered offline instead of by pre-baked cohorts.

## Acceptance test

Pure Python:

```python
from transformers import AutoTokenizer
from utils import compute_repetition_metrics
tok = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")  # HF_HUB_OFFLINE=1
target = "3 8 3 5 2 1 6 3 7 9 4 1"
ids = tok(target, add_special_tokens=False)["input_ids"]
m = compute_repetition_metrics(ids)          # after the fix: item-level or target-relative
assert m["unique_token_fraction"] >= 0.9 and m["repeat_2gram_rate"] <= 0.1, m
```

At HEAD this gives about 0.32 and 0.42. Also: `grep -c get_prompt_format_flags utils.py train.py` returns 0.

## Related

- #180: dead code list; add the cohorts and the flag.
- #161: per-example fields would replace the cohorts.
- #198 and #167: other keys in the same flattened eval record that mislead.
- #144: the oracle test is the natural place to assert a perfect answer reads as non-repetitive.

## Sources

- Audit report code-eval, findings E4 and E7, metric audit table; `oracle_probe.py`, `oracle_out.txt`, `metrics_probe_out.txt`
- Audit report gh-prs-b, finding A-9 and PR #106 row; codex-b PR #106 paragraph; chatgpt-b (the prompt that asked for the cohorts)
- Merged findings NF-43

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `eval/degenerate-eval-diagnostics`.*
