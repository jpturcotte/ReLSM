---
slug: eval-json-examples-are-failures
title: Eval JSON `examples` are the first 10 failures per condition, not samples
issue: 228
severity: P3
labels: [bug, P3-minor, area: evaluation, telemetry, audit-2026-10-27]
nf: [NF-51]
found_by: [artifacts]
status: Demonstrated
related_issues: [#161, #144, #162]
---

## Summary

`evaluate_condition` adds an example to `collected_examples` only when `exact_match != 1.0`, up to 10 per condition, and the result file stores that list under the key `examples`. The seven Dec-23 files show it: successor has 0 `examples` in the four `5steps`/`20steps` files (accuracy 1.0) and 10 in the other three; every task with accuracy below 0.95 has exactly 10. Only the 5 `sampled_examples` per condition are an unbiased reservoir sample. Any statement made from `examples` ("the model echoes operands", "every copy prediction is 6 items") describes the failure set.

## What happens

- [utils.py:1238-1248](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1238-L1248): `if metrics["exact_match"] == 1.0: total_correct += 1 else: if len(collected_examples) < 10: collected_examples.append({...})`. The dict has `prompt`, `target`, `expected_output`, `prediction`; no correctness flag.
- [utils.py:1310-1315](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1310-L1315): the reservoir `sampled_examples` (seeded `Random(seed + 17)`, 5 per condition) is the unbiased list; `eval_hub.py` merges it per task ([eval_hub.py:611-615](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L611-L615)).
- Audit probe `artifacts/p5_eval_json.py`, section 3, all seven files: `examples` per condition are 10 for every task except successor in `5steps.v1/v2` and `20steps.v1/v2` (0, accuracy 1.000); `sampled_examples` are 5 everywhere.
- For copy and reverse at accuracy 0.0 the failure set is the whole set; for addition at 0.66 it is the 34 % that failed; for successor at 1.0 it is empty.

## Why it matters for the experiments

- Readers of the Dec-23 files (the counter-review's section 4 and several audit transcripts) reasoned about typical model behaviour from `examples`. For tasks with partial accuracy that reasoning is about failures only. The claim that is safe from `examples` is "here are failures"; the typical prediction must come from `sampled_examples` (5 per condition).
- #161 says the stored predictions are normalized and lose the raw text; it calls the list "failure examples" once but keeps the key name `examples`, and the fix it proposes ("first N per condition") would keep the bias.

## How it got here

- PR [#19](https://github.com/jpturcotte/ReLSM/pull/19) (2025-12-12, "Add unified evaluation hub for modular tasks", commit `d8240df`) wrote the `len(collected_examples) < 10` failure collection.
- PR [#83](https://github.com/jpturcotte/ReLSM/pull/83) (2025-12-20, "… per-task sampling logs") added the 5-item reservoir `sampled_examples` beside it without renaming the first list. Testing: "Not run".

## How we verified it, step by step

1. artifacts read `utils.py:1238-1248` and all seven `eval_results` files with `artifacts/p5_eval_json.py`: per-condition counts of `examples` and `sampled_examples`, empty-prediction counts, and the copy/reverse item-count distributions (`p5_out.txt` sections 3 and 4) (AR-9).
2. The drafter re-read `utils.py:1238-1248` and `1310-1315` at HEAD: the branch structure is as described.
3. Disagreements: none.

## Suggested fix

- Rename the key to `failures` and add `exact_match` (and the raw fields of #161) to every stored example.
- Store a seeded sample of all examples, correct and incorrect, with `exact_match` as a field; or write every example to a JSONL (#161's `--save_all_examples`) and drop the two partial lists.
- Add a line to #161 saying that `examples` are failures only.

## Acceptance test

CPU, pure Python with the oracle answerer from #144: run `evaluate_condition` for one condition with a model stub that always returns the target. After the fix the record has `failures == []` and `samples` of length `min(n, sample_count)` with `exact_match == 1.0` on each. At HEAD the same run yields `examples == []` and gives no sign that the condition was perfect except the accuracy field.

## Related

- #161: raw outputs and token ids are lost; the same two lists.
- #144: oracle regression test; it would show `examples` empty for every condition.
- #162: readings of the Dec-23 copy/reverse outputs; for those two tasks at 0 % the failure set is the whole set, so those readings stand.

## Sources

- Audit report artifacts, finding AR-9; `p5_eval_json.py`, `p5_out.txt`
- Merged findings NF-51; extension row for #161

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `eval/eval-json-examples-are-failures`.*
