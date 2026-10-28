---
slug: metrics-json-is-jsonl-with-bare-nan
title: metrics.json is JSON Lines under a .json name and writes NaN as a bare token
issue: 240
severity: P3
labels: [bug, P3-minor, area: tooling, tech-debt, audit-2026-10-27]
nf: [NF-26]
found_by: [code-train-a, cli-commands, codex-b]
status: Demonstrated
related_issues: [#175, #151, #169, #150]
---

## Summary

`MetricsLogger._append_snapshot` writes one JSON object per line into a file named `metrics.json`, so `json.load` fails with "Extra data". `json.dump` runs with the default `allow_nan=True`, so a NaN loss is written as the bare token `NaN`, which strict JSON parsers reject. #169 item 4 reads the surviving `metrics.0.json` (JSONL) as evidence of an older code version; HEAD writes the same format, only the name changed.

## What happens

- [train.py:410-449](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L410-L449): `_append_snapshot` writes a `hyperparameters` row, then one `record` row per log, then `task_accuracy` rows, each followed by `\n`, and renames the temp file to `metrics.json`.
- Audit probe `code-train-a/p4_metrics_logger.py`: after logging `train/loss/ce = float("nan")` and saving, "NaN written literally: True"; Python's `json.loads` accepts the row, a strict parser (`parse_constant` raising) rejects it.
- Audit run R1 (cli-commands): `metrics.json` has 31 lines (one `hyperparameters`, then `record` and `task_accuracy` rows); `json.load` raises "Extra data".
- [README.md:94](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/README.md#L94) says `metrics.json` and `summary.json` "capture scalar logs and evaluation outputs"; nothing says the file is JSON Lines.
- DAG-v12's `metrics.0.json` contains no `NaN` (grep count 0), so the NaN case has not bitten a kept run. #151 shows a NaN `train/loss/ce` is reachable at `alg_seq_len 48`.

## Why it matters for the experiments

- Any reader that opens the file with `json.load`, or with a strict parser (most non-Python tooling), fails. The run-record tooling of #189 would inherit the trap.
- One NaN row from #151 makes the whole file unreadable for strict readers, after the fact and without warning.
- #169 item 4 draws a wrong conclusion from the format ("points to an older code version"); the correction belongs there.

## How it got here

PR [#118](https://github.com/jpturcotte/ReLSM/pull/118) ([task](https://chatgpt.com/remote/task_e_69517d36cb34832e945dea9172a5b354), 2025-12-28, "Write metrics.json as JSONL and flatten per-task accuracies") switched the writer to one object per line and kept the `.json` name and the README sentence. `allow_nan` was never set in any version.

## How we verified it, step by step

1. code-train-a (T6) read `_append_snapshot` and ran audit probe `code-train-a/p4_metrics_logger.py`: NaN written literally; strict parser rejects; `json.loads` per line works.
2. cli-commands counted the lines of R1's `metrics.json` (31) and confirmed `json.load` fails with "Extra data" (section 3c); it flagged #169 item 4 as wrong.
3. codex-b (#175 row) confirmed PR #118 introduced JSONL on Dec 28 and that the file is still rewritten in full each save (#175).
4. code-train-a grepped DAG-v12's `metrics.0.json` for `NaN`: 0 hits.
5. Coordinator: not re-checked.

## Suggested fix

- Map non-finite floats to `null` in `MetricsLogger.log` and call `json.dump(..., allow_nan=False)` so a regression fails loudly.
- Rename the file to `metrics.jsonl` (update the resume reader of #150 and the README), or add one README sentence saying the file is JSON Lines.
- Correct #169 item 4.

If #175 replaces the full rewrite with an append, make both changes in the same PR.

## Acceptance test

Pure Python: log a record with `float("nan")`, call `save()`, then read every line with `json.loads(line, parse_constant=...)` where `parse_constant` raises. No exception; the NaN field is `null`. The file name ends in `.jsonl`, or the README names the format.

## Related

- #175: the same writer rewrites the whole history at every log.
- #150: the resume path that reads this file back.
- #151: the NaN loss that would be written literally.
- #169: item 4 misreads the format; correct it.

## Sources

- Audit report code-train-a, finding T6; probe `p4_metrics_logger.py`
- Audit report cli-commands, section 3c (README table) and the #169 correction row
- Audit report codex-b, changes-to-existing-issues row #175
