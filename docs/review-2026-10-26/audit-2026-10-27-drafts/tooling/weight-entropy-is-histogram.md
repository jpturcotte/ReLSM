---
slug: weight-entropy-is-histogram
title: eval/weight/entropy is a 256-bin histogram entropy of |w|, not spectral entropy
issue: 245
severity: P3
labels: [bug, P3-minor, area: tooling, area: docs, telemetry, audit-2026-10-27]
nf: [NF-31]
found_by: [agent-reports]
status: Demonstrated
related_issues: [#189, #157, #180, #149]
---

## Summary

`compute_weight_entropy` bins `abs(param)` of a module into 256 equal-width buckets scaled to the module's largest weight and returns the Shannon entropy of that histogram. It is logged as `eval/weight/entropy/last_block` and `eval/weight/entropy/head`. Every Nash-SEC design document behind the DAG defines "SVD entropy" as the normalised entropy of a singular-value spectrum and gates phase transitions on it (`H_spec > 0.7`). Nothing at HEAD computes a singular value. The key name invites reading the histogram as the spectral quantity.

## What happens

- [utils.py:341-370](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L341-L370): find `abs_max` over the module's parameters, `torch.histc(values, bins=256, min=0.0, max=abs_max)` per parameter, sum, normalise, `-(p * log p).sum()`.
- [train.py:820-823](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L820-L823): logged for the last block and the head.
- Properties: the maximum is `ln 256 = 5.545`; the bin width follows the single largest `|w|` in the module, so one outlier weight moves the number for all the others; it is invariant to any permutation of the weights and says nothing about rank, spectrum or structure.
- Grep over `*.py` for `svd|spectral|hessian|nash|famo|exp3|bandit|learning_progress` hits only a comment at `model.py:11`.
- Logged values: DAG-v12 `weight_entropy.last_block` series present at every eval; the Jan-11 run printed `eval/weight/entropy/last_block=2.713` at step 11,000.

## Why it matters for the experiments

- The design documents that supplied the task graph (Nash-SEC and its follow-ups) make SVD entropy the structural gate for unlocking the next phase. A reader who matches the log key to those documents would believe the gate exists and read 2.7 against a 0.7 threshold on a different scale. It does not exist; the gates at HEAD are accuracy EMAs (#157).
- Spectral entropy, effective rank and related measures are the grokking-literature probes for the memorisation-to-generalisation switch. The project has none; the log suggests it has one.

## How it got here

PR [#73](https://github.com/jpturcotte/ReLSM/pull/73) (2025-12-20) added `compute_weight_entropy` with the other lightweight diagnostics. The Nash-SEC documents (Dec 24 to 26) came later and asked for "SVD entropy"; agent-reports' 69-row table of design recommendations against HEAD records that nothing of the SVD/Hessian gates reached the code (agent-reports section 2.2). The two were never reconciled.

## How we verified it, step by step

1. agent-reports (AR-6) read `utils.py:341-363` and `train.py:820-823`, grepped the repository for any spectral computation (one comment), and read the Nash-SEC definition and the `H_spec > 0.7` gate in the thread markdown and the report PDFs.
2. The Jan-11 value comes from gemini-c's reading of the pasted log (section 2.6).
3. Coordinator: not re-checked. Single finder; code reading plus grep, no probe needed.

## Suggested fix

Rename the key to `eval/weight/hist_entropy/*` (one string change, plus the plot label), and state in the metrics README that no spectral metric is logged. If the spectral version is wanted: for each 2-D weight `W`, `s = torch.linalg.svdvals(W.float())`, `p = s / s.sum()`, `H = -(p * log p).sum() / log(len(s))`, logged as `eval/weight/spectral_entropy/<module>`; cost is one SVD per matrix per eval on CPU. Record the decision in the ADR of #189 together with the other Nash-SEC items that were not implemented.

## Acceptance test

Pure PyTorch: for a random Gaussian 512 x 512 matrix the new `spectral_entropy` is between 0.95 and 1.0 and for a rank-1 matrix it is 0.0; `grep -rn "eval/weight/entropy" *.py` returns nothing. If only renamed: the grep returns nothing and `hist_entropy` appears in the plot legend.

## Related

- #189: the ADR should list which Nash-SEC design items were adopted; SVD entropy was not.
- #157: the gates that do exist are accuracy EMAs.
- #180: misleading names and dead code.
- #149: other telemetry whose names do not match their definitions.
- #234, #238: the two other PR #73 diagnostics.

## Sources

- Audit report agent-reports, finding AR-6, section 2.2 (recommendation table) and section 2.5 (C4, C6 on how the Dec-24 reports read the metrics)
- Audit report gemini-c, section 2.6 (Jan-11 log value)
- Audit report gh-prs-b, per-PR table row #73 (provenance)
