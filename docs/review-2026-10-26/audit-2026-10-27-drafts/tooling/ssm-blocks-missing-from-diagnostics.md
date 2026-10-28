---
slug: ssm-blocks-missing-from-diagnostics
title: "get_transformer_blocks returns [] for ssm/ssm_mem; block diagnostics are absent"
issue: 241
severity: P3
labels: [bug, P3-minor, area: tooling, telemetry, variants-only, audit-2026-10-27]
nf: [NF-06]
found_by: [code-model]
status: Demonstrated
related_issues: [#149, #147, #133, #184]
---

## Summary

`get_transformer_blocks` takes the model's `layers` and keeps only modules that have both `attn` and `ff`. `SSMBlock` has `ff` and no `attn`, so the list is empty and `resolve_diagnostic_modules` returns only `embedding` and `head`. For `ssm` and `ssm_mem` runs the per-block gradient and weight norms and the attention probes are missing from the metrics, with no warning.

## What happens

- [utils.py:268-287](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L268-L287): the first matching attribute (`layers`) is filtered with `hasattr(block, "attn") and hasattr(block, "ff")`; the filtered (empty) list is returned without trying the other names.
- [utils.py:299-302](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L299-L302): `block0`, `block_mid`, `block_last` are added only `if blocks`.
- [train.py:2734-2739](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2734-L2739): the loop logs `train/grad/norm/<key>` and `train/weight/norm/<key>` for whatever keys exist, so nothing flags the gap.

Measured (audit probe `code-model/p5_generate_utils.py`, diagnostics section): `baseline`, `shared_loop`, `latent` → `blocks=2`, keys `['block0', 'block_last', 'block_mid', 'embedding', 'head']`; `ssm`, `ssm_mem` → `blocks=0`, keys `['embedding', 'head']`.

## Why it matters for the experiments

- An SSM run's `metrics.json` has no block-level norms or entropies, so the diagnostics #149 relies on cannot be compared across variants. The variant with the most open correctness issues (#133, #134, #147) is the one with the least telemetry.
- No recorded run used `ssm`/`ssm_mem`, so no artifact is affected.

## How it got here

- PR [#73](https://github.com/jpturcotte/ReLSM/pull/73) (2025-12-20, "Add lightweight training diagnostics and eval-probe metrics") introduced the diagnostics; PR [#77](https://github.com/jpturcotte/ReLSM/pull/77) (same day) edited `resolve_diagnostic_modules` for tied weights. Which of them wrote the `attn`/`ff` filter was not traced by this audit; neither ran an SSM model.

## How we verified it, step by step

1. code-model (CM-8) read the filter and `SSMBlock`'s attributes.
2. Audit probe `code-model/p5_generate_utils.py` built all five variants on CPU and printed `get_transformer_blocks` length and the diagnostic keys.

## Suggested fix

Filter on `hasattr(block, "ff")` only, or match `SSMBlock` explicitly (`isinstance(block, (TransformerBlock, SSMBlock))`). Log a warning when a model has layers but the diagnostic block list is empty.

## Acceptance test

CPU: `resolve_diagnostic_modules(create_model(..., variant="ssm"))` has keys `block0`, `block_mid`, `block_last`, and the same for `ssm_mem`; `baseline` keys unchanged.

## Related

- #149: the telemetry these keys feed.
- #147, #133: SSM-specific issues that need per-block norms to diagnose.
- #184: CPU test home.

## Sources

- Audit report code-model, finding CM-8; probe `code-model/p5_generate_utils.py`
