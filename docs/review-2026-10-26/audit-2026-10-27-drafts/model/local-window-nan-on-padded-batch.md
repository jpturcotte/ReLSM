---
slug: local-window-nan-on-padded-batch
title: "--local_window > 0 gives NaN loss on padded batches for latent, act and ssm"
issue: 212
severity: P2
labels: [bug, P2-important, area: model, correctness, variants-only, audit-2026-10-27]
nf: [NF-01]
found_by: [code-model]
status: Demonstrated
related_issues: [#135, #184, #172, #126]
---

## Summary

With `--local_window N`, the thought loop reads the last `N` positions of the padded batch tensor, not the last `N` valid tokens of each row. For any row whose last `N` positions are all pad, every key in the window is masked to `-inf`, the hand-written softmax in `CrossAttention` returns NaN, and the sequence-mean injection spreads the NaN over the whole row. The batch loss is NaN. Training pads every batch to its longest row, so with the flag on nearly every real batch has such a row; `latent`, `act` and `ssm` cannot train with it. No recorded run used the flag. code-model rated it P1 conditional on the flag; by the brief's rule (it breaks a conditional path, not a default one) this draft keeps P2.

## What happens

- [model.py:730-731](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L730-L731): `_select_kv_source` takes `toks = h[:, -cfg.local_window:, :]`, the tail of the padded tensor.
- [model.py:753-754](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L753-L754): `_apply_thought_loop` slices the padding mask the same way, `kv_mask[..., -cfg.local_window:]`.
- [model.py:419-422](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L419-L422): `CrossAttention` adds the mask and calls `F.softmax` directly, not SDPA. An all-`-inf` row gives NaN.
- [model.py:811-812](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L811-L812): `z.mean(dim=1)` and `thought_to_tok` broadcast the NaN to every position of that row.
- [model.py:755-761](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L755-L761): `ssm_mem` is spared only because its memory tokens are unmasked keys prepended to the window.
- [data.py:1613-1617](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1613-L1617): training collates with right padding to the longest row (`pad_id` is the EOS id, see #232). Algorithmic examples vary in length by far more than any sensible window.
- The flag exists at [train.py:3782](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3782) with default 0.

Measured (audit probe `code-model/p1_causal_padding_cache.py`, section E; tiny model, CPU, batch of two rows: 10 real tokens, and 5 real + 5 pad):

| variant | `local_window` | row 1 logits NaN | loss |
|---|---|---|---|
| latent | 2 / 4 | yes / yes | nan / nan |
| act | 2 / 4 | yes / yes | nan / nan |
| ssm | 2 / 4 | yes / yes | nan / nan |
| ssm_mem | 2 / 4 | no / no | 4.137 / 4.138 |

Section E2: the same configs without padding give finite loss (latent 4.159, ssm_mem 4.154).

Even without the NaN the semantics are wrong: for a shorter row the window covers pad positions and misses the row's own last tokens.

## Why it matters for the experiments

- `--local_window` is the one knob the design offers to make the thought read cheaper, and the natural first step toward a causal thought contract once #135 is addressed. As written it cannot train on real batches for three of the four thought variants.
- Nothing in the DAG-v12 or Dec-23 numbers is affected: every recorded command has `local_window 0` (DAG-v12 hyperparameter row; grep of transcripts and `eval_results`).
- The Dec-31 review of PR #126's mask diff wrote "Handling of `local_window` for thought masks is correct". That sign-off did not consider an all-pad window, so the hazard is undocumented.

## How it got here

- PR [#1](https://github.com/jpturcotte/ReLSM/pull/1) (2025-12-10) introduced `local_window` and the slice `toks = h[:, -cfg.local_window:, :]` with the comment "if >0, thought attends only to last local_window tokens (plus mem, if enabled)". The "last tokens" were taken from the tensor, not per row. No documented run used it.
- PR [#126](https://github.com/jpturcotte/ReLSM/pull/126) (2025-12-31, "Mask padding in thought and memory attention") added the matching mask slice `kv_mask[..., -cfg.local_window:]`. The padding mask made the all-pad window produce `-inf` everywhere instead of attending to pad vectors. The Dec-31 Claude chat that reviewed the diff called the window handling correct.

## How we verified it, step by step

1. code-model read `_select_kv_source`, `_apply_thought_loop` and `CrossAttention` and predicted the NaN for an all-pad window.
2. Audit probe `code-model/p1_causal_padding_cache.py` (section E) built `latent`, `act`, `ssm`, `ssm_mem` at `d_model=32` with `local_window` 2 and 4 and a training-style right-padded batch: NaN logits in the padded row and `loss=nan` for the first three; `ssm_mem` finite. Section E2 confirmed finite loss without padding.
3. code-model grepped the transcripts, `eval_results` and the DAG-v12 `summary.json` for `--local_window`: no run used it. It read the Dec-31 chat's sign-off on the PR #126 diff.
4. The coordinator re-read the slicing at `model.py:751-754` and the hand softmax at 730-732 and confirmed the mechanism; the NaN was not re-run.

## Suggested fix

- Select the window per row from valid positions: gather the last `local_window` indices from `attention_mask.cumsum(-1)` (or from the row lengths the collator already knows), so each row sees its own last tokens.
- Use `F.scaled_dot_product_attention` in `CrossAttention` (safe softmax in torch >= 2.5) or assert that every row has at least one unmasked key.
- Fold this into the #135 redesign of the thought read; the window is the natural place to make it causal. Add a padded-batch `local_window` case to the #184 suite.

## Acceptance test

CPU, tiny `latent` model (`d_model=32`, 2 layers), `local_window=2`, batch of two rows (10 real tokens; 5 real + 5 pad): `loss` is finite, and the two keys the thought loop reads for row 1 are its tokens at positions 3 and 4, not positions 8 and 9. Repeat for `act` and `ssm`.

## Related

- #135: the thought loop's future leak; this fix belongs inside that redesign.
- #184: the CPU invariant suite that should carry the padded-window case.
- #172: cached decoding drops the padding mask; same mask plumbing.
- #232: the pad id the collator uses here is the EOS id.

## Sources

- Audit report code-model, finding CM-1; probe `code-model/p1_causal_padding_cache.py` sections E and E2
- VERIFIED_BY_COORDINATOR.md (`--local_window` slicing row)
- Transcript `2025-12-31_26632ad7_Masking_and_padding_issues_in_shared_loop_variants.md` (line 616)
