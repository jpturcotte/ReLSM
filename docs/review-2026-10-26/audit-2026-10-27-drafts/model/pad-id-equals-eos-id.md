---
slug: pad-id-equals-eos-id
title: "pad_token_id == eos_token_id: real EOS keys are masked; no-pad fast path is dead"
issue: 232
severity: P3
labels: [bug, P3-minor, area: model, tech-debt, audit-2026-10-27]
nf: [NF-03]
found_by: [gh-prs-b, code-model, review-files, codex-b, gemini-a, claude-b2]
status: Demonstrated
related_issues: [#172, #189, #135, #159, #184]
---

## Summary

Llemma has no pad token, so `train.py` and `prepare_tokenizer` set `pad_token = eos_token`; both ids are 2. PR #95 appends EOS to every example and PR #126 builds the model's padding mask from `input_ids == pad_token_id`. So the real EOS that ends every answer is masked as a key for every later position, the thought and memory cross-attention never see the end-of-answer position, `_build_padding_mask`'s early exit never fires in training (every batch takes the dense-mask SDPA path), and "pad or real EOS" is undecidable from `input_ids`. Current numbers are unaffected: one example per row, EOS last, its label dropped. It becomes a defect for any packed or multi-document layout, and it defeats any "with mask vs without mask" comparison.

## What happens

- [train.py:1954-1955](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1954-L1955) and [utils.py:255-256](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L255-L256): `tokenizer.pad_token = tokenizer.eos_token`. [train.py:1975](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1975): that id becomes `config.pad_token_id`.
- [data.py:1121-1122](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1121-L1122): EOS appended to every example. [data.py:1644](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1644): the collator pads with the same id.
- [model.py:820-827](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L820-L827): `is_pad = input_ids.eq(pad_token_id)`; `if not is_pad.any(): return None` (never true in training); otherwise `-inf` at every pad position, EOS included.
- [model.py:852](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L852): `if past_len > 0 or padding_mask is not None` builds the dense causal+padding mask, so `is_causal=True` is dead for every training batch (a host-device sync and no flash kernel).

Measured:

- Audit probe `gh-prs-b/probe_pad_eq_eos_mask.py`: `pad_token_id=2 eos_token_id=2`; in a two-row batch the real EOS at index 5 (row 0) and index 7 (row 1) is masked.
- Audit probe `gemini-a/probe_recipe_params_and_eos_mask.py` (B): tiny baseline, EOS at index 3 of 8; logits at positions 0–2 unchanged, positions 3–7 move by up to 0.046 versus the unmasked model.
- Audit probe `code-model/p1_causal_padding_cache.py` (B, H): right padding by 3 or 7 pads changes real-position logits by at most 1.0e-7, and a row that finishes early in batched `generate()` equals the same row alone. So today's effect on results is nil.
- `claude-b2/tokenizer_pad_eos.py` and codex-b probe P6 confirm the equal ids.

## Why it matters for the experiments

- Nothing in DAG-v12 or the Dec-23 files changes: every row holds one example, EOS is last, its label is `-100`.
- For `latent`, `act`, `ssm_mem` the pooled thought vector never includes the end-of-answer position. Small today, unintended, and undocumented.
- Any packed layout (several examples or documents per row, the language phase of #159) makes the token after an EOS unable to see it. Exact-length bucketing was proposed and declined on Dec 31 in favour of masks; the mask now hits EOS.
- Performance: every training batch takes the explicit-mask SDPA path.

## How it got here

- PR [#1](https://github.com/jpturcotte/ReLSM/pull/1) (2025-12-10) wrote `tokenizer.pad_token = tokenizer.eos_token` in `train.py` for a GPT-2 tokenizer that also lacks a pad token. PR [#48](https://github.com/jpturcotte/ReLSM/pull/48) (2025-12-18) switched to Llemma, same situation.
- PR [#95](https://github.com/jpturcotte/ReLSM/pull/95) (2025-12-21, "Append EOS token to algorithmic datasets") made every row end in id 2.
- PR [#126](https://github.com/jpturcotte/ReLSM/pull/126) (2025-12-31, "Mask padding in thought and memory attention") derived the mask from token identity. The Dec-31 Claude chat set `pad_token_id` to `Optional[int] = None` because "Default of 0 would incorrectly mask token 0"; nobody noted that the configured id is the EOS id. The same chat's bucketing proposal ("identical lengths -> no padding needed -> no mask needed") was not taken.
- `project-history-gemini.md` §7 says the pad fallback is fine "because pad is set to EOS"; that is the cause, not a defence.

## How we verified it, step by step

1. gh-prs-b (A-2) traced the three PRs and ran the mask probe: both ids 2, real EOS masked in both rows.
2. code-model (CM-9) confirmed ids from the loaded tokenizer, measured padding invariance (1e-7) and the batched-generate equality, and concluded "harmless today, hazard for packed layouts".
3. gemini-a (G-A-3) measured the logit change behind an interior EOS (up to 0.046).
4. codex-b (N5), claude-b2 (B2-1) and review-files (RF-4) found the same by reading; review-files noted the previous review had planned a draft `eos-as-pad-masks-real-eos` and dropped it to plain text.
5. The coordinator re-read `train.py:1954-1955` and `model.py:820-827` and confirmed.
6. Disagreement on where it belongs: gh-prs-b, review-files and codex-b would rescope #172 to own it; code-model wants an ADR line only; claude-b2 and gemini-a want a distinct pad id or a collator mask. This draft opens it as its own P3 so the fix has an owner.

## Suggested fix

- Give the tokenizer a distinct pad id: a reserved id (the checkpoint vocab is 32,016 against 32,000 tokens) or an added `[PAD]` with a one-row embedding resize. Or build the padding mask from the collator's lengths (`attention_mask`) instead of token identity, which also removes the `input_ids.eq` sync.
- Alternative that removes the mask entirely for training: exact-length bucketing of the algorithmic stream (sequences are 15–25 tokens with few distinct lengths).
- Either way, write the pad/EOS contract into the vocabulary ADR (#189) and keep `_build_padding_mask` returning `None` for an unpadded batch.
- Independent of #172's cached-mask fix, but the two should use the same mask source.

## Acceptance test

CPU, Llemma tokenizer: `tokenizer.pad_token_id != tokenizer.eos_token_id`; and for a batch of two rows (lengths 6 and 8, each ending in EOS, padded to 8) `_build_padding_mask` is `-inf` only at row 0 positions 6–7, and returns `None` for a batch with no padding.

## Related

- #172: cached decoding drops the padding mask; same mask derivation.
- #189: the vocabulary/pad ADR.
- #135: the thought read whose keys this mask filters.
- #159: the language pipeline, where packed documents would hit this.
- #184: CPU test home.
- #212: the same pad id fills the all-pad window.
- #218: the other half of the vocabulary contract.

## Sources

- Audit report gh-prs-b, finding A-2; probe `gh-prs-b/probe_pad_eq_eos_mask.py`
- Audit report code-model, finding CM-9; probe `code-model/p1_causal_padding_cache.py` (B, H), `p5_generate_utils.py`
- Audit report gemini-a, finding G-A-3; probe `gemini-a/probe_recipe_params_and_eos_mask.py`
- Audit report codex-b, finding N5; claude-b2, finding B2-1 and avenues 1–2; review-files, finding RF-4
- VERIFIED_BY_COORDINATOR.md (pad token = EOS row)
- Transcript `2025-12-31_26632ad7_Masking_and_padding_issues_in_shared_loop_variants.md`
