---
slug: position-ids-past-chunk-indexerror
title: "Explicit position_ids past the chunk length raise IndexError in the RoPE slice"
issue: 233
severity: P3
labels: [bug, P3-minor, area: model, tech-debt, audit-2026-10-27]
nf: [NF-05]
found_by: [code-model]
status: Demonstrated
related_issues: [#172, #184]
---

## Summary

`Attention.forward` slices the rotary cache to `T + past_len` and then indexes it with the caller's `position_ids`. Any position larger than `T + past_len − 1` raises `IndexError`. The usual reason to pass `position_ids` is per-row offsets (left padding, packed sequences), which is exactly the case that fails. No current caller does this; the one caller forwards an `arange` captured by a hook. Latent hazard, not a result defect.

## What happens

- [model.py:296-300](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L296-L300): `seq_len = T` (+ cache length); `cos, sin = self.rotary(seq_len, x.device)` returns a cache of `seq_len` rows.
- [model.py:238-239](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L238-L239): `cos = cos[position_ids]` indexes that slice directly.
- [train.py:795](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L795): the only caller passing `position_ids` forwards the hook-captured `arange`, so it never exceeds the slice.

Measured (audit probe `code-model/p5_generate_utils.py`, last section): tiny baseline, `B=2, T=6`, `position_ids = arange(6) + 3` → `IndexError: index 6 is out of bounds for dimension 0 with size 6` at `model.py:238`.

## Why it matters for the experiments

- No recorded number is affected. The parameter exists on the public `forward` signature and fails for its main use.
- Left-padded batched generation (the fix path for #172's padded prompts) or any packing would hit it immediately.

## How it got here

- PR [#1](https://github.com/jpturcotte/ReLSM/pull/1) (2025-12-10) added `position_ids: Optional[torch.Tensor] = None` to `apply_rotary_emb` and `Attention.forward` and the direct `cos[position_ids]` index, with the cache sized to the chunk. No task asked for offset positions.

## How we verified it, step by step

1. code-model (CM-3) read the slice and the index and predicted the failure for any offset.
2. Audit probe `code-model/p5_generate_utils.py` ran the call; traceback recorded in `p5_out.txt`.
3. code-model grepped the callers: only `train.py:795`, which forwards the hook's own positions.

## Suggested fix

Size the rotary cache to `max(seq_len, int(position_ids.max()) + 1)` when `position_ids` is given, or remove the parameter until a caller needs it. One line either way.

## Acceptance test

CPU, tiny baseline, `B=2, T=6`: `model(x, position_ids=torch.arange(6).expand(2, -1) + 3)` runs, and its logits equal a forward of the same tokens placed at positions 3–8 inside a length-9 sequence (compare the last six positions).

## Related

- #172: cached decoding and padded prompts, the first caller that would need offsets.
- #184: add the offset case to the CPU suite.

## Sources

- Audit report code-model, finding CM-3; probe `code-model/p5_generate_utils.py`
