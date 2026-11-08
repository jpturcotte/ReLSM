# Restore tests: final report (2026-11-03)

**Question:** late in training, do diagonal pairs (a, a) lose margin because:
- layer-1 attention at '=' reads the operands less (operand mass about 0.6–0.75 vs 0.8), or
- the final norm scales their logits down (inverse RMS 6–12 % lower)?

**Answer:** neither is the main cause.
- **Layer-1 operand attention is not specific to diagonals.** Changing it moves diagonal and off-diagonal margins together.
- **The final-norm scale accounts for 15–30 % of the gap,** and only by rescaling logits; predictions don't change.
- **The remaining 70–85 % is in the direction of the diagonal residual, not its size.**

## What was run

**Two versions of the test, 12 final checkpoints each:**
- E1 add ×3 and mul ×3;
- diag_control ×3;
- diag_sepops ×3 (control).

Both ran on the GPU through the memory guard: about 30 s each, peak RAM 1.2 GB, about 1.2 GB GPU.

**Version 1:** swaps of group means.

**Codex code review:**
- no bug that changed results; base gaps match the earlier probe within 3e-6;
- two latent bugs, in options v1 didn't use;
- design limits: the norm swap is mechanical, the separate-token sanity bound was ill posed, and the mass result depends on the norm.

**Version 2:**
- the bugs fixed;
- new checks: forward vs model within 3e-5, exact no-ops, row sums, accuracy invariance;
- a cleaner attention test: small offsets on the operand scores, per head and for all heads, with the final norm free or fixed at each example's own value, applied to diagonal or off-diagonal pairs.

## Results (all-heads offset δ = +1 on layer-1 operand scores; change in lse margin)

| | diagonal, norm fixed | off-diagonal, norm fixed | diagonal, norm free | off-diagonal, norm free |
|---|---|---|---|---|
| add s0/s1/s2 | 0.15 / 0.39 / 0.38 | 0.14 / 0.39 / 0.37 | −0.48 / −0.05 / −0.15 | −0.26 / −0.06 / −0.13 |
| mul s0/s1/s2 | −0.01 / 0.26 / 0.14 | 0.13 / 0.25 / 0.24 | −0.62 / −0.21 / −0.41 | −0.27 / −0.18 / −0.14 |
| control s0/s1/s2 | −0.06 / 0.24 / 0.34 | 0.19 / 0.32 / 0.32 | −0.74 / −0.36 / −0.10 | −0.23 / −0.16 / −0.14 |

**Final-norm scale:**
- giving diagonals the off-diagonal scale closes 15.5–30.3 % of the gap;
- the flattening control (diagonals given their own mean) changes them by ≤ 0.05;
- accuracy is identical under every norm swap.

## Verdicts at the registered thresholds (H6c)

Both analyses agree on all three.

- **Attention:** MIXED. 4/9 above +0.25 and 2/9 at or below +0.1, where 7/9 is needed either way; diagonals beat off-diagonals in only 4–5/9, by ≤ 0.02.
- **Norm flattening:** passes in all 9 shared-token runs.
- **Accuracy invariance:** passes.

## What it means

**Where both analyses agree:**
1. **More operand attention gives a small, general gain** when the norm is held fixed (≤ 0.39, closing at most 7.8 % of a gap), about the same for diagonal and off-diagonal pairs. So diagonals' lower operand mass is not what costs them margin.
2. **With the norm free, more operand attention hurts, and hurts diagonals more.** The extra content raises the residual norm, and the final RMSNorm shrinks every logit. That explains the first test's odd "more attention, worse diagonal" result, largely but not entirely: in control s0, the group-mean swap hurts even at fixed norm (−0.59).
3. **On diagonals the residual at '=' carries extra energy that doesn't help the answer.**
   - Codex: "redundant residual content".
   - Claude: the coherent doubling of the identical operand value (2·v_a).

   Both mark this as inference, not yet measured.

**Where they differ: only in the next step**, and the two are complementary.
- **Claude:** locate where the extra norm comes from, by splitting the residual at '=' into its components.
  - Predicted under doubling: layer-0 attention output about √2 ≈ 1.4× larger on diagonals.
  - The alternative: the extra norm first appears in the MLPs.
- **Codex:** split the residual into the part the answer readout can see and the part it can't; remove the invisible part and redo the norm.
  - Under the redundancy account: diagonals carry more invisible energy, and removing it shrinks both their scale disadvantage and the free-norm penalty.

Both run on existing checkpoints, take about a minute on the GPU, and use about 1.2 GB of RAM.

## Files

All in `grok_runs\mechanism_2026-11-03\`:
- code: `restore_test.py` (v1), `restore_test2.py` (v2);
- results: `restore_results.md`, `restore2_tables.md`, raw data in `restore\`;
- reviews: `codex_restore_review.md` (code review), `codex_restore2_analysis.md` (blind analysis);
- my analysis: `claude_restore2_analysis.md`;
- predictions and results: `hypotheses.md` (H6, H6c).
