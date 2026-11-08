# Claude's analysis of restore tests v2

Written 2026-11-03, before reading `codex_restore2_analysis.md`. Source: `restore2_tables.md`.

## Validity

All checks pass in all 12 runs:
- the manual forward matches `model(x)` within 3.1e-5;
- both self-target no-ops are exactly 0;
- attention rows sum to 1 within 2.1e-4;
- accuracy is unchanged by every norm-only swap.

The rerun group-mean tests reproduce the first version's numbers exactly (e.g. add_s0: 4.37, 2.94, 2.84).

## H6c verdicts at the written thresholds

**Attention (all heads, δ = +1, norm fixed, Δ diag_train):**
- add: 0.15 / 0.39 / 0.38;
- mul: −0.01 / 0.26 / 0.14;
- control: −0.06 / 0.24 / 0.34.

> 0.25 in 4 of 9 (needs 7 for "limiting"); ≤ 0.1 in 2 of 9 (needs 7 for "not limiting"). **MIXED.**

The second condition also fails. Diagonal Δ beats off-diagonal Δ in only 5 of 9, by at most 0.02 (add 0.15 vs 0.14, 0.39 vs 0.39, 0.38 vs 0.37; mul 0.26 vs 0.25; control 0.34 vs 0.32). In the other four, the off-diagonal gains more (mul −0.01 vs 0.13, 0.14 vs 0.24; control −0.06 vs 0.19, 0.24 vs 0.32).

**Norm flattening control:** |Δ diag_train| ≤ 0.05 in all 9 shared-token runs. So the diag→off swap's effect is the scale difference itself, not the flattening. (Separate-token s0 is 0.30.)

**Accuracy invariance:** holds everywhere.

## What the offset curves show

1. **Norm fixed: more operand attention helps every pair about equally.** At δ = +1, diagonal and off-diagonal gain about the same: within ±0.15 in every shared-token run, and within 0.02 in 5 of them. So nothing about the layer-1 operand read is specifically starving diagonals. Their lower operand mass (0.60–0.75 vs 0.77–0.84) is not what costs them margin.
2. **Norm free: more operand attention hurts, and hurts diagonals more.** At δ = +1, diagonal Δ is −0.05 to −0.74 vs off-diagonal −0.06 to −0.27 (e.g. mul s0 −0.62 vs −0.27; control s0 −0.74 vs −0.23). The extra operand content raises the residual norm at '=', and the final RMSNorm then shrinks every logit; the norm-fixed curves show the pre-norm effect was positive. The penalty is larger on diagonals. My inference: on a diagonal, the two operand positions carry very similar layer-1 values, so extra weight on both adds norm coherently.
3. **This explains the first test's "raising mass lowers diagonal margins".** That test let the norm recompute.
4. **Heads:** with norm fixed, the per-head effects are small (−0.10 to +0.16), roughly additive, and none is specific to diagonals.
5. **Separate tokens respond more strongly with norm fixed** (+0.45 to +1.72 at δ = +1) on both groups. Those models lean more on the operand read, but symmetrically: no diagonal-specific deficit, consistent with their gap being about 0.

## My account

**Shown by the data:**
- Layer-1 operand attention is not the cause. Changing it moves diagonal and off-diagonal margins together.
- The diagonal's final-norm scale is 6–12 % lower (its residual at '=' is 6–12 % larger in RMS). Restoring the scale closes 15–30 % of the gap, purely by rescaling logits; accuracy doesn't change.
- **The remaining 70–85 % is direction, not size.** The diagonal residual at '=' is less aligned with the correct answer's unembedding direction relative to its competitors.

**Inferred:**
- The larger diagonal residual is extra content that doesn't help the answer.
- The simplest source: layer 0 reads a and b at about 0.43 each. On a diagonal the two value vectors are identical, so they add coherently (|2v| = 2|v|) instead of partly cancelling as for a ≠ b (about √2|v| when nearly orthogonal).
- That gives a larger, differently shaped input to everything downstream, which is the "(a, a) is a special input" account the mirror test favoured.

## Next step (no training, about a minute on the GPU)

Decompose the diagonal's extra residual norm at '=' by component (emb, attn0, mlp0, attn1, mlp1), and measure each component's projection onto the margin direction.

**Prediction under coherent doubling:**
- the layer-0 attention output at '=' has RMS about √2 ≈ 1.4 times larger on diagonals than off-diagonals;
- that excess carries little margin-direction signal;
- most of the residual's extra norm traces back to attn0 and the MLP0 response to it.

**Under the alternative** (the deficit is created later, in the MLPs), attn0's norm ratio is about 1, and the extra norm first appears in mlp0 or mlp1.
