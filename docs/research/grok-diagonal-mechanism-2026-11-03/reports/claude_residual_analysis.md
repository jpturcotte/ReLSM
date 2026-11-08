# Claude's analysis of the residual tests (H7)

Written 2026-11-03, before reading `codex_resid_analysis.md`. Sources: `residual_results.md`, `residual\all.jsonl`.

## Validity

- **Components add up exactly.** The five writes sum to the final residual with error 0 in all 12 runs.
- **Stripping leaves the visible logits alone.** Removing the invisible part changes the centered pre-norm answer logits by at most 5e-4.
- **The excess-share attribution is exact but signed.** ||h||² = Σ c_i·h holds exactly, so the five shares sum to 1. A share says how much a component's write lies along the final residual. It is not the component's own size. A negative share (mlp0) means the write points against the excess and cancels part of it.
- **A design limit I missed when registering H7b.** Stripping keeps the visible logits fixed by construction (see the check above). So it can change margins only through the final-norm scale. Its ceiling is therefore the full scale swap from H6, which closed 15–30 % of the gap. It can never test the other 70–85 %, the direction part.

## Verdicts at the written thresholds

| prediction | result | verdict |
|---|---|---|
| H7a (i) attn0 RMS ratio ≥ 1.25 in all 9 shared runs; in [0.9, 1.1] in the separate-token runs | 1.409–1.435 in all 9; separate 1.080, 0.929, 0.978 | **PASS**, by a wide margin |
| H7a (ii) attn0 + mlp0 share ≥ 0.5 in ≥ 7/9 | 7/9 (fails: add_s1 0.45, control_s2 0.42) | **PASS, borderline** (exactly at threshold) |
| H7b (i) invisible share diag > off in ≥ 7/9 | 9/9 (diag 0.133–0.229 vs off 0.090–0.153) | **PASS** |
| H7b (ii) scale gap halved by stripping in ≥ 7/9 | 3/9 (add_s0, control_s0, control_s1) | **FAIL** |
| H7b (iii) offset response ≥ 50 % less negative after stripping in ≥ 7 runs | 0/8 (e.g. −0.48 → −0.46, −0.74 → −0.74) | **FAIL** |

## What the components show

1. **attn0 is the source, and its size is exactly coherent doubling.**
   - Layer 0 reads a and b at about 0.43 each.
   - If the two value vectors have equal norm and are nearly orthogonal for a ≠ b, the write is 0.43·√2·|v| off the diagonal and 0.43·2·|v| on it, a ratio of √2 = 1.414.
   - Observed: 1.409 to 1.435 in all 9 shared runs, with a spread of only ±0.013.
   - In separate-token runs the two operands are different tokens even when a = b, so nothing doubles: 0.93 to 1.08.
2. **attn0 overshoots, then mlp0 cancels part of it.** (Corrected after reading Codex: a negative share means mlp0 adds less along h on diagonals. Its RMS is also lower, so "smaller write" is shown, and "active cancellation" is not.)
   - attn0 alone carries 0.88–2.18 of the excess; mlp0 carries −0.27 to −0.88.
   - mlp0's own RMS is 12–26 % smaller on diagonals (ratio 0.74–0.88).
   - So layer 0's MLP partly compensates for the larger operand write, but not fully.
3. **attn1 adds to the excess in every shared run.**
   - Its share is 0.28–0.59 and its RMS ratio 1.02–1.27.
   - This fits H6's free-norm finding: layer-1 operand content raises diagonal norm more than off-diagonal norm.
4. **mlp1 varies.** Its share ranges from −1.00 to +0.12, mostly negative, so it also cancels some excess.
5. **Emb is 1.000**, as it must be, since '=' is the same token for every pair.

## Visible vs invisible

- The diagonal residual is 10–28 % larger in squared norm.
- **The invisible part of that excess** (diag invisible energy minus off invisible energy, over the excess) **varies a lot by run: 0.33 to 0.99.**

| run | excess sq / off | invisible part of excess | scale gap halved? |
|---|---|---|---|
| control_s0 | 0.100 | 0.99 | yes (0.050 → 0.004) |
| control_s1 | 0.118 | 0.82 | yes (0.059 → 0.016) |
| add_s0 | 0.122 | 0.61 | yes (0.059 → 0.029) |
| mul_s0 | 0.152 | 0.58 | just misses (0.070 → 0.036) |
| control_s2 | 0.204 | 0.47 | no |
| add_s2 | 0.237 | 0.45 | no |
| mul_s2 | 0.250 | 0.37 | no |
| mul_s1 | 0.195 | 0.35 | no |
| add_s1 | 0.276 | 0.33 | no |

**H7b (ii) sorts exactly by this column.** Runs whose excess is more than about 0.6 invisible pass; the rest don't. So H7b (ii) didn't fail at random. In 6 of 9 runs, most of the extra energy sits in directions the readout does see.

**H7b (iii):** the extra norm from a layer-1 operand offset is visible content, so stripping can't touch it. The 0/8 result is what the design would give regardless of mechanism.

**Gap closed by stripping: −1 % to 7 %.** That roughly equals the scale share H6 found (15–30 %) times the invisible fraction of the excess, as expected given the ceiling above.

## My account

**Shown:**
- The diagonal's extra residual norm starts at layer-0 attention.
- Its size is exactly √2, as coherent doubling predicts.
- MLP0 cancels part of it; attn1 adds more.
- Part of the excess is invisible to the readout (9/9), but removing it closes ≤ 7 % of the gap.
- **In most runs the excess is mainly visible, and the direction part of the gap (70–85 %) lies entirely in the visible logits.** No invisible-energy story can explain it.

**Inferred:**
- The direction deficit is downstream of the doubled operand write, in how mlp0, attn1 and mlp1 process a residual that holds 2·0.43·v_a instead of 0.43·(v_a + v_b).
- Two candidates:
  1. **Magnitude:** the operand part is √2 too large relative to emb('='), so the pre-norm MLP0 input lands in a different region. MLP0's under-response (ratio 0.74–0.88) is a symptom of this.
  2. **Content:** (a, a) carries one operand direction rather than two. Features the model builds from the interaction of two different operand vectors (e.g. products of a and b Fourier components) get a degenerate input on the diagonal.
- **I can't separate these two with the current data.**

## Next step (no training, about a minute on the GPU)

**Rescale the layer-0 attention write on diagonal pairs by 1/√2** before it enters the residual. This matches its norm to off-diagonal pairs while keeping its direction. Run the rest of the network and score the train gap. As a control, apply the same 1/√2 to off-diagonal pairs.

- **If magnitude is the cause:** the diagonal margin rises and closes ≥ 50 % of the gap. Off-diagonal margins drop under the control.
- **If content is the cause:** closure ≤ 20 %, because the operand direction is still single.
- **A sharper follow-up for the content account:** replace the diagonal's attn0 write with 0.43·(v_a + v_c) for c ≠ a, at matched norm. Then measure whether the margin for the target a + a recovers.
