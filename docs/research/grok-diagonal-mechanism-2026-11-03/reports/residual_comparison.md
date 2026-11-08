# H7: Claude vs blind Codex

Files: `claude_residual_analysis.md` (written first), `codex_resid_analysis.md` (blind).

## Agreement
- **Verdicts are identical:**
  - H7a (i) pass;
  - H7a (ii) pass at exactly 7/9;
  - H7b (i) pass, 9/9;
  - H7b (ii) fail, 3/9;
  - H7b (iii) fail, 0/8.
- **attn0's ratio of about 1.42 is coherent doubling** (√2). Separate-token runs lack it.
- **mlp0's write is smaller on diagonals** (RMS ratio 0.74–0.88). attn1 adds 28–59 % of the excess.
- **Invisible energy is a minor factor.** Stripping closes −1 % to 7 % of the gap. The deficit lies in the visible directions.
- **Neither analysis shows that the early doubling causes the margin deficit.** That needs a causal test.

## Codex raised; my response
- **"Rank 98 is impossible for 97 classes."** Checked; it's not a bug. `cols` = `arange(n + 2)` = 99 columns: 97 answers plus the op and '=' tokens. That is the class set every margin in this study uses, so the centered rank is ≤ 98. The visible subspace also includes those two tokens' contrasts, which makes stripping slightly conservative and doesn't change any verdict.
- **Shares are signed accounting, not causal attribution; negative ≠ active cancellation.** Accepted; I corrected my wording.
- **Borderlines, from the unrounded values:**
  - add_s0 keeps 48.25 % of its scale gap (passes); mul_s0 keeps 51.26 % (fails).
  - control_s2 barely qualifies for (iii) at −0.103, with 33.7 % attenuation.
  - None of these changes a verdict.

## I raised; Codex didn't
- **Stripping has a built-in ceiling.** By construction it leaves the visible logits unchanged, so it acts only through the final-norm scale. It could never close more than H6's scale swap did (15–30 %), and (iii) was bound to fail, since the operand offset adds visible content. H7b tested less than its registration implied. The registration was my design, so I take that on.
- **H7b (ii) sorts by how much of the excess is invisible.** The invisible part of the excess is 0.61–0.99 in the three passing runs and 0.33–0.58 in the six failing ones. In most runs the excess energy itself is mostly visible.

## Next experiment: Codex's design is sharper; merged version
**Scale the operand-only part of the layer-0 attention write at '='** (the a and b terms, not the op or '=' terms) by gains 0.707, 0.85, 1.00 and 1.15. Do this separately on diagonal and off-diagonal pairs. Run three modes:
1. **Replay:** recompute everything downstream. This measures the total effect, including how mlp0, attn1 and mlp1 respond.
2. **Final-only:** add the same change to the final residual, with downstream writes cached. This measures the direct effect only.
3. **Both modes again with the final inverse RMS fixed** at baseline, to remove the scale part.

**Readings:**
- **Doubling causes the deficit through downstream processing:** replay at 0.707 lifts diagonal margins well beyond final-only, even with the norm fixed. The same gain on off-diagonal pairs hurts.
- **Later discrimination is deficient on its own:** gain 0.707 gives little selective rescue in either mode.
- **If replay helps:** follow up with a content swap. Replace the diagonal's operand write with 0.43·(v_a + v_c), c ≠ a, at matched norm. This separates magnitude from single-direction content.
