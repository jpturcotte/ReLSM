# Claude's analysis of same-answer patching (H9)

Written 2026-11-03, before reading `codex_patch_analysis.md`. Source: `patch_results.md` (including T2b, which I added after the smoke test and which is not registered).

## Validity
- **Checks pass in all 12 runs:**
  - P_both reproduces the donor within 2.3e-5;
  - self-patches change nothing beyond 1.9e-5;
  - the forward matches model(x) within 3.1e-5.
- **P_op0 and P_eq agree to within 0.01 nats everywhere. I read this as expected, not a bug.**
  - After layer 0, the '=' residual is emb('=') + attn0 + mlp0.
  - emb('=') is fixed. The op and '=' terms of attn0 barely vary across pairs. mlp0 is a function of the '=' residual.
  - So the operand write carries all the pair-specific content of the '=' stream after layer 0.

## Registered verdicts
| | result | verdict |
|---|---|---|
| C1 P_op0 closure ≥ 40 % | −19 % to −236 % | fail, 0/9 |
| C2 P_op0 − P_size ≥ 25 pts | all negative | fail, 0/9 |
| C3 P_eq − P_ops ≥ 25 pts | 5/9 | fail |
| Alternative (P_ops ≥ 40 %, P_op0 ≤ 20 %) | 0/9 | fail |

**Overall: mixed, and the design can't answer its question.**
- Patching parts of one off-diagonal pair into another off-diagonal pair with the same answer costs 3.3–7.2 nats (T2, P_op0 / P_eq / P_ops). Separate-token runs pay 1.2–1.5.
- That mismatch cost is larger than the diagonal deficit itself (gap 3.6–6.2), so it swamps every closure.
- I should have predicted this. If the two streams have to agree about which pair is present, mixing them breaks the computation. I flagged it at the smoke test and ran the test as registered anyway, rather than amending after a peek.

## What is still learned
**Shown:**
1. **The answer is not stored in either stream after layer 0.**
   - Layer 1 needs the '=' stream and its reads of the operand positions to describe the *same* pair.
   - Two pairs with the same answer can't be mixed: accuracy on mixed diagonals drops to 0.65–0.95.
   - So the model computes the answer jointly in layer 1 (and mlp1) from both streams, not by reading a finished answer.
2. **Size alone helps little.** P_size changes diagonal margins by −0.04 to +0.71 (closure −1 % to +15 %), matching H8.
3. **Mixing costs far more in shared-token models** (3.3–7.2 nats) than in separate-token ones (1.2–1.5). Shared-token models tie the two streams together more tightly.

**Inferred, from T2b (not registered).** This is a difference-in-differences: diagonal Δ minus off-diagonal Δ, scaled by the diagonal gap.
- **P_ops** (the diagonal's own '=' stream with a donor's operand positions) hurts diagonals more than it hurts off-diagonals in 7 of 9 runs (−4 % to −54 %; the other two are −0 % and +1 %).
  - So a diagonal's '=' stream fits other pairs' operand reads worse than an off-diagonal's '=' stream does.
  - That fits "the diagonal's '=' stream is atypical": the content account.
- **P_op0 / P_eq** (a donor's '=' stream with the diagonal's own operand positions) has mixed sign: +22 % to +43 % in 5 runs, −5 % to −51 % in 4.
  - If the deficit lived only in the '=' stream, this should be clearly positive. It isn't.
  - The likely reason: the diagonal's operand positions are atypical too (position 2 attends over a, op, a in layer 0), so both streams carry the (a, a) peculiarity.
- These differences use recipients with different baseline margins (diag about 10 vs off about 14.5), so they are suggestive only.

## Corrected design: small moves instead of swaps
**Mechanics:**
- Move one stream a small step toward the donor: h ← h + α · (h_donor − h), with α = 0.1, 0.2, 0.3 in that stream only.
- Measure the margin slope with respect to α, per stream, for diagonal recipients and for off-diagonal recipients. Same donors as H9.
- Small α keeps the mismatch small.
- The off-diagonal slope is the reference for the mismatch cost. The diagonal minus off-diagonal slope is the signal.

**Predictions:**
- **Deficit in the '=' stream:** for the '=' stream, diagonal slope minus off slope is clearly positive (moving a diagonal toward typical helps it). For the operand positions, the difference is ≤ 0.
- **Deficit in layer 1's reads of the operand positions:** the reverse.
- **Both:** both differences positive.

Thresholds would be registered before running.
