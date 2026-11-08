# Claude's analysis of the v2 probe results

Written 2026-11-03, before reading the blind Codex review (`codex_results_review.md`). Source: `results_tables.md`, plus a direct look at the per-position layer-1 attention on diagonals.

## 1. Validity

**What checks out**
- **Logged vs probe:** lse gap_train agrees within 0.02 in all 27 runs.
- **Counts and errors:**
  - group counts match every run record;
  - split reconstruction error ≤ 4e-5;
  - forward error about 2e-5;
  - chunked vs unchunked: largest difference 1e-6 over 1,009 numbers.
- **ratio_L0 behaves as designed:**
  - shared-token runs: diagonal logits change by ≤ 2e-5;
  - separate-token runs: they change by 1.0 to 2.9, as they should, since the two operand values differ there.

**Caveats**
- **ratio_L1 isn't comparable across tasks.** Sub, div and s5 read the b position only in layer 1 (operand ratio about 0.002–0.01). Forcing it to 50/50 wrecks them: off-diagonal margins drop by 2 to 10. Their "shrink" over 100 % is damage, not a diagonal effect. Only add, mul and diag_control already sit near 50/50 in layer 1 (mean ratio 0.47–0.50), so only there is ratio_L1 a like-for-like test.
- **Small samples.** There are about 50 diagonal train pairs per run, so single-checkpoint gaps (T6) move by roughly ±0.5 between checkpoints. Percentages in T6 with a base under about 1 are noise.

## 2. Verdicts against the written thresholds

| Hypothesis | Verdict | Deciding numbers |
|---|---|---|
| H1, prediction 1 (token dependence ≥ 2× in add/mul vs sub/div/a+2b) | **FAIL** in layer 0, the only layer the premise covers | Largest head's ab_diff_std is 0.070–0.118 for add/mul vs 0.034–0.071 for sub/div/a+2b (add_s1 0.070 vs div_s2 0.071). Layer 1 passes (0.54–0.60 vs ≤ 0.27), but layer 1 is outside the premise. |
| H1, prediction 2 (freezing shrinks the gap > 50 %) | **FAIL** | freeze_L0: −2 to 0 %. freeze_L1: 27–52 % (only add_s1 passes). |
| H1b (ratio_L0 shrinks ≥ 50 %) | **FAIL** | −2 to 0 % in all six add/mul seeds. |
| H2 (one component ≥ 60 %), descriptive | **FAIL** | No single component in every seed: add_s2's largest is mlp0 at 57 %, mul_s0's is attn1 at 54 %. |
| H3 (gap < 1 at update 500; grows only once generalization starts) | **FAIL** | Logged train gap at 500 is 1.99–2.53 in every add/mul seed. It reaches ≥ 1 at update 100, before off-diagonal test accuracy reaches 0.5 (updates 300–500). Caveat: in E1, add/mul have almost no memorization-only phase (off-diagonal test 0.72–0.98 at update 500). |
| H4, removal | **FAIL** | Removing operand classes: −1 to 0 %; random removal is the same. |
| H4, runner-up | **FAIL** | The runner-up is the operand on 0–2 % of add/mul diagonals. |

## 3. Mechanism account

**What the data shows**

**(a) Two phases in add/mul.**
- **Early (updates 500–2,500):**
  - the gap is already 2–4;
  - in the split it sits mostly in mlp1 (+1.2 to +2.5), with mlp0 growing;
  - attention interventions do nothing then: ratio_L1 and freeze each shrink the gap by ≤ 2 %.
- **Late (5,000–25,000):**
  - the gap grows to 3.7–6.2;
  - attn1 becomes a steady contributor, +2.0 to +2.5 in all nine add/mul/diag_control seeds (3.8 in one);
  - mlp1 turns negative and partly cancels it.

**(b) The late part is in layer-1 attention at '=', and it is diagonal-sensitive.**
- Off the diagonal, layer-1 heads put about 0.80–0.85 of their weight on the two operand positions, split about evenly, with a strongly token-dependent ratio (std 0.26–0.31; sub/div/s5/a+2b/sepops: ≤ 0.14).
- On diagonals, the same heads drop operand mass to 0.58–0.76 and move it to `op` and `=`. Example, add_s0 L1h0: off-diagonal weights (a, op, b, =) = (0.40, 0.08, 0.41, 0.11); diagonal = (0.28, 0.20, 0.30, 0.22).
- Equalizing the layer-1 ratio removes 1.6–2.3 from off-diagonal margins and nothing from diagonal ones. That is 32–49 % of the gap, matching the size of the attn1 term.
- So a late, token-dependent layer-1 read helps off-diagonal pairs and is not available on diagonals.

**(c) Layer 0 reads the operands symmetrically in add/mul:** ratio 0.50 ± 0.04, the same on and off the diagonal.

**(d) Sub and div build the order-dependent part at the b position.** In layer 1, '=' reads only the b position. A non-commutative task can't use the symmetric read.

**What I infer (not shown)**

**(e) The early gap may come from how many training examples each fact gets under the paired split.**
- `make_split(pairs="paired")` keeps (a, b) and (b, a) together, so each unordered off-diagonal fact has 2 training examples and each diagonal fact has 1.
- A model whose first layer reads a and b symmetrically sees (a, b) and (b, a) as nearly the same input. Off-diagonal facts then get about twice the gradient, so a higher margin from the start.
- This also fits the controls:
  - **sub/div:** the mirror is a different fact, so nothing is doubled. Their diagonal answers are also constant (0 for sub, 1 for div), which explains the negative early gap of −1.8 to −4.1: the diagonal is easier.
  - **separate tokens:** (a, b) and (b, a) are different token sets, so no doubling; gap about 0.
  - **a+2b shared:** not commutative; gap ≤ 1.
- **Weak point:** doubling explains the train gap, not the larger test gap (7–8.8), since neither held-out group was trained. For that I'd add a geometric reading. With a symmetric read, (a, a) is the extreme point v_a + v_b = 2·v_a, a thin region of input space (p of p² pairs) that the generalizing circuit covers poorly.

**Ruled out**
- the operand token as the competitor (H4);
- computing in the layer-0 a/b ratio (H1b);
- one component carrying the deficit (H2);
- the gap appearing only with generalization (H3).

**Left open**
- whether the early gap is the doubling, the geometry, or both;
- whether the late layer-1 operand-mass drop on diagonals causes margin loss, or only correlates with it.

## 4. Surprises

- **sub_s2:**
  - its final gap is negative (−0.57);
  - its layer-1 solution differs from the other sub seeds: ratio 0.32 and mass 0.80, instead of routing to b.
- **s5_s1 and a+2b shared use an asymmetric layer 0:**
  - their layer-0 ratios are 0.008, 0.258, 0.258 and 0.012, and ratio_L0 destroys them (off-diagonal −7 to −12);
  - a+2b s0/s1 put 0.26 on a, near the 1:2 coefficient (1/3), while s2 found another solution (0.012).
- **mul_s2:** diagonal test accuracy is 0.848 while off-diagonal test is 1.000.
- **Test vs train:** in every add/mul run, the test gap (7.0–8.8) exceeds the train gap.

## 5. Next experiments

**1. Decisive, and needs no training: the mirror test on the diag_control checkpoints.** That split (ordered) puts (a, b) and (b, a) independently. Split the off-diagonal train pairs by whether the mirror is also in train.
- *Doubling:* pairs with the mirror absent sit near the diagonal margin (the gap to the diagonal shrinks ≥ 50 %); pairs with the mirror present carry the gap.
- *Geometry:* both kinds sit well above the diagonal.

**2. Backup, needs training: weight the diagonal training examples 2× in paired E1 add.**
- *Doubling:* the early train gap goes to about 0.
- *Geometry:* it persists.

**3. Backup, no training: restore layer-1 operand mass on diagonals.** Set their layer-1 operand weights at '=' to the off-diagonal mean, keeping the ratio.
- *Causal:* the late layer-1 mechanism raises diagonal margins by about 2.
- *Correlational:* little change.
