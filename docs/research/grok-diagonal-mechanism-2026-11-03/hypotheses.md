# Why are diagonal pairs (a, a) weak? Hypotheses, written 2026-11-03 before any probe ran

## What is known

**Large gap** (train gap 4.5–5.8; diagonal errors) when both hold:
- the operands share tokens;
- the task is commutative (add, mul).

**Small gap** (0.6–2.5) in the non-commutative tasks (sub, div, s5, a+2b), and about 0 with separate operand tokens.

**Fact about the model:** RoPE rotates only queries and keys. So with (a, a), every head reads the same value vector from both operand positions, whatever its attention weights.

## Hypotheses and the readout that decides each

### H1. Token-dependent attention
The commutative solution does part of its computation in the attention weights. How much weight goes to a versus b depends on the tokens themselves; Nanda et al. 2023 saw this in their one-layer add model.

On (a, a) the two value vectors are identical, so that contribution collapses to one fixed mix. Non-commutative solutions instead attend by position.

**Predictions:**
- **Token dependence of attention at '='** = the spread across off-diagonal pairs of w(a) − w(b). Taking each run's most token-dependent head, add and mul should come out at least 2× higher than sub, div and a+2b.
- **Freezing that layer's attention to its mean pattern** should shrink the add/mul train gap by more than half, by lowering off-diagonal margins more than diagonal ones.

**H1 fails if:** add/mul attention is no more token-dependent than sub/div's, or freezing it leaves the gap.

### H2. One component carries the deficit
Split the margin at '=' into what each part writes into the final residual: embedding, attention L1, MLP L1, attention L2, MLP L2. This split is exact.

**Prediction:** in add and mul, one component carries at least 60 % of the diagonal deficit, in every seed.

**H2 fails if:** the deficit is spread over components.

### H3. The gap appears with generalization, not with memorization
This uses the logged metrics, not the probe.

**Prediction:** in add and mul, the train gap is under 1 at update 500 (before grokking), and grows only once the off-diagonal test accuracy rises.

**H3 fails if:** the diagonal is already behind during memorization.

## Controls

- **Separate tokens** (diag-arms sepops runs, a+2b sepops runs): no deficit in any component, and freezing attention changes nothing specific to the diagonal.
- **a+2b with shared tokens:** a non-commutative task on the same `ordered` split as the diagonal-arms add control.

## Runs probed

- E1 (paired split, `--tasks`): add, sub, mul, div, s5 × 3 seeds. Final weights, plus checkpoints at updates 500, 1,000, 2,500, 5,000 and 10,000.
- Diagonal arms: control and sepops × 3. Final weights only.
- a+2b runs: shared and sepops × 3. Final weights only.

## Precision

Margins are recomputed in fp32 on CPU, so they differ slightly from the bf16-logged ones.

---

# Revision after the Codex review (GPT 6.1 Sol), 2026-11-03, before the revised probe ran

**Seen before this revision:** first-version probe numbers (top-competitor margin only) for these final weights:
- add s0–s2, mul s0 and sub s0–s2;
- their checkpoints at updates 500, 1,000, 2,500, 5,000 and 10,000.

So the new predictions below are partly informed by those numbers. Read them as **revised, not blind**.

## Corrections

1. **Margin definition.**
   - The primary margin is now "lse": the correct logit minus the logsumexp of the wrong ones, which is what grok_lab logs. The first probe used "top", the correct logit minus the strongest wrong one; both are reported.
   - The probe also reads the logged margins at the same update, so precision and aggregation differences can be seen.
   - E1's headline gaps are late-window averages (the last 10 % of evals). The probe reads single checkpoints.
2. **H1's premise holds in layer 0 only.**
   - In layer 0, at evaluation, the two operand values of (a, a) are identical. Only the split *between* the operand positions stops mattering; the total operand mass can still depend on a.
   - In layer 1 the two positions differ. In training, embedding dropout also makes them differ.
   - So layer-1 token dependence does not test H1.
3. **The registered H1 intervention** (whole-layer freezing) **failed its threshold.** add s0 shrank by 40 %, below the required 50 %. It also changes many things at once, so a narrower test is added (H1b).
4. **H2 is descriptive accounting, not cause.** Two components can each pass the 60 % rule while another cancels them. It is reported as an accounting of the margin only.
5. **H3, operational definitions.**
   - Memorization at update 500 means train accuracy ≥ 0.9 there.
   - Generalization starts at the first eval with off-diagonal test accuracy ≥ 0.5.
   - The registered prediction stands as written.

## New tests

### H1b. Computing in the operand ratio (explanation a)
**Intervention:** at '=' in layer 0 only, set each head's operand weights to (m/2, m/2), keeping its operand mass m. Nothing else changes, and diagonals are unchanged by construction.

**Prediction:** in add and mul, this lowers the off-diagonal train margin (lse) enough to shrink the train gap by ≥ 50 %, in every seed.

**Fails if:** the shrink is under 50 % in any add/mul seed. The first probe's near-constant layer-0 attention (std of w_a − w_b about 0.08) makes failure the likely outcome.

### H4. The operand token as the competitor (explanation b)
**Predictions:**
- **Removal:** in add and mul, removing the operand classes from the competitors shrinks the train gap (lse) by ≥ 50 % in every seed. Removing the same number of random wrong classes shrinks it by < 15 %.
- **Runner-up:** on train diagonals of add and mul, the runner-up is the operand token in ≥ 50 % of pairs. In sub and div it is ≤ 20 %.

**Fails if:** either the removal or the runner-up prediction misses in any add/mul seed.

### Explanation c (symmetric read)
No direct test here. If H1b and H4 both fail, c is what remains and needs its own experiment.

## Controls kept
- separate-token runs;
- a+2b shared;
- group counts checked against each run record.

---

# H5. The mirror test (exposure vs representation), registered 2026-11-03 before it ran

**Seen before this was written:** all v2 tables (`results_tables.md`), the blind Codex review, and my analysis. No per-pair margins split by mirror status have been computed yet.

## Setup

**Runs:** the ordered-split runs (final weights only, since no checkpoints were saved), where (a, b) and (b, a) land in train independently:
- diag_control (add, shared tokens) ×3: **the test**;
- diag_sepops (add, separate tokens) ×3: control;
- add2b shared ×3 and add2b sepops ×3: controls.

E1 add s0 (paired split) is a sanity check: every off-diagonal train pair there has its mirror in train.

**Groups (train pairs, lse margin):**
- `mirror`: off-diagonal, and (b, a) is also in train;
- `lone`: off-diagonal, and (b, a) is in test;
- `diag`: diagonal pairs in train.

**Quantities:**
- D_mirror = M_mirror − M_diag;
- D_lone = M_lone − M_diag;
- R = D_lone / D_mirror.

## Predictions for diag_control

- **Exposure** (under a symmetric read, (a, b) and (b, a) are one fact, so mirror pairs get about 2× the examples): lone pairs sit near the diagonal. **R ≤ 0.5 in all 3 seeds.**
- **Representation** ((a, a) is a distinctive input; exposure doesn't matter): lone pairs sit with mirror pairs. **R ≥ 0.85 in all 3 seeds.**
- **Anything else:** both accounts are partly right, or the seeds disagree. Report it as mixed, with R per seed.

## Controls

**add2b shared and diag_sepops:**
- add2b shared: the mirror is a different fact;
- diag_sepops: (a, b) and (b, a) are different token sets, so nothing merges.

Both accounts predict |M_mirror − M_lone| < 0.5 there. If a control shows a mirror effect ≥ 0.5, then "mirror in train" is confounded with something else, and the diag_control result must be discounted.

## Also reported (descriptive, no threshold)

- held-out off-diagonal pairs split by whether their mirror is in train (seen vs unseen);
- the standard error of each group mean.

## Limit

Final weights only. The early-training gap, where exposure should matter most, is not covered here.

## H5 result at final weights (2026-11-03 10:50)

**Representation.** R = 0.96 / 0.89 / 0.91 (diag_control s0/s1/s2), so all three seeds are ≥ 0.85. No control shows |mirror − lone| ≥ 0.5.

## H5b. The mirror test over training, registered before the reruns

The final-weights test left the early gap uncovered. diag_control and diag_sepops (3 seeds each) are rerun on CPU, with the same settings as the originals except for the device and no bf16. Checkpoints are saved at 100, 200, 300, 500, 1,000, 2,500, 5,000 and 10,000 updates. The LR is constant, so the early path does not depend on run length. These are new runs (CPU numerics), not copies of the GPU runs.

**Scoring:** a checkpoint counts only if D_mirror ≥ 1. The diagonal mean has a standard error of about 0.4–0.5, so smaller gaps give a meaningless R.

**Predictions for diag_control:**
- **Exposure explains the early gap:** R ≤ 0.5 at every scored checkpoint ≤ 1,000, in all 3 seeds.
- **Representation from the start:** R ≥ 0.85 at every scored checkpoint, in all 3 seeds.
- **Exposure early, then representation:** R rises from ≤ 0.5 to ≥ 0.85 over training.
- **Anything else:** report it as mixed.

**Control:** in diag_sepops, |mirror − lone| < 0.5 at every checkpoint.

## H5b result (2026-11-03 12:20)

**diag_control R by update (s0 / s1 / s2).** Brackets mark checkpoints where D_mirror < 1, which the rule leaves unscored.

| update | s0 | s1 | s2 |
|---|---|---|---|
| 100 | (0.22) | 0.24 | (0.22) |
| 200 | (0.57) | 0.56 | 0.68 |
| 300 | (0.82) | 0.83 | 0.88 |
| 500–10,000 | 0.87–0.95 | 0.87–0.95 | 0.87–0.95 |

**By the registered scoring: mixed across seeds.**
- s1 meets "exposure early, then representation";
- s0 meets "representation" (its scored checkpoints start at 500);
- s2 meets neither (scored from 200, at 0.68).

**Descriptively, all three seeds trace the same curve.** The seed differences come only from the D_mirror ≥ 1 gate. The diagonal's standard error early on is 0.05–0.13, far below the 0.4–0.5 that the gate was set for.

**Control:** diag_sepops shows |mirror − lone| ≤ 0.11 at every checkpoint (registered bound 0.5), so the control is clean.

---

# H6. Restore tests for the late gap, registered 2026-11-03 before they ran

**Script:** `restore_test.py`.

**Runs:** final weights of E1 add ×3, E1 mul ×3, diag_control ×3, plus diag_sepops ×3 as a control (its gap is about 0, so every change should be about 0).

**Margin:** lse, train groups. "Closes X %" = (base gap_train − gap_train after) / base gap_train.

**H6a. Layer-1 operand mass at '='.**
- *Intervention:* on diagonal examples, set each layer-1 head's operand mass to that head's off-diagonal train mean, keeping the a/b ratio and rescaling `op` and `=`.
- **Causal** if it closes ≥ 25 % of the train gap in all 9 add/mul/control runs.
- **Not causal** if it closes < 10 % in all 9.
- Anything else is mixed.
- *Control direction, reported without a threshold:* lower the off-diagonal mass to the diagonal mean and record how far the off-diagonal margin moves.

**H6b. Final-norm scale at '='.**
- *Intervention:* on diagonal examples, set the final RMSNorm inverse RMS to the off-diagonal train mean.
- Same thresholds as H6a: ≥ 25 % in all 9 means it matters; < 10 % in all 9 means it doesn't.
- *Control direction:* lower the off-diagonal scale to the diagonal mean.

**Also reported:** both interventions together.

**Sanity:**
- the base gap must match the v2 probe within 0.02;
- separate-token runs: |Δ gap| < 0.3 for every intervention.

## H6 result (2026-11-03 13:16), before the code review

**Registered verdicts:**
- H6a: NOT CAUSAL (−32 % to +3 %);
- H6b: MIXED (15–30 %);
- the separate-token sanity bound failed (|Δ| up to 1.72).

**The Codex review (`codex_restore_review.md`) found no bug that affected these numbers:**
- base gaps match v2 within 3e-6;
- two latent bugs, in unused options (chunk argument; per-example norm targets).

**But:**
- **The sanity bound was ill posed.** The separate-token runs have their diagonal scale above the off-diagonal one.
- **The norm swap mostly measures mechanical logit rescaling.**
- **The mass result depends on the final norm.** Mass and norm swapped together, minus the norm swap alone, gives +0.08 to +0.36 on diagonals in 7/9 runs.

---

# H6c. Cleaner tests after the review, registered 2026-11-03 before they ran

**Fixes:**
- chunk argument;
- per-example norm targets;
- checks: row sums, self-target no-ops (mass and norm), an examplewise match of the unmodified forward against `model(x)`.

**Attention test (per head and all heads): score offsets.**
- At layer 1, query '=', add δ ∈ {−1, −0.5, +0.5, +1} to the scores of both operand positions. This changes operand mass smoothly and keeps the a/b ratio.
- Apply it to diagonal train examples, and separately to off-diagonal train examples, as a comparison.
- Each runs with the final-norm inverse RMS either free, or fixed at each example's own baseline value.

**Predictions (all-heads offset, norm fixed, add/mul/control runs):**
- **Too little operand mass limits diagonals** if Δ diag_train margin at δ = +1 is > +0.25 in ≥ 7 of 9 runs, and larger than Δ off_train at δ = +1 in ≥ 7 of 9.
- **Not limiting** if Δ diag at δ = +1 ≤ +0.1 in ≥ 7 of 9.
- Anything else is mixed.

**Norm tests:**
- **Flattening control:** set each diagonal's inverse RMS to the diagonal mean. Its effect (|Δ diag_train|) is reported; if < 0.2, the diag→off swap's effect is the scale difference itself, not flattening.
- **Accuracy:** must be unchanged by any norm-only swap (it is a positive rescale per example).

**Then:** a blind Codex analysis of these results, compared with mine.

## H6c result (2026-11-03 13:30)

**Attention: MIXED.**
- Δ diag at δ = +1 with norm fixed is > 0.25 in 4/9 and ≤ 0.1 in 2/9.
- Diagonal beats off-diagonal in only 4–5/9, by ≤ 0.02.

**Norm flattening control:** |Δ| ≤ 0.05 in all 9 shared-token runs. The scale swap's effect is the scale difference itself. (Separate-token s0 is 0.30.)

**Accuracy invariance:** holds.

**The blind Codex analysis (`codex_restore2_analysis.md`) agrees on all three verdicts.** Compared with mine in `restore_final_report.md`.

---

# H7. Where the diagonal's extra residual energy comes from, and whether the answer can see it. Registered 2026-11-03 before it ran

**Script:** `residual_test.py`.

**Runs:** final weights of the same 12 runs (E1 add ×3, mul ×3, diag_control ×3; diag_sepops ×3 as the control).

**Readout:** the final residual h at '=' (fp32, pre-norm) = emb + attn0 + mlp0 + attn1 + mlp1. Train groups.

## H7a (Claude): the source of the extra norm

**Readouts:**
- each component's RMS at '=', and its ratio diagonal / off-diagonal;
- each component's share of the diagonal's excess squared norm. Since ||h||² = Σ_i c_i·h, the share is (mean c_i·h on diag − mean on off) / (mean ||h||² on diag − mean on off).

**Predictions under coherent doubling** (layer 0 reads a and b at about 0.43 each; on (a, a) the two values are identical and add coherently):
- (i) attn0's RMS ratio diag/off is ≥ 1.25 in all 9 shared-token runs, and between 0.9 and 1.1 in all 3 separate-token runs;
- (ii) attn0 + mlp0 together carry ≥ 50 % of the excess squared norm in ≥ 7 of 9 shared-token runs.

**Alternative:** the excess first appears later. attn0's ratio is about 1, and attn1 + mlp1 carry most of the excess.

## H7b (Codex): energy the answer readout cannot see

**Definitions:**
- **Readout matrix:** A = W_U[answer columns] · diag(final-norm gain), centered over classes. Centering removes shifts that softmax ignores.
- **Visible part** of h: its projection onto the row space of A. **Invisible part:** the rest. Removing the invisible part leaves the pre-norm centered logits unchanged.
- **Stripping:** remove the invisible part, recompute the final RMSNorm, score again.

**Predictions under the redundancy account:**
- (i) the invisible share of ||h||² is higher on diagonals than off-diagonals in ≥ 7 of 9 shared-token runs;
- (ii) after stripping, the diagonal's scale disadvantage, |1 − r_diag / r_off|, is at most half its baseline value in ≥ 7 of 9;
- (iii) the free-norm response to the layer-1 operand offset δ = +1 (all heads, on diagonals) becomes at least 50 % less negative with stripping, in ≥ 7 of the shared-token runs where the baseline response was < −0.1 (fewer runs if fewer qualify; all must then meet it).

**Main alternative:** weaker diagonal discrimination in the visible part, so none of (i)–(iii) holds.

**Also reported:**
- the train gap after stripping;
- the share of the gap closed.

**Checks:**
- stripping leaves pre-norm centered logits unchanged (max abs difference reported);
- the component sum equals h.

## H7 result (2026-11-03 13:55)

- **H7a (i) PASS.** attn0 RMS ratio:
  - shared runs 1.409–1.435 (√2, coherent doubling);
  - separate-token runs 1.080, 0.929, 0.978.
- **H7a (ii) PASS at threshold, 7/9.** Fails: add_s1 0.45, control_s2 0.42.
- **H7b (i) PASS, 9/9.**
- **H7b (ii) FAIL, 3/9.** The passing runs are those whose excess is more than 0.6 invisible.
- **H7b (iii) FAIL, 0/8.**
- **Stripping closes −1 % to 7 % of the gap.**
- **Design limit:** stripping fixes the visible logits, so it acts only through the norm scale (capped at H6's 15–30 %). (iii) could not pass under any mechanism.
- **Codex blind analysis agrees on all verdicts.** Its rank-98 flag was checked: cols has 99 classes (answers, op, '='), not a bug.
- **Conclusion:** the excess starts at attn0 (doubling), and the deficit lives in visible directions; invisible energy is minor. The causal link from doubling to deficit is untested.
- **Next:** operand-gain sweep (see `residual_final_report.md`).

# H8. Does the doubled layer-0 operand write cause the diagonal deficit? Registered 2026-11-03 before it ran

**Intervention (no training, final weights):**
- At '=', in layer 0, multiply the attention weights on the two operand positions (a and b), all heads, by a gain g, with no renormalization.
- o_proj has no bias, so this scales exactly the operand part of attn0's write at '=' and leaves its op and '=' terms alone. Other positions are untouched.
- Gains: 0.5, 0.707, 0.85, 1.0, 1.15, 1.414. Thresholds use g = 0.707 (1/√2, which matches the diagonal's operand write to an off-diagonal's in norm).
- Each example's forward is independent, so the gain is applied to all pairs at once and diagonal and off-diagonal train groups are read separately.

**Modes:**
- **replay:** recompute everything downstream (mlp0, attn1, mlp1 respond).
- **final-only:** baseline final residual + (g − 1) · the operand part of attn0 at '='. Downstream writes are kept at baseline, so this is the direct effect only.
- Each mode has two norm settings:
  - **norm free:** final inverse RMS recomputed;
  - **norm fixed:** final inverse RMS held at each example's baseline value.

**Measures:**
- **Δ diag** = change in mean lse margin on diag_train vs g = 1.
- **Δ off** = the same on off_train.
- **Closure** = Δ diag / baseline gap_train.

**Predictions under the doubling-causal account** (the √2 operand write on (a, a) causes the direction deficit through downstream processing), all at g = 0.707, replay, norm fixed, in the 9 shared-token runs:
- **D1:** closure ≥ 25 % in ≥ 7 of 9.
- **D2:** replay Δ diag exceeds final-only Δ diag (norm fixed) by ≥ 0.5 nats in ≥ 7 of 9.
- **D3:** selectivity: Δ diag − Δ off ≥ 0.5 nats in ≥ 7 of 9.

**Main alternative** (the later layers discriminate poorly on diagonals regardless of the doubling): closure ≤ 10 % in ≥ 7 of 9. Results between the two are mixed.

**Separate-token reference:** |Δ diag − Δ off| ≤ 0.3 nats at g = 0.707 (replay, norm fixed) in 3 of 3, since nothing doubles there.

**Also reported (no threshold):**
- the full dose curves, both modes, both norm settings;
- train accuracy on each group;
- the diagonal's mlp0, attn1 and mlp1 RMS at g = 0.707 vs the off-diagonal's at g = 1;
- operand attention mass at '=' in layer 0, diagonal and off-diagonal.

**Checks:**
- the manual forward matches model(x);
- replay and final-only at g = 1 equal the baseline;
- final-only and replay agree in the attn0 write itself.

## H8 result (2026-11-03 14:00)

- **Checks pass:**
  - forward vs model(x) ≤ 3.1e-5;
  - attn0 write error ≤ 2.4e-7;
  - g = 1 reproduces the baseline exactly.
- **D1 FAIL, 0/9.** Closure at g = 0.707 (replay, norm fixed) is −4 % to −23 %. Diagonal margins drop.
- **D2 FAIL, 0/9.** Replay is worse than final-only by 0.36–1.26 nats.
- **D3 FAIL, 1/9.**
- **Alternative (closure ≤ 10 %) SUPPORTED, 9/9.**
- **Separate-token reference PASS, 3/3.**
- **Conclusion:** the size of the doubled operand write does not cause the deficit. Matching it to off-diagonal size weakens both groups' signal about equally. Downstream layers process a size-matched (a, a) differently (mlp0 smaller, attn1 larger), which points to content.
- **Not ambiguous**, so no blind Codex pass.
- **Side note:** the separate-token models' layer-1 MLP write is exactly 0.

# H9. Where does the diagonal deficit live? Same-answer patching. Registered 2026-11-03 before it ran

**Idea:**
- Each diagonal train pair (a, a) is a **recipient**.
- Its **donors** are off-diagonal train pairs (b, c) with the same answer (for add, b + c = 2a; for mul, bc = a²).
- The target is unchanged, so patching parts of the donor's computation into the recipient shows which part carries the deficit.
- Up to 8 donors per diagonal, drawn with a fixed seed.

**Patches** (recipient forward, final norm free, final weights):
- **P_op0:** replace the operand part of attn0's write at '=' (the a and b terms) with the donor's. Then recompute mlp0 at '=' and everything after.
- **P_size:** the size-only control. Rescale the recipient's own operand part to the donor's RMS, per pair, keeping its direction.
- **P_eq:** replace the whole '=' residual after layer 0 (emb + attn0 + mlp0) with the donor's. Layer-1 keys and values at positions a, op, b stay the recipient's.
- **P_ops:** replace the residuals after layer 0 at positions a, op, b (layer 1's keys and values) with the donor's. The '=' stream stays the recipient's.
- **P_both:** replace all of them. This must reproduce the donor exactly (check).

**Measure:**
- closure = mean(patched margin − recipient margin) / mean(donor margin − recipient margin), over all (recipient, donor) pairs in a run;
- lse margin over the run's class columns, train diagonals only.

**Predictions under the content account** (the single-direction operand content that enters '=' at layer 0 causes the deficit), in the 9 shared-token runs:
- **C1:** P_op0 closure ≥ 40 % in ≥ 7 of 9.
- **C2:** P_op0 closure exceeds P_size closure by ≥ 25 points in ≥ 7 of 9.
- **C3:** P_eq closure exceeds P_ops closure by ≥ 25 points in ≥ 7 of 9.

**Main alternative** (the deficit comes from layer 1 reading the identical operand positions): P_ops closure ≥ 40 % and P_op0 closure ≤ 20 %, in ≥ 7 of 9. Anything else is mixed.

**Off-diagonal reference (reported, no threshold):**
- the same patches on up to 300 off-diagonal train recipients;
- each with up to 2 same-answer off-diagonal donors, excluding its mirror;
- mean Δ margin per patch. Expected small if same-type swaps are benign.

**Separate-token runs:** the gap is about 0, so closures are not meaningful there. Only Δ is reported.

**Checks:**
- P_both equals the donor (max logit difference);
- patching a recipient with itself changes nothing;
- the manual forward matches model(x).

## H9 result (2026-11-03 14:20)

- **Checks pass:**
  - forward ≤ 3.1e-5;
  - P_both = donor ≤ 2.3e-5;
  - self-patch ≤ 1.9e-5.
- **Verdicts:** C1 FAIL 0/9; C2 FAIL 0/9; C3 FAIL 5/9; alternative FAIL 0/9 → mixed.
- **Design flaw, owned:** any partial patch between two different pairs with the same answer costs 3.3–7.2 nats, even off the diagonal. That is as large as the deficit, so the closures can't localize it. The flaw showed in the smoke test; the test ran as registered anyway.
- **Shown:** layer 1 needs the '=' stream and the operand reads to describe the same pair (interaction 10.9–20.7 nats). Size alone closes at most 15 %.
- **Blind Codex agrees on every point.**
- **Next design (Codex):** route-resolved integrated-gradient attribution. See `patch_comparison.md`.

# H10. Route-resolved attribution of the diagonal deficit (Codex's design). Registered 2026-11-03 before it ran

**Setup:**
- Layer-0 states are cached for each recipient (a diagonal train pair) and each donor (a same-answer off-diagonal train pair; 2 per diagonal, seed 0; all diagonal train recipients).
- The rest of the model is a function F of three groups of inputs:
  - **E:** the '=' state after layer 0. It supplies the residual, the layer-1 query, and the '=' self key/value.
  - **ops:** the layer-1 inputs at the operand positions 0 and 2.
  - **opr:** the layer-1 input at the op position 1.
- F returns the lse margin of the true answer, over the run's class columns.

**Path and attribution:**
- All three groups move together along a straight line from recipient (t = 0) to donor (t = 1).
- 32-point Gauss-Legendre quadrature on [0, 1].
- Attribution of a route = ∫ ∇_route F · Δ_route dt. The three routes sum to F(donor) − F(recipient).

**Two interpolation choices for the prefix positions:**
- **(A) pre-norm:** interpolate the residuals; layer 1's RMSNorm, K and V are recomputed.
- **(B) post-norm:** interpolate the normalized inputs, which is the same as interpolating K and V.
- E is interpolated as a residual in both.

**Controls:**
- up to 100 off-diagonal train recipients, each with 2 same-answer off-diagonal donors (no mirror);
- the separate-token runs are reported in nats only.

**Per-run classification** (9 shared-token runs; shares = route mean / mean total gap, diagonal recipients):
- **'=' account:** E ≥ 80 % and ops ≤ 20 %, under both A and B.
- **Operand-read account:** ops ≥ 80 % and E ≤ 20 %, under both A and B.
- **Third route:** opr ≥ 25 % under both A and B.
- **Coupled / unresolved**, if any of these holds:
  - cancellation: Σ |route means| > 2 × |total|;
  - any route mean differs by > 0.5 nats between A and B;
  - none of the above.

**Overall verdict:**
- an account wins if ≥ 7 of 9 runs get its label;
- otherwise mixed, with the labels counted.

**Also reported:**
- per-route nats for the diagonals and the off-diagonal controls;
- the fraction of diagonal pairs whose largest positive route is E vs ops;
- Σ |route| for the controls (control-path damage).

**Checks:**
- completeness: |Σ routes − (F(donor) − F(recipient))|, run mean ≤ 0.05 nats (max per pair reported);
- F(recipient) and F(donor) equal the model's own margins (≤ 1e-3);
- the manual forward matches model(x).

## H10 result (2026-11-03 14:15)

- **Checks pass:**
  - endpoints ≤ 3.5e-5;
  - completeness ≤ 0.028 (32 nodes), ≤ 0.006 (64 nodes);
  - route means stable to 0.022 nats between 32 and 64 nodes.
- **Registered verdict: MIXED.** 7 runs "unresolved (none)", 2 "'=' account" (control_s1, control_s2: E 82–85 %).
  - No cancellation trigger.
  - A vs B differ ≤ 0.36 nats.
- **Described, not registered (the registration had no "split" category):**
  - the deficit splits between the '=' stream (60–85 % in 8 of 9) and layer-1 reads of the operand positions (15–40 %);
  - both are positive;
  - the op position is ≈ 0 everywhere;
  - mul_s0 is the exception (ops 57–59 %).
- **Per pair,** routes are large and noisy in both diagonals and controls (Σ |route| 5–8 nats). The diagonal split is an average.
- **Blind Codex agrees on all verdicts.** It flagged two reporting flaws (signed top route; Σ |mean|), now fixed in v2.
- **Codex's next step:** split E into residual / query / self-K/V and ops into K / V on the same path.
- **Claude's next step:** a 2× diagonal-weight training arm.
