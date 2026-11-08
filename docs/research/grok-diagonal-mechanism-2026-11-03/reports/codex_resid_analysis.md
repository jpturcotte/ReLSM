1. **Validity.** The manual forward matches the stated block structure and RMSNorm formula. Component sums agree exactly; stripping changes centered pre-norm logits by only approximately \(1.7–5\times10^{-4}\). These support internal consistency, but the component-sum check does not establish equivalence to the model’s native forward.

There is one unresolved inconsistency: **a centered 97-class readout has rank at most 96, whereas every run reports rank 98.** The allowed code obtains `cols` from an unread helper. Extra classes or mismatched experiment metadata need checking. H7b’s answer-only interpretation is therefore provisional.

The projection itself is correct: multiply readout weights by the norm gain, center across classes, construct the SVD row-space projector, project the residual, then recompute RMSNorm. “Invisible” means invisible to the selected centered logits before normalization.

The \(c_i\cdot h\) shares are sound **signed accounting**, not causal attribution:

\[
c_i\cdot h=\|c_i\|^2+\sum_{j\ne i}c_i\cdot c_j.
\]

They allocate each pairwise interaction equally between its two components. Negative shares and shares above 100% are legitimate. They do not measure what ablating a component would do, because later writes depend on earlier ones.

2. **Registered verdicts, using unrounded JSON.** H7b verdicts below describe the implemented projection, subject to the class-set issue.

| Prediction | Result | Verdict |
|---|---|---|
| H7a(i) | Shared attn0 ratios 1.409–1.435, all ≥1.25; separate 1.080, 0.929, 0.978 | Pass |
| H7a(ii) | attn0+mlp0 shares: 116.8%, 44.6%, 61.3%, 127.0%, 64.6%, 78.2%, 161.1%, 96.5%, 42.3% | Pass, exactly 7/9 |
| H7b(i) | Diagonal invisible fraction exceeds off-diagonal by 4.28–7.62 percentage points | Pass, 9/9 |
| H7b(ii) | Scale disadvantage halves in add_s0, control_s0, control_s1 | Fail, 3/9 |
| H7b(iii) | No qualifying run achieves 50% attenuation | Fail, 0/8; seven required |

H7a(ii) sits exactly at the required run count, although individual shares are not especially close to 50%.

H7b(ii) has two borderline cases: add_s0 retains **48.25%** of its original scale disadvantage and passes; mul_s0 retains **51.26%** and fails. Even granting the latter would leave only 4/9.

For H7b(iii), control_s2 barely qualifies: baseline response −0.10289, stripped −0.06824, a **33.68%** attenuation. Excluding it still gives 0/7. Add_s1 improves 53.98%, but its baseline −0.04720 does not qualify.

3. **What the components show.** Attn0’s approximately 1.42 ratio is strikingly close to \(\sqrt2\). Equal-weight addition of identical vectors has twice the squared norm of adding two equal-energy, uncorrelated vectors. Given identical layer-0 values for repeated tokens, this strongly supports coherent addition as the source of the inflated attention write. Nonoperand contributions and learned correlations prevent treating \(\sqrt2\) as an exact prediction.

Mlp0 moves oppositely: RMS ratios are **0.735–0.882**, and its shares are **−27% to −88%**. Thus the larger attention write is accompanied by a smaller MLP write. Negative shares establish reduced contribution to the final norm excess, not active cancellation.

Attn1 contributes **28–59%** of final excess energy despite RMS ratios as low as **1.018**. Alignment with other components matters substantially. This rules out treating the final excess as an isolated attn0 phenomenon.

Separate-token runs lack the coherent signature: attn0 ratios stay near one, mlp0 ratios are **0.953–0.982**, and diagonal invisible fractions are only **2.1–3.0%**, versus **13.3–22.9%** shared. Their total norm ratios are **0.919, 1.009, 0.983**. Two have negative squared-norm differences; the remaining difference is small. Their “excess shares,” especially attn1’s 176%, therefore require denominator and sign caution.

4. **My account.** The experiment establishes coherent amplification early, substantial downstream interactions, and excess invisible energy. It does **not** establish that the early amplification causes the margin deficit.

Conditional on the projection audit, invisible energy is a minor contributor to the train gap: stripping closes **−0.63% to 7.20%**, leaving gaps of **3.53–6.16**. Control_s0 is particularly informative: its scale disadvantage falls from **5.01% to 0.38%**, yet its margin gap remains **3.77**, versus 4.06 initially. Null-space dilution cannot explain most of this deficit.

My working hypothesis is that coherent operand amplification changes the balance of features entering subsequent normalized nonlinear computations, producing poorer answer discrimination. Mlp0’s consistently smaller write is a clue, not proof. The remaining problem can reside within the visible subspace: visibility says a direction changes some class contrasts, not that it favors the correct answer. Neither stronger correct evidence nor fewer strong competitors follows from greater visible energy.

5. **Next experiment.** Run a causal gain sweep on the **operand-only contribution to attn0 at '='**, retaining its other contributions. Use gains **0.70, 0.85, 1.00, 1.15**, then replay downstream computation without training.

Compare this with the identical additive change applied only at the final residual while retaining cached downstream writes. Measure margins both normally and with final inverse RMS fixed to baseline; include off-diagonal controls.

Under my hypothesis, reducing diagonal gain toward **\(1/\sqrt2\approx0.707\)** should improve discrimination through changed downstream writes, exceeding the final-only intervention even with normalization fixed. Under the main surviving alternative, later discrimination is deficient independently of early amplification: early gain reduction should offer little selective rescue or worsen performance. This directly tests the causal bridge that the present norm accounting leaves unproven.
