The measurements are credible, and the registered causal explanations fail. My best working account is a special representation produced by shared tokens and symmetric operand reads, with later attention amplifying an earlier deficit. The exact mechanism remains unproven.

I read only the permitted files.

**Measurement validity**

The numerical checks are strong:

- Across 102 valid checkpoint records, maximum manual-forward error is \(3.25\times10^{-5}\), and maximum margin-split reconstruction error is \(4.01\times10^{-5}\).
- Logged versus probe train-gap disagreement is at most 0.027 across checkpoints, negligible beside add/mul gaps of roughly 2–6.
- Every count check passes. Diagonal totals correctly reflect the domains: 97 for add/sub, 96 for mul/div, and 120 for S5.

The count check compares total training count and train/test diagonal counts, not exact membership or off-test counts. Nevertheless, reconstruction uses the training split function, and agreement with logged margins provides additional validation.

The interventions match their descriptions. Ratio interventions alter only operand weights at '=' in the selected layer, preserve operand mass, and recompute downstream computation. Shared-token L0 diagonals change by at most approximately \(2\times10^{-5}\). Whole-layer freezing replaces attention probabilities at **every query position**, using means over all inputs. Head-mean interventions replace projected head outputs at '='. These are substantially different interventions.

Class removal masks only wrong operand classes, preserving the correct answer. Random removal matches the number removed per example and samples other wrong numeric classes. It uses one deterministic random draw, sufficient here because the observed effects are tiny. In separate-token runs, “operand b class” means its numeric answer class, not the separate input token.

“Runner-up” means the strongest wrong class, including on misclassified examples. Its operand frequency is conditional on having a wrong operand competitor, rather than covering every diagonal. Excluding the few operand-equals-answer cases cannot turn frequencies near 0–2% into 50%.

The split is exact for **top margin**, with each example’s actual final RMSNorm scale and competitor held fixed. It is neither an LSE decomposition nor a causal attribution.

Separate tokens are a material confound: they add input parameters, and second-operand embeddings are excluded from output competition. Thus they change tied-embedding training pressures as well as repeated-token identity.

T7’s late averages reproduce using updates 22,500–25,000 inclusive. They must not be compared directly with final-checkpoint interventions. For example, add s0 is 4.46 averaged versus 3.70 final; sub s2 is +0.62 averaged versus −0.57 final.

A minor packaging defect: two raw JSONL files contain seven non-JSON status lines. Filtering those leaves all 102 records.

**Threshold verdicts**

Numbers separated by slashes are seeds 0/1/2.

| Prediction | Verdict and deciding numbers |
|---|---|
| H1: ≥2× token dependence | **FAIL**, using the revision’s L0 scope. Maximum-head SD: add .086/.070/.071; mul .118/.086/.089; sub .057/.049/.060; div .039/.057/.071; shared a+2b .034/.056/.037. Only mul s0 exceeds twice all three matched controls. |
| H1: whole-layer freezing shrinks gap >50% | **FAIL.** L0 freezing produces approximately −2/−1/0% shrink for add and 0/−1/−1% for mul. Even allowing L1 gives add 37/52/34%, mul 32/31/27%; only add s1 passes. |
| H1b: L0 ratio equalization shrinks ≥50% | **FAIL in every seed.** Add −2/−1/−1%; mul approximately 0/−1/−1%. Off-diagonal margins instead rise by .06/.06/.03 and .02/.08/.03. |
| H2: one component carries ≥60% | **FAIL overall**, descriptively. Add: **PASS 67.4%** attention0; **PASS 82.9%** MLP0; **FAIL 56.7%** MLP0. Mul: **FAIL 54.4%** attention1; **PASS 83.6%** MLP0; **PASS 68.3%** MLP0. |
| H3: gap <1 at 500, growing only after generalization starts | **FAIL in every seed on both conditions.** Logged gaps at 500: add 2.08/2.53/2.25, mul 2.51/2.29/1.99, with train accuracy ≥.999. Every run first exceeds gap 1 at update 100; generalization starts at add 300/300/500, mul 400/300/300. |
| H4: operand removal ≥50%, random removal <15% | **FAIL.** Operand removal slightly enlarges every add/mul gap, with shrink between −1.10% and −0.015%. Random removal independently passes its control threshold, ranging from −1.06% to +0.48%. |
| H4: diagonal operand runner-up ≥50%; sub/div ≤20% | **FAIL.** Add 0/0/0%; mul 2.0/2.1/2.0%. Sub 0/0/0%; div 0/0/4.1%, satisfying the control bound. |

H2’s passing percentages can coexist: add s0 attention0 and attention1 contribute about 67% and 61%, while MLP1 cancels −3.02 margin units.

**What the data shows, and my interpretation**

The strongest controlled result is the interaction between shared tokens and commutative computation. Ordered-split add controls retain gaps 4.06/4.64/4.72, so paired splitting and the several-task code path are unnecessary. Separate-token add gives −.25/.59/.34. Shared a+2b gives .16/.96/.99, versus .19/−.30/−.15 with separate tokens.

Shared tokens alone therefore do not produce the large deficit. Nor does token-dependent L0 attention: its ratio can be erased almost harmlessly in add/mul.

My inference is that nearly symmetric reads create a distinctive equal-operand representation. Downstream nonlinear computation learns weaker answer separation for that representation. The diagonal answer remains computable from a alone; these results do not demonstrate information loss.

The positional-read controls support this account. Shared a+2b uses strongly asymmetric L0 ratios, .258/.258/.012. Equalizing them lowers off-diagonal margins by 8.17/6.92/10.83 while leaving diagonals unchanged. S5 s1 likewise uses ratio .008 and loses 11.61 off-diagonal margin units. Small token-dependent variation can coexist with essential **constant positional asymmetry**.

Later routing contributes, but cannot explain onset. Add/mul L1 ratio equalization lowers off-diagonal margins by 1.61–2.29, barely affecting diagonals, accounting for 32–49% of final gaps. L1 operand mass is also consistently lower on diagonals. Yet at update 500 the gap is already substantial and L1 ratio shrink is approximately zero. At update 100, gaps are 1.14–1.29 while off-diagonal test accuracy is only 0–.2%.

The component accounting changes accordingly: early deficits sit largely in MLP1; late MLP1 contributions oppose the deficit, while MLP0 and attention1 carry positive contributions. This argues against one stationary deficient component.

Normalization may contribute late: raw final inverse RMS is 6–12% lower on train diagonals in all six add/mul runs. At 500 the relationship reverses, excluding that scale difference as a unified explanation.

Still open are training dynamics, effective exposure of symmetric input pairs, representation geometry, and distributed tied-embedding effects. Removing operand competitors rules out their **direct readout competition**, not their influence during training.

**Surprises and inconsistencies**

Noncommutative gaps are substantial and unstable: sub ends at 2.40/2.17/−.57; div at 2.34/.85/2.26. Their early negative gaps are plausibly helped by constant diagonal answers, 0 for subtraction and 1 for division, unlike addition.

Separate-token controls do not have zero deficit in every component: add s1 MLP0 contributes +1.74, partly cancelled by attention1 at −.80. Their small aggregate gaps conceal internal differences.

Enormous percentage shrinks often represent damage: div L1 equalization produces gaps −6.44/−1.74/−7.92 by disproportionately lowering off-diagonal margins. Add/mul seed spread is also real: final train gaps span 3.70–6.17, and mul s2 diagonal test accuracy is only .848.

**Next experiment**

Run a matched factorial retraining experiment on shared-token add/mul: force L0 operand ratios at '=' to .5 versus .25, preserving mass, crossed with diagonal loss weights 1 versus 2. Keep parameter count, output classes, splits and seeds fixed; compare trajectories at matched off-diagonal accuracy.

Under my symmetric-read account, asymmetric routing should reduce the gap even with unchanged exposure. Under the main alternative, insufficient effective diagonal training exposure, doubling diagonal weight should dominate, with little benefit from routing asymmetry itself.

A backup is diagonal oversampling across a wider dose range. Persistent gaps despite matched diagonal training performance favor a representation effect; systematic closure with exposure favors the training-exposure account.
