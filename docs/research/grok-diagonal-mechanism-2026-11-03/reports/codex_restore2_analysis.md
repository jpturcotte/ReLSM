**1. Validity**

The checks support trusting the reported effects. All 756 numeric table entries checked against the raw JSON agree within rounding. Manual-forward logits match `model(x)` within \(3.1\times10^{-5}\); both self-target interventions produce exactly zero logit change.

The largest attention row-sum error is \(2.1\times10^{-4}\). That warrants qualification, but the direct mass-rescaling hook can amplify floating-point error through division by \(1-m\). Score offsets instead pass through softmax. The pooled check does not identify which intervention produced the maximum.

Static inspection confirms that offsets preserve the operand ratio, selection and chunk slicing are consistent, and fixed norm uses each example’s baseline inverse RMS. No table/code inconsistency emerged. T4’s intervention columns contain **resulting gaps**, except the explicitly labelled flattening delta.

These checks validate implementation consistency, not an explanation of the deficit. I did not inspect imported helpers or independently reproduce checkpoint outputs.

**2. H6c verdicts at the registered thresholds**

**Attention: mixed.** All-heads, \(\delta=+1\), norm fixed:

| Run | Δ diagonal | Δ off-diagonal |
|---|---:|---:|
| add_s0 | +0.146 | +0.145 |
| add_s1 | +0.389 | +0.390 |
| add_s2 | +0.381 | +0.371 |
| mul_s0 | −0.012 | +0.128 |
| mul_s1 | +0.256 | +0.249 |
| mul_s2 | +0.136 | +0.243 |
| control_s0 | −0.058 | +0.189 |
| control_s1 | +0.240 | +0.322 |
| control_s2 | +0.341 | +0.320 |

Only **4/9** exceed +0.25, and diagonals improve more than off-diagonals in only **4/9**. The limiting criterion requires at least 7/9 for each. Only **2/9** diagonal effects are ≤+0.1, so the registered “not limiting” criterion also fails.

**Norm flattening:** all nine shared-token runs pass the <0.2 threshold. Their effects range from **+0.008 to +0.053**, versus **+0.669 to +1.510** for the diagonal-to-off-diagonal scale swap. Thus the swap’s benefit principally reflects the scale difference, rather than flattening.

Among separate-token runs, flattening effects are **+0.296, +0.103, +0.126**. The first fails the threshold, so the flattening conclusion does not hold universally across all twelve runs.

**Accuracy invariance: passes.** Raw accuracies match exactly under every norm-only intervention in all twelve runs, including all recorded train/test groups.

**3. What the offset curves show**

With **norm free**, increasing operand scores hurts shared-token diagonal margins in **9/9** runs: at +1, changes span **−0.74 to −0.05**. Off-diagonals also worsen in 9/9, by **−0.27 to −0.06**. Negative offsets sometimes help, but several curves peak around −0.5 or zero. These are locally curved tradeoffs, not globally monotonic relationships.

With **norm fixed**, positive offsets mostly help. Every shared-token diagonal curve is positive at +0.5, but mul_s0 and control_s0 turn slightly negative by +1. Negative offsets hurt all nine. The asymmetry indicates diminishing returns and, in those two runs, an optimum within the tested range.

There is little diagonal-specific rescue. At +1, diagonal operand mass increases by **0.134–0.186**, versus **0.088–0.125** off-diagonal, yet margin gains are usually comparable or smaller. Even the largest diagonal gain closes only **7.8%** of its baseline gap.

Per-head results reinforce this. Fixed-norm diagonal effects range from **−0.097 to +0.163**; no individual head hides a large rescue. All-head effects differ from summed individual-head effects by at most **0.029**, suggesting little interaction at this offset. With norm free, **31/36** diagonal and **35/36** off-diagonal head interventions are negative.

Separate-token runs show stronger fixed-norm gains despite their small baseline gaps: diagonal **+1.72, +0.45, +1.08**, and off-diagonal **+1.05, +0.81, +1.16**. Free-norm effects shrink to **+0.16, +0.02, −0.05** diagonally. Thus normalization suppressing attention-induced margin gains is a general phenomenon, not sufficient evidence for the shared-token diagonal deficit.

**4. My account**

The evidence supports two distinct contributions.

First, operand attention supplies some useful readout signal. Holding final scale fixed usually reveals a positive effect. But these interventions do not establish insufficient attention as the main diagonal bottleneck: gains are small, heterogeneous, and generally not selective for diagonals.

Second, final normalization materially affects the measured margin deficit. Replacing diagonal scale with off-diagonal scale closes **15.5–30.3%** of the shared-token gap while preserving predictions. Most of the gap remains. This demonstrates scale sensitivity of confidence, without identifying why diagonal residuals have greater RMS.

The free-minus-fixed effect at +1 is **−0.44 to −0.68** diagonally. Allowing normalization to respond therefore more than cancels the small fixed-scale benefit in every shared-token run.

My working hypothesis is that shared operands produce redundant residual content: additional operand weighting increases residual magnitude more than useful class separation. Lower operand attention could partly compensate for that geometry. **Redundancy and compensation are inferences**, not directly measured here.

The earlier “raising mass lowers diagonal margins” result is consequently largely explained by normalization, but not completely. In T4, comparing both interventions with the norm swap alone gives positive effects in **7/9** runs, approximately zero in mul_s0, but **−0.587** in control_s0. The exact group-mean intervention can therefore damage useful representation even at a common fixed scale. It also differs from a uniform score offset.

**5. Single most informative next step**

Decompose the final pre-norm residual into components visible and invisible to the **centered answer-logit readout**, including learned norm weights. Remove the invisible component and recompute RMSNorm, for baseline and offset examples. This preserves pre-norm relative answer logits while testing whether answer-irrelevant energy causes the normalization penalty.

Under my redundancy account, diagonals should contain more surplus invisible energy; removing it should reduce their scale disadvantage and substantially weaken the negative free-norm offset response.

The main alternative is weaker diagonal discrimination plus normalization of largely answer-visible signal. It predicts little preferential diagonal rescue, with the gap and negative offset response largely persisting.
