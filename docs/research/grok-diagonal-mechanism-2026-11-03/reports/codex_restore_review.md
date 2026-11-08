**The norm oddities are expected design artefacts. The attention result is plausibly real for this intervention. I found two latent API bugs, neither triggered by these runs.** Static review and JSON arithmetic only; no checkpoints were loaded.

All line references below refer to [restore_test.py](grok_runs/mechanism_2026-11-03/restore_test.py).

1. **Implementation review**

- **Mass hook, lines 79–89:** Correct for these inputs. Operands occupy positions 0 and 2, including with separate tokens. `[H]` targets broadcast across selected examples; `[B,H]` targets are sliced and selected correctly. Multiplying operand weights by `M/m` and remaining weights by `(1-M)/(1-m)` preserves both within-group ratios and gives total mass `M+(1-M)=1`. The final query can attend to all four positions, so the causal mask introduces no complication. Earlier query rows remain unchanged.
- **Numerical robustness, line 88:** There is no protection against `m=0` or `m=1`, which could produce division by zero and NaNs. This is a potential failure, with no evidence that it affected these finite results. Assert positive support or define boundary behavior explicitly.
- **Output projection, line 91:** Correct. Transposing to `[B,T,H,dh]` and flattening concatenates heads in the same order as the probe’s weight slices. Applying `o_proj` equals summing the per-head projections, including its bias once. Floating-point accumulation can differ slightly.
- **Norm, lines 94–100:** Correct for this model. The calculation uses fp32 RMS, the actual `model.norm.eps`, casts the normalized activation back before multiplying the learned gain, and calls the actual `lm_head`. This matches [RMSNorm](model.py:212). Bypassing `model.norm(...)` is intentional here. It does assume RMSNorm rather than Identity.
- **Actual latent bugs:** `run_all` ignores its `chunk` argument at line 105 and always uses `mp.CHUNK`. Also, advertised per-example inverse-RMS targets are broken: line 117 neither slices them by chunk, nor does line 99 select target entries with `sel`. Fix both: slice nonscalar norm targets by `s`, then assign `tgt[sel]`. Current norm targets are scalars, so neither bug explains these results.
- **Targets and groups, lines 131–149:** Targets use only the unmodified training groups. Mass means average examples separately for each head; norm means average examples. There is no held-out leakage. The probe’s `off_test` excludes held-out pairs whose transpose is in training, so `off` does not include every off-diagonal example. That does not affect the registered training comparison.
- **Margins, lines 120–124:** Correct probe function, class columns, labels, and gap sign. Competitors include the numeric answers plus `op` and `=`, matching training. Empty groups would produce NaN rather than the probe’s `None`, but current groups are populated.
- **Baseline verification:** All 12 saved base gaps match v2 within **0.000003**, comfortably below 0.02. Every reported unselected group is unchanged. This strongly supports implementation consistency, although matching aggregate gaps alone is not an examplewise equivalence check.

2. **Raising operand mass: plausibly real, but narrower than “not causal”**

Seven of nine shared-token runs lose diagonal margin, ranging from **−0.075 to −1.306**. The other two gain **+0.131** and **+0.051**. Thus all nine satisfy the registered “less than 10% closed” criterion.

The intervention tests whether replacing each head’s mass with an off-diagonal mean rescues diagonals. It simultaneously removes attention from `op` and `=`, changes the residual, and changes the downstream MLP and final normalization. Layer-1 operand values also differ on diagonals. More operand weight need not mean more useful evidence.

Lower mass could therefore be compensatory. Group-mean replacement additionally removes example-specific variation, and changing all heads together can hide opposing effects.

**The existing combined intervention supplies useful evidence:** comparing `both_up_diag` with `norm_up_diag` holds the final inverse RMS identical. Under that comparison, adding the mass intervention **improves diagonals in seven of nine runs**, by **0.078–0.361**. Mul s0 is effectively unchanged, **−0.0005**; control s0 remains worse, **−0.587**.

So much of the mass-only harm depends on final normalization. The broad “NOT CAUSAL” wording overstates the result. The registered rescue failed; operand mass still has effects and interactions.

3. **Separate-token norm control: design artefact**

A near-zero baseline gap does not imply equal sensitivity to an intervention.

For sepops s0, diagonal/off-diagonal inverse RMS is **3.4060/3.1329**. “Norm up” actually lowers diagonal scale, producing **−1.093** margin change. “Norm down” raises off-diagonal scale, producing **+1.721**. The gap moves from **−0.249 to +1.472**.

Sepops s2 also has reversed scales and an off-diagonal gain of **+0.418**. S1 has the original ordering and a diagonal gain of **+0.313**.

The registered 0.3 sanity bound failed, but it was not a valid null prediction. Equal margins can coexist with different scales and compensating representations.

4. **Norm asymmetry: expected scaling, not evidence of a bug**

With the residual fixed and this bias-free head,

\[
z_i^{new}=(r_{\mathrm{target}}/r_i)z_i^{base}.
\]

Top margins scale exactly. LSE margins follow

\[
m_i(s)=s z_{iy}-\log\sum_{j\ne y}e^{s z_{ij}},
\]

so they are not exactly proportional.

Off-diagonal margins start larger. For mul s0, diagonal/off-diagonal margins are **11.095/15.655**; norm replacement changes them by **+0.930/−1.109**. Larger absolute off-diagonal changes are unsurprising.

Replacing individual scales also removes within-group variation and its correlation with logits. Reciprocal percentage changes differ too. Aggregate JSON cannot separate these contributions exactly.

The norm test measures sensitivity to an imposed logit scale. It does not independently establish why diagonal representations are weaker. In ordinary RMSNorm, inverse RMS compensates residual magnitude; it is not a standalone confidence variable.

5. **Minimal fixes and reruns**

- Fix chunk handling and vector norm targets. Add guarded checks for row sums, target mass, unchanged rows, self-target no-ops, and examplewise baseline agreement. These API fixes alone do not require repeating the existing scalar-target experiment.
- **Rerun attention tests** with small positive/negative operand-score offsets, one head at a time. This preserves ratios and gives controlled mass changes without group-mean flattening. Positive local slopes support insufficient mass; flat/negative slopes weaken that account. Repeat with each example’s **baseline inverse RMS held fixed** to distinguish normalization effects.
- **Replace the norm sanity test** with per-example self-target no-ops and exact logit-rescaling checks. Add own-group-mean replacements to measure the flattening effect.
- Save per-example logits and inverse RMS during the next memory-guarded pass. Subsequent norm sweeps need no transformer reruns. Report accuracy and scale-invariant margins: pure norm replacement should preserve both, apart from numerical ties.
