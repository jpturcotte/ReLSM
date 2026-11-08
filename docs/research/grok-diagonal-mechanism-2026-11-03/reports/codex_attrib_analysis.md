The result is a distributed deficit with a usually larger E contribution. It does not establish a single faulty site, and “unresolved” understates how consistent the measured split is.

Static inspection found no substantive forward or decomposition bug:

- Layer 0 reproduces attention, residual addition and the feedforward block.
- E supplies the layer-1 residual, query and '=' self K/V, with gradients through all those uses.
- RoPE applies to Q/K at the correct positions, including '='; V remains unrotated. The last query can causally attend to all four keys.
- A interpolates prefix residuals before RMSNorm. B interpolates normalized prefix inputs, equivalently their K/V, including rotated K because rotation at fixed positions is linear. E remains a residual interpolation in both.
- Donors share the answer; off-diagonal controls exclude self and mirror pairs.

Two reporting problems matter. `argmax` selects the largest **signed** attribution, even if all routes are negative, so “largest positive route” is not guaranteed. Also, “control-path damage” is Σ|route **means**|, not mean Σ|per-pair routes|. It can hide substantial cancellation across pairs.

The checks support aggregate accounting. Endpoint errors are at most 3.46×10⁻⁵ nats, well below 10⁻³. Every completeness mean passes 0.05; the worst is 0.02828. But individual errors reach 2.471 nats. Moreover, completeness constrains the sum, not each route’s numerical accuracy, so quadrature refinement would strengthen localization claims.

Separate-token completeness errors are **rounded to 0.000**, not exactly zero. Raw means are 2.89×10⁻⁶ to 3.77×10⁻⁶ nats; maxima are 7.22×10⁻⁶ to 1.39×10⁻⁵. This is compatible with easier numerical integration along those paths, not proof of linearity.

The registered verdicts are correct. Shares below are route means divided by mean total:

| Shared-token run | Gap, nats | A: E / ops | B: E / ops | Registered label |
|---|---:|---:|---:|---|
| add_s0 | 4.05 | 69.5 / 30.7% | 71.4 / 28.5% | Unresolved |
| add_s1 | 5.33 | 71.1 / 28.4% | 66.5 / 33.9% | Unresolved |
| add_s2 | 6.24 | 62.3 / 37.6% | 65.8 / 34.1% | Unresolved |
| mul_s0 | 3.83 | 42.6 / 57.4% | 41.4 / 58.5% | Unresolved |
| mul_s1 | 6.19 | 68.1 / 31.8% | 69.2 / 30.7% | Unresolved |
| mul_s2 | 4.04 | 60.0 / 40.3% | 63.7 / 36.1% | Unresolved |
| control_s0 | 4.15 | 62.5 / 36.9% | 66.0 / 33.7% | Unresolved |
| control_s1 | 4.73 | 84.8 / 15.5% | 83.7 / 16.2% | '=' account |
| control_s2 | 4.42 | 83.3 / 15.4% | 82.5 / 17.7% | '=' account |

Thus seven unresolved, two '=' accounts, overall mixed. Neither cancellation nor interpolation disagreement triggers: cancellation ratios stay near one, and the largest A/B route difference is 0.2904 nats.

control_s2 is the closest qualifying label: B’s ops share is only 2.25 percentage points below 20%. control_s1 also clears its thresholds modestly. Neither is numerically on the boundary, but sampling uncertainty is unreported. The unresolved runs are comfortably outside the 80/20 criterion.

“Unresolved (none)” is mechanically fair for 65/35, but descriptively poor. “Distributed, E-majority attribution” captures the positive finding. It does **not** establish nonlinear coupling.

Across eight of nine runs, E contributes more than ops; both contribute positively and substantially. A/B agreement makes this robust to these two normalization-path choices. Nevertheless, E is the per-pair argmax only about 53–69% of the time in those runs: mean dominance is not universal dominance.

mul_s0 is a reproducible exception: ops contributes 2.20–2.24 nats versus E’s 1.59–1.63, with ops winning 51–55% of pairs. This defeats a universal E-dominant account.

The op-position route contributes less than 0.71% in magnitude to every shared-token mean. That excludes a substantial **signed average** contribution here, not op processing elsewhere or cancelling individual contributions.

Off-diagonal mean gaps span roughly −0.58 to +0.65 nats, versus diagonal gaps of 3.83–6.24. They support a diagonal-associated directional signal. They do not establish harmless control paths, given the averaging problem.

Separate-token gaps are −0.39, +0.59 and −0.27 nats. There is no consistent deficit, although routes can cancel: s1 has E +1.60–1.63 and ops −1.01–1.04.

Together with the earlier findings, simple amplitude excess is inadequate. Representational direction or compatibility remains plausible. E attribution includes downstream normalization effects, so its share cannot be added to the earlier 15–30% scale estimate. Failed transplants show hybrid incompatibility; IG allocates endpoint differences without proving isolated repairability. Both straight-line paths may traverse unnatural states, and their agreement does not establish path independence or where the defect originally arose.

The most informative next step is finer same-path edge attribution: split E into residual, query and self K/V, and ops into K versus V. Query/read compatibility predicts E-query plus operand-key contributions; deficient carried content predicts residual and value contributions. mul_s0 should reveal whether its exception shifts the same mechanism toward operand edges or uses a different mechanism.
