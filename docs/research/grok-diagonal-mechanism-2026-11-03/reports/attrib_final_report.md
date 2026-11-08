# Route attribution (H10): final report, 2026-11-03

> **Corrections (2026-11-03 consolidation):** the study-status table (exposure, layer-1 attention, size) and the "wrong content at layer 0" interpretation overstate. Marked below. Source: `grok_runs\consolidation_2026-11-03\CONSOLIDATED_REPORT.md`.

> **Scope (added 2026-11-03):** these route shares were measured on models trained at lr 6e-4 (the E-series base), at 25k updates. At lr 1e-3 the mix is different. These are long-stack runs that differ from the 6e-4 control models only in lr, scored at 1e5 updates:
> - **The operand route carries 103–131 % of the diagonal deficit, and the '=' route is negative (−3 to −31 %),** in 3/3 seeds, under path choice A. Under choice B the ranges are 116–174 % and −16 to −74 %, and s1 is rated coupled; both falsifiers fire 3/3 under either (corrected 2026-11-08, misread audit). The registered prediction (K2b) is falsified; Codex agreed blind.
> - **In 2 of those 3 seeds, the layer-1 MLP had decayed to zero.** lr and lr × wd are confounded in this comparison.
> - **Even at 6e-4, by 1e5 updates** the '=' share is 48–70 %.
> - **The routes cancel heavily from pair to pair,** so these shares are path averages, not a fixed circuit.
> - **The lr difference doesn't depend on the method** (per-pair replay, RESULTS.md R; Codex agreed blind). An exact Shapley split, which uses no path, gives '=' shares of −82 % to −134 % at 1e-3 against 39–61 % at 6e-4. So '=' takes much more at 6e-4, but under Shapley it isn't the majority in 2/3 seeds.
>   - The point shares do depend on the method: about 10 points at 6e-4, and 50–117 at 1e-3.
>   - Per-pair signs often disagree between IG and Shapley.
>   - At 6e-4, 2–8 % of pairs need more than 32 integration nodes; the means are unaffected.
>   - The attribution has no gradient bug.
>
> Details: `grok_runs\checks_2026-11-03\RESULTS.md`, K2.

**Bottom line.**
- **The diagonal deficit splits between two routes into layer 1.** About two thirds goes through the '=' stream after layer 0 (60–85 % in 8 of 9 runs). About one third goes through layer 1's reads of the operand positions. The op position carries none.
- **This is mixed by the registration:** 7 runs "unresolved", 2 "'=' account". The registered categories had no label for a stable two-route split, which is what 7 runs show.

## Validity
- The tail function reproduces the model's margins within 3.5e-5.
- The routes sum to the measured gap within 0.028 nats (32 nodes) and 0.006 (64 nodes).
- Route means move ≤ 0.022 nats between 32 and 64 nodes, and ≤ 0.36 nats between the two interpolation choices.
- Blind Codex found no bug and agrees on all verdicts. Its two reporting fixes were applied and the test rerun (v2).

## Results (diagonal recipients, shares of the gap, choices A / B)
| run | gap (nats) | '=' stream | operand reads | op position |
|---|---|---|---|---|
| add_s0 | 4.05 | 70 / 71 % | 31 / 28 % | 0 % |
| add_s1 | 5.33 | 71 / 66 % | 28 / 34 % | 1 / 0 % |
| add_s2 | 6.24 | 62 / 66 % | 38 / 34 % | 0 % |
| mul_s0 | 3.83 | 43 / 41 % | 57 / 59 % | 0 % |
| mul_s1 | 6.19 | 68 / 69 % | 32 / 31 % | 0 % |
| mul_s2 | 4.04 | 60 / 64 % | 40 / 36 % | 0 % |
| control_s0 | 4.15 | 62 / 66 % | 37 / 34 % | 0 % |
| control_s1 | 4.73 | 85 / 84 % | 15 / 16 % | 0 % |
| control_s2 | 4.42 | 83 / 82 % | 15 / 18 % | 1 / 0 % |

**Separate-token runs:** gaps −0.39 to +0.59, with no consistent route.

## Caveats
- **Per pair the routes are large and noisy, on diagonals and healthy controls alike** (5–8 nats of route movement). The split above is an average. The controls average to about 0; the diagonals average to a consistent positive split.
- **Integrated gradients depend on the path.** The shares mean "along a straight line to same-answer off-diagonal donors", not an intrinsic property of the network.
- **mul_s0 runs mostly through the operand reads,** so "'=' stream dominant" is not universal.

## What the study now says (H1–H10)
| question | answer |
|---|---|
| Exposure? | only in the first ~200 updates (H5) |
| Layer-1 attention allocation? | not the limiter (H6c) |
| Final-norm scale? | 15–30 % of the gap (H6) |
| Readout-invisible energy? | ≤ 7 % (H7) |
| Size of the doubled layer-0 write? | no (H8) |
| Which route carries it? | mostly the '=' stream after layer 0, partly layer 1's operand reads, never the op position (H10) |
| Can it be localized by swapping parts? | no, the parts are tightly coupled (H9) |

**Interpretation (inferred):** what (a, a) writes into '=' at layer 0 is the wrong content for the answer, not the wrong size, and the operand positions add a smaller share.

> **Correction (2026-11-03 consolidation):**
> - **Table:** exposure explains most of the gap at update 100 and 32–44 % at 200, fading by about 500 with a 5–15 % residual (CPU fp32 reruns, which act as new seeds). Layer-1 attention is registered MIXED: no diagonal-specific effect at final weights, not "not the limiter". Size (H8) is a minor factor at final weights, not "no".
> - **Interpretation (inferred, path-averaged):** averaged along one integration path, about 2/3 of the deficit flows through the '=' state after layer 0. That state includes the layer-1 query and the '=' self K/V, so "layer-0 content" is not localized. It is not a per-pair mechanism, and "not the wrong size" overstates H8. Finer attribution is the test that would localize it.
> - All 9 shared-token models are at lr 6e-4, `repo`, dropout 0.1.

## Next options (your call)
1. **Finer attribution (Codex), no training, about a minute of GPU.** Split the '=' route into residual / query / self K-V, and the operand route into K / V. This tells "layer 1 reads the diagonal wrongly" apart from "layer 0 carries the wrong content".
2. **2× diagonal-weight training arm (Claude), about 15 minutes of GPU for 3 seeds.** It tests whether the deficit is a learned trade-off, which more weight on diagonals would shrink, or a structural limit, which it wouldn't.

**Files:**
- `attrib_results_v2.md`
- `claude_attrib_analysis.md`
- `codex_attrib_analysis.md`
- `hypotheses.md` (H10)
