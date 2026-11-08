# Route attribution (H10): final report, 2026-11-03

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

## Next options (your call)
1. **Finer attribution (Codex), no training, about a minute of GPU.** Split the '=' route into residual / query / self K-V, and the operand route into K / V. This tells "layer 1 reads the diagonal wrongly" apart from "layer 0 carries the wrong content".
2. **2× diagonal-weight training arm (Claude), about 15 minutes of GPU for 3 seeds.** It tests whether the deficit is a learned trade-off, which more weight on diagonals would shrink, or a structural limit, which it wouldn't.

**Files:**
- `attrib_results_v2.md`
- `claude_attrib_analysis.md`
- `codex_attrib_analysis.md`
- `hypotheses.md` (H10)
