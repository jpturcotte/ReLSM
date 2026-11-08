# Claude's analysis of route attribution (H10)

Written 2026-11-03, before reading `codex_attrib_analysis.md`. Source: `attrib_results.md`.

## Validity
- **Endpoints:** F(recipient) and F(donor) match the model's own margins within 3.5e-5 in all 12 runs, so the tail function is the model.
- **Completeness:** the routes sum to the measured change within 0.028 nats (worst run mean), so the check passes.
  - Per-pair maxima reach 1.9–2.5 nats in a few shared-token pairs (control_s1/s2, choice A). Some paths are steep enough that 32 nodes under-resolve them. Run means are unaffected.
- **Separate-token completeness is exactly 0.000.** Their layer-1 MLP is dead (output 0, found in H8). That leaves a smoother function, which 32-point Gauss-Legendre integrates almost exactly. This is my inference, not checked.
- **Controls show the method doesn't create route effects.** Healthy off-diagonal pairs get small totals (−0.58 to +0.65) with Σ |route| ≤ 0.65, unlike the 3–7 nat damage from H9's transplants.

## Registered verdicts
| run | E share (A / B) | ops share (A / B) | label |
|---|---|---|---|
| add_s0 | 70 / 71 % | 31 / 28 % | unresolved (none) |
| add_s1 | 71 / 66 % | 28 / 34 % | unresolved |
| add_s2 | 62 / 66 % | 38 / 34 % | unresolved |
| mul_s0 | 43 / 41 % | 57 / 59 % | unresolved |
| mul_s1 | 68 / 69 % | 32 / 31 % | unresolved |
| mul_s2 | 60 / 64 % | 40 / 36 % | unresolved |
| control_s0 | 62 / 66 % | 37 / 34 % | unresolved |
| control_s1 | 85 / 84 % | 15 / 16 % | '=' account |
| control_s2 | 83 / 82 % | 15 / 18 % | '=' account |

**Overall: mixed** (7 unresolved, 2 '=').
- **The registration had no label for "both routes, same sign, stable".** That is what 7 of 9 runs show. The label "unresolved" undersells a clean result.
- **The fault is in the registered categories, not the data.** I adopted the thresholds without adding a "split" category.
- The thresholds stay as registered. The split is reported as a description, not a verdict.

## What it shows
**Shown:**
1. **The deficit is split between two routes, with the same sign, and none of it goes through the op position.**
   - The '=' stream after layer 0 carries 60–85 % in 8 of 9 runs.
   - Layer 1's reads of the operand positions carry 15–40 %.
   - The op position carries 0–1 % everywhere.
2. **No cancellation.** Both routes are positive in every shared-token run, and Σ |route| equals the total.
3. **Robust to the interpolation choice.** A and B differ by at most 0.36 nats on any route (add_s1 ops 1.51 vs 1.81).
4. **Per pair, the '=' stream is the largest route for 53–69 % of diagonal pairs**, against about 40 % for off-diagonal controls.
5. **mul_s0 is the one exception** (ops 57–59 %). It also has the smallest gap among mul runs (3.83).
6. **The separate-token runs have near-zero totals** (−0.39 to +0.59). sepops_s1 shows +1.6 / −1.0 cancellation at the noise level.

**Inferred:**
- **The '=' route** is the doubled-operand content that enters '=' at layer 0. H9 showed that the '=' state is close to a function of its operand write. H8 showed its size isn't the problem. So the '=' stream holds the wrong content for the answer: a single operand direction.
- **The operand-read route** fits position 2 being atypical too. On (a, a), position 2's layer-0 state attends over (a, op, a).
- **The final-norm scale (15–30 % in H6) is not a separate route here.** It acts through whichever route changes the final residual, mostly E.

**Limits of the method:**
- The attribution is relative to the chosen baseline (same-answer off-diagonal donors) and the straight-line path. Another baseline or path could split the deficit differently.
- Integrated gradients is unique only given the path. "60–85 % '='" means "along this path", not an intrinsic property.

## Together with H1–H9
- The diagonal's lower margin is not about exposure (after the first ~200 updates), layer-1 attention allocation, readout-invisible energy, or the size of the doubled write.
- It is mostly about **what the (a, a) input puts into the '=' stream at layer 0** (about two thirds), plus how layer 1 reads the operand positions (about one third).
- The network learned a solution that serves (a, b) pairs and handles (a, a) less well. That points to a training question, not more mechanism work.

## Most informative next step
**This is training, not more mechanism work. It needs GPU time and your OK.** The earlier analysis proposed a **2× diagonal-weight arm**: the same runs with diagonal train pairs upweighted, 3 seeds.

- **If the deficit is a learned trade-off** (the solution under-serves a small input class): the train gap shrinks a lot, and the '=' route share falls most.
- **If it is a capacity or geometry limit** of a shared embedding with a commutative readout: the gap barely moves, and off-diagonal margins may drop.

## Corrections after reading Codex and rerunning (v2: `attrib_results_v2.md`)
- **Controls:** my claim above, "Σ |route| ≤ 0.65", used route *means*. Measured per pair, the controls move 5.1–6.5 nats of total route against 3.1–3.9 nats of |total|. Diagonals move 6.7–7.8 against 4.4–6.5.
  - So routes are large and noisy for every pair, and the diagonal signal lives in the average.
  - Per-pair cancellation is modest: Σ |route| / |total| is 1.2–1.6 on diagonals.
- **"Top route"** was the largest signed route. v2 reports the largest positive route, and the fractions barely change.
- **Quadrature:** 32 vs 64 nodes moves no route mean by more than 0.022 nats, and the 64-node completeness mean is ≤ 0.006. The route split is numerically stable.
