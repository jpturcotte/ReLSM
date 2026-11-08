# Operand-gain sweep (H8): final report, 2026-11-03

> **Corrections (2026-11-03 consolidation):** the bottom line and "hurts both alike" (wrong in every mode); "content, not size" (inferred, not shown); the side note's "exactly 0"; the study-status table (exposure, layer-1 attention, size). Each is marked below. Source: `grok_runs\consolidation_2026-11-03\CONSOLIDATED_REPORT.md`.

**Bottom line: the doubled operand write is not what costs diagonals their margin.**
- Shrinking the diagonal's layer-0 operand write to off-diagonal size (g = 1/√2), and letting the rest of the network respond, *lowers* diagonal margins in all 9 shared-token runs. The losses are −0.17 to −1.03 nats, with the norm fixed.
- Off-diagonals fall by about as much.
- The registered alternative wins in 9 of 9 runs: the later layers handle (a, a) worse than (a, b), and this does not depend on how large the doubled write is.

> **Correction (2026-11-03 consolidation):** "hurts both alike" is wrong in every mode. The same gain is applied to every pair, so off-diagonals also move off their operating point: at g = 0.707 (replay, norm fixed) they lose 0.67–1.08 nats. Net of that cost, the diagonal fares better in 6/9 runs (Δdiag − Δoff +0.08 to +0.56, ≤ ~12 % of the gap) and worse in 3/9 (−0.01 to −0.20). With the norm free, replay raises diagonal margins in 9/9 (registered closure +1.3 to +16.1 %, which counts the diagonal change only); net of the off-diagonal change the gap narrows in only 2/9 (−10.8 % to +5.5 %) and widens in 7/9 (corrected 2026-11-08). The registered verdicts (D1 0/9, D2 0/9, D3 1/9, alternative 9/9) stand as counted, but the alternative is met by the generic off-operating-point cost. Supported reading: size alone doesn't selectively rescue diagonals; at final weights it is a minor factor, not ruled out. A final-weights test can't rule out a cause acting during training. All 9 models are at lr 6e-4, `repo`, dropout 0.1.

## Validity
- The manual forward matches model(x) within 3.1e-5.
- The gain changes attn0's write exactly as intended (error ≤ 2.4e-7).
- At g = 1, both modes reproduce the baseline exactly.
- Baseline gaps match the earlier tests (e.g. add_s0 3.70).
- The operand write RMS ratio diag/off is 1.41–1.43 again.
- Layer-0 operand mass is identical on and off the diagonal (e.g. 0.854 vs 0.856).

## Registered verdicts (g = 0.707, replay, norm fixed, 9 shared-token runs)
| | prediction | result | verdict |
|---|---|---|---|
| D1 | closure ≥ 25 % in ≥ 7/9 | −4 % to −23 % (all negative) | **fail, 0/9** |
| D2 | replay beats final-only on diag by ≥ 0.5 in ≥ 7/9 | −0.36 to −1.26 (replay is *worse*) | **fail, 0/9** |
| D3 | Δ diag − Δ off ≥ 0.5 in ≥ 7/9 | −0.20 to +0.56 | **fail, 1/9** |
| Alternative | closure ≤ 10 % in ≥ 7/9 | 9/9 | **supported** |
| Separate tokens | \|Δ diag − Δ off\| ≤ 0.3 in 3/3 | 0.08, 0.14, 0.10 | **pass** |

## What the curves show
1. **Replay, norm fixed.** Lowering the operand gain hurts both groups about equally, and the effect grows with the cut:
   - g = 0.5 gives −1 to −2.7 nats;
   - raising it to 1.15 helps a little (+0.0 to +0.3);
   - at 1.414 the diagonal falls again in 6 of 9.
   
   The downstream layers are tuned to the operand write as it is. Making the diagonal's write off-diagonal-sized just weakens its signal.

   > **Correction (2026-11-03 consolidation):** "about equally" hides a matched difference: net of the off-diagonal loss, the gap narrows in 6/9 runs (+0.08 to +0.56 nats). See the correction under the bottom line.

2. **Final-only, norm fixed** (the direct readout of the write):
   - a cut gives the diagonal a small gain, +0.01 to +0.45, in 8 of 9;
   - off-diagonals lose a little.
   
   So the extra half of the diagonal's operand write points slightly *away* from the answer when read directly. But downstream recomputation more than undoes that gain.
3. **Norm free.** Cutting the gain mostly helps through the norm (final-only: +1 to +2 nats for both groups). Replay with the norm free gives small, mixed changes. Neither is selective for diagonals in a way that closes the gap.
4. **Downstream responses at g = 0.707 don't converge to off-diagonal behaviour.**
   - With attn0 matched in size (0.081 vs 0.080), the diagonal's mlp0 write gets *smaller* still (0.042 vs off 0.055 in add_s0).
   - Its attn1 write is larger than off-diagonal (0.076 vs 0.066).
   
   A diagonal input of matched size is still processed differently. What sets (a, a) apart is its content, not its size.

   > **Correction (2026-11-03 consolidation):** "content, not size" is an inference, not a shown result. Size is a minor factor at final weights (see above), and H10 later placed about 2/3 of the deficit in the '=' state after layer 0, path-averaged, which is not localized to layer-0 content.

5. **Accuracy** stays at 0.96–1.00 throughout. These are margin changes, not failures.
6. **Side note:** in all three separate-token models the layer-1 MLP write is exactly 0. That MLP is dead in those runs. It doesn't affect H8, but it's worth knowing for the design doc.

   > **Correction (2026-11-03 consolidation):** not exactly 0. The write RMS is about 5e-14 to 3e-13: functionally dead. It is dead in all 6 separate-token models (diag and add2b), and near-dead in 4/6 E1 sub/div runs. Separate-token models are therefore a different circuit, not the shared-token one minus the diagonal problem.

## Where the study stands
| question | answer | test |
|---|---|---|
| Exposure (how often (a, a) is seen)? | only in the first ~200 updates | H5, H5b |
| Layer-1 operand attention? | not the limiter | H6c |
| Final-norm scale? | 15–30 % of the gap | H6 |
| Readout-invisible energy? | ≤ 7 % | H7 |
| Size of the doubled layer-0 write? | **no**: matching it makes things worse | H8 |
| What remains | content: (a, a) gives one operand direction instead of two, and the later layers handle that input worse | inferred |

> **Correction (2026-11-03 consolidation):** three rows overstate.
> - **Exposure:** most of the gap at update 100, 32–44 % at 200, fading by about 500 (R 0.87–0.90) with a 5–15 % residual through 10k. These are CPU fp32 reruns, which act as new seeds.
> - **Layer-1 operand attention:** registered MIXED. At final weights, raising operand mass helps diagonals and off-diagonals alike. That is no diagonal-specific effect, not "not the limiter".
> - **Size:** minor at final weights (≤ ~12 %): net of the off-diagonal cost the gap narrows in 6/9; with the norm free, net of the off-diagonal change the gap narrows in only 2/9 (−10.8 % to +5.5 %; corrected 2026-11-08: the 1–16 % counted the diagonal change alone). Not "no".
> - **Final-norm scale:** the 15–30 % is measured under `repo`, which leaves the norm gains undecayed.

## Next (no training, about a minute on the GPU; needs your go-ahead)
**Content swap:**
- On diagonal pairs, replace the operand part of attn0's write with the one from (a, c), for c ≠ a, then replay.
- That supplies two different operand directions.
- If content is the cause, the margin for the true answer 2a should *fall* (the input now encodes a + c). The other answers' logits then show whether the network reads "two directions" more sharply.

**A cleaner variant:** compare (a, a) with the average of the (a, c) and (c, a) operand writes over many c. That shows how far the diagonal's operand content sits from the off-diagonal family.

I'd register both before running.

Files:
- `gain_results.md`
- `gain_test.py`
- `gain_tables.py`
- `gain\all.jsonl`
- `hypotheses.md` (H8 registration and result)
