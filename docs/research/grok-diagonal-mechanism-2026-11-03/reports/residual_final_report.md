# Residual tests (H7): final report, 2026-11-03

> **Corrections (2026-11-03 consolidation):** "ruled out" is scoped to final weights; "exposure only very early" and "layer-1 attention is not the limiter" overstate; "visible direction" was not measured. Each is marked below. Source: `grok_runs\consolidation_2026-11-03\CONSOLIDATED_REPORT.md`.

**Bottom line.**
- **Confirmed: the diagonal's extra residual energy starts in layer-0 attention.** Its size is exactly what coherent doubling predicts: √2 larger on (a, a), in all 9 shared-token runs (1.409–1.435), and absent with separate tokens (0.93–1.08).
- **Ruled out: invisible energy.** Some of that extra energy is invisible to the answer readout (9/9 runs), but removing it closes only −1 % to 7 % of the margin gap.
- **The deficit is in directions the readout does see.** Whether the early doubling causes it has not been tested yet.

> **Correction (2026-11-03 consolidation):** ruled out **at final weights**. Invisible energy reaches the output only through the final norm, so stripping it measures its whole final-weights effect: −1 to 7 %. That is firmer than an intervention ceiling. Its role during training is untested. The 9 models are all at lr 6e-4, `repo`, dropout 0.1.

## Validity
- The components sum to the final residual exactly (error 0).
- Stripping changes the visible logits by ≤ 5e-4.
- Codex reviewed blind and agreed with every verdict.
- Its one suspected bug (rank 98 with 97 answers) is not a bug: the class set has 99 columns (97 answers, op and '='), the same set every margin in this study uses.

## Registered verdicts
| | prediction | result | verdict |
|---|---|---|---|
| H7a (i) | attn0 RMS ratio ≥ 1.25 (shared), 0.9–1.1 (separate) | 1.409–1.435; 1.080, 0.929, 0.978 | **pass** |
| H7a (ii) | attn0 + mlp0 ≥ 50 % of excess in ≥ 7/9 | 7/9 | **pass, at threshold** |
| H7b (i) | invisible share diag > off in ≥ 7/9 | 9/9 (0.13–0.23 vs 0.09–0.15) | **pass** |
| H7b (ii) | stripping halves the scale gap in ≥ 7/9 | 3/9 | **fail** |
| H7b (iii) | stripping weakens the free-norm offset penalty ≥ 50 % | 0/8 | **fail** |

## What we learned
1. **Coherent doubling is real and exact.** Layer 0 reads a and b at about 0.43 each. On (a, a) the two value vectors are identical and add; off the diagonal they are nearly orthogonal. The result is the √2 ratio.
2. **Downstream reshapes the excess.**
   - mlp0 writes less on diagonals (RMS ratio 0.74–0.88);
   - attn1 adds 28–59 % of the final excess;
   - mlp1 mostly writes less.
   - The diagonal's final residual ends up 10–28 % larger in squared norm.
3. **Most of that excess is visible.** The invisible fraction of the excess ranges from 0.33 to 0.99. H7b (ii) passed only in the three runs where it was above 0.6.
4. **Invisible energy can't be the main cause, by design as well as by data.** Stripping keeps the visible logits fixed, so it can only act through the final-norm scale. That scale part was already capped at 15–30 % of the gap in H6. The remaining 70–85 % sits in the visible logits.
5. **Design limit, owned:** H7b (iii) could not have passed under any mechanism, because the operand offset adds visible content. The test still ruled out "redundancy" as the main story.

## Where the study stands
- **Exposure explains the gap only very early in training** (mirror test, H5/H5b). After that it's representation.
- **Layer-1 operand attention is not the limiter** (H6c).
- **Final-norm scale explains 15–30 %** of the gap (H6).
- **The rest is visible direction.** Its likely root is the √2 doubled operand write at layer 0. That link is inferred, not shown.

> **Correction (2026-11-03 consolidation):**
> - **Exposure:** it explains most of the gap at update 100 and 32–44 % at 200. It fades by about 500 (R 0.87–0.90) but leaves a 5–15 % residual through 10k. These are CPU fp32 reruns, which act as new seeds.
> - **Layer-1 operand attention:** H6c is registered MIXED. At final weights, raising operand mass helps diagonals and off-diagonals alike: no diagonal-specific effect, which is weaker than "not the limiter".
> - **Final-norm scale:** the 15–30 % is measured under `repo`, which also leaves the norm gains undecayed.
> - **"Visible direction":** the 70–85 % is an arithmetic complement of the scale share. Direction was not measured. H8 later found the doubled write's size a minor factor at final weights.

## Next (no training, about a minute on the GPU; needs your go-ahead)
**Operand-gain sweep:** scale only the operand part of attn0's write at '=' by 0.707 / 0.85 / 1.0 / 1.15, on diagonal and off-diagonal pairs separately. Run it in two modes, each with the norm free and with the norm fixed:
- **replay**, which recomputes everything downstream;
- **final-only**, which adds the same change to the final residual with downstream writes cached.

**How to read it:**
- If replay at 0.707 rescues diagonals well beyond final-only, the doubling causes the deficit through downstream processing.
- If neither mode rescues them, the deficit lies in later layers independent of the doubling.

Registration would be written before the run, as before.

Files:
- `residual_results.md`
- `claude_residual_analysis.md`
- `codex_resid_analysis.md`
- `residual_comparison.md`
