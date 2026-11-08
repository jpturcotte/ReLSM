# Why diagonal pairs (a, a) are weak: what the mirror test settled (2026-11-03)

> **Corrections (2026-11-03 consolidation):** exposure fades by about update 500, not 300, and leaves a 5–15 % residual; the over-training curve comes from CPU fp32 reruns that act as new seeds; the layer-1 lead conflates operand mass with the a/b ratio. Marked below. Source: `grok_runs\consolidation_2026-11-03\CONSOLIDATED_REPORT.md`.

**The question:** add and mul models with shared operand tokens have weak diagonal margins. Is that because each diagonal fact gets fewer training examples (exposure), or because (a, a) is a special input (representation)?

**The test:** with the ordered split, some training pairs (a, b) also have their mirror (b, a) in train ("mirror" pairs) and some don't ("lone" pairs).
- If exposure drives the gap, lone pairs should sit near the diagonal.
- If representation does, lone pairs should sit with the mirror pairs.

R = (lone − diagonal) / (mirror − diagonal).
- R near 0 means exposure.
- R near 1 means representation.

## Results

**Final weights (GPU runs, 25,000 updates):** R = 0.96 / 0.89 / 0.91. Representation: lone pairs score almost as high as mirror pairs, about 4 logits above the diagonal.

**Over training** (CPU reruns, checkpoints at 100 to 10,000 updates):

| update | R, s0 / s1 / s2 | gap mirror − diag (s0 / s1 / s2) |
|---|---|---|
| 100 | 0.22 / 0.24 / 0.22 | 0.84 / 1.00 / 0.96 |
| 200 | 0.57 / 0.56 / 0.68 | 0.92 / 1.28 / 1.45 |
| 300 | 0.82 / 0.83 / 0.88 | 0.97 / 1.14 / 1.45 |
| 500 | 0.90 / 0.89 / 0.87 | 1.68 / 1.73 / 1.46 |
| 10,000 | 0.91 / 0.93 / 0.95 | 3.53 / 4.24 / 5.29 |

Separate operand tokens (control): the mirror effect is ≤ 0.11 at every checkpoint, and the gap is about 0.

## What it means

1. **During memorization (the first ~200 updates), exposure explains most of the small gap.**
   - A model that reads a and b symmetrically treats (a, b) and (b, a) as one fact, so facts whose mirror is also trained are learned faster.
   - Diagonal facts have no separate mirror, so they lag.
2. **From about update 300 on, exposure stops mattering**, and the diagonal keeps falling further behind (the gap grows from about 1 to 3.5–5.3). That growth is about (a, a) as an input, which matches Codex's lead account.
3. **Held-out pairs whose mirror is trained score like trained pairs** (e.g. 14.61 vs lone-train 14.64 in control s0). So the model really does treat (a, b) and (b, a) as the same input.

> **Correction (2026-11-03 consolidation):**
> - **Points 1 and 2:** exposure explains most of the gap at update 100 (1 − R = 0.76–0.78) and 32–44 % at 200. It fades by about 500 (R 0.87–0.90), not 300. It does not vanish: from 300 to 10k, mirror − lone stays +0.17 to +0.39 nats (separate tokens ≤ 0.11), a 5–15 % residual.
> - **The over-training rows are CPU fp32 reruns,** not the GPU models. They act as new seeds: at 9–10k their train gap is larger in all 3 seeds, by 0.33–1.13, and control_s2 first reaches unseen 0.99 at 3,200 vs 300. Transfer of the R curve to the GPU runs is plausible but not shown.
> - All runs are at lr 6e-4, `repo`, dropout 0.1.

## By the registered rule

The verdict is mixed across seeds: s1 shows the full early-exposure-then-representation pattern, s0 counts as representation, and s2 matches neither cleanly. That is only because the D ≥ 1 gate hides the earliest checkpoints differently per seed; the curves themselves match.

## Still open

Why the symmetric (a, a) input is handled worse. Leads from the probe:
- layer-1 attention moves weight off the operands on diagonals (32–49 % of the late gap);
- the final-norm scale is 6–12 % lower on diagonals late in training.

> **Correction (2026-11-03 consolidation):** the first lead joins two different measurements. On diagonals, head-mean layer-1 operand mass drops 8–25 % (median 18 %). The 32–49 % comes from equalizing the a/b ratio, which lowers *off-diagonal* margins by 1.6–2.3 nats and moves diagonals ≤ 0.006. H6c later found the mass effect non-selective (registered MIXED).
