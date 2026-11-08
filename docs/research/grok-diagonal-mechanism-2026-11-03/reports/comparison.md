# Blind Codex review vs Claude's analysis, 2026-11-03

**Files:**
- `codex_results_review.md` (Codex, gpt-6.1-sol, high effort): saw only the hypotheses, the tables, the probe code and the raw logs. Its log shows it never opened `claude_analysis.md`.
- `claude_analysis.md` (Claude): written before Codex's review was read.

## Where they agree

- **The measurements are sound.**
  - logged vs probe within 0.03;
  - all count checks pass;
  - reconstruction error ≤ 4e-5;
  - ratio_L0 leaves shared-token diagonals unchanged.
- **Every registered hypothesis fails at its threshold:**
  - H1 (both predictions);
  - H1b (−2 to 0 %);
  - H2 (no single component in every seed);
  - H3 (gap is 2.0–2.5 at update 500, and ≥ 1 already at update 100);
  - H4 (operand runner-up 0–2 %; removing operands changes nothing).

  Same deciding numbers on both sides.
- **The deficit has an early part and a late part.**
  - Early, the gap sits mainly in MLP1 and no attention intervention touches it.
  - Late, layer-1 attention adds to it: equalizing the layer-1 ratio removes 1.6–2.3 from off-diagonal margins and nothing from diagonals, 32–49 % of the final gap.
  - Layer-1 operand mass is lower on diagonals.
- **Big ratio_L1 "shrinks" in sub/div are damage**, not a diagonal effect.
- **Sub/div diagonals have constant answers** (0 and 1). That plausibly explains their early negative gap.
- **a+2b shared and s5 s1 use an asymmetric layer 0**, and equalizing it destroys them.
- **Both name the same two candidates** for the early gap:
  - (i) a representation effect of the symmetric read;
  - (ii) diagonal facts getting less effective training.

## Where they differ

| | Codex | Claude |
|---|---|---|
| Lead account of the early gap | (i) Representation: the symmetric read makes (a, a) a distinctive input that downstream nonlinear layers separate less well. Exposure is the main alternative. | (ii) Exposure: (a, b) and (b, a) look nearly identical after a symmetric read, so each off-diagonal fact gets about 2 training examples per diagonal fact. Geometry is the secondary account. |
| Reads the ordered-split control (gap 4.1–4.7) as | Evidence that paired splitting isn't needed. | Still consistent with exposure. In an ordered split the mirror (b, a) is in train with probability about frac, so off-diagonal facts still get about 2× the examples on average. |
| Next experiment | Retrain add/mul with the layer-0 ratio forced to 0.5 vs 0.25, crossed with diagonal loss weight 1 vs 2. | First, a mirror test on existing diag_control checkpoints, no training: compare off-diagonal train pairs whose mirror is in train with those whose mirror is not. Then the 2× diagonal-weight retraining, which both proposed. |

## What only one side caught

**Only Codex:**
- freeze_Lk replaces attention at **every** query position, not just '=', so it is a broader change than the table header suggests;
- separate tokens confound more than repetition: extra input parameters, and second-operand embeddings never compete as outputs;
- late in training, the final-norm inverse RMS is 6–12 % lower on diagonals in all six add/mul runs (reversed at update 500), a candidate late contributor;
- the T7 late averages (e.g. add s0 4.46) must not be compared with final-checkpoint numbers (3.70);
- two JSONL files carry seven non-JSON status lines (harmless; the readers skip them).

**Only Claude:**
- the shape of the layer-1 diagonal shift: weight moves from the operands to `op` and `=`, e.g. add_s0 L1h0 (0.40, 0.08, 0.41, 0.11) → (0.28, 0.20, 0.30, 0.22);
- a+2b s0/s1 put 0.26 of layer-0 operand weight on a, close to the 1:2 coefficient (1/3);
- the weak point of the exposure account: it does not explain why the test gap (7–8.8) is bigger than the train gap.

## Resolution

The disagreement is a ranking of the same two accounts, and one cheap test separates them: **the mirror test**.

In diag_control (ordered split), some off-diagonal train pairs have their mirror in train and some don't.
- **Exposure:** pairs whose mirror is absent sit near the diagonal margin.
- **Representation:** both kinds of off-diagonal pair sit well above the diagonal, because neither is an equal-operand input.

It needs no training: about a minute of CPU per checkpoint, run through the guard. If it is ambiguous, Codex's factorial retraining is the next step; it also covers the 2× diagonal-weight arm both sides proposed.
