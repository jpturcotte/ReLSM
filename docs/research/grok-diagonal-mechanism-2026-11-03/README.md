# Why diagonal pairs (a, a) are weak: mechanism study, 2026-11-03

In `grok_lab` models trained on a commutative task (add, mul) with shared operand tokens, diagonal pairs (a, a) end training with lse margins about 3.7–6.2 nats lower than off-diagonal pairs (train, final weights, E1 add and mul) at lr 6e-4, `--wd_scope repo`, dropout 0.1. At lr 1e-3 the long-run a+b train gap is 2.6–3.4 (corrected 2026-11-03, consolidation). With separate operand tokens the gap is about 0. This folder is the study of why.

**Corrections from the 2026-11-03 consolidation** (full analysis: `grok_runs\consolidation_2026-11-03\CONSOLIDATED_REPORT.md`):
- The gap is now stated with its base (lr 6e-4, `repo`, dropout 0.1). Every model here was trained at that one base. No-training checks on older checkpoints (2026-11-03, `grok_runs\checks_2026-11-03\RESULTS.md`) since showed:
  - **Under full decay** (C1, lr 1e-3, dropout 0, 1e4 updates): the gap exists in 3/3 seeds, but it is small, 0.3–0.6 nats (8–34 % of the off-diagonal margin). As a share of the off-diagonal margin one seed matches the base's 0.27–0.39 (E1) and two are about 3× smaller, so most of the drop in nats follows the logit scale; this does not show that the stack causes most of the deficit (corrected 2026-11-08, misread audit; this said most of its size comes from the training stack).
  - **At lr 1e-3 on this base:** the gap is about half the 6e-4 gap at 1e5 updates, the route mix is different, and the layer-1 MLP decayed to zero in 2/3 seeds (see the scope note in the Summary).
  - **Dropout:** at C1's settings, adding dropout 0.1 makes every unseen error diagonal (3/3 seeds). Dropout 0 at this base is still untested.
- H2's "one component" rule was ill-posed; H1, H3 and H4 failed as registered.
- H5: exposure fades by about update 500, not 200, and leaves a 5–15 % residual. The early curve comes from CPU fp32 reruns, which act as new seeds.
- H6/H6c: "not diagonal-specific" restated as the registered MIXED verdict at final weights. H6a had no matched off-diagonal arm.
- H8: "hurts diagonals and off-diagonals alike" was wrong in every mode. Size is a minor factor at final weights, not ruled out.
- Summary: "content, not size, at layer 0" replaced by the path-averaged H10 statement.
- The probe's 32–49 % (a/b ratio) and H6c (operand mass) measure different quantities; the correction paragraph is rewritten.

**Method:**
- Every test was pre-registered in `hypotheses.md` before it ran, with thresholds.
- Each ambiguous result got a blind analysis by Codex (gpt-6.1-sol), which never saw this session's analysis.
- The two analyses were then compared.
- Torch jobs ran on the GPU through the memory guard. Each takes seconds per model.

**Models:** the final weights of 12 runs:
- E1 add and mul, seeds 0–2;
- diag-arms control (ordered split), seeds 0–2;
- separate-token control, seeds 0–2.

**Scope** (added 2026-11-03, consolidation): all 12 models were trained at lr 6e-4 constant, `--wd_scope repo`, dropout 0.1, under bf16 autocast, and rescored in fp32. The 3 control models are the same trajectories, bit for bit, as long-stack `stack_lr6e4` to 25k, so they are not an independent replication. Every intervention is at final weights, so none rules out a training-time cause. The separate-token models have a functionally dead layer-1 MLP (write RMS 5e-14 to 3e-13), so they are a different circuit, not the shared-token one minus the diagonal problem.

## Results

| test | question | registered verdict | what it shows |
|---|---|---|---|
| H1–H4 (probe) | operand as competitor, a/b-ratio computing, one component | all failed | H1, H3 and H4 failed as registered. H2's "one component" rule is ill-posed: shares exceed 100 % because mlp1 cancels, and in 4/6 seeds one component is ≥ 60 % (corrected 2026-11-03, consolidation) |
| H5, H5b (mirror) | exposure (diagonals have no swapped twin) vs representation | mixed by the noise gate; the curves agree | exposure explains most of the gap at update 100 and 32–44 % at 200. It fades by about 500 (R 0.87–0.90) but leaves a 5–15 % residual through 10k. These are CPU fp32 reruns, which act as new seeds (corrected 2026-11-03, consolidation) |
| H6, H6c (restore) | layer-1 operand attention; final-norm scale | attention: mixed; scale: 15–30 % of the gap | at final weights, raising layer-1 operand mass helps diagonals and off-diagonals alike (registered MIXED). H6a ("not causal") had no matched off-diagonal arm (corrected 2026-11-03, consolidation). The final-norm scale is part of it, measured under `repo`, which leaves the norm gains undecayed |
| H7 (residual) | where the extra residual norm comes from; is it readout-invisible | H7a pass; H7b (ii), (iii) fail | layer-0 attention writes √2 more on (a, a) (coherent doubling); invisible energy closes ≤ 7 % |
| H8 (gain) | does the doubled write's size cause it | fail, 0/9; alternative 9/9 | size alone doesn't selectively rescue diagonals. Replay, norm fixed: net of the off-diagonal cost the gap narrows in 6/9 (≤ ~12 %); in the final-only modes it narrows in 8/9, by up to ~18 % (norm fixed) and ~30 % (norm free). Replay, norm free: removing the doubling's size closes 1–16 % of the gap in 9/9, the registered closure, and H9's size-only test (P_size, −1 to +15 %) matches it run by run. It is not diagonal-selective: net of the off-diagonal change the gap narrows in only 2/9 (−10.8 % to +5.5 %). At final weights, size is a minor factor, not ruled out (corrected 2026-11-03, consolidation; 2026-11-08, misread audit: an earlier 2026-11-08 note read the net figure as closure and called size weaker than written; that is withdrawn) |
| H9 (patching) | '=' stream vs layer-1 operand reads, by transplant | mixed; design flaw | transplants between same-answer pairs cost 3–7 nats even off the diagonal, so they can't localize |
| H10 (route attribution) | the same question, by integrated gradients | mixed (7 unresolved, 2 '='); no "split" category was registered | about 2/3 '=' stream after layer 0 (60–85 % in 8/9), 1/3 layer-1 operand reads, 0 op position; per pair it is noisy |

**Summary** (corrected 2026-11-03, consolidation):
- **Scope (added 2026-11-03):** the H10 route shares were measured at lr 6e-4 and 25k updates, and don't carry over to lr 1e-3.
  - At lr 1e-3 (long-stack runs that differ from the 6e-4 control only in lr, scored at 1e5 updates), the operand route carries 103–131 % of the deficit and the '=' route is negative (−3 to −31 %) under path choice A; under choice B the ranges are 116–174 % and −16 to −74 %, and s1 is rated coupled; both falsifiers fire in 3/3 seeds under either choice (corrected 2026-11-08, misread audit). The registered prediction (K2b in `checks_2026-11-03\RESULTS.md`) is falsified; Codex agreed blind.
  - In 2 of those seeds, the layer-1 MLP had decayed to zero. lr and lr × wd are confounded here.
  - Even at 6e-4, by 1e5 updates the '=' share is 48–70 %.
  - The routes cancel heavily from pair to pair, so the shares are path averages.
  - **The lr difference doesn't depend on the method** (per-pair replay, `checks_2026-11-03\RESULTS.md` R; Codex agreed blind). An exact Shapley split, which uses no path, gives '=' shares of −82 % to −134 % at 1e-3 against 39–61 % at 6e-4.
    - The point shares do depend on the method: Shapley moves about 10 points from '=' to operands at 6e-4, and 50–117 at 1e-3.
    - Per-pair signs often disagree between methods.
    - Report the shares as ranges across methods, and don't read single pairs.
- At lr 6e-4 and 25k updates, averaged along one integration path, about 2/3 of the deficit flows through the '=' state after layer 0. That state includes the layer-1 query and the '=' self K/V, so "layer-0 content" is not localized. The rest goes through layer 1's reads of the operand positions.
- Exposure fades early but leaves a residual. Layer-1 attention and size show no diagonal-specific effect at final weights. Invisible energy accounts for ≤ 7 % at final weights.
- Training-time roles are untested. All of this is at lr 6e-4, `repo`, dropout 0.1.

**A correction to the probe's reading** (rewritten 2026-11-03, consolidation): removing the token-dependent layer-1 a/b split lowers *off-diagonal* margins by 1.6–2.3 nats (32–49 % of the gap) and leaves diagonals unchanged. Raising operand mass at a fixed split (H6c) gives a small, non-selective gain (MIXED). The two tests measure different quantities. Neither shows that layer-1 attention causes the deficit, and neither rules it out.

## Files
- **`hypotheses.md`:** every registration and result, H1–H10, in order.
- **`reports/`, the main report per test:**
  - mirror: `mirror_summary.md`;
  - restore: `restore_final_report.md`;
  - residual: `residual_final_report.md`;
  - gain: `gain_final_report.md`;
  - patching: `patch_comparison.md`;
  - route attribution: `attrib_final_report.md`.
- **`reports/`, supporting files:**
  - the tables behind each test: `*_results.md`, `*_tables.md`;
  - this session's analyses: `claude_*`;
  - Codex's blind analyses: `codex_*`;
  - the comparisons: `*_comparison.md`, `comparison.md`.
- **`code/`:** the test and table scripts.
  - They import `tools/grok_mech_probe.py`, and their paths point at this machine's run folders.
  - `watch_guard.sh` is the guard-log watcher used to monitor runs.
- **`data/`:** the raw results (JSONL, one record per run), so most tables can be rebuilt without the checkpoints. `attrib_v2` supersedes `attrib_v1`; it adds per-pair cancellation, the largest positive route, and 64-node quadrature.
  Run from `code/`. These reproduce their reports exactly: `residual_tables.py`, `restore2_tables.py` and `mirror_score.py` (they read `data/` directly), and `attrib_tables.py ../data/attrib_v2.jsonl`, `gain_tables.py ../data/gain.jsonl` and `patch_tables.py ../data/patch.jsonl`. `attrib_tables.py ../data/attrib_v1.jsonl` matches `attrib_results.md` apart from two headings that changed for v2. Not rebuildable from `data/`: `restore_score.py` (the v1 restore records), `mirror_score_early.py` (the early mirror records) and `results_tables.py` (the probe records and the E1 training logs); their inputs were not committed.

## Open
Two next steps were proposed and neither has run:
- **Finer attribution, no training:** split the '=' route into residual / query / self K-V, and the operand route into K / V. (2026-11-03: run it on the 6e-4 and the 1e-3 models together, since the route mix differs between them. A per-pair path replay on the saved checkpoints comes first: fixed pair IDs, finite-difference route checks, and attention, norm and MLP activity along the path.)
- **A 2× diagonal-weight training arm:** is the deficit a learned trade-off or a structural limit? (2026-11-03: it waits for the scope and lr checks, and needs a matched off-diagonal control when it runs. Its lower priority in the consolidation rested partly on the H8 size figures. An earlier 2026-11-08 note said those were misread and the arm's motivation less weakened; the misread audit found the figures stand (see H8), so its priority is as the consolidation set it.)
