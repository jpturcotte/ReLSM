# Why diagonal pairs (a, a) are weak: mechanism study, 2026-11-03

In `grok_lab` models trained on a commutative task (add, mul) with shared operand tokens, diagonal pairs (a, a) end training with lse margins 4–6 nats lower than off-diagonal pairs. With separate operand tokens the gap is about 0. This folder is the study of why.

**Method:**
- Every test was pre-registered in `hypotheses.md` before it ran, with thresholds.
- Each ambiguous result got a blind analysis by Codex (gpt-6.1-sol), which never saw this session's analysis.
- The two analyses were then compared.
- Torch jobs ran on the GPU through the memory guard. Each takes seconds per model.

**Models:** the final weights of 12 runs:
- E1 add and mul, seeds 0–2;
- diag-arms control (ordered split), seeds 0–2;
- separate-token control, seeds 0–2.

## Results

| test | question | registered verdict | what it shows |
|---|---|---|---|
| H1–H4 (probe) | operand as competitor, a/b-ratio computing, one component | all failed | none of these explain it |
| H5, H5b (mirror) | exposure (diagonals have no swapped twin) vs representation | mixed by the noise gate; the curves agree | exposure explains it only in the first ~200 updates |
| H6, H6c (restore) | layer-1 operand attention; final-norm scale | attention: mixed; scale: 15–30 % of the gap | layer-1 operand attention is not diagonal-specific; the final-norm scale is part of it |
| H7 (residual) | where the extra residual norm comes from; is it readout-invisible | H7a pass; H7b (ii), (iii) fail | layer-0 attention writes √2 more on (a, a) (coherent doubling); invisible energy closes ≤ 7 % |
| H8 (gain) | does the doubled write's size cause it | fail, 0/9; alternative 9/9 | no: shrinking it hurts diagonals and off-diagonals alike |
| H9 (patching) | '=' stream vs layer-1 operand reads, by transplant | mixed; design flaw | transplants between same-answer pairs cost 3–7 nats even off the diagonal, so they can't localize |
| H10 (route attribution) | the same question, by integrated gradients | mixed (7 unresolved, 2 '='); no "split" category was registered | about 2/3 '=' stream after layer 0 (60–85 % in 8/9), 1/3 layer-1 operand reads, 0 op position; per pair it is noisy |

**Summary:**
- The diagonal deficit is about **what (a, a) puts into the '=' stream at layer 0** (content, not size), plus a smaller share in how layer 1 reads the operand positions.
- It is not about exposure (after early training), layer-1 attention allocation, readout-invisible energy, or the size of the doubled write.

**A correction to the probe's reading:** the probe found that equalizing the layer-1 a/b attention ratio removes 32–49 % of the final gap. H6c later showed that raising operand attention helps diagonal and off-diagonal pairs about equally. So layer-1 attention is not a diagonal-specific limiter.

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
- **Finer attribution, no training:** split the '=' route into residual / query / self K-V, and the operand route into K / V.
- **A 2× diagonal-weight training arm:** is the deficit a learned trade-off or a structural limit?
