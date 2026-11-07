# Grokking criteria and better metrics: an analysis (2026-11-07)

**Question (owner):** we did research before on grokking signs or criteria. Is it still relevant? Are there better metrics? Which would help the project in general, not just for grokking?

**Data used:** the 48 phase 1 runs of the task × format study (`grok_runs/taskfmt_2026-11-07/runs`), the 6 reused add runs (`format_2026-11-05/runs/{TA,D}_s0-2`), and the 15 E1 runs (`e1_2026-11-02`) as an out-of-sample check. Everything was computed from `metrics.jsonl`, `final_errors.json` and `split_counts.json`, with stdlib and numpy only. No torch, no checkpoints. The script is `curve_metrics.py` in this folder; its full output is `curve_results.json`. Step notes are in `progress.md`.

**Labels used below.** **[data]** means tested here on the project's runs. **[lit]** means it comes from the literature only. **[post hoc]** means a rule or cut-off chosen after looking at these runs; it needs pre-registration before it decides anything.

## Top 5 recommendations

1. **Replace the single 0.99 state with an offline curve-summary readout**, for grok_lab and train.py alike. It reports the late level (raw and chance-corrected), a trend β with a projected 0.99 crossing, t50 and the transition width in decades, the delay ratio, and the loss hump. It costs one small script and no training time.
2. **Log per-task balanced accuracy, worst-class recall and the majority baseline at every eval.** For gcd, plain accuracy ranks the digit arm (D) above the one-token arm (TA) in 3/3 seeds; balanced accuracy ranks them the other way in 3/3. Kappa hides the gap.
3. **Always pair exact match with per-task held-out answer loss and margin.** train.py lacks per-task answer loss. Held-out loss peaks and turns down 200–2,500 updates before accuracy moves, in 18/18 runs that have a memorisation phase.
4. **Call it "grokked" only when generalisation comes after memorisation** (delay ratio ≥ 2). 19 of the 33 runs that today's rule scores "groks" have no memorisation phase. The harness mostly measures generalisation time.
5. **For train.py, add a resume-equivalence check**: first diverging update, largest loss-trace gap, and a parameter hash. Also add a dormant-unit fraction, per-layer norms and a measured update ratio (the #149 fix) for the layer-1 MLP disengagement question.

**Earlier criteria to retire (as decision rules):**
- **The 0.99 first crossing as the only run state.** Keep it as a secondary number.
- **Sustained ≥ 0.99 through the horizon.** It has the worst seed spread here, log-sd 0.35.
- **"Memorises" = late median ≤ 0.05.**
- **Margin-cliff fractions as a failure sign.** Keep them as a description only.
- **Plain accuracy for skewed tasks.**

Everything else in the earlier research still stands as logging advice. Only the advice on what to conclude from it needs changing.

## 1. Conclusions first

1. **The 0.99 first-crossing rule measures "generalised by the horizon", not grokking.** Grokking means held-out accuracy rises long *after* the model has fitted its training set (Power et al. 2022). The rule never checks that delay. In phase 1, 19 of the 33 runs it scores "groks" show no memorisation phase: held-out accuracy reaches half its final level within 2× of the update at which training accuracy reaches 0.99, and held-out loss never rises above its starting value. Examples are D_add, D_sub, D_add2b, TA_div, TA_sub and TA_sq. [data]
2. **The rule's three states lump unlike runs together and split like ones apart.** "Partial" covers D_mul (0.97 and climbing fast) and TA_cubeab (0.05–0.09, creeping up from chance). Meanwhile TA_sqab's three seeds, which share one shape, come out as partial / memorises / memorises: their late medians sit at 0.054, 0.033 and 0.023, against a 0.05 cut-off. TA_cubeab_s0 is "partial" by 0.002. [data]
3. **A small set of threshold-free numbers separates all five problem cases.** The set is:
   - the **late level**: the median accuracy over the last 1,000 updates, also corrected for chance;
   - the **trend β**: the slope of logit(accuracy) against ln(update) over the last 60 % of the run;
   - the **timing** t50: when the run reaches half of its own final level;
   - the **width**: t10 to t90, in decades of updates;
   - the **delay ratio**: t50 divided by the update at which training accuracy reaches 0.99;
   - the **loss hump**: the peak held-out loss divided by the loss at the start.

   Section 4 gives the numbers. [data, post hoc cut-offs]
4. **For skewed tasks, accuracy is the wrong scale, and so is Cohen's kappa.** For gcd, plain accuracy ranks D above TA in 3 of 3 seeds (0.978–0.985 against 0.976–0.980). Balanced accuracy ranks TA above D in 3 of 3 (0.81–0.90 against 0.71–0.72). Kappa sits at 0.96–0.975 in all six runs and hides the gap. Balanced accuracy and macro-F1 show it. Near chance the roles flip: kappa is the useful one there, and it separates TA_cubeab (0.046–0.079) from D_cubeab (0.003–0.006). [data]
5. **"Sustained ≥ 0.99 through the horizon" is the least reliable timing here.** Its spread across seeds is the largest of the timings tested (pooled log-sd 0.35, against 0.19 for the first crossing and 0.13 for t50). It also turns a near-line wobble into a 10× delay: D_add's sustained times are 21,600 / 4,200 / 12,800, against first crossings of 1,300 / 900 / 1,600. E1 found the opposite (0.16 against 0.20), because its plateaus sat well above the line. So that advantage was not general. [data]
6. **Extrapolating past the horizon is useful but biased optimistic.** Fitted at 15k, β put D_mul's 0.99 crossing at 24–28k, yet no seed crossed by 25k. Fitted at 25k it says 36–43k. A sigmoid fitted in log time says D_mul levels off at 0.97–0.98 and never crosses. The two models disagree, so only the 100k rerun the E1 rule already asks for can settle D_mul. Record the 36–43k projection as a prediction before that run. [data]
7. **Most of the earlier research still stands as logging advice.** Its advice on what to *conclude* needs changing. Keep:
   - held-out loss and margin next to accuracy;
   - transpose-seen and diagonal splits;
   - no early stopping;
   - weight norm as a description, never a trigger;
   - "call a defect on a positive sign, not on elapsed steps".

   Retire, as decision rules: the 0.99 first crossing as the *sole* state, sustained-0.99, the 0.05 "memorises" line, and margin-cliff fractions. The cliff fractions were already demoted for CUDA in the 11-03 and 11-06 consolidations. [data + earlier docs]
8. **Beyond grokking, the best value for cost is an offline curve-summary tool plus three cheap eval-time additions:** per-task balanced accuracy, per-task held-out answer loss, and a resume-equivalence check. Dormant-unit fractions per MLP layer come next; they are the direct probe for the layer-1 MLP disengagement in the 11-06 consolidation. LLC, CKA, restricted/excluded loss and memorisation scores are expensive. No current decision waits on them. [lit + project needs]

## 2. Recommendation table

Cost: **S** means under a day and under about 150 lines. **M** means one to three days. **L** means more. "Offline" means it runs on logged files after the run, at no training cost.

| # | Metric | Verdict | Where | Cost | Needs | Evidence |
|---|---|---|---|---|---|---|
| 1 | Curve summary: late level (raw and chance-corrected), β trend with projected 0.99 crossing, t10/t50/t90, width in decades, delay ratio, share of late evals ≥ 0.99, loss hump and its update | **Adopt** | an offline tool over grok_lab `metrics.jsonl` and train.py `metrics.json`, per task | S | logged scalars only | [data] |
| 2 | Per-task balanced accuracy, worst-class recall, majority baseline | **Adopt** | grok_lab `evaluate` (a bincount); train.py per-task eval, by answer class or by answer-length/first-digit bucket | S | an eval pass already done | [data] gcd |
| 3 | Per-task held-out loss and mean or 5th-percentile margin, as the continuous partner of exact match | **Adopt**: grok_lab has it; add per-task teacher-forced answer loss to train.py | both | S | an eval pass | [data] + [lit] Schaeffer |
| 4 | Cohen's kappa, or chance-corrected accuracy, to call "above chance" | **Adopt** for near-chance calls only, not for skewed tasks | both | S | the eval confusion | [data] |
| 5 | Resume-equivalence check: first diverging update, largest loss-trace gap, parameter hash at matched updates | **Adopt** | a train.py test, run once per resume-path change | S–M | short runs | project fact: same seed gives bit-identical results |
| 6 | Dormant-unit fraction per MLP layer, per-layer weight norm, measured update ratio (fixes #149) | **Adopt** | train.py (the norms exist in grok_lab) | S–M | one forward hook at eval | [lit] Sokar; project §3 of 11-06 |
| 7 | First crossing of 0.99 | **Keep, demoted** to a secondary, backward-compatible number | both | none | scalars | [data] |
| 8 | Sustained crossing of 0.99 | **Drop as a primary.** Use the share of late evals ≥ 0.99, or sustained ≥ 0.95 (log-sd 0.155) | both | none | scalars | [data] |
| 9 | "Memorises" = late median ≤ 0.05 | **Drop**; use flat β plus kappa ≈ 0 | score scripts | none | scalars + confusion | [data] |
| 10 | Area under the accuracy curve on log time | **Maybe**: a one-number paired summary (lowest seed spread, 0.09), but it mixes timing and level | offline | S | scalars | [data] |
| 11 | Fitted sigmoid (onset, width, top) | **Maybe**: descriptive only. Good fits on transitions (R² ≥ 0.99), meaningless on flat runs; its "top" disagrees with β on D_mul | offline | S | scalars | [data] |
| 12 | Loss-peak early warning (held-out loss has passed its peak) | **Maybe**: pre-register, then test | offline | S | scalars | [data, post hoc] 18/18 |
| 13 | Spectral early warning on the per-update training loss (Notsawo et al.) | **Maybe**: free to test on `train_obj_trace`; not tested here | offline | S | scalars | [lit] |
| 14 | Fourier and log-Fourier top-5 share of the number embeddings | **Keep in grok_lab** (it exists); harness only | grok_lab | none | the eval | [lit] Nanda |
| 15 | Restricted and excluded loss | **Maybe, harness only**, for a mechanism question on add or mul | offline from checkpoints | M | checkpoints | [lit] Nanda |
| 16 | Margin-cliff fractions, zero-target-gradient share | **Keep as descriptors only** (already decided) | grok_lab | none | the eval | earlier docs |
| 17 | Weight norm as a grokking trigger | **Drop as a trigger**; log it as a description | both | none | scalars | [lit] |
| 18 | Effective rank of the embedding or weights | **Maybe**: descriptive, never read at the grok step | both | S | weights | [lit] |
| 19 | Expected calibration error on held-out answers | **Maybe for train.py lexical evals; drop for grokking** | train.py | S | the eval | [lit] |
| 20 | Gradient noise scale | **Maybe for train.py batch-size decisions**; cheap from the 4 × 128 micro-batches | train.py | S–M | gradients | [lit] |
| 21 | CKA or other representation similarity | **Maybe**, for multitask interference questions | offline from checkpoints | M | checkpoints + activations | [lit] |
| 22 | Local learning coefficient | **Not now**: costly and tuning-sensitive, and no decision waits on it | offline | L | checkpoints + sampling | [lit] |
| 23 | Memorisation scores (Feldman and Zhang) | **Drop**: they need very many retrained models, and the finite-table split already gives the exact seen/unseen contrast | — | L | many runs | [lit] |

## 3. What from the earlier research still holds

Sources: `grokking-lit-2026-10-28/report.md`, `grokking-fix-plan-2026-10-28.md`, `grok-multitask-design-2026-11-02.md`, `grok-consolidation-2026-11-06.md`, and the diagonal-mechanism study.

**Still relevant, and supported by later data:**
- **Score unseen off-diagonal pairs; report diagonals and transpose-seen pairs separately.** The multitask design showed that diagonals and transpose leakage change crossing times by hundreds to thousands of updates. Every number here uses unseen off-diagonal accuracy.
- **Log held-out loss and confidence, because they move before accuracy** (lit report rec. 10, citing [20]). Supported [data]: in all 18 runs with a memorisation phase, held-out loss peaks and starts falling 200–2,500 updates before accuracy reaches 10 % of its final level. For D_mul that lead is 2,400–2,500 updates, and for D_div 1,200–1,800.
- **Report several thresholds, never just one** (lit report, "Grokking time"). Supported, and here pushed further, to threshold-free summaries.
- **"Call a defect on a positive sign, not on elapsed steps"** (fix plan §3.4). Still right. "Training accuracy never reaches 0.99" (optimisation failure) is the one state of the old rule to keep unchanged.
- **Budget to 1e5 and rerun anything under 0.99 at 25k** (E1 rule). Still right. D_mul now has a quantitative prediction for that rerun (section 4.3).
- **Weight norm per layer as a description, not a trigger** ([11][42][45] in the lit report). Not contradicted. Norms as a trigger were not tested here.
- **Representation metrics after convergence, not at the grok step** ([66]). Not tested, but it argues against reading effective rank or Fourier share at a crossing time.
- **Three seeds screen only large effects** (11-06 §7). Reinforced: the spread of every timing measure across seeds is log-sd 0.09–0.35 (section 4.4).

**Outdated or contradicted by the project's own data:**
- **"Sustained crossing is the lower-variance outcome"** (E2 pre-registration, from E1: 0.16 against 0.20). Contradicted in phase 1 (0.35 against 0.19), where several plateaus sit just above 0.99. Its variance depends on how far the plateau is from the line. It is not a general property.
- **The margin cliff (16.6 in fp32, 5.5 in bf16) as a failure sign.** Already retired for CUDA (11-03). Keep it as a description only.
- **"Grokking time" as the name of what the 0.99 rule measures.** Section 1, point 1: most crossing times in the harness are generalisation times with no memorisation gap. The E1 runs show the same: delay ratios 0.8–2.7, with held-out loss rising at most 8 % above its start, except s5_s2. So results like E2's "add+mul slows add 8×" are slowdowns of *generalisation*. That is still a valid result, but "grokking" overstates the mechanism.
- **The 0.05 "memorises" line, and memorises against partial.** Section 1, point 2.
- **gcd read by plain accuracy.** Already caught in the pre-registration (R5 uses balanced accuracy). The data confirm the reversal: plain and balanced accuracy rank TA and D in opposite orders.

## 4. Test on the phase 1 runs: do threshold-free metrics separate the five cases?

### 4.1 Definitions (all computed by `curve_metrics.py`)

- **Curve:** `test_acc_unseen_offdiag@task`, one eval every 100 updates, 0 to 25,000.
- **Late level L:** the median over updates 24,000–25,000. The single final eval is noisy: D_mul_s0 reads 0.958 at the last eval, against a late median of 0.976.
- **Chance-corrected level:** (L − majority baseline) / (1 − baseline). The baseline comes from `split_counts.json`.
- **β (trend):** the ordinary least-squares slope of logit(acc) on ln(update), over 10,000–25,000, using every fifth eval (one per 500 updates) to cut autocorrelation.
  - Near accuracy 1, logit(acc) ≈ −ln(error), so β is the power-law exponent at which the error falls.
  - Near 0, β is the growth exponent of accuracy itself.
  - **Projected crossing x99:** where the fitted line reaches logit(0.99).
  - **"Rising"** means β > 3 standard errors and β > 0.3 [post hoc].
- **t10, t50, t90:** the first update at which a 5-eval running median reaches 10 %, 50 % or 90 % of the way from baseline to L. These are computed only when the chance-corrected level is above 0.2.
- **Width:** log10(t90 / t10), in decades.
- **Delay ratio:** t50 / Tmem, where Tmem is the first update with training accuracy ≥ 0.99. "Delayed" means a ratio ≥ 2 [post hoc].
- **Loss hump:** the peak unseen held-out loss divided by its value at update 0, which is about ln 97.
- **Balanced accuracy, macro-F1, kappa:** the full confusion matrix on the scored set at update 25,000, rebuilt from the exact answer counts and the wrong rows in `final_errors.json`. This reproduces score.py's gcd balanced accuracy exactly (D_gcd_s0: 0.7222).

### 4.2 The five cases, plus TA_sqab

Ranges are across seeds 0–2.

| | TA_mul (sharp grok) | D_mul (slow climb) | D_div (late jump) | D_sq (near the line) | gcd, TA / D (skewed) | TA_cubeab (creep) |
|---|---|---|---|---|---|---|
| **0.99 rule** | groks ×3 (1,600–2,200) | partial ×3 | groks ×3 (4,100–5,100) | groks, groks, partial (16,000; 8,200; none) | TA: groks once (a single eval at 600), not grokked ×2. D: not grokked ×3 | partial ×3 (s0 by 0.002) |
| **Late level L** | 1.000 | 0.971–0.976 | 0.999 | 0.991 / 0.996 / 0.982 | plain 0.975–0.977 / 0.978–0.984 | 0.052–0.086 |
| **Late evals ≥ 0.99 (last 5k)** | 100 % | 0 % | 100 % | 41 % / 100 % / 0 % | 0 % | 0 % |
| **β, 10k–25k** | at ceiling | **2.24–2.39** (se ≤ 0.16) | at ceiling | 1.34 / 0.77 / 1.25 (se 0.13–0.23) | −0.25 to −0.08 / −0.23 to 0.21 | **1.11–1.64** (se ≤ 0.08); D_cubeab −0.05 to 0.00 |
| **t50** | 700–900 | **7,200–7,500** | 3,300–4,300 | 1,300–1,900 | 100 | not defined (still near chance) |
| **Width, decades** | 0.23–0.30 | 0.42–0.47 | **0.09–0.12** | **0.85–1.04** | — | — |
| **Delay ratio** | 3.5–4.5 | **12.0–12.5** | 3.7–4.8 | 2.6–3.8 | **0.1–0.2** | — |
| **Loss hump** | 1.17–1.39 | 2.11–2.25 | 2.10–2.18 | 1.15–1.23 | 1.00 (no hump) | 2.95–3.02, peak at 9.2–10.2k, now falling |
| **Balanced accuracy** | 1.000 | 0.960–0.971 | 0.999 | 0.982–0.994 | **0.81–0.90 / 0.71–0.72** | 0.049–0.085 |
| **Kappa** | 1.000 | 0.958–0.969 | 0.999 | 0.980–0.994 | 0.960–0.967 / 0.962–0.975 | **0.046–0.079** (D: 0.003–0.006) |
| **New readout** [post hoc] | converged; delayed, sharp | rising ×3; delayed, gradual | converged; delayed, sharp, late | rising / converged / rising; delayed, very gradual | plain accuracy at a plateau (β ≈ 0); minority classes fail; read by balanced accuracy | rising from near chance; above chance by kappa |

TA_sqab is the same story as TA_cubeab, one level down. The rule reads partial / memorises / memorises. β is 1.01 / 1.02 / 0.89 (se 0.06–0.09), rising in all three seeds. D_sqab's β is 0.07–0.12, flat. Kappa adds a check that matters: TA_sqab_s2 rises from 0.013 to 0.024, which only reaches the majority baseline of 0.025. Its kappa is 0.016, against 0.054 and 0.026 for the other seeds. So s2 may be learning the answer frequencies rather than the function. [data]

### 4.3 Reading the table

- **Sharp grokking against slow climbing (TA_mul against D_mul).**
  - The rule's only distinction is groks against partial.
  - The new numbers say *how* they differ. D_mul starts about 10× later (t50 7,200–7,500 against 700–900). Its transition is about 1.6× wider on log time (0.42–0.47 against 0.23–0.30 decades). It then has a long tail in which the error falls as roughly (update)^−2.3.
  - D_mul's memorisation gap is the largest in the study (delay ratio about 12). So D_mul is the clearest *grokking* run in phase 1, and the 0.99 rule is the one that misses it.
- **Late sharp jump (D_div).** Width 0.09–0.12 decades: the narrowest transition in the study. It is not "slow"; it is delayed. The rule calls D_div and TA_mul both "groks" and cannot tell a 4,000-update delay from a 700-update one. Only its timing number can, and that number depends on the line.
- **Near the line (D_sq).** No level rule removes the problem, because D_sq really does sit at 0.98–0.996. The new readout reports what is there: a very gradual transition (0.85–1.04 decades, the widest of any run that converges), still rising in 2 of 3 seeds, at 41 %, 100 % and 0 % of late evals ≥ 0.99. Then "is D_sq disadvantaged by digits?" becomes a comparison of levels and widths, not a coin toss at 0.99. Truncation shows the rule's fragility. At a 15k horizon, D_sq_s0 would read "partial": its first crossing came at 16,000.
- **Skewed task (gcd).** Curve metrics on plain accuracy see only a plateau near 0.98. The real failure is in the small classes. The worst classes of gcd 16, 19 and 23 have 6–12 examples and recall of 0–0.5. Balanced accuracy and macro-F1 (TA 0.77–0.84, D 0.72–0.75) carry the signal. Kappa does not. Separately, TA_gcd's held-out loss rises in the late phase (log-slope +0.01 to +0.055 per 1k updates). That is mild overfitting, which also explains why TA_gcd_s0's one 0.990 eval at update 500–600 never returned. Per-eval balanced accuracy is not logged, so the time course cannot be checked without checkpoints.
- **Creep (TA_cubeab).** β separates TA (1.1–1.6, about 15–40 se) from D (about 0) cleanly, and kappa agrees. Projecting a 0.99 crossing from 5 % is meaningless (the fits give over 10⁶ updates), because grokking onsets are not power laws. Read TA_cubeab as "onset not reached by 25k; held-out loss past its peak and falling". The loss-peak sign predicts an accuracy onset next. That is a testable prediction if cubeab is ever run to 100k.

**Prediction to register before the D_mul 100k rerun** [data, model-dependent]:
- **β at 25k projects the first 0.99 crossing at 37.7k / 42.6k / 35.5k** (seeds 0, 1, 2).
- **The same fit at 15k projected 24–28k and was wrong (too early).** So read 36–43k as optimistic.
- **The log-time sigmoid predicts a plateau at 0.97–0.98 and no crossing.**
- **What the outcomes would mean:**
  - a crossing by about 60k means the tail is a slow power law;
  - a level stuck under 0.98 at 100k means a real plateau, not slow grokking.

### 4.4 How reproducible each timing is across seeds

Pooled across-seed sd of ln(time) on the 10 cells where every measure is defined (TA/D add, TA_mul, TA/D div, TA/D sub, TA/D add2b, TA_sq):

| first 0.99 | sustained 0.99 | first 0.95 | sustained 0.95 | t50 | t90 | sigmoid onset | 1 − AUC on log time |
|---|---|---|---|---|---|---|---|
| 0.19 | **0.35** | 0.155 | 0.155 | **0.133** | 0.157 | 0.14 | 0.092 |

Caveat: evals every 100 updates quantise times of 300–2,000, which gives some zero spreads (D_add2b 400 / 400 / 400). A smaller spread helps paired tests only if the seeds measure the same event. t50 does, by construction. AUC mixes level and timing.

### 4.5 Out-of-sample check on E1

E1 is another study, not used to set the cut-offs. 14 of its 15 runs read "converged". s5_s2, which E1 itself flagged ("never sustains 0.99", final 0.9864), reads "rising". It has β = 0.34 ± 0.07, level 0.988, 8 % of late evals ≥ 0.99, delay ratio 5.6 and a loss hump of 1.38. So the readout agrees with the hand reading, and adds that s5_s2 is still improving. E1's other runs have delay ratios of 0.8–2.7 and loss humps of 1.01–1.08: little or no memorisation phase (section 3).

**What is not settled:** the cut-offs (β > 0.3 and 3 se; 90 % of late evals; L ≥ 0.97; width 0.35 decades; delay ratio 2) were chosen after seeing phase 1. Width is unreliable for fast runs: one eval step at t = 500 is 0.08–0.15 decades. It needs evals at least 5× denser than t10. Before these readouts decide a pre-registered reading, fix the cut-offs in a pre-registration and check them on one more study, such as `e2_2026-11-04` or `staged_2026-11-05`.

## 5. The broader metric menu: what each measures, its cost, and how robust it is

"Scalars" means it can be computed from logged numbers. "Checkpoints" means it needs saved weights.

| Metric | What it measures | Cost | Scalars or checkpoints | Robustness | Notes |
|---|---|---|---|---|---|
| Curve summary (late level, β, t50, width, delay ratio, loss hump, AUC) | when, how fast and how completely a run generalises; whether it is still improving | about 1 s per run, offline | scalars | good with evals ≤ 1/5 of the transition time; extrapolation biased optimistic | [data] This is what the 0.99 rule was trying to approximate |
| Held-out loss and margin | continuous progress, including on examples still wrong | already logged in grok_lab | scalars | high; dominated by confidently wrong examples far from chance (TA_sqab's loss is flat while its accuracy creeps up) | [data] Schaeffer et al. 2023: discontinuous metrics such as exact match manufacture apparent sharp jumps [lit, checked] |
| Margins of the wrong examples | how near the errors are to flipping | free from `final_errors.json` | final only | high | [data] Errors in near-line runs are low-confidence (median margin −1.4 to −2.5 nats in D_mul and D_sq). gcd errors are more confident (−3.9 to −4.6, 10th percentile −15). Failing tasks are very confident (−12 to −16) |
| Balanced accuracy, macro-F1, worst-class recall | minority-class failure | trivial | the eval's confusion | small classes are noisy (n = 6 means steps of 1/6); report class sizes | [data] |
| Kappa or chance-corrected accuracy | learning beyond the answer frequencies | trivial | the confusion | blind to minority failure when classes are skewed | [data] |
| Calibration error (ECE) | whether confidence matches accuracy | trivial | the eval | binning-dependent | [lit] Guo et al. 2017 (not re-checked). Little use post-grok, where margins are about 10 nats |
| Weight norm per layer | the scale weight decay controls | logged in grok_lab | scalars | descriptive; not a reliable proxy for generalisation | [lit] Tikeng Notsawo et al., ICML 2025 ([11] in the lit report) |
| Effective rank (Roy and Vetterli) | how many directions a weight matrix uses | one SVD of a 128-d matrix | weights | read at the grok step it overstates the converged value 3–5× | [lit] [66] in the lit report |
| Dormant-unit fraction | MLP units that no longer fire: a unit's mean absolute activation over the layer mean, at or below τ (0 to 0.1) | one forward hook | eval activations | depends on τ; simple | [lit] Sokar et al. 2023, definition checked. The direct probe for 11-06 §3's layer-1 MLP disengagement |
| Fourier or log-Fourier top-5 share | whether the number embedding built the Nanda-style circuit | in grok_lab | weights at eval | harness-only, task-specific (add-like or mul-like) | [lit] Nanda et al. 2023 |
| Restricted and excluded loss | loss with only the key frequencies kept, or with them removed: the circuit forms before the test jump and memorisation is cleaned up after it | a 2-D DFT of the logits over the full table, per checkpoint | checkpoints | needs the key frequencies, found from the trained model; add/mul only | [lit] Nanda et al. 2023, definitions checked |
| Local learning coefficient (LLC) | effective complexity, or loss-landscape degeneracy | a sampling run per checkpoint (SGLD-style), with step size and localisation to tune | checkpoints | "suggestive" stage detection in small transformers | [lit] Lau et al. 2023; Hoogland et al. 2024 (abstracts checked; the sampler detail is from memory) |
| CKA or representation similarity | whether two runs or two tasks share representations | activations on a probe batch, per checkpoint pair | checkpoints | good for "same or different", poor at saying why | [lit] Kornblith et al. 2019 (not re-checked). Possible use: multitask interference (E2's add+mul) |
| Gradient noise scale | the batch size above which bigger batches stop helping | cheap from the micro-batch gradients train.py already accumulates | gradients | noisy, needs smoothing | [lit] McCandlish et al. 2018 (not re-checked) |
| Per-task gradient cosine and norm | conflict and starvation between tasks | one extra backward per task per eval | eval | already in grok_lab | [lit] + design doc |
| Spectral early warning | low-frequency oscillation in the early learning curve predicts later grokking | an FFT of `train_obj_trace` | scalars (dense) | unvalidated here | [lit] Notsawo et al. 2023, abstract checked |
| Memorisation score | how much an example's own presence decides its correctness | about 10³ models trained on subsamples | many runs | robust but very costly | [lit] Feldman and Zhang 2020 (abstract checked; the run count is from memory). The finite split already gives this contrast exactly |
| Ungrokking and semi-grokking checks | a generalised run losing test accuracy; late generalisation that stops at partial accuracy | covered by late level + β + share of late evals ≥ 0.99 | scalars | — | [lit] Varma et al. 2023, checked. D_sq, D_mul and gcd fit the semi-grokking shape at 25k; whether they stay there is open |

## 6. What helps train.py too: ranked by value against cost

train.py logs `eval/acc/{exact_match,token,prefix,distance}`, per-task accuracy and MAE, `eval/loss/ce`, perplexity, attention and weight entropy, gradient norm, and an update norm that is still the SGD estimate lr·‖g‖ (#149). It has a format-crash warning: exact match drops while token accuracy holds.

1. **The offline curve-summary tool** (rec. 1). It reads `metrics.json`, which has per-task accuracy at each eval, and prints, per task: late level, β, t50, width, delay ratio, loss hump and projected crossing. No training cost. It helps:
   - lexical learning, where curves are gradual and exact match is threshold-like;
   - multitask runs: "which task stopped improving, and when";
   - curriculum runs: a task's t50 measured from its own unlock.

   Value high, cost S.
2. **Per-task held-out answer loss** (teacher-forced, over the answer tokens), next to exact match. This is the continuous partner that Schaeffer et al. argue for, and it shows progress before exact match moves. In phase 1, held-out loss leads accuracy by 200–2,500 updates. Value high, cost S.
3. **The resume-equivalence check.** Run N updates. Separately, run N/2, save, resume, and run to N. Report the first update at which the loss traces differ, the largest gap, and whether the parameter hashes match at N. Same-seed runs on the bridge path are already bit-identical (11-06 §1), so anything short of identity is a bug signal. Today the bridge mode refuses to resume. This turns "resume correctness" into a pass/fail number. Value high (resume is untested), cost S–M.
4. **Per-task balanced accuracy and worst-bucket accuracy.** For digit answers, bucket by answer length or first digit. This catches the DAG-v12 failure mode, where 98 % of examples went to four easy tasks, and any task whose aggregate accuracy hides a failing subgroup. Value medium-high, cost S.
5. **Dormant fraction, per-layer weight norm and measured update ratio.** This is the instrument 11-06 §3 asked for to see whether train.py's MLP disengages. Fixing #149 belongs here. Value medium-high, cost S–M.
6. **Calibration error on lexical evals.** Cheap. It tells overconfident memorisation apart from honest uncertainty. Value medium, cost S.
7. **Gradient noise scale from the micro-batches.** It informs batch size and accumulation choices for LM runs. Value medium, cost S–M.
8. **CKA between checkpoints or tasks.** Useful for multitask interference studies, not for routine runs. Value medium, cost M.
9. **LLC, restricted and excluded loss, memorisation scores.** Research tools with no current train.py decision behind them. Value low now, cost M–L.

## 7. Proposed replacement for the run-state rule

To pre-register before use. The cut-offs below are the post hoc ones from this analysis.

- **Level-and-trend state**, from the last 1,000 updates (level) and the last 60 % of the run (β):
  - **converged:** L ≥ 0.99 and at least 90 % of the last 5,000 updates' evals ≥ 0.99;
  - **rising:** β > max(3 se, 0.3). Report the projected crossing, flagged as optimistic;
  - **near-line plateau:** L ≥ 0.97 and not rising;
  - **flat at chance:** chance-corrected L < 0.05, kappa < 0.01, and not rising;
  - **stalled partial:** anything else.
- **Shape**, when chance-corrected L > 0.2:
  - delayed (delay ratio ≥ 2) or concurrent;
  - sharp (width < 0.35 decades) or gradual;
  - plus t50 as the timing.
- **"Grokked"** = converged and delayed. **"Generalised directly"** = converged and concurrent.
- **Skewed tasks** (majority baseline > 0.2): run the same state rule on balanced accuracy logged per eval, and report worst-class recall with class sizes.
- **Keep** the first 0.99 crossing and the optimisation-failure state for continuity.

## 8. Limits of this analysis

- **Every cut-off in sections 4 and 7 is post hoc.** E1 is the only out-of-sample check, and it contains almost no hard cases.
- **The OLS standard errors on β ignore the remaining autocorrelation**, even with thinning to one eval per 500 updates. Treat "3 se" as a rough screen.
- **Per-eval balanced accuracy and per-class curves could not be computed** without loading checkpoints, which this task forbade. So gcd's time course is unknown.
- **The literature claims above were checked against abstracts or HTML pages**, not full PDFs, except where noted. The sampler detail for LLC, the run count for memorisation scores, and the claims attributed to Guo, Kornblith, McCandlish and Roy–Vetterli are from memory and were not re-checked.

## References

Checked means the page was opened on 2026-11-07 and the claim used here confirmed.

- Power, Burda, Edwards, Babuschkin, Misra (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* https://arxiv.org/abs/2201.02177. Checked: abstract (generalisation long after overfitting).
- Nanda, Chan, Lieberum, Smith, Steinhardt (2023). *Progress measures for grokking via mechanistic interpretability.* https://arxiv.org/abs/2301.05217. Checked: HTML (restricted and excluded loss, key frequencies found post hoc, the memorisation / circuit formation / cleanup phases).
- Schaeffer, Miranda, Koyejo (2023). *Are Emergent Abilities of Large Language Models a Mirage?* https://arxiv.org/abs/2304.15004. Checked: abstract (nonlinear or discontinuous metrics produce apparent emergence).
- Varma, Shah, Kenton, Kramár, Kumar (2023). *Explaining grokking through circuit efficiency.* https://arxiv.org/abs/2309.02390. Checked: abstract (ungrokking, semi-grokking).
- Lau, Furman, Wang, Murfet, Wei (2023). *The Local Learning Coefficient: A Singularity-Aware Complexity Measure.* https://arxiv.org/abs/2308.12108. Checked: abstract.
- Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet (2024). *Loss Landscape Degeneracy and Stagewise Development in Transformers.* https://arxiv.org/abs/2402.02364. Checked: abstract ("suggestive evidence").
- Kumar, Bordelon, Gershman, Pehlevan (2023). *Grokking as the Transition from Lazy to Rich Training Dynamics.* https://arxiv.org/abs/2310.06110. Checked: abstract.
- Tikeng Notsawo, Zhou, Pezeshki, Rish, Dumas (2023). *Predicting Grokking Long Before it Happens.* https://arxiv.org/abs/2306.13253. Checked: abstract.
- Sokar, Agarwal, Castro, Evci (2023). *The Dormant Neuron Phenomenon in Deep Reinforcement Learning.* https://arxiv.org/abs/2302.12902. Checked: HTML (the τ-dormant definition and the τ values).
- Feldman, Zhang (2020). *What Neural Networks Memorize and Why.* https://arxiv.org/abs/2008.03703. Checked: abstract only.
- Not re-checked, cited from memory:
  - Guo et al. (2017), *On Calibration of Modern Neural Networks*, https://arxiv.org/abs/1706.04599;
  - Kornblith et al. (2019), *Similarity of Neural Network Representations Revisited*, https://arxiv.org/abs/1905.00414;
  - McCandlish et al. (2018), *An Empirical Model of Large-Batch Training*, https://arxiv.org/abs/1812.06162;
  - Roy and Vetterli (2007), *The effective rank*, EUSIPCO.
- From the project's earlier sweep (`grokking-lit-2026-10-28/report.md`, checked there): [11] Tikeng Notsawo et al., ICML 2025 (the ℓ2 norm is not a reliable proxy); [64] commutator defect; [65] LLC; [66] Truong, at-grok measurement audit; [67] Ngnawé and Notsawo, margin-based progress measures.
