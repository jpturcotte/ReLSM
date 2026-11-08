# Grokking metrics review: consolidated report (2026-11-07)

**Question (owner):** we did research before on grokking signs or criteria. Is it still relevant? Are there better metrics? Which would help the project in general, not just for grokking?

**What this report is.** One place for the final answer, after three rounds of work in this folder:
1. `analysis.md`: the original analysis of the 48 phase 1 runs, the 6 reused add runs and the 15 E1 runs. Its script is `curve_metrics.py`; its output is `curve_results.json`.
2. `recheck_2026-11-07.md`: an independent recheck. Every number was recomputed with separate code, every literature claim was checked against its source, and the 2025 to November 2026 literature was swept.
3. Two red teams on both documents: a Fable agent with full numerical access, and Codex, which reviewed code and sources. Their 32 findings and what was done with each are in section 8 of the recheck.

Where this report and `analysis.md` disagree, this report has the checked value. `analysis.md` is left as written; the recheck lists the edits it needs.

**Labels.** **[post hoc]**: a cut-off chosen after seeing these runs. It must be pre-registered before it decides anything. **[env]**: holds for this machine and software environment.

## The answer in brief

- **The earlier research still stands as logging advice.** Keep held-out loss and margin next to accuracy, the transpose-seen and diagonal splits, no early stopping, and weight norm as a description, never a trigger.
- **Its decision rules need replacing.** The 0.99 first-crossing rule measures "generalised by the horizon", not grokking. It never checks that generalisation came after memorisation. 19 of the 33 runs it scores "groks" show no sign of a memorisation phase by the halfway test; 8 of those 19 still take at least 2× Tmem to reach 0.99 (corrected 2026-11-08, misread audit).
- **Better metrics exist and are cheap:**
  - a threshold-free curve summary: late level, trend, timing, width and delay;
  - balanced accuracy for skewed tasks;
  - held-out loss logged on the same examples as the accuracy;
  - for train.py, per-task answer loss and a resume-equivalence test.

  None of them costs training time beyond one eval pass.
- **Treat the rest as research tools with no decision waiting on them:** LLC, CKA, restricted and excluded loss, and memorisation scores.

## 1. Do now

Each is cheap and helps the runs in flight. S means under a day and under about 150 lines.

| # | Action | Why | Cost |
|---|---|---|---|
| 1 | **Apply the four meaning-changing corrections to `analysis.md`** (recheck corrections 1, 2, 4, 4b and 4c) before anyone cites it. | Section 4 below | 15 min |
| 2 | **grok_lab eval: log per-task balanced accuracy, worst-class recall with class sizes, and the predicted-class histogram (or kappa) at every eval.** | gcd's ranking flips under balanced accuracy. "Flat at chance" needs kappa, which today exists only at the final eval. | S |
| 3 | **grok_lab eval: log held-out loss on the accuracy's own examples** (unseen, off-diagonal). | Today's `test_loss_unseen` includes the diagonals (`grok_lab.py` line 534), so every loss-against-accuracy comparison mixes two sets. | S |
| 4 | **Use the curve summary as a description only, after three fixes:** <br>(i) width 0 reads as "missing" in `classify()`, so TA_gcd is called "gradual"; <br>(ii) print the eval spacing next to every timing, and mark a value censored when Tmem or t10 is under 5 eval intervals; <br>(iii) give the loss-peak flag a minimum drop, for example 5 % below the peak. | Delay ratio and loss hump both depend on eval spacing. | S |
| 5 | **train.py: add per-task teacher-forced answer loss next to exact match.** | Only aggregate `eval/loss/ce` exists. It is the cheapest continuous partner for exact match. | S |
| 6 | **Register the D_mul 100k prediction with both fits before the run, scored against the sustained crossing.** | Full-window β projects 37.7k / 42.6k / 35.5k. Late-window β projects 41.4k / 42.9k / 70.5k, because β has fallen in 2 of 3 seeds. | 0 |
| 7 | **State the scope of every verdict, and add the code hash to `run_record.json`** (its `commit` is null). | Same-seed reruns are exact here [env], but environment changes flip a minority of same-seed outcomes (Ootani 2026). | 0 + S |
| 8 | **Name the definitions you quote.** grok_lab's margin is z_t − logsumexp over the other classes. State macro-F1's label set too. | For D_gcd_s0, macro-F1 is 0.732 over true labels only and 0.565 over true and predicted labels. | 0 |

## 2. Do going forward

- **Pre-register the readout's cut-offs and test them out of sample**, on `e2_2026-11-04` or `staged_2026-11-05`. The cut-offs: β > max(3 se, 0.3), 90 % of late evals, L ≥ 0.97, width 0.35, delay 2. E1, the only check so far, has almost no hard cases.
- **Define "grokked" by a gap the evals can resolve.** Use delay ratio ≥ 2 only where Tmem is at least 5 eval intervals, and report it as censored otherwise. Pair it with the loss hump, measured on the accuracy's own examples and with evals dense enough to catch the peak. A hump cut-off of 1.1 agrees with the delay ratio in all 42 runs where both are defined [post hoc].
- **A resume-equivalence test for train.py, the highest-value train.py item**, built to the Megatron Core spec. Fingerprint loss and gradient norm every step, and find the first step that differs. After a restore, byte-compare parameters, optimizer state, every RNG stream and the data position, including the sampler's own RNG. Run it on CPU with pinned threads. Bridge mode refuses to resume today.
- **Dormant-unit fraction per MLP layer, normalised to the layer mean with τ stated, plus the measured update ratio (#149).** train.py already logs per-module weight norms and grad/weight ratios for the embedding, head and three blocks, so this extends them.
- **Readout flags:**
  - "censored", for runs still moving at the horizon. At 25k that is D_mul, D_sq_s0 and s2, TA_sqab and TA_cubeab.
  - "plateau verified", before comparing a representation metric across runs.
  - "declining", for a late trend that is significantly negative (anti-grokking: Prakash and Martin 2026).
- **A hindcast check inside the curve tool:** fit at an earlier horizon, then score the projection against the sustained crossing. Report projections only for task families whose hindcasts land within ×1.5.
- **Optional:**
  - train.py health scalars: the OLMo 2 spike score from logged values (free); max attention logit, output log-Z and per-layer cos(Δθ, θ) as new instrumentation;
  - a time difference with its η·λ next to the delay ratio (Truong et al. 2026);
  - ILDR as a raw value, not a trigger.

## 3. Retire as decision rules

- **The 0.99 first crossing as the only run state.** Keep it as a secondary, backward-compatible number.
- **Sustained ≥ 0.99 as a primary timing.** It has the worst seed spread under every exclusion rule tried (pooled log-sd 0.35; 0.26 without the floor cells). It turns near-line wobble into 10× delays: D_add's sustained times are 21,600 / 4,200 / 12,800 against first crossings of 1,300 / 900 / 1,600.
- **"Memorises" = late median ≤ 0.05.** TA_sqab's three seeds share one shape but read partial / memorises / memorises (0.054, 0.033, 0.023).
- **Plain accuracy for skewed tasks.**
- **Point forecasts of a crossing time as decision inputs.** Register them and score them; don't act on them.
- **"t50 is the most reproducible timing".** Its edge comes from cells at the eval floor (see 4.4). Report 1 − AUC on log time and the first 0.95 crossing next to it.
- **Held-out loss as an early-warning sign.** Keep it as a level-and-shape measure. Re-admit it as a warning only after a pre-registered test on the accuracy's own examples, with a drop-size rule.
- **The Fourier top-5 share as a progress measure.** Keep it as a descriptor for the TA arm only.
- **Margin-cliff fractions as a failure sign, and weight norm as a trigger.** These were already retired in earlier docs, and nothing revives them.
- **Unscoped per-seed claims.** A claim meant to hold across machines or thread counts must be a seed rate.

**Not retired:** LLC stays "not now". Its estimates shift by orders of magnitude with sampler settings, but at fixed settings its trajectories are informative (Cullen et al. 2026). Use it only for a concrete question, with fixed settings and sampler diagnostics.

## 4. What the runs show (checked numbers)

All accuracy numbers are held-out, unseen, off-diagonal, from one eval every 100 updates to 25,000. "Seed spread" is the pooled log-sd of a timing across the three seeds of each cell.

### 4.1 Most of the "groks" are not grokking

- **19 of the 33 runs that the 0.99 rule scores "groks" show no memorisation phase** by the halfway test (held-out accuracy at half its final level within 2× Tmem). 8 of the 19 still take at least 2× Tmem to reach 0.99 (TA_sq 8.5–10.5×, D_add 2.25–4×), so on a 0.99 delay it is about 11 of 33 (corrected 2026-11-08, misread audit). The 33 are phase 1 plus the reused add runs. Examples: D_add, D_sub, D_add2b, TA_div, TA_sub and TA_sq.
- **The evidence is the loss hump.** Peak held-out loss over the starting loss is 1.00 in 13 of the 19, and 1.04–1.08 in TA_div and TA_sub.
- **The delay ratio can't carry it on its own.** In 11 of the 19, training accuracy reaches 0.99 within 500 updates, so the ratio is censored. That loss also includes the diagonals.
- **D_mul is the clearest delayed-generalisation run.** It is still rising at 25k.

### 4.2 The states lump unlike runs together

- "Partial" covers both D_mul, at 0.97 and climbing, and TA_cubeab, at 0.05–0.09 and creeping up from chance. TA_cubeab_s0 is "partial" by 0.002.
- A level-and-trend readout separates all five problem cases:
  - late median over the last 1k updates;
  - β, the slope of logit(acc) against ln(update) over 10–25k;
  - t50, the time to half of the run's own final level;
  - width, t10 to t90 in decades;
  - delay ratio;
  - loss hump.

  Its cut-offs are [post hoc].
- **The "rising" screen (β above 3 se) is robust to a simple autocorrelation check.** With AR(1)-inflated standard errors, every "rising" call stays above 3 se, the lowest at 3.3 (D_sq_s1). This is a heuristic, not a HAC or bootstrap estimate.
- **TA_cubeab is the "creeping from chance" case.** Its β sits 20–27 se above zero.

### 4.3 Skewed tasks need balanced accuracy

For gcd, the plain and balanced metrics disagree:

| gcd, three seeds | TA | D | Ranking |
|---|---|---|---|
| Balanced accuracy | 0.81–0.90 | 0.71–0.72 | TA ahead, 3 of 3 seeds |
| Plain accuracy, final eval | | | D ahead, 2 of 3 (s1: TA 0.9800, D 0.9776) |
| Plain accuracy, late median | | | D ahead, 3 of 3 |
| Kappa | 0.96–0.975 | 0.96–0.975 | hides the gap |

The balanced-accuracy gap (0.09 / 0.19 / 0.15) comes almost entirely from classes with 12 or fewer held-out examples (100–128 of about 4,560 per seed); the larger classes contribute 0.000–0.020. D makes fewer total errors than TA in s0 and s2. So "TA ahead 3 of 3" stands, but the cost of digits here is about 20 rare-gcd examples, and one example in a 2-example class moves balanced accuracy by about 0.013: report class sizes with it (corrected 2026-11-08, misread audit).

- **Wrong examples are confidently wrong.** Their median margins are −2.9 to −6.2 nats, with 10th percentiles from −8 to −18.
- **Near chance, kappa is the useful one.** It separates TA_cubeab (0.046–0.079) from D_cubeab (0.003–0.006).

### 4.4 Timing: what's reproducible

Pooled log-sd of timing across seeds, over the 10 cells (arm × task) where every timing is defined in all three seeds:

| Timing | All 10 cells | Without the 5 floor cells | Without the 6 cells with zero spread in any measure |
|---|---|---|---|
| t50 | 0.133 | 0.173 | 0.188 |
| first 0.99 crossing | 0.190 | 0.155 | 0.141 |
| first 0.95 crossing | 0.155 | 0.135 | 0.138 |
| t90 | 0.157 | 0.139 | 0.155 |
| sustained 0.99 | 0.348 | 0.262 | 0.274 |
| 1 − AUC on log time | 0.092 | 0.083 | 0.082 |

- The 5 floor cells are 3 with zero spread (D_add, D_add2b, TA_sq) and 2 near the floor (D_sub, TA_div). The third column also drops TA_sub.
- **t50's apparent lead depends on those cells.** Sustained 0.99 is worst and 1 − AUC lowest under every rule.
- Lower spread is not the same as more power to detect an arm effect, so compare paired effect sizes before choosing a primary timing.

### 4.5 Projections run early

- **The β-line forecast of the 0.99 crossing has been optimistic in all six runs where it could be checked** (D_mul and D_sq, three seeds each), scored against the sustained crossing.
- **D_mul.** Forecasts made at 8k–12k (22k–25k) were wrong in all three seeds. At 15k, s2's 24.0k forecast was wrong; s0's 27.5k and s1's 28.3k lie past the horizon and are censored.
- **A sigmoid in log time disagrees.** It says D_mul levels off at 0.97–0.98. Only the 100k rerun can settle it.

### 4.6 The loss peak is not yet a warning sign

- **The peak-then-fall pattern appears in 5 of 6 flat D runs**, so on its own it doesn't mark progress.
- **A drop-size condition separates one pair but not another.** TA_cubeab falls 9.5–17.7 % below its peak and D_cubeab 0.7–0.9 %, which separates them. TA_sqab (2–4 %) and D_sqab (1–2 %) stay mixed.
- **The analysis's "loss leads accuracy by 200–2,500 updates in 18 of 18 runs" is a mixed-set description** [post hoc]. It needs re-testing with loss on the accuracy's own examples.

### 4.7 Eval spacing changes the verdicts

- **In TA runs, Tmem is 200 updates, two evals.** At one eval per 500 updates, TA_mul's delay ratio reads 2.0, flipping it to "delayed".
- **The hump vanishes too.** TA_mul_s0's loss peak of 1.17× at update 300 reads 1.00 at that spacing.
- **D_div and D_sq are stable** under the coarser spacing.
- **The project's actual spacing:**
  - grok_lab and train.py's table scorer evaluate every 100 updates;
  - train.py's LM eval runs every 500 micro-batches, which is 125 optimizer updates at 4× accumulation.

### 4.8 Two project facts

- **train.py already logs per-module weight norms and grad/weight ratios**: `train/weight/norm/{key}` and `train/grad/ratio/{key}` for 5 modules. Still missing: per-MLP norms, a dormant fraction and the measured update ratio (#149).
- **Phase 1 logs Fourier and log-Fourier top-5 shares, but only for the TA arm.** The share leads accuracy in 2 runs and lags in 2, so it can't be a cross-arm progress measure.

## 5. Proposed run-state rule (to pre-register)

This is `analysis.md` §7 with the recheck's amendments. All cut-offs are [post hoc].

- **Level-and-trend state**, from the last 1,000 updates (level L) and the last 60 % of the run (β):
  - **converged:** L ≥ 0.99, and at least 90 % of the evals in the last 5,000 updates are ≥ 0.99;
  - **rising:** β > max(3 se, 0.3). Report the projected crossing, flagged optimistic and scored later against the sustained crossing;
  - **declining** *(new)*: a significantly negative late trend;
  - **near-line plateau:** L ≥ 0.97 and not rising;
  - **flat at chance:** chance-corrected L < 0.05, per-eval kappa < 0.01 (needs do-now item 2), and not rising;
  - **stalled partial:** anything else.
- **Shape**, when chance-corrected L > 0.2:
  - delayed (delay ratio ≥ 2) or concurrent; censored when Tmem is under 5 eval intervals;
  - sharp (width < 0.35 decades) or gradual; a width of 0 is sharp;
  - t50 as the timing, with 1 − AUC and the first 0.95 crossing next to it.
- **"Grokked"** = converged and delayed, with a loss hump on the accuracy's own examples. **"Generalised directly"** = converged and concurrent.
- **Skewed tasks** (majority baseline > 0.2): run the same rule on balanced accuracy, and report worst-class recall with class sizes.
- **Keep** the first 0.99 crossing and the optimisation-failure state for continuity.
- **Known gap:** "rising" is computed on raw accuracy, so a run climbing only to the majority baseline reads "rising". TA_sqab_s2 does this: 0.023 against a baseline of 0.025. Per-eval kappa is the fix.

## 6. Corrections to `analysis.md`

The recheck lists 26 numbered items. These change meaning:
1. **gcd plain accuracy:** D above TA in 2 of 3 seeds at the final eval, not 3 of 3. It is 3 of 3 only on the late median. The balanced-accuracy reversal stands.
2. **"Held-out loss never rises above its start"** in the 19 runs: true for 13. TA_div and TA_sub reach 1.04–1.08.
3. **Loss and accuracy use different examples** (diagonals in the loss only). Every loss-against-accuracy statement needs that label, or a recompute.
4. **t50's spread of 0.133** rests on floor cells (see 4.4).

"Biased optimistic" in §1.6 stands, and is stronger than the analysis states (see 4.5). The other corrections are range edits, a citation fix and scope notes.

## 7. Literature

- **The "from memory" claims hold in substance:** Guo, Kornblith, McCandlish, Roy–Vetterli, the LLC sampler and Feldman–Zhang. Each was checked against the full PDF. Corrected:
  - Feldman–Zhang trained 2,000–4,000 models on 70 % subsets, not "about 10³";
  - Varma's ungrokking needs further training on a subset below the critical size;
  - Furman and Lau 2024 is withdrawn into Lau et al.;
  - Guo states the strong binning sensitivity for MCE, not ECE.
- **New work that changes a recommendation:**
  - Ootani 2026 (arXiv 2607.05104): the numerical environment flips a minority of same-seed outcomes. Hence the environment scope in do-now item 7.
  - The NVIDIA Megatron Core determinism post (Oct 2026): a concrete resume-equivalence spec. It warns that printed losses "are not sensitive enough".
  - Truong 2026 (arXiv 2607.06639): don't read a representation metric at the transition as converged, and flag censored runs.
- **Most of the rest supports the direction here:** threshold-free level-and-shape summaries, margin and loss next to exact match, and LLC left for later. That includes Cullen et al. 2026, Prakash and Martin 2026, Kataria 2026 and Gning and Dohmatob 2026. Section 6 of the recheck has the full list.

## 8. How sure this is

- **Checked twice:**
  - Every number in sections 1–4 above was recomputed by the recheck. Fable recomputed all 54 runs independently and matched.
  - Codex's sandbox blocked Python, so its numerical points rest on code reading, which I then verified.
  - Several of the recheck's own first claims were withdrawn after the red teams: the hindcast verdict, "same-seed reruns are inexact", retiring LLC, a β < 1 plateau rule, and a "spacing-free" loss hump.
- **Settled here:**
  - The 0.99 first-crossing rule doesn't measure grokking.
  - gcd needs balanced accuracy.
  - Sustained 0.99 has the worst seed spread.
  - D_mul is the clearest delayed run.
  - β-line forecasts run early.
- **Not settled:**
  - Every cut-off (pre-register and test out of sample).
  - Whether D_mul crosses 0.99 (100k run).
  - Every loss-against-accuracy result (needs loss on the accuracy's own examples).
  - The β standard errors (a HAC or bootstrap check is still to do).
  - Whether any of this transfers to train.py's eval spacing.
  - gcd's time course (needs per-eval balanced accuracy).

## Files in this folder

| File | What it is |
|---|---|
| `report.md` | this consolidated report |
| `dmul_100k_prediction.md` | the registered D_mul 100k forecasts and how they are scored |
| `analysis.md` | the original analysis; unedited except a pointer line at the top |
| `curve_metrics.py`, `curve_results.json` | the analysis's script and output |
| `progress.md` | the analysis's step notes |
| `recheck_2026-11-07.md` | the full recheck: number-by-number audit, literature audit, new literature, the three lists, 26 corrections, red-team table |
| `recheck_progress.md` | the recheck's dated step log |
