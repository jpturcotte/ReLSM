# Grok metrics review: progress notes (2026-11-07)

Running log. Final deliverable: analysis.md in this folder.

## Step 0: setup
- numpy 2.2.6 available. No torch used.

## Step 1: earlier research read (lit report, fix plan, consolidation, multitask design)
- Lit report grokking-time definitions: first >50/90/95/99.5/100 %; it recommended "several thresholds plus a sustained one (stays >= 95 %)". Logging recs: log-axis loss, held-out loss + mean correct-class confidence ("move before accuracy", [20]), margin (mean/min/frac past 16.6), zero-target-grad fraction, Adam v, ||dθ||/||θ||, per-layer weight norm "descriptive, never a trigger". Leads only: linear-probe R2, Fourier power, logit scale, LLC [65], commutator defect [64], train margin [67]; eff. rank at grok step overstates converged value 3-5x [66].
- Consolidation 11-06: bf16 cliff CPU-only; embedding decay not needed; sustained crossing used as E2 primary (lower variance: pooled log-sd 0.16 vs 0.20 first). 3 seeds pin a ratio only to x/÷1.5-1.9.
- Multitask doc: first/sustained 0.99 crossings, unseen off-diagonal; diagonals separate; min_test_acc_after_first.
## Step 2: phase 1 data
- metrics.jsonl: eval every 100 updates, 251 rows to 25k. Keys include test_loss_unseen, test_margin_mean/p5_unseen_offdiag, train margins, norms (table/matrices/head/gains/final_gain), rel_update, adam_v, grad_norm, spikes, train_obj_trace.
- score.json states: D_mul 3x partial (final 0.958/0.970/0.968); D_sq groks/groks/partial (first 16000/8200/None, sust 25000/12300); D_div groks first 5100/4100/4400; TA_gcd s0 "groks" at 600 on plain accuracy (majority 0.61) but s1,s2 not; balanced acc TA 0.81/0.90/0.87, D 0.72/0.71/0.72. D_add sustained 21600 vs first 1300 (late dip).
- grok_lab already logs: per-group margins (mean, p5), fp64 test loss, unseen loss, norms per group, Adam v, measured rel_update, spikes; emb Fourier/log-Fourier top5 share exists (diagnostics flag, not in phase-1 rows). summarize() = first crossing 0.99 + frac of last-10% evals >= 0.99.
- train.py logs: eval/acc/{exact_match,token,prefix,distance}, per-task accuracy + MAE, loss/ce, ppl, attention entropy, weight entropy, grad norm; update norm is still the SGD estimate lr*grad (#149). Format-crash warning: exact match drop >= threshold with stable token acc.
- final_errors.json has (a,b,answer,pred,margin) for wrong rows; with split_counts hist we can rebuild the full confusion -> balanced acc, macro-F1, kappa for every task at the final eval.
- add runs live in grok_runs/format_2026-11-05/runs/{TA,D}_s0-2 (key @add).

## Step 3: first pass of threshold-free metrics (curve_metrics.py, 54 runs incl. add from format_2026-11-05)
- Sampled curves: TA_sq and TA_gcd generalise *with* memorisation (TA_sq unseen 0.875 at u200 when train=1.0; TA_gcd 0.98 at 200) -> "groks" label covers runs with no delay at all.
- TA_gcd s0 "groks" = one eval at 0.990 (u500) then slow decline to 0.976; test loss rises 0.06 -> 0.15-0.19 (mild overfitting). Threshold rule rewarded a transient.
- D_mul: classic memorise-then-generalise; test loss hump 4.6 -> 10.2 at u2000, then falls. t50 7.2-7.5k, t90 11.9-12.5k, width 0.42-0.47 decades (TA_mul 0.23-0.30): on log time only ~1.5x wider; difference is onset (10x) and a power-law tail (error ~ u^-2.1..2.3). Projected 1 % error at 37-42k (power law) / 34-62k (logit-linear).
- D_div: sharp (width 0.09-0.12 dec), late (t50 3.3-4.3k), delay ratio 3.7-4.8.
- D_sq: wide (0.85-1.04 dec). s0 crosses 0.99 24x in last 5k (41 % of evals >= 0.99); logit slope s0 +0.076±0.026/k, s1 +0.013±0.034, s2 +0.024±0.022.
- TA_sqab all 3 seeds rising (logit slope 0.044/0.050/0.027 ±0.01) but rule says partial/memorises/memorises (0.054 vs 0.05 line). D_sqab flat. TA_cubeab rising 0.08-0.12±0.005; s0 "partial" by 0.002 over 0.05. D_cubeab flat.
- Delay ratio t50/Tmem: TA_add 3.5-6, TA_mul 3.5-4.5, TA_add2b 4-5, D_div 3.7-4.8, D_sq 2.6-3.8, D_mul 12; ~1 for D_add, TA/D_sub, D_add2b, TA_div; 0.5 TA_sq; 0.1-0.2 gcd.

## Step 4: confusion metrics, taxonomy, truncation, seed spread
- Confusion rebuilt from final_errors + split hist reproduces score.py balanced acc exactly (D_gcd_s0 0.7222).
- gcd: plain acc D 0.978-0.985 > TA 0.976-0.980, but balanced acc TA 0.81-0.90 > D 0.71-0.72 (3/3 reversal). macro-F1 TA 0.77-0.84, D 0.72-0.75. Cohen's kappa 0.96-0.975 for all six: kappa does NOT expose minority-class failure.
- Near chance, kappa helps: TA_sqab 0.054/0.026/0.016, D_sqab 0.002-0.005; TA_cubeab 0.046-0.079, D_cubeab 0.003-0.006. (TA_sqab_s2's rise 0.013->0.024 only reaches the majority baseline 0.025.)
- On balanced tasks balanced acc == acc within 0.002: free to log always.
- Trend beta = slope of logit(acc) vs ln(update), evals thinned to every 500, window 10k-25k: D_mul 2.39±0.16 / 2.24±0.09 / 2.38±0.15 (x99 extrapolated 36-43k); D_sq 1.34/0.77/1.25 (±0.13-0.23); TA_cubeab 1.1-1.6; TA_sqab 0.9-1.0; D_sqab 0.07, D_cubeab -0.04. 5k-window logit-vs-u slope is noisier (D_mul_s2 fails 3 se).
- Extrapolations disagree for D_mul: log-time sigmoid top 0.968-0.983 (never crosses), power law / beta -> 36-43k. Only a longer run settles it.
- Cross-seed pooled log-sd on 10 cells where all are defined: first_0.99 0.19, sustained_0.99 0.35, first/sust_0.95 0.155, t50_own 0.133, t90_own 0.157, sigmoid u0 0.14, (1-AUC_log) 0.092. Sustained-0.99 is worst here (E1 found the reverse: 0.16 vs 0.20) because D_add/TA_add dip near the line (D_add sust 21600 vs first 1300).
- Loss hump (peak unseen loss / init loss): ~2.1-2.25 for D_mul, D_div; 1.1-1.5 TA_add/TA_mul; 1.6-2.0 TA_add2b; 1.00 for D_add, D_sub, D_add2b, TA_sq, gcd (no memorisation phase); ~3 for sqab/cubeab (still memorising).

## Step 5: literature checks (abstract / HTML pages fetched 2026-11-07)
- Checked: Power 2201.02177 (abstract), Nanda 2301.05217 (HTML: restricted/excluded loss need key frequencies, found post hoc from the trained model), Schaeffer 2304.15004 (discontinuous metrics create apparent emergence), Varma 2309.02390 (ungrokking, semi-grokking), Lau 2308.12108 and Hoogland 2402.02364 (LLC; stage detection "suggestive"), Kumar 2310.06110, Notsawo 2306.13253 (spectral signature of early learning curve), Sokar 2302.12902 (tau-dormant: score = mean |act| / layer mean, tau 0-0.1), Feldman-Zhang 2008.03703 (abstract only).
## Step 6: beta trend + truncation + loss-peak lead
- beta added to curve_metrics.py; new taxonomy: D_mul rising x3 (at 10k, 15k, 20k and 25k); D_sq rising/converged/rising; TA_sqab rising x3; TA_cubeab rising x3 (from 15k); D_sqab, D_cubeab flat x3; gcd near-line plateau x6 (on plain acc).
- x99 projected at 15k for D_mul: 24-28k, but no crossing by 25k -> projections run optimistic (curve decelerates). At 25k: 36-43k.
- 19 of 33 rule-"groks" runs have delay ratio < 2 (no memorisation gap): D_add, TA_div, TA_sub, D_sub, D_add2b, TA_sq, TA_gcd_s0.
- Unseen test-loss peak precedes the 10 % accuracy point in 18/18 delayed runs, by 200-2,500 updates (post hoc).
- Results saved: curve_results.json (this folder).

## Step 7: out-of-sample check on E1 (grok_runs/e1_2026-11-02, 15 runs; script in scratchpad e1_check.py)
- 14/15 converged; s5_s2 (E1: "never sustains 0.99", final 0.9864) -> rising, beta 0.34±0.07, level 0.988, 8 % late evals >= 0.99, delay 5.6, hump 1.38.
- E1 delay ratios 0.8-2.7, humps 1.01-1.08 except s5_s2: most E1 "grokking" runs have little or no memorisation gap.
- Width for fast runs is quantised by the 100-update eval spacing (one eval step at t=500 is 0.08-0.15 decades): width needs evals denser than ~t10/5.
- Next: write analysis.md.

## Step 8: deliverable
- The harness refused a subagent write of analysis.md ("return findings as text"), so the full analysis went back to the coordinator in the final report, to be saved as analysis.md here.
