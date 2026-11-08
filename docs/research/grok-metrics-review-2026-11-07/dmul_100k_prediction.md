# D_mul at 100k updates: registered prediction (2026-11-08)

**Status.** Registered before any 100k run of D_mul. On 2026-11-08 no such run existed under `grok_runs`. The commit that adds this file to `main` is its timestamp. Scoring uses only the rules below; nothing here may be changed after the run starts.

**Why.** All three D_mul seeds of the task × format study (phase 1) end at 25k updates below 0.99 and still rising. The design doc's E1 rule says a task under 0.99 at 25k gets a 100k run before anything is concluded. The metrics review (`report.md`, do-now item 6) says to register the forecasts first and score them against the sustained crossing.

## The run this applies to

- **Arm and task:** D_mul (digit inputs, `--tasks mul --number_format digits`), seeds 0, 1, 2.
- **Settings:** the same arguments as phase 1's `runs/D_mul_s<seed>`, except `--updates 100000`. Phase 1's other arguments:
  `--device cuda --wd_scope repo --dropout 0.1 --beta2 0.95 --clip 1 --batch 512 --bf16 autocast --lr 6e-4 --pairs paired --answers per_task --tasks mul --number_format digits`.
- **Eval:** one every 100 updates, as in phase 1.
- **Curve metric:** `test_acc_unseen_offdiag@mul`.

**The run is not planned yet** (checked with the study's planning session, 2026-11-08):
- Taskfmt phase 1's pre-registration treats "censored at 25k" as a registered outcome and has no 100k rule.
- The E1 100k rule covers only E1's tasks.
- A D_mul 100k run is therefore new work and the owner's call. It needs its own registered amendment.
- Phase 2 has first claim on the GPU. The cost is about 20 GPU-min per seed, 60 min for three.

This note fixes the forecasts and scoring now, so they stand whenever the run happens.

**It must be a fresh 100k run, not an extension.** grok_lab keeps no state past 10k (`--ckpt_at` stops at 10,000), so nothing at 25k can be continued. The learning rate is the same as phase 1's at every update: `--lr_schedule constant`, `--lr 6e-4`, `--warmup 10`, and with a constant schedule `lr_at()` does not read `--updates`. So the forecasts, which were fitted under phase 1's schedule, apply to the new run. Scoring uses the new run's curve alone.

## Where the runs stand at 25k (phase 1)

| Seed | Late level L (24–25k median) | Eval at 25k | Highest single eval | First eval ≥ 0.99 | Train ≥ 0.99 at |
|---|---|---|---|---|---|
| 0 | 0.976 | 0.958 | 0.982 | none | 600 |
| 1 | 0.971 | 0.970 | 0.984 | none | 600 |
| 2 | 0.972 | 0.968 | 0.979 | none | 600 |

## The forecasts

How the trend forecasts are computed:
- fit logit(accuracy) against ln(update) by least squares, over evals at multiples of 500 updates in the window;
- clip accuracy to [10⁻⁴, 1 − 10⁻⁴] before the logit;
- project the crossing as x₉₉ = h · exp((logit 0.99 − ẑ(h)) / β), where h is the window's end and ẑ(h) the fitted value there.

This is the method of `analysis.md` §4, recomputed independently in the recheck.

| Forecast | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| **A. Full-window trend (10k–25k):** β ± se | 2.39 ± 0.16 | 2.24 ± 0.09 | 2.38 ± 0.15 |
| A: projected 0.99 crossing | **37,700** | **42,600** | **35,500** |
| **B. Late-window trend (17.5k–25k):** β ± se | 2.07 ± 0.67 | 2.19 ± 0.35 | 1.03 ± 0.47 |
| B: projected 0.99 crossing | **41,400** | **42,900** | **70,500** |
| **C. Sigmoid in log time:** fitted top | 0.983 | 0.968 | 0.970 |
| C: forecast | **never crosses 0.99** | **never crosses 0.99** | **never crosses 0.99** |

**Prior expectation, stated now:** if A or B is right, the run will cross later than it projects. Scored against the sustained crossing, this kind of forecast was optimistic in all six phase 1 runs where it could be checked.

## How each forecast is scored (fixed now)

1. **The event.** The sustained crossing T is the first eval u at which every eval from u through 100,000 is ≥ 0.99. If there is none, T is censored: "after 100k".
   - The first single eval ≥ 0.99 is reported too, but not scored.
2. **Forecasts A and B, per seed.** Take r = T / projection.
   - **Hit:** 1/1.5 ≤ r ≤ 1.5.
   - **Miss:** r falls outside that range.
   - **Censored seed:**
     - a miss if the projection is under 66,700 (100,000 / 1.5);
     - undetermined if the projection is 66,700 or more. Only B's seed 2 (70,500) is in that case.
3. **A against B.** Compare the mean |ln r| over the three seeds; the smaller wins.
   - For a censored seed, use the lower bound ln(100,000 / projection).
   - If the bounds leave the order open, the comparison is reported as undetermined.
4. **The late trend, per seed (scored, used by rule 5).** β over 75k–100k, by the same method as the forecasts (evals at multiples of 500 updates in that window). The seed is:
   - **still rising** if β > max(3 se, 0.3), the readout's "rising" test;
   - **declining** if β < −3 se;
   - **flat** otherwise.
5. **Forecast C, per seed.**
   - **Right:** no sustained crossing by 100k, the median over 99k–100k is < 0.99, **and** the late trend is flat or declining.
   - **Wrong:** there is a sustained crossing.
   - **Censored at 100k:** no sustained crossing, but the late trend is still rising. This confirms neither C nor a plateau. The seed is reported as censored, and no plateau verdict is drawn for it.
   - **Mixed:** no sustained crossing, but the 99k–100k median is ≥ 0.99. This is reported as mixed, not as right.
6. **Also reported, not scored:**
   - the late level L;
   - whether the first 25k reproduce phase 1 exactly (see below).

**What the outcomes mean:**
- **A or B hits in all three seeds:** the trend forecast can be used for D_mul, with its known optimism.
- **C is right in all three** (no crossing, and a flat or declining late trend): D_mul is a near-line plateau below 0.99, not a delayed grokking still on its way.
- **Any seed censored at 100k:** for that seed the question stays open. The run says D_mul is still climbing at 100k, and nothing more.
- **Seeds split:** the outcome is reported per seed, and no single verdict is drawn.

## Reproducibility check

Same-seed CUDA reruns have been bit-identical on this machine (47 pairs; E1 add_s0 rerun 251/251 rows). With a constant learning rate, the 100k runs' first 25k may match phase 1's `metrics.jsonl` exactly. Nobody has checked that nothing else in grok_lab depends on `--updates`, so a match is expected but not assumed. Either way, scoring uses the new run alone:
- **Exact match:** report it. The forecasts were then made on the same trajectory's first 25k.
- **Not exact:** record what differs: device, threads, torch, matmul precision, code hash, and the first eval that differs. Score the forecasts anyway, and mark the result "first 25k differ from phase 1". Ootani 2026 found that environment changes flip a minority of same-seed outcomes.
