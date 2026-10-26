# ReLSM counter-review

**Date:** 2026-10-26. **Code:** this repository at HEAD `acd48ff`. Nothing in the repo was modified.
**Reviewed:** the external report `ReLSM_Code_Review_2026-10-26.md` (F01–F30 plus its telemetry, test, documentation and language-data sections), and eight independent agent reports.

| Key | Agent | Saw the external report? | Scope |
|---|---|---|---|
| VM | verify_model | yes | F01–F03, F15–F20 |
| VT | verify_train | yes | F05, F07–F11, F21, F23–F28 |
| VE | verify_eval | yes | F04, F06, F12–F14, F22, F29, F30, tests, language data |
| VA | verify_artifacts | yes | telemetry tables, IID result files, parameter counts |
| BM ★ | blind_model | **no** | model.py |
| BT ★ | blind_train | **no** | train.py, data.py, curriculum.py |
| BE ★ | blind_eval | **no** | eval, metrics, tests |
| BA ★ | blind_artifacts | **no** | the surviving run, the seven IID files, README |

★ marks a blind agent. When a blind agent found a problem on its own, that is the strongest evidence in this document. The eight agent reports and their probe scripts are kept outside the repository (local review bundle, `ReLSM-review/`).

Two tags are used throughout:
- **Demonstrated** means someone reproduced it with a probe, a run, or the saved artifacts.
- **Plausible** means it is consistent with the evidence but not proven.

---

## 1. Bottom line

1. **The external report is reliable on facts.** No finding was refuted, and every telemetry number was recomputed exactly. Most line citations are within ±2 lines. Its weakness is understated severity and causes it left unconnected, not false claims.
2. **The DAG-v12 run's main failure was a chain of causes the report did not connect.**
   - The 48-token training limit truncated copy and reverse targets.
   - That pinned their curriculum accuracy at about 0.61 and 0.64.
   - That paused the DAG from micro-step 13,000 to the end, so six of ten tasks got 1.94% of examples between them.
   - A probe of the checkpoint shows the cliff directly: copy and reverse score 1.0 at lengths 6–10 and 0.0 at lengths 11–14.
3. **DAG-v12 was never a grokking experiment.** The data was procedural and its distribution kept moving. The run made only 6,458 optimizer updates. Embeddings were excluded from weight decay, and the weight norm grew from 152.6 to 358.1. Its "IID" score of 29% mostly measures the setup, not the model.
4. **Grokfast as implemented diverges exponentially whenever gradient accumulation is 2 or more.** The gradient grows ×1.545 per update at the default accumulation of 4, and ×16.08 at 12. Two agents found this on their own, one of them blind. Any Grokfast run with accumulation ≥ 2 was invalid. That probably covers your deleted Grokfast run, but we cannot prove it.
5. **The evaluation harness is not trustworthy yet.**
   - A perfect "oracle" model scores only 0.779.
   - 7 of the 21 OOD conditions cannot be answered within the hard-coded 32-token cap.
   - Eval sets change in every Python process.
   - Training-time evals reset the global RNGs.
6. **The "infinite procedural data" is a replayed set.** Each new iterator reseeds its RNG. With DataLoader workers, every epoch is byte-identical and the difficulty schedule never advances, even though the logs say it reached 1.0.
7. **The latent, act, ssm and ssm_mem variants leak future tokens, and trained models exploit the leak.** Under teacher forcing they beat the entropy floor; given only the prefix, they do worse than chance. Baseline and shared_loop are clean. Do not compare architectures until the leak is fixed.
8. **Fix order.** First the truncation guard, the eval budget, determinism and an oracle test, Grokfast placement, data reseeding, and the fixed-data crash. Then run a grokking experiment built on purpose: a fixed finite training set, a disjoint test set from the same distribution, and many full-batch updates with weight decay on all weights.

---

## 2. Scorecard of the external report

**Verdicts:**
- **Confirmed:** right as stated.
- **Understated:** right, but the real impact or scope is larger.
- **Overstated:** part of the claim does not hold.
- **Refuted:** wrong.
- **Unverified:** nobody checked it.

**Overall: 0 refuted. 12 findings understated. 1 partly overstated (F23). The rest confirmed.**

| ID | Topic | Report pri. | Verdict | Corroborated by | Notes / citation errors |
|---|---|---|---|---|---|
| F01 | Future/target leakage in latent/act/ssm/ssm_mem | P1 | **Confirmed, understated** | VM, **BM★** | BM trained on i.i.d. tokens (floor ln 8 = 2.079). Teacher-forced loss was latent 1.841, ssm 1.927, ssm_mem 1.831, act 2.016. Prefix-only loss was 2.588, 2.744, 3.275 and 2.255. Trained models *exploit* the leak, which goes beyond the report's random-initialization demonstration. `local_window` does not fix it (VM). |
| F02 | SSM fallback scan mixes batch and time | P1 | **Confirmed, understated** | VM, **BM★** | It is also a causality leak *within one example* (future → past) whenever B > 1. The recurrence itself is right; only the layout and reshape at :624 are wrong. Cite fix: the broadcast is at model.py:83/87, and :84 is blank. |
| F03 | ACT halt head never learns | P1 | **Confirmed** | VM, **BM★**, BT | Gradients are identical for λ = 0 and λ = 10. Missed: outputs depend on batch composition (VM and BM★ found this independently), and the ponder term rewards *not* halting (VM). |
| F04 | OOD 32-token generation cap | P1 | **Confirmed, understated** | VE, **BE★**, **BT★**, **BA★** | 7 of 21 OOD conditions are impossible. Copy/reverse at lengths 20 and 24 are *certain* to fail, not "suspect". The cap was already present in the Dec-23 eval code (8f01eca:eval_hub.py:407). The metadata records 64 while 32 was used (VE, BE★, BA★). OOD is off by default, and none of the seven files is OOD. |
| F05 | Fixed-data train-eval crash | P1 | **Confirmed, understated** | VT | Real `--fixed_data` run (train_eval_mode `auto`): it crashed at the first eval, and no checkpoint is written. With the default eval_interval, every fixed-data run without the task curriculum dies at micro-step 500. That is the canonical grokking configuration. No blind agent ran this path end to end. |
| F06 | 48-token truncation removes answer supervision | P1/P2 | **Confirmed, understated** | VE, VA, **BA★** | Rates agree (copy at 0.65: 200/400 report, 214/400 VE, 48.9% of 2,000 BA★). **The report did not connect this to the curriculum stall. It was the root cause (§4).** Cite: data.py:1123 should be 1124. |
| F07 | Fresh-process resume rejects its own checkpoint | P1 | **Confirmed** | VT, **BT★** | The real `--resume_from` exits with an UnpicklingError. eval/self_test has the same bug (VE, BE★). |
| F08 | Final partial accumulation discarded | P2 | **Confirmed** | VT, **BT★**, VA | The DAG-v12 checkpoint proves it directly: Adam step 6,458 = floor(77,503/12). The report said this needed the original source. |
| F09 | Resume not reproducible | P2 | **Confirmed, understated** | VT, **BT★** | Resume **replays the data stream from the beginning**: identical batch hashes, and difficulty progress reset to 0. That is worse than "changes order". |
| F10 | Resume overwrites metrics | P2 | **Confirmed** | VT, **BT★** | The full file is also rewritten at every log, so I/O grows quadratically. |
| F11 | Grokfast re-filters accumulated / AMP-scaled grads | P2 | **Confirmed, severely understated** | VT, **BT★** | It is exponential divergence, not just "different filtering". ×1.545 per update at accum 4 (measured end to end), ×16.08 at accum 12. One fp16 overflow poisons the EMA permanently. Should be P1 (§3, A1). |
| F12 | Eval seeds vary per process | P2 | **Confirmed** | VE, **BE★**, **BT★**, **BA★** | The v1/v2 re-evals of the same checkpoint share 0 prompts (BA★). |
| F13 | Parser false positives/negatives | P2 | **Confirmed** | VE, **BE★** | More cases were found. Practical frequency is low: parse failures ≤ 0.009 in the seven files (BA★). |
| F14 | Long-integer distance contradicts EM | P2 | **Confirmed, understated** | VE, **BE★** | Every value with ≥ 16 digits gets distance 1, so all successor and chain IID answers do, even when correct. The oracle's overall distance is 0.196 and its MAE 3.4e35 (BE★). |
| F15 | Mamba kernel ABI mismatch | P2 | **Confirmed (latent locally)** | VM, BM★ (source only) | `mamba_ssm` is not installed in the venv, so this never ran locally. |
| F16 | Fallback ignores input multiplier | P2 | **Confirmed, understated** | VM, **BM★** | The real consequence: every channel in a 128-channel group is identical. At nano scale the "SSM" emits 8 distinct scalars per token. This is P1 for SSM experiments. |
| F17 | Generic init erases SSM dt init | P2 | **Confirmed** | VM, **BM★** | BM★ adds that `A_log` starts at all zeros, so every state has A = −1. |
| F18 | KV cache loses padding history | P2 | **Confirmed (low impact)** | VM, BM★ | Eval buckets prompts so they are unpadded, and training never uses the cache. Arguably P3. |
| F19 | SSM cache never initialized | P2 | **Confirmed** | VM, BM★ | Cite: the gating line is model.py:646, not :633. |
| F20 | create_model override collisions | P2 | **Confirmed (broader)** | VM, BM★ | Fails for every size. The standard CLI is unaffected. |
| F21 | Worker RNG and tokens_seen reset | P2 | **Confirmed, understated** | VT, **BT★** | With workers, every epoch is byte-identical and the `smooth` difficulty stays near progress ≈ 0 for the whole run, while the logs claim 1.0. Arguably P1 for default non-curriculum runs. |
| F22 | Addition OOD overlaps training | P2 | **Confirmed** | VE, **BE★**, **BT★**, **BA★** | Nuance: IID addition is capped at 3–4 digits, so the "OOD" 6-digit condition is *inside* training but *outside* IID. |
| F23 | Language PPL uses the training stream | P2 | **Confirmed in substance; "affects dataset state" overstated** | VT, **BT★** | LanguageDataset has no mutable state. PPL runs only in the language phase. |
| F24 | PPL includes ACT ponder | P2 | **Confirmed (small)** | VT, BM★ | At most about 0.08 nats. |
| F25 | Full-parameter `torch.cat` at each log | P2 | **Confirmed (scale-dependent)** | VT | 166 MB at nano. Cite: about :2770, not :2762. |
| F26 | Misaligned gradient telemetry | P2 | **Confirmed, slightly understated** | VT | Module norms are post-clip while the global norm is pre-clip. NaN norms are logged as 0.0. |
| F27 | Update-norm proxy / WD pressure | P2 | **Confirmed** | VT, BA★ | WD pressure is overstated about 1.52×. Logged keys have no "est" label. Cite: :2843–2857, not :2790. |
| F28 | Warmup == total gives ZeroDivisionError | P2 | **Confirmed (cosine only)** | VT, **BT★** | The default `wsd` and `plateau` schedules do not crash. |
| F29 | Raw outputs not saved | P2 | **Confirmed** | VE, **BE★** | The resolved `max_new_tokens` is not saved either. |
| F30 | Longctx budget override / AUC 0 | P2 | **Confirmed, understated** | VE, **BE★** | The override *always* happens. Nano never runs longctx, so it always reports AUC 0. Also a regression: scoring is now raw string equality. |

### Other sections of the report

| Section | Verdict | Notes |
|---|---|---|
| DAG-v12 property table (all 17 rows) | **Confirmed exactly** (VA) | |
| Task exposure table | **Confirmed exactly** (VA; BA★ reproduced the same counts) | |
| "Locked floor 0.02 did not give 2% per task" | **Overstated** (VA) | By design the floor is a *total*. Locked tasks received 1.944% in total, so the code did what it says. Starving six tasks with 2% is still a design problem. |
| "10 of 1,550 windows clipped" | **Confirmed count, weak inference** (VA) | The flag samples only the last optimizer step per window (about 24% of steps). 7 of the 10 fall in warmup. |
| "Procedural train-eval" 81.8% / 71.3% | **Numbers confirmed, interpretation missed** (VA, BA★) | This is a fixed probe set, 68% of it locked tasks at difficulty 0.2. It is neither a training-split metric nor a memorization metric. |
| Seven IID files (scores, baseline variant) | **Confirmed, understated** (VA, BA★) | Copy/reverse are 0 in *all seven* files, and successor is 100% in all four 5/20-step files. |
| Test suite (collection error; 29 pass / 3 fail; classifications) | **Confirmed exactly** (VE, **BE★**) | |
| Tooling bullets (self_test, import test, profiler) | **Confirmed** (VE, BE★, BT★) | The profiler also cannot run as a script. The import test is vacuous without mamba. |
| Parameter counts / README | **Confirmed** (VA, **BM★**, **BA★**) | 41,564,672 unique parameters (measured). 300M and 350M are the same 435.5M model; "760M" (984.8M) is bigger than "1B" (976.3M). |
| Shared-loop fairness | **Confirmed, understated** (VM, BM★) | `train.py` exposes no unroll flags, so shared_loop is always 6 unique layers × 24 applications: 4× the compute at the same parameter count. |
| Language-data points 1–8 | **Confirmed** (VE, BT★) | The infinite rebuild loop has now been exercised (BT★: about 41k rebuilds in 20 s). Citations are off by about 10 lines. |
| "Rechecking earlier bug discussions" table | **Unverified** | No agent re-adjudicated the historical transcripts. |

### What the report got right, and where it was weak

**Right:**
- Careful evidence labels.
- Exact telemetry numbers.
- Correctly cleared the opt-in variants as an explanation for the baseline runs.
- Correctly refused to claim any grokking or transfer success.
- Correctly flagged that the "mix" phase was aliased algorithmic data.

**Weak:**
1. It listed truncation (F06) and the curriculum stall as two separate observations. The artifacts link them causally, and that link is the main explanation for DAG-v12.
2. It treated Grokfast as a filtering nuance. In fact it is a runaway feedback loop.
3. It missed that the procedural data replays, that difficulty stalls with workers, that eval resets the RNGs, and that the compare "=" answer is never supervised.
4. It did not run an oracle through the eval, which would have exposed the harness's ceiling at once.
5. It did not ask whether DAG-v12 could have grokked at all: 6,458 updates, a moving distribution, no weight decay on embeddings, and a growing weight norm.

---

## 3. New issues the report missed (deduplicated, ranked by impact on your experiments)

### Tier A: silently invalidates or changes results

| # | Issue | Where | Evidence | Found by |
|---|---|---|---|---|
| A1 | **Grokfast diverges exponentially with accumulation ≥ 2.** The filter runs after every micro-batch on `.grad`, which already contains λ·EMA. Per-update factor: accum 1 → 0.98, 2 → 1.058, 4 → 1.544, 8 → 4.76, 12 → 16.08. The float32 EMA overflows after about 204 updates at accum 4 (about 32 at accum 12), then clip turns inf into NaN weights. Under fp16, one overflow writes inf into the EMA and every later step is skipped. | train.py:2718–2726 (EMA at 2722–2726) | VT: grad norm 25 → 1,481 over 11 updates (ratio 1.545). BT★: 27 → 161k in 88 micro-steps. Loss 3.54 (accum 4) vs 2.17 (accum 1). fp16 toy: steps skipped `[F,T,T,T,T,T]`. | VT, **BT★** |
| A2 | **Truncation causes the curriculum deadlock.** At alg_seq_len 48, truncated copy/reverse targets cap train-eval exact, and the DAG gates need 0.98–0.99. Details in §4. | data.py:1124 (slice); train-eval re-splits truncated tensors | Logged samples show truncated targets. Train-eval exact was frozen at 39/64 (copy) and 53/83 (reverse) for 22 evals. Length cliff at 11 items in the checkpoint probe. | VA, **BA★** |
| A3 | **The procedural data is a replayed set.** `__iter__` reseeds `random.Random(seed + worker_id)` on every new iterator, and an epoch is only `alg_examples` (default 100k). With workers, epochs are byte-identical. With `num_workers=0`, they diverge only through curriculum jitter. DAG-v12 restarted the stream about 92 times. | data.py:1069–1076, 1087; sampler data.py:1751–1758 | VT: 11/11 batch hashes identical. BT★: overlap 1.000 with workers, 66.8% without. VA: 9.16M examples ÷ 100k ≈ 91.6 passes. | VT, **BT★**, VA, **BA★** |
| A4 | **The difficulty schedule never advances with workers, and the logs claim it did.** Progress is computed from `tokens_seen` inside each worker's pickled copy and resets every epoch. It ends an epoch at about 0.0047. The logged difficulty uses the parent's counter and rises to 1.0. Applies to `smooth` (default), `warmup_ramp` and `phased` with `num_workers > 0` and no task curriculum. **The Dec-23 code (8f01eca:data.py:1022–1025) had the same logic, with the same defaults: num_workers 4, smooth.** | data.py:1089, 1135; train.py:2598–2602 | BT★ run A: logged difficulty 1.000 while trained examples stayed easy. VT: the arithmetic and the probe. | VT, **BT★** |
| A5 | **Training-time eval resets the global RNGs** (Python, NumPy, torch CPU/CUDA) at every eval. Fixed-data shuffles become identical every epoch, and dropout and jitter streams restart. | eval_hub.py:568, 664, 750; utils.py:1129 | BT★: three post-eval epochs got the same permutation `[11,5,2,8,6,1,7,4,10,3,9,0]`. BE★: the first `torch.rand` after an eval was identical from two different starting states. | **BT★**, **BE★** |
| A6 | **Compare "=" is never supervised.** "Answer:" + "=" tokenizes as the single token `:=`, so the prompt mask covers it and only EOS is trained. This caps compare exact at 93.9% at difficulty 0.1, 96.1% at 0.3 and 97.0% at 0.7, which is below the DAG gate. | data.py:1121–1131, 1271–1284 | VE: 41/600 examples at 0.1 supervise only `</s>`. BT★: 35/35 "=" examples mismatched, 0/5,965 for "<"/">". | VE, **BT★** |
| A7 | **The SSM ignores its input stream, and channels collapse.** The input `v` never enters the recurrence. B, C, dt and A are shared per 128-channel group, so all channels in a group are identical. At nano scale that is 8 distinct values per token per layer. | model.py:87, 615–618 (v at 564, unused) | Spread within a group = 0.0 (VM, BM★). | VM, **BM★** |
| A8 | **CE returns NaN on batches with zero labels.** `nan * 0` poisons the eval aggregates, and training logs a NaN loss. This is realistic: chain at 48 tokens has zero labels in 213–218 of 400 examples at difficulty 1.0. | evaluate_perplexity / evaluate_training_split; model.py:903–910 | VT probe | VT |

### Tier B: confounds how results are read

| # | Issue | Where | Evidence | Found by |
|---|---|---|---|---|
| B1 | **A perfect oracle scores 0.779 overall** (about 0.67 on OOD), with distance 0.196 and MAE 3.4e35. The eval has a built-in ceiling. | eval_hub `run_eval_suite`, grid `all` | p5_oracle | **BE★** |
| B2 | **The "IID" grid is not the training distribution.** It uses final-phase ranges at a fixed difficulty of 0.5. Addition is capped at 3–4 digits with digits 0–9; training's final phase uses digits 5–9. Dyck has depth ≤ 3 and length 10–18; training uses 18–30. Locked tasks trained at 0.2 are scored at final-phase sizes. Fixed-data sets (difficulty 0.5) are scored on a different distribution, e.g. successor 7–16 vs 19–32 digits. | eval/ood_grid.py:19–31; eval_hub.py:591 | VE, BE★, BT★ (p4_fixed), BA★ | VE, **BE★**, **BT★**, **BA★** |
| B3 | **Dyck "OOD depth 5/6" is not OOD.** At difficulty 0.5 the depth cap is min(3, max_depth), so ood_depth5, ood_depth6 and max_depth=50 produce byte-identical sets. | data.py:631–636; eval_hub.py:591 | Measured depth histograms | VE, **BE★**, **BT★** |
| B4 | **Train-eval is a fixed probe, weighted toward locked tasks.** Seed `seed+123`, re-iterated each eval. It contains the same 800 examples whenever the curriculum is frozen, and 68% of them are locked tasks. The gate EMAs re-measure this one small set (often about 31–65 per task) against 0.98–0.99 thresholds. | train.py curriculum block (~3450) | Identical per-task counts at 14k, 15k and 16k (BA★). 68% locked (VA). | VA, **BA★**, **BT★** |
| B5 | **The DAG design is fragile even without truncation.** The controller raises difficulty whenever EMA > 0.85, so tasks hover between 0.55 and 0.85, while the gates need 0.98–0.99 for 4 consecutive evals. Other problems: <ul><li>one backslide pauses *all* unlocks, with no timeout;</li><li>successor is a single choke point for 6 tasks;</li><li>the EMA snaps to the first nonzero accuracy;</li><li>the IID-sync bump resets the EMA to 0.85;</li><li>task subsets without their prerequisites never unlock;</li><li>`train_eval_mode none` freezes the curriculum.</li></ul> | curriculum.py:74–77, 103; train.py:3272–3285 | BT★ p3 §1/§3; BA★ timeline | **BT★**, **BA★** |
| B6 | **Positions beyond 48 were never trained, but IID needs them.** IID needs them for copy in 162/200 examples, reverse 144/200, successor 195/200 and chain 200/200; 104/200 chain prompts alone are ≥ 48 tokens. With `lang_loader` aliased, no other data covered those positions. | — | VE M3 | VE |
| B7 | **Grokking preconditions were absent in DAG-v12.** <ul><li>6,458 optimizer updates; all counters are in micro-steps.</li><li>tok_emb/lm_head (39% of parameters) excluded from weight decay.</li><li>Weight norm grew 152.6 → 358.1 and never contracted.</li><li>Eval every 83 updates; curriculum cooldown about 42 updates.</li></ul> | optimizer groups; train.py:2729 | Checkpoint and logs | VA, **BA★** |
| B8 | **Best-checkpoint selection is noise and leaks.** The same 800-example IID set selects `best_model.pt` and is also reported. Best is 0.2925 vs final 0.2900, with SE ≈ 1.6 pp. IID results also feed back into training through the sync bump. | train.py:3288, ~3277 | | VA, **BA★** |
| B9 | **Soft metrics are biased by SentencePiece's dummy prefix.** A wrong parity label gets token accuracy 0.5. A perfect "579" gets length_ratio 0.75. | utils.py:858, 989 | p4_metrics | **BE★** |
| B10 | **EM and the numeric diagnostics read different text.** EM uses the first line; MAE uses all of it. "579\n580" gets EM 1 with abs_err 1. | utils.py:777 vs 1264/1283 | | **BE★** |
| B11 | **The ACT variant depends on batch composition.** Halting breaks only when the whole batch halts. Alone: 5 steps; batched: 6; logits move by 7.7e-3 to 1.6e-2. | model.py:789, 792 | | VM, **BM★** |
| B12 | **The aliased `lang_loader` replays the start of the stream.** 8.3% exact duplicates vs 0.2% for a single stream. "mix" is mislabeled. | train.py:2242–2243 | BT★ p3 §7; VT identical first-5 hashes | VT, **BT★** |
| B13 | **Metadata misstates the eval.** `meta.decoding.max_new_tokens` = 64 while 32 is used. `gather_metadata` records the git commit of the current working directory, not the code. | eval_hub.py:752, 780; utils.py:1384 | All seven files | VE, **BE★**, **BA★** |
| B14 | **Logging bugs.** <ul><li>`train_eval/acc/token` appears twice in one dict, so the generated value overwrites the teacher-forced one.</li><li>`train_eval.difficulty.by_task` is always 0.</li><li>In the old schema, `train_accuracy` means teacher-forced token accuracy, while `train_accuracy.<task>` means generated exact match.</li></ul> | train.py:3603, 3606 | | VA, **BA★** |
| B15 | **Dyck can be solved by counting.** 100% of invalid strings have unequal counts. Compare at difficulty ≥ 0.6 almost never produces "=" (6 of 20k). | data.py dyck generator | 30k samples | **BA★** |

### Tier C: hygiene and robustness

- **Unhandled exceptions write no final checkpoint**, so F05- and F28-style crashes lose all progress. Ctrl-C writes `final_model.pt` and `summary.json` as if the run had finished. (VT, BT★)
- **Loss normalization:** micro-batch means are weighted equally, so one token can carry up to about 9× the weight of another inside a single update. (VT)
- **Long-context:** the numeric-tail parser was removed in 17a72e8, so a continuation like " 1234." now scores 0. Needle prompts have no BOS. (VE, BE★)
- **Eval autocast** defaults to fp16 on CUDA while training uses bf16. Not verified on a GPU. (BE★, BA★)
- **`--compile` checkpoints** carry an `_orig_mod.` prefix and will not load. Source reading only. (VT)
- **Language documents get no EOS.** One pass is at most 1.0M documents, after which the order repeats. `datasets` 4.4.1 likely drops the script-based `wikipedia` source. (VE, BT★)
- **Manager-proxy sampling is about 7× slower** than plain dicts. (BT★)
- **Other model issues:** `generate()` leaves the model in eval mode. The SSM ignores the padding mask. GQA head counts are not validated. (VM, BM★)
- **Tests do not cover the live scoring path:** `evaluate_condition`, the OOD grid, generation budgets or determinism. (BE★)
- **Code drift:** DAG-v12 started about 50 minutes before commit a868d71 ("Prevent double curriculum bump after IID eval"), so it probably ran with that bug. Unverified. (VA, BA★)

---

## 4. Why the experiments failed

### 4.1 DAG-v12 (`exp0_baseline_100M_5steps_wd1-e1_dag_v12`)

**Setup (all Demonstrated, from the checkpoint and hyperparameter row):**
- Baseline nano: 41.6M unique parameters, context 128.
- alg_seq_len 48.
- 100M *supervised* tokens.
- Accumulation 12, giving 6,458 optimizer updates.
- AdamW with weight decay 0.1, excluding embeddings.
- WSD learning rate, no Grokfast.
- DAG task curriculum, `num_workers=0`.
- No language phase: `lang_loader` is aliased to `alg_loader`.
- Eval: 80 examples per task on the "IID" grid, 32-token cap.

**Causal chain:**

| Step | What happened | Status |
|---|---|---|
| 1 | Copy, reverse and compare start unlocked. Copy and reverse climb to about 0.99 EMA at difficulty 0.2–0.45. Parity unlocks at step 10,000. | Demonstrated (VA, BA★) |
| 2 | By step 12k, copy and reverse reach difficulty 0.65 (8–12 items). At 48 tokens, about half of those examples are truncated: copy 48.9%, reverse 43.7%; every 11- and 12-item example is truncated, and truncated examples lose EOS. | Demonstrated (BA★ trunc_check; VE, report) |
| 3 | Train-eval scores the model against the *truncated* target. The model continues past it and is marked wrong. | Demonstrated from logged samples (VA); the mechanism is inferred from source, not re-run |
| 4 | Copy and reverse EMAs fall below the prerequisite threshold minus the margin. **The DAG pauses at step 13,000 and stays paused for all remaining evals.** No further unlocks are considered. | Demonstrated (VA, BA★) |
| 5 | Copy and reverse settle at EMA 0.609 / 0.639, inside the controller's dead band (0.55–0.85), so their difficulty oscillates between 0.60 and 0.65. The frozen train-eval set gives exactly 39/64 and 53/83 at 22 straight evals, even while teacher-forced accuracy rises from 0.79 to 0.82. | Demonstrated (VA, BA★) |
| 6 | Successor needs copy ≥ 0.99, reverse ≥ 0.98 and compare ≥ 0.98 *and* an unpaused DAG. Every other locked task hangs off successor, so six tasks stay locked for the whole run. Together they get 1.94% of examples, always at difficulty 0.2. By answer tokens, copy and reverse take about 95%. | Demonstrated (VA, BA★) |
| 7 | The model really cannot do 11+ items. A CPU probe of both checkpoints gives 1.0 exact at lengths 6–10 and 0.0 at lengths 11–14: it copies 10 items, then garbles. | **Demonstrated behaviourally** (BA★ gen_probe) |
| 8 | The compare "=" ceiling (A6) did *not* bind in this run: compare reached difficulty 1.0, where "=" is about 0.09% of targets. | Demonstrated (BT★ table + VA) |
| 9 | The double-bump bug (pre-a868d71) may have perturbed the difficulty jumps. | Plausible, unverified |

**Why the final "IID" score is 29.0%.** These are the last-eval per-task scores; the sum is 2.90 over 10 tasks.

| Task | Last IID | Main reason | Status |
|---|---:|---|---|
| copy / reverse | 27.5% / 31.25% | The grid uses 10–12 items, and only the 10-item third is solvable (cliff at 11). Positions beyond 48 were never trained (B6). | Demonstrated |
| compare / parity | 91.25% / 93.75% | Genuinely learned. Parity rose as its curriculum difficulty reached the eval's range. | Demonstrated |
| dyck | 46.25% | Below the majority-class baseline of 51.25%: it predicts "no" 72 times out of 80. Locked, trained at 0.2. | Demonstrated |
| successor, chain, addition, mod_add, multiplication | 0% each | Locked and trained only at difficulty 0.2, but scored at final-phase sizes (20–32-digit successors, for example). Successor and chain also have zero slack under the 32-token cap; successor hit max_new_tokens on 71–90% of examples at some evals. | Demonstrated |

**Why there was no grokking.** This run could not test grokking:
- The training distribution kept moving, with no fixed train set.
- It made 6,458 updates.
- Training loss was about 0.04 within the first few thousand micro-steps and then flat.
- The weight norm grew instead of shrinking under decay, because embeddings were exempt.
- The eval curve is a staircase that follows curriculum events, not a delayed jump.

(VA, BA★. That these conditions *prevent* grokking is a plausible reading of the literature, not something demonstrated here.)

### 4.2 The seven Dec-23 IID result files

- **They come from five checkpoints**, v15, v16, v18, 5steps and 20steps. All are baseline. All have been deleted, and none is DAG-v12.
- **They used Dec-23 code**, in which chain and successor were different tasks. They cannot be compared with DAG-v12.
- **Their training configs are gone.**

| Observation | Explanation | Status |
|---|---|---|
| Copy/reverse = 0% in all seven files | **5steps and 20steps:** every prediction is exactly 6 items, ending in EOS. Reverse gives the *correct* first six; copy gives the first 4 plus the last 2. The eval used 8–12 items. This signature points to training at ≤ 6 items (the then-first phase, 4–6). **v15–v18:** they ramble with wrong digits (copy hit max_new_tokens in 92/200 for .1 and 172/200 for .2), which is a genuine model failure. | Signature demonstrated (VA, BA★). "Trained only on ≤ 6 items" is inferred. |
| Why stuck in the first phase? | **Candidate:** A4. The Dec-23 code has the same per-worker progress reset, with defaults `num_workers=4` and `smooth`, which pins difficulty near progress ≈ 0. mod_add predictions are always 2 digits, which fits the first phase (mod 17–50) against an eval at mod 500–997. | **Plausible.** We confirmed the code and defaults in 8f01eca, but the actual run flags are lost. |
| Successor = 100% (5/20-step) | The Dec-23 final phase was 3–6 digits, so it was easy. | Demonstrated |
| Compare 0.5–17.5% (12–17.5% for 5/20-step) | The model echoes operands (`12266`, `3906`); invalid_label_rate is 0.88 and 0.805 in v2. This is a real model failure, not the parser. | Demonstrated |
| Parity 0.515–0.655, dyck 0.01–0.42 | Near chance. Dyck shows confusion with parity's label space. | Demonstrated |
| Addition / multiplication | Real carry errors (6455+4488 → 10933). | Demonstrated |
| v1 vs v2 (30.05 vs 28.05, and so on) | Different example sets (0 shared prompts, because of salted `hash()`) with SE ≈ 3.5 pp per task. This is resampling noise, not an effect of the parser change. | Demonstrated |
| "5 vs 20 steps" | Not a depth ablation: both are baseline with the same architecture, and they score within noise (30.05 vs 30.55). | Demonstrated |
| The 32-token cap | Present in the Dec-23 code but harmless for those tasks. | Demonstrated |

### 4.3 Your deleted Grokfast run

**Demonstrated:** with any gradient accumulation ≥ 2, the Grokfast EMA in this code grows by a fixed factor every optimizer update. The update direction becomes a stale, exploding EMA, and the weights eventually go NaN. With fp16, a single overflow freezes training permanently. The default accumulation is 4, and DAG-v12 used 12.

**Not demonstrable:** whether your run used accumulation ≥ 2 or fp16, or whether something else killed it. What we can say: if it used accum ≥ 2, the run never tested Grokfast. Grokfast should therefore be treated as **untested**, not as "tried and failed".

---

## 5. Prioritized fix list

### Must-fix before any next run

| # | Fix | Acceptance test (one line) |
|---|---|---|
| 1 | Make truncation impossible or loud. Derive `alg_seq_len` from the hardest phase per task, or reject and resample. Log truncation and zero-label rates per task. | For every task at its final phase, 1,000 generated examples fit within the run's seq len with answer and EOS intact (0 truncated). |
| 2 | Per-condition generation budget = tokenized reference + EOS + slack. Honour the CLI and save the resolved value. | An oracle model scores 1.000 exact match on every IID and OOD condition. |
| 3 | Stable per-example seeds (digest-based). Save each eval set's content hash. | Two fresh processes with different PYTHONHASHSEED produce identical IID-set hashes. |
| 4 | Eval must save and restore the global RNG state (or use local generators). | The torch, Python and NumPy RNG states are bit-identical before and after a training-time eval. |
| 5 | Grokfast: filter once per optimizer step, after `unscale_`; skip non-finite grads; checkpoint the EMA. | Constant-gradient test: accum 4 and accum 1 give the same filtered gradient, and the norm stays bounded for 1,000 updates. |
| 6 | Fixed-data train-eval: use normal batching for map-style datasets (F05). | A tiny `--fixed_data` run passes two eval intervals and writes `final_model.pt`. |
| 7 | Counter-based example generation `(seed, epoch, index)` instead of reseeding in `__iter__`. Keep difficulty progress in shared state. | Epochs 0 and 1 share no examples, and `num_workers` 0 vs 2 give the same difficulty histogram at 50% of tokens. |
| 8 | Prompt/answer tokenization guard: a separator before the answer, or assert that the prompt tokens are a prefix. | For 10k examples per task, `encode(prompt)` is a prefix of `encode(prompt + answer)` (0 mismatches; compare "=" included). |
| 9 | Guard all-ignored CE (skip the batch or weight it by token count). | An all-`-100` batch gives a finite aggregate in `evaluate_training_split` and `evaluate_perplexity`. |
| 10 | Wrap the training loop so any exception writes a clearly labelled crash checkpoint. | An injected exception at step N leaves a loadable `crash_step_N.pt`. |
| 11 | *(Only if you resume.)* Save the config as a primitive dict, plus RNG states, data cursor, Grokfast EMA and accumulation boundary. Append to metrics instead of overwriting (F07, F09, F10). | A run that is interrupted and resumed matches the uninterrupted run's batch hashes and its final weights within 1e-6. |

### Before any further DAG / curriculum run

| # | Fix | Acceptance test |
|---|---|---|
| 12 | Train-eval: draw a fresh, task-balanced sample at the current difficulty each eval (≥ 200 per task), scored against untruncated targets. | Per-task counts are equal, and two consecutive evals share no prompts. |
| 13 | DAG: gate on competence at a stated difficulty, not on the moving EMA. Add a pause timeout. Remove the EMA snap and the sync-bump EMA reset. Revisit the successor choke point. | A scripted accuracy trace with one prerequisite backslide still unlocks downstream tasks within K evals. |
| 14 | Count `opt_step`, and express eval_interval and cooldown in optimizer steps. | The logs show opt_step, and the cooldown is a whole number of updates. |
| 15 | Split eval into three grids: current difficulty, a fixed target ladder, and true OOD (dyck depth actually > 3; addition boundaries derived from generator support). Use a separate set for best-checkpoint selection. | The dyck ood_depth5 set contains strings of depth 5, and the selection set has no overlap with the report set. |

### Before architecture comparisons

| # | Fix | Acceptance test |
|---|---|---|
| 16 | Causal latent/memory design (F01). | Suffix and target perturbation changes earlier logits by 0 for all six variants. |
| 17 | SSM: fix the layout/reshape (F02), put the input `v` into the recurrence and un-share the groups (A7, F16), and preserve the dt init (F17). | B=2 vs independent runs match exactly; channels within a group differ; post-construction dt lies in 0.001–0.1. |
| 18 | ACT: a differentiable, per-example halting formulation; ponder kept as a tensor; batch independence. | The halt head has nonzero gradients, and an example gives the same logits alone as in a batch. |
| 19 | Fair comparisons: expose `n_unique_layers`/`n_unroll`; report unique parameters, activated depth and FLOPs. | The run summary lists measured parameters and FLOPs per variant. |
| 20 | Report CE separately from the ponder term (F24). Fix the SSM cache init (F19) and the Mamba ABI (F15), or disable the accelerated path. | PPL is computed from CE only, and SSM incremental and full logits match. |

### Nice-to-have

- Parser grammar and integer-exact distance (F13, F14, B10).
- Fix the dummy-prefix bias in soft metrics (B9).
- Save raw outputs and token IDs (F29).
- Longctx fixes (F30) and accurate metadata (B13).
- Streamed norm computation (F25) and aligned telemetry (F26, F27, B14).
- Cosine edge case (F28).
- Fix or delete the stale tests, and add tests for the live scoring path.
- Make mamba-ssm an optional dependency and regenerate the README preset table from code.
- Language pipeline: chunking, EOS, a hard failure instead of an infinite loop, and pinned sources.
- Make Dyck invalid strings require stack structure.

---

## 6. Recommended next experiments

Run them in this order, each after the must-fix list above.

1. **Eval sanity: minutes, no training.** Run an oracle model and a constant-output model through the full grid.
   - Expect the oracle to score 1.0 everywhere, and the constant model to score majority-class on the label tasks.
   - This checks the harness before you trust any model number.
2. **Single-task learnability: short runs.** Copy, parity and addition, each alone, at a fixed difficulty, with full-length targets, a fixed sampling distribution and accum 1.
   - This establishes what nano can learn in this budget before the DAG is added. (Recommended by the report, step 3, and by BA★.)
3. **A deliberate grokking test, built to grok in principle.**
   - One algorithmic task (for example modular addition at a single fixed modulus) with a **fixed, finite training set**, a **disjoint held-out set from the same distribution**, and **no curriculum**.
   - Full-batch or near-full-batch training, accum 1.
   - Weight decay applied to **all** weights, embeddings included (or untied embeddings).
   - Typical grokking setups use 10⁴–10⁵ optimizer updates (BA★). Strong weight decay and a small training fraction are the standard ingredients in the published modular-arithmetic grokking work; that is literature guidance, not evidence from this review.
   - Log train and held-out exact match, the total weight norm and the actual parameter deltas at every eval. Use several seeds.
   - Success criterion: held-out accuracy rises sharply long after training accuracy saturates, while the weight norm falls.
4. **Grokfast as a matched ablation of experiment 3.** Same seeds, after fix 5. This is the first valid test of Grokfast in this codebase.
5. **Only then:** curriculum vs fixed balanced sampling at matched compute (after fixes 12–15), and architecture comparisons (after fixes 16–20).

---

## 7. Disagreements between agents, and open questions

### Disagreements, and how they resolve

| Topic | Positions | Resolution |
|---|---|---|
| Number of OOD conditions | VE: "7 of 26"; BE★: "7 of 21" | **21.** Counted from eval/ood_grid.py: addition 3, successor 3, multiplication 2, copy/reverse/parity 9, dyck 2, mod_add 2. The 7 impossible conditions agree. |
| Share of successor ood_33digit that is infeasible | 85% (VE), 91% (BE★), 82% (BT★) | Different samples (F12 itself makes eval sets differ per process) and slightly different measures (answer only vs answer + EOS). The oracle EM of 0.15 agrees across agents. |
| How hard the "IID" grid is | BA★: locked tasks are scored at "difficulty 0.85–1.0, the final phase". VE and BE★: final-phase *ranges* at difficulty 0.5, which caps addition at 3–4 digits and dyck at depth 3. | Both are partly right. Copy, reverse, successor and chain ranges match the final phase; addition, dyck and chain digit distributions are easier. Either way, IID ≠ what the learner trained on. |
| Impact of compare "=" | BT★: it delays the DAG by thousands of steps. | That holds in general but did not bind in DAG-v12, where compare reached difficulty 1.0. The truncation of copy/reverse was the binding constraint. |
| "Procedural data is a fixed set" | BT★: byte-identical epochs. BA★/VA: DAG-v12 ran with `num_workers=0`. | It is strictly fixed only with workers. DAG-v12 restarted the same seeded stream about 92 times, with curriculum jitter making the passes diverge. |
| Locked floor | Report: "didn't deliver 2% per task". VA: it is a total, and it was delivered. | VA is right about the code. The design choice is still bad. |
| F18 priority | Report P2; VM and BM★ call it latent. | Low impact for current workflows; P3. |
| F23 "affects dataset state" | Report: yes. VT: no. | VT is right: LanguageDataset has no mutable state. |
| Copy@0.65 truncation rate | 50% (report), 53.5% (VE), 48.9% (BA★) | Within sampling noise. |
| Failures in the Dec-23 files: model or eval? | BA★: mostly model plus a mismatched "IID" definition; parse failures ≤ 0.009. | Agreed. The parser did not cause them. |

### Open questions (not resolvable from the available material)

- **The Dec-23 runs' flags are lost** (num_workers, schedule, alg_seq_len, accum, Grokfast), and so are their checkpoints. The first-phase explanation (A4) cannot be confirmed.
- **Your Grokfast run's settings are lost:** accum, AMP dtype, and when it failed.
- **The double-bump bug:** whether DAG-v12 actually hit it.
- **GPU paths are unvalidated:** fp16/bf16 AMP, Flash Attention, torch.compile, real Mamba kernels, eval autocast mismatch. All probes ran on CPU.
- **No trained SSM, latent or ACT checkpoint survives,** so the real-world impact of F01, F02 and A7 on past variant runs cannot be measured.
- **The report's "Rechecking earlier bug discussions" table** has not been re-verified against the transcripts.
- **Live language-stream behaviour** (which sources actually load under `datasets` 4.4.1) has not been tested online.

---

## 8. Project history (from Codex/ChatGPT transcripts)

Full write-up with per-PR links, quotes and the run table: [`project-history-codex.md`](project-history-codex.md). Sources: the 131 Codex task exports (PR #1, 2025-12-10 → PR #131, 2026-01-12). Nothing in the repo was read for this section.

**Arcs.** Five phases. (1) Dec 10–13, #1–#33: the owner pasted an ablation-ladder `model.py` and Codex built the harness around it (streaming data, SSM + JIT scan, curriculum sampler, `eval_hub.py`). (2) Dec 14–19, #34–#65: Windows hardening and grokking instrumentation (SDPA, eval cap 32, `FixedAlgorithmicDataset` + Grokfast + wd 1.0, metrics.json, checkpoint rotation). (3) Dec 20, #66–#84: a 19-PR day repairing eval after a 25K-step run scored compare 76.6%, dyck 61.5%, parity 10.2%, successor 2.0%, rest 0%; hash seeding, diagnostics, HF-style labels, IID/OOD split. (4) Dec 21–25, #85–#110: per-task EMA curriculum, "Answer: " prompts, resume payload, DAG curriculum, first DAG run, prompt masking. (5) Dec 26–Jan 12, #111–#131: keeping the DAG run alive (dynamic padding, 32-digit successor, chain rewrite, trailing-space fix, JSONL, token batching, Windows workers, padding masks in thought/memory attention).

**Test discipline.** 14 of 131 tasks ran anything; 6 ran a passing test (#7, #17, #43, #108, #114, #118); 7 were compile-only; 23 attempts were skipped because torch was absent in the sandbox; 94 ended "Not run (not requested)" or "(per instructions)". No transcript shows Codex running `train.py`. Every behavioural check was the owner pasting a console log into the next task.

**Prompt → defect links (strongest first).**
- #45 (Dec 17): "Apply Grokfast (after loss.backward(), before optimizer.step()) … # Before: scaler.unscale_(optimizer)" → F11/A1, the per-micro-batch filter inside accumulation. Never run.
- #71 (Dec 20): `example_seed = hash((seed, task, i)) & 0x7FFFFFFF` → F12; pytest skipped (no torch).
- #40/#43 (Dec 16): "--eval_max_new_tokens (type=int, default=32 … prevent infinite loops)" → F04; #20's signature default of 64 is what B13's metadata still records.
- #110 (Dec 25): `labels[:prompt_len] = -100` with the prompt encoded separately; #116 (Dec 28) found the space merge but asserted the parser was "already robust enough" → A6 (compare `:=` never raised) and the train-eval outage fixed in #119.
- #95/#109/#113/#115: EOS appended before `tokens[:self.max_seq_len]`; `--alg_seq_len 32` then `48` while the curriculum was pushed to 32-digit successor/chain → F06/A2, the DAG-v12 stall. The mismatch is never discussed.
- #13: after Codex said "Embeddings do get weight decay", the owner added `"tok_emb"`/`"lm_head"` to the no-decay group → B7.
- #5 ("repetition period … likely acceptable") and #17 ("Reset tokens_seen = 0 in iter … repeating easy → hard cycle every time the dataset is exhausted") → A3/A4 specified as intended behaviour.
- #12: the pasted `ssm_scan_jit` recurrence `state = A_bar * state + dt_t * B_t_exp[:, t]` has no input stream → A7.
- #103 "DO NOT introduce or plumb attention_mask" plus #36 `mask=None` → no padding mask in thought/memory attention until #126 (Dec 31), which then shipped a non-causal path the owner caught.
- #104 "Prefer the safe fix if possible" → F07 (safe globals never run in a fresh process). #59 atomic writes + #90 "every log interval" → F10.

**Diagnoses that later proved wrong.** #119 (Dec 28): "IID/OOD evals: fine" and "No obvious padding/masking bugs found", three days before #126. #108/#110: the DAG unlock margin was reversed, fixed, then reversed again. #93's cached worker snapshots were the stale copies #130 had to replace with Manager dicts.

**What this resolves in §7.** Little. `v15`, `v16`, `v18`, `20steps`, `dag_v12` and the Grokfast run never appear in any transcript, so the Dec-23 run flags and the Grokfast run's accumulation/AMP settings are unrecoverable from this source. The only Grokfast command is #45's suggested test (`--fixed_data_size 3000 --grokfast --weight_decay 1.0 --alg_tokens 50000000`, no accumulation stated); every later logged config says "Grokfast: False". The documented DAG commands are Dec 25 (`exp0_baseline_100M_5steps_wd1-e1_dag`: alg_seq_len 32, accum 16, eval cap 32, workers 0), Dec 27 (`…_dag_v6`: alg_seq_len 48, accum 16, eval cap 48) and Jan 11 (`…_dag`, accum 12, workers 4, crashed at start). Accumulation 12 appears only on Jan 11, consistent with the DAG-v12 checkpoint. The double-bump guard (#123) is dated Dec 29, so whether DAG-v12 ran with the bug depends on its start time, which the transcripts do not give. The report's "Rechecking earlier bug discussions" table remains unverified beyond the items above.


The ChatGPT side of the history (48 chats, advice-to-defect links, leads opened and dropped) is in `project-history-chatgpt.md`.

---

## 9. Recommendations going forward: keeping the codebase trustworthy

Sections 5 and 6 say what to fix and what to run. This section says how to work so the same classes of defect do not come back. Each item is tied to something this review found.

### 9.1 Encode the invariants as tests (CPU, minutes, run on every change)

Every P1 in this review violates an invariant that a tiny CPU test can check. Put these in `tests/invariants/` and make them the first thing any change runs:

| Invariant | Test | Would have caught |
|---|---|---|
| Causality | For each variant, perturb tokens after position t and assert logits at positions ≤ t change by exactly 0 (train and eval mode, B=1 and B=2). | F01 thought-loop leak, F02 SSM reshape |
| Batch independence | A batch of two examples gives the same logits as each example alone. | F02, ACT batch-wide halting |
| Every parameter learns | After one backward pass, no trainable parameter has `grad is None`. | F03 ACT halt head |
| Oracle eval | A model that emits the exact reference + EOS scores 1.0 exact match, distance 0, MAE 0 on every IID/OOD condition. | F04 token cap, F13/F14 parser and distance, dyck/addition OOD definitions |
| Prompt is a token prefix | For 10k examples per task, `encode(prompt)` is a prefix of `encode(prompt + answer)`. | compare `:=` merge, the Dec-28 "Answer: " masking bug |
| No truncation | At every task's hardest phase, 1,000 examples fit in `alg_seq_len` with answer and EOS intact. | F06 / the DAG stall |
| Fresh data | Epochs 0 and 1 share no examples; `num_workers` 0 and 2 give the same difficulty histogram. | per-iterator reseeding, worker progress reset |
| Resume equivalence | Interrupt at step N and resume; batch hashes and final weights match the uninterrupted run. | F07, F09, F10 |
| Gradient-filter equivalence | With a constant gradient, accumulation 1 and 4 give the same filtered gradient and a bounded norm over 1,000 updates. | Grokfast divergence |
| RNG untouched by eval | torch, Python and NumPy RNG states are identical before and after a training-time eval. | eval reseeding |
| Finite aggregates | An all-`-100` batch yields finite loss and metrics. | NaN CE |
| Config round-trip | Save a checkpoint in one process and load it in a fresh interpreter. | F07 `TransformerConfig` unpickling |

Also keep an end-to-end smoke test: `train.py` on the nano preset for about 30 microsteps with eval, resume and `--fixed_data`, under 2 minutes on CPU, with `num_workers` 0 and 2.

### 9.2 CI and environment

- GitHub Actions on every push and PR: lint, `pytest` (a collection failure is red), the invariant tests, the smoke run. Matrix: Ubuntu and Windows, because the worker-seeding and `CUDA_VISIBLE_DEVICES=""` problems are Windows-specific.
- Pin the environment: a lock file (`uv lock` or `pip-compile`), `torch`, `transformers` and `datasets` pinned, `mamba-ssm` as an optional extra, `pytest` in dev dependencies. The review found `datasets` 4.4.1 silently dropping script datasets; a pin turns that into a visible decision.
- `pre-commit` with `ruff` and the quick test subset. A failing collection should never reach `main` again: `tests/test_long_context_parse_continuation.py` has been un-importable since commit 17a72e8.

### 9.3 Experiment hygiene

- Every run writes a manifest: git SHA plus a hash of the dirty tree, the full resolved config, measured unique parameter count, tokenizer name and hash, generator version, environment versions, and the parent checkpoint. The Dec-23 runs cannot be interpreted today because none of this was saved.
- Eval sets are materialized once, hashed, and stored next to the results; results files live next to the checkpoint they score. Never overwrite metrics; append.
- Keep every run, including failed ones, in an archive with a one-paragraph README (what was tried, what happened). The deleted Grokfast run is the clearest example of lost evidence.
- One ledger (`experiments/LEDGER.md`, committed) with one row per run: date, commit, config summary, headline numbers, verdict, link to artifacts. Hypotheses stay in a separate section until measured.
- Count optimizer steps everywhere. A "100M-token, 77k-step" run was 6,458 updates.

### 9.4 Repository structure and documentation

- `docs/architecture.md`, `docs/data.md`, `docs/evaluation.md`, `docs/curriculum.md`: one page each, with the generated numbers (parameter table, OOD grid) produced by a script so they cannot drift from the code. The README's "~18M nano" is 2.3× off today.
- `docs/decisions/`: short ADRs for choices that shaped results (embeddings excluded from weight decay; DAG thresholds 0.98–0.99; `alg_seq_len` 32→48; the "Answer:" prompt format). Each ADR names the expected effect and how it will be checked.
- Mark every number in the README as *measured* (with a run id) or *hypothesis*. The expected-results table is hypothetical today and reads as results.
- Add this review bundle under `docs/review-2026-10-26/` so the issues can link to it.
- Remove dead code (`nano_baseline.py`, `thought_use_rope`) rather than documenting around it.

### 9.5 Working with coding agents (what the transcripts show)

The 131 Codex tasks and 48 ChatGPT chats show one pattern: the code was written fast and almost never executed. 94 of 131 tasks ended with "Testing: not run (not requested)"; Codex never ran `train.py`; design pseudocode from a Dec 9 chat went into PR #1 with the thought-loop leak and the batch-wide ACT halting already in it; the `hash()` seeding, the 32-token cap, the Grokfast placement and the embedding/no-decay choice all came from prompts or advice that nobody checked by running. Concrete rules:

1. **An agent task is not done until it has run something.** Put this in `AGENTS.md` / `CLAUDE.md`: every change runs `pytest`, the invariant tests and the smoke run, and the output goes in the PR. Give the agent an environment where that is possible: the Codex cloud sandbox had no `torch`, which is why every "Testing" section says not run. A setup script that installs the pinned CPU environment fixes that.
2. **Bug fixes start with a failing test.** The fix PR adds the test that reproduces the defect, then the fix. Each issue from this review carries an acceptance test for that purpose.
3. **Never paste chat-written code straight into a PR.** Design chats produce proposals; proposals get an adversarial review (a second model told to break it, as the blind reviewers did here) before implementation.
4. **One defect per PR, one issue per defect.** Several Dec-20 PRs touched eval, curriculum and logging at once; the provenance of five defects could not be pinned because of it.
5. **Keep the prompt and the transcript.** The PR descriptions already link the Codex task; also export the transcript into `docs/agent-logs/` so the history survives the tool. The task links broke once already.
6. **Give agents the invariants list, not just the code.** An agent that knows "logit t must not depend on token t+1" writes a different thought-loop than one asked to "add latent reasoning tokens".
7. **Ask for the test plan before the code.** The first message of a task should state the acceptance test the agent will run.
8. **Blind double review for anything that changes training or evaluation.** This review's strongest evidence came from reviewers who had not seen the claims. Two cheap, independent reviews beat one expensive one that anchors on the first opinion.

### 9.6 Skills and project files to add

- `CLAUDE.md` / `AGENTS.md` at the repo root: how to run tests, the smoke run, the invariants, the definition of done, and the rule that every number is measured or labelled hypothesis.
- Project skills (`.claude/skills/<name>/SKILL.md`), each a short checklist the agent follows:
  - `smoke-train`: run the tiny CPU training loop with eval, resume and fixed data, and report the numbers.
  - `oracle-eval`: run the oracle and constant-output models through the full grid; fail if the oracle scores below 1.0.
  - `add-task`: steps for a new algorithmic task: generator, prefix-tokenization check, max-length check at every phase, OOD grid entry, eval reference check.
  - `review-against-invariants`: the §9.1 table as a review checklist for any PR touching `model.py`, `train.py`, `data.py` or `eval_hub.py`.
  - `run-manifest`: write the §9.3 manifest and ledger row for a run.
- Hooks: a Claude Code `PostToolUse` hook (or a pre-commit hook) that runs the quick invariant subset whenever `model.py`, `data.py`, `train.py` or `eval_hub.py` changes.

### 9.7 A two-week sequence

- **Week 1.** CI, lock file, `CLAUDE.md`, the §9.1 invariant tests (expected to fail on today's code), then must-fix items 1–11 from §5, each PR turning one invariant test green.
- **Week 2.** Experiments 1–3 from §6 (oracle eval, single-task learnability, the deliberate grokking test), each with a manifest and a ledger row. Grokfast and the variants wait until their tests pass.
