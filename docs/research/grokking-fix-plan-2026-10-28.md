# Grokking fix plan: blocker map and PR sequence for the first finite-set run

Date: 2026-10-28. Status: **draft, corrected after an independent red-team review on 2026-10-28.** Nothing here has been posted, committed or applied. The red-team changed: the precision guard (§2.6, Q7), the reading of the step thresholds (§3.4), the scope of PRs 2 to 7 (§4.1), the #145, #218, #256 rows (§2.3), and the open questions (§6). The blocker verdicts did not change.

Code revision: `origin/main` at `be08cde`. The worktree's `.py` files differ from main only by `docs/research/unseen_token_rows_probe.py` (checked with `git diff origin/main --stat -- '*.py'` after `git fetch`). Every `file:line` below is at `be08cde`.

Issues: all of jpturcotte/ReLSM#132 to jpturcotte/ReLSM#259 are open (checked with `gh issue view` on Oct 28). No fix has landed since the Oct 27 audit.

Items marked **(uncertain)** were not measured in this session or rest on one weak source. Section 8 lists them together.

---

## 0. Short answer

- **The fastest valid first run needs one PR: a standalone harness.** It builds `BaselineTransformer` directly, with its own task vocabulary, an enumerated split, AdamW with decay on every weight, fp32, full batch and constant LR. It needs none of the open defects fixed. It is arm A1-H of jpturcotte/ReLSM#249.
- **The train.py arm needs five PRs** before it can run the recipe and be read: optimizer switches, a precision flag with a run record, a small preset with a task tokenizer, the enumerated split with split scoring, and full-batch mode with update-based stopping.
- **The audit's order is too strict for the first run.** Most of #259 step 1 and #182 steps 0 to 2 fix the procedural stream, the generation-based eval grid, accumulation and resume. The first run uses none of them. Two items are missing from the audit's list: the vocabulary is a hard memory blocker for full batch on 8 GB, and training precision has no flag.
- **#249's own dependency list, checked:** #214, #136, #186 and #246 are real blockers for the train.py arm. #143 and #148 are not, at full batch with dropout 0 and accumulation 1.
- **Recommendation:** harness first, train.py arm in parallel, and read A1-50 against A1-H. Proposed: 8 PRs in total. PR 1 (#166), PR 2 (harness) and PR 3 (optimizer switches) can start today, in parallel.
- **The primary arm is the canonical recipe:** standard cross-entropy, Adam eps 1e-8, fp32 logits and loss, weight decay on every parameter (as in Power 2022), untied head and dropout 0 (the model defaults since 2026-10-28, `docs/model-defaults.md`; the head was tied by default when this plan was first written), full batch. The numerical guards (margin-form cross-entropy, fp64 loss, larger eps) are arms beside it, not the default. The first run asks whether the standard regime groks on this stack; a non-standard default loss would answer a different question.
- **The first sweep (owner's decision, 2026-10-28) has four core arms on the same seeds:** (1) the standard recipe; (2) standard + margin-form cross-entropy; (3) standard + `--untie`; (4) standard with `--wd_scope repo`, the repo's current exemption. Optional extras: final RMSNorm off, fp64 loss, train fraction 0.3. See §3.2.
- **Two departures from the reference recipes, stated on purpose.** Power 2022 trains with minibatch `min(512, N/2)` = 512, not full batch; full batch is Nanda 2023's setting. The #143 and #148 verdicts below hold only at full batch with accumulation 1. And the repo's model has a tied head and a final RMSNorm; Power's is untied with post-norm LayerNorm, Nanda's is untied with no norm. An untied arm and a final-norm-off arm run in the first sweep, so a negative on the tied model is not read as a negative on the recipe.

---

## 1. What the first run must do

This is the contract the blocker map is checked against. R1 to R8 come from #249. R9 to R14 come from the literature sweep in `docs/research/grokking-lit-2026-10-28/workers/*.md`.

| Id | Requirement | Source |
|---|---|---|
| R1 | `a + b mod 97`, the full 9,409-pair table, a seeded split into disjoint train and held-out halves, overlap asserted 0 | #249, #214 items 1-2 |
| R2 | d = 128, 2 layers, 4 heads, dropout 0, a task-sized vocabulary | #249, #214 items 3-4 |
| R3 | AdamW, lr 1e-3, betas (0.9, 0.98), weight decay 1.0 on all weights. "All" means every parameter, norm gains included: `openai/grok` `training.py` builds its AdamW over `self.parameters()` with no exempt group | #249, #186, #246; Power 2022 code |
| R4 | Short warmup, then constant LR; no decay phase. `get_lr` applies the warmup ramp before the schedule branch (`train.py:309-313`), so `constant` does warm up | #249 |
| R5 | Full batch, accumulation 1. This is Nanda 2023's setting; Power 2022 uses minibatch `min(512, N/2)` = 512 | #249 |
| R6 | At least 1e5 updates, counted as updates | #249, #217, #248 |
| R7 | Every 100 updates: train and held-out exact match and loss, decayed weight norm | #249 |
| R8 | 3 seeds per arm; full args and commit recorded | #249, #187, #189 |
| R9 | Decoupled AdamW (decay not added to the gradient) | ja lane, takeaway 2 |
| R10 | Logits and loss in fp32 | fr lane 3, ja lane 1, zh lane 1 |
| R11 | Log the logit margin, the fraction of exactly-zero target gradients, and Adam's `v` | fr lane 3, ja lane 1, zh lane 1 |
| R12 | Handle commutative pairs on purpose; report held-out accuracy split into "transpose seen" and "transpose unseen" | fr lane 4, de lane 4 |
| R13 | No early stopping, no plateau scheduler, no rollback on loss spikes | fr lane 5, de lane 1, ko lane 2 |
| R14 | A fast CPU smoke test before the run: the harness's own 512-pair overfit (T3) and a timed 100 updates. The de lane's MLP script (mod 97, AdamW lr 1e-3, wd 1.0, val about 0.98 by 20k steps, 0.00 at wd 0) is optional; it tests nothing in the repo | de lane 3 (re-run by that lane on torch 2.8 CPU), ko lane 2 |

---

## 2. Blocker map

### 2.1 Legend

- **H (hard blocker):** the train.py arm cannot run, crashes, or is not the recipe without it.
- **R (needed to read results):** the run executes, but some result cannot be trusted or attributed without it.
- **N (not needed for the first run):** no effect on a correct first finite-set run. A workaround is named where one is used.
- The **harness** arm (A1-H) needs none of the issues below. It builds the model, data, optimizer and eval itself. Where an issue still matters for the harness, the row says so.
- Size: **S** under about 150 changed lines; **M** 150 to 500; **L** over 500 or a redesign.

### 2.2 #249's own dependency list, checked against the code

jpturcotte/ReLSM#249 lists #214, #136, #186, #143, #148 and #246, and says arm A1-H needs none of them. Checked one by one:

| Issue | Verdict for the train.py arm | Why, at `be08cde` |
|---|---|---|
| jpturcotte/ReLSM#214 | **H** | No split: `FixedAlgorithmicDataset` draws with replacement (`data.py:1264`). The modulus comes from `mod_range` per example (`data.py:958-959`, `data.py:358-363`). The smallest preset is nano, d 512 / 6 layers (`model.py:1036`). There is no full-batch mode. |
| jpturcotte/ReLSM#136 | **H** | The train-eval loader uses `batch_size=None` (`train.py:2296-2298`) on a map-style dataset. The collator then iterates the dict's keys (`data.py:1614`) and the run dies at the first eval. `--train_eval_mode none` avoids the crash but leaves no held-out score at all. |
| jpturcotte/ReLSM#186 | **H** | `build_optimizer` exempts `tok_emb`, `lm_head` and every tensor with "norm" in its name (`train.py:1810-1817`). There is no flag. The recipe decays all weights. |
| jpturcotte/ReLSM#246 | **H** (recipe parity) | `AdamW(param_groups, lr=lr, betas=(0.9, 0.95))` with the default eps (`train.py:1844`). The recipe uses (0.9, 0.98). The eps also matters to one recent preprint (see R10 row in §2.5) **(uncertain)**. Size S. |
| jpturcotte/ReLSM#143 | **N** | `seed_all` is reached only through the in-run IID grid: `evaluate_algorithmic` → `eval_hub.py:568` (`run_algorithmic_suite`) → `utils.py:1129` (`evaluate_condition`); `eval_hub.py:664` and `750` are eval-only entry points. The finite run turns that grid off (`--no-eval_algorithmic`, `train.py:4006`), because the grid is another distribution (#162). Even with it on: at full batch with dropout 0 the RNG never enters the gradient. The only consumer is the shuffle at `train.py:2195`, and a full-batch sum does not depend on order, up to float rounding. It matters for every mini-batch arm, including a Power-style batch-512 arm. PR 6 carries #143's own test (RNG state identical before and after an eval), so the verdict cannot go stale when the new scorer lands. |
| jpturcotte/ReLSM#148 | **N** | With `--grad_accum_steps 1`, the step condition at `train.py:2729` fires on every micro-step. Nothing is dropped, and the Adam step count equals the micro-step count. **The default is 4** (`train.py:3987`); the command must pass 1, and PR 7's `--full_batch` forces it. It matters for accumulation ≥ 2 and before the #137 fix. |

Missing from #249's list, but needed for the train.py arm: #225 (vocabulary; H on 8 GB at full batch), #145 (H: the `grok` preset needs its factory fix, see §2.3), a training precision flag (new, R), the `--grad_clip` semantics (new, workaround exists), #187 (R), #215 (R), #214 item 7 (split scoring, H), #243 (eval cost, R: the train-eval block runs a generation pass at every eval) and #218 (met by PR 5's vocabulary, see §2.3).

### 2.3 Issues that bear on the first run

All rows: state **open**.

| Issue | Title (short) | First run | Reason, checked against the code | Size | Depends on |
|---|---|---|---|---|---|
| jpturcotte/ReLSM#214 | Finite-set regime cannot be run from the CLI | **H** | See §2.2. Items 1-7 are all needed; item 8 (README, run log) is R. | L (split into PRs 5-7) | #136, #186, #246 |
| jpturcotte/ReLSM#136 | `--fixed_data` crashes at the first train-eval | **H** | See §2.2. | S | — |
| jpturcotte/ReLSM#186 | Embeddings and head excluded from decay, no switch | **H** | See §2.2. With a tied head `named_parameters()` yields the table once, as `tok_emb.weight`; the `lm_head` test at `train.py:1818` is dead code. "All weights" also covers the RMSNorm gains, which the same name test exempts: 13 gains at nano, 5 for the grok preset (table + gains = 13,312 of 537,600 numel). Power 2022 decays every parameter, so `all` is the recipe's scope. | S | — |
| jpturcotte/ReLSM#246 | AdamW betas and eps hard-coded; cuDNN flags forced | **H** | See §2.2. The second half of #246 is `cudnn.deterministic = True; benchmark = False` on every CUDA run (`train.py:1935`) with no flag; PR 3 adds `--deterministic` or marks #246 advanced, not closed. | S | — |
| jpturcotte/ReLSM#225 | Llemma tokenizer adopted on a false premise | **H** at full batch on 8 GB | Measured this session: a `mod_add` example from today's fixed set is 21 Llemma tokens with EOS (note: that set uses moduli 100 to 997, never 97; `difficulty=0.5` maps to `mod_range (100, 997)`), and `encode(prompt)` is a prefix of `encode(prompt + answer)`. Full batch at train fraction 0.5 is 4,704 × 21 × 32,016 ≈ 3.2e9 logits: about 6.3 GB in bf16, **about 12.6 GB in fp32**, which is the setting PR 4 makes real, before the copies inside cross-entropy **(computed, not run)**. That does not fit on an 8 GB card. The table is also 88.6 % of a d 128 model (#214). Workaround: emulate full batch with chunked accumulation, which then needs #174. The probe batch (`train.py:2308`) is a second `alg_batch_size`-sized set held for the whole run, so the figure applies twice. | M (code: task tokenizer); S (ADR) | — |
| jpturcotte/ReLSM#145 | `create_model` overrides raise TypeError | **H** | `train.py:1971-1984` passes no width or dropout. `create_model` does `base.update(kwargs)` and then builds **every** preset eagerly with `**base` (`model.py:1033-1055`); the `300M`, `1B` and `1B-16k` entries pass `dropout=` themselves, so `create_model(size, dropout=0.0)` raises for every size, exactly as #145 says. A `grok` entry with `dropout=0` inside the dict works; a `--dropout` passthrough does not, until the factory builds only the requested preset and merges kwargs once. PR 5 includes that fix (about 20 lines) and closes #145. | S (preset + factory fix) | — |
| jpturcotte/ReLSM#215 | `--fixed_data` has no contract | **R** | The finite mode must refuse or fix three traps. `--use_task_curriculum` runs a cosmetic curriculum. The default `--total_tokens` (500M, `train.py:3787`) exceeds `--alg_tokens` (100M, `train.py:3785`), so `LanguageDataset` is built (`train.py:2229`). `--easy_mix_frac` is stored and ignored. | S | #214 |
| jpturcotte/ReLSM#187 | Run args and code version not saved | **R** | Args reach only the first row of `metrics.json` (`train.py:1946`). No commit, no dirty flag. A result cannot be attributed to code without it. | S | (#138 for the checkpoint part only) |
| jpturcotte/ReLSM#189 | Experiment hygiene, run log | **R** (run-log entry only) | #249 asks for a run-log entry saying no earlier run used this regime. | S for that part | #187 |
| jpturcotte/ReLSM#256 | Tiny-overfit check | **R** (precondition) | Cheap. If the stack cannot memorise 512 examples, no grokking result means anything. #256 tests **the training stack**: data.py labels, the collator, train.py's step, with a sabotage arm (a label shift must turn it red). PR 2's T3 runs the harness on the mod-p table and never touches data.py or train.py, so it is a harness smoke test, not #256. The #256 test ships in PR 6, once the enumerated split and the char tokenizer make it run in seconds. | S | #136, PR 5 |
| jpturcotte/ReLSM#243 | Train-eval generation pass is slow | **R** (feasibility) | The train-eval block runs `evaluate_task_accuracy` (`train.py:3461-3471`), a generation pass over `--train_eval_samples` (default 2,000) at every eval; DAG-v12 spent 61 s per eval at 800 samples. #249 asks for an eval every 100 updates: 1,000 evals per seed, hours on top of the 1-2 h of training. PR 6's finite mode skips the generation pass and scores with the teacher-forced split scorer only. | S | PR 6 |
| jpturcotte/ReLSM#149 | Gradient and update telemetry misaligned | **R** (diagnostics only) | `update_norm_est = lr × grad_norm` (`train.py:2854`) is the SGD step, not AdamW's; at least 3,000× too small on DAG-v12 (Oct 28 addendum). The primary result, the accuracy curves, does not use it. Do not read the `train/update/*` keys until fixed. | M | — |
| jpturcotte/ReLSM#218 | Unused embedding rows dominate the norm | **R** with Llemma; **met by PR 5** with a task vocabulary | #182 lists #218 "before any grokking rerun". #218's own fix list names "a compact vocabulary (~200 tokens for the algorithmic phase)", which is PR 5, so PR 5 is credited to #218. The head is tied (`model.py:704`), so every unused row gets a softmax gradient. With 31,879 unused rows the table norm says nothing about the circuit. With a vocabulary of about 100 symbols the effect shrinks to the ids no template uses; only the harness's p + 2 vocabulary has no unused rows. The telemetry split is PR 8. | M (telemetry split) | #186 |
| jpturcotte/ReLSM#174 | Accumulated loss weights micro-batches equally | **N** at accumulation 1; **H** if full batch is emulated by chunks | `loss / grad_accum_steps` (`train.py:2694`). Matters only for the Llemma workaround above. | S | — |
| jpturcotte/ReLSM#217 | Budgets count supervised tokens only | **N** (workaround) | The loop stops on supervised tokens (`train.py:2630`, counted at `data.py:1773`). At full batch every update sees the same set, so tokens per update are constant. `--total_tokens = N × (tokens per update)` gives exactly N updates. PR 7 adds `--max_opt_steps` to remove the arithmetic. | S | — |
| jpturcotte/ReLSM#248 | Warmup assumes 40 tokens per example | **N** (workaround) | `--warmup_steps` converts through `train.py:2540` and `train.py:2555`. Use `--warmup_tokens`, or PR 7's `--warmup_opt_steps`. | S | — |
| jpturcotte/ReLSM#139 | Truncation at `alg_seq_len` | **N** | Default `--alg_seq_len` is 128 (`train.py:3875`); a `mod_add` example is 21 tokens. Add a startup assert that no answer is cut (PR 6). Updated 2026-10-28: per #139's corrected history, the Dec 21–25 runs trained at 32 and their train-eval targets were cut too, so the check covers every target a scorer or gate reads, not only training examples. | S | — |
| jpturcotte/ReLSM#158 | `lang_loader` alias replays the alg stream | **N** | With `--total_tokens == --alg_tokens`, `lang_loader = alg_loader` (`train.py:2243`). In the mix band the sampler draws "lang" batches from a second iterator over the same fixed set (region B, `data.py:1714`), so the data does not change. The perplexity eval runs only in phase "language" (`train.py:3690`), which this setting never reaches. | S | — |
| jpturcotte/ReLSM#143 | Eval reseeds global RNGs | **N** | See §2.2. Needed before #252 stage 2 and #253. | S | — |
| jpturcotte/ReLSM#148 | Partial accumulation dropped | **N** | See §2.2. Needed before #137. | M | #138 |
| jpturcotte/ReLSM#213 | `--alg_batch_size` is a token budget | **N** | Under `--fixed_data` it is a literal `DataLoader` batch (`train.py:2192-2195`), so full batch is `len(train split)`. Side effect: the eval probe builds `alg_batch_size` examples (`train.py:2308-2310`); harmless with a small vocabulary. | — | — |
| jpturcotte/ReLSM#151 | NaN loss on zero-label batches | **N** | Every `mod_add` example has at least two supervised tokens. | S | — |
| jpturcotte/ReLSM#247 | No non-finite guard without GradScaler | **N** (low risk) | fp32 at 0.5M parameters and lr 1e-3. If a guard is added, it must skip only non-finite steps, not loss spikes (R13). | S | — |
| jpturcotte/ReLSM#137 | Grokfast diverges under accumulation | **N** | Grokfast is off in the first run. Needed for #254 only if #254 runs through train.py (§4.4). | M | #148 |
| jpturcotte/ReLSM#144 | Oracle-model eval regression test | **N** | It gates the generation-based grid and `eval_hub`. The first run is scored by teacher-forced exact match on the enumerated split, a new scorer. That scorer needs its own oracle test (PR 6). | M | #142, #164, #165, #178 |
| jpturcotte/ReLSM#142, #156, #160, #164, #165, #178 | Fixes gated by #144 | **N** | All six fix the generation-based IID/OOD grid. None changes teacher-forced scoring of `mod_add`. #156 affects only `compare`. | S-M each | — |
| jpturcotte/ReLSM#162 | "IID" grid is not the training distribution | **N** (turn the grid off) | The grid uses final-phase parameters and other moduli. For the finite run it is a different task. | M | — |
| jpturcotte/ReLSM#167 | Duplicate teacher-forced key | **N** | PR 6 writes new, distinct keys. | S | — |
| jpturcotte/ReLSM#242 | Eval labels wrong | **N** | PR 6 names splits `train` and `test`. | S | — |
| jpturcotte/ReLSM#235 | `best_model.pt` missing at IID 0.0 | **N** | Selection runs off the IID grid (`train.py:3292`), which is off. Use `final_model.pt` and the rotating checkpoints. Not early stopping: training continues. | S | — |
| jpturcotte/ReLSM#175, #150 | `metrics.json` rewritten in full each log | **N** (workaround) | `save()` rewrites the whole file at each log and eval (`train.py:2990`, `train.py:3711`, writer at `train.py:410`). At 1e5 updates and `--log_interval 50` that is 2,000 full rewrites. Use `--log_interval 500` or more. Resume is not used. | S | — |
| jpturcotte/ReLSM#244, #216 | Warnings silent; Unicode crash on redirect | **R** (workaround is mandatory) | Run with `--extended_logging` and `PYTHONIOENCODING=utf-8`. The `Δ` warning at `train.py:1925` fires whenever a task's exact match drops by 0.2 or more with stable token accuracy, which is what a loss spike after memorisation looks like. With stdout redirected on Windows the run then dies with no crash checkpoint (#185), and an interrupted run still writes `final_model.pt` with exit 0. The env var goes in the README command (PR 7), or PR 6 writes that line in ASCII. | S | — |
| jpturcotte/ReLSM#232 | `pad_token_id == eos_token_id` | **N** | One example per row with EOS last; #232 itself says current numbers are unaffected. A task tokenizer gets a distinct pad id anyway (PR 5). | S | — |
| jpturcotte/ReLSM#193 | Eval autocasts fp16 | **N** | Teacher-forced split scoring runs under the training context. With `--precision fp32` there is no autocast. | S | — |
| jpturcotte/ReLSM#152, #173 | Weight-norm concat; LR edge cases | **N** | 0.5M parameters; constant schedule. | S | — |
| jpturcotte/ReLSM#166 | pytest stops at collection | **N**, but do it first | Measured today: `--co -x` stops at `tests/test_long_context_parse_continuation.py` (stale import of `_prediction_from_output_ids`); with collection errors skipped, 29 pass and 3 fail. New test files can run on their own, so this is not a blocker. It is S and makes `python -m pytest` usable as the acceptance command. | S | — |
| jpturcotte/ReLSM#184 | CPU invariant suite | **N** in full; **R** for four rows | The first run needs four rows: split disjointness, optimizer groups, tiny overfit (#256), and an oracle for the split scorer. Each ships with the PR that adds the code. The other 32 rows cover paths the first run does not use. | L | #166 |
| jpturcotte/ReLSM#188 | CI and locked environment | **N** | Useful later. Signed commits go through the gh skill either way. | M | #166, #184, #168 |
| jpturcotte/ReLSM#138, #154, #185 | Resume crash; resume state; crash checkpoint | **N** | Runs are 1-2 h at most (#249 cost section); no resume. A crash loses at most one eval interval. | S-M | — |
| jpturcotte/ReLSM#140, #141 | Procedural replay; difficulty stuck with workers | **N** | Procedural path only. Use `--num_workers 0` anyway: on Windows, workers respawn at each epoch and train-eval pass (#188 comment). | M | — |

### 2.4 Issues in the audit's orders that the first run does not touch

All open. All **N** for the first run, because the run uses the baseline variant, one task, no curriculum, no language phase and no generation grid.

| Group | Issues | Why N |
|---|---|---|
| Eval grid and `eval_hub` telemetry | #161, #163, #198, #226, #227, #228, #229, #230, #194, #237, #240 | Generation grid, result JSON, `eval_hub` CLI. Not used. |
| Eval-diagnostic keys | #234, #238, #245 | Keys from `run_eval_diagnostics` (called at `train.py:3086`). Mislabelled; ignore them in the first run. |
| Curriculum controller | #157, #201, #202, #203, #204, #205, #206, #207, #219, #220, #221, #222, #223, #236, #243 | Task curriculum off. |
| Generators | #155, #176, #208, #209, #210, #211 | Other tasks. `mod_add` with a fixed modulus is enumerated, not sampled. |
| Snapshots and workers | #192, #197, #224 | Procedural path, workers. |
| Language path | #159 | No language phase. |
| Architecture and variants | #132, #133, #134, #135, #146, #147, #153, #171, #172, #190, #191, #212, #231, #233, #239, #241 | Baseline variant only; no generation, no `--compile`, no `--local_window`. |
| Docs | #169, #170, #180 | Update #170 when the `grok` preset lands (PR 5). |

### 2.5 Gaps: needed, but no issue owns them

| Id | Gap | First run | Evidence at `be08cde` | Proposed owner |
|---|---|---|---|---|
| G1 | No training precision flag | **R** | On CUDA, training always autocasts to bf16, or fp16 with GradScaler (`train.py:2356-2358`). The lanes ask for fp32 logits and loss (R10). Under autocast the logits come out of a bf16 matmul; cross-entropy's `log_softmax` runs in fp32 per PyTorch's autocast policy, so the loss is fp32 but the logits are bf16-rounded **(policy read from PyTorch docs, effect not measured)**. | Fold into #255 as its code prerequisite, or a new P3. |
| G2 | `--grad_clip 0` zeroes every gradient | **N** (workaround `--grad_clip 1e9`) | `clip_grad_norm_` is always called (`train.py:2731`). Measured on torch 2.9.1 CPU: `clip_grad_norm_(p, 0.0)` sets the gradient to all zeros. With AdamW the run then only decays. The recipe has no clipping, so the natural "turn it off" value silently breaks training. | New P3 defect, fixed in PR 3. |
| G3 | No margin, zero-gradient fraction or Adam `v` telemetry | **R** (diagnostics) | Nothing in train.py logs the logit margin or the fraction of exactly-zero target gradients. #255 asks for max logit and zero-gradient fraction as experiment logging. #218 shows `exp_avg_sq` falling below eps. | Fold into #149 (measured update norm) and #255, or one new issue "grokking diagnostics". |
| G4 | Commutative pairs not handled in any split | **R** | No split exists yet. With a random split of ordered pairs at fraction 0.5, about half the off-diagonal held-out pairs have their transpose in train (fr lane 4). | Fold into #214 item 1 and #249's metrics. |
| G5 | No check that every residue appears as a train answer | **R** | From zh lane 4: a missing residue caps test accuracy below 100 %. At 4,704 train pairs this is almost certain to hold, but it costs one line to assert. | Fold into #214 item 1. |
| G6 | No CPU smoke test of split, eval and decay wiring | **R** | R14. Neither #184 nor #256 has it. | Harness side: PR 2 (T1-T3). Training-stack side: the #256 test in PR 6. |
| G7 | Full-batch memory with the Llemma vocabulary | **H** for the train.py arm | §2.3, #225 row: about 12.6 GB of fp32 logits per full pass, twice with the probe batch. | Add as a comment on #225 and #214 (not yet posted; only #249 carries it). |
| G8 | Defaults the first run must override | **R** | `--grad_accum_steps 4`, `--lr_schedule wsd`, `--max_lr 6e-4`, `--num_workers 4`, `--train_eval_samples 2000` (wraps the set in a `Subset`, same #136 crash), `--fixed_data_size 5000`, `--task_weighting adaptive`, `--difficulty_schedule smooth`, `--eval_interval 500`, `--mix_band_tokens` (half of `alg_tokens`, so the logged phase reads `mix` for the second half of the run). | PR 7's README command lists every override; PR 6's finite-mode validation refuses accumulation > 1 and a non-`constant` schedule. |
| G9 | Harness and train.py do not share an objective | **R** | train.py's loss is a token-mean cross-entropy over the answer digits **and EOS** (`data.py:1275-1289`, `model.py:903-910`), over four prompt templates, with padding. The harness scores one answer token per example. A difference between A1-H and A1-50 is not attributable until a harness arm reproduces the EOS target and the token-mean, or PR 6's scorer reports the single-token margin beside the sequence exact match. | PR 6 reports both; a `digits` harness arm, if ever built, copies train.py's objective. |

### 2.6 Literature guards, mapped

| Guard | Status at `be08cde` | Owner |
|---|---|---|
| Decay on embedding and output head | Missing: exempt at `train.py:1810-1817` | #186; arms in #251 |
| Decoupled AdamW | **Already true**: `torch.optim.AdamW` (`train.py:1844`). A one-step test pins it: a zero-gradient weight shrinks by exactly `1 − lr·wd`. | Test in PR 2 and PR 3 |
| Logits and loss in fp32 | Missing on CUDA | G1 |
| Log margin, zero-gradient fraction, Adam `v` | Missing | G3 |
| Commutative pairs; transpose seen / unseen | Missing | G4 |
| No early stopping or plateau scheduler | **Already true**: no early-stop code; `plateau` is a fixed hold-then-cosine (`train.py:326-331`), not ReduceLROnPlateau; `constant` exists (`train.py:333`); best-checkpoint selection does not stop training (`train.py:3292`). | Nothing to fix; state it in the run record |
| 3+ seeds | In #249 | #249 |
| CPU smoke test (harness T3; #256 test of the training stack) | Missing | G6: PR 2 and PR 6 |
| Adam eps: **keep 1e-8 in the primary arm; sweep eps in #255.** Barron et al. (arXiv 2610.31642) derive a margin equilibrium of order ln(1/eps) per parameter, with λ, the parameter size and the loss's 1/N dropped inside the "≈" (§5, Eq. 10-11). At N = 4,704 the 1/N term alone moves the estimate by about 2.3 nats, more than the 1.8-nat gap between ln(1e8) = 18.4 and the fp32 cliff 16.6. They measured a mean margin of 17.9 with **their fixed loss** on a 1-layer mod-31 model (N ≈ 480); with standard cross-entropy the mean margin ended at 13.0 (Table 1). They ran eps ∈ {1e-8, 1e-12, 1e-16} only; "eps ≥ 1e-7 settles without spiking" is one untested sentence. So whether this run crosses the cliff must be measured (R11), not assumed. The guards with direct evidence are the margin-form cross-entropy (Table 1, 100 ± 0 vs 94.3 ± 5.4, 5 seeds) and an fp64 loss (Table 1, fp64 standard CE 100 ± 0; Liu et al. Fig. 1a). | eps fixed at default (`train.py:1844`); no loss option | #246 flag; margin-form CE, fp64 loss and eps arms in #255 (`--loss_fp64` from #268) |
| Expect loss spikes; log loss on a log axis; no rollback | No rollback exists | State in #249 |

---

## 3. Minimal path to the first valid grokking run

### 3.1 Three options

| Criterion | (a) Fix train.py first | (b) Standalone harness first | (c) Both: harness first, train.py in parallel |
|---|---|---|---|
| Work before the first run | PRs 3-7: about 800-1,200 changed lines in `train.py`, `data.py`, `model.py`, plus a tokenizer | PR 2: one new file of about 400 lines, plus tests | PR 2 gates the first run; PRs 3-7 run beside it |
| Touches the 4,061-line `train.py` | Yes, with no test harness (#166, #184) | No | Later, against a known-good reference |
| Open defects it must fix or dodge | #214, #136, #186, #246, #225 (or #174), #145 (factory fix), #215, #243, G1, G2, G8 | None | — |
| Fits the 8 GB card at full batch | Only with a task tokenizer or chunked accumulation | Yes: vocabulary p + 2, about 4 tokens per example | — |
| Can run on CPU while the GPU sweep runs | Impractical with Llemma | Probably **(uncertain: not timed, training was not allowed in this session)** | — |
| Tests the repo's model code (`BaselineTransformer`: RoPE, RMSNorm, SwiGLU, init, tied head) | Yes | Yes | Yes |
| Tests `data.py`, the train loop and the eval path | Yes | No | Yes, through the A1-H vs A1-50 difference |
| Risk of a false negative | High: many moving parts, none tested | Low | Low, and a negative in A1-50 becomes a localised defect |
| Long-term cost | One entry point | A second entry point to maintain | Owner decision (§6, Q1) |

### 3.2 Recommendation: (c), harness first

- PR 2 alone gives a correct first run. It answers "can this model with AdamW grok here" with nothing in the repo's data, curriculum or eval paths in the way. That is the positive control #249 already asks for (arm A1-H, lexical note §5).
- The train.py arm still matters. It is how #249 tells "the stack cannot grok" apart from "the pipeline breaks it". But it should not gate A1-H.
- PRs 3 and 4 are small and independent. Start them at the same time as PR 2.
- The harness's first version uses one input format, `token` (one token per number, the literature setting), and standard cross-entropy. A `digits` format was dropped from v1: to be a bridge it would need the repo's tokenizer, templates, EOS target and token-mean objective (G9), which is PR 5 and PR 6's work. If A1-H groks and A1-50 does not, PR 6's oracle tests and PR 3's group assertions localise the difference; the suspects, in order, are the objective (EOS and token-mean), the templates, the tokenizer and the scorer.
- The harness runs the arms that make a negative readable in the same sweep as the primary. All arms use the same seeds, fp32, full batch and decay on every parameter unless noted. The owner's first sweep (decided 2026-10-28) is four core arms:
  1. **Standard recipe** (the primary arm). Can this setup grok at all?
  2. **Standard + margin-form cross-entropy** (Barron Eq. 6). Does the softmax fix help?
  3. **Standard + tied head** (`--tie`, the previous model default). Does tying hurt grokking? Both reference recipes and Llama are untied. (Updated 2026-10-28: the model default is now untied, see `docs/model-defaults.md`, so the primary arm is untied and this arm tests the old tied setting.)
  4. **Standard with `--wd_scope repo`.** The repo's current embedding and norm exemption: was the old setting hurting us?

  Optional extra arms, if time allows: the old ReLSM model as a whole (tied head, dropout 0.1; every ReLSM run before 2026-10-28 used it), final RMSNorm off (#249's C7 arm), fp64 loss, train fraction 0.3. The SwiGLU width change does not apply at this size (512 either way).

  Each arm answers one question. If arm 1 groks, the replication works and the other arms rank the fixes. If arm 1 fails but another arm groks, that arm names the blocker. If nothing groks, the margin, zero-gradient, Adam-v and norm logs in every arm show where training stalls.

### 3.3 The minimal ordered path

For A1-H (the first valid run):

1. PR 2, the harness, with its CPU tests green (split, decoupled decay, harness overfit smoke, update-norm check, determinism with pinned threads). Then time 100 updates on CPU before choosing CPU or GPU for the sweep.
2. Run A1-H: p = 97, train fraction 0.5, tied head, decay on every parameter, fp32, standard cross-entropy, eps 1e-8, full batch, 10-update warmup, 3 seeds, 1e5 updates, eval every 100 updates. Same seeds, same sweep, the four core arms of §3.2 (standard; + margin-form CE; + untied; `--wd_scope repo`), then the optional arms (final norm off, fp64 loss, fraction 0.3) if time allows.

For A1-50 / A1-30 (the train.py arm), in order:

1. PR 3, optimizer switches (#186, #246, G2).
2. PR 4, precision flag and run record (G1, #187 part).
3. PR 5, `grok` preset and task tokenizer (#214 items 3-4, #145 part, #225 code side).
4. PR 6, enumerated split and split scoring (#214 items 1, 2, 7; #136; #215 part; G4, G5).
5. PR 7, full batch and update-based stopping (#214 items 5, 6, 8).
6. Run A1-50 and A1-30.

PR 8 (telemetry: #149, #218, G3) is needed to diagnose a failure in the train.py arm, not to read its accuracy curves. It can land beside PRs 5-7.

### 3.4 Reading the result

From #249: groks in at least 2 of 3 seeds, or not by 1e5. The step counts in the literature are not a calibration for this model:

- Every source with a stated batch size and model that groks within a few thousand updates at wd 1 is a **1-layer** transformer (Truong 2603.13331: 1.6-2.2k at fraction 0.5, batch 512; Truong 2605.18845: 2-4k at fraction 0.4, full batch; Nanda: about 10k at fraction 0.3). The only 2-layer, fraction-0.5 number is one blog run (ja-13) with no batch size and no code, whose wd 0 arm also grokked. The 2-layer, fraction-0.3 number is one seed (zh-13: first above 50 % near 21k, 90 % near 26k). Power 2022's only quantified p = 97, fraction-0.5 timing is division, at 1e5 to 1e6 steps. A 15-seed 1-layer mod-31 replication had only 8 of 15 seeds above 95 % at 30k full-batch steps.
- So 1e5 updates is the budget (Power), and no elapsed step count is a failure line. At 2e4 and 5e4 updates, **inspect**: the margin trace and the fraction of examples past 16.6, the optimizer groups, the split overlap, the transpose-seen curve, train exact match. Then finish the budget. Call a defect only on a positive sign: train exact match never reaches 99 %, transpose-unseen accuracy high from the first eval, a decayed group with zero numel, a margin stuck at the bf16 cliff of about 5.5.
- Loss spikes after memorisation are expected and are not divergence. Each one is classified by the margin log: at the fp32 cliff, or elsewhere.
- At fraction 0.5 a short or absent plateau is a valid outcome (zh lane 5).

---

## 4. Proposed PR sequence

### 4.1 Table

| PR | Scope | Issues closed (C) or advanced (A) | Files | Acceptance test (CPU unless noted) | Risk | Size |
|---|---|---|---|---|---|---|
| **1** | Make pytest collect and pass. Fix or delete the stale import test; fix the two stale tests; mark the parser failure `xfail(strict=True)` naming #165. | C #166; A #184 | `tests/test_long_context_parse_continuation.py`, `tests/test_batch_generate_no_padding.py`, `tests/test_eval_no_padding.py`, `tests/test_normalize_prediction.py` | `python -m pytest` collects every module, 0 failures, 1 strict xfail naming #165. | Low | S |
| **2** | Standalone harness `tools/grok_lab.py`, kept small (about 200 lines). Enumerated mod-p table; seeded split of ordered pairs with a transpose-seen tag per test pair, `--pairs ordered\|paired` (paired keeps (a,b) and (b,a) together); assert every residue in train answers. Model: `BaselineTransformer(TransformerConfig(d_model=128, n_layers=2, n_heads=4, dropout=0, vocab_size=p+2))`, with `--untie` and `--no_final_norm`. AdamW with `--wd_scope all\|embeddings\|repo` (repo mirrors `build_optimizer`'s exemption, as an arm), `--beta2`, `--eps`. Warmup in updates, then constant LR. No clipping. fp32 only in v1. Loss: standard cross-entropy at the answer position (default) or `--loss margin` (Barron Eq. 6) or `--loss_fp64`. Eval every k updates: train and test loss and exact match; test exact match by transpose seen / unseen; weight norm total, decayed set and table; per-example logit margin (mean, 5th percentile, fraction above 16.6); fraction of examples with exactly-zero target gradient; Adam `exp_avg_sq` min and median; measured ‖θ_t − θ_{t−1}‖ / ‖θ‖; loss-spike count. Append-only JSONL metrics; `run_record.json` with args, commit, dirty flag, torch version, device, seed, thread count, examples per update, updates per run. Checkpoint every N updates. No early stopping. One input format, `token`. | A #249 (A1-H), #251, #255, #254 (harness arm); G3-G5 | `tools/grok_lab.py` (new), `tests/test_grok_lab.py` (new) | T1: train ∪ test = all p² pairs, disjoint; in `paired` mode no pair is split from its transpose; every residue is a train answer. T2: with `--wd_scope all` the decayed numel equals the model numel; one update with zero gradient multiplies a weight by exactly `1 − lr·wd`. T3 (harness smoke, not #256): 512 pairs, wd 0, ≤ 500 updates: train loss < 0.1 and exact match ≥ 0.99, under 60 s. T5: logged update norm equals ‖θ_after − θ_before‖ within 1 %. T6: two runs with the same seed and `torch.set_num_threads(1)` write identical metrics. Then **GPU or CPU (manual)**: time 100 updates, then A1-H and its arms, 3 seeds. | Low: new code only. It tests `BaselineTransformer`, not data.py or train.py. | S-M |
| **3** | Optimizer switches in `train.py`: `--wd_scope repo\|embeddings\|all` (`repo` is today's behaviour, unchanged; `embeddings` moves `tok_emb.weight`; `all` moves the table and the norm gains); `--adam_beta1`, `--adam_beta2`, `--adam_eps`; `--grad_clip <= 0` disables clipping but still logs the norm and reports `train/grad/clipped = 0`; `--deterministic` (default off) around the cuDNN flags at 1935; always print the group sizes and store them in the hyperparameter row. | C #246; C #267; C #186 part (A) (switch; rationale rewrite stays in the issue); A #218 | `train.py` (`build_optimizer` at 1795-1844, its call at 2348-2353, cuDNN at 1935, parser near 3985-3987), `tests/test_build_optimizer.py` (new) | Default parameter-group id lists identical to today. `--wd_scope all`: no-decay group empty. Betas and eps reach `param_groups` and the hyperparameter row. `--grad_clip 0` leaves gradients unchanged (today it zeroes them) and logs `clipped = 0`. Without `--deterministic`, `cudnn.deterministic` is unchanged after setup (#246's test). Decoupled check as in T2. | Low | S |
| **4** | `--precision auto\|fp32\|bf16\|fp16` for training (fp32 = no autocast on CUDA); `--loss_fp32` and `--loss_fp64` (cast the final logits before cross-entropy; the arm #255 and #268 point at); one record file, `run_manifest.json` (the name #187 uses), written at start with: args, argv, commit, dirty flag, torch and CUDA versions, device, `torch.get_float32_matmul_precision()` (expected `highest`; nothing in the repo sets TF32), thread count, betas, eps, precision and loss dtype, parameter-group sizes, `n_used_ids`; PR 7 adds examples per update and updates per run. The same fields go in the hyperparameter row of `metrics.json`. | C #268, C G1; A #255; A #187 (file half), A #189 | `train.py` (2356-2358; the loss call at 2682 or `model.py:906`; startup near 1946) | `--precision fp32` yields a null context and a disabled scaler. `--loss_fp32` under bf16: the tensor passed to cross-entropy is float32 (#268's test). A 2-step CPU run writes `run_manifest.json` with a 40-hex commit and a boolean dirty flag, and the same fields appear in the hyperparameter row. | Low; small conflict with PR 3 in the parser | S |
| **5** | `grok` preset (d 128, 2 layers, 4 heads; dropout is now 0 by default). #145's factory fix (build only the requested preset, let kwargs override it) and the `--dropout` flag already landed in #273 on 2026-10-28; this PR only adds the preset entry. A task tokenizer selected by `--tokenizer char` with the interface the first-run path uses: `encode(text, add_special_tokens=)`, `decode(ids, skip_special_tokens=)`, `__call__(text, return_tensors="pt"\|None, truncation=, padding=, max_length=, add_special_tokens=)` returning a dict with `input_ids` (`train.py:1120, 1153`), `eos_token_id`, `pad_token_id`, settable `pad_token`, `eos_token`, `vocab_size`; `model_max_length` and `name_or_path` may be absent. A pad id distinct from EOS. If PR 6 removes the generation pass from the finite path, `__call__` can go. | A #214 items 3-4; C #145; A #225 (ADR stays separate), credited to **#218** (its vocabulary-contract remedy); C #232 for `--tokenizer char`; A #170 | `model.py` (`create_model` at 1026-1060), `train.py` (tokenizer at 1953, model at 1971-1984, parser), `utils.py` (`prepare_tokenizer` at 245-257), a new small tokenizer module | `grok` + `char` builds a model under 1M parameters with dropout 0; `create_model("nano", dropout=0.0)` no longer raises. Encode/decode round-trips every `mod_add` template. `encode(prompt)` is a prefix of `encode(prompt + answer)` for all 9,409 pairs. `pad_token_id != eos_token_id`. `tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"]` returns a 2-D tensor. | Medium: the tokenizer reaches `evaluate_task_accuracy` and the helpers in `utils.py` | M |
| **6** | Enumerated split for `mod_add`: `--mod_add_modulus`, `--mod_add_template`, `--fixed_data_split_frac`, `--split_seed`, `--split_pairs ordered\|paired`, a transpose-seen tag per test pair, residue check. Fix the train-eval loader batching for map-style datasets (#136). New split scorer: teacher-forced loss, token accuracy, **sequence exact match** (state whether EOS counts; the training loss includes it) on train and test, test exact match by transpose seen / unseen, and the per-example margin at the first answer token. **Finite mode skips the generation pass** (`evaluate_task_accuracy`, #243) and the IID grid. Finite-mode validation: `--total_tokens == --alg_tokens`, accumulation 1, `--lr_schedule constant`, task curriculum off, IID grid off, `--difficulty_schedule` ignored, `--mix_band_tokens 0`. **No truncated target anywhere** (#139, widened 2026-10-28): the startup check covers every sequence a scorer or a gate reads, not only training examples. That means the train split, the test split, the train-eval set and any probe batch: each prompt + answer + EOS must fit `--alg_seq_len`, or the run refuses to start and names the first offending example. Every run from Dec 21 to Dec 25, 2025 trained at `alg_seq_len` 32, and the train-eval targets were cut with the training data. So a check on training examples alone would have missed half of that bug. The `Δ` warning line (1925) written in ASCII. The #256 tiny-overfit test of the training stack: 512 pairs, wd 0, ≤ 500 updates, train exact match ≥ 0.99, with a `--sabotage label_shift` arm that must fail. #143's RNG-identity test. | C #136; A #214 items 1, 2, 7; A #215; C #256 (train.py test); A #243 (finite path); G4, G5, G8, G9; A #184 (rows 5, 10, 22) | `data.py` (near 1230-1311), `train.py` (2178-2303; eval block 3449-3630), `tests/test_fixed_split.py` (new), `tests/test_tiny_overfit.py` (new) | Split sizes sum to 9,409 with zero overlap; every residue is a train answer; transpose tags match a brute-force check. Oracle for the scorer: a stub model whose logits put all mass on the label scores exact match 1.0 on both splits; a constant-answer stub scores about 1/97. A tiny `--fixed_data` run (grok preset, char tokenizer) passes two evals, writes `final_model.pt`, and runs no generation (#136's own test, plus a timing assertion). Tiny overfit passes and its sabotage arm fails. RNG states identical before and after an eval. Truncation check: with `--alg_seq_len` set one token too short, startup fails and names an example from each of the train split, the test split and the train-eval set. | Medium: edits the eval block | M |
| **7** | `--full_batch` (batch = train split, accumulation forced to 1, error otherwise; `--max_opt_steps` is defined only in this mode, so #214 item 6 is done for full batch only); `--max_opt_steps`; `--warmup_opt_steps`, with LR progress read from updates when set. Record examples per update and updates per run in `run_manifest.json`. README command that lists **every** override (G8) and sets `PYTHONIOENCODING=utf-8`; a run-log row in the ledger #189 names (`experiments/LEDGER.md`: date, run id, commit, config summary, headline numbers, `opt_steps`, verdict). Run #214's own end-to-end acceptance command once. | A #214 items 5, 6, 8; sidesteps #148, #217, #248 for this mode; A #187, #189 | `train.py` (loop at 2630, LR at 2669-2676, parser), `README.md`, `experiments/LEDGER.md` | Tiny run with `--full_batch --max_opt_steps 20`: `optimizer_state_dict["state"][i]["step"]` in `final_model.pt` is 20 for every parameter; every update used exactly `len(train split)` examples; LR equals `max_lr` from update `warmup_opt_steps` on; the logged phase never reads `mix`. #214's single command: split sizes sum to 9,409 with zero overlap, one optimizer step per full pass, test-split exact match at the eval, model under 1M parameters. Then **GPU (manual)**: A1-50 and A1-30, 3 seeds each. | Low | S |
| **8** | Grokking telemetry in train.py, ported from the harness: measured AdamW update norm (replaces `lr × grad_norm`), decayed-set norm, table norm split by used and unused rows, logit margin, zero-target-gradient fraction, Adam `exp_avg_sq` quantiles. | C #149 (F27 part); A #218 (telemetry); G3 | `train.py` (2840-2900 and the eval block) | #149's own test: logged update norm equals ‖Δθ‖ within 1 % at `--grad_clip 1e-3` and `1e9`. Margin on a hand-built logits tensor. Zero-gradient fraction is 1.0 for an fp32 example with margin 30. | Low-medium | M |

Later, after A1 (not needed for the first run): #148 then #137 (Grokfast once per update, EMA from the first gradient, a recorded clip order) if #254 runs through train.py; #184 in full and #188; the measurement step (#142, #143, #160, #164, #165, #178, then #144) and #139, #141, #217, #248 before #252 stage 2 and #253 at nano.

### 4.2 Dependencies and parallel work

```
PR1 ─────────────────────────────────────────────── (independent)
PR2 ──► run A1-H (3 seeds)
PR3 ─┬─► PR5 ──► PR6 ──► PR7 ──► run A1-50 / A1-30
PR4 ─┘                 ▲
PR3 ──────► PR8 ───────┘ (parallel with PR5-7; needed to diagnose, not to read accuracy)
```

- **Start together:** PR 1, PR 2, PR 3, PR 4. PR 3 and PR 4 both touch the parser and the optimizer setup in `train.py`; land PR 3 first and rebase PR 4.
- **PR 5 before PR 6:** PR 6's end-to-end test needs the small preset and the task tokenizer to run in seconds on CPU.
- **PR 8** needs PR 3's groups for the decayed-set norm. Otherwise it is independent of PRs 5-7.
- Total: 8 PRs. The first valid run needs only PR 2. The train.py arm needs PRs 3-7.

### 4.3 Experiments after A1, and what they need

| Experiment | In the harness | Through train.py |
|---|---|---|
| jpturcotte/ReLSM#251 weight-decay sweep, tied and untied | Flags only (`--wd`, `--wd_scope`, `--untie`) | PR 3, plus an untie flag |
| jpturcotte/ReLSM#255 StableMax / precision / eps | `--loss margin\|stablemax`, `--loss_fp64`, `--eps`; ⊥Grad is an optimizer change, a small addition | PR 4 (`--loss_fp32`, `--loss_fp64`) and PR 8, plus a loss option and ⊥Grad |
| jpturcotte/ReLSM#254 Grokfast | A once-per-update EMA filter seeded from the first gradient, written in the harness; #137 is not needed. Compare against the tuned AdamW wd 1 baseline; the Grokfast README's baseline is unregularized Adam (wd 0), so its 39,890 → 910 figure does not transfer | #148 then #137, and #154, #216 |
| jpturcotte/ReLSM#252 stage 1 (A1 task, staged reveal) | Yes | PR 7 plus a staging option |
| jpturcotte/ReLSM#252 stage 2 (nano, procedural, `dag` arm) | No | #141, #157 **fixed** (the `dag` arm needs the controller working), #139, #203, #217, #248, #143, and #144 before trusting grid scores |
| jpturcotte/ReLSM#253 (single task at nano) | No | #139, #141 or `--num_workers 0`, #157 **off**, #203, #136, #144 |
| jpturcotte/ReLSM#256 tiny overfit | T3 is a harness smoke only | The #256 test of the training stack ships in PR 6 |
| jpturcotte/ReLSM#258 (loops) | No | #145, #170, A1, A5, #217, #248 |
| Weight-decay-off-at-memorisation control (report rec. 7; Truong 2605.18845: 0/3 vs 3/3 on a 1-layer, fraction-0.4, β2 0.999 model) | A `--wd_off_at_train_acc 0.99` flag, after A1 has grokked once | PR 7 plus the flag |

---

## 5. Suggested edits to the audit's ordering

These refine #259 "Order of work" and #182 "Fix order". They do not replace them for the nano work.

1. **#259 step 1 is not a gate for A1.** Keep #136, #186, #246 and #256 before the train.py arm. Move #144 with its five fixes, #143, #148, #217, #248 and #149 to "before #252 stage 2 and #253". The first run does not use the generation grid, accumulation or token budgets.
2. **#182 step 0 is not a gate for A1.** Do #166 (S). Ship the four #184 rows the first run needs with PRs 2, 3 and 6. Leave the rest of #184 and #188 for after A1.
3. **Add #225 as a hard prerequisite for full batch through train.py**, with the memory figure above (G7). #182 already lists it "before any grokking rerun"; the reason is now concrete. #182 also lists #218 there; PR 5's task vocabulary is #218's own vocabulary-contract remedy, so credit PR 5 to #218 rather than skip the gate.
4. **Add G1 (precision flag) and G2 (`--grad_clip 0`)** as small issues, or fold G1 into #255. (Done: #268 and #267.)
4a. **Record the two departures from the reference recipes** in #249: full batch (Nanda) rather than Power's batch 512; a tied head and a final RMSNorm rather than Power's untied, post-norm model. Add the untied and final-norm-off arms to the A1 sweep, not to a later experiment.
4b. **Keep the canonical recipe as the primary arm** (standard cross-entropy, eps 1e-8, fp32). The literature report's "margin-form cross-entropy as default" is an arm. Log the per-example margin; do not assume the cliff is crossed.
5. **Add commutative handling and the residue check (G4, G5) to #214 item 1**, and the transpose seen / unseen split to #249's metrics.
6. **#254 does not need #137 if it runs in the harness.** Say so in #254.
7. **#249's dependency list:** drop #143 and #148 for the full-batch arms; add #225 (or #174), #145 (or the `grok` preset), #215 and G1.

### 5.1 Additions from the audit session's review (2026-10-28)

The audit session re-checked this plan against main `be08cde` and agreed with it, including G1, G2 and the #143 / #148 / #225 verdicts. That check, the #249 comment of Oct 28, #267 and #268 were all written by the same session as this plan and cite it, so they are one source, not an independent one; the independent review is the red-team of 2026-10-28 (not in the repo). The #143 verdict needs `--no-eval_algorithmic`: the `seed_all` calls train.py can reach are `eval_hub.py:568` and `utils.py:1129`, both inside the IID grid. The audit session's own size estimate for #225 is about 4.7k examples × 20 positions × 32,016 logits ≈ 3e9 fp32 values ≈ 12 GB. It added four points:

1. **PR 6 and PR 7: no `best_model.pt` without the algorithmic eval (#235).** With `--no-eval_algorithmic`, `best_model.pt` is never written. The A1 run record must use `final_model.pt` and the split scorer's log. Both PRs should state this, and the run record should point at those two files.
2. **A1 runs set `--extended_logging` (#244).** Every warning is silent without it. Combining it with `--grokfast` on Windows with stdout redirected crashes at startup (#216). That matters only for #254.
3. **PR 5 also closes #232 for the task tokenizer.** The distinct pad id ends the pad == EOS masking for `--tokenizer char`. List it as "C #232 for `--tokenizer char`".
4. **Out of scope for A1: #212 and the variant issues.** #212 is the `local_window` NaN on padded batches. Neither touches A1, which is baseline-only.

Filed 2026-10-28:
- G2 as jpturcotte/ReLSM#267, "--grad_clip 0 zeroes every gradient";
- G1 as jpturcotte/ReLSM#268, "Training precision cannot be chosen", linked from #255.

The dependency corrections above were posted on #249. The memory figure (G7) was not yet posted on #225 or #214. #268 says "#249 specifies fp32" and #267 says "#249 uses no clipping"; both are true of the Oct 28 comment, not of #249's body, which says nothing about precision, eps, clipping, the head or commutative pairs.

---

## 6. Decisions for the owner

**Decided by the owner on 2026-10-28: all five as recommended below**, with the four-arm first sweep in §3.2.

Five decisions gate code:

1. **Primary arm = the canonical recipe?** Standard cross-entropy, eps 1e-8, fp32, decay on every parameter, untied head and dropout 0 (model defaults since 2026-10-28), full batch. The first sweep runs four core arms on the same seeds: (1) the standard recipe, (2) standard + margin-form CE, (3) standard + tied head (the previous default), (4) standard with `--wd_scope repo`; optional extras are the old ReLSM model as a whole (tied, dropout 0.1), final norm off, fp64 loss and fraction 0.3. Recommended: yes. The literature report's "margin-form CE as default" would be no.
2. **Full batch (Nanda) or minibatch 512 (Power)?** Recommended: full batch, recorded as a departure from Power. Batch 512 flips the #143 and #148 verdicts and the step calibration.
3. **Entry points.** Is `tools/grok_lab.py` a permanent second entry point for small-scale experiments (#251, #254, #255, #252 stage 1), or a throwaway positive control? Recommended: permanent but small (§4.1 PR 2); it is where #251, #255 and #254 are cheapest.
4. **Tokenizer for the train.py finite path (#225).** A small character/digit tokenizer with an ADR, or keep Llemma and emulate full batch by chunks (needs #174 and about 12 GB of fp32 logits per full pass)? Recommended: the char tokenizer.
5. **Process.** Accept relaxing #182 step 0 and #259 step 1 as gates for A1, as in §5, with #218 credited to PR 5 and the #256 test placed in PR 6? Recommended: yes.

Settled or recommended, no decision needed:

- **Tied or untied head.** The repo's model is tied (`model.py:704`); Power 2022 and Nanda 2023 are both untied. Tied as the A1 primary, so it tests the repo's model; untied in the same sweep.
- **Input format for A1-H.** `token` (one token per number). No `digits` format in v1 (§3.2).
- **Decay scope.** Every parameter, norm gains and biases included: `openai/grok` `training.py` builds its AdamW over all parameters with no exempt group and applies decay as `p.mul_(1 − lr·wd)`. The French lane reports both bias conventions grokking. **(Checked at the source; no longer uncertain.)**
- **Commutative split.** Ordered pairs (Power's convention, 9,409 pairs) with the transpose seen / unseen report; the unordered split is an arm.
- **Adam eps.** 1e-8 in the primary arm; eps is a swept factor in #255. See §2.6: the ln(1/eps) equilibrium is a leading-order scale, not a measured value for this run.
- **Precision.** fp32 for all A-series runs. It costs nothing at 0.5M parameters.
- **Clipping.** None. `openai/grok` `training.py` and `transformer.py` contain no clipping (the Lightning trainer settings are not in the repo); Nanda uses none. Log the norm.
- **Train fraction.** 0.5 primary (Power), 0.3 as an arm in the first sweep, as #249 says. At 0.5 the plateau may be short or absent.
- **Compute.** Time 100 harness updates on CPU first; the GPU sweep and the laptop's memory decide the rest.

---

## 7. What this session checked, and how

- Read #259, #182, #214, #249, #251-#256 and the summaries and Oct 27-28 comments of every issue in §2. Read `audit-2026-10-27-issue-changes.md` §(a) and §(b) from `origin/main`.
- Read the code paths: `build_optimizer`, `get_lr`, the setup in `train()`, the data loaders, the training step, logging, the eval block, `evaluate_training_split`, `MetricsLogger`, `FixedAlgorithmicDataset`, `modular_arithmetic`, `generate_example`, `CurriculumSampler`, the collator, `TransformerConfig`, `BaselineTransformer.forward` and `create_model`.
- Ran on CPU: `pytest` (29 pass, 3 fail, 1 collection error, as #166 says); the Llemma tokenizer from the local cache on two `mod_add` examples (21 tokens with EOS; prompt encoding is a prefix); `clip_grad_norm_` at `max_norm=0` on torch 2.9.1 (gradient becomes all zeros).
- Did not run training or any GPU code.

## 8. Uncertain items, for the reviewer

| Item | Why uncertain |
|---|---|
| Full-batch logits memory, 6.3 GB in bf16, 12.6 GB in fp32 | Computed from a measured 21 tokens per example (from today's 100-997 modulus set); not run. |
| Harness fits a CPU run of 1e5 updates in reasonable time | Not timed; training was not allowed in this session. |
| Whether this run's margin crosses the fp32 cliff at eps 1e-8 | Not predictable from Barron et al.'s Eq. 11 (λ, parameter size and 1/N dropped; their measurement is mod 31, 1 layer, fixed loss, mean margin 17.9). Must be measured (R11). |
| Expected grokking times | No source with a stated batch size gives a 2-layer, fraction-0.5, wd 1 number. See §3.4. |
| bf16 autocast leaves cross-entropy in fp32 but rounds the logits | PyTorch autocast policy as documented; not measured on the 4060. |
| Whether "all weights" in Power 2022 includes LayerNorm gains | **Resolved:** yes. `openai/grok` `training.py` decays every parameter. |
| #158 is harmless in the finite mode | Read from `data.py` region B and `train.py:2243`, `train.py:3690`; not run. |
| Size estimates (S / M / L) | Judgement, not counted from diffs. |

## Sources

- Issues jpturcotte/ReLSM#132 to jpturcotte/ReLSM#259, bodies and comments, as of 2026-10-28.
- `docs/review-2026-10-26/audit-2026-10-27-issue-changes.md` on `origin/main`, §(a) and §(b).
- `docs/research/lexical-grokking-2026-10.md` §2 and §5 (harness proposal, E2 arm A).
- `docs/research/grokking-lit-2026-10-28/workers/de.md`, `fr.md`, `ja.md`, `ko.md`, `zh.md` (takeaways sections).
- Code at `be08cde`: `train.py`, `data.py`, `model.py`, `utils.py`.
