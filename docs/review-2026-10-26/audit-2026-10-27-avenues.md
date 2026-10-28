# ReLSM audit 2026-10-27: research and code avenues (synthesis)

Sources: the "Research and code avenues" sections (and "dropped ideas" appendices) of the 26 reports under `<audit bundle>\reports\` (`agent-reports.md` is partial: it has no section 4 yet, so its avenues are taken from its recommendation table, rows 1-69; `artifacts.md` does not exist in the folder at the time of writing), plus `docs/review-2026-10-26/counter-review.md` section 6 ("Recommended next experiments", lines 300-317) and the `lexical.md` design critique (section 5). Report names below are the file stems. "N reports" counts the reports that proposed the avenue or a close variant.

Fixed context used for the cost and fit judgements: one RTX 4060 laptop, 8 GB VRAM, 15.7 GB RAM; nano is 41.6M parameters, 16.4M of them the 32,016-row Llemma embedding, excluded from weight decay (#186); measured nano throughput 10,259 supervised tokens/s in training windows, 100M supervised tokens = 4.8 h, peak VRAM 4.15 GB (review-files 4.2, from the DAG-v12 artifacts); every recorded run streamed procedural data with a moving difficulty and made at most ~10k optimizer updates (DAG-v12: 6,458); the standard grokking recipe was never run; the owner received that recipe on 2025-12-17 (Gemini "Grokking du Langage" report, agent-reports rows 29-35) and the Dec-24 "Curriculum Étagé" report warned that easy-to-hard curricula can block grokking (gh-issues, #157 comment).

Issue numbers refer to the 2026-10-26 review (#132-#198). "Depends on" lists the fixes that must land first, or says "standalone harness" when the avenue can bypass them by building `BaselineTransformer` directly (lexical section 5: "Running it in a standalone harness that builds `BaselineTransformer` directly avoids #136, #137, #143, #145").

---

## A. Experiments that test the grokking question directly

### A1. Run the standard finite-set grokking recipe once, at the classic scale
- **Statement.** One task (modular addition at one fixed modulus, p = 97 or 113), the full enumerated table split into a 30-60% train set and a disjoint held-out set, a d=128 / 2-layer / 4-head body with a digit-sized vocabulary, AdamW with weight decay 1.0 on all weights including embeddings, betas (0.9, 0.98), lr 1e-3, warmup then constant LR, full batch, accumulation 1, dropout 0, at least 1e5 optimizer updates, no early stopping, 3-5 seeds; log train and held-out exact match and the decayed weight norm every 100 updates.
- **Why it fits.** It is the control the project skipped. At d=128 the body is about 0.5M parameters (gemini-a probe A); claude-a1 and claude-a2 say "minutes per run"; review-files 4.2 estimates 1-2 h per seed for 1e5 full-batch updates at d=128 and 8-14 h per seed at nano, "feasible, but start small"; lexical puts E1 (60 runs of this shape) at 5-10 GPU-hours. If this groks and the procedural DAG does not, "the DAG is the problem, not the models" (gemini-a A-1).
- **Depends on.** Either a standalone harness (nothing), or in `train.py`: #136 (`--fixed_data` crash at first eval), #186 (embedding weight-decay switch), #143 (eval resets RNGs), #148 / counter-review fix 14 (budget in optimizer steps), plus three features review-files 4.2 found missing: an enumerate-and-split generator with a same-distribution held-out half (see A2), a fixed-modulus flag for `mod_add`, and a full-batch mode (today `alg_batch_size` is a token budget, `data.py:1028-1029`). A `micro` preset (d=128, 2 layers) and dropout 0 (nano has 0.1, `model.py:115`).
- **Evidence.** claude-a1 #1 (file 4, Dec 17: "try the original 2-layer, 128-dim configuration"; "weight decay 1.0"); claude-a2 #1 (chat 23d47afd: "Power et al. used WD=1.0"; the Dec-19 v4 run "differed on every axis"); gemini-a A-1 ("Nous ferons varier λ de 0.0 à 2.0" [G:46f1 L235]); gemini-b #3 (91349111 L742-744); codex-a (PR045 command `--fixed_data --fixed_data_size 3000 --grokfast --weight_decay 1.0`, "no output exists"); chatgpt-c #4 (files 42/47, the "infinite compute" recipe); claude-b1 #9 (`6e635fde` line 109: "missing two-phase training"); gh-issues (#157 comment, counter-review experiment 3); review-files 4.2 experiment 3 (ten missing pieces listed); lexical #7 and E2 arm A ("must reproduce Nanda-style grokking before anything else counts"); code-train-b #5; agent-reports rows 29-34 (Grokking-Langage L56, L57, L155-165, L184-202); counter-review §6 item 3.
- **Proposed by 13 reports** (plus the counter-review).

### A2. A same-distribution held-out split inside the repo's own data path
- **Statement.** Add an enumerate-and-split generator (all `(a, b)` pairs at a fixed modulus or fixed digit count, split by seed) and score the held-out half with the same code that scores the train half; or hold out a rule class ("famine": all pairs with `a mod 7 == 3`) from the procedural stream.
- **Why it fits.** Grokking is defined by the gap between a memorised set and a disjoint set from the same rule. At HEAD `--fixed_data` has no held-out split (lexical L4) and the "IID" grid is a different distribution (#162); the two overlap by chance 0-3% (review-files 4.2). One generator filter, CPU-only.
- **Depends on.** #136; #162 (or bypass the grid entirely).
- **Evidence.** gemini-a A-2 ("Seulement 20% des verbes réguliers (pour créer une 'famine' ...)" [G:46f1 L217]); lexical L4 and #7 ("One modulus for mod_add (97 or 113), fixed digit counts"); review-files 4.2 items 1-2 ("No disjoint same-distribution held-out set exists"); claude-a2 #1 and #9 ("for a 97² table the IID complement can be evaluated exhaustively"); chatgpt-a #7 (chat 4: "Hold out by seed (train seeds != eval seeds)"); code-train-a ("run the fixed set as `train_eval` and the procedural stream as `heldout_eval` in the same run").
- **Proposed by 6 reports.**

### A3. Weight-decay sweep on all weights, embeddings included, same seed
- **Statement.** On the A1 task, sweep wd in {0, 0.1, 0.5, 1.0, 2.0} with decay on the embedding table (tied head) and in an untied-head arm; watch for the regime where the decayed norm falls while held-out accuracy rises.
- **Why it fits.** No recorded run ever entered the norm-shrinking regime (claude-a2 #6: v4 norm 152.6 -> 409.2, DAG-v12 152.6 -> 358.1). The Dec-17 report named the sweep as the main lever and it never ran. Free at d=128; at nano the embedding is 39% of parameters and exempt (#186), so the sweep is meaningless there until B1 or #186 lands.
- **Depends on.** #186 (switch), A1 harness.
- **Evidence.** claude-c1 #6 ("An ablation at wd in {0, 0.1, 1.0} on the same seed is the obvious next step"); claude-a2 #6 (chat 23d47afd: "If weight norm is staying flat or decreasing while loss plateaus, weight decay is winning"); gemini-a A-1 (WD sweep {0.1, 0.5, 1.0, 2.0}); gemini-b #2 ("the owner's own Dec 17 report asked for a weight-decay sweep up to 2.0 that never ran"); agent-reports row 29 ("Weight-decay sweep 0.1-2.0 as the main grokking lever ... no sweep tooling in repo") and AR-2 (the Nash-SEC source said wd 1e-4, ">1e-3 eliminates grokking", the opposite of the literature); chatgpt-c #4 (file 42: "roughly 30 times the conventional weight decay"); review-files H2; lexical E1 arm (ii) and #12 ("untied head with decay on both tables ... never run in any rung").
- **Proposed by 8 reports.**

### A4. Curriculum versus fixed balanced sampling at matched compute
- **Statement.** Same task set, same model, same number of optimizer updates and examples: one arm with the DAG/difficulty curriculum, one arm with a fixed difficulty and uniform task mix. First on the A1 task (does the curriculum delay or block the transition?), then at nano on the ten tasks.
- **Why it fits.** This is the direct test of the Dec-24 warning that curricula can block grokking, and of chat 7's finding that a static mix beat every curriculum for a GPT-2-70M. The flags exist (`--difficulty_schedule fixed`, static task weights since PR #129). At micro scale it is hours; at nano a matched pair with three seeds is about 30 h (review-files 4.2 experiment 5).
- **Depends on.** A1 first (there must be a grokking arm to compare against); #141 (difficulty schedule with workers), #157 (DAG gates), #139 (`alg_seq_len` truncation) for the nano version.
- **Evidence.** chatgpt-a #15 (chat 7 research: "the best static mix achieved 2.7x lower perplexity ... than the best curriculum strategy"; "The repo never ran a no-curriculum control"); gh-issues ("Curricula can prevent grokking ... 'peuvent parfois empêcher le grokking'; 'Nobody acted on that'"); counter-review §6 item 5; review-files 4.2 experiment 5; agent-reports (Étagé L28, Éval L208, Audit-MTL L153-154: the sources themselves asked to drop binary gates); claude-b1 #14 (`--alg_tasks compare dyck parity` "The DAG that followed never ran this baseline on the fixed objective").
- **Proposed by 5 reports** (plus the counter-review).

### A5. Single-task learnability at nano, fixed difficulty, no curriculum
- **Statement.** Copy, parity, addition (and the "sanity curriculum" successor + copy) each alone, at one stated difficulty, full-length targets, accumulation 1, `alg_seq_len >= 128` for copy, three seeds, 5-20M supervised tokens each.
- **Why it fits.** Establishes what nano can learn at all before any curriculum or variant is compared. 10-30 min per run at the measured rate; "three tasks x 3 seeds under a day" (review-files 4.2).
- **Depends on.** A stated difficulty (the `fixed` schedule through the shared `Value`, `train.py:2048-2051`); no task curriculum (subsets without prerequisites never unlock, #157); #139 for copy at 12 items; #136 if `--fixed_data` is used.
- **Evidence.** counter-review §6 item 2; chatgpt-a #12 (chat 11: "a tiny 'sanity' curriculum (only successor + copy) with very small ranges ... verify end-to-end that this curriculum infrastructure can produce grokking on an easier subproblem"); chatgpt-b #5 (chat 69484386: "only parity, dyck, compare, copy, reverse ... pin difficulty in [0.1, 0.4]"); claude-b1 #14; review-files 4.2 experiment 2 (five missing specifications); gemini-a A-13 ("test the ssm variant on copy/reverse alone at fixed difficulty"); gh-prs-b ("A fixed-data grokking run on chain alone (`--alg_tasks chain --fixed_data`)"); gh-issues.
- **Proposed by 7 reports** (plus the counter-review).

### A6. Grokfast as a matched ablation, after #137
- **Statement.** Filter once per optimizer step after `unscale_`, checkpoint the EMA, accumulation 1, alpha 0.98, lambda 2 primary with 5 as a sensitivity arm, same seeds as A1, judged with a sustained-threshold metric (steps to hold 95% held-out, not first touch).
- **Why it fits.** Grokfast was the only acceleration ever planned and has never been tested in this repo (both of code-train-b's probes died at the banner; the Dec-19 run diverged, #137). Same cost as A1. Literature note: the official defaults are alpha 0.98, lambda 2; the lambda-5 default is the moving-average variant's; one public issue reports AdamW beating Grokfast+Adam (lexical §9), so the baseline must be tuned first.
- **Depends on.** #137 (filter placement), #154 (EMA in checkpoints), A1.
- **Evidence.** counter-review §6 item 4; claude-a1 #2 ("Grokfast done right"); chatgpt-a #14 (chat 11: "Start with amp = 2-3, EMA decay ~0.98 ... One with Grokfast off; one with Grokfast on"); code-train-b #5 ("the cheapest grokking experiment available on one GPU"); review-files experiment 4 and H7 ("lambda 2-3 'not 5-10'"); lexical #6 and L9 ("show that Grokfast buys anything over AdamW wd 1 on E2 arm A with 5 seeds"); codex-a (PR045 recipe).
- **Proposed by 7 reports.**

### A7. Softmax-collapse controls: StableMax / perpendicular gradient, fp32 vs bf16, max-logit logging
- **Statement.** On the A1 task, add arms with StableMax (or ⊥Grad) at wd 0, and train the control in fp32 and bf16; log the max logit and the fraction of zero gradients every eval.
- **Why it fits.** The numerical-stability account of grokking failure (Prieto et al. 2025) says logits grow until float absorption zeroes gradients; ReLSM trains in bf16 and evaluates in fp16 autocast (#193). A 0.5M model in fp32 costs nothing; StableMax is a loss-level swap with no extra compute.
- **Depends on.** A1.
- **Evidence.** claude-a1 #5 ("Log max logit and the fraction of zero gradients; try StableMax as a drop-in with wd 0"); chatgpt-b #8 (chat 6942ce5e: StableMax / perpendicular gradient, arXiv 2501.04697); gemini-a A-8 ("Softmax collapse under low precision is a documented grokking blocker"); review-files H1 (chat "Analyze code and grokking", Dec 17, "dropped from the merged doc entirely"); lexical §9 (Prieto 2025, TRM and HRM use stable-max).
- **Proposed by 5 reports.**

### A8. Tiny-overfit invariant (memorisation in minutes)
- **Statement.** A fixed set of 256-1024 examples, nano or smaller, 200-500 updates on CPU or GPU; assert train loss < 0.1 and teacher-forced exact match near 1.0. Ship it as a test in #184.
- **Why it fits.** Grokking needs memorisation first; the repo never checked it can memorise. chatgpt-b says it "would have caught the double-shift (B-3) and the prompt-digit loss (Dec 25) in minutes". Minutes, no GPU required.
- **Depends on.** #136 (or a standalone loader).
- **Evidence.** chatgpt-b #1 (chat 6945b243, Dec 19: "Create a small fixed train set of 256-1024 examples. Track training loss vs eval loss over time. If training doesn't hit near-zero loss quickly, something is wrong"); claude-a2 #7 (chat 23d47afd: `--fixed_data_size 500 --max_lr 3e-4 --alg_tokens 20M --eval_interval 200 --eval_samples 200`, "Never run"); counter-review 9.1 (CPU invariants); code-train-b #2 (CPU determinism is bit-exact, so the hash test is cheap).
- **Proposed by 4 reports.**

### A9. Repeated-subset mixing (two-set training) as the bridge between the stream and grokking
- **Statement.** Mix a fixed repeated subset (say 25%) into every epoch of the procedural stream and vary the fraction; `--persistent_alg_frac` (`train.py:3868`, default 0.15) is the hook.
- **Why it fits.** Reconciles the infinite stream the DAG needs with the repetition grokking needs; the sampler already mixes loaders by probability. Nano cost per arm is one 100M-token run (4.8 h), or minutes at micro scale.
- **Depends on.** #136; #158 (the `lang_loader` alias already replayed 44% of DAG-v12's second half by accident, so the "replay" arm must be deliberate).
- **Evidence.** claude-a1 #3 (file 4: "select a random 25% of your training data as a repeated subset sampled every epoch"); chatgpt-c #5 (file 46, Charton and Kempe: "mix a repeated subset with fresh examples"); code-train-b #3 ("a deliberate replay buffer ... could be tested against no replay, on the fixed-data path"); agent-reports row 20 (Nash-SEC L289-294 replay buffer, capacity 1000, "none; everything is sampled fresh").
- **Proposed by 4 reports.**

### A10. GrokTransfer / two-stage transfer
- **Statement.** Train the micro model to ~70% on the A1 task, transfer its embedding table (or the whole body) into nano, and measure whether the delay to generalisation shrinks.
- **Why it fits.** Cheap once A1 works; it is the ladder the project wanted (micro -> nano). Cited as "eliminate delayed generalization in some cases".
- **Depends on.** A1 groks first; B1 if the two models must share a vocabulary.
- **Evidence.** claude-a1 #4 (file 4: "Train a smaller model to non-trivial accuracy (~70%), then transfer its embedding layer"); chatgpt-a #16 (chat 11 research, Chen & Wang 2025 as cited); chatgpt-b #8 ("embedding transfer from a smaller model"); agent-reports row 52 (Ramping L40-43: "none"); gemini-b #10 (lowest fit until there is a teacher).
- **Proposed by 5 reports.**

### A11. The PR #196 lexical-grokking designs E1-E4, with the critique's corrections
- **Statement.** E2 arm A as the positive control (Nanda-style modular addition in the standalone harness), then E1 (paradigm grokking) with the identifiability ceiling computed from the data, an untied-head arm and a final-norm on/off arm; E3 (two-hop relations) as the calibration test for `shared_loop` once #145/#135 are fixed; E4 last.
- **Why it fits.** These are "the only written grokking experiment plans in the repo" (gh-prs-b). Costs from the designs: E1 5-10 GPU-hours for 60 runs; E3 one GPU-day; E4 most expensive.
- **Depends on.** Nothing for E2-A/E1 (harness); #145 and #135 for the loop arms of E3; fold in the three Codex review comments on #196 (RMSNorm-scale decay identical across arms, mini-batch E3, test the row-reset assumption).
- **Evidence.** lexical section 5 and #9-#10; gh-prs-b ("PR #196's E1/E2 designs are the only written grokking experiment plans in the repo. The three Codex review comments on #196 should be folded in before anything is run").
- **Proposed by 2 reports.**

---

## B. Curriculum and data design

### B1. A small vocabulary for the algorithmic phase (digit + operator + template words, <= 256 to 1k ids), or a logit mask
- **Statement.** Replace the 32k Llemma tokenizer with a few-hundred-symbol tokenizer for algorithmic-only runs (`--tokenizer` already exists, `README.md:46`), or mask the logits to the 137 ids the curriculum uses; a 16k or factorized table is the fallback if a language phase is kept.
- **Why it fits.** Cuts nano from 41.6M to about 25M, moves the embedding share from 39% to under 2% (gemini-c probe: 16.4M -> 0.5M at d=512), makes "weight decay on all weights" a small matter, removes the `▁` prefix token (#178), the `:=` merge (#156) and the junk-row drift that dominates the logged embedding norm (lexical L1), shortens every example ~20% (helps #139/#142) and speeds the `lm_head`. code-model: "the cheapest change with the largest effect on the grokking telemetry".
- **Depends on.** Nothing blocking; tokenizer file plus `--tokenizer`. Interacts with #186 (makes the switch almost moot) and #156/#178 (fixes them at the source).
- **Evidence.** claude-a2 #2 (chat 090b121f: "Consider character-level or digit-level tokenization"); claude-c1 #4 ("a small custom vocabulary (<= 256 symbols) would cut parameters ~40%"); code-model #1 ("137 of 32,016 ids are used"; the owner's #186 comment quoting Gemini's "Embedding Tax"); gemini-a A-1 and §5.7 ("the real fix is a ~100-token digit vocabulary for the algorithmic phase, not 16k"); gemini-b #1 (91349111 L620, LRF/ALBERT factorization); gemini-c #2 ("Vocabulaire cible de 16 384" [L549]; "Pure Digit ... Efficience Paramétrique (50M): Excellente (<2%)" [L499]); gh-issues ("A digit-plus-ASCII tokenizer (a few hundred tokens) would shrink nano to about 25M"); lexical #1; agent-reports row 35 (Grokking-Langage L54-55; P L2454 "digit-level, not BPE").
- **Proposed by 9 reports.**

### B2. Curriculum controller redesign (one bundle, because the pieces interact)
- **Statement.** (a) a hard floor at `init_difficulty` and hysteresis (step down only after k consecutive sub-band evals); (b) count cooldown, eval interval and EMA decay in optimizer updates, with a floor so a sweep cannot switch the EMA off; (c) bias-corrected EMA instead of the first-nonzero snap; (d) gate on competence at a stated difficulty measured on a fresh balanced probe, not on the moving train-eval; (e) learning-progress sampling or a separate unlock EMA / root-mastery gating instead of binary gates; (f) two-sided jitter around the frontier; (g) log `curriculum.weight.<task>` and `curriculum.prob.<task>`; (h) re-run the DAG once with the never-merged 70/50 band and gates at 0.90 to see whether the Dec-29 stall was the band/gate conflict or truncation.
- **Why it fits.** Zero compute; every item is a few lines in `curriculum.py` / `train.py`; several are half built (`ema_loss` stored but never read, `curriculum.py:82-88` vs `:110-139`). It only matters if the curriculum is kept at all (see A4).
- **Depends on.** #157 (the gate deadlock), #141, #143; decide A4 first.
- **Evidence.** chatgpt-b #4 (chat 694b4280, Dec 24: floor-escape rule, learning-progress sampling, confidence weighting, KL cap); claude-b1 #3-#7, #13 (`6e635fde` lines 39, 56-58, 245-359; `b823ca58` lines 453, 531-536); claude-c1 #1-#2 (the Dec-22 log: train-eval addition 61% while IID addition 1.2%); claude-c2 #1-#3, #5 (bias-corrected EMA; "four consecutive 0.05 drops in 250 optimizer updates"); codex-b #2, #3, #7 (#109 "Alternative 1" and "Alternative 3"; #97's decay formula clamps to 0 at 2000; "Re-apply #107 as an experiment"); code-data (gate on competence; "do not step the controller for a prerequisite while a dependent task's streak is non-zero"); gh-prs-b ("Make the controller band and the DAG gates one design"); review-files H3 (owner: "Let me think about it") and H9; agent-reports AR-4 (two-sided jitter per Compass L259-266 and Ramping L38) and rows 17, 21-22, 39, 42, 60 (patience/inertia, 10% floor, learning-progress weighting, plateau-triggered advance, soft gates: all "No").
- **Proposed by 10 reports.**

### B3. Carry-structured difficulty and intermediate-trace supervision
- **Statement.** Make successor and addition sample the trailing-9 run and the number of carry columns explicitly (as chain already does, `data.py:765-776`), and add an optional trace target (carry chain or per-digit partial sums) so latent step k can be supervised on step k; carry pressure becomes an OOD axis separate from length.
- **Why it fits.** Turns the 32-digit "copy task" back into a carry task so the 48-token window can stay; the generators already compute the intermediate values; a known route to length generalisation on addition. Costs tokens (interacts with #139).
- **Depends on.** #139 (`alg_seq_len`), #155 (addition bands).
- **Evidence.** code-data ("Make the 'hard' phases hard in the intended way ... The comment at data.py:58 states the intent"); gemini-a A-6 ("MSE(h₁, Step₁) + ..." [G:5905 L1481-1527]); gemini-b #7 (834815ac L276-282, Least-to-Most 99% vs 16% on SCAN); agent-reports rows 37-38 (Nash-SEC L444, Ramping L33-36, Pondéré L126-127: "vary carry presence, not digit count"); codex-a ("OOD-carry: digits=4, digit_bounds=(7,9)" from PR018, dropped in PR66's rebuild); gh-prs-b ("Carry-propagation chain as the grokking task", #114's `999->1` format); chatgpt-c #7 (file 42: test "what the model is asked to predict and how structure is presented before attributing failures to insufficient reasoning depth").
- **Proposed by 7 reports.**

### B4. Reversed-digit (least-significant-first) answers for successor, addition, chain
- **Statement.** Emit numeric answers right-to-left in the target string so the carry propagates in reading order.
- **Why it fits.** A target-string change in `data.py`, zero compute; removes the carry look-ahead that makes decimal addition hard for a causal LM. The Gemini report rejected it only for a mixed maths-and-language model, and ReLSM in practice has no language phase.
- **Depends on.** Nothing; pairs with B3 and the parser (#165) must accept the reversed form.
- **Evidence.** gemini-c #1 ("Certaines études suggèrent d'inverser les nombres lors de la tokenisation (ex: 123 devient 3, 2, 1)" [G:ba6c L478]; "never mentioned again in any chat or PR"); claude-b1 #10 (a `reverse_order` format-OOD condition was in the Dec-20 coding prompt, `df40d43e` line 1744, and dropped).
- **Proposed by 2 reports** (one as training format, one as an OOD condition).

### B5. A task-structure report at startup, and the empirical task mix in the log
- **Statement.** Before training, print per task and phase: carry rate, operand-length mismatch rate, trailing-9 distribution, distinct-example support, class balance and shortcut ceilings; during training, count examples per task over the last 1k batches.
- **Why it fits.** CPU-only, seconds; would have exposed the 98%/2% task split (counter-review 4.1), the small supports in `uniform(0, 0.3)` (code-data D4) and #139/#155/#156 before any GPU hour was spent.
- **Depends on.** Nothing.
- **Evidence.** code-data ("A task-structure report as a startup artifact"); chatgpt-a #13 (chat 11: "Log the empirical task mix: counts per task over, say, 1k batches"); cli-commands #4 (the run record should carry "the measured quantities (examples per batch, supervised density, optimizer-step count), not the raw flags").
- **Proposed by 3 reports.**

### B6. Tasks and probe sets that separate "can compute" from "can echo"
- **Statement.** Count-balanced dyck corruptions (swap one `(` with a later `)` so counts stay equal); hard negatives near boundaries (many carries, deep nesting); counterfactual sets for compare/arithmetic (change one operand, add a distractor); tasks never built: kth-token retrieval, Dyck-2, subtraction, sort, permutation / exactly-one-selection tasks scored by a checker; mixed-length operands as an OOD axis.
- **Why it fits.** Generator-level, CPU, no new model code; exact match currently hides operand echoing (88% invalid compare outputs) and dyck measures counting, not depth (#176), so the recursion hypothesis cannot be tested with the present tasks.
- **Depends on.** #176 (dyck), #162 (grid), #165 (parser).
- **Evidence.** gh-issues ("Count-balanced dyck corruptions (#176)"); chatgpt-a #7-#8 (chat 4: "Hard negative cases (near-boundaries ...)"; "select the kth token", "Dyck-2 with two bracket types"); chatgpt-c #3 and #13 (file 37 "multiple-valid-output tasks"; file 45's counterfactual table); codex-a (PR017: "Optionally add Dyck-2 later (two bracket types) as OOD"); code-data ("Mixed-length operands as a cheap generalisation test"); agent-reports row 67 (Subitizing, Sort: "none").
- **Proposed by 6 reports.**

### B7. Prompt-format and tokenization invariants before any model comparison
- **Statement.** A unit test that `encode(prompt)` is a prefix of `encode(prompt + answer)` for every template and target form; a train/eval prompt diff tool; a 10-minute per-checkpoint ablation with and without the `Answer:` marker and with multi-template prompts on/off.
- **Why it fits.** The `:=` merge (#156), the empty-output cohort on `=`/`?` prompts and the Dec-28 trailing-space drift are all format effects that were read as learning effects. CPU, minutes.
- **Depends on.** Nothing; belongs in #184.
- **Evidence.** chatgpt-b #2 (chat 6945e825: "Print 5 train samples + 5 eval prompts per task and diff them"); claude-c1 #3 (the `=`-cohort: `stop_reason=eos pred_len=0`); claude-c2 #4; gemini-c #5 ([G:d75f L114, L404]); code-eval #5; gh-prs-a (#16's multi-template prompts "removed later ... A cheap ablation: templates on/off at nano scale").
- **Proposed by 6 reports.**

### B8. Padding done right: exact-length bucketing, a distinct pad token, an explicit attention mask, sequence packing
- **Statement.** Bucket training batches by exact length (as eval already does), add a `[PAD]` id or derive the mask from the collator instead of token identity (pad == EOS today), and optionally pack sequences with a block-diagonal mask.
- **Why it fits.** Less padding is less VRAM and more supervised tokens per step (padded/supervised ratio 4.5-5.8, codex-b probe P1); restores the flash SDPA path; removes the EOS-masking side effect. Sequences are 15-25 tokens with few distinct lengths, so fragmentation is small.
- **Depends on.** #172, #126 follow-up; #139 (decouple truncation limit from batch budget).
- **Evidence.** claude-b2 #1-#2, #6 (Dec 31 chat lines 553-580, 643-717: "Use exact length as bucket key ... no padding needed -> no mask needed"); codex-b #4 and #10 (#120 "Do not implement Sequence Packing"; distinct pad token); gh-prs-b ("Distinct pad token ... removes the dense-mask path from every training batch"); gh-issues (#139 comment 2: `alg_seq_len` 128 doubles examples per micro-batch, 108 -> 220); code-data (decouple the three uses of `alg_seq_len`).
- **Proposed by 5 reports.**

### B9. Easy mix drawn from a large-support phase, not `uniform(0, 0.3)`
- **Statement.** Draw the easy-mix and jitter samples from a fixed "easy but not tiny" phase per task (parity length >= 8, compare 3 digits) so the regulariser cannot be memorised.
- **Why it fits.** The small supports live in `uniform(0, 0.3)` (code-data D4); one line.
- **Depends on.** Nothing.
- **Evidence.** code-data ("Easy mix from a large-support phase"); agent-reports AR-4 (jitter is one-sided easy replay).
- **Proposed by 2 reports.**

### B10. Lexical-noise stream and a real-data anchor (only if a language phase is revived)
- **Statement.** Wire the `lex_loader` (5% short real-text snippets, 64-128 tokens) or delete the dead flag; keep >= 10% real text from the start of any language phase.
- **Why it fits.** The only mechanism that would touch the 32k rows the algorithmic data never uses (#196); data, not compute. Not a grokking experiment; lowest priority here.
- **Depends on.** #158/#159 (language loader); decide B1 first (a small vocabulary removes the reason).
- **Evidence.** chatgpt-a #10 (chat 7 prompt 3 is the spec); codex-a (PR014: "lex_loader: optional lexical noise (short real text snippets just to exercise the full vocabulary)"); gh-prs-a; gemini-b #5 (af708951 L175-179: "retain a minimum of 10% high-quality ... real data"); gemini-a A-14; claude-a1 #14 (the creative-writing trace corpus).
- **Proposed by 6 reports.**

---

## C. Model and optimizer

### C1. Compute-matched `shared_loop` ladder
- **Statement.** Expose `n_unique_layers` / `n_unroll`; compare 1, 2 and 4 passes at equal parameters and equal training budget, with full versus truncated backprop; the TRM corner (2 unique layers x 8-16 unrolls) and the parameter-matched corner (1-3 unique layers x 6 unrolls vs the 6-layer baseline); progressive or stochastic loop depth during training.
- **Why it fits.** `shared_loop` is hard-wired 6 x 24 (#145), 4x the FLOPs of the baseline at equal parameters; nothing recorded compares the variants at matched compute. Nano fits the 4060; smaller unique depth makes steps faster. This is the project's central claim, but it should be run only after A1 shows anything in this stack can grok (code-model #7: only `baseline` and `shared_loop` are clean enough to carry a result today).
- **Depends on.** #145 (overrides raise TypeError), #170; A1 first.
- **Evidence.** chatgpt-c #1 (file 38: "partir d'un noyau simple à 1, 2 et 4 passes, avec contrôles à paramètres égaux et à budget d'entraînement égal, puis comparer gradients complets et tronqués"); chatgpt-a #2, #18 (chat 6: "same FLOPs budget? Then fix unroll and compare accuracy"); chatgpt-b #9 (chat 6a56e38e L1-L4 ladder); claude-a1 #8, #12 (file 1: "Start with 2 loop iterations, increase to 4 at 50% training"); code-model #4; gemini-a A-7 ([G:5905 L864]); codex-b #8 ("A 3-variant x 1-seed smoke at nano fits one GPU-night"); review-files H4; lexical #10 (Kohli 2026 as the setting with a known answer).
- **Proposed by 9 reports.**

### C2. Per-iteration readout and test-time K scaling for latent / act
- **Statement.** Decode at every k on the eval set; add `--eval_K` / `--eval_max_K` and evaluate a trained latent checkpoint at K in {1, 2, 4, 8, 16} on longer or deeper OOD inputs; make OOD length extrapolation the headline metric.
- **Why it fits.** Eval-time only, ~20 lines; the only experiment that distinguishes the recursive variants from the baseline on the project's own terms.
- **Depends on.** #135 (the thought loop is non-causal, so Exp2-4 accuracies are not comparable until fixed), #132 (halt head never learned), #142/#162 (grid).
- **Evidence.** chatgpt-a #3 (chat 1: "whether the model is improving over iterations"); gemini-a A-4 ("Allow the loop to run for up to 30 steps. Hypothesis: The model should generalize ... by simply 'thinking longer'" [G:5905 L1575-1581]); claude-a1 #11 (file 2: "Test on sequence lengths longer than trained ... The ReLSM (being recurrent) should theoretically extrapolate better").
- **Proposed by 3 reports.**

### C3. Supervised halting, deep supervision, per-position ACT with a step embedding
- **Statement.** Give the halt head a target (BCE: 0 until the step where the answer is correct), add intermediate-output losses, and move halting to per-position with a step embedding so it has a gradient through the accumulated output.
- **Why it fits.** Fixes #132 by construction; one extra head loss, no extra forward passes.
- **Depends on.** #132, #135.
- **Evidence.** chatgpt-a #1 (chat 1: "supervise it to stay 0 until the final step when the answer is correct"; "compute loss at intermediate outputs as well"); claude-a1 #6 (file 1 STRM: "Halt gate: predicts per-position halting probability", "Step embedding"); gemini-a A-6; agent-reports row 57 (anneal the ACT threshold: "moot under #132").
- **Proposed by 3 reports.**

### C4. A causal thought loop (per-row causal window or interleaved latent tokens)
- **Statement.** Restrict thought states / kv to the prefix per position (gather on `cumsum` of the attention mask instead of `h[:, -local_window:, :]`), or interleave COCONUT-style latent tokens in the sequence.
- **Why it fits.** The fix for #135 by design; cheap first step. Until then the latent variants are non-causal models.
- **Depends on.** It is the #135 fix.
- **Evidence.** claude-a1 #7 (file 1 NEXUS, file 2); code-model #3 (`model.py:730-731`: "the current window is 'the end of the padded tensor'").
- **Proposed by 2 reports.**

### C5. Optimizer knobs the grokking recipes assume: beta2, embedding LR, Adam eps, Coupled Adam
- **Statement.** `--adam_beta2` (0.98 for replication; HEAD uses 0.95, `train.py:1848`), an embedding-specific learning rate, a vocabulary-mean second moment for the table (Coupled Adam), and the eps regime for rare rows.
- **Why it fits.** Zero cost flags; removes unlogged confounds before any sweep; Coupled Adam is "the only candidate that attacks the mechanism rather than its symptom" of unused-row drift (lexical). The beta2 0.95 vs 0.999 rare-token effect is a single-author preprint (unverified).
- **Depends on.** Nothing.
- **Evidence.** code-train-a ("Power et al. used (0.9, 0.98) and decayed all weights ... a `--adam_beta2` flag costs nothing"); codex-a ("Optimizer betas/eps as flags ... Zero cost"); lexical #4-#5 and §9 (AlQuabeh 2025: embedding LR; TRM lr 1e-2 for embeddings; Stollenwerk 2025).
- **Proposed by 3 reports.**

### C6. Embedding-table hygiene from the lexical sweep: gate unused rows, first-use init, freeze used rows, untied head
- **Statement.** Behind a flag: a usage mask updated per batch, a gradient hook on `tok_emb.weight`, first-use init from the final-norm hidden state with per-row clearing of Adam state and the Grokfast EMA; a "freeze used rows in the language phase" arm; an untied head with decay on both tables.
- **Why it fits.** The sweep found gating plus first-use init best (rung 2: 0.745 vs scratch 2.519); one boolean vector. Only matters while the 32k table is kept (see B1).
- **Depends on.** #186 decision; B1 makes most of it unnecessary for algorithmic-only runs.
- **Evidence.** lexical #2, #8, #11, #12 and section 5 "reset recipe" (rung 2 `P1_P2a`; Jia 2026 relapse 19.40% -> 0.07%; "mean of used rows plus noise ... never run"; "untied head with decay on both tables ... never run in any rung").
- **Proposed by 1 report** (gh-prs-b points at the same PR #196 designs).

### C7. Final RMSNorm on/off arm versus norm-based grokking diagnostics
- **Statement.** Run the A1 control with and without the final norm (`model.py:899`) and see whether the weight-norm signature appears at all.
- **Why it fits.** If a final norm removes the weight-norm timescale law (Truong 2026, as cited), ReLSM's norm telemetry cannot show the grokking signature; minutes per run in the harness.
- **Depends on.** A1.
- **Evidence.** lexical #3 (note §3.4: "A final LayerNorm removes the dependence").
- **Proposed by 1 report.**

### C8. Backbone and position controls: RoPE-nano, a tiny pure SSM, Mamba on copy alone, an external reference model
- **Statement.** A RoPE-vs-learned-position nano under one harness (replacing the dead `nano_baseline.py`, #180); a tiny SSM without the thought loop; the ssm variant on copy/reverse alone; SmolLM2-360M or Qwen2.5-0.5B through `eval_hub` on copy/successor/compare as an upper-bound and harness calibration.
- **Why it fits.** Inference-only or small; separates backbone from loop effects; calibrates the tasks (#144).
- **Depends on.** #133/#134 (SSM scan) for the SSM arms; #144 for the reference model.
- **Evidence.** chatgpt-a #6, #17 (chat 3: "Off-the-shelf open model (good 'reference / upper bound')"; chat 6: "Add a RoPE-nano baseline"); claude-a1 #13; codex-a (PR008: "test the hypothesis that absolute embeddings fail OOD length generalization", never run); gemini-a A-13 ("Mamba struggles with 'Copying' and 'Associative Recall'" [G:5905 L1670]).
- **Proposed by 4 reports.**

### C9. Loop-stability details parked in Dec 2025: orthogonal thought init, sandwich norm, consistency loss, self-distillation across iterations, MLP core ablation
- **Statement.** Orthogonal init for thought tokens (still `N(0, 0.02)`, `model.py:694-695`), a post-residual norm in the loop, an auxiliary loss for smooth progression across steps, later iterations teaching earlier ones, and a plain-MLP thought core as a control.
- **Why it fits.** Lines of code; meaningful only after #135 is redesigned.
- **Depends on.** #135, #132.
- **Evidence.** chatgpt-a #5, #9 (chats 7/16: "Orthogonal init: yes, adopt it"; TRM "replacing self-attention with a plain MLP improved accuracy"); claude-a1 #9 (file 2: "later iterations teach earlier ones"); agent-reports rows 55-56 (Bugs L33-38: "No"); review-files H5.
- **Proposed by 4 reports.**

### C10. Layer reset / shrink-and-perturb at phase transitions, cosine warm restarts
- **Statement.** Reset the last layer or shrink-and-perturb (alpha 0.9) when a task is stuck at difficulty 0.2, with a warm LR restart.
- **Why it fits.** Cheap on nano; aimed at the stuck tasks; untested here and the sources' numbers are unsourced (claude-b1 Appendix F).
- **Depends on.** B2; A4 first.
- **Evidence.** claude-b1 #8 (`6e635fde` lines 445-493); agent-reports rows 23, 47 (Nash-SEC L267, Compass L288-335: "none").
- **Proposed by 2 reports.**

### C11. Far-field model ideas recorded so they are not lost
- Multi-token / token-order prediction head, MIR auxiliary, LAW/Loopholing recurrence, Rho-1 selective token loss, VQ concept auxiliary on thought tokens, mode-seeking distillation, memory tokens + local KV window, thought beams, COCONUT latent-step curriculum. None fits the current grokking question; all are nano-affordable.
- **Evidence.** chatgpt-c #7-#9 (files 36, 42, 47); chatgpt-a #19 ("Phase 2" items); claude-a1 #10; gemini-a A-12 ([G:60df L81]); gemini-b #10.
- **Proposed by 5 reports** (as parked items).

---

## D. Measurement and tooling

### D1. Oracle and constant-output model through every eval path, as the first acceptance test
- **Statement.** Run a perfect answerer and a majority-class model through `eval_hub` and through both in-training paths (teacher-forced `evaluate_training_split`, generation train-eval with `_normalize_text`); expect exact 1.0 / distance 0.0 / MAE 0.0 / 0 `max_new_tokens` stops for the oracle and the stated majority-class scores for the constant model; ship as `tests/test_oracle_grid.py`.
- **Why it fits.** Under two minutes on CPU, no model (`oracle_probe.py` already exists); nothing else should be read before it passes. It cannot pass before #142, #160 (or pin `PYTHONHASHSEED`), #164 are fixed, so it is an acceptance test for those fixes.
- **Depends on.** #142, #144, #156, #160, #164, #178.
- **Evidence.** counter-review §6 item 1; code-eval #1 ("A real oracle test is cheap and already written"); code-train-a (`p5_train_eval_oracle.py` "exposes #156, #164, #178, #198 and T2 in one run"); gh-issues ("the oracle currently scores 0.779"); review-files 4.2 experiment 1 (the constant-model definition is missing: dyck 0.51, parity 0.5, compare near 0.5); codex-b #9 (make #106's crash detector a test); chatgpt-c #11.
- **Proposed by 6 reports** (plus the counter-review).

### D2. Teacher-forced answer-token exact match as the memorisation monitor
- **Statement.** Rebuild `_evaluate_numeric_answer` (`train.py:1108-1140`, dead) so it encodes prompt+target as one string and compares the tail, generalise it to all tasks, log it per task at every eval next to the generation metric.
- **Why it fits.** One forward pass, no decoding, no parser; it is the memorisation curve grokking needs and halves eval time (eval was ~25% of DAG-v12's wall time, claude-c1 #5).
- **Depends on.** #156 (tokenization boundary) so the tail compare is exact.
- **Evidence.** claude-a2 #3 (chat 23d47afd "Probability-Based Evaluation"); codex-a (PR053: "Generation is slow ... check if the model assigns high probability to the correct answer"); gh-prs-a ("the cheapest exact-match signal available"); chatgpt-c #11 ("teacher-forced per-task loss as the gate signal would remove the 64-83-sample exact-match noise in the DAG controller").
- **Proposed by 4 reports.**

### D3. Budget and schedule in optimizer updates; log supervised tokens, processed tokens and examples per update
- **Statement.** Define runs by `opt_step` and a fixed dataset size; make `eval_interval`, `curriculum_cooldown` and the EMA decay multiples of `grad_accum_steps`; log both token kinds (supervised density is ~12.5%) and FLOPs; fix the 40-tokens-per-step `--warmup_steps` estimate.
- **Why it fits.** The literature's "steps x batch" accounting cannot be compared with "100M supervised tokens" (DAG-v12 made 6,458 updates); the step estimate is off 4x; every curriculum timer is in micro-steps. Two counters in `CurriculumSampler.next_batch`.
- **Depends on.** It is counter-review fix 14 / #148; nothing blocks it.
- **Evidence.** code-train-b #1 ("Budget in optimizer updates, not answer tokens"); codex-b #1 and #3; cli-commands #1 ("the budgets are supervised-token budgets at ~12.5% density"); gemini-c #4 (GC-1); codex-a ("Budget semantics ... say which"); gh-issues (#139 comment 2: the batch-token coupling "silently changes the number of optimizer updates per token budget"); review-files 4.2 item 7 ("10⁵ full-batch updates ... is `--total_tokens 2e9`. Say so"); chatgpt-b (B-1, the hard-coded 40 tokens/step).
- **Proposed by 8 reports.**

### D4. Grokking-signature instrumentation: train/held-out gap as one number, norm split, measured parameter deltas, update-scale ratio
- **Statement.** Log per task: train-set and held-out exact match and their gap; weight norm split into seen-token rows, unseen-token rows and the rest; actual parameter deltas per update (not `lr * grad_norm`); the `train/update/ratio/postclip_global` already at `train.py:2984`; plateau detection on EMA loss (state exists, `curriculum.py:82-88`, never read).
- **Why it fits.** "A grokking experiment is defined by this gap; the repo never logged it as one number" (chatgpt-a). The logged embedding norm is dominated by unused rows, so the total norm "is not even a measurement of the circuit" (lexical L1). All free.
- **Depends on.** #149 (norm keys), A2 (a held-out set to measure).
- **Evidence.** chatgpt-a #11 (chat 7: "gap between training and validation loss grows"; chat 11 "EMA-based plateau detection"); chatgpt-b #6-#7 (chat 69484386; chat 69488a01: "if those update proxies are collapsing while loss is flat, you've got a true vanishing-update situation"); claude-b1 #4-#5 (`b823ca58` line 455: "Monitor train-eval gap as a curriculum health metric"); code-model #2; codex-b #6 (#96's owner asked for "actual parameter deltas"); claude-a2 #6; lexical L1.
- **Proposed by 7 reports.**

### D5. Representation-geometry order parameters next to accuracy
- **Statement.** Per eval: Fourier energy of the digit embeddings for modular tasks or the Ω orthogonality of digit-position embeddings; NC1 within-class variability on the train-eval features; a real SVD / spectral entropy (the key named "entropy" today is a 256-bin histogram of `abs(weights)`, `utils.py:341-363`); top-k Hessian eigenvalues by power iteration on a fixed batch; attention entropy on real positions only; prediction entropy / distinct-answer count to flag mode collapse (every output was `9` at step 1000 in the Dec-22 log).
- **Why it fits.** One SVD of a 10 x d matrix or one power iteration per eval; gives a grokking signal that does not depend on the parser (#165) or the grid (#162). A 41.6M model is small enough for the Hessian probe on GPU.
- **Depends on.** Nothing for the cheap ones; rename the histogram key.
- **Evidence.** gemini-a A-3, A-11 ("Ω = (1/N) Σ |w_root·w_suffix|" [G:46f1 L262-323]; "Signature de Grokking : Spectre concentré près de zéro" [L449-454]); gemini-c #6 ([G:dd31 L290-302]); codex-b #5 (#73 excluded "sharpness/perturbation tests"); code-train-a ("attention sharpening is one of the reported grokking precursors"; the current key is padding-driven); claude-c1 #7 (mode collapse at step 1000); agent-reports AR-6 and rows 14-16 (SVD-entropy gate, Hessian trace, "Linéarité Locale" PCA: all "No").
- **Proposed by 6 reports.**

### D6. Per-task soft metrics that read correctly
- **Statement.** Cyclic distance `min(|p-t|, mod-|p-t|)` for mod_add; digit-level accuracy; tolerance (+-1) accuracy; a compare confusion matrix with a "missed_equality" rate; dyck false-positive/negative by depth; carry error rate; "forgot to reverse" rate; a string edit distance in place of the saturating `log1p` MAE (#164).
- **Why it fits.** One-line metric changes; the mod_add distance is the one that matters for the classic grokking task; the compare confusion matrix "would have exposed #156 on day one".
- **Depends on.** #164, #165.
- **Evidence.** claude-a2 #4-#5, #8 (chat d95daf7e: cyclic distance "proposed, dropped when the owner asked for 3-4 metrics"; chat 090b121f: "training accuracy evaluation with some tolerance (+-1)"); codex-a (PR050, owner: "96 and 0 are 'far' linearly but 'close' modularly"); chatgpt-a #13 (chat 11: "Digit-level accuracy for arithmetic"); gh-prs-a ("a digit-wise edit distance on the string ... would be a drop-in replacement").
- **Proposed by 4 reports.**

### D7. Cheaper evaluation: left-padded batched generation, whole-condition length bucketing, gate OOD on IID, eval every N updates on a smaller fixed probe
- **Statement.** Batch all prompts of a condition with left padding and a pad mask; sort once and chunk (5-8x fewer `generate` calls); run OOD only for tasks above 70% IID; evaluate every ~250 updates on a fixed probe set instead of 77 evals x 61 s per 100M-token run.
- **Why it fits.** Eval cost was a quarter of DAG-v12's wall time and "eval takes 20 min!" at 256 samples; all of this is bookkeeping.
- **Depends on.** #172 (pad mask), #142 (budget).
- **Evidence.** claude-b1 #2, #11 (`571b07d4` Fix 3: "Consider left-padding for batch generation"; `df40d43e` line 2148: "OOD eval only runs on tasks that achieved >70% IID accuracy"); code-eval #2-#3; claude-c1 #5 ("77 evals x 60.8 s = 78 min per 100M-token run").
- **Proposed by 3 reports.**

### D8. Run hygiene: `validate_args()`, a `--fixed_data` contract, a crash/interrupt contract, a run record from measured quantities, grid content hash, CPU determinism test, metric-schema migration
- **Statement.** One `validate_args()` after `parse_args` (silent DAG flags, `--grokfast` with accum > 1, `--compile_dynamic` without `--compile`, `--train_eval_samples 0`); make the fixed path reject or honour every curriculum flag and write `best_model.pt` unconditionally; write `summary.json` from `global_step`, `tokens_seen`, `opt_step` and a `status`; hash the grid content into `grid_version`; a two-run CPU determinism test (bit-identical today); a migration script for the old flat `metrics.json` keys.
- **Why it fits.** Half-day changes that would have caught several of the 63 issues at launch; CPU determinism is already exact, so the test is cheap.
- **Depends on.** #136, #137, #138, #184, #185, #187, #189, #194.
- **Evidence.** cli-commands #2-#4; code-train-b #2, #4; claude-b2 #3; gh-issues ("Grid versioning by content hash"); gh-prs-b ("pin the eval code version in every result file"); chatgpt-b #13 (per-size config files, chat 6941b170: "don't accidentally drift"); chatgpt-c #15 (file 39's acceptance gates); gemini-b #4 (stability-aware checkpoint selection: task score + OOD probe + NaN flag).
- **Proposed by 8 reports.**

### D9. Attention / padding mask as an explicit ablation variable
- **Statement.** Build the `kv_mask` that `CrossAttention` already accepts (`model.py:410`) from the collate pad mask and run latent/act/ssm_mem with and without it.
- **Why it fits.** The Dec-20 decision not to add masks "mid-study" assumed all variants were equally affected; they are not (thought cross-attention reads padded keys, causal self-attention never does). Small models, no new data.
- **Depends on.** #172, #126 follow-up.
- **Evidence.** claude-b1 #1 (`571b07d4` lines 592-604: "Option C: Defer to Future Study ... treat it as a new ablation variable").
- **Proposed by 1 report.**

### D10. Throughput attribution and the worker question
- **Statement.** Split the per-window time into data wait, step time and eval time and log the GPU clock; default `--num_workers 0` on this machine until #140/#141/#192 are fixed, or restore the trainer-owned difficulty `Value` from PR #17 and shared-memory tensors so workers are correct.
- **Why it fits.** On a 16 GB laptop sharing the GPU, four extra torch processes are the first thing to cut; a slow generator is indistinguishable from GPU throttling today.
- **Depends on.** #141, #192, #197.
- **Evidence.** claude-b2 #4; cli-commands #6 ("the generator costs microseconds per example; workers only add processes"); gemini-c #7 (Gemini's Option 2, page file, [G:ce77 L536-556], skipped); review-files H17; gh-prs-a (PR #17: "drive difficulty from a single global token counter in the training loop").
- **Proposed by 5 reports.**

### D11. Black-box search over the grokking knobs
- **Statement.** Optuna or CMA-ES with Hyperband pruning over wd, LR, batch and train fraction, 20-minute nano trials or minute-scale micro trials, a surrogate after ~100 evaluations.
- **Why it fits.** Grokking depends on exactly these knobs; feasible once A1's harness exists. Speculative until a single grokking run exists to anchor the objective.
- **Depends on.** A1, #136.
- **Evidence.** gemini-b #2 (6db3c20f L2272-2320).
- **Proposed by 1 report.**

---

## E. Ideas from the transcripts that were only touched on and dropped

Each row: idea, where it came from (report, transcript file, quote), what became of it.

| # | Idea | Source (file and quote) | Status at HEAD | Folded into |
|---|---|---|---|---|
| E1 | Fixed-data grokking command | codex-a: PR045 task, `--fixed_data --fixed_data_size 3000 --grokfast --weight_decay 1.0 --alg_tokens 50000000`; "no output exists" | `--fixed_data` crashes at first eval (#136) | A1 |
| E2 | Tiny-overfit invariant | chatgpt-b: chat 6945b243 (Dec 19), "Create a small fixed train set of 256-1024 examples ... If training doesn't hit near-zero loss quickly, something is wrong" | never built | A8 |
| E3 | Quick-iteration recipe | claude-a2: chat 23d47afd, `--fixed_data_size 500 --max_lr 3e-4 --alg_tokens 20M --eval_interval 200 --eval_samples 200`; "Never run" | not run | A8 |
| E4 | Sanity curriculum successor + copy | chatgpt-a: chat 11, "a tiny 'sanity' curriculum (only successor + copy) with very small ranges, plus ID vs OOD eval" | not run | A5 |
| E5 | Minimal-curriculum run | chatgpt-b: chat 69484386, "only parity, dyck, compare, copy, reverse ... pin difficulty in [0.1, 0.4] ... confirm strong OOD within 20-30M tokens" | not run | A5 |
| E6 | Static mix vs curriculum control | chatgpt-a: chat 7 research, "the best static mix achieved 2.7x lower perplexity ... than the best curriculum strategy" | never run | A4 |
| E7 | "Curricula can block grokking" warning | gh-issues / agent-reports: "Curriculum Étagé : Prometteur ou Risqué ?" (Dec 24), easy-to-hard curricula "peuvent parfois empêcher le grokking"; #157 comment: "Nobody acted on that" | DAG kept | A4 |
| E8 | Data famine on a rule class | gemini-a: [G:46f1 L217], "Seulement 20% des verbes réguliers (pour créer une 'famine' ...)" | nothing | A2 |
| E9 | Seed-split holdout, hard negatives | chatgpt-a: chat 4, "Hold out by seed (train seeds != eval seeds)"; "Hard negative cases (near-boundaries: lots of carries, deep nesting, long sequences)" | nothing | A2, B6 |
| E10 | Weight-decay sweep 0-2.0 | gemini-a / agent-reports: Grokking-Langage L56, "faire varier λ de 0.0 à 2.0 ... le grokking n'apparaîtra que pour λ > λ_critique" (Dec 17) | no sweep tooling | A3 |
| E11 | Constant LR after warmup for grokking | gemini-a: [G:46f1 L259], "warmup suivi d'une valeur constante, pour éviter un refroidissement trop rapide" | `constant` exists, default is `wsd` | A1 |
| E12 | Small model d=128 / 2 layers | claude-a1 file 4; agent-reports row 32 (Grokking-Langage L155-165; Validation L6 "2-4 couches, 128-256") | smallest preset is nano | A1 |
| E13 | Grokfast lambda 2-3, paired same-seed ablation | chatgpt-a: chat 11, "Start with amp = 2-3, EMA decay ~0.98 ... One with Grokfast off; one with Grokfast on"; review-files H7 | default 5.0; no paired record | A6 |
| E14 | StableMax / perpendicular gradient | chatgpt-b: chat 6942ce5e (Dec 17); review-files H1: "dropped from the merged doc entirely" | nothing | A7 |
| E15 | ~30x conventional weight decay for repeated data; 0.1-0.3% procedural warm-up; structured prediction before blaming depth | chatgpt-c: files 42/47 (Oct 2026 survey), "some highly overparameterized configurations needed roughly 30 times the conventional weight decay" | nothing | A3, B3 |
| E16 | Two-set training (25% repeated subset) | claude-a1: file 4, "select a random 25% of your training data as a repeated subset sampled every epoch" | `--persistent_alg_frac` unexercised | A9 |
| E17 | Reversed-digit answers | gemini-c: [G:ba6c L478], "inverser les nombres lors de la tokenisation (ex: 123 devient 3, 2, 1)"; "never mentioned again" | nothing | B4 |
| E18 | Format-OOD `reverse_order` condition | claude-b1: `df40d43e` line 1744, `Condition("addition", "ood_format", params={"digits": 4, "format": "reverse_order"})`; "Dropped from the coding prompt" | nothing | B4, B6 |
| E19 | Carry-pressure OOD axis | codex-a: PR018, "OOD-carry: digits=4, digit_bounds=(7,9)"; dropped in PR66's rebuild | gone | B3 |
| E20 | Digit / character tokenizer; 16k target | claude-a2 chat 090b121f, "Consider character-level or digit-level tokenization"; gemini-c [L549] "Vocabulaire cible de 16 384"; gh-issues (#186 comment, "The Embedding Tax") | 32k Llemma kept the same day (PR #48) | B1 |
| E21 | Curriculum controller redesign A/B | chatgpt-b: chat 694b4280 (Dec 24), floor-escape, learning-progress sampling; review-files H3: owner "Let me think about it" | Dec-23 controller kept under the DAG | B2 |
| E22 | Separate unlock EMA; gate on root mastery | codex-b: #109 "Alternative 1" / "Alternative 3", offered and dropped | Option A kept | B2 |
| E23 | 70/50 band with CLI flags | codex-b, gh-prs-b, agent-reports AR-1: PR #107 closed unmerged 2025-12-25; #157 and the codex history describe it as if it shipped | 0.85/0.55, no flag | B2 |
| E24 | Two-sided jitter around the frontier | agent-reports AR-4: Compass L259-266 `target + Normal(0, 0.2 * max)`; claude-b1 #13: code's jitter is `uniform(0, d)`, "replay, not frontier noise" | one-sided | B2 |
| E25 | Plateau detection on EMA loss; entropy as curriculum signal | claude-b1: `6e635fde` lines 56-58, 304-359; `ema_loss` stored, never read | half built | B2, D4 |
| E26 | Replay buffer, inertia, soft gates, plateau-triggered advance, 10% probability floor | agent-reports rows 20-22, 42, 60 (Nash-SEC L203, L281-294; Compass L393-400; P L383 "Ne faites pas de coupure brutale ('Hard Gating')") | binary gates, 0.02 floors | B2, A9 |
| E27 | Train/eval prompt diff tool | chatgpt-b: chat 6945e825, "Print 5 train samples + 5 eval prompts per task and diff them ... same separators, same spaces" | never built; #156 is what it would show | B7 |
| E28 | Multi-template prompts | gh-prs-a: PR #16 prompt, "Increase format diversity for each task (so the model doesn't overfit to a single prompt pattern)"; reversed by #103 | single `Answer:` template | B7 |
| E29 | Exact-length bucketing; explicit attention mask from the collate | claude-b2: Dec 31 chat lines 553-580, 643-717, "Use exact length as bucket key ... no padding needed -> no mask needed" | three coarse bands | B8 |
| E30 | Sequence packing | codex-b: #120, "Do not implement Sequence Packing" | not implemented | B8 |
| E31 | Trainer-owned difficulty via a shared `Value` | gh-prs-a: PR #17 task, "drive difficulty from a single global token counter in the training loop"; abandoned by #69's default | per-worker drift (#141) | D10 |
| E32 | Lexical-noise loader | chatgpt-a / codex-a / gh-prs-a: PR #14, "lex_loader: optional lexical noise (short real text snippets just to exercise the full vocabulary)" | `--lexical_frac_phase1` is a no-op | B10 |
| E33 | Dyck-2, kth-token, stack-machine traces, subtraction, sort, subitizing | chatgpt-a chat 4; codex-a PR017 ("Optionally add Dyck-2 later (two bracket types) as OOD"); agent-reports row 67 (Structuré L31-35) | none built | B6 |
| E34 | Count-balanced dyck corruptions | gh-issues (#176) | dyck measures counting | B6 |
| E35 | Owner's "sliding window + mask numbers linked by a causal relation" | chatgpt-c: file 45; "ChatGPT talked him out of in favour of random masking" | nothing | none (recorded) |
| E36 | Cyclic distance for mod_add | claude-a2: chat d95daf7e; codex-a: PR050 owner, "96 and 0 are 'far' linearly but 'close' modularly" | `grep -i cyclic` empty | D6 |
| E37 | Compare confusion matrix, missed_equality, dyck depth-stratified accuracy, carry error rate | claude-a2 #5: chat d95daf7e, "dropped with the 4-metric cut" | none | D6 |
| E38 | Teacher-forced answer check | codex-a: PR053 owner, "Generation is slow ... check if the model assigns high probability to the correct answer" | `_evaluate_numeric_answer` dead | D2 |
| E39 | Measured parameter deltas | codex-b: #96 owner asked for "actual parameter deltas" instead of `lr * grad_norm`; dropped | proxy only | D4 |
| E40 | Sharpness / perturbation diagnostics | codex-b: #73, "no sharpness/perturbation tests" | none | D5 |
| E41 | Ω order parameter; Hessian signature; "Linéarité Locale" PCA gate | gemini-a [G:46f1 L262-323, L449-454]; agent-reports row 16 (P L369) | none | D5 |
| E42 | Halting histograms by task; "avg inner steps" | chatgpt-a chat 4 Exp3 ("histogram of halting steps by task type; correlation of steps with correctness"); review-files H6 | only a window mean | C3 |
| E43 | Test-time "think longer" at K up to 30 | gemini-a: [G:5905 L1575-1581], "Allow the loop to run for up to 30 steps" | fixed K | C2 |
| E44 | Pause-token baseline | gemini-a: [G:5905 L947], "appending dummy tokens <pause> to an input sequence improves performance on reasoning tasks" | none | C1/C2 control |
| E45 | Eight-condition thought-mechanism ablation | chatgpt-c: file 35, "Report both fixed training-budget comparisons and inference accuracy-cost curves" | 4 of 8 exist, broken | C1 |
| E46 | Orthogonal thought init; sandwich norm; consistency loss | chatgpt-a chats 7/16, "Orthogonal init: yes, adopt it"; agent-reports rows 55-56 (Bugs L33-38) | `N(0, 0.02)`; pre-norm only | C9 |
| E47 | Absolute-position control | codex-a: PR008, "Use nano_baseline.py specifically to test the hypothesis that absolute embeddings fail OOD length generalization" | `nano_baseline.py` dead (#180) | C8 |
| E48 | Attention mask as ablation variable | claude-b1: `571b07d4` lines 592-604, "Option C: Defer to Future Study ... treat it as a new ablation variable" | `kv_mask` accepted, never fed | D9 |
| E49 | Gate OOD on IID > 70%; `--eval_iid_matched`; go/no-go "3 tasks > 50% IID by 30% of training" | claude-b1: `df40d43e` lines 2147-2148, 1761 | none | D7 |
| E50 | Newline stop token | claude-b1: `df40d43e` line 92, `eos_token_id = tokenizer.encode("\n")[0]` | EOS appended instead (PR #95) | none (recorded) |
| E51 | "Mean of used rows plus noise" reset target; untied head with decay on both tables | lexical #11-#12: "never run" | not run | C6 |
| E52 | Final RMSNorm vs norm diagnostics | lexical #3: note §3.4, "A final LayerNorm removes the dependence" | untested | C7 |
| E53 | Coupled Adam for the table | lexical #4: v3 `coupled` arm "written, not yet run" | not run | C5 |
| E54 | Loop-transformer ladder L0-L6 | chatgpt-b: chat 6a56e38e (Jul 2026), "a full ReLSM research-plan rewrite" | nothing built | C1 |
| E55 | Mode-collapse detector | claude-c1 #7: every prediction `9`/`99`/`9 9 9` at step 1000; `repeat_1gram_rate` reads 0.0 for it | none | D5 |
| E56 | Learnable task embedding to discover the hierarchy; multiple seeds and model soups | agent-reports rows 66, 68 (P L1865; P L1844, L1868) | none | D8 (seeds) |
| E57 | Nash-SEC's weight decay 1e-4 (">1e-3 eliminates grokking") | agent-reports AR-2: [P L1118] | not followed, never flagged | a warning, not an avenue: the DAG's source document contradicts the grokking literature on its one optimizer number |

---

## Ranked top 10

Ranked by (1) how directly the avenue answers "can this setup grok at all", (2) cost on the 4060, (3) how much prior evidence supports it.

1. **A1. The standard finite-set recipe at the classic scale.** It is the only experiment that answers the question directly; minutes to a few hours per seed at d=128; the recipe has been reproduced many times in the literature and was handed to the owner on Dec 17; 13 reports proposed it.
2. **A2. A same-distribution held-out split in the data path.** Without it A1 cannot be scored and every "IID" number stays a different-distribution number; one generator and a split, CPU-only; 6 reports.
3. **A3. Weight-decay sweep on all weights, embeddings included.** The one knob the literature agrees on, never swept here because the embedding was exempt; free on the A1 harness; 8 reports plus the Dec-17 report's own request.
4. **A4. Curriculum vs fixed sampling at matched compute.** The direct test of the warning the project ignored; hours at micro scale once A1 groks, about 30 h for a nano pair; 5 reports plus the counter-review.
5. **D1. Oracle and constant-model eval as the acceptance gate.** Not a grokking test but the precondition for reading any number (the oracle scored 0.833 in this audit, 0.779 in the previous one); two minutes on CPU; 6 reports plus the counter-review.
6. **A5. Single-task learnability at nano, no curriculum.** Tells what the actual model can learn in the actual budget before variants are compared; under a day for three tasks x three seeds; 7 reports.
7. **B1. A small vocabulary for the algorithmic phase.** Removes 39% of nano's parameters from outside the decayed set and three tokenizer bugs at once; a tokenizer file; 9 reports.
8. **D3 + D4. Count optimizer updates and log the train/held-out gap and the norm split.** Without these a grokking run cannot be recognised or compared with the literature; bookkeeping only; 8 and 7 reports.
9. **A6. Grokfast as a matched ablation after #137.** The only acceleration ever planned and never validly tested; same cost as A1; 7 reports, with the literature caveat that it must beat a tuned AdamW first.
10. **A7. StableMax / softmax-collapse controls.** A cheap alternative account of why nothing grokked (bf16 training, fp16 eval, no decay on the table); zero extra compute on the A1 harness; 5 reports.

C1 (compute-matched `shared_loop`) is the project's own hypothesis and would be next, but every report that ranked it put it after a positive grokking control, and it is blocked on #145 and #135.
