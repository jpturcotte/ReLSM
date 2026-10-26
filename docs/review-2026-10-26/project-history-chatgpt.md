# ReLSM project history from the owner's ChatGPT conversations

Merged from `history/chats_slice_1.md` to `chats_slice_4.md` (48 chats, 12 per slice), checked against sections 2-4 of `counter-review.md`. Tags: **[S1]-[S4]** = slice file and section (for example `[S2 §3.5]`); **CR** = counter-review, with its IDs (F01-F30, A1-A8, B1-B15). One extra source, `issues/model/act-halting-not-differentiable.md`, is cited only for the "ACT float" detail. Years are not printed on the chat pages. Dec = 2025; the 2026 dates are inferred by the slices from content and hex chat IDs. "(inference)" marks a reading that no slice or CR states outright. Several chats are forks of each other, so one run or piece of advice can appear in two chats.

---

## 1. Chat inventory

30 chats are ReLSM-related, 13 are partly related (literature sessions with a few ReLSM asides), and 5 are not related. Within each group they are sorted by first visible date.

| Title | Link | Date | ReLSM-related? | Topic |
|---|---|---|---|---|
| ReLSM architecture analysis | https://chatgpt.com/c/69378f5b-5ff0-832b-8934-d5078feacde6 | 2025-12-08 | yes | Report plus attachments; saturating a 4060 and a B200; "up to 2B tokens", "16k context"; ends in a "ReLSM-16k" design with PyTorch snippets |
| Critical analysis of proposals | https://chatgpt.com/c/69386da6-9c18-8328-a2db-f9522dd648ec | 2025-12-09 | yes | What the Exp0 baseline should be (1-3B, 2B tokens, 10 h); ~1B GQA config and Axolotl YAML |
| Training data mix | https://chatgpt.com/c/69388926-1e78-8329-8edc-21cbf4ec0d69 | 2025-12-09 | yes | Origin of the Exp0-Exp4 ladder; ~1B baseline; synthetic generator code |
| Branche, Branche: Training data mix | https://chatgpt.com/c/6938c171-31bc-8332-adc1-0c0c181f0873 | 2025-12-09 | yes | Ladder variants, `model_updated.py`, owner pastes PR #1 |
| Branche · Training data mix | https://chatgpt.com/c/6938adff-a640-832e-990e-8f558d732ad5 | 2025-12-09 | yes | "Ladder-ready" model.py (baseline/shared_loop/latent/act/ssm/ssm_mem); Codex integration prompt (PR #1/#2) |
| Branche · Branche · ReLSM dataset feedback (Gutenberg) | https://chatgpt.com/c/6941b6b1-4964-8327-b7d7-2afd363fb283 | 2025-12-10 | yes | Gutenberg and synthetic data, 3-phase token budgets, "do I modify my code?" (only `LANGUAGE_CONFIGS`) |
| Branche · Branche · ReLSM dataset feedback (6941b170) | https://chatgpt.com/c/6941b170-c138-8332-9c20-6d888935a72f | 2025-12-10 | yes | Phase 1/2/2b budgets per ladder size, 4K vs 16K, ablation fairness, disk usage |
| ReLSM dataset feedback | https://chatgpt.com/c/693a2b2b-042c-8327-bc52-11ed611b68c4 | 2025-12-11 | yes | Mixing research, token ladder, review of "Hybrid Architecture Research Plan", 6 Codex prompts |
| Branche · ReLSM dataset feedback (693b3166) | https://chatgpt.com/c/693b3166-b92c-8332-b04e-e2e06b16e6d9 | 2025-12-11 | yes | Same six prompts; review of Codex's `build_optimizer` diff |
| Branche · ReLSM dataset feedback (693b3ef6) | https://chatgpt.com/c/693b3ef6-33b4-8333-a9a9-e75c2bd59bf8 | 2025-12-11 | yes | Same, plus review of the annealed-sampler diff (PR #14) |
| Fixing DataLoader memory issue | https://chatgpt.com/c/693b46bf-b378-832f-9d66-5d8d78e79152 | 2025-12-11 | yes | Reviews `cycle(iter(loader))` replacement, tests, commit message (PR #15) |
| Branche: Code review ReLSM evaluation | https://chatgpt.com/c/693ccedd-53a0-8325-ba12-8baf7731191b | 2025-12-12 | yes | Eval harness review plus 8 Codex prompts; three re-reviews |
| Branche, Branche: Code review ReLSM evaluation | https://chatgpt.com/c/693cd148-4b98-8328-9312-e4fcbe4575ed | 2025-12-12 | yes | Two reviews of the rewritten eval harness; 8 prompts (PRs #21-23, #28) |
| Branche x3: Code review ReLSM evaluation | https://chatgpt.com/c/693cd442-e2a0-8325-b5fa-d06747df70da | 2025-12-12 | yes | Fork: batch_size, config coupling, pytest collection, one-shot prompt |
| Code review ReLSM evaluation | https://chatgpt.com/c/693c6f60-92bc-832e-b2eb-59a3f4c210a9 | 2025-12-12 | yes | No-padding and continuation-parsing prompts; reviews of `utils.py` zips |
| Check code bugs | https://chatgpt.com/c/6941a570-20b8-8325-aa2c-d95cd360e877 | 2025-12-16 | yes | Repo audit against two design PDFs; "no runtime bugs" |
| Analyze code and grokking | https://chatgpt.com/c/6942ce5e-dc3c-832c-bf2d-30c84a6268c2 | 2025-12-17 | yes | Why Exp0 and Exp2 did not grok |
| Model convergence analysis | https://chatgpt.com/c/6945b243-752c-8328-8703-3f5d71e4591c | 2025-12-19 | yes | exp0_baseline 20M, v2 (diverged), v4 (slow); fixed-dataset advice |
| Training issues analysis | https://chatgpt.com/c/6945e825-80a4-832f-8de9-f070e72c533c | 2025-12-19 | yes | v4 log: eval truncation, MAE explosion, warmup 25.6% |
| Branche, Branche: AI model metrics suggestions | https://chatgpt.com/c/69472e56-f0a8-832f-a57b-403ea7658931 | 2025-12-20 | yes | Label double-shift, answer-span metrics, NaN module norms (PR #80) |
| AI model metrics suggestions | https://chatgpt.com/c/6946d7ab-184c-8330-9aaf-28df7fdc4166 | 2025-12-20 | yes | Same run; ragged MetricsLogger; grad-clip-fraction naming |
| Branche · ReLSM dataset feedback (693b54b7) | https://chatgpt.com/c/693b54b7-3538-8330-a780-87ee7823c39a | 2025-12-20 | yes | "Difficulty ramps too fast?"; curriculum V1 prompt; Grokfast "highest-ROI" |
| Branche: AI model metrics suggestions | https://chatgpt.com/c/694728fe-39e4-8326-97ee-b22ec47e81d2 | 2025-12-20 | yes | Codex prompt for 4 diagnostic groups; diagnostics not reaching metrics.json |
| Branche · Branche · ReLSM dataset feedback (Dec 21 thread) | https://chatgpt.com/c/69484386-5130-832e-9750-5533a8ea4a3b | 2025-12-21 | yes | "Difficulty: 0.000" in a log line; "You didn't look at the latest file!" |
| Log optimization metrics | https://chatgpt.com/c/69488a01-f7a8-8333-8308-15c09d0a17cc | 2025-12-21 | yes | LR / update-norm logging diffs in train.py (PR #96) |
| Compare run metrics | https://chatgpt.com/c/6949cb7d-007c-8328-89b5-b956b8ea9c79 | 2025-12-23 | yes | Why Dyck, Chain, Successor "crashed" v16 to v18; parser debate |
| Branche · Compare run metrics | https://chatgpt.com/c/694b2b13-c49c-8332-9384-f4fec4874973 | 2025-12-23 | yes | 5steps/20steps v1/v2 files, "assess the new parser" |
| Compare training runs | https://chatgpt.com/c/694b4280-5500-8328-8d9d-2f980e24fec1 | 2025-12-23 | yes | run4/run5/run6 per-task difficulty curves |
| Plot task accuracy difficulty | https://chatgpt.com/c/694c894c-05cc-8332-bcb7-179d75b93f73 | 2025-12-25 | yes | First DAG-curriculum run; copy/reverse stuck; prompt masking |
| Efficient Small Data Training | https://chatgpt.com/c/6adf95f7-4c74-83e9-9946-44dadd627a1b | 2026-10-26 | yes | Data-efficiency survey; seven IID files; six-sub-agent review behind the CR |
| Recursive model architecture research | https://chatgpt.com/c/693745a8-255c-832c-baff-a28857884cb3 | 2025-12-08 | partly | Origin survey (HRM, TRM, MoR, Huginn, COCONUT, SSMs); single-B200, <1M-token design; no code |
| Recurrent AI Papers | https://chatgpt.com/c/6a56e38e-8dec-83ea-abe5-7580ff072a91 | 2026-07-15 | partly | Re-audit of the owner's literature report; L0-L6 ladder; "the current broken ACT mechanism" |
| Compare concept models | https://chatgpt.com/c/6adbf8f4-7acc-83ea-88b7-be1e4eef173a | 2026-10-23 | partly | Research Session 01; a few "for ReLSM" suggestions |
| Compare latent reasoning methods | https://chatgpt.com/c/6adbfb87-1048-83ea-bc91-358dd6a81a99 | 2026-10-23 | partly | Session 02; applies to ReLSM and flags a causality-leak risk |
| Comparaison des modèles de diffusion | https://chatgpt.com/c/6adbfdba-8df0-83ea-ac21-e64a20cf8373 | 2026-10-23 | partly | Session 03; diffusion / flow LM survey |
| Compare generation strategies | https://chatgpt.com/c/6adbff8d-b990-83ea-b670-a5ec65481aff | 2026-10-23 | partly | Session 04; one paragraph proposes a copy/reverse experiment |
| Taxonomie de récurrence | https://chatgpt.com/c/6adc056f-c49c-83ea-bcd4-f9cb4392659f | 2026-10-23 | partly | Session 05; red-team audit of recurrence |
| Performance comparison report | https://chatgpt.com/c/6adc0c7b-4468-83e9-97aa-7137a25892c6 | 2026-10-23 | partly | Session 06; owner: "Don't box yourself in with the project." |
| Structured Concepts Framework | https://chatgpt.com/c/6adccafc-f538-83ea-861d-a03c0042201c | 2026-10-24 | partly | Session 07; the string "ReLSM" never appears |
| Rapport de recherche complet | https://chatgpt.com/c/6adce7ea-176c-83e9-8cdb-f7f0b4b28ce3 | 2026-10-24 | partly | Merges seven sessions into an integrated report (body not in chat) |
| Méthodes d'entraînement frugal en données | https://chatgpt.com/c/6adcf1ea-3b50-83ea-873e-aabd9a7236ad | 2026-10-24 | partly | Data-frugal training survey; ReLSM only in asides |
| JEPA Explanation Mistake Correction | https://chatgpt.com/c/6adeb9ac-8054-83e9-914f-25731f0589a8 | 2026-10-26 | partly | Conceptual Q&A on JEPA for text |
| Streaming Dataset Training | https://chatgpt.com/c/6adf912d-2ba0-83e9-9026-9311c91ba3b4 | 2026-10-26 | partly | General streaming-dataset answer; no experiments |
| Jev & Laya Re-audit | https://chatgpt.com/c/6ab2e6fa-e080-83e9-b956-b1e2102c4f40 | 2026-09-28 | no | JevBench, routing and game-task evaluations |
| Latest Distillation Models | https://chatgpt.com/c/6adbf078-004c-83ea-a24c-175a80d1d3cf | 2026-10-23 | no (adjacent) | Concept-prediction models; assistant replies empty in the scrape |
| AI Memory Methods | https://chatgpt.com/c/6adcfef5-5d5c-83e9-b978-e08ac95043d0 | 2026-10-24 | no | PEER / UltraMem, beyond-VRAM models |
| Continuer la recherche ML | https://chatgpt.com/c/6add204a-ffd4-83e8-8aaa-05529bdb868d | 2026-10-24/25 | no | Research-agent progress on disk-backed training |
| Technologies web 2026 | https://chatgpt.com/c/6adfb260-d6c4-83ea-856f-21c452d7481e | 2026-10 | no | Web frameworks as of Oct 2026 |

---

## 2. Timeline of experiments as reported in chat

Hardware throughout: RTX 4060 Laptop, 8 GB [S3 §2]. **No chat before Oct 26, 2026 names DAG-v12, and no chat reports a Grokfast run** (only `Grokfast: False` in printed configs [S3 §3.8]). S1 reports no runs at all [S1 §2].

**2025-12-17: Exp0 vs Exp2** ("Analyze code and grokking") [S2 §2.1]. Files `metrics_baseline.json`, `metrics_latent.json`. Owner: "I'm using 20M of synthetic algorithmic data." ChatGPT's description of the config: infinite procedural data, curriculum, weight decay 0.1, phase 1 of about 10M tokens. Baseline: loss about 10 to about 0.5, then flat; addition 0%; best Dyck about 27%. Latent: loss 0.2-0.3, under 0.18 mid-run; Dyck peaked about 15%, ended about 6%. Quote (owner): "I tried Exp0 & Exp2 on my ladder, but I didn't see the posited grokking."

**2025-12-19: exp0_baseline (20M tokens)** ("Model convergence analysis") [S3 §2.1]. Nano, about 20M tokens, difficulty 50% to about 99.9%. Loss about 10 to about 0.55; best acc 6.0%. Quote (ChatGPT): "Exp0-20M underfit the data: training never converged to low loss, and no task was mastered."

**2025-12-19: exp0_baseline_100M_v2** (same chat) [S3 §2.2]. `--weight_decay 0.001 --max_lr 6e-3 --warmup_steps 30000 --grad_clip 0.5 --alg_tokens 100000000 --alg_batch_size 24 --grad_accum_steps 12 --eval_max_new_tokens 32 --num_workers 0`; 41.6M params; decay group 25,165,824, no-decay 16,398,848. Step 2,500: loss 1.207, acc 12.9%, weight norm 166.2. Step 22,500: acc 11.1%, weight norm 424.6, Dyck 43.1%, compare 39.6%; MAE about 1e28-1e60. Quote (owner): "Wow, something weird just happened in my current run".

**2025-12-19: exp0_baseline_100M_v4** ("Model convergence analysis"; "Training issues analysis") [S3 §2.3]. `--weight_decay 0.0 --max_lr 1e-4 --warmup_steps 10000 --grad_clip 0.1 --alg_batch_size 64 --eval_max_new_tokens 24 --difficulty_schedule linear`. Loss 10.44 (step 50) to 4.75 (3,000); weight norm 152.6 to 152.8. By about 20k steps (per ChatGPT): loss about 0.9, acc about 16%, Dyck 76-78%, compare about 72%, arithmetic/copy/reverse/successor about 0%. Quote (ChatGPT): "your eval signals scream 'format/metric + truncation + task-spec mismatch' more than 'model can't learn'."

**2025-12-19: single-task runs** exp0_grokking_mod_add_v4 (about 30M tokens, best acc 4.17%), Exp0_parity_v8, Exp0_grokking_v7 [S3 §2.4]. Judged from plots; low evidence.

**2025-12-20: three runs on one metrics file** ("AI model metrics suggestions", two forks) [S2 §2.2; S4 §2.1]. Run A (pre-fix): about 4.4k tok/s; difficulty 0.25 to about 0.44; accuracy 6-7% at every eval (0.0685 at step 14,000). Run B (post label fix): loss to 0.94 at 10k; Dyck about 0.80, compare about 0.70, the rest about 0. Run C: per-module grad norms NaN at steps 6000 and 8000. Quote (owner): "It seems the model sometimes gets the first few numbers right and then start hallucinating."

**2025-12-21: run with flat difficulty** ("Branche · ReLSM dataset feedback"; "Dec 21 thread") [S3 §2.5; S4 §2.2]. Schedule `smooth`, `alg_tokens=100M`, adaptive task weighting, WSD LR. Log line: "Difficulty: 0.000 (easy_mix: 0.2)"; logged 0.250001 at step 50, 0.250026 at step 3000. Per-task exact at 1000/2000/3000 (S4): compare 0.177/0.427/0.333, Dyck 0.497/0.537/0.503, parity 0.102/0.146/0.078, the rest 0. Quote (owner): "I still think difficulty ramps up too fast."

**2025-12-23: checkpoints v15, v16, v18, 5steps, 20steps; seven IID files** ("Compare run metrics"; "Branche · Compare run metrics"; revisited in "Efficient Small Data Training") [S2 §2.5; S4 §2.3-2.4]. Baseline, 6 layers, d=512, context 128, 200 examples/task, commit 66ba4bec. Overall exact: v15 8.60%, v16 13.80%, v18 13.80%, 5steps 30.05%/28.05% (v1/v2), 20steps 30.55%/30.35%. v2 files: addition 66.0, successor 100, copy 0, reverse 0. Dec 23 chat: chain exact 0.015/0.035/0.005, Dyck 0.17/0.21/0.01. S4's per-task table for the 5/20-step files is unreliable [S4 §2.4]. Quote (owner): "Do you have an idea or hypothesis for why Dyck, Chain, and Successor crashed?" Quote (ChatGPT, Oct 26): "filenames alone do not establish the training-token budget, parameter count, or meaning of '5steps' and '20steps'".

**2025-12-23/24: run4, run5, run6** ("Compare training runs") [S4 §2.5]. run4 `min_task_evals=5`; run5 `=20`; run6 as run4 plus `min_difficulty=0.2`. run6: 8 of 10 tasks pinned at 0.2; addition about 0.45; successor to 1.0. Quote (ChatGPT): "raising the minimum just changes where it gets stuck".

**2025-12-25: first DAG run** ("Plot task accuracy difficulty") [S2 §2.4]. Gates 0.99/0.98, margin 0.01, patience 4, ramp 3, replay 0.20. Step 6000: copy EMA about 0.38, reverse about 0.60. IID eval: acc 10.8%, token 0.366, MAE 52954.8; copy/reverse predictions always exactly 11 tokens against 16-24-token targets. Quote (owner): "That's the IID evaluation, not the train evaluation." The last ChatGPT turn shows only the status line "Investigating target truncation issues".

**2026-10-26: DAG-v12, as summarized** ("Efficient Small Data Training") [S2 §2.6-2.7]. Quote (ChatGPT): "recorded 98.06% of examples on copy, reverse, compare and parity. Six other task gates never opened. Its best broad IID exact match was 29.25%." Owner: "all of my grokking experiments failed. I don't know why, but is the state of affairs for now."

---

## 3. Advice that led to defects

| Advice (quote, chat, date) | What it caused | CR defect |
|---|---|---|
| Latent pseudocode: `z` cross-attends to `kv_source=h`, then `z_sum = z.mean(dim=1)`, `h = h + thought_to_token(z_sum)` "broadcast" (Training data mix, Dec 9) [S1 §3.3] | Thought state reads all positions and writes back to earlier ones; PR #1 implemented the design (inference) | **F01** future-token leak. Teacher-forced loss latent 1.841 vs prefix-only 2.588 |
| ACT pseudocode: `ponder += p.mean()`, break only on `(cum_halt >= tau).all()` (Training data mix, Dec 9) [S1 §3.3] | Halt head gets no gradient; ponder term rewards not halting; batch-wide break. The "ACT float" is the `float()` conversion for `model.aux` logging that detaches halting (issue file) | **F03**, **B11**. Dec 11 advice to anneal the halting threshold [S2 §3.8] was moot |
| "generated on the fly so it never overfits" (Training data mix, Dec 9) [S1 §5]; "the model never sees repeating samples, which disables memorization" (Model convergence analysis, Dec 19) [S3 §3.7] | Owner built the grokking plan on "infinite data" | **A3**: the set is replayed; CR §4.1: DAG-v12 "was never a grokking experiment" |
| "Both embedding + head are excluded from weight decay." (ReLSM dataset feedback, Dec 11) [S3 §3.6] | PR #13 no-decay group; 16,398,848/41,564,672 = 39.5% of parameters | **B7**: weight norm 152.6 to 358.1, never contracted |
| "itertools.cycle stores every batch it has seen... That's the correct fix" (Fixing DataLoader memory issue, Dec 11) [S2 §3.5] | PR #15 recreates iterators, with no check of RNG or worker state (inference) | **A3**, **A4**, **F21** |
| "No obvious runtime bugs (exceptions or tensor errors) are present in the code itself" (Check code bugs, Dec 16) [S4 §3.2] | Cleared SSM and latent variants; only tuning advice followed | **F01**, **F02**, **F16**, **A7**, **F17** |
| "I'd consider the evaluation pipeline 'clean' for the custom-model-only target" (Code review ReLSM evaluation, Dec 12-13) [S1 §3.1] | Eval trusted for Dec 19-25 diagnoses | **B1** oracle 0.779; **F04** 7/21 OOD impossible; **F12**; **A5**; **B2**; **B3** |
| "Deterministic eval is actually deterministic now" (Code review, Dec 12) [S2 §3.3] | Only decoding checked; `hash()` seeding approved in later diffs | **F12** (0 shared prompts between v1 and v2); **A5** |
| `--fixed_data True --persistent_alg_frac 1.0` (Model convergence analysis, Dec 19) [S3 §3.7] | The canonical grokking config | **F05** crash at micro-step 500, no checkpoint; **A3** |
| MAE "clearly broken (or dominated by invalid parses)"; `safe_parse_int` 40-digit cap; `--eval_max_new_tokens 256` (Training issues analysis, Dec 19) [S3 §3.2-3.3] | Wrong cause chased; the flag would probably not reach the hard-coded OOD path | **F14**/**B1** (oracle MAE 3.4e35); **F04** |
| "This is probably the single highest-ROI idea in the doc." (Grokfast; ReLSM dataset feedback, Dec 20-21); then "ship it" on four diffs containing the Grokfast block and the telemetry code behind F25-F27 (Log optimization metrics, Dec 21) [S4 §3.2; S2 §3.4, §3.10] | Endorsed without checking accumulation or AMP. PR #45 (Dec 17) predates this, so it endorsed rather than originated | **A1**/**F11**: x1.545 per update at accum 4; run probably invalid |
| "Correct and improves ... in exactly the right way." (AI model metrics suggestions, Dec 20) [S1 §3.2] | Diffs carried `hash((sample_seed, task))`, `max_new_tokens=32`, `tokens[:max_seq_len]`, `random.Random(seed)` in `__iter__` [S1 §3.2; S4 §3.2] | **F12** (low impact there), **F04**, **F06**, **A3** |
| "Your training labels are almost certainly misaligned (double-shift)" (AI model metrics suggestions, Dec 20) [S4 §3.1] | PR #80 fixed it. Slices do not say who introduced it. Dec 17 and Dec 19 verdicts rest on pre-fix runs (inference) | Not in CR. Weakens the Dec 17 Exp0/Exp2 and Dec 19 v2/v4 analyses |
| "Exp2 simply memorized more aggressively, whereas Exp0 plateaued at a higher error" (Analyze code and grokking, Dec 17) [S2 §3.6] | Owner read a leak-exploiting model as memorization | **F01**; CR: discard the Exp0 vs Exp2 comparison |
| "Difficulty ramp: absolutely not 'too fast' according to the logs; it's barely moved" (ReLSM dataset feedback, Dec 20-21) [S4 §5] | Overrode the owner's instinct; a flat 0.250001-0.250026 fits A4 but is not provable | **A4** |
| "That's the signature of 'model is kinda right, scorer is strict'." (Compare run metrics, Dec 23) [S4 §2.3] | Normalizer and extractor patches; 5steps vs 20steps v1/v2 gaps attributed to the parser | **F12** (0 shared prompts; resampling noise); **F13**, **B10** |
| Prompt-loss masking (PR #110) as the Dec 25 fix [S2 §3.9, §3.15] | Mask covers the tokenized `:=`, so "=" is never supervised (inference); truncation not found | **A6**; **A2** |
| "Your worry is partly correct, but the 'they'll be too slow to recover' part is usually less bad than it sounds." (Plot task accuracy difficulty, Dec 25) [S2 §3.14] | Owner's worry proved right | **B5**; CR §4.1: six tasks got 1.94% of examples |

---

## 4. Leads that were opened and dropped

1. **Dec 25, "Plot task accuracy difficulty".** The final turn shows only "Investigating target truncation issues" and "Examining evaluation truncation process", with no answer [S2 §2.4]. Truncation was the root cause of the DAG stall (CR A2, F06). The owner's symptom, copy predictions stopping at exactly 6 items, was visible the same day.
2. **Dec 19, "Training issues analysis".** ChatGPT named "truncation" and suggested `--lang_seq_len 256` if copy/reverse exceed 128, plus `--eval_max_new_tokens 256` and a newline stop token. Right direction, wrong parameters, and the chat ends before the owner reports applying anything [S3 §3.1-3.2, §4]. Warmup tokens equal exactly 40 x batch, which hints at about 40 tokens per sequence (S3 inference, unverified).
3. **Dec 20, stop-reason metric.** ChatGPT designed `stop_reason` "to explain 'gets first digits right then hallucinates / truncates'". The instrument was right (PR #103/#106), but nobody tied it to the 32-token cap [S1 §3.2; S2 §3.2].
4. **Dec 25, train vs IID distribution.** ChatGPT said copy was trained at difficulty about 0.10 but judged at 0.5. The owner objected ("That's the IID evaluation, not the train evaluation") and the point was dropped [S2 §2.4, §3.15]. CR **B2** shows the IID grid is not the training distribution.
5. **Dec 25, gate thresholds.** ChatGPT said 0.99/0.98 are "very strict", suggested copy 0.95, reverse 0.93 and patience 4 to 2, and asked for stage-aware runway [S2 §3.14]. CR **B5**: thresholds were never loosened.
6. **Dec 21, "Difficulty: 0.000" vs 0.25.** Never resolved; the final owner message "You didn't look at the latest file!" has no reply [S3 §2.5, §3.9]. Nearest documented cause is **A4**.
7. **Dec 21, parity 0.10/0.15/0.08.** ChatGPT noted it was "worse than chance" and did not investigate [S4 §3.1 row 7].
8. **Dec 19, grad norms in the hundreds at clip 0.5.** ChatGPT guessed pre-clip logging but recommended clip 0.1 instead [S3 §3.14]. CR **F26** confirms the pre-clip/post-clip mismatch.
9. **Dec 9, shared-loop fairness.** "same FLOPs budget? Then fix unroll and compare accuracy" was never carried into `train.py`; `shared_loop` is always 6 x 24 (CR §2) [S1 §3.3].
10. **Dec 11 Codex prompts** for lexical noise, orthogonal thought init, `--grad_clip 0.5` and the README: no matching PR found [S2 §4].
11. **Oct 2026, causality check.** "Compare latent reasoning methods" flagged "a risk to verify in the actual implementation" [S1 §3.3]. It was verified only by the six-agent review.

---

## 5. What ChatGPT never raised

"Raised" means a Dec 2025 chat. Items 1-5 first appear in the Oct 26, 2026 review chat ("Efficient Small Data Training") or CR; the rest appear in no chat at all.

| Defect (CR) | Chat record |
|---|---|
| 48-token truncation (A2, F06, B6) | Dec 2025 diffs with `tokens[:max_seq_len]` approved silently [S4 §3.2]; Dec 25 lead died. Oct 26: "177/400 copy, 158/400 reverse" [S2 §3.1] |
| 32-token eval cap (F04, B13, F30) | `max_new_tokens=32` visible in Dec 12/20 diffs, uncommented; named Oct 26 [S1 §3.2; S2 §3.2] |
| `hash()` eval seeding (F12) | Approved Dec 12/20; named Oct 26 [S2 §3.3] |
| Future-token leak (F01), ACT halt gradient (F03) | Introduced by the Dec 9 designs; named Oct 26 [S2 §3.6, §3.8] |
| Resume (F07, F09, F10), SSM fallback scan (F02) | Named Oct 26 only [S2 §3.13, §3.7] |
| Grokfast divergence under accumulation (A1/F11) | Praised and approved Dec 20-21; absent from the Oct 26 visible summary [S2 §3.4] |
| Worker replay and difficulty reset (A3, A4, F21); aliased `lang_loader` (B12) | Diffs reviewed, not flagged; not in Oct 26 visible text [S4 §3.2] |
| Compare "=" never supervised (A6) | Never; ChatGPT saw only `Answer[:=]?` regex risk [S3 §3.11] |
| Eval resets global RNGs (A5); fixed-data crash (F05) | Never; F05's flag was recommended [S3 §3.7] |
| SSM ignores input, channel collapse, init, cache (A7, F16, F17, F19) | Never; Dec 16 audit found "No obvious runtime bugs" |
| Oracle ceiling 0.779 (B1); IID grid not training distribution (B2); Dyck depth 5/6 identical (B3) | Never; "clean" verdict instead |
| Final partial accumulation discarded (F08); CE NaN on zero-label batches (A8); best-checkpoint noise (B8) | Never |
| Train-eval fixed probe misread (B4); duplicate `train_eval/acc/token` key (B14); metadata says 64 not 32 (B13) | ChatGPT read train-eval as a procedural split [S3 §3.12] |

---

## 6. Gap between plan and runs

| Item | Planned (chat) | Actual |
|---|---|---|
| Scale | Dec 8: under 1M tokens, 10 h on one B200 [S1 §2.2]. Dec 9: ~300M ladder; Exp0 at 1B (1.01B / 1.055B / 1.08B variants) [S1 §2.2] | Nano baseline, 41.6M unique parameters [S3 §2]; a "ReLSM-Nano (~18M)" was the proposal's match target [S1 §2.2] |
| Tokens | 2B (0.4B algorithmic, 1.6B text); "2B tokens in 10 h ~ 55,556 tokens/s" [S1 §2.2]. Ladder: nano 50M, 125M 200M, 300M 400M, 760M-1B 600-800M [S2 §2.8]. Regime A 100M, B 500M [S3 §2.6]. Oct 26: "100 billion is feasible... Maybe 200" [S2 §2.7] | 100M supervised tokens, no language phase [S3 §2.2; CR §4.1]. The Axolotl YAML itself gives only about 0.79B tokens (S1 arithmetic) |
| Context | 4k, with the best variant re-run at 16k [S1 §2.2] | Context 128 [CR §4.1]; `--lang_seq_len 128` [S3 §2.2] |
| Sequence length | Never planned or discussed | `alg_seq_len` 48 [S2 §2.6; CR §4.1] |
| Updates | 104,166 estimated steps (v2), 39,062 (v4) [S3 §2.2-2.3] | Counters are micro-steps at accum 12: v4 about 3,250 updates (S3 inference); DAG-v12 6,458 updates [CR B7] |
| Ladder coverage | Exp0 to Exp4 [S1 §2.2] | Only Exp0 and Exp2 reported [S2 §2.1]; Exp1, Exp3, Exp4 absent from the chats |
| Throughput | ChatGPT: "~38 tokens/sec" and "around 30-40 tokens/sec" on the 4060 [S3 §2.6; S4 §5] | Owner logs about 4.4k tok/s (S4) and 5,000-7,600 tok/s (S3): about 100x off |
| Grokking design | Dec 9 "infinite" synthetic phase [S1 §3.3] | Moving distribution, embeddings exempt from WD; CR recommends a fixed finite set |

Net: planned at 300M-1B parameters, 2B tokens and 4k-16k context; everything that ran was a 41.6M, 128-context nano baseline with 48-token algorithmic sequences at about 5% of the token plan.
