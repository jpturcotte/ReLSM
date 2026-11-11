# ReLSM: grokking research and the owner's ideas, Oct 2025 to Oct 2026

Written 2026-10-28. This file merges six reader reports (res-chatgpt-dec, res-chatgpt-2026, res-gemini-perplexity, res-claude, res-codex-lit, res-new-exports), three notes on the DAG curriculum (dag-origin-idea, dag-origin-implementation, dag-runs-and-language-switch) and the account-wide search log of 2026-10-28. It is centred on the owner's own prompts. Two later inputs are merged too: res-chatgpt-outside (the 37 ChatGPT chats outside the ReLSM project, section 7) and four readers of the full ChatGPT re-export, reexport-g1 to reexport-g4 (the older messages the first exports missed, section 7b). Section 9 lists what they changed.

How sources are named:
- "ChatGPT chat 6942ce5e", "Gemini chat 46f11897", "Claude chat dc94954c": the chat's export ID. For ChatGPT and Codex the ID is the creation time in hex seconds.
- "Claude.ai chat a460a854": a chat found only in the 2026-10-28 Claude.ai export bundle.
- "Perplexity Nash-SEC thread turn 3": turns counted 1 to 14.
- "Codex task PR108": the Codex task that produced pull request #108.
- Gemini Deep Research reports whose prompt is lost are named by their title.
- "Rep-NN" refers to a research report in section 4. "G3", "C12" and so on refer to idea rows in section 2.

Code status is from `git grep` on `main` at `be08cde` (2026-10-28). No model code was run for this file. Quotes are short; French has an English gloss.

---

## 1. Summary

**What the owner set out to do.** On Dec 8, 2025 he asked three AIs to "find a novel but simple architecture" for reasoning, inspired by HRM and TRM, "easily implemented and testable with minimal means": under 1M tokens, under 10 h on one B200, inference on an RTX 4060 laptop. Within a day the plan was a general multi-domain text model ("No, it needs to be multi-domain") with a latent loop, tested on a five-rung ablation ladder. "Grokking" entered through an AI report he adopted: Phase 1 would be "algorithmic grokking" on infinite procedural data, then a language phase.

**What research he received.** 71 research reports or literature answers (section 4): 53 between Oct 2025 and Jan 2026, 18 in 2026. Three answers on the same day, Dec 17, 2025 (ChatGPT chat 6942ce5e, Claude chat dc94954c, Gemini chat 46f11897), said the same thing: grokking needs a finite training set that is memorised first, strong weight decay and long training; infinite procedural data "preempts the classic grokking dynamic". The Gemini report gave a full recipe (d = 128, 2 layers, weight-decay sweep 0 to 2.0, constant LR, at least 1e5 steps). 51 of the 71 reports contain at least one wrong citation, invented number, wrong code or false claim (section 4.3 lists the 59 errors that matter most). Three ChatGPT code reviews said the code was fine while it had serious bugs: Dec 16, Dec 19 ("No fatal coding bugs", one day before the double label shift was fixed) and Dec 21 ("nothing is obviously broken", quoting six defaults that were not in the repo).

**What was built.** Six model variants (baseline, `shared_loop`, `latent`, `act`, `ssm`, `ssm_mem`); ten procedural tasks; a `--fixed_data` flag, Grokfast and a weight-decay default of 1.0 (Dec 17); a per-task competence controller (Dec 20 to 22); a prerequisite DAG curriculum (Dec 24 to 27). ChatGPT wrote its graph and every gate value in one answer (Dec 25, 01:19 UTC, chat 694c894c), on top of AI reports, some written with no web search. The recurrent variants have serious defects: the thought loop leaks future tokens (#135), the ACT halt head gets no gradient (#132), the SSM fallback mixes batch and time (#134).

**What was never tried.** The finite-set recipe at the classic scale (#249). A same-distribution held-out split (#214). A weight-decay sweep on all weights (#251). A curriculum versus fixed-mix control (#252). A compute-matched loop ladder (#258). A language phase: even the Dec 17 "language" batches were replays of the algorithmic stream (#158). Language grokking, the owner's own Dec 17 question. Every recorded run streamed procedural data and made at most about 10k optimizer updates (DAG-v12: 6,458).

**The five things that matter most now.**
1. **Grokking was never tested in its documented regime.** The Dec 17 recipe was replaced by curriculum work on the infinite stream within three days, with no recorded decision ("Make sure task curriculum work with infinite algorithmic generation", Codex task PR087). "All of my grokking experiments failed" (Oct 26, 2026) is not yet established. Run #256, then #214 and #249, before anything else.
2. **The weight-decay advice contradicted itself, and the owner ended on the weakest setting.** In 48 hours he got wd 1.0, then wd 0, then "raise it to 1.0 or higher", then wd 1e-4 from the Nash-SEC report. Every run after Dec 19 used wd 0.0 or 0.1, with 39 % of nano (the 32k embedding) exempt from decay. #251 is the fix.
3. **The DAG came from AI reports, and their warnings were dropped.** ChatGPT wrote the PR #108 spec in 43 minutes on Dec 25 and made up every gate value in that answer. The data it read, copy and reverse at 0.0, were a truncation artefact of the 32-token sequence limit (#139), not a task-order problem. The upstream reports said "don't hard-gate", "a static mixture often outperforms a stepwise curriculum" and "easy-to-hard curricula can sometimes prevent grokking". The code has binary gates; DAG-v12 unlocked one task and paused for the rest of the run. #252 should run before any more controller work.
4. **The recurrence hypothesis is untested, and the literature now says when to test it.** The only loop-versus-baseline record (an unreported `shared_loop` twin of DAG-v12) stopped at 3.7 % of its budget with no edge. Wang et al. 2024 and Kohli et al. 2026 find loops help out-of-distribution composition only through a later grokking stage. #258 belongs after a positive grokking control.
5. **Two of the owner's own ideas have no issue.** Language grokking (Dec 17, 2025; revived as lexical grokking in PR #196) and algorithm-to-language transfer ("Could we mix algorithmic & language tasks so they build off each other?", Dec 24). Section 6 lists 58 ideas with no issue: 27 are grouped into 7 proposed new issues, 21 into 15 proposed comments on existing issues, and 10 are recorded only. Section 6.1 adds 11 from the re-export: 8 proposed comments, 3 recorded only.

**What the full ChatGPT re-export changed (section 7b).** It recovered 12 run records (among them the Dec 17 Exp0/Exp2 metrics and the full v15/v16 and run4/run5 settings), the chat where the DAG spec was written, and the July 2026 claim ledger. It moved the 5steps/20steps copy/reverse ceiling from the worker bug (#141) to truncation (#139). It found four owner ideas stated before an AI proposed them to him again: a repeated subset (Dec 19, 2025), memorise first (Dec 19), a reasoning language the model invents (May 2, 2026), and internal concepts that need not be human (Oct 24, 2026).

---

## 2. The owner's ideas, by theme

161 rows: 131 from the first readers, 30 added from the outside chats and the full re-export. A row is one idea in the owner's words; some rows join two prompts on the same idea. "(adopted)" marks text an AI wrote that the owner pasted or accepted. "Issue" is the GitHub issue that owns the idea today; "none" means no issue does (section 6). The 30 added rows cite issues up to #272; the first 131 cite issues up to #259. Times in the added rows are UTC (EST = UTC − 5 h in December).

### 2.1 Grokking mechanics

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| G1 | 2025-12-09 | Claude chat 3ee613da | "Take feedback into account." (accepting the split curriculum) | (adopted) Phase 1 = "algorithmic grokking" on infinite procedural data, then language | Built as an infinite stream. Contradicted Dec 17. The stream also replays (seed 42). | #214, #140, #169 |
| G2 | 2025-12-17 | ChatGPT chat 6942ce5e | "I tried Exp0 & Exp2 on my ladder, but I didn't see the posited grokking." | Grokking expected from the design, with no stated criterion | Rep-18. The two metrics files are in the re-export: after 19.95M tokens train loss 0.550 (Exp0) and 0.174 (Exp2), no late drop, no validation loss. The runs used a broken objective (labels shifted twice, prompt in the loss), so they could not grok. | #249, #259 |
| G3 | 2025-12-17 | Gemini chat 46f11897 | « je veux essayer de répliquer le grokking mais pour des datasets de langage » ("I want to try to replicate grokking but for language datasets") | Language grokking; testable hypotheses from physics, maths, linguistics | Rep-20 gave the finite-set recipe and a French-conjugation design. Never run. Revived Oct 27, 2026 (PR #196). | none |
| G4 | 2025-12-17 | Claude chat dc94954c | "What are the benefits of grokking?" | Why grokking matters | Answer tied grokking to longer-input generalisation. Unsupported; the tie stayed in the project goal. | none |
| G5 | 2025-12-17 | Claude chat dc94954c | "StableMax vs. weight decay (+ Grokfast)" | Choose between two accounts of grokking | Grokfast and wd 1.0 built (PR #45). StableMax never built; the chat's StableMax code was plain softmax. | #255, #254 |
| G6 | 2025-12-17 | Claude chat dc94954c | "add a knob for fixed vs. infinite data, plus add Grokfast & WD" | A finite training set as a flag | Built (Codex task PR045). 11 runs Dec 17 to 19, none grokked, all on the broken objective; Grokfast NaNs. Crashes today at the first train-eval. | #136, #214, #215, #137 |
| G7 | 2025-12-17 | Claude chat dc94954c; Codex task PR046 | "I'm trying with just parity" / "Add a knob to choose one algorithmic task to focus on" | Isolate one task | Built (`--alg_tasks`). Parity and mod_add runs, before the label fix. | #253 |
| G8 | 2025-12-18 | Codex task PR052 | "a fixed evaluation based on the fixed data" | Measure the train-vs-test gap on the fixed set | Built for `--fixed_data` only, which crashes. On the stream, "train_eval" is a second fresh sample. | #136, #242 |
| G9 | 2025-12-19 | Claude chat 23d47afd | "Isn't it concerning that loss plateaued this high?" | The 0.53 loss floor is suspicious | Right: the floor was random-operand entropy under the double label shift. Fixed Dec 20 (PR #80). | done |
| G10 | 2025-12-19 | ChatGPT chat 6945b243 | "Here are a couple runs that I have tested, so you see what I can mean." (runs `exp0_grokking_mod_add_v4`, `exp0_parity_v8`) | Repeated single-task grokking attempts | All before the label fix, so invalid. No record kept. | #253, #189 |
| G11 | 2025-12-23 | Perplexity Nash-SEC thread turn 1 | « travaux sur le "grokking" … (addition modulaire vs multiplication) » ("work on grokking … modular addition vs multiplication") | How networks decompose arithmetic | Answered with Gromov 2023 and Fourier circuits. No experiment. | none |
| G12 | 2025-12-24 | Gemini chat dd319d3d | « qu'est-ce qui peut causer l'effondrement neural » ("what can cause neural collapse") | Real-time metrics for collapse | Rep-36 covered class-level neural collapse, not LM training collapse. Only `warn_on_format_crash` built. | #226, #245 |
| G13 | 2026-10-26 | ChatGPT chat 6adf95f7 | "all of my grokking experiments failed. I don't know why" | The grokking line has failed | Led to the code review (#132 to #198) and the Oct 27 audit, which says the regime was never run. | #259, #249 |
| G14 | 2026-10-27 | PR #196 | "Add lexical-grokking research note and unseen-token-rows probe" (PR title) | "Paradigm grokking" on a synthetic inflection language | Open PR with a literature note and probes. It does not cite the Dec 17 report. | none |
| G15 | 2025-12-19 20:14 | ChatGPT chat 6945b243 | "The model won't grok, probably because it is too large. And, apparently, it won't even memorize." / "it won't ever grok, because it won't memorize, so that is the source issue we need to identify." | Memorise first: find why memorisation fails before chasing grokking | He set this scope for Rep-23a; the tiny-overfit test came later in the same chat (Rep-23). Never run. | #256 |
| G16 | 2025-12-19 20:14 | ChatGPT chat 6945b243 | "Maybe we have to only introduce a small subset of new problems every epoch, or something of the like." | A repeated subset inside the stream | Two days after Claude's Rep-19 suggested a 25 % repeated subset; ten months before the Oct 26, 2026 advice in D10. Never built. | #257 |
| G17 | 2025-12-17 19:10 | ChatGPT chat 69430040 (outside the project) | « Valider, analyser, commenter et critiquer » ("validate, analyse, comment and critique") | Send one AI's grokking design (Rep-20) to a second AI for audit | ChatGPT's critique asked for λ = 0 controls and a pilot (Rep-21); it accepted three unsupported claims. Never run. | #260, #249 |
| G18 | 2025-12-12 03:22 | ChatGPT chat 693b54b7 | "Wait, do we generate a new dataset every epoch?" | Is the "infinite" stream really fresh? | First doubt about stream freshness, ten months before #140. ChatGPT's answer was on stale files (Rep-12a). | #140 |

### 2.2 Recursion, loops and latent reasoning

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| R1 | 2025-12-08 | ChatGPT chat 693745a8; Gemini chat 5905ea6a; Claude chat f50fd8de | "Look at recursive models such as HRM and TRM … find a novel but simple architecture" | The founding idea: a small model that reasons by looping | Built as five recurrent variants. All have defects. Never compared at matched compute. | #258, #132, #133, #134, #135, #145 |
| R2 | 2025-12-09 | Claude chat f50fd8de | "You need to push the envelope much further." | A more radical design | NEXUS (byte patching, test-time memory, COCONUT, multi-token prediction). Rejected the same day. | none |
| R3 | 2025-12-09 | Claude chat 3ee613da | "Give me the baseline." | A control first, then one mechanism at a time | Built. The "≥5 % better at ≤2× time" rule is only in the README. | #258, #169 |
| R4 | 2025-12-09 | ChatGPT chat 69388926 | (adopted) the Exp0 to Exp4 ladder; "Give me an experiment zero baseline" | Baseline, shared depth 6×24, latent loop, ACT, SSM | Built; depth hard-wired. Only Exp0, Exp2 (Dec 17) and Exp1 (Dec 22, Dec 30) ever ran. The v1 report (ChatGPT chat 69378f5b) had a per-token thought state; the sequence-level thought tokens that leak (#135) came with this ladder. | #145, #258 |
| R5 | 2025-12-30 | run record `metrics.loop.json` | `--variant shared_loop` (run flag) | A `shared_loop` twin of DAG-v12 | Stopped at 4,350 micro-steps (3.7 %). No edge over baseline; 2.3× slower per token. Never reported. | #258 |
| R6 | 2025-12-31 | Claude chat 26632ad7 | "Is it possible masking or padding is broken for shared loop and latent variants?" | A variant-specific confound | Padding masks added (PR #126). The future-token leak was missed until Oct 26, 2026. | #135, #232 |
| R7 | 2026-05-01 | Perplexity thread on alternatives to chain of thought | "Are there hypothetical ways to use latent reasoning or induce hidden reasoning … in already trained models?" | Retrofit latent reasoning onto a pretrained model | Answer: true latent loops need custom training, not a LoRA recipe. Not pursued. | none |
| R8 | 2026-07-15 | ChatGPT chat 6a56e38e | "be broader-minded, especially with loop transformers" | Make loop transformers the main axis | ChatGPT's L0 to L6 loop ladder, preceded the same night by a 0 to 8 ladder in the claim ledger (msg 77, §XIII); `shared_loop` becomes a negative control. Not built. | #258 (in part) |
| R9 | 2026-10-23 | ChatGPT chat 6adbfb87 (Session 02) | (adopted brief) "Can models reason in an internal representation more effectively than by generating natural-language chains of thought?" | Latent reasoning vs text chain of thought | Found that recurrent depth and a thought stream are separate hypotheses, and flagged a possible future-token leak, confirmed three days later (#135). | #135, #258 |
| R10 | 2026-10-23 | ChatGPT chat 6adc056f (Session 05) | "Red team. Not redeem." | Attack the recurrence claim | Plan: 1, 2 and 4 passes at equal parameters and budget; full vs truncated gradients. | #258 |
| R11 | 2026-10-23 | ChatGPT chat 6adc056f | "building blocks that might not have been put together yet, or incompletely, or wrongly?" | Known parts combined badly or not at all | History plus step conditioning work together; richer loop links lose to a simple stabilised loop. Not built. | none |

### 2.3 ACT and halting

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| H1 | 2025-12-09 | ChatGPT chat 69388926 | (adopted) the ladder's Exp3 (ACT) rung | Learned halting over thought loops | Built. The halt head has no gradient; halting is batch-wide; the ponder term rewards not halting. | #132 |
| H2 | 2025-12-09 | Claude chat 3ee613da | "Take feedback into account." (reviews with `min_K`, ponder ramp, K histogram) | ACT stability: minimum steps, ramp the ponder cost, log steps | `min_K` built. Ramp and histogram not built. | #132, #231 |
| H3 | 2025-12-11 | ChatGPT chat 693a2b2b | "What do you think of this supplemental analysis? Go point by point." (Hybrid Research Plan) | Anneal the halting threshold 0.7 to 0.99 | Not built. Threshold fixed at 0.99, no flag. | #231 |
| H4 | 2026-07-15 | ChatGPT chat 6a56e38e | "be broader-minded, especially with loop transformers" | Loops before ACT; ChatGPT calls ACT "broken" | ChatGPT's July history report (msg 74) gave all three #132 defects: no differentiable path from the halt head, ponder turned into a Python float, batch-wide halting. ACT deprioritised; nothing filed until Oct 26. | #132 |
| H5 | 2026-10-23 | ChatGPT chat 6adc056f | "Red team. Not redeem." | Adaptive halting only after a useful trajectory is shown; fixed exit rules as controls | Not built. | #231, #132 |

### 2.4 SSM

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| S1 | 2025-11-23 | Gemini chat 6db3c20f | "Can the RNN architecture RWKV be used in a mixture-of-experts model" | Recurrent-state models with sparsity (pre-ReLSM) | Research only. Its "init decay in log space" advice was later broken in ReLSM (#147). | #147 |
| S2 | 2025-12-09 | ChatGPT chat 69388926 | (adopted) the ladder's Exp4 (SSM) rung | SSM rung of the ladder | Built (`ssm`, `ssm_mem`). | #133, #134, #146 |
| S3 | 2025-12-11 | Codex task PR010 | "Add Mamba-2 SSM implementation instead of current placeholder." | Real selective scan with a fallback | Built. The fallback ignores the input and mixes batch and time; no generation cache. | #133, #134, #146, #168 |
| S4 | 2025-12-11 | ChatGPT chat 693b3166 | "How's this for #1?" (optimizer groups) | SSM `A_log` and `dt_bias` out of weight decay | Built (PR #13). | #186 |
| S5 | 2026-07-15 | ChatGPT chat 6a56e38e | asked to add "linear-attention papers" | A modern linear operator (Gated DeltaNet family) at an attention ratio, not a custom Mamba | Not built. | none |

### 2.5 Curriculum and the DAG

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| C1 | 2025-12-11 | ChatGPT chat 693a2b2b | "curriculum design, dataset mixing, training stability" | Deep Research scope | Rep-13: a static mix often beats a staged curriculum; algorithmic data ≈ 15 % of tokens. Neither followed. | #252 |
| C2 | 2025-12-11 | ChatGPT chat 693b3ef6 | "How's this for #2?" | Annealed algorithmic-to-language sampler, 15 % persistent, 5 % lexical noise | Sampler built. Lexical loader never wired (`lex_loader=None`). | #158 |
| C3 | 2025-12-20 | ChatGPT chat 693b54b7 | "I still think difficulty ramps up too fast." | The ramp outpaces learning | Contradicted by the logs: difficulty sat near 0.25. The run was v10 (`num_workers 0`, 2.6 % into its budget), so this is the `smooth` schedule's first plateau (#223), not the worker bug (#141). He came back to it in five prompts. | #223, #141 |
| C4 | 2025-12-20 | ChatGPT chat 693b54b7 | "Ideally, the model should learn all the tasks, but maybe that's not possible." | Goal moves from grokking to "learn all tasks" | Drove the controller and the DAG. | #157 |
| C5 | 2025-12-20 | ChatGPT chat 693b54b7 | "Here is another research report. I think it may be too ambitious." | Trim the Claude curriculum report to a V1 | V1 built: EMA, 85/55 band, cooldown, one-sided jitter, plateau logged only. | #157, #202, #220 |
| C6 | 2025-12-20 | Claude chat df40d43e | "What would a better difficulty/algorithmic mix look like, while preserving the capacity to test for OOD?" | A curriculum that keeps a clean OOD test | Per-task phases and an IID/OOD grid built. The "IID" grid is not the training distribution; addition OOD overlaps training. | #162, #155 |
| C7 | 2025-12-21 | Codex task PR087 | "Make sure task curriculum work with infinite algorithmic generation." | The curriculum runs on the infinite stream | Built. This locked the project out of the finite-set regime. | #214 |
| C8 | 2025-12-22 | Codex task PR102 | "ensuring no task gets starved" | A minimum sampling probability at low difficulty | Built. Never fires (start difficulty is exactly 0.2); dead under the DAG. | #221 |
| C9 | 2025-12-23 | Claude chat 8f01c17a; Codex task PR105 | "The default for that new argument should be 0" | A difficulty floor | `--min_difficulty` built with default 0.0 (Claude advised 0.1). Collapse to 0.0 still possible. | #201 |
| C10 | 2025-12-23 | ChatGPT chat 6949cb7d | "Do you have an idea or hypothesis for why Dyck, Chain, and Successor crashed?" | Tasks collapse while arithmetic takes over | The problem the DAG was built for. The first causes found were parser and format bugs. | #165, #157 |
| C11 | 2025-12-23 | Perplexity Nash-SEC thread turn 1 | « Synthétiser une architecture hybride "Nash-SEC" » ("synthesise a hybrid Nash-SEC architecture") | Nash-MTL gradient balancing plus a bandit over tasks; SVD-entropy collapse metrics | Name came from a Gemini report. No Nash-MTL, FAMO or bandit was built. | none |
| C12 | 2025-12-23 | Perplexity Nash-SEC thread turn 3 | « Doit-on décomposer ou introduire en phases les tâches? » ("Should we break up or phase in the tasks?") | The core curriculum question | Answered with no web search. Upstream of the DAG: the PR #108 spec was written by ChatGPT chat 694c894c on Dec 25 from an uploaded PDF (section 7b.2). | #157 |
| C13 | 2025-12-23 | Perplexity Nash-SEC thread turn 5 | « Où se situerait ces autres tâches? » ("Where would these other tasks go?") | Place copy, reverse, parity, chain, dyck | Graph built in PR #108. The parity → addition edge rests on binary addition; ReLSM's addition is decimal. "Chain" meant pointer-following there. | #157 |
| C14 | 2025-12-24 | ChatGPT chat 694b4280 | "propose better sampling, weighting, and difficulty scheduling logic" / "Let me think about it." | Redesign the controller | Two redesigns offered; not built. The DAG was built that evening: ChatGPT chat 694c894c wrote its spec at 20:19 EST (C25). | #157, #201, #202 |
| C15 | 2025-12-24 | ChatGPT chat 694b4280 | "here's another run with minimum difficulty set at 0.2" | His own one-flag control | 8 of 10 tasks pinned at 0.2. | #201 |
| C16 | 2025-12-24 | Codex task PR107 | "change their defaults from 85 and 55 to 70 and 50" | The gates are too strict | PR closed unmerged, 11 minutes before the DAG task. #157 and the Codex history say it shipped. | #157 |
| C17 | 2025-12-24 | Gemini report "Curriculum Étagé" (prompt lost) | « la pertinence des curriculum étagés » ("whether staged curricula are worthwhile") | Whether to phase at all | The answer warned easy-to-hard can block grokking. Not tested. | #252 |
| C18 | 2025-12-25 | Codex task PR109 | "we will not see meaningful change before the training run is over" | Locked tasks starve and recover too late | Right. ChatGPT had told him the same day that the worry "isn't the real killer" (chat 694c894c, 18:01 UTC). Locked floor 0.02 and post-unlock warm-up built. In DAG-v12 six locked tasks got 1.94 % of examples; only parity unlocked. | #205 |
| C19 | 2025-12-26 | Gemini report "Apprentissage Algorithmique et Mathématique Structuré" (prompt lost) | « leur séquençage optimal ("le quand" et "le pourquoi") » ("their best order: the when and the why") | A final ordered task list | "The paper" of Codex task PR114: graph rebuilt; chain as carry propagation; successor to 32 digits. Its only inline citation is the owner's own PDF. | #157, #211, #139 |
| C20 | 2026-01-01 | Codex task PR129 | "the gates should unlock in the specified order, it's just that the sampling should be done uniformly" | Separate DAG order from task weighting | Built (`--disable_dynamic_task_weighting`). The seed of a controlled comparison. | #222, #252 |
| C21 | 2025-12-12 02:18–02:32 | ChatGPT chat 693b54b7 | "For addition, should we set default for 0.5 difficulty to 4 digits or bump evaluation?" / "Do we need all the experiments to test the OOD hypothesis?" | Keep a clean "train small, test big" probe | ChatGPT: freeze addition (≤ 4 digits) and parity (length 8) as fixed probes; "Parity OOD + Addition OOD on baseline vs 1–2 ReLSM variants is enough". Built as defaults, then lost: by Dec 20 addition trained to 5 digits, later 6. | #155, #162 |
| C22 | 2025-12-12 13:42 | ChatGPT chat 693b54b7 | "What do you suggest for the difficulty ramping based on the literature and what we're trying to achieve?" | A literature-based ramp, eight days before Rep-26 | Three-phase ramp proposed (Rep-12a). Not built; the code got the four-plateau staircase. | #223 |
| C23 | 2025-12-21 18:59 | ChatGPT chat 69484386 | "Please tell me if EMA and tasks weighting are wired correctly." | A wiring audit of the V1 controller | The review quoted six defaults that were not in the repo (Rep-29a). 84 minutes later: "It says difficulty is 0." | #201, #222 |
| C24 | 2025-12-24 02:36 | ChatGPT chat 694b4280 | "Wait, isn't task weighting changing?" | The logged task probabilities never move | Right: under the curriculum `train.task_prob.*` logs the static base table (Rep-35a). | #222 |
| C25 | 2025-12-25 01:15–01:18 | ChatGPT chat 694c894c | "Does that change anything to the idea and order of a DAG instead of introducing all tasks at the same time like we currently do?" / "propose an exact DAG + unlock rules using the metrics you already log" | He already had the DAG idea; he asks for the spec | ChatGPT wrote the whole spec at 01:19 and the Codex prompt at 01:58 (Rep-35b). PR #108 opened at 02:18. | #157 |
| C26 | 2025-12-25 18:49 | ChatGPT chat 694d7ffc (outside the project) | "do we need to hit current full difficulty to open the gates?" | Should a gate require mastery at hard difficulty? | Answer: gates read the EMA at the current difficulty; gating on a fixed hard difficulty suggested. Not built. | #157 |
| C27 | 2025-12-25 18:58–19:10 | ChatGPT chat 694d7ffc (outside the project) | "Why isn't TASK_DIFFICULTY_CONFIG used?" / "I want to only keep task weighting config." / "No, I want each task to follow these:" | One source of truth for per-task phases, no silent overrides | Right: `--use_task_curriculum` forces `linear` and `--fixed_data` passes progress 1.0. He reversed his first answer in 4 minutes. Overrides still exist. | #203, #215, #236 |
| C28 | 2025-12-11 18:29–18:35 | ChatGPT chat 693a2b2b | "Should I add warm-up steps between the phases…" / "Sorry, I meant dataset-mixing inter-phases." | Soften the algorithmic-to-language switch | Origin of `mix_band_tokens`. ChatGPT advised "e.g. 5% of alg tokens", or 0; the default became 50 %, so mixing starts at half the budget. | #158 |

### 2.6 Tokenizer and vocabulary

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| T1 | 2025-12-17 | Gemini chat ba6cfceb | « quel est le meilleur tokenizer … pour les maths et le langage » ("which is the best tokenizer … for maths and language") | Tokenizer for a mixed maths and language model on a 4060 | Rep-22 advised a custom 16k vocabulary and said not to use the Llama tokenizer. | #225 |
| T2 | 2025-12-18 | Codex task PR048 | "Use the EleutherAI/llemma_7b tokenizer for everything." | Llemma 32k for all tasks; no reason given | Built (default). 138 of 32,016 ids are used; the table is 39 % of nano and exempt from decay. | #225, #186, #218 |
| T3 | 2025-12-20 | Claude chat 571b07d4 | "Review the tokenization and padding logic" | Audit the pipeline | Later fixes for trailing spaces created `:=` (#156); pad = EOS left in (#232). | #232, #156 |
| T4 | 2025-12-13 | Codex task PR026 | "Remember not to strip the operators or special characters" | The normaliser must keep task symbols | Built. | done |
| T5 | 2025-12-11 | ChatGPT chat 693b3166 (owner's Hybrid Research Plan) | (from his plan, as quoted) "most embeddings never get gradients" | Cold embeddings at the language switch | Remedies: lexical noise (never wired) and no decay on the table (built). With a tied head the unused rows do get gradients and drift together. | #186, #218, #158 |

### 2.7 Concept-level units

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| K1 | 2026-02-20 | Gemini chat 834815ac | "papers, old and new, about decomposing text into concepts and other quanta/units of information?" | Units of meaning above the token | Research only. | none |
| K2 | 2026-02-21 | Gemini chat 60df30f8 | "reason over structured semantic concepts prior to textual decoding" | Reason first, decode after, at concept scale | DLCM named as the closest published shape to ReLSM's memory compressor plus loop. Not built. | none |
| K3 | 2026-02-23 | Claude.ai chats 13087a35, a460a854 | (from an AI summary) "Why not make concepts into discrete tokens?" | A concept codebook; literals (numbers, names) kept as typed pointers | Claude called his proposals "largely ahead of the curve": flattery, not evidence. ChatGPT also recalls a Feb 23 chat of its own on the same idea (chat 6adbf078 msg 26; that chat is not in the export). Not built. | none |
| K4 | 2026-10-23 | ChatGPT chat 6adbf078 | "I'm not asking you to butter me up" | Did his earlier concept prompts anticipate Next-Concept Modeling? | Answer recovered (msg 26): a "very strong conceptual match", but not ahead. ConceptLM appeared 14 days before his Feb 23 chat; DLCM on Dec 31, 2025; Meta's LCM in Dec 2024. His split of logic from factual literals is not covered by those methods. ChatGPT also said the detailed ReLSM and NEXUS designs came from AI proposals, not from him. | none |
| K5 | 2026-10-23 | ChatGPT chat 6adbf8f4 (Session 01) | (adopted brief) "Can language models learn meaningful conceptual units instead of relying entirely on words or subword tokens?" | A learned vocabulary of concepts | Plan: pooled continuous branch vs quantised branch, with a future-summary control; then learned boundaries; then cross with recurrence. Not built. | none |
| K6 | 2026-10-24 15:30 | ChatGPT chat 6adccafc (Session 07) | "why is it important that the concept actually mean something for humans or fit with the codebook established by humans? … would it be more natural to let the model come up with its own representation, or is this point moot?" | Internal codes need not be human concepts | ChatGPT: "Your objection is right." It gave the invertible re-coding argument, a collision test for lost information, and the "preservation contract" with an old-vs-new decoder test. A probe on the thought state would apply it. Not built. | none |
| K7 | 2026-10-26 | ChatGPT chat 6adeb9ac | "mask the numbers in the second sentence if there's a causal link" | Text JEPA with a sliding window and meaning-targeted masks | He then rejected his own linguistic masks: "The whole premise is moot, isn't it?" ChatGPT agreed. | none |
| K8 | 2026-05-02 16:04 | ChatGPT chat "Emoji-based Reasoning Ideas" (outside the project) | "give a series of dense tokens ... and have the model create its own language ... specifically for reasoning" | A reasoning language the model invents, judged only by its answers | ChatGPT turned it into neutral `<Z###>` code tokens with six kill tests (Rep-57). Five months before Hybrid Reasoning Agents (Oct 2026), which ChatGPT later called close in philosophy. Not built. | #266 |
| K9 | 2026-05-02 16:14 | same | "I find the interpretability idea completely moot. AI models are already black boxes." | Judge intermediate codes by performance and validity only | ChatGPT kept causal ablations (remove, shuffle, swap codes) as "experimental hygiene". | #264 |
| K10 | 2026-10-24 15:43 | ChatGPT chat 6adccafc (Session 07) | "Explore mathematics and machine representation and programming languages … compilers, ASTs, machine language, intermediate representations" | The compiler analogy for internal codes | His request, not ChatGPT's idea. Answer: merged states must have compatible futures (bisimulation); abstract interpretation; e-graphs. Not built. | none |

### 2.8 Weight decay and optimizers

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| W1 | 2025-12-11 | ChatGPT chat 693b3166 | "How's this for #1?" (second version exempts `tok_emb` and `lm_head`) | Embeddings and head out of weight decay | Built (PR #13). The reason, lexical collapse at the switch, lapsed: no language phase ran. | #186, #251 |
| W2 | 2025-12-17 | Codex task PR045 | "Is shuffle correct?" (on a pasted recipe: wd 1.0, Grokfast λ 5, "50x speedup") | Default wd 1.0 plus Grokfast | Built. No completed run used both. | #251, #254, #137 |
| W3 | 2025-12-17 | Claude chats dc94954c, 23d47afd | `--fixed_data_size 4000 --grokfast --weight_decay 1.0`, then `--weight_decay 0.1` | Fast search: wd 1.0, 0.3, 0.25, 0.1; λ 5, 1, 0.5; sets of 2k to 40k | NaNs (Grokfast under accumulation) and the double label shift. Most runs stopped at 10 to 25 %. | #137, #251 |
| W4 | 2025-12-19 | ChatGPT chat 6945e825 | `--weight_decay 0.0 --max_lr 1e-4 --grad_clip 0.1` (run command) | Follow the advice to drop weight decay | Against Rep-18 two days earlier. Every later run used wd 0.0 or 0.1. | #251 |
| W5 | 2025-12-19 | Codex task PR056 | "Add Gradient Norm, Weight Norm and Weight Decay Pressure to console logging." | Watch norms as the grokking signal | Built. In DAG-v12 the total norm is mostly 31,879 unused embedding rows drifting. | #218, #149 |
| W6 | 2025-12-17 | Claude chat dc94954c | "Can we play with the LR, warmup steps & warmup algorithm?" | The schedule blocks learning | `--warmup_steps` built. The step estimate assumes 40 tokens per example (measured 4.6 to 18). | #248 |
| W7 | 2025-12-20 | Claude chat df40d43e | "Isn't LR decaying too fast?" | LR schedule | Right. WSD and `min_lr` 1.5e-4 built on the broken objective; never re-tuned. | #173 |
| W8 | 2026-01-11 | Gemini chat ce77dd7d | `--weight_decay 0.1 … --task_curriculum_strategy dag` (run command) | DAG baseline at wd 0.1, not the 1.0 default | Three crashes (pickling, paging file, duplicate key). | #251 |
| W9 | 2025-10-16 | Claude.ai chat 62127e14 | "That aggressive 1e-4 strategy has worked twice in a row now!" | From LoRA work: high LR, few epochs | ReLSM's max_lr 1.5e-3 was never swept. | none |

### 2.9 Data and tasks

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| D1 | 2025-12-08 | ChatGPT chat 693745a8 | "Small datasets under 1 million tokens, but think hard about the issue of scalability" | Train from scratch on very little data | Raised to 2B tokens on Dec 9. Runs used 20M to 100M supervised tokens. | none |
| D2 | 2025-12-09 | ChatGPT chat 69386da6 | "this is also an issue we are trying to solve … training on very small datasets" | Sample efficiency is the research goal | Came back on Oct 24 to 26, 2026 as data efficiency. | none |
| D3 | 2025-12-09 | ChatGPT chat 69388926 | "general multi-domain text with raw text, non-reasoning, and reasoning data" | A language mix plus a synthetic generator | Ten generators built. The language mix never ran after Dec 17. | #158, #159 |
| D4 | 2025-12-11 02:34 | ChatGPT chat 693a2b2b (also in its fork 6941b170) | "What about adding the Project Gutenberg dataset for better creative capabilities?" | Creative writing; teacher-made synthetic data | Not built. | none |
| D5 | 2025-12-11 | ChatGPT chat 693a2b2b | "What about grouping data by theme instead of by dataset" | Topic-based mixing | Not built. | none |
| D6 | 2025-12-13 | Gemini chat 4ddca337 | "Are there medium or large corpora of fiction … that are liberally licensed?" | Clean-licence fiction | Not used. The language config is TinyStories, Simple Wikipedia, MiniPile. | none |
| D7 | 2025-12-22 | Gemini chat d75ff83a | "What are the different templates such as jinja to train an AI model from scratch" | Prompt templates | Prompt masking built (Dec 25). The bit-for-bit template identity test was not. | #184, #156 |
| D8 | 2025-12-25 18:19 | ChatGPT chat 694d7ffc (outside the project); chat 694c894c | "Copy and Reverse don't seem to be training well at all. Try to find why that is." / "alg_seq_len=64 fixed it!" | Copy and reverse fail | In 694c894c it led to prompt masking (PR #110) and the truncation question went unanswered. In a new chat the same day ChatGPT found the cause: examples cut at `alg_seq_len` 32, so about 11 answer tokens survive (Rep-35c). He fixed it at 64. DAG-v12 ran at 48 four days later and truncated again. | #139 |
| D9 | 2025-12-26 | Codex task PR112 | "Could we do dynamic padding & batching of sequences of different lengths?" | Less padding | Built. Pad = EOS then masks the real EOS. | #232 |
| D10 | 2026-10-26 | ChatGPT chat 6adf912d | "Can I train a model by streaming the dataset, or do I have to download large datasets in advance?" | Streaming under storage limits | Answer: mix a repeated subset into the stream (Charton and Kempe). He had proposed this himself on Dec 19, 2025 (G16). | #257 |
| D11 | 2026-10-26 | ChatGPT chat 6adf95f7 | "to make sure I keep healthy sources and don't run into copyright problems" | Licence-clean data (FineWeb-Edu, Wikipedia) | Advice only. | none |
| D12 | 2025-12-11 02:23 | ChatGPT chat 693a2b2b (also 6941b170) | "I have pinpointed the datasets in datasets.txt for training, and I would also like to use subsets of fineweb 1 & 2 and finewiki." | His own 16-source list (NuminaMath-1.5, orca-math, MegaScience, WildSci, smoltalk2, MedReason and others) | ChatGPT made it a "Phase 2b" SFT cap of 50 to 100M tokens over a FineWeb-Edu and FineWiki backbone. Never built; the list is only in this chat. | #261 |
| D13 | 2025-12-11 16:00 | ChatGPT chat 693a2b2b | "All the phases for all the models of the ladder. As well as context length. As we already planned, I believe 4K vs 16K." | A token plan for every rung and phase | Pilot 100M (20M algorithmic, 60M language, 20M SFT) and serious 500M plans. Never run; context never left 128. | #213 |
| D14 | 2026-03-02 22:26 | ChatGPT chat "Largest De-duplicated Dataset" (outside the project) | "what is better between huge token numbers, even with noise, or distillation or curated datasets with a lot less tokens?" | Quality vs quantity, eight months before the October data-frugal chats | Answer: repeating a filtered set up to 10 epochs beats one pass over a 10× larger unfiltered set (Rep-56). | #257 |
| D15 | 2026-03-02 20:08 | same | "is there a way to stream the data, especially do deduplication on the fly while streaming it?" | Streaming with online dedup on small disks | StreamingDataset or WebDataset; offline MinHash/LSHBloom plus a Bloom filter while streaming. Not built. | #159 |

### 2.10 Evaluation and rigor

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| E1 | 2025-12-09 | Claude chat f50fd8de | "I have submitted this problem to 8 experts … Critically analyze their research" | One brief to several AIs, then a judge | His method for the whole project. It found the double shift (Dec 20); it also produced contradictory advice (section 5). | #189 |
| E2 | 2025-12-09 | ChatGPT chat 69386da6 | "Can the Baseline Transformer be an out-of-the-box open source model?" | A pretrained open model as a reference | Dropped Dec 12: "This will only be used with custom models for now." | none |
| E3 | 2025-12-11 16:47 | ChatGPT chat 693a2b2b (also in its fork 6941b170) | "if we vary the number of tokens during training, won't that screw our ablation experiments?" | Token budget as a confound | Answer: keep total tokens, phase split, mix and LR schedule identical within a rung; at 4k vs 16k match tokens, not steps. A rule, not code. Budgets count answer tokens only. | #217, #189 |
| E4 | 2025-12-16 | ChatGPT chat 6941a570 | "check if we haven't strayed from the original design" | Drift from the design | Rep-16 was weak. README still describes the old design. | #169 |
| E5 | 2025-12-19 | Claude chat d95daf7e | "What would be good closeness-to-answer metrics for every task?" / "Try to keep 3-4 metrics" | Partial credit before exact match | Four universal metrics built. Cyclic distance for mod_add dropped. Distance saturates at 16 digits. | #164, #178 |
| E6 | 2025-12-19 | Claude chat 23d47afd | "Is there a better way to evaluate?" | A teacher-forced memorisation check | Built (PR #53), broken by tokenisation, now dead code; a duplicate key drops the accuracy. | #167, #214 |
| E7 | 2025-12-20 | Gemini chat 45e61603 | "Predictions get punctuation stripped but targets don't, causing false negatives." | Low scores are partly a meter artefact | Right; fixed (PR #71). His doubt about the last-number rule ("or should be 42?") was right too. His hash seeding created #160. | #165, #160 |
| E8 | 2025-12-20 | Gemini chat 45e61603 | "Pad tokens are added to input_ids but the model's attention mechanism doesn't mask them out." | Padding treated as context | On Dec 23 he chose to wait (E21); ChatGPT had called the missing mask harmless for right padding. Masks added Dec 31 (PR #126). | #232, #172 |
| E9 | 2025-12-20 | Claude chat df40d43e | "Are you sure our evaluations are correct?" | Doubt the eval itself | Right. A perfect answerer scores 0.78 to 0.83 today. | #144 |
| E10 | 2025-12-20 | Claude chat df40d43e | "What do you think of this other analysis?" | Bring an outside analysis | It found the two-ahead label bug (PR #80). | done |
| E11 | 2025-12-20 | Codex task PR082 | "Separate IID from OOD evaluation. Gate OOD evaluation behind a flag." | Evaluation protocol | Built. | #162, #142 |
| E12 | 2025-12-24 | Gemini report "Stratégies d'évaluation" (prompt lost, restated) | « évaluer … sur des problèmes générés qu'il n'a jamais vus, utiliser un cache de problèmes déjà rencontrés » ("evaluate on unseen generated problems, or a cache of seen ones?") | What a gate on fresh data measures | The sharpest question in the corpus. On an infinite stream, memorisation vs generalisation is undefined. Gates read a fixed seeded set. | #162, #242, #206 |
| E13 | 2025-12-28 | Codex task PR119 | "Are there other bugs like this one elsewhere?" / "No other padding or masking bugs?" | A class of silent bugs | Right. PR #126 found them three days later. | #184 |
| E14 | 2026-01-01 | Claude chat 81a65579 | "Logs seen to be hard to parse by AI because of contradictory metrics" | One metric schema readable by agents | Built (PR #128); it crashed every eval until PR #131. | #167, #198, #189 |
| E15 | 2026-01-11 | Gemini chat f5bb0569 | "How come evaluation results at the same difficulty differ so much?" | Eval vs train difficulty | The printed difficulty was a placeholder. | #237 |
| E16 | 2026-05-01 | Perplexity thread on alternatives to chain of thought | "demonstrated Statistically Seniorficant [significant] gains" (dictation) | Statistical significance on local tests | ReLSM evals use one seed. | #189 |
| E17 | 2026-10-23 | ChatGPT chat 6adc056f | "actually have some gates and completeness criteria" | Research with completeness gates | Applied in the October sessions. | #189 |
| E18 | 2026-10-26 | ChatGPT chat 6adf95f7 | "I have not kept a lot of experiments" | Run records were lost | Only DAG-v12 has a checkpoint folder. | #189 |
| E19 | 2026-10-26 | ChatGPT chat 6adf95f7 | "I know there were already some problems and bugs found … I've never followed through until today." | A full code review | Done: #132 to #198, then the Oct 27 audit. | #132 to #198 |
| E20 | 2025-12-23 02:44 | ChatGPT chat 6949cb7d | "Compare is at 60% on the IID difficulty 0.5 evaluation, but at 14% for the train evaluation at difficulty 0. That doesn't make sense." | The train-eval scores are wrong | Right: the train-eval targets were corrupt (successor expected `'> 7'`, a separator tie-break; a copy target cut to `'8 4 1 7'` by the 32-token limit). The controller had driven compare to difficulty 0.00 on those scores. | #201, #206, #139 |
| E21 | 2025-12-23 01:32–01:43 | ChatGPT chat 6949cb7d | "Are you sure we are using attention mask?" / "leave the masks the way they are. We do not want to introduce attention masks at this point in time." | Padding and the attention mask | Right that masks were only on the loss. His decision deferred them; reversed Dec 31 (PR #126). | #232, #172 |
| E22 | 2025-12-23 01:15–01:27 | ChatGPT chat 6949cb7d | "There are a lot of empty predictions too." / "change all templates to add answer and colon, to correctly pad and parse this new format, and to patch evaluation to use tokens instead of string in the parser." | Empty predictions; one answer marker | Right: two causes (parser on colon-ended prompts, EOS first). Origin of PR #103; ChatGPT wrote its task text (msg 30). The string parser survived. | #156, #165 |
| E23 | 2025-12-20 17:06–17:25 | ChatGPT chat 6946d7ab | "my model doesn't seem to memorize anything … Are there other fine-grained AI metrics I should be adding" / "I would leave the over 5,000 steps one." | Real-time diagnostics on consumer hardware, with a cost cadence | Rep-25a; its Codex spec became PR #73. Two of the three keys measure something else (#238, #245). | #238, #245, #234 |
| E24 | 2026-03-02 23:06 | ChatGPT chat "Largest De-duplicated Dataset" (outside the project) | "for the same compute level, a below 5% change on benchmarks is ... totally within the statistical error" | A minimum effect size of his own | His first explicit significance rule. ReLSM evals still use one seed. | #259 |
| E25 | 2026-09-28 | ChatGPT chat "AI Harness Comparison" (outside the project) | "I just want it to be statistically correct and telling." / "I don't want any benchmaxing" | Small-n evaluation done right | Per-task × per-seed matrix, paired design, evaluator preflight, contamination needs evidence. Not applied to ReLSM. | #259, #189, #144 |
| E26 | 2025-10-16 17:43 | ChatGPT chat "Axolotl config for QLoRa" (outside the project) | "I tested two configs on a single H200, and surprisingly, this one gave the best results." | Trust his own run over the AI's prior | One run per config, and the eval never fired (about 3 optimizer steps per epoch against `eval_steps: 25`). | #189 |
| E27 | 2025-10-17 | ChatGPT chat "OOM fix for evaluation" (outside the project) | "Hmm, you forgot the other changes we talked about." / "I'm supposed to be in qlora 4-bit." | AI agents edit configs silently | The "full" config added `use_rslora: true` (about 11× larger updates) and dropped 4-bit loading. He caught it from the printed resolved config. | #187, #189 |

### 2.11 Language phase and transfer

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| L1 | 2025-12-08 | Gemini chat 5905ea6a | "This doesn't seem to cater to multi-domain text generation tasks." | ReLSM as a general text model, not a puzzle solver | The language loader is a prototype. No language run after Dec 17. | #158, #159 |
| L2 | 2025-12-09 | Claude chat 3ee613da | "No, it needs to be multi-domain." / "I will use available datasets, except for arithmetic." | Real corpora; synthetic data only for arithmetic | As L1. | #158, #159 |
| L3 | 2025-12-17 | Claude chat dc94954c | "With how many tokens should I retest?" | Retest budget | Answer: "100M tokens on algorithmic data only". Every later run had 0 language tokens. | #158 |
| L4 | 2025-12-23 | Perplexity Nash-SEC thread turn 7 | "What could that look like?" (Nash-SEC for language) | Carry the curriculum to language | NLP report (BERT-base, 19 tasks), written with no web search. Never committed. | none |
| L5 | 2025-12-24 | Perplexity Nash-SEC thread turn 11 | "Could we mix algorithmic & language tasks so they build off each other?" | Cross-domain scaffolding and transfer | Not built. | none |
| L6 | 2025-12-24 | Perplexity Nash-SEC thread turns 12 to 14 | "What about maths?" … "Yes!" | Maths as a third domain (45 tasks) | Accepted; never committed. | none |
| L7 | 2026-10-26 | ChatGPT chat 6adf95f7 | "I think 100 billion is feasible … Maybe 200, but probably not much more." | A future language model on 100 to 200B clean tokens | Advice: 500M to 1.5B parameters; grokking not a prerequisite. | none |
| L8 | 2026-10-26 | ChatGPT chat 6adf95f7 | "can we get to a competitive small local model with little training data?" | A competitive small local model | Open. | none |

### 2.12 Hardware and budget

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| X1 | 2025-12-08 | Claude chat f50fd8de | "<10 hrs of training on a single B200 or less & inference on a RTX 4060" | Founding constraints | Every run was on the 4060 laptop. No B200 run on record. | none |
| X2 | 2025-12-09 | Claude chat f50fd8de | "I'm looking to match or beat 4B models and scale up." | Ambition jump | A 1.5B looped hybrid plan; never built. | none |
| X3 | 2025-12-09 | ChatGPT chat 69378f5b | "Up to 2B; what if we trained on up to 16k context?" | 2B tokens, 16k context | Context never left 128 (algorithmic runs used 48). | #213 |
| X4 | 2025-12-17 | Claude chat dc94954c | "it's slow (1500 tok/s) & taking 7.6GB VRAM" | Throughput limits | No action. | none |
| X5 | 2025-12-21 | Claude chat 3a6ca91d | "Tok/s has been cut by 4!" / "We now upcast gradients to f32. Could that be it?" | Throughput regression | The cause was Manager IPC; PR #130 brought part of it back. | #192 |
| X6 | 2025-12-21 | Claude chat 3a6ca91d | "I have always used 0 worker because I'm on Windows." / "What about when running on Linux?" | DataLoader workers | Workers above 0 freeze difficulty. | #141, #197 |
| X7 | 2025-12-28 | Claude chat 135e58ea | "optimizations targeting a laptop RTX 4060 … preserving the integrity of our ablation ladder" | Speed without breaking comparability | Generic answer; nothing built. | none |
| X8 | 2026-10-24 | ChatGPT chat 6adcfef5 | "What about the super sparse architectures?" | Capacity beyond 8 GB VRAM | 65-source register (now on disk). One ReLSM link: per-depth KV cache in looped models. | #145 |

### 2.13 Later directions in 2026

| ID | Date | Source | Owner's words | Idea | What happened | Issue |
|---|---|---|---|---|---|---|
| Z1 | 2026-02-08 | Claude.ai chat 3074e0e1 | "What's the non-financial bleeding-edge of extracting faint signals from noise?" | Signal-to-noise as a general question | Came back Oct 24 (module gradient SNR, spectral clipping). | none |
| Z2 | 2026-05-01 | Perplexity thread on alternatives to chain of thought | "look at self-guided and evolutive approaches as well" | Self-training and evolutionary methods | Survey only. | none |
| Z3 | 2026-07-15 | ChatGPT chat 6a56e38e | "Yes, can you set it up?" / "There is also the J-SPACE paper by Anthropic" | A monthly frontier-architecture watch; the J-Space workspace result | Runs in ChatGPT; nothing in the repo. ChatGPT's scheduler wrote the task prompt from his requests (three versions, final at 20:33 UTC). Reports on Aug 1, Sep 1, Oct 1 and Oct 23 (Rep-55, section 7). | none |
| Z4 | 2026-07-29 | Gemini chat af708951; Claude.ai chat 9116181b | "I want to distill a large model as cheaply and easily as possible" / "You invented my goals, model choice, and budget." | Cheap distillation (another project) | Claude had imported ReLSM's constraints; he caught it. | none |
| Z5 | 2026-10-23 | ChatGPT chat 6adbf078 | "What about NCM and another approach that was really hyped up on the internet in the past month?" | Next-Concept Modeling; Hybrid Reasoning Agents | Answers recovered (Rep-47a): NCM predicts hierarchical residual VQ-VAE codes level by level; the hyped item was a video on NCM with Hybrid Reasoning Agents. The chat had opened on diffusion LMs ("Sorry, not distillation. Diffusion."). 2026 items not checked. | none |
| Z6 | 2026-10-23 | ChatGPT chats 6adbfdba, 6adbff8d, 6adc0c7b (Sessions 03, 04, 06) | "I want you to do the research." / "Don't box yourself in with the project." | Diffusion LMs; parallel generation; end-to-end cost accounting | Reports not on disk. Small diffusion LMs and checker-scored tasks suggested. | none |
| Z7 | 2026-10-24 | ChatGPT chat 6adcf1ea | "train models from the smallest number of data, no matter the type of model" | Data-frugal training across fields | Eight items for ReLSM; a short procedural warm-up beats "a long preliminary grokking phase". | #251, #257; none for the warm-up |
| Z8 | 2026-10-24 | ChatGPT chat 6adcf1ea | "What about quantization-aware training, as well as pruning and distillation in training pipelines?" | QAT, pruning, distillation | Three separate questions; out of ReLSM's scope. | none |
| Z9 | 2026-10-24 | ChatGPT chat 6adce7ea | "connect the dots … create a full complete, full context, self-contained report" | One integrated report over Sessions 01 to 07 | 15,800-word report, not on disk. It "corrects the historical grokking prescriptions". | none |
| Z10 | 2026-07-15 01:34–01:51 | ChatGPT chat 6a56e38e | "@GitHub Summarize the project history and state from chats, sources & code." / "The field has evolved considerably since. List all claims, proposed or rejected, and compare with the state of the art." | A project history and a claim ledger | The history report (msg 74) and the claim ledger (msg 77), both in the export (Rep-45, Rep-45a). Nothing filed for three months. | #272 |
| Z11 | 2026-07-14 21:09 | ChatGPT chat 6a56a593 (outside the project) | "Can you put all our conversations from December 2025 about ReLSM into one project?" | Gather his own history before re-auditing | ChatGPT cannot move chats; it listed the December groups. The re-audit started the next day. | none |
| Z12 | 2026-10-24 21:17 | ChatGPT chat 6add204a | (pasted brief; who drafted it is not recorded) "Do not import an old ≤10 GB disk constraint from another project" / old transcripts "are not scientific evidence. Do not inherit their unsupported claims about 'effective parameters,' 'effective data,' multiplicative improvements" | Research rules for the beyond-VRAM brief | The brief is now recovered in full (10 KB). It also fixes two controls: the same oversized checkpoint in a normal runtime, and a strong smaller resident model. | none |

---

## 3. Timeline of the owner's thinking

| Date | Turning point | Owner's words |
|---|---|---|
| Oct to Nov 2025 | Before ReLSM: LoRA fine-tuning on rented H100/H200 with $50 to $90 budgets; how to shrink models; RWKV-7 mixture of experts. He already cross-checks AIs. | "Think deeply, explore every angle, be creative" (Gemini chat 6db3c20f) |
| Dec 8 | **Founding brief**, to ChatGPT, Gemini and Claude. Recursion for reasoning on minimal means. | "find a novel but simple architecture for LLM-like tasks such as chain of thought and reasoning" (ChatGPT chat 693745a8) |
| Dec 8 | First pushback: a general text model, not a puzzle solver. | "This doesn't seem to cater to multi-domain text generation tasks." (Gemini chat 5905ea6a) |
| Dec 9 | Ambition jumps, then narrows. An 8-AI tournament, then the baseline and the ladder. "Algorithmic grokking" enters from an AI report. | "I'm looking to match or beat 4B models" … "Give me the baseline." (Claude chats f50fd8de, 3ee613da) |
| Dec 9 to 16 | Building. Codex writes the code; ChatGPT reviews and writes Codex prompts. He worries about drift. | "check if we haven't strayed from the original design" (ChatGPT chat 6941a570) |
| Dec 11 to 12 | Data and probes. His own 16-source dataset list, a token plan per rung, token budgets matched per rung, the mix band. He caps addition at 4 digits and parity at length 8 as clean OOD probes, and first doubts the stream. | "Wait, do we generate a new dataset every epoch?" (ChatGPT chat 693b54b7) |
| **Dec 17** | **"I didn't see the posited grokking."** Same day: language grokking asked of Gemini; three research answers say a finite set is needed; the fixed-data flag, Grokfast and wd 1.0 are added. | "I had no expectations except to see it happen." (ChatGPT chat 6942ce5e); « répliquer le grokking mais pour des datasets de langage » ("replicate grokking but for language datasets", Gemini chat 46f11897) |
| Dec 17 to 19 | Fast search, short runs: wd 1.0 to 0.1, three Grokfast λ, set sizes 2k to 40k, one task, a new tokenizer. All on a broken objective. | "I'm trying with just parity, but it's slow" (Claude chat dc94954c) |
| Dec 19 | Doubt moves from the model to the measurement. He names memorisation as the first problem and proposes a repeated subset. A ChatGPT code audit finds "No fatal coding bugs" one day before the double shift is fixed. Weight decay goes to 0 on ChatGPT's advice. | "Isn't it concerning that loss plateaued this high?" (Claude chat 23d47afd); "it won't ever grok, because it won't memorize" (ChatGPT chat 6945b243) |
| **Dec 20 to 21** | **Curriculum pivot.** The label double shift is found. The goal becomes "learn all tasks". A ChatGPT wiring review of the new controller quotes code that is not in the repo; difficulty reaches 0 the same afternoon. The finite-set recipe is dropped without a recorded decision; the curriculum is tied to the infinite stream. | "Ideally, the model should learn all the tasks, but maybe that's not possible." (ChatGPT chat 693b54b7); "Make sure task curriculum work with infinite algorithmic generation." (Codex task PR087) |
| Dec 23 | Task collapse ("monopolisation"). Nash-SEC research plan in Perplexity. | « Doit-on décomposer ou introduire en phases les tâches? » ("Should we break up or phase in the tasks?", Perplexity turn 3) |
| **Dec 24** | **The DAG.** He defers ChatGPT's two controller redesigns. That evening he asks ChatGPT for "an exact DAG + unlock rules". ChatGPT writes the graph and every value at 20:19 EST and the Codex prompt at 20:58; PR #108 opens at 21:18 EST (chat 694c894c). | "Let me think about it." (ChatGPT chat 694b4280) |
| Dec 25 | His own correct worry about starved locked tasks; prompt masking found. He finds the 32-token truncation of copy and reverse and fixes it at 64. | "we will not see meaningful change before the training run is over" (Codex task PR109); "alg_seq_len=64 fixed it!" (ChatGPT chat 694d7ffc) |
| Dec 26 to 27 | The DAG is rebuilt on an AI report he calls "the paper". | "We are aligning the curriculum with a research paper" (Codex task PR114) |
| Dec 29 to 30 | DAG-v12 runs its full budget: best IID 29.25 %, only parity unlocked, 98 % of examples on four tasks, 6,458 updates. The loop twin stops at 3.7 %. | (run records) |
| Jan 1 to 2 | First step toward a controlled comparison: DAG order with uniform sampling. | "the gates should unlock in the specified order, it's just that the sampling should be done uniformly" (Codex task PR129) |
| **Jan 11 to 12** | **Last run.** Three crashes, then a run to 11,000 micro-steps whose folder is gone. Codex work stops Jan 12. No owner turn after Dec 19 asks about the learning regime. | "How come evaluation results at the same difficulty differ so much?" (Gemini chat f5bb0569) |
| Feb 2026 | Same intuition at concept scale; ReLSM not named. | "reason over structured semantic concepts prior to textual decoding" (Gemini chat 60df30f8) |
| Mar 2026 | Data at scale: quality vs quantity, streaming dedup; his own 5 % significance rule. | "a below 5% change on benchmarks is ... totally within the statistical error" (ChatGPT, outside the project, Mar 2) |
| May 2026 | Latent reasoning on pretrained models, by fine-tuning only. His own idea: a reasoning language the model invents. | "induce hidden reasoning or conceptual reasoning states In already trained models?" (Perplexity, May 1); "have the model create its own language ... specifically for reasoning" (ChatGPT, outside the project, May 2) |
| Jul 15, 2026 | ChatGPT reads the repo and writes a project history (it finds #132's three ACT defects three months before #132) and a claim ledger. Loops to the centre; four grokking conditions stated; the monthly watch is set up. | "be broader-minded, especially with loop transformers" (ChatGPT chat 6a56e38e) |
| Aug to Oct 2026 | The monthly watch reports four times. Its "next decisive experiment" changes each month and never names a grokking control. No ReLSM run. | (ChatGPT chat 6a6de5a9) |
| Oct 23 to 25, 2026 | Seven themed research sessions. A stricter style: red team, completeness gates, no flattery. ChatGPT withdraws its own Dec 2025 multipliers and grokking prescriptions. | "Red team. Not redeem." (ChatGPT chat 6adc056f); "I'm not asking you to butter me up" (ChatGPT chat 6adbf078) |
| **Oct 26, 2026** | **"All of my grokking experiments failed."** New target: a competitive small local model on 100 to 200B clean tokens. First full code review. | "I wasn't able to reproduce it, no matter what I tried." / "I've never followed through until today." (ChatGPT chat 6adf95f7) |
| **Oct 26 to 28, 2026** | **Reset.** Code review (#132 to #198), full audit and research plan (#201 to #259), lexical-grokking session (PR #196). The audit's view: the regime was never run, so "failed" is not established. | — |

Ideas the owner kept returning to:
- **Latent reasoning before decoding**: Dec 8 → Dec 17 → Feb 2026 → Jul and Oct 2026.
- **Language as the real target**: Dec 8, Dec 9, Dec 17, Dec 24, Oct 26, 2026.
- **Is the measurement right?**: Dec 19 to Jan 11, about 20 prompts. He was right each time.
- **Fair comparisons in the ladder**: Dec 9, Dec 16, Dec 20, Dec 28, Dec 31.
- **Tasks must not starve**: Codex tasks PR100, PR102, PR105, PR107, PR109.
- **Memorise first, repeat the data**: Dec 19, 2025 (G15, G16) → Mar 2, 2026 (D14) → Oct 26, 2026 (D10).
- **Several AIs on one question**: from Oct 2025 to Oct 2026.
- **Units above the token**: Feb 2026, May 2026 (dense reasoning codes) and Oct 2026 (5 of the 18 ChatGPT files of 2026).
- **Small models on his own hardware**: Oct 2025 to Oct 2026.

---

## 4. Research received

### 4.1 Every report, by date

Verdict: **right** (main claims hold), **partly** (useful core, wrong parts), **wrong** (wrong on points that matter), **unsupported** (cannot be checked, or no evidence given). Most citations were checked from the readers' memory, offline; 2026 papers mostly cannot be checked.

**Background, Oct to Nov 2025**

| Rep | Date | Who, source | Question | Main claims and recommendations | Papers cited | Verdict |
|---|---|---|---|---|---|---|
| 01 | 2025-10 | Claude Research, Claude.ai chats 41b8e9b1, 62127e14, ded4e71a, 7b45f1cf | QLoRA fine-tuning recipes on rented GPUs | Axolotl configs, costs; "50–58 % MATH-500" for one run | — | unsupported: projections, nothing measured |
| 02 | 2025-10-22 | Gemini, chat 91349111 | How to downsize an LLM | Width over depth pruning, Wanda, distillation; factorised embedding and ALBERT sharing (never linked) | Minitron, Wanda, ALBERT | partly: mixes two Minitron lines; its "LRF" row is LoRA |
| 03 | 2025-11-23 | Gemini Deep Research, chat 6db3c20f | RWKV-7 with mixture of experts; merging | Decay rates in log space; dense frozen memory path; needle fitness ("perplexity: a flawed proxy"); CMA-ES search | RWKV-7, Flock of Finches | partly: its "geometric mean" code is a harmonic mean; RWKV-6 tensor names |
| 01a | 2025-10-16 to 18 | ChatGPT, chats "Axolotl config for QLoRa" and "OOM fix for evaluation" with three branches (outside the project) | QLoRA configs and runs on rented H100/H200 | Think in tokens per update; keep warmup constant in tokens; run a step-0 eval; set the eval cadence from the real step count | — | partly: invented config keys; said long samples are "truncated, not dropped" against a log that said "Dropped 92 samples"; "You're not losing training" after a failed save; silently added rsLoRA |

**Design, Dec 8 to 16, 2025**

| Rep | Date | Who, source | Question | Main claims and recommendations | Papers cited | Verdict |
|---|---|---|---|---|---|---|
| 04 | Dec 8 | ChatGPT Deep Research, chat 693745a8 | Recursive architectures for reasoning | Survey; proposes "ReLoRA": 5–20M, 2-layer shared core, latent tokens, halting trained with BCE, deep supervision, 1 → N loop curriculum | TRM 2510.04871; MoR 2507.10524; Huginn 2502.05171; COCONUT 2412.06769 | partly: numbers right; four wrong attributions; the name clashes with ReLoRA (2023); no grokking |
| 05 | Dec 8 | Gemini Deep Research, chat 5905ea6a | Same | "Iterative Latent Refinement": one think token over a cached context, p_stop halting, 3-phase loop curriculum, test at depth 20 with 30 loops; then adapters on GPT-2 | TRM; UT 1807.03819; ACT; 2402.00976; Mamba copy limits 2410.03810; pause tokens; ListOps 1804.06028 | partly: "deep supervision, unlike HRM" is wrong; λ·T ponder cost has no gradient; its halt head gets no gradient (the #132 defect). Its three defining parts were never built |
| 06 | Dec 9 | Claude, chat f50fd8de | Judge 8 AI proposals | None is simple; HRM/TRM won on puzzles, not text; none proposes ablations | — | right |
| 06a | Dec 9 | ChatGPT, chat 69386da6 | Critique RH-SSM, ReLSM and NEXUS, then Claude's critique of them | ReLSM "most buildable + falsifiable"; L2 residency "numerically suspicious"; NEXUS's gains "wishful multiplication"; the ReLSM PDF's dual-stream attention rule is blank; NEXUS's "Llama-3.2-3B 45% GSM8K" is the 1B score; "define one primary compute budget knob" | — | right as far as checked (the blank rule is probably math lost in PDF extraction) |
| 07 | Dec 9 | Claude Research, chat f50fd8de | 4B-model quality from under 2B tokens | 1.5B "Looped Hybrid Transformer"; progressive loop depth; Rho-1 selection | Saunshi 2502.17416; Relaxed Recursive 2410.20672; Hymba 2411.13676; Griffin 2402.19427; Phi-1 2306.11644; Rho-1 2404.07965; Muennighoff 2305.16264 | partly: paper claims fair; MMLU 52–58 % on 2B tokens unsupported; "wd 0.1 important for looped" unsourced |
| 08 | Dec 9 | Claude, chat f50fd8de | "Push the envelope" | NEXUS: byte patching, test-time memory, multi-token prediction, COCONUT; "2B tokens → ~200B equivalent" | BLT 2412.09871; Titans 2501.00663 | wrong: gains do not multiply (ChatGPT withdrew the same kind of claim on Oct 26, 2026) |
| 09 | Dec 8–9 | AI-written "ReLSM" report the owner brought (ChatGPT chat 69378f5b, inlined in the re-export; Claude chat 3ee613da) | A unified design | Token and thought streams, shared weights K = 32–64, ACT, memory tokens, "core residency" in L2 cache, Phase 1 "algorithmic grokking" then language. v1 keeps a per-token thought state z, but the token stream never reads z. ChatGPT's reply added code (`recurrent_think_with_ponder`, `MemoryTokens`, `HaltHead`) that later appears in the Dec 16 ReLSM PDF | — | wrong in premise: grokking on infinite data; 50M fp16 weights are 100 MB vs 24 MB L2; as specified, v1's thinking cannot change the token; ChatGPT's added code has #132's batch-wide halt and called the design "a very strong consolidation" |
| 10 | Dec 9 | Reviews pasted into Claude chat 3ee613da | The baselines | Split curriculum; RoPE nano and tiny SSM controls; ACT stability (`min_K`, ponder ramp, K histogram); thought-loop pseudocode with mean pooling | Hahn 1906.06755; Chiang and Cholak 2202.12172 | partly: predicted ACT collapse (#132); its pseudocode is the #135 leak |
| 11 | Dec 9 | "Counter-analysis" PDF the owner brought (ChatGPT chat 69378f5b) | Critique | "Grokking headroom": train beyond memorisation | — | unsupported as used: assumes a finite set |
| 12 | Dec 11 | "ReLSM Hybrid Architecture Research Plan" PDF the owner brought (ChatGPT chats 693a2b2b, 693b3166) | Plan | Hard switch causes forgetting; SSM weight-decay trap; thought-token collapse → orthogonal init; cold embeddings; replay; sandwich norm; ACT annealing; WSD | — | partly: SSM decay point right; "embeddings never get gradients" is wrong with a tied head |
| 12a | Dec 11–12 | ChatGPT, chats 693b54b7 and 69484386 (shared part, msgs 113–244) | Review of the `data.py` diffs; OOD probes; a ramp from the literature | Keep addition (≤ 4 digits) and parity (length 8) as fixed OOD probes and let difficulty vary on the other eight tasks; "Parity OOD + Addition OOD on baseline vs 1–2 ReLSM variants is enough"; three-phase ramp (U(0, 0.3) for 10 %, a linear ramp 0.3 → 1.0, then a 70/30 mix); `cycle(iter(loader))` replays the first pass | Bengio 2009; Graves 2017 | partly: the probe design is sound and was never written down; the replay finding was on stale files (fix 30e0069 already committed); four reviews missed the shared digit count (#210) |
| 13 | Dec 11 | ChatGPT Deep Research, chat 693a2b2b | Curriculum, data mixing, stability | Avoid hard switches; a static mix often beats a staged curriculum; short synthetic warm-up; algorithmic ≈ 15 % of tokens; re-warm the LR at phase changes | DoReMi 2023; RegMix 2024; SPAM 2025; SkyLadder 2025 | partly: GPT-3 peak LR "2.8e-5" is wrong (0.6e-4); one cited project is probably ReLSM's own README; "Perfume/PerfRe" unidentified |
| 14 | Dec 13 | Gemini, chat 4ddca337 | Liberally licensed fiction | PG-19, SCP, ELTeC; a risk ladder to grey data | — | partly: Royal Road is not in The Pile; Common Pile and Common Corpus confused |
| 15 | Dec 15 | Claude, chat 2f16ee49 | Creative-writing dataset with reasoning traces (other project) | LIMA; an "unslop" pipeline | LIMA | partly: budget arithmetic inconsistent |
| 16 | Dec 16 | ChatGPT Deep Research "Bugs", chat 6941a570 | Code vs design | Global decay on SSM parameters; batch-pooled halting | — | wrong: its lead bug was fixed Dec 11; "no obvious runtime bugs" while #133–#135 were present |
| 17 | Dec 11 | ChatGPT, chat 693a2b2b (16:15 UTC; inherited by the later fork 6941b170) | Creative datasets and throughput | LongPage, Neural-Story-v1, COIG-Writer, the AO3 dump; "30–40 tokens/sec" on the 4060 | — | wrong: off by about 150× (logs show 5,000–7,600 tok/s); nano "~18M parameters" (41.6M); a 300M model on 200M tokens in "~3.5 sec on B200"; every rung at 4,096 context (runs used 32 to 128); the same chat said to avoid AO3 2.7 h later (Rep-13) |

**Grokking, Dec 17 to 19, 2025**

| Rep | Date | Who, source | Question | Main claims and recommendations | Papers cited | Verdict |
|---|---|---|---|---|---|---|
| 18 | Dec 17 | ChatGPT Deep Research, chat 6942ce5e | Why Exp0 and Exp2 did not grok | Infinite procedural data "preempts the classic grokking dynamic"; moderate weight decay; training too short; curricula smear one transition; StableMax and ⊥Grad; decoder-only decay; embedding transfer; synergy metrics | Power 2022; Liu 2022; Carvalho 2502.01774; Clauw 2024; Prieto 2501.04697; Li 2506.21551 | partly: **the key diagnosis is right**; three papers get wrong authors; it misread the curves (broken objective, #135 leak) |
| 19 | Dec 17 | Claude Research, chat dc94954c | Same | Infinite data prevents grokking; a 25 % repeated subset; wd 1.0; train fraction 35–50 %; 2-layer d = 128; train 10–100× past fit; Grokfast; GrokTransfer; StableMax; slingshot | Charton and Kempe 2410.07041; Power 2201.02177; Nanda 2301.05217; Liu 2205.10343; Grokfast 2405.20233; GrokTransfer 2504.13292; Prieto 2501.04697; Thilak 2206.04817; Kumar 2310.06110 | partly: recipe right; **its StableMax code is plain softmax**; Grokfast "50×" overstated; its follow-up (grokking ⇒ longer-input generalisation) unsupported |
| 20 | Dec 17 | Gemini Deep Research "Grokking du Langage", chat 46f11897 | Language grokking | A finite memorised set; decay as "pressure", noise as "temperature"; lazy → rich; per-circuit grokking; French conjugation with "famine" of regular verbs, Wug test, wd sweep 0–2.0; recipe d = 128, L = 2, H = 4, constant LR, ≥ 1e5 steps | Power; 2506.21551; Kumar; Rubin 2310.03789; Liu 2022; Murty 2305.18741; 2401.10463; 2505.11411 | partly, with a right core: **the most useful research item in the corpus; never run**. Murty's depth result inverted; one source cited for the opposite of its title; 17 of 42 sources uncited; "structural, not metaphorical" physics is qualitative |
| 21 | Dec 17 | ChatGPT review "Validation méthodologique"; by content very probably ChatGPT chat 69430040 (outside the project, 19:10 UTC, prompt « Valider, analyser, commenter et critiquer ») | Review Rep-20 | Experiment A feasible; controls in every arm (λ = 0, no bottleneck); pilot first; the metric Ω is undefined in practice; too strong a "famine" leaves no rule; physics analogies "largely heuristic" | Arnett and Bergen 2411.14198 (cited correctly) | partly: right on controls; cites a "Poincaré" source that does not exist; accepted three unsupported claims (shallow nets memorise less; full batch is better; French gives a stronger signal) |
| 22 | Dec 17 | Gemini Deep Research, chat ba6cfceb | Best tokenizer for a 50M–1B maths and language model | Embedding tax; split digits; custom 16k vocabulary; size a multiple of 128; do not use GPT-2 or Llama tokenizers | Tokenization counts 2402.14903; Scaling Laws with Vocabulary (NeurIPS 2024); Llemma | partly: its key comparison is false (Llama 2 already splits digits and Llemma reuses that vocabulary); its digit regex does not split digits; the tax arithmetic is right |
| 23a | Dec 19 | ChatGPT Deep Research, chat 6945b243 msg 7 | The owner's scope: baseline only, "why it won't memorise" | "No fatal coding bugs"; wd 1.0 "the single biggest oversight", use 0.1–0.01; **loss on the answer only** (prompt labels −100); raise the clip to 5–10; dropout 0; an absolute-position control; static difficulty 0.3–0.5; `alg_seq_len` 32–64; a fixed set of 1,000–5,000; nano "about 18M", size "likely not the bottleneck" | — | partly: answer-only loss right and the first to say it (built Dec 25, PR #110); "no fatal bugs" wrong (the double shift was in that code); nano is 41.6M; its clip advice is the opposite of Rep-23's; the short `alg_seq_len` later truncated answers (#139) |
| 23 | Dec 19 | ChatGPT, chats 6945b243 (msgs 14, 18), 6945e825 | Analyse the runs | Underfit; lower LR; **wd 0 in the algorithmic phase**; lower the clip to 0.1; fixed data first; one task; **tiny-overfit test on 256–1,024 examples**; nano "might simply be at the edge" | — | partly: tiny-overfit test right (#256); wd 0 contradicts Rep-18; calls nano 12-layer (it has 6; probably taken from the attached plan PDF); "peak LR around 8e-5" (it was 1e-4); read v2's rising LR as a missing decay schedule (v2 was still in warmup) |
| 24 | Dec 19 | Claude, chats 23d47afd, 090b121f | The mod_add runs | Loss floor blamed on wd 1.0; a fresh chat says raise wd to 1.0 or more; small 2–4-layer model; digit tokenizer | — | wrong on the cause (same floor at wd 0.1: the double shift); the two chats contradict each other |
| 25 | Dec 19 | Claude, chat d95daf7e | Code review; closeness metrics | 35 findings; four universal metrics | — | partly: the "+1000" normalisation became #164 |

**Curriculum and DAG, Dec 20 to 30, 2025**

| Rep | Date | Who, source | Question | Main claims and recommendations | Papers cited | Verdict |
|---|---|---|---|---|---|---|
| 25a | Dec 20 | ChatGPT Deep Research "Fine-Grained Metrics for Diagnosing Transformer Training", chat 6946d7ab (17:17 UTC), then a Codex spec (17:26) | Real-time diagnostics on consumer hardware | Activation sparsity "plateaus right before" generalisation; Shannon entropy of the absolute weights; local circuit complexity by weight perturbation; attention-head entropy; mean attention distance; gradient-direction consistency; sharpness; a fixed probe set | markers only; by content Golechha 2024, 2405.12755 (from memory) | partly: the report meant inactive hidden units, but its own spec measured near-zero entries of the FF output (#238). The source of PR #73 |
| 26 | Dec 20 | ChatGPT Deep Research "Curriculum Learning & Difficulty Ramping", chat 693b54b7 | Curricula from the literature | Accuracy-gated advance with patience and backslide; per-task ramps; token accuracy as signal; transfer from a smaller model; a curriculum "can mitigate" grokking | Wilson 2019; Cho 2024; "Chen & Wang (2025)" | partly: GrokTransfer authors wrong; three different target success rates; **treats delayed generalisation as a stall to avoid**, the opposite of the project's goal; never read the logs it was given |
| 27 | Dec 20 | Claude Research "compass" curriculum guide, chat 6e635fde | Curriculum implementation | 85/55 band, cooldown, EMA; Grokfast λ 5; two-sided Gaussian jitter; layer reset; plateau-triggered advance; GradNorm | Wilson 2019; Platanios 1903.09848; GradNorm 1711.02257; Grokfast; Ash and Adams 1910.08475 | partly: thresholds invented; its layer reset can never fire; "EMA 0.98 ≈ 100 samples" is about 50 |
| 28 | Dec 20 | Outside analysis pasted into Claude chat df40d43e | Why runs fail | Labels shifted twice; logger columns ragged; teacher-forced accuracy as a live check | — | **right**: the most valuable outside input of the month (PR #80) |
| 29 | Dec 21 | ChatGPT, chat 69484386 | Run analysis | mod_add groks "only very late … single task, small dataset, strong weight decay"; phased task introduction; weight ∝ max(ε, 1 − EMA); a five-task sanity run | — | partly: summary right; "train-eval rises first" holds only on a repeated set |
| 29a | Dec 21 | ChatGPT, chat 69484386 msg 485 (19:18 UTC) | "Please tell me if EMA and tasks weighting are wired correctly." | "nothing is obviously broken"; with the curriculum on, `weighting_fn` overrides `--task_weighting` (#222); `phased` acts like `smooth` (#203); train-eval at a fixed 0.5 is "expected and not a bug" (#237) | — | wrong: it quoted six defaults that were not in the repo (EMA 0.9, low gate 0.6, a clamp floor of 0.05, `min_weight` 0.2, cooldown 1000, equal base weights with a "sort" task). The real clamp was [0, 1]; 84 minutes later difficulty was 0 (#201) |
| 29b | Dec 21 | ChatGPT, chat 69484386 msg 973 (23:51 UTC) | "You didn't look at the latest file!" (run v12) | It admits it had not opened the new file. Fix `algorithmic_mae`; add a parse-fail rate; per-layer residual RMS; attention distance and memory-token mass; an inner-step histogram; a `grad_to_weight` EMA per block | — | partly: right on MAE (#164, #198) and the halting histogram (#231) |
| 29c | Dec 22 (Dec 21 EST) | ChatGPT, chat 69488a01 msg 1 | "Analyze this run extensively." (the 43,000-step run) | Put wd 1e-3 to 1e-2 back "because weight_norm is running away"; evaluate on a difficulty sweep 0.2/0.4/0.6/0.8; log LR and update scale | — | partly: missed that the "mix" phase was the alias replay (#158); the wd advice was not followed |
| 30 | Dec 23 | Gemini Deep Research "Curriculum d'Apprentissage Algorithmique Pondéré" (prompt lost) | Task weighting and fairness | Weight tasks by a dependency graph (Skill-It); carry as a variable; learning-progress weights | Skill-It; FairGrad 2402.15638; DoReMi | wrong citations: FairGrad, DoReMi and Cornacchia and Mossel described wrongly |
| 31 | Dec 23 | Gemini Deep Research "Recherche Transdisciplinaire sur l'Équité Pédagogique" (prompt lost) | Collapse and monopolisation | Coins "Nash-Sec"; subitizing → successor → an SVD check; a 50 % rule | SEC 2505.14970 | wrong: invented definitions; wrong neural-collapse formula |
| 32 | Dec 23–24 | Perplexity Nash-SEC thread (14 turns) and its four reports | The owner's 6-point plan | Nash-MTL plus a bandit; three phases; copy → reverse; chain + parity → addition; 99.9 % gate; the word "DAG"; 80/20 replay; 10 % floor; wd 1e-4, ">1e-3 eliminates grokking"; NLP and 45-task plans | Nash-MTL 2202.01017; FAMO 2306.03792; FairGrad 2402.15638; LibMTL 2203.14338; Gromov 2301.02679; 2310.03789; Graves 2017 | wrong: **six turns (3, 5, 7, 10, 13, 14) did no web search**; the Nash product on the simplex always returns uniform weights; invented gains; parity → addition is false for decimal sums; its NLP Phase I can never unlock. **Upstream source of the DAG**; the direct source is Rep-35b |
| 33 | Dec 23 | ChatGPT Deep Research, chat 694b2b13 | Assess the new parser | A per-task table | — | wrong: 7 of 10 rows do not match the files |
| 33a | Dec 22–23 | ChatGPT Deep Research, chat 6949cb7d (shared with 694b2b13), msgs 8, 17, 19, 29 | v15 vs v16; copy/reverse termination; empty predictions; attention mask | v16 cascades to the difficulty floor; copy/reverse never emit EOS (blamed on a missing "Answer:" marker and "nondeterminism"); two empty-prediction modes and token slicing; no attention mask is "harmless" for right padding | — | partly: the empty-prediction diagnosis is right; "language modeling data at 50M tokens" is wrong (alias, #158); it missed the 32-token truncation in its own samples (#139) |
| 34 | Dec 22 | Gemini, chat d75ff83a | Training templates | The template is part of the model; train and inference must match bit for bit; mask the prompt | — | partly: two false claims (BOS, ChatML injection) |
| 35 | Dec 24 | ChatGPT, chat 694b4280 | Controller redesign | The controller "slams tasks to the minimum"; redesign A (one controller) and B (learning-progress sampling, floor escape) | — | partly: diagnosis right; not built |
| 35a | Dec 24 | ChatGPT Deep Research, chat 694b4280 (01:31–03:02 UTC) | Compare run4 and run5 | With the task curriculum on, `train.task_prob.*` and `train.task_weight.*` log the static base table; the real distribution is only in `train.task_count_window.*`; log `curriculum.weight.<task>` | — | right |
| 35b | Dec 25 | ChatGPT, chat 694c894c (01:19, 01:58, 02:24, 16:40 UTC) | "propose an exact DAG + unlock rules using the metrics you already log" | The graph (roots successor, copy, reverse; dyck ← copy, reverse; compare ← addition), thresholds 0.99/0.98/0.95/0.97, patience 4, unlock margin +0.01, lock margin 0.03, backslide replay 35 %, ramp 3, floors; the Codex prompt for PR #108; a review of Codex's first diff | — (an uploaded PDF, no filename; by content Rep-38 or the Nash-SEC report) | wrong: every value invented in the answer; it read copy/reverse at 0.0 (a truncation artefact) as a task-order problem; it designed `+ margin`, then called it a bug 14 h later; it told the owner his cold-start worry "isn't the real killer" (#205 shows it was). **Direct source of PR #108** |
| 35c | Dec 25 | ChatGPT, chat 694d7ffc (outside the project, 18:19–19:10 UTC) | "Copy and Reverse don't seem to be training well at all." | Examples are cut to `max_seq_len` before the labels are built; at 32 about 11 answer tokens survive; raise the length, cap the tasks, or skip and resample; per-task `truncation_rate` and `supervised_tokens` counters; gate at a fixed hard difficulty | — | right: the owner confirmed "alg_seq_len=64 fixed it!"; the counters were never built |
| 36 | Dec 24 | Gemini Deep Research, chat dd319d3d | Neural collapse; real-time metrics | NC1–NC4; decay speeds collapse; OOD-detection metrics | Papyan 2020 | partly: misattributes a result; misses LM training collapse and Wu and Papyan 2024 |
| 37 | Dec 24 | Gemini "Amélioration des Approches d'Apprentissage Multi-tâches" | Improve Nash-SEC | FAMO above five tasks; drop learning-progress reward; structural signals; a parity "grokking trap" test | FAMO; CAGrad | partly: rewards a rising weight norm, but grokking under decay comes with a falling norm |
| 38 | Dec 24 | Gemini "Curriculum Étagé : Prometteur ou Risqué ?" | Is a staged curriculum worthwhile? | Only under strict conditions; easy-to-hard can block grokking; decompose, don't simplify; hard gates for arithmetic | Charton and Kempe 2410.07041; Cornacchia and Mossel 2301.13833 | partly: **the one warning against the plan**; not tested |
| 39 | Dec 24 | Gemini "Stratégies d'évaluation pour l'architecture Nash-SEC" | Unseen vs seen gates | Unseen accuracy lags; a soft structural gate; drop binary thresholds | — | partly: NC1 lacks normalisation; power iteration gives the top eigenvalue, not the trace; misses that the stream has no finite set |
| 40 | Dec 26 | Gemini "Apprentissage Algorithmique et Mathématique Structuré" ("the paper") | Ideal tasks and their order | 13 tasks; 99.9 % gates; addition = parity + chain; carry-weighted loss | — | wrong: its only inline citation is the owner's own PDF; grokking papers listed, never cited |
| 41 | ≥ Dec 30 | Gemini, chat 216cd96a | Review the codebase | "9/10" | — | wrong: missed #132–#135, which were in the code |

**2026**

| Rep | Date | Who, source | Question | Main claims and recommendations | Papers cited | Verdict |
|---|---|---|---|---|---|---|
| 42 | Feb 20–21 | Gemini, chats 834815ac, 60df30f8 | Text decomposed into concepts; LLMs with graphs; Large Concept Models | Least-to-Most; LCM in SONAR space; DLCM +2.69 % at matched FLOPs; VQ concepts added to token loss | LCM 2412.08821 | partly: numbers right; 60df padded (about 10 of 247 sources primary) |
| 43 | Feb 21–23 | Claude Research, Claude.ai chats 13087a35, 21757722, a460a854 | Concept models at scale; fact-check of the owner's proposals | Tokens won through a dense signal; hybrids win; a pause-token control nearly matches COCONUT | LCM; SONAR 2308.11466; LLaDA 2502.09992; H-Net 2507.07955; Ouro 2510.25741 | partly: surveys fair; the fact-check is flattery |
| 56 | Mar 2 | ChatGPT, chat "Largest De-duplicated Dataset" (outside the project) | Open text data at trillion-token scale; streaming; dedup; quality vs quantity; curation | FineWeb about 18.5T, RedPajama-v2 about 30T, about 75.7T in all; StreamingDataset, WebDataset, LSHBloom; repeating a filtered set up to 10 epochs beats one pass over a 10× larger unfiltered set; DCLM; Ask-LLM; LIMO | DCLM; LSHBloom; ProX 2409.17115; Ask-LLM 2402.09668; LIMO 2502.03387 | partly: the known items are right; Ask-LLM's "ICLR 2026" venue doubtful; LIMO's figure depends on the version; three Deep Research files were not exported |
| 44 | May 1 | Perplexity thread on alternatives to chain of thought | Post-CoT reasoning; local fine-tuning | Latent loops need custom training; rStar-Math | Ouro; rStar-Math | unsupported (2026 items unchecked); framing sensible |
| 57 | May 2 | ChatGPT, chat "Emoji-based Reasoning Ideas" (outside the project) | Dense reasoning codes the model invents | 64–256 neutral `<Z###>` tokens as a fixed-length bottleneck; bootstrap from compressed CoT; evolve the codebook under a verifier; six kill tests (equal-length filler, compressed CoT, continuous latent, no transfer, shuffled codes, verifier exploitation); OLMo 2 1B with LoRA and trainable tokens; $0–5 smoke test to $30–200 ablations | Dot by Dot 2404.15758; Coconut 2412.06769; pause tokens 2310.02226; TokenSkip | right as far as checked; the 2026 items in the uploaded report are not checkable |
| 45 | Jul 15 | ChatGPT, chat 6a56e38e (msg 74, 18,499 characters; later turns to msg 92) | "Summarize the project history and state from chats, sources & code."; then latent reasoning and loops | Read the repo at `acd48ff`: the three #132 ACT defects; `shared_loop` runs 24 block applications against the README's "144 effective layers" (#145, #170); duplicate weight tables (#180, #222); no CI (#188); a hypothetical README table (#169). One frozen-curriculum, three-seed Exp0 vs Exp1 study, token- and FLOP-matched (#258). Six kinds of "latent reasoning"; a loop taxonomy; **four grokking conditions** (finite repeated set, memorisation phase, delayed held-out gain, several seeds); L0–L6 ladder | Huginn; Ouro; MoR; ITT 2502.13842; Titans | partly: the code findings, the four conditions and the older papers are right; one false claim ("Later fixed-dataset experiments demonstrated memorization"); 2026 items not checkable. Nothing was filed for three months |
| 45a | Jul 15 | ChatGPT, chat 6a56e38e msg 77 (41,946 characters) | "List all claims, proposed or rejected, and compare with the state of the art." | The ReLSM claim ledger, sections I–XIII. Grokking: infinite fresh data "prevents grokking" is too absolute; wd 1.0 is not transferable; grokking does not imply length extrapolation; one run is not evidence. Compare uniform, static, adaptive and DAG schedules; freeze the DAG for architecture tests. A 0–8 ladder; fixed-set grokking as a separate side study | Huginn (about 800B tokens); Ouro (7.7T); STaR; Mixture-of-Depths; RULER; Mamba-in-Llama; position coupling | partly: the checkable claims are right; it says "58 distinct claims" but numbers 1–66 plus 28 more rows; 2026 items not checkable |
| 46 | Jul 29 | Gemini chat af708951; Claude.ai chat 9116181b | Cheap distillation (another project) | On-policy over off-policy; prototype small; same-stack evaluation | 2305.15717; GKD 2306.13649; MiniLLM 2306.08543; Minitron 2407.14679 | partly: one result backwards; QLoRA LR about 100× low; Claude invented the budget |
| 55 | Aug 1 – Oct 23 | ChatGPT scheduled task "Frontier Architecture Watch", chat 6a6de5a9 (outside the project) | A monthly update of the claim ledger, ending in "the next smallest decisive ReLSM experiment" | ACT and the custom SSM are invalid as evidence; loop count is not a parameter multiplier; recurrence is not latent chain of thought; grokking "is neither required for algorithm learning nor evidence" of a general algorithm. The decisive experiment changed four times: vertical vs temporal recurrence; naive vs stabilised vs gated loops; a middle loop with a feedback-off control; a middle loop at 2 passes, full vs truncated backprop, evaluated at 1–4 loops | about 60 2026 preprints | unsupported: almost all items are 2026 preprints not checked; two non-arXiv hosts; one chronology conflict (a September "reproduction" of a paper dated Oct 6); each month followed the newest preprint; it never names the grokking control (#249) |
| 47 | Oct 23 | ChatGPT Sessions 01–07 (chats 6adbf8f4, 6adbfb87, 6adbfdba, 6adbff8d, 6adc056f, 6adc0c7b, 6adccafc) | Concepts; latent reasoning; diffusion; parallel generation; recurrence red team; cost accounting; structured concepts | Depth and thought stream are separate hypotheses, and the thought summary may leak (later #135); no general superiority of recurrence; 1/2/4 passes at equal budget; ten acceptance criteria. Session 07: five structural commitments and six controls (same evidence as plain text, exact copy or pointers, oracle parsing among them) | ACT 1603.08983; UT; PonderNet 2107.05407; ALBERT 1909.11942; Bansal 2202.05826; Coconut; LLaDA; MDLM; about 25 2026 core readings with arXiv ids in the briefs | partly: older papers and arithmetic right; Session 02 predicted #135; Session 05's first answer overstated Closing the Loop and SMELT before the red-team turn corrected it; 8 sessions were planned (6adbf078), 01–07 ran, 08 became Rep-48; reports not on disk; two of the 2026 ids spot-checked and real |
| 47a | Oct 23 | ChatGPT, chat 6adbf078 | Diffusion LMs for text; Next-Concept Modeling; "did my earlier concept prompts anticipate it?" | A diffusion survey (model families, releases, skeptical items, a 10-paper list); NCM; not ahead of the literature, but the logic-vs-literal split is uncovered; the ReLSM and NEXUS designs "came from assistant-generated proposals"; the 8-session plan | ConceptLM 2602.08984; DLCM 2512.24617; NCM 2610.14162; Hybrid Reasoning Agents 2610.18861 | unsupported: 2026 items not checked; the attribution answer is honest |
| 48 | Oct 24 | ChatGPT, chat 6adce7ea | One integrated report | "Corrects the historical grokking prescriptions" | — | unsupported: report not on disk |
| 49 | Oct 24 | ChatGPT, chat 6adcf1ea | Data-frugal training; frontier pretraining; QAT, pruning | Repeated data needs much stronger regularisation (about 30× weight decay); a 0.1–0.3 % procedural warm-up; multi-token prediction; eight items for ReLSM | Pre-training under infinite compute 2509.14786; Rho-1; Signal and Noise (2025) | partly: numbers right as recalled; says 2.09× here and 2.29× in chat 6adf95f7, never reconciled |
| 50 | Oct 24–25 | ChatGPT, chats 6adcfef5, 6add204a, and the Oct 24 literature register | Capacity beyond 8 GB | Compute sparsity vs storage sparsity; PEER, UltraMem, memory layers; per-depth KV cost in looped models | PKM 1907.05242; Memory Layers 2412.09764; LLM in a Flash 2312.11514 | partly: older papers right; cautions sober |
| 51 | Oct 26 | ChatGPT, chat 6adeb9ac | JEPA for text | How masking teaches; random spans as good as linguistic ones; data2vec; V-JEPA | SpanBERT 1907.10529; data2vec 2202.03555; V-JEPA 2404.08471; I-JEPA 2301.08243 | partly: first backed linguistic masks against its own evidence; conceded to the owner |
| 52 | Oct 26 | ChatGPT, chat 6adf912d | Streaming datasets | Resume is not an identical continuation; budget in tokens; mix a repeated subset | Charton and Kempe 2410.07041 | right |
| 53 | Oct 26 | ChatGPT, chat 6adf95f7 | Small-data LLM; 100–200B tokens; the Dec 23 results; full code review | Grokking "not a prerequisite"; README grokking text = intentions; 500M–1.5B on FineWeb-Edu or Common Pile; 30 findings | FineWeb-Edu; Common Pile; Phi-1.5 2309.05463 | right: the review found real bugs; "not a prerequisite" is a judgment the audit does not share |
| 54 | Oct 27–28 | Claude lexical-grokking session (PR #196) and side-session notes lit-lexical-grokking, lit-embedding-table, lit-puzzles-concepts, research_grokfast, research_adam_resets | Grokking literature 2022–2026; changing the token table | See section 4.2 | about 100 papers | partly: most checked at abstract only; Grokfast paper and code read in full; its own experiments contradict its first recommendation (reset only unused rows) |

### 4.2 Consolidated grokking literature map

Merged from res-codex-lit's map and the papers in the other reports. "Checked": **F** full text, **A** abstract, **C** code or GitHub issue read, **S** snippet or summary, **U** from memory, **AI** cited only by an AI chat. Ids starting 2601 to 2610 are 2026 preprints that rest on the source agents' checks.

**Classic grokking: mechanism and progress measures**

| Paper | Year | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|---|
| Power et al., Grokking | 2022 | 2201.02177 | Finite-set grokking on algorithmic tables; decay on all weights | U | The finite set is missing; embedding exemption and β2 0.95 depart from the recipe (#249, #251) |
| Nanda et al., Progress measures | 2023 | 2301.05217 | Fourier circuit; memorise → circuit → clean-up; restricted and excluded loss | A | The positive control must reproduce it first (#249) |
| Liu et al., Towards Understanding Grokking | 2022 | 2205.10343 | Structured embeddings; phase diagram; four phases | A | The table carries the structure |
| Liu, Michaud, Tegmark, Omnigrok | 2022 | 2210.01117 | Weight-norm mechanism; IMDb groks weakly | F | Norm signal is unreliable here: the norm is unused-row drift (#218) |
| Varma et al., Circuit efficiency | 2023 | 2309.02390 | Memorising and generalising circuits compete; critical data size; ungrokking | A | Size the fixed set near the critical size |
| Gromov, Grokking modular arithmetic | 2023 | 2301.02679 | An MLP groks under plain gradient descent | A | — |
| Zhong et al., Clock and Pizza | 2023 | 2306.17844 | Different algorithms by width | A | Seeds may pick different circuits |
| Morwani et al. | 2023 | 2311.07568 | Margin maximisation predicts Fourier features | A | — |
| Kumar et al., lazy to rich | 2023 | 2310.06110 | Grokking is a lazy-to-rich transition | A | Init scale is a knob |
| Rubin et al., first-order phase transition | 2023 | 2310.03789 | Grokking as a phase transition in two-layer nets | AI | — |
| Thilak et al., Slingshot | 2022 | 2206.04817 | Grokking at Adam's cyclic norm spikes | U | Weak support for "grad-norm spikes come first" |
| Miller et al. | 2023 | 2310.17247 | Grokking beyond neural nets; needs a complexity penalty | A | — |
| Fan, Pascanu, Jaggi, Deep Grokking | 2024 | 2405.19454 | Deep MLPs grok more easily; feature rank tracks it | A | Log feature rank |
| Carvalho et al., Grokking Explained | 2025 | 2502.01774 | Distribution shift is the key factor | AI | — |
| Clauw et al., Emergent phase transitions | 2024 | — | Synergy as an information-theoretic precursor | AI | The "weight entropy" key is not this (#245) |
| Hidajat et al., Structural inference | 2026 | 2605.15787 | Early MLP memorisation starves attention; a KL attention intervention skips the wait | A | The delay may be attention structure, not only norm |
| Song and Ye | 2026 | 2605.09724 | Bigger models can avoid grokking | A | Nano (41.6M) may memorise too fast to show a delay |
| Sweet spot (anonymous) | 2026 | 2605.14659 | Fastest generalisation at an intermediate data size | A/S | Sweep the fixed-set size |
| Kazanskii, GeomDR | 2026 | 2607.11666 | Dimensionality collapse precedes grokking | A | Log effective rank |
| Is grokking a computational glass relaxation? | 2025 | 2505.11411 | Glass-relaxation view | AI | — |
| Golechha, Progress measures for grokking on real-world tasks | 2024 | 2405.12755 | Activation sparsity, absolute-weight entropy and approximate local circuit complexity track grokking | U (matched to Rep-25a by content; the chat gives only citation markers) | The intended meaning of the PR #73 keys: hidden-unit sparsity (#238), absolute-weight entropy (#245) |

**Weight decay, norms, numerics, fragility**

| Paper | Year | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|---|
| Prieto, Barsbey, Mediano, Birdal, Grokking at the Edge of Numerical Stability | 2025 | 2501.04697 | Softmax collapse stops learning; StableMax and ⊥Grad grok without decay; too large an Adam eps halts learning | A | bf16 training vs fp16 eval and a 32k softmax: run #255. The right StableMax is s(x) = x + 1 for x ≥ 0, 1/(1 − x) below |
| Truong et al., Weight norm sets the timescale | 2026 | 2606.13753 | A final LayerNorm removes the norm dependence | A | ReLSM ends in a final RMSNorm; test with and without it (optional arm of #249) |
| Manir and Rupa | 2026 | 2603.25009 | Only a narrow decay band groks | A | Sweep decay (#251) |
| Verma, WD regimes | 2026 | 2605.20441 | Memorise, grok, collapse regimes; λc ≈ 0.016 in its parametrisation | A | Decay values do not transfer across parametrisations |
| Ootani, Conditional and fragile | 2026 | 2607.05104 | CPU thread count flips same-seed outcomes; single runs mislead | A | ≥ 5 seeds; report grok rates (#187, #189) |
| Ootani, Small Enough to Know Everything | 2026 | 2609.20166 | The decay response steepens with scale | A | A micro-harness decay value may not carry to nano |
| Prakash and Martin; extension | 2025, 2026 | 2506.04434; 2602.02859 | "Anti-grokking" after very long training | A | Keep evaluating; do not stop at the peak |
| Park et al., Competition dynamics | 2024 | 2412.01003 | Generalisation can be transient | A | Same |

**Grokfast and optimizers**

| Paper | Year | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|---|
| Lee et al., Grokfast | 2024 | 2405.20233 | Low-pass gradient filter, ">50×" against Adam with coupled L2 wd 0.005; **EMA setting α 0.98, λ 2** (λ 5 is the moving-average default) | C, F | `train.py` uses λ 5, a zero EMA start, clipping after the filter, filtering inside accumulation (#137, #254) |
| NeuralGrok (Zhou et al.) | 2025 | 2504.17243 | Its own table: a tuned baseline groks x·y mod 97 in 1,280 steps vs 39,890 | F | Grokfast's gain may vanish against a tuned AdamW |
| Saheb Pasand and Dohmatob, EGD | 2025 | 2510.04930 | Reruns Grokfast on the same weak baseline | F | A reproduction, not a replication |
| Wang, Muon active ingredient | 2026 | 2607.20512 | "Faster" claims can flip under a hold-the-threshold metric | A | Judge by steps to hold 95 % |
| Tveit et al., Muon accelerates grokking | 2025 | 2504.16041 | Mean grokking epoch 153 (AdamW) vs 103 (Muon) | A | — |
| AdEMAMix | 2024 | 2409.03137 | Fast plus slow EMA in Adam | A | Closest principled relative of Grokfast + Adam |

**Grokking and the embedding layer**

| Paper | Year | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|---|
| AlQuabeh et al. | 2025 | 2505.15624 | Embeddings cause the delay; rare tokens stall; embedding-specific LR | A | An embedding-LR flag (#246) |
| Xu, Ni, Wang, Hu, GrokTransfer ("Let Me Grok for You") | 2025 | 2504.13292 | A weak model's embedding removes the delay | A | #250 (whose body has the wrong authors) |
| Furuta et al. | 2024 | 2402.16726 | Retraining only the tables of a grokked model failed for subtraction | F | Relearning a table under a frozen body is not guaranteed |
| Zhou et al., LLMs use Fourier features to add | 2024 | 2406.03445 | Number-token rows carry the Fourier basis | A | ReLSM's digit tokens put the circuit in the body instead |
| Jia | 2026 | 2609.18078 | Freezing transferred tables cuts relapse 19.40 % → 0.07 % | A | Freeze used rows |

**Grokking beyond arithmetic: language, grammar, structure**

| Paper | Year | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|---|
| Murty et al., Grokking of Hierarchical Structure | 2023 | 2305.18741 | Dyck and tense inflection: OOD rises long after ID saturates; depth is an inverted U; weight norm does not predict it | F | Never early-stop; norm telemetry is weak |
| Zhu et al., Critical Data Size | 2024 | 2401.10463 | IMDb and Yelp grok near a critical size, smoother on language; wd 1e-2 | F | Expect a smooth jump on language |
| Muckatira et al. | 2026 | 2606.00230 | Delayed generalisation on five BLiMP phenomena | A | Closest to "grokking grammar" |
| Chen et al., Sudden drops in the loss | 2023 | 2309.07311 | Syntax heads appear with a loss drop | A | An attention-level progress measure |
| Lv et al., LMs grok to copy | 2024 | 2409.09281 | Induction heads form in a grokking-like way | A | Copy and reverse may show this |
| Li, Fan, Zhou, Grokking in LLM pretraining | 2025 | 2506.21551 | Data groups grok at different times | A | Per-task timing differs |
| Sabatelli | 2026 | 2609.26018 | Rule plus exceptions; over-regularisation | A | Irregular stems in a lexical test |
| Ahuja et al. | 2024 | 2404.16367 | Full next-token loss gives hierarchical generalisation; prefix-LM often does not | A | ReLSM trains answer-only; try a full-loss arm |
| Schulz et al. | 2025 | 2510.02524 | Sub-grammars are learned in parallel | A | Argues against strict simplest-first |
| Not found by any sweep | — | — | Grokking on inflection data, SCAN or COGS; tokenizer or vocabulary-size effects | — | Open niche for lexical grokking |

**Composition, recurrence and puzzles (ReLSM's own hypothesis)**

| Paper | Year | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|---|
| Wang (Boshi) et al., Grokked transformers are implicit reasoners | 2024 | 2405.15071 | Composition appears only through grokking; speed set by the inferred/atomic ratio; parameter sharing helps OOD | F | Direct support for `shared_loop`, after a grokking control (#258) |
| Kohli et al., Loop, Think & Generalize | 2026 | 2604.07822 | Recurrent-depth models pass OOD composition via a three-stage grokking; wd 0.01; halt when KL < 0.01 and entropy < 3; zero-init output projections | F | The most relevant paper for the loop variant; a halting rule for #132 |
| Abramov et al., Grokking in the wild | 2025 | 2504.20752 | Raising the inferred/atomic ratio induces grokking | A | — |
| He et al., Is grokking worthwhile? | 2026 | 2601.09049 | Accuracy can decouple from circuit formation | A | Every claim needs a causal probe |
| HRM (Wang, Guan et al.) | 2025 | 2506.21734 | 27M parameters, about 1k examples, deep supervision, Q-learning halting | A/F | Closest public analogue to `latent` and `act` |
| ARC Prize HRM analysis | 2025 | blog | The outer loop and augmentation drive results | A | No per-example ID embeddings |
| TRM (Jolicoeur-Martineau) | 2025 | 2510.04871 | 2-layer recursed net; wd 1.0; weight EMA; StableMax; embedding LR 1e-2 vs 1e-4 | A | A grokking recipe in all but name |
| URM (Sapunov et al.) | 2025–26 | 2512.14693, 2604.21999 | ACT needs memory tokens and a halting bias of −3 | A/S | A bias init for #132 |
| Looped transformers recognise context-free languages | 2026 | 2601.01754 | O(log n) loops suffice | S | Theory for the loop variant |
| Saunshi et al., Looped transformers | 2025 | 2502.17416 | A k-layer net looped L times nearly matches a kL-layer net on reasoning | U | — |
| Geiping et al., Huginn | 2025 | 2502.05171 | 3.5B recurrent depth on 800B tokens | U | Loops still need large pretraining |
| Ouro | 2025 | 2510.25741 | 1.4B/2.6B looped models on 7.7T tokens | U | Same |
| Bansal et al., End-to-end algorithm synthesis | 2022 | 2202.05826 | Input recall and progressive training for recurrent nets | U | Loop-stability options |

**Curricula and data regime**

| Paper | Year | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|---|
| Charton and Kempe, Emergent properties with repeated examples | 2024 | 2410.07041 (from readers' memory) | Repeated small sets beat unlimited data; a repeated subset plus fresh data | AI/U | The DAG stream never repeated a set (#257) |
| Cornacchia and Mossel, curriculum for parity | 2023 | 2301.13833 | A curriculum helps learn parity | AI/U | Directly about a ReLSM task; unused |
| Wilson et al., The Eighty Five Percent Rule | 2019 | — | 85 % accuracy maximises learning (binary tasks) | U | Basis of the 0.85 band, stretched |
| Platanios et al., competence-based curriculum | 2019 | 1903.09848 | Sqrt pacing | U | — |
| Muennighoff et al., data-constrained scaling | 2023 | 2305.16264 | Up to 4 epochs ≈ unique data | U | — |
| Pre-training under infinite compute | 2025 | 2509.14786 | Repetition needs much stronger regularisation | U | #251 range |
| Nash-MTL (Navon et al.); FAMO; FairGrad | 2022, 2023, 2024 | 2202.01017; 2306.03792; 2402.15638 | Multi-task gradient balancing | U | Nash-MTL costs one backward per task; FAMO is cheap |

**Changing the token table (from the lexical-grokking session; condensed)**

| Papers | arXiv | Claim | Checked | Implication for ReLSM |
|---|---|---|---|---|
| Chen et al., Active forgetting | 2307.01163 | Reset embeddings and their optimizer state periodically | F | Belongs in a language phase, not during maths |
| Land and Bartolo, Magikarp | 2405.05417 | In tied models unused tokens drift together | mixed (U to A) | The DAG-v12 finding (31,879 rows, cosine 0.915) |
| Lazić et al. | 2604.21632 | Unseen unembeddings collapse to one vector, worse when tied | A/S | Confirms #218 |
| Han et al., A Deafening Silence | 2610.09835 | Forgetting sits in absent-token output rows; a larger eps helps | A/S | Lost to scratch in the session's own test |
| Stollenwerk, Coupled Adam | 2502.08441 | Shared second moment across the vocabulary | F | Arm written, not run (#246) |
| WECHSEL, FOCUS, OFA, ZeTT and others | 2112.06598, 2305.14481, 2311.08849, 2405.07883 | Transplants anchor on shared tokens | A | Keep the digit rows |
| TokenAdapt | 2505.09738 | — | A; a mirror says withdrawn | Do not cite |

### 4.3 Citation and content errors found

| # | Where | What it said | What is right | Spread to |
|---|---|---|---|---|
| 1 | ChatGPT chat 693b54b7 (Rep-26) | GrokTransfer is "Chen & Wang (2025)" | Xu, Ni, Wang, Hu, ICLR 2025, 2504.13292 | avenue A10 and the **#250 body** |
| 2 | ChatGPT chat 6942ce5e (Rep-18) | Softmax Collapse by "Dureau et al." | Prieto, Barsbey, Mediano, Birdal | — |
| 3 | same | "Zhang et al., Grokking in LLM Pretraining?" | Li, Fan, Zhou (2506.21551) | — |
| 4 | same | "Thilak et al. 2023, role of weight decay" | Decoder-only decay result is Liu et al. 2022; Thilak 2022 is Slingshot | — |
| 5 | same | Liu et al. effective theory "2023" | NeurIPS 2022 | — |
| 6 | ChatGPT chat 693745a8 (Rep-04) | RMT "Ding et al., 2022"; Mamba "Pankaj et al."; BLT "Peng et al., 2023"; Mercury "late 2025" | Bulatov et al.; Gu and Dao; Pagnoni et al., Dec 2024; Feb 2025 | — |
| 7 | ChatGPT chat 693a2b2b (Rep-13) | GPT-3 peak LR 2.8e-5 | 0.6e-4 | — |
| 8 | Claude chat dc94954c (Rep-19) | StableMax code: `exp(x − max) / sum` | That is ordinary softmax | #255 should use the right formula |
| 9 | Claude chat dc94954c; Codex task PR045; Rep-27 | Grokfast "50×", λ 5 | The EMA setting is λ 2; 50× is against a weak Adam baseline | `train.py` default λ 5 |
| 10 | Claude chat dc94954c | "weight_decay / learning_rate … 10,000, which is extreme" | Wrong way round for AdamW: per-step shrink is lr × wd | — |
| 11 | Claude chat dc94954c | A grokked model goes from 8-bit to 64-bit parity | Unsupported: grokking is in-distribution | the project's goal |
| 12 | Gemini chat 46f11897 (Rep-20) | Murty: "shallow is better" | Inverted U in depth | — |
| 13 | same | Arnett and Bergen cited for the opposite of its title; another source cited for U-shaped learning | Mis-cited | — |
| 14 | ChatGPT review "Validation méthodologique" (Rep-21) | A "Poincaré" source; a "Moses bias" | Neither exists | — |
| 15 | Gemini chat ba6cfceb (Rep-22) | Llemma beats Llama on digit splitting | Llemma reuses the Llama 2 vocabulary (32,016), which already splits digits | owner's Llemma choice (#225) |
| 16 | same | Its digit regex splits digits; reversal paragraph | The regex does not; the source is about chunk direction | — |
| 17 | Gemini chat 4ddca337 (Rep-14) | Royal Road text is in The Pile | It is not | — |
| 18 | Gemini "Pondéré" (Rep-30) | FairGrad = Nash-MTL + PCGrad; DoReMi is online | Both wrong; Cornacchia and Mossel mis-cited | — |
| 19 | Gemini "Équité" (Rep-31) | Rawlsian disagreement point; neural-collapse formula | Invented; wrong | — |
| 20 | Perplexity Nash-SEC (Rep-32) | ∏ αᵢ‖gᵢ‖ on the simplex for Nash weights | Always returns uniform weights | — |
| 21 | same | Long Range Arena has copy, reverse, parity, dyck; Manning 2020 in Nature | Wrong; PNAS | — |
| 22 | same | "30–50 % less data", "+13.6 %", "+39.6 %" | Invented | — |
| 23 | same | wd 1e-4; ">1e-3 eliminates grokking" | Contradicts Power, Nanda, Omnigrok, TRM (wd 1.0) | never flagged until the audit (E57) |
| 24 | same | Parity is the sum bit of addition | True for binary sums; ReLSM adds decimals | the DAG edge parity → addition (#157) |
| 25 | Gemini "Audit-MTL" (Rep-37) | Reward a rising weight norm | Under decay, grokking comes with a falling norm | — |
| 26 | Gemini "Éval" (Rep-39) | NC1; power iteration for the Hessian trace | NC1 needs between-class normalisation; power iteration gives the top eigenvalue | — |
| 27 | Claude Research compass guide (Rep-27) | EMA 0.98 ≈ 100 samples; "difficulty_step = 8 aligns with FP16" | About 50; a non sequitur | — |
| 28 | Gemini chat dd319d3d (Rep-36) | "Unique global minimiser" by Mixon | Misattributed | — |
| 29 | Gemini chat 5905ea6a (Rep-05) | Deep supervision "unlike HRM" | HRM uses it | — |
| 30 | Gemini chat 6db3c20f (Rep-03) | "Geometric mean" code | It computes a harmonic mean | the owner's README of that project |
| 31 | ChatGPT chat 693a2b2b, Dec 11 (Rep-17; inherited by 6941b170) | 4060 trains at 30–40 tok/s | Logs show 5,000–7,600 | — |
| 32 | ChatGPT chat 694b2b13 (Rep-33) | Per-task parser table | Wrong in 7 of 10 rows | — |
| 33 | Gemini chat af708951 (Rep-46) | DeepSeek-R1 shows distillation insufficient; QLoRA LR 1e-6 | Backwards; about 100× too low | — |
| 34 | Claude chat f50fd8de (Rep-07, 08) | MMLU 52–58 % at 1.5B on 2B tokens; "2B → 200B equivalent" | No derivation; gains do not multiply | — |
| 35 | Gemini chat 60df30f8 (Rep-42) | Shortest path is NP-hard | It is polynomial | — |
| 36 | Claude.ai chat a460a854 (Rep-43) | Owner's proposals "largely ahead of the curve" | Rests on its own earlier reports | — |
| 37 | Chats 6adcf1ea vs 6adf95f7 (Rep-49, 53) | 2.09× vs 2.29× for the same paper | Never reconciled | — |
| 38 | Lexical-grokking note (PR #196) | Manir and Rupa "2026-05"; Muckatira "May" | Ids say March and June | — |
| 39 | Lexical-grokking notes | Magikarp check level: "link only", "abstract", "peer reviewed" | Inconsistent | — |
| 40 | Codex task PR045 paste | "wd 0.1 prevents grokking"; grokking "After ~10K–50K steps" | Wang 2024 groks at 0.1, Kohli 2026 and Zhu 2024 at 0.01; the forecast is invented | — |
| 41 | ChatGPT chat 6945b243 msg 7 (Rep-23a) | "No fatal coding bugs were found in the baseline path"; nano "about 18M parameters" | The double label shift was in that code (fixed the next day, PR #80); nano is 41.6M | — |
| 42 | ChatGPT chat 6945b243 (Rep-23) | Peak LR "around 8e-5 near 20k"; v2's rising LR means no decay schedule was used | It was 1e-4; v2 was still in warmup, which really ended near micro-step 56,000 (#248) | — |
| 43 | ChatGPT chat 693a2b2b (Rep-17) | Nano "~18M parameters"; a 300M model on 200M tokens in "~3.5 sec on B200"; every rung at 4,096 context | 41.6M; off by orders of magnitude; the runs used 32 to 128 | — |
| 44 | ChatGPT chat 693b54b7 msg 242 (Rep-12a) | `cycle(iter(loader))` caches the first pass and replays it | Already fixed about 4 h earlier (commit 30e0069); it read stale uploaded files | — |
| 45 | ChatGPT chats 693b54b7, 69484386 (Rep-12a), four reviews | "No correctness issues" for addition, multiplication and compare | One digit count is drawn for both operands; that starts in this diff | #210 |
| 46 | ChatGPT chat 6946d7ab, Codex spec (Rep-25a) | Near-zero entries (below 1e-4) "in the MLP output of 3 blocks" | Its own report meant inactive hidden units; the FF output is dense by construction | #238 |
| 47 | ChatGPT chat 69484386 msg 485 (Rep-29a) | EMA 0.9, low gate 0.6, clamp floor 0.05, `min_weight` 0.2, cooldown 1000, equal base weights including "sort" | Repo: 0.98, 0.55, 0.0, 0.05, 500, task-specific weights, no sort task | #201: difficulty hit 0 the same day |
| 48 | ChatGPT chat 6949cb7d msg 8 (Rep-33a) | The mix phase introduces "language modeling data at 50M tokens" | The "language" batches were the algorithmic loader again (#158) | — |
| 49 | ChatGPT chat 6949cb7d msg 17 (Rep-33a) | Copy and reverse never emit EOS because of a missing marker and "nondeterminism" | Examples were cut at `alg_seq_len` 32 before their EOS (#139) | — |
| 50 | ChatGPT chat 694c894c 01:19 (Rep-35b) | The DAG edges, thresholds, patience, margins, floors and replay values | No source; invented in that answer | PR #108, #157 |
| 51 | ChatGPT chat 694c894c 00:51 (Rep-35b) | Copy and reverse at 0.0 are "a red flag for all-at-once mixing" | Truncation at 32 tokens (#139) | The DAG's task order |
| 52 | ChatGPT chat 694c894c 02:24 and 16:40 (Rep-35b) | `minimum − margin` is "backwards (too lenient)", use `+`; then `+` is a bug | One chat contradicting itself within 14 h | The −, +, − flip (section 5) |
| 53 | ChatGPT chat 69378f5b (Rep-09 reply) | "This is actually a very strong consolidation"; its own code with a batch-wide halt | v1's thought stream cannot change the token; the code has #132's defects and was folded into the ReLSM PDF | #132 |
| 54 | ChatGPT chat 693745a8, Deep Research brief (Rep-04) | HRM = "Hierarchical Recurrent Memory"; TRM = "Transformer with Recursive Memory" | Hierarchical Reasoning Model; Tiny Recursive Model (the report itself uses the right names) | — |
| 55 | ChatGPT chat 6a56e38e msg 74 (Rep-45) | "Later fixed-dataset experiments demonstrated memorization" | No record: the fixed-data runs crashed or were stopped (#136, G6) | — |
| 56 | ChatGPT chat 6a56e38e msg 77 (Rep-45a) | "58 distinct claims" | Numbered 1–66, plus 28 more rows | — |
| 57 | ChatGPT outside chats "Axolotl config" and "OOM fix" (Rep-01a) | Config keys `include_tkps`, `eval_pad_to_sequence_len` and others; long samples "truncated, not dropped"; "You're not losing training" | The keys do not exist; Axolotl logged "Dropped 92 samples"; the checkpoint was lost to a full disk | — |
| 58 | ChatGPT chat 6a6de5a9 (Rep-55) | A September "reproduction" of the Recurrent Looped Transformer; then the paper dated Oct 6 | One of the two is wrong | #272 |
| 59 | ChatGPT chat 6adc056f, Session 05 first answer (Rep-47) | Closing the Loop has a "parameter-matched dense counterpart" | That control had about 4× less training compute (corrected in the same session's red-team turn) | — |

Charton and Kempe (2410.07041) is cited in four reports but its id comes only from the readers' memory; no source in the corpus gives it.

---

## 5. Contradictions and errors in the advice

| Topic | What the AIs said | What the owner did | Where it stands |
|---|---|---|---|
| **Weight decay** | Dec 17 ChatGPT (Rep-18): grokking needs "moderate regularization". Dec 17 Claude (Rep-19): wd 1.0 is "the empirical standard". Dec 17 Codex paste: "wd 0.1 prevents grokking". Dec 19 ChatGPT Deep Research (Rep-23a): wd 1.0 is "the single biggest oversight", use 0.1–0.01. Dec 19 ChatGPT (Rep-23): wd 0 in the algorithmic phase. Dec 21 ChatGPT (Rep-29c): put wd 1e-3 to 1e-2 back "because weight_norm is running away". Dec 19 Claude chat 23d47afd: wd 1.0 is "extremely high", the root cause. Dec 19 Claude chat 090b121f: raise to "1.0 or even higher". Dec 23 Perplexity: wd 1e-4, ">1e-3 eliminates grokking". Dec 9 Claude: "0.1 (important for looped architectures)". Oct 24, 2026 ChatGPT: repeated data needs about 30× the usual decay. | wd 1.0 (Dec 17–18, fixed data, NaNs), 0.1 (Dec 19), 0.0 (Dec 19–24: v4, v10, v12, the 43k run, v15, v16, run4, run5), 0.1 (every DAG run, Jan 11). The 32k embedding and head stay exempt throughout. | Literature: Power, Nanda, TRM use 1.0; Wang 2024 0.1; Kohli 2026 and Zhu 2024 0.01. Values do not transfer across optimizer parametrisations. #251 sweeps 0 to 2.0 on all weights. |
| **Gradient clipping** | Dec 19 ChatGPT Deep Research (Rep-23a): raise the clip "to 5 or 10". Same chat, next report (Rep-23): lower it further, "e.g. 0.1". | 0.1 on Dec 19 (v4), then 5.0 from Dec 21 (the 43k run, v10, v15, v16, run4, run5). | Never compared. The recipes in #249 use no clipping (#267). |
| **Model size** | Dec 19, the owner: "probably because it is too large". Rep-23a the same day: nano is "about 18M" and size is "likely not the bottleneck". Rep-23 later in the same chat: nano "might simply be at the edge", try a larger model. | No change. | Nano is 41.6M (25.2M without the table). A d = 128, 2-layer model is #249's setting. |
| **"The code is fine"** | Dec 16 ChatGPT (Rep-16): "no obvious runtime bugs". Dec 19 ChatGPT (Rep-23a): "No fatal coding bugs". Dec 21 ChatGPT (Rep-29a): "nothing is obviously broken", quoting code not in the repo. Dec 30 Gemini (Rep-41): "9/10". | Trusted each review. | #133–#135 and the double shift were present; the Dec 21 review read stale or invented code. Any AI review must name the commit it read (#189). |
| **Which evaluator to believe** | Dec 22 in-training IID: v15 27.83 % > v16 20.00 %. Dec 23 `eval_hub` on the same checkpoints: v15 8.6 % < v16 13.8 % (compare 65.0/55.0 % vs 0.5/6.5 %). | Got both within a day; the comparison was left there. | The two evaluators reverse the conclusion (#160, #227). No Dec 22–23 curriculum conclusion holds. |
| **Finite vs infinite data** | Dec 10 Codex paste: fresh data avoids overfitting "rather than grokking". Dec 17, three reports: infinite data prevents grokking. Dec 19 ChatGPT: fixed data for the first 5–10M tokens. | Built `--fixed_data` (Dec 17), then tied the curriculum to the infinite stream (Dec 21). | The finite regime still cannot be run (#214, #136). |
| **What grokking is for** | Project goal: produce grokking. Curriculum research (Rep-26): keep the train–validation gap "minimal", i.e. avoid delayed generalisation. Claude (Dec 17): grokking gives longer-input generalisation. | Followed the curriculum research; kept the length tie. | Nobody noticed the conflict until the audit. The length tie is unsupported. |
| **Staging the tasks** | Dec 11 ChatGPT: a static mix often beats a stepwise curriculum. Dec 23 Perplexity turn 3: phasing is « absolument impérative » ("absolutely essential"), then « Ne faites pas de coupure brutale » ("don't cut off abruptly") and the dependent task « doit être forcée à zéro » ("must be forced to zero") in one answer. Dec 24 "Étagé": easy-to-hard can block grokking, but hard gates are safer for arithmetic. Dec 24 "Éval": « Abandonnez les seuils binaires rigides » ("drop rigid binary thresholds"). | Binary gates with a 3-eval ramp and a 2 % floor. | DAG-v12 paused from eval 13 to the end, as "Éval" warned (« blocage total », "total blocking"). No static-mix control exists (#252). |
| **Jitter** | Compass (Rep-27): two-sided Gaussian around the frontier. Ramping (Rep-26) and ChatGPT's V1 prompt: one-sided. | One-sided `uniform(0, d)`. | #220 says the code misread Compass; in fact it followed ChatGPT's advice. |
| **Unlock margin sign** | ChatGPT chat 694c894c designed `+ 0.01` at 01:19 UTC on Dec 25 ("you must exceed threshold by +0.01"). Its review of Codex's first diff at 02:24 said `minimum − margin` is "backwards (too lenient)", use `+`. At 16:40 the same chat called `+` a bug. One ChatGPT chat contradicted itself; it was not Codex against ChatGPT. | Applied both within 14 hours (−, +, −). | Lenient `−` on main; the flag's name reads as strictness. |
| **StableMax** | ChatGPT (Rep-18): StableMax and ⊥Grad, correct id. Claude (Rep-19): code that is plain softmax. | Asked "StableMax vs. weight decay"; built neither StableMax nor ⊥Grad. | #255. |
| **Grokfast λ** | Compass and Codex paste: λ 5, "50×". ChatGPT V2 (Dec 20): λ 2–3 with a paired same-seed ablation. Paper's EMA setting: λ 2. | Default λ 5; no paired run. | #254 uses λ 2 with 5 as sensitivity. |
| **Language phase share** | Dec 11 ChatGPT: algorithmic ≈ 15 % of tokens; a mixing band of "e.g. 5% of alg tokens", or 0 for a hard switch (the default became 50 %). Owner's plan PDF: 95 % → 20 % algorithmic. Dec 17 Claude: "100M tokens on algorithmic data only". Oct 24, 2026 ChatGPT: a 0.1–0.3 % warm-up. | Followed Claude: 100 % algorithmic from Dec 17. | README still says "Phase 1: Algorithmic Grokking" (#169). |
| **Tokenizer** | Dec 17 Gemini: custom 16k; "do not use the GPT-2 or Llama tokenizer". Dec 19 Claude: digit or character level. | Llemma 32k the next day, no reason given. | #225. |
| **"IID" evaluation** | Dec 20 Claude: IID = held-out examples from the training distribution. | Code built a grid at the final phase's ranges. | #162. |
| **Weight norm as a grokking signal** | Omnigrok and Nanda: yes. Murty 2023: no. Truong 2026: a final LayerNorm removes it. "Audit-MTL": reward a rising norm. | Logged norm and "WD pressure" as the main signal. | The norm is unused-row drift (#218). |
| **Controller band vs gates** | Compass: 85/55 band. Nash-SEC: 99 % gates. ChatGPT Dec 24: lower the band; Dec 25: lower the gates (copy 0.95, reverse 0.93). | Kept 85/55 and 0.95–0.99 gates; his own 70/50 PR was never merged. | A prerequisite holds 0.97 for four evals only once it saturates (#157). |
| **Parity in the DAG** | Perplexity turn 5: force parity in with a guaranteed minimum weight. | Parity locked behind copy and reverse at 0.99/0.98. | Parity was the only task that ever unlocked. |
| **"Chain"** | Codebase: chained arithmetic. Perplexity: pointer-following. "The paper": carry propagation. | PR #114 rewrote the task to match "the paper". | The PR #108 edges referred to a task ReLSM did not have. |
| **Same assistant, two days apart** | ChatGPT Dec 17 (moderate decay) vs Dec 19 (no decay); Claude chat 23d47afd vs 090b121f (lower vs raise); ChatGPT chat 693a2b2b on Dec 11 (use AO3 at 16:15 UTC, avoid it at 18:56). | Followed the latest answer each time. | Each chat saw only its own run. |
| **Large Adam eps** | Han 2026: a larger output eps reduces forgetting. Prieto 2025: too large an eps halts learning. | — | The lexical session's own test sided with Prieto. |
| **Multi-token prediction** | Jul 15, 2026 ChatGPT: keep it out of the first loop experiment. Oct 24 and 26: rank it high for data efficiency. | — | Both can hold; no file reconciles them. |
| **JEPA masks** | Oct 26 ChatGPT: cites evidence for random masks, then recommends linguistic masks. | Objected: "The whole premise is moot, isn't it?" | ChatGPT conceded; random masks by default. |

---

## 6. Ideas with no GitHub issue

Every idea in sections 2 to 4 that no issue from #132 to #259 owns. 58 rows. "Action" points to the drafts file: **I1 to I7** are proposed new issues; **comment #N** is a proposed comment; **record** means keep it here only (out of scope, trivia, or another project).

| # | Idea | Source | Why it matters (one line) | Action |
|---|---|---|---|---|
| 1 | Language grokking (French conjugation, famine, Wug test) | G3; Rep-20; Gemini chat 46f11897 | The owner's own grokking question; the literature has no inflection grokking paper | I1 |
| 2 | Lexical "paradigm" grokking on a synthetic inflection language | G14; PR #196 | Same idea, now with a design and probes; owner: none in #259 (A11) | I1 |
| 3 | "Pidgin first, then full grammar" staged language curriculum | Gemini chat 46f11897 (reasoning trace) | A curriculum arm for I1 | I1 |
| 4 | Creolisation by iterated learning (experiment B of Rep-20) | Rep-20 | Exploratory; optional arm | I1 |
| 5 | Inferred/atomic ratio as the knob for composition (Wang 2024) | §4.2; PR #196 E3 | Cheap knob for a composition test | I1 |
| 6 | Mix algorithmic and language tasks so they build on each other | L5; Perplexity turn 11 | The owner's transfer idea; never tested | I2 |
| 7 | Maths as a third domain | L6; Perplexity turns 12–14 | Accepted in chat, never committed | I2 |
| 8 | Short procedural warm-up (0.1–0.3 %) before language | Z7; Rep-49 | Replaces the "long grokking phase" premise with a testable claim | I2 |
| 9 | Topic-based mixing | D5; Rep-13 | Cheap arm for a language run | I2 |
| 10 | Licence-clean corpora (fiction, FineWeb-Edu, Wikipedia) | D6, D11; Rep-14, Rep-53 | Needed for any language run | I2 |
| 11 | Creative-writing data (Gutenberg, teacher-made) | D4; Rep-17 | Later; data choice for I2 | I2 (later) |
| 12 | Reversed-digit (least-significant first) answers | Rep-22; avenue B4 | Carry flows in reading order; avenue B4 has no owner | I3 |
| 13 | Loop-step (iteration) embedding, LayerScale or identity-biased residual | Rep-05; Rep-45 (L2 loop) | Standard loop stabilisers; none built | I4 |
| 14 | Orthogonal thought-token init; sandwich norm; consistency loss | Rep-12; ChatGPT chats 693a2b2b, 6941a570 | Avenue C9, owner none | I4 |
| 15 | Self-distillation: the last loop teaches earlier loops | Rep-02; Claude chat 3ee613da | C9, owner none | I4 |
| 16 | MLP core control (TRM 87.4 % vs 74.7 % with attention) | Rep-04 | A cheap control for the loop core | I4 |
| 17 | Loop-count sampling in training | Rep-07; Rep-45 | Train at varied depth so eval-time depth can vary | I4 |
| 18 | Linear and causal probes of the thought state vs baseline hidden states | Rep-45; K6 (Rep-47 Session 07); He 2026 | Tests whether loops build usable state; needs #135 fixed | I5 |
| 19 | "Linéarité Locale": PCA ordering of number embeddings | Perplexity turn 3 | An offline geometry probe | I5 |
| 20 | Test `ssm_mem` memory on exact recall vs iterative state | Rep-45 | Separates two uses of memory | I5 |
| 21 | Off-the-shelf pretrained model through the same eval harness | E2; ChatGPT chat 69386da6 | Owner's own idea, dropped Dec 12; gives a reference point | I6 |
| 22 | Small-model leaderboards as a baseline | Perplexity, Jun 27, 2026 | Same need, another project | I6 |
| 23 | Concept-level units: pooled vs quantised span branch, then learned boundaries, then cross with recurrence | K5; Rep-47 Session 01 | The owner's most frequent 2026 idea | I7 |
| 24 | VQ "concept" codebook on thought states, added to the token loss | Rep-42 (ConceptLM) | Avenue C11, recorded only | I7 |
| 25 | Variable-length concept pooling instead of fixed memory queries (DLCM) | K2; Rep-42 | Closest published shape to ReLSM's compressor | I7 |
| 26 | Concepts as discrete tokens; literals as typed pointers | K3 | Owner's Feb 2026 proposal | I7 |
| 27 | Text JEPA vs token model at matched data and compute | K7; Rep-51 | A matched test; low priority | I7 (later arm) |
| 28 | Nash-MTL / FAMO gradient balancing between tasks | C11; Rep-32, Rep-37 | The owner's Dec 23 idea; FAMO is cheap | comment #252 |
| 29 | A bandit task sampler with a learning-progress reward | C11; Rep-32 | Owner's idea; arm of a sampling comparison | comment #252 |
| 30 | Layer reset or shrink-and-perturb at stuck phases | Rep-27, Rep-32; avenue C10 | Owner none | comment #252 |
| 31 | Train-fraction sweep (critical data size, sweet spot) | Rep-20 step 6; Varma 2023; 2605.14659 | The data-size lever never became an experiment | comment #249 |
| 32 | KL attention intervention to skip the memorisation wait (Hidajat 2026) | §4.2 | Optional arm once #249 groks | comment #249 |
| 33 | Black-box search (Optuna or CMA-ES) over decay, LR, batch, train fraction | Gemini chat 6db3c20f; avenue D11 | After A1 anchors the objective | comment #251 |
| 34 | LR in the sweep grid (owner's LoRA belief in a high LR) | W9 | max_lr never swept | comment #251 |
| 35 | Full-LM-loss arm on dyck (Ahuja 2024) | §4.2 | ReLSM trains answer-only | comment #253 |
| 36 | Halting rules: KL < 0.01 and entropy < 3 (Kohli 2026); halting bias −3 (URM) | §4.2 | Known-good settings for #132 | comment #132 |
| 37 | Cosine fixed-point halting at eval time | Gemini chat 60df30f8 | Cheap eval-time option | comment #231 |
| 38 | Correct StableMax formula | Rep-19 error | The chat's code was plain softmax | comment #255 |
| 39 | GrokTransfer authors wrong in the issue body | §4.3 #1 | The issue cites a paper that does not exist under that name | comment #250 |
| 40 | Parity → addition edge is false for decimal addition; "chain" changed meaning | C13; Rep-32 | The graph's key edge has no basis | comment #157 |
| 41 | `dag_locked_floor` provenance; six Perplexity turns had no web search | dag-origin notes; res-new-exports; reexport-g3 | Records where the gate values came from: ChatGPT chat 694c894c invented them on Dec 25 (section 7b.2) | comment #157 |
| 42 | One-sided jitter followed ChatGPT's advice, not a misreading | Rep-26; ChatGPT V1 prompt | Corrects #220's provenance | comment #220 |
| 43 | Vocabulary 32,016 is not a multiple of 128 | Rep-22 | Small Tensor-Core cost | comment #225 |
| 44 | Unreported `shared_loop` twin of DAG-v12 (`metrics.loop.json`) | R5 | The only loop-vs-baseline record | comment #258 |
| 45 | Loops help OOD composition only after grokking; zero-init output projections | Kohli 2026; Wang 2024 | Order and settings for #258 | comment #258 |
| 46 | Grokking ⇒ longer-input generalisation is unsupported | G4; Rep-19 | It shaped the goal; the research plan should say so | comment #259 |
| 47 | Position-weighted loss on carry digits | Rep-39, Rep-40 | Pairs with the carry-structure work | comment #208 |
| 48 | Modern linear operator instead of the custom Mamba fallback; Mamba's copy weakness | S5; Rep-05 | An alternative to fixing #133/#134 | comment #133 |
| 49 | Learned task embeddings or a meta-learned controller | Perplexity turn 6, 8 | Speculative | record |
| 50 | Bruner spiral: replay old tasks at higher difficulty | Perplexity turn 8 | Speculative | record |
| 51 | Thermodynamic annealing (gradient noise or softmax temperature, then cooling) | Rep-20 experiment C | Speculative; risks noted by Rep-21 | record |
| 52 | Dataset distillation: "the minimal, most potent dataset" | Rep-02 | No ReLSM path | record |
| 53 | Retrofit latent reasoning onto a pretrained model | R7 | The answer says ReLSM's from-scratch path is the right place | record |
| 54 | Multi-token / token-order prediction; Rho-1 token selection | Rep-07, Rep-49 | Data-efficiency track; avenue C11 | record |
| 55 | Small diffusion LMs (MDLM, Duo) and LAW feedback as recurrence | Z6; Rep-47 Session 03 | Diffusion costs 14–27× AR compute | record |
| 56 | Capacity beyond VRAM (PEER, UltraMem); QAT, pruning, distillation | X8, Z8 | Deployment questions | record |
| 57 | NEXUS, the 1.5B looped hybrid, "beat 4B", B200 runs, 16k context | R2, X1–X3 | Dropped by the project; context recorded | record |
| 58 | Monthly frontier watch; "assemble the pieces"; J-Space | Z3, R11 | Method and reading, not repo work | record |

### 6.1 Added after the full re-export

Ideas from the outside chats and the re-export that no issue from #132 to #272 owns. 11 rows. "Action" points to part B of the proposal file `reexport-issue-changes.md` (not in the repo; nothing is posted).

| # | Idea | Source | Why it matters (one line) | Action |
|---|---|---|---|---|
| 59 | Fixed-regime OOD probe tasks (addition ≤ 4 digits, parity length 8) whose training range never moves, read by eval from one constant; a minimal OOD pack (parity + addition OOD, baseline vs one or two variants); a fixed difficulty sweep for run comparisons | C21; Rep-12a; Rep-29c | The owner designed a clean "train small, test big" probe on Dec 12 and lost it by Dec 20 | B1: comment #162 |
| 60 | Cross-task gradient-direction consistency (cosine of per-task gradients on a random projection) as an interference diagnostic; a rehearsal trigger when a task's eval drops while train-eval stays high | Rep-25a; chat 6949cb7d msg 10 | Measures the "monopolisation" the DAG was built against | B2: comment #252 |
| 61 | The v1 per-token thought design as the reference for the #135 redesign, and its gap (z needs a read path into h); a tokenwise latent state as a ladder rung | Rep-09 (chat 69378f5b); Rep-45a §XIII rung 4 | v1 was causal; the leak came with the Dec 9 ladder | B3: comment #135 |
| 62 | A halting control at matched average compute: fixed K, random per-token K and confidence stopping at the same mean K as learned ACT | Rep-47 (Session 05 first answer) | Without it a learned-halting gain cannot be told from extra compute | B4: comment #231 |
| 63 | Keep the owner's 16-source `datasets.txt` as a Phase 2b SFT list, and add a non-story perplexity eval (a held-out FineWeb or MiniPile shard) | D12 | The list exists only in one chat | B5: comment #261 |
| 64 | Mixing-band width: ChatGPT advised 5 % of `alg_tokens` or 0; the default is 50 % | C28 | Any language run starts mixing at half the budget unless the flag is set | B6: comment #158 |
| 65 | Golechha-style progress measures done properly: hidden-unit sparsity, absolute-weight entropy as defined, and a weight-perturbation complexity probe on one probe batch | Rep-25a; §4.2 | #238 and #245 only fix or rename keys | B7: with the #238 and #245 corrections |
| 66 | Session 07 controls for any concept-unit arm: the same evidence given as plain text or JSON, an exact copy or pointer baseline, oracle parsing | Rep-47 (Session 07) | Separates the unit from the information it carries | B8: comment #266 |
| 67 | Dec 17 Exp0 throughput fell 42 % (7,980 → 4,608 tok/s) while Exp2 fell 7 % on the same stream | 7b.1 | Unexplained; logging costs would hit both runs | record |
| 68 | Long-context eval beyond single-needle NIAH (multi-needle, aggregation, multi-hop; RULER and BABILong style) | Rep-45a §X | #163 fixes the current NIAH code only | record |
| 69 | Training-time internals: per-layer residual RMS, attention mean distance, memory-token attention mass, a `grad_to_weight` EMA per block | Rep-29b; Rep-25a | Useful once a run is worth diagnosing; low priority now | record |

---

## 7. ChatGPT chats outside the ReLSM project

Source: res-chatgpt-outside, which read 37 chats from Oct 16, 2025 to Oct 23, 2026 (file `chatgpt_outside_project_37.json`; the same chats are inside `chatgpt_full_export_2026-10-28.json`). Six chats bear on ReLSM. Five are the owner's fine-tuning work before ReLSM (Oct 2025). The other 26 are other projects: finance, philosophy, game design, game QA and agent harnesses.

### 7.1 The six chats about ReLSM or research that bears on it

| Chat | Date (UTC) | Owner's words | What it adds | Issues |
|---|---|---|---|---|
| 69430040 "Analyse et critique rapport" | 2025-12-17 19:10 | « Valider, analyser, commenter et critiquer » ("validate, analyse, comment and critique"); « Je veux tout ça. » ("I want all of it.") | He sent the Gemini "Grokking du Langage" report (Rep-20) to ChatGPT the same day. The critique (very probably Rep-21) asked for λ = 0 controls and a few-verb pilot. It called the physics analogies "largely heuristic" but accepted three unsupported claims. Nothing was built. | #260, #249 |
| 694d7ffc "Training Issues with Copy Reverse" | 2025-12-25 18:19–19:10 | "Copy and Reverse don't seem to be training well at all." / "alg_seq_len=64 fixed it!" / "do we need to hit current full difficulty to open the gates?" / "Why isn't TASK_DIFFICULTY_CONFIG used?" | ChatGPT found the 32-token truncation; he fixed it at 64 within 30 minutes (Rep-35c). It proposed per-task truncation counters and a gate at a fixed hard difficulty; neither was built. DAG-v12 ran at 48 four days later and truncated again. He then asked for one source of truth for the per-task phases (C27). | #139, #157, #203 |
| "Largest De-duplicated Dataset" | 2026-03-02 | "a below 5% change on benchmarks is ... totally within the statistical error" | His first significance rule (E24); quality vs quantity (D14); streaming dedup (D15); Rep-56. | #259, #257, #159 |
| "Emoji-based Reasoning Ideas" | 2026-05-02 | "have the model create its own language ... specifically for reasoning" / "I find the interpretability idea completely moot." | His own latent-reasoning idea (K8, K9). ChatGPT made it a neutral code bottleneck with six kill tests and an OLMo 2 1B pipeline (Rep-57). Nothing built. | #266, #264 |
| 6a56a593 "ReLSM December 2025 Project" | 2026-07-14 21:09 | "Can you put all our conversations from December 2025 about ReLSM into one project?" | The day before the July re-audit (Z11). | — |
| 6a6de5a9 "ReLSM ledger update and decisive experiment" | 2026-08-01 to 10-23 | (the scheduled task prompt) | Four "Frontier Architecture Watch" reports: Aug 1, Sep 1, Oct 1, Oct 23 (Rep-55). No ReLSM runs or numbers. The decisive experiment changed every month; the Oct 23 version is close to #258. Grokking stays a side topic. | #258, #263, #270, #271, #272 |

Two points in res-chatgpt-outside are wrong, shown by the re-export (section 7b.5):
- **The Oct 23 task prompt is not an owner rewrite.** Its 3,095 characters are identical to the prompt saved on Jul 15, 2026 at 20:33 UTC in chat 6a56e38e (msg 93). That was the third version in 16 minutes, and ChatGPT's scheduler wrote it from his requests. So the ten-family loop taxonomy and the evidence classes quoted from it (res-chatgpt-outside A56–A61) are adopted wording, not his own words.
- **The ledger body is in the export.** It is chat 6a56e38e msg 77, inside the project.

### 7.2 Before ReLSM, Oct 2025 (5 chats)

QLoRA fine-tuning of Qwen3 and a Mistral-Nemo 12B on rented H100 and H200 GPUs, with $50 to $90 budgets. Three habits are already there:
- he pastes other AIs' configs and asks one AI to judge them;
- he trusts his own run over the AI's prior, though on one run with no working eval (E26);
- he catches silent drift: "you forgot the other changes we talked about" (E27).

He also asked whether a choice would "hurt the acquisition of reasoning capabilities". ChatGPT made several errors he had to absorb (Rep-01a; §4.3 #57).

### 7.3 Other projects, Feb to Sep 2026: method rules that transfer

None of these is about ReLSM. The rules are his:
- Finance (Feb 9): "Avoid MDPI, that's a predatory publication."; ask for failed experiments; "only theorems with formal mathematical proofs"; one win condition, not several.
- Agent plans and games (Feb to Jul): "Am I seduced by their marketing?"; blind plans from two agents, then criteria written first and applied the same way to both; "red team versus previous version".
- Harness comparison (Sep 28–29): "I don't want any benchmaxing"; with about 10 tasks report the per-task matrix, not a percentage; contamination needs evidence, not a release date; cost decides inclusion.

None was applied to ReLSM's own evaluation at the time. The Oct 26–27 audit found defects these rules would have caught (#160, #206, #237).

### 7.4 Answers to the placeholder's questions

- **Opening prompts.** 693a2b2b's first 10 prompts are recovered (Dec 11, 02:23 to 18:44); "What are the best practices as of December 2025?" led to Rep-13. 694c894c's first turns hold the PR #108 spec (section 7b.2). The Session 05 brief and the Session 07 prompts are recovered, and the beyond-VRAM brief is complete (Z12). The JEPA chat's original question is still lost: the export starts with "Not GPT-8. JEPA."
- **Chat 6adbf078.** The answers are recovered (K4, Z5, Rep-47a).
- **Grokking or recursion outside the project.** Only three places: the Dec 17 language-grokking audit, the May 2 dense codes, and the Aug–Oct watch. No ReLSM run appears in the outside chats after Dec 25, 2025.
- **The July 2026 literature report.** Yes: chat 6a56e38e msg 77, the claim ledger. "Claims 21–30" is its Section III.

---

## 7b. What the full ChatGPT re-export added

The first ChatGPT exports were page text. 21 of the 30 Dec 2025 files and 7 of the 18 2026 files started with « Chargement des messages plus anciens… » ("loading older messages…") and missed their oldest messages. The API re-export `chatgpt_full_export_2026-10-28.json` holds 85 chats: the 46 relsm-project chats with every message, the 37 outside chats of section 7, and 2 unrelated chats. Four readers (reexport-g1 to reexport-g4) compared each message with the old export and read what was new. Times are UTC.

### 7b.1 Recovered run records

| Date | Run | Settings | Outcome | Chat |
|---|---|---|---|---|
| Dec 17 | Exp0 baseline and Exp2 latent (`metrics_baseline.json`, `metrics_latent.json`, inlined, cut after `avg_inner_steps`) | about 20M algorithmic tokens; 228 log points each; 2-ahead objective | Train loss 10.24 → 0.550 (Exp0) and 10.35 → 0.174 (Exp2). One jump at 7.99M tokens in both (0.53 → 0.86; 0.18 → 0.45). No late drop. No validation loss. Exp2 ran 4.0 inner steps throughout. VRAM 5.9 and 6.27 GB. The "language" batches were aliased algorithmic replays (#158) | 6942ce5e |
| Dec 19 | `exp0_baseline_100M_v2` | wd 0.001, max_lr 6e-3, `warmup_steps` 30,000, clip 0.5, batch 24 × 12 | Best 12.9 % at step 2,500. Grad norm 128.6 at 35,000 and 326.1 at 37,500, the last line, exact 0.0. The LR was still rising: the warmup really ended near micro-step 56,000 (#248) | 6945b243 |
| Dec 19 | `exp0_baseline_100M_v4` | wd 0.0, max_lr 1e-4, warmup 10,000, clip 0.1, batch 64 × 12, `linear` schedule | Best 18.0 % at 20,000; 16.9 % at 24,000, where the log ends (32.8 % of the budget). Parity 10–42 %, below the 50 % chance level. Addition, multiplication, chain, copy and reverse 0 at all 24 evals | 6945b243, 6945e825 |
| Dec 20 | a run before the label fix | `eval_max_new_tokens` 24 | Accuracy 0.065, 0.067, 0.063 at 2k, 4k, 6k. Attention entropy 1.92 → 0.84. FF "sparsity" about 3e-4 → 5e-5. MAE about 6e28 | 6946d7ab |
| Dec 20 | `exp0_baseline_100M_v10` | nano baseline, 1e8 algorithmic tokens, `smooth`, adaptive weighting, `alg_seq_len` 128, batch 72 × 16, `num_workers 0`, max_lr 1.5e-3 WSD, wd 0.0, clip 5.0 | Loss 10.05 → 1.14 by step 2,000; token accuracy 0.55; exact 0.085. Difficulty 0.250001 → 0.250010 at 2.6 % of the budget. The in-training eval mixed 6- to 10-digit additions into one score. Ran just after the label fix | 693b54b7 msg 246 |
| Dec 21 | `exp0_baseline_100M_v12` | as v10 | Stopped at step 500 (6.19e5 tokens); exact 0.079 at eval difficulty 0.2. **Not DAG-v12** (Dec 29–30): any run index must keep the two names apart | 69484386 msg 973 |
| Dec 21 | an unnamed 43,000-step run | max_lr 1.5e-3 WSD, wd 0.0, clip 5.0, `alg_seq_len` 32, `eval_samples` 96 | Loss about 0.78 from step 5k on. Difficulty about 0.2, then about 0.0 (#201). "Mix" from 38,500 was alias replay (#158). Final IID mean about 0.25 (compare 0.635 … chain 0.021). Train-eval reached about 1.0 on easy tasks at difficulty about 0 | 69488a01 msg 1 |
| Dec 22 | v15 and v16 | `smooth`, adaptive, task curriculum on, cooldown 500, jitter 0.1, mix band 50M, `alg_seq_len` 32, batch 72 × 16, `num_workers 0`, Grokfast off, max_lr 1.5e-3 → 1.5e-4 WSD, wd 0.0, clip 5.0, `eval_max_new_tokens` 32, seed 42. v16 adds `curriculum_min_task_evals` 5, 800 train-eval and 80 eval samples | In-training IID 27.83 % (v15) and 20.00 % (v16); `eval_hub` 8.6 % and 13.8 % (7b.4). v18's flags are still unknown | 6949cb7d tool msgs 1–2 |
| Dec 23 | step 68,000, probably v16 | — | IID 20.4 %. Compare at difficulty 0.00 on a train-eval score of 14.3 % while IID compare was 60 %. The train-eval targets were corrupt: `'> 7'` from a separator tie-break; a copy target cut to `'8 4 1 7'` | 6949cb7d |
| Dec 23–24 | run4 (`5steps`) and run5 (`20steps`) | identical except `curriculum_min_task_evals` 5 and 20; `alg_seq_len` 32, batch 72 × 16, `num_workers 0`, wd 0.0, clip 5.0, `eval_max_new_tokens` 32; no `min_difficulty` key (before PR #105) | Token accuracy 75.8 % and 75.3 %. "Mix" from 50M tokens (alias). A copy example averaged 21.2 prompt + 12.8 answer tokens | 694b4280 |
| Dec 25 | the run on hand before the DAG (probably run6) | `alg_seq_len` 32 | Solve accuracy: successor 0.9875, addition 0.975, parity 0.975, dyck 0.4375, multiplication 0.3625, compare 0.1125, chain 0.025, mod_add 0.0125, **copy 0.0, reverse 0.0** | 694c894c 00:51 |
| Dec 25 | the first DAG run, after prompt masking | `alg_seq_len` 32 | IID 10.8 %: successor 56.2, parity 51.2, the other eight 0.0. Train-eval targets truncated (reverse "0 3 2 3 4 5 2 0" expected `0 2 5 4`) | 694c894c 18:16 |

### 7b.2 Where the DAG spec came from

ChatGPT chat 694c894c, Dec 25, 2025 (the evening of Dec 24 in Montréal):
- **01:15.** He asks: "Does that change anything to the idea and order of a DAG instead of introducing all tasks at the same time like we currently do?" He already had the idea.
- **01:18.** "propose an exact DAG + unlock rules using the metrics you already log (EMA per-task accuracy + patience + replay mix), so it's a small code change". He reused ChatGPT's own offer word for word.
- **01:19.** ChatGPT writes the whole spec. Roots: successor, copy, reverse. Dyck ← copy, reverse; compare ← addition. Thresholds 0.99 / 0.98 / 0.95 / 0.97, "strict but not insane". Patience 4, unlock margin +0.01, lock margin 0.03, backslide replay 35 %, ramp 3, floors per task. No source gives these values. ChatGPT made them up in this answer. It read an uploaded PDF with no filename; by content the "Curriculum Étagé" critique (Rep-38) or the Nash-SEC report. Rep-38 is the one report that warned against staging.
- **01:58.** The Codex prompt, verbatim: "You are Codex working in the repo from this zip: ReLSM-main(7).zip". PR #108 was opened at 02:18. 43 minutes from question to prompt.
- **02:24.** The "review turns" of PR #108 are ChatGPT's review of Codex's first diff, item 1: "Unlock margin sign is backwards (too lenient)".
- **16:40.** The same chat calls `+` a bug. PR #110 goes back to `−`.

So Perplexity and Gemini are upstream of the PDF, and ChatGPT is the direct author of the graph and every value. The margin flip is ChatGPT contradicting itself within 14 hours.

### 7b.3 Truncation behind the first DAG and the 6-item ceiling

- **The data behind the DAG were a truncation artefact.** At `alg_seq_len` 32 a copy example averages about 35 tokens with EOS. Copy and reverse sat at 0.0 while successor, addition and parity were at 0.975 to 0.9875. ChatGPT read this as a task-order problem. The graph then locked addition (0.975) behind chain (0.025).
- **The first DAG run's gates scored cut-off answers.** Its train-eval targets were truncated too.
- **The 5steps/20steps ceiling is #139, not #141.** Every copy and reverse prediction in those files is exactly 6 items. #141 and #162 explained it as training stuck at the first phase, through the worker bug. But run4 and run5 used `num_workers 0`, and #141 needs workers. 21.2 + 12.8 tokens + EOS = 35, above the 32-token window. About 11 answer tokens fit: 6 digit-plus-space items, exactly the observed `pred_len` 11. The IID-grid point of #162 still stands. The earlier claim that these files cannot be truncation (chatgpt-c C-1, and #139's audit caution) looked at the eval commit's default of 128, not at the training flag.
- **The same cut appears in v15/v16** (a copy target `'8 4 1 7'`) and in the Dec 21 43k run. So every run from Dec 21 to Dec 25 trained at 32 tokens.
- **The owner fixed it on Dec 25 at 64** (chat 694d7ffc). DAG-v12 ran at 48 four days later.

### 7b.4 Two evaluators, opposite verdicts on v15/v16

| Run | In-training IID (`eval_max_new_tokens` 32) | `eval_hub` IID, Dec 23 (`max_new_tokens` 64, seed 1234, 200 per task) |
|---|---:|---:|
| v15 | 27.83 % (token 37.21 %) | 8.6 % (19 parse failures) |
| v16 | 20.00 % (peak 20.75 %) | 13.8 % (1 parse failure) |
| v18 | — | 13.8 % |

The in-training eval says the `min_task_evals` change cost 7.8 points; `eval_hub` says it gained 5.2. Compare is 65.0 % and 55.0 % in one, 0.5 % and 6.5 % in the other. The owner was told both within a day. No curriculum conclusion from Dec 22–23 survives this (#160, #227).

### 7b.5 The claim ledger and the July 2026 history report

Both are in ChatGPT chat 6a56e38e, written on Jul 15, 2026 (the evening of Jul 14 in Montréal):
- **msg 74, "ReLSM project history and current state"** (18,499 characters), from the prompt "Summarize the project history and state from chats, sources & code" (01:34 UTC). It read the repo at `acd48ff` through the GitHub connector and found the three #132 ACT defects, the `shared_loop` depth mismatch (#145, #170), the duplicate weight tables (#180, #222), the missing CI (#188) and the hypothetical README table (#169). It proposed one frozen-curriculum, three-seed Exp0 vs Exp1 study, token- and FLOP-matched: #258 in all but name. Nothing was filed for three months. One false claim: "Later fixed-dataset experiments demonstrated memorization".
- **msg 77, "ReLSM claim ledger versus the July 2026 state of the art"** (41,946 characters), 6 minutes after "List all claims, proposed or rejected, and compare with the state of the art" (01:51). It says 58 claims and numbers 1–66, plus 28 more rows. Sections I–XIII; Section VIII is grokking; §XIII is a 0–8 ladder.
- **msgs 80, 88, 93: the watch task.** Three versions in 16 minutes (20:17, 20:28, 20:33). ChatGPT's scheduler wrote the final prompt ("Produce a monthly, source-grounded update of the entire ReLSM claim ledger …"). First run: Aug 1, 2026, 08:00 Montréal time.

So the ledger is no longer only inside ChatGPT: it is in the export, and #272's first step can copy it.

### 7b.6 Owner ideas stated before an AI said them to him

| Idea | His words, date, chat | What came later |
|---|---|---|
| A repeated subset in the stream | "Maybe we have to only introduce a small subset of new problems every epoch, or something of the like." Dec 19, 2025, 20:14, ChatGPT chat 6945b243 (G16) | Claude's Rep-19 had suggested a 25 % repeated subset two days earlier, so this may be his restatement. ChatGPT gave it as new advice on Oct 26, 2026 (D10), and #257 credits that. |
| Memorise first | "it won't ever grok, because it won't memorize, so that is the source issue we need to identify." Dec 19, 20:16, chat 6945b243 (G15) | ChatGPT's tiny-overfit test later in the same chat (Rep-23); #256 on Oct 27, 2026. |
| A reasoning language the model invents | "have the model create its own language ... specifically for reasoning", May 2, 2026, outside chat "Emoji-based Reasoning Ideas" (K8) | ChatGPT on Oct 23, 2026 called Hybrid Reasoning Agents (Oct 2026) close in philosophy (Rep-47a). |
| Internal concepts need not be human | "would it be more natural to let the model come up with its own representation, or is this point moot?" Oct 24, 2026, 15:30, chat 6adccafc (K6) | ChatGPT: "Your objection is right." The merged doc had only ChatGPT's restatement. |

Two more came early but not before every AI: "Wait, do we generate a new dataset every epoch?" (Dec 12, G18), ten months before #140; and the DAG idea itself (Dec 25, 01:15, C25), after Perplexity's phases of Dec 23 but before ChatGPT's spec. The compiler analogy of Session 07 was his request (K10).

### 7b.7 Other research the re-export added

- **Rep-23a**, a Dec 19 ChatGPT code audit: the first to prescribe answer-only loss, and wrong that there were no fatal bugs.
- **Rep-25a**, a Dec 20 ChatGPT Deep Research report on grokking progress measures: the source of PR #73. The `|x| < 1e-4` FF-output threshold of #238 was in ChatGPT's own spec. #245's key implements what this report asked for ("Shannon entropy of the absolute weights"), so comparing it with the later Nash-SEC SVD entropy is out of date order. The rename still stands.
- **Rep-29a**, the Dec 21 wiring review that quoted invented code.
- **Rep-12a**, the Dec 12 probe design and three-phase ramp.
- **The mixing band** (Dec 11, C28): advised at 5 % of algorithmic tokens or 0; the default is 50 %.
- **`datasets.txt`** (D12): the owner's 16 sources, recorded nowhere else.
- **The v1 ReLSM design** (Rep-09): a per-token thought state, causal in principle, but with no path to the token output. The leaking sequence-level thought tokens (#135) came with the Dec 9 ladder rewrite. Part of the Dec 16 "ReLSM paper" is ChatGPT's own Dec 9 code, including #132's batch-wide halt.
- **The 2026 session briefs** carry arXiv ids for about 25 core readings. Two spot checks resolve to real papers dated Sep 30, 2026 (2610.00673, 2609.38806).

### 7b.8 Date corrections

- Rep-17, D4 and E3: Dec 11 in chat 693a2b2b (16:15, 02:34, 16:47 UTC), not Dec 16 in 6941b170, a later fork that inherited them. res-chatgpt-dec rows 29–31 have the same error; only the hard-drive question (row 32) is Dec 16.
- The AO3 "contradiction" is one chat on one day: AO3 proposed at 16:15, avoided at 18:56.
- Chat 6a56e38e began Jul 15, 2026 01:34 UTC (Jul 14, 21:34 EDT). Chat 6adccafc began at 15:13 UTC (11:13 EDT), not 12:50.
- The watch prompt dates from Jul 15, 2026 20:33 UTC, not Oct 23.
- The PR #108 spec: Dec 25 01:19 UTC (Dec 24, 20:19 EST).
- Rep-47: 8 sessions were planned (6adbf078 msgs 32–33); 01 to 07 ran; 08 became the integrated report (Rep-48).
- D8: the truncation was found at 32 tokens, not 48, and fixed at 64 on Dec 25.

---

## 8. Sources and coverage

| Service | What was read | By | Known gaps |
|---|---|---|---|
| ChatGPT, ReLSM project | First 48 page-text files: Dec 8–25, 2025 (30) and Jul–Oct 2026 (18). Then the full API re-export of the 46 project chats, every message, with the parsed text of attached PDFs and many metrics files as tool messages | res-chatgpt-dec, res-chatgpt-2026; reexport-g1 to reexport-g4 | Still missing: the name of the PDF behind the Dec 25 DAG spec; v18's flags; the Oct 24 integrated report and the 2026 saved Deep Research files; the JEPA chat's first question; most tool outputs in chat 69484386 (one-line file reads). Attached metrics files are cut at about 42 KB |
| ChatGPT, outside the project | 37 chats (81 relevant of 233 candidates); 6 bear on ReLSM | res-chatgpt-outside | Game chats: owner prompts in full, answers grepped; three Deep Research files in the dataset chat not exported; nearly all 2026 papers not checkable |
| Claude.ai | 17 ReLSM chats, Dec 9, 2025 – Jan 1, 2026; plus the 2026-10-28 bundle: 40 relevant of 267 chats | res-claude, res-new-exports | Jan 13–15, 2026 screenwriting chats not read; attachments not inlined (the ReLSM PDF, NEXUS, metrics files); Research tool output not exported; Feb 2026 concept proposals known only second-hand |
| Gemini | 18 chats, Oct 2025 – Jul 2026 | res-gemini-perplexity | No API; in-app search found no other ReLSM chat; 6 report prompts lost; some chats undated |
| Perplexity | Nash-SEC thread (14 turns, Dec 23–24, 2025); 53 threads in the 2026-10-28 bundle, of which 1 is ReLSM | res-gemini-perplexity, res-new-exports | "Computer" mode deliverables are files outside the thread; `metrics.4.json` not on disk |
| Agent reports | 13 reports (PDF text): Compass, Ramping, Bugs, Pondéré, Équité, Nash-SEC design, NLP, Triadic, RECH, Audit-MTL, Étagé, Éval, Structuré; Grokking du Langage and its Validation | res-gemini-perplexity | Read through text extractions only |
| Codex | 131 task files, PR001 (2025-12-10) to PR131 (2026-01-12) | res-codex-lit | About 45 files cut long prompts at « Afficher plus » ("show more"); most paste sources are style guesses |
| Literature notes | lit-lexical-grokking, lit-embedding-table, lit-puzzles-concepts, the lexical-grokking note in PR #196, research_grokfast, research_adam_resets, counter-review §6 | res-codex-lit | No paper re-checked; 2026 ids rest on the source agents |
| DAG notes | dag-origin-idea, dag-origin-implementation, dag-runs-and-language-switch | this file | Confirmed by the re-export: ChatGPT chat 694c894c wrote the PR #108 spec (01:19 UTC, Dec 25) and the Codex prompt (01:58); the review turns are ChatGPT's (02:24) |
| Stray files | `metrics.loop.json` (Dec 30, 2025 loop run); `literature_register_2026-10-24.json` (65 sources); `eval_results_3.json`, `eval_results_4.json` (not ReLSM: an NPC-narrator benchmark) | res-new-exports | `metrics.loop.json` has no end record |
| Google Jules | Codebase list and repo page | search log | ReLSM was never connected |
| Code | `main` at `be08cde`, by `git grep` | all readers | Nothing was run; "built" means present, not shown to work |

General limits:
- Citations were checked from the readers' memory, without network access. Corrections are confident for the well-known papers (Mamba, RMT, BLT, Softmax Collapse, GrokTransfer, GPT-3, StableMax, Grokfast); "Thilak et al. 2023", "Carvalho et al. 2025" and "Perfume/PerfRe" could not be checked.
- Run files: `metrics_baseline.json` and `metrics_latent.json` (Dec 17) are inlined, cut short, in ChatGPT chat 6942ce5e; their "language" batches were algorithmic replays, so no run on record had a language phase. The run headers of `metrics.4.json` and `metrics.5.json` are quoted in chat 694b4280. Still lost: `metrics.6.json`, the files themselves, and every checkpoint folder except DAG-v12.
- Owner idea counts are rows, not prompts. Logistics turns ("Continue", commit messages) are not counted.

---

## 9. Changes after the full re-export

Every spot below was updated 2026-10-28 from the full re-export (reexport-g1 to reexport-g4) or from res-chatgpt-outside. Nothing is marked inline.

**Header and section 1**
- Header: the seventh reader and the four re-export readers are merged; the placeholder sentence is gone.
- Research count 54 → 71 (2025: 41 → 53; 2026: 13 → 18). Reports with an error 40 → 51: 10 new reports with errors, plus Rep-45, which now has one false claim. Section 4.3 rows 40 → 59.
- New sentence on the three "the code is fine" reviews (Dec 16, 19, 21).
- "What was built": the DAG's graph and values are ChatGPT's (694c894c), not only "from AI reports".
- "What was never tried": the Dec 17 "language" batches were replays, so no language phase ever ran.
- Point 3: the PR #108 spec and the truncation behind its data.
- New paragraph on what the re-export changed; section 6.1 count added to point 5.

**Section 2 (131 → 161 rows)**
- Corrected: G2 (Dec 17 logs), R4 (v1 design and the leak), R8 (the 0–8 ladder), H4 (the July ACT diagnosis), C3 (v10 had `num_workers 0`: plateau one of #223, not #141; issue order now #223, #141), C12 (upstream, not the source, of PR #108), C14 and C18 (694c894c), K3 (ChatGPT's own Feb 23 chat), K4 and Z5 (answers recovered), K6 (his own words, 15:30), D4 and E3 (Dec 11, chat 693a2b2b), D8 (truncation at 32 found and fixed at 64, chat 694d7ffc), D10 (his Dec 19 idea), E8 (his Dec 23 decision), Z3 (the scheduler wrote the prompt).
- Added: G15–G18, C21–C28, K8–K10, D12–D15, E20–E27, Z10–Z12.

**Section 3**
- New rows: Dec 11–12, Mar 2026, Aug–Oct 2026. Changed rows: Dec 19, Dec 20–21, Dec 24 (the spec times), Dec 25 (the 64-token fix), May 2026, Jul 15, 2026. Two "kept returning" bullets updated.

**Section 4**
- 4.1 corrected: Rep-09 (inlined; v1 thought stream; ChatGPT's code), Rep-17 (Dec 11, chat 693a2b2b; three more errors), Rep-21 (source very probably chat 69430040), Rep-23 (more errors), Rep-32 ("upstream source"), Rep-45 (the history report; one false claim; verdict partly), Rep-47 (8 planned sessions; Session 05's first answer; arXiv ids).
- 4.1 added: Rep-01a, 06a, 12a, 23a, 25a, 29a, 29b, 29c, 33a, 35a, 35b, 35c, 45a, 47a, 55, 56, 57.
- 4.2 added: Golechha 2024.
- 4.3: row 31 re-dated; rows 41–59 added.

**Section 5**
- Weight decay row: Rep-23a and Rep-29c added; the runs at wd 0.0 named.
- New rows: gradient clipping, model size, "the code is fine", which evaluator to believe.
- Unlock margin sign: ChatGPT contradicted itself; it was not Codex against ChatGPT.
- Same assistant: the AO3 pair is one chat, one day, opposite order.
- Language phase share: the 5 % mixing-band advice.

**Section 6**
- Row 41: provenance now ChatGPT 694c894c.
- New 6.1 with rows 59–69.

**Sections 7 and 7b**
- Section 7 filled from res-chatgpt-outside, with two corrections to that report: the Oct 23 watch prompt is the Jul 15 text, and the ledger body is in the export.
- Section 7b is new.

**Section 8**
- ChatGPT rows: the re-export and the outside chats are read; gaps restated.
- DAG notes row: the PR #108 author is confirmed.
- Run files: the Dec 17 metrics and the run4/run5 headers are recovered.

**Wording that changed meaning**
- "Probably ChatGPT chat 694c894c" → confirmed.
- "Only inside ChatGPT" (the ledger) → in the export (6a56e38e msg 77).
- "v15/v16 flags lost" → recovered; only v18's are unknown.
- "The 6-item ceiling is the worker bug" → truncation at 32 tokens (#139).
- "The owner rewrote the watch prompt on Oct 23" → set on Jul 15 by ChatGPT's scheduler; repeated unchanged on Oct 23.
