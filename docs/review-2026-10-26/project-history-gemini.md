# ReLSM project history from the owner's Gemini conversations

Built from the 18 exported Gemini chats in `ReLSM-transcripts/gemini_chats/` (local bundle, not in the repository) and, for sections 6 and 7, the agent research reports in `ReLSM-transcripts/agent_reports/`. Checked against `counter-review.md` (CR: F01–F30, A1–A8, B1–B15), `issues-index.md` (#132–#194), the Codex task exports and the code at HEAD `e6724c3` (code identical to the audited `acd48ff`).

Tags: **[G:xxxx L123]** = Gemini export whose file name starts with `xxxx`, line 123. **[R:name L12]** = agent report `.txt`/`.md`, line 12. **PR #n** = the Codex task behind that PR. Years are not printed in the Gemini chats; "(inferred)" marks a date or link that the text does not state outright. Quotes are exact; French is quoted as written.

---

## 1. Chat inventory

7 chats are ReLSM-related, 4 are partly related, 7 are not. Sorted by date within each group.

The owner says six chats had the repo `jpturcotte/relsm` attached. The export shows the attachment marker in **four**: 216cd96a, ce77dd7d, bfb14b23, f5bb0569. The other two cannot be identified from the text; 45e61603 and 86665c16 work on repo code but carry no marker.

| Title | Link | Date | ReLSM-related? | Topic |
|---|---|---|---|---|
| Tokenizer pour Transformers Maths Langage | https://gemini.google.com/app/ba6cfceb8e1156a5 | 2025-12-17 (printed "17 déc., 16:05") | yes (not named) | Best tokenizer for a 50M–1B math+language model on an RTX 4060. Commit e9de18b "Default to llemma tokenizer" is the same day |
| Fixing Evaluation and Scoring Logic | https://gemini.google.com/app/45e61603cdc5059b | 2025-12-20 (inferred: text pasted into PR #71, created Dec 20) | yes | Cuts an eval-fix prompt down to 4 tasks: normalization, last-number extraction, parse-failure tracking, `hash()` seeding |
| Implement Algorithmic Curriculum V1 | https://gemini.google.com/app/86665c167958f409 | 2025-12-20/21 (inferred: PR #84 Dec 20, PR #85 Dec 21) | yes | NaN per-module grad norms ("safe" snippet → PR #84); review of the per-task curriculum diff ("Manager Pickling Bomb", "Amnesia" → PR #85) |
| ReLSM Codebase Review: Strengths & Weaknesses | https://gemini.google.com/app/216cd96a6459b802 | on or after 2025-12-30 (inferred: names `compute_dag_weighting`, added Dec 30) | yes, repo attached | Full repo review: "Rating: 9/10 (Research Quality)" |
| Fixing Pickle Error on Windows | https://gemini.google.com/app/ce77dd7dea3f3876 | 2026-01-11 (inferred: commits 26062f9, 10863d8, dd5e7d4 and PR #130 that day; the index's "1 mai" is a parsing artefact) | yes, repo attached | Unpicklable lambda; stale worker snapshots → Manager dicts; `ListProxy.clear`; Windows paging file → `--num_workers 0` |
| Fix Duplicate Metric Logging Error | https://gemini.google.com/app/bfb14b23a550b186 | 2026-01-11 (inferred: commit acd48ff 19:31, PR #131) | yes, repo attached | Duplicate `*/numeric/mae` keys crash every IID eval; agent prompt; review of the fix diff |
| Training vs. Evaluation Difficulty Mismatch | https://gemini.google.com/app/f5bb0569b7096e15 | on or after 2026-01-11 (inferred: the pasted log finishes an IID eval with the Jan-1 key names, which crashed before acd48ff) | yes, repo attached | Why IID scores differ at "difficulty 0.500": final-phase grid vs current curriculum |
| Exploring Recursive AI Architectures | https://gemini.google.com/app/5905ea6ad5c2fddd | 2025-12-08 (printed "8 déc., 16:51") | partly | HRM/TRM survey; proposes a latent "thinking" loop with a halting head; pivot to a "Latent Recurrent Adapter" for text. The design family of the latent/act variants |
| Liberally Licensed Fiction Corpora | https://gemini.google.com/app/4ddca33763ad1e95 | 2025-12-13 (printed) | partly (inferred) | Fiction corpora and "grey" datasets for a language phase. None of them is in `data.py` |
| Grokking du Langage : Plan de Recherche | https://gemini.google.com/app/46f1189745f32b28 | 2025-12-17 (printed "17 déc., 13:55") | partly | Grokking for language: French conjugation, creolization, annealing experiments. Not the source of the DAG (section 5) |
| Diagnostic de l'effondrement neural IA | https://gemini.google.com/app/dd319d3d75eab329 | 2025-12-24 (printed) | partly (inferred) | Neural collapse and real-time metrics. Nothing traced to code (the diagnostics were added Dec 20, aa0b66b) |
| Research Plan: TF32 on B200 | https://gemini.google.com/app/af53de468c5acddb | 2025-10-20 (printed "20 oct."; year inferred) | no | B200 TF32 speed |
| Downsizing Large Language Models | https://gemini.google.com/app/913491115de3aa0e | 2025-10-22 (inferred) | no | Pruning and distillation pipeline |
| RWKV Mixture-of-Experts Architecture Exploration | https://gemini.google.com/app/6db3c20f03b5cec9 | 2025-11-23 to 25 (printed) | no | A separate RWKV-7 MoE merge project with its own README |
| AI Model Training Templates | https://gemini.google.com/app/d75ff83a2de86eda | 2025-12-22 (printed) | no (adjacent) | Jinja chat templates and training recipes |
| AI/ML Text Decomposition Concepts | https://gemini.google.com/app/834815ac09e7768d | 2026-02-20 (printed; year from content) | no (adjacent) | Text → concepts literature |
| LLM-Graph Integration Research Plan | https://gemini.google.com/app/60df30f88a9a7845 | 2026-02-21 (printed) | no (adjacent) | LLMs with concept and semantic graphs |
| Distilling Large Models: A Guide | https://gemini.google.com/app/af708951f66c3210 | 2026-07-29 (printed; prompt says "as of July 2026") | no | Distillation tooling |

---

## 2. Timeline as it bears on ReLSM

**2025-12-08, "Exploring Recursive AI Architectures."** The owner asked for "a novel but simple architecture" for reasoning, testable "with minimal means". Gemini proposed a latent token looped through one block, with a halting head: `if halt_score > 0.9: break` [G:5905 L561] and "Introduce a Ponder Cost penalty" over a step count [G:5905 L1535]. Its plan used "Synthetic Datasets that provide infinite, controllable samples" [G:5905 L1429] and a three-phase loop curriculum ("Curriculum Phase 1 (Fixed Loop, Teacher Forcing)" [G:5905 L1471]). Owner: "This doesn't seem to cater to multi-domain text generation tasks." Gemini pivoted to a "Latent Recurrent Adapter" on GPT-2. This chat came one day before the ChatGPT "Training data mix" pseudocode that the ChatGPT history links to F01/F03. No transfer from this chat to PR #1 is shown.

**2025-12-13, "Liberally Licensed Fiction Corpora."** Two Deep Research reports on fiction corpora. No effect on `data.py`.

**2025-12-17, "Grokking du Langage" (13:55).** The owner asked whether grokking had been tried on language and for testable hypotheses. The report describes grokking as generalization "après avoir rapidement mémorisé les données d'entraînement" [G:46f1 L59], recommends full batch, weight decay sweeps and no early stopping, and proposes three experiments. It contains no task DAG (section 5).

**2025-12-17, "Tokenizer pour Transformers Maths Langage" (16:05).** The report called Llemma "Verdict : Le candidat le plus fort." [G:ba6c L529], but its final recommendation for a 50–200M model was a custom "Vocabulaire cible de 16 384" [G:ba6c L549]. Commit e9de18b "Default to llemma tokenizer" landed the same day.

**2025-12-20, "Implement Algorithmic Curriculum V1" (first turn).** Owner: per-module grad norms "still show intermittent NaNs". Gemini answered with a "Robust Safe-Logging Snippet" that replaces a NaN with `norms[name] = 0.0 # Or handle appropriately` [G:8666 L96]. Pasted verbatim into PR #84.

**2025-12-20, "Fixing Evaluation and Scoring Logic."** The owner pasted a long eval-fix prompt and asked for a shorter one with five tasks. Gemini's version kept "Task 4: Fix Dataset Seeding (Determinism)" with `# Use hash for better seed isolation` [G:45e6 L881, L894] and "Take the LAST number" [G:45e6 L828]. It became the PR #71 prompt word for word (role line changed).

**2025-12-20/21, same curriculum chat.** The owner pasted the per-task curriculum diff (PR #85). Gemini praised the "Thermostat" logic (85% up, 55% down) and found three bugs: the Manager pickling risk, no curriculum checkpoint ("Amnesia"), and an f-string syntax error. All three were pasted into PR #85's follow-up and fixed.

**2025-12-22 and 12-24.** Templates and neural-collapse research chats. No traced effect.

**On or after 2025-12-30, "ReLSM Codebase Review."** With the repo attached, Gemini called the ACT code "textbook correctness" [G:216c L34], the thought loop "a clean implementation" [G:216c L36], and the SSM scan "sound for a diagonal SSM" [G:216c L118]. It flagged Grokfast as a divergence risk [G:216c L87] and gave "Rating: 9/10".

**2026-01-11, "Fixing Pickle Error on Windows."** The Jan 11 DAG command (accum 12, workers 4) crashed with "Can't pickle local object 'train.<locals>.<lambda>'". Gemini fixed it, then on "Look for similar bugs." found that worker snapshots were stale: "The curriculum effectively freezes at step 0." [G:ce77 L279]. It told the owner "You must use a shared dictionary (via multiprocessing.Manager)" [G:ce77 L283]. It confirmed that `ListProxy` has no `clear()` (true on the venv's Python 3.11.14). For the next crash (paging file) it recommended `--num_workers 0` [G:ce77 L529]. This text went into PR #130.

**2026-01-11, "Fix Duplicate Metric Logging Error."** Every IID eval crashed with "got multiple values for keyword argument 'eval/numeric/mae'". The pasted log also shows successor targets of 30–33 tokens stopped at `pred_len=32 stop_reason=max_new_tokens` [G:bfb1 L12–20]. Gemini did not comment on that. It said "Yes, it is only the MAE metrics" [G:bfb1 L319], wrote the agent prompt used in PR #131, and approved the diff as "maintaining identical logging behavior" [G:bfb1 L933].

**On or after 2026-01-11, "Training vs. Evaluation Difficulty Mismatch."** The owner pasted a DAG run at step 11,000 (IID 19.0%, copy/reverse/compare at difficulty 0.5–0.55, the rest at 0.2). The surviving DAG-v12 checkpoint is dated Dec 29, 2025, so this is a later run whose artifacts are not in `checkpoints/`. Gemini: the IID grid uses the final phase while training is easier. It recommended "an evaluation grid that uses the current curriculum parameters" [G:f5bb L505].

**2026-02 and later.** Concept-model and distillation research. No ReLSM content.

---

## 3. Advice that led to defects

| Advice (quote, chat, date) | What it caused | CR defect / issue |
|---|---|---|
| "Synthetic Datasets that provide infinite, controllable samples" (Exploring Recursive AI Architectures, Dec 8) [G:5905 L1429]; "designed to support "infinite" data to force generalization (grokking)" (Codebase Review, ≥Dec 30) [G:216c L93] | Earliest "infinite data" framing; reinforced after the code existed. No transfer to code shown (inference) | **A3** / #140; CR §4.1 |
| `if halt_score > 0.9: break` and a ponder cost on the step count (Exploring Recursive AI Architectures, Dec 8) [G:5905 L561, L1535] | Same failure class as ReLSM's ACT: a hard break gives the halt head no gradient. No transfer shown. Gemini's latent loop wrote the thought back to the last position only, so it is **not** the origin of the F01 broadcast leak | **F03**, **B11** / #132 |
| "Task 4: Fix Dataset Seeding (Determinism)" … `# Use hash for better seed isolation` (Fixing Evaluation and Scoring Logic, Dec 20) [G:45e6 L881, L894] | PR #71 verbatim; `utils.py:968` still has it. Gemini kept a line already in the owner's draft and called it a determinism fix | **F12** / #160 |
| "Take the LAST number (more likely to be the final answer)" and `text.split("\n")[0]` (same chat) [G:45e6 L828, L811] | PR #71. Today `extract_integer_token` returns `matches[-1]` (`utils.py:130`), so "1,234" → "234"; `_base_normalize` keeps only line 1 (`utils.py:777`) while MAE reads the whole output. The link from #71 to today's function is inference | **F13**, **B10** / #165 |
| `norms[name] = 0.0 # Or handle appropriately` (Implement Algorithmic Curriculum V1, Dec 20) [G:8666 L96] | PR #84 verbatim. `module_grad_weight_norm` prints Gemini's warning and sets a NaN norm to 0.0 (`utils.py:333`) | **F26** / #149 |
| "Verdict : Le candidat le plus fort." for Llemma (Tokenizer, Dec 17) [G:ba6c L529] | Llemma's 32k SentencePiece vocabulary, adopted the same day. The report never mentions the SentencePiece leading-space prefix or `:` + `=` merging. Contributed, not caused (inference) | **A6** / #156, **B9** / #178 |
| "Shared Memory (TaskCurriculumState): Uses multiprocessing.Manager. This is crucial" (Dec 20) [G:8666 L2224]; "You must use a shared dictionary (via multiprocessing.Manager)" (Pickle chat, Jan 11) [G:ce77 L283] | PR #130 Manager proxies, read for every example. No warning about per-call IPC cost | #192 (Tier C); new problem N1 (#197) |
| "The implementation of ACT (Exp3) is textbook correctness" [G:216c L34]; "This is a clean implementation" (thought loop) [G:216c L36]; "The logic here is sound for a diagonal SSM" [G:216c L118]; "Rating: 9/10" (Codebase Review, ≥Dec 30) | Cleared the variants, like ChatGPT's Dec 16 "No obvious runtime bugs" | **F03**/#132, **F01**/#135, **A7**/**F16**/#133, **F02**/#134 |
| "correctly separated to ensure the model never sees long sequences during training" (same review) [G:216c L99] | Wrong: addition "OOD" overlaps training; IID needs untrained positions | **F22**/#155, **B6** |
| "Yes, it is only the MAE metrics" [G:bfb1 L319]; "maintaining identical logging behavior" [G:bfb1 L933] (Jan 11) | The reviewed diff shows `"train_eval/acc/token"` twice in one dict; it survived. The fix also changed MAE logging (N2) | **B14** / #167; N2 (#198) |
| "The "Difficulty" Label in Logs is a Default" … "For addition, this means 3–6 digits." (Training vs. Evaluation, ≥Jan 11) [G:f5bb L471, L485] | Half wrong: difficulty 0.5 is passed to the generators (`eval_hub.py:591`) and caps IID addition at 3–4 digits. The main diagnosis was right (section 4) | **B2**/**B3** / #162 |

---

## 4. Correct calls that were not acted on

1. **Jan 11+, IID grid vs training distribution.** Gemini: "your "IID" evaluation measures performance on the hardest data, while your model is still training on easy/medium data", and the fix is a current-difficulty grid [G:f5bb L483, L505]. This is CR **B2** and fix 15. No commit followed; #162 is open.
2. **≥Dec 30, Grokfast.** "If the model diverges, this is a likely culprit." [G:216c L87]. Right target, wrong cause (it blamed hyperparameter sensitivity, not accumulation). CR **A1** / #137.
3. **Dec 17, grokking recipe.** The language-grokking report: "Taille du Batch	Full Batch ou Large" [G:46f1 L518], "Le bruit du SGD (petits batchs) peut interférer avec la dynamique fine du grokking." [L520], "L'arrêt précoce (Early Stopping) est l'ennemi du grokking." [L528], a weight-decay sweep up to 2.0, and a held-out test set of unseen verbs and pseudo-verbs [L221–223]. That is CR §6 experiment 3. DAG-v12 instead used moving procedural data and 6,458 updates. The report also contradicts the "infinite data" framing: it describes grokking as coming after memorizing a finite training set [L59].
4. **Dec 17, vocabulary size.** "Pour un modèle ~50M - 200M : Vocabulaire cible de 16 384" [G:ba6c L549] and "Ne pas utiliser le tokenizer pré-entraîné de GPT-2 ou Llama!" [L591]. The owner took the 32k Llemma vocabulary, so embeddings are about 39% of nano's parameters (CR **B7**; #186, #170).
5. **Dec 20, AMP and NaN norms.** Gemini said that with GradScaler, "logging the raw .grad will show inf or NaN" when a step is skipped [G:8666 L111]. Its own snippet then hid NaN as 0.0. Logging NaN as NaN is the #149 acceptance test.
6. **Dec 24, critique of the staged curriculum (agent report, section 6).** "les curricula "Easy-to-Hard" (du facile au difficile) peuvent parfois empêcher le grokking" [R:Étagé L46]. Not acted on.

Acted on: the Manager pickling and "Amnesia" warnings (PR #85, later #98/#111), the lambda and stale-snapshot fixes (PR #130), and `--num_workers 0` (DAG-v12 ran with workers 0, CR §4.1).

---

## 5. The DAG curriculum in the Gemini chats

**Verdict: "Grokking du Langage : Plan de Recherche" does not contain the DAG hierarchy, and nothing in it could reasonably be read as that hierarchy.** I read the chat and its attached report in full, in French and English. It names none of the ReLSM tasks except as literature (modular arithmetic, Dyck), states no prerequisite between tasks, and never uses "Variety / Atomic / Composition" or "Variété / Atomique / Composition" as phases. The closest passages:

- "Les Transformers montrent un "grokking structurel" où la compréhension de la hiérarchie émerge tardivement." [G:46f1 L170]. This is about bracket nesting in Dyck, not task order.
- "L'arithmétique modulaire : Addition, multiplication, division modulaire." [L168]. A list of tasks where grokking was seen.
- "Composition du Train Set :" [L213]. This means dataset make-up.
- "Score de Compositionnalité" [L395]. A metric for the multi-agent experiment.
- Three training phases, "Phase "Liquide"", "Phase de Recuit", "Phase "Cristalline"" [L432–447]. A temperature schedule, not a task order.
- "notamment sur des tâches hiérarchiques" [L1485]. A reasoning-trace line about Dyck-style tasks.

**Other Gemini chats:**
- "Exploring Recursive AI Architectures" (Dec 8) has a three-phase curriculum, but over loop counts: "Curriculum Phase 1 (Fixed Loop, Teacher Forcing)", "Phase 2 (Adaptive Loop)", "Phase 3 (Generalization Test)" [G:5905 L1471–1573].
- "Tokenizer" (Dec 17) is the only Gemini chat that mentions carries: "propagation des retenues arithmétiques" [G:ba6c L478]. No task graph.
- "Implement Algorithmic Curriculum V1" (Dec 20) is a per-task difficulty controller (85%/55%). No prerequisites.
- "ReLSM Codebase Review" (≥Dec 30) describes the DAG after it existed: "hard tasks (e.g., multiplication) only unlock after prerequisites (e.g., addition, reverse) are mastered" [G:216c L75].
- "Fixing Pickle Error on Windows" (Jan 11) covers only the plumbing of DAG snapshots.
- "AI/ML Text Decomposition Concepts" (Feb 2026) uses "Atomic Fact Decomposition", unrelated and after PR #114.
- The other files contain nothing on task order.

The wording of PR #114 ("Variety, Atomic, Composition", "Explicitly stated in paper") appears in no Gemini or ChatGPT export. It comes from the agent reports in section 6.

---

## 6. Where the DAG task hierarchy came from

**Sources.** "Nash-SEC" is ReLSM under another name (owner). The reports are in `ReLSM-transcripts/agent_reports/`. The French PDFs without a chat are **probably** Gemini Deep Research exports: "Grokking du Langage _ Recherche et Hypothèses.pdf" is the PDF of Gemini chat 46f11897, and the others share its format. Their chats are no longer in the account list.

| Report | Date | Made by | Role |
|---|---|---|---|
| compass_artifact_wf-92da85b9… ("Curriculum Learning for Algorithmic Tasks") | Dec 20 | Claude Research | `self.advance_threshold = 0.85` / `self.retreat_threshold = 0.55` [R:compass L14–15] match `curriculum.py:116–117` (added Dec 20, bb081bc). "Spiral curricula with jittered difficulty" [L248] matches `curriculum_jitter` (same commit). Its Grokfast class [L209–234] postdates ReLSM's Grokfast (7b96230, Dec 17), so it is not the origin of #137 |
| Perplexity thread "(1) Rechercher des détails d'implémentation pratiques pour l'algorithme Nash-MTL…" (14 turns) | Dec 23–24 | Perplexity Deep Research, project "Multi-Task Phased LLM" | **Where the hierarchy started** (section 6.0). The owner asks whether to phase the tasks, then where copy, reverse, parity, chain and dyck go. All Nash-SEC reports were generated in this thread |
| Nash-SEC: Complete Design & Implementation Report | Dec 23 | Perplexity (cites `ppl-ai-file-upload.s3.amazonaws.com`) | First task graph: Copy→Reverse; Chain, Parity→Addition; 80/20 replay; copy > 0.99 gate; 10% floor |
| Curriculum Étagé : Prometteur ou Risqué ? | Dec 24 | probably Gemini Deep Research | Restates and critiques the Nash-SEC graph the day before PR #108 |
| Apprentissage Algorithmique et Mathématique Structuré | Dec 26 | probably Gemini Deep Research | Variété / Atomique / Compositionnel phases with successor and compare added. The "paper" of PR #114 (Dec 27) |

The Dec 26 report cites "Curriculum d'Apprentissage Algorithmique Pondéré.pdf" for nearly every claim [R:Structuré L145]. That file has no copy, reverse, parity, chain or successor task; its only graph example is "Addition → Multiplication → Exponentiation" [R:Pondéré L102]. The hierarchy matches the uncited Nash-SEC report instead.

### 6.0 The conversation that started it (Perplexity, Dec 23–24)

Found on 2026-10-27 after this document's first draft. The thread is saved as `agent_reports/perplexity_nash_sec_thread.md` (14 turns; timestamps 2025-12-24 03:09 UTC, the evening of Dec 23 in Québec). **[P:tN]** = turn N.

- **[P:t2]** Owner: "Doit-on décomposer ou introduire en phases les tâches? Si oui, comment?" Perplexity: phasing is "absolument impératives". It proposes a three-phase hierarchy:
  - "Phase 1 : Ancrage du Manifold", with tasks "Auto-encodage, Fonction Successeur ($n 	o n+1$), Comparaison ($a > b ?$)";
  - "Phase 2 : Opérateurs Atomiques" (addition, subtraction, simple modulo);
  - "Phase 3 : Opérateurs Compositionnels" (multiplication, …).

  It recommends a replay "Buffer de Replay (environ 10-20% du batch)" and hard gating: "Si le modèle échoue à la tâche $N$ (prerequis), la probabilité d'échantillonner la tâche $N+1$ (dépendante) doit être forcée à zéro".
- **[P:t4]** Owner: "Où se situerait ces autres tâches? "reverse" "copy" "parity" "chain" "dyck"". These are ReLSM's own task names. Perplexity's table:
  - Copy: "Phase I (Début)";
  - Reverse: "Phase I (Fin)";
  - Chain: "Phase II (Début)", "Précurseur direct de la Retenue (Carry)";
  - Parity: "Phase II (Milieu)", "l'opérateur atomique de l'addition binaire (Sum = XOR)";
  - Dyck: "Phase III".

  It asks for "99.9% de précision" on copy and reverse. Its graph: `A[Copy] --> B[Reverse]`, then `D[Chain] -- Prépare la mémoire --> E[Addition]`, `D --> F[Parity]`, `F -- Opérateur atomique --> E`.
- **[P:t5]–[P:t13]** produce the English "Complete Design & Implementation Report" and the NLP and Triadic versions.

So the phase vocabulary, successor and comparison as Phase 1 tasks, copy → reverse, chain and parity → addition, the binary-addition argument for parity, the 99.9% gate, the replay share and zero probability for locked tasks were all chosen by Perplexity in answer to these two questions. The later reports (Dec 24 and Dec 26) rework this answer.

### 6.1 PR #108 (Dec 25): each element traced

"Not in any report" means the value or edge appears in no report dated before the PR, so it was added in the prompt. The Nash-SEC prerequisite column is scrambled in both text extractions. The best column reading is Copy: None, Reverse: Copy, Chain: Reverse, Parity: Copy, Addition: Chain, Parity, Dyck: Addition, Multiplication: Addition. Its graph differs: `D[Chain] --> E[Parity]` [R:Nash-SEC L147].

| PR #108 element | Source sentence | Status |
|---|---|---|
| Roots copy, reverse | "Phase I: Manifold" `A[Copy] --> B[Reverse]` [R:Nash-SEC L143] | from Nash-SEC |
| Root successor | "Phase 1 : Ancrage du Manifold … Fonction Successeur ($n 	o n+1$), Comparaison" [P:t2]. Later reports move successor to Phase II ("Phase II (Fonction Successeur)" [R:Équité L116]) | **from the Perplexity thread** |
| chain ← copy | thread: Copy → Reverse → gate → Chain [P:t4]; report table reading: Chain ← Reverse | **approximation** (chain follows reverse, not copy) |
| parity ← copy, reverse | table reading: Parity ← Copy; graph: Chain → Parity. Reverse edge absent | partly |
| addition ← chain, parity | "Composition of Chain (carry) + Parity (sum bit)." [R:Nash-SEC L181] | from Nash-SEC |
| multiplication ← addition, chain | addition: table reading and "Repeated addition." [L189]; chain: none | addition only |
| mod_add ← addition, parity | the thread puts "Modulo simple" in Phase 2 next to addition [P:t2]; no parity edge | **addition only, loosely** |
| compare ← addition | the thread puts comparison in Phase 1, before addition [P:t2] | **contradicts the source** |
| dyck ← copy, reverse | table reading: Dyck ← Addition (Phase III) | **not in any report** |
| copy 0.99 | `val_acc['copy'] > 0.99` [R:Nash-SEC L137]; "(ex: >99% sur "Copy" pour débloquer la Phase II)" [R:Étagé L29] | from Nash-SEC |
| reverse 0.98 | report: "99.5% accuracy" for Reverse | **not in any report** |
| chain 0.95, parity 0.95 | Chain success "95% accuracy"; Parity "90% accuracy (grokking expected)" | chain yes; parity changed |
| addition 0.97 (for mult, mod_add, compare) | report: `val_acc['addition'] > 0.95` [L137] | **0.97 not in any report** |
| patience 4 evals | "Phase Unlock Patience … 3 consecutive epochs meeting criteria" [R:Nash-SEC L346–352] | idea yes, value changed |
| ramp 3 evals | closest: "transitions "sigmoïdales" douces entre les phases" [R:Étagé L69] | **value not in any report** |
| replay 0.20 | "Buffer de Replay (environ 10-20% du batch)" [P:t2]; "sample 80% from current phase + 20% from replay buffer." [R:Nash-SEC L293] | from the thread and Nash-SEC |
| backslide replay 0.35 | none | **not in any report** |
| unlock margin 0.01, lock margin 0.03, mastery margin 0.02, frontier 4 evals, floors 0.02 unlocked / 0.03 frontier | none | **not in any report** |
| locked floor 0.02 (PR #109, Dec 25) | none. Nash-SEC's floor is for active tasks: "force minimum 10% probability for any task in its active phase" [L203] | **not in any report** |

### 6.2 PR #114 (Dec 27): each element traced to the Dec 26 report

| PR #114 element | Source sentence in "Apprentissage Algorithmique et Mathématique Structuré" | Status |
|---|---|---|
| Phase names "Variety, Atomic, Composition" | "Formation de la Variété, Opérations Atomiques, Raisonnement Compositionnel" [L9] | from report |
| Roots copy, reverse, compare ("Primitive Comparison") | The thread puts "Comparaison ($a > b ?$)" in Phase 1 [P:t2]. §3 lists "Tâche 1.3 : Subitisation et Comparaison Primitive (Compare - Scalar)" in Phase I [L31]. **But its own summary table puts "4. Compare (Primitive)" in "II. Atomique"** [L119–121] | report contradicts itself; the PR took §3 |
| successor ← copy, reverse, compare ("The bridge to dynamic arithmetic") | "C'est le lien manquant entre la comparaison statique et l'addition dynamique." [L41]. Edges from copy and reverse only follow from phase order | compare edge from report; others inferred |
| parity ← copy, reverse | phase order only | **no stated edge** |
| chain ← successor | none. Successor is described with "logique de retenue élémentaire" [L43] and listed before Chain [L121] | **no stated edge** |
| addition ← chain, parity ("Explicitly stated in paper") | "Elle synthétise la compétence "Parity" (pour calculer le chiffre courant) et "Chain" (pour gérer la retenue venant de droite)." [L57] | from report |
| mod_add ← addition | §4 makes Mod_Add "Tâche 2.5" after addition [L62], but the table puts "9. Mod_Add" in Phase III [L130]. No stated edge | **no stated edge** |
| multiplication ← addition, reverse | "alignement parfait des produits partiels (dépendant de "Reverse") et une sommation robuste (dépendant de "Addition")" [L69–71] | from report |
| dyck ← multiplication | none. The PR says "or simply place at the end of the chain"; the report gives Dyck no prerequisite | **not in report** |
| Thresholds 0.95–0.99 per edge | the report gives one phase gate: "Précision > 99.9% sur Copy/Reverse." [L86] | **values not in report** |
| Chain format "Chain A->B" | "Entrée : Chain 9 -> 1" [L55] | from report |

### 6.3 Where the implemented DAG departs from the sources

Checked against `train.py:2090–2120`, `data.py` and the report texts.

1. **Compare is not the report's primitive.** The report's root is a scalar comparison ("Compare 3 5") plus subitizing [R:Structuré L33–35]. ReLSM's compare uses two integers with up to 4 digits and three labels (`data.py:137–143`), and there is no subitizing task. The successor gate needs compare at 0.98 (`train.py:2101`) while compare's difficulty keeps rising, so the "root" demands 3–4-digit mastery. The report's Phase III "Sort (Compare Complex)" [L132] is not implemented.
2. **The parity → addition edge rests on binary addition.** "Dans l'addition binaire, le calcul du bit de somme est une opération de parité." [R:Structuré L46]. The same report decomposes addition mod 10 [L60]. ReLSM's addition is decimal (`data.py:427` ff., `result = a + b`) and parity is over bit strings (`data.py:404–405`). The edge has no basis in ReLSM's tasks.
3. **dyck ← multiplication is invented** (section 6.2). PR #108's dyck ← copy, reverse was also invented, and its compare ← addition contradicts the thread, which puts comparison in Phase 1.
4. **Four edges are phase-order guesses**, not stated dependencies: successor ← copy/reverse, parity ← copy/reverse, chain ← successor, mod_add ← addition. Chain ← successor makes successor the single choke point for six tasks (CR **B5**).
5. **Gates measure different things.** The report says the curriculum "doit donc être piloté non par la performance de surface, mais par des métriques structurelles "profondes"" (SVD entropy, Hessian trace) [R:Structuré L17], at phase level [L84–87]. ReLSM gates each edge on the EMA of generated exact match at the task's current difficulty (`train.py:2101–2109`), with no SVD or Hessian signal.
6. **Gate levels differ.** Report: 99.9% on Copy/Reverse. ReLSM: 0.99/0.98, minus the 0.01 unlock margin, on other edges 0.95–0.99. These sit above the 0.85/0.55 controller band taken from the compass report (CR **B5**, #157).
7. **Replay differs.** Nash-SEC replays stored past-phase examples from a buffer ("Replay Buffer de capacité fixe ($B=1000$)" [R:Étagé L75]). ReLSM's 0.20 is a sampling split between frontier and other unlocked tasks, drawn fresh (`train.py:136–160`).
8. **Locked tasks are shown, active tasks get less.** Nash-SEC hard-gates: "no task from Phase III (multiplication) is presented until Phase I (number representation) stabilizes" [R:Nash-SEC L40]. ReLSM gives locked tasks a 0.02 total floor (`train.py:3854`, `180–191`). The report's 10% per active task became 0.02 total over unlocked tasks and 0.03 over the frontier (`train.py:166–176`).
9. **No plateau protection.** Both reports warn against stopping on a plateau ("Grokking Abandonment" [R:Structuré L17]). ReLSM pauses all unlocks on any backslide, with no timeout (CR **B5**).
10. **Gradient negotiation (Nash-MTL/FAMO), bandits and the SVD guard**, the core of Nash-SEC, were never implemented. Only the task graph, the copy gate and the 20% replay figure survived.

---

## 7. New problems not covered by #132–#194

Each candidate was checked against the code at HEAD.

**N1 (posted as [#197](https://github.com/jpturcotte/ReLSM/issues/197)). Curriculum snapshot refresh is not atomic for DataLoader workers. Confirmed (code and probe); impact small.**
- The refresh does `clear()` then `update()` on the shared proxies (`train.py:2129–2132`, `2136–2139`).
- Workers read the same proxies key by key for every example: gates `train.py:118`, frontier `train.py:136`, difficulty `train.py:219`, weights `train.py:244, 261`.
- A worker that reads between the two calls sees missing keys. Gates then default to 0.0 (task treated as locked), and difficulty defaults to 0.2. If every gate reads 0.0, `compute_dag_weighting` falls back to unweighted sampling over all tasks (`train.py:120–126`).
- Probe (venv Python 3.11.14, Manager dict, one reader process, 10 `.get` calls per read like `train.py:118`): 477 refreshes, 16,383 reads, **197 partial reads** (unlocked roots seen as locked), 0 all-locked.
- Real runs refresh once per eval, so expect a few wrongly-sampled examples per eval, and only with `num_workers > 0` (`train.py:2064`).
- It comes from PR #130, which followed Gemini's Jan 11 advice. Related to #192, which covers speed only. Fix: publish one immutable snapshot under a single key and read it once per example; this also cuts #192's 11–21 round trips.

**N2 (posted as [#198](https://github.com/jpturcotte/ReLSM/issues/198)). Since the Jan 11 duplicate-key fix, logged MAE is 0.0 when nothing was measured. Confirmed (code).**
- PR #131 / acd48ff removed the explicit MAE keys and kept `_flatten_eval_diagnostics` "as the source of truth".
- The diagnostic MAE is `_mean(...)`, which returns 0.0 for an empty list (`eval_hub.py:296–297`, used at `eval_hub.py:409` and `441`).
- The flattener writes `{prefix}/numeric/mae` and `{prefix}/numeric/mae/{task}` for **every** task in the diagnostics (`train.py:971`, `989`), including copy, reverse, compare, dyck and parity, and numeric tasks where no prediction parsed.
- Before the fix, per-task MAE was skipped when it was `None` (`if mae is not None` in the removed lines, [G:bfb1 L505–513]).
- So `metrics.json` now shows a "perfect" MAE of 0.0 for tasks with no numeric measurement. Gemini called the change "identical logging behavior". `log_task_accuracy` still gets the None-aware value (`train.py:381`), so two MAE records disagree.
- P3; related to #164 and #165, not covered by them.

**Checked and not a defect:**
- WSD holds max LR for 40% and decays over 60%, as documented (`train.py:305`, `319–324`). Gemini's comment [G:216c L132] is a design note.
- No KV cache for the latent and act variants: speed only, and documented in `generate`.
- `ListProxy.clear`: handled by `_set_frontier_snapshot` (`train.py:91–106`).
- The "Amnesia" bug: curriculum and DAG state are saved and restored (`train.py:2460–2514`).
- The pad fallback to token 0 in the owner's Dec 20 draft: pad is set to EOS (`train.py:1955`).
- Sign stripping in numeric token accuracy: that code is gone.
- Gemini's claim that eval difficulty 0.5 is only a label: wrong, but the real behaviour is already #162.

**Not a code defect, but evidence for #189:** the f5bb0569 log shows a DAG run on post-Jan-11 code: step 11,000, IID 19.0%, "New best! Saved checkpoint." Its output folder is not in `checkpoints/`, and no review document mentions it.

---

## 8. Existing issues that deserve a comment

Posted on 2026-10-27 as comments on #142, #149, #157, #160, #165, #167, #186, #189 and #192, and summarized on #182. The rest of this list is provenance only and was not posted separately.

| Issue | Comment |
|---|---|
| #160 | The `hash((seed, task, i))` line reached PR #71 through a Gemini-condensed prompt (chat 45e61603, ~2025-12-20) that labelled it "Fix Dataset Seeding (Determinism)"; the owner's own draft already contained the line. |
| #165 | The "take the LAST number" rule and the first-line-only `_base_normalize` came from the same Gemini prompt pasted verbatim into PR #71. |
| #149 | The NaN→0.0 replacement in `module_grad_weight_norm`, including its warning text, is Gemini's "Robust Safe-Logging Snippet" pasted verbatim into PR #84 (2025-12-20). |
| #192 | PR #130's per-example Manager proxies followed Gemini's Jan 11 advice; the same `clear()`-then-`update()` refresh also lets workers read partial snapshots (197 of 16,383 reads in a probe). |
| #162 | Gemini diagnosed this on a post-Jan-11 DAG log and proposed a current-difficulty grid; its claim that difficulty 0.5 is only a label is wrong (it is passed to the generators). |
| #167 | Gemini reviewed the Jan 11 diff that shows both `train_eval/acc/token` entries and answered "it is only the MAE metrics". |
| #142 | The Jan 11 chat log shows successor targets of 30–33 tokens stopping at `max_new_tokens` 32, with no comment from Gemini or the owner. |
| #132 | Gemini's Dec 8 design already used a hard `halt_score > 0.9` break and a step-count ponder cost, and its repo review called ReLSM's ACT "textbook correctness". |
| #135 | A Gemini repo review (≥2025-12-30) called `_apply_thought_loop` "a clean implementation"; Gemini's own Dec 8 loop wrote only to the last position and is not the origin of the leak. |
| #133 / #134 | The same review said of `ssm_scan_jit`: "The logic here is sound for a diagonal SSM." |
| #137 | Gemini's repo review flagged Grokfast as "a likely culprit" for divergence but blamed hyperparameter sensitivity, not accumulation. |
| #140 | Gemini framed procedural data as "infinite" (Dec 8, and the repo review), while its own Dec 17 grokking report describes grokking after memorizing a finite set. |
| #186 | Gemini's Dec 17 tokenizer report recommended a 16,384-token vocabulary for 50–200M models; the 32k Llemma vocabulary adopted that day puts about 39% of nano's parameters in embeddings. |
| #157 | The DAG graph, the copy 0.99 gate and the 20% replay come from the Perplexity Nash-SEC report (Dec 23); reverse 0.98, the 0.97 thresholds, ramp, margins, 0.35 backslide replay and the locked floor appear in no report, and the 0.85/0.55 band comes from the Dec 20 Claude Research report. |
| #189 | A post-Jan-11 DAG run (step 11,000, IID 19.0%, saved a best checkpoint) appears in Gemini chat f5bb0569 but its artifacts are not in `checkpoints/`. |
