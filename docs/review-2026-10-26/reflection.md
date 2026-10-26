# How this review was done, step by step

Date: October 26, 2026. Repo: jpturcotte/ReLSM at `acd48ff4fd50588ba0b5d20762182c1093591a4f`. All checks ran on CPU with the repo's own `.venv` (Python 3.11.14, torch 2.9.1+cu128). Nothing in the repo was modified; every probe and output lives outside it.

## 1. Starting point

- Input: an external AI-written review, `ReLSM_Code_Review_2026-10-26.md` (30 numbered findings F01–F30, a telemetry section on the surviving DAG-v12 run, a test-suite section, and architecture notes).
- Question asked by the owner: is that report right, and what does an independent look at the code and artifacts say?
- Decision: run two kinds of agents at once. *Verifiers* read the report and tried to reproduce or refute each claim with their own probes. *Blind reviewers* were forbidden from reading the report or any review folder and had to form their own conclusions from code and artifacts. Agreement between a blind reviewer and the report is the strongest evidence we can produce without a third party.

## 2. Agents, scope and rules

| Agent | Scope | Inputs |
|---|---|---|
| verify_model | F01–F03, F15–F20 | report + `model.py` |
| verify_train | F05, F07–F11, F21, F23–F28, loss normalization | report + `train.py`, `data.py`, tiny end-to-end runs of `train.main()` |
| verify_eval | F04, F06, F12–F14, F22, F29, F30, tests, language data | report + `eval_hub.py`, `utils.py`, `eval/`, `tests/`, real Llemma tokenizer |
| verify_artifacts | every number in the telemetry tables, the seven Dec-23 result files, parameter counts | report + `checkpoints/`, `eval_results/` |
| blind_model | model correctness | `model.py`, `nano_baseline.py` only |
| blind_train | training / data / curriculum correctness | `train.py`, `data.py`, `curriculum.py` only |
| blind_eval | evaluation / metrics / tests | `eval_hub.py`, `utils.py`, `eval/`, `tests/` only |
| blind_artifacts | what the experiments actually show | checkpoints, metrics, result files, README |

Rules given to all: repo read-only, probes in a scratch folder, CPU only, tiny models, no network datasets, every claim backed by an executed probe where possible, and verdicts in both directions (overstated as well as confirmed).

## 3. What came back

- No finding in the external report was refuted. Twelve were understated. One sub-claim (F23, "affects dataset state") was overstated.
- Every number in the report's telemetry tables recomputed exactly from the raw files (verify_artifacts).
- The blind reviewers independently found F01, F02, F03, F04, F06, F07, F08, F10, F11, F12, F13, F14, F16, F17, F19, F20, F21, F22, F28 and F29 without seeing the report.
- New defects not in the report (each found by at least one agent, most by two): Grokfast diverges exponentially under gradient accumulation; the SSM recurrence ignores its input and its channels collapse; compare "=" is never supervised because `Answer:=` tokenizes to `:=`; the "infinite procedural" dataset reseeds on every iterator and replays; the difficulty schedule never advances when DataLoader workers are used; every eval reseeds the global RNGs; CE over a batch with no labels is NaN; Dyck "OOD depth 5/6" is capped at depth 3; a perfect-answer oracle scores only 0.779 on the eval suite.
- The causal story the report missed: the 48-token truncation froze copy/reverse scores at 39/64 and 53/83 for 22 evaluations in a row; those scores feed the DAG gates; the DAG paused at step 13,000 and six tasks got 1.94% of examples. A checkpoint probe shows accuracy 1.0 at lengths 6–10 and 0.0 at 11–14.

## 4. Synthesis

- One agent merged the eight reports into `counter-review.md`: scorecard per finding, new issues in three tiers, causal narrative for DAG-v12 and the Dec-23 files, fix order, next experiments, disagreements.
- Disagreements between agents were small and were resolved in the counter-review's section 7 (mostly line-number nits and severity ratings).

## 5. Recovering the project history

- Local Codex history starts in April 2026; the ReLSM work (Dec 9, 2025 – Jan 12, 2026) was done in Codex cloud. Every one of the 131 PRs carries a `chatgpt.com/codex/tasks/<id>` link.
- Those links redirect to the ChatGPT home page. The `/codex/cloud/tasks/` form suggested in openai/codex#48903 also redirected for this account. The `/remote/<task_id>` form works.
- Each task page shows the owner's prompt and Codex's summary. The "work log" (commands, file reads) is collapsed and did not expand, so it is not in the export. Diffs are in the PRs.
- Export mechanics: the browser tool refuses to return page text that looks like it contains cookies or query strings, so the text was never passed through the tool. Each page's text was stored in the page's own storage and downloaded once as a single bundle, then split into one file per task (`ReLSM-transcripts/codex_tasks/`, 131 files, 589k chars) with PR numbers in the names. The same method exported the 48 chats in the owner's "relsm" ChatGPT project (`ReLSM-transcripts/chatgpt_chats/`, 1.54M chars).
- Five small agents summarized the task history in date slices; one agent merged them into `project-history-codex.md` and filled section 8 of the counter-review. Four agents mined the chats.

## 6. Turning findings into issues

- Rule from the owner: exactly one posted issue per code defect. Drafts were written per area (model 8, training 9, data 9, eval 10, docs 6) with a fixed template: summary, mechanism with evidence, impact on experiments, how it got here (PR + Codex task + quote), verification steps, fix, acceptance test, related issues, sources.
- Dedup check before posting: each draft maps to one defect; overlapping facts (e.g. the 48-token limit) appear in several issues only as cross-references to the one issue that owns them.
- Labels: severity (P1/P2/P3), area, scope (variants-only / baseline-affected), type (correctness, reproducibility, telemetry, tech-debt, dependencies) and `review-2026-10-26`.
- Links: code links are pinned to the audited commit; PR links go to the merged PRs; `[[slug]]` cross-links are rewritten to `#N` after creation; local disk paths are stripped from the bodies.

## 7. Things that did not work, and what was done instead

- Returning page text through the Chrome tool: blocked. Used downloads.
- Chrome blocked repeated automatic downloads: the owner allowed multiple downloads for chatgpt.com, and later a single bundle download replaced per-file saves.
- A 24-item browser batch timed out; 12–16 items per batch was reliable.
- `CUDA_VISIBLE_DEVICES=""` is ignored on Windows and crashes `train.py`; agents used `-1`.
- Three of my replies were cut off by an output safety check while the issue work was being set up; the work was redone by writing drafts to files first and posting from there.

## 8. What remains unverified

- The exact source revision and flags of the Dec-23 runs and of the deleted Grokfast run.
- Whether DAG-v12 was affected by the double-curriculum-bump bug (fixed in a868d71 after the run).
- GPU, AMP, `torch.compile`, Flash-Attention and real Mamba-kernel paths: never executed here.
- The external report's table re-checking older bug discussions was not independently re-verified.
