---
slug: finite-set-grokking-regime-missing
title: The finite-set grokking regime cannot be run from the CLI
issue: 214
severity: P2
labels: [enhancement, P2-important, area: training, reproducibility, audit-2026-10-27]
nf: [NF-13]
found_by: [lexical, claude-a2, claude-a1, gemini-a, review-files]
status: Demonstrated
related_issues: [#136, #186, #162, #140, #170, #169, #189]
---

## Summary

Grokking as reported in the literature (Power et al. 2022, Nanda et al. 2023) and in the owner's own Dec 17 research report needs five things: a finite training table, a disjoint held-out test set drawn from the same distribution, a fixed task space small enough to memorise (one modulus), a small model, and long training at constant learning rate with large or full batches. At HEAD none of the five is available from the command line. `--fixed_data` draws 5,000 examples with replacement from a space of 10^5 to 10^8 strings per task and has no complement set; `mod_add` draws a new modulus per example and no flag fixes it; the smallest preset is nano (41.6M, 6 layers); batching is a token budget with no full-batch mode; the default LR schedule decays over the last 60 % of the run. The repo's conclusion that "every grokking attempt failed" was reached without the regime ever being runnable. This is an enhancement because the missing pieces are features, not broken code; it is P2 because the project's stated goal depends on them.

## What happens

1. No same-distribution held-out split. `FixedAlgorithmicDataset` ([data.py:1264-1274](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1264-L1274)) draws `fixed_data_size` examples with replacement and builds no complement. The in-run "IID" eval ([train.py:3109-3119](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3109-L3119) via [eval/ood_grid.py:19-33](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval/ood_grid.py#L19-L33)) uses each task's final-phase parameters while the fixed set is the difficulty-0.5 phase ([data.py:934-942](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L934-L942)). Audit probe `lexical/fixed_data_split_probe.py`: the 5,000 fixed examples are 4,860 distinct texts; exact-text overlap between the 200-per-task IID eval and training is 0/200 for 8 of 10 tasks (parity 5/198, dyck 6/196). The zero is a distribution shift, not a split: successor trains on 8-16 digits and is evaluated on 20-32. The train-eval loader in `fixed` mode is the training set itself ([train.py:2262-2277](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2262-L2277)).
2. No fixed modulus. `mod_add` samples `operand_digits` from the difficulty bucket and then a modulus with that many digits ([data.py:350-361](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L350-L361)); the phase table passes `mod_range` (17-50 up to 500-997) through [data.py:958-959](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L958-L959), so the `mod=97` default never applies. `train.py` has no `--mod` flag. The fixed set at seed 42 holds 479 distinct `(a, b)` pairs under about 300 different moduli (lexical probe). For a single modulus p = 97, 2,000 draws cover about 1,803 of 9,409 pairs and an independently drawn test pair is a training pair with probability about 0.19 (claude-a2 arithmetic). The Dec 19 grid was two-thirds out-of-modulus (`ood_mod_1000`, `ood_mod_10000`).
3. No preset below nano. [model.py:1036](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L1036) is the smallest entry: d 512, 6 layers, 41.6M parameters, dropout 0.1 ([model.py:115](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L115)). The Dec 19 chats recommended `--model_size micro`, which never existed. Audit probe `gemini-a/probe_recipe_params_and_eos_mask.py` (A) builds the recipe size (d 128, 2 layers, 4 heads) on the 32,000-row Llemma vocabulary: 4,620,928 parameters, of which 4,096,000 are the tied embedding (88.6 %; 80.0 % at a 16,384 vocabulary). The non-embedding body is 524,928 parameters.
4. No full-batch mode. `--alg_batch_size` is a token budget capped at 4x examples ([data.py:1028-1029](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1028-L1029)); the fixed loader takes it as a literal batch ([train.py:2192-2199](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2192-L2199)) but crashes at the first eval (#136). The default is `--grad_accum_steps 4` ([train.py:3987](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3987)).
5. Schedule and stopping. `--lr_schedule` defaults to `wsd`, which decays from 40 % of the run ([train.py:319-324](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L319-L324), [3952-3957](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3952-L3957)); `constant` exists ([train.py:333](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L333)) but no recorded run used it. No run reached 10^5 updates (DAG-v12: 6,458). Weight decay excludes the embedding and head with no switch (#186).

The recipe the repo was given, and the standard one:

| Lever | Dec 17 Gemini report (gemini-a G-A-1) | Power 2022 / Nanda 2023 (claude-a2 §4) | HEAD |
|---|---|---|---|
| Data | finite set, memorised first [L59]; 20 % "famine" of the rule [L217] | full p^2 table, p = 97 / 113; train fraction 30-80 % / 30 % | 5,000 draws with replacement, per-example modulus |
| Test | held-out unseen items of the same rule [L221-223] | the complement of the train split | IID grid at a harder phase; 0/200 overlap by shift |
| Model | d 128, 2 layers, 4 heads [L491-513] | 2 layers d 128 4 heads (about 4e5 params) / 1 layer d 128 | nano 41.6M; recipe body would be 11 % of its own parameters |
| Tokens | character or rule-aligned tokenizer [L229] | vocabulary p + 2 | Llemma 32,016 |
| Optimiser | AdamW, WD sweep 0.1-2.0 [L517] | AdamW lr 1e-3, betas (0.9, 0.98), wd 1 on all parameters | wd 1.0 default, embeddings exempt (#186), betas (0.9, 0.95) |
| LR | warmup then constant [L259] | constant after 10-step warmup | `wsd`, decays over the last 60 % |
| Batch | full batch or large [L518] | 512 / full batch | token-budget micro-batches, accum 4 |
| Length | no early stop, at least 1e5 steps [L523-528] | 1e5 / 4e4 steps | thousands of updates; eval-selected best checkpoint |
| Regularisation | - | dropout 0 | dropout 0.1 |

Dec 19 `exp0_grokking_mod_add_v4`, the closest attempt, differed from the standard recipe on every row of this table and also trained a 2-token-ahead objective (historical fact 1 in the merged findings; fixed by PR #80 on Dec 20).

## Why it matters for the experiments

- "All of my grokking experiments failed" (owner, Oct 24-26) is the project's standing conclusion. It was reached with the regime unrunnable: no run had a held-out same-distribution set to grok on, and the Dec 17-19 runs that came closest were on a broken objective.
- The regime is the cheapest experiment the repo could run. review-files §4.2 scales from DAG-v12's throughput: at the classic size (p about 100, a 1-2 layer d 128 model) 10^5 full-batch updates take 1-2 hours per seed on the RTX 4060; with nano it is 8-14 hours per seed. The control that would tell the DAG and the variants apart from the data pipeline does not exist.
- Every downstream comparison (variants in #132-#135, the DAG in #157, weight decay in #186) is interpreted against "grokking failed", so this gap decides how the whole result table must be read.

## How it got here

- The Dec 17 Gemini Deep Research report (`46f11897_Grokking_du_Langage_Plan_de_Recherche.md`) prescribed the finite-set recipe in full. The same day the tokenizer moved to Llemma (commit `e9de18b`), which makes the recipe-sized body impossible (point 3).
- `--fixed_data` was built on Dec 17-18 as "fixed dataset for grokking" ([train.py:3892-3900](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3892-L3900)) with a sampled set, not an enumerated split; its PR was not traced by this audit.
- The Dec 19 Claude chats (`2025-12-19_23d47afd_Grokking_experiment_training_analysis.md`, `2025-12-19_090b121f_Debugging_failed_modular_addition_grokking.md`) recommended "a 2-4 layer transformer with hidden dim 128-256", "35-50 % of all examples" of a fixed p, and a 500-example quick recipe; none was built. claude-a1 (A3) confirms the per-example modulus was already in the Dec 17 code (`7b96230:data.py` 111-121).
- The 135 PR titles contain no PR for a constant-LR run, a finite-set split, a weight-decay sweep, a small preset or a digit tokenizer (gemini-a). Not traced to a single PR: this is an omission across the Dec 17-Jan 11 build.

## How we verified it, step by step

1. lexical ran audit probe `lexical/fixed_data_split_probe.py` (no tokenizer; replays the fixed loop and the IID grid through `utils.build_dataset`): distinct texts 4,860 of 5,000; 479 distinct mod_add pairs; IID-vs-train overlap per task as above.
2. gemini-a ran audit probe `gemini-a/probe_recipe_params_and_eos_mask.py` (A): parameter counts for d 128 / L 2 / H 4 at vocab 32,000 and 16,384.
3. code-data ran `code-data/probe_fixed.py`: 5.6 % duplicates and 5.6 % cross-seed overlap in the fixed set (details in #215).
4. claude-a2 computed the expected 19 % test-pair overlap for one modulus and read the Dec 19 grid (`94f9f3b:eval/ood_grid.py:184-188`); claude-a1 read `mod_add` at HEAD and at `7b96230`.
5. cli-commands ran R10 (`--fixed_data` crashes at the first eval, #136) and R11 (`--fixed_data --use_task_curriculum` runs with a cosmetic curriculum).
6. gemini-a read the Dec 17 report line by line and tabulated the recipe against HEAD defaults (`train.py:3785, 3952-3957, 3987, 319-335`). review-files worked the feasibility numbers in §4.2. No coordinator re-check of this row.

## Suggested fix

A minimal feature list, in dependency order. Each item is small; together they make one documented command.

1. `FixedAlgorithmicDataset(split="train" | "test", train_fraction=f)`: one enumerated pool per task, a seeded permutation, disjoint halves; `--fixed_data_split_frac` on the CLI; the train-eval loader for `--fixed_data` runs scores the test split; assert overlap 0 at startup. (Lexical L4, review-files §4.2 item 1.)
2. `--mod_add_modulus p`: fixed p, pool = all p^2 pairs, one prompt template or the template as a split key. (claude-a1 A3, claude-a2 A2-3.)
3. A `grok` preset in `create_model`: 1-2 layers, d 128, 4 heads, dropout 0. (claude-a2 A2-6, gemini-a Tableau 2.)
4. A digit-and-symbol tokenizer option (about 100 ids) for algorithmic-only runs, so the body is not 11 % of the model. (gemini-a G-A-2.) Until then, #186's switch to decay the embedding is a prerequisite.
5. A full-batch map-style loader for the fixed set: `--alg_batch_size` equal to the train split with `--grad_accum_steps 1`. Needs #136 (the `batch_size=None` collator bug) fixed first.
6. `--lr_schedule constant` plus `--adam_beta2`, and a stopping rule in optimizer steps (`--max_opt_steps`), which needs the `opt_step` counter from #148/#189.
7. Log train-split and test-split answer-token exact match every eval (teacher-forced, no generation; claude-a2 avenue 3). Keep the IID grid as a separate "shifted" number (#162).
8. One README command and a run-log entry (#189) saying no run before it used this regime.

Depends on #136, #186, #148/#189 as noted, and on #140/#143 (RNG resets) for data order to be part of the dynamics. Overlaps #215 only on the split.

## Acceptance test

CPU, under a minute: `python train.py --fixed_data --alg_tasks mod_add --mod_add_modulus 97 --fixed_data_split_frac 0.5 --model_size grok --lr_schedule constant --grad_accum_steps 1 ...` for 20 steps must (a) print train and test split sizes that sum to 9,409 with zero overlap (a pure-Python check on the two example lists), (b) take one optimizer step per full pass over the train split, (c) report a test-split exact match at the eval, and (d) build a model under 1M parameters with the digit tokenizer. Today step (a) is impossible: there is no split, no modulus flag and no `grok` preset.

## Related

- #136: `--fixed_data` crashes at the first eval; blocks every item above.
- #186: embeddings and head excluded from weight decay with no switch; the recipe decays all parameters.
- #162: the IID grid is not the training distribution; for `--fixed_data` it is harder for 8 of 10 tasks.
- #140: reseeding replays data; a finite-set run wants exact replay by design.
- #170: README has no preset below nano and no grokking preset.
- #169: README's "Phase 1: Algorithmic Grokking" with "infinite procedural generation" contradicts the recipe.
- #189: the dated invalidation list (2-token-ahead objective before PR #80).
- #215: what the existing fixed path does with its flags.
- #213: why "full batch" needs a new loader mode.

## Sources

- Audit report lexical, finding L4; audit probe `lexical/fixed_data_split_probe.py`
- Audit report claude-a2, findings A2-3, A2-6, §4 avenue 1 (standard-recipe comparison); chats `2025-12-19_23d47afd_Grokking_experiment_training_analysis.md`, `2025-12-19_090b121f_Debugging_failed_modular_addition_grokking.md`
- Audit report claude-a1, finding A3
- Audit report gemini-a, findings G-A-1, G-A-2, File 3 table; audit probe `gemini-a/probe_recipe_params_and_eos_mask.py`; chat `46f11897_Grokking_du_Langage_Plan_de_Recherche.md`
- Audit report review-files, §4.2 Experiment 3
- Audit report code-data, finding D8; audit report cli-commands, runs R10 and R11
- Merged findings §5, historical fact 1 (2-token-ahead objective)
