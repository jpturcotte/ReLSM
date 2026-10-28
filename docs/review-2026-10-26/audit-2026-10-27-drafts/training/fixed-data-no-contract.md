---
slug: fixed-data-no-contract
title: "--fixed_data has no contract: curriculum flags are stored, ignored or misread"
issue: 215
severity: P2
labels: [bug, P2-important, area: training, correctness, reproducibility, audit-2026-10-27]
nf: [NF-12]
found_by: [cli-commands, code-data]
status: Demonstrated
related_issues: [#136, #140, #162, #157]
---

## Summary

`--fixed_data` is the repo's grokking path, and nothing says what it does with the curriculum flags. `--easy_mix_frac` is stored on the dataset and never read. `--difficulty_schedule phased` silently builds the hardest set while the other four schedules all build the difficulty-0.5 set. `--use_task_curriculum --fixed_data`, the only `--fixed_data` form that does not crash (#136), prints a moving curriculum while the model sees the same 5,000 examples throughout. The set has 5.6 % internal duplicates and shares 5.6 % of its texts with the same set built from the next seed, so a "test split" from another seed is not disjoint. Batch-size semantics under `--fixed_data` are a separate defect (NF-50).

## What happens

- `FixedAlgorithmicDataset` stores `easy_mix_frac` at [data.py:1256](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1256) and never reads it; `generate_example` has no such argument ([data.py:907-917](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L907-L917)). Audit probe `cli-commands/flag_semantics.py` F07b: two 300-example sets built with `easy_mix_frac` 0.0 and 0.3 are byte-identical.
- The constructor calls `generate_example(difficulty=0.5, progress=1.0, difficulty_schedule=...)` at [data.py:1264-1274](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1264-L1274). At [data.py:934-942](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L934-L942) `phased` takes `get_difficulty_params(task, 1.0)` (the final phase); every other schedule takes difficulty 0.5. F07: mean answer length 8.16 tokens for `smooth`, `linear`, `fixed` and `warmup_ramp`, 11.07 for `phased` (max 33). Under `phased` successor has 20-32 digits and chain 24-32, which overflow `alg_seq_len 48` (#139).
- `--use_task_curriculum --fixed_data`: the training loader is the `FixedAlgorithmicDataset` built at [train.py:2178-2191](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2178-L2191), which takes no `difficulty_fn`, `weighting_fn` or `weighting_adjust_fn`; those go only to the procedural dataset at [train.py:2201-2218](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2201-L2218). Probe run R11 (tiny model, CPU) finished with rc 0, printed `Fixed data: True`, `Train eval: procedural` and `Curriculum: addition: d=0.20 ema=0.00 ...` at every eval, and wrote `final_model.pt`. The per-task difficulties move only the train-eval sampler and the `eval/curriculum/*` keys.
- Duplicates (audit probe `code-data/probe_fixed.py`, seed 42, 5,000 examples, default `smooth`): 281 duplicate texts (5.6 %; multiplication 195 of 1,142, addition 55 of 889, parity 27 of 470). The same set from seed 43 shares 265 of its 4,748 distinct texts with seed 42 (5.6 %). Nothing in the class dedupes ([data.py:1264-1303](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L1264-L1303) appends unconditionally). Under `phased` the duplicates fall to 19 (0.4 %) because the phase is larger.

## Why it matters for the experiments

- Every `--fixed_data` command the owner ran ("the canonical grokking configuration", #136) had one difficulty and no curriculum, whatever the flags said, and one schedule name silently changes which difficulty.
- A grokking study needs to know what is memorisable: 5.6 % of the set is repeated, and a held-out set drawn from another seed overlaps training by 5.6 %.
- The one `--fixed_data` form that runs produces logs that look like a curriculum run. A reader of its `metrics.json` would see `eval/curriculum/difficulty/*` climb while the training data never changed.

## How it got here

- `--fixed_data` comes from the Dec 17-18 grokking work; all nine Dec 17-18 runs used it (claude-a1). The flag's own PR was not traced by this audit.
- PR [#59](https://github.com/jpturcotte/ReLSM/pull/59) (Dec 19) gave the class a `difficulty` argument after a Claude code review listed "`FixedAlgorithmicDataset` gets no difficulty" (claude-a2, review item 7). PR [#66](https://github.com/jpturcotte/ReLSM/pull/66) and PR [#69](https://github.com/jpturcotte/ReLSM/pull/69) (Dec 20) added `difficulty_schedule` and `easy_mix_frac` for the procedural path and threaded them into this constructor; the second is never read. PR [#105](https://github.com/jpturcotte/ReLSM/pull/105) added `min_difficulty`, which only lifts 0.5 upward. No PR body states what the fixed path should do with these flags.

## How we verified it, step by step

1. cli-commands ran `cli-commands/flag_semantics.py` (F07: answer-length means per schedule; F07b: byte-identical sets at `easy_mix_frac` 0.0 vs 0.3) and the tiny end-to-end run R11 (`--use_task_curriculum --fixed_data`, rc 0, curriculum lines printed, `final_model.pt` written). Logs `r11_fixed_curr.log` lines 26-30 and 70.
2. code-data ran `code-data/probe_fixed.py`: duplicate counts by task, seed-42 vs seed-43 overlap, phase parameters for `smooth` and `phased`.
3. Code read: `train.py:2178-2191` vs `2201-2218` (which loader gets the curriculum functions). No coordinator re-check of this row.

## Suggested fix

- Reject `--easy_mix_frac` and `--difficulty_schedule` under `--fixed_data`, or add `--fixed_data_difficulty <float|phase>` and make the schedule flag irrelevant.
- Dedupe by text at construction and report the distinct count.
- Refuse `--use_task_curriculum --fixed_data`, or pass `difficulty_fn`/`weighting_fn` to a procedural-style fixed set and say in the log that the curriculum applies.
- Batch semantics: see #213 (`--alg_batch_size` is a token budget on the procedural path and a literal batch on this one).
- Depends on #136 (first-eval crash) for any `--fixed_data` run to complete; the held-out split belongs to #214.

## Acceptance test

Pure Python: `FixedAlgorithmicDataset(seed=42, num_examples=5000)` has zero duplicate `text` values; building it with `easy_mix_frac=0.0` and `0.3` either raises or gives different sets; building with `difficulty_schedule="phased"` either raises or logs the phase it used. A 20-step CPU run with `--fixed_data --use_task_curriculum` either exits at argument parsing or logs that the curriculum drives the training set.

## Related

- #136: the crash this path hits at the first eval.
- #140: replay is half the repetition; the fixed set's 5.6 % duplicates are the other half.
- #162: the IID grid is a harder distribution than the fixed set for 8 of 10 tasks.
- #157: the cosmetic curriculum under R11 logs the same keys the DAG issue reads.
- #214: the split, modulus and preset this path would need.
- #203: the same flags on the procedural path.
- #213: `--alg_batch_size` means a different thing on this path.

## Sources

- Audit report cli-commands, findings C3 and C4; audit probe `cli-commands/flag_semantics.py` (F07, F07b); run R11
- Audit report code-data, finding D8; audit probe `code-data/probe_fixed.py`
- Audit report claude-a2, review item 7 (PR #59 provenance); audit report claude-a1, runs A-I
