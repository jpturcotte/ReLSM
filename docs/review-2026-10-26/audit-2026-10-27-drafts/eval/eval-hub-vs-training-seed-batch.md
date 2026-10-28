---
slug: eval-hub-vs-training-seed-batch
title: eval_hub and the in-training IID eval use different seeds and batch sizes
issue: 227
severity: P3
labels: [bug, P3-minor, area: evaluation, reproducibility, audit-2026-10-27]
nf: [NF-44]
found_by: [cli-commands]
status: Plausible
related_issues: [#160, #194, #187, #143]
---

## Summary

`eval_hub.py` defaults to `--seed 1234` and `--batch_size 32`. The training-time IID eval (`evaluate_algorithmic` in `train.py`) passes `seed=args.seed` (default 42) and a hard-coded `batch_size=8`. Per-condition seeds are derived from the seed, so the two paths draw different example sets. A README post-hoc number from `eval_hub` and the IID curve of the training log are measured on different examples, before #160's per-process hash salting is even considered. The batch size changes only speed, not results.

## What happens

- [eval_hub.py:854-855](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L854-L855): `--seed` default 1234, `--batch_size` default 32.
- [train.py:617-633](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L617-L633): `run_algorithmic_suite(model, tokenizer, device_obj, seed=seed, batch_size=8, ...)`; the caller at [train.py:3117](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3117) passes `seed=args.seed`.
- [eval_hub.py:588-589](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L588-L589): `cond_seed = seed + int.from_bytes(sha256(f"{task}:{name}:{seed}")[:4])`, then `build_dataset` seeds each example from `(seed, task, i)` ([utils.py:968](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L968), the salted `hash()` of #160). Different seed, different examples.
- `generate_batch` buckets by exact length with no padding, so the batch size does not change any prediction; it changes the number of `generate` calls (#229).

## Why it matters for the experiments

- DAG-v12's `best_alg_accuracy 0.2925` (80 examples per task) and an `eval_hub` rerun of `best_model.pt` (200 per task, seed 1234) are on different example sets. At these sizes one example is 1.25 points (80) or 0.5 points (200), and the 95 % interval on a 0.3 accuracy is about ±10 points at 80. Differences between the two numbers cannot be attributed to the model.
- The seven Dec-23 result files (seed 1234) and the training-time IID numbers of the same runs are likewise not on the same examples.

## How it got here

- PR [#18](https://github.com/jpturcotte/ReLSM/pull/18) (2025-12-12, "Improve eval metadata", commit `2a20678`) added the `train.py` `evaluate_algorithmic` wrapper with `seed=42` and `batch_size=8`.
- PR [#20](https://github.com/jpturcotte/ReLSM/pull/20) (2025-12-12, "Refactor evaluation into single deterministic hub", commit `7cfedae`) gave the hub `--seed 1234` and `--batch_size 32`. Both are Codex PRs of the same day; no task reconciled the two defaults.

## How we verified it, step by step

1. cli-commands read both call sites and the per-condition seed derivation (C13). Not run: no probe compared the two example sets or ran both paths on one model.
2. The drafter re-read `eval_hub.py:854-855`, `train.py:617-633`, `train.py:3117` and `eval_hub.py:586-591` at HEAD; the lines say what the row claims.
3. A probe that settles it: call `build_dataset("addition", n=200, params=<iid params>, seed=S)` for the two `cond_seed` values derived from 42 and 1234 and compare the prompt sets (expected: disjoint apart from chance); then run `evaluate_algorithmic` and `eval_hub.run_algorithmic_suite` on a tiny CPU model and compare `sampled_examples`.
4. Disagreements: none.

## Suggested fix

- One seed source: `eval_hub.py` should default its seed to the seed recorded in the checkpoint (#187 adds the run arguments) and fall back to 42, the trainer's default; or both should take an explicit `--eval_seed`.
- Replace the hard-coded `batch_size=8` with a flag shared by both paths.
- Record `seed`, `batch_size` and the per-condition `cond_seed` in the result metadata (#194) and in the training log at each IID eval.
- Depends on #160 (process-salted `hash()`) for the sets to be identical across processes at all.

## Acceptance test

CPU, tiny model, after #160 is fixed: run `evaluate_algorithmic(model, tokenizer, "cpu", n_examples=20, max_new_tokens=8, tasks=["addition"], seed=S, ...)` and `run_algorithmic_suite(model, tokenizer, torch.device("cpu"), seed=S, batch_size=32, generation_kwargs=..., limit=20, tasks=["addition"], sample_count_per_task=20)` with the same `S`; assert the sorted prompt lists in `sampled_examples_by_task["addition"]` are identical and the per-task accuracy is equal.

## Related

- #160: eval sets differ per process because of the salted `hash()`.
- #194: eval metadata; add seed and batch size.
- #187: run record; the trainer's seed should be readable by `eval_hub`.
- #143: the training-time eval reseeds the global RNGs.
- #229: what the batch size does affect.

## Sources

- Audit report cli-commands, finding C13 and the `--seed` flag table rows
- Audit report gh-prs-a, PR #18 and PR #20 rows
- Merged findings NF-44

---
*Draft from the Oct 27, 2026 audit. Code links point at commit `774513d`. Draft id: `eval/eval-hub-vs-training-seed-batch`.*
