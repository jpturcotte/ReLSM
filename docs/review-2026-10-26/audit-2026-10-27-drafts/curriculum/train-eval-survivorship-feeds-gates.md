---
slug: train-eval-survivorship-feeds-gates
title: Train-eval drops unscorable examples silently; the curriculum sees survivors
issue: 207
severity: P2
labels: [bug, P2-important, area: curriculum, correctness, audit-2026-10-27]
nf: [NF-16]
found_by: [code-train-a]
status: Demonstrated
related_issues: [#139, #151, #157, #144]
---

## Summary

`evaluate_task_accuracy` skips an example when its task is missing, when every label is -100, when the decoded text has no `Answer:` marker, or when the target is empty. The skipped example has already been counted against `max_samples`, and it vanishes from `total[task]`, `task_counts` and the per-task accuracy that feeds `task_curriculum.update_metrics` and the DAG gates. At `alg_seq_len 48` and difficulty 1.0 the dropped examples are exactly the longest chains, so the curriculum sees chain accuracy on the short ones. #139 covers scoring truncated targets and #151 the NaN loss; neither says the curriculum signal is computed on a biased subset.

## What happens

- [train.py:1091-1105](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1091-L1105): `_iter_eval_examples` increments `count` for every example it yields.
- [train.py:1065-1073](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1065-L1073): `_decode_example_text` returns `None` when no label is valid (the whole answer was cut by the sequence cap).
- [train.py:1273-1286](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1273-L1286): the scoring loop `continue`s on a missing task, an empty text, no `Answer:` marker, or an empty prompt or target. Nothing is counted for the skipped example.
- [train.py:3477-3480](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3477-L3480): the resulting `per_task_accuracy` goes into `task_curriculum.update_metrics` and from there into `dag_state.update_from_ema` and `step_curriculum`.
- Audit probe `code-train-a/p5_train_eval_oracle.py` (real Llemma tokenizer, an oracle model that replays the training continuation): at `alg_seq_len 48`, difficulty 1.0 (DAG-v12's length and final phase), chain had 18 of 47 examples with zero labels; 382 of the requested 400 examples were scored and `train_eval/throughput/task_count/chain` was 29. At `alg_seq_len 128` the same 400 are all scored (chain 44 and 47). Hand-built case: `max_samples=2` with the first example unparseable scores one example.
- code-data's batching probe at the default schedule and 48 tokens found 36 zero-label chain examples per 4,000 and chain 100 % over 48 tokens at difficulty 0.8 and above, so any run that reaches high chain difficulty at 48 is affected. In DAG-v12 chain never left difficulty 0.2 (#157), so the effect there was nil; the only trace a run leaves is a smaller `task_count`.

## Why it matters for the experiments

- A task whose long examples cannot be scored looks easier than it is. The controller then raises its difficulty on the survivors' accuracy, and the DAG gate reads the same number. The signal that is supposed to stop the curriculum from outrunning the model is blind to exactly the examples where it is outrunning it.
- The per-task counts in `train_eval/throughput/task_count/*` are the only evidence; they are not plotted or summarised.
- #144's oracle test, as posted, runs through `eval_hub` and would not see this; the training-time path drives the curriculum and needs its own oracle.

## How it got here

- The generation-based train-eval (`_iter_eval_examples`, `evaluate_task_accuracy`, `_predict_with_tokens`) came with PR [#83](https://github.com/jpturcotte/ReLSM/pull/83) (Dec 20); the `Answer:` extraction with `rfind` came with PR [#119](https://github.com/jpturcotte/ReLSM/pull/119) (Dec 28). Each `continue` was a guard against a crash, not a scoring decision. Not traced to a prompt by this audit.

## How we verified it, step by step

1. code-train-a ran `code-train-a/p5_train_eval_oracle.py` at seq 128 (difficulty 0.5 and 1.0) and seq 48 (difficulty 1.0): counts of re-tokenised prompts, no-marker skips and all-(-100) examples per task, the scored total against `max_samples`, and the hand-built `max_samples=2` case.
2. The coordinator re-read `train.py:1276-1292` and confirmed the five `continue` paths after `_iter_eval_examples` has consumed the sample (`VERIFIED_BY_COORDINATOR.md`).
3. code-data's `code-data/probe_batching.py` supplies the whole-pass zero-label count at 48 tokens (36 of 4,000, all chain).

## Suggested fix

- Count skips per task and per reason and log them (`train_eval/skipped/<reason>/<task>`); print the per-task scored count next to the accuracy.
- Do not charge skipped examples to `max_samples`, or report the shortfall.
- For curriculum purposes treat a zero-label or unparseable example as wrong (accuracy 0), so the EMA and gates cannot rise by dropping hard examples; or refuse the configuration, which is #139's fix.
- Depends on #139 (sequence cap); does not conflict with #157.

## Acceptance test

Run the oracle harness from audit probe `code-train-a/p5_train_eval_oracle.py` (CPU, tiny model, real tokenizer) at `alg_seq_len 48`, difficulty 1.0, `max_samples 400`: the per-task scored counts plus the logged skip counts sum to 400, and chain's `per_task_accuracy` counts its 18 zero-label examples as incorrect (so the oracle scores below 1.0 on chain) or the run refuses the length.

## Related

- #139: `alg_seq_len 48` truncates answers; this is what the truncation does to the curriculum signal.
- #151: zero-label batches give NaN loss; the same examples are dropped here.
- #157: the gates read the survivors' accuracy.
- #144: the oracle regression test must also drive the training-time path.

## Sources

- Audit report code-train-a, finding T2; audit probe `code-train-a/p5_train_eval_oracle.py`
- Audit report code-data, §2 batching (zero-label chain count); audit probe `code-data/probe_batching.py`
- `VERIFIED_BY_COORDINATOR.md`, "Train-eval skips"
