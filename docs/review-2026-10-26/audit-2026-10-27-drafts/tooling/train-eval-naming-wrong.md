---
slug: train-eval-naming-wrong
title: "Training-time eval labels are wrong: no val loss; held-out set called training"
issue: 242
severity: P3
labels: [bug, P3-minor, area: tooling, area: docs, telemetry, audit-2026-10-27]
nf: [NF-27, NF-28]
found_by: [code-train-a, review-files, code-train-b]
status: Demonstrated
related_issues: [#167, #185, #153, #158, #136, #162, #169]
---

## Summary

Two labels in the training-time evaluation say the opposite of what is measured. (a) `summary.final_val_loss` and the "val" curve of `loss.png` read `eval/loss/ce`, which only the language phase writes; every algorithmic-only run, DAG-v12 included, reports `final_val_loss: null` and draws one curve under the title "Training and Validation Loss", while the teacher-forced held-out loss that does exist (`train_eval/loss/ce`) is neither plotted nor summarised. (b) The procedural train-eval stream is built from its own generator with `seed = args.seed + 123`, so none of its examples were trained on, yet code, keys and console call it "Training accuracy" and "memorization tracking". Both are naming defects in the same eval block with one fix, so they are one draft.

## What happens

Validation loss:

- [train.py:452-453](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L452-L453): `summary()` takes `final_val_loss` from `eval/loss/ce`. [train.py:476-478](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L476-L478): `plot()` draws the same series as "val"; [train.py:483](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L483) titles the figure "Training and Validation Loss".
- The only writer is [train.py:3699](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3699), inside the language phase: `math.log(ppl)` of `evaluate_perplexity` on `lang_loader`, a loss that includes the ACT ponder term (#153) over a loader that may alias the algorithmic stream (#158).
- DAG-v12 (`total_tokens == alg_tokens`): `summary.json` has `"final_val_loss": null`. Audit probe `code-train-a/p4_metrics_logger.py`: a logger with no `eval/loss/ce` returns `final_val_loss: None`. Audit run R1 (cli-commands): same.
- [train.py:3604](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3604): `train_eval/loss/ce` (teacher-forced loss on the held-out stream) is logged at every eval and read by nothing in `summary()` or `plot()`.
- [README.md:95](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/README.md#L95): `loss.png` shows "log-scale loss curves".

Held-out stream called "training":

- [train.py:2279-2294](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2279-L2294): the procedural train-eval dataset is a separate `AlgorithmicDataset` with `seed=args.seed + 123` and `difficulty_value=None`. Nothing in it is a training example.
- Labels: `# Training-eval loader (for grokking memorization tracking)` ([train.py:2245](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2245)); `# Training data accuracy (memorization tracking)` ([train.py:3449](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3449)); console `Training accuracy (procedural): ...` ([train.py:3563](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3563)); metric keys `train_eval/*`. Only the CLI help is right: "procedural=fresh samples" ([train.py:3924-3927](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3924-L3927)).
- Since PR #87, `--train_eval_mode auto` selects `procedural` for every `--use_task_curriculum` run ([train.py:2246-2248](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2246-L2248)), so every DAG run has carried the wrong label.

## Why it matters for the experiments

- The grokking signal the project is looking for is a gap between a memorised training set and held-out items of the same rule. Today the run summary has no held-out loss at all (`null`), and the one key that is a held-out measure is named as if it were the training set. A reader of DAG-v12 who sees `train_eval/acc/exact_match` 0.46 for addition against IID 0.0 reads it as "memorised but not generalising"; it is "held-out at the current curriculum difficulty vs a harder grid" (#162).
- The Jan-11 log line "train-eval teacher-forced 69.27 % / loss 0.8711" is the number a summary should carry; `summary.json` carries `null` instead.
- Once #136 is fixed and a fixed set can be evaluated, the project needs both keys side by side, and neither name currently promises the right thing.

## How it got here

- PR [#41](https://github.com/jpturcotte/ReLSM/pull/41) (2025-12-16) added `MetricsLogger.plot()` and `loss.png` (the title string itself was not traced to a PR by this audit); PR [#79](https://github.com/jpturcotte/ReLSM/pull/79) ([task](https://chatgpt.com/remote/task_e_694707b7b5b8832e8c9266cb11119b55), 2025-12-20) introduced the record-based `summary()` with `final_val_loss` from `eval/loss/ce`, at a time when a language phase was the plan for every run.
- PR [#57](https://github.com/jpturcotte/ReLSM/pull/57) ([task](https://chatgpt.com/remote/task_e_6944b5513f4c832e9968d658bb5bb96e), 2025-12-19) added the "Training data accuracy (memorization tracking)" block for the fixed-data grokking recipe, where the label was right. PR [#87](https://github.com/jpturcotte/ReLSM/pull/87) ([task](https://chatgpt.com/remote/task_e_694748de2120832e95eac05ae6205486), 2025-12-21, "Enable procedural train-eval when task curriculum is active") switched curriculum runs to the fresh-sample stream without renaming anything.

## How we verified it, step by step

1. code-train-a (T5) read `summary()`, `plot()` and the single writer at 3699, and ran audit probe `code-train-a/p4_metrics_logger.py` (summary with no `eval/loss/ce` gives `None`). It read DAG-v12's `summary.json` (`final_val_loss: null`) and grepped `metrics.0.json` for `train_eval/loss/ce` (present at every eval). It did not open the PNGs.
2. review-files (RF-6) grepped `train.py:483` for the title and cross-checked the DAG-v12 summary.
3. code-train-b (N2) confirmed `final_val_loss` is `null` in all of its algorithmic-only probe runs and in DAG-v12 (same observation, folded into #185 for the other summary fields).
4. code-train-a (T8) read `train.py:2244-2294`, `3449`, `3563` and the help text, and confirmed the seed offset and the separate generator.
5. Coordinator: not re-checked. No disagreement.

## Suggested fix

One PR, naming only:

- Plot and summarise `train_eval/loss/ce` as the held-out curve (`final_heldout_loss`, legend "held-out (teacher-forced)"); rename the language-phase key `eval/lang/loss/ce` and title the figure "Training loss and held-out eval loss".
- Rename the procedural stream's keys and messages to `heldout_eval/*` and "Held-out accuracy (procedural, seed+123)"; keep `train_eval/*` for the fixed set once #136 is fixed, so the two can appear together.
- Fix the two comments and README line 95.

The key rename touches the duplicate `train_eval/acc/token` key (#167); do both in the same change. Any consumer of `train_eval/*` (plots, `log_task_accuracy`) must follow the rename.

## Acceptance test

A tiny CPU run with `--use_task_curriculum --alg_tokens 2000 --eval_interval 10`: `summary.json` has a non-null held-out loss equal to the last `heldout_eval/loss/ce` record; `loss.png` legend names it; `grep -n "memorization\|Training accuracy" train.py` returns nothing; no key or console line containing `train` describes the `seed + 123` stream.

## Related

- #167: duplicate `train_eval/acc/token` key in the same `logger.log` call.
- #185: the other `summary.json` fields that describe the last window, not the run.
- #153, #158: why the language-phase `eval/loss/ce` is not a clean validation loss either.
- #136: the fixed-set train-eval that would deserve the name `train_eval`.
- #162: the IID grid is not the training distribution; the held-out stream is the closer comparison.
- #169: README line 95.

## Sources

- Audit report code-train-a, findings T5 and T8; probe `p4_metrics_logger.py`
- Audit report review-files, finding RF-6
- Audit report code-train-b, finding N2
- Audit report gemini-c, section 2.6 (Jan-11 log numbers)
