---
slug: act-knobs-not-controllable
title: "ACT has no --halt_threshold flag, no eval-time K override, no halting histogram"
issue: 231
severity: P3
labels: [enhancement, P3-minor, area: model, telemetry, variants-only, audit-2026-10-27]
nf: [NF-07]
found_by: [chatgpt-a, gemini-a]
status: Demonstrated
related_issues: [#132, #145, #153, #161, #187]
---

## Summary

Three controls the design documents named as the test of the recursive variants do not exist. `halt_threshold` is a dataclass default (0.99) with no CLI path. `K`, `min_K` and `max_K` are config fields read from the checkpoint, and neither `eval_hub.py` nor `utils.py` can override them, so "think longer at test time" cannot be run. The only ACT telemetry is a window mean of inner steps; the per-sequence `steps_used` tensor is averaged and discarded, so the per-task halting histogram and the steps-vs-correctness correlation that the Dec-9 ladder made the Exp3 win condition were never logged. With #132 (no gradient to the halt head) that histogram would have shown the collapse in the first log.

## What happens

- [model.py:133-136](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L133-L136): `K=4`, `min_K=2`, `max_K=8`, `halt_threshold=0.99` as config defaults. [train.py:3777-3778](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3777-L3778): flags for `min_K`/`max_K` exist; there is no `--halt_threshold` anywhere in `train.py` (grep).
- [model.py:777](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L777) and [model.py:792](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L792): the halting probability and the threshold test; [model.py:800](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L800): the per-sequence `steps_used` is stored in `model.aux["inner_steps"]`.
- [train.py:2683-2691](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2683-L2691): the tensor is reduced to `.float().mean().item()`; [train.py:2970](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2970): logged as the window mean `train/act/inner_steps`. No per-task, per-example or eval-time record.
- `grep -n "max_K\|halt_threshold" eval_hub.py utils.py` returns nothing: the eval tool loads the checkpoint's config as is.

## Why it matters for the experiments

- The Dec-8 design's own validation was "Freeze the model. Feed inputs with nesting depth 20 ... Allow the loop to run for up to 30 steps" and the hypothesis that the model generalises "by simply thinking longer". That experiment cannot be run from the eval tool at HEAD, for any trained `latent` or `act` checkpoint.
- The Dec-9 ladder's Exp3 win condition was a "histogram of halting steps by task type" and "correlation of steps with correctness (should go up)", with "no 'always halt' / 'never halt' collapse". Only `min_K` was built. #132 shows every sequence runs `max_K`; a histogram would have made that visible at step 50.
- Chats 7, 11 and 16 asked for a threshold anneal ("Early training: lower threshold (e.g. 0.7) ... anneal to 0.95-0.99"); nothing exists.

## How it got here

- PR [#1](https://github.com/jpturcotte/ReLSM/pull/1) (2025-12-10) wrote the ACT loop with "Store auxiliary stats (inner steps, ponder) on self.aux for logging"; chat 5 (Dec 9) said `model.aux` would "hold step stats for your ACT histograms". The consumer was never written.
- PR [#2](https://github.com/jpturcotte/ReLSM/pull/2) (2025-12-10, "Fix ACT inner step tracking") made `steps_used` per-sequence in form; `train.py` averaged it from then on.
- No Codex task asked for `--halt_threshold`, an eval override of `K`/`max_K`, or the histogram.

## How we verified it, step by step

1. chatgpt-a (CA-2) read chats 4, 5, 7, 11 and 16 for the stated requirements and grepped `train.py` for `halt_threshold` (absent) and the `inner_steps` consumers (window mean only).
2. gemini-a (G-A-4) read the Dec-8 Gemini design (lines 1575–1581) and grepped `eval_hub.py` and `utils.py` for `max_K`, `.K`, `min_K`, `halt_threshold` (nothing).
3. This draft re-ran the greps at `774513d`: `train.py` has `--min_K`/`--max_K` (3777–3778) and no `--halt_threshold`; `eval_hub.py`/`utils.py` have no override.
4. No training probe was needed: the finding is the absence of three code paths. Status is Demonstrated by grep.

## Suggested fix

- `--halt_threshold` passed through `create_model`, with an optional linear anneal (0.7 → 0.99) over the algorithmic budget.
- `--eval_K` / `--eval_max_K` in `eval_hub.py` that override the checkpoint config, written into the eval JSON `meta`.
- At eval, log a per-task histogram of `steps_used` and the mean steps for correct vs incorrect examples; at train, log the window histogram, not only the mean.
- Depends on #132 for the histogram to carry information (today it is a spike at `max_K`). Adds three acceptance items to #132 rather than changing its fix.

## Acceptance test

CPU, tiny `act` checkpoint: `python train.py --help` lists `--halt_threshold`; `eval_hub.py --eval_max_K 16` on that checkpoint writes `meta.eval_max_K: 16` and `act.steps_hist.<task>` (a dict of step count → examples) into the results JSON, and `act.steps_mean_correct` / `act.steps_mean_incorrect` per task.

## Related

- #132: the halt head has no gradient; this adds the controls and telemetry that would have exposed it.
- #145: `create_model` override collisions; the new flags must not go through `**base`.
- #153: ponder term in reported PPL.
- #161: eval JSON keeps too little; the histogram belongs in the same record.
- #187: run record should carry `halt_threshold`.

## Sources

- Audit report chatgpt-a, finding CA-2 (chats 4 `69388926`, 5 `6938adff`, 7 `693a2b2b`, 11 `693b54b7`, 16 `6941a570`)
- Audit report gemini-a, finding G-A-4 and avenue A-4 (Dec-8 design file G:5905, lines 1575–1581)
