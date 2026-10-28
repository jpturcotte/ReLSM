---
slug: controller-limit-cycle-copy
title: "Controller limit cycle: copy toggled 0.60/0.65 for 40k micro-steps in DAG-v12"
issue: 202
severity: P2
labels: [bug, P2-important, area: curriculum, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-49]
found_by: [artifacts, code-data]
status: Demonstrated
related_issues: [#157, #139, #162]
---

## Summary

With a step of 0.05, a band of 0.55/0.85, an EMA decay of 0.5 and a cooldown (500 micro-steps) shorter than the eval interval (1000), a task whose accuracy is about 1.0 at difficulty d and about 0.5 at d + 0.05 cannot settle. At d the EMA clears 0.85 within two evals and the controller bumps; at d + 0.05 it falls under 0.55 within two or three and the controller demotes. Copy in DAG-v12 did this between 0.60 and 0.65 from step 12,000 to 52,000, with its raw train-eval accuracy swinging between 0.96-0.99 and 0.44-0.59. Its EMA peaked at 0.87-0.88 for one eval per cycle and never held the 0.98 the successor gate needs for four evals, so even a perfect 10-item copier could not have opened that gate. This is the dynamic form of #157's "gates above the controller band".

## What happens

- [curriculum.py:110-139](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L110-L139): bump when `ema_acc > 0.85`, demote when `ema_acc < 0.55`, step 0.05, cooldown check against `last_update_step` in micro-steps. Decay 0.5 from [train.py:2066-2069](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2066-L2069) (#204). The cooldown of 500 never blocks an eval 1000 steps later.
- DAG-v12 copy, `curriculum.difficulty.copy` (audit probe `artifacts/p2_timeline.py`, B5): 0.20 at 1,000, then +0.05 per eval from 4,000 to 0.65 at 12,000; 0.60 at 18,000, 0.65 at 20,000, 0.60 at 22,000, 0.65 at 24,000, 0.60 at 27,000, 0.65 at 29,000, 0.60 at 32,000, 0.65 at 34,000, 0.60 at 37,000, 0.65 at 39,000, 0.60 at 43,000, 0.65 at 45,000, 0.60 at 50,000, 0.65 at 52,000. That is fourteen changes of direction after the ramp (AR-3 counts eleven toggles over 18k-52k; the full list is above). Raw `train_accuracy.copy` at 0.60: 0.96-0.99; at 0.65: 0.44-0.59. EMA peaks 0.87-0.88, troughs 0.53-0.58. From 56,000 the raw score is exactly 0.609 (39 of 64) at every eval and the cycle stops because 0.609 sits inside the band; final difficulty 0.65, final EMA 0.609.
- Reverse never cycled: its raw accuracy at 0.65 was 0.61-0.68, inside the band from 13,000 on (final 0.639).
- The three training-loss spikes of the run (steps 3,000, 6,050, 10,050: 0.24, 0.49, 0.50 against a 0.03-0.08 baseline) each follow a difficulty bump.
- The cliff is the 11-item copy: from the logged IID samples at 20k and later, copy at 10 items 115 of 116 correct, at 11 items 2 of 58, at 12 items 0 of 116 (#139, #162). Difficulty 0.65 brings 11-12-item copies into the train-eval; 0.60 does not.
- Audit probe `code-data/probe_curriculum.py` section D re-implements the `train.py` glue: a learner with the copy/reverse cliff at 0.65 and an otherwise exposure-driven competence flips copy and reverse 99 times each between 0.60 and 0.65 in 300 evals, the EMA pinned at 0.597, and nothing ever unlocks (D5c); with a perfect learner plus the cliff, everything unlocks at eval 4 and then `paused` is set for 293 of 300 evals (D3); a cooldown of 1500 does not change it (D8).

## Why it matters for the experiments

- The successor gate needs copy's EMA at or above 0.98 for four consecutive evals (`0.99 - unlock_margin`). In the cycle the EMA touches 0.87-0.88 once per cycle and is pushed back. Successor, and through it chain, addition, mod_add, multiplication and dyck (#157), could not have unlocked in DAG-v12 even if the model had been perfect at 10 items.
- `dag_paused` went to 1 at step 13,000 because copy and reverse fell below their lock margins after the 0.65 bump (0.729 / 0.833) and stayed paused for 65 evals: the pause #157 describes is the first demotion of this cycle.
- The three loss spikes, read elsewhere as instability, are the controller's bumps.

## How it got here

- PR [#85](https://github.com/jpturcotte/ReLSM/pull/85) (Dec 21) set the band, the step and a cooldown of 500 after Claude's Dec 20 guide ("advance at 0.85 / retreat at 0.55 with a 100-step cooldown and EMA 0.99", claude-b1); PR [#97](https://github.com/jpturcotte/ReLSM/pull/97) (Dec 22) made the EMA 0.5 at this eval interval; PR [#107](https://github.com/jpturcotte/ReLSM/pull/107) (0.70/0.50 plus flags) was closed unmerged on Dec 25. The cliff itself comes from `--alg_seq_len 48` truncating 11-12-item copies (#139). Nobody asked for hysteresis; the design documents assumed a slow EMA.

## How we verified it, step by step

1. artifacts ran `artifacts/p2_timeline.py` over DAG-v12 `metrics.0.json`: the copy difficulty path (B5), raw vs EMA per eval (B1), the pause trace (B2), the loss spikes (A), and `artifacts/p2b_samples.py` for the per-length IID copy counts.
2. code-data ran `code-data/probe_curriculum.py` section D (synthetic learners through the real `curriculum.py` and a re-implementation of `train.py:3273-3284` and `3521-3545`): D3, D5c, D8 as above. The structural statement (bump at every eval above 0.85, unlock needs four evals at 0.97-0.98) is read directly from `curriculum.py:129-130`, `325`, `361` and `train.py:3541`.
3. The coordinator confirmed the hard-coded band and the unmerged PR #107 (`VERIFIED_BY_COORDINATOR.md`); the cycle was not re-derived by the coordinator.

## Suggested fix

With #157:

- Hysteresis: bump only after two consecutive evals above the high threshold, demote only after two below the low one; use a smaller step near the top of the band.
- Decide difficulty from per-length accuracy, not the mean over a length mix. The train-eval already records `target_len_tokens` per sample; "10 items 1.0, 11 items 0.0" is a different signal from "mean 0.6".
- Gate on accuracy at a fixed length (for example copy at 10 items for four evals), not at the moving difficulty.
- Express cooldown in optimizer steps and require it to span at least two eval intervals (#148/#189).
- Depends on #204 (a decay of 0.5 makes any band twitchy) and #201 (same function); conflicts with nothing.

## Acceptance test

Pure Python with `curriculum.py`: a scripted learner with accuracy 1.0 at difficulty 0.60 and below, 0.5 at 0.65, run through `update_metrics` and `step_curriculum` for 50 evals at decay 0.5. With the fix the difficulty changes direction at most twice and the EMA at 0.60 holds at or above 0.98 for four consecutive evals; today it changes direction more than ten times and the EMA never exceeds 0.88.

## Related

- #157: "gates above the controller band", the pause at 13,000 and the successor choke point; this is their dynamic mechanism.
- #139: the 48-token cap makes the cliff at 11 items.
- #162: the IID copy numbers by length.
- #204: the decay that lets the EMA swing 0.53 to 0.88 in two evals.
- #201 and #205: the other two controller defects on the same function.
- #220: the stream never previews d + 0.05 before the bump.

## Sources

- Audit report artifacts, finding AR-3 and §2b; audit probes `artifacts/p2_timeline.py` (A, B1, B2, B5) and `artifacts/p2b_samples.py`
- Audit report code-data, finding D5; audit probe `code-data/probe_curriculum.py` (D3, D5c, D8)
- Audit report claude-b1, §"Task curriculum review" (PR #85 provenance)
- `VERIFIED_BY_COORDINATOR.md`, "Controller band 0.85/0.55"
