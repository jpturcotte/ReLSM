---
slug: ema-decay-clamp-formula
title: Curriculum EMA decay auto-tune is 0.5 in every DAG run and 0.0 at interval 2000
issue: 204
severity: P2
labels: [bug, P2-important, area: curriculum, correctness, audit-2026-10-27]
nf: [NF-15]
found_by: [code-data, codex-b, gemini-c, claude-c1, claude-c2, gh-prs-b]
status: Demonstrated
related_issues: [#157, #148, #189, #180]
---

## Summary

`train.py` sets the curriculum EMA decay to `clamp(1 - eval_interval / 2000, 0, 0.99)` with `eval_interval` in micro-steps. That is 0.75 at the default interval of 500, 0.5 at the 1000 every DAG run used, and 0.0 at any interval of 2000 or more, where the "EMA" is the last 31-166-example train-eval with no memory. The `ema_decay=0.98` default in `curriculum.py` is never used by `train.py`. #157 reasons about the gates as if the EMA were slow; with decay 0.5 a single eval at 0.96 pulls an EMA of 1.00 down to 0.98, so the gate regime is harsher than #157 describes, and the Dec 22 diagnosis that led to the EMA snap assumed 0.98 while the run printed 0.5. Several agents would fold this into #157; it is kept separate because the formula has its own fix.

## What happens

- [train.py:2066-2069](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2066-L2069): `target_reaction_steps = 2000.0`, `update_interval = float(args.eval_interval)`, `calculated_decay = 1.0 - update_interval / target_reaction_steps`, `ema_decay = min(max(calculated_decay, 0.0), 0.99)`; passed to `TaskCurriculumState` at line 2078. The default `--eval_interval` is 500 ([train.py:3992](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3992)).
- Values (audit probe `code-data/probe_curriculum.py` section A): 100 to 0.95, 250 to 0.875, 500 to 0.75, 1000 to 0.5, 1500 to 0.25, 2000 to 0.0, 4000 to 0.0. At 0.0, `update_metrics` ([curriculum.py:70-78](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L70-L78)) sets `ema_acc = accuracy`, and nothing in `curriculum.py` guards against it.
- Every DAG-era log printed `Auto-tuned Curriculum | Update Interval: 1000.0 | Target Reaction: 2000.0 | Calculated Decay: 0.5000` (Dec 22, Dec 25, Dec 27, Jan 11). "Target Reaction: 2000" is micro-steps: 125 optimizer updates at accumulation 16, 167 at 12.
- The `ema_decay: float = 0.98` default at [curriculum.py:20](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L20) is dead under `train.py`.
- Audit probe `claude-c2/ema_check.py`: the Dec 22 printed EMAs (`parity ema=0.29, 0.46, 0.46, 0.44` after raw 58.3 %, 63.3 %, 46.7 %, 41.7 %) reproduce for 8 of 8 tasks with decay 0.5 and for 1 of 8 with 0.98. Artifacts: recomputing DAG-v12's `curriculum.ema_acc.*` from the raw `train_accuracy.*` with decay 0.5 and the snap rule matches every logged value to 1e-4, except compare at step 9,000 (the IID override).

## Why it matters for the experiments

- With decay 0.5 the "EMA" is a two-eval average. Copy's gate for successor needs EMA at or above 0.98 (0.99 minus the 0.01 margin) for four consecutive evals on about 64 examples: one eval at 0.96 after a 1.00 gives 0.98, the edge; two give 0.97, a fail. The DAG's unlock condition is therefore "four near-perfect raw train-evals in a row", not the slow competence signal #157 assumes.
- The Dec 22 reply that produced the EMA snap and the 0.85 reset (PR #100) said "with EMA starting at 0 and building slowly (`ema_decay=0.98`)"; the run was at 0.5. The snap, now a #157 defect, fixed a cold start that did not exist.
- Anyone sweeping `--eval_interval` to 2000 or more (a natural choice at 60 s per eval, 78 minutes per 100M-token run) silently turns smoothing off: every gate, bump and demotion then reacts to one noisy 800-sample eval.

## How it got here

- PR [#97](https://github.com/jpturcotte/ReLSM/pull/97) (Dec 22, 01:40Z, three turns): the owner wrote that decay 0.98 at `eval_interval 500` "takes ~25,000 steps to react" and asked for `decay = clamp(1 - eval_interval/2000, 0, 0.99)` plus a log line. Codex implemented it as asked; tests "Not run" (twice). The formula's units follow the micro-step counters of #148/#189.
- Chat `2025-12-22_8f01c17a` (15 hours later) misread the banner, and PR [#100](https://github.com/jpturcotte/ReLSM/pull/100) added the snap (claude-c2 C2-1).

## How we verified it, step by step

1. code-data (`code-data/probe_curriculum.py` A), codex-b (probe P3) and claude-c1 (`adaptive_ratchet_probe.py` case E) each evaluated the formula at several intervals: 0.75 / 0.5 / 0.0 as above.
2. claude-c2 ran `claude-c2/ema_check.py` against the Dec 22 printed EMAs (8/8 at 0.5, 1/8 at 0.98, 2/8 with the HEAD snap rule).
3. artifacts recomputed DAG-v12's logged EMAs from the raw per-task accuracies (`artifacts/p2_timeline.py`, B3): match to 1e-4 with decay 0.5.
4. gemini-c and gh-prs-b read the printed banner in the Dec 25, Dec 27 and Jan 11 logs. The coordinator re-read `train.py:2066-2069` and confirmed the formula and the dead `curriculum.py:20` default.

## Suggested fix

- Compute the decay from optimizer steps (`eval_interval / grad_accum_steps`) against a reaction time in updates, floor it at 0.5 (or expose `--curriculum_ema_decay` and make the auto-tune the default only), and warn when the clamp engages.
- Log the decay next to the gate metrics every eval, not only in the banner.
- Remove the dead 0.98 default or make it the fallback when no auto-tune is requested (#180).
- #157's text should state the decay actually in effect. Depends on #148/#189 for an `opt_step` counter; combine with #201 and #202, which read the same EMA.

## Acceptance test

Pure Python: for `eval_interval` in {500, 1000, 2000, 4000} and `grad_accum_steps` in {4, 12, 16}, the decay handed to `TaskCurriculumState` lies in [0.5, 0.99] and `TaskCurriculumState(ema_decay=0.0)` raises. Today 2000 and 4000 give 0.0 and the constructor accepts it.

## Related

- #157: assumes a slow EMA; add the decay in effect and the Dec 22 provenance of the snap.
- #148 / #189: all curriculum counters are micro-steps.
- #180: the dead `ema_decay=0.98` default.
- #201: the controller that consumes this EMA.
- #202: the two-eval average is what lets copy's EMA swing 0.53 to 0.88 and back.

## Sources

- Audit report code-data, finding D10; audit probe `code-data/probe_curriculum.py`
- Audit report codex-b, finding N2 and PR #97 notes
- Audit report gemini-c, finding GC-3
- Audit report claude-c1, finding C1-5; audit report claude-c2, findings C2-1 and C2-4; audit probe `claude-c2/ema_check.py`
- Audit report gh-prs-b, PR #97 row; audit report artifacts, §2b (EMA consistency)
- `VERIFIED_BY_COORDINATOR.md`, "Curriculum EMA decay"
