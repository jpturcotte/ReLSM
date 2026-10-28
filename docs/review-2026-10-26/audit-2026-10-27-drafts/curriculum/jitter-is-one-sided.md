---
slug: jitter-is-one-sided
title: "Curriculum jitter is one-sided (uniform(0, d)): easy replay, not frontier noise"
issue: 220
severity: P3
labels: [bug, P3-minor, area: curriculum, audit-2026-10-27]
nf: [NF-20]
found_by: [agent-reports, claude-b1]
status: Demonstrated
related_issues: [#157, #140]
---

## Summary

Both jitter sites draw `uniform(0.0, difficulty)`: a jittered example is never harder than the task's current difficulty. The two documents the controller was built from describe jitter as two-sided noise around the target, so that the model meets slightly harder examples before the level advances. ReLSM's `--curriculum_jitter` is therefore a second easy-replay knob on top of `--easy_mix_frac`, and a jitter ablation measures something other than what the sources describe. The help text says "replaying easier samples", so the code matches its own flag; what is wrong is the name and the idea it stands in for.

## What happens

- [curriculum.py:145-146](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L145-L146) and [train.py:220-221](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L220-L221): `random.uniform(0.0, difficulty)` with probability `jitter_prob` (default 0.1, [train.py:3844-3849](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3844-L3849), help "Probability of replaying easier samples per task"). The easy mix at [train.py:216-217](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L216-L217) already draws `uniform(0.0, 0.3)` for 20 % of examples.
- Audit probe `code-data/probe_schedules.py`: at a snapshot difficulty of 0.65 the realised buckets are easy and medium only, never hard; at 1.0, 3 % of draws land in the medium bucket and none above. Audit probe `code-train-a/p6_dag_weighting_misc.py`: with `min_difficulty 0.4` the sampler still returns a minimum of 0.000, because the jitter ignores the floor by design (PR #105).
- The sources (agent-reports AR-4): the Compass report, whose 0.85/0.55 thresholds the controller took in the same commit that added `jitter_prob` (`bb081bc`, Dec 20), defines jitter as `target + Normal(0, 0.2 * max_difficulty)` clipped (Compass L259-266, L421-424); the Ramping report says jitter "exposes the model to slightly harder and slightly easier instances" (Ramping L38). claude-b1 lists "jitter centred on the frontier, `[d - eps, d + eps]`" (chat `6e635fde` line 39) among the ideas discussed and dropped.

## Why it matters for the experiments

- The controller bumps a task to d + 0.05 the first time its EMA clears 0.85, and the model meets d + 0.05 cold: nothing in the stream previewed it. That is one ingredient of the bump-then-crash cycle in #202 (copy at 0.65: 0.44-0.59 right after each bump).
- An ablation of `--curriculum_jitter` compares 10 % replay against 0 % replay, not look-ahead against none. Any write-up that cites the Compass or Ramping rationale for the flag is describing a feature the repo does not have.

## How it got here

- Commit `bb081bc` / PR [#85](https://github.com/jpturcotte/ReLSM/pull/85) (Dec 20-21) added `jitter_prob` as `uniform(0, d)` alongside the Compass thresholds.
- Chat `2025-12-22_8f01c17a` recommended that jitter respect the floor; PR [#105](https://github.com/jpturcotte/ReLSM/pull/105) (Dec 23) adopted it and then reverted it at the owner's request ("Restore jitter behavior independent of minimum difficulty"), keeping `uniform(0, d)`.

## How we verified it, step by step

1. agent-reports read both source documents and both code sites (AR-4).
2. code-data's `code-data/probe_schedules.py` and code-train-a's `p6` give the measured bucket shares and the below-floor minimum; no draw above the snapshot in 20,000.
3. claude-b1 read chat `6e635fde` for the dropped two-sided variant.

## Suggested fix

- Two-sided: with probability `jitter_prob`, `clip(d + uniform(-eps, eps), floor, 1.0)` with `eps` a flag (0.05-0.1), or the Compass Gaussian; respect `min_difficulty`.
- Or rename the flag `--curriculum_easy_replay` and fold it into `--easy_mix_frac`, and say so in the README.
- Small; land with #219, which changes the same two lines.

## Acceptance test

Pure Python: 10,000 calls to `CurriculumDifficultySampler` at snapshot 0.5 with `curriculum_jitter 1.0` and `easy_mix_frac 0.0` produce values above 0.5 (two-sided) and none below `min_difficulty`; or, if the rename is chosen, the flag `--curriculum_jitter` no longer exists.

## Related

- #157: the controller's band and the gates; two-sided jitter is one of the smaller levers on the bump dynamics.
- #140: the easy fifth of the stream is already replay; the jitter adds up to 10 % more.
- #219: the RNG these draws come from.
- #202: the cycle the one-sided draw feeds.
- #201: the floor the jitter ignores.

## Sources

- Audit report agent-reports, finding AR-4 (Compass L259-266, L421-424; Ramping L38)
- Audit report claude-b1, §4 avenue 13 (chat `6e635fde`)
- Audit report code-data, finding D9; audit probe `code-data/probe_schedules.py`
- Audit report code-train-a, audit probe `code-train-a/p6_dag_weighting_misc.py`
- Audit report gh-prs-b, PR #105 row
