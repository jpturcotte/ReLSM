---
slug: adaptive-controller-no-floor-ratchets-to-zero
title: Adaptive task controller has no floor and ratchets every task to difficulty 0.0
issue: 201
severity: P2
labels: [bug, P2-important, area: curriculum, correctness, baseline-affected, audit-2026-10-27]
nf: [NF-14]
found_by: [claude-c1, claude-c2, chatgpt-b, gh-prs-b]
status: Demonstrated
related_issues: [#157, #141, #162, #189]
---

## Summary

The default task controller (`--task_curriculum_strategy adaptive`) lowers a task's difficulty by 0.05 at every eval where its EMA accuracy is below 0.55, and clamps at `min_difficulty`, whose default is 0.0 in both `curriculum.py` and the CLI. The cooldown (500 micro-steps) is shorter than the eval interval (1000), so every task is eligible at every eval. A task that has not yet learned anything therefore walks 0.20, 0.15, 0.10, 0.05, 0.00 in four evals and trains on the trivial band from then on, while the IID eval keeps scoring it at difficulty 0.5. The Dec 22 run did exactly this for all ten tasks. The floor Claude recommended that day (default 0.1) shipped the next day with default 0.0, and the one surviving run recorded `min_difficulty = 0.0`. #157 covers the DAG gates, the EMA snap and the pause; the ratchet is the controller's own behaviour and hits unlocked tasks under the DAG too.

## What happens

- `step_curriculum` at [curriculum.py:110-139](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L110-L139): `elif ema_acc < acc_threshold_low: difficulty -= self.step_size` (line 131), then `min(max(difficulty, self.min_difficulty), 1.0)` (line 136). Thresholds are keyword defaults 0.85/0.55 ([curriculum.py:116-117](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L116-L117)); the call at [train.py:3541-3545](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3541-L3545) does not override them and no CLI flag exists.
- `min_difficulty` defaults to 0.0 at [curriculum.py:19](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/curriculum.py#L19) and [train.py:3802-3806](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3802-L3806). The default strategy is `adaptive` ([train.py:3825-3830](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3825-L3830)).
- The Dec 22 log (run `exp1_loop_100M_v1`, nano `shared_loop`, `--use_task_curriculum`, eval every 1000): all ten tasks move 0.200 to 0.150 (step 1000), 0.100 (2000), 0.050 (3000), 0.000 (4000), while the IID block directly above each line reads `[eval] addition/iid difficulty=0.500`. At step 4000 train-eval addition is 61.2 % and successor 61.0 %, but their EMAs are 0.35 and 0.33 (decay 0.5), still under 0.55, so the controller pushed down again. From step 4050 every task trained at 0.000 with `easy_mix 0.2` on top. The owner's only message: "Something seems to be wrong with the per-task difficulty scaling."
- Audit probe `claude-c1/adaptive_ratchet_probe.py` on HEAD `curriculum.py` (defaults, decay 0.5, `min_task_evals` 5, evals every 1000): accuracy rising 0.0 to 0.80 over 11 evals gives difficulty 0.20 until eval 4, then 0.15, 0.10, 0.05, 0.00 at evals 5-8, and 0.00 while accuracy climbs to 0.80 (EMA 0.71 < 0.85). Accuracy 0.50 forever (chance for parity and dyck) ends at 0.00 by eval 8. `--min_difficulty 0.1` stops at 0.10. Accuracy 0.60 never moves.
- Difficulty 0.0 is a real band, not a label. `get_difficulty_params` uses `start <= progress < end` ([data.py:150-159](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L150-L159)) and the 0.0 phase differs from 0.2 for every task ([data.py:35-110](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L35-L110)): copy and reverse 1-3 items vs 3-6, successor 1-4 digits vs 2-8, addition 1-2 digits vs 1-3, parity 2-5 bits vs 4-8, mod_add modulus 17-50 vs 50-200, chain 1-2 ops vs 2-3.
- History (chatgpt-b, chat `694b4280`): run4 (`min_task_evals` 5) had parity, chain, compare, dyck and mod_add hit 0.05 then 0.00 together at step 8k; run5 (`min_task_evals` 20) the same cascade at 23k; run6 (run4 plus `--min_difficulty 0.2`) pinned 8 of 10 tasks at exactly 0.2 for the whole run. The first DAG run (Dec 25, gh-prs-b) drove the locked tasks to 0.05 by step 8000 before PR #109 stopped stepping locked tasks. Under the DAG, unlocked tasks remain exposed: DAG-v12 recorded `min_difficulty 0.0` and parity went 0.20 to 0.10 right after its unlock (see #205).

## Why it matters for the experiments

- Every adaptive-mode run from Dec 20 to Dec 24 (the only mode before PR #108) trained the easiest bucket after a few evals and was scored at difficulty 0.5. Their IID numbers measure a distribution the model had stopped seeing; any "curriculum vs no curriculum" comparison from that window is confounded.
- The controller is still the default strategy at HEAD. A new user who runs `--use_task_curriculum` gets the Dec 22 behaviour, delayed by `min_task_evals` (5 evals) and masked by the EMA snap.
- Under the DAG the same rule is what demotes freshly unlocked tasks (NF-48) and what drives the 0.60/0.65 limit cycle (NF-49); a floor is the first of the four controller fixes those drafts need.

## How it got here

- PR [#85](https://github.com/jpturcotte/ReLSM/pull/85) (Dec 21) created `TaskCurriculumState` with 0.85/0.55, step 0.05, cooldown 500 and no floor, three hours after Claude's Dec 20 "Implementation Guide" (chat `6e635fde`) recommended "advance at 0.85 / retreat at 0.55 with a 100-step cooldown and EMA 0.99" (claude-b1).
- Dec 22, chat `2025-12-22_8f01c17a_Curriculum_learning_difficulty_threshold_tuning.md`: the owner pasted the log four times; Claude's reply (lines 6106-6114) proposed a `min_difficulty` floor with default 0.1, step-down only on sustained regression, and `acc_threshold_low` 0.40.
- PR [#100](https://github.com/jpturcotte/ReLSM/pull/100) (Dec 22, two hours later) added `min_task_evals` and the EMA snap, not the floor. Commit `4a469d6` / PR [#105](https://github.com/jpturcotte/ReLSM/pull/105) (Dec 23, titled "Restore jitter behavior independent of minimum difficulty") added `--min_difficulty` with default 0.0 and `tests/test_min_difficulty.py`. PR [#107](https://github.com/jpturcotte/ReLSM/pull/107) (Dec 24, thresholds 0.70/0.50 plus flags) was closed unmerged on 2025-12-25T01:55Z. PR [#109](https://github.com/jpturcotte/ReLSM/pull/109) (Dec 25) stopped stepping locked tasks under the DAG only.

## How we verified it, step by step

1. claude-c1 read the Dec 22 log (lines 351, 709, 1066, 1424, 1503 for the difficulty transitions; 216, 575, 933, 1290 for the IID lines at 0.5; 1491-1492 for the step-4000 accuracies) and ran audit probe `claude-c1/adaptive_ratchet_probe.py` (cases A-D above) against HEAD `curriculum.py`.
2. claude-c2 ran audit probe `claude-c2/ema_check.py`: the printed EMAs reproduce with decay 0.5 for 8 of 8 tasks and the controller trace gives 0.15, 0.10, 0.05, 0.00, matching the log; it read `git show 4a469d6` for the default-0.0 provenance and the DAG-v12 `metrics.0.json` first row for `min_difficulty = 0.0`.
3. chatgpt-b read chat `694b4280` for run4/run5/run6; gh-prs-b read PR #109's pasted first-DAG-run log (0.05 for locked tasks at step 8000).
4. The coordinator confirmed the hard-coded band and the unmerged PR #107 (`VERIFIED_BY_COORDINATOR.md`); the ratchet itself was not re-run by the coordinator.

## Suggested fix

- Default `--min_difficulty` to `init_difficulty` (0.2), or to 0.1 as recommended on Dec 22; never below the difficulty a task was unlocked at.
- Step down only after N consecutive evals below the band (hysteresis), and make the down-step smaller than the up-step or require the cooldown to be at least two eval intervals.
- Express `curriculum_cooldown` and `eval_interval` in optimizer steps (#148/#189); today both are micro-steps and the cooldown is inert.
- Re-land PR #107's threshold flags so the band can be set per run.
- Depends on nothing; #157's fix list should reference this as the floor item. Coordinated with #205 and #202, which change the same function.

## Acceptance test

Pure Python, importing `curriculum.py` only (extend `tests/test_min_difficulty.py`): at default arguments, feed `update_metrics` an accuracy of 0.50 for 12 evals 1000 steps apart and call `step_curriculum` each time; the difficulty must never fall below the documented floor (0.2 or 0.1), and for an accuracy trace rising 0.0 to 0.8 it must never fall below its starting value. Today both end at 0.00 by eval 8.

## Related

- #157: DAG gates, snap and pause; this is the controller underneath them.
- #141: the schedule path has its own stuck-at-easy failure; this one is the curriculum path.
- #162: the IID grid at 0.5 is what the ratcheted runs were scored on.
- #189: `exp1_loop_100M_v1` has no artifact; the pasted log is its only record.
- #205: the same rule applied to a task that has just unlocked.
- #202: the same rule oscillating at the top of the band.
- #204: why the EMAs that drive the rule are a two-eval average.

## Sources

- Audit report claude-c1, finding C1-1; audit probe `claude-c1/adaptive_ratchet_probe.py`
- Audit report claude-c2, finding C2-2 and §5; audit probe `claude-c2/ema_check.py`
- Audit report chatgpt-b, file 29 notes (run4/run5/run6; chat `694b4280`)
- Audit report gh-prs-b, PR #105, #107, #109 rows
- Audit report claude-b1, §"Task curriculum review" (chat `6e635fde`, PR #85 provenance)
- Transcript `2025-12-22_8f01c17a_Curriculum_learning_difficulty_threshold_tuning.md`
