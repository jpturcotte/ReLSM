---
slug: warmup-steps-40-token-estimate
title: "Step estimate and --warmup_steps assume 40 tokens per example; measured 4.6-18"
issue: 248
severity: P3
labels: [bug, P3-minor, area: training, reproducibility, audit-2026-10-27]
nf: [NF-10]
found_by: [code-train-b, cli-commands, gh-prs-a, codex-a, codex-b, chatgpt-b, claude-a2, claude-b1, claude-c1]
status: Demonstrated
related_issues: [#173, #148, #149, #139, #169]
---

## Summary

`algo_steps = alg_tokens // (alg_batch_size * 40)` assumes 40 counted tokens per algorithmic example. The counter it is compared with counts answer tokens only (#217), and the measured mean is 4.6–8 per example at the defaults and 11–18 in late phases. The printed "Estimated steps" is 2–8× low, and `--warmup_steps N` is converted into tokens through the same estimate, so it buys 2–9× N micro-steps. The default path (`--warmup_frac`) is unaffected and DAG-v12 had `warmup_steps: null`; the Dec 19–21 runs all passed `--warmup_steps` and warmed up about twice as long as asked. chatgpt-b and claude-b1 rated it P2 for those runs; this draft keeps the merged table's P3 because no surviving artifact used the flag and the default path is correct.

## What happens

- [train.py:2540](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2540): `algo_steps = args.alg_tokens // (args.alg_batch_size * 40)`. [train.py:2542](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2542): `lang_steps` uses `alg_batch_size`, not `lang_batch_size`.
- [train.py:2546](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2546): `estimated_tokens_per_step = total_tokens / estimated_steps`. [train.py:2555](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2555): `warmup_tokens = int(warmup_steps * estimated_tokens_per_step)`. [train.py:2570](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2570): the estimate is printed.
- `alg_batch_size` is a token-budget base in the procedural path, not an example count (216–241 examples at 64 × 128), which moves the per-batch count again.

Measured counted tokens per example: 4.6–4.8 at the defaults (cli F04); 4.63 at progress 0 and 8.20 over progress 0–1 (claude-b1); 6.9–8.4 (code-data); 4.8 / 7.7 / 7.5 at d 0.2 / 0.5 / 1.0 (codex-b P1); 8.9 (code-train-b `base_w0`); 11.1 at the final phase, where total tokens are 37.2, so the heuristic was right for total tokens (codex-a); 12.4 from the DAG-v12 artifact and 18.3 from the Dec-22 log at nominal batch 72 (claude-c1).

Consequences in logs:

| run | requested | real |
|---|---|---|
| Dec-19 `_v4` (`--warmup_steps 10000`) | 10,000 steps | LR reached `max_lr` at micro-step ~19,000 |
| Dec-19 `_v2` (`--warmup_steps 30000`) | 30,000 | ~56,000 |
| Dec-19 mod_add v4/v5 (`--warmup_steps 2000`) | 2,000 | LR 1.00e-3 first at 4,450 |
| Dec-22 run, "Estimated steps 34722" | 34,722 | ~75,900 micro-steps |
| DAG-v12, estimate 100M // (96 × 40) = 26,041 | 26,041 | 77,503 micro-steps |
| audit run R13 (`--warmup_steps 5`, 1,200-token budget) | 5 | "Warmup: 857 tokens", 71.4% of training, 19 micro-steps |

## Why it matters for the experiments

- Every run record that quotes `warmup_steps` or "Estimated steps" means something else. ChatGPT's "warmup is 25.6% of training" diagnosis (Dec 19) and "the warmup doesn't complete until ~step 1900" (Dec 20) were read off this conversion; the warmup choices that followed rest on it.
- The 5%-of-training warning at `train.py:2556` is computed in tokens and is right, which hides the step mismatch.
- Budget planning from the printed estimate is off by the same factor (README "~10 min", #169).

## How it got here

- PR [#42](https://github.com/jpturcotte/ReLSM/pull/42) (2025-12-16, "Improve hybrid estimated steps calculation"); owner's prompt: "Algorithmic data is sparse … Effective length is approx 40 tokens". The constant was a guess for total tokens per example at the time.
- PR [#49](https://github.com/jpturcotte/ReLSM/pull/49) (2025-12-18) added `--warmup_steps`; PR [#51](https://github.com/jpturcotte/ReLSM/pull/51) (2025-12-18) made the schedule token-based and kept `--warmup_steps` "meaning steps" by converting through the estimate.
- PR [#80](https://github.com/jpturcotte/ReLSM/pull/80) (Dec 20) and PR [#110](https://github.com/jpturcotte/ReLSM/pull/110) (Dec 25) changed what the counter counts (padding, then prompt excluded) and left the 40 in place, which took the ratio from ~2× to ~4–8×.

## How we verified it, step by step

1. code-train-b (N4) ran `train.main()` on CPU: estimate 3 steps, run took 11; 638 counted tokens over 72 examples = 8.9.
2. cli-commands (C7) ran R13 with `--warmup_steps 5` and recorded the printed warmup and the real step count; F04/F05 gave 4.6–4.8 per example and the 2,560-per-micro-step assumption.
3. claude-b1 (B1-1), codex-b (P1), code-data and codex-a (F1) measured the per-example count from the real generators and tokenizer at several difficulties and phases.
4. chatgpt-b (B-1), claude-a2 (A2-4) and claude-c1 (C1-2) recomputed the Dec-19 and Dec-22 warmups from the logged LR and Progress lines and the DAG-v12 metrics (step 41,900 at 49.99M tokens = 12.4 per example).
5. gh-prs-a (A7) and codex-a traced the constant to PR #42 and the conversion to PR #51.
6. The coordinator re-read `train.py:2540`, `2546`, `2555` and confirmed; measured range 4.6–11 per example in the agents' probes.

## Suggested fix

- Estimate tokens per step from a measured sample (the probe batch at `train.py:2308` already exists) or from the first 50 micro-steps, and print the measured value; or drop `--warmup_steps` and keep `--warmup_tokens`/`--warmup_frac`.
- Better: accept warmup in optimizer steps and apply it as a step count, which also removes the `lang_batch_size` slip at 2542.
- Independent of #217 (that fixes the unit; this fixes a stale constant on top of it) and of #173 (the schedule's edge cases). One change could cover all three if warmup becomes step-based.

## Acceptance test

CPU, tiny config, `--warmup_steps 20`: the first micro-step at which `get_lr` returns `max_lr` is between 18 and 22 (today, with 4.6 counted tokens per example at the defaults, it is about 170).

## Related

- #173: cosine zero division and the 40% jump, same code block, different defect.
- #148 / #149: the micro-step counter also feeds this conversion.
- #139: log realised examples and tokens per batch, which gives the measured estimate for free.
- #169: README run-time estimates inherit the error.
- #217: the unit the constant should match.

## Sources

- Audit report code-train-b, finding N4; probe `code-train-b/run_train.py` (`base_w0`)
- Audit report cli-commands, finding C7; probes `cli-commands/flag_semantics.py` (F04, F05), `cli-commands/tiny_train.py` (R13)
- Audit report claude-b1, finding B1-1; probe `claude-b1/probe_token_budget.py`
- Audit report codex-b, finding N1; probe `codex-b/probe_codex_b.py` (P1)
- Audit report codex-a, finding F1; gh-prs-a, finding A7; chatgpt-b, finding B-1 (chats `6945b243`, `6945e825`); claude-a2, finding A2-4; claude-c1, finding C1-2; code-data
- VERIFIED_BY_COORDINATOR.md (40-token row)
