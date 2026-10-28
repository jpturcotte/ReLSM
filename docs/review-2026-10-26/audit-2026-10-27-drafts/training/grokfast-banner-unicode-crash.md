---
slug: grokfast-banner-unicode-crash
title: "--grokfast --extended_logging dies with UnicodeEncodeError when stdout is a file"
issue: 216
severity: P2
labels: [bug, P2-important, area: training, reproducibility, audit-2026-10-27]
nf: [NF-08]
found_by: [code-train-b]
status: Demonstrated
related_issues: [#137, #185, #188, #184]
---

## Summary

The startup banner prints the Grokfast settings with `α` and `λ`. On Windows, when stdout is redirected to a file or a pipe, Python encodes it with the locale code page (cp1252), which has no Greek letters, and the run dies before the first batch with `UnicodeEncodeError`. Both of the audit's Grokfast probes ended that way with `batches 0 opt_steps 0` and an empty `metrics.json`. The same hazard sits in the extended-logging format-crash warning (`Δ`), which would kill a run after an eval. An interactive console is unaffected, so a run started by hand survives and a run whose log is redirected or launched by an agent harness does not.

## What happens

- [train.py:2382-2385](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2382-L2385): `log_print(f"  Grokfast: {args.grokfast}" + (f" (α={args.grokfast_alpha}, λ={args.grokfast_lambda})" if args.grokfast else ""))`. The non-ASCII characters are only emitted when `--grokfast` is on.
- [train.py:1923-1927](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1923-L1927): the `[warn] ... exact-match dropped by ... (Δ=...)` message, reached under `--extended_logging` when a task's exact match falls by ≥ 0.2 with stable token accuracy. That one fires after an eval and before the rotating checkpoint, so it would end a long run with no crash checkpoint (#185).

Measured (audit probe `code-train-b/run_train.py`, runs `gf1` and `gf3`, real `train.main()` on CPU with stdout to a file): both logs end with `UnicodeEncodeError: 'charmap' codec can't encode character 'α' in position 18: character maps to <undefined>`, `batches 0 opt_steps 0`; the output directory holds only an empty `metrics.json`.

## Why it matters for the experiments

- Grokfast is the one grokking-specific optimiser option in the repo, and #137 says it has never been validly tested. With a redirected log on this machine it cannot start at all. The owner's deleted Grokfast run "didn't work" (#137); if its log was redirected, it never trained. That cannot be checked now.
- DAG-v12 ran with `extended_logging: true` on this Windows machine and completed, so either no `Δ` warning fired or its stdout was a console. Any future `--extended_logging` run under a harness carries the mid-run hazard.
- Both failure modes leave no useful artifact (#185), which is how a crash becomes "the run didn't work".

## How it got here

- PR [#45](https://github.com/jpturcotte/ReLSM/pull/45) (2025-12-17, "Add grokking dataset mode and Grokfast option") added the banner line with `α` and `λ`. Testing: "Not run".
- PR [#106](https://github.com/jpturcotte/ReLSM/pull/106) (2025-12-23) added the format-crash detector with `Δ` in its message. Testing: "Not run" (seven turns).
- Every documented run was launched from a console, where Python uses UTF-8 on Windows, so the characters never failed before this audit's harness redirected stdout.

## How we verified it, step by step

1. code-train-b (N1) ran `--grokfast --extended_logging` twice (accumulation 1 and 3) through a driver that redirects stdout to `runs/gf1.log` and `runs/gf3.log`; both died at `train.py:2384` before any batch.
2. code-train-b located the second occurrence at `train.py:1925` by reading and grepping; it was not triggered (needs an eval-time accuracy drop).
3. The coordinator read both lines with `cat -A` and confirmed the non-ASCII bytes; the crash was not re-run.
4. Not verified: whether the owner's deleted Grokfast run was launched with a redirected log.

## Suggested fix

- Use ASCII in every `print`/`log_print` string (`alpha=`, `lambda=`, `delta=`), or call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the top of `main()`. Setting `PYTHONUTF8=1` in the environment is a workaround, not a fix.
- Add a smoke test that runs `train.py --grokfast --extended_logging` for a few steps with stdout redirected to a file (the #188 CI smoke run). Independent of #137's filter fix; both are needed before any Grokfast result counts.

## Acceptance test

Pure Python, no GPU: for every string literal passed to `print`/`log_print` in `train.py`, `s.encode("cp1252")` succeeds (or, after the `reconfigure` fix, a tiny `--grokfast --extended_logging` CPU run with `PYTHONIOENCODING=cp1252` and stdout to a file exits 0 and writes a non-empty `metrics.json`).

## Related

- #137: Grokfast diverges with accumulation; this crash sits in front of that test.
- #185: crashes leave no checkpoint, which turns this into a silent failure.
- #188: the CI smoke run that should catch it.
- #184: CPU suite.
- #247: the other reason a Grokfast overflow reaches the weights.

## Sources

- Audit report code-train-b, finding N1; probe `code-train-b/run_train.py`, logs `runs/gf1.log`, `runs/gf3.log`
- VERIFIED_BY_COORDINATOR.md (non-ASCII characters row)
