---
slug: warnings-silenced-without-extended-logging
title: Every log_print warning is a no-op unless --extended_logging is set
issue: 244
severity: P3
labels: [bug, P3-minor, area: tooling, telemetry, audit-2026-10-27]
nf: [NF-23]
found_by: [cli-commands, gh-prs-b]
status: Demonstrated
related_issues: [#185, #187, #173, #157]
---

## Summary

`log_print` is `print` only with `--extended_logging`; otherwise it is a lambda that discards its arguments. Every warning in `train.py` goes through it: the warmup-fraction warning, the exact-match crash warning, the task restriction notice, the resume notice, the config block, every `Curriculum:` line and "New best!". A default run prints `Loss | LR | Tok/s`, the `[eval]` headers (bare `print` in `eval_hub.py`) and `Eval duration`, and nothing else.

## What happens

- [train.py:1888](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1888): `log_print = print if args.extended_logging else (lambda *args, **kwargs: None)`.
- Silenced by default: `WARNING: warmup_steps=... is X% of training` ([train.py:2557](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2557)); `[warn] ... exact-match dropped ... Possible format/strictness issue` ([train.py:1923](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1923)); `Restricting algorithmic tasks to:` ([train.py:2061](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2061)); `Dynamic task weighting disabled` ([train.py:2037](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2037)); `Resumed from checkpoint` ([train.py:2531](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L2531)); every `Curriculum:` line ([train.py:3559](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3559)); the `Training config:` block; `New best! Saved checkpoint.` ([train.py:3293-3294](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L3293-L3294)).
- Audit run R2 (default flags, tiny CPU model): apart from library warnings, the console held four `Loss: ... | LR: ... (expected ...) | Tok/s: ...` lines, the `[eval] <task>/iid difficulty=0.500 samples=2` blocks and `Eval duration`. No step number, no phase, no config.
- Audit run R13 (`--warmup_steps 5`, 1,200-token budget): `WARNING: warmup_steps=5 is 71.4% of training` printed only because the flag was on.

## Why it matters for the experiments

- A user running the README commands cannot see which tasks are training, whether a resume took effect, or what configuration ran. #187 asks to save the arguments; until then the console block is the only record, and it is off by default.
- The two warnings written to catch bad runs early (71 % warmup, exact-match crash with stable token accuracy) never reach the person who needs them. DAG-v12 ran with `extended_logging: True`, so its log is complete; any run started from the README is not.

## How it got here

PR [#73](https://github.com/jpturcotte/ReLSM/pull/73) (2025-12-20, "Add lightweight training diagnostics and eval-probe metrics") added `--extended_logging` for the new diagnostics; its second commit gated every console print except the `Loss | LR` line behind the flag, including the step counter. The PR body lists the diagnostics only and does not mention the change to the default console (gh-prs-b 3c). PR [#117](https://github.com/jpturcotte/ReLSM/pull/117) restored `Tok/s` to the one default line. #185 asked that the interrupt and crash messages print regardless; the rest was not revisited.

## How we verified it, step by step

1. cli-commands (C11) ran the same tiny CPU run twice: R1 with `--extended_logging` (config block, step numbers, `Difficulty by task`, per-task eval lines, `Training complete!`) and R2 without (audit log `cli-commands/r2_default_verbosity.log`: only the lines listed above). R13 showed the warmup warning depends on the flag.
2. gh-prs-b read PR #73's two commits and confirmed the gating was introduced there without mention in the body.
3. Coordinator: not re-checked (a code read confirms `train.py:1888`).

## Suggested fix

Use `print` (or `warnings.warn` / a `log_warn` helper that always prints) for warnings, the config block, the resume notice and the task restriction; keep `log_print` for per-step detail and diagnostics. One-line change at each of the sites above. This extends the "print regardless of `--extended_logging`" item of #185 to all warnings.

## Acceptance test

Without `--extended_logging`: `python train.py --warmup_steps 5 --alg_tokens 1200 ...` prints the `WARNING: warmup_steps=` line, and `--alg_tasks parity` prints `Restricting algorithmic tasks to: parity`. A grep of `train.py` finds no `log_print(` whose string starts with `WARNING` or `[warn]`.

## Related

- #185: already asks for the interrupt and crash messages to print regardless.
- #187: run arguments are not saved; the console config block is the only copy today.
- #173: the `--warmup_steps` arithmetic that the silenced warning reports on.
- #157: the `Curriculum:` lines are the only live view of the gates.
- #235: "New best!" is silent too.

## Sources

- Audit report cli-commands, finding C11 and flag table row `--extended_logging`; logs `r1_baseline.log`, `r2_default_verbosity.log`, `r13_misc.log`
- Audit report gh-prs-b, per-PR table row #73 and section 3c
