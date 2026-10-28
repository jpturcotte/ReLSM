---
slug: latent-hazards-unreachable-today
title: Five latent hazards no current flag reaches, in eval helpers and the model
issue: 239
severity: P3
labels: [bug, P3-minor, area: evaluation, area: model, tech-debt, audit-2026-10-27]
nf: [NF-45, NF-04]
found_by: [code-eval, code-model, codex-a, claude-b2]
status: Demonstrated
related_issues: [#184, #147, #172, #146, #176, #163]
---

## Summary

A checklist, in the style of #180, of five defects that no current flag, grid condition or caller reaches, so they change nothing today. Each is a one-line fix and a one-line invariant for the #184 suite. Three were demonstrated by probe (a, c, d); two by reading (b, e). Left alone, each is a trap for the next diagnostic, test or variant.

## What happens

1. **`build_dataset` raises `TypeError` if a dyck condition carries `valid_ratio`.** [utils.py:970-974](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L970-L974): `kwargs = dict(params)` keeps the key, `params.get("valid_ratio", 0.5)` reads it, and `dyck_language` ([data.py:607](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/data.py#L607)) has no such argument. Probe: `build_dataset("dyck", 2, {"max_depth": 5, "valid_ratio": 0.5}, 1)` -> `TypeError: ... got an unexpected keyword argument 'valid_ratio'`. No grid condition sets the key, so every dyck condition runs at the 0.5 default and the knob cannot be used.
2. **An unknown `suite` writes a meta-only results file.** [eval_hub.py:761-763](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L761-L763): `suites_to_run = [suite]`; the loop at [eval_hub.py:792-830](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/eval_hub.py#L792-L830) has no `else`, so an unmatched string runs nothing and still calls `save_json` on `{"meta": ...}`. The CLI restricts `--suite` to four choices; the Python API does not. Plausible (not executed).
3. **`_resolve_max_length` right-truncates long prompts and drops `Answer:`.** [utils.py:404-416](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L404-L416) returns `config.max_seq_len`; `generate_text` ([utils.py:534-543](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L534-L543)), `batch_generate` ([:575-577](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L575-L577)), `compute_perplexity` ([:699-700](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L699-L700)) and `generate_batch` ([:1051-1053](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L1051-L1053)) pass it with `truncation=True`. The Llemma tokenizer truncates on the right (`truncation_side=right`). Probe at `max_length=8`: the prompt becomes `<s> ▁Comput e ▁ 1 2 3 4`, the `Answer:` cue is gone, silently. Unreachable today: the longest IID+OOD grid prompt is 58 tokens (codex-a) against `max_seq_len` 128 (DAG runs) or 1,024.
4. **`CausalDepthwiseConv1d` with `kernel_size=1` caches the whole input.** [model.py:505](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L505): `x_pad[..., -(self.kernel - 1):]` is `x_pad[..., -0:]`, the full tensor. Probe: cache shape `(1, 4, 3)` after chunk 1 and `(1, 4, 6)` after chunk 2, expected `(1, 4, 0)`; outputs stay correct (diff 0.0) because a 1-tap conv is pointwise. `ssm_kernel` is not on the CLI.
5. **`Attention.forward` keeps a padding-only branch that drops causality.** [model.py:319-322](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L319-L322): `elif padding_mask is not None: mask = padding_mask`; [model.py:331](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L331) then runs SDPA with `is_causal=False`, full bidirectional attention. Dead today because `forward()` builds the causal mask whenever padding exists ([model.py:852-856](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L852-L856)) and both callers pass both masks ([:874](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L874), [:886](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L886)). Any caller that passes only a padding mask gets a non-causal model with no error. Plausible (read).

## Why it matters for the experiments

None of the five changes a number at HEAD. Each is the kind of defect the Oct-26 review found live elsewhere: (1) and (2) are silent no-ops of the shape of #163 and #176; (3) is the shape of #139 (silent truncation); (4) is the shape of #146 (cache semantics); (5) is the shape of #135 and #172 (a mask path that loses causality). A one-line test for each costs less than one more day lost to a diagnostic that attends to the future.

## How it got here

- (1) PR [#89](https://github.com/jpturcotte/ReLSM/pull/89) ([task](https://chatgpt.com/remote/task_e_6947504b85d8832ebe1c3c9a01ff6e8d), 2025-12-21, "randomize Dyck generation") added the `valid_ratio` draw without popping the key.
- (2) PR [#20](https://github.com/jpturcotte/ReLSM/pull/20) (2025-12-12) built `run_eval_suite` around a fixed list; PR [#44](https://github.com/jpturcotte/ReLSM/pull/44) added the per-suite save that writes even when nothing ran.
- (3) PR [#33](https://github.com/jpturcotte/ReLSM/pull/33) (2025-12-13, "Harden evaluation tokenization for custom models") added `_resolve_max_length` and the `truncation=True` calls.
- (4) The depthwise conv came with PR [#1](https://github.com/jpturcotte/ReLSM/pull/1) and PR [#10](https://github.com/jpturcotte/ReLSM/pull/10); the `-(kernel - 1)` slice was never tested at kernel 1.
- (5) The Dec-31 Claude chat found "Lost Causal Masking When Only Padding Mask Present" and proposed building the causal mask upstream and removing the fallback; PR [#126](https://github.com/jpturcotte/ReLSM/pull/126) ([task](https://chatgpt.com/remote/task_e_6955827d86ac832ea0c8f051653eb80b)) did the first and kept the fallback (claude-b2, chat 3 row 8).

## How we verified it, step by step

1. code-eval (E8) ran `build_dataset("dyck", 2, {"max_depth": 5, "valid_ratio": 0.5}, 1)` in audit probe `code-eval/metrics_probe.py`: `TypeError`. (E11) read `run_eval_suite`; not executed.
2. code-model (CM-10) ran audit probe `code-model/p5_generate_utils.py`: `_resolve_max_length -> 8` on a tiny config, right-truncated prompt shown; `truncation_side=right`. codex-a measured the longest grid prompt (58 tokens) and confirmed no truncation on the DAG checkpoints.
3. code-model (CM-4) ran audit probe `code-model/p4_ssm_init.py`: kernel 7 cache `(1, 4, 6)` both chunks; kernel 1 cache `(1, 4, 3)` then `(1, 4, 6)`; outputs equal to the full forward in both cases.
4. claude-b2 (B2-5) read `model.py:319-331` and the two call sites, and matched them to the Dec-31 chat's finding and proposed fix.
5. Coordinator: not re-checked. No disagreements.

## Suggested fix

- [ ] (1) `kwargs.pop("valid_ratio", None)` before calling the generator.
- [ ] (2) `else: raise ValueError(f"unknown suite {selected!r}")` in the loop; do not save when `results` is empty.
- [ ] (3) Refuse prompts longer than `max_seq_len - max_new_tokens` (raise) or set `tokenizer.truncation_side = "left"` for eval; log a count of truncated prompts in `meta`.
- [ ] (4) `new_cache = x_pad[..., x_pad.size(-1) - (self.kernel - 1):]`, or assert `kernel >= 2`.
- [ ] (5) Replace the `elif` with `assert mask is not None` or build the causal mask inside `Attention.forward` when `mask is None and padding_mask is not None`.

(4) and the conv init gap belong to the same module as #147. None conflicts with another issue's fix.

## Acceptance test

Five invariants for the #184 CPU suite: (1) the dyck call above returns one example; (2) `run_eval_suite(..., suite="nope")` raises and writes no file; (3) a 1,030-token prompt at `max_seq_len` 1,024 raises, or keeps its last token; (4) `CausalDepthwiseConv1d(kernel_size=1)` cache has shape `(B, C, 0)` after two chunks; (5) `Attention.forward(x, mask=None, padding_mask=pm)` equals `forward(x, mask=causal, padding_mask=pm)` to 1e-6 on an unpadded batch.

## Related

- #184: the invariant suite these checks belong in.
- #147: `_init_weights` does not touch the same `Conv1d`.
- #172, #146: cache and mask paths in the same module.
- #176, #163: existing silent-no-op issues in dyck and long-context eval.
- #180: the sibling checklist of dead code.
- #234: a diagnostic that already rebuilds attention outside `forward()`, which is where item 5 would bite.

## Sources

- Audit report code-eval, findings E8 and E11; probe `metrics_probe.py`
- Audit report code-model, findings CM-4 and CM-10; probes `p4_ssm_init.py`, `p5_generate_utils.py`
- Audit report codex-a, verified non-findings (58-token longest prompt)
- Audit report claude-b2, finding B2-5 and working notes on chat 3 (Dec 31)
- Audit report gh-prs-a, per-PR table rows #20, #33, #44; gh-prs-b rows #89, #126
