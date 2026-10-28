---
slug: unused-embedding-rows-dominate-weight-norm
title: "DAG-v12 embedding-norm growth is 31,879 unused rows drifting, then frozen by eps"
issue: 218
severity: P2
labels: [bug, P2-important, area: training, telemetry, baseline-affected, audit-2026-10-27]
nf: [NF-02]
found_by: [code-model, artifacts, lexical]
status: Demonstrated
related_issues: [#186, #149, #187, #189, #184, #170, #137]
---

## Summary

The head is tied to the embedding, so every one of the 32,016 vocabulary rows gets a gradient through the softmax denominator on every step, whether or not its token is in the batch. AdamW normalises that tiny, sign-consistent push into a step of about `lr`, and `tok_emb` is exempt from weight decay. In DAG-v12 the curriculum uses 137 token ids; the other 31,879 rows moved together to one direction, reached norm 1.08 (init 0.45), and now carry 98.7% of the embedding's squared norm. The Adam state in the same checkpoint shows the drift was early and self-limiting: once the dead logits sat near −20, the gradient fell below `eps` and the rows froze (update ratio 0.000; relative change 8.7e-4 over the last 9,500 micro-steps). The logged `weight_norm.embedding` series (81 → 175 by step 10k, 194 at the end) is this event, and the decayed blocks grew more (129 → 307). So the total weight norm is not a grokking signal for this model, and #186's reading that the norm grew "because embeddings were exempt" is at most half the story.

## What happens

Mechanism:

- [model.py:703-704](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L703-L704): `lm_head.weight = tok_emb.weight` (tied by default). [model.py:900](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/model.py#L900): the logits are `lm_head(x)`, so for an absent token v, `grad(row_v) = Σ_t p_t(v) h_t`, never zero.
- [train.py:1817-1818](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1817-L1818): `tok_emb` and `lm_head` are in the no-decay group (#186). [train.py:1844](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/train.py#L1844): AdamW, betas (0.9, 0.95), eps 1e-8.

Measured on DAG-v12 `final_model.pt` (step 77,503, Adam step 6,458; audit probes `code-model/p3_embed_grad.py` and `artifacts/p3_checkpoints.py`):

| quantity | used rows (137) | unused rows (31,879) |
|---|---|---|
| mean row norm (init 0.453) | 1.893 | 1.082 (std 0.008, range 1.054–1.134) |
| cosine to the unused rows' mean direction | 0.107 | 0.915 (min 0.777) |
| share of embedding squared norm | 1.3% | 98.7% |
| Adam `exp_avg_sq` median | 1.7e-8 | 1.7e-23 (mean 1.6e-22, max 4.1e-18) |
| Adam `\|exp_avg\|` mean | 2.4e-3 | 4.0e-11 |
| mean `\|m\| / (sqrt(v) + eps)` | 0.17 | 0.000 |
| relative change, step 68,000 → 77,503 | 8.2e-2 | 8.7e-4 |

- With `sqrt(v)` ≈ 4e-12 ≪ eps 1e-8, a dead row's per-element step is `lr·m/eps` ≈ 4e-3·lr: nothing. Along the common direction, with `|h| = |norm.weight| = 20.5`, a dead token's logit is about −20 (p ≈ 2e-9).
- The row-norm histogram has 31,882 rows below 1.26 and 134 above, matching the 137 used ids to within 3.
- Logged `weight_norm.embedding`: 81.0 (step 50), 175.2 (10k), 186.4 (20k), 190.4 (30k), 193.7 (50k), 194.45 (77.5k). The drift happened in the first ~10k micro-steps (~830 updates) and is flat after 20k.
- Logged global norm (tied tensor counted once): 152.6 → 358.1; its non-embedding part 129.3 → 307.2 (50k) → 300.7. The 42 decayed tensors grew by 178 in norm, the 14 exempt ones (embedding + 13 RMSNorm) by 113.
- Weight decay 0.1 at peak lr 1.5e-3 removes 1.5e-4 per update; over 6,458 updates the cumulative shrink is `exp(−0.1·Σlr)` ≈ 0.53. Decay could at most halve a weight with no gradient, and could not act during the ~800 updates in which the drift happened. The 2% fall from 50k to 77.5k during LR decay is its only sign.
- The dead rows' momentum points partly back along the common direction (cos −0.6), consistent with an overshoot that stopped.

Scale checks:

- Tiny tied model (vocab 256, tokens 0–15 in the batch): 241 of 241 absent rows have non-zero gradient (mean norm 7.2e-3 vs 3.5e-1 for present rows); untied, 0 of 241 (`code-model/p3_embed_grad.py` (a)).
- Tiny AdamW run (vocab 2,000, 20 used): unused-row norm 0.159 → 4.232 in 1,500 steps, cosine 0.999, no plateau. That probe never reached the eps regime; DAG-v12 did.
- Nano with the Llemma tokenizer and real batches (lexical, `rung1g`): 138 of 32,016 ids used; unused rows 0.452 → 1.371 (3.0×) in 2,000 updates with the table exempt, 1.038 (2.3×) with the table decayed; cosine 0.898 vs 0.984. `adam_sqrtv_unused` is below eps from about update 200. Decaying the table tightens the collapse direction; it does not prevent it.

## Why it matters for the experiments

- The counter-review's grokking criterion ("the weight norm falls") and #186's table read the total and embedding norms as a signal about the circuit. 39.4% of nano's parameters are a block whose norm is dominated by rows that encode nothing, moved once in the first 830 updates and then stopped. No norm-based grokking reading of DAG-v12 is possible from the logged series.
- #186's "How it got here" rationale ("embeddings drifting because most tokens get no gradient") has the sign wrong for a tied model: the dead rows drift because they get a gradient every step. Exempting them from decay makes the drift worse, not better; and decaying them would not stop the direction collapse (nano cosine 0.984 with decay). The #186 switch is justified by literature parity, not by this mechanism.
- Any later language phase starts from 31,879 rows pushed to one corner of embedding space with logits near −20: the "lexical collapse" the Dec-11 design feared, produced by the hedge chosen against it. The lexical research sweep (PR #196) finds that gating never-seen rows plus initialising a row at first use beats both a reset and decay; a one-time reset of never-used rows only matches a fresh model.
- The PR #196 toy (0.159 → 4.232, cosine 0.999) overstates the speed; at 32k rows the drift is eps-limited after ~200 updates. The operative claim is the direction collapse, which holds with or without decay.

## How it got here

- PR [#13](https://github.com/jpturcotte/ReLSM/pull/13) (2025-12-11, "Refine optimizer parameter grouping") put `tok_emb`/`lm_head` in the no-decay group after the owner pasted a review citing the research plan's "lexical collapse" concern (details in #186). The hedge assumed absent tokens get no gradient.
- PR [#48](https://github.com/jpturcotte/ReLSM/pull/48) (2025-12-18, "Default to llemma tokenizer") set the 32,016-row vocabulary of which the curriculum uses 137 ids. Gemini's Dec-17 "Embedding Tax" report had recommended 16,384 the same day (comment on #186).
- The tied head is the model's default since PR [#1](https://github.com/jpturcotte/ReLSM/pull/1) (`tie_weights: bool = True`).
- No Codex task or chat asked what AdamW does to unused rows under a tied head. The lexical research worktree (PR [#196](https://github.com/jpturcotte/ReLSM/pull/196), open) found the collapse in a toy in Oct 2026; this audit measured it in the checkpoint.

## How we verified it, step by step

1. code-model (CM-2) derived the softmax-denominator gradient, demonstrated it on a tied vs untied tiny model (241/241 vs 0/241 absent rows with gradient), ran the tiny AdamW drift (0.159 → 4.232, cosine 0.999), then opened `final_model.pt` read-only (mmap) and split `tok_emb.weight` by the 137 ids used in 3,000 generated curriculum examples: norms, cosines, 98.7% share, logit ≈ −20 (`code-model/p3_embed_grad.py`, `p3_out.txt`).
2. artifacts (AR-5) opened the same checkpoint's `optimizer_state_dict` and split the Adam moments by 135 ids from 2,000 examples: `exp_avg_sq` median 1.7e-23 for unused rows, update ratio 0.000; compared `best_model.pt` (step 68,000) with `final_model.pt`: unused rows changed 8.7e-4 relative, used rows 8.2e-2 (`artifacts/p3_checkpoints.py`, `p3_out.txt`). AR-6 read the logged `weight_norm` and `weight_norm.embedding` series from `metrics.0.json` and computed the non-embedding part and the decay bound (`artifacts/p2_timeline.py`, `p2_out.txt` A4).
3. lexical (L1) reproduced the 1.08 unused-row norm by arithmetic from the values quoted in #186 and the `rung1g` usage count (1.09 estimated vs 1.082 measured), reran the PR #196 toy probe, and read the nano-scale `rung1g` sweep (3.0× / 2.3×, cosine 0.898 / 0.984, eps dominance from update ~200) and the `rung2g` sweep ranking (L2, L5; audit probe `lexical/unseen_token_rows_probe_copy.py`).
4. The coordinator re-read `code-model/p3_out.txt` (137 of 32,016 ids; 1.082 ± 0.008; 0.987; 0.915) and confirmed the drift; the eps freeze rests on artifacts' probe.
5. Disagreement: code-model wrote that 6,458 updates at `lr·wd` cannot counter a drift of `lr` per step; artifacts' Adam-state reading refines this to "decay cannot act during the ~800 updates in which the drift happens, and after the freeze there is nothing to counter". This draft uses the second form.

## Suggested fix

- Telemetry first: log `weight_norm.embedding_used`, `weight_norm.embedding_unused` and `n_used_ids` (a boolean per token id, updated from each batch), plus the decayed-tensor norm separately. Without the split no norm-based criterion can be read. This is the #149 telemetry fix.
- Decide the vocabulary contract: a logits mask restricted to the ids the curriculum can emit, or a compact vocabulary (~200 tokens for the algorithmic phase). Either removes the early drift and 16.3M dead parameters from the effective model.
- If the full vocabulary stays: gate the gradient of token rows that have not appeared yet (one boolean mask, re-applied after Grokfast), initialise a row at its first use and clear its Adam/Grokfast state; for checkpoints trained without gating, reset the never-used rows once at the switch (lexical §5, from the PR #196 `rung2g` sweep). Do not rely on table decay or a table-only eps for this purpose.
- Rewrite #186 as the lexical report proposes: keep the `--decay_embeddings` / `--untie_head` switch for literature parity, and state that the exemption does not protect unseen tokens. This draft owns the dead-row mechanism; the switch stays in #186. Record the decision in the ADR (#189).

## Acceptance test

CPU, tied tiny model (`d_model=32`, vocab 256), AdamW with the repo's `build_optimizer`, 200 updates on batches drawn from tokens 0–15 only. With unseen-row gating (or a logits mask) on: the mean norm of rows 16–255 equals its init value to 1e-6 and their cosine to their mean direction is below 0.1. With it off (the control): the unused-row norm exceeds 2× init. The logged metrics contain `weight_norm.embedding_used` and `weight_norm.embedding_unused`.

## Related

- #186: the decay exemption and its switch; its rationale is inverted by this finding and the lexical report's rewrite text applies.
- #149: weight-norm telemetry must split used rows / unused rows / decayed tensors.
- #187: record `n_used_ids` and the optimizer group sizes in the run record.
- #189: the vocabulary and pad ADR.
- #184: the gating invariant belongs in the CPU suite.
- #170: the 41,564,672 count of which 16,392,192 is the table.
- #137: Grokfast's EMA must respect the row gate.
- #232: the other half of the vocabulary contract.

## Sources

- Audit report code-model, finding CM-2; probe `code-model/p3_embed_grad.py`
- Audit report artifacts, findings AR-5 and AR-6, §2b; probes `artifacts/p3_checkpoints.py`, `artifacts/p2_timeline.py`
- Audit report lexical, findings L1, L2, L5, §5 recipe, §6 #186 rewrite, §7 draft 1; probe `lexical/unseen_token_rows_probe_copy.py`; sweep snapshots `rung1g_baseline_snapshot.jsonl`, `rung1g_decay_table_snapshot.jsonl`, `rung2g_results_snapshot.jsonl`
- VERIFIED_BY_COORDINATOR.md (embedding drift row)
- Issue #186 body and its 2026-10-27 comment; PR #196 body
