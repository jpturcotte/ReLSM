# Model defaults (changed 2026-10-28)

The model defaults now follow the Llama reference. The previous values are kept, documented, and selectable. **Nobody knows why the previous values were chosen:** no commit message, PR, issue or chat records a reason.

## What changed

| Setting | Before (until 2026-10-28) | Now | Reference |
|---|---|---|---|
| Output head | tied to the token embedding (`tie_weights=True`) | separate (`tie_weights=False`) | Llama (HF `LlamaConfig.tie_word_embeddings=False`); also the grokking references: Power et al. 2022 and Nanda et al. 2023 are untied |
| Dropout | 0.1 (0.05 in the 300M, 1B and 1B-16k presets) | 0.0 | Llama (HF `attention_dropout=0.0`; no residual or FFN dropout) |
| SwiGLU width (`d_ff`, when a preset does not fix it) | 4 × `d_model` | Llama's rule: `int(2 × 4 × d_model / 3)` rounded up to a multiple of 256 | Meta's Llama code (`multiple_of=256`); for 4096 it gives 11008, the HF default |
| 300M preset `d_ff` | 4096 (4×) | 2816 (the rule) | as above |

These already matched the reference and are unchanged: RMSNorm ε 1e-6, RoPE base 10,000, init N(0, 0.02), no biases, pre-norm blocks, SwiGLU, final RMSNorm. The 1B and 1B-16k presets keep `d_ff = 5504`, which is the Llama rule with `multiple_of=64`.

### Width per preset

| Preset | `d_model` | `d_ff` before | `d_ff` now |
|---|---|---|---|
| nano, 50M | 512 | 2048 | 1536 |
| 125M | 768 | 3072 | 2048 |
| 300M, 350M | 1024 | 4096 | 2816 |
| 760M | 1280 | 5120 | 3584 |
| 1B, 1B-16k | 2048 | 5504 | 5504 |
| grokking size (`d_model` 128) | 128 | 512 | 512 (unchanged) |

### Parameter counts

- **Untying adds `vocab_size × d_model` parameters.** With the 32,016-token Llemma vocabulary at nano that is 16.4M. The narrower SwiGLU removes `3 × n_layers × d_model × (old − new d_ff)`, which is 4.7M at nano.
- README figures predate this change and were already wrong (#170).

## Where the previous values came from

- **Initial commit, `6ba960e` (2025-12-09):** the model was always tied (`self.lm_head.weight = self.tok_emb.weight`), with dropout 0.1. Its SwiGLU width was `int(2.67 × d_model)`, which is Llama-like.
- **PR #1, `31356c1` (2025-12-09, "Integrate ladder variants and long-context evaluation"):** a pasted "ablation-ladder" `model.py` replaced it. That version made tying a switch defaulting to on, changed the width to `4 × d_model`, and added the presets' explicit `dropout=0.05` and the 300M preset's `d_ff=4096`. The project history (`docs/review-2026-10-26/project-history-codex.md`) records the paste but not its source or reasoning.
- **No later PR changed these values.** Every training run before 2026-10-28, DAG-v12 included, used them, because `train.py` had no flag for tying or dropout.

## Using the previous setup

- **Python:**
  - `create_model(size, legacy_defaults=True)` rebuilds the pre-change setup exactly, including the old per-preset values (`LEGACY_DEFAULTS` and `LEGACY_PRESET_VALUES` in `model.py`).
  - Individual settings can be overridden, for example `create_model(size, tie_weights=True)`. Explicit arguments win over presets and over `legacy_defaults`.
- **train.py:**
  - `--legacy_model_defaults` restores the old setup.
  - The individual flags are `--tie_embeddings` / `--untie_embeddings`, `--dropout`, and `--ffn_rule {llama,4x}`.

## What this does not break

- **Old checkpoints still load and evaluate unchanged.** `utils.load_model_from_checkpoint` rebuilds the model from the `config` saved in the checkpoint, so DAG-v12 loads tied, with 4× width and dropout 0.1, and gives the same outputs. This is tested (`test_old_style_checkpoint_loads_with_its_own_settings`).
- **Resuming an old run fails loudly instead of silently.** Before, `train.py`'s resume path rebuilt the model from the current defaults; with the new defaults that would fail on the width, or load silently and untie a tied model. `check_resume_architecture` now compares the checkpoint's settings and stops with a message that names the flag to use (`--legacy_model_defaults`).
- **#145, the factory half:** `create_model` used to build every preset eagerly, so passing `dropout=` raised `TypeError` for every size. It now builds only the requested preset and lets keyword arguments override it. #145's `shared_loop` depth flags in `train.py` are not part of this change.

### Interaction with weight decay (#186)

`build_optimizer` exempts parameters whose names contain `tok_emb` or `lm_head` from weight decay. With a tied head that exempted one tensor. With the untied default it exempts two: the embedding and the head. The grokking recipe (#249) decays every parameter and needs `--wd_scope all` from the fix plan; the default exemption itself is #186's question.

## Checked against the reference

`tests/test_baseline_matches_llama.py` loads identical weights into `BaselineTransformer` and Hugging Face's `LlamaForCausalLM`, then compares logits and gradients. It covers 7 settings:
- the grokking size, tied and untied;
- the nano shape;
- grouped-query attention;
- biases;
- a different RoPE base and width;
- 256-token sequences.

They agree to float32 rounding: relative difference below 1e-6 on the 2026-10-28 run, with a test tolerance of 1e-5 for logits and 1e-4 for gradients. The file also checks causality, right-padding invariance with a distinct pad id (#232), KV-cache decoding against a full forward pass, and memorisation of 512 random labels. The whole file runs in about 10 s on CPU.

## For the grokking experiments

The first-run plan (`docs/research/grokking-fix-plan-2026-10-28.md`, §3.2) was written when the default was tied. With the new defaults:
- The **primary arm** uses the default untied head and dropout 0, matching Power, Nanda and Llama.
- The **"tied head" arm** is the comparison: does tying hurt grokking?
- An optional **"old ReLSM model" arm**, tied with dropout 0.1, tests the previous setup as a whole. It is relevant because every ReLSM run so far used it.
- **The width rule does not matter at the grokking size** (512 either way). It matters at nano and above: #252 stage 2, #253 and the #258 `shared_loop` ladder.
