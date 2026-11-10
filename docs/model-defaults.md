# Model defaults (changed 2026-10-28)

Two defaults changed on 2026-10-28: **dropout is now 0** and **the SwiGLU width follows Llama's rule**. The output head **stays tied** to the token embedding. The previous values are kept, documented, and selectable.

> **Correction, 2026-10-28 (same day).** The first version of this change (PR #273) also untied the head. It said "nobody knows why the previous values were chosen" and "Llama is untied". Both were wrong:
> - The chat history records who chose each value and, for three of them, why (see below).
> - Small Llamas are tied: Llama 3.2 1B and 3B ("Shared Embeddings: Yes"; the parameter totals match the tied count).
>
> The owner then restored the tied default. Sources: the audit session's provenance and reference reports, `model-defaults-provenance.md` and `model-defaults-references.md` (2026-10-28), in the owner's audit folder.

## What changed

| Setting | Before (until 2026-10-28) | Now | Why |
|---|---|---|---|
| Output head | tied to the token embedding | **tied (unchanged)** | Tying is the norm for small models in 2025–2026: Llama 3.2 1B/3B, Qwen3 up to 4B ("Tie Embedding: Yes" for 0.6B–4B, "No" from 8B; arXiv 2505.09388 Table 1), Gemma 3, SmolLM2/3, Phi-4-mini, MobileLLM. MobileLLM tested nano's shape (d 512, 32k vocab): sharing cost 0.2 points and saved 16M parameters, and spending them on depth recovered the points (arXiv 2402.14905). Untied, nano would hold 61.6% of its parameters in the two vocabulary tables. `tie_weights=False` (`--untie_embeddings`) unties it. |
| Dropout | 0.1 (0.05 in the 300M, 1B and 1B-16k presets) | 0.0 | Llama and the small models above use none. For grokking, Power et al. 2022 found weight decay beats dropout, and Nanda et al. 2023 found dropout alone groks more slowly and less cleanly. |
| SwiGLU width (`d_ff`, when a preset does not fix it) | 4 × `d_model` | Llama's rule: `int(2 × 4 × d_model / 3)` rounded up to a multiple of 256 | A parameter-budget choice, not a quality fix: Llama 3.2 1B itself uses a 4× SwiGLU. The rule gives 3.0× at d 512. |
| 300M preset `d_ff` | 4096 (4×) | 2816 (the rule) | as above |

These already matched and are unchanged: RMSNorm ε 1e-6, RoPE base 10,000, init N(0, 0.02), no biases, pre-norm blocks, SwiGLU, final RMSNorm. The 1B and 1B-16k presets keep `d_ff = 5504`, which is the Llama rule with `multiple_of=64`.

### Width per preset

| Preset | `d_model` | `d_ff` before | `d_ff` now |
|---|---|---|---|
| nano, 50M | 512 | 2048 | 1536 |
| 125M | 768 | 3072 | 2048 |
| 300M, 350M | 1024 | 4096 | 2816 |
| 760M | 1280 | 5120 | 3584 |
| 1B, 1B-16k | 2048 | 5504 | 5504 |
| grokking size (`d_model` 128) | 128 | 512 | 512 (unchanged) |

**Parameter counts:** the narrower SwiGLU removes `3 × n_layers × d_model × (old − new d_ff)` parameters, which is 4.7M at nano. Nano goes from 41.6M to about 36.9M. README figures were already wrong (#170).

## Where the previous values came from

From the audit session's provenance report (2026-10-28), checked against git and byte-identical copies of the files. Times are UTC.

- **Two baselines on 2025-12-09.** The owner built two baselines that evening, both starting from Claude's GPT-2-style `baseline_v2` (125M; 4× width, dropout 0.1, tied).
  - Claude's revision, the "unified ablation ladder", became the initial commit `6ba960e`. It used a 2.67× width, stated in its code as "SwiGLU has 3 matrices, so reduce expansion for param parity". It was always tied, with dropout 0.1.
  - ChatGPT's revision of the older `baseline_v2`, `model_updated.py` (chat 6938c171, 23:40), was pasted through Codex 27 minutes later as PR #1 (`31356c1`). It is byte-identical to that commit. It brought back 4× width, made tying a switch defaulting to on, and added the presets' `dropout=0.05` and the 300M preset's `d_ff=4096`.
- **Tied head:** Claude, Dec 9 00:40 (chat f50fd8de): tying "nearly halves the embedding-related parameters—a standard trick that works well." Every Dec 9 baseline kept it. No owner prompt named it.
- **Dropout 0.1 / 0.05:** Claude's code had 0.1, with no stated reason. ChatGPT, Dec 9 20:46, advised "Dropout: 0.05–0.1 (yes, even if you'd normally run 0)" as regularisation for a 1B model on 2B tokens, not for grokking. A Dec 19 Deep Research report advised dropout 0 for grokking, but `train.py` had no `--dropout` flag. So every December run, the grokking runs included, used 0.1.
- **4× width:** no stated reason; it is GPT-2's MLP habit, kept from `baseline_v2`. Claude's own later file had moved to 2.67×. With 2.67×, nano would have had about 18.9M non-embedding parameters, the README's "~18M".
- **Decay exemption for the tied table** (Dec 11, ChatGPT 693b3166): exempted because of a "lexical collapse" concern, which is now #186.

## Using the previous setup

- **Python:**
  - `create_model(size, legacy_defaults=True)` rebuilds the pre-change setup exactly, including the old per-preset values (`LEGACY_DEFAULTS` and `LEGACY_PRESET_VALUES` in `model.py`).
  - Individual settings can be overridden, for example `create_model(size, tie_weights=False)` for an untied head. Explicit arguments win over presets and over `legacy_defaults`.
- **train.py:**
  - `--legacy_model_defaults` restores the old setup.
  - The individual flags are `--tie_embeddings` / `--untie_embeddings`, `--dropout`, and `--ffn_rule {llama,4x}`.

## What this does not break

- **Old checkpoints still load and evaluate unchanged.** `utils.load_model_from_checkpoint` rebuilds the model from the `config` saved in the checkpoint, so DAG-v12 loads with its 4× width and dropout 0.1 and gives the same outputs. This is tested (`test_old_style_checkpoint_loads_with_its_own_settings`).
- **Resuming an old run fails loudly instead of silently.** `check_resume_architecture` compares the checkpoint's saved settings (width, tying, dropout and the shape fields) with the run's, and stops with a message naming `--legacy_model_defaults`.
- **#145, the factory half:** `create_model` builds only the requested preset and lets keyword arguments override it. #145's `shared_loop` depth flags in `train.py` are not part of this change.
- **Weight decay (#186):** `build_optimizer` exempts the tied table from weight decay, as before. The grokking recipe (#249) decays every parameter through `--wd_scope all` in the fix plan.

## Checked against the reference

`tests/test_baseline_matches_llama.py` loads identical weights into `BaselineTransformer` and Hugging Face's `LlamaForCausalLM`, then compares logits and gradients. It covers 7 settings:
- the grokking size, tied and untied;
- the nano shape;
- grouped-query attention;
- biases;
- a different RoPE base and width;
- 256-token sequences.

They agree to float32 rounding (relative difference below 1e-6). The file also checks causality, right-padding invariance with a distinct pad id (#232), KV-cache decoding against a full forward pass, and memorisation of 512 random labels. It runs in about 10 s on CPU.

## For the grokking experiments

The first-run plan (`docs/research/grokking-fix-plan-2026-10-28.md`, §3.2):
- The **primary arm** is the standard recipe with the repo's tied head and dropout 0.
- The **untied-head arm** is the comparison. Power 2022 and Nanda 2023 are untied, but with vocabularies of about 115 tokens, and no paper compares tied and untied heads for grokking.
- An optional **"old ReLSM model" arm** (tied, dropout 0.1) tests the setup every ReLSM run so far used.
- **The width rule does not matter at the grokking size** (512 either way). It matters at nano and above: #252 stage 2, #253 and the #258 `shared_loop` ladder.
