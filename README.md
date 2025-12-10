# Unified Ablation Ladder for Recursive Architectures

## Overview

This codebase implements a systematic ablation study comparing recursive architectures (ReLSM-style) against standard transformers. It supports multiple scales and variants in a single unified framework.

**Research Question**: Does recursion/latent compute actually help, or is it just adding complexity?

## The Ablation Ladder

| Exp | Variant | What it tests | Key hypothesis |
|-----|---------|---------------|----------------|
| **0** | `baseline` | Standard transformer | Control model |
| **1** | `shared_loop` | Parameter-shared depth | Does weight reuse help? |
| **2** | `latent` | Dual-stream + thought tokens | Does latent scratchpad help? |
| **3** | `act` | Adaptive halting | Does adaptive K help more than fixed? |
| **4a** | `ssm` | Mamba-2 backbone | Does O(N) efficiency enable more iterations? |
| **4b** | `ssm_mem` | SSM + memory tokens | Can SSM + attention hybrid work? |

## Model Scales

| Size | d_model | Layers | Params | Purpose |
|------|---------|--------|--------|---------|
| `nano` | 512 | 6 | ~18M | Match ReLSM-Nano, quick experiments |
| `300M` | 1024 | 24 | ~300M | Fast ladder iteration |
| `1B` | 2048 | 18 | ~1B | ReLSM-16k comparison (target) |
| `1B-16k` | 2048 | 18 | ~1B | Long context with GQA |

## Quick Start

```bash
# Install
pip install torch transformers datasets

# Train nano baseline (quick test, ~10 min)
python train.py --model_size nano --variant baseline \
    --alg_tokens 10000000 --total_tokens 20000000 \
    --output_dir ./runs/nano_baseline

# Train nano with latent thought stream
python train.py --model_size nano --variant latent \
    --alg_tokens 10000000 --total_tokens 20000000 \
    --output_dir ./runs/nano_latent

# Evaluate
python evaluate.py --checkpoint ./runs/nano_baseline/best_model.pt
python evaluate.py --checkpoint ./runs/nano_latent/best_model.pt
```

## Split Curriculum Training

Following ReLSM's strategy:

### Phase 1: Algorithmic Grokking
- Synthetic logic, math, code
- Infinite procedural generation (no overfitting)
- Many epochs on synthetic data
- **Goal**: Force recursive core to learn algorithms

### Phase 2: Language Generalization  
- TinyStories, filtered web text
- Few epochs (1-3 passes)
- **Goal**: Map natural language to learned logic

```bash
# Example: 100M tokens algorithmic, then 400M tokens language
python train.py --model_size 300M --variant shared_loop \
    --alg_tokens 100000000 \
    --total_tokens 500000000 \
    --output_dir ./runs/300M_shared_loop
```

## Evaluation Metrics

The evaluation suite tests what actually matters for the research:

### 1. Algorithmic Exact-Match
Per-task accuracy on synthetic problems:
- Modular arithmetic
- Parity
- Addition/multiplication
- Copy/reverse sequences
- Dyck language (balanced parens)
- Chain arithmetic

### 2. OOD Length Generalization
**Key test for recursion**: Does the model generalize to longer sequences than training?

- Parity: trained on 8-bit, test up to 64-bit
- Addition: trained on 4-digit, test up to 8-digit

**Expected**: Recurrent/recursive models should extrapolate better than positional-embedding transformers.

### 3. Needle-in-Haystack Retrieval
Tests long-context memory:
- Insert secret code at various depths
- Test retrieval at 1k, 2k, 4k context

### 4. Perplexity
Language modeling quality on TinyStories validation.

## Running the Full Ladder

```bash
# === NANO SCALE (quick validation) ===

# Exp0: Baseline
python train.py --model_size nano --variant baseline \
    --alg_tokens 10000000 --total_tokens 20000000 \
    --output_dir ./runs/nano/exp0_baseline

# Exp1: Shared loop
python train.py --model_size nano --variant shared_loop \
    --alg_tokens 10000000 --total_tokens 20000000 \
    --output_dir ./runs/nano/exp1_shared_loop

# Exp2: Latent
python train.py --model_size nano --variant latent \
    --alg_tokens 10000000 --total_tokens 20000000 \
    --output_dir ./runs/nano/exp2_latent

# Exp3: ACT
python train.py --model_size nano --variant act \
    --alg_tokens 10000000 --total_tokens 20000000 \
    --output_dir ./runs/nano/exp3_act

# === COMPARE ===
for exp in exp0_baseline exp1_shared_loop exp2_latent exp3_act; do
    echo "=== $exp ==="
    python evaluate.py --checkpoint ./runs/nano/$exp/best_model.pt \
        --output ./runs/nano/$exp/eval_results.json
done
```

## Success Criteria

A variant "wins" if it achieves:

1. **≥5% improvement** on algorithmic accuracy vs baseline
2. **Better OOD generalization** (recursive should extrapolate)
3. **≤2× training time** (complexity must pay rent)
4. **No PPL collapse** (still generates coherent text)

## Key Implementation Details

### Thought Stream (Exp2/Exp3)

```python
# model.py - ThoughtCore
z = thought_tokens(B)           # (B, Z, thought_d)
for k in range(K):
    z = thought_core(z, h)      # Cross-attend to token stream
# Inject back
h = h + thought_to_token(z.mean(dim=1, keepdim=True))
```

### Adaptive Halting (Exp3)

```python
# model.py - ACT loop
cum_halt = zeros(B, 1)
for k in range(max_K):
    z = thought_core(z, h)
    p_halt = halt_head(z)
    cum_halt = cum_halt + (1 - cum_halt) * p_halt
    if (cum_halt >= 0.99).all():
        break
loss += lambda_ponder * ponder_cost
```

### Shared-Layer Looping (Exp1)

```python
# model.py - forward()
for unroll_idx in range(n_unroll):      # e.g., 24 iterations
    for layer_idx, layer in enumerate(self.layers):  # e.g., 6 unique layers
        x = layer(x, mask)
# Effective depth: 6 × 24 = 144 layers, but only 6 layers of params
```

## File Structure

```
unified/
├── model.py      # All variants in one file
├── data.py       # Split curriculum + evaluation data
├── train.py      # Training with curriculum support
├── evaluate.py   # Full evaluation suite
├── README.md     # This file
└── requirements.txt
```

## Hardware Requirements

### Nano (~18M params)
- Training: Any GPU, ~10-30 min
- Inference: CPU OK

### 300M
- Training: RTX 4060+ (~2-4 hrs)
- Inference: RTX 4060

### 1B
- Training: B200/A100 (~4-8 hrs)
- Inference: RTX 4060 (tight) or better

## Comparison to Proposals

This codebase synthesizes and tests ideas from:

### ReLSM
- ✅ Dual-stream (token + thought)
- ✅ Split curriculum (algorithmic → language)
- ✅ Adaptive halting (ACT)
- ✅ Memory tokens for long context

### RH-SSM
- ✅ SSM backbone option
- ✅ Mamba-2 integration (placeholder)
- ❌ "Plug-and-play memory" (underspecified in original)

### NEXUS
- ❌ TTT (too complex for constraints)
- ❌ Entropy patching (adds another model)
- ❌ Multiplicative gains (not empirically supported)

## Expected Results (Hypotheses)

| Metric | Baseline | Shared Loop | Latent | ACT |
|--------|----------|-------------|--------|-----|
| Algorithmic acc | 70% | 75% | 80% | 82% |
| OOD parity @32 | 20% | 40% | 60% | 65% |
| Needle @2k | 80% | 80% | 85% | 85% |
| PPL | 20 | 22 | 21 | 21 |
| Train time | 1× | 1.2× | 1.5× | 1.8× |

**If these hypotheses are wrong, that's valuable data.**

## Citation

This codebase is designed to test the core claims of:
- ReLSM (Recursive Latent-Stack Model)
- RH-SSM (Recursive Hybrid SSM)
- NEXUS (Neural EXponential Unified System)

The goal is empirical validation, not advocacy for any particular approach.

## License

MIT
