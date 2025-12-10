"""
Unified Transformer Baseline + Variants
========================================
Supports the full ablation ladder:

SCALES:
  - Nano (~18M): ReLSM-Nano comparison, algorithmic tasks
  - 300M: Fast iteration
  - 1B: ReLSM-16k comparison, full target

VARIANTS (--variant flag):
  - baseline: Standard transformer (Exp0)
  - shared_loop: Parameter-shared depth (Exp1)
  - latent: Dual-stream with thought tokens (Exp2)
  - act: Adaptive halting on thought loop (Exp3)
  - ssm: Mamba-2 backbone (Exp4a)
  - ssm_mem: SSM + memory token attention (Exp4b)

The baseline variant with 1B config is the CONTROL MODEL.
All other variants must beat it by ≥5% on target metrics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal


# =============================================================================
# CONFIGURATION
# =============================================================================

VariantType = Literal["baseline", "shared_loop", "latent", "act", "ssm", "ssm_mem"]

@dataclass
class ModelConfig:
    """Unified configuration for all model variants."""
    
    # Architecture
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 2048
    n_layers: int = 18
    n_heads: int = 16
    d_ff: Optional[int] = None  # Defaults to ~2.67 * d_model (SwiGLU adjusted)
    dropout: float = 0.1
    bias: bool = False
    rope_base: int = 10000
    
    # Variant selection
    variant: VariantType = "baseline"
    
    # Shared loop (Exp1)
    n_unique_layers: int = 6       # Number of unique layer blocks
    n_unroll: int = 24             # Times to unroll (effective depth)
    
    # Latent thought stream (Exp2/Exp3)
    n_thought_tokens: int = 16     # Number of thought tokens
    thought_d_model: int = 512     # Thought stream dimension (smaller for L2)
    K: int = 8                     # Fixed inner loop iterations
    
    # ACT (Exp3)
    max_K: int = 16                # Maximum iterations
    min_K: int = 2                 # Minimum iterations (stability)
    lambda_ponder: float = 1e-4   # Ponder cost weight
    halt_threshold: float = 0.99   # Cumulative halt threshold
    
    # Memory tokens for long context (Exp4b, ReLSM-16k)
    n_memory_tokens: int = 64      # Compressed memory slots
    local_window: int = 2048       # Local attention window
    
    # GQA (for 16k context efficiency)
    use_gqa: bool = False
    n_kv_heads: Optional[int] = None
    
    def __post_init__(self):
        if self.d_ff is None:
            # SwiGLU has 3 matrices, so reduce expansion for param parity
            self.d_ff = int(2.67 * self.d_model)
        
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
        
        if self.use_gqa and self.n_kv_heads is None:
            self.n_kv_heads = max(1, self.n_heads // 4)


# Predefined configurations for different scales
CONFIGS = {
    # Nano: Match ReLSM-Nano (~18M) for fair comparison
    # Width+Recursion (ReLSM) vs Depth+Attention (this)
    "nano": ModelConfig(
        vocab_size=32000,  # Smaller vocab for nano
        max_seq_len=1024,
        d_model=512,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        # Shared loop defaults for nano
        n_unique_layers=2,
        n_unroll=6,
        # Thought stream
        n_thought_tokens=8,
        thought_d_model=256,
        K=4,
    ),
    
    # 300M: Fast iteration on the ladder
    "300M": ModelConfig(
        vocab_size=50257,
        max_seq_len=2048,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        dropout=0.1,
        n_unique_layers=6,
        n_unroll=24,
        n_thought_tokens=16,
        thought_d_model=512,
        K=8,
    ),
    
    # 1B: ReLSM-16k comparison (THE CONTROL MODEL)
    "1B": ModelConfig(
        vocab_size=50257,
        max_seq_len=4096,  # Can extend to 16k with memory tokens
        d_model=2048,
        n_layers=18,
        n_heads=16,
        dropout=0.1,
        n_unique_layers=6,
        n_unroll=18,
        n_thought_tokens=32,
        thought_d_model=512,  # L2-friendly recursive core
        K=8,
        max_K=16,
        n_memory_tokens=64,
        local_window=2048,
    ),
    
    # 1B-16k: Long context variant with GQA
    "1B-16k": ModelConfig(
        vocab_size=50257,
        max_seq_len=16384,
        d_model=2048,
        n_layers=18,
        n_heads=16,
        dropout=0.1,
        use_gqa=True,
        n_kv_heads=4,
        n_unique_layers=6,
        n_unroll=18,
        n_thought_tokens=32,
        thought_d_model=512,
        K=8,
        max_K=16,
        n_memory_tokens=128,
        local_window=2048,
    ),
}


# =============================================================================
# COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin, position_ids=None):
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads if config.use_gqa else config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=config.bias)
        
        self.rotary = RotaryEmbedding(self.d_head, config.max_seq_len, config.rope_base)
        self.dropout = nn.Dropout(config.dropout)
    
    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, D).reshape(B, self.n_heads, T, D)
    
    def forward(self, x, mask=None, kv_cache=None, use_cache=False, position_ids=None):
        B, T, _ = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        seq_len = T
        if kv_cache is not None:
            seq_len += kv_cache[0].size(2)
        
        cos, sin = self.rotary(seq_len, x.device)
        
        if position_ids is None:
            if kv_cache is not None:
                position_ids = torch.arange(kv_cache[0].size(2), seq_len, device=x.device).unsqueeze(0)
            else:
                position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        
        q, k = apply_rotary_emb(q, k, cos, sin, position_ids)
        
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config, layer_idx)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = SwiGLU(config)
    
    def forward(self, x, mask=None, kv_cache=None, use_cache=False, position_ids=None):
        h, new_cache = self.attn(self.attn_norm(x), mask, kv_cache, use_cache, position_ids)
        x = x + h
        x = x + self.ff(self.ff_norm(x))
        return x, new_cache


# =============================================================================
# THOUGHT STREAM COMPONENTS (Exp2/Exp3)
# =============================================================================

class ThoughtTokens(nn.Module):
    """Learnable thought tokens for the latent reasoning stream."""
    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
    
    def forward(self, batch_size: int):
        return self.tokens.expand(batch_size, -1, -1)


class MemoryTokens(nn.Module):
    """Learnable memory tokens for long-context compression."""
    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
    
    def forward(self, batch_size: int):
        return self.mem.expand(batch_size, -1, -1)


class ThoughtCore(nn.Module):
    """
    Compact recursive core for thought stream.
    Designed to be L2-cache friendly (~50M params).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.thought_d_model
        
        # Project from token stream dimension to thought dimension
        self.proj_in = nn.Linear(config.d_model, d)
        self.proj_out = nn.Linear(d, config.d_model)
        
        # Cross-attention: thought queries, token keys/values
        self.cross_norm = RMSNorm(d)
        self.cross_q = nn.Linear(d, d)
        self.cross_kv = nn.Linear(config.d_model, 2 * d)
        self.cross_out = nn.Linear(d, d)
        
        # Self-update MLP
        self.ff_norm = RMSNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )
        
        self.scale = (d // 8) ** -0.5  # 8 heads equivalent
    
    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        z: thought stream (B, Z, thought_d)
        h: token stream (B, T, d_model)
        Returns: updated z
        """
        B, Z, D = z.shape
        
        # Cross-attention to token stream
        z_norm = self.cross_norm(z)
        q = self.cross_q(z_norm)  # (B, Z, D)
        kv = self.cross_kv(h)     # (B, T, 2D)
        k, v = kv.chunk(2, dim=-1)
        
        # Simple attention (no multi-head for compactness)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        cross_out = self.cross_out(attn @ v)
        
        z = z + cross_out
        z = z + self.ff(self.ff_norm(z))
        
        return z


class HaltHead(nn.Module):
    """Predicts halting probability for ACT."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Pool over thought tokens, predict halt prob
        z_pooled = z.mean(dim=1)  # (B, D)
        return torch.sigmoid(self.proj(z_pooled))  # (B, 1)


# =============================================================================
# MAIN MODEL
# =============================================================================

class UnifiedTransformer(nn.Module):
    """
    Unified transformer supporting all ablation variants.
    
    Variants:
        baseline: Standard transformer (Exp0 control)
        shared_loop: Parameter-shared depth (Exp1)
        latent: Dual-stream with thought tokens (Exp2)
        act: Adaptive halting (Exp3)
        ssm: SSM backbone (Exp4a) - placeholder
        ssm_mem: SSM + memory tokens (Exp4b) - placeholder
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.variant = config.variant
        
        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Build layers based on variant
        if config.variant == "shared_loop":
            # Exp1: Fewer unique layers, unrolled multiple times
            self.layers = nn.ModuleList([
                TransformerBlock(config, i) for i in range(config.n_unique_layers)
            ])
            self.n_unroll = config.n_unroll
        else:
            # baseline, latent, act: Standard layer stack
            self.layers = nn.ModuleList([
                TransformerBlock(config, i) for i in range(config.n_layers)
            ])
            self.n_unroll = 1
        
        # Thought stream components (Exp2/Exp3)
        if config.variant in ["latent", "act"]:
            self.thought_tokens = ThoughtTokens(config.n_thought_tokens, config.thought_d_model)
            self.thought_core = ThoughtCore(config)
            self.thought_to_token = nn.Linear(config.thought_d_model, config.d_model)
            
            if config.variant == "act":
                self.halt_head = HaltHead(config.thought_d_model)
        
        # Memory tokens for long context
        if config.variant == "ssm_mem" or (config.variant in ["latent", "act"] and config.n_memory_tokens > 0):
            self.memory_tokens = MemoryTokens(config.n_memory_tokens, config.d_model)
        
        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying
        
        # Initialize
        self.apply(self._init_weights)
        
        # Report
        n_params = sum(p.numel() for p in self.parameters())
        print(f"UnifiedTransformer ({config.variant}): {n_params/1e6:.1f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[List] = None,
        use_cache: bool = False,
        return_thought_stats: bool = False,
    ):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.tok_emb(input_ids)
        x = self.drop(x)
        
        # Causal mask
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].size(2)
            mask = torch.zeros((T, past_len + T), device=device)
            mask[:, past_len:] = self._make_causal_mask(T, device)
        else:
            mask = self._make_causal_mask(T, device)
        
        # Stats tracking
        thought_stats = {}
        
        # =====================================================================
        # VARIANT: baseline / shared_loop
        # =====================================================================
        if self.variant in ["baseline", "shared_loop"]:
            new_cache = [] if use_cache else None
            
            for unroll_idx in range(self.n_unroll):
                for layer_idx, layer in enumerate(self.layers):
                    cache_idx = unroll_idx * len(self.layers) + layer_idx
                    layer_cache = kv_cache[cache_idx] if kv_cache else None
                    x, cache = layer(x, mask, layer_cache, use_cache)
                    if use_cache:
                        new_cache.append(cache)
        
        # =====================================================================
        # VARIANT: latent (fixed K) / act (adaptive K)
        # =====================================================================
        elif self.variant in ["latent", "act"]:
            new_cache = [] if use_cache else None
            
            # Run backbone to get token representations
            for layer_idx, layer in enumerate(self.layers):
                layer_cache = kv_cache[layer_idx] if kv_cache else None
                x, cache = layer(x, mask, layer_cache, use_cache)
                if use_cache:
                    new_cache.append(cache)
            
            h = x  # Token stream (B, T, D)
            
            # Initialize thought stream
            z = self.thought_tokens(B)  # (B, Z, thought_d)
            
            # Project h to thought dimension for cross-attention
            h_for_thought = h  # Keep at d_model, ThoughtCore handles projection
            
            if self.variant == "latent":
                # Fixed K iterations
                for k in range(self.config.K):
                    z = self.thought_core(z, h_for_thought)
                thought_stats["avg_K"] = self.config.K
            
            else:  # act
                # Adaptive halting
                cum_halt = torch.zeros(B, 1, device=device)
                ponder_cost = 0.0
                actual_K = 0
                
                for k in range(self.config.max_K):
                    if k < self.config.min_K or not (cum_halt >= self.config.halt_threshold).all():
                        z = self.thought_core(z, h_for_thought)
                        actual_K = k + 1
                        
                        if k >= self.config.min_K:
                            p_halt = self.halt_head(z)
                            cum_halt = cum_halt + (1.0 - cum_halt) * p_halt
                            ponder_cost = ponder_cost + p_halt.mean()
                    else:
                        break
                
                thought_stats["avg_K"] = actual_K
                thought_stats["ponder_cost"] = ponder_cost
            
            # Inject thought summary back to token stream
            z_summary = z.mean(dim=1, keepdim=True)  # (B, 1, thought_d)
            z_proj = self.thought_to_token(z_summary)  # (B, 1, D)
            x = h + z_proj  # Broadcast across sequence
        
        # =====================================================================
        # VARIANT: ssm / ssm_mem (placeholder)
        # =====================================================================
        elif self.variant in ["ssm", "ssm_mem"]:
            raise NotImplementedError("SSM variants require mamba-ssm package. Placeholder for Exp4.")
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Add ponder cost for ACT
            if self.variant == "act" and "ponder_cost" in thought_stats:
                loss = loss + self.config.lambda_ponder * thought_stats["ponder_cost"]
        
        if return_thought_stats:
            return logits, loss, new_cache, thought_stats
        return logits, loss, new_cache
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9, eos_token_id=None):
        self.eval()
        generated = input_ids
        
        # Note: KV cache with variants is complex, using simple generation for now
        for _ in range(max_new_tokens):
            logits, _, _ = self(generated[:, -self.config.max_seq_len:])
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated


# =============================================================================
# FACTORY
# =============================================================================

def create_model(
    size: str = "1B",
    variant: VariantType = "baseline",
    **kwargs
) -> UnifiedTransformer:
    """
    Create model with specified size and variant.
    
    Args:
        size: "nano", "300M", "1B", "1B-16k"
        variant: "baseline", "shared_loop", "latent", "act", "ssm", "ssm_mem"
        **kwargs: Override any config parameter
    """
    if size not in CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(CONFIGS.keys())}")
    
    config = CONFIGS[size]
    config.variant = variant
    
    # Apply overrides
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    return UnifiedTransformer(config)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing all configurations and variants")
    print("="*60)
    
    # Test nano baseline
    print("\n[Nano Baseline]")
    model = create_model("nano", "baseline")
    x = torch.randint(0, 32000, (2, 64))
    logits, loss, _ = model(x, labels=x)
    print(f"Output: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test nano shared_loop
    print("\n[Nano Shared Loop]")
    model = create_model("nano", "shared_loop")
    logits, loss, _ = model(x, labels=x)
    print(f"Output: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test nano latent
    print("\n[Nano Latent]")
    model = create_model("nano", "latent")
    logits, loss, _, stats = model(x, labels=x, return_thought_stats=True)
    print(f"Output: {logits.shape}, Loss: {loss.item():.4f}, K: {stats['avg_K']}")
    
    # Test nano ACT
    print("\n[Nano ACT]")
    model = create_model("nano", "act")
    logits, loss, _, stats = model(x, labels=x, return_thought_stats=True)
    print(f"Output: {logits.shape}, Loss: {loss.item():.4f}, K: {stats['avg_K']}")
    
    # Test 1B baseline (just config, don't instantiate full model)
    print("\n[1B Config]")
    config = CONFIGS["1B"]
    print(f"d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    estimated_params = (
        config.vocab_size * config.d_model +  # embeddings
        config.n_layers * (4 * config.d_model ** 2 + 3 * config.d_model * config.d_ff)  # layers
    )
    print(f"Estimated params: {estimated_params/1e9:.2f}B")
    
    print("\n✓ All tests passed!")
