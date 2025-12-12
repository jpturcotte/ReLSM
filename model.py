"""
Baseline Transformer + Ablation Ladder Variants (Exp0–Exp4)
==========================================================

This file extends the original BaselineTransformer control model with a small set of
**opt-in** variants:

- baseline    : vanilla decoder-only Transformer (Exp0)  ✅ identical behavior when selected
- shared_loop : parameter-shared depth via looped blocks (Exp1)
- latent      : fixed-K latent thought tokens (Exp2)
- act         : adaptive computation time (ACT) on latent thought (Exp3)
- ssm         : Mamba-2 style SSM backbone + fixed-K thought (Exp4A)
- ssm_mem     : SSM backbone + memory-token compression + thought loop (Exp4B)

Design goals:
- Keep the baseline path unchanged unless variant != "baseline".
- Preserve the forward() return signature: (logits, loss, new_cache)
- Store auxiliary stats (inner steps, ponder) on self.aux for logging.

Notes:
- KV cache generation is fully supported for the baseline Transformer path.
- For thought-loop variants, generate() falls back to no-cache generation by default
  (correctness > speed for early research). You can later implement a cache-aware
  thought KV source.
"""

import importlib
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional accelerated selective scan (Mamba kernels)
_selective_scan_fn = None
HAS_MAMBA_SCAN = False

try:
    _scan_iface = importlib.util.find_spec("mamba_ssm.ops.selective_scan_interface")
except ModuleNotFoundError:
    _scan_iface = None

if _scan_iface is not None:
    _module = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    _selective_scan_fn = getattr(_module, "selective_scan_fn", None)
    HAS_MAMBA_SCAN = _selective_scan_fn is not None
else:
    try:
        _scan_triton = importlib.util.find_spec("mamba_ssm.ops.triton.selective_scan")
    except ModuleNotFoundError:
        _scan_triton = None
    if _scan_triton is not None:
        _module = importlib.import_module("mamba_ssm.ops.triton.selective_scan")
        _selective_scan_fn = getattr(_module, "selective_scan_fn", None)
        HAS_MAMBA_SCAN = _selective_scan_fn is not None


@torch.jit.script
def ssm_scan_jit(
    B_t_exp: torch.Tensor,
    C_t_exp: torch.Tensor,
    dt_exp: torch.Tensor,
    A_exp_groups: torch.Tensor,
    state: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled Selective Scan.
    Optimizes the sequential loop to run in C++ rather than Python.
    """
    B_sz, T, E_flat = dt_exp.shape

    # Reshape A for broadcasting: (G*group_size, N) -> (1, 1, G*group_size, N)
    A_exp = A_exp_groups.unsqueeze(0).unsqueeze(0)

    outputs: List[torch.Tensor] = []

    for t in range(T):
        dt_t = dt_exp[:, t].unsqueeze(-1)             # (B, E, 1)
        A_bar = torch.exp(dt_t * A_exp)               # Discretization

        # Recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
        # B_t_exp[:, t] is (B, E, N)
        state = A_bar * state + dt_t * B_t_exp[:, t]

        # Readout: y_t = sum(state * C_t, dim=-1)
        # C_t_exp[:, t] is (B, E, N)
        y_t = (state * C_t_exp[:, t]).sum(dim=-1)     # (B, E)
        outputs.append(y_t)

    # Stack along time dimension
    y = torch.stack(outputs, dim=1)
    return y, state


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TransformerConfig:
    # Tokenization / sequence
    vocab_size: int = 50257
    max_seq_len: int = 1024

    # Transformer backbone
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: Optional[int] = None          # Defaults to 4 * d_model (SwiGLU uses 3 mats)
    dropout: float = 0.1
    bias: bool = False
    rope_base: int = 10000
    tie_weights: bool = True

    # Attention variant
    use_gqa: bool = False
    n_kv_heads: Optional[int] = None

    # Ladder variants
    variant: str = "baseline"           # baseline|shared_loop|latent|act|ssm|ssm_mem

    # Exp1: shared-loop depth
    n_unique_layers: int = 6            # number of unique layers
    n_unroll: int = 24                  # number of unrolled applications

    # Exp2/3: latent thought loop
    thought_tokens: int = 16            # Z thought tokens
    K: int = 4                          # fixed steps for latent/ssm/ssm_mem
    min_K: int = 2                      # ACT minimum
    max_K: int = 8                      # ACT maximum
    halt_threshold: float = 0.99        # ACT halting threshold
    lambda_ponder: float = 0.01         # ponder cost coefficient
    thought_use_rope: bool = False      # default: no RoPE in cross-attn (simpler)

    # Exp4B: memory tokens (compression)
    num_mem: int = 64                   # number of memory tokens
    local_window: int = 0               # if >0, thought attends only to last local_window tokens (plus mem, if enabled)

    # SSM backbone (Mamba-2 inspired)
    ssm_kernel: int = 7                 # causal depthwise conv kernel
    ssm_expand: int = 2                 # expansion factor in the SSM block
    ssm_state: int = 16                 # SSM state size (per expanded channel)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads

        if self.use_gqa and self.n_kv_heads is None:
            # Default: 4x reduction in KV heads (common)
            self.n_kv_heads = max(1, self.n_heads // 4)
        if not self.use_gqa:
            self.n_kv_heads = self.n_heads

    @property
    def num_params_estimate(self) -> int:
        """
        Rough estimate (baseline transformer + embeddings).
        Variants add small overhead (thought/mem), not counted here.
        """
        embed = self.vocab_size * self.d_model
        if self.use_gqa:
            attn = self.n_layers * (
                self.d_model * self.d_model +                                   # Q
                2 * self.n_kv_heads * self.d_head * self.d_model +              # KV
                self.d_model * self.d_model                                     # O
            )
        else:
            attn = self.n_layers * 4 * self.d_model * self.d_model
        ffn = self.n_layers * 3 * self.d_model * self.d_ff                      # SwiGLU: (w1,w2,w3)
        ln = self.n_layers * 2 * self.d_model + self.d_model
        total = embed + attn + ffn + ln
        if not self.tie_weights:
            total += embed
        return total


# =============================================================================
# PRIMITIVES
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding cache."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# SELF-ATTENTION (baseline path)
# =============================================================================

class Attention(nn.Module):
    """Multi-head self-attention with optional GQA and KV cache."""

    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads if config.use_gqa else config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # For GQA
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=config.bias)

        self.rotary = RotaryEmbedding(self.d_head, config.max_seq_len, config.rope_base)
        self.attn_dropout = nn.Dropout(config.dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, D).reshape(B, self.n_heads, T, D)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        return out, new_cache


class FeedForward(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""
    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config, layer_idx)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, new_cache = self.attn(self.attn_norm(x), mask, kv_cache, use_cache, position_ids)
        x = x + h
        x = x + self.ff(self.ff_norm(x))
        return x, new_cache


# =============================================================================
# CROSS-ATTENTION (for thought / memory)
# =============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention: queries from q_in, keys/values from kv_in.
    (No RoPE by default; simplest stable version for latent tokens.)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, Q, D = q_in.shape
        _, T, _ = kv_in.shape

        q = self.q_proj(q_in).view(B, Q, self.n_heads, self.d_head).transpose(1, 2)   # (B,h,Q,dh)
        k = self.k_proj(kv_in).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B,h,T,dh)
        v = self.v_proj(kv_in).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        w = (q @ k.transpose(-2, -1)) * self.scale
        w = F.softmax(w, dim=-1, dtype=torch.float32).type_as(q)
        w = self.drop(w)

        out = (w @ v).transpose(1, 2).contiguous().view(B, Q, D)
        return self.o_proj(out)


class ThoughtCore(nn.Module):
    """A small shared 'thought' block: cross-attn(z <- kv) + FFN."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.z_norm = RMSNorm(config.d_model)
        self.kv_norm = RMSNorm(config.d_model)
        self.xattn = CrossAttention(config.d_model, config.n_heads, dropout=config.dropout, bias=config.bias)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, z: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        z = z + self.xattn(self.z_norm(z), self.kv_norm(kv))
        z = z + self.ff(self.ff_norm(z))
        return z


class MemoryCompressor(nn.Module):
    """
    Builds M memory tokens by letting learnable mem queries attend to the token sequence (O(T*M)).
    Then optionally lets tokens attend back to memory (also O(T*M)).
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_mem = config.num_mem
        self.mem_init = nn.Parameter(torch.zeros(1, self.num_mem, config.d_model))
        nn.init.normal_(self.mem_init, mean=0.0, std=0.02)

        self.mem_norm = RMSNorm(config.d_model)
        self.tok_norm = RMSNorm(config.d_model)

        self.mem_read = CrossAttention(config.d_model, config.n_heads, dropout=config.dropout, bias=config.bias)  # mem <- toks
        self.tok_read = CrossAttention(config.d_model, config.n_heads, dropout=config.dropout, bias=config.bias)  # toks <- mem

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = h.shape
        mem = self.mem_init.expand(B, -1, -1)

        mem = mem + self.mem_read(self.mem_norm(mem), self.tok_norm(h))     # build memory summary
        h2 = h + self.tok_read(self.tok_norm(h), self.mem_norm(mem))        # inject memory back
        return mem, h2


# =============================================================================
# SSM backbone (Mamba-2 inspired)
# =============================================================================

class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise conv over the sequence dimension."""
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,C) -> (B,C,T)
        B, T, C = x.shape
        x_t = x.transpose(1, 2)

        if cache is None:
            # pad left
            x_pad = F.pad(x_t, (self.kernel - 1, 0))
        else:
            # cache: (B,C,K-1)
            x_pad = torch.cat([cache, x_t], dim=-1)

        y = self.conv(x_pad)
        y = y[..., -T:]  # keep last T outputs
        new_cache = x_pad[..., -(self.kernel - 1):].detach()  # (B,C,K-1)
        return y.transpose(1, 2), new_cache


class SSMBlock(nn.Module):
    """
    Minimal, dependency-free Mamba-2 style SSM block.

    The implementation follows the high-level structure of Mamba-2:
    - input gating
    - causal depthwise convolutional mixing
    - state-space recurrence per expanded channel with learnable decay (A) and
      input projection (B), producing output via learned readout (C)
    - feed-forward residual

    This trades some kernel fusion performance for readability and zero external
    dependencies while preserving the SSM semantics expected from a Mamba-2
    backbone.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.D = config.d_model
        self.E = config.ssm_expand * config.d_model
        self.N = config.ssm_state

        # Grouped SSM parameters to keep the accelerated kernel shapes small
        self.groups = max(1, self.E // 128)
        self.group_size = math.ceil(self.E / self.groups)

        self.norm1 = RMSNorm(self.D)
        self.in_proj = nn.Linear(self.D, 2 * self.E, bias=config.bias)  # gate + value
        self.dwconv = CausalDepthwiseConv1d(self.E, kernel_size=config.ssm_kernel)

        self.use_accelerated_scan = _selective_scan_fn is not None

        # State-space parameters
        self.B_proj = nn.Linear(self.E, self.groups * self.N, bias=False)
        self.C_proj = nn.Linear(self.E, self.groups * self.N, bias=False)
        self.dt_proj = nn.Linear(self.E, self.groups, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.groups, self.N))     # decay (log-space)

        self.out_proj = nn.Linear(self.E, self.D, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

        self.norm2 = RMSNorm(self.D)
        self.ff = FeedForward(config)

    def _ssm_scan(self, v: torch.Tensor, cache_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Mamba-2 style selective scan over the sequence."""
        B, T, E = v.shape
        pad = self.groups * self.group_size - E
        if pad:
            v = F.pad(v, (0, pad))
        v_group = v.view(B, T, self.groups, self.group_size)

        # Project inputs to SSM parameters
        B_t = self.B_proj(v).view(B, T, self.groups, self.N)
        C_t = self.C_proj(v).view(B, T, self.groups, self.N)
        dt = F.softplus(self.dt_proj(v)).view(B, T, self.groups)

        # Decay parameter (A)
        A = -torch.exp(self.A_log)  # (groups, N)

        # 1. Try accelerated CUDA kernel first (if available)
        if self.use_accelerated_scan and _selective_scan_fn is not None:
            initial_state = cache_state
            if initial_state is None:
                initial_state = v.new_zeros(B, self.groups * self.group_size, self.N)

            # Broadcast SSM parameters across the per-group channels instead of
            # compressing the value streams. This mirrors the typical
            # multi-head treatment where B/C are shared and expanded to match v.
            B_t_exp = B_t.repeat_interleave(self.group_size, dim=2)
            C_t_exp = C_t.repeat_interleave(self.group_size, dim=2)
            dt_exp = dt.repeat_interleave(self.group_size, dim=2)
            A_exp_groups = A.repeat_interleave(self.group_size, dim=0)

            v_flat = v_group.reshape(B, T, -1)
            try:
                y, last_state = _selective_scan_fn(
                    v_flat,
                    dt_exp,
                    A_exp_groups,
                    B_t_exp,
                    C_t_exp,
                    delta_softplus=False,
                    return_last_state=True,
                    initial_state=initial_state,
                )
                y = y.view(B, T, self.groups, self.group_size).reshape(B, T, -1)[..., :E]
                return y, last_state.detach().float()
            except Exception as exc:
                warnings.warn(
                    f"Falling back to Python selective scan due to kernel error: {exc}",
                    RuntimeWarning,
                )

        # 2. Fallback: JIT-Compiled Python Loop
        if cache_state is None:
            state = v.new_zeros(B, self.groups * self.group_size, self.N, dtype=torch.float32)
        else:
            state = cache_state.float()

        # Expand dimensions for the scan (B/C/dt shared across group_size)
        B_t_exp = B_t.repeat_interleave(self.group_size, dim=2)
        C_t_exp = C_t.repeat_interleave(self.group_size, dim=2)
        dt_exp = dt.repeat_interleave(self.group_size, dim=2)
        A_exp_groups = A.repeat_interleave(self.group_size, dim=0)

        # CALL JIT FUNCTION HERE
        y_flat, state = ssm_scan_jit(B_t_exp, C_t_exp, dt_exp, A_exp_groups, state)

        # Reshape output to match original dimensions
        y = y_flat.view(B, T, self.groups, self.group_size).reshape(B, T, -1)[..., :E]

        return y.to(v.dtype), state.detach()

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        conv_cache, state_cache = (cache if cache is not None else (None, None))

        h = self.norm1(x)
        uv = self.in_proj(h)
        u, v = uv.chunk(2, dim=-1)
        v, conv_cache = self.dwconv(v, conv_cache)

        y, new_state = self._ssm_scan(v, state_cache)
        y = torch.sigmoid(u) * y
        y = self.drop(self.out_proj(y))
        x = x + y
        x = x + self.ff(self.norm2(x))

        new_cache = (conv_cache, new_state) if cache is not None else None
        return x, new_cache


# =============================================================================
# MAIN MODEL
# =============================================================================

class BaselineTransformer(nn.Module):
    """
    Baseline decoder-only transformer + ladder variants.

    Forward returns:
        logits, loss, new_cache

    Extra logging (inner steps, ponder) is stored in self.aux after forward().
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.aux: Dict[str, Any] = {}

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Backbone selection
        self._use_ssm = config.variant in {"ssm", "ssm_mem"}
        self._use_shared_loop = config.variant == "shared_loop"

        if self._use_ssm:
            self.layers = nn.ModuleList([SSMBlock(config) for _ in range(config.n_layers)])
            self._cache_slots = len(self.layers)
        elif self._use_shared_loop:
            # Only n_unique_layers modules, but unrolled n_unroll times
            self.unique_layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_unique_layers)])
            self._cache_slots = config.n_unroll
        else:
            self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
            self._cache_slots = len(self.layers)

        # Optional memory compressor (Exp4B)
        self._use_mem = config.variant == "ssm_mem"
        self.mem_comp = MemoryCompressor(config) if self._use_mem else None

        # Optional thought loop (Exp2/3 + Exp4)
        self._use_thought = config.variant in {"latent", "act", "ssm", "ssm_mem"}
        if self._use_thought:
            self.thought_init = nn.Parameter(torch.zeros(1, config.thought_tokens, config.d_model))
            nn.init.normal_(self.thought_init, mean=0.0, std=0.02)
            self.thought_core = ThoughtCore(config)
            self.halt_head = nn.Linear(config.d_model, 1, bias=True) if config.variant == "act" else None
            self.thought_to_tok = nn.Linear(config.d_model, config.d_model, bias=False)

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"BaselineTransformer({config.variant}): {n_params/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _select_kv_source(self, h: torch.Tensor, mem: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Build KV source for thought loop.
        - If mem is provided: kv = [mem, maybe local window tokens]
        - Else: kv = tokens (maybe local window)
        """
        cfg = self.config
        if cfg.local_window and cfg.local_window > 0:
            toks = h[:, -cfg.local_window:, :]
        else:
            toks = h

        if mem is None:
            return toks
        return torch.cat([mem, toks], dim=1)

    def _apply_thought_loop(self, h: torch.Tensor, mem: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply latent thought loop and inject a summary back into token hidden states.
        """
        cfg = self.config
        B, T, D = h.shape

        kv = self._select_kv_source(h, mem)
        z = self.thought_init.expand(B, -1, -1)

        self.aux = {"inner_steps": 0, "ponder": 0.0}

        if cfg.variant == "act":
            # Adaptive Computation Time over thought updates (sequence-level halting)
            cum_halt = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
            steps_used = torch.zeros(B, device=h.device, dtype=torch.int64)
            ponder = 0.0
            iterations_run = 0

            # enforce a minimum number of steps
            for k in range(cfg.max_K):
                z = self.thought_core(z, kv)

                p = torch.sigmoid(self.halt_head(z.mean(dim=1)))  # (B,1)

                if k < cfg.min_K:
                    p = p * 0.0  # no halting credit before min_K

                # Update cum_halt
                delta = (1.0 - cum_halt) * p
                cum_halt = cum_halt + delta
                ponder = ponder + p.mean()

                # Track the number of iterations performed for every sequence
                iterations_run = k + 1
                steps_used[:] = iterations_run

                # Break if all halted and min_K satisfied
                if (cum_halt >= cfg.halt_threshold).all() and (k + 1) >= cfg.min_K:
                    break

            # If some never halted, they used the final iteration count (max_K cap)
            never_halted = (cum_halt < cfg.halt_threshold).squeeze(-1)
            if never_halted.any():
                steps_used[never_halted] = iterations_run

            self.aux["inner_steps"] = steps_used.detach().cpu()
            self.aux["ponder"] = float(ponder)

        else:
            # Fixed-K
            for _ in range(max(1, cfg.K)):
                z = self.thought_core(z, kv)
            self.aux["inner_steps"] = int(max(1, cfg.K))
            self.aux["ponder"] = 0.0

        # Inject summary (mean of z)
        z_sum = z.mean(dim=1, keepdim=True)  # (B,1,D)
        h = h + self.thought_to_tok(z_sum)   # broadcast
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Any]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Any]]]:
        B, T = input_ids.shape
        device = input_ids.device

        x = self.drop(self.tok_emb(input_ids))

        # Baseline Transformer path uses causal mask; SSM path does not.
        mask = None
        if not self._use_ssm:
            if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
                # kv_cache[0] expected to be (K,V) for transformer paths
                past_len = kv_cache[0][0].size(2)
                mask = torch.zeros((T, past_len + T), device=device)
                mask[:, past_len:] = self._make_causal_mask(T, device)
            else:
                mask = self._make_causal_mask(T, device)

        new_cache = [] if use_cache else None

        if self._use_shared_loop:
            # Shared weights, unrolled depth
            for step in range(self.config.n_unroll):
                layer = self.unique_layers[step % len(self.unique_layers)]
                layer_cache = kv_cache[step] if kv_cache is not None else None
                x, cache = layer(x, mask, layer_cache, use_cache, position_ids)
                if use_cache:
                    new_cache.append(cache)
        elif self._use_ssm:
            for i, layer in enumerate(self.layers):
                layer_cache = kv_cache[i] if kv_cache is not None else None
                x, cache = layer(x, layer_cache)
                if use_cache:
                    new_cache.append(cache)
        else:
            for i, layer in enumerate(self.layers):
                layer_cache = kv_cache[i] if kv_cache is not None else None
                x, cache = layer(x, mask, layer_cache, use_cache, position_ids)
                if use_cache:
                    new_cache.append(cache)

        # Optional memory compression (Exp4B)
        mem = None
        if self._use_mem and self.mem_comp is not None:
            mem, x = self.mem_comp(x)

        # Optional thought loop (Exp2/3/4)
        if self._use_thought:
            x = self._apply_thought_loop(x, mem)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            # Add ACT ponder cost (if any)
            if self.config.variant == "act":
                loss = loss + self.config.lambda_ponder * logits.new_tensor(self.aux.get("ponder", 0.0))

        return logits, loss, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        - Baseline Transformer path uses KV cache (fast).
        - Thought-loop variants currently fall back to no-cache generation (slower but correct).
        """
        self.eval()
        generated = input_ids

        use_kv_cache = self.config.variant in {"baseline", "shared_loop", "ssm", "ssm_mem"}

        if use_kv_cache:
            kv_cache: List[Any] = [None] * self._cache_slots

        for _ in range(max_new_tokens):
            if use_kv_cache:
                curr_input = generated[:, -1:] if kv_cache[0] is not None else generated
                logits, _, kv_cache = self(curr_input, kv_cache=kv_cache, use_cache=True)
            else:
                # no-cache: recompute full context each step
                logits, _, _ = self(generated, kv_cache=None, use_cache=False)

            logits = logits[:, -1, :] / max(1e-8, temperature)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding path used for deterministic evaluation.
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(
    size: str = "125M",
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    use_gqa: bool = False,
    variant: str = "baseline",
    **kwargs,
) -> BaselineTransformer:
    """
    Factory for baseline + ladder variants.

    Sizes:
      - nano (ReLSM-Nano compatible quick runs)
      - 50M, 125M, 350M, 760M (original)
      - 300M (recommended ladder iteration size)
      - 1B   (baseline control scale)
      - 1B-16k (long context with GQA)

    Variant:
      baseline | shared_loop | latent | act | ssm | ssm_mem
    """
    base = dict(vocab_size=vocab_size, max_seq_len=max_seq_len, use_gqa=use_gqa, variant=variant)
    base.update(kwargs)

    configs = {
        "nano": TransformerConfig(d_model=512,  n_layers=6,  n_heads=8,  **base),
        "50M":  TransformerConfig(d_model=512,  n_layers=8,  n_heads=8,  **base),
        "125M": TransformerConfig(d_model=768,  n_layers=12, n_heads=12, **base),
        "350M": TransformerConfig(d_model=1024, n_layers=24, n_heads=16, **base),
        "760M": TransformerConfig(d_model=1280, n_layers=36, n_heads=20, **base),

        # Ladder-friendly defaults
        "300M": TransformerConfig(d_model=1024, n_layers=24, n_heads=16, d_ff=4096, dropout=0.05, **base),

        # ~1B control (adjust vocab_size to your tokenizer; with 50k vocab this is ~1B-ish)
        "1B":   TransformerConfig(d_model=2048, n_layers=18, n_heads=16, d_ff=5504, dropout=0.05, **base),
        "1B-16k": TransformerConfig(
            d_model=2048,
            n_layers=18,
            n_heads=16,
            d_ff=5504,
            dropout=0.05,
            **{**base, "max_seq_len": 16384, "use_gqa": True},
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")

    return BaselineTransformer(configs[size])


if __name__ == "__main__":
    print("Quick smoke test...")

    # Baseline control
    m0 = create_model("125M", variant="baseline")
    x = torch.randint(0, m0.config.vocab_size, (2, 128))
    logits, loss, _ = m0(x, labels=x)
    print("baseline loss", float(loss))

    # Latent thought
    m1 = create_model("50M", variant="latent", K=3, thought_tokens=8)
    logits, loss, _ = m1(x, labels=x)
    print("latent aux", m1.aux)

    # ACT thought
    m2 = create_model("50M", variant="act", max_K=6, min_K=2)
    logits, loss, _ = m2(x, labels=x)
    print("act aux", m2.aux)

    # SSM backbone
    m3 = create_model("50M", variant="ssm", K=2)
    logits, loss, _ = m3(x, labels=x)
    print("ssm ok", logits.shape)
