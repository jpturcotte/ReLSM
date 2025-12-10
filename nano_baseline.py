"""
Nano Baseline Transformer (Control Group)
==========================================
Matched to ReLSM-Nano (~18M params) for fair comparison.

This uses learned positional embeddings (like original GPT),
which will fail on OOD length tests - that's the point.
If ReLSM-Nano extrapolates better, recursion is validated.

From baseline_proposal_2.md:
  - ReLSM-Nano: ~18M params, d=1024, 1 physical layer looped K=8
  - This baseline: ~19M params, d=512, 6 physical layers

Trade-off being tested: Width+Recursion (ReLSM) vs Depth+Attention (this)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_len: int):
        super().__init__()
        assert d_model % n_head == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Calculate Query, Key, Value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        # Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * expansion)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_len: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoBaseline(nn.Module):
    """
    Nano-scale baseline transformer for ReLSM-Nano comparison.
    
    ~19M parameters to match ReLSM-Nano's ~18M.
    Uses LEARNED positional embeddings - will fail OOD length tests.
    """
    
    def __init__(
        self, 
        vocab_size: int = 32000,
        d_model: int = 512, 
        n_layer: int = 6, 
        n_head: int = 8, 
        max_len: int = 1024
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)  # LEARNED positions
        
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, max_len) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.head.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        self._print_params()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"NanoBaseline: {total/1e6:.2f}M Parameters")
        print("Note: Uses LEARNED positional embeddings (will fail OOD length tests)")

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor = None
    ):
        B, T = idx.size()
        
        # Check sequence length
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_len
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compare_architectures():
    """Print comparison between NanoBaseline and hypothetical ReLSM-Nano."""
    print("="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)
    
    print("\n[NanoBaseline (Control)]")
    baseline = NanoBaseline()
    print(f"  d_model: 512")
    print(f"  n_layers: 6 (physical)")
    print(f"  n_heads: 8")
    print(f"  Positional: LEARNED (breaks OOD)")
    print(f"  Parameters: {count_parameters(baseline)/1e6:.2f}M")
    
    print("\n[ReLSM-Nano (Treatment)]")
    print(f"  d_model: 1024 (wider)")
    print(f"  n_layers: 1 (looped K=8)")
    print(f"  Positional: RECURRENT (should extrapolate)")
    print(f"  Parameters: ~18M")
    
    print("\n[Key Difference]")
    print("  Baseline: 6 physical layers, learned positions")
    print("  ReLSM: 1 layer × 8 iterations, recurrent state")
    print("  Test: OOD length generalization reveals which is better")
    print("="*60)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    compare_architectures()
    
    print("\n[Sanity Check]")
    model = NanoBaseline()
    
    # Test forward pass
    dummy_input = torch.randint(0, 32000, (2, 64))
    logits, loss = model(dummy_input, targets=dummy_input)
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, 32000, (1, 10))
    output = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {output.shape}")
    
    # Test OOD (should work up to max_len, then fail)
    print("\n[OOD Length Test]")
    for T in [512, 1024, 1025]:
        try:
            x = torch.randint(0, 32000, (1, T))
            model(x)
            print(f"  T={T}: OK")
        except ValueError as e:
            print(f"  T={T}: FAIL (expected) - {e}")
    
    print("\n✓ NanoBaseline ready for comparison")
