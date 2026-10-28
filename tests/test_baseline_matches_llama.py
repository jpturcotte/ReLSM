"""BaselineTransformer is a Llama-style decoder: check it against Hugging Face's LlamaForCausalLM.

Same weights are loaded into both models; logits and gradients must agree to float32 rounding.
Also checks causality, right-padding invariance, KV-cache decoding and a tiny memorisation run.
CPU only, about 30 s. See docs/model-defaults.md.
"""
import pytest
import torch
import torch.nn.functional as F

from model import BaselineTransformer, TransformerConfig

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

# name: (vocab, d_model, n_layers, n_heads, n_kv_heads or None, d_ff or None, bias, rope_base, seq_len, tie)
CONFIGS = {
    "grok_tied": (99, 128, 2, 4, None, None, False, 10000, 4, True),
    "grok_untied": (99, 128, 2, 4, None, None, False, 10000, 4, False),
    "nano_shape": (1000, 512, 6, 8, None, None, False, 10000, 64, True),
    "gqa": (500, 256, 2, 8, 2, None, False, 10000, 32, True),
    "bias": (300, 128, 2, 4, None, None, True, 10000, 16, True),
    "rope_and_dff": (300, 128, 2, 4, None, 344, False, 500000, 16, False),
    "long_seq": (300, 128, 2, 4, None, None, False, 10000, 256, True),
}


def relsm(vocab, d, layers, heads, kv, dff, bias, rope, seq, tie, pad_token_id=None):
    cfg = TransformerConfig(
        vocab_size=vocab, max_seq_len=max(seq, 16), d_model=d, n_layers=layers, n_heads=heads,
        use_gqa=kv is not None, n_kv_heads=kv, d_ff=dff, bias=bias, rope_base=rope, dropout=0.0,
        tie_weights=tie, variant="baseline", pad_token_id=pad_token_id,
    )
    m = BaselineTransformer(cfg).eval()
    with torch.no_grad():  # non-trivial norm gains and biases, so the comparison covers them
        for mod in m.modules():
            if mod.__class__.__name__ == "RMSNorm":
                mod.weight.add_(torch.randn_like(mod.weight) * 0.3)
            if isinstance(mod, torch.nn.Linear) and mod.bias is not None:
                mod.bias.normal_(0, 0.02)
    return m


def llama_from(m):
    c = m.config
    cfg = LlamaConfig(
        vocab_size=c.vocab_size, hidden_size=c.d_model, intermediate_size=c.d_ff,
        num_hidden_layers=c.n_layers, num_attention_heads=c.n_heads, num_key_value_heads=c.n_kv_heads,
        max_position_embeddings=c.max_seq_len, rms_norm_eps=1e-6, rope_theta=float(c.rope_base),
        tie_word_embeddings=c.tie_weights, attention_bias=c.bias, mlp_bias=c.bias, hidden_act="silu",
        attn_implementation="eager",
    )
    ref = LlamaForCausalLM(cfg).eval()
    sd = {"model.embed_tokens.weight": m.tok_emb.weight, "model.norm.weight": m.norm.weight}
    for i, blk in enumerate(m.layers):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = blk.attn_norm.weight
        sd[p + "post_attention_layernorm.weight"] = blk.ff_norm.weight
        for n in ("q", "k", "v", "o"):
            lin = getattr(blk.attn, f"{n}_proj")
            sd[p + f"self_attn.{n}_proj.weight"] = lin.weight
            if lin.bias is not None:
                sd[p + f"self_attn.{n}_proj.bias"] = lin.bias
        for ours, theirs in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            lin = getattr(blk.ff, ours)
            sd[p + f"mlp.{theirs}.weight"] = lin.weight
            if lin.bias is not None:
                sd[p + f"mlp.{theirs}.bias"] = lin.bias
    if not c.tie_weights:
        sd["lm_head.weight"] = m.lm_head.weight
    missing, unexpected = ref.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
    missing = [k for k in missing if not (c.tie_weights and k == "lm_head.weight")]
    assert not missing and not unexpected, (missing, unexpected)
    if c.tie_weights:
        ref.tie_weights()
    return ref


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_forward_and_gradients_match_llama(name):
    torch.manual_seed(0)
    spec = CONFIGS[name]
    m = relsm(*spec)
    ref = llama_from(m)
    vocab, seq = spec[0], spec[8]
    x = torch.randint(0, vocab, (4, seq))
    ours, _, _ = m(x)
    theirs = ref(input_ids=x).logits
    assert rel(ours, theirs) < 1e-5, f"{name}: logits differ"
    tgt = torch.randint(0, vocab, (4, seq))
    F.cross_entropy(ours.reshape(-1, vocab), tgt.reshape(-1)).backward()
    F.cross_entropy(theirs.reshape(-1, vocab), tgt.reshape(-1)).backward()
    pairs = [(m.tok_emb.weight, ref.model.embed_tokens.weight),
             (m.layers[0].attn.q_proj.weight, ref.model.layers[0].self_attn.q_proj.weight),
             (m.layers[-1].ff.w2.weight, ref.model.layers[-1].mlp.down_proj.weight),
             (m.norm.weight, ref.model.norm.weight)]
    for a, b in pairs:
        assert rel(a.grad, b.grad) < 1e-4, f"{name}: gradients differ"


def test_causal():
    torch.manual_seed(1)
    m = relsm(*CONFIGS["grok_tied"])
    x = torch.randint(0, 99, (8, 4))
    y = x.clone()
    y[:, 2:] = torch.randint(0, 99, (8, 2))
    with torch.no_grad():
        a, b = m(x)[0], m(y)[0]
    assert torch.equal(a[:, :2], b[:, :2])


def test_right_padding_does_not_change_real_positions():
    torch.manual_seed(2)
    pad = 299  # a dedicated pad id, distinct from every real token (see #232)
    m = relsm(*CONFIGS["bias"], pad_token_id=pad)
    x = torch.randint(0, 299, (3, 10))
    padded = torch.cat([x, torch.full((3, 6), pad)], dim=1)
    with torch.no_grad():
        a = m(x)[0]
        b = m(padded)[0][:, :10]
    assert rel(b, a) < 1e-5


def test_kv_cache_decoding_matches_full_forward():
    torch.manual_seed(3)
    m = relsm(*CONFIGS["gqa"])
    x = torch.randint(0, 500, (2, 12))
    with torch.no_grad():
        full = m(x)[0]
        logits, _, cache = m(x[:, :8], use_cache=True)
        steps = [logits]
        for t in range(8, 12):
            logits, _, cache = m(x[:, t:t + 1], kv_cache=cache, use_cache=True)
            steps.append(logits)
    inc = torch.cat(steps, dim=1)
    assert rel(inc, full) < 1e-5


def test_memorises_random_labels():
    torch.manual_seed(4)
    m = relsm(*CONFIGS["grok_tied"]).train()
    X = torch.randint(0, 99, (512, 4))
    Y = torch.randint(0, 99, (512,))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.0)
    for _ in range(1500):
        loss = F.cross_entropy(m(X)[0][:, -1], Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.01:
            break
    acc = (m(X)[0][:, -1].argmax(-1) == Y).float().mean().item()
    assert acc > 0.99
