"""Model defaults (changed 2026-10-28 to the Llama reference) and the legacy path.

See docs/model-defaults.md. The legacy table below is the pre-change factory, copied from
model.py at be08cde, so legacy_defaults=True must reproduce it exactly.
"""
import pytest

torch = pytest.importorskip("torch")

import model as model_module  # noqa: E402
from model import (  # noqa: E402
    LEGACY_DEFAULTS, PRESETS, BaselineTransformer, TransformerConfig, create_model, ffn_dim,
)

# size -> (d_model, d_ff, dropout, tie_weights) under the factory at be08cde (before the change)
OLD_FACTORY = {
    "nano": (512, 2048, 0.1, True),
    "50M": (512, 2048, 0.1, True),
    "125M": (768, 3072, 0.1, True),
    "350M": (1024, 4096, 0.1, True),
    "760M": (1280, 5120, 0.1, True),
    "300M": (1024, 4096, 0.05, True),
    "1B": (2048, 5504, 0.05, True),
    "1B-16k": (2048, 5504, 0.05, True),
}


@pytest.fixture
def config_only(monkeypatch):
    """Make create_model return the config instead of building large models."""
    monkeypatch.setattr(model_module, "BaselineTransformer", lambda cfg: cfg)


def test_config_defaults_follow_llama():
    cfg = TransformerConfig()
    assert cfg.tie_weights is False
    assert cfg.dropout == 0.0
    assert cfg.ffn_rule == "llama"
    assert cfg.d_ff == ffn_dim(cfg.d_model)


def test_ffn_rule_values():
    assert ffn_dim(4096) == 11008  # Llama 7B
    assert ffn_dim(512) == 1536
    assert ffn_dim(128) == 512
    assert ffn_dim(2048) == 5632
    assert ffn_dim(512, "4x") == 2048
    with pytest.raises(ValueError):
        ffn_dim(512, "8x")


def test_presets_cover_the_old_sizes():
    assert set(PRESETS) == set(OLD_FACTORY)


@pytest.mark.parametrize("size", list(OLD_FACTORY))
def test_legacy_defaults_reproduce_the_old_factory(size, config_only):
    cfg = create_model(size, legacy_defaults=True)
    d_model, d_ff, dropout, tie = OLD_FACTORY[size]
    assert (cfg.d_model, cfg.d_ff, cfg.dropout, cfg.tie_weights) == (d_model, d_ff, dropout, tie)
    if size == "1B-16k":
        assert cfg.max_seq_len == 16384 and cfg.use_gqa


@pytest.mark.parametrize("size", list(OLD_FACTORY))
def test_new_defaults_per_preset(size, config_only):
    cfg = create_model(size)
    assert cfg.tie_weights is False and cfg.dropout == 0.0
    expected_dff = 5504 if size.startswith("1B") else ffn_dim(OLD_FACTORY[size][0])
    assert cfg.d_ff == expected_dff


@pytest.mark.parametrize("size", list(OLD_FACTORY))
def test_overrides_work_for_every_size(size, config_only):
    """#145: create_model(size, dropout=..., d_ff=...) used to raise TypeError for every size."""
    cfg = create_model(size, dropout=0.0, d_ff=1024, tie_weights=True)
    assert (cfg.dropout, cfg.d_ff, cfg.tie_weights) == (0.0, 1024, True)
    cfg = create_model(size, legacy_defaults=True, dropout=0.0)  # explicit beats legacy
    assert cfg.dropout == 0.0 and cfg.tie_weights is True


def test_tying_flag_shares_or_separates_the_head():
    small = dict(vocab_size=50, max_seq_len=8, d_model=32, n_layers=1, n_heads=2)
    untied = BaselineTransformer(TransformerConfig(**small))
    tied = BaselineTransformer(TransformerConfig(**small, tie_weights=True))
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()
    assert tied.lm_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()


def test_old_style_checkpoint_loads_with_its_own_settings(tmp_path):
    """Checkpoints saved before the change hold a tied, 4x, dropout-0.1 config (no ffn_rule field).
    utils.load_model_from_checkpoint rebuilds from that config, so they keep loading unchanged."""
    small = dict(vocab_size=50, max_seq_len=8, d_model=32, n_layers=2, n_heads=2)
    old_cfg = TransformerConfig(**small, tie_weights=True, dropout=0.1, d_ff=128)
    del old_cfg.__dict__["ffn_rule"]  # an old pickle has no such field
    old = BaselineTransformer(old_cfg).eval()
    path = tmp_path / "old.pt"
    torch.save({"config": old_cfg, "model_state_dict": old.state_dict()}, path)

    torch.serialization.add_safe_globals([TransformerConfig])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    restored = BaselineTransformer(ckpt["config"]).eval()
    restored.load_state_dict(ckpt["model_state_dict"])
    assert restored.config.tie_weights is True and restored.config.d_ff == 128
    assert restored.lm_head.weight.data_ptr() == restored.tok_emb.weight.data_ptr()
    x = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        assert torch.equal(old(x)[0], restored(x)[0])


def test_resume_guard_refuses_a_changed_architecture():
    train = pytest.importorskip("train")
    small = dict(vocab_size=50, max_seq_len=8, d_model=32, n_layers=2, n_heads=2)
    old_cfg = TransformerConfig(**small, **{k: v for k, v in LEGACY_DEFAULTS.items()})
    new_cfg = TransformerConfig(**small)
    with pytest.raises(ValueError, match="legacy_model_defaults"):
        train.check_resume_architecture(old_cfg, new_cfg)
    train.check_resume_architecture(old_cfg, TransformerConfig(**small, **LEGACY_DEFAULTS))
    train.check_resume_architecture(None, new_cfg)  # checkpoints without a config are not checked
