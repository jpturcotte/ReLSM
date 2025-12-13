import pytest

torch = pytest.importorskip("torch")

from model import create_model


def run_variant(variant: str):
    model = create_model(
        size="50M",
        variant=variant,
        K=2,
        min_K=1,
        max_K=3,
        thought_tokens=4,
        num_mem=8,
    )
    model.eval()

    x = torch.randint(0, model.config.vocab_size, (2, 16))
    logits, loss, cache = model(x, labels=x, use_cache=True)

    assert logits.shape == (2, 16, model.config.vocab_size)
    assert torch.isfinite(loss).all()
    assert cache is None or isinstance(cache, list)


def test_all_variants_smoke():
    variants = ["baseline", "shared_loop", "latent", "act", "ssm", "ssm_mem"]
    for v in variants:
        run_variant(v)
