"""Lightweight sanity checks for the evaluation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

from model import BaselineTransformer, TransformerConfig
from utils import generate_batch, seed_all


def run_self_test_suite(
    model,
    tokenizer,
    device: torch.device,
    *,
    seed: int,
    generation_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run minimal deterministic checks without heavy computation."""

    seed_all(seed)
    prompts = ["Hello, world!", "2 + 2 ="]
    preds, tokens, elapsed = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=generation_kwargs.get("max_new_tokens", 8),
        device=device,
        use_autocast=device.type == "cuda",
        task="self_test",
        generation_kwargs=generation_kwargs,
    )
    return {
        "status": "ok",
        "samples": [
            {"prompt": p, "completion": c} for p, c in zip(prompts, preds)
        ],
        "tokens_generated": tokens,
        "elapsed_seconds": elapsed,
    }


def _make_tiny_checkpoint(tmpdir: Path) -> Path:
    seed_all(123)
    config = TransformerConfig(
        vocab_size=50257,
        max_seq_len=128,
        d_model=64,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    model = BaselineTransformer(config)
    ckpt_path = tmpdir / "tiny_ckpt.pt"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


def run_self_checks() -> None:
    """Internal regression check to ensure deterministic outputs."""

    from eval_hub import run_eval_suite
    from utils import prepare_tokenizer

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_tiny_checkpoint(tmp_path)
        tokenizer = prepare_tokenizer("gpt2")
        checkpoint = torch.load(ckpt, map_location="cpu")
        model = BaselineTransformer(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        payload = run_eval_suite(
            model,
            tokenizer,
            torch.device("cpu"),
            suite="self_test",
            out_dir=tmp_path / "eval_out",
            seed=42,
            max_new_tokens=8,
        )
        assert "self_test" in payload
        second = run_eval_suite(
            model,
            tokenizer,
            torch.device("cpu"),
            suite="self_test",
            out_dir=tmp_path / "eval_out2",
            seed=42,
            max_new_tokens=8,
        )
        assert payload == second, "Self-test outputs should be deterministic"
    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    run_self_checks()
