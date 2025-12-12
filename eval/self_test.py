"""Lightweight determinism and schema checks for eval_hub."""

import json
import tempfile
from pathlib import Path

import torch

from eval_hub import EvaluatorHub
from model import BaselineTransformer, TransformerConfig
from utils import seed_all


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


def _run_once(tmpdir: Path) -> dict:
    ckpt_path = _make_tiny_checkpoint(tmpdir)
    output_dir = tmpdir / "eval_out"
    hub = EvaluatorHub(
        checkpoint=str(ckpt_path),
        tokenizer_name="gpt2",
        device=torch.device("cpu"),
        tasks=["algorithmic", "ood", "needle", "tinystories"],
        grid_tasks=["addition", "parity"],
        output_dir=output_dir,
        seed=42,
        batch_size=2,
        n_override=4,
        needle_contexts=[64],
        needle_depths=[0.5],
        needle_samples=1,
        ppl_samples=2,
        max_new_tokens=8,
    )
    summary = hub.run_all()
    results_path = output_dir / "results.json"
    assert results_path.exists()
    with results_path.open() as f:
        data = json.load(f)
    # key presence
    assert "metadata" in data and "results" in data
    assert "algorithmic" in data["results"]
    assert "ood" in data["results"]
    assert "longctx" in data["results"]
    assert "ppl" in data["results"]
    assert (output_dir / "results_ood.csv").exists()
    assert (output_dir / "summary.md").exists()
    return summary


def run_self_checks():
    with tempfile.TemporaryDirectory() as tmp:
        first = _run_once(Path(tmp))
        second = _run_once(Path(tmp))
        assert first == second, "Deterministic runs should match exactly"
    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    run_self_checks()
