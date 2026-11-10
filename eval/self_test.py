"""Lightweight sanity checks for the evaluation pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

if __package__ in (None, ""):
    # Run as `python eval/self_test.py`: put the repo root on the path so the
    # root modules below import (same as `python -m eval.self_test`).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model import BaselineTransformer, TransformerConfig, config_to_payload
from utils import DEFAULT_TOKENIZER_NAME, generate_batch, load_checkpoint, seed_all


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
    preds, gen_lengths, elapsed, info = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=generation_kwargs.get("max_new_tokens", 8),
        device=device,
        use_autocast=device.type == "cuda",
        task="self_test",
        generation_kwargs=generation_kwargs,
    )
    tokens = sum(gen_lengths)
    # "completion" is normalized (first line only), so two different generations
    # can share it; the token ids are what the determinism check compares.
    token_ids = [[int(t) for t in ids] for ids in info["pred_token_ids"]]
    return {
        "status": "ok",
        "samples": [
            {"prompt": p, "completion": c, "output_token_ids": ids}
            for p, c, ids in zip(prompts, preds, token_ids)
        ],
        "tokens_generated": tokens,
        "elapsed_seconds": elapsed,
    }


# Fields that change from run to run without saying anything about determinism:
# the wall-clock time the payload was built and the generation time. Everything
# else in the payload (samples and their token ids, token counts, seed, decoding,
# model summary) is compared.
VOLATILE_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("meta", "timestamp"),
    ("self_test", "elapsed_seconds"),
)


def weights_sha256(model: torch.nn.Module) -> str:
    """Hash every parameter and buffer (name, dtype, shape, bytes).

    Buffers registered with persistent=False (the RoPE tables) are not in the
    state dict, so this walks the parameters and buffers instead. Take it before
    evaluation: run_eval_suite casts the model to bf16, which can map different
    fp32 weights to the same values.
    """

    tensors = [("param:" + n, t) for n, t in model.named_parameters()]
    tensors += [("buffer:" + n, t) for n, t in model.named_buffers()]
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        t = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(t.dtype).encode())
        digest.update(str(tuple(t.shape)).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def determinism_view(
    payload: Dict[str, Any],
    loaded_weights_sha256: str,
    volatile: Iterable[Tuple[str, ...]] = VOLATILE_FIELDS,
) -> Dict[str, Any]:
    """Return the payload minus the wall-clock fields, plus the loaded weights' hash."""

    view = copy.deepcopy(payload)
    for path in volatile:
        node: Any = view
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    view["weights_sha256"] = loaded_weights_sha256
    return view


def _diff_keys(a: Any, b: Any, prefix: str = "") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            where = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(where)
            else:
                out.extend(_diff_keys(a[key], b[key], where))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_diff_keys(x, y, f"{prefix}.{i}" if prefix else str(i)))
        return out
    return [] if a == b else [prefix or "<root>"]


def assert_deterministic(first: Dict[str, Any], second: Dict[str, Any]) -> None:
    """Fail if two determinism views differ, naming the fields that differ."""

    if first != second:
        raise AssertionError(
            "Self-test outputs should be deterministic; differing fields: "
            + ", ".join(_diff_keys(first, second))
        )


def _make_tiny_checkpoint(tmpdir: Path, vocab_size: int) -> Path:
    seed_all(123)
    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=64,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    model = BaselineTransformer(config)
    ckpt_path = tmpdir / "tiny_ckpt.pt"
    torch.save({"config": config_to_payload(config), "model_state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


def run_self_checks(tokenizer=None) -> None:
    """Internal regression check to ensure deterministic outputs.

    Two runs, each loading its own model from the same checkpoint, must give
    the same weights, token ids and payload apart from VOLATILE_FIELDS.
    ``tokenizer`` defaults to DEFAULT_TOKENIZER_NAME (tests pass a small one).
    """

    from eval_hub import run_eval_suite
    from utils import prepare_tokenizer

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if tokenizer is None:
            tokenizer = prepare_tokenizer(DEFAULT_TOKENIZER_NAME)
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        ckpt = _make_tiny_checkpoint(tmp_path, vocab_size)

        def _run(out_name: str) -> Dict[str, Any]:
            # Each run loads its own model from the checkpoint, so the check
            # covers loading as well as generation.
            checkpoint = load_checkpoint(ckpt, map_location="cpu")  # weights_only (#138)
            model = BaselineTransformer(checkpoint["config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            loaded = weights_sha256(model)  # before run_eval_suite's bf16 cast
            payload = run_eval_suite(
                model,
                tokenizer,
                torch.device("cpu"),
                suite="self_test",
                out_dir=tmp_path / out_name,
                seed=42,
                max_new_tokens=8,
            )
            assert "self_test" in payload
            return determinism_view(payload, loaded)

        assert_deterministic(_run("eval_out"), _run("eval_out2"))
    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    run_self_checks()
