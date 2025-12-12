"""Canonical evaluation orchestrator for ReLSM.

This module exposes ``run_eval_suite`` as the single entrypoint for all
evaluation flows (algorithmic OOD grid, long-context retrieval, and
lightweight self tests). The accompanying CLI mirrors the same API and
writes standardized JSON outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from eval import ood_grid
from eval.self_test import run_self_test_suite
from utils import (
    EvalResult,
    NeedleInHaystackGenerator,
    area_under_depth_curve,
    evaluate_condition,
    gather_metadata,
    get_eval_generation_kwargs,
    load_model_and_tokenizer,
    save_json,
    seed_all,
    select_device,
)


def _normalize_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return select_device(device)


def _dtype_from_string(dtype: str) -> Optional[torch.dtype]:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(dtype.lower(), None)


def _prepare_model(model, device: torch.device, dtype: Optional[str], compile_model: bool):
    torch_dtype = _dtype_from_string(dtype) if dtype else None
    try:
        if torch_dtype is not None:
            model = model.to(device=device, dtype=torch_dtype)
        else:
            model = model.to(device)
    except Exception:
        model = model.to(device)
    model.eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass
    return model


def _algorithmic_results_to_dict(results: List[EvalResult]) -> Dict[str, Any]:
    conditions: List[Dict[str, Any]] = []
    per_task: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    total_seen = 0

    for res in results:
        conditions.append(
            {
                "task": res.task,
                "condition": res.condition,
                "accuracy": res.accuracy,
                "n": res.n,
                "avg_gen_len": res.avg_gen_len,
                "tokens_per_sec": res.tokens_per_sec,
                "tokens_generated": res.tokens_generated,
                "elapsed_seconds": res.elapsed,
                "examples": res.examples,
            }
        )
        total_correct += res.correct
        total_seen += res.n
        per_task.setdefault(res.task, {"conditions": [], "correct": 0, "total": 0})
        per_task[res.task]["conditions"].append(res.condition)
        per_task[res.task]["correct"] += res.correct
        per_task[res.task]["total"] += res.n

    per_task_acc = {
        task: (payload["correct"] / payload["total"] if payload["total"] else 0.0)
        for task, payload in per_task.items()
    }
    overall = total_correct / total_seen if total_seen else 0.0
    return {
        "grid_version": ood_grid.OOD_GRID_VERSION,
        "conditions": conditions,
        "per_task_accuracy": per_task_acc,
        "overall_accuracy": overall,
    }


def run_algorithmic_suite(
    model,
    tokenizer,
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    generation_kwargs: Dict[str, Any],
    limit: Optional[int],
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    seed_all(seed)
    grid = ood_grid.build_grid(list(tasks) if tasks else None)
    gen_kwargs = dict(generation_kwargs)
    gen_kwargs.pop("max_new_tokens", None)
    use_autocast = device.type == "cuda"

    results: List[EvalResult] = []
    for cond in grid:
        cond_seed_bytes = hashlib.sha256(f"{cond.task}:{cond.name}:{seed}".encode()).digest()
        cond_seed = seed + int.from_bytes(cond_seed_bytes[:4], "little")
        n = limit if limit is not None else cond.n
        res = evaluate_condition(
            model,
            tokenizer,
            task=cond.task,
            condition=cond.name,
            params=cond.params,
            n=n,
            device=device,
            max_new_tokens=cond.max_new_tokens,
            seed=cond_seed,
            batch_size=batch_size,
            generation_kwargs=gen_kwargs,
        )
        results.append(res)

    return _algorithmic_results_to_dict(results)


@torch.no_grad()
def run_longctx_suite(
    model,
    tokenizer,
    device: torch.device,
    *,
    seed: int,
    generation_kwargs: Dict[str, Any],
    ctx_lengths: Sequence[int] = (4096, 16384),
    depths: Sequence[float] = (0.25, 0.5, 0.75, 0.9),
    samples_per_depth: int = 10,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    seed_all(seed)
    results = []
    gen_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        extra_kwargs=generation_kwargs,
    )

    for ctx_len in ctx_lengths:
        if hasattr(model, "config") and ctx_len > getattr(model.config, "max_seq_len", ctx_len):
            continue
        generator = NeedleInHaystackGenerator(tokenizer, context_length=ctx_len)
        by_depth = {}
        correct = 0
        total = 0
        for depth in depths:
            depth_correct = 0
            for _ in range(samples_per_depth):
                example = generator.generate(needle_depth=depth)
                input_ids = example["input_ids"].to(device).unsqueeze(0)
                outputs = model.generate(input_ids, **gen_kwargs)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Answer:" in decoded:
                    tail = decoded.split("Answer:")[-1].strip().split()
                else:
                    tail = decoded.split()
                predicted = tail[0] if tail else ""
                if predicted == example["answer"]:
                    depth_correct += 1
                    correct += 1
                total += 1
            by_depth[depth] = depth_correct / max(samples_per_depth, 1)
        retrieval_acc = correct / max(total, 1)
        results.append(
            {
                "context_length": ctx_len,
                "metrics": {
                    "retrieval_accuracy": retrieval_acc,
                    "by_depth": by_depth,
                    "area_under_depth_curve": area_under_depth_curve(by_depth),
                },
                "config": {
                    "depths": list(depths),
                    "samples_per_depth": samples_per_depth,
                    "max_new_tokens": max_new_tokens,
                },
            }
        )

    auc = 0.0
    if results:
        auc = sum(r["metrics"]["area_under_depth_curve"] for r in results) / len(results)

    return {"per_context": results, "auc": auc}


def run_eval_suite(
    model,
    tokenizer,
    device,
    *,
    suite: str,
    out_dir: str,
    seed: int = 1234,
    batch_size: int = 32,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = False,
    dtype: str = "bf16",
    compile_model: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the requested evaluation suite and persist standardized JSON."""

    device_obj = _normalize_device(device)
    model = _prepare_model(model, device_obj, dtype, compile_model)
    seed_all(seed)

    base_generation_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    suites_to_run = [suite]
    if suite == "all":
        suites_to_run = ["algorithmic", "longctx", "self_test"]

    results: Dict[str, Any] = {}
    for selected in suites_to_run:
        if selected == "algorithmic":
            results["algorithmic"] = run_algorithmic_suite(
                model,
                tokenizer,
                device_obj,
                seed=seed,
                batch_size=batch_size,
                generation_kwargs=base_generation_kwargs,
                limit=limit,
            )
        elif selected == "longctx":
            results["longctx"] = run_longctx_suite(
                model,
                tokenizer,
                device_obj,
                seed=seed,
                generation_kwargs=base_generation_kwargs,
                max_new_tokens=min(10, max_new_tokens),
            )
        elif selected == "self_test":
            results["self_test"] = run_self_test_suite(
                model,
                tokenizer,
                device_obj,
                seed=seed,
                generation_kwargs=base_generation_kwargs,
            )

    meta = gather_metadata(
        checkpoint=getattr(model, "checkpoint_path", None),
        tokenizer_name=getattr(tokenizer, "name_or_path", None),
        device=device_obj,
        model_config=getattr(model, "config", None),
        generation_kwargs=base_generation_kwargs,
        suite=suite,
        seed=seed,
        grid_version=ood_grid.OOD_GRID_VERSION if "algorithmic" in results else None,
        model_id=getattr(getattr(model, "config", None), "variant", None),
    )

    payload = {"meta": meta, **results}
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"results_{suite}.json" if suite != "all" else "results_all.json"
    save_json(payload, output_path / filename)
    return payload


def _load_for_cli(checkpoint: str, tokenizer_name: str, device: torch.device):
    model, config, tok = load_model_and_tokenizer(checkpoint, tokenizer_name, device)
    setattr(model, "checkpoint_path", checkpoint)
    return model, tok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation hub")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or path")
    parser.add_argument(
        "--suite",
        default="algorithmic",
        choices=["algorithmic", "longctx", "self_test", "all"],
    )
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional example cap")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = _normalize_device(args.device)
    model, tokenizer = _load_for_cli(args.checkpoint, args.tokenizer, device)
    payload = run_eval_suite(
        model,
        tokenizer,
        device,
        suite=args.suite,
        out_dir=args.out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        dtype=args.dtype,
        compile_model=args.compile_model,
        limit=args.limit,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
