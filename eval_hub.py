"""Canonical evaluation orchestrator for ReLSM.

This module exposes ``run_eval_suite`` as the single entrypoint for all
evaluation flows (algorithmic IID/OOD grids, long-context retrieval, and
lightweight self tests). The accompanying CLI mirrors the same API and
writes standardized JSON outputs.
"""

from __future__ import annotations

import argparse
import random
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from eval import ood_grid
from eval.self_test import run_self_test_suite
from utils import (
    DEFAULT_TOKENIZER_NAME,
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


def _parse_answer_from_continuation(decoded: str) -> str:
    """Extract the first token of the answer from a decoded continuation."""

    if "Answer:" in decoded:
        tail = decoded.split("Answer:")[-1].strip()
    else:
        tail = decoded.strip()

    number_match = re.search(r"\b\d+\b", tail)
    if number_match:
        return number_match.group(0)

    words = tail.split()
    return words[0] if words else ""


def _prediction_from_output_ids(
    tokenizer: Any, output_ids: Sequence[int] | torch.Tensor, prompt_len: int
) -> str:
    """Decode and parse only the continuation portion of a generation."""

    continuation_ids = output_ids[prompt_len:]
    if isinstance(continuation_ids, torch.Tensor):
        continuation_ids = continuation_ids.tolist()
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return _parse_answer_from_continuation(continuation)


def _algorithmic_results_to_dict(results: List[EvalResult]) -> Dict[str, Any]:
    conditions: List[Dict[str, Any]] = []
    per_task: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    total_seen = 0
    total_token_accuracy = 0.0
    total_distance = 0.0
    total_prefix_accuracy = 0.0
    total_abs_error = 0.0
    total_numeric = 0
    total_parse_failures = 0

    for res in results:
        conditions.append(
            {
                "task": res.task,
                "condition": res.condition,
                "difficulty": res.difficulty,
                "accuracy": res.accuracy,
                "mean_token_accuracy": res.mean_token_accuracy,
                "mean_distance": res.mean_distance,
                "mean_prefix_accuracy": res.mean_prefix_accuracy,
                "mae": res.mae,
                "numeric_count": res.numeric_count,
                "parse_failures": res.parse_failures,
                "parse_failure_rate": (res.parse_failures / res.n if res.n else 0.0),
                "n": res.n,
                "avg_gen_len": res.avg_gen_len,
                "tokens_per_sec": res.tokens_per_sec,
                "tokens_generated": res.tokens_generated,
                "elapsed_seconds": res.elapsed,
                "examples": res.examples,
                "sampled_examples": res.sampled_examples,
            }
        )
        total_correct += res.correct
        total_seen += res.n
        total_token_accuracy += res.mean_token_accuracy * res.n
        total_distance += res.mean_distance * res.n
        total_prefix_accuracy += res.mean_prefix_accuracy * res.n
        per_task.setdefault(
            res.task,
            {
                "conditions": [],
                "correct": 0,
                "total": 0,
                "token_accuracy": 0.0,
                "distance": 0.0,
                "prefix_accuracy": 0.0,
                "mae_error": 0.0,
                "mae_count": 0,
                "parse_failures": 0,
            },
        )
        per_task[res.task]["conditions"].append(res.condition)
        per_task[res.task]["correct"] += res.correct
        per_task[res.task]["total"] += res.n
        per_task[res.task]["token_accuracy"] += res.mean_token_accuracy * res.n
        per_task[res.task]["distance"] += res.mean_distance * res.n
        per_task[res.task]["prefix_accuracy"] += res.mean_prefix_accuracy * res.n
        if res.mae is not None and res.numeric_count > 0:
            error_sum = res.mae * res.numeric_count
            per_task[res.task]["mae_error"] += error_sum
            per_task[res.task]["mae_count"] += res.numeric_count
            total_abs_error += error_sum
            total_numeric += res.numeric_count
        per_task[res.task]["parse_failures"] += res.parse_failures
        total_parse_failures += res.parse_failures

    per_task_acc = {
        task: (payload["correct"] / payload["total"] if payload["total"] else 0.0)
        for task, payload in per_task.items()
    }
    per_task_token_accuracy = {
        task: (payload["token_accuracy"] / payload["total"] if payload["total"] else 0.0)
        for task, payload in per_task.items()
    }
    per_task_distance = {
        task: (payload["distance"] / payload["total"] if payload["total"] else 0.0)
        for task, payload in per_task.items()
    }
    per_task_prefix_accuracy = {
        task: (payload["prefix_accuracy"] / payload["total"] if payload["total"] else 0.0)
        for task, payload in per_task.items()
    }
    per_task_mae: Dict[str, Optional[float]] = {
        task: (payload["mae_error"] / payload["mae_count"] if payload["mae_count"] else None)
        for task, payload in per_task.items()
    }
    per_task_parse_failure_rate = {
        task: (payload["parse_failures"] / payload["total"] if payload["total"] else 0.0)
        for task, payload in per_task.items()
    }
    overall = total_correct / total_seen if total_seen else 0.0
    overall_token_accuracy = total_token_accuracy / total_seen if total_seen else 0.0
    overall_distance = total_distance / total_seen if total_seen else 1.0
    overall_prefix_accuracy = total_prefix_accuracy / total_seen if total_seen else 0.0
    overall_mae = total_abs_error / total_numeric if total_numeric else None
    overall_parse_failure_rate = total_parse_failures / total_seen if total_seen else 0.0
    return {
        "grid_version": ood_grid.OOD_GRID_VERSION,
        "conditions": conditions,
        "per_task_accuracy": per_task_acc,
        "per_task_token_accuracy": per_task_token_accuracy,
        "per_task_distance": per_task_distance,
        "per_task_prefix_accuracy": per_task_prefix_accuracy,
        "per_task_mae": per_task_mae,
        "per_task_parse_failure_rate": per_task_parse_failure_rate,
        "overall_accuracy": overall,
        "overall_token_accuracy": overall_token_accuracy,
        "overall_distance": overall_distance,
        "overall_prefix_accuracy": overall_prefix_accuracy,
        "overall_mae": overall_mae,
        "overall_parse_failures": total_parse_failures,
        "overall_parse_failure_rate": overall_parse_failure_rate,
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
    sample_count_per_task: int = 5,
    grid: str = "iid",
) -> Dict[str, Any]:
    seed_all(seed)
    grid_key = grid.lower()
    include_iid = grid_key in {"iid", "all"}
    include_ood = grid_key in {"ood", "all"}
    if not include_iid and not include_ood:
        raise ValueError(f"Unknown algorithmic grid selection: {grid}")
    grid_conditions = ood_grid.build_grid(
        list(tasks) if tasks else None,
        include_iid=include_iid,
        include_ood=include_ood,
    )
    gen_kwargs = dict(generation_kwargs)
    gen_kwargs.pop("max_new_tokens", None)
    use_autocast = device.type == "cuda"

    results: List[EvalResult] = []
    sampled_examples_by_task: Dict[str, List[Dict[str, str]]] = {}
    sample_seen_by_task: Dict[str, int] = {}
    sample_rng = random.Random(seed + 101)
    for cond in grid_conditions:
        cond_seed_bytes = hashlib.sha256(f"{cond.task}:{cond.name}:{seed}".encode()).digest()
        cond_seed = seed + int.from_bytes(cond_seed_bytes[:4], "little")
        n = limit if limit is not None else cond.n
        difficulty = cond.params.get("difficulty", 0.5)
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
            difficulty=difficulty,
            sample_count=sample_count_per_task,
        )
        results.append(res)
        if sample_count_per_task:
            task_samples = sampled_examples_by_task.setdefault(cond.task, [])
            seen = sample_seen_by_task.get(cond.task, 0)
            for sample in res.sampled_examples:
                if len(task_samples) < sample_count_per_task:
                    task_samples.append(sample)
                else:
                    idx = sample_rng.randint(0, seen)
                    if idx < sample_count_per_task:
                        task_samples[idx] = sample
                seen += 1
            sample_seen_by_task[cond.task] = seen
            if res.sampled_examples:
                print(
                    f"[eval] {cond.task}/{cond.name} difficulty={difficulty:.3f} "
                    f"samples={len(res.sampled_examples)}"
                )
                for idx, sample in enumerate(res.sampled_examples, start=1):
                    expected_output = sample.get("expected_output", sample.get("target"))
                    print(
                        "  "
                        f"{idx}. prompt={sample['prompt']!r} "
                        f"prediction={sample['prediction']!r} "
                        f"expected_output={expected_output!r}"
                    )

    payload = _algorithmic_results_to_dict(results)
    payload["grid"] = grid_key
    if sample_count_per_task:
        payload["sampled_examples_by_task"] = sampled_examples_by_task
    return payload


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
                outputs = model.generate(input_ids=input_ids, **gen_kwargs)
                prompt_len = input_ids.shape[1]
                predicted = _prediction_from_output_ids(
                    tokenizer, outputs[0], prompt_len
                )
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
    tasks: Optional[Sequence[str]] = None,
    sample_count_per_task: int = 5,
    algorithmic_grid: str = "iid",
) -> Dict[str, Any]:
    """Run the requested evaluation suite and persist standardized JSON.

    Results are saved after each suite finishes and again if evaluation is
    interrupted via ``KeyboardInterrupt`` so partial progress is always
    captured.
    """

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

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if suite == "algorithmic":
        filename = f"results_{suite}_{algorithmic_grid}.json"
    else:
        filename = f"results_{suite}.json" if suite != "all" else "results_all.json"

    results: Dict[str, Any] = {}

    def _build_payload(interrupted: bool = False) -> Dict[str, Any]:
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
        if "algorithmic" in results:
            meta["algorithmic_grid"] = algorithmic_grid
        if interrupted:
            meta["status"] = "interrupted"
        return {"meta": meta, **results}

    for selected in suites_to_run:
        try:
            if selected == "algorithmic":
                results["algorithmic"] = run_algorithmic_suite(
                    model,
                    tokenizer,
                    device_obj,
                    seed=seed,
                    batch_size=batch_size,
                    generation_kwargs=base_generation_kwargs,
                    limit=limit,
                    tasks=tasks,
                    sample_count_per_task=sample_count_per_task,
                    grid=algorithmic_grid,
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
        except KeyboardInterrupt:
            payload = _build_payload(interrupted=True)
            save_json(payload, output_path / filename)
            raise

        payload = _build_payload()
        save_json(payload, output_path / filename)

    return payload


def _load_for_cli(checkpoint: str, tokenizer_name: str, device: torch.device):
    model, config, tok = load_model_and_tokenizer(checkpoint, tokenizer_name, device)
    setattr(model, "checkpoint_path", checkpoint)
    return model, tok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation hub")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER_NAME, help="Tokenizer name or path"
    )
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
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of algorithmic tasks (e.g., addition dyck copy)",
    )
    parser.add_argument(
        "--eval_sample_count_per_task",
        type=int,
        default=5,
        help="Number of sample prompts to log per active task",
    )
    parser.add_argument(
        "--algorithmic_grid",
        default="iid",
        choices=["iid", "ood", "all"],
        help="Select IID vs OOD conditions for the algorithmic suite",
    )
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
        tasks=args.tasks,
        sample_count_per_task=args.eval_sample_count_per_task,
        algorithmic_grid=args.algorithmic_grid,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
