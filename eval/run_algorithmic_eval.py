"""CLI for canonical algorithmic IID/OOD evaluation."""
import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch

from eval import ood_grid
from utils import EvalResult, evaluate_condition, seed_all, select_device


def _load_model(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        from model import NanoGPT

        config = checkpoint.get("config")
        model = NanoGPT(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = checkpoint
    model.to(device)
    model.eval()
    return model


def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def summarize(results: List[EvalResult]) -> Dict[str, float]:
    task_acc: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for res in results:
        correct = res.correct
        prev_correct, prev_total = task_acc[res.task]
        task_acc[res.task] = (prev_correct + correct, prev_total + res.n)
    summary = {
        task: (correct / total if total else 0.0)
        for task, (correct, total) in task_acc.items()
    }
    total_correct = sum(c for c, _ in task_acc.values())
    total_seen = sum(t for _, t in task_acc.values())
    summary["overall"] = total_correct / total_seen if total_seen else 0.0
    return summary


def run_evaluation(
    ckpt_path: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 123,
    tasks: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    batch_size: int = 16,
    model=None,
    tokenizer=None,
) -> Dict:
    seed_all(seed)
    device_obj = select_device(device)
    tokenizer = tokenizer or ( _load_tokenizer(tokenizer_name) if tokenizer_name else None)
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided or tokenizer_name must be set")
    model = model or (_load_model(ckpt_path, device_obj) if ckpt_path else None)
    if model is None:
        raise ValueError("Model must be provided or ckpt_path must be set")

    grid = ood_grid.build_grid(tasks)
    results: List[EvalResult] = []

    for cond in grid:
        h = hashlib.sha256(f"{cond.task}:{cond.name}:{seed}".encode()).digest()
        cond_seed = seed + int.from_bytes(h[:4], "little")
        res = evaluate_condition(
            model,
            tokenizer,
            task=cond.task,
            condition=cond.name,
            params=cond.params,
            n=cond.n,
            device=device_obj,
            max_new_tokens=cond.max_new_tokens,
            seed=cond_seed,
            batch_size=batch_size,
        )
        results.append(res)

    summary = summarize(results)

    if out_path:
        commit = None
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
                .decode()
                .strip()
            )
        except Exception:
            commit = None
        metadata = {
            "checkpoint": ckpt_path,
            "device": str(device_obj),
            "dtype": str(next(model.parameters()).dtype),
            "timestamp": datetime.utcnow().isoformat(),
            "commit": commit,
        }
        output = {
            "summary": summary,
            "results": [res.__dict__ for res in results],
            "metadata": metadata,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    # Print table
    header = f"{'task':<12}{'condition':<30}{'N':<6}{'acc':<8}{'avg_gen':<12}{'tok/s':<10}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.task:<12}{res.condition:<30}{res.n:<6}{res.accuracy*100:>6.2f}  {res.avg_gen_len:>8.2f}    {res.tokens_per_sec:>8.1f}"
        )
    print("Overall accuracy:", summary["overall"])

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate algorithmic generalization (IID + OOD)")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--device", default=None, help="Device to run on")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks to evaluate")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_evaluation(
        ckpt_path=args.ckpt,
        tokenizer_name=args.tokenizer,
        device=args.device,
        seed=args.seed,
        tasks=args.tasks,
        out_path=args.out,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
