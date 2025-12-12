"""Unified evaluation entrypoint for algorithmic, long-context, and PPL tasks.

Example usage:
    python eval_hub.py \
        --checkpoint checkpoints/model.pt \
        --tokenizer gpt2 \
        --tasks addition dyck needle tinystories \
        --context_lengths 512 2048 4096 \
        --output_dir eval_results/ \
        --max_examples 200
"""
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from data import AlgorithmicGenerator
from utils import (
    NeedleInHaystackGenerator,
    batch_generate,
    load_model_and_tokenizer,
    save_json,
    seed_all,
    select_device,
)


def _extract_answer(text: str) -> str:
    """Extract a conservative answer token from model output."""
    if not text:
        return ""
    first_line = text.split("\n")[0].strip()
    import re

    numbers = re.findall(r"-?\d+", first_line)
    if numbers:
        return numbers[0]
    words = first_line.split()
    return words[0] if words else ""


def evaluate_algorithmic_task(
    model,
    tokenizer,
    device: torch.device,
    task: str,
    n_examples: int,
    max_new_tokens: int,
) -> Dict:
    """Evaluate a single algorithmic task with exact-match accuracy."""
    examples = AlgorithmicGenerator.generate_batch(n_examples, tasks=[task])
    prompts = [ex["input"] for ex in examples]
    targets = [ex["target"].strip() for ex in examples]

    generations = batch_generate(
        model,
        tokenizer,
        prompts,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    correct = 0
    for pred, tgt in zip(generations, targets):
        if _extract_answer(pred) == _extract_answer(tgt):
            correct += 1

    accuracy = correct / max(len(targets), 1)
    return {
        "task": task,
        "metrics": {"accuracy": accuracy},
        "config": {
            "n_examples": n_examples,
            "max_new_tokens": max_new_tokens,
        },
    }


def evaluate_longctx_needle(
    model,
    tokenizer,
    device: torch.device,
    context_lengths: Sequence[int],
    depths: Sequence[float],
    samples_per_depth: int,
    max_new_tokens: int = 10,
) -> List[Dict]:
    """Evaluate needle retrieval across multiple context lengths."""
    results: List[Dict] = []
    for ctx_len in context_lengths:
        if ctx_len > model.config.max_seq_len:
            continue
        generator = NeedleInHaystackGenerator(tokenizer, context_length=ctx_len)
        by_depth: Dict[float, float] = {}
        total_correct = 0
        total_seen = 0
        for depth in depths:
            depth_correct = 0
            for _ in range(samples_per_depth):
                ex = generator.generate(needle_depth=depth)
                input_ids = ex["input_ids"].to(device).unsqueeze(0)
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    top_k=1,
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Answer:" in decoded:
                    tail = decoded.split("Answer:")[-1].strip().split()
                else:
                    tail = decoded.split()
                predicted = tail[0] if tail else ""
                if predicted == ex["answer"]:
                    depth_correct += 1
                    total_correct += 1
                total_seen += 1
            by_depth[depth] = depth_correct / max(samples_per_depth, 1)

        retrieval_acc = total_correct / max(total_seen, 1)
        results.append(
            {
                "task": "needle",
                "metrics": {
                    "retrieval_accuracy": retrieval_acc,
                    "by_depth": by_depth,
                },
                "config": {
                    "context_length": ctx_len,
                    "depths": list(depths),
                    "samples_per_depth": samples_per_depth,
                    "max_new_tokens": max_new_tokens,
                },
            }
        )
    return results


def _compute_perplexity(model, tokenizer, texts: Iterable[str], device: torch.device) -> float:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(
                text,
                max_length=model.config.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            if tokens.size(1) < 2:
                continue
            labels = tokens.clone()
            _, loss, _ = model(tokens, labels=labels)
            n_tokens = tokens.size(1) - 1
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    average_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(min(average_loss, 20.0))).item())


def evaluate_tinystories_ppl(
    model,
    tokenizer,
    device: torch.device,
    n_examples: int,
) -> Dict:
    """Compute perplexity on a subset of TinyStories validation split."""
    from datasets import load_dataset

    try:
        dataset = load_dataset("roneneldan/TinyStories", split="validation")
        texts = [dataset[i]["text"] for i in range(min(n_examples, len(dataset)))]
        ppl = _compute_perplexity(model, tokenizer, texts, device)
    except Exception as exc:  # noqa: BLE001
        ppl = float("inf")
        texts = []
        print(f"Failed to compute TinyStories perplexity: {exc}")

    return {
        "task": "tinystories",
        "metrics": {"perplexity": ppl},
        "config": {"n_examples": len(texts)},
    }


class EvaluatorHub:
    """Coordinated dispatcher for multiple evaluation tasks."""

    def __init__(
        self,
        checkpoint: str,
        tokenizer_name: str,
        device: torch.device,
        tasks: Sequence[str],
        context_lengths: Sequence[int],
        output_dir: Path,
        max_examples: int,
        max_new_tokens: int = 32,
        log_to_wandb: bool = False,
    ) -> None:
        self.device = device
        self.output_dir = output_dir
        self.tasks = list(tasks)
        self.context_lengths = list(context_lengths)
        self.max_examples = max_examples
        self.max_new_tokens = max_new_tokens
        self.log_to_wandb = log_to_wandb

        self.model, self.config, self.tokenizer = load_model_and_tokenizer(
            checkpoint, tokenizer_name, device
        )

        if log_to_wandb:
            import wandb

            wandb.init(project="relsm-eval", config={"checkpoint": checkpoint, "tasks": tasks})
            self.wandb = wandb
        else:
            self.wandb = None

    def _record(self, result: Dict) -> None:
        if self.wandb:
            self.wandb.log({f"{result['task']}/{k}": v for k, v in result["metrics"].items()})

    def run_all(self) -> Dict:
        """Dispatch requested tasks and aggregate results."""
        aggregated: List[Dict] = []
        for task in self.tasks:
            if task in {"addition", "dyck", "chain", "mod_add", "copy", "reverse", "parity", "multiplication", "compare", "successor"}:
                print(f"[Algorithmic] {task}")
                res = evaluate_algorithmic_task(
                    self.model,
                    self.tokenizer,
                    self.device,
                    task=task,
                    n_examples=self.max_examples,
                    max_new_tokens=self.max_new_tokens,
                )
                aggregated.append(res)
                self._record(res)
                print(f"  accuracy: {res['metrics']['accuracy']*100:.2f}%")
            elif task == "needle":
                print("[Needle retrieval]")
                needle_results = evaluate_longctx_needle(
                    self.model,
                    self.tokenizer,
                    self.device,
                    context_lengths=self.context_lengths,
                    depths=[0.25, 0.5, 0.75, 0.9],
                    samples_per_depth=max(1, self.max_examples // len(self.context_lengths) // 4),
                )
                for res in needle_results:
                    aggregated.append(res)
                    self._record(res)
                    ctx = res["config"]["context_length"]
                    print(f"  ctx={ctx} retrieval: {res['metrics']['retrieval_accuracy']*100:.2f}%")
            elif task == "tinystories":
                print("[Perplexity] TinyStories")
                res = evaluate_tinystories_ppl(
                    self.model, self.tokenizer, self.device, n_examples=self.max_examples
                )
                aggregated.append(res)
                self._record(res)
                print(f"  ppl: {res['metrics']['perplexity']:.2f}")
            else:
                print(f"Unknown task '{task}', skipping")

        summary = {"results": aggregated, "model": str(self.config), "tasks": self.tasks}
        output_path = self.output_dir / "results.json"
        save_json(summary, output_path)
        print(f"Saved results to {output_path}")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation hub")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--tasks", nargs="*", default=["addition", "needle", "tinystories"], help="Tasks to run")
    parser.add_argument("--context_lengths", nargs="*", type=int, default=[512, 2048, 4096], help="Context lengths for needle task")
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = select_device(args.device)
    EvaluatorHub(
        checkpoint=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=device,
        tasks=args.tasks,
        context_lengths=args.context_lengths,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        log_to_wandb=args.wandb,
    ).run_all()


if __name__ == "__main__":
    main()
