"""Unified evaluation entrypoint for algorithmic, long-context, and PPL tasks.

Example usage:
    python eval_hub.py \
        --checkpoint checkpoints/model.pt \
        --tokenizer gpt2 \
        --tasks algorithmic parity_ood addition_ood needle tinystories \
        --context_lengths 512 1024 2048 \
        --algorithmic_examples 100 \
        --output_dir eval_results/ \
        --max_examples 200
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from data import AlgorithmicGenerator
from utils import (
    NeedleInHaystackGenerator,
    batch_generate,
    compute_perplexity,
    generate_text,
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


def evaluate_algorithmic_suite(
    model,
    tokenizer,
    device: torch.device,
    tasks: Sequence[str],
    n_examples: int,
    max_new_tokens: int,
) -> List[Dict]:
    """Run all requested algorithmic tasks and compute an overall score."""

    results: List[Dict] = []
    total_correct = 0
    total_seen = 0

    for task in tasks:
        res = evaluate_algorithmic_task(
            model,
            tokenizer,
            device,
            task=task,
            n_examples=n_examples,
            max_new_tokens=max_new_tokens,
        )
        results.append(res)
        total_correct += res["metrics"]["accuracy"] * n_examples
        total_seen += n_examples

    results.append(
        {
            "task": "algorithmic_overall",
            "metrics": {"accuracy": total_correct / max(total_seen, 1)},
            "config": {
                "tasks": list(tasks),
                "n_examples_per_task": n_examples,
                "max_new_tokens": max_new_tokens,
            },
        }
    )

    return results


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


def evaluate_parity_ood(
    model,
    tokenizer,
    device: torch.device,
    lengths: Sequence[int],
    samples_per_length: int,
    max_new_tokens: int,
) -> Dict:
    """Evaluate parity extrapolation to longer bit strings."""

    results_by_length: Dict[int, float] = {}
    total_correct = 0
    total_seen = 0

    for length in lengths:
        correct = 0
        for _ in range(samples_per_length):
            bits = [torch.randint(0, 2, ()).item() for _ in range(length)]
            expected = str(sum(bits) % 2)
            prompt = f"parity({''.join(map(str, bits))}) ="
            response = generate_text(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                generation_kwargs={"temperature": 0.1, "top_k": 1},
            )
            predicted = _extract_answer(response)
            if predicted == expected:
                correct += 1
                total_correct += 1
            total_seen += 1
        results_by_length[length] = correct / max(samples_per_length, 1)

    return {
        "task": "parity_ood",
        "metrics": {
            "accuracy": total_correct / max(total_seen, 1),
            "by_length": results_by_length,
        },
        "config": {
            "lengths": list(lengths),
            "samples_per_length": samples_per_length,
            "max_new_tokens": max_new_tokens,
        },
    }


def evaluate_addition_ood(
    model,
    tokenizer,
    device: torch.device,
    digit_counts: Sequence[int],
    samples_per_count: int,
    max_new_tokens: int,
) -> Dict:
    """Evaluate multi-digit addition beyond training lengths."""

    results_by_digits: Dict[int, float] = {}
    total_correct = 0
    total_seen = 0

    for n_digits in digit_counts:
        correct = 0
        for _ in range(samples_per_count):
            a = torch.randint(10 ** (n_digits - 1), 10**n_digits, ()).item()
            b = torch.randint(10 ** (n_digits - 1), 10**n_digits, ()).item()
            expected = str(a + b)
            prompt = f"{a} + {b} ="
            response = generate_text(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                generation_kwargs={"temperature": 0.1, "top_k": 1},
            )
            predicted = _extract_answer(response)
            if predicted == expected:
                correct += 1
                total_correct += 1
            total_seen += 1
        results_by_digits[n_digits] = correct / max(samples_per_count, 1)

    return {
        "task": "addition_ood",
        "metrics": {
            "accuracy": total_correct / max(total_seen, 1),
            "by_digits": results_by_digits,
        },
        "config": {
            "digit_counts": list(digit_counts),
            "samples_per_count": samples_per_count,
            "max_new_tokens": max_new_tokens,
        },
    }


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
        ppl = compute_perplexity(model, tokenizer, texts, device)
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
        algorithmic_tasks: Sequence[str],
        algorithmic_examples: int,
        ood_lengths: Sequence[int],
        ood_digit_counts: Sequence[int],
        ood_examples: int,
        max_new_tokens: int = 32,
        log_to_wandb: bool = False,
    ) -> None:
        self.device = device
        self.output_dir = output_dir
        self.tasks = list(tasks)
        self.context_lengths = list(context_lengths)
        self.max_examples = max_examples
        self.algorithmic_tasks = list(algorithmic_tasks)
        self.algorithmic_examples = algorithmic_examples
        self.ood_lengths = list(ood_lengths)
        self.ood_digit_counts = list(ood_digit_counts)
        self.ood_examples = ood_examples
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

    def _print_summary(self, aggregated: List[Dict]) -> None:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        overall = next((r for r in aggregated if r["task"] == "algorithmic_overall"), None)
        if overall:
            print(f"Algorithmic accuracy: {overall['metrics']['accuracy']*100:.1f}%")

        parity = next((r for r in aggregated if r["task"] == "parity_ood"), None)
        if parity:
            print(f"Parity OOD: {parity['metrics']['accuracy']*100:.1f}%")
            for length, acc in parity["metrics"].get("by_length", {}).items():
                print(f"  len={length}: {acc*100:.1f}%")

        addition = next((r for r in aggregated if r["task"] == "addition_ood"), None)
        if addition:
            print(f"Addition OOD: {addition['metrics']['accuracy']*100:.1f}%")
            for digits, acc in addition["metrics"].get("by_digits", {}).items():
                print(f"  {digits}-digit: {acc*100:.1f}%")

        for res in aggregated:
            if res["task"] == "needle":
                ctx = res["config"].get("context_length", "?")
                print(f"Needle @{ctx}: {res['metrics']['retrieval_accuracy']*100:.1f}%")

        for res in aggregated:
            if res["task"].startswith("tinystories"):
                print(f"TinyStories perplexity: {res['metrics']['perplexity']:.2f}")

        print("=" * 60)

    def run_all(self) -> Dict:
        """Dispatch requested tasks and aggregate results."""
        aggregated: List[Dict] = []
        for task in self.tasks:
            if task == "algorithmic":
                print("[Algorithmic Exact-Match]")
                algo_results = evaluate_algorithmic_suite(
                    self.model,
                    self.tokenizer,
                    self.device,
                    tasks=self.algorithmic_tasks,
                    n_examples=self.algorithmic_examples,
                    max_new_tokens=self.max_new_tokens,
                )
                for res in algo_results:
                    aggregated.append(res)
                    self._record(res)
                    if res["task"] != "algorithmic_overall":
                        print(f"  {res['task']}: {res['metrics']['accuracy']*100:.2f}%")
                overall = next(r for r in algo_results if r["task"] == "algorithmic_overall")
                print(f"  Overall: {overall['metrics']['accuracy']*100:.2f}%")
            elif task in set(self.algorithmic_tasks):
                print(f"[Algorithmic] {task}")
                res = evaluate_algorithmic_task(
                    self.model,
                    self.tokenizer,
                    self.device,
                    task=task,
                    n_examples=self.algorithmic_examples,
                    max_new_tokens=self.max_new_tokens,
                )
                aggregated.append(res)
                self._record(res)
                print(f"  accuracy: {res['metrics']['accuracy']*100:.2f}%")
            elif task == "parity_ood":
                print("[OOD Length] Parity")
                res = evaluate_parity_ood(
                    self.model,
                    self.tokenizer,
                    self.device,
                    lengths=self.ood_lengths,
                    samples_per_length=self.ood_examples,
                    max_new_tokens=self.max_new_tokens,
                )
                aggregated.append(res)
                self._record(res)
                print(f"  overall: {res['metrics']['accuracy']*100:.2f}%")
            elif task == "addition_ood":
                print("[OOD Length] Addition")
                res = evaluate_addition_ood(
                    self.model,
                    self.tokenizer,
                    self.device,
                    digit_counts=self.ood_digit_counts,
                    samples_per_count=self.ood_examples,
                    max_new_tokens=self.max_new_tokens,
                )
                aggregated.append(res)
                self._record(res)
                print(f"  overall: {res['metrics']['accuracy']*100:.2f}%")
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

        summary = {
            "results": aggregated,
            "model": json.loads(json.dumps(getattr(self.config, "__dict__", str(self.config)), default=str)),
            "tasks": self.tasks,
        }
        output_path = self.output_dir / "results.json"
        save_json(summary, output_path)
        print(f"Saved results to {output_path}")
        self._print_summary(aggregated)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation hub")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or path")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["algorithmic", "parity_ood", "addition_ood", "needle", "tinystories"],
        help="Tasks to run",
    )
    parser.add_argument("--context_lengths", nargs="*", type=int, default=[512, 1024, 2048], help="Context lengths for needle task")
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--max_examples", type=int, default=500, help="Examples for needle/tinystories")
    parser.add_argument("--algorithmic_tasks", nargs="*", default=["mod_add", "parity", "addition", "multiplication", "copy", "reverse", "chain", "compare", "successor"], help="Algorithmic tasks to include")
    parser.add_argument("--algorithmic_examples", type=int, default=100, help="Examples per algorithmic task")
    parser.add_argument("--ood_lengths", nargs="*", type=int, default=[8, 16, 24, 32, 48, 64], help="Lengths for parity OOD evaluation")
    parser.add_argument("--ood_digit_counts", nargs="*", type=int, default=[2, 4, 5, 6, 7, 8], help="Digit counts for addition OOD evaluation")
    parser.add_argument("--ood_examples", type=int, default=20, help="Examples per OOD setting")
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
        algorithmic_tasks=args.algorithmic_tasks,
        algorithmic_examples=args.algorithmic_examples,
        ood_lengths=args.ood_lengths,
        ood_digit_counts=args.ood_digit_counts,
        ood_examples=args.ood_examples,
        max_new_tokens=args.max_new_tokens,
        log_to_wandb=args.wandb,
    ).run_all()


if __name__ == "__main__":
    main()
