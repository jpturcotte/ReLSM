"""Canonical evaluation hub for ReLSM models.

Usage:
    python eval_hub.py --checkpoint ./runs/nano_baseline/best_model.pt --all
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from eval.run_algorithmic_eval import run_evaluation as run_algo_ood
from evaluate_longctx import run_longctx_eval
from utils import (
    EvalResult,
    compute_perplexity,
    gather_metadata,
    get_eval_generation_kwargs,
    load_model_and_tokenizer,
    save_json,
    seed_all,
    select_device,
    write_ood_csv,
    write_summary_md,
)

DEFAULT_TASKS = ["algorithmic", "ood", "needle", "tinystories"]


def evaluate_tinystories_ppl(
    model,
    tokenizer,
    device: torch.device,
    n_examples: int,
    seed: int,
    texts: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute TinyStories perplexity on a small subset."""

    seed_all(seed)
    samples: List[str] = []
    if texts is not None:
        samples = list(texts)[:n_examples]
    else:
        try:
            from datasets import load_dataset

            dataset = load_dataset("roneneldan/TinyStories", split="validation")
            samples = [dataset[i]["text"] for i in range(min(n_examples, len(dataset)))]
        except Exception:
            samples = []

    ppl = float("inf")
    if samples:
        ppl = compute_perplexity(model, tokenizer, samples, device)

    return {"perplexity": ppl, "n": len(samples)}


class EvaluatorHub:
    """Unified dispatcher over algorithmic IID/OOD, long-context, and PPL tasks."""

    def __init__(
        self,
        checkpoint: str,
        tokenizer_name: str,
        device: torch.device,
        tasks: Sequence[str],
        grid_tasks: Sequence[str],
        output_dir: Path,
        seed: int,
        batch_size: int,
        n_override: Optional[int],
        needle_contexts: Sequence[int],
        needle_depths: Sequence[float],
        needle_samples: int,
        ppl_samples: int,
        max_new_tokens: int,
    ) -> None:
        self.device = device
        self.tasks = list(tasks)
        self.grid_tasks = list(grid_tasks)
        self.output_dir = output_dir
        self.seed = seed
        self.batch_size = batch_size
        self.n_override = n_override
        self.needle_contexts = list(needle_contexts)
        self.needle_depths = list(needle_depths)
        self.needle_samples = needle_samples
        self.ppl_samples = ppl_samples
        self.max_new_tokens = max_new_tokens

        self.model, self.config, self.tokenizer = load_model_and_tokenizer(
            checkpoint, tokenizer_name, device
        )
        self.generation_overrides = get_eval_generation_kwargs(tokenizer=self.tokenizer)
        self.generation_overrides.pop("max_new_tokens", None)
        self.metadata = gather_metadata(
            checkpoint=checkpoint,
            tokenizer_name=tokenizer_name,
            device=device,
            model_config=self.config,
            generation_kwargs={"do_sample": False, "top_k": 1, "top_p": None, "temperature": 1.0},
        )

    def _convert_algo_results(self, results: List[EvalResult]) -> Dict:
        per_task_iid: Dict[str, float] = {}
        details = []
        total_iid = 0.0
        total_tasks = 0

        for res in results:
            row = {
                "task": res.task,
                "condition": res.condition,
                "accuracy": res.accuracy,
                "n": res.n,
                "avg_gen_len": res.avg_gen_len,
                "tokens_per_sec": res.tokens_per_sec,
            }
            details.append(row)
            if res.condition == "iid":
                per_task_iid[res.task] = res.accuracy
                total_iid += res.accuracy
                total_tasks += 1

        overall_iid = total_iid / total_tasks if total_tasks else 0.0
        overall_all = sum(r.correct for r in results) / max(
            sum(r.n for r in results), 1
        )
        return {
            "per_task": per_task_iid,
            "overall_accuracy": overall_iid,
            "overall_all_conditions": overall_all,
            "details": details,
        }

    def _ood_table(self, results: List[EvalResult]) -> Dict:
        table = [
            {
                "task": res.task,
                "condition": res.condition,
                "accuracy": res.accuracy,
                "n": res.n,
                "avg_gen_len": res.avg_gen_len,
                "tokens_per_sec": res.tokens_per_sec,
            }
            for res in results
        ]
        per_task: Dict[str, float] = {}
        for res in results:
            per_task[res.task] = max(per_task.get(res.task, 0.0), res.accuracy)
        overall = sum(r.correct for r in results) / max(sum(r.n for r in results), 1)
        return {"table": table, "per_task_max": per_task, "overall_accuracy": overall}

    def run_algorithmic_and_ood(self) -> Dict:
        output = run_algo_ood(
            ckpt_path=None,
            tokenizer_name=None,
            device=str(self.device),
            seed=self.seed,
            tasks=self.grid_tasks,
            out_path=None,
            batch_size=self.batch_size,
            n_override=self.n_override,
            model=self.model,
            tokenizer=self.tokenizer,
            verbose=False,
            iid_only="ood" not in self.tasks,
        )
        results: List[EvalResult] = output["results"]
        algorithmic = self._convert_algo_results(results)
        ood = self._ood_table(results)
        return {"algorithmic": algorithmic, "ood": ood}

    def run_longctx(self) -> Dict:
        longctx = run_longctx_eval(
            self.model,
            self.tokenizer,
            self.device,
            ctx_lengths=self.needle_contexts,
            depths=self.needle_depths,
            samples_per_depth=self.needle_samples,
            seed=self.seed,
            max_new_tokens=min(10, self.max_new_tokens),
        )
        return longctx

    def run_ppl(self) -> Dict:
        return evaluate_tinystories_ppl(
            self.model,
            self.tokenizer,
            self.device,
            n_examples=self.ppl_samples,
            seed=self.seed,
        )

    def run_all(self) -> Dict:
        seed_all(self.seed)
        results: Dict[str, Dict] = {}

        if "algorithmic" in self.tasks or "ood" in self.tasks:
            algo_payload = self.run_algorithmic_and_ood()
            results.update(algo_payload)

        if "needle" in self.tasks:
            results["longctx"] = self.run_longctx()

        if "tinystories" in self.tasks:
            results["ppl"] = self.run_ppl()

        summary = {"metadata": self.metadata, "results": results}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(summary, self.output_dir / "results.json")

        if "ood" in results:
            write_ood_csv(self.output_dir / "results_ood.csv", results["ood"].get("table", []))
        write_summary_md(self.output_dir / "summary.md", summary)

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: Dict) -> None:
        print("\n=== Evaluation Summary ===")
        algo = summary.get("results", {}).get("algorithmic", {})
        if algo:
            print(f"Algorithmic IID overall: {algo.get('overall_accuracy', 0.0)*100:.2f}%")
            for task, acc in sorted(algo.get("per_task", {}).items()):
                print(f"  {task}: {acc*100:.2f}%")
        ood = summary.get("results", {}).get("ood", {})
        if ood:
            print(f"OOD overall: {ood.get('overall_accuracy', 0.0)*100:.2f}% across {len(ood.get('table', []))} conditions")
        longctx = summary.get("results", {}).get("longctx", {})
        if longctx:
            print(f"Needle AUC: {longctx.get('auc', 0.0)*100:.2f}%")
            for ctx in longctx.get("per_context", []):
                print(
                    f"  ctx={ctx['context_length']}: {ctx['metrics']['retrieval_accuracy']*100:.2f}%"
                )
        ppl = summary.get("results", {}).get("ppl", {})
        if ppl:
            print(f"TinyStories PPL: {ppl.get('perplexity', float('inf')):.2f} (n={ppl.get('n', 0)})")
        print("==========================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation hub")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="Tasks to run: algorithmic ood needle tinystories")
    parser.add_argument("--all", action="store_true", help="Convenience flag for all tasks")
    parser.add_argument("--grid_tasks", nargs="*", default=None, help="Subset of algorithmic/OOD tasks to evaluate")
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n", type=int, default=None, help="Override per-condition example count for algorithmic/OOD")
    parser.add_argument("--needle_contexts", nargs="*", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--needle_depths", nargs="*", type=float, default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--needle_samples", type=int, default=8)
    parser.add_argument("--ppl_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = DEFAULT_TASKS if args.all else args.tasks
    device = select_device(args.device)

    hub = EvaluatorHub(
        checkpoint=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=device,
        tasks=tasks,
        grid_tasks=args.grid_tasks or [],
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        n_override=args.n,
        needle_contexts=args.needle_contexts,
        needle_depths=args.needle_depths,
        needle_samples=args.needle_samples,
        ppl_samples=args.ppl_samples,
        max_new_tokens=args.max_new_tokens,
    )
    hub.run_all()


if __name__ == "__main__":
    main()
