"""
Split Curriculum Training
=========================
Implements the ReLSM training strategy:

Phase 1 - Algorithmic Grokking:
  - Synthetic data, high epochs, force algorithm learning
  - Log: algorithmic accuracy per task, convergence speed

Phase 2 - Language Generalization:
  - Real text data, few epochs, learn language mapping
  - Log: perplexity, avoid overfitting

Metrics tracked for ladder comparison:
  - tokens/sec
  - peak VRAM
  - avg inner steps (for latent/act variants)
  - per-task accuracy
  - needle retrieval @ 4k/16k
"""

import os
import sys
import time
import math
import json
import random
import argparse
import re
from collections import defaultdict
from multiprocessing import Value
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, Dict, List, Sequence, Any

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset

from utils import (
    DEFAULT_TOKENIZER_NAME,
    aggregate,
    compute_metrics,
    normalize_prediction,
    normalize_target,
)

plt.switch_backend("Agg")

def get_lr(progress: int, warmup_progress: int, max_progress: int, max_lr: float, min_lr: float) -> float:
    """Cosine schedule with warmup based on generic progress units (e.g., tokens)."""
    current_progress = min(progress, max_progress)
    if progress < warmup_progress:
        return max_lr * (progress + 1) / max(1, warmup_progress)
    progress_frac = (current_progress - warmup_progress) / max(1, max_progress - warmup_progress)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress_frac))


class MetricsLogger:
    """Tracks metrics for the ablation ladder."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.hyperparameters = {}
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "tokens_seen": [],
            "tokens_per_sec": [],
            "peak_vram_gb": [],
            "avg_inner_steps": [],
            "phase": [],
            "step": [],
        }
        self.evaluations = []
        self.metric_steps = {}
        self.task_accuracies = {}
        self.train_task_accuracies = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.output_dir / "metrics.json"
        if not self._metrics_path.exists():
            self._append_snapshot()
    
    def log(self, step: int, phase: str, **kwargs):
        entry = {"step": step, "phase": phase}
        entry.update(kwargs)
        self.evaluations.append(entry)

        self.metrics["step"].append(step)
        self.metrics["phase"].append(phase)
        
        for k, v in kwargs.items():
            if k not in self.metrics:
                self.metrics[k] = []
            if k not in self.metric_steps:
                self.metric_steps[k] = []
            self.metrics[k].append(v)
            self.metric_steps[k].append(step)
    
    def log_task_accuracy(
        self,
        task: str,
        accuracy: float,
        step: int,
        target: str = "eval",
        mae: Optional[float] = None,
        mean_token_accuracy: Optional[float] = None,
        mean_distance: Optional[float] = None,
        mean_prefix_accuracy: Optional[float] = None,
    ):
        if target == "train":
            store = self.train_task_accuracies
        else:
            store = self.task_accuracies
        if task not in store:
            store[task] = []
        entry = {"step": step, "accuracy": accuracy}
        if mae is not None:
            entry["mae"] = mae
        if mean_token_accuracy is not None:
            entry["mean_token_accuracy"] = mean_token_accuracy
        if mean_distance is not None:
            entry["mean_distance"] = mean_distance
        if mean_prefix_accuracy is not None:
            entry["mean_prefix_accuracy"] = mean_prefix_accuracy
        store[task].append(entry)
    
    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._append_snapshot()

    def set_hyperparameters(self, hyperparameters: Dict) -> None:
        self.hyperparameters = hyperparameters

    def _append_snapshot(self):
        snapshot = {
            "hyperparameters": self.hyperparameters,
            "training": self.metrics,
            "evaluations": self.evaluations,
            "task_accuracies": self.task_accuracies,
            "train_task_accuracies": self.train_task_accuracies,
        }
        tmp_path = self._metrics_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(snapshot, f)
            f.write("\n")
        os.replace(tmp_path, self._metrics_path)
    
    def summary(self) -> Dict:
        return {
            "final_train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else None,
            "final_val_loss": self.metrics["val_loss"][-1] if self.metrics["val_loss"] else None,
            "peak_vram_gb": max(self.metrics["peak_vram_gb"]) if self.metrics["peak_vram_gb"] else None,
            "avg_tokens_per_sec": sum(self.metrics["tokens_per_sec"]) / len(self.metrics["tokens_per_sec"]) if self.metrics["tokens_per_sec"] else None,
            "total_tokens": self.metrics["tokens_seen"][-1] if self.metrics["tokens_seen"] else 0,
        }

    def plot(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss plot
        plt.figure()
        train_steps = self.metric_steps.get("train_loss", [])
        train_losses = self.metrics.get("train_loss", [])
        if train_steps and train_losses:
            plt.plot(train_steps, train_losses, label="train")

        val_steps = self.metric_steps.get("val_loss", [])
        val_losses = self.metrics.get("val_loss", [])
        if val_steps and val_losses:
            plt.plot(val_steps, val_losses, label="val")

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training and Validation Loss")
        if train_steps or val_steps:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss.png")
        plt.close()

        # Accuracy plot
        plt.figure()
        for task, records in self.task_accuracies.items():
            steps = [entry["step"] for entry in records]
            accuracies = [entry["accuracy"] for entry in records]
            if steps and accuracies:
                plt.plot(steps, accuracies, label=task)

        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Task Accuracy")
        if self.task_accuracies:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy.png")
        plt.close()

        # Training task accuracy plot
        if self.train_task_accuracies:
            plt.figure()
            for task, records in self.train_task_accuracies.items():
                steps = [entry["step"] for entry in records]
                accuracies = [entry["accuracy"] for entry in records]
                if steps and accuracies:
                    plt.plot(steps, accuracies, label=task)

            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.title("Training Task Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / "train_task_accuracy.png")
            plt.close()


@torch.no_grad()
def evaluate_algorithmic(
    model,
    tokenizer,
    device,
    n_examples: int = 100,
    max_new_tokens: int = 32,
    tasks: Optional[Sequence[str]] = None,
    seed: int = 42,
    sample_count_per_task: int = 5,
) -> Dict[str, Any]:
    """Run the canonical algorithmic OOD grid using ``eval_hub``.

    The ``n_examples`` argument caps the number of examples per condition
    (kept for backward compatibility with older training scripts), while
    ``max_new_tokens`` constrains generation length to avoid runaway decoding
    when EOS is missed.
    """

    from eval_hub import run_algorithmic_suite
    from utils import get_eval_generation_kwargs

    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    generation_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer, max_new_tokens=max_new_tokens
    )

    results = run_algorithmic_suite(
        model,
        tokenizer,
        device_obj,
        seed=seed,
        batch_size=8,
        generation_kwargs=generation_kwargs,
        limit=n_examples,
        tasks=tasks,
        sample_count_per_task=sample_count_per_task,
    )

    return {
        "overall_accuracy": results.get("overall_accuracy", 0.0),
        "overall_token_accuracy": results.get("overall_token_accuracy", 0.0),
        "overall_distance": results.get("overall_distance", 1.0),
        "overall_prefix_accuracy": results.get("overall_prefix_accuracy", 0.0),
        "per_task_accuracy": results.get("per_task_accuracy", {}),
        "per_task_token_accuracy": results.get("per_task_token_accuracy", {}),
        "per_task_distance": results.get("per_task_distance", {}),
        "per_task_prefix_accuracy": results.get("per_task_prefix_accuracy", {}),
        "overall_mae": results.get("overall_mae"),
        "per_task_mae": results.get("per_task_mae", {}),
        "sampled_examples_by_task": results.get("sampled_examples_by_task", {}),
    }


@torch.no_grad()
def evaluate_perplexity(model, val_loader, device, ctx, max_batches: int = 50) -> float:
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        with ctx:
            _, loss, _ = model(input_ids, labels=labels)
        
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    
    model.train()
    
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))


@torch.no_grad()
def evaluate_training_split(
    model,
    loader,
    device,
    ctx,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Compute token-level accuracy and loss on a fixed loader."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with ctx:
            logits, loss, _ = model(input_ids, labels=labels)

        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        total_correct += (predictions[mask] == labels[mask]).sum().item()

        n_tokens = mask.sum().item()
        total_tokens += n_tokens
        total_loss += loss.item() * n_tokens

    model.train()

    denom = max(total_tokens, 1)
    avg_loss = total_loss / denom
    accuracy = total_correct / denom
    return {"loss": avg_loss, "accuracy": accuracy}


NUMERIC_TASKS = {"mod_add", "addition", "multiplication", "chain", "successor", "parity"}
SEQUENCE_TASKS = {"copy", "reverse"}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _decode_example_text(example: Dict[str, torch.Tensor], tokenizer) -> Optional[str]:
    input_ids = example["input_ids"]
    labels = example["labels"]
    valid = (labels != -100).nonzero(as_tuple=False).view(-1)
    if valid.numel() == 0:
        return None
    last_pos = int(valid[-1].item())
    full_tokens = torch.cat([input_ids[: last_pos + 1], labels[last_pos : last_pos + 1]])
    return tokenizer.decode(full_tokens, skip_special_tokens=True)


def _extract_prompt_target(full_text: str, task: str) -> Optional[Dict[str, str]]:
    if task in SEQUENCE_TASKS:
        separators = ["->", "=>", "=", ":"]
        sep_index = -1
        chosen_sep = None
        for sep in separators:
            idx = full_text.rfind(sep)
            if idx > sep_index:
                sep_index = idx
                chosen_sep = sep
        if sep_index == -1 or chosen_sep is None:
            return None
        prompt = full_text[: sep_index + len(chosen_sep)].rstrip() + " "
        target = full_text[sep_index + len(chosen_sep) :].strip()
        return {"prompt": prompt, "target": target}

    if " " not in full_text:
        return None
    prompt, target = full_text.rsplit(" ", 1)
    return {"prompt": prompt.rstrip() + " ", "target": target.strip()}


def _iter_eval_examples(loader, max_samples: Optional[int]):
    count = 0
    for batch in loader:
        batch_size = batch["input_ids"].shape[0]
        for idx in range(batch_size):
            if max_samples is not None and count >= max_samples:
                return
            example = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    example[key] = value[idx]
                else:
                    example[key] = value[idx]
            count += 1
            yield example


def _evaluate_numeric_answer(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device,
    ctx,
) -> Optional[bool]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"].to(device)
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return None

    if len(target_ids) == 1:
        input_ids = prompt_ids
    else:
        target_prefix = torch.tensor(target_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        input_ids = torch.cat([prompt_ids, target_prefix], dim=1)

    with ctx:
        logits, _, _ = model(input_ids)

    prompt_len = prompt_ids.shape[1]
    for offset, token_id in enumerate(target_ids):
        pos = prompt_len + offset - 1
        if pos < 0 or pos >= logits.shape[1]:
            return None
        pred_token = logits[0, pos].argmax().item()
        if pred_token != token_id:
            return False

    return True


def _predict_generation_answer(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int,
) -> Optional[str]:
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"].to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    pred = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
    pred_norm = _normalize_text(pred)
    return pred_norm if pred_norm else None


def _predict_numeric_answer(
    model,
    tokenizer,
    prompt: str,
    task: str,
    device,
    max_new_tokens: int,
) -> Optional[str]:
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"].to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    pred = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
    pred_norm = normalize_prediction(task, pred)
    return pred_norm if pred_norm else None


@torch.no_grad()
def evaluate_task_accuracy(
    model,
    tokenizer,
    loader,
    device,
    ctx,
    max_samples: Optional[int] = 200,
    max_new_tokens: int = 32,
    numeric_tasks: Optional[set] = None,
    sample_count_per_task: int = 5,
    sample_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate task completion accuracy on a training split."""
    model.eval()
    numeric_tasks = numeric_tasks or NUMERIC_TASKS
    correct = defaultdict(int)
    total = defaultdict(int)
    metrics_by_task: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    total_abs_error = 0.0
    total_numeric = 0
    per_task_abs_error = defaultdict(float)
    per_task_numeric = defaultdict(int)
    sampled_examples_by_task: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_samples_by_task: Dict[str, int] = defaultdict(int)
    sample_seed = 0 if sample_seed is None else sample_seed
    sample_rngs: Dict[str, random.Random] = {}

    for example in _iter_eval_examples(loader, max_samples):
        task = example.get("task")
        if task is None:
            continue
        full_text = _decode_example_text(example, tokenizer)
        if not full_text:
            continue
        parsed = _extract_prompt_target(full_text, task)
        if not parsed:
            continue
        prompt = parsed["prompt"]
        target = parsed["target"]
        if not prompt or not target:
            continue

        if task in numeric_tasks:
            pred_norm = _predict_numeric_answer(
                model, tokenizer, prompt, task, device, max_new_tokens
            )
            if pred_norm is not None:
                target_norm = normalize_target(task, target)
                try:
                    pred_val = float(pred_norm)
                    target_val = float(target_norm)
                except ValueError:
                    pass
                else:
                    abs_error = abs(pred_val - target_val)
                    total_abs_error += abs_error
                    total_numeric += 1
                    per_task_abs_error[task] += abs_error
                    per_task_numeric[task] += 1
        else:
            pred_norm = _predict_generation_answer(
                model, tokenizer, prompt, device, max_new_tokens
            )

        if pred_norm is None:
            continue

        metrics = compute_metrics(task, pred_norm, target)
        metrics_by_task[task].append(metrics)
        total[task] += 1
        if metrics["exact_match"] == 1.0:
            correct[task] += 1
        if sample_count_per_task > 0:
            rng = sample_rngs.get(task)
            if rng is None:
                rng = random.Random(hash((sample_seed, task)))
                sample_rngs[task] = rng
            sample_item = {
                "prompt": prompt,
                "target": target,
                "prediction": pred_norm,
            }
            seen = seen_samples_by_task[task]
            if len(sampled_examples_by_task[task]) < sample_count_per_task:
                sampled_examples_by_task[task].append(sample_item)
            else:
                idx = rng.randint(0, seen)
                if idx < sample_count_per_task:
                    sampled_examples_by_task[task][idx] = sample_item
            seen_samples_by_task[task] = seen + 1

    model.train()

    per_task = {task: correct[task] / total[task] for task in total}
    per_task_token_accuracy = {}
    per_task_distance = {}
    per_task_prefix_accuracy = {}
    total_metrics: List[Dict[str, float]] = []
    for task, metrics in metrics_by_task.items():
        aggregated = aggregate(task, metrics)
        per_task_token_accuracy[task] = aggregated.mean_token_accuracy
        per_task_distance[task] = aggregated.mean_distance
        per_task_prefix_accuracy[task] = aggregated.mean_prefix_accuracy
        total_metrics.extend(metrics)
    per_task_mae = {
        task: (
            per_task_abs_error[task] / per_task_numeric[task]
            if per_task_numeric[task]
            else None
        )
        for task in per_task_abs_error
    }
    overall = sum(correct.values()) / max(sum(total.values()), 1)
    overall_aggregated = aggregate("overall", total_metrics)
    overall_mae = total_abs_error / total_numeric if total_numeric else None
    return {
        "overall_accuracy": overall,
        "per_task_accuracy": per_task,
        "overall_token_accuracy": overall_aggregated.mean_token_accuracy,
        "overall_distance": overall_aggregated.mean_distance,
        "overall_prefix_accuracy": overall_aggregated.mean_prefix_accuracy,
        "per_task_token_accuracy": per_task_token_accuracy,
        "per_task_distance": per_task_distance,
        "per_task_prefix_accuracy": per_task_prefix_accuracy,
        "overall_mae": overall_mae,
        "per_task_mae": per_task_mae,
        "sampled_examples_by_task": dict(sampled_examples_by_task),
    }


def _print_sampled_examples(label: str, samples_by_task: Dict[str, List[Dict[str, str]]]) -> None:
    if not samples_by_task:
        return
    print(f"  {label} samples:")
    for task in sorted(samples_by_task):
        samples = samples_by_task[task]
        if not samples:
            continue
        print(f"    {task}:")
        for idx, sample in enumerate(samples, start=1):
            print(
                f"      {idx}. prompt={sample['prompt']!r} "
                f"prediction={sample['prediction']!r} "
                f"target={sample['target']!r}"
            )


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create AdamW optimizer with decay and no-decay parameter groups."""

    decay_params = []
    no_decay_params = []

    ssm_terms = ["a_log", "dt_bias"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()
        if (
            name.endswith(".bias")
            or "norm" in name_lower
            or "tok_emb" in name_lower
            or "lm_head" in name_lower
            or name_lower.endswith(".d")
            or any(term in name_lower for term in ssm_terms)
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                # Exclude SSM internals and normalization scales from weight decay
                # to avoid distorting their specialized parameterization.
                "weight_decay": 0.0,
            }
        )

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"[optimizer] decay params: {n_decay:,}, no_decay params: {n_no_decay:,}")

    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


def difficulty_schedule(tokens_seen: int, alg_tokens: int, schedule: str = "linear") -> float:
    """Compute difficulty based on progress through algorithmic tokens."""

    progress = min(tokens_seen / max(alg_tokens, 1), 1.0)

    if schedule == "linear":
        return 0.5 + 0.5 * progress
    if schedule == "phased":
        return progress
    if schedule == "fixed":
        return 0.5

    raise ValueError(f"Unknown difficulty schedule: {schedule}")


def train(args):
    """Main training loop with split curriculum."""

    # =========================================================================
    # SETUP
    # =========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = MetricsLogger(output_dir)
    logger.set_hyperparameters(vars(args))
    
    # =========================================================================
    # TOKENIZER
    # =========================================================================
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # =========================================================================
    # MODEL
    # =========================================================================
    
    from model import create_model

    size = args.model_size
    extra_kwargs = {}
    if size == "nano":
        extra_kwargs["max_seq_len"] = args.lang_seq_len
    elif size == "1B-16k":
        extra_kwargs["max_seq_len"] = 16384
        extra_kwargs["use_gqa"] = True

    model = create_model(
        size=size,
        variant=args.variant,
        vocab_size=tokenizer.vocab_size,
        K=args.K,
        min_K=args.min_K,
        max_K=args.max_K,
        lambda_ponder=args.lambda_ponder,
        thought_tokens=args.thought_tokens,
        num_mem=args.num_mem,
        local_window=args.local_window,
        **extra_kwargs,
    )
    model = model.to(device)

    # Optimize model execution
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        # 'reduce-overhead' is excellent for smaller models or CPU training
        # If this errors on your setup, try mode="default"
        model = torch.compile(model, mode="reduce-overhead")

    config = model.config
    
    print(f"\nModel: {args.model_size} / {args.variant}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # =========================================================================
    # DATA
    # =========================================================================
    
    from data import (
        AlgorithmicDataset,
        AlgorithmicGenerator,
        CurriculumSampler,
        FixedAlgorithmicDataset,
        LanguageDataset,
        create_curriculum_loaders,
    )
    
    print("\nLoading datasets...")

    difficulty_value = Value("d", 0.0) if not args.fixed_data else None

    available_tasks = set(AlgorithmicGenerator._get_generators().keys())
    alg_tasks = list(args.alg_tasks) if args.alg_tasks is not None else None
    if alg_tasks:
        invalid = [task for task in alg_tasks if task not in available_tasks]
        if invalid:
            raise ValueError(
                f"Invalid alg_tasks {invalid}; choose from {sorted(available_tasks)}"
            )
        print(f"Restricting algorithmic tasks to: {', '.join(alg_tasks)}")

    # Phase 1: Algorithmic
    if args.fixed_data:
        alg_dataset = FixedAlgorithmicDataset(
            tokenizer=tokenizer,
            num_examples=args.fixed_data_size,
            max_seq_len=args.alg_seq_len,
            tasks=alg_tasks,
            seed=args.seed,
            difficulty=0.5,
            difficulty_schedule=args.difficulty_schedule,
            task_weighting=args.task_weighting,
        )
        alg_loader = DataLoader(
            alg_dataset,
            batch_size=args.alg_batch_size,
            shuffle=True,  # reshuffle fixed set each epoch to avoid ordering artifacts
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        alg_dataset = AlgorithmicDataset(
            tokenizer=tokenizer,
            num_examples=args.alg_examples,
            max_seq_len=args.alg_seq_len,
            tasks=alg_tasks,
            seed=args.seed,
            difficulty_value=difficulty_value,
            difficulty_schedule=args.difficulty_schedule,
            task_weighting=args.task_weighting,
            total_tokens=args.alg_tokens,
        )
        alg_loader = DataLoader(
            alg_dataset,
            batch_size=args.alg_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Phase 2: Language (only load if we'll use it)
    if args.total_tokens > args.alg_tokens:
        lang_dataset = LanguageDataset(
            tokenizer=tokenizer,
            max_seq_len=args.lang_seq_len,
        )
        lang_loader = DataLoader(
            lang_dataset,
            batch_size=args.lang_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        lang_loader = alg_loader  # Fallback

    # Training-eval loader (for grokking memorization tracking)
    if args.train_eval_mode == "auto":
        train_eval_mode = "fixed" if args.fixed_data else "none"
    else:
        train_eval_mode = args.train_eval_mode

    if train_eval_mode == "fixed" and not args.fixed_data:
        raise ValueError("train_eval_mode='fixed' requires --fixed_data")

    train_eval_loader = None
    train_eval_label = None
    if train_eval_mode != "none":
        train_eval_label = train_eval_mode
        if train_eval_mode == "fixed":
            base_dataset = alg_dataset if isinstance(alg_dataset, FixedAlgorithmicDataset) else FixedAlgorithmicDataset(
                tokenizer=tokenizer,
                num_examples=args.fixed_data_size,
                max_seq_len=args.alg_seq_len,
                tasks=alg_tasks,
                seed=args.seed,
                difficulty=0.5,
                difficulty_schedule=args.difficulty_schedule,
                task_weighting=args.task_weighting,
            )
            if args.train_eval_samples is not None:
                capped = min(args.train_eval_samples, len(base_dataset))
                base_dataset = Subset(base_dataset, range(capped))
        else:
            base_dataset = AlgorithmicDataset(
                tokenizer=tokenizer,
                num_examples=args.train_eval_samples or args.eval_samples,
                max_seq_len=args.alg_seq_len,
                tasks=alg_tasks,
                seed=args.seed + 123,
                difficulty_value=None,
                difficulty_schedule=args.difficulty_schedule,
                task_weighting=args.task_weighting,
                total_tokens=args.alg_tokens,
            )

        train_eval_loader = DataLoader(
            base_dataset,
            batch_size=args.train_eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Curriculum sampler
    curriculum = CurriculumSampler(
        alg_loader=alg_loader,
        lang_loader=lang_loader,
        total_tokens=args.total_tokens,
        alg_tokens=args.alg_tokens,
        mix_band_tokens=args.mix_band_tokens,
        persistent_alg_frac=args.persistent_alg_frac,
        lexical_frac_phase1=args.lexical_frac_phase1,
        seed=args.seed,
        lex_loader=None,
    )
    
    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = build_optimizer(model, lr=args.max_lr, weight_decay=args.weight_decay)

    # Mixed precision
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
    scaler = GradScaler(enabled=(dtype == torch.float16))

    grokfast_ema: Dict[str, torch.Tensor] = {}
    if args.grokfast:
        for name, param in model.named_parameters():
            if param.requires_grad:
                grokfast_ema[name] = torch.zeros_like(param.data)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print(f"\nTraining config:")
    print(f"  Phase 1 (algorithmic): {args.alg_tokens/1e6:.1f}M tokens")
    print(f"  Phase 2 (language): {(args.total_tokens - args.alg_tokens)/1e6:.1f}M tokens")
    print(f"  Total: {args.total_tokens/1e6:.1f}M tokens")
    print(f"  Mix band: {args.mix_band_tokens/1e6:.2f}M tokens")
    print(f"  Persistent alg frac: {args.persistent_alg_frac:.2f}")
    print(f"  Lexical frac (phase1): {args.lexical_frac_phase1:.2f}")
    print(f"  Variant: {args.variant}")
    print(
        f"  Fixed data: {args.fixed_data}" +
        (f" ({args.fixed_data_size} examples)" if args.fixed_data else "")
    )
    print(
        f"  Grokfast: {args.grokfast}" +
        (f" (α={args.grokfast_alpha}, λ={args.grokfast_lambda})" if args.grokfast else "")
    )
    print(f"  Weight decay: {args.weight_decay}")
    print(
        f"  Train eval: {train_eval_mode}"
        + (f" ({args.train_eval_samples} samples)" if train_eval_mode != "none" else "")
    )
    
    global_step = 0
    tokens_seen = 0
    start_time = time.time()
    best_alg_acc = 0.0
    last_grad_norm = 0.0
    checkpoint_paths: List[Path] = []

    def save_rotating_checkpoint(step: int, tokens_seen: int, tag: Optional[str] = None) -> Optional[Path]:
        if args.eval_checkpoints <= 0:
            return None
        suffix = f"_{tag}" if tag else ""
        checkpoint_path = output_dir / f"checkpoint_step_{step}{suffix}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": step,
            "tokens_seen": tokens_seen,
        }, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)
        while len(checkpoint_paths) > args.eval_checkpoints:
            old_checkpoint = checkpoint_paths.pop(0)
            try:
                old_checkpoint.unlink()
            except FileNotFoundError:
                pass
        return checkpoint_path
    
    # Estimate steps with hybrid calculation for sparse algorithmic and dense language data
    algo_steps = args.alg_tokens // (args.alg_batch_size * 40)
    lang_tokens = max(0, args.total_tokens - args.alg_tokens)
    lang_steps = lang_tokens // (args.alg_batch_size * args.lang_seq_len)
    estimated_steps = algo_steps + lang_steps
    warmup_steps = args.warmup_steps
    warmup_tokens_arg = args.warmup_tokens
    estimated_tokens_per_step = args.total_tokens / max(estimated_steps, 1)

    if warmup_tokens_arg is not None:
        warmup_tokens = warmup_tokens_arg
        warmup_source = "manual warmup_tokens override"
    elif warmup_steps is not None:
        # Preserve the CLI contract for --warmup_steps by translating the requested
        # step count into the equivalent token budget using the hybrid step
        # estimate above.
        warmup_tokens = int(warmup_steps * estimated_tokens_per_step)
        if warmup_tokens > 0.05 * args.total_tokens:
            print(
                f"WARNING: warmup_steps={warmup_steps} is "
                f"{warmup_tokens/args.total_tokens:.1%} of training. Consider reducing."
            )
        warmup_source = f"derived from {warmup_steps} manual warmup steps"
    else:
        warmup_tokens = int(args.warmup_frac * args.total_tokens)
        warmup_source = f"{args.warmup_frac:.2f} of total tokens"

    warmup_tokens = min(max(warmup_tokens, 1), args.total_tokens)
    warmup_display = f"{warmup_tokens} tokens ({warmup_source})"

    print(
        f"  Estimated steps (Hybrid calc: Algo ~{algo_steps} + Lang ~{lang_steps}): {estimated_steps}"
    )
    print(f"  Warmup: {warmup_display}")
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    running_inner_steps = 0.0
    running_ponder = 0.0
    
    try:
        while curriculum.tokens_seen < args.total_tokens:
            if difficulty_value is not None:
                if curriculum.tokens_seen < args.alg_tokens:
                    difficulty_value.value = difficulty_schedule(
                        curriculum.tokens_seen, args.alg_tokens, args.difficulty_schedule
                    )
                else:
                    difficulty_value.value = 1.0
            batch = curriculum.next_batch()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tokens_seen = curriculum.tokens_seen
            
            # Learning rate (token-based decay to align with actual budget consumption)
            lr = get_lr(tokens_seen, warmup_tokens, args.total_tokens, args.max_lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            
            # Forward
            with ctx:
                logits, loss, _ = model(input_ids, labels=labels)
                inner_steps = model.aux.get("inner_steps", 0)
                ponder = model.aux.get("ponder", 0.0)

                if isinstance(inner_steps, torch.Tensor):
                    avg_inner = inner_steps.float().mean().item()
                else:
                    avg_inner = float(inner_steps) if inner_steps else 0.0

                running_inner_steps += avg_inner
                running_ponder += float(ponder)

                loss = loss / args.grad_accum_steps
            
            # Backward
            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum_steps

            if args.grokfast:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            ema = grokfast_ema[name]
                            ema.mul_(args.grokfast_alpha).add_(
                                param.grad, alpha=1 - args.grokfast_alpha
                            )
                            param.grad.add_(ema, alpha=args.grokfast_lambda)

            # Step
            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                last_grad_norm = float(grad_norm)
            
            global_step += 1
            
            # Logging
            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                toks_per_sec = tokens_seen / max(elapsed, 1e-6)
                peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                avg_loss = running_loss / args.log_interval
                avg_K = running_inner_steps / args.log_interval if running_inner_steps > 0 else 0.0
                avg_ponder = running_ponder / args.log_interval if running_ponder > 0 else 0.0

                phase = curriculum.phase
                progress = curriculum.progress * 100

                with torch.no_grad():
                    weight_norm = math.sqrt(
                        sum((param.detach().float() ** 2).sum().item() for param in model.parameters())
                    )

                print(f"Step {global_step:>6} | "
                      f"Phase: {phase[:4]:>4} | "
                      f"Progress: {progress:>5.1f}% | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Tok/s: {toks_per_sec:.0f} | "
                      f"VRAM: {peak_vram:.1f}GB" +
                      (f" | K: {avg_K:.1f}" if avg_K > 0 else "") +
                      (f" | ponder: {avg_ponder:.3f}" if avg_ponder > 0 else ""))

                difficulty_logged = (
                    float(difficulty_value.value) if difficulty_value is not None else 0.5
                )

                logger.log(
                    step=global_step,
                    phase=phase,
                    train_loss=avg_loss,
                    tokens_seen=tokens_seen,
                    tokens_per_sec=toks_per_sec,
                    peak_vram_gb=peak_vram,
                    avg_inner_steps=avg_K,
                    ponder=avg_ponder,
                    difficulty=difficulty_logged,
                    grad_norm=last_grad_norm,
                    weight_norm=weight_norm,
                )

                running_loss = 0.0
                running_inner_steps = 0.0
                running_ponder = 0.0
            
            # Evaluation
            if global_step % args.eval_interval == 0:
                print("\nRunning evaluation...")
                difficulty_logged = (
                    float(difficulty_value.value) if difficulty_value is not None else 0.5
                )
                print(f"  Eval difficulty: {difficulty_logged:.3f}")
                with torch.no_grad():
                    weight_norm = math.sqrt(
                        sum((param.detach().float() ** 2).sum().item() for param in model.parameters())
                    )
                effective_wd_pressure = args.weight_decay * weight_norm
                print(f"  Grad norm: {last_grad_norm:.1f}")
                print(f"  Weight norm: {weight_norm:.1f}")
                print(f"  WD pressure: {effective_wd_pressure:.1f}")

                # Algorithmic accuracy (OOD grid)
                if args.eval_algorithmic:
                    alg_results = evaluate_algorithmic(
                        model,
                        tokenizer,
                        device,
                        n_examples=args.eval_samples,
                        max_new_tokens=args.eval_max_new_tokens,
                        tasks=alg_tasks,
                        seed=args.seed,
                        sample_count_per_task=args.eval_sample_count_per_task,
                    )
                    overall_acc = alg_results.get("overall_accuracy", 0.0)
                    overall_mae = alg_results.get("overall_mae")
                    overall_token_accuracy = alg_results.get("overall_token_accuracy", 0.0)
                    overall_distance = alg_results.get("overall_distance", 1.0)
                    overall_prefix_accuracy = alg_results.get("overall_prefix_accuracy", 0.0)
                    print(
                        "  Algorithmic metrics: "
                        f"acc={overall_acc*100:.1f}% | "
                        f"token={overall_token_accuracy:.3f} | "
                        f"dist={overall_distance:.3f} | "
                        f"prefix={overall_prefix_accuracy:.3f}"
                    )
                    if overall_mae is not None:
                        print(f"  Algorithmic MAE (numeric tasks): {overall_mae:.3f}")

                    _print_sampled_examples(
                        "Algorithmic eval",
                        alg_results.get("sampled_examples_by_task", {}),
                    )

                    logger.log(
                        step=global_step,
                        phase="eval",
                        algorithmic_accuracy=overall_acc,
                        algorithmic_token_accuracy=overall_token_accuracy,
                        algorithmic_distance=overall_distance,
                        algorithmic_prefix_accuracy=overall_prefix_accuracy,
                        algorithmic_mae=overall_mae,
                        eval_difficulty=difficulty_logged,
                        algorithmic_samples=alg_results.get("sampled_examples_by_task", {}),
                    )

                    per_task_mae = alg_results.get("per_task_mae", {}) or {}
                    per_task_token_accuracy = alg_results.get("per_task_token_accuracy", {}) or {}
                    per_task_distance = alg_results.get("per_task_distance", {}) or {}
                    per_task_prefix_accuracy = (
                        alg_results.get("per_task_prefix_accuracy", {}) or {}
                    )
                    for task, acc in alg_results.get("per_task_accuracy", {}).items():
                        task_mae = per_task_mae.get(task)
                        logger.log_task_accuracy(
                            task,
                            acc,
                            global_step,
                            target="eval",
                            mae=task_mae,
                            mean_token_accuracy=per_task_token_accuracy.get(task),
                            mean_distance=per_task_distance.get(task),
                            mean_prefix_accuracy=per_task_prefix_accuracy.get(task),
                        )
                        token_acc = per_task_token_accuracy.get(task, 0.0)
                        distance = per_task_distance.get(task, 1.0)
                        prefix_acc = per_task_prefix_accuracy.get(task, 0.0)
                        line = (
                            f"    {task}: {acc*100:.1f}%"
                            f" | token={token_acc:.3f}"
                            f" | dist={distance:.3f}"
                            f" | prefix={prefix_acc:.3f}"
                        )
                        if task_mae is not None:
                            line += f" | MAE: {task_mae:.3f}"
                        print(line)

                    # Track best
                    if overall_acc > best_alg_acc:
                        best_alg_acc = overall_acc
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "step": global_step,
                            "alg_accuracy": best_alg_acc,
                        }, output_dir / "best_model.pt")
                        print(f"  New best! Saved checkpoint.")

                # Training data accuracy (memorization tracking)
                if train_eval_loader is not None:
                    train_eval_metrics = evaluate_training_split(
                        model,
                        train_eval_loader,
                        device,
                        ctx,
                        max_batches=args.train_eval_max_batches,
                    )
                    train_acc = train_eval_metrics["accuracy"]
                    train_loss_eval = train_eval_metrics["loss"]

                    task_eval = evaluate_task_accuracy(
                        model,
                        tokenizer,
                        train_eval_loader,
                        device,
                        ctx,
                        max_samples=args.train_eval_samples,
                        max_new_tokens=args.eval_max_new_tokens,
                        sample_count_per_task=args.eval_sample_count_per_task,
                        sample_seed=args.seed,
                    )
                    overall_task_acc = task_eval.get("overall_accuracy", 0.0)
                    overall_task_mae = task_eval.get("overall_mae")
                    print(
                        f"  Training accuracy ({train_eval_label}): {train_acc*100:.2f}% | "
                        f"Loss: {train_loss_eval:.4f}"
                    )
                    print(f"  Training task accuracy: {overall_task_acc*100:.1f}%")
                    if overall_task_mae is not None:
                        print(f"  Training task MAE (numeric tasks): {overall_task_mae:.3f}")

                    _print_sampled_examples(
                        "Training eval",
                        task_eval.get("sampled_examples_by_task", {}),
                    )

                    logger.log(
                        step=global_step,
                        phase="eval",
                        train_accuracy=train_acc,
                        train_eval_loss=train_loss_eval,
                        train_eval_mae=overall_task_mae,
                        train_eval_token_accuracy=task_eval.get("overall_token_accuracy"),
                        train_eval_distance=task_eval.get("overall_distance"),
                        train_eval_prefix_accuracy=task_eval.get("overall_prefix_accuracy"),
                        eval_difficulty=difficulty_logged,
                        train_eval_samples=task_eval.get("sampled_examples_by_task", {}),
                    )

                    per_task_mae = task_eval.get("per_task_mae", {}) or {}
                    per_task_token_accuracy = task_eval.get("per_task_token_accuracy", {}) or {}
                    per_task_distance = task_eval.get("per_task_distance", {}) or {}
                    per_task_prefix_accuracy = (
                        task_eval.get("per_task_prefix_accuracy", {}) or {}
                    )
                    for task, acc in task_eval.get("per_task_accuracy", {}).items():
                        task_mae = per_task_mae.get(task)
                        logger.log_task_accuracy(
                            task,
                            acc,
                            global_step,
                            target="train",
                            mae=task_mae,
                            mean_token_accuracy=per_task_token_accuracy.get(task),
                            mean_distance=per_task_distance.get(task),
                            mean_prefix_accuracy=per_task_prefix_accuracy.get(task),
                        )
                        line = f"    {task}: {acc*100:.1f}%"
                        if task_mae is not None:
                            line += f" | MAE: {task_mae:.3f}"
                        print(line)

                # Perplexity (only in Phase 2)
                if curriculum.phase == "language":
                    ppl = evaluate_perplexity(model, lang_loader, device, ctx)
                    print(f"  Language PPL: {ppl:.2f}")
                    logger.log(step=global_step, phase="eval", val_loss=math.log(ppl))

                logger.plot()
                logger.save()

                checkpoint_path = save_rotating_checkpoint(global_step, tokens_seen)
                if checkpoint_path is not None:
                    print(
                        f"  Saved checkpoint: {checkpoint_path.name} "
                        f"(keeping last {args.eval_checkpoints})"
                    )

                print()
                model.train()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving checkpoint before exit.")
        interrupt_checkpoint = save_rotating_checkpoint(global_step, tokens_seen, tag="interrupt")
        if interrupt_checkpoint is not None:
            print(
                f"  Saved checkpoint: {interrupt_checkpoint.name} "
                f"(keeping last {args.eval_checkpoints})"
            )
    
    # =========================================================================
    # FINISH
    # =========================================================================
    
    total_time = time.time() - start_time
    
    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": global_step,
        "tokens_seen": tokens_seen,
    }, output_dir / "final_model.pt")
    
    # Save metrics
    logger.save()
    
    summary = logger.summary()
    summary["total_time_hours"] = total_time / 3600
    summary["best_alg_accuracy"] = best_alg_acc
    summary["variant"] = args.variant
    summary["model_size"] = args.model_size
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Tokens seen: {tokens_seen/1e9:.2f}B")
    print(f"  Best algorithmic accuracy: {best_alg_acc*100:.1f}%")
    print(f"  Output: {output_dir}")
    print("="*60)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Split curriculum training")
    
    # Model
    parser.add_argument("--model_size", type=str, default="nano",
                       choices=["nano", "50M", "125M", "300M", "350M", "760M", "1B", "1B-16k"])
    parser.add_argument("--variant", type=str, default="baseline",
                       choices=["baseline", "shared_loop", "latent", "act", "ssm", "ssm_mem"])
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--K", type=int, default=4, help="Fixed inner steps for latent/ssm")
    parser.add_argument("--min_K", type=int, default=2, help="Minimum ACT steps")
    parser.add_argument("--max_K", type=int, default=8, help="Maximum ACT steps")
    parser.add_argument("--lambda_ponder", type=float, default=0.01, help="Ponder cost weight for ACT")
    parser.add_argument("--thought_tokens", type=int, default=16, help="Number of latent thought tokens")
    parser.add_argument("--num_mem", type=int, default=64, help="Memory tokens for ssm_mem")
    parser.add_argument("--local_window", type=int, default=0, help="Local window for thought attention")
    
    # Curriculum
    parser.add_argument("--alg_tokens", type=int, default=100_000_000,
                       help="Tokens for Phase 1 (algorithmic)")
    parser.add_argument("--total_tokens", type=int, default=500_000_000,
                       help="Total token budget")
    parser.add_argument(
        "--difficulty_schedule",
        type=str,
        default="phased",
        choices=["linear", "phased", "fixed"],
        help="Difficulty curriculum type",
    )
    parser.add_argument(
        "--task_weighting",
        type=str,
        default="adaptive",
        choices=["uniform", "adaptive"],
        help="Task sampling weights",
    )
    parser.add_argument("--mix_band_tokens", type=int, default=None,
                       help="Transition band (tokens) between algorithmic and language phases")
    parser.add_argument("--persistent_alg_frac", type=float, default=0.15,
                       help="Long-run fraction of algorithmic data after transition (recommended 0.05–0.3)")
    parser.add_argument("--lexical_frac_phase1", type=float, default=0.05,
                       help="Lexical noise fraction during initial algorithmic phase")
    
    # Data
    parser.add_argument("--alg_examples", type=int, default=100_000)
    parser.add_argument("--alg_seq_len", type=int, default=128)
    parser.add_argument(
        "--alg_tasks",
        nargs="+",
        default=None,
        help=(
            "Optional subset of algorithmic tasks for training (e.g., parity addition). "
            "If provided, examples are drawn only from this set."
        ),
    )
    parser.add_argument("--alg_batch_size", type=int, default=64)
    parser.add_argument("--lang_seq_len", type=int, default=1024)
    parser.add_argument("--lang_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Grokking
    parser.add_argument(
        "--fixed_data",
        action="store_true",
        help="Use fixed dataset for grokking (vs infinite procedural)",
    )
    parser.add_argument(
        "--fixed_data_size",
        type=int,
        default=5000,
        help="Number of examples in fixed dataset",
    )
    parser.add_argument(
        "--grokfast",
        action="store_true",
        help="Enable Grokfast gradient filtering for faster grokking",
    )
    parser.add_argument(
        "--grokfast_alpha",
        type=float,
        default=0.98,
        help="EMA decay for Grokfast",
    )
    parser.add_argument(
        "--grokfast_lambda",
        type=float,
        default=5.0,
        help="Amplification factor for Grokfast",
    )
    parser.add_argument(
        "--train_eval_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "fixed", "procedural"],
        help=(
            "What to evaluate for training accuracy: auto=use fixed when available, "
            "none=skip, fixed=only the memorized set, procedural=fresh samples"
        ),
    )
    parser.add_argument(
        "--train_eval_samples",
        type=int,
        default=2000,
        help="Number of samples to use for training-set accuracy checks",
    )
    parser.add_argument(
        "--train_eval_batch_size",
        type=int,
        default=128,
        help="Batch size for training accuracy evaluation",
    )
    parser.add_argument(
        "--train_eval_max_batches",
        type=int,
        default=None,
        help="Optional cap on batches when evaluating training accuracy",
    )

    # Optimizer
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help=(
            "Manual override for warmup expressed in steps (converted to tokens); "
            "ignored when warmup_tokens is set"
        ),
    )
    parser.add_argument(
        "--warmup_tokens",
        type=int,
        default=None,
        help=(
            "Manual override for warmup expressed directly in tokens. Overrides "
            "warmup_steps and warmup_frac when set."
        ),
    )
    parser.add_argument(
        "--warmup_frac",
        type=float,
        default=0.02,
        help=(
            "Warmup fraction (default 2%) when warmup_tokens and warmup_steps "
            "are not set"
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=4)

    # Logging
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument(
        "--eval_checkpoints",
        type=int,
        default=2,
        help="Number of most recent evaluation checkpoints to keep",
    )
    parser.add_argument(
        "--eval_algorithmic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the algorithmic OOD eval suite (disable with --no-eval_algorithmic)",
    )
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=32,
        help="Limit generated tokens per example to prevent infinite loops",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Number of examples to evaluate per interval",
    )
    parser.add_argument(
        "--eval_sample_count_per_task",
        type=int,
        default=5,
        help="Number of sampled eval prompts to log per task",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (recommended for Linux only)")

    args = parser.parse_args()

    if args.mix_band_tokens is None:
        args.mix_band_tokens = int(0.5 * args.alg_tokens)
    train(args)


if __name__ == "__main__":
    main()
