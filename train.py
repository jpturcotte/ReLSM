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
    compute_attention_probs,
    compute_weight_entropy,
    compute_metrics,
    get_transformer_blocks,
    module_grad_weight_norm,
    normalize_prediction,
    normalize_target,
    resolve_diagnostic_modules,
    safe_parse_number,
)

plt.switch_backend("Agg")

def get_lr(
    tokens_seen: int,
    warmup_tokens: int,
    total_tokens: int,
    max_lr: float,
    min_lr: float,
    schedule: str = "cosine",
) -> float:
    """
    Learning rate schedule with multiple options.

    Schedules:
      - cosine: Original cosine decay (aggressive, loses capacity fast)
      - wsd: Warmup-Stable-Decay (hold at max 40%, then linear decay)
      - plateau: Hold at max 40%, then cosine decay
      - constant: No decay after warmup (for debugging)
    """
    progress = min(tokens_seen / max(total_tokens, 1), 1.0)
    warmup_frac = warmup_tokens / max(total_tokens, 1)

    if progress < warmup_frac:
        return max_lr * (progress / warmup_frac)

    if schedule == "cosine":
        decay_progress = (progress - warmup_frac) / (1 - warmup_frac)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_progress))

    if schedule == "wsd":
        stable_end = 0.4
        if progress < stable_end:
            return max_lr
        decay_progress = (progress - stable_end) / (1 - stable_end)
        return max_lr - (max_lr - min_lr) * min(decay_progress, 1.0)

    if schedule == "plateau":
        plateau_end = 0.4
        if progress < plateau_end:
            return max_lr
        decay_progress = (progress - plateau_end) / (1 - plateau_end)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_progress))

    if schedule == "constant":
        return max_lr

    raise ValueError(f"Unknown LR schedule: {schedule}")


class MetricsLogger:
    """Tracks metrics for the ablation ladder."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.hyperparameters = {}
        self.records = []
        self.evaluations = []
        self.task_accuracies = {}
        self.train_task_accuracies = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.output_dir / "metrics.json"
        if not self._metrics_path.exists():
            self._append_snapshot()
    
    def log(self, step: int, phase: str, **kwargs):
        record = {"step": step, "phase": phase}
        record.update(kwargs)
        self.records.append(record)
        self.evaluations.append(record)
    
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

    def series(self, key: str, phase: Optional[str] = None):
        steps = [
            record["step"]
            for record in self.records
            if key in record and (phase is None or record.get("phase") == phase)
        ]
        vals = [
            record[key]
            for record in self.records
            if key in record and (phase is None or record.get("phase") == phase)
        ]
        return steps, vals

    def _append_snapshot(self):
        snapshot = {
            "hyperparameters": self.hyperparameters,
            "records": self.records,
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
        train_losses = self.series("train_loss")[1]
        val_losses = self.series("val_loss")[1]
        peak_vram = self.series("peak_vram_gb")[1]
        tokens_per_sec = self.series("tokens_per_sec")[1]
        tokens_seen = self.series("tokens_seen")[1]
        return {
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "peak_vram_gb": max(peak_vram) if peak_vram else None,
            "avg_tokens_per_sec": (
                sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else None
            ),
            "total_tokens": tokens_seen[-1] if tokens_seen else 0,
        }

    def plot(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss plot
        plt.figure()
        train_steps, train_losses = self.series("train_loss", phase="train")
        if train_steps and train_losses:
            plt.plot(train_steps, train_losses, label="train")

        val_steps, val_losses = self.series("val_loss", phase="eval")
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


@torch.no_grad()
def run_eval_diagnostics(
    model: nn.Module,
    probe_batch: Dict[str, torch.Tensor],
    device: torch.device,
    ctx,
    diagnostic_blocks: Sequence[nn.Module],
    diagnostic_modules: Dict[str, Optional[nn.Module]],
    probe_seed: int,
) -> Dict[str, float]:
    """DIAGNOSTIC METRICS (eval-only): inexpensive probe batch diagnostics."""

    metrics: Dict[str, float] = {}
    if not probe_batch:
        return metrics

    was_training = model.training
    model.eval()

    input_ids = probe_batch["input_ids"].to(device)
    labels = probe_batch.get("labels")
    if labels is not None:
        labels = labels.to(device)

    ff_outputs: Dict[str, torch.Tensor] = {}
    last_block_input: Dict[str, torch.Tensor] = {}
    hooks = []

    def _make_ff_hook(label: str):
        def _hook(_module, _inputs, output):
            ff_outputs[label] = output.detach()
        return _hook

    def _capture_block_input(_module, inputs, _output):
        if inputs:
            last_block_input["x"] = inputs[0].detach()
        if len(inputs) > 4 and inputs[4] is not None:
            last_block_input["position_ids"] = inputs[4].detach()

    if diagnostic_blocks:
        indices = {
            "early": 0,
            "mid": len(diagnostic_blocks) // 2,
            "late": len(diagnostic_blocks) - 1,
        }
        for label, idx in indices.items():
            block = diagnostic_blocks[idx]
            ff_module = getattr(block, "ff", None)
            if isinstance(ff_module, nn.Module):
                hooks.append(ff_module.register_forward_hook(_make_ff_hook(label)))

        last_block = diagnostic_blocks[-1]
        hooks.append(last_block.register_forward_hook(_capture_block_input))

    with ctx:
        model(input_ids, labels=labels)

    for hook in hooks:
        hook.remove()

    for label in ("early", "mid", "late"):
        ff_out = ff_outputs.get(label)
        if ff_out is not None:
            metrics[f"act_sparsity.ff.{label}"] = (
                (ff_out.abs() < 1e-4).float().mean().item()
            )

    last_block = diagnostic_blocks[-1] if diagnostic_blocks else None
    attn_module = getattr(last_block, "attn", None) if last_block is not None else None
    if attn_module is not None and "x" in last_block_input:
        attn_input = last_block.attn_norm(last_block_input["x"])
        probs = compute_attention_probs(
            attn_module,
            attn_input,
            position_ids=last_block_input.get("position_ids"),
        )
        B, H, T, _ = probs.shape
        total_positions = B * T
        if total_positions > 0:
            gen = torch.Generator(device=probs.device)
            gen.manual_seed(probe_seed)
            num_samples = min(32, total_positions)
            indices = torch.randint(
                0,
                total_positions,
                (H, num_samples),
                generator=gen,
                device=probs.device,
            )
            batch_ids = indices // T
            pos_ids = indices % T
            head_ids = torch.arange(H, device=probs.device).unsqueeze(1).expand(H, num_samples)
            selected = probs[batch_ids, head_ids, pos_ids, :]
            entropy = -(selected * (selected + 1e-12).log()).sum(dim=-1)
            mean_by_head = entropy.mean(dim=1)
            metrics["attn_entropy_last_mean"] = mean_by_head.mean().item()
            metrics["attn_entropy_last_std"] = mean_by_head.std(unbiased=False).item()

    if last_block is not None:
        metrics["weight_entropy.last_block"] = compute_weight_entropy(last_block)
    head_module = diagnostic_modules.get("head")
    if head_module is not None:
        metrics["weight_entropy.head"] = compute_weight_entropy(head_module)

    if was_training:
        model.train()

    return metrics


NUMERIC_TASKS = {"mod_add", "addition", "multiplication", "chain", "successor", "parity"}
MAE_TASKS = {"mod_add", "addition", "multiplication", "chain", "successor"}
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
    target_val = safe_parse_number(target)
    if math.isnan(target_val):
        return None
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
            if pred_norm is not None and task in MAE_TASKS:
                target_norm = normalize_target(task, target)
                pred_val = safe_parse_number(pred_norm)
                target_val = safe_parse_number(target_norm)
                if not (math.isnan(pred_val) or math.isnan(target_val)):
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
                "expected_output": target,
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
            expected_output = sample.get("expected_output", sample.get("target"))
            print(
                f"      {idx}. prompt={sample['prompt']!r} "
                f"prediction={sample['prediction']!r} "
                f"expected_output={expected_output!r}"
            )


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    *,
    extended_logging: bool = False,
) -> torch.optim.Optimizer:
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
    if extended_logging:
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
    if schedule == "smooth":
        boundaries = [0.15, 0.35, 0.60]
        target_difficulties = [0.25, 0.45, 0.70, 1.0]
        result = target_difficulties[0]
        for i, boundary in enumerate(boundaries):
            transition_width = 0.05
            blend = 1.0 / (1.0 + math.exp(-(progress - boundary) / (transition_width / 4)))
            result = result * (1 - blend) + target_difficulties[i + 1] * blend
        return result
    if schedule == "warmup_ramp":
        warmup_frac = 0.1
        hold_frac = 0.2
        if progress < warmup_frac:
            return 0.175
        if progress < warmup_frac + hold_frac:
            return 0.3
        ramp_progress = (progress - warmup_frac - hold_frac) / (1 - warmup_frac - hold_frac)
        return 0.3 + 0.7 * ramp_progress

    raise ValueError(f"Unknown difficulty schedule: {schedule}")


def train(args):
    """Main training loop with split curriculum."""

    # =========================================================================
    # SETUP
    # =========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print = print if args.extended_logging else (lambda *args, **kwargs: None)

    def format_metric_value(value: Any) -> str:
        if isinstance(value, (float, int)):
            if math.isnan(value):
                return "nan"
            if value == 0:
                return "0"
            if abs(value) >= 1000 or abs(value) < 0.01:
                return f"{value:.2e}"
            return f"{value:.3f}"
        return str(value)

    def format_metrics_line(metrics: Dict[str, float]) -> str:
        return " | ".join(
            f"{key}={format_metric_value(metrics[key])}" for key in sorted(metrics)
        )
    log_print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device.type == "cuda":
        log_print(f"GPU: {torch.cuda.get_device_name()}")
        log_print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
        log_print("Compiling model with torch.compile...")
        # 'reduce-overhead' is excellent for smaller models or CPU training
        # If this errors on your setup, try mode="default"
        model = torch.compile(model, mode="reduce-overhead")

    config = model.config

    diagnostic_modules = resolve_diagnostic_modules(model)
    diagnostic_blocks = get_transformer_blocks(model)
    if diagnostic_modules.get("embedding") and diagnostic_modules.get("head"):
        emb_params = {id(p) for p in diagnostic_modules["embedding"].parameters(recurse=True)}
        head_params = {id(p) for p in diagnostic_modules["head"].parameters(recurse=True)}
        if emb_params & head_params:
            diagnostic_modules["head"] = None
    
    log_print(f"\nModel: {args.model_size} / {args.variant}")
    log_print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
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
    
    log_print("\nLoading datasets...")

    if not args.fixed_data and args.difficulty_schedule in {"linear", "phased", "fixed"}:
        difficulty_value = Value("d", 0.0)
    else:
        difficulty_value = None

    available_tasks = set(AlgorithmicGenerator._get_generators().keys())
    alg_tasks = list(args.alg_tasks) if args.alg_tasks is not None else None
    if alg_tasks:
        invalid = [task for task in alg_tasks if task not in available_tasks]
        if invalid:
            raise ValueError(
                f"Invalid alg_tasks {invalid}; choose from {sorted(available_tasks)}"
            )
        log_print(f"Restricting algorithmic tasks to: {', '.join(alg_tasks)}")

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
            easy_mix_frac=args.easy_mix_frac,
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
            easy_mix_frac=args.easy_mix_frac,
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
                easy_mix_frac=args.easy_mix_frac,
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
                easy_mix_frac=args.easy_mix_frac,
            )

        train_eval_loader = DataLoader(
            base_dataset,
            batch_size=args.train_eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Fixed probe batch for eval-only diagnostics
    probe_seed = args.seed + 4242
    probe_batch = None
    probe_dataset = FixedAlgorithmicDataset(
        tokenizer=tokenizer,
        num_examples=args.alg_batch_size,
        max_seq_len=args.alg_seq_len,
        tasks=alg_tasks,
        seed=probe_seed,
        difficulty=0.5,
        difficulty_schedule=args.difficulty_schedule,
        task_weighting=args.task_weighting,
        easy_mix_frac=args.easy_mix_frac,
    )
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.alg_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    probe_batch = next(iter(probe_loader))
    
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

    optimizer = build_optimizer(
        model,
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        extended_logging=args.extended_logging,
    )

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
    
    log_print(f"\nTraining config:")
    log_print(f"  Phase 1 (algorithmic): {args.alg_tokens/1e6:.1f}M tokens")
    log_print(f"  Phase 2 (language): {(args.total_tokens - args.alg_tokens)/1e6:.1f}M tokens")
    log_print(f"  Total: {args.total_tokens/1e6:.1f}M tokens")
    log_print(f"  Mix band: {args.mix_band_tokens/1e6:.2f}M tokens")
    log_print(f"  Persistent alg frac: {args.persistent_alg_frac:.2f}")
    log_print(f"  Lexical frac (phase1): {args.lexical_frac_phase1:.2f}")
    log_print(f"  Variant: {args.variant}")
    log_print(
        f"  Fixed data: {args.fixed_data}" +
        (f" ({args.fixed_data_size} examples)" if args.fixed_data else "")
    )
    log_print(
        f"  Grokfast: {args.grokfast}" +
        (f" (α={args.grokfast_alpha}, λ={args.grokfast_lambda})" if args.grokfast else "")
    )
    log_print(f"  Weight decay: {args.weight_decay}")
    log_print(
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
            log_print(
                f"WARNING: warmup_steps={warmup_steps} is "
                f"{warmup_tokens/args.total_tokens:.1%} of training. Consider reducing."
            )
        warmup_source = f"derived from {warmup_steps} manual warmup steps"
    else:
        warmup_tokens = int(args.warmup_frac * args.total_tokens)
        warmup_source = f"{args.warmup_frac:.2f} of total tokens"

    warmup_tokens = min(max(warmup_tokens, 1), args.total_tokens)
    warmup_display = f"{warmup_tokens} tokens ({warmup_source})"

    log_print(
        f"  Estimated steps (Hybrid calc: Algo ~{algo_steps} + Lang ~{lang_steps}): {estimated_steps}"
    )
    log_print(f"  Warmup: {warmup_display}")
    log_print("\n" + "="*60)
    log_print("Starting training...")
    log_print("="*60 + "\n")
    
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    running_inner_steps = 0.0
    running_ponder = 0.0
    last_grad_norm = float("nan")
    flush_diagnostic_metrics: Optional[Dict[str, float]] = None
    
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
            lr = get_lr(
                tokens_seen,
                warmup_tokens,
                args.total_tokens,
                args.max_lr,
                args.min_lr,
                schedule=args.lr_schedule,
            )
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
                if (global_step + 1) % args.log_interval == 0:
                    flush_diagnostic_metrics = {}
                    for key, module in diagnostic_modules.items():
                        if module is None:
                            continue
                        mod_grad_norm, mod_weight_norm = module_grad_weight_norm(module)
                        flush_diagnostic_metrics[f"grad_norm.{key}"] = mod_grad_norm
                        if not math.isnan(mod_grad_norm) and not math.isnan(mod_weight_norm):
                            flush_diagnostic_metrics[f"grad_to_weight.{key}"] = (
                                mod_grad_norm / (mod_weight_norm + 1e-12)
                            )
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
                effective_wd_pressure = args.weight_decay * weight_norm

                print(f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                difficulty_logged = (
                    float(difficulty_value.value)
                    if difficulty_value is not None
                    else difficulty_schedule(
                        curriculum.tokens_seen, args.alg_tokens, args.difficulty_schedule
                    )
                )

                if args.extended_logging:
                    log_print(f"Step {global_step:>6} | "
                              f"Phase: {phase[:4]:>4} | "
                              f"Progress: {progress:>5.1f}% | "
                              f"Tok/s: {toks_per_sec:.0f} | "
                              f"VRAM: {peak_vram:.1f}GB" +
                              (f" | K: {avg_K:.1f}" if avg_K > 0 else "") +
                              (f" | ponder: {avg_ponder:.3f}" if avg_ponder > 0 else ""))
                    log_print(f"  Difficulty: {difficulty_logged:.3f} (easy_mix: {args.easy_mix_frac})")
                    expected_lr = get_lr(
                        tokens_seen,
                        warmup_tokens,
                        args.total_tokens,
                        args.max_lr,
                        args.min_lr,
                        schedule=args.lr_schedule,
                    )
                    log_print(f"  LR: {expected_lr:.2e} (schedule: {args.lr_schedule})")
                    log_print(f"  Grad norm: {last_grad_norm:.1f}")
                    log_print(f"  Weight norm: {weight_norm:.1f}")
                    if args.weight_decay > 0.0:
                        log_print(f"  WD pressure: {effective_wd_pressure:.1f}")

                # DIAGNOSTIC METRICS (cheap)
                diagnostic_metrics: Dict[str, float] = {}
                if flush_diagnostic_metrics is not None:
                    diagnostic_metrics.update(flush_diagnostic_metrics)
                    flush_diagnostic_metrics = None
                else:
                    for key, module in diagnostic_modules.items():
                        if module is None:
                            continue
                        mod_grad_norm, mod_weight_norm = module_grad_weight_norm(module)
                        diagnostic_metrics[f"grad_norm.{key}"] = mod_grad_norm
                        if not math.isnan(mod_grad_norm) and not math.isnan(mod_weight_norm):
                            diagnostic_metrics[f"grad_to_weight.{key}"] = (
                                mod_grad_norm / (mod_weight_norm + 1e-12)
                            )

                if math.isnan(last_grad_norm):
                    diagnostic_metrics["grad_clipped"] = float("nan")
                    diagnostic_metrics["grad_clip_frac"] = float("nan")
                else:
                    grad_clipped = 1.0 if last_grad_norm > args.grad_clip else 0.0
                    diagnostic_metrics["grad_clipped"] = grad_clipped
                    diagnostic_metrics["grad_clip_frac"] = (
                        args.grad_clip / (last_grad_norm + 1e-12) if grad_clipped else 1.0
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
                    **diagnostic_metrics,
                )
                if args.extended_logging and diagnostic_metrics:
                    log_print(f"  Diagnostics: {format_metrics_line(diagnostic_metrics)}")

                running_loss = 0.0
                running_inner_steps = 0.0
                running_ponder = 0.0
            
            # Evaluation
            if global_step % args.eval_interval == 0:
                if args.extended_logging:
                    log_print("\nRunning evaluation...")
                difficulty_logged = (
                    float(difficulty_value.value)
                    if difficulty_value is not None
                    else difficulty_schedule(
                        curriculum.tokens_seen, args.alg_tokens, args.difficulty_schedule
                    )
                )
                if args.extended_logging:
                    log_print(f"  Eval difficulty: {difficulty_logged:.3f}")
                with torch.no_grad():
                    weight_norm = math.sqrt(
                        sum((param.detach().float() ** 2).sum().item() for param in model.parameters())
                    )
                effective_wd_pressure = args.weight_decay * weight_norm
                if args.extended_logging:
                    log_print(f"  Grad norm: {last_grad_norm:.1f}")
                    log_print(f"  Weight norm: {weight_norm:.1f}")
                    if args.weight_decay > 0.0:
                        log_print(f"  WD pressure: {effective_wd_pressure:.1f}")

                # DIAGNOSTIC METRICS (eval-only)
                eval_diagnostic_metrics = run_eval_diagnostics(
                    model=model,
                    probe_batch=probe_batch,
                    device=device,
                    ctx=ctx,
                    diagnostic_blocks=diagnostic_blocks,
                    diagnostic_modules=diagnostic_modules,
                    probe_seed=probe_seed,
                )
                if eval_diagnostic_metrics:
                    logger.log(step=global_step, phase="eval", **eval_diagnostic_metrics)
                    if args.extended_logging:
                        log_print(
                            "  Eval diagnostics: "
                            f"{format_metrics_line(eval_diagnostic_metrics)}"
                        )

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
                    if args.extended_logging:
                        log_print(
                            "  Algorithmic metrics: "
                            f"acc={overall_acc*100:.1f}% | "
                            f"token={overall_token_accuracy:.3f} | "
                            f"dist={overall_distance:.3f} | "
                            f"prefix={overall_prefix_accuracy:.3f}"
                        )
                        if overall_mae is not None:
                            log_print(f"  Algorithmic MAE (numeric tasks): {overall_mae:.3f}")

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
                        if args.extended_logging:
                            line = (
                                f"    {task}: {acc*100:.1f}%"
                                f" | token={token_acc:.3f}"
                                f" | dist={distance:.3f}"
                                f" | prefix={prefix_acc:.3f}"
                            )
                            if task_mae is not None:
                                line += f" | MAE: {task_mae:.3f}"
                            log_print(line)

                    # Track best
                    if overall_acc > best_alg_acc:
                        best_alg_acc = overall_acc
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "step": global_step,
                            "alg_accuracy": best_alg_acc,
                        }, output_dir / "best_model.pt")
                        if args.extended_logging:
                            log_print(f"  New best! Saved checkpoint.")

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
                    if args.extended_logging:
                        log_print(
                            f"  Training accuracy ({train_eval_label}): {train_acc*100:.2f}% | "
                            f"Loss: {train_loss_eval:.4f}"
                        )
                        log_print(f"  Training task accuracy: {overall_task_acc*100:.1f}%")
                        if overall_task_mae is not None:
                            log_print(f"  Training task MAE (numeric tasks): {overall_task_mae:.3f}")

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
                        if args.extended_logging:
                            line = f"    {task}: {acc*100:.1f}%"
                            if task_mae is not None:
                                line += f" | MAE: {task_mae:.3f}"
                            log_print(line)

                # Perplexity (only in Phase 2)
                if curriculum.phase == "language":
                    ppl = evaluate_perplexity(model, lang_loader, device, ctx)
                    if args.extended_logging:
                        log_print(f"  Language PPL: {ppl:.2f}")
                    logger.log(step=global_step, phase="eval", val_loss=math.log(ppl))

                logger.plot()
                logger.save()

                checkpoint_path = save_rotating_checkpoint(global_step, tokens_seen)
                if checkpoint_path is not None and args.extended_logging:
                    log_print(
                        f"  Saved checkpoint: {checkpoint_path.name} "
                        f"(keeping last {args.eval_checkpoints})"
                    )

                if args.extended_logging:
                    log_print()
                model.train()
    except KeyboardInterrupt:
        log_print("\nKeyboard interrupt detected. Saving checkpoint before exit.")
        interrupt_checkpoint = save_rotating_checkpoint(global_step, tokens_seen, tag="interrupt")
        if interrupt_checkpoint is not None:
            log_print(
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
    
    log_print("\n" + "="*60)
    log_print("Training complete!")
    log_print(f"  Total time: {total_time/3600:.2f} hours")
    log_print(f"  Tokens seen: {tokens_seen/1e9:.2f}B")
    log_print(f"  Best algorithmic accuracy: {best_alg_acc*100:.1f}%")
    log_print(f"  Output: {output_dir}")
    log_print("="*60)
    
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
        default="smooth",
        choices=["linear", "phased", "fixed", "smooth", "warmup_ramp"],
        help="Difficulty curriculum type (default: smooth)",
    )
    parser.add_argument(
        "--easy_mix_frac",
        type=float,
        default=0.2,
        help="Fraction of training samples that should always be easy (0.0-0.3 difficulty)",
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
    parser.add_argument("--min_lr", type=float, default=1.5e-4)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="wsd",
        choices=["cosine", "wsd", "plateau", "constant"],
        help="Learning rate schedule type (default: wsd for Warmup-Stable-Decay)",
    )
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
    parser.add_argument(
        "--extended_logging",
        action="store_true",
        help="Print extended metrics beyond loss/LR to the console.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (recommended for Linux only)")

    args = parser.parse_args()

    if args.mix_band_tokens is None:
        args.mix_band_tokens = int(0.5 * args.alg_tokens)
    train(args)


if __name__ == "__main__":
    main()
