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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import math
import json
import random
import argparse
import re
from collections import defaultdict
from functools import partial
from multiprocessing import Manager, Value
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, Dict, List, Sequence, Any, Tuple, Set

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset

from curriculum import DagUnlockState, TaskCurriculumState
from data import get_task_weight, make_collate_fn
from utils import (
    DEFAULT_TOKENIZER_NAME,
    aggregate,
    compute_repetition_metrics,
    compute_attention_probs,
    compute_weight_entropy,
    compute_metrics,
    digit_length_from_token,
    extract_prediction_from_generate,
    extract_integer_token,
    get_transformer_blocks,
    get_prompt_format_flags,
    module_grad_weight_norm,
    normalize_prediction,
    normalize_label_prediction,
    normalize_target,
    parse_numeric_prediction,
    PROMPT_FORMAT_KEYS,
    resolve_diagnostic_modules,
    safe_parse_number,
)

plt.switch_backend("Agg")

def _uniform_weighting(_: str) -> float:
    return 1.0

def _create_shared_dict(manager: Optional[Manager]) -> Dict[str, float]:
    if manager is None:
        return {}
    return manager.dict()


def _create_shared_list(manager: Optional[Manager], initial: Optional[List[float]] = None) -> List[float]:
    if manager is None:
        return initial or []
    shared = manager.list(initial or [])
    return shared


def _set_frontier_snapshot(snapshot: Any, frontier: Set[str]) -> None:
    if hasattr(snapshot, "clear"):
        snapshot.clear()
    else:
        try:
            snapshot[:] = []
        except TypeError:
            try:
                del snapshot[:]
            except TypeError:
                while len(snapshot):
                    snapshot.pop()
    if hasattr(snapshot, "extend") and not hasattr(snapshot, "update"):
        snapshot.extend(sorted(frontier))
    else:
        snapshot.update(frontier)

def compute_dag_weighting(
    tasks: Sequence[str],
    weights: Sequence[float],
    dag_gate_snapshot: Dict[str, float],
    dag_frontier_snapshot: Set[str],
    dag_replay_ratio_snapshot: float,
    dag_locked_floor: float,
) -> Tuple[List[float], Dict[str, float], Dict[str, float]]:
    if not tasks:
        return [], {}, {}
    gates = [dag_gate_snapshot.get(task, 0.0) for task in tasks]
    unlocked = [idx for idx, gate in enumerate(gates) if gate > 0.0]
    locked = [idx for idx, gate in enumerate(gates) if gate <= 0.0]
    if not unlocked:
        total = sum(weights)
        probs = {
            task: (weight / total if total else 0.0)
            for task, weight in zip(tasks, weights)
        }
        return list(weights), probs, {task: weight for task, weight in zip(tasks, weights)}

    gated_weights = [weight * gate for weight, gate in zip(weights, gates)]
    total_gated = sum(gated_weights[idx] for idx in unlocked)
    if total_gated <= 0.0:
        for idx in unlocked:
            gated_weights[idx] = 1.0
        total_gated = float(len(unlocked))

    frontier_set = set(dag_frontier_snapshot)
    frontier_indices = [
        idx for idx, task in enumerate(tasks) if task in frontier_set and gates[idx] > 0.0
    ]
    replay_indices = [
        idx for idx in unlocked if idx not in frontier_indices
    ]

    total_frontier = sum(gated_weights[idx] for idx in frontier_indices)
    total_replay = sum(gated_weights[idx] for idx in replay_indices)

    probs: List[float] = [0.0] * len(tasks)
    replay_ratio = dag_replay_ratio_snapshot
    if not frontier_indices:
        replay_ratio = 1.0
    elif not replay_indices:
        replay_ratio = 0.0
    if total_frontier > 0.0:
        for idx in frontier_indices:
            probs[idx] += (1.0 - replay_ratio) * (gated_weights[idx] / total_frontier)
    if total_replay > 0.0:
        for idx in replay_indices:
            probs[idx] += replay_ratio * (gated_weights[idx] / total_replay)

    if sum(probs) <= 0.0:
        uniform = 1.0 / len(unlocked)
        for idx in unlocked:
            probs[idx] = uniform

    min_prob_unlocked_total = 0.02
    floor_unlocked = min_prob_unlocked_total / len(unlocked)
    for idx in unlocked:
        probs[idx] = max(probs[idx], floor_unlocked)

    min_prob_frontier_total = 0.03
    if frontier_indices:
        floor_frontier = min_prob_frontier_total / len(frontier_indices)
        for idx in frontier_indices:
            probs[idx] = max(probs[idx], floor_frontier)

    locked_floor_total = max(dag_locked_floor, 0.0)
    if locked and locked_floor_total > 0.0:
        locked_weight_total = sum(weights[idx] for idx in locked)
        if locked_weight_total <= 0.0:
            locked_weight_total = float(len(locked))
            for idx in locked:
                weights[idx] = 1.0
        for idx in locked:
            probs[idx] = max(
                probs[idx],
                locked_floor_total * (weights[idx] / locked_weight_total),
            )

    total = sum(probs)
    if total <= 0.0:
        uniform = 1.0 / len(unlocked)
        for idx in unlocked:
            probs[idx] = uniform
        total = 1.0
    probs = [prob / total for prob in probs]
    prob_map = {task: prob for task, prob in zip(tasks, probs)}
    gated_map = {task: weight for task, weight in zip(tasks, gated_weights)}
    return probs, prob_map, gated_map


class CurriculumDifficultySampler:
    def __init__(
        self,
        difficulty_snapshot: Dict[str, float],
        *,
        easy_mix_frac: float,
        curriculum_jitter: float,
        min_difficulty: float,
    ) -> None:
        self._difficulty_snapshot = difficulty_snapshot
        self._easy_mix_frac = easy_mix_frac
        self._curriculum_jitter = curriculum_jitter
        self._min_difficulty = min_difficulty

    def __call__(self, task: str) -> float:
        if self._easy_mix_frac > 0.0 and random.random() < self._easy_mix_frac:
            return random.uniform(0.0, 0.3)

        difficulty = self._difficulty_snapshot.get(task, 0.2)
        if self._curriculum_jitter > 0.0 and random.random() < self._curriculum_jitter:
            return random.uniform(0.0, difficulty)
        return max(difficulty, self._min_difficulty)


class CurriculumEvalDifficultySampler:
    def __init__(
        self,
        difficulty_snapshot: Dict[str, float],
        *,
        min_difficulty: float,
    ) -> None:
        self._difficulty_snapshot = difficulty_snapshot
        self._min_difficulty = min_difficulty

    def __call__(self, task: str) -> float:
        return max(self._difficulty_snapshot.get(task, 0.2), self._min_difficulty)


class CurriculumWeightingSampler:
    def __init__(self, weights_snapshot: Dict[str, float]) -> None:
        self._weights_snapshot = weights_snapshot

    def __call__(self, task: str) -> float:
        return self._weights_snapshot.get(task, 1.0)


class CurriculumWeightingAdjuster:
    def __init__(
        self,
        difficulty_snapshot: Dict[str, float],
        *,
        min_task_prob: float,
    ) -> None:
        self._difficulty_snapshot = difficulty_snapshot
        self._min_task_prob = min_task_prob

    def __call__(self, tasks: Sequence[str], weights: List[float]) -> List[float]:
        if not tasks:
            return weights
        mean_difficulty = sum(
            self._difficulty_snapshot.get(task, 0.2) for task in tasks
        ) / len(tasks)
        if mean_difficulty >= 0.2:
            return weights
        return _apply_minimum_probability(weights, self._min_task_prob)


class DagWeightingAdjuster:
    def __init__(
        self,
        dag_gate_snapshot: Dict[str, float],
        dag_frontier_snapshot: Set[str],
        dag_replay_ratio_snapshot: List[float],
        dag_locked_floor: float,
    ) -> None:
        self._dag_gate_snapshot = dag_gate_snapshot
        self._dag_frontier_snapshot = dag_frontier_snapshot
        self._dag_replay_ratio_snapshot = dag_replay_ratio_snapshot
        self._dag_locked_floor = dag_locked_floor

    def __call__(self, tasks: Sequence[str], weights: List[float]) -> List[float]:
        adjusted, _, _ = compute_dag_weighting(
            tasks,
            weights,
            self._dag_gate_snapshot,
            self._dag_frontier_snapshot,
            self._dag_replay_ratio_snapshot[0],
            self._dag_locked_floor,
        )
        return adjusted

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
        self.task_accuracies = {}
        self.ood_task_accuracies = {}
        self.train_eval_task_accuracies = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.output_dir / "metrics.json"
        if not self._metrics_path.exists():
            self._append_snapshot()
    
    def log(self, step: int, phase: str, **kwargs):
        record = {"step": step, "phase": phase}
        record.update(kwargs)
        self.records.append(record)
    
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
        if target == "train_eval":
            store = self.train_eval_task_accuracies
        elif target == "eval_ood":
            store = self.ood_task_accuracies
        else:
            store = self.task_accuracies
        if task not in store:
            store[task] = []
        prefix = target
        entry = {"step": step, f"{prefix}/acc/exact_match": accuracy}
        if mae is not None:
            entry[f"{prefix}/numeric/mae"] = mae
        if mean_token_accuracy is not None:
            entry[f"{prefix}/acc/token"] = mean_token_accuracy
        if mean_distance is not None:
            entry[f"{prefix}/acc/distance"] = mean_distance
        if mean_prefix_accuracy is not None:
            entry[f"{prefix}/acc/prefix"] = mean_prefix_accuracy
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
        tmp_path = self._metrics_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump({"type": "hyperparameters", "data": self.hyperparameters}, f)
            f.write("\n")
            for record in self.records:
                json.dump({"type": "record", **record}, f)
                f.write("\n")
            for task, records in self.task_accuracies.items():
                for entry in records:
                    json.dump(
                        {"type": "task_accuracy", "target": "eval", "task": task, **entry},
                        f,
                    )
                    f.write("\n")
            for task, records in self.ood_task_accuracies.items():
                for entry in records:
                    json.dump(
                        {
                            "type": "task_accuracy",
                            "target": "eval_ood",
                            "task": task,
                            **entry,
                        },
                        f,
                    )
                    f.write("\n")
            for task, records in self.train_eval_task_accuracies.items():
                for entry in records:
                    json.dump(
                        {
                            "type": "task_accuracy",
                            "target": "train_eval",
                            "task": task,
                            **entry,
                        },
                        f,
                    )
                    f.write("\n")
        os.replace(tmp_path, self._metrics_path)
    
    def summary(self) -> Dict:
        train_losses = self.series("train/loss/ce")[1]
        val_losses = self.series("eval/loss/ce")[1]
        peak_vram = self.series("train/throughput/vram_gb")[1]
        tokens_per_sec = self.series("train/throughput/tokens_per_sec")[1]
        tokens_seen = self.series("train/throughput/tokens_seen")[1]
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
        train_steps, train_losses = self.series("train/loss/ce")
        if train_steps and train_losses:
            plt.plot(train_steps, train_losses, label="train")

        val_steps, val_losses = self.series("eval/loss/ce", phase="eval")
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

        # Update-scale proxies vs loss
        update_steps, update_postclip = self.series("train/update/norm/postclip")
        loss_steps, loss_values = self.series("train/loss/ce")
        if update_steps and update_postclip:
            fig, ax1 = plt.subplots()
            ax1.plot(update_steps, update_postclip, label="train/update/norm/postclip")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Update norm (postclip)")
            ax1.grid(True)
            ax2 = ax1.twinx()
            if loss_steps and loss_values:
                ax2.plot(loss_steps, loss_values, alpha=0.6, label="train/loss/ce")
                ax2.set_ylabel("Loss")
                ax2.set_yscale("log")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines or lines2:
                ax1.legend(lines + lines2, labels + labels2, loc="best")
            fig.tight_layout()
            fig.savefig(self.output_dir / "update_norm_postclip.png")
            plt.close(fig)

        update_weight_steps, update_weight_postclip = self.series(
            "train/update/ratio/postclip"
        )
        if update_weight_steps and update_weight_postclip:
            fig, ax1 = plt.subplots()
            ax1.plot(
                update_weight_steps,
                update_weight_postclip,
                label="train/update/ratio/postclip",
            )
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Update/weight (postclip)")
            ax1.grid(True)
            ax2 = ax1.twinx()
            if loss_steps and loss_values:
                ax2.plot(loss_steps, loss_values, alpha=0.6, label="train/loss/ce")
                ax2.set_ylabel("Loss")
                ax2.set_yscale("log")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines or lines2:
                ax1.legend(lines + lines2, labels + labels2, loc="best")
            fig.tight_layout()
            fig.savefig(self.output_dir / "update_to_weight_postclip.png")
            plt.close(fig)

        # Accuracy plot
        plt.figure()
        for task, records in self.task_accuracies.items():
            steps = [entry["step"] for entry in records]
            accuracies = [entry["eval/acc/exact_match"] for entry in records]
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

        # OOD accuracy plot
        if self.ood_task_accuracies:
            plt.figure()
            for task, records in self.ood_task_accuracies.items():
                steps = [entry["step"] for entry in records]
                accuracies = [entry["eval_ood/acc/exact_match"] for entry in records]
                if steps and accuracies:
                    plt.plot(steps, accuracies, label=task)

            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.title("OOD Task Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / "accuracy_ood.png")
            plt.close()

        # Training task accuracy plot
        if self.train_eval_task_accuracies:
            plt.figure()
            for task, records in self.train_eval_task_accuracies.items():
                steps = [entry["step"] for entry in records]
                accuracies = [entry["train_eval/acc/exact_match"] for entry in records]
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
    grid: str = "iid",
) -> Dict[str, Any]:
    """Run the canonical algorithmic grid using ``eval_hub``.

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
        grid=grid,
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
        "diagnostics": results.get("diagnostics", {}),
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
        
        n_tokens = (labels[..., 1:] != -100).sum().item()
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
        shift_labels = labels[..., 1:]
        shift_predictions = predictions[..., :-1]
        mask = shift_labels != -100
        total_correct += (shift_predictions[mask] == shift_labels[mask]).sum().item()

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
        if len(inputs) > 5 and inputs[5] is not None:
            last_block_input["position_ids"] = inputs[5].detach()

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
            metrics[f"eval/act/sparsity/ff/{label}"] = (
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
            metrics["eval/attn/entropy/last/mean"] = mean_by_head.mean().item()
            metrics["eval/attn/entropy/last/std"] = mean_by_head.std(unbiased=False).item()

    if last_block is not None:
        metrics["eval/weight/entropy/last_block"] = compute_weight_entropy(last_block)
    head_module = diagnostic_modules.get("head")
    if head_module is not None:
        metrics["eval/weight/entropy/head"] = compute_weight_entropy(head_module)

    if was_training:
        model.train()

    return metrics


NUMERIC_TASKS = {"mod_add", "addition", "multiplication", "chain", "successor"}
MAE_TASKS = {"mod_add", "addition", "multiplication", "chain", "successor"}
SEQUENCE_TASKS = {"copy", "reverse"}
CLASSIFICATION_TASKS = {"compare", "dyck", "parity"}
NUMERIC_OUTPUT_TASKS = {"mod_add", "addition", "multiplication", "chain", "successor"}
assert NUMERIC_TASKS.isdisjoint(CLASSIFICATION_TASKS)
assert MAE_TASKS.issubset(NUMERIC_TASKS)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _token_length(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        return int(encoded["input_ids"].shape[1])


def _compute_task_sampling(
    tasks: Sequence[str],
    progress: float,
    task_weighting: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not tasks:
        return {}, {}
    if task_weighting == "adaptive":
        weights = {task: get_task_weight(task, progress) for task in tasks}
    else:
        weights = {task: 1.0 for task in tasks}
    total = sum(weights.values())
    probs = {task: (weights[task] / total if total else 0.0) for task in weights}
    return weights, probs


def _apply_minimum_probability(weights: Sequence[float], min_prob: float) -> List[float]:
    if not weights:
        return []
    if min_prob <= 0.0:
        return list(weights)
    total = sum(weights)
    if total <= 0.0:
        return [1.0] * len(weights)
    probs = [weight / total for weight in weights]
    floor_total = min_prob * len(probs)
    if floor_total >= 1.0:
        return [1.0] * len(probs)
    adjusted = [max(prob, min_prob) for prob in probs]
    excess = sum(adjusted) - 1.0
    if excess > 1e-12:
        above = [max(prob - min_prob, 0.0) for prob in adjusted]
        above_total = sum(above)
        if above_total <= 0.0:
            return [1.0] * len(probs)
        scale = (1.0 - floor_total) / above_total
        adjusted = [
            min_prob + (prob - min_prob) * scale if prob > min_prob else min_prob
            for prob in adjusted
        ]
    return adjusted


def _relabel_phase(metrics: Dict[str, float], phase: str) -> Dict[str, float]:
    relabeled = {}
    for key, value in metrics.items():
        if "/" in key:
            relabeled[f"{phase}/{key.split('/', 1)[1]}"] = value
        else:
            relabeled[f"{phase}/{key}"] = value
    return relabeled


def _flatten_eval_diagnostics(prefix: str, diagnostics: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    if not diagnostics:
        return flat
    answer_len = diagnostics.get("answer_length", {})
    mean_lengths = answer_len.get("mean", {})
    for key, value in mean_lengths.items():
        metric = {
            "target_len_tokens": "gen/len/target/mean",
            "pred_len_tokens": "gen/len/pred/mean",
            "length_ratio": "gen/len/ratio/mean",
            "abs_len_error": "gen/len/error/mean",
            "pred_len_tokens_p50": "gen/len/pred/p50",
            "pred_len_tokens_p90": "gen/len/pred/p90",
        }.get(key, f"gen/len/{key}/mean")
        flat[f"{prefix}/{metric}"] = float(value)
    for task, payload in answer_len.get("by_task", {}).items():
        for key, value in payload.items():
            metric = {
                "target_len_tokens": "gen/len/target",
                "pred_len_tokens": "gen/len/pred",
                "length_ratio": "gen/len/ratio",
                "abs_len_error": "gen/len/error",
                "pred_len_tokens_p50": "gen/len/pred/p50",
                "pred_len_tokens_p90": "gen/len/pred/p90",
            }.get(key, f"gen/len/{key}")
            flat[f"{prefix}/{metric}/{task}"] = float(value)
    flat[f"{prefix}/gen/stop/eos_rate"] = float(answer_len.get("eos_emitted_rate", 0.0))
    flat[f"{prefix}/gen/stop/eos_first_rate"] = float(
        answer_len.get("eos_first_rate", 0.0)
    )
    flat[f"{prefix}/gen/empty_rate"] = float(answer_len.get("empty_rate", 0.0))
    flat[f"{prefix}/gen/stop/truncated_rate"] = float(
        answer_len.get("max_new_tokens_rate", 0.0)
    )
    for task, rate in answer_len.get("eos_emitted_rate_by_task", {}).items():
        flat[f"{prefix}/gen/stop/eos_rate/{task}"] = float(rate)
    for task, rate in answer_len.get("eos_first_rate_by_task", {}).items():
        flat[f"{prefix}/gen/stop/eos_first_rate/{task}"] = float(rate)
    for task, rate in answer_len.get("empty_rate_by_task", {}).items():
        flat[f"{prefix}/gen/empty_rate/{task}"] = float(rate)
    for task, rate in answer_len.get("max_new_tokens_rate_by_task", {}).items():
        flat[f"{prefix}/gen/stop/truncated_rate/{task}"] = float(rate)
    for reason, count in answer_len.get("stop_reason_counts", {}).items():
        flat[f"{prefix}/gen/stop/reason/{reason}/count"] = float(count)
    for task, reasons in answer_len.get("stop_reason_counts_by_task", {}).items():
        for reason, count in reasons.items():
            flat[f"{prefix}/gen/stop/reason/{reason}/{task}/count"] = float(count)

    repetition = diagnostics.get("repetition", {})
    for metric_name, payload in repetition.items():
        metric = {
            "repeat_1gram_rate": "gen/repeat/1gram",
            "repeat_2gram_rate": "gen/repeat/2gram",
            "unique_token_fraction": "gen/repeat/unique_fraction",
            "max_run_length": "gen/repeat/max_run_length",
        }.get(metric_name, f"gen/repeat/{metric_name}")
        flat[f"{prefix}/{metric}/mean"] = float(payload.get("mean", 0.0))
        if "median" in payload:
            flat[f"{prefix}/{metric}/median"] = float(payload.get("median", 0.0))
        for task, value in payload.get("by_task", {}).items():
            flat[f"{prefix}/{metric}/{task}"] = float(value)

    parse = diagnostics.get("parse", {})
    flat[f"{prefix}/acc/parse_success"] = float(parse.get("parse_success_rate", 0.0))
    flat[f"{prefix}/numeric/non_numeric_rate"] = float(parse.get("non_numeric_rate", 0.0))
    flat[f"{prefix}/numeric/mae"] = float(parse.get("numeric_abs_error", 0.0))
    flat[f"{prefix}/numeric/mae_median"] = float(
        parse.get("numeric_abs_error_median", 0.0)
    )
    flat[f"{prefix}/numeric/rel_error"] = float(parse.get("numeric_rel_error", 0.0))
    flat[f"{prefix}/numeric/parse_fail_rate"] = float(
        parse.get("numeric_parse_fail_rate", 0.0)
    )
    flat[f"{prefix}/numeric/len_mismatch_rate"] = float(
        parse.get("length_mismatch_rate", 0.0)
    )
    for task, payload in parse.get("by_task", {}).items():
        flat[f"{prefix}/acc/parse_success/{task}"] = float(
            payload.get("parse_success_rate", 0.0)
        )
        flat[f"{prefix}/numeric/non_numeric_rate/{task}"] = float(
            payload.get("non_numeric_rate", 0.0)
        )
        flat[f"{prefix}/numeric/mae/{task}"] = float(
            payload.get("numeric_abs_error", 0.0)
        )
        flat[f"{prefix}/numeric/mae_median/{task}"] = float(
            payload.get("numeric_abs_error_median", 0.0)
        )
        flat[f"{prefix}/numeric/rel_error/{task}"] = float(
            payload.get("numeric_rel_error", 0.0)
        )
        flat[f"{prefix}/numeric/parse_fail_rate/{task}"] = float(
            payload.get("numeric_parse_fail_rate", 0.0)
        )
        flat[f"{prefix}/numeric/len_mismatch_rate/{task}"] = float(
            payload.get("length_mismatch_rate", 0.0)
        )
    for reason, count in parse.get("failure_counts", {}).items():
        flat[f"{prefix}/numeric/parse_failure/{reason}/count"] = float(count)
    for task, reasons in parse.get("failure_counts_by_task", {}).items():
        for reason, count in reasons.items():
            flat[f"{prefix}/numeric/parse_failure/{reason}/{task}/count"] = float(count)

    labels = diagnostics.get("labels", {})
    flat[f"{prefix}/class/invalid_label_rate"] = float(
        labels.get("invalid_label_rate", 0.0)
    )
    for task, rate in labels.get("invalid_label_rate_by_task", {}).items():
        flat[f"{prefix}/class/invalid_label_rate/{task}"] = float(rate)
    for key, count in labels.get("confusion_counts", {}).items():
        flat[f"{prefix}/class/confusion/{key}"] = float(count)
    for task, counts in labels.get("confusion_counts_by_task", {}).items():
        for key, count in counts.items():
            flat[f"{prefix}/class/confusion/{task}/{key}"] = float(count)

    prompt_format = diagnostics.get("prompt_format", {})
    for key, payload in prompt_format.get("overall", {}).items():
        flat[f"{prefix}/gen/format/{key}/count"] = float(payload.get("count", 0.0))
        flat[f"{prefix}/gen/format/{key}/acc"] = float(
            payload.get("accuracy", 0.0)
        )
        flat[f"{prefix}/gen/format/{key}/empty_rate"] = float(
            payload.get("empty_rate", 0.0)
        )
        flat[f"{prefix}/gen/format/{key}/eos_first_rate"] = float(
            payload.get("eos_first_rate", 0.0)
        )
    for task, stats in prompt_format.get("by_task", {}).items():
        for key, payload in stats.items():
            flat[f"{prefix}/gen/format/{key}/{task}/count"] = float(
                payload.get("count", 0.0)
            )
            flat[f"{prefix}/gen/format/{key}/{task}/acc"] = float(
                payload.get("accuracy", 0.0)
            )
            flat[f"{prefix}/gen/format/{key}/{task}/empty_rate"] = float(
                payload.get("empty_rate", 0.0)
            )
            flat[f"{prefix}/gen/format/{key}/{task}/eos_first_rate"] = float(
                payload.get("eos_first_rate", 0.0)
            )

    soft_hard_gap = diagnostics.get("soft_hard_gap", {})
    for metric in ("token", "prefix"):
        gap = soft_hard_gap.get(metric, {})
        flat[f"{prefix}/acc/soft_hard_gap/{metric}/overall"] = float(
            gap.get("overall", 0.0)
        )
        for task, value in gap.get("by_task", {}).items():
            flat[f"{prefix}/acc/soft_hard_gap/{metric}/{task}"] = float(value)

    for task, count in diagnostics.get("task_counts", {}).items():
        flat[f"{prefix}/throughput/task_count/{task}"] = float(count)
    for task, value in diagnostics.get("difficulty_by_task", {}).items():
        flat[f"{prefix}/curriculum/difficulty/{task}"] = float(value)
    return flat


def _decode_example_text(example: Dict[str, torch.Tensor], tokenizer) -> Optional[str]:
    input_ids = example["input_ids"]
    labels = example["labels"]
    valid = (labels != -100).nonzero(as_tuple=False).view(-1)
    if valid.numel() == 0:
        return None
    last_pos = int(valid[-1].item())
    full_tokens = input_ids[: last_pos + 1]
    return tokenizer.decode(full_tokens, skip_special_tokens=True)


def _extract_prompt_target(full_text: str) -> Optional[Dict[str, str]]:
    marker = "Answer:"
    idx = full_text.rfind(marker)
    if idx == -1:
        return None
    prompt_end = idx + len(marker)
    if prompt_end < len(full_text) and full_text[prompt_end] == " ":
        prompt_end += 1
    prompt = full_text[:prompt_end]
    target = full_text[prompt_end:].strip()
    if not prompt or target == "":
        return None
    return {"prompt": prompt, "target": target}


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


def _predict_with_tokens(
    model,
    tokenizer,
    prompt: str,
    task: str,
    device,
    max_new_tokens: int,
) -> Optional[Dict[str, Any]]:
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"].to(device)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=eos_token_id,
    )
    pred_raw, gen_tokens, stop_reason, first_token_is_eos = extract_prediction_from_generate(
        input_ids[0],
        output[0],
        eos_token_id,
        tokenizer,
    )
    pred_norm = (
        normalize_prediction(task, pred_raw)
        if task in NUMERIC_TASKS or task in CLASSIFICATION_TASKS
        else _normalize_text(pred_raw)
    )
    return {
        "pred_norm": pred_norm,
        "pred_raw": pred_raw,
        "pred_tokens": gen_tokens,
        "stop_reason": stop_reason,
        "first_token_is_eos": first_token_is_eos,
    }


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
    target_len_tokens: List[int] = []
    pred_len_tokens: List[int] = []
    length_ratio: List[float] = []
    abs_len_error: List[float] = []
    per_task_lengths: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "target_len_tokens": [],
            "pred_len_tokens": [],
            "length_ratio": [],
            "abs_len_error": [],
        }
    )
    stop_reason_counts = {"eos": 0, "max_new_tokens": 0, "other": 0}
    stop_reason_counts_by_task: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"eos": 0, "max_new_tokens": 0, "other": 0}
    )
    eos_emitted = 0
    eos_emitted_by_task: Dict[str, int] = defaultdict(int)
    repeat_1gram_rates: List[float] = []
    repeat_2gram_rates: List[float] = []
    unique_token_fractions: List[float] = []
    max_run_lengths: List[float] = []
    repetition_by_task: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "repeat_1gram_rate": [],
            "repeat_2gram_rate": [],
            "unique_token_fraction": [],
            "max_run_length": [],
        }
    )
    parse_successes = 0
    parse_failure_counts = defaultdict(int)
    parse_successes_by_task: Dict[str, int] = defaultdict(int)
    parse_failure_counts_by_task: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    numeric_abs_errors: List[float] = []
    numeric_rel_errors: List[float] = []
    numeric_abs_errors_by_task: Dict[str, List[float]] = defaultdict(list)
    numeric_rel_errors_by_task: Dict[str, List[float]] = defaultdict(list)
    task_difficulty_sum: Dict[str, float] = defaultdict(float)
    empty_prediction_count = 0
    first_token_is_eos_count = 0
    empty_predictions_by_task: Dict[str, int] = defaultdict(int)
    first_token_is_eos_by_task: Dict[str, int] = defaultdict(int)
    invalid_label_count = 0
    invalid_label_by_task: Dict[str, int] = defaultdict(int)
    label_confusion_counts: Dict[str, int] = defaultdict(int)
    label_confusion_by_task: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    format_stats = {
        key: {"count": 0, "correct": 0, "empty": 0, "eos_first": 0}
        for key in PROMPT_FORMAT_KEYS
    }
    format_stats_by_task: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: {
            key: {"count": 0, "correct": 0, "empty": 0, "eos_first": 0}
            for key in PROMPT_FORMAT_KEYS
        }
    )
    numeric_length_mismatch_count = 0
    numeric_length_total = 0
    numeric_length_mismatch_by_task: Dict[str, int] = defaultdict(int)
    numeric_length_total_by_task: Dict[str, int] = defaultdict(int)

    for example in _iter_eval_examples(loader, max_samples):
        task = example.get("task")
        if task is None:
            continue
        full_text = _decode_example_text(example, tokenizer)
        if not full_text:
            continue
        parsed = _extract_prompt_target(full_text)
        if not parsed:
            continue
        prompt = parsed["prompt"]
        target = parsed["target"]
        if not prompt or not target:
            continue

        pred_bundle = _predict_with_tokens(
            model, tokenizer, prompt, task, device, max_new_tokens
        )
        if pred_bundle is None:
            continue
        pred_norm = pred_bundle["pred_norm"]
        pred_raw = pred_bundle["pred_raw"]
        empty_prediction = pred_raw.strip() == ""

        metrics = compute_metrics(task, pred_norm, target, tokenizer)
        metrics_by_task[task].append(metrics)
        total[task] += 1
        difficulty = example.get("difficulty")
        if difficulty is not None:
            task_difficulty_sum[task] += float(difficulty)
        if metrics["exact_match"] == 1.0:
            correct[task] += 1

        target_len = _token_length(tokenizer, normalize_target(task, target))
        pred_tokens = pred_bundle["pred_tokens"]
        pred_len = len(pred_tokens)
        target_len_tokens.append(target_len)
        pred_len_tokens.append(pred_len)
        length_ratio.append(pred_len / max(target_len, 1))
        abs_len_error.append(abs(pred_len - target_len))
        stop_reason = pred_bundle["stop_reason"]
        first_token_is_eos = pred_bundle["first_token_is_eos"]
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        task_stop_counts = stop_reason_counts_by_task[task]
        task_stop_counts[stop_reason] = task_stop_counts.get(stop_reason, 0) + 1
        eos_emitted += 1 if stop_reason == "eos" else 0
        eos_emitted_by_task[task] += 1 if stop_reason == "eos" else 0
        empty_prediction_count += 1 if empty_prediction else 0
        first_token_is_eos_count += 1 if first_token_is_eos else 0
        empty_predictions_by_task[task] += 1 if empty_prediction else 0
        first_token_is_eos_by_task[task] += 1 if first_token_is_eos else 0
        per_task_lengths[task]["target_len_tokens"].append(target_len)
        per_task_lengths[task]["pred_len_tokens"].append(pred_len)
        per_task_lengths[task]["length_ratio"].append(pred_len / max(target_len, 1))
        per_task_lengths[task]["abs_len_error"].append(abs(pred_len - target_len))
        repetition = compute_repetition_metrics(pred_tokens)
        repeat_1gram_rates.append(repetition["repeat_1gram_rate"])
        repeat_2gram_rates.append(repetition["repeat_2gram_rate"])
        unique_token_fractions.append(repetition["unique_token_fraction"])
        max_run_lengths.append(repetition["max_run_length"])
        repetition_by_task[task]["repeat_1gram_rate"].append(repetition["repeat_1gram_rate"])
        repetition_by_task[task]["repeat_2gram_rate"].append(repetition["repeat_2gram_rate"])
        repetition_by_task[task]["unique_token_fraction"].append(
            repetition["unique_token_fraction"]
        )
        repetition_by_task[task]["max_run_length"].append(repetition["max_run_length"])

        prompt_flags = get_prompt_format_flags(prompt)
        for key, enabled in prompt_flags.items():
            if not enabled:
                continue
            format_stats[key]["count"] += 1
            format_stats[key]["correct"] += 1 if metrics["exact_match"] == 1.0 else 0
            format_stats[key]["empty"] += 1 if empty_prediction else 0
            format_stats[key]["eos_first"] += 1 if first_token_is_eos else 0
            format_stats_by_task[task][key]["count"] += 1
            format_stats_by_task[task][key]["correct"] += (
                1 if metrics["exact_match"] == 1.0 else 0
            )
            format_stats_by_task[task][key]["empty"] += 1 if empty_prediction else 0
            format_stats_by_task[task][key]["eos_first"] += (
                1 if first_token_is_eos else 0
            )

        valid_label = None
        if task in {"compare", "dyck", "parity"}:
            pred_label = normalize_label_prediction(task, pred_raw)
            target_label = normalize_label_prediction(task, target)
            valid_label = pred_label is not None
            if not valid_label:
                invalid_label_count += 1
                invalid_label_by_task[task] += 1
            elif target_label is not None:
                key = f"pred_{pred_label}_when_{target_label}"
                label_confusion_counts[key] += 1
                label_confusion_by_task[task][key] += 1

        parse_ok = None
        digit_length_pred = None
        digit_length_target = None
        if task in numeric_tasks:
            parsed_pred, failure_reason = parse_numeric_prediction(pred_bundle["pred_raw"])
            parsed_tgt, tgt_failure = parse_numeric_prediction(target)
            parse_ok = failure_reason is None
            if not parse_ok:
                if failure_reason is not None:
                    parse_failure_counts[failure_reason] += 1
                    parse_failure_counts_by_task[task][failure_reason] += 1
            elif tgt_failure is not None:
                parse_failure_counts[tgt_failure] += 1
                parse_failure_counts_by_task[task][tgt_failure] += 1
            else:
                parse_successes += 1
                parse_successes_by_task[task] += 1
            if parse_ok and tgt_failure is None and task in MAE_TASKS:
                abs_error = abs(parsed_pred - parsed_tgt)
                rel_error = abs_error / max(abs(parsed_tgt), 1.0)
                total_abs_error += abs_error
                total_numeric += 1
                per_task_abs_error[task] += abs_error
                per_task_numeric[task] += 1
                numeric_abs_errors.append(abs_error)
                numeric_rel_errors.append(rel_error)
                numeric_abs_errors_by_task[task].append(abs_error)
                numeric_rel_errors_by_task[task].append(rel_error)
            pred_token = extract_integer_token(pred_bundle["pred_raw"])
            tgt_token = extract_integer_token(target)
            digit_length_pred = digit_length_from_token(pred_token)
            digit_length_target = digit_length_from_token(tgt_token)
            if digit_length_pred is not None and digit_length_target is not None:
                numeric_length_total += 1
                numeric_length_total_by_task[task] += 1
                if digit_length_pred != digit_length_target:
                    numeric_length_mismatch_count += 1
                    numeric_length_mismatch_by_task[task] += 1
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
                "target_len_tokens": target_len,
                "pred_len_tokens": pred_len,
                "stop_reason": stop_reason,
                "first_token_is_eos": first_token_is_eos,
                "empty_prediction": empty_prediction,
                "valid_label": valid_label,
                "digit_length_target": digit_length_target,
                "digit_length_pred": digit_length_pred,
                "parse_ok": parse_ok,
                "repeat_1gram_rate": repetition["repeat_1gram_rate"],
                "difficulty": difficulty,
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
    total_numeric_samples = parse_successes + sum(parse_failure_counts.values())
    assert sum(total.values()) == len(target_len_tokens) == len(pred_len_tokens)
    task_counts = dict(total)
    task_difficulty_mean = {
        task: (
            task_difficulty_sum[task] / task_counts[task]
            if task_counts[task]
            else 0.0
        )
        for task in task_counts
    }
    if task_counts and not task_difficulty_sum:
        task_difficulty_mean = {task: 0.0 for task in task_counts}

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return float(ordered[mid])
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = int(round((len(ordered) - 1) * pct))
        return float(ordered[max(0, min(idx, len(ordered) - 1))])

    answer_length = {
        "mean": {
            "target_len_tokens": _mean(target_len_tokens),
            "pred_len_tokens": _mean(pred_len_tokens),
            "length_ratio": _mean(length_ratio),
            "abs_len_error": _mean(abs_len_error),
            "pred_len_tokens_p50": _percentile(pred_len_tokens, 0.5),
            "pred_len_tokens_p90": _percentile(pred_len_tokens, 0.9),
        },
        "by_task": {
            task: {
                "target_len_tokens": _mean(payload["target_len_tokens"]),
                "pred_len_tokens": _mean(payload["pred_len_tokens"]),
                "length_ratio": _mean(payload["length_ratio"]),
                "abs_len_error": _mean(payload["abs_len_error"]),
                "pred_len_tokens_p50": _percentile(payload["pred_len_tokens"], 0.5),
                "pred_len_tokens_p90": _percentile(payload["pred_len_tokens"], 0.9),
            }
            for task, payload in per_task_lengths.items()
        },
        "eos_emitted_rate": eos_emitted / sum(total.values()) if total else 0.0,
        "eos_emitted_rate_by_task": {
            task: (eos_emitted_by_task.get(task, 0) / task_counts.get(task, 1))
            for task in task_counts
        },
        "eos_first_rate": (
            first_token_is_eos_count / sum(total.values()) if total else 0.0
        ),
        "eos_first_rate_by_task": {
            task: (first_token_is_eos_by_task.get(task, 0) / task_counts.get(task, 1))
            for task in task_counts
        },
        "empty_rate": empty_prediction_count / sum(total.values()) if total else 0.0,
        "empty_rate_by_task": {
            task: (empty_predictions_by_task.get(task, 0) / task_counts.get(task, 1))
            for task in task_counts
        },
        "max_new_tokens_rate": (
            stop_reason_counts.get("max_new_tokens", 0) / sum(total.values())
            if total
            else 0.0
        ),
        "max_new_tokens_rate_by_task": {
            task: (
                stop_reason_counts_by_task.get(task, {}).get("max_new_tokens", 0)
                / task_counts.get(task, 1)
            )
            for task in task_counts
        },
        "stop_reason_counts": stop_reason_counts,
        "stop_reason_counts_by_task": dict(stop_reason_counts_by_task),
    }

    repetition = {
        "repeat_1gram_rate": {
            "mean": _mean(repeat_1gram_rates),
            "by_task": {
                task: _mean(payload["repeat_1gram_rate"])
                for task, payload in repetition_by_task.items()
            },
        },
        "repeat_2gram_rate": {
            "mean": _mean(repeat_2gram_rates),
            "by_task": {
                task: _mean(payload["repeat_2gram_rate"])
                for task, payload in repetition_by_task.items()
            },
        },
        "unique_token_fraction": {
            "mean": _mean(unique_token_fractions),
            "by_task": {
                task: _mean(payload["unique_token_fraction"])
                for task, payload in repetition_by_task.items()
            },
        },
        "max_run_length": {
            "mean": _mean(max_run_lengths),
            "by_task": {
                task: _mean(payload["max_run_length"])
                for task, payload in repetition_by_task.items()
            },
        },
    }

    parse_metrics = {
        "parse_success_rate": (
            parse_successes / total_numeric_samples if total_numeric_samples else 0.0
        ),
        "non_numeric_rate": (
            1.0 - (parse_successes / total_numeric_samples)
            if total_numeric_samples
            else 0.0
        ),
        "numeric_abs_error": _mean(numeric_abs_errors),
        "numeric_abs_error_median": _median(numeric_abs_errors),
        "numeric_rel_error": _mean(numeric_rel_errors),
        "numeric_parse_fail_rate": (
            sum(parse_failure_counts.values()) / total_numeric_samples
            if total_numeric_samples
            else 0.0
        ),
        "length_mismatch_rate": (
            numeric_length_mismatch_count / numeric_length_total
            if numeric_length_total
            else 0.0
        ),
        "by_task": {
            task: {
                "parse_success_rate": (
                    parse_successes_by_task.get(task, 0)
                    / (
                        parse_successes_by_task.get(task, 0)
                        + sum(parse_failure_counts_by_task[task].values())
                    )
                    if (parse_successes_by_task.get(task, 0)
                    + sum(parse_failure_counts_by_task[task].values()))
                    else 0.0
                ),
                "non_numeric_rate": (
                    1.0
                    - (
                        parse_successes_by_task.get(task, 0)
                        / (
                            parse_successes_by_task.get(task, 0)
                            + sum(parse_failure_counts_by_task[task].values())
                        )
                    )
                    if (parse_successes_by_task.get(task, 0)
                    + sum(parse_failure_counts_by_task[task].values()))
                    else 0.0
                ),
                "numeric_abs_error": _mean(numeric_abs_errors_by_task.get(task, [])),
                "numeric_abs_error_median": _median(numeric_abs_errors_by_task.get(task, [])),
                "numeric_rel_error": _mean(numeric_rel_errors_by_task.get(task, [])),
                "numeric_parse_fail_rate": (
                    sum(parse_failure_counts_by_task[task].values())
                    / (
                        parse_successes_by_task.get(task, 0)
                        + sum(parse_failure_counts_by_task[task].values())
                    )
                    if (
                        parse_successes_by_task.get(task, 0)
                        + sum(parse_failure_counts_by_task[task].values())
                    )
                    else 0.0
                ),
                "length_mismatch_rate": (
                    numeric_length_mismatch_by_task.get(task, 0)
                    / numeric_length_total_by_task.get(task, 0)
                    if numeric_length_total_by_task.get(task, 0)
                    else 0.0
                ),
            }
            for task in task_counts
        },
        "failure_counts": dict(parse_failure_counts),
        "failure_counts_by_task": {
            task: dict(counts) for task, counts in parse_failure_counts_by_task.items()
        },
    }

    label_metrics = {
        "invalid_label_rate": (
            invalid_label_count / sum(total.values()) if total else 0.0
        ),
        "invalid_label_rate_by_task": {
            task: (invalid_label_by_task.get(task, 0) / task_counts.get(task, 1))
            for task in task_counts
        },
        "confusion_counts": dict(label_confusion_counts),
        "confusion_counts_by_task": {
            task: dict(counts) for task, counts in label_confusion_by_task.items()
        },
    }

    def _format_rates(stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for key, values in stats.items():
            count = values.get("count", 0)
            result[key] = {
                "count": float(count),
                "accuracy": (values.get("correct", 0) / count if count else 0.0),
                "empty_rate": (values.get("empty", 0) / count if count else 0.0),
                "eos_first_rate": (values.get("eos_first", 0) / count if count else 0.0),
            }
        return result

    prompt_format = {
        "overall": _format_rates(format_stats),
        "by_task": {
            task: _format_rates(stats) for task, stats in format_stats_by_task.items()
        },
    }

    soft_hard_gap = {
        "token": {
            "overall": overall_aggregated.mean_token_accuracy - overall,
            "by_task": {
                task: (per_task_token_accuracy[task] - per_task[task])
                for task in per_task
            },
        },
        "prefix": {
            "overall": overall_aggregated.mean_prefix_accuracy - overall,
            "by_task": {
                task: (per_task_prefix_accuracy[task] - per_task[task])
                for task in per_task
            },
        },
    }
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
        "diagnostics": {
            "answer_length": answer_length,
            "repetition": repetition,
            "parse": parse_metrics,
            "labels": label_metrics,
            "prompt_format": prompt_format,
            "soft_hard_gap": soft_hard_gap,
            "task_counts": dict(task_counts),
            "difficulty_by_task": task_difficulty_mean,
        },
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
            target_len = sample.get("target_len_tokens")
            pred_len = sample.get("pred_len_tokens")
            stop_reason = sample.get("stop_reason")
            parse_ok = sample.get("parse_ok")
            repeat_1gram = sample.get("repeat_1gram_rate")
            difficulty = sample.get("difficulty")
            print(
                f"      {idx}. prompt={sample['prompt']!r} "
                f"prediction={sample['prediction']!r} "
                f"expected_output={expected_output!r} "
                f"target_len={target_len} pred_len={pred_len} "
                f"stop_reason={stop_reason} parse_ok={parse_ok} "
                f"repeat_1gram_rate={repeat_1gram} "
                f"difficulty={difficulty}"
            )


def _print_prompt_format_stats(label: str, prompt_format: Dict[str, Any]) -> None:
    if not prompt_format:
        return
    overall = prompt_format.get("overall", {})
    if overall:
        print(f"  {label} prompt format (overall):")
        for key in PROMPT_FORMAT_KEYS:
            stats = overall.get(key)
            if not stats:
                continue
            print(
                f"    {key}: n={int(stats.get('count', 0))} "
                f"acc={stats.get('accuracy', 0.0):.3f} "
                f"empty={stats.get('empty_rate', 0.0):.3f} "
                f"eos_first={stats.get('eos_first_rate', 0.0):.3f}"
            )
    by_task = prompt_format.get("by_task", {})
    if by_task:
        print(f"  {label} prompt format (by task):")
        for task in sorted(by_task):
            task_stats = by_task[task]
            if not task_stats:
                continue
            print(f"    {task}:")
            for key in PROMPT_FORMAT_KEYS:
                stats = task_stats.get(key)
                if not stats:
                    continue
                print(
                    f"      {key}: n={int(stats.get('count', 0))} "
                    f"acc={stats.get('accuracy', 0.0):.3f} "
                    f"empty={stats.get('empty_rate', 0.0):.3f} "
                    f"eos_first={stats.get('eos_first_rate', 0.0):.3f}"
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

    def clamp_difficulty(value: float) -> float:
        """Clamp difficulty to the configured minimum and valid range."""
        return min(max(value, args.min_difficulty), 1.0)

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

    def warn_on_format_crash(
        label: str,
        prev_acc: Dict[str, float],
        prev_token_acc: Dict[str, float],
        current_acc: Dict[str, float],
        current_token_acc: Dict[str, float],
    ) -> None:
        for task, acc in current_acc.items():
            if task not in prev_acc:
                continue
            drop = prev_acc[task] - acc
            token_delta = abs(current_token_acc.get(task, 0.0) - prev_token_acc.get(task, 0.0))
            if drop >= crash_drop_threshold and token_delta <= crash_token_stability_threshold:
                log_print(
                    f"[warn] {label} {task} exact-match dropped by {drop:.3f} "
                    f"with token accuracy stable (Δ={token_delta:.3f}). "
                    "Possible format/strictness issue."
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
        pad_token_id=tokenizer.pad_token_id,
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
        if not args.compile_dynamic:
            log_print(
                "Warning: dynamic padding enables variable sequence lengths; "
                "torch.compile may recompile on shape changes. Consider "
                "--compile_dynamic or disabling --compile if you see stalls."
            )
        # 'reduce-overhead' is excellent for smaller models or CPU training
        # If this errors on your setup, try mode='default'
        compile_kwargs = {"mode": "reduce-overhead"}
        if args.compile_dynamic:
            compile_kwargs["dynamic"] = True
        try:
            model = torch.compile(model, **compile_kwargs)
        except TypeError:
            compile_kwargs.pop("dynamic", None)
            model = torch.compile(model, **compile_kwargs)

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

    if args.disable_dynamic_task_weighting:
        if args.task_weighting != "uniform":
            log_print("Dynamic task weighting disabled; using uniform task sampling.")
        args.task_weighting = "uniform"

    task_curriculum = None
    difficulty_fn = None
    eval_difficulty_fn = None
    weighting_fn = None
    weighting_adjust_fn = None
    dag_state = None
    if args.use_task_curriculum:
        difficulty_value = None
    elif not args.fixed_data and args.difficulty_schedule in {"linear", "phased", "fixed"}:
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

    if args.use_task_curriculum:
        manager = None if args.num_workers == 0 else Manager()
        curriculum_tasks = alg_tasks or sorted(available_tasks)
        target_reaction_steps = 2000.0
        update_interval = float(args.eval_interval)
        calculated_decay = 1.0 - (update_interval / target_reaction_steps)
        ema_decay = min(max(calculated_decay, 0.0), 0.99)
        log_print(
            "Auto-tuned Curriculum | Update Interval: %.1f | Target Reaction: %.1f | Calculated Decay: %.4f"
            % (update_interval, target_reaction_steps, ema_decay)
        )
        task_curriculum = TaskCurriculumState(
            manager,
            tasks=curriculum_tasks,
            min_difficulty=args.min_difficulty,
            ema_decay=ema_decay,
            min_task_evals=args.curriculum_min_task_evals,
        )

        dag_gate_snapshot = _create_shared_dict(manager)
        dag_frontier_snapshot = _create_shared_list(manager)
        dag_replay_ratio_snapshot = _create_shared_list(manager, [0.0])
        dag_unlocked_snapshot: Set[str] = set()
        dag_prob_snapshot: Dict[str, float] = {}
        dag_gated_weight_snapshot: Dict[str, float] = {}
        dag_ema_snapshot: Dict[str, float] = {}

        if args.task_curriculum_strategy == "dag":
            dag_roots = ["copy", "reverse", "compare"]
            dag_prereqs = {
                "successor": ["copy", "reverse", "compare"],
                "parity": ["copy", "reverse"],
                "chain": ["successor"],
                "addition": ["chain", "parity"],
                "mod_add": ["addition"],
                "multiplication": ["addition", "reverse"],
                "dyck": ["multiplication"],
            }
            dag_thresholds = {
                "successor": {"copy": 0.99, "reverse": 0.98, "compare": 0.98},
                "parity": {"copy": 0.99, "reverse": 0.98},
                "chain": {"successor": 0.99},
                "addition": {"chain": 0.95, "parity": 0.95},
                "mod_add": {"addition": 0.97},
                "multiplication": {"addition": 0.97, "reverse": 0.98},
                "dyck": {"multiplication": 0.97},
            }
            dag_state = DagUnlockState(
                roots=dag_roots,
                prereqs=dag_prereqs,
                thresholds=dag_thresholds,
                patience_evals=args.dag_patience_evals,
                ramp_evals=args.dag_ramp_evals,
                replay_ratio=args.dag_replay_ratio,
                replay_ratio_backslide=args.dag_replay_ratio_backslide,
                unlock_margin=args.dag_unlock_margin,
                lock_margin=args.dag_lock_margin,
                frontier_recent_evals=args.dag_frontier_recent_evals,
                mastery_margin=args.dag_mastery_margin,
            )

        difficulty_snapshot = _create_shared_dict(manager)
        weights_snapshot = _create_shared_dict(manager)

        def refresh_curriculum_snapshots() -> None:
            nonlocal difficulty_snapshot, weights_snapshot
            difficulty_snapshot.clear()
            difficulty_snapshot.update(task_curriculum.get_difficulty_snapshot())
            weights_snapshot.clear()
            weights_snapshot.update(task_curriculum.get_sampling_weights_snapshot())

        refresh_curriculum_snapshots()
        if dag_state is not None:
            dag_gate_snapshot.clear()
            dag_gate_snapshot.update(dag_state.get_gate_snapshot())
            _set_frontier_snapshot(dag_frontier_snapshot, dag_state.compute_frontier({}))
            dag_replay_ratio_snapshot[0] = dag_state.compute_replay_ratio({})
            dag_unlocked_snapshot.clear()
            dag_unlocked_snapshot.update(
                {task for task, gate in dag_gate_snapshot.items() if gate > 0.0}
            )

        difficulty_fn = CurriculumDifficultySampler(
            difficulty_snapshot,
            easy_mix_frac=args.easy_mix_frac,
            curriculum_jitter=args.curriculum_jitter,
            min_difficulty=args.min_difficulty,
        )
        eval_difficulty_fn = CurriculumEvalDifficultySampler(
            difficulty_snapshot,
            min_difficulty=args.min_difficulty,
        )

        min_task_prob = 0.05

        if not args.disable_dynamic_task_weighting:
            weighting_fn = CurriculumWeightingSampler(weights_snapshot)
            weighting_adjust_fn = CurriculumWeightingAdjuster(
                difficulty_snapshot,
                min_task_prob=min_task_prob,
            )
        elif dag_state is not None:
            weighting_fn = _uniform_weighting

    if dag_state is not None:
        weighting_adjust_fn = DagWeightingAdjuster(
            dag_gate_snapshot,
            dag_frontier_snapshot,
            dag_replay_ratio_snapshot,
            args.dag_locked_floor,
        )

    collate_fn = make_collate_fn(tokenizer)

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
            min_difficulty=args.min_difficulty,
            include_lengths=True,
        )
        alg_loader = DataLoader(
            alg_dataset,
            batch_size=args.alg_batch_size,
            shuffle=True,  # reshuffle fixed set each epoch to avoid ordering artifacts
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        alg_dataset = AlgorithmicDataset(
            tokenizer=tokenizer,
            num_examples=args.alg_examples,
            max_seq_len=args.alg_seq_len,
            base_batch_size=args.alg_batch_size,
            tasks=alg_tasks,
            seed=args.seed,
            difficulty_value=difficulty_value,
            difficulty_fn=difficulty_fn,
            weighting_fn=weighting_fn,
            weighting_adjust_fn=weighting_adjust_fn,
            difficulty_schedule=args.difficulty_schedule,
            task_weighting=args.task_weighting,
            total_tokens=args.alg_tokens,
            easy_mix_frac=args.easy_mix_frac,
            min_difficulty=args.min_difficulty,
            include_lengths=True,
        )
        alg_loader = DataLoader(
            alg_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
        )
    else:
        lang_loader = alg_loader  # Fallback

    # Training-eval loader (for grokking memorization tracking)
    if args.train_eval_mode == "auto":
        if args.use_task_curriculum:
            train_eval_mode = "procedural"
        else:
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
                min_difficulty=args.min_difficulty,
                include_lengths=True,
            )
            if args.train_eval_samples is not None:
                capped = min(args.train_eval_samples, len(base_dataset))
                base_dataset = Subset(base_dataset, range(capped))
        else:
            base_dataset = AlgorithmicDataset(
                tokenizer=tokenizer,
                num_examples=args.train_eval_samples or args.eval_samples,
                max_seq_len=args.alg_seq_len,
                base_batch_size=args.train_eval_batch_size,
                tasks=alg_tasks,
                seed=args.seed + 123,
                difficulty_value=None,
                difficulty_fn=eval_difficulty_fn,
                difficulty_schedule=args.difficulty_schedule,
                task_weighting=args.task_weighting,
                total_tokens=args.alg_tokens,
                easy_mix_frac=args.easy_mix_frac,
                min_difficulty=args.min_difficulty,
                include_lengths=True,
            )

        train_eval_loader = DataLoader(
            base_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
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
        min_difficulty=args.min_difficulty,
        include_lengths=True,
    )
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.alg_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
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
    log_print(
        f"  Task curriculum: {args.use_task_curriculum}"
        + (
            f" (cooldown={args.curriculum_cooldown}, jitter={args.curriculum_jitter})"
            if args.use_task_curriculum
            else ""
        )
    )
    
    global_step = 0
    tokens_seen = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = tokens_seen
    best_alg_acc = 0.0
    last_grad_norm = 0.0
    checkpoint_paths: List[Path] = []
    prev_eval_task_acc: Dict[str, float] = {}
    prev_eval_task_token_acc: Dict[str, float] = {}
    prev_eval_ood_task_acc: Dict[str, float] = {}
    prev_eval_ood_task_token_acc: Dict[str, float] = {}
    prev_train_eval_task_acc: Dict[str, float] = {}
    prev_train_eval_task_token_acc: Dict[str, float] = {}
    crash_drop_threshold = 0.2
    crash_token_stability_threshold = 0.05

    def _task_curriculum_config() -> Optional[Dict[str, float]]:
        if task_curriculum is None:
            return None
        return {
            "init_difficulty": float(task_curriculum.init_difficulty),
            "min_difficulty": float(task_curriculum.min_difficulty),
            "ema_decay": float(task_curriculum.ema_decay),
            "step_size": float(task_curriculum.step_size),
            "min_task_evals": float(task_curriculum.min_task_evals),
        }

    def _curriculum_config() -> Dict[str, float]:
        return {
            "alg_tokens": int(args.alg_tokens),
            "total_tokens": int(args.total_tokens),
            "mix_band_tokens": int(args.mix_band_tokens),
            "persistent_alg_frac": float(args.persistent_alg_frac),
            "lexical_frac_phase1": float(args.lexical_frac_phase1),
            "difficulty_schedule": str(args.difficulty_schedule),
            "min_difficulty": float(args.min_difficulty),
            "use_task_curriculum": bool(args.use_task_curriculum),
            "curriculum_cooldown": int(args.curriculum_cooldown),
            "curriculum_jitter": float(args.curriculum_jitter),
        }

    def build_checkpoint_payload(step: int, tokens_seen: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_alg_acc": best_alg_acc,
            "curriculum_tokens_seen": int(curriculum.tokens_seen),
            "curriculum_rng_state": curriculum.rng.getstate(),
            "difficulty_value": (
                float(difficulty_value.value) if difficulty_value is not None else None
            ),
            "curriculum_config": _curriculum_config(),
            "task_curriculum_config": _task_curriculum_config(),
        }
        if task_curriculum is not None:
            payload["curriculum_state"] = task_curriculum.state_dict()
        if dag_state is not None:
            payload["dag_state"] = dag_state.state_dict()
        return payload

    def save_rotating_checkpoint(step: int, tokens_seen: int, tag: Optional[str] = None) -> Optional[Path]:
        if args.eval_checkpoints <= 0:
            return None
        suffix = f"_{tag}" if tag else ""
        checkpoint_path = output_dir / f"checkpoint_step_{step}{suffix}.pt"
        payload = build_checkpoint_payload(step, tokens_seen)
        torch.save(payload, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)
        while len(checkpoint_paths) > args.eval_checkpoints:
            old_checkpoint = checkpoint_paths.pop(0)
            try:
                old_checkpoint.unlink()
            except FileNotFoundError:
                pass
        return checkpoint_path

    def _apply_task_curriculum_config(config_payload: Dict[str, float]) -> None:
        if task_curriculum is None or not config_payload:
            return
        if "init_difficulty" in config_payload:
            task_curriculum.init_difficulty = float(config_payload["init_difficulty"])
        if "min_difficulty" in config_payload:
            task_curriculum.min_difficulty = float(config_payload["min_difficulty"])
        if "ema_decay" in config_payload:
            task_curriculum.ema_decay = float(config_payload["ema_decay"])
        if "step_size" in config_payload:
            task_curriculum.step_size = float(config_payload["step_size"])
        if "min_task_evals" in config_payload:
            task_curriculum.min_task_evals = int(config_payload["min_task_evals"])
        elif "warmup_evals" in config_payload:
            task_curriculum.min_task_evals = int(config_payload["warmup_evals"])
        task_curriculum.init_difficulty = max(
            task_curriculum.init_difficulty, task_curriculum.min_difficulty
        )

    def load_checkpoint_state(checkpoint_path: str) -> Tuple[int, int, float]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        if task_curriculum is not None and checkpoint.get("curriculum_state"):
            task_curriculum.load_state_dict(checkpoint["curriculum_state"])
        if checkpoint.get("task_curriculum_config"):
            _apply_task_curriculum_config(checkpoint["task_curriculum_config"])
        if dag_state is not None and checkpoint.get("dag_state"):
            dag_state.load_state_dict(checkpoint["dag_state"])
        if checkpoint.get("curriculum_rng_state") is not None:
            curriculum.rng.setstate(checkpoint["curriculum_rng_state"])
        resume_tokens = int(checkpoint.get("tokens_seen", 0))
        curriculum_tokens = int(checkpoint.get("curriculum_tokens_seen", resume_tokens))
        curriculum.tokens_seen = curriculum_tokens
        if difficulty_value is not None and checkpoint.get("difficulty_value") is not None:
            difficulty_value.value = float(checkpoint["difficulty_value"])
        resume_step = int(checkpoint.get("step", 0))
        resume_best_alg = float(checkpoint.get("best_alg_acc", 0.0))
        return resume_step, curriculum_tokens, resume_best_alg

    if args.resume_from:
        resume_step, resume_tokens, resume_best_alg = load_checkpoint_state(args.resume_from)
        global_step = resume_step
        tokens_seen = resume_tokens
        best_alg_acc = resume_best_alg
        log_print(
            f"Resumed from checkpoint: {args.resume_from} "
            f"(step={global_step}, tokens_seen={tokens_seen}, "
            f"best_alg_acc={best_alg_acc:.4f})"
        )
        last_log_time = time.time()
        last_log_tokens = tokens_seen
    
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
    last_token_accuracy = float("nan")
    last_pct_masked_tokens = float("nan")
    flush_diagnostic_metrics: Optional[Dict[str, float]] = None
    lr_mismatch_warned = False
    task_window_counts = defaultdict(int)
    task_window_input_len = defaultdict(int)
    task_window_target_len = defaultdict(int)

    def _current_difficulty() -> float:
        if task_curriculum is not None:
            active = alg_tasks or list(AlgorithmicGenerator._get_generators().keys())
            return clamp_difficulty(task_curriculum.get_mean_difficulty(active))
        if difficulty_value is not None:
            return clamp_difficulty(float(difficulty_value.value))
        return clamp_difficulty(
            difficulty_schedule(
                curriculum.tokens_seen, args.alg_tokens, args.difficulty_schedule
            )
        )

    def _difficulty_by_task() -> Dict[str, float]:
        active = alg_tasks or list(AlgorithmicGenerator._get_generators().keys())
        if task_curriculum is not None:
            return {
                task: clamp_difficulty(task_curriculum.get_task_state(task)["difficulty"])
                for task in active
            }
        if difficulty_value is not None:
            difficulty = clamp_difficulty(float(difficulty_value.value))
        else:
            difficulty = clamp_difficulty(
                difficulty_schedule(
                    curriculum.tokens_seen, args.alg_tokens, args.difficulty_schedule
                )
            )
        return {task: difficulty for task in active}

    def _format_difficulty_by_task(difficulty_by_task: Dict[str, float]) -> str:
        if not difficulty_by_task:
            return "n/a"
        return ", ".join(
            f"{task}={value:.3f}"
            for task, value in sorted(difficulty_by_task.items())
        )
    
    try:
        while curriculum.tokens_seen < args.total_tokens:
            if difficulty_value is not None:
                if curriculum.tokens_seen < args.alg_tokens:
                    difficulty_value.value = clamp_difficulty(
                        difficulty_schedule(
                            curriculum.tokens_seen,
                            args.alg_tokens,
                            args.difficulty_schedule,
                        )
                    )
                else:
                    difficulty_value.value = clamp_difficulty(1.0)
            batch = curriculum.next_batch()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tokens_seen = curriculum.tokens_seen

            batch_tasks = batch.get("task")
            if batch_tasks is not None:
                if torch.is_tensor(batch_tasks):
                    task_list = batch_tasks.tolist()
                elif isinstance(batch_tasks, (list, tuple)):
                    task_list = list(batch_tasks)
                else:
                    task_list = [batch_tasks]
                input_lens = batch.get("input_len_tokens")
                target_lens = batch.get("target_len_tokens")
                if torch.is_tensor(input_lens):
                    input_lens = input_lens.tolist()
                if torch.is_tensor(target_lens):
                    target_lens = target_lens.tolist()
                for idx, task in enumerate(task_list):
                    task_window_counts[task] += 1
                    if isinstance(input_lens, list) and idx < len(input_lens):
                        task_window_input_len[task] += int(input_lens[idx])
                    if isinstance(target_lens, list) and idx < len(target_lens):
                        task_window_target_len[task] += int(target_lens[idx])
            
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

            with torch.no_grad():
                if labels.size(1) > 1:
                    shift_labels = labels[..., 1:]
                    shift_logits = logits[..., :-1, :]
                    mask = shift_labels != -100
                    denom = mask.sum().item()
                    if denom > 0:
                        predictions = shift_logits.argmax(dim=-1)
                        correct = (predictions[mask] == shift_labels[mask]).sum().item()
                        last_token_accuracy = correct / denom
                        last_pct_masked_tokens = 1.0 - (denom / mask.numel())
                    else:
                        last_token_accuracy = float("nan")
                        last_pct_masked_tokens = float("nan")
                else:
                    last_token_accuracy = float("nan")
                    last_pct_masked_tokens = float("nan")
            
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
                        flush_diagnostic_metrics[f"train/grad/norm/{key}"] = mod_grad_norm
                        flush_diagnostic_metrics[f"train/weight/norm/{key}"] = mod_weight_norm
                        if not math.isnan(mod_grad_norm) and not math.isnan(mod_weight_norm):
                            flush_diagnostic_metrics[f"train/grad/ratio/{key}"] = (
                                mod_grad_norm / (mod_weight_norm + 1e-12)
                            )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                last_grad_norm = float(grad_norm)
            
            global_step += 1
            
            # Logging
            if global_step % args.log_interval == 0:
                elapsed = time.time() - last_log_time
                tokens_since_log = tokens_seen - last_log_tokens
                toks_per_sec = tokens_since_log / max(elapsed, 1e-6)
                peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                avg_loss = running_loss / args.log_interval
                avg_K = running_inner_steps / args.log_interval if running_inner_steps > 0 else 0.0
                avg_ponder = running_ponder / args.log_interval if running_ponder > 0 else 0.0

                phase = curriculum.phase
                progress = curriculum.progress * 100

                with torch.no_grad():
                    params = [
                        param.detach().view(-1)
                        for param in model.parameters()
                        if param.numel() > 0
                    ]
                    if params:
                        all_params = torch.cat(params)
                        weight_norm = all_params.norm(2.0, dtype=torch.float32).item()
                    else:
                        weight_norm = float("nan")
                effective_wd_pressure = args.weight_decay * weight_norm

                difficulty_logged = _current_difficulty()

                expected_lr = get_lr(
                    tokens_seen,
                    warmup_tokens,
                    args.total_tokens,
                    args.max_lr,
                    args.min_lr,
                    schedule=args.lr_schedule,
                )
                lr_actuals = [pg.get("lr", float("nan")) for pg in optimizer.param_groups]
                lr_actuals_clean = [value for value in lr_actuals if not math.isnan(value)]
                lr_actual = (
                    sum(lr_actuals_clean) / len(lr_actuals_clean)
                    if lr_actuals_clean
                    else float("nan")
                )
                lr_actual_min = (
                    min(lr_actuals_clean) if lr_actuals_clean else float("nan")
                )
                lr_actual_max = (
                    max(lr_actuals_clean) if lr_actuals_clean else float("nan")
                )
                lr_for_print = lr_actual if not math.isnan(lr_actual) else expected_lr
                base_log_line = (
                    f"Loss: {avg_loss:.4f} | LR: {lr_for_print:.2e} "
                    f"(expected {expected_lr:.2e}) | Tok/s: {toks_per_sec:.0f}"
                )
                if args.extended_logging:
                    log_print(
                        f"Step {global_step:>6} | "
                        f"Phase: {phase[:4]:>4} | "
                        f"Progress: {progress:>5.1f}% | "
                        f"{base_log_line} | "
                        f"VRAM: {peak_vram:.1f}GB"
                        + (f" | K: {avg_K:.1f}" if avg_K > 0 else "")
                        + (f" | ponder: {avg_ponder:.3f}" if avg_ponder > 0 else "")
                    )
                else:
                    print(base_log_line)
                if (
                    not lr_mismatch_warned
                    and not math.isnan(expected_lr)
                    and not math.isnan(lr_actual)
                    and abs(expected_lr - lr_actual)
                    / max(abs(expected_lr), abs(lr_actual), 1e-12)
                    > 1e-3
                ):
                    print(
                        "[warn] expected LR differs from optimizer LR "
                        f"(expected={expected_lr:.3e}, actual={lr_actual:.3e})."
                    )
                    lr_mismatch_warned = True

                if args.extended_logging:
                    difficulty_by_task = _difficulty_by_task()
                    log_print(
                        "  Difficulty by task: "
                        f"{_format_difficulty_by_task(difficulty_by_task)} "
                        f"(easy_mix: {args.easy_mix_frac})"
                    )
                    log_print(f"  Grad norm: {last_grad_norm:.1f}")
                    log_print(f"  Weight norm: {weight_norm:.1f}")
                    if args.weight_decay > 0.0:
                        log_print(f"  WD pressure: {effective_wd_pressure:.1f}")

                # Update-scale proxies (cheap, no extra parameter copies).
                # update_norm_est ~= ||lr * grad||, update_to_weight ~= ||lr * grad|| / ||w||.
                if math.isnan(last_grad_norm) or math.isnan(weight_norm):
                    update_norm_est_preclip = float("nan")
                    update_norm_est_postclip = float("nan")
                    update_to_weight_preclip = float("nan")
                    update_to_weight_postclip = float("nan")
                    grad_to_weight_global = float("nan")
                    update_to_weight_global_postclip = float("nan")
                else:
                    update_norm_est_preclip = lr_actual * last_grad_norm
                    update_norm_est_postclip = lr_actual * min(last_grad_norm, args.grad_clip)
                    update_to_weight_preclip = update_norm_est_preclip / (weight_norm + 1e-12)
                    update_to_weight_postclip = update_norm_est_postclip / (weight_norm + 1e-12)
                    grad_to_weight_global = last_grad_norm / (weight_norm + 1e-12)
                    update_to_weight_global_postclip = (
                        lr_actual
                        * min(last_grad_norm, args.grad_clip)
                        / (weight_norm + 1e-12)
                    )

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
                        diagnostic_metrics[f"train/grad/norm/{key}"] = mod_grad_norm
                        diagnostic_metrics[f"train/weight/norm/{key}"] = mod_weight_norm
                        if not math.isnan(mod_grad_norm) and not math.isnan(mod_weight_norm):
                            diagnostic_metrics[f"train/grad/ratio/{key}"] = (
                                mod_grad_norm / (mod_weight_norm + 1e-12)
                            )

                # Per-module update proxies for diagnostic modules.
                for key, mod_grad_norm in list(diagnostic_metrics.items()):
                    if not key.startswith("train/grad/norm/"):
                        continue
                    if math.isnan(mod_grad_norm):
                        continue
                    module_key = key.split("train/grad/norm/", 1)[1]
                    mod_weight_norm = diagnostic_metrics.get(
                        f"train/weight/norm/{module_key}"
                    )
                    if mod_weight_norm is None or math.isnan(mod_weight_norm):
                        continue
                    diagnostic_metrics[f"train/update/norm/{module_key}"] = (
                        lr_actual * mod_grad_norm
                    )
                    diagnostic_metrics[f"train/update/ratio/{module_key}"] = (
                        lr_actual * mod_grad_norm / (mod_weight_norm + 1e-12)
                    )

                if math.isnan(last_grad_norm):
                    diagnostic_metrics["train/grad/clipped"] = float("nan")
                    diagnostic_metrics["train/grad/clip_ratio"] = float("nan")
                else:
                    grad_clipped = 1.0 if last_grad_norm > args.grad_clip else 0.0
                    diagnostic_metrics["train/grad/clipped"] = grad_clipped
                    diagnostic_metrics["train/grad/clip_ratio"] = (
                        args.grad_clip / (last_grad_norm + 1e-12) if grad_clipped else 1.0
                    )

                lr_actual_metrics: Dict[str, float] = {}
                if not math.isnan(lr_actual_min) and not math.isnan(lr_actual_max):
                    lr_actual_metrics["train/optim/lr_actual_min"] = lr_actual_min
                    lr_actual_metrics["train/optim/lr_actual_max"] = lr_actual_max

                task_metrics: Dict[str, float] = {}
                active_tasks = alg_tasks or list(AlgorithmicGenerator._get_generators().keys())
                progress_ratio = min(curriculum.tokens_seen / max(args.alg_tokens, 1), 1.0)
                if dag_state is not None:
                    if args.disable_dynamic_task_weighting:
                        base_weights = [1.0] * len(active_tasks)
                    else:
                        base_weights = [
                            weights_snapshot.get(task, 1.0) for task in active_tasks
                        ]
                    final_weights, final_probs, gated_weights = compute_dag_weighting(
                        active_tasks,
                        base_weights,
                        dag_gate_snapshot,
                        dag_frontier_snapshot,
                        dag_replay_ratio_snapshot[0],
                        args.dag_locked_floor,
                    )
                    dag_prob_snapshot = dict(final_probs)
                    dag_gated_weight_snapshot = dict(gated_weights)
                    for task, prob in final_probs.items():
                        task_metrics[f"train/curriculum/prob/{task}"] = prob
                    for task, weight in gated_weights.items():
                        task_metrics[f"train/curriculum/weight/{task}"] = weight
                    for task, gate in dag_gate_snapshot.items():
                        task_metrics[f"train/curriculum/gate/{task}"] = gate
                else:
                    task_weights, task_probs = _compute_task_sampling(
                        active_tasks, progress_ratio, args.task_weighting
                    )
                    for task, prob in task_probs.items():
                        task_metrics[f"train/curriculum/prob/{task}"] = prob
                    for task, weight in task_weights.items():
                        task_metrics[f"train/curriculum/weight/{task}"] = weight
                for task, count in task_window_counts.items():
                    task_metrics[f"train/throughput/task_count_window/{task}"] = float(
                        count
                    )
                    if count > 0:
                        task_metrics[f"train/throughput/task_input_len/{task}"] = (
                            task_window_input_len[task] / count
                        )
                        task_metrics[f"train/throughput/task_target_len/{task}"] = (
                            task_window_target_len[task] / count
                        )

                logger.log(
                    step=global_step,
                    phase=phase,
                    **{
                        "train/loss/ce": avg_loss,
                        "train/throughput/tokens_seen": tokens_seen,
                        "train/throughput/tokens_per_sec": toks_per_sec,
                        "train/throughput/vram_gb": peak_vram,
                        "train/act/inner_steps": avg_K,
                        "train/loss/ponder": avg_ponder,
                        "train/curriculum/difficulty": difficulty_logged,
                        "train/acc/token": last_token_accuracy,
                        "train/acc/masked_token_rate": last_pct_masked_tokens,
                        "train/grad/norm/global": last_grad_norm,
                        "train/weight/norm/global": weight_norm,
                        "train/optim/lr_expected": expected_lr,
                        "train/optim/lr_actual": lr_actual,
                        "train/update/norm/preclip": update_norm_est_preclip,
                        "train/update/norm/postclip": update_norm_est_postclip,
                        "train/update/ratio/preclip": update_to_weight_preclip,
                        "train/update/ratio/postclip": update_to_weight_postclip,
                        "train/grad/ratio/global": grad_to_weight_global,
                        "train/update/ratio/postclip_global": update_to_weight_global_postclip,
                    },
                    **diagnostic_metrics,
                    **lr_actual_metrics,
                    **task_metrics,
                )
                logger.save()
                if args.extended_logging and diagnostic_metrics:
                    log_print(f"  Diagnostics: {format_metrics_line(diagnostic_metrics)}")

                running_loss = 0.0
                running_inner_steps = 0.0
                running_ponder = 0.0
                task_window_counts.clear()
                task_window_input_len.clear()
                task_window_target_len.clear()
                last_log_time = time.time()
                last_log_tokens = tokens_seen
            
            # Evaluation
            if global_step % args.eval_interval == 0:
                eval_start = time.time()
                if args.extended_logging:
                    log_print("\nRunning evaluation...")
                iid_bumped_tasks: Set[str] = set()
                difficulty_logged = _current_difficulty()
                if args.extended_logging:
                    difficulty_by_task = _difficulty_by_task()
                    log_print(
                        "  Eval difficulty by task: "
                        f"{_format_difficulty_by_task(difficulty_by_task)}"
                    )
                    log_print(
                        f"  Train token acc: {last_token_accuracy:.3f} | "
                        f"pct masked: {last_pct_masked_tokens:.3f}"
                    )
                active_tasks = alg_tasks or list(AlgorithmicGenerator._get_generators().keys())
                progress_ratio = min(curriculum.tokens_seen / max(args.alg_tokens, 1), 1.0)
                if dag_state is not None:
                    if args.disable_dynamic_task_weighting:
                        base_weights = [1.0] * len(active_tasks)
                    else:
                        base_weights = [
                            weights_snapshot.get(task, 1.0) for task in active_tasks
                        ]
                    _, eval_task_probs, eval_gated_weights = compute_dag_weighting(
                        active_tasks,
                        base_weights,
                        dag_gate_snapshot,
                        dag_frontier_snapshot,
                        dag_replay_ratio_snapshot[0],
                        args.dag_locked_floor,
                    )
                    eval_task_metrics = {
                        f"eval/curriculum/prob/{task}": prob
                        for task, prob in eval_task_probs.items()
                    }
                    eval_task_metrics.update(
                        {
                            f"eval/curriculum/weight/{task}": weight
                            for task, weight in eval_gated_weights.items()
                        }
                    )
                    eval_task_metrics.update(
                        {
                            f"eval/curriculum/gate/{task}": gate
                            for task, gate in dag_gate_snapshot.items()
                        }
                    )
                else:
                    eval_task_weights, eval_task_probs = _compute_task_sampling(
                        active_tasks, progress_ratio, args.task_weighting
                    )
                    eval_task_metrics = {
                        f"eval/curriculum/prob/{task}": prob
                        for task, prob in eval_task_probs.items()
                    }
                    eval_task_metrics.update(
                        {
                            f"eval/curriculum/weight/{task}": weight
                            for task, weight in eval_task_weights.items()
                        }
                    )
                with torch.no_grad():
                    params = [
                        param.detach().view(-1)
                        for param in model.parameters()
                        if param.numel() > 0
                    ]
                    if params:
                        all_params = torch.cat(params)
                        weight_norm = all_params.norm(2.0, dtype=torch.float32).item()
                    else:
                        weight_norm = float("nan")
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
                    logger.log(
                        step=global_step,
                        phase="eval",
                        **eval_diagnostic_metrics,
                        **eval_task_metrics,
                    )
                    if args.extended_logging:
                        log_print(
                            "  Eval diagnostics: "
                            f"{format_metrics_line(eval_diagnostic_metrics)}"
                        )

                # Algorithmic accuracy (IID grid)
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
                        grid="iid",
                    )
                    overall_acc = alg_results.get("overall_accuracy", 0.0)
                    overall_mae = alg_results.get("overall_mae")
                    overall_token_accuracy = alg_results.get("overall_token_accuracy", 0.0)
                    overall_distance = alg_results.get("overall_distance", 1.0)
                    overall_prefix_accuracy = alg_results.get("overall_prefix_accuracy", 0.0)
                    if args.extended_logging:
                        log_print(
                            "  Algorithmic IID metrics: "
                            f"acc={overall_acc*100:.1f}% | "
                            f"token={overall_token_accuracy:.3f} | "
                            f"dist={overall_distance:.3f} | "
                            f"prefix={overall_prefix_accuracy:.3f}"
                        )
                        diagnostics = alg_results.get("diagnostics", {})
                        answer_diag = diagnostics.get("answer_length", {})
                        labels_diag = diagnostics.get("labels", {})
                        parse_diag = diagnostics.get("parse", {})
                        log_print(
                            "  Algorithmic IID diagnostics: "
                            f"empty={answer_diag.get('empty_rate', 0.0):.3f} | "
                            f"eos_first={answer_diag.get('eos_first_rate', 0.0):.3f} | "
                            f"invalid_label={labels_diag.get('invalid_label_rate', 0.0):.3f} | "
                            f"numeric_parse_fail={parse_diag.get('numeric_parse_fail_rate', 0.0):.3f}"
                        )
                        if overall_mae is not None:
                            log_print(f"  Algorithmic MAE (numeric tasks): {overall_mae:.3f}")

                        _print_sampled_examples(
                            "Algorithmic IID eval",
                            alg_results.get("sampled_examples_by_task", {}),
                        )
                        if args.debug_eval_format_stats:
                            _print_prompt_format_stats(
                                "Algorithmic IID eval",
                                diagnostics.get("prompt_format", {}),
                            )

                    flat_eval_diagnostics = _flatten_eval_diagnostics(
                        "eval", alg_results.get("diagnostics", {})
                    )
                    if not any(key.startswith("eval/") for key in flat_eval_diagnostics):
                        print(
                            "[warn] eval diagnostics are empty; check diagnostic wiring for eval."
                        )

                    per_task_mae = alg_results.get("per_task_mae", {}) or {}
                    per_task_token_accuracy = alg_results.get("per_task_token_accuracy", {}) or {}
                    per_task_distance = alg_results.get("per_task_distance", {}) or {}
                    per_task_prefix_accuracy = (
                        alg_results.get("per_task_prefix_accuracy", {}) or {}
                    )
                    per_task_accuracy = alg_results.get("per_task_accuracy", {}) or {}

                    flat_per_task_accuracy = {
                        f"eval/acc/exact_match/{task}": acc
                        for task, acc in per_task_accuracy.items()
                    }
                    flat_per_task_token_accuracy = {
                        f"eval/acc/token/{task}": acc
                        for task, acc in per_task_token_accuracy.items()
                    }
                    flat_per_task_prefix_accuracy = {
                        f"eval/acc/prefix/{task}": acc
                        for task, acc in per_task_prefix_accuracy.items()
                    }
                    flat_per_task_distance = {
                        f"eval/acc/distance/{task}": dist
                        for task, dist in per_task_distance.items()
                    }
                    flat_per_task_mae = {
                        f"eval/numeric/mae/{task}": mae
                        for task, mae in per_task_mae.items()
                        if mae is not None
                    }

                    logger.log(
                        step=global_step,
                        phase="eval",
                        **{
                            "eval/acc/exact_match": overall_acc,
                            "eval/acc/token": overall_token_accuracy,
                            "eval/acc/distance": overall_distance,
                            "eval/acc/prefix": overall_prefix_accuracy,
                            "eval/numeric/mae": overall_mae,
                            "eval/curriculum/difficulty": difficulty_logged,
                            "eval/samples": alg_results.get("sampled_examples_by_task", {}),
                        },
                        **flat_per_task_accuracy,
                        **flat_per_task_token_accuracy,
                        **flat_per_task_prefix_accuracy,
                        **flat_per_task_distance,
                        **flat_per_task_mae,
                        **flat_eval_diagnostics,
                        **eval_task_metrics,
                    )
                    if per_task_accuracy:
                        warn_on_format_crash(
                            "Algorithmic IID",
                            prev_eval_task_acc,
                            prev_eval_task_token_acc,
                            per_task_accuracy,
                            per_task_token_accuracy,
                        )
                        prev_eval_task_acc = dict(per_task_accuracy)
                        prev_eval_task_token_acc = dict(per_task_token_accuracy)
                    for task, acc in per_task_accuracy.items():
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
                            answer_diag = alg_results.get("diagnostics", {}).get(
                                "answer_length", {}
                            )
                            labels_diag = alg_results.get("diagnostics", {}).get("labels", {})
                            parse_diag = alg_results.get("diagnostics", {}).get("parse", {})
                            empty_rate = answer_diag.get("empty_rate_by_task", {}).get(task)
                            eos_first_rate = answer_diag.get("eos_first_rate_by_task", {}).get(task)
                            max_new_tokens_rate = answer_diag.get(
                                "max_new_tokens_rate_by_task", {}
                            ).get(task)
                            if empty_rate is not None:
                                line += f" | empty={empty_rate:.3f}"
                            if eos_first_rate is not None:
                                line += f" | eos_first={eos_first_rate:.3f}"
                            if max_new_tokens_rate is not None:
                                line += f" | max_new={max_new_tokens_rate:.3f}"
                            if task in CLASSIFICATION_TASKS:
                                invalid_rate = labels_diag.get(
                                    "invalid_label_rate_by_task", {}
                                ).get(task)
                                if invalid_rate is not None:
                                    line += f" | invalid_label={invalid_rate:.3f}"
                            elif task in NUMERIC_OUTPUT_TASKS:
                                parse_fail_rate = parse_diag.get(
                                    "by_task", {}
                                ).get(task, {}).get("numeric_parse_fail_rate")
                                if parse_fail_rate is not None:
                                    line += f" | parse_fail={parse_fail_rate:.3f}"
                            if task_mae is not None:
                                line += f" | MAE: {task_mae:.3f}"
                            log_print(line)

                    if task_curriculum is not None and per_task_accuracy:
                        did_sync = False
                        for task, acc in per_task_accuracy.items():
                            state = task_curriculum.get_task_state(task)
                            if acc > 0.85 and state["difficulty"] < 0.40:
                                log_print(
                                    f"  Syncing curriculum for {task}: Boosting difficulty to 0.5"
                                )
                                task_curriculum.override_difficulty(task, 0.5)
                                task_curriculum.update_metrics(task, acc, None, global_step)
                                iid_bumped_tasks.add(task)
                                did_sync = True
                        if did_sync:
                            refresh_curriculum_snapshots()

                    # Track best
                    if overall_acc > best_alg_acc:
                        best_alg_acc = overall_acc
                        best_payload = build_checkpoint_payload(global_step, tokens_seen)
                        best_payload["alg_accuracy"] = best_alg_acc
                        torch.save(best_payload, output_dir / "best_model.pt")
                        if args.extended_logging:
                            log_print(f"  New best! Saved checkpoint.")

                # Algorithmic accuracy (OOD grid)
                if args.eval_algorithmic_ood:
                    ood_results = evaluate_algorithmic(
                        model,
                        tokenizer,
                        device,
                        n_examples=args.eval_samples,
                        max_new_tokens=args.eval_max_new_tokens,
                        tasks=alg_tasks,
                        seed=args.seed,
                        sample_count_per_task=args.eval_sample_count_per_task,
                        grid="ood",
                    )
                    overall_acc = ood_results.get("overall_accuracy", 0.0)
                    overall_mae = ood_results.get("overall_mae")
                    overall_token_accuracy = ood_results.get("overall_token_accuracy", 0.0)
                    overall_distance = ood_results.get("overall_distance", 1.0)
                    overall_prefix_accuracy = ood_results.get("overall_prefix_accuracy", 0.0)
                    per_task_mae = ood_results.get("per_task_mae", {}) or {}
                    per_task_token_accuracy = ood_results.get("per_task_token_accuracy", {}) or {}
                    per_task_distance = ood_results.get("per_task_distance", {}) or {}
                    per_task_prefix_accuracy = (
                        ood_results.get("per_task_prefix_accuracy", {}) or {}
                    )
                    per_task_accuracy = ood_results.get("per_task_accuracy", {}) or {}
                    if args.extended_logging:
                        log_print(
                            "  Algorithmic OOD metrics: "
                            f"acc={overall_acc*100:.1f}% | "
                            f"token={overall_token_accuracy:.3f} | "
                            f"dist={overall_distance:.3f} | "
                            f"prefix={overall_prefix_accuracy:.3f}"
                        )
                        diagnostics = ood_results.get("diagnostics", {})
                        answer_diag = diagnostics.get("answer_length", {})
                        labels_diag = diagnostics.get("labels", {})
                        parse_diag = diagnostics.get("parse", {})
                        log_print(
                            "  Algorithmic OOD diagnostics: "
                            f"empty={answer_diag.get('empty_rate', 0.0):.3f} | "
                            f"eos_first={answer_diag.get('eos_first_rate', 0.0):.3f} | "
                            f"invalid_label={labels_diag.get('invalid_label_rate', 0.0):.3f} | "
                            f"numeric_parse_fail={parse_diag.get('numeric_parse_fail_rate', 0.0):.3f}"
                        )
                        if overall_mae is not None:
                            log_print(f"  Algorithmic OOD MAE (numeric tasks): {overall_mae:.3f}")

                        _print_sampled_examples(
                            "Algorithmic OOD eval",
                            ood_results.get("sampled_examples_by_task", {}),
                        )
                        if args.debug_eval_format_stats:
                            _print_prompt_format_stats(
                                "Algorithmic OOD eval",
                                diagnostics.get("prompt_format", {}),
                            )

                    logger.log(
                        step=global_step,
                        phase="eval",
                        **{
                            "eval_ood/acc/exact_match": overall_acc,
                            "eval_ood/acc/token": overall_token_accuracy,
                            "eval_ood/acc/distance": overall_distance,
                            "eval_ood/acc/prefix": overall_prefix_accuracy,
                            "eval_ood/numeric/mae": overall_mae,
                            "eval_ood/curriculum/difficulty": difficulty_logged,
                            "eval_ood/samples": ood_results.get("sampled_examples_by_task", {}),
                        },
                        **{
                            f"eval_ood/acc/exact_match/{task}": acc
                            for task, acc in per_task_accuracy.items()
                        },
                        **{
                            f"eval_ood/acc/token/{task}": acc
                            for task, acc in per_task_token_accuracy.items()
                        },
                        **{
                            f"eval_ood/acc/prefix/{task}": acc
                            for task, acc in per_task_prefix_accuracy.items()
                        },
                        **{
                            f"eval_ood/acc/distance/{task}": dist
                            for task, dist in per_task_distance.items()
                        },
                        **{
                            f"eval_ood/numeric/mae/{task}": mae
                            for task, mae in per_task_mae.items()
                            if mae is not None
                        },
                        **_flatten_eval_diagnostics(
                            "eval_ood", ood_results.get("diagnostics", {})
                        ),
                        **_relabel_phase(eval_task_metrics, "eval_ood"),
                    )
                    if per_task_accuracy:
                        warn_on_format_crash(
                            "Algorithmic OOD",
                            prev_eval_ood_task_acc,
                            prev_eval_ood_task_token_acc,
                            per_task_accuracy,
                            per_task_token_accuracy,
                        )
                        prev_eval_ood_task_acc = dict(per_task_accuracy)
                        prev_eval_ood_task_token_acc = dict(per_task_token_accuracy)
                    for task, acc in per_task_accuracy.items():
                        task_mae = per_task_mae.get(task)
                        logger.log_task_accuracy(
                            task,
                            acc,
                            global_step,
                            target="eval_ood",
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
                            answer_diag = ood_results.get("diagnostics", {}).get(
                                "answer_length", {}
                            )
                            labels_diag = ood_results.get("diagnostics", {}).get("labels", {})
                            parse_diag = ood_results.get("diagnostics", {}).get("parse", {})
                            empty_rate = answer_diag.get("empty_rate_by_task", {}).get(task)
                            eos_first_rate = answer_diag.get("eos_first_rate_by_task", {}).get(task)
                            max_new_tokens_rate = answer_diag.get(
                                "max_new_tokens_rate_by_task", {}
                            ).get(task)
                            if empty_rate is not None:
                                line += f" | empty={empty_rate:.3f}"
                            if eos_first_rate is not None:
                                line += f" | eos_first={eos_first_rate:.3f}"
                            if max_new_tokens_rate is not None:
                                line += f" | max_new={max_new_tokens_rate:.3f}"
                            if task in CLASSIFICATION_TASKS:
                                invalid_rate = labels_diag.get(
                                    "invalid_label_rate_by_task", {}
                                ).get(task)
                                if invalid_rate is not None:
                                    line += f" | invalid_label={invalid_rate:.3f}"
                            elif task in NUMERIC_OUTPUT_TASKS:
                                parse_fail_rate = parse_diag.get(
                                    "by_task", {}
                                ).get(task, {}).get("numeric_parse_fail_rate")
                                if parse_fail_rate is not None:
                                    line += f" | parse_fail={parse_fail_rate:.3f}"
                            if task_mae is not None:
                                line += f" | MAE: {task_mae:.3f}"
                            log_print(line)

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
                    curriculum_metrics: Dict[str, float] = {}
                    curriculum_log_lines: List[str] = []
                    if task_curriculum is not None:
                        per_task_accuracy = task_eval.get("per_task_accuracy", {})
                        for task, acc in per_task_accuracy.items():
                            loss_proxy = 1.0 - acc
                            task_curriculum.update_metrics(task, acc, loss_proxy, global_step)
                        if dag_state is not None:
                            ema_snapshot = {
                                task: task_curriculum.get_task_state(task)["ema_acc"]
                                for task in dag_state.tasks
                            }
                            dag_state.update_from_ema(ema_snapshot)
                            dag_gate_snapshot.clear()
                            dag_gate_snapshot.update(dag_state.get_gate_snapshot())
                            _set_frontier_snapshot(
                                dag_frontier_snapshot,
                                dag_state.compute_frontier(ema_snapshot),
                            )
                            dag_replay_ratio_snapshot[0] = dag_state.compute_replay_ratio(
                                ema_snapshot
                            )
                            dag_unlocked_snapshot.clear()
                            dag_unlocked_snapshot.update(
                                {task for task, gate in dag_gate_snapshot.items() if gate > 0.0}
                            )
                            dag_ema_snapshot = dict(ema_snapshot)
                            dag_metrics = {
                                f"eval/curriculum/gate/{task}": gate
                                for task, gate in dag_gate_snapshot.items()
                            }
                            dag_metrics["eval/curriculum/dag_paused"] = (
                                1.0 if dag_state.paused else 0.0
                            )
                            dag_metrics["eval/curriculum/dag_frontier_size"] = float(
                                len(dag_frontier_snapshot)
                            )
                            dag_metrics["eval/curriculum/dag_unlocked_size"] = float(
                                len(dag_unlocked_snapshot)
                            )
                            dag_metrics["eval/curriculum/dag_replay_ratio"] = (
                                dag_replay_ratio_snapshot[0]
                            )
                            curriculum_metrics.update(dag_metrics)
                        unlock_warmup_evals = (
                            args.dag_unlock_warmup_evals
                            if args.dag_unlock_warmup_evals is not None
                            else args.curriculum_min_task_evals
                        )
                        unlock_warmup_evals = max(int(unlock_warmup_evals), 0)
                        for task, acc in per_task_accuracy.items():
                            should_step_curriculum = True
                            if task in iid_bumped_tasks:
                                should_step_curriculum = False
                            if dag_state is not None:
                                gate = dag_gate_snapshot.get(task, 0.0)
                                if gate <= 0.0:
                                    should_step_curriculum = False
                                else:
                                    unlocked_index = dag_state.unlocked_eval_index.get(task)
                                    if unlocked_index is not None:
                                        evals_since_unlock = (
                                            dag_state.eval_index - unlocked_index
                                        )
                                        if evals_since_unlock < unlock_warmup_evals:
                                            should_step_curriculum = False
                            if should_step_curriculum:
                                task_curriculum.step_curriculum(
                                    task,
                                    global_step,
                                    args.curriculum_cooldown,
                                )
                            state = task_curriculum.get_task_state(task)
                            curriculum_metrics[f"eval/curriculum/difficulty/{task}"] = state[
                                "difficulty"
                            ]
                            curriculum_metrics[f"eval/curriculum/ema_acc/{task}"] = state[
                                "ema_acc"
                            ]
                            curriculum_log_lines.append(
                                f"{task}: d={state['difficulty']:.2f} "
                                f"ema={state['ema_acc']:.2f}"
                            )
                        if curriculum_metrics:
                            logger.log(step=global_step, phase="eval", **curriculum_metrics)
                            log_print("  Curriculum: " + " | ".join(sorted(curriculum_log_lines)))
                        refresh_curriculum_snapshots()
                    if args.extended_logging:
                        log_print(
                            f"  Training accuracy ({train_eval_label}): {train_acc*100:.2f}% | "
                            f"Loss: {train_loss_eval:.4f}"
                        )
                        log_print(f"  Training task accuracy: {overall_task_acc*100:.1f}%")
                        diagnostics = task_eval.get("diagnostics", {})
                        answer_diag = diagnostics.get("answer_length", {})
                        labels_diag = diagnostics.get("labels", {})
                        parse_diag = diagnostics.get("parse", {})
                        log_print(
                            "  Training eval diagnostics: "
                            f"empty={answer_diag.get('empty_rate', 0.0):.3f} | "
                            f"eos_first={answer_diag.get('eos_first_rate', 0.0):.3f} | "
                            f"invalid_label={labels_diag.get('invalid_label_rate', 0.0):.3f} | "
                            f"numeric_parse_fail={parse_diag.get('numeric_parse_fail_rate', 0.0):.3f}"
                        )
                        if overall_task_mae is not None:
                            log_print(f"  Training task MAE (numeric tasks): {overall_task_mae:.3f}")

                        _print_sampled_examples(
                            "Training eval",
                            task_eval.get("sampled_examples_by_task", {}),
                        )
                        if args.debug_eval_format_stats:
                            _print_prompt_format_stats(
                                "Training eval",
                                diagnostics.get("prompt_format", {}),
                            )

                    per_task_mae = task_eval.get("per_task_mae", {}) or {}
                    per_task_token_accuracy = task_eval.get("per_task_token_accuracy", {}) or {}
                    per_task_distance = task_eval.get("per_task_distance", {}) or {}
                    per_task_prefix_accuracy = (
                        task_eval.get("per_task_prefix_accuracy", {}) or {}
                    )
                    per_task_accuracy = task_eval.get("per_task_accuracy", {}) or {}

                    logger.log(
                        step=global_step,
                        phase="eval",
                        **{
                            "train_eval/acc/token": train_acc,
                            "train_eval/loss/ce": train_loss_eval,
                            "train_eval/acc/exact_match": overall_task_acc,
                            "train_eval/numeric/mae": overall_task_mae,
                            "train_eval/acc/token": task_eval.get("overall_token_accuracy"),
                            "train_eval/acc/distance": task_eval.get("overall_distance"),
                            "train_eval/acc/prefix": task_eval.get("overall_prefix_accuracy"),
                            "train_eval/curriculum/difficulty": difficulty_logged,
                            "train_eval/samples": task_eval.get("sampled_examples_by_task", {}),
                        },
                        **{
                            f"train_eval/acc/exact_match/{task}": acc
                            for task, acc in per_task_accuracy.items()
                        },
                        **{
                            f"train_eval/acc/token/{task}": acc
                            for task, acc in per_task_token_accuracy.items()
                        },
                        **{
                            f"train_eval/acc/prefix/{task}": acc
                            for task, acc in per_task_prefix_accuracy.items()
                        },
                        **{
                            f"train_eval/acc/distance/{task}": dist
                            for task, dist in per_task_distance.items()
                        },
                        **{
                            f"train_eval/numeric/mae/{task}": mae
                            for task, mae in per_task_mae.items()
                            if mae is not None
                        },
                        **_flatten_eval_diagnostics(
                            "train_eval", task_eval.get("diagnostics", {})
                        ),
                        **_relabel_phase(eval_task_metrics, "train_eval"),
                    )
                    if per_task_accuracy:
                        warn_on_format_crash(
                            f"Training eval ({train_eval_label})",
                            prev_train_eval_task_acc,
                            prev_train_eval_task_token_acc,
                            per_task_accuracy,
                            per_task_token_accuracy,
                        )
                        prev_train_eval_task_acc = dict(per_task_accuracy)
                        prev_train_eval_task_token_acc = dict(per_task_token_accuracy)
                    for task, acc in per_task_accuracy.items():
                        task_mae = per_task_mae.get(task)
                        logger.log_task_accuracy(
                            task,
                            acc,
                            global_step,
                            target="train_eval",
                            mae=task_mae,
                            mean_token_accuracy=per_task_token_accuracy.get(task),
                            mean_distance=per_task_distance.get(task),
                            mean_prefix_accuracy=per_task_prefix_accuracy.get(task),
                        )
                        if args.extended_logging:
                            line = f"    {task}: {acc*100:.1f}%"
                            answer_diag = task_eval.get("diagnostics", {}).get(
                                "answer_length", {}
                            )
                            labels_diag = task_eval.get("diagnostics", {}).get("labels", {})
                            parse_diag = task_eval.get("diagnostics", {}).get("parse", {})
                            empty_rate = answer_diag.get("empty_rate_by_task", {}).get(task)
                            eos_first_rate = answer_diag.get("eos_first_rate_by_task", {}).get(task)
                            max_new_tokens_rate = answer_diag.get(
                                "max_new_tokens_rate_by_task", {}
                            ).get(task)
                            if empty_rate is not None:
                                line += f" | empty={empty_rate:.3f}"
                            if eos_first_rate is not None:
                                line += f" | eos_first={eos_first_rate:.3f}"
                            if max_new_tokens_rate is not None:
                                line += f" | max_new={max_new_tokens_rate:.3f}"
                            if task in CLASSIFICATION_TASKS:
                                invalid_rate = labels_diag.get(
                                    "invalid_label_rate_by_task", {}
                                ).get(task)
                                if invalid_rate is not None:
                                    line += f" | invalid_label={invalid_rate:.3f}"
                            elif task in NUMERIC_OUTPUT_TASKS:
                                parse_fail_rate = parse_diag.get(
                                    "by_task", {}
                                ).get(task, {}).get("numeric_parse_fail_rate")
                                if parse_fail_rate is not None:
                                    line += f" | parse_fail={parse_fail_rate:.3f}"
                            if task_mae is not None:
                                line += f" | MAE: {task_mae:.3f}"
                            log_print(line)

                # Perplexity (only in Phase 2)
                if curriculum.phase == "language":
                    ppl = evaluate_perplexity(model, lang_loader, device, ctx)
                    if args.extended_logging:
                        log_print(f"  Language PPL: {ppl:.2f}")
                    logger.log(
                        step=global_step,
                        phase="eval",
                        **{
                            "eval/loss/ppl": ppl,
                            "eval/loss/ce": math.log(ppl),
                        },
                    )

                eval_duration = time.time() - eval_start
                print(f"Eval duration: {eval_duration:.2f}s")
                logger.log(
                    step=global_step,
                    phase="eval",
                    **{"eval/throughput/eval_duration": eval_duration},
                )
                logger.plot()
                logger.save()
                torch.cuda.empty_cache()
                last_log_time += eval_duration

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
    final_payload = build_checkpoint_payload(global_step, tokens_seen)
    torch.save(final_payload, output_dir / "final_model.pt")
    
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
        "--min_difficulty",
        type=float,
        default=0.0,
        help="Minimum difficulty floor applied to schedules and per-task curricula",
    )
    parser.add_argument(
        "--task_weighting",
        type=str,
        default="adaptive",
        choices=["uniform", "adaptive"],
        help="Task sampling weights",
    )
    parser.add_argument(
        "--disable_dynamic_task_weighting",
        action="store_true",
        help="Disable dynamic task weighting and use uniform task sampling.",
    )
    parser.add_argument(
        "--use_task_curriculum",
        action="store_true",
        help="Enable per-task competence curriculum for algorithmic data",
    )
    parser.add_argument(
        "--task_curriculum_strategy",
        type=str,
        default="adaptive",
        choices=["adaptive", "dag"],
        help="Task-level curriculum strategy: adaptive (current) or dag (staged unlock + replay).",
    )
    parser.add_argument(
        "--curriculum_cooldown",
        type=int,
        default=500,
        help="Steps to wait between curriculum difficulty updates per task",
    )
    parser.add_argument(
        "--curriculum_min_task_evals",
        type=int,
        default=5,
        help="Minimum evals per task before curriculum difficulty can adjust",
    )
    parser.add_argument(
        "--curriculum_jitter",
        type=float,
        default=0.1,
        help="Probability of replaying easier samples per task",
    )
    parser.add_argument("--dag_patience_evals", type=int, default=4)
    parser.add_argument("--dag_ramp_evals", type=int, default=3)
    parser.add_argument("--dag_replay_ratio", type=float, default=0.20)
    parser.add_argument("--dag_replay_ratio_backslide", type=float, default=0.35)
    parser.add_argument("--dag_locked_floor", type=float, default=0.02)
    parser.add_argument("--dag_unlock_margin", type=float, default=0.01)
    parser.add_argument("--dag_lock_margin", type=float, default=0.03)
    parser.add_argument("--dag_frontier_recent_evals", type=int, default=4)
    parser.add_argument("--dag_mastery_margin", type=float, default=0.02)
    parser.add_argument(
        "--dag_unlock_warmup_evals",
        type=int,
        default=None,
        help="Eval count to wait after a task unlocks before adjusting difficulty "
        "(defaults to --curriculum_min_task_evals).",
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
        "--resume_from",
        type=str,
        default=None,
        help="Path to a training checkpoint to resume from",
    )
    parser.add_argument(
        "--eval_algorithmic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the algorithmic IID eval suite (disable with --no-eval_algorithmic)",
    )
    parser.add_argument(
        "--eval_algorithmic_ood",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the algorithmic OOD eval suite (disabled by default)",
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
        "--debug_eval_format_stats",
        action="store_true",
        help="Print prompt-format cohort tables during eval logging",
    )
    parser.add_argument(
        "--extended_logging",
        action="store_true",
        help="Print extended metrics beyond loss/LR to the console.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (recommended for Linux only)")
    parser.add_argument(
        "--compile_dynamic",
        action="store_true",
        help="Enable dynamic shapes for torch.compile when batches vary in length.",
    )

    args = parser.parse_args()

    if args.mix_band_tokens is None:
        args.mix_band_tokens = int(0.5 * args.alg_tokens)
    train(args)


if __name__ == "__main__":
    main()
