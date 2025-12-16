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
from multiprocessing import Value
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, Dict, List

import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine schedule with warmup."""
    current_step = min(step, max_steps)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class MetricsLogger:
    """Tracks metrics for the ablation ladder."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
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
        self.task_accuracies = {}
    
    def log(self, step: int, phase: str, **kwargs):
        self.metrics["step"].append(step)
        self.metrics["phase"].append(phase)
        
        for k, v in kwargs.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
    
    def log_task_accuracy(self, task: str, accuracy: float, step: int):
        if task not in self.task_accuracies:
            self.task_accuracies[task] = []
        self.task_accuracies[task].append({"step": step, "accuracy": accuracy})
    
    def save(self):
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump({
                "training": self.metrics,
                "task_accuracies": self.task_accuracies,
            }, f, indent=2)
    
    def summary(self) -> Dict:
        return {
            "final_train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else None,
            "final_val_loss": self.metrics["val_loss"][-1] if self.metrics["val_loss"] else None,
            "peak_vram_gb": max(self.metrics["peak_vram_gb"]) if self.metrics["peak_vram_gb"] else None,
            "avg_tokens_per_sec": sum(self.metrics["tokens_per_sec"]) / len(self.metrics["tokens_per_sec"]) if self.metrics["tokens_per_sec"] else None,
            "total_tokens": self.metrics["tokens_seen"][-1] if self.metrics["tokens_seen"] else 0,
        }


@torch.no_grad()
def evaluate_algorithmic(model, tokenizer, device, n_examples: int = 100) -> Dict[str, float]:
    """Run the canonical algorithmic OOD grid using ``eval_hub``.

    The ``n_examples`` argument caps the number of examples per condition
    (kept for backward compatibility with older training scripts).
    """

    from eval_hub import run_algorithmic_suite
    from utils import get_eval_generation_kwargs

    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    generation_kwargs = get_eval_generation_kwargs(tokenizer=tokenizer, max_new_tokens=0)

    results = run_algorithmic_suite(
        model,
        tokenizer,
        device_obj,
        seed=42,
        batch_size=8,
        generation_kwargs=generation_kwargs,
        limit=n_examples,
    )

    summary = {"overall": results.get("overall_accuracy", 0.0)}
    summary.update(results.get("per_task_accuracy", {}))
    return summary


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
    if schedule == "cosine":
        return 0.5 + 0.5 * (1 - math.cos(math.pi * progress)) / 2
    if schedule == "step":
        return 0.5 if progress < 0.5 else 1.0

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
        AlgorithmicDataset, LanguageDataset, CurriculumSampler,
        create_curriculum_loaders
    )
    
    print("\nLoading datasets...")

    difficulty_value = Value("d", 0.0)
    
    # Phase 1: Algorithmic
    alg_dataset = AlgorithmicDataset(
        tokenizer=tokenizer,
        num_examples=args.alg_examples,
        max_seq_len=args.alg_seq_len,
        seed=args.seed,
        difficulty_value=difficulty_value,
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
    
    global_step = 0
    tokens_seen = 0
    start_time = time.time()
    best_alg_acc = 0.0
    
    # Estimate steps
    tokens_per_step = args.alg_batch_size * args.alg_seq_len * args.grad_accum_steps
    max_steps = args.total_tokens // tokens_per_step
    warmup_steps = int(0.1 * max_steps)
    
    print(f"  Estimated steps: {max_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    running_inner_steps = 0.0
    running_ponder = 0.0
    
    while curriculum.tokens_seen < args.total_tokens:
        if curriculum.tokens_seen < args.alg_tokens:
            difficulty_value.value = difficulty_schedule(
                curriculum.tokens_seen, args.alg_tokens, args.difficulty_schedule
            )
        else:
            difficulty_value.value = 1.0
        batch = curriculum.next_batch()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Learning rate
        lr = get_lr(global_step, warmup_steps, max_steps, args.max_lr, args.min_lr)
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
        
        # Step
        if (global_step + 1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        global_step += 1
        tokens_seen = curriculum.tokens_seen
        
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

            print(f"Step {global_step:>6} | "
                  f"Phase: {phase[:4]:>4} | "
                  f"Progress: {progress:>5.1f}% | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tok/s: {toks_per_sec:.0f} | "
                  f"VRAM: {peak_vram:.1f}GB" +
                  (f" | K: {avg_K:.1f}" if avg_K > 0 else "") +
                  (f" | ponder: {avg_ponder:.3f}" if avg_ponder > 0 else ""))

            logger.log(
                step=global_step,
                phase=phase,
                train_loss=avg_loss,
                tokens_seen=tokens_seen,
                tokens_per_sec=toks_per_sec,
                peak_vram_gb=peak_vram,
                avg_inner_steps=avg_K,
                ponder=avg_ponder,
                difficulty=float(difficulty_value.value),
            )

            running_loss = 0.0
            running_inner_steps = 0.0
            running_ponder = 0.0
        
        # Evaluation
        if global_step % args.eval_interval == 0:
            print("\nRunning evaluation...")
            
            # Algorithmic accuracy
            alg_results = evaluate_algorithmic(model, tokenizer, device, n_examples=200)
            print(f"  Algorithmic accuracy: {alg_results['overall']*100:.1f}%")
            for task, acc in alg_results.items():
                if task != "overall":
                    logger.log_task_accuracy(task, acc, global_step)
                    print(f"    {task}: {acc*100:.1f}%")
            
            # Track best
            if alg_results["overall"] > best_alg_acc:
                best_alg_acc = alg_results["overall"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "step": global_step,
                    "alg_accuracy": best_alg_acc,
                }, output_dir / "best_model.pt")
                print(f"  New best! Saved checkpoint.")
            
            # Perplexity (only in Phase 2)
            if curriculum.phase == "language":
                ppl = evaluate_perplexity(model, lang_loader, device, ctx)
                print(f"  Language PPL: {ppl:.2f}")
                logger.log(step=global_step, phase="eval", val_loss=math.log(ppl))
            
            print()
            model.train()
    
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
    parser.add_argument("--tokenizer", type=str, default="gpt2")
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
        default="linear",
        choices=["linear", "cosine", "step"],
        help="Schedule for ramping algorithmic difficulty",
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
    parser.add_argument("--alg_batch_size", type=int, default=64)
    parser.add_argument("--lang_seq_len", type=int, default=1024)
    parser.add_argument("--lang_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Optimizer
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=4)

    # Logging
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (recommended for Linux only)")

    args = parser.parse_args()

    if args.mix_band_tokens is None:
        args.mix_band_tokens = int(0.5 * args.alg_tokens)
    train(args)


if __name__ == "__main__":
    main()
