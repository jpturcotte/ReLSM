"""Utility helpers shared across evaluation entrypoints.

This module centralizes model/tokenizer loading, deterministic generation
defaults, random seeding, long-context data creation, algorithmic scoring,
and metadata recording so that all evaluation paths share consistent
behavior and output schemas.
"""

import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from data import AlgorithmicGenerator
from model import BaselineTransformer, TransformerConfig


def seed_all(seed: int) -> None:
    """Seed Python, NumPy (if available), and torch RNGs."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: Optional[str] = None) -> torch.device:
    """Return a CUDA device when available unless overridden."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


SPACE_PATTERN = re.compile(r"\s+")


def prepare_tokenizer(tokenizer_name: str):
    """Load a tokenizer and ensure padding is defined."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.no_grad()
def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_name: str,
    device: torch.device,
) -> Tuple[BaselineTransformer, Any, Any]:
    """Load a BaselineTransformer checkpoint and matching tokenizer.

    Returns
    -------
    model : BaselineTransformer
        The model loaded in evaluation mode on the requested device.
    config : Any
        The serialized configuration stored alongside the checkpoint.
    tokenizer : PreTrainedTokenizer
        Tokenizer ready for generation with a defined pad token.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = BaselineTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = prepare_tokenizer(tokenizer_name)
    return model, config, tokenizer


def get_eval_generation_kwargs(
    tokenizer=None,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return generation kwargs compatible with ``BaselineTransformer.generate``.

    HF-style kwargs are intentionally filtered out so evaluation only forwards
    arguments our custom ``model.generate`` understands.
    """

    allowed_keys = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        # Explicitly surface the EOS token ID our custom generator supports.
        "eos_token_id": getattr(tokenizer, "eos_token_id", None) if tokenizer else None,
    }

    if extra_kwargs:
        # Strip out any kwargs meant for HF generation APIs to avoid leaking
        # unsupported parameters into custom models.
        for key, value in extra_kwargs.items():
            if key in allowed_keys:
                allowed_keys[key] = value

    # Only forward parameters our custom generate implementation accepts and
    # that have concrete values (drop optional None entries like eos_token_id).
    return {k: v for k, v in allowed_keys.items() if v is not None}


def generate_text(
    model: BaselineTransformer,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 32,
    generation_kwargs: Optional[Dict] = None,
) -> str:
    """Generate a continuation for a single prompt."""
    generation_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        extra_kwargs=generation_kwargs,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_len = input_ids.shape[1]

    output_ids = model.generate(input_ids=input_ids, **generation_kwargs)
    generated_tokens = output_ids[0, input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def batch_generate(
    model: BaselineTransformer,
    tokenizer,
    prompts: Iterable[str],
    device: torch.device,
    max_new_tokens: int = 32,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Vectorized generation helper for multiple prompts."""
    prompts_list = list(prompts)
    gen_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        extra_kwargs=generation_kwargs,
    )

    # Tokenize each prompt individually to avoid introducing padding tokens that
    # custom models may treat as real context.
    encoded: List[Tuple[int, torch.Tensor]] = []
    for idx, prompt in enumerate(prompts_list):
        tokens = tokenizer(prompt, truncation=True, return_tensors="pt")
        encoded.append((idx, tokens["input_ids"].squeeze(0)))

    # Group prompts by exact tokenized length to preserve batching without
    # padding. Each micro-batch shares the same sequence length and can be
    # stacked without inserting pad tokens.
    buckets: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    for item in encoded:
        _, input_ids = item
        buckets.setdefault(input_ids.size(0), []).append(item)

    generations: List[str] = [""] * len(prompts_list)
    for prompt_len, bucket in buckets.items():
        batch_ids = torch.stack([ids for _, ids in bucket], dim=0).to(device)
        output_ids = model.generate(input_ids=batch_ids, **gen_kwargs)
        for i, (orig_idx, _) in enumerate(bucket):
            tail = output_ids[i, prompt_len:]
            generations[orig_idx] = tokenizer.decode(tail, skip_special_tokens=True).strip()
    return generations


class NeedleInHaystackGenerator:
    """Generate long-context retrieval examples with controllable depth."""

    def __init__(self, tokenizer, context_length: int = 4096):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.filler_sentences = [
            "The best way to learn is by doing.",
            "Ideas are not precious, execution is.",
            "Start with something small and iterate.",
            "The market is the ultimate judge.",
            "Good taste is essential for great work.",
            "Curiosity drives innovation forward.",
            "Simple solutions are often the best.",
            "Focus on what matters most first.",
            "Learn from failure, celebrate success.",
            "Build something people actually want.",
        ]

    def _build_haystack(self, needle_tokens: List[int], depth: float) -> List[int]:
        target_length = self.context_length - len(needle_tokens) - 50
        haystack_tokens: List[int] = []
        while len(haystack_tokens) < target_length:
            sentence = random.choice(self.filler_sentences)
            haystack_tokens.extend(self.tokenizer.encode(" " + sentence, add_special_tokens=False))
        haystack_tokens = haystack_tokens[:target_length]
        insert_pos = int(len(haystack_tokens) * depth)
        return haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]

    def generate(self, needle_depth: float = 0.5) -> Dict:
        secret_number = random.randint(1000, 9999)
        needle = f"The secret code is {secret_number}."
        needle_tokens = self.tokenizer.encode(needle, add_special_tokens=False)

        haystack = self._build_haystack(needle_tokens, needle_depth)
        question = " Question: What is the secret code? Answer:"
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        full_tokens = haystack + question_tokens

        return {
            "input_ids": torch.tensor(full_tokens),
            "answer": str(secret_number),
            "needle_depth": needle_depth,
            "context_length": len(full_tokens),
        }

    def generate_sweep(self, n_depths: int = 10) -> List[Dict]:
        examples = []
        for i in range(n_depths):
            depth = i / (n_depths - 1) if n_depths > 1 else 0.5
            examples.append(self.generate(needle_depth=depth))
        return examples


def save_json(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def compute_perplexity(
    model: BaselineTransformer,
    tokenizer,
    texts: Iterable[str],
    device: torch.device,
) -> float:
    """Compute perplexity for a collection of texts."""

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

    avg_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(min(avg_loss, 20.0))).item())


@dataclass
class EvalResult:
    """Container for algorithmic evaluation statistics."""

    condition: str
    task: str
    accuracy: float
    n: int
    avg_gen_len: float
    tokens_per_sec: float
    examples: List[Dict[str, str]]
    correct: int
    tokens_generated: int
    elapsed: float


def normalize_prediction(task: str, text: str) -> str:
    """Normalize model outputs for fair comparisons across tasks."""

    text = text.split("\n")[0]
    text = SPACE_PATTERN.sub(" ", text).strip()
    if task == "dyck":
        return text.lower()
    return text


def normalize_target(task: str, text: str) -> str:
    """Normalize ground-truth targets for matching."""

    if task == "dyck":
        return text.lower().strip()
    return SPACE_PATTERN.sub(" ", text).strip()


def build_dataset(task: str, n: int, params: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    """Generate a deterministic dataset for a given task/condition."""

    examples: List[Dict[str, Any]] = []
    gen_fn = AlgorithmicGenerator._get_generators()[task]

    for i in range(n):
        rng = random.Random(seed + i)
        kwargs = dict(params)
        if task == "dyck":
            kwargs["force_valid"] = i < n / 2
        example = gen_fn(rng=rng, difficulty=0.5, **kwargs)
        examples.append(example)
    return examples


def batch_tokenize(tokenizer, prompts: List[str], device: torch.device):
    """Tokenize a list of prompts on the specified device."""

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def decode_generated(tokenizer, input_ids, attention_mask, output_ids, task: str) -> List[str]:
    """Decode generated continuations and normalize predictions."""

    preds: List[str] = []
    for inp, mask, out in zip(input_ids, attention_mask, output_ids):
        prompt_len = int(mask.sum().item())
        gen_tokens = out[prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        preds.append(normalize_prediction(task, text))
    return preds


def score_predictions(task: str, preds: List[str], targets: List[str]) -> Tuple[int, int]:
    """Compute correct counts for a batch of predictions."""

    correct = 0
    for pred, tgt in zip(preds, targets):
        if normalize_prediction(task, pred) == normalize_target(task, tgt):
            correct += 1
    return correct, len(targets)


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: torch.device,
    use_autocast: bool,
    task: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], int, float]:
    """Generate model completions and return predictions and throughput stats."""

    tokenized = batch_tokenize(tokenizer, prompts, device)
    input_ids = tokenized["input_ids"]
    gen_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        extra_kwargs=generation_kwargs,
    )
    start = time.time()
    with torch.no_grad():
        if use_autocast:
            with torch.cuda.amp.autocast():
                outputs = model.generate(input_ids=input_ids, **gen_kwargs)
        else:
            outputs = model.generate(input_ids=input_ids, **gen_kwargs)
    elapsed = max(1e-6, time.time() - start)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    preds = decode_generated(tokenizer, input_ids, attention_mask, outputs, task=task)
    total_tokens = int((outputs.shape[1] - input_ids.shape[1]) * outputs.shape[0])
    return preds, total_tokens, elapsed


def evaluate_condition(
    model,
    tokenizer,
    task: str,
    condition: str,
    params: Dict[str, Any],
    n: int,
    device: torch.device,
    max_new_tokens: int,
    seed: int,
    batch_size: int = 16,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """Evaluate a single algorithmic condition (IID or OOD)."""

    seed_all(seed)
    dataset = build_dataset(task, n, params, seed)
    use_autocast = device.type == "cuda"

    total_correct = 0
    total_examples = 0
    total_tokens = 0
    total_time = 0.0
    total_gen_len = 0.0
    collected_examples: List[Dict[str, str]] = []

    for start_idx in range(0, n, batch_size):
        batch = dataset[start_idx : start_idx + batch_size]
        prompts = [ex["input"] for ex in batch]
        targets = [ex["target"] for ex in batch]
        preds, tok_count, elapsed = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            device=device,
            use_autocast=use_autocast,
            task=task,
            generation_kwargs=generation_kwargs,
        )
        total_tokens += tok_count
        total_time += elapsed
        avg_batch_gen = tok_count / max(len(batch), 1)
        total_gen_len += avg_batch_gen
        for pred, tgt, ex in zip(preds, targets, batch):
            norm_pred = normalize_prediction(task, pred)
            norm_tgt = normalize_target(task, tgt)
            if norm_pred == norm_tgt:
                total_correct += 1
            else:
                if len(collected_examples) < 10:
                    collected_examples.append(
                        {
                            "prompt": ex["input"],
                            "target": tgt,
                            "prediction": pred,
                        }
                    )
            total_examples += 1

    accuracy = total_correct / total_examples if total_examples else 0.0
    avg_gen_len = total_gen_len / math.ceil(n / batch_size) if n else 0.0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    return EvalResult(
        condition=condition,
        task=task,
        accuracy=accuracy,
        n=total_examples,
        avg_gen_len=avg_gen_len,
        tokens_per_sec=tokens_per_sec,
        examples=collected_examples,
        correct=total_correct,
        tokens_generated=total_tokens,
        elapsed=total_time,
    )


def gather_metadata(
    checkpoint: Optional[str],
    tokenizer_name: Optional[str],
    device: torch.device,
    model_config: Optional[TransformerConfig],
    generation_kwargs: Dict[str, Any],
    suite: Optional[str] = None,
    seed: Optional[int] = None,
    grid_version: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect run metadata for auditing and reproducibility."""

    commit = None
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
            .decode()
            .strip()
        )
    except Exception:
        commit = None

    cuda_version = torch.version.cuda or None
    torch_version = torch.__version__

    model_summary = None
    if model_config is not None:
        getter = (
            (lambda k: getattr(model_config, k))
            if not isinstance(model_config, dict)
            else (lambda k: model_config.get(k))
        )
        try:
            model_summary = {
                "variant": getter("variant"),
                "d_model": getter("d_model"),
                "n_layers": getter("n_layers"),
                "n_heads": getter("n_heads"),
                "max_seq_len": getter("max_seq_len"),
            }
        except Exception:
            model_summary = None

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "commit": commit,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cuda": cuda_version,
        "torch": torch_version,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer_name,
        "device": str(device),
        "model": model_summary,
        "model_id": model_id,
        "suite": suite,
        "seed": seed,
        "grid_version": grid_version,
        "decoding": generation_kwargs,
    }


def area_under_depth_curve(by_depth: Dict[float, float]) -> float:
    """Compute the average retrieval accuracy across depths."""

    if not by_depth:
        return 0.0
    depths = sorted(by_depth)
    total = 0.0
    for depth in depths:
        total += by_depth[depth]
    return total / len(depths)


def write_summary_md(output_path: Path, summary: Dict[str, Any]) -> None:
    """Write a lightweight human-readable summary file."""

    lines = ["# Evaluation Summary", ""]
    metadata = summary.get("metadata", {})
    if metadata:
        lines.append("## Metadata")
        for k in ["checkpoint", "commit", "device", "tokenizer", "timestamp"]:
            if metadata.get(k) is not None:
                lines.append(f"- **{k}**: {metadata[k]}")
        lines.append("")

    algo = summary.get("results", {}).get("algorithmic", {})
    if algo:
        lines.append("## Algorithmic (IID)")
        overall = algo.get("overall_accuracy")
        if overall is not None:
            lines.append(f"- Overall accuracy: {overall*100:.2f}%")
        for task, acc in sorted(algo.get("per_task", {}).items()):
            lines.append(f"- {task}: {acc*100:.2f}%")
        lines.append("")

    ood = summary.get("results", {}).get("ood", {})
    if ood:
        lines.append("## OOD Grid")
        overall = ood.get("overall_accuracy")
        if overall is not None:
            lines.append(f"- Overall accuracy: {overall*100:.2f}%")
        lines.append("- Conditions: {}".format(len(ood.get("table", []))))
        lines.append("")

    longctx = summary.get("results", {}).get("longctx", {})
    if longctx:
        lines.append("## Needle Retrieval")
        auc = longctx.get("auc")
        if auc is not None:
            lines.append(f"- Avg depth accuracy: {auc*100:.2f}%")
        for ctx in longctx.get("per_context", []):
            lines.append(
                f"- ctx={ctx['context_length']}: {ctx['metrics']['retrieval_accuracy']*100:.2f}%"
            )
        lines.append("")

    ppl = summary.get("results", {}).get("ppl", {})
    if ppl:
        lines.append("## TinyStories PPL")
        if "perplexity" in ppl:
            lines.append(f"- Perplexity: {ppl['perplexity']:.2f} on {ppl.get('n', 0)} samples")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def write_ood_csv(output_path: Path, table: Sequence[Dict[str, Any]]) -> None:
    """Write OOD grid results to CSV for convenient analysis."""

    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "condition",
        "accuracy",
        "n",
        "avg_gen_len",
        "tokens_per_sec",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row.get(k) for k in fieldnames})
