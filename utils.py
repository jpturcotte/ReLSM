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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from data import AlgorithmicGenerator
from model import BaselineTransformer, TransformerConfig

DEFAULT_TOKENIZER_NAME = "EleutherAI/llemma_7b"


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


def select_device(device: Optional[Union[str, int]] = None) -> torch.device:
    """Return a CUDA device when available unless overridden."""
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        if torch.cuda.is_available():
            return torch.device("cuda", device)
        return torch.device("cpu")
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


SPACE_PATTERN = re.compile(r"\s+")
_INT_RE = re.compile(r"^-?\d{1,40}$")


def safe_parse_number(s: str, default: float = float("nan")) -> float:
    """Parse numeric string with overflow protection."""
    s = s.strip()
    if not s or not _INT_RE.match(s):
        return default
    try:
        val = float(s)
    except (ValueError, OverflowError):
        return default
    if math.isinf(val) or math.isnan(val):
        return default
    return val


def compute_mae(predictions: List[str], targets: List[str]) -> float:
    """Compute MAE with NaN filtering for invalid parses."""
    errors: List[float] = []
    for pred, tgt in zip(predictions, targets):
        pred_val = safe_parse_number(pred)
        tgt_val = safe_parse_number(tgt)
        if not (math.isnan(pred_val) or math.isnan(tgt_val)):
            errors.append(abs(pred_val - tgt_val))
    return sum(errors) / len(errors) if errors else float("nan")


def prepare_tokenizer(tokenizer_name: str):
    """Load a tokenizer and ensure padding is defined."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - defensive import guard
        raise ImportError(
            "transformers is required for loading tokenizers; install it to proceed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_max_length(model: BaselineTransformer, tokenizer: Any) -> int | None:
    """Best-effort maximum length derived from model config or tokenizer."""

    config = getattr(model, "config", None)
    if config is not None:
        max_len = getattr(config, "max_seq_len", None)
        if max_len is not None:
            return max_len

    model_max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_len, int) and model_max_len < 1e12:
        return model_max_len
    return None


def _tokenize_prompt(tokenizer: Any, text: str, max_length: Optional[int] = None) -> Dict[str, Any]:
    """Call a tokenizer while gracefully handling unsupported kwargs."""

    base_kwargs: Dict[str, Any] = {"truncation": True, "padding": False, "return_tensors": "pt"}
    if max_length is not None:
        base_kwargs["max_length"] = max_length

    try:
        return tokenizer(text, **base_kwargs)
    except TypeError:
        reduced_kwargs = {k: v for k, v in base_kwargs.items() if k != "max_length"}
        try:
            return tokenizer(text, **reduced_kwargs)
        except TypeError:
            minimal_kwargs = {
                "padding": base_kwargs.get("padding"),
                "return_tensors": base_kwargs.get("return_tensors"),
            }
            try:
                return tokenizer(text, **minimal_kwargs)
            except TypeError:
                return tokenizer(text)


def _ensure_2d_tensor(value: Any) -> torch.Tensor:
    """Convert arbitrary tensor-like ``input_ids`` to a 2D torch.Tensor."""

    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


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
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
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
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
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
    max_length = _resolve_max_length(model, tokenizer)
    tokenize_kwargs: Dict[str, Any] = {"return_tensors": "pt", "truncation": True}
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length

    try:
        encoded = tokenizer(prompt, **tokenize_kwargs)
    except TypeError:
        tokenize_kwargs.pop("max_length", None)
        encoded = tokenizer(prompt, **tokenize_kwargs)

    input_ids = _ensure_2d_tensor(encoded["input_ids"]).to(device)
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
    # Be explicit about the decoding limit in case downstream code mutates
    # ``generation_kwargs`` before calling ``model.generate``.
    gen_kwargs.setdefault("max_new_tokens", max_new_tokens)

    # Tokenize each prompt individually to avoid introducing padding tokens that
    # custom models may treat as real context.
    encoded: List[Tuple[int, torch.Tensor]] = []
    max_length = _resolve_max_length(model, tokenizer)
    for idx, prompt in enumerate(prompts_list):
        tokens = _tokenize_prompt(tokenizer, prompt, max_length=max_length)
        encoded.append((idx, _ensure_2d_tensor(tokens["input_ids"]).squeeze(0)))

    # Group prompts by exact tokenized length to preserve batching without
    # padding. Each micro-batch shares the same sequence length and can be
    # stacked without inserting pad tokens.
    buckets: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    for item in encoded:
        _, input_ids = item
        buckets.setdefault(input_ids.size(0), []).append(item)

    generations: List[str] = [""] * len(prompts_list)
    for prompt_len in sorted(buckets):
        bucket = buckets[prompt_len]
        batch_ids = torch.stack([ids for _, ids in bucket], dim=0).to(device)
        output_ids = model.generate(input_ids=batch_ids, **gen_kwargs)
        for i, (orig_idx, _) in enumerate(bucket):
            tail = output_ids[i, prompt_len:]
            generations[orig_idx] = tokenizer.decode(tail, skip_special_tokens=True).strip()
    return generations


@torch.no_grad()
def evaluate_model(
    model: BaselineTransformer,
    tokenizer,
    prompts: Iterable[str],
    device: torch.device,
    max_new_tokens: int = 32,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate model completions for a collection of prompts.

    The ``max_new_tokens`` argument bounds generation length to avoid runaway
    decoding when EOS markers are missing. Additional generation kwargs are
    forwarded to the model's ``generate`` implementation when provided.
    """

    return batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=max_new_tokens,
        generation_kwargs=generation_kwargs,
    )


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
            max_length = _resolve_max_length(model, tokenizer)
            encoded = _tokenize_prompt(tokenizer, text, max_length=max_length)
            input_ids = _ensure_2d_tensor(encoded["input_ids"]).to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = _ensure_2d_tensor(attention_mask).to(device)
            else:
                attention_mask = torch.ones_like(input_ids)

            if input_ids.size(1) < 2:
                continue

            labels = input_ids.clone()
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)

            _, loss, _ = model(input_ids, labels=labels)

            if attention_mask is not None:
                n_tokens = int(attention_mask[..., 1:].sum().item())
            else:
                n_tokens = input_ids.size(1) - 1

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(min(avg_loss, 20.0))).item())


@dataclass
class EvalResult:
    """Container for algorithmic evaluation statistics."""

    condition: str
    task: str
    difficulty: float
    accuracy: float
    mean_token_accuracy: float
    mean_distance: float
    mean_prefix_accuracy: float
    mae: Optional[float]
    numeric_count: int
    n: int
    avg_gen_len: float
    tokens_per_sec: float
    examples: List[Dict[str, str]]
    sampled_examples: List[Dict[str, str]]
    correct: int
    tokens_generated: int
    elapsed: float


def normalize_prediction(task: str, text: str) -> str:
    """Normalize model outputs for fair comparisons across tasks."""

    text = text.split("\n")[0]
    text = SPACE_PATTERN.sub(" ", text).strip()
    text = text.strip(" \t\n\r\"'`.,;:!?")

    if task == "dyck":
        lowered = text.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
        return lowered

    numeric_tasks = {"mod_add", "parity", "addition", "multiplication", "chain", "successor"}
    if task in numeric_tasks:
        match = re.search(r"[+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?", text)
        if match:
            return match.group(0)

    return text


def normalize_target(task: str, text: str) -> str:
    """Normalize ground-truth targets for matching."""

    if task == "dyck":
        return text.lower().strip()
    return SPACE_PATTERN.sub(" ", text).strip()


def compute_metrics(task: str, pred: str, target: str) -> Dict[str, float]:
    """Compute universal metrics for algorithmic and fixed training tasks."""
    pred = _metrics_normalize(task, pred)
    target = _metrics_normalize(task, target)

    exact_match = float(pred == target)

    if _is_numeric_task(task):
        return _numeric_metrics(pred, target, exact_match)
    if _is_classification_task(task):
        return _classification_metrics(pred, target, exact_match)
    return _sequence_metrics(pred, target, exact_match)


def _metrics_normalize(task: str, text: str) -> str:
    text = text.split("\n")[0].strip().lower()
    text = SPACE_PATTERN.sub(" ", text)

    if task in {"dyck", "parity", "compare"}:
        text = text.split()[0] if text else ""
        text = text.strip(".,!?\"'-")

    return text


def _is_numeric_task(task: str) -> bool:
    return task in {"addition", "multiplication", "chain", "successor", "mod_add"}


def _is_classification_task(task: str) -> bool:
    return task in {"parity", "dyck", "compare"}


def _numeric_metrics(pred: str, target: str, exact_match: float) -> Dict[str, float]:
    pred_num = safe_parse_number(pred)
    target_num = safe_parse_number(target)
    parseable = not (math.isnan(pred_num) or math.isnan(target_num))

    pred_digits = re.sub(r"[^0-9\-]", "", pred).lstrip("-") or "0"
    target_digits = re.sub(r"[^0-9\-]", "", target).lstrip("-") or "0"
    max_len = max(len(pred_digits), len(target_digits))
    pred_padded = pred_digits.zfill(max_len)
    target_padded = target_digits.zfill(max_len)
    token_accuracy = sum(p == t for p, t in zip(pred_padded, target_padded)) / max_len

    if parseable:
        abs_error = abs(pred_num - target_num)
        normalized_distance = min(
            1.0,
            math.log1p(abs_error)
            / math.log1p(abs(target_num) + 1000),
        )
    else:
        normalized_distance = 1.0

    prefix_len = 0
    for p, t in zip(pred_padded, target_padded):
        if p == t:
            prefix_len += 1
        else:
            break
    prefix_accuracy = prefix_len / max_len

    return {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "normalized_distance": normalized_distance,
        "prefix_accuracy": prefix_accuracy,
    }


def _sequence_metrics(pred: str, target: str, exact_match: float) -> Dict[str, float]:
    pred_tokens = pred.split()
    target_tokens = target.split()

    if not target_tokens:
        return {
            "exact_match": exact_match,
            "token_accuracy": 1.0 if not pred_tokens else 0.0,
            "normalized_distance": 0.0 if not pred_tokens else 1.0,
            "prefix_accuracy": 1.0 if not pred_tokens else 0.0,
        }

    max_len = max(len(pred_tokens), len(target_tokens))
    matches = sum(p == t for p, t in zip(pred_tokens, target_tokens))
    token_accuracy = matches / max_len
    normalized_distance = _edit_distance(pred_tokens, target_tokens) / max_len

    prefix_len = 0
    for p, t in zip(pred_tokens, target_tokens):
        if p == t:
            prefix_len += 1
        else:
            break
    prefix_accuracy = prefix_len / len(target_tokens)

    return {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "normalized_distance": min(1.0, normalized_distance),
        "prefix_accuracy": prefix_accuracy,
    }


def _edit_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (a[i - 1] != b[j - 1]),
            )
            prev = temp
    return dp[n]


def _classification_metrics(pred: str, target: str, exact_match: float) -> Dict[str, float]:
    token_accuracy = exact_match
    normalized_distance = 1.0 - exact_match

    if pred and target:
        max_len = max(len(pred), len(target))
        prefix_len = 0
        for p, t in zip(pred, target):
            if p == t:
                prefix_len += 1
            else:
                break
        prefix_accuracy = prefix_len / max_len
    else:
        prefix_accuracy = exact_match

    return {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "normalized_distance": normalized_distance,
        "prefix_accuracy": prefix_accuracy,
    }


@dataclass
class AggregatedMetrics:
    task: str
    n: int
    accuracy: float
    mean_token_accuracy: float
    mean_distance: float
    mean_prefix_accuracy: float


def aggregate(task: str, results: List[Dict[str, float]]) -> AggregatedMetrics:
    n = len(results)
    if n == 0:
        return AggregatedMetrics(task, 0, 0.0, 0.0, 1.0, 0.0)

    return AggregatedMetrics(
        task=task,
        n=n,
        accuracy=sum(r["exact_match"] for r in results) / n,
        mean_token_accuracy=sum(r["token_accuracy"] for r in results) / n,
        mean_distance=sum(r["normalized_distance"] for r in results) / n,
        mean_prefix_accuracy=sum(r["prefix_accuracy"] for r in results) / n,
    )


def build_dataset(
    task: str,
    n: int,
    params: Dict[str, Any],
    seed: int,
    difficulty: float = 0.5,
) -> List[Dict[str, Any]]:
    """Generate a deterministic dataset for a given task/condition."""

    examples: List[Dict[str, Any]] = []
    gen_fn = AlgorithmicGenerator._get_generators()[task]

    for i in range(n):
        rng = random.Random(seed + i)
        kwargs = dict(params)
        if task == "dyck":
            kwargs["force_valid"] = i < n / 2
        example = gen_fn(rng=rng, difficulty=difficulty, **kwargs)
        examples.append(example)
    return examples


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
    batch_size: int = 16,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[int], float]:
    """Generate model completions and return predictions and throughput stats."""

    prompts_list = list(prompts)
    gen_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        extra_kwargs=generation_kwargs,
    )

    # Tokenize each prompt individually to avoid inserting padding tokens that
    # custom models would treat as real context.
    encoded: List[Tuple[int, torch.Tensor]] = []
    max_length = _resolve_max_length(model, tokenizer)
    for idx, prompt in enumerate(prompts_list):
        tokens = _tokenize_prompt(tokenizer, prompt, max_length=max_length)
        encoded.append((idx, _ensure_2d_tensor(tokens["input_ids"]).squeeze(0)))

    # Bucket by exact sequence length so we can batch prompts without padding.
    buckets: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    for item in encoded:
        _, input_ids = item
        seq_len = input_ids.size(0)
        buckets.setdefault(seq_len, []).append(item)

    preds: List[str] = [""] * len(prompts_list)
    generation_lengths: List[int] = [0] * len(prompts_list)
    eos_token_id = gen_kwargs.get("eos_token_id")

    start = time.time()
    with torch.no_grad():
        for prompt_len in sorted(buckets):
            bucket = buckets[prompt_len]
            for start_idx in range(0, len(bucket), batch_size):
                microbatch = bucket[start_idx : start_idx + batch_size]
                batch_ids = torch.stack([ids for _, ids in microbatch], dim=0).to(device)
                if use_autocast:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model.generate(input_ids=batch_ids, **gen_kwargs)
                else:
                    outputs = model.generate(input_ids=batch_ids, **gen_kwargs)

                for i, (orig_idx, _) in enumerate(microbatch):
                    gen_tokens = outputs[i, prompt_len:]
                    if eos_token_id is not None:
                        eos_positions = (gen_tokens == eos_token_id).nonzero(as_tuple=False)
                        if eos_positions.numel() > 0:
                            gen_tokens = gen_tokens[: eos_positions[0].item() + 1]

                    generation_lengths[orig_idx] = int(gen_tokens.size(0))
                    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    preds[orig_idx] = normalize_prediction(task, decoded)

    elapsed = max(1e-6, time.time() - start)
    return preds, generation_lengths, elapsed


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
    difficulty: float = 0.5,
    sample_count: int = 0,
) -> EvalResult:
    """Evaluate a single algorithmic condition (IID or OOD)."""

    seed_all(seed)
    dataset = build_dataset(task, n, params, seed, difficulty=difficulty)
    use_autocast = device.type == "cuda"

    total_correct = 0
    total_absolute_error = 0.0
    total_numeric_samples = 0
    total_examples = 0
    total_tokens = 0
    total_time = 0.0
    total_gen_len = 0.0
    collected_examples: List[Dict[str, str]] = []
    metrics_samples: List[Dict[str, float]] = []
    sampled_examples: List[Dict[str, str]] = []
    sample_seen = 0
    sample_target = max(sample_count, 0)
    sample_rng = random.Random(seed + 17)

    for start_idx in range(0, n, batch_size):
        batch = dataset[start_idx : start_idx + batch_size]
        prompts = [ex["input"] for ex in batch]
        targets = [ex["target"] for ex in batch]
        preds, gen_lengths, elapsed = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            device=device,
            use_autocast=use_autocast,
            task=task,
            batch_size=batch_size,
            generation_kwargs=generation_kwargs,
        )
        batch_generated = sum(gen_lengths)
        total_tokens += batch_generated
        total_time += elapsed
        total_gen_len += batch_generated
        for pred, tgt, ex in zip(preds, targets, batch):
            norm_tgt = normalize_target(task, tgt)
            metrics = compute_metrics(task, pred, tgt)
            metrics_samples.append(metrics)
            if sample_target:
                sample_item = {
                    "prompt": ex["input"],
                    "target": tgt,
                    "expected_output": tgt,
                    "prediction": pred,
                }
                if len(sampled_examples) < sample_target:
                    sampled_examples.append(sample_item)
                else:
                    idx = sample_rng.randint(0, sample_seen)
                    if idx < sample_target:
                        sampled_examples[idx] = sample_item
                sample_seen += 1
            if metrics["exact_match"] == 1.0:
                total_correct += 1
            else:
                if len(collected_examples) < 10:
                    collected_examples.append(
                        {
                            "prompt": ex["input"],
                            "target": tgt,
                            "expected_output": tgt,
                            "prediction": pred,
                        }
                    )

            val_pred = safe_parse_number(pred)
            val_tgt = safe_parse_number(norm_tgt)
            if not (math.isnan(val_pred) or math.isnan(val_tgt)):
                total_absolute_error += abs(val_pred - val_tgt)
                total_numeric_samples += 1
            total_examples += 1

    aggregated = aggregate(task, metrics_samples)
    accuracy = aggregated.accuracy
    mae = total_absolute_error / total_numeric_samples if total_numeric_samples > 0 else None
    avg_gen_len = total_gen_len / total_examples if total_examples else 0.0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    return EvalResult(
        condition=condition,
        task=task,
        difficulty=difficulty,
        accuracy=accuracy,
        mean_token_accuracy=aggregated.mean_token_accuracy,
        mean_distance=aggregated.mean_distance,
        mean_prefix_accuracy=aggregated.mean_prefix_accuracy,
        mae=mae,
        numeric_count=total_numeric_samples,
        n=total_examples,
        avg_gen_len=avg_gen_len,
        tokens_per_sec=tokens_per_sec,
        examples=collected_examples,
        sampled_examples=sampled_examples,
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
