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
import torch.nn as nn

from data import AlgorithmicGenerator
from model import BaselineTransformer, TransformerConfig, apply_rotary_emb

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
_INT_RE = re.compile(r"^-?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
PROMPT_FORMAT_KEYS = ("colon", "answer", "arrow", "fatarrow", "equals")
LABEL_TASKS = {"dyck", "parity", "compare"}


def safe_parse_number(
    s: str,
    default: float = float("nan"),
    *,
    max_digits: int = 20,
    max_magnitude: float = 1e15,
) -> float:
    """Parse numeric string with overflow protection."""
    s = s.strip()
    if not s or not _INT_RE.match(s):
        return default
    if len(s.lstrip("+-")) > max_digits:
        return default
    try:
        val = float(s)
    except (ValueError, OverflowError):
        return default
    if math.isinf(val) or math.isnan(val):
        return default
    if abs(val) > max_magnitude:
        return default
    return val


def parse_numeric_prediction(text: str) -> Tuple[float, Optional[str]]:
    """Parse a numeric prediction with failure categorization."""
    cleaned = text.strip()
    if not cleaned:
        return float("nan"), "empty_output"
    candidate = extract_integer_token(cleaned)
    if candidate is None:
        return float("nan"), "contains_non_digit"
    if candidate in {"+", "-"}:
        return float("nan"), "wrong_sign_or_format"
    if not _INTEGER_RE.match(candidate):
        return float("nan"), "contains_non_digit"
    try:
        return float(int(candidate)), None
    except ValueError:
        return float("nan"), "wrong_sign_or_format"


def extract_integer_token(text: str) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None

    answer_matches = list(re.finditer(r"answer:", cleaned, flags=re.IGNORECASE))
    if answer_matches:
        cleaned = cleaned[answer_matches[-1].end() :]
    else:
        separator_positions = []
        for sep in ("=>", "->", "="):
            idx = cleaned.rfind(sep)
            if idx >= 0:
                separator_positions.append((idx, sep))
        if separator_positions:
            idx, sep = max(separator_positions, key=lambda item: item[0])
            cleaned = cleaned[idx + len(sep) :]

    matches = list(re.finditer(r"[+-]?\d+", cleaned))
    if not matches:
        return None
    return matches[-1].group(0)


def digit_length_from_token(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    return len(token.lstrip("+-"))


def get_prompt_format_flags(prompt: str) -> Dict[str, bool]:
    stripped = prompt.rstrip()
    return {
        "colon": stripped.endswith(":"),
        "answer": "Answer:" in prompt,
        "arrow": "->" in prompt,
        "fatarrow": "=>" in prompt,
        "equals": "=" in prompt,
    }


def normalize_label_prediction(task: str, text: str) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None

    if task == "compare":
        for token in cleaned.split():
            stripped = token.strip().strip(".,!?\"'`()[]{}")
            if stripped in {">", "<", "="}:
                return stripped
        return None

    if task == "dyck":
        for token in re.findall(r"[A-Za-z]+", cleaned):
            lowered = token.lower()
            if lowered == "yes":
                return "yes"
            if lowered == "no":
                return "no"
        return None

    if task == "parity":
        match = re.search(r"\b[01]\b", cleaned)
        return match.group(0) if match else None

    return None


def compute_repetition_metrics(token_ids: Sequence[int]) -> Dict[str, float]:
    """Compute repetition/degeneration metrics for a token sequence."""
    length = len(token_ids)
    if length <= 0:
        return {
            "repeat_1gram_rate": 0.0,
            "repeat_2gram_rate": 0.0,
            "unique_token_fraction": 0.0,
            "max_run_length": 0.0,
        }

    repeat_1 = 0
    max_run = 1
    current_run = 1
    bigram_counts: Dict[Tuple[int, int], int] = {}
    unique_tokens = set()

    prev = token_ids[0]
    unique_tokens.add(prev)
    for idx in range(1, length):
        token = token_ids[idx]
        unique_tokens.add(token)
        if token == prev:
            repeat_1 += 1
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
        bigram = (prev, token)
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
        prev = token

    max_run = max(max_run, current_run)
    bigram_total = max(length - 1, 1)
    repeated_bigrams = sum(count - 1 for count in bigram_counts.values() if count > 1)
    repeat_2_rate = repeated_bigrams / bigram_total
    repeat_1_rate = repeat_1 / bigram_total
    unique_fraction = len(unique_tokens) / max(length, 1)

    return {
        "repeat_1gram_rate": repeat_1_rate,
        "repeat_2gram_rate": repeat_2_rate,
        "unique_token_fraction": unique_fraction,
        "max_run_length": float(max_run),
    }


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


def _find_first_module(root: nn.Module, candidates: Sequence[str]) -> Optional[nn.Module]:
    for name in candidates:
        module = getattr(root, name, None)
        if isinstance(module, nn.Module):
            return module
    return None


def get_transformer_blocks(model: nn.Module) -> List[nn.Module]:
    candidates = [
        "layers",
        "blocks",
        "h",
        "transformer_blocks",
        "encoder_layers",
        "decoder_layers",
        "unique_layers",
    ]
    for name in candidates:
        blocks = getattr(model, name, None)
        if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0:
            filtered = [
                block
                for block in blocks
                if isinstance(block, nn.Module) and hasattr(block, "attn") and hasattr(block, "ff")
            ]
            return filtered
    return []


def resolve_diagnostic_modules(model: nn.Module) -> Dict[str, Optional[nn.Module]]:
    blocks = get_transformer_blocks(model)
    diagnostic: Dict[str, Optional[nn.Module]] = {
        "embedding": _find_first_module(
            model, ("tok_emb", "token_embedding", "embedding", "embed_tokens", "wte")
        ),
        "head": _find_first_module(model, ("lm_head", "head", "output_head", "output_proj", "proj")),
    }

    if blocks:
        diagnostic["block0"] = blocks[0]
        diagnostic["block_mid"] = blocks[len(blocks) // 2]
        diagnostic["block_last"] = blocks[-1]

    return diagnostic


def module_grad_weight_norm(module: Optional[nn.Module]) -> Tuple[float, float]:
    if module is None:
        return float("nan"), float("nan")
    params = list(module.parameters(recurse=True))
    if not params:
        return float("nan"), float("nan")
    with torch.no_grad():
        weight_tensors = [
            param.detach().view(-1) for param in params if param.numel() > 0
        ]
        if weight_tensors:
            all_weights = torch.cat(weight_tensors)
            weight_norm = all_weights.norm(2.0, dtype=torch.float32).item()
        else:
            weight_norm = float("nan")

        grads = [
            param.grad.detach().view(-1)
            for param in params
            if param.grad is not None and param.grad.numel() > 0
        ]
        if grads:
            all_grads = torch.cat(grads)
            grad_norm = all_grads.norm(2.0, dtype=torch.float32).item()
        else:
            grad_norm = 0.0
    if math.isnan(grad_norm):
        print(
            f"WARNING: NaN detected in logging for module: {module.__class__.__name__}"
        )
        grad_norm = 0.0
    return grad_norm, weight_norm


def compute_weight_entropy(module: Optional[nn.Module], bins: int = 256) -> float:
    if module is None:
        return float("nan")

    abs_max = 0.0
    has_param = False
    for param in module.parameters(recurse=True):
        if param.numel() == 0:
            continue
        has_param = True
        abs_max = max(abs_max, float(param.detach().abs().max().item()))

    if not has_param:
        return float("nan")
    if abs_max == 0.0:
        return 0.0

    hist = torch.zeros(bins, dtype=torch.float64)
    for param in module.parameters(recurse=True):
        if param.numel() == 0:
            continue
        values = param.detach().abs().float().cpu()
        hist += torch.histc(values, bins=bins, min=0.0, max=abs_max).double()

    total = float(hist.sum().item())
    if total <= 0:
        return 0.0
    probs = hist / total
    entropy = -(probs * (probs + 1e-12).log()).sum().item()
    return float(entropy)


def compute_attention_probs(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T, _ = hidden_states.shape

    q = attn_module.q_proj(hidden_states).view(B, T, attn_module.n_heads, attn_module.d_head).transpose(1, 2)
    k = attn_module.k_proj(hidden_states).view(B, T, attn_module.n_kv_heads, attn_module.d_head).transpose(1, 2)

    cos, sin = attn_module.rotary(T, hidden_states.device)
    if position_ids is None:
        position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0)

    q, k = apply_rotary_emb(q, k, cos, sin, position_ids)

    k = attn_module._repeat_kv(k)

    q = q.float()
    k = k.float()
    scores = torch.matmul(q, k.transpose(-2, -1)) * float(attn_module.scale)

    if mask is None:
        causal_mask = torch.full((T, T), float("-inf"), device=hidden_states.device)
        mask = torch.triu(causal_mask, diagonal=1)
    scores = scores + mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return probs


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
    torch.serialization.add_safe_globals([TransformerConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
    parse_failures: int
    n: int
    avg_gen_len: float
    tokens_per_sec: float
    examples: List[Dict[str, str]]
    sampled_examples: List[Dict[str, str]]
    correct: int
    tokens_generated: int
    elapsed: float
    target_len_tokens: List[int]
    pred_len_tokens: List[int]
    length_ratio: List[float]
    abs_len_error: List[float]
    stop_reason_counts: Dict[str, int]
    eos_emitted: int
    repeat_1gram_rate: List[float]
    repeat_2gram_rate: List[float]
    unique_token_fraction: List[float]
    max_run_length: List[float]
    parse_successes: int
    parse_failure_counts: Dict[str, int]
    numeric_abs_errors: List[float]
    numeric_rel_errors: List[float]
    empty_predictions: int
    first_token_is_eos_count: int
    invalid_label_count: int
    label_confusion_counts: Dict[str, int]
    format_stats: Dict[str, Dict[str, int]]
    numeric_length_mismatch_count: int
    numeric_length_total: int


def _base_normalize(text: str) -> str:
    """Common normalization applied to both predictions and targets."""

    text = text.split("\n")[0]
    text = SPACE_PATTERN.sub(" ", text).strip()
    return text.strip(" \t\n\r\"'`.,;:!?")


def normalize_prediction(task: str, text: str) -> str:
    """Normalize model outputs for fair comparisons across tasks."""

    text = _base_normalize(text)

    if task in LABEL_TASKS:
        label = normalize_label_prediction(task, text)
        return label if label is not None else ""

    numeric_tasks = {"mod_add", "addition", "multiplication", "chain", "successor"}
    if task in numeric_tasks:
        token = extract_integer_token(text)
        return token if token is not None else ""

    return text


def normalize_target(task: str, text: str) -> str:
    """Normalize ground-truth targets for matching."""

    text = _base_normalize(text)
    if task == "dyck":
        return text.lower()
    return text


def compute_metrics(task: str, pred: str, target: str, tokenizer: Any) -> Dict[str, float]:
    """Compute universal metrics for algorithmic and fixed training tasks."""
    pred = _metrics_normalize(task, pred)
    target = _metrics_normalize(task, target)

    exact_match = float(pred == target)

    pred_ids = _encode_for_metrics(tokenizer, pred)
    target_ids = _encode_for_metrics(tokenizer, target)
    token_accuracy, prefix_accuracy = _token_accuracy(pred_ids, target_ids)

    if _is_numeric_task(task):
        normalized_distance = _numeric_distance(pred, target)
    elif _is_classification_task(task):
        normalized_distance = 1.0 - exact_match
    else:
        normalized_distance = _sequence_distance(pred_ids, target_ids)

    return {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "normalized_distance": normalized_distance,
        "prefix_accuracy": prefix_accuracy,
    }
    pred = _metrics_normalize(task, pred)
    target = _metrics_normalize(task, target)

    exact_match = float(pred == target)

    if _is_numeric_task(task):
        return _numeric_metrics(pred, target, exact_match)
    if _is_classification_task(task):
        return _classification_metrics(pred, target, exact_match)
    return _sequence_metrics(pred, target, exact_match)


def _metrics_normalize(task: str, text: str) -> str:
    text = text.split("\n")[0].strip()
    text = SPACE_PATTERN.sub(" ", text)

    if task in LABEL_TASKS:
        label = normalize_label_prediction(task, text)
        return label if label is not None else ""

    text = text.lower()
    if task in {"mod_add", "addition", "multiplication", "chain", "successor"}:
        token = extract_integer_token(text)
        return token if token is not None else ""

    return text


def _is_numeric_task(task: str) -> bool:
    return task in {"addition", "multiplication", "chain", "successor", "mod_add"}


def _is_classification_task(task: str) -> bool:
    return task in LABEL_TASKS


def _encode_for_metrics(tokenizer: Any, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, add_special_tokens=False)
        if isinstance(encoded, dict):
            return encoded["input_ids"][0].tolist()
        return encoded[0].tolist()


def _token_accuracy(pred_ids: List[int], target_ids: List[int]) -> Tuple[float, float]:
    if not target_ids:
        token_accuracy = 1.0 if not pred_ids else 0.0
        prefix_accuracy = 1.0 if not pred_ids else 0.0
        return token_accuracy, prefix_accuracy

    max_len = max(len(pred_ids), len(target_ids))
    matches = sum(p == t for p, t in zip(pred_ids, target_ids))
    token_accuracy = matches / max_len

    prefix_len = 0
    for p, t in zip(pred_ids, target_ids):
        if p == t:
            prefix_len += 1
        else:
            break
    prefix_accuracy = prefix_len / max(len(target_ids), 1)
    return token_accuracy, prefix_accuracy


def _numeric_distance(pred: str, target: str) -> float:
    pred_num = safe_parse_number(pred)
    target_num = safe_parse_number(target)
    parseable = not (math.isnan(pred_num) or math.isnan(target_num))

    if parseable:
        abs_error = abs(pred_num - target_num)
        return min(
            1.0,
            math.log1p(abs_error)
            / math.log1p(abs(target_num) + 1000),
        )
    else:
        return 1.0


def _sequence_distance(pred_ids: List[int], target_ids: List[int]) -> float:
    if not target_ids:
        return 0.0 if not pred_ids else 1.0
    max_len = max(len(pred_ids), len(target_ids))
    normalized_distance = _edit_distance(pred_ids, target_ids) / max_len
    return min(1.0, normalized_distance)


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
        example_seed = hash((seed, task, i)) & 0x7FFFFFFF
        rng = random.Random(example_seed)
        kwargs = dict(params)
        if task == "dyck":
            valid_ratio = params.get("valid_ratio", 0.5)
            kwargs["force_valid"] = rng.random() < valid_ratio
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


def _token_length(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        return int(encoded["input_ids"].shape[1])


def extract_prediction_from_generate(
    prompt_input_ids: Sequence[int] | torch.Tensor,
    generated_ids: Sequence[int] | torch.Tensor,
    eos_token_id: Optional[int],
    tokenizer: Any,
) -> Tuple[str, List[int], str, bool]:
    if isinstance(prompt_input_ids, torch.Tensor):
        prompt_list = prompt_input_ids.tolist()
    else:
        prompt_list = list(prompt_input_ids)
    if isinstance(generated_ids, torch.Tensor):
        generated_list = generated_ids.tolist()
    else:
        generated_list = list(generated_ids)

    prompt_len = len(prompt_list)
    continuation_ids = generated_list[prompt_len:]
    first_token_is_eos = bool(
        continuation_ids and eos_token_id is not None and continuation_ids[0] == eos_token_id
    )
    stop_reason = "max_new_tokens"
    if eos_token_id is not None:
        for idx, token_id in enumerate(continuation_ids):
            if token_id == eos_token_id:
                continuation_ids = continuation_ids[:idx]
                stop_reason = "eos"
                break
    prediction = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
    return prediction, continuation_ids, stop_reason, first_token_is_eos


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
) -> Tuple[List[str], List[int], float, Dict[str, List[Any]]]:
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
    raw_preds: List[str] = [""] * len(prompts_list)
    generation_lengths: List[int] = [0] * len(prompts_list)
    stop_reasons: List[str] = ["max_new_tokens"] * len(prompts_list)
    eos_emitted: List[bool] = [False] * len(prompts_list)
    first_token_is_eos: List[bool] = [False] * len(prompts_list)
    pred_token_ids: List[List[int]] = [[] for _ in prompts_list]
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
                    prediction, continuation_ids, stop_reason, first_is_eos = (
                        extract_prediction_from_generate(
                            batch_ids[i],
                            outputs[i],
                            eos_token_id,
                            tokenizer,
                        )
                    )
                    generation_lengths[orig_idx] = len(continuation_ids)
                    pred_token_ids[orig_idx] = continuation_ids
                    raw_preds[orig_idx] = prediction
                    preds[orig_idx] = normalize_prediction(task, prediction)
                    stop_reasons[orig_idx] = stop_reason
                    eos_emitted[orig_idx] = stop_reason == "eos"
                    first_token_is_eos[orig_idx] = first_is_eos

    elapsed = max(1e-6, time.time() - start)
    return preds, generation_lengths, elapsed, {
        "raw_predictions": raw_preds,
        "pred_token_ids": pred_token_ids,
        "stop_reasons": stop_reasons,
        "eos_emitted": eos_emitted,
        "first_token_is_eos": first_token_is_eos,
    }


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
    total_parse_failures = 0
    total_examples = 0
    total_tokens = 0
    total_time = 0.0
    total_gen_len = 0.0
    collected_examples: List[Dict[str, str]] = []
    metrics_samples: List[Dict[str, float]] = []
    sampled_examples: List[Dict[str, str]] = []
    target_len_tokens_list: List[int] = []
    pred_len_tokens_list: List[int] = []
    length_ratio_list: List[float] = []
    abs_len_error_list: List[float] = []
    stop_reason_counts = {"eos": 0, "max_new_tokens": 0, "other": 0}
    eos_emitted_count = 0
    repeat_1gram_list: List[float] = []
    repeat_2gram_list: List[float] = []
    unique_token_fraction_list: List[float] = []
    max_run_length_list: List[float] = []
    parse_successes = 0
    parse_failure_counts: Dict[str, int] = {}
    numeric_abs_errors: List[float] = []
    numeric_rel_errors: List[float] = []
    empty_prediction_count = 0
    first_token_is_eos_count = 0
    invalid_label_count = 0
    label_confusion_counts: Dict[str, int] = {}
    format_stats = {
        key: {"count": 0, "correct": 0, "empty": 0, "eos_first": 0}
        for key in PROMPT_FORMAT_KEYS
    }
    numeric_length_mismatch_count = 0
    numeric_length_total = 0
    sample_seen = 0
    sample_target = max(sample_count, 0)
    sample_rng = random.Random(seed + 17)

    for start_idx in range(0, n, batch_size):
        batch = dataset[start_idx : start_idx + batch_size]
        prompts = [ex["input"] for ex in batch]
        targets = [ex["target"] for ex in batch]
        preds, gen_lengths, elapsed, gen_info = generate_batch(
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
        raw_preds = gen_info["raw_predictions"]
        pred_token_ids = gen_info["pred_token_ids"]
        stop_reasons = gen_info["stop_reasons"]
        eos_emitted = gen_info["eos_emitted"]
        first_token_is_eos = gen_info["first_token_is_eos"]
        for idx, (pred, tgt, ex) in enumerate(zip(preds, targets, batch)):
            norm_tgt = normalize_target(task, tgt)
            metrics = compute_metrics(task, pred, tgt, tokenizer)
            metrics_samples.append(metrics)
            target_len_tokens = _token_length(tokenizer, norm_tgt)
            pred_tokens = pred_token_ids[idx]
            pred_len_tokens = len(pred_tokens)
            raw_pred = raw_preds[idx]
            empty_prediction = raw_pred.strip() == ""
            length_ratio = pred_len_tokens / max(target_len_tokens, 1)
            abs_len_error = abs(pred_len_tokens - target_len_tokens)
            repeat_metrics = compute_repetition_metrics(pred_tokens)
            stop_reason = stop_reasons[idx]
            eos_hit = eos_emitted[idx]
            first_is_eos = first_token_is_eos[idx]
            if stop_reason not in stop_reason_counts:
                stop_reason = "other"
            stop_reason_counts[stop_reason] += 1
            eos_emitted_count += 1 if eos_hit else 0
            empty_prediction_count += 1 if empty_prediction else 0
            first_token_is_eos_count += 1 if first_is_eos else 0

            prompt_flags = get_prompt_format_flags(ex["input"])
            for key, enabled in prompt_flags.items():
                if not enabled:
                    continue
                format_stats[key]["count"] += 1
                format_stats[key]["correct"] += 1 if metrics["exact_match"] == 1.0 else 0
                format_stats[key]["empty"] += 1 if empty_prediction else 0
                format_stats[key]["eos_first"] += 1 if first_is_eos else 0

            valid_label = None
            if _is_classification_task(task):
                pred_label = normalize_label_prediction(task, raw_pred)
                target_label = normalize_label_prediction(task, norm_tgt)
                valid_label = pred_label is not None
                if not valid_label:
                    invalid_label_count += 1
                elif target_label is not None:
                    key = f"pred_{pred_label}_when_{target_label}"
                    label_confusion_counts[key] = label_confusion_counts.get(key, 0) + 1

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

            target_len_tokens_list.append(target_len_tokens)
            pred_len_tokens_list.append(pred_len_tokens)
            length_ratio_list.append(length_ratio)
            abs_len_error_list.append(abs_len_error)
            repeat_1gram_list.append(repeat_metrics["repeat_1gram_rate"])
            repeat_2gram_list.append(repeat_metrics["repeat_2gram_rate"])
            unique_token_fraction_list.append(repeat_metrics["unique_token_fraction"])
            max_run_length_list.append(repeat_metrics["max_run_length"])

            parse_ok = None
            digit_length_pred = None
            digit_length_target = None
            if _is_numeric_task(task):
                parsed_pred, failure_reason = parse_numeric_prediction(raw_preds[idx])
                parsed_tgt, tgt_failure = parse_numeric_prediction(norm_tgt)
                parse_ok = failure_reason is None
                if not parse_ok:
                    total_parse_failures += 1
                    parse_failure_counts[failure_reason] = (
                        parse_failure_counts.get(failure_reason, 0) + 1
                    )
                elif tgt_failure is None:
                    abs_error = abs(parsed_pred - parsed_tgt)
                    rel_error = abs_error / max(abs(parsed_tgt), 1.0)
                    total_absolute_error += abs_error
                    total_numeric_samples += 1
                    parse_successes += 1
                    numeric_abs_errors.append(abs_error)
                    numeric_rel_errors.append(rel_error)
                else:
                    total_parse_failures += 1

                pred_token = extract_integer_token(raw_preds[idx])
                tgt_token = extract_integer_token(norm_tgt)
                digit_length_pred = digit_length_from_token(pred_token)
                digit_length_target = digit_length_from_token(tgt_token)
                if digit_length_pred is not None and digit_length_target is not None:
                    numeric_length_total += 1
                    if digit_length_pred != digit_length_target:
                        numeric_length_mismatch_count += 1

            if sample_target:
                sample_item = {
                    "prompt": ex["input"],
                    "target": tgt,
                    "expected_output": tgt,
                    "prediction": pred,
                    "target_len_tokens": target_len_tokens,
                    "pred_len_tokens": pred_len_tokens,
                    "stop_reason": stop_reason,
                    "first_token_is_eos": first_is_eos,
                    "empty_prediction": empty_prediction,
                    "valid_label": valid_label,
                    "digit_length_target": digit_length_target,
                    "digit_length_pred": digit_length_pred,
                    "parse_ok": parse_ok,
                    "repeat_1gram_rate": repeat_metrics["repeat_1gram_rate"],
                    "difficulty": difficulty,
                }
                if len(sampled_examples) < sample_target:
                    sampled_examples.append(sample_item)
                else:
                    idx = sample_rng.randint(0, sample_seen)
                    if idx < sample_target:
                        sampled_examples[idx] = sample_item
                sample_seen += 1
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
        parse_failures=total_parse_failures,
        n=total_examples,
        avg_gen_len=avg_gen_len,
        tokens_per_sec=tokens_per_sec,
        examples=collected_examples,
        sampled_examples=sampled_examples,
        correct=total_correct,
        tokens_generated=total_tokens,
        elapsed=total_time,
        target_len_tokens=target_len_tokens_list,
        pred_len_tokens=pred_len_tokens_list,
        length_ratio=length_ratio_list,
        abs_len_error=abs_len_error_list,
        stop_reason_counts=stop_reason_counts,
        eos_emitted=eos_emitted_count,
        repeat_1gram_rate=repeat_1gram_list,
        repeat_2gram_rate=repeat_2gram_list,
        unique_token_fraction=unique_token_fraction_list,
        max_run_length=max_run_length_list,
        parse_successes=parse_successes,
        parse_failure_counts=parse_failure_counts,
        numeric_abs_errors=numeric_abs_errors,
        numeric_rel_errors=numeric_rel_errors,
        empty_predictions=empty_prediction_count,
        first_token_is_eos_count=first_token_is_eos_count,
        invalid_label_count=invalid_label_count,
        label_confusion_counts=label_confusion_counts,
        format_stats=format_stats,
        numeric_length_mismatch_count=numeric_length_mismatch_count,
        numeric_length_total=numeric_length_total,
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
