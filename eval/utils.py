"""Shared helpers for algorithmic evaluation."""
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from data import AlgorithmicGenerator

SPACE_PATTERN = re.compile(r"\s+")


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EvalResult:
    condition: str
    task: str
    accuracy: float
    n: int
    avg_gen_len: float
    tokens_per_sec: float
    examples: List[Dict[str, str]]
    correct: int


def normalize_prediction(task: str, text: str) -> str:
    text = text.split("\n")[0]
    text = SPACE_PATTERN.sub(" ", text).strip()
    if task == "dyck":
        return text.lower()
    return text


def normalize_target(task: str, text: str) -> str:
    if task == "dyck":
        return text.lower().strip()
    return SPACE_PATTERN.sub(" ", text).strip()


def select_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def decode_generated(tokenizer, input_ids, attention_mask, output_ids, task: str) -> List[str]:
    preds: List[str] = []
    for inp, mask, out in zip(input_ids, attention_mask, output_ids):
        prompt_len = int(mask.sum().item())
        gen_tokens = out[prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        preds.append(normalize_prediction(task, text))
    return preds


def score_predictions(task: str, preds: List[str], targets: List[str]) -> Tuple[int, int]:
    correct = 0
    for pred, tgt in zip(preds, targets):
        if normalize_prediction(task, pred) == normalize_target(task, tgt):
            correct += 1
    return correct, len(targets)


def generate_outputs(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: torch.device,
    use_autocast: bool,
    task: str,
) -> Tuple[List[str], int, float]:
    tokenized = batch_tokenize(tokenizer, prompts, device)
    input_ids = tokenized["input_ids"]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    start = time.time()
    with torch.no_grad():
        if use_autocast:
            with torch.cuda.amp.autocast():
                outputs = model.generate(**tokenized, **gen_kwargs)
        else:
            outputs = model.generate(**tokenized, **gen_kwargs)
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
) -> EvalResult:
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
        preds, tok_count, elapsed = generate_outputs(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            device=device,
            use_autocast=use_autocast,
            task=task,
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
    )
