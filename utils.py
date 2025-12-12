"""Utility helpers shared across evaluation entrypoints.

This module centralizes model/tokenizer loading, generation helpers,
random seeding, and long-context data generation so that individual
scripts remain lightweight.
"""
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from model import BaselineTransformer


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
) -> Tuple[BaselineTransformer, any, any]:
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


def generate_text(
    model: BaselineTransformer,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 32,
    generation_kwargs: Optional[Dict] = None,
) -> str:
    """Generate a continuation for a single prompt."""
    generation_kwargs = generation_kwargs or {}
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_len = input_ids.shape[1]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        **generation_kwargs,
    )
    generated_tokens = output_ids[0, input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def batch_generate(
    model: BaselineTransformer,
    tokenizer,
    prompts: Iterable[str],
    device: torch.device,
    max_new_tokens: int = 32,
) -> List[str]:
    """Vectorized generation helper for multiple prompts."""
    tokenized = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    input_ids = tokenized["input_ids"]
    output_ids = model.generate(
        **tokenized,
        max_new_tokens=max_new_tokens,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
    prompt_lengths = tokenized.get("attention_mask", torch.ones_like(input_ids)).sum(dim=1)
    generations: List[str] = []
    for i, prompt_len in enumerate(prompt_lengths):
        tail = output_ids[i, prompt_len:]
        generations.append(tokenizer.decode(tail, skip_special_tokens=True).strip())
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
