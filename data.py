"""
Split Curriculum Data Loading
=============================
Implements the ReLSM training strategy:

PHASE 1 - Algorithmic Grokking:
  - Synthetic logic, math, code
  - Infinite procedural generation (no overfitting risk)
  - Massive epochs (1000+) on small model
  - Goal: Force recursive core to learn generalizable algorithms

PHASE 2 - Language Generalization:
  - TinyStories, filtered web text
  - Standard epochs (1-3 passes)
  - Goal: Map natural language into the learned logic space

Also includes:
  - OOD length testing for algorithmic tasks
"""

import random
import math
from functools import partial
from typing import Any, List, Dict, Optional, Tuple, Iterator, Callable, Iterable, Union, Sequence
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler, get_worker_info
from torch.nn.utils.rnn import pad_sequence


# =============================================================================
# ALGORITHMIC DATA GENERATORS (Phase 1)
# =============================================================================

TASK_DIFFICULTY_CONFIG = {
    "addition": {
        "phases": [
            (0.00, 0.15, {"digit_range": (1, 2)}),
            (0.15, 0.40, {"digit_range": (1, 3)}),
            (0.40, 0.70, {"digit_range": (1, 4)}),
            (0.70, 0.90, {"digit_range": (1, 5)}),
            (0.90, 1.00, {"digit_range": (3, 6)}),
        ],
        "train_max": 5,
        "ood_start": 6,
    },
    "multiplication": {
        "phases": [
            (0.00, 0.20, {"digit_range": (1, 2)}),
            (0.20, 0.50, {"digit_range": (1, 2)}),
            (0.50, 0.80, {"digit_range": (1, 3)}),
            (0.80, 1.00, {"digit_range": (2, 3)}),
        ],
        "train_max": 3,
        "ood_start": 4,
    },
    "successor": {
        # INCREASED DIFFICULTY: Ramps up to 32 digits to force long-range carry propagation.
        "phases": [
            (0.00, 0.10, {"digit_range": (1, 4)}),
            (0.10, 0.25, {"digit_range": (2, 8)}),
            (0.25, 0.40, {"digit_range": (4, 12)}),
            (0.40, 0.55, {"digit_range": (8, 16)}),
            (0.55, 0.70, {"digit_range": (12, 20)}),
            (0.70, 0.85, {"digit_range": (16, 26)}),
            (0.85, 1.00, {"digit_range": (20, 32)}),
        ],
        "train_max": 32,
        "ood_start": 33,
    },
    "copy": {
        "phases": [
            (0.00, 0.10, {"length_range": (1, 3)}),
            (0.10, 0.20, {"length_range": (2, 4)}),
            (0.20, 0.35, {"length_range": (3, 6)}),
            (0.35, 0.50, {"length_range": (4, 8)}),
            (0.50, 0.65, {"length_range": (6, 10)}),
            (0.65, 0.80, {"length_range": (8, 12)}),
            (0.80, 1.00, {"length_range": (10, 12)}),
        ],
        "train_max": 12,
        "ood_start": 16,
    },
    "reverse": {
        "phases": [
            (0.00, 0.10, {"length_range": (1, 3)}),
            (0.10, 0.20, {"length_range": (2, 4)}),
            (0.20, 0.35, {"length_range": (3, 6)}),
            (0.35, 0.50, {"length_range": (4, 8)}),
            (0.50, 0.65, {"length_range": (6, 10)}),
            (0.65, 0.80, {"length_range": (8, 12)}),
            (0.80, 1.00, {"length_range": (10, 12)}),
        ],
        "train_max": 12,
        "ood_start": 16,
    },
    "parity": {
        "phases": [
            (0.00, 0.15, {"length_range": (2, 5)}),
            (0.15, 0.30, {"length_range": (4, 8)}),
            (0.30, 0.60, {"length_range": (6, 10)}),
            (0.60, 1.00, {"length_range": (8, 12)}),
        ],
        "train_max": 12,
        "ood_start": 16,
    },
    "dyck": {
        "phases": [
            (0.00, 0.15, {"depth_range": (1, 2)}),
            (0.15, 0.40, {"depth_range": (2, 3)}),
            (0.40, 0.70, {"depth_range": (2, 4)}),
            (0.70, 1.00, {"depth_range": (3, 4)}),
        ],
        "train_max": 4,
        "ood_start": 5,
    },
    "mod_add": {
        "phases": [
            (0.00, 0.20, {"mod_range": (17, 50)}),
            (0.20, 0.50, {"mod_range": (50, 200)}),
            (0.50, 0.80, {"mod_range": (100, 997)}),
            (0.80, 1.00, {"mod_range": (500, 997)}),
        ],
        "train_max": 997,
        "ood_start": 2003,
    },
    "chain": {
        "phases": [
            (0.00, 0.20, {"length_range": (1, 2), "digit_range": (2, 4)}),
            (0.20, 0.50, {"length_range": (2, 3), "digit_range": (4, 12)}),
            (0.50, 0.80, {"length_range": (3, 5), "digit_range": (12, 24)}),
            (0.80, 1.00, {"length_range": (5, 8), "digit_range": (24, 32)}),
        ],
        "train_max": 8,
        "ood_start": 9,
    },
    "compare": {
        "phases": [
            (0.00, 0.20, {"digit_range": (1, 2)}),
            (0.20, 0.60, {"digit_range": (1, 3)}),
            (0.60, 0.90, {"digit_range": (1, 4)}),
            (0.90, 1.00, {"digit_range": (3, 4)}),
        ],
        "train_max": 4,
        "ood_start": 5,
    },
}


def get_difficulty_params(task: str, progress: float) -> Dict[str, Tuple[int, int]]:
    """Get sampling params for current training progress."""
    config = TASK_DIFFICULTY_CONFIG.get(task)
    if not config:
        return {}

    for start, end, params in config["phases"]:
        if start <= progress < end:
            return params
    return config["phases"][-1][2]


TASK_BASE_WEIGHTS = {
    "addition": 2.0,
    "multiplication": 2.5,
    "reverse": 1.5,
    "successor": 1.0,
    "mod_add": 1.0,
    "copy": 1.0,
    "parity": 1.0,
    "chain": 0.8,
    "dyck": 0.7,
    "compare": 0.7,
}


def get_task_weight(task: str, progress: float) -> float:
    """Get task sampling weight. Decay easy tasks after 50%."""
    base = TASK_BASE_WEIGHTS.get(task, 1.0)
    if task in ["dyck", "compare", "chain"] and progress > 0.5:
        return base * 0.5
    return base


def sample_difficulty_smooth(
    progress: float,
    easy_mix_frac: float = 0.2,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Sample difficulty with smooth phase transitions and guaranteed easy mixing.

    Args:
        progress: Training progress from 0.0 to 1.0
        easy_mix_frac: Fraction of samples that should be easy (0.0-0.3 difficulty)
        rng: Random number generator

    Returns:
        Difficulty value from 0.0 to 1.0
    """
    rng = rng or random

    if rng.random() < easy_mix_frac:
        return rng.uniform(0.0, 0.3)

    boundaries = [0.15, 0.35, 0.60]
    target_difficulties = [0.25, 0.45, 0.70, 1.0]

    result = target_difficulties[0]

    for i, boundary in enumerate(boundaries):
        transition_width = 0.05
        blend = 1.0 / (1.0 + math.exp(-(progress - boundary) / (transition_width / 4)))
        result = result * (1 - blend) + target_difficulties[i + 1] * blend

    noise = rng.gauss(0, 0.05)
    result = max(0.0, min(1.0, result + noise))

    return result


def sample_difficulty_warmup_ramp(
    progress: float,
    warmup_frac: float = 0.1,
    hold_frac: float = 0.2,
    easy_mix_frac: float = 0.2,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Stay easy during warmup, hold, then ramp difficulty.

    This is more conservative than smooth_phased - good for harder tasks.
    """
    rng = rng or random

    if rng.random() < easy_mix_frac:
        return rng.uniform(0.0, 0.3)

    if progress < warmup_frac:
        return rng.uniform(0.1, 0.25)
    if progress < warmup_frac + hold_frac:
        return rng.uniform(0.2, 0.4)

    ramp_progress = (progress - warmup_frac - hold_frac) / (1 - warmup_frac - hold_frac)
    base = 0.3 + 0.7 * ramp_progress
    noise = rng.gauss(0, 0.05)
    return max(0.0, min(1.0, base + noise))

# =============================================================================
# TASK WEIGHTING LOGIC
# =============================================================================

TASK_BASE_WEIGHTS = {
    "addition": 2.0,
    "multiplication": 2.5,
    "reverse": 1.5,
    "successor": 1.0,
    "mod_add": 1.0,
    "copy": 1.0,
    "parity": 1.0,
    "chain": 0.8,
    "dyck": 0.7,
    "compare": 0.7,
}


def get_task_weight(task: str, progress: float) -> float:
    """Get task sampling weight. Decay easy tasks after 50%."""
    base = TASK_BASE_WEIGHTS.get(task, 1.0)
    # Decay simple algorithmic tasks in the second half of training
    # to focus capacity on harder tasks or language data.
    if task in ["dyck", "compare", "chain"] and progress > 0.5:
        return base * 0.5
    return base


# mod_add's prompt templates, in the order rng.choice picks from (shared with ModTableDataset)
MOD_ADD_TEMPLATES = (
    "Compute ({a} + {b}) mod {mod}. Answer:",
    "What is ({a} + {b}) modulo {mod}? Answer:",
    "Add {a} and {b}, then take mod {mod}. Answer:",
    "({a} plus {b}) % {mod} equals. Answer:",
)


class AlgorithmicGenerator:
    """
    Generates synthetic algorithmic tasks for grokking.
    Infinite, procedural - train for 1000s of epochs without overfitting.
    """

    @staticmethod
    def _difficulty_bucket(difficulty: float) -> str:
        """Map difficulty scalar to a coarse bucket."""
        d = min(max(difficulty, 0.0), 1.0)
        if d < 0.3:
            return "easy"
        if d < 0.7:
            return "medium"
        return "hard"

    @classmethod
    def _digit_range(cls, difficulty: float, *, cap: int = 8) -> Tuple[int, int]:
        bucket = cls._difficulty_bucket(difficulty)
        if bucket == "easy":
            low, high = 2, 3
        elif bucket == "medium":
            low, high = 4, 6
        else:
            low, high = 7, 8

        high = min(high, cap)
        low = min(low, high)
        return low, high

    @classmethod
    def _length_range(cls, difficulty: float, *, easy: Tuple[int, int], medium: Tuple[int, int], hard: Tuple[int, int]) -> Tuple[int, int]:
        bucket = cls._difficulty_bucket(difficulty)
        if bucket == "easy":
            return easy
        if bucket == "medium":
            return medium
        return hard

    @staticmethod
    def _sample_int_with_digits(rng: random.Random, digits: int) -> int:
        if digits <= 1:
            return rng.randint(0, 9)
        low = 10 ** (digits - 1)
        high = 10**digits - 1
        return rng.randint(low, high)

    @staticmethod
    def _sample_int_with_digit_bounds(
        rng: random.Random, digits: int, low_digit: int, high_digit: int
    ) -> int:
        """Sample an integer with a fixed digit count and per-digit bounds."""
        digits = max(1, digits)
        low_digit = max(0, low_digit)
        high_digit = min(9, max(low_digit, high_digit))

        first_low = max(low_digit, 1) if high_digit > 0 else 0
        first = rng.randint(first_low, high_digit)
        remaining = [rng.randint(low_digit, high_digit) for _ in range(digits - 1)]
        all_digits = [first] + remaining
        return int("".join(map(str, all_digits)))

    @staticmethod
    def _choose_prompt(rng: random.Random, templates: List[str], **kwargs) -> str:
        template = rng.choice(templates)
        return template.format(**kwargs)
    
    @staticmethod
    def modular_arithmetic(
        mod: Optional[int] = 97,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        mod_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Modular addition: (a + b) mod p"""
        rng = rng or random
        digits_low, digits_high = AlgorithmicGenerator._digit_range(difficulty)
        operand_digits = rng.randint(digits_low, digits_high)
        modulus_digits = max(operand_digits, digits_low)
        sampled_mod = AlgorithmicGenerator._sample_int_with_digits(
            rng, min(modulus_digits, 5)
        )
        if mod_range is not None:
            mod_val = rng.randint(mod_range[0], mod_range[1])
        elif mod is None:
            mod_val = max(3, sampled_mod)
        else:
            mod_val = max(3, mod)
        a = rng.randint(0, mod_val - 1)
        b = rng.randint(0, mod_val - 1)
        result = (a + b) % mod_val

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            list(MOD_ADD_TEMPLATES),
            a=a,
            b=b,
            mod=mod_val,
        )

        return {
            "text": f"{prompt}{result}",
            "input": prompt,
            "target": str(result),
            "task": "mod_add",
        }

    @staticmethod
    def parity(
        length: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        length_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Parity of binary string"""
        rng = rng or random
        # NOTE: For OOD evaluation we keep training length fixed at 8 by default.
        if length is not None:
            seq_len = length
        elif length_range is not None:
            seq_len = rng.randint(length_range[0], length_range[1])
        else:
            seq_len = 8
        bits = [rng.randint(0, 1) for _ in range(seq_len)]
        parity = sum(bits) % 2
        bit_str = "".join(map(str, bits))

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "parity({bits}). Answer:",
                "Is the number of ones in {bits} even (0) or odd (1)? Answer:",
                "Compute the parity bit for {bits}. Answer:",
                "Parity of {bits}? Answer:",
            ],
            bits=bit_str,
        )

        return {
            "text": f"{prompt}{parity}",
            "input": prompt,
            "target": str(parity),
            "task": "parity",
        }

    @staticmethod
    def addition(
        max_digits: int = 4,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        digits: Optional[int] = None,
        digit_bounds: Optional[Tuple[int, int]] = None,
        digit_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Multi-digit addition.

        The ``digits`` and ``digit_bounds`` arguments are optional overrides used
        by evaluation scripts to create deterministic OOD settings while keeping
        the training defaults intact.
        """
        # NOTE: For OOD addition eval we cap training digits at 4.
        rng = rng or random
        bucket = AlgorithmicGenerator._difficulty_bucket(difficulty)
        if bucket == "easy":
            default_digit_range = (1, 2)
            default_digit_bounds = (0, 4)  # low carry pressure
        elif bucket == "medium":
            default_digit_range = (2, 3)
            default_digit_bounds = (0, 9)  # neutral carry pressure
        else:
            default_digit_range = (4, 4)
            default_digit_bounds = (5, 9)  # high carry pressure

        active_range = digit_range if digit_range is not None else default_digit_range
        if digits is None:
            digits = rng.randint(active_range[0], min(active_range[1], max_digits))
        active_bounds = digit_bounds if digit_bounds is not None else default_digit_bounds
        a = AlgorithmicGenerator._sample_int_with_digit_bounds(
            rng, digits, *active_bounds
        )
        b = AlgorithmicGenerator._sample_int_with_digit_bounds(
            rng, digits, *active_bounds
        )
        result = a + b

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "Compute: {a} + {b}. Answer:",
                "What is {a} plus {b}? Answer:",
                "Add {a} and {b}. Answer:",
                "Sum {a} with {b}. Answer:",
                "{a} + {b} equals. Answer:",
            ],
            a=a,
            b=b,
        )

        return {
            "text": f"{prompt}{result}",
            "input": prompt,
            "target": str(result),
            "task": "addition",
        }

    @staticmethod
    def multiplication(
        max_val: int = 99,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        digits: Optional[int] = None,
        digit_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Two-digit multiplication"""
        rng = rng or random
        default_range = AlgorithmicGenerator._digit_range(difficulty, cap=4)
        active_range = digit_range if digit_range is not None else default_range
        if digits is None:
            digits = rng.randint(active_range[0], active_range[1])
        if digits is not None and max_val < 10**digits - 1:
            max_val = 10**digits - 1
        high = min(max_val, 10**digits - 1)
        low = max(2, 10 ** (digits - 1))
        if low > high:
            low = 2
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        result = a * b

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "Multiply: {a} * {b}. Answer:",
                "What is {a} times {b}? Answer:",
                "Product of {a} and {b}. Answer:",
                "Compute {a} multiplied by {b}. Answer:",
            ],
            a=a,
            b=b,
        )

        return {
            "text": f"{prompt}{result}",
            "input": prompt,
            "target": str(result),
            "task": "multiplication",
        }

    @staticmethod
    def copy_sequence(
        length: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        length_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Copy task - tests basic sequence modeling"""
        rng = rng or random
        default_range = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(4, 6),
            medium=(7, 10),
            hard=(11, 14),
        )
        active_range = length_range if length_range is not None else default_range
        seq_len = length if length is not None else rng.randint(active_range[0], active_range[1])
        seq = [rng.randint(0, 9) for _ in range(seq_len)]
        seq_str = " ".join(map(str, seq))

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "Copy the sequence {seq}. Answer:",
                "Repeat the sequence {seq}. Answer:",
                "Copy this list exactly: {seq}. Answer:",
                "Write the same numbers again: {seq}. Answer:",
            ],
            seq=seq_str,
        )

        return {
            "text": f"{prompt}{seq_str}",
            "input": prompt,
            "target": seq_str,
            "task": "copy",
        }

    @staticmethod
    def reverse_sequence(
        length: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        length_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Reverse task - tests sequential reasoning"""
        rng = rng or random
        default_range = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(4, 6),
            medium=(7, 10),
            hard=(11, 14),
        )
        active_range = length_range if length_range is not None else default_range
        seq_len = length if length is not None else rng.randint(active_range[0], active_range[1])
        seq = [rng.randint(0, 9) for _ in range(seq_len)]
        seq_str = " ".join(map(str, seq))
        rev_str = " ".join(map(str, reversed(seq)))

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "Reverse: {seq}. Answer:",
                "Write {seq} backwards. Answer:",
                "Reverse the order of {seq}. Answer:",
                "Flip the sequence {seq}. Answer:",
            ],
            seq=seq_str,
        )

        return {
            "text": f"{prompt}{rev_str}",
            "input": prompt,
            "target": rev_str,
            "task": "reverse",
        }

    @staticmethod
    def dyck_language(
        max_depth: int = 4,
        length: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        force_valid: Optional[bool] = None,
        depth_range: Optional[Tuple[int, int]] = None,
        length_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Dyck language (balanced parentheses) - tests stack operations"""
        rng = rng or random
        default_range = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(6, 10),
            medium=(10, 18),
            hard=(18, 30),
        )
        active_range = length_range if length_range is not None else default_range
        seq_len = length if length is not None else rng.randint(active_range[0], active_range[1])
        depth_bucket = AlgorithmicGenerator._difficulty_bucket(difficulty)
        bucket = AlgorithmicGenerator._difficulty_bucket(difficulty)
        if depth_range is not None:
            max_depth = rng.randint(depth_range[0], depth_range[1])
        if depth_bucket == "easy":
            depth_cap = min(2, max_depth)
        elif depth_bucket == "medium":
            depth_cap = min(3, max_depth)
        else:
            depth_cap = max_depth

        # Generate valid Dyck sequence
        seq = []
        depth = 0
        for _ in range(seq_len):
            if depth == 0:
                seq.append("(")
                depth += 1
            elif depth >= depth_cap:
                seq.append(")")
                depth -= 1
            else:
                if rng.random() < 0.5:
                    seq.append("(")
                    depth += 1
                else:
                    seq.append(")")
                    depth -= 1

        # Close remaining
        while depth > 0:
            seq.append(")")
            depth -= 1

        seq_str = "".join(seq)

        def _is_balanced(s: str) -> bool:
            depth = 0
            for ch in s:
                depth += 1 if ch == "(" else -1
                if depth < 0:
                    return False
            return depth == 0

        # Optionally corrupt to create invalid strings for decision task
        is_valid = force_valid if force_valid is not None else (rng.random() < 0.5)
        if not is_valid:
            def _corruption_idx(length: int) -> int:
                if length <= 1:
                    return 0
                if bucket == "hard":
                    # Bias toward early prefix violations at higher difficulty
                    return min(length - 1, int(rng.random() ** 2 * length))
                return rng.randint(0, length - 1)

            corruption_type = rng.choice(["drop", "flip", "append"])
            seq_list = list(seq_str)
            if corruption_type == "drop" and len(seq_list) > 1:
                drop_idx = _corruption_idx(len(seq_list))
                seq_list.pop(drop_idx)
            elif corruption_type == "flip":
                flip_idx = _corruption_idx(len(seq_list))
                seq_list[flip_idx] = "(" if seq_list[flip_idx] == ")" else ")"
            else:
                seq_list.append(")")

            seq_str = "".join(seq_list)
            if _is_balanced(seq_str):
                seq_str += "("  # force imbalance

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "Is '{seq}' balanced? Answer:",
                "Does {seq} form a correct parentheses string? Answer:",
                "Dyck check for {seq}. Answer:",
                "Check Dyck validity for: {seq}. Answer:",
            ],
            seq=seq_str,
        )

        return {
            "text": f"{prompt}{'yes' if is_valid else 'no'}",
            "input": prompt,
            "target": "yes" if is_valid else "no",
            "task": "dyck",
        }

    @staticmethod
    def chain_arithmetic(
        n_ops: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        operand_high: Optional[int] = None,
        allow_negative: Optional[bool] = None,
        length_range: Optional[Tuple[int, int]] = None,
        digits: Optional[int] = None,
        digit_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Carry propagation chains for long-range state maintenance."""
        rng = rng or random
        default_range = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(1, 2),
            medium=(2, 4),
            hard=(3, 6),
        )
        bucket = AlgorithmicGenerator._difficulty_bucket(difficulty)
        op_counts = length_range if length_range is not None else default_range
        sampled_ops = rng.randint(op_counts[0], op_counts[1])
        ops = max(1, n_ops if n_ops is not None else sampled_ops)

        if bucket == "easy":
            default_digit_range = (2, 3)
            suffix_range = (1, 2)
            small_max = 3
            one_prob = 0.8
        elif bucket == "medium":
            default_digit_range = (3, 5)
            suffix_range = (2, 3)
            small_max = 5
            one_prob = 0.7
        else:
            default_digit_range = (5, 8)
            suffix_range = (3, 5)
            small_max = 9
            one_prob = 0.6

        active_digit_range = digit_range if digit_range is not None else default_digit_range

        if operand_high is not None:
            small_max = max(2, operand_high)
        if allow_negative:
            allow_negative = False

        if digits is None:
            total_digits = rng.randint(active_digit_range[0], active_digit_range[1])
        else:
            total_digits = max(1, digits)
        max_suffix = min(total_digits, suffix_range[1])
        min_suffix = min(suffix_range[0], max_suffix)
        suffix_len = rng.randint(min_suffix, max_suffix)
        prefix_len = total_digits - suffix_len
        if prefix_len > 0:
            prefix_val = AlgorithmicGenerator._sample_int_with_digits(rng, prefix_len)
            prefix_str = str(prefix_val)
        else:
            prefix_str = ""

        base_str = f"{prefix_str}{'9' * suffix_len}"
        base_val = int(base_str)

        addends: List[int] = []
        for _ in range(ops):
            if rng.random() < one_prob:
                addends.append(1)
            else:
                addends.append(rng.randint(2, small_max))

        chain_terms = [base_str] + [str(addend) for addend in addends]
        chain_expr = "->".join(chain_terms)
        result = base_val + sum(addends)
        prompt = f"Chain {chain_expr}. Answer:"

        return {
            "text": f"{prompt}{result}",
            "input": prompt,
            "target": str(result),
            "task": "chain",
        }

    @staticmethod
    def comparison(
        max_val: int = 1000,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        digits: Optional[int] = None,
        digit_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Number comparison"""
        rng = rng or random
        default_range = AlgorithmicGenerator._digit_range(difficulty, cap=6)
        active_range = digit_range if digit_range is not None else default_range
        if digits is None:
            digits = rng.randint(active_range[0], active_range[1])
        if digits is not None and max_val < 10**digits - 1:
            max_val = 10**digits - 1
        high = min(max_val, 10**digits - 1)
        low = max(1, 10 ** (digits - 1))
        if low > high:
            low = 1
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        result = ">" if a > b else ("<" if a < b else "=")

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "Compare {a} and {b}. Answer:",
                "Which relation holds between {a} and {b} (>, <, or =)? Answer:",
                "Compare {a} to {b} (>, <, or =). Answer:",
                "Between {a} and {b}, choose the relation. Answer:",
            ],
            a=a,
            b=b,
        )

        return {
            "text": f"{prompt}{result}",
            "input": prompt,
            "target": result,
            "task": "compare",
        }

    @staticmethod
    def successor(
        max_val: int = 1000,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
        digits: Optional[int] = None,
        digit_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        """Successor function"""
        rng = rng or random
        default_range = AlgorithmicGenerator._digit_range(difficulty, cap=6)
        active_range = digit_range if digit_range is not None else default_range
        if digits is None:
            digits = rng.randint(active_range[0], active_range[1])
        if digits is not None and max_val < 10**digits - 1:
            max_val = 10**digits - 1
        max_n = min(max_val, 10**digits - 1)
        n = rng.randint(0, max_n)

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "succ({n}). Answer:",
                "What is the successor of {n}? Answer:",
                "Next integer after {n} is. Answer:",
                "Increment {n} by one. Answer:",
            ],
            n=n,
        )
        return {
            "text": f"{prompt}{n + 1}",
            "input": prompt,
            "target": str(n + 1),
            "task": "successor",
        }

    @classmethod
    def generate_batch(
        cls,
        n: int,
        tasks: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
        difficulty: Optional[float] = None,
        min_difficulty: float = 0.0,
    ) -> List[Dict]:
        """Generate mixed batch of algorithmic tasks."""
        rng = rng or random
        generators = cls._get_generators()

        if difficulty is None:
            difficulty = 0.5

        if tasks is None:
            tasks = list(generators.keys())

        return [
            cls.generate_example(
                tasks=tasks,
                rng=rng,
                generators=generators,
                difficulty=difficulty,
                min_difficulty=min_difficulty,
            )
            for _ in range(n)
        ]

    @classmethod
    def generate_example(
        cls,
        tasks: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
        generators: Optional[Dict[str, Callable]] = None,
        difficulty: Optional[float] = None,
        min_difficulty: float = 0.0,
        progress: Optional[float] = None,
        difficulty_schedule: str = "linear",
        task_weighting: str = "uniform",
    ) -> Dict:
        """Generate a single example honoring the difficulty schedule."""
        rng = rng or random
        if difficulty is None:
            difficulty = 0.5
        if generators is None:
            generators = cls._get_generators()

        if tasks is None:
            tasks = list(generators.keys())

        if task_weighting == "adaptive" and progress is not None:
            weights = [get_task_weight(task, progress) for task in tasks]
            task = rng.choices(tasks, weights=weights, k=1)[0]
        else:
            task = rng.choice(tasks)

        if difficulty_schedule == "phased" and progress is not None:
            params = get_difficulty_params(task, progress)
            difficulty_value = progress
        else:
            params = get_difficulty_params(task, difficulty)
            if difficulty_schedule == "fixed":
                difficulty_value = 0.5
            else:
                difficulty_value = difficulty
        difficulty_value = min(max(difficulty_value, min_difficulty), 1.0)

        kwargs: Dict[str, int | Tuple[int, int]] = {}
        if "digit_range" in params:
            kwargs["digits"] = rng.randint(params["digit_range"][0], params["digit_range"][1])
        if "length_range" in params:
            sampled_length = rng.randint(params["length_range"][0], params["length_range"][1])
            if task in {"copy", "reverse", "parity"}:
                kwargs["length"] = sampled_length
            elif task == "chain":
                kwargs["n_ops"] = sampled_length
            elif task == "dyck":
                kwargs["length"] = sampled_length
        if "depth_range" in params:
            kwargs["max_depth"] = rng.randint(params["depth_range"][0], params["depth_range"][1])
        if "mod_range" in params:
            kwargs["mod"] = rng.randint(params["mod_range"][0], params["mod_range"][1])

        return generators[task](rng=rng, difficulty=difficulty_value, **kwargs)

    @classmethod
    def _get_generators(cls) -> Dict[str, Callable]:
        return {
            "mod_add": cls.modular_arithmetic,
            "parity": cls.parity,
            "addition": cls.addition,
            "multiplication": cls.multiplication,
            "copy": cls.copy_sequence,
            "reverse": cls.reverse_sequence,
            "dyck": cls.dyck_language,
            "chain": cls.chain_arithmetic,
            "compare": cls.comparison,
            "successor": cls.successor,
        }


class AlgorithmicDataset(IterableDataset):
    """
    Dataset for Phase 1: Algorithmic Grokking.
    Generates fresh data each epoch for true procedural sampling.
    Uses token-based dynamic batching with length bucketing to keep per-step
    token counts stable while minimizing padding overhead.
    """

    def __init__(
        self,
        tokenizer,
        num_examples: int = 100000,
        max_seq_len: int = 128,
        base_batch_size: int = 1,
        max_batch_multiplier: int = 4,
        tasks: Optional[List[str]] = None,
        seed: Optional[int] = None,
        difficulty_value=None,
        difficulty_fn: Optional[Callable[[str], float]] = None,
        weighting_fn: Optional[Callable[[str], float]] = None,
        weighting_adjust_fn: Optional[Callable[[Sequence[str], List[float]], List[float]]] = None,
        difficulty_schedule: str = "smooth",
        task_weighting: str = "uniform",
        total_tokens: Optional[int] = None,
        easy_mix_frac: float = 0.2,
        min_difficulty: float = 0.0,
        include_lengths: bool = False,
        length_bucket_boundaries: Optional[Sequence[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        self.base_batch_size = max(1, int(base_batch_size))
        self.max_batch_multiplier = max(1, int(max_batch_multiplier))
        self.tasks = tasks
        self.seed = seed
        self.difficulty_value = difficulty_value
        self.difficulty_fn = difficulty_fn
        self.weighting_fn = weighting_fn
        self.weighting_adjust_fn = weighting_adjust_fn
        self.difficulty_schedule = difficulty_schedule
        self.task_weighting = task_weighting
        self.easy_mix_frac = easy_mix_frac
        # Enforce a minimum difficulty floor across schedules/samplers.
        self.min_difficulty = min_difficulty
        self.include_lengths = include_lengths
        self.tokens_seen = 0
        # Target a fixed token budget per batch while capping the number of
        # examples to avoid metadata overhead from very short sequences.
        self._target_batch_tokens = self.base_batch_size * self.max_seq_len
        self._max_batch_size = self.base_batch_size * self.max_batch_multiplier
        # Default to splitting into 3 roughly equal buckets.
        if length_bucket_boundaries is None:
            first = max(1, self.max_seq_len // 3)
            second = max(first + 1, (2 * self.max_seq_len) // 3)
            length_bucket_boundaries = (first, second)
        self._length_bucket_boundaries = tuple(sorted(set(length_bucket_boundaries)))
        self._token_budget = max(
            1,
            total_tokens
            if total_tokens is not None
            else self.num_examples * max(self.max_seq_len - 1, 1),
        )
        self._worker_token_budget = self._token_budget

    def _encode_text(self, text: str) -> List[int]:
        """Encode text with the tokenizer to align training and evaluation."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def _encode_length(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(self.tokenizer(text, truncation=True)["input_ids"])

    def __len__(self):
        return self.num_examples

    def _length_bucket_index(self, seq_len: int) -> int:
        for idx, boundary in enumerate(self._length_bucket_boundaries):
            if seq_len <= boundary:
                return idx
        return len(self._length_bucket_boundaries)

    def __iter__(self) -> "AlgorithmicIterator":
        # An object with explicit state rather than a generator, so a checkpoint can save where it is (#154).
        # Same calls in the same order as the generator it replaces.
        it = AlgorithmicIterator(self)
        pending, self._pending_state = getattr(self, "_pending_state", None), None
        if pending is not None:
            it.load_state_dict(pending)
        self.last_iter = it
        return it

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("last_iter", None)  # DataLoader workers get the dataset, never an iterator of the main process
        state.pop("_pending_state", None)
        return state

    def exact_dataset_state(self) -> Dict[str, Any]:
        """The counters every iterator of this dataset shares (saved once per dataset)."""
        return {"tokens_seen": int(self.tokens_seen), "worker_token_budget": float(self._worker_token_budget)}

    def load_exact_dataset_state(self, state: Dict[str, Any]) -> None:
        self.tokens_seen = int(state["tokens_seen"])
        self._worker_token_budget = float(state["worker_token_budget"])

    def _sample_task(
        self,
        rng: random.Random,
        progress: float,
        generators: Dict[str, Callable],
    ) -> str:
        tasks = self.tasks or list(generators.keys())
        if self.weighting_fn is not None:
            weights = [self.weighting_fn(task) for task in tasks]
            if self.weighting_adjust_fn is not None:
                weights = list(self.weighting_adjust_fn(tasks, weights))
            if sum(weights) <= 0:
                weights = [1.0] * len(tasks)
            return rng.choices(tasks, weights=weights, k=1)[0]
        if self.task_weighting == "adaptive":
            weights = [get_task_weight(task, progress) for task in tasks]
            return rng.choices(tasks, weights=weights, k=1)[0]
        return rng.choice(tasks)

    def _sample_difficulty(self, rng: random.Random, progress: float) -> float:
        """Sample difficulty based on schedule type."""
        if self.difficulty_value is not None:
            try:
                with self.difficulty_value.get_lock():
                    value = float(self.difficulty_value.value)
            except Exception:
                value = 0.0
            return min(max(value, self.min_difficulty), 1.0)

        if self.difficulty_schedule == "fixed":
            return max(0.5, self.min_difficulty)

        if self.difficulty_schedule == "linear":
            upper = min(1.0, 0.5 + 0.5 * progress)
            upper = max(upper, self.min_difficulty)
            return rng.uniform(self.min_difficulty, upper)

        if self.difficulty_schedule == "phased":
            return min(max(progress, self.min_difficulty), 1.0)

        if self.difficulty_schedule == "smooth":
            return max(
                sample_difficulty_smooth(
                    progress,
                    easy_mix_frac=self.easy_mix_frac,
                    rng=rng,
                ),
                self.min_difficulty,
            )

        if self.difficulty_schedule == "warmup_ramp":
            return max(
                sample_difficulty_warmup_ramp(
                    progress,
                    warmup_frac=0.1,
                    hold_frac=0.2,
                    easy_mix_frac=self.easy_mix_frac,
                    rng=rng,
                ),
                self.min_difficulty,
            )

        return max(sample_difficulty_smooth(progress, rng=rng), self.min_difficulty)


class AlgorithmicIterator:
    """AlgorithmicDataset.__iter__ as an object whose position can be saved and restored (exact resume, #154).

    The body is the old generator's, cut at its yields: phase "unstarted" (nothing drawn yet: seeding and the
    worker's token budget happen on the first next(), as they did in the generator), "generate" (examples
    index < num_examples left), "flush" (the length buckets still to yield, from flush_index), "done".
    Dataset-level counters (tokens_seen, _worker_token_budget) live on the dataset and are shared by every
    iterator of it, as before.
    """

    def __init__(self, dataset: "AlgorithmicDataset"):
        self.dataset = dataset
        self.phase = "unstarted"
        self.rng: Optional[random.Random] = None
        self.seeded = dataset.seed is not None
        self.worker_id = 0
        self.num_workers = 1
        self.index = 0
        self.flush_index = 0
        self.buffers: List[List[Dict[str, Any]]] = []
        self.buffer_max_len: List[int] = []
        self._generators: Optional[Dict[str, Callable]] = None

    def __iter__(self) -> "AlgorithmicIterator":
        return self

    def _start(self) -> None:
        ds = self.dataset
        worker_info = get_worker_info()
        if ds.seed is not None:
            seed = ds.seed + (worker_info.id if worker_info is not None else 0)
        else:
            seed = random.SystemRandom().randint(0, 2**63 - 1)
            if worker_info is not None:
                seed += worker_info.id
        self.rng = random.Random(seed)
        self.worker_id = worker_info.id if worker_info is not None else 0
        self.num_workers = worker_info.num_workers if worker_info is not None else 1
        # Keep tokens_seen cumulative across epochs so difficulty schedules carry
        # forward instead of resetting each pass over the dataset.
        ds._worker_token_budget = max(1, ds._token_budget / self.num_workers)
        num_buckets = len(ds._length_bucket_boundaries) + 1
        self.buffers = [[] for _ in range(num_buckets)]
        self.buffer_max_len = [0 for _ in range(num_buckets)]
        self.index = self.worker_id
        self.phase = "generate"

    def __next__(self) -> List[Dict[str, Any]]:
        if self.phase == "unstarted":
            self._start()
        ds = self.dataset
        if self._generators is None:
            self._generators = AlgorithmicGenerator._get_generators()
        generators = self._generators
        rng = self.rng
        if self.phase == "generate":
            while self.index < ds.num_examples:
                self.index += self.num_workers
                progress = min(ds.tokens_seen / max(ds._worker_token_budget, 1), 1.0)
                if ds.difficulty_fn is not None:
                    task = ds._sample_task(rng, progress, generators)
                    difficulty = max(ds.difficulty_fn(task), ds.min_difficulty)
                    effective_schedule = ds.difficulty_schedule
                    if effective_schedule in {"phased", "fixed"}:
                        effective_schedule = "linear"
                    example = AlgorithmicGenerator.generate_example(
                        tasks=[task],
                        rng=rng,
                        generators=generators,
                        difficulty=difficulty,
                        min_difficulty=ds.min_difficulty,
                        progress=progress,
                        difficulty_schedule=effective_schedule,
                        task_weighting=ds.task_weighting,
                    )
                else:
                    difficulty = ds._sample_difficulty(rng, progress)
                    example = AlgorithmicGenerator.generate_example(
                        tasks=ds.tasks,
                        rng=rng,
                        generators=generators,
                        difficulty=difficulty,
                        min_difficulty=ds.min_difficulty,
                        progress=progress,
                        difficulty_schedule=ds.difficulty_schedule,
                        task_weighting=ds.task_weighting,
                    )
                prompt_tokens = ds._encode_text(example["input"])
                prompt_len = len(prompt_tokens)
                tokens = ds._encode_text(example["text"])
                if ds.tokenizer.eos_token_id is not None:
                    tokens.append(ds.tokenizer.eos_token_id)

                if len(tokens) > ds.max_seq_len:
                    tokens = tokens[:ds.max_seq_len]
                prompt_len = min(prompt_len, len(tokens))

                input_ids = torch.tensor(tokens, dtype=torch.long)

                labels = input_ids.clone()
                if prompt_len > 0:
                    labels[:prompt_len] = -100

                n_tokens = int((labels[1:] != -100).sum().item())
                ds.tokens_seen += n_tokens

                payload = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "task": example["task"],
                }
                if ds.include_lengths:
                    payload["input_len_tokens"] = ds._encode_length(example["input"])
                    payload["target_len_tokens"] = ds._encode_length(example["target"])
                seq_len = int(input_ids.numel())
                bucket_idx = ds._length_bucket_index(seq_len)
                self.buffers[bucket_idx].append(payload)
                self.buffer_max_len[bucket_idx] = max(self.buffer_max_len[bucket_idx], seq_len)

                current_batch_size = len(self.buffers[bucket_idx])
                estimated_tensor_size = current_batch_size * self.buffer_max_len[bucket_idx]

                if (
                    current_batch_size >= ds._max_batch_size
                    or estimated_tensor_size >= ds._target_batch_tokens
                ):
                    out = self.buffers[bucket_idx]
                    self.buffers[bucket_idx] = []
                    self.buffer_max_len[bucket_idx] = 0
                    return out
            self.phase = "flush"
            self.flush_index = 0
        if self.phase == "flush":
            while self.flush_index < len(self.buffers):
                out = self.buffers[self.flush_index]
                self.flush_index += 1
                if out:
                    self.buffers[self.flush_index - 1] = []
                    return out
            self.phase = "done"
        raise StopIteration

    def state_dict(self) -> Dict[str, Any]:
        """Plain values and tensors only (torch.load weights_only=True)."""
        return {
            "kind": "algorithmic",
            "phase": self.phase,
            "seeded": bool(self.seeded),
            "rng": self.rng.getstate() if self.rng is not None else None,
            "worker_id": int(self.worker_id),
            "num_workers": int(self.num_workers),
            "index": int(self.index),
            "flush_index": int(self.flush_index),
            "buffers": [[dict(p) for p in b] for b in self.buffers],
            "buffer_max_len": [int(v) for v in self.buffer_max_len],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("kind") != "algorithmic":
            raise ValueError(f"not an AlgorithmicIterator state: {state.get('kind')!r}")
        phase = state["phase"]
        if phase not in ("unstarted", "generate", "flush", "done"):
            raise ValueError(f"AlgorithmicIterator state: unknown phase {phase!r}")
        num_buckets = len(self.dataset._length_bucket_boundaries) + 1
        if phase != "unstarted" and (len(state["buffers"]) != num_buckets
                                     or len(state["buffer_max_len"]) != num_buckets):
            raise ValueError("AlgorithmicIterator state: bucket count differs from this dataset's")
        if phase != "unstarted" and state["rng"] is None:
            raise ValueError("AlgorithmicIterator state: a started iterator without its rng")
        if not state["seeded"]:
            raise ValueError("AlgorithmicIterator state: an unseeded dataset (seed=None) cannot be resumed exactly")
        self.phase = phase
        self.seeded = bool(state["seeded"])
        if state["rng"] is None:
            self.rng = None
        else:
            version, internal, gauss = state["rng"]
            self.rng = random.Random()
            self.rng.setstate((version, tuple(internal), gauss))
        self.worker_id = int(state["worker_id"])
        self.num_workers = int(state["num_workers"])
        self.index = int(state["index"])
        self.flush_index = int(state["flush_index"])
        self.buffers = [[dict(p) for p in b] for b in state["buffers"]]
        self.buffer_max_len = [int(v) for v in state["buffer_max_len"]]


class FixedAlgorithmicDataset(Dataset):
    """Fixed dataset for grokking experiments. Same examples every epoch."""

    def __init__(
        self,
        tokenizer,
        num_examples: int = 5000,
        max_seq_len: int = 128,
        tasks: Optional[List[str]] = None,
        seed: int = 42,
        difficulty: float = 0.5,
        difficulty_schedule: str = "linear",
        task_weighting: str = "uniform",
        easy_mix_frac: float = 0.2,
        min_difficulty: float = 0.0,
        include_lengths: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.seed = seed
        self.min_difficulty = min_difficulty
        self.difficulty = max(difficulty, min_difficulty)
        self.difficulty_schedule = difficulty_schedule
        self.task_weighting = task_weighting
        self.easy_mix_frac = easy_mix_frac
        self.include_lengths = include_lengths

        self.examples: List[Dict[str, torch.Tensor]] = []

        rng = random.Random(seed)
        generators = AlgorithmicGenerator._get_generators()

        for _ in range(self.num_examples):
            example = AlgorithmicGenerator.generate_example(
                tasks=self.tasks,
                rng=rng,
                generators=generators,
                difficulty=self.difficulty,
                min_difficulty=self.min_difficulty,
                progress=1.0,
                difficulty_schedule=self.difficulty_schedule,
                task_weighting=self.task_weighting,
            )
            self.examples.append(self._encode_example(example))

    def _encode_example(self, example: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """One example dict (text, input, target, task) -> the training payload: BOS + prompt + answer + EOS,
        prompt positions masked to -100 in the labels. ModTableDataset uses this same method."""
        prompt_tokens = self._encode_text(example["input"])
        prompt_len = len(prompt_tokens)
        tokens = self._encode_text(example["text"])
        if self.tokenizer.eos_token_id is not None:
            tokens.append(self.tokenizer.eos_token_id)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        prompt_len = min(prompt_len, len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)

        labels = input_ids.clone()
        if prompt_len > 0:
            labels[:prompt_len] = -100

        payload = {
            "input_ids": input_ids,
            "labels": labels,
            "task": example["task"],
        }
        if self.include_lengths:
            payload["input_len_tokens"] = len(
                self.tokenizer.encode(example["input"], add_special_tokens=False)
            )
            payload["target_len_tokens"] = len(
                self.tokenizer.encode(example["target"], add_special_tokens=False)
            )
        return payload

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx % len(self.examples)]


class _PermutationBatchIterator:
    """One epoch of PermutationBatchSampler. The seed is drawn on the first next(), as RandomSampler draws it."""

    def __init__(self, n: int, batch_size: int):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.perm: Optional[List[int]] = None
        self.pos = 0

    def __iter__(self) -> "_PermutationBatchIterator":
        return self

    def __next__(self) -> List[int]:
        if self.perm is None:
            # RandomSampler.__iter__ without a generator (torch 2.9.1 sampler.py:173-198): one draw from the
            # global torch RNG, then a permutation from a fresh generator seeded with it
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.perm = torch.randperm(self.n, generator=generator).tolist()
        if self.pos >= self.n:
            raise StopIteration
        batch = self.perm[self.pos:self.pos + self.batch_size]
        self.pos += len(batch)
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "permutation",
            "n": self.n,
            "batch_size": self.batch_size,
            "perm": None if self.perm is None else torch.tensor(self.perm, dtype=torch.int64),
            "pos": int(self.pos),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("kind") != "permutation":
            raise ValueError(f"not a permutation sampler state: {state.get('kind')!r}")
        if int(state["n"]) != self.n or int(state["batch_size"]) != self.batch_size:
            raise ValueError("permutation sampler state: dataset size or batch size differs from this run's")
        perm = state["perm"]
        pos = int(state["pos"])
        if perm is None:
            if pos != 0:
                raise ValueError("permutation sampler state: a position without a permutation")
            self.perm = None
        else:
            values = perm.tolist()
            if sorted(values) != list(range(self.n)):
                raise ValueError("permutation sampler state: not a permutation of the dataset's indices")
            self.perm = values
        if not 0 <= pos <= self.n:
            raise ValueError("permutation sampler state: position out of range")
        self.pos = pos


class PermutationBatchSampler(Sampler):
    """What DataLoader(shuffle=True, batch_size=B) uses (BatchSampler over RandomSampler, drop_last False), with a
    position a checkpoint can save (exact resume of --fixed_data, #154). Same global RNG draws, same batches."""

    def __init__(self, n: int, batch_size: int):
        if int(n) <= 0:
            raise ValueError(f"PermutationBatchSampler needs a non-empty dataset, got n={n}")
        if int(batch_size) <= 0:
            raise ValueError(f"PermutationBatchSampler needs a positive batch size, got {batch_size}")
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.last_iter: Optional[_PermutationBatchIterator] = None
        self._pending_state: Optional[Dict[str, Any]] = None

    def __iter__(self) -> _PermutationBatchIterator:
        it = _PermutationBatchIterator(self.n, self.batch_size)
        pending, self._pending_state = self._pending_state, None
        if pending is not None:
            it.load_state_dict(pending)
        self.last_iter = it
        return it

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getstate__(self):
        state = self.__dict__.copy()
        state["last_iter"] = None
        state["_pending_state"] = None
        return state


def create_algorithmic_dataset(
    tokenizer,
    fixed: bool = False,
    num_examples: int = 100000,
    max_seq_len: int = 128,
    base_batch_size: int = 1,
    max_batch_multiplier: int = 4,
    tasks: Optional[List[str]] = None,
    seed: int = 42,
    difficulty_value=None,
    difficulty_fn: Optional[Callable[[str], float]] = None,
    weighting_fn: Optional[Callable[[str], float]] = None,
    weighting_adjust_fn: Optional[Callable[[Sequence[str], List[float]], List[float]]] = None,
    difficulty_schedule: str = "smooth",
    task_weighting: str = "uniform",
    total_tokens: Optional[int] = None,
    easy_mix_frac: float = 0.2,
    min_difficulty: float = 0.0,
    length_bucket_boundaries: Optional[Sequence[int]] = None,
) -> Union[AlgorithmicDataset, FixedAlgorithmicDataset]:
    """Factory that returns fixed or infinite dataset based on flag."""

    if fixed:
        if difficulty_value is None:
            difficulty = 0.5
        else:
            try:
                difficulty = float(difficulty_value.value)
            except Exception:
                difficulty = float(difficulty_value)
        return FixedAlgorithmicDataset(
            tokenizer=tokenizer,
            num_examples=num_examples,
            max_seq_len=max_seq_len,
            tasks=tasks,
            seed=seed,
            difficulty=difficulty,
            difficulty_schedule=difficulty_schedule,
            task_weighting=task_weighting,
            easy_mix_frac=easy_mix_frac,
            min_difficulty=min_difficulty,
        )

    return AlgorithmicDataset(
        tokenizer=tokenizer,
        num_examples=num_examples,
        max_seq_len=max_seq_len,
        base_batch_size=base_batch_size,
        max_batch_multiplier=max_batch_multiplier,
        tasks=tasks,
        seed=seed,
        difficulty_value=difficulty_value,
        difficulty_fn=difficulty_fn,
        weighting_fn=weighting_fn,
        weighting_adjust_fn=weighting_adjust_fn,
        difficulty_schedule=difficulty_schedule,
        task_weighting=task_weighting,
        total_tokens=total_tokens,
        easy_mix_frac=easy_mix_frac,
        min_difficulty=min_difficulty,
        length_bucket_boundaries=length_bucket_boundaries,
    )


# =============================================================================
# LANGUAGE DATA (Phase 2)
# =============================================================================

@dataclass
class LanguageDataConfig:
    name: str
    subset: Optional[str]
    split: str
    text_field: str
    weight: float
    max_samples: Optional[int]
    # Hub commit to load; None means the latest. A pin keeps the data fixed across runs.
    revision: Optional[str] = None


LANGUAGE_CONFIGS = {
    "tinystories": LanguageDataConfig(
        name="roneneldan/TinyStories",
        subset=None,
        split="train",
        text_field="text",
        weight=0.5,
        max_samples=500_000,
    ),
    # Simple English Wikipedia from FineWiki (parquet, ~2025 snapshot, 265k articles). It
    # replaces "wikipedia" 20220301.simple, a script dataset that datasets>=4.0 cannot load.
    "wikipedia": LanguageDataConfig(
        name="HuggingFaceFW/finewiki",
        subset="simple",
        split="train",
        text_field="text",
        weight=0.3,
        max_samples=200_000,
        revision="8bd13e72e6a002407649b3e898535f42ceb1aeb9",
    ),
    "minipile": LanguageDataConfig(
        name="JeanKaddour/minipile",
        subset=None,
        split="train",
        text_field="text",
        weight=0.2,
        max_samples=300_000,
    ),
}


class _EmptyLanguageSource:
    """An empty stream standing in for a language source that has no data on this worker.

    ``failed`` is True when the source could not be loaded and False when this worker skips
    it on purpose. ``reason`` keeps the text, so LanguageDataset can report why it got no data.
    """

    def __init__(self, name: str, reason: str, *, failed: bool):
        self.name = name
        self.reason = reason
        self.failed = failed

    def __iter__(self) -> "_EmptyLanguageSource":
        return self

    def __next__(self) -> Dict[str, str]:
        raise StopIteration


def load_language_dataset(
    config: LanguageDataConfig,
    tokenizer,
    max_seq_len: int,
    *,
    log: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
    max_samples_override: Optional[int] = None,
) -> Iterable[Dict[str, str]]:
    """Stream a language dataset without loading it fully into memory."""
    from datasets import load_dataset

    if log:
        print(f"Loading {config.name}...")

    # No trust_remote_code: datasets>=4.0 ignores it, and on older versions it would run
    # any loading script found in a source's repo.
    extra = {"revision": config.revision} if config.revision else {}
    try:
        if config.subset:
            ds = load_dataset(
                config.name,
                config.subset,
                split=config.split,
                streaming=True,
                **extra,
            )
        else:
            ds = load_dataset(
                config.name,
                split=config.split,
                streaming=True,
                **extra,
            )
    except Exception as e:
        if log:
            print(f"  Failed: {e}")
        return _EmptyLanguageSource(
            config.name, f"failed to load: {type(e).__name__}: {e}", failed=True
        )

    num_shards = max(1, num_shards)
    shard_index = min(max(shard_index, 0), num_shards - 1)
    requested_shard_index = shard_index
    shardable = True
    n_files = getattr(ds, "num_shards", None)
    if num_shards > 1 and isinstance(n_files, int) and n_files < num_shards:
        # Fewer files than workers: datasets cannot give every worker a file (shard() raises
        # IndexError on the ones left over), so worker 0 streams the source unsharded.
        if log:
            print(
                f"  Warning: {config.name} has {n_files} file(s) for {num_shards} workers; "
                "falling back to unsharded stream."
            )
        shardable = False
        num_shards = 1
    elif num_shards > 1:
        try:
            ds = ds.shard(num_shards=num_shards, index=shard_index, contiguous=True)
        except TypeError:
            ds = ds.shard(num_shards=num_shards, index=shard_index)
        except AttributeError:
            if log:
                print(
                    f"  Warning: {config.name} does not support sharding; "
                    "falling back to unsharded stream."
                )
            shardable = False
            num_shards = 1

    if not shardable and requested_shard_index > 0:
        if log:
            print(
                f"  Skipping worker {requested_shard_index} "
                f"for {config.name} (no sharding support)"
            )
        return _EmptyLanguageSource(
            config.name,
            f"skipped on worker {requested_shard_index} (no sharding support)",
            failed=False,
        )

    if not shardable and config.max_samples is not None:
        effective_max_samples = config.max_samples
    else:
        effective_max_samples = max_samples_override

    max_samples = effective_max_samples if effective_max_samples is not None else config.max_samples

    buffer_size = 10_000
    if max_samples:
        buffer_size = min(buffer_size, max_samples)
        ds = ds.shuffle(seed=42, buffer_size=buffer_size).take(max_samples)
    else:
        ds = ds.shuffle(seed=42, buffer_size=buffer_size)

    def generator() -> Iterator[Dict[str, str]]:
        count = 0
        for item in ds:
            text = item.get(config.text_field, "")
            if text and len(text) > 50:
                count += 1
                yield {"text": text, "source": config.name}

        if log:
            print(f"  Streamed {count} examples from {config.name}")

    return generator()


def _example_sha256(example: Dict[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in ("input_ids", "labels"):
        t = example[key].detach().cpu().contiguous()
        h.update(str((key, t.dtype, tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class LanguageIterator:
    """LanguageDataset.__iter__ with a position a checkpoint can save (exact resume, #154).

    The Hub/arrow streams themselves cannot be saved, so the position is how many examples were yielded, the hash
    of the last one, and every routing event of the stream (a source popped, with why, and each rebuild), each with
    the yield count it happened at. A restore rebuilds the streams and runs them forward that many examples; it must
    see exactly the same events and end on the same example, else it raises (a source changed, or failed
    differently this time). Running forward reads and tokenizes every consumed example once.
    """

    def __init__(self, dataset: "LanguageDataset"):
        self.dataset = dataset
        self.events: List[List[Any]] = []
        self._gen = dataset._stream(self.events)
        self.yielded = 0
        self.last_sha256: Optional[str] = None

    def __iter__(self) -> "LanguageIterator":
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        example = next(self._gen)
        self.yielded += 1
        self.last_sha256 = _example_sha256(example)
        return example

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "language",
            "yielded": int(self.yielded),
            "last_sha256": self.last_sha256,
            "events": [list(e) for e in self.events],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("kind") != "language":
            raise ValueError(f"not a LanguageIterator state: {state.get('kind')!r}")
        target = int(state["yielded"])
        if self.yielded != 0:
            raise ValueError("LanguageIterator: restore needs a fresh iterator")
        for _ in range(target):
            try:
                next(self)
            except StopIteration:
                raise ValueError(
                    f"LanguageIterator restore: the language stream ended after {self.yielded} of the "
                    f"{target} examples the checkpoint had consumed"
                ) from None
        want_events = [list(e) for e in state["events"]]
        if self.events != want_events:
            raise ValueError(
                "LanguageIterator restore: the language stream routed differently this time "
                f"(saved events {want_events[:5]}, now {self.events[:5]})"
            )
        if self.last_sha256 != state["last_sha256"]:
            raise ValueError("LanguageIterator restore: the last example differs from the checkpoint's")


class LanguageDataset(IterableDataset):
    """Dataset for Phase 2: Language Generalization, streamed to limit RAM."""

    def __init__(
        self,
        tokenizer,
        configs: Optional[Dict[str, LanguageDataConfig]] = None,
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.configs = configs or LANGUAGE_CONFIGS

    def __iter__(self) -> "LanguageIterator":
        it = LanguageIterator(self)
        pending, self._pending_state = getattr(self, "_pending_state", None), None
        if pending is not None:
            it.load_state_dict(pending)
        self.last_iter = it
        return it

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("last_iter", None)
        state.pop("_pending_state", None)
        return state

    def _stream(self, events: List[List[Any]]) -> Iterator[Dict[str, torch.Tensor]]:
        """The language stream. events gets [kind, yield count, source name, reason] for every source popped
        and every rebuild, so a restore can check it routed the same way (LanguageIterator)."""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rng = random.Random(42 + worker_id)
        yielded = 0

        config_items = list(self.configs.items())
        rng.shuffle(config_items)

        def _build_streams() -> List[Tuple[LanguageDataConfig, Iterator[Dict[str, str]]]]:
            streams: List[Tuple[LanguageDataConfig, Iterator[Dict[str, str]]]] = []
            for _, config in config_items:
                if config.max_samples is not None and num_workers > 1:
                    base = config.max_samples // num_workers
                    remainder = config.max_samples % num_workers
                    worker_max_samples = base + (1 if worker_id < remainder else 0)
                    if worker_max_samples == 0:
                        continue
                else:
                    worker_max_samples = config.max_samples

                stream = load_language_dataset(
                    config,
                    self.tokenizer,
                    self.max_seq_len,
                    log=(worker_id == 0),
                    num_shards=num_workers,
                    shard_index=worker_id,
                    max_samples_override=worker_max_samples,
                )
                streams.append((config, iter(stream)))
            return streams

        worker_tag = f" (worker {worker_id} of {num_workers})" if num_workers > 1 else ""
        warned_sources = set()

        def _start_pass(streams) -> None:
            # A pass is one build of the streams, until all of them are used up. A pass that
            # ends without one example means every source failed or was empty. Rebuilding
            # used to loop on that with no limit, so the check at the end of the pass stops.
            nonlocal pass_yielded, pass_reasons, pass_all_skipped
            pass_yielded = False
            pass_reasons = []
            pass_all_skipped = bool(streams) and all(
                isinstance(s, _EmptyLanguageSource) and not s.failed for _, s in streams
            )
            for config, s in streams:
                if isinstance(s, _EmptyLanguageSource) and s.failed:
                    _warn_once(config, s.reason)

        def _warn_once(config: LanguageDataConfig, reason: str) -> None:
            key = (config.name, config.subset, config.split)
            if key not in warned_sources:
                warned_sources.add(key)
                print(
                    f"Warning: language source {config.name} {reason}{worker_tag}; "
                    "continuing without it, so the language mix changes."
                )

        pass_yielded = False
        pass_reasons: List[str] = []
        pass_all_skipped = False

        active_streams = _build_streams()
        _start_pass(active_streams)
        rebuild_count = 0

        while True:
            if not active_streams:
                if not pass_yielded and pass_reasons:
                    if pass_all_skipped:
                        print(
                            f"Language streams{worker_tag}: every source is skipped on this "
                            "worker; stopping."
                        )
                        return
                    which = "first pass" if rebuild_count == 0 else f"pass after rebuild {rebuild_count}"
                    details = "\n".join(f"  - {r}" for r in pass_reasons)
                    raise RuntimeError(
                        f"LanguageDataset{worker_tag}: the {which} over the language sources "
                        "yielded no example; stopping instead of rebuilding again. Sources:\n"
                        f"{details}\n"
                        "Check the network or the HF offline settings, and LANGUAGE_CONFIGS."
                    )
                rebuild_count += 1
                events.append(["rebuild", yielded, "", str(rebuild_count)])
                print(
                    "Warning: language streams exhausted; rebuilding streams "
                    f"(count={rebuild_count})."
                )
                active_streams = _build_streams()
                _start_pass(active_streams)
                if not active_streams:
                    print("Warning: language stream rebuild produced no streams; stopping.")
                    return

            weights = [config.weight for config, _ in active_streams]
            total_weight = sum(weights)
            pick = rng.uniform(0, total_weight)

            chosen_index = 0
            cumulative = 0.0
            for i, w in enumerate(weights):
                cumulative += w
                if pick <= cumulative:
                    chosen_index = i
                    break

            config, stream = active_streams[chosen_index]

            try:
                example = next(stream)
            except StopIteration:
                active_streams.pop(chosen_index)
                if isinstance(stream, _EmptyLanguageSource):
                    pass_reasons.append(f"{config.name}: {stream.reason}")
                    events.append(["pop", yielded, config.name, stream.reason])
                else:
                    pass_reasons.append(f"{config.name}: stream ended with no example")
                    events.append(["pop", yielded, config.name, "ended"])
                continue
            except Exception as exc:
                active_streams.pop(chosen_index)
                reason = f"raised while streaming: {type(exc).__name__}: {exc}"
                pass_reasons.append(f"{config.name}: {reason}")
                events.append(["pop", yielded, config.name, f"raised {type(exc).__name__}"])
                _warn_once(config, reason)
                continue
            pass_yielded = True

            tokens = self.tokenizer.encode(example["text"], add_special_tokens=True)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]

            input_ids = torch.tensor(tokens, dtype=torch.long)

            labels = input_ids.clone()

            yielded += 1
            yield {"input_ids": input_ids, "labels": labels}


def _collate_with_padding(
    examples: Sequence[Dict[str, torch.Tensor]],
    *,
    pad_id: int,
    label_pad_id: int = -100,
) -> Dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [ex["input_ids"] for ex in examples],
        batch_first=True,
        padding_value=pad_id,
    )
    labels = pad_sequence(
        [ex["labels"] for ex in examples],
        batch_first=True,
        padding_value=label_pad_id,
    )
    batch: Dict[str, torch.Tensor] = {"input_ids": input_ids, "labels": labels}

    for key in examples[0]:
        if key in {"input_ids", "labels"}:
            continue
        values = [ex[key] for ex in examples]
        first = values[0]
        if torch.is_tensor(first):
            try:
                batch[key] = torch.stack(values)
            except RuntimeError:
                batch[key] = values
        elif isinstance(first, (int, float, bool)):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = values

    return batch


def make_collate_fn(tokenizer) -> Callable[[Sequence[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return partial(_collate_with_padding, pad_id=pad_id)


# =============================================================================
# CURRICULUM SAMPLER
# =============================================================================

def _generator_state(loader) -> Optional[torch.Tensor]:
    """A copy of the loader's own shuffle generator state (None without one). Reading it draws nothing."""
    gen = getattr(loader, "generator", None)
    return None if gen is None else gen.get_state()


def _position_holder(loader: Optional[DataLoader]):
    """The object a loader's position lives in after iter(loader), for exact resume: its IterableDataset (which
    records the iterator it made as last_iter) or, for a map-style dataset, its batch sampler. None when the loader
    keeps it in worker processes (num_workers > 0)."""
    if loader is None or getattr(loader, "num_workers", 0) != 0:
        return None
    if isinstance(loader.dataset, IterableDataset):
        return loader.dataset
    return loader.batch_sampler


def language_manifest(configs: Dict[str, "LanguageDataConfig"]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The language sources as an ordered list (insertion order drives the stream), each with its config and, for a
    local directory, the sha256 of every file in it by relative path. Second value: why a resume of a stream over
    these sources cannot be exact (a source that is not a local directory: Hub data can change under the same name;
    a directory with a dataset card, script or YAML file, or with links out of it: datasets could then read files the
    hashes do not cover). Exact language sources are plain directories of data files, read by datasets' default
    patterns inside the directory."""
    import hashlib
    import os
    from dataclasses import asdict

    manifest: List[Dict[str, Any]] = []
    why: List[str] = []
    for key, config in configs.items():
        entry: Dict[str, Any] = {"key": str(key), "config": {k: v for k, v in asdict(config).items()}}
        if os.path.isdir(config.name):
            root = Path(config.name)
            files = []
            redirects, outside = [], []
            real_root = root.resolve()
            for f in sorted(q for q in root.rglob("*") if q.is_file()):
                rel = f.relative_to(root).as_posix()
                # a dataset card (README YAML data_files / configs), a loading script or a YAML file can make
                # datasets read files outside this directory (absolute or ../ paths, URLs); a link can too
                if f.name.lower().startswith("readme") or f.suffix.lower() in (".py", ".yaml", ".yml"):
                    redirects.append(rel)
                if f.is_symlink() or real_root not in f.resolve().parents:
                    outside.append(rel)
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for block in iter(lambda: fh.read(1 << 20), b""):
                        h.update(block)
                files.append([rel, h.hexdigest()])
            entry["files"] = files
            if redirects:
                why.append(f"language source {config.name} holds {', '.join(redirects[:3])}: a dataset card, "
                           f"script or YAML file can point datasets at files outside the hashed directory")
            if outside:
                why.append(f"language source {config.name} holds links that resolve outside it: "
                           f"{', '.join(outside[:3])}")
        else:
            entry["files"] = None
            why.append(f"language source {config.name} is not a local directory")
        manifest.append(entry)
    return manifest, why


class CurriculumSampler:
    """
    Flexible curriculum sampler that blends algorithmic, language, and optional
    lexical noise data according to token-based schedules.
    """

    def __init__(
        self,
        alg_loader: DataLoader,
        lang_loader: DataLoader,
        total_tokens: int,
        alg_tokens: int,
        mix_band_tokens: int,
        persistent_alg_frac: float = 0.15,
        lexical_frac_phase1: float = 0.05,
        seed: int = 42,
        lex_loader: Optional[DataLoader] = None,
        track_position: bool = False,
    ):
        if not 0.0 <= persistent_alg_frac <= 0.9:
            raise ValueError(
                "persistent_alg_frac must be between 0.0 and 0.9 (recommended 0.05–0.3)"
            )

        self.alg_loader = alg_loader
        self.lang_loader = lang_loader
        self.lex_loader = lex_loader

        # Exact resume (train.py --mod_table_resume auto): the alg loader's position is the state of its shuffle
        # generator just before the current alg iterator was made (the "anchor"), how that iterator was made, and
        # how many batches it has given. Read-only bookkeeping: it never draws from any generator.
        self.track_position = track_position
        if track_position:
            self._alg_anchor = _generator_state(self.alg_loader)
            self._alg_anchor_kind = "init"  # made by this constructor, before lang_iter and lex_iter
            self.alg_drawn = 0

        # exact resume (train.py --resume_from): the object holding each source's position, read right after each
        # iter() below; see exact_state()
        self._inner: Dict[str, Any] = {"alg": None, "lang": None, "lex": None}
        self.alg_iter = self._fresh_iter("alg")
        self.lang_iter = self._fresh_iter("lang")
        self.lex_iter = self._fresh_iter("lex") if self.lex_loader is not None else None

        self.total_tokens = total_tokens
        self.alg_tokens = alg_tokens
        self.mix_band_tokens = int(min(max(mix_band_tokens, 0), self.alg_tokens))
        self.persistent_alg_frac = persistent_alg_frac
        self.lexical_frac_phase1 = lexical_frac_phase1 if lex_loader is not None else 0.0
        self.tokens_seen = 0

        self.rng = random.Random(seed)

    def _loader(self, which: str) -> Optional[DataLoader]:
        return {"alg": self.alg_loader, "lang": self.lang_loader, "lex": self.lex_loader}[which]

    def _fresh_iter(self, which: str):
        """iter() of one source's loader, noting the object its position lives in (made eagerly inside iter())."""
        loader = self._loader(which)
        iterator = iter(loader)
        holder = _position_holder(loader)
        self._inner[which] = getattr(holder, "last_iter", None) if holder is not None else None
        return iterator

    @property
    def phase(self) -> str:
        # Phase indicator is approximate; probabilities may mix.
        if self.tokens_seen < self.alg_tokens - self.mix_band_tokens:
            return "algorithmic"
        if self.tokens_seen < self.alg_tokens + self.mix_band_tokens:
            return "mix"
        return "language"

    @property
    def progress(self) -> float:
        return self.tokens_seen / self.total_tokens

    def _compute_probs(self) -> Dict[str, float]:
        t = self.tokens_seen

        if t < self.alg_tokens - self.mix_band_tokens:
            # Region A: early algorithmic focus with optional lexical noise.
            p_alg = 1.0 - self.lexical_frac_phase1
            p_lex = self.lexical_frac_phase1
            p_lang = 0.0
        elif t < self.alg_tokens + self.mix_band_tokens:
            # Region B: linear transition from algorithmic to language.
            if self.mix_band_tokens == 0:
                u = 1.0
            else:
                u = (t - (self.alg_tokens - self.mix_band_tokens)) / (2.0 * self.mix_band_tokens)
                u = min(max(u, 0.0), 1.0)

            p_alg = (1.0 - self.lexical_frac_phase1) * (1.0 - u) + self.persistent_alg_frac * u
            p_lex = self.lexical_frac_phase1 * (1.0 - u)
            p_lang = 1.0 - p_alg - p_lex
        else:
            # Region C: predominantly language with persistent algorithmic fraction.
            p_alg = self.persistent_alg_frac
            p_lex = 0.0
            p_lang = 1.0 - p_alg

        # If no lexical loader, renormalize to exclude lexical probability.
        if self.lex_iter is None:
            p_lex = 0.0
            total = p_alg + p_lang
            if total > 0:
                p_alg /= total
                p_lang /= total

        return {"alg": p_alg, "lang": p_lang, "lex": p_lex}

    def _sample_source(self) -> str:
        probs = self._compute_probs()
        return self.rng.choices(
            ["alg", "lang", "lex"],
            weights=[probs["alg"], probs["lang"], probs["lex"]],
        )[0]

    def _next_from(self, which: str) -> Dict[str, torch.Tensor]:
        iter_attr = f"{which}_iter"

        iterator = getattr(self, iter_attr)
        if self.track_position and which == "alg":
            return self._next_alg_tracked(iterator)
        try:
            return next(iterator)
        except StopIteration:
            iterator = self._fresh_iter(which)
            setattr(self, iter_attr, iterator)
            return next(iterator)

    def _next_alg_tracked(self, iterator) -> Dict[str, torch.Tensor]:
        """_next_from("alg") with the position bookkeeping; the same calls in the same order."""
        try:
            batch = next(iterator)
        except StopIteration:
            # the exhausted sampler has already drawn its tail from the generator: the anchor comes after that
            anchor = _generator_state(self.alg_loader)
            self.alg_iter = self._fresh_iter("alg")
            batch = next(self.alg_iter)
            self._alg_anchor, self._alg_anchor_kind, self.alg_drawn = anchor, "epoch", 0
        self.alg_drawn += 1
        return batch

    def set_alg_loader(self, loader: DataLoader) -> None:
        """Replace the alg loader and start a fresh iterator over it (what train.py's mod-table join does)."""
        if self.track_position:
            self._alg_anchor, self._alg_anchor_kind, self.alg_drawn = _generator_state(loader), "join", 0
        self.alg_loader = loader
        self.alg_iter = self._fresh_iter("alg")

    def alg_position(self) -> Dict[str, Any]:
        if not self.track_position:
            raise RuntimeError("CurriculumSampler(track_position=True) is needed for alg_position()")
        return {"anchor": self._alg_anchor, "kind": self._alg_anchor_kind, "drawn": int(self.alg_drawn),
                "generator_now": _generator_state(self.alg_loader)}

    def restore_alg_position(self, pos: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Put the alg iterator where alg_position() saw it. self.alg_loader must already be the loader that
        position belongs to (same dataset, batch size, sampler), freshly built. Replays the same calls the original
        run made since the anchor (iter(), plus the constructor's lang/lex iter() for an "init" anchor, then
        `drawn` batches), checks the generator ends in the saved state, and returns the last replayed batch."""
        gen = self.alg_loader.generator
        if gen is None or pos["anchor"] is None:
            raise RuntimeError("restore_alg_position needs an alg loader with its own torch.Generator")
        gen.set_state(pos["anchor"].clone())
        self.alg_iter = self._fresh_iter("alg")
        if pos["kind"] == "init":
            self.lang_iter = self._fresh_iter("lang")
            self.lex_iter = self._fresh_iter("lex") if self.lex_loader is not None else None
        last = None
        for _ in range(int(pos["drawn"])):
            last = next(self.alg_iter)  # StopIteration here means the saved count is wrong: fail loudly
        if not torch.equal(gen.get_state(), pos["generator_now"]):
            raise RuntimeError("restore_alg_position: the alg generator does not end in the saved state")
        self.track_position = True
        self._alg_anchor, self._alg_anchor_kind, self.alg_drawn = pos["anchor"].clone(), pos["kind"], int(pos["drawn"])
        return last

    def next_batch(self) -> Dict[str, torch.Tensor]:
        source = self._sample_source()
        if source == "alg":
            batch = self._next_from("alg")
        elif source == "lex" and self.lex_iter is not None and self.lex_loader is not None:
            batch = self._next_from("lex")
        else:
            batch = self._next_from("lang")

        # Count tokens (non-padding)
        n_tokens = (batch["labels"][..., 1:] != -100).sum().item()
        self.tokens_seen += n_tokens

        return batch

    def exact_state(self) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """(state, why not): where every source's current iterator is, for an exact resume. state is None, with the
        reasons, when a position cannot be captured (workers, or an iterator type without state)."""
        why: List[str] = []
        sources: Dict[str, Any] = {}
        for which in ("alg", "lang", "lex"):
            loader = self._loader(which)
            if loader is None:
                sources[which] = None
                continue
            if getattr(loader, "num_workers", 0) != 0:
                why.append(f"{which} loader has num_workers {loader.num_workers}")
                continue
            inner = self._inner.get(which)
            if inner is None or not hasattr(inner, "state_dict"):
                why.append(f"{which} loader position is not capturable ({type(inner).__name__})")
                continue
            sources[which] = inner.state_dict()
        if why:
            return None, why
        return {
            "format": 1,
            "tokens_seen": int(self.tokens_seen),
            "rng": self.rng.getstate(),
            "alias": self.lang_loader is self.alg_loader,
            "sources": sources,
        }, []

    def restore_exact_state(self, state: Dict[str, Any]) -> None:
        """Put every source's iterator where exact_state() saw it. The loaders must be the ones that state belongs
        to, freshly built (same datasets, batch sizes, samplers). Each iterator is made by iter(loader) with the
        saved position armed on its holder, in the constructor's order."""
        if state.get("format") != 1:
            raise ValueError(f"sampler state format {state.get('format')!r} is not 1")
        if bool(state["alias"]) != (self.lang_loader is self.alg_loader):
            raise ValueError("sampler state: the language loader is the algorithmic one in only one of the runs")
        for which in ("alg", "lang", "lex"):
            loader = self._loader(which)
            saved = state["sources"].get(which)
            if (loader is None) != (saved is None):
                raise ValueError(f"sampler state: the {which} source exists in only one of the runs")
            if loader is None:
                continue
            holder = _position_holder(loader)
            if holder is None:
                raise ValueError(f"sampler state: the {which} loader cannot take a saved position here")
            holder._pending_state = saved
            iterator = self._fresh_iter(which)
            if holder._pending_state is not None or self._inner[which] is None:
                raise ValueError(f"sampler state: the {which} iterator did not take the saved position")
            setattr(self, f"{which}_iter", iterator)
        self.tokens_seen = int(state["tokens_seen"])
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))

    def reset(self):
        self.tokens_seen = 0
        self.alg_iter = self._fresh_iter("alg")
        self.lang_iter = self._fresh_iter("lang")
        self.lex_iter = self._fresh_iter("lex") if self.lex_loader is not None else None


# =============================================================================
# OOD LENGTH TESTING
# =============================================================================

class OODLengthGenerator:
    """
    Generates algorithmic tasks at longer-than-training lengths.
    Tests whether recursive architectures generalize better than positional-embedding transformers.
    """
    
    @staticmethod
    def addition_ood(n_digits: int) -> Dict:
        """Addition with more digits than training."""
        a = random.randint(10**(n_digits-1), 10**n_digits - 1)
        b = random.randint(10**(n_digits-1), 10**n_digits - 1)
        prompt = f"{a} + {b}. Answer:"
        target = str(a + b)
        return {
            "input": prompt,
            "target": target,
            "text": f"{prompt}{target}",
            "answer": target,
            "length": n_digits,
            "task": "addition_ood",
        }
    
    @staticmethod
    def parity_ood(length: int) -> Dict:
        """Parity with longer sequences than training."""
        bits = [random.randint(0, 1) for _ in range(length)]
        parity = sum(bits) % 2
        prompt = f"parity({''.join(map(str, bits))}). Answer:"
        target = str(parity)
        return {
            "input": prompt,
            "target": target,
            "text": f"{prompt}{target}",
            "answer": target,
            "length": length,
            "task": "parity_ood",
        }
    
    @staticmethod
    def copy_ood(length: int) -> Dict:
        """Copy with longer sequences than training."""
        seq = [random.randint(0, 9) for _ in range(length)]
        seq_str = " ".join(map(str, seq))
        prompt = f"Copy the sequence {seq_str}. Answer:"
        target = seq_str
        return {
            "input": prompt,
            "target": target,
            "text": f"{prompt}{target}",
            "answer": target,
            "length": length,
            "task": "copy_ood",
        }
    
    @classmethod
    def generate_sweep(cls, task: str, min_len: int, max_len: int, step: int = 2, n_per_len: int = 10) -> List[Dict]:
        """Generate examples across length range."""
        generators = {
            "addition": cls.addition_ood,
            "parity": cls.parity_ood,
            "copy": cls.copy_ood,
        }
        
        if task not in generators:
            raise ValueError(f"Unknown task: {task}")
        
        gen = generators[task]
        examples = []
        
        for length in range(min_len, max_len + 1, step):
            for _ in range(n_per_len):
                examples.append(gen(length))
        
        return examples


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_curriculum_loaders(
    tokenizer,
    alg_batch_size: int = 64,
    lang_batch_size: int = 32,
    alg_seq_len: int = 128,
    lang_seq_len: int = 1024,
    alg_examples: int = 100000,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for both curriculum phases."""
    
    alg_dataset = AlgorithmicDataset(
        tokenizer=tokenizer,
        num_examples=alg_examples,
        max_seq_len=alg_seq_len,
        base_batch_size=alg_batch_size,
        seed=42,
    )
    
    lang_dataset = LanguageDataset(
        tokenizer=tokenizer,
        max_seq_len=lang_seq_len,
    )

    collate_fn = make_collate_fn(tokenizer)
    alg_loader = DataLoader(
        alg_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    lang_loader = DataLoader(
        lang_dataset,
        batch_size=lang_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return alg_loader, lang_loader


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing algorithmic generators...")
    
    examples = AlgorithmicGenerator.generate_batch(10)
    for ex in examples[:5]:
        print(f"  [{ex['task']}] {ex['text']}")
    
    print("\nTesting OOD length generation...")
    ood = OODLengthGenerator.generate_sweep("parity", 8, 32, step=8, n_per_len=2)
    for ex in ood[:4]:
        print(f"  [len={ex['length']}] {ex['input'][:50]}... -> {ex['answer']}")
    
    print("\n✓ Data generators working!")
