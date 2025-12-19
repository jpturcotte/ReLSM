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
  - Needle-in-haystack evaluation data
  - OOD length testing for algorithmic tasks
"""

import random
import math
from typing import List, Dict, Optional, Tuple, Iterator, Callable, Iterable, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info


# =============================================================================
# ALGORITHMIC DATA GENERATORS (Phase 1)
# =============================================================================

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
        return template.format(**kwargs).strip()
    
    @staticmethod
    def modular_arithmetic(
        mod: Optional[int] = 97,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
    ) -> Dict[str, str]:
        """Modular addition: (a + b) mod p"""
        rng = rng or random
        digits_low, digits_high = AlgorithmicGenerator._digit_range(difficulty)
        operand_digits = rng.randint(digits_low, digits_high)
        modulus_digits = max(operand_digits, digits_low)
        sampled_mod = AlgorithmicGenerator._sample_int_with_digits(
            rng, min(modulus_digits, 5)
        )
        if mod is None:
            mod_val = max(3, sampled_mod)
        else:
            mod_val = max(3, mod)
        a = rng.randint(0, mod_val - 1)
        b = rng.randint(0, mod_val - 1)
        result = (a + b) % mod_val

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "Compute ({a} + {b}) mod {mod} =",
                "What is ({a} + {b}) modulo {mod}?",
                "Add {a} and {b}, then take mod {mod}. Result:",
                "({a} plus {b}) % {mod} equals",
            ],
            a=a,
            b=b,
            mod=mod_val,
        )

        return {
            "text": f"{prompt} {result}",
            "input": prompt,
            "target": str(result),
            "task": "mod_add",
        }

    @staticmethod
    def parity(
        length: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
    ) -> Dict[str, str]:
        """Parity of binary string"""
        rng = rng or random
        # NOTE: For OOD evaluation we keep training length fixed at 8 by default.
        seq_len = length if length is not None else 8
        bits = [rng.randint(0, 1) for _ in range(seq_len)]
        parity = sum(bits) % 2
        bit_str = "".join(map(str, bits))

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "parity({bits}) =",
                "Is the number of ones in {bits} even (0) or odd (1)?",
                "Compute the parity bit for {bits}:",
                "Parity of {bits}?",
            ],
            bits=bit_str,
        )

        return {
            "text": f"{prompt} {parity}",
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
            digit_range = (1, 2)
            default_digit_bounds = (0, 4)  # low carry pressure
        elif bucket == "medium":
            digit_range = (2, 3)
            default_digit_bounds = (0, 9)  # neutral carry pressure
        else:
            digit_range = (4, 4)
            default_digit_bounds = (5, 9)  # high carry pressure

        digits = digits if digits is not None else rng.randint(digit_range[0], min(digit_range[1], max_digits))
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
                "Compute: {a} + {b} =",
                "What is {a} plus {b}?",
                "Add {a} and {b}. Answer:",
                "Sum {a} with {b} =",
                "{a} + {b} equals",
            ],
            a=a,
            b=b,
        )

        return {
            "text": f"{prompt} {result}",
            "input": prompt,
            "target": str(result),
            "task": "addition",
        }

    @staticmethod
    def multiplication(max_val: int = 99, rng: Optional[random.Random] = None, difficulty: float = 0.5) -> Dict[str, str]:
        """Two-digit multiplication"""
        rng = rng or random
        digit_range = AlgorithmicGenerator._digit_range(difficulty, cap=4)
        digits = rng.randint(digit_range[0], digit_range[1])
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
                "Multiply: {a} * {b} =",
                "What is {a} times {b}?",
                "Product of {a} and {b}:",
                "Compute {a} multiplied by {b} =",
            ],
            a=a,
            b=b,
        )

        return {
            "text": f"{prompt} {result}",
            "input": prompt,
            "target": str(result),
            "task": "multiplication",
        }

    @staticmethod
    def copy_sequence(
        length: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
    ) -> Dict[str, str]:
        """Copy task - tests basic sequence modeling"""
        rng = rng or random
        len_range = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(4, 6),
            medium=(7, 10),
            hard=(11, 14),
        )
        seq_len = length if length is not None else rng.randint(len_range[0], len_range[1])
        seq = [rng.randint(0, 9) for _ in range(seq_len)]
        seq_str = " ".join(map(str, seq))

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "copy: {seq} ->",
                "Repeat the sequence {seq}:",
                "Copy this list exactly: {seq} =>",
                "Write the same numbers again: {seq} =",
            ],
            seq=seq_str,
        )

        return {
            "text": f"{prompt} {seq_str}",
            "input": prompt,
            "target": seq_str,
            "task": "copy",
        }

    @staticmethod
    def reverse_sequence(
        length: Optional[int] = None,
        rng: Optional[random.Random] = None,
        difficulty: float = 0.5,
    ) -> Dict[str, str]:
        """Reverse task - tests sequential reasoning"""
        rng = rng or random
        len_range = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(4, 6),
            medium=(7, 10),
            hard=(11, 14),
        )
        seq_len = length if length is not None else rng.randint(len_range[0], len_range[1])
        seq = [rng.randint(0, 9) for _ in range(seq_len)]
        seq_str = " ".join(map(str, seq))
        rev_str = " ".join(map(str, reversed(seq)))

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "reverse: {seq} ->",
                "Write {seq} backwards:",
                "Reverse the order of {seq}:",
                "Flip the sequence {seq} =",
            ],
            seq=seq_str,
        )

        return {
            "text": f"{prompt} {rev_str}",
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
    ) -> Dict[str, str]:
        """Dyck language (balanced parentheses) - tests stack operations"""
        rng = rng or random
        len_range = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(6, 10),
            medium=(10, 18),
            hard=(18, 30),
        )
        seq_len = length if length is not None else rng.randint(len_range[0], len_range[1])
        depth_bucket = AlgorithmicGenerator._difficulty_bucket(difficulty)
        bucket = AlgorithmicGenerator._difficulty_bucket(difficulty)
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
                "Is '{seq}' balanced? Answer yes or no.",
                "Does {seq} form a correct parentheses string?",
                "dyck: {seq} -> valid?",
                "Check Dyck validity for: {seq}",
            ],
            seq=seq_str,
        )

        return {
            "text": f"{prompt} {'yes' if is_valid else 'no'}",
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
    ) -> Dict[str, str]:
        """Chain of arithmetic operations"""
        rng = rng or random
        op_counts = AlgorithmicGenerator._length_range(
            difficulty,
            easy=(2, 3),
            medium=(3, 5),
            hard=(5, 8),
        )
        bucket = AlgorithmicGenerator._difficulty_bucket(difficulty)
        if bucket == "easy":
            default_operand_high = 9
            start_range = (5, 40)
            default_allow_negative = False
        elif bucket == "medium":
            default_operand_high = 20
            start_range = (10, 60)
            default_allow_negative = rng.random() < 0.5
        else:
            default_operand_high = 50
            start_range = (20, 100)
            default_allow_negative = True

        operand_high = operand_high if operand_high is not None else default_operand_high
        allow_negative = allow_negative if allow_negative is not None else default_allow_negative

        sampled_ops = rng.randint(op_counts[0], op_counts[1])
        ops = n_ops if n_ops is not None else sampled_ops
        val = rng.randint(start_range[0], start_range[1])
        expr = str(val)

        for _ in range(ops):
            op = rng.choice(["+", "-"])
            operand = rng.randint(1, operand_high)
            if not allow_negative and op == "-" and val - operand < 0:
                op = "+"
            expr += f" {op} {operand}"
            val = val + operand if op == "+" else val - operand

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "calc: {expr} =",
                "Evaluate {expr}.",
                "Compute the result of {expr}:",
                "What does {expr} simplify to?",
            ],
            expr=expr,
        )

        return {
            "text": f"{prompt} {val}",
            "input": prompt,
            "target": str(val),
            "task": "chain",
        }

    @staticmethod
    def comparison(max_val: int = 1000, rng: Optional[random.Random] = None, difficulty: float = 0.5) -> Dict[str, str]:
        """Number comparison"""
        rng = rng or random
        digit_range = AlgorithmicGenerator._digit_range(difficulty, cap=6)
        digits = rng.randint(digit_range[0], digit_range[1])
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
                "compare: {a} ? {b} ->",
                "Which is larger, {a} or {b}?",
                "Compare {a} to {b} (>, <, or =):",
                "Between {a} and {b}, choose the relation:",
            ],
            a=a,
            b=b,
        )

        return {
            "text": f"{prompt} {result}",
            "input": prompt,
            "target": result,
            "task": "compare",
        }

    @staticmethod
    def successor(max_val: int = 1000, rng: Optional[random.Random] = None, difficulty: float = 0.5) -> Dict[str, str]:
        """Successor function"""
        rng = rng or random
        digit_range = AlgorithmicGenerator._digit_range(difficulty, cap=6)
        digits = rng.randint(digit_range[0], digit_range[1])
        max_n = min(max_val, 10**digits - 1)
        n = rng.randint(0, max_n)

        prompt = AlgorithmicGenerator._choose_prompt(
            rng,
            [
                "succ({n}) =",
                "What is the successor of {n}:",
                "Next integer after {n} is",
                "Increment {n} by one =>",
            ],
            n=n,
        )
        return {
            "text": f"{prompt} {n + 1}",
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
                tasks=tasks, rng=rng, generators=generators, difficulty=difficulty
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
    ) -> Dict:
        rng = rng or random
        if difficulty is None:
            difficulty = 0.5
        if generators is None:
            generators = cls._get_generators()

        if tasks is None:
            tasks = list(generators.keys())

        task = rng.choice(tasks)
        return generators[task](rng=rng, difficulty=difficulty)

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
    """

    def __init__(
        self,
        tokenizer,
        num_examples: int = 100000,
        max_seq_len: int = 128,
        tasks: Optional[List[str]] = None,
        seed: Optional[int] = None,
        difficulty_value=None,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.seed = seed
        self.difficulty_value = difficulty_value
        self.tokens_seen = 0
        self._token_budget = max(1, self.num_examples * max(self.max_seq_len - 1, 1))
        self._worker_token_budget = self._token_budget

        self._char_token_map = self._build_char_token_map()

    def _build_char_token_map(self) -> Dict[str, int]:
        """Precompute a fast char->id map for synthetic strings."""
        chars = "0123456789()+-*=? "
        chars += "abcdefghijklmnopqrstuvwxyz"
        chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # For future-proofing
        chars += ":,%<>'"  # punctuation used in prompts

        token_map = {}
        for ch in set(chars):
            encoded = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(encoded) == 1:
                token_map[ch] = encoded[0]
        return token_map

    def _encode_text(self, text: str) -> List[int]:
        """Encode text with the tokenizer to align training and evaluation."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def __len__(self):
        return self.num_examples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()

        # Keep tokens_seen cumulative across epochs so difficulty schedules carry
        # forward instead of resetting each pass over the dataset.

        if self.seed is not None:
            seed = self.seed + (worker_info.id if worker_info is not None else 0)
        else:
            seed = random.SystemRandom().randint(0, 2**63 - 1)
            if worker_info is not None:
                seed += worker_info.id

        rng = random.Random(seed)
        generators = AlgorithmicGenerator._get_generators()

        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        self._worker_token_budget = max(1, self._token_budget / num_workers)

        for _ in range(worker_id, self.num_examples, num_workers):
            difficulty = self._sample_difficulty(rng)
            example = AlgorithmicGenerator.generate_example(
                tasks=self.tasks, rng=rng, generators=generators, difficulty=difficulty
            )
            tokens = self._encode_text(example["text"])

            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)

            pad_len = self.max_seq_len - 1 - len(input_ids)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id or 0
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id)])
                labels = torch.cat([labels, torch.full((pad_len,), -100)])

            n_tokens = int((labels != -100).sum().item())
            self.tokens_seen += n_tokens

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "task": example["task"],
            }

    def _sample_difficulty(self, rng: random.Random) -> float:
        if self.difficulty_value is not None:
            try:
                with self.difficulty_value.get_lock():
                    value = float(self.difficulty_value.value)
            except Exception:
                value = 0.0
            return min(max(value, 0.0), 1.0)

        budget = getattr(self, "_worker_token_budget", self._token_budget)
        progress = min(self.tokens_seen / budget, 1.0)
        upper = min(1.0, 0.5 + 0.5 * progress)
        lower = 0.0
        return rng.uniform(lower, upper)


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
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.seed = seed
        self.difficulty = difficulty

        self._char_token_map = self._build_char_token_map()
        self.examples: List[Dict[str, torch.Tensor]] = []

        rng = random.Random(seed)
        generators = AlgorithmicGenerator._get_generators()

        for _ in range(self.num_examples):
            example = AlgorithmicGenerator.generate_example(
                tasks=self.tasks, rng=rng, generators=generators, difficulty=self.difficulty
            )
            tokens = self._encode_text(example["text"])

            if len(tokens) > self.max_seq_len:
                tokens = tokens[: self.max_seq_len]

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)

            pad_len = self.max_seq_len - 1 - len(input_ids)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id or 0
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id)])
                labels = torch.cat([labels, torch.full((pad_len,), -100)])

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "task": example["task"],
                }
            )

    def _build_char_token_map(self) -> Dict[str, int]:
        chars = "0123456789()+-*=? "
        chars += "abcdefghijklmnopqrstuvwxyz"
        chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # For future-proofing
        chars += ":,%<>'"  # punctuation used in prompts

        token_map = {}
        for ch in set(chars):
            encoded = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(encoded) == 1:
                token_map[ch] = encoded[0]
        return token_map

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx % len(self.examples)]


def create_algorithmic_dataset(
    tokenizer,
    fixed: bool = False,
    num_examples: int = 100000,
    max_seq_len: int = 128,
    tasks: Optional[List[str]] = None,
    seed: int = 42,
    difficulty_value=None,
) -> Union[AlgorithmicDataset, FixedAlgorithmicDataset]:
    """Factory that returns fixed or infinite dataset based on flag."""

    if fixed:
        return FixedAlgorithmicDataset(
            tokenizer=tokenizer,
            num_examples=num_examples,
            max_seq_len=max_seq_len,
            tasks=tasks,
            seed=seed,
        )

    return AlgorithmicDataset(
        tokenizer=tokenizer,
        num_examples=num_examples,
        max_seq_len=max_seq_len,
        tasks=tasks,
        seed=seed,
        difficulty_value=difficulty_value,
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


LANGUAGE_CONFIGS = {
    "tinystories": LanguageDataConfig(
        name="roneneldan/TinyStories",
        subset=None,
        split="train",
        text_field="text",
        weight=0.5,
        max_samples=500_000,
    ),
    "wikipedia": LanguageDataConfig(
        name="wikipedia",
        subset="20220301.simple",  # Simple English for efficiency
        split="train",
        text_field="text",
        weight=0.3,
        max_samples=200_000,
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

    try:
        if config.subset:
            ds = load_dataset(
                config.name,
                config.subset,
                split=config.split,
                streaming=True,
                trust_remote_code=True,
            )
        else:
            ds = load_dataset(
                config.name,
                split=config.split,
                streaming=True,
                trust_remote_code=True,
            )
    except Exception as e:
        if log:
            print(f"  Failed: {e}")
        return iter(())

    num_shards = max(1, num_shards)
    shard_index = min(max(shard_index, 0), num_shards - 1)
    requested_shard_index = shard_index
    shardable = True
    if num_shards > 1:
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
        return iter(())

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

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rng = random.Random(42 + worker_id)

        config_items = list(self.configs.items())
        rng.shuffle(config_items)

        active_streams: List[Tuple[LanguageDataConfig, Iterator[Dict[str, str]]]] = []

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
            active_streams.append((config, iter(stream)))

        while active_streams:
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
                continue

            tokens = self.tokenizer.encode(example["text"], add_special_tokens=True)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)

            pad_len = self.max_seq_len - 1 - len(input_ids)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id or 0
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id)])
                labels = torch.cat([labels, torch.full((pad_len,), -100)])

            yield {"input_ids": input_ids, "labels": labels}

            active_streams[chosen_index] = (config, stream)


# =============================================================================
# CURRICULUM SAMPLER
# =============================================================================

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
    ):
        if not 0.0 <= persistent_alg_frac <= 0.9:
            raise ValueError(
                "persistent_alg_frac must be between 0.0 and 0.9 (recommended 0.05–0.3)"
            )

        self.alg_loader = alg_loader
        self.lang_loader = lang_loader
        self.lex_loader = lex_loader

        self.alg_iter = iter(self.alg_loader)
        self.lang_iter = iter(self.lang_loader)
        self.lex_iter = iter(self.lex_loader) if self.lex_loader is not None else None

        self.total_tokens = total_tokens
        self.alg_tokens = alg_tokens
        self.mix_band_tokens = int(min(max(mix_band_tokens, 0), self.alg_tokens))
        self.persistent_alg_frac = persistent_alg_frac
        self.lexical_frac_phase1 = lexical_frac_phase1 if lex_loader is not None else 0.0
        self.tokens_seen = 0

        self.rng = random.Random(seed)

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
        choices = ["alg", "lang", "lex"]
        cumulative = 0.0
        pick = self.rng.random()
        for choice in choices:
            cumulative += probs[choice]
            if pick <= cumulative:
                return choice
        return "lang"  # Fallback

    def _next_from(self, which: str) -> Dict[str, torch.Tensor]:
        if which == "alg":
            loader, iter_attr = self.alg_loader, "alg_iter"
        elif which == "lang":
            loader, iter_attr = self.lang_loader, "lang_iter"
        else:
            loader, iter_attr = self.lex_loader, "lex_iter"

        iterator = getattr(self, iter_attr)
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            setattr(self, iter_attr, iterator)
            return next(iterator)

    def next_batch(self) -> Dict[str, torch.Tensor]:
        source = self._sample_source()
        if source == "alg":
            batch = self._next_from("alg")
        elif source == "lex" and self.lex_iter is not None and self.lex_loader is not None:
            batch = self._next_from("lex")
        else:
            batch = self._next_from("lang")

        # Count tokens (non-padding)
        n_tokens = (batch["labels"] != -100).sum().item()
        self.tokens_seen += n_tokens

        return batch

    def reset(self):
        self.tokens_seen = 0
        self.alg_iter = iter(self.alg_loader)
        self.lang_iter = iter(self.lang_loader)
        self.lex_iter = iter(self.lex_loader) if self.lex_loader is not None else None


# =============================================================================
# NEEDLE-IN-HAYSTACK EVALUATION
# =============================================================================

class NeedleInHaystackGenerator:
    """
    Generates needle-in-haystack retrieval tasks for evaluating long-context.
    """
    
    def __init__(self, tokenizer, context_length: int = 4096):
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        # Filler text (Paul Graham essays style)
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
    
    def generate(self, needle_depth: float = 0.5) -> Dict:
        """
        Generate a needle-in-haystack example.
        
        Args:
            needle_depth: Position of needle (0.0 = start, 1.0 = end)
        """
        # Generate random needle
        secret_number = random.randint(1000, 9999)
        needle = f"The secret code is {secret_number}."
        
        # Build haystack
        haystack_tokens = []
        needle_tokens = self.tokenizer.encode(needle, add_special_tokens=False)
        target_length = self.context_length - len(needle_tokens) - 50  # Buffer for question
        
        while len(haystack_tokens) < target_length:
            sentence = random.choice(self.filler_sentences)
            tokens = self.tokenizer.encode(" " + sentence, add_special_tokens=False)
            haystack_tokens.extend(tokens)
        
        haystack_tokens = haystack_tokens[:target_length]
        
        # Insert needle at specified depth
        insert_pos = int(len(haystack_tokens) * needle_depth)
        full_tokens = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
        
        # Add question
        question = " Question: What is the secret code? Answer:"
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        full_tokens = full_tokens + question_tokens
        
        return {
            "input_ids": torch.tensor(full_tokens),
            "answer": str(secret_number),
            "needle_depth": needle_depth,
            "context_length": len(full_tokens),
        }
    
    def generate_sweep(self, n_depths: int = 10) -> List[Dict]:
        """Generate examples across depth positions."""
        examples = []
        for i in range(n_depths):
            depth = i / (n_depths - 1) if n_depths > 1 else 0.5
            examples.append(self.generate(needle_depth=depth))
        return examples


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
        return {
            "input": f"{a} + {b} =",
            "answer": str(a + b),
            "length": n_digits,
            "task": "addition_ood",
        }
    
    @staticmethod
    def parity_ood(length: int) -> Dict:
        """Parity with longer sequences than training."""
        bits = [random.randint(0, 1) for _ in range(length)]
        parity = sum(bits) % 2
        return {
            "input": f"parity({''.join(map(str, bits))}) =",
            "answer": str(parity),
            "length": length,
            "task": "parity_ood",
        }
    
    @staticmethod
    def copy_ood(length: int) -> Dict:
        """Copy with longer sequences than training."""
        seq = [random.randint(0, 9) for _ in range(length)]
        seq_str = " ".join(map(str, seq))
        return {
            "input": f"copy: {seq_str} ->",
            "answer": seq_str,
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
        seed=42,
    )
    
    lang_dataset = LanguageDataset(
        tokenizer=tokenizer,
        max_seq_len=lang_seq_len,
    )
    
    alg_loader = DataLoader(
        alg_dataset,
        batch_size=alg_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    lang_loader = DataLoader(
        lang_dataset,
        batch_size=lang_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
