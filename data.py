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
from typing import List, Dict, Optional, Tuple, Iterator, Callable, Iterable
from dataclasses import dataclass
from itertools import cycle
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
    def modular_arithmetic(mod: int = 97, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Modular addition: (a + b) mod p"""
        rng = rng or random
        a = rng.randint(0, mod - 1)
        b = rng.randint(0, mod - 1)
        result = (a + b) % mod
        
        return {
            "text": f"({a} + {b}) mod {mod} = {result}",
            "input": f"({a} + {b}) mod {mod} =",
            "target": str(result),
            "task": "mod_add",
        }
    
    @staticmethod
    def parity(length: int = 8, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Parity of binary string"""
        rng = rng or random
        bits = [rng.randint(0, 1) for _ in range(length)]
        parity = sum(bits) % 2
        bit_str = "".join(map(str, bits))
        
        return {
            "text": f"parity({bit_str}) = {parity}",
            "input": f"parity({bit_str}) =",
            "target": str(parity),
            "task": "parity",
        }
    
    @staticmethod
    def addition(max_digits: int = 4, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Multi-digit addition"""
        rng = rng or random
        digits = rng.randint(1, max_digits)
        a = rng.randint(10**(digits-1), 10**digits - 1) if digits > 1 else rng.randint(0, 9)
        b = rng.randint(10**(digits-1), 10**digits - 1) if digits > 1 else rng.randint(0, 9)
        result = a + b
        
        return {
            "text": f"{a} + {b} = {result}",
            "input": f"{a} + {b} =",
            "target": str(result),
            "task": "addition",
        }
    
    @staticmethod
    def multiplication(max_val: int = 99, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Two-digit multiplication"""
        rng = rng or random
        a = rng.randint(2, max_val)
        b = rng.randint(2, max_val)
        result = a * b
        
        return {
            "text": f"{a} * {b} = {result}",
            "input": f"{a} * {b} =",
            "target": str(result),
            "task": "multiplication",
        }
    
    @staticmethod
    def copy_sequence(length: int = 8, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Copy task - tests basic sequence modeling"""
        rng = rng or random
        seq = [rng.randint(0, 9) for _ in range(length)]
        seq_str = " ".join(map(str, seq))
        
        return {
            "text": f"copy: {seq_str} -> {seq_str}",
            "input": f"copy: {seq_str} ->",
            "target": seq_str,
            "task": "copy",
        }
    
    @staticmethod
    def reverse_sequence(length: int = 8, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Reverse task - tests sequential reasoning"""
        rng = rng or random
        seq = [rng.randint(0, 9) for _ in range(length)]
        seq_str = " ".join(map(str, seq))
        rev_str = " ".join(map(str, reversed(seq)))
        
        return {
            "text": f"reverse: {seq_str} -> {rev_str}",
            "input": f"reverse: {seq_str} ->",
            "target": rev_str,
            "task": "reverse",
        }
    
    @staticmethod
    def dyck_language(max_depth: int = 4, length: int = 8, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Dyck language (balanced parentheses) - tests stack operations"""
        rng = rng or random
        # Generate valid Dyck sequence
        seq = []
        depth = 0
        for _ in range(length):
            if depth == 0:
                seq.append("(")
                depth += 1
            elif depth >= max_depth:
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
        
        return {
            "text": f"dyck: {seq_str} valid",
            "input": f"dyck: {seq_str}",
            "target": "valid",
            "task": "dyck",
        }
    
    @staticmethod
    def chain_arithmetic(n_ops: int = 3, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Chain of arithmetic operations"""
        rng = rng or random
        val = rng.randint(1, 50)
        expr = str(val)

        for _ in range(n_ops):
            op = rng.choice(["+", "-"])
            operand = rng.randint(1, 20)
            expr += f" {op} {operand}"
            val = val + operand if op == "+" else val - operand
        
        return {
            "text": f"calc: {expr} = {val}",
            "input": f"calc: {expr} =",
            "target": str(val),
            "task": "chain",
        }
    
    @staticmethod
    def comparison(max_val: int = 1000, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Number comparison"""
        rng = rng or random
        a = rng.randint(1, max_val)
        b = rng.randint(1, max_val)
        result = ">" if a > b else ("<" if a < b else "=")
        
        return {
            "text": f"compare: {a} ? {b} -> {result}",
            "input": f"compare: {a} ? {b} ->",
            "target": result,
            "task": "compare",
        }
    
    @staticmethod
    def successor(max_val: int = 1000, rng: Optional[random.Random] = None) -> Dict[str, str]:
        """Successor function"""
        rng = rng or random
        n = rng.randint(0, max_val)
        return {
            "text": f"succ({n}) = {n + 1}",
            "input": f"succ({n}) =",
            "target": str(n + 1),
            "task": "successor",
        }

    @classmethod
    def generate_batch(
        cls,
        n: int,
        tasks: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Dict]:
        """Generate mixed batch of algorithmic tasks."""
        rng = rng or random
        generators = cls._get_generators()
        
        if tasks is None:
            tasks = list(generators.keys())
        
        return [cls.generate_example(tasks=tasks, rng=rng, generators=generators) for _ in range(n)]

    @classmethod
    def generate_example(
        cls,
        tasks: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
        generators: Optional[Dict[str, Callable]] = None,
    ) -> Dict:
        rng = rng or random
        if generators is None:
            generators = cls._get_generators()

        if tasks is None:
            tasks = list(generators.keys())

        task = rng.choice(tasks)
        return generators[task](rng=rng)

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
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.seed = seed

        self._char_token_map = self._build_char_token_map()

    def _build_char_token_map(self) -> Dict[str, int]:
        """Precompute a fast char->id map for synthetic strings."""
        chars = "0123456789()+-*=? "
        chars += "abcdefghijklmnopqrstuvwxyz"
        chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # For future-proofing
        chars += ":,"  # punctuation used in prompts

        token_map = {}
        for ch in set(chars):
            encoded = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(encoded) == 1:
                token_map[ch] = encoded[0]
        return token_map

    def _encode_text(self, text: str) -> List[int]:
        """Encode text quickly using the char map when possible."""
        if self._char_token_map and all(ch in self._char_token_map for ch in text):
            core_tokens = [self._char_token_map[ch] for ch in text]
            return self.tokenizer.build_inputs_with_special_tokens(core_tokens)
        return self.tokenizer.encode(text, add_special_tokens=True)

    def __len__(self):
        return self.num_examples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()

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

        for _ in range(worker_id, self.num_examples, num_workers):
            example = AlgorithmicGenerator.generate_example(
                tasks=self.tasks, rng=rng, generators=generators
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

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "task": example["task"],
            }


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
            num_shards = 1
            shard_index = 0

    max_samples = max_samples_override if max_samples_override is not None else config.max_samples

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

        config_items = list(self.configs.items())
        random.Random(42 + worker_id).shuffle(config_items)

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
            next_round: List[Tuple[LanguageDataConfig, Iterator[Dict[str, str]]]] = []
            for config, stream in active_streams:
                try:
                    example = next(stream)
                except StopIteration:
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

                next_round.append((config, stream))

            active_streams = next_round


# =============================================================================
# CURRICULUM SAMPLER
# =============================================================================

class CurriculumSampler:
    """
    Implements the split curriculum:
    - First N tokens: algorithmic data (Phase 1)
    - Remaining tokens: language data (Phase 2)
    """
    
    def __init__(
        self,
        alg_loader: DataLoader,
        lang_loader: DataLoader,
        alg_tokens: int,      # Tokens for Phase 1
        total_tokens: int,    # Total token budget
    ):
        self.alg_iter = cycle(iter(alg_loader))
        self.lang_iter = cycle(iter(lang_loader))
        self.alg_tokens = alg_tokens
        self.total_tokens = total_tokens
        self.tokens_seen = 0
    
    @property
    def phase(self) -> str:
        return "algorithmic" if self.tokens_seen < self.alg_tokens else "language"
    
    @property
    def progress(self) -> float:
        return self.tokens_seen / self.total_tokens
    
    def next_batch(self) -> Dict[str, torch.Tensor]:
        if self.tokens_seen < self.alg_tokens:
            batch = next(self.alg_iter)
        else:
            batch = next(self.lang_iter)
        
        # Count tokens (non-padding)
        n_tokens = (batch["labels"] != -100).sum().item()
        self.tokens_seen += n_tokens
        
        return batch
    
    def reset(self):
        self.tokens_seen = 0


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
