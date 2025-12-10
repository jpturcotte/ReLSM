"""
Ablation Ladder Evaluation
==========================
Comprehensive evaluation for comparing variants:

1. ALGORITHMIC EXACT-MATCH: Per-task accuracy on synthetic tasks
2. OOD LENGTH: Generalization to longer sequences than training
3. NEEDLE RETRIEVAL: Long-context recall at 4k/16k
4. PERPLEXITY: Language modeling quality
5. REASONING QA: Word problem accuracy

Outputs structured JSON for easy comparison across variants.
"""

import os
import re
import json
import math
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


@dataclass
class EvalResult:
    task: str
    metric: str
    score: float
    n_examples: int
    details: Optional[Dict] = None


class AlgorithmicEvaluator:
    """Evaluate exact-match accuracy on algorithmic tasks."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def _generate(self, prompt: str, max_tokens: int = 20) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_k=1,
            )
        
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()
    
    def _extract_answer(self, text: str) -> str:
        """Extract first number or word from response."""
        text = text.split("\n")[0].strip()
        # Try to extract number
        numbers = re.findall(r'-?\d+', text)
        if numbers:
            return numbers[0]
        # Otherwise first word
        words = text.split()
        return words[0] if words else ""
    
    def evaluate_task(self, task: str, n_examples: int = 100) -> EvalResult:
        """Evaluate a single task type."""
        from data import AlgorithmicGenerator
        
        correct = 0
        examples = AlgorithmicGenerator.generate_batch(n_examples, tasks=[task])
        
        for ex in examples:
            response = self._generate(ex["input"])
            predicted = self._extract_answer(response)
            expected = ex["target"].strip()
            
            if predicted == expected:
                correct += 1
        
        return EvalResult(
            task=task,
            metric="accuracy",
            score=correct / n_examples,
            n_examples=n_examples,
        )
    
    def run_all(self, n_per_task: int = 100) -> List[EvalResult]:
        """Run all algorithmic evaluations."""
        tasks = [
            "mod_add", "parity", "addition", "multiplication",
            "copy", "reverse", "chain", "compare", "successor"
        ]
        
        results = []
        for task in tasks:
            result = self.evaluate_task(task, n_per_task)
            results.append(result)
            print(f"  {task}: {result.score*100:.1f}%")
        
        # Overall
        total_correct = sum(r.score * r.n_examples for r in results)
        total_examples = sum(r.n_examples for r in results)
        
        results.append(EvalResult(
            task="overall",
            metric="accuracy",
            score=total_correct / total_examples,
            n_examples=total_examples,
        ))
        
        return results


class OODLengthEvaluator:
    """
    Evaluate generalization to longer-than-training sequences.
    Key test: Does recursion help extrapolation?
    """
    
    def __init__(self, model, tokenizer, device, train_length: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.train_length = train_length
    
    def _generate(self, prompt: str, max_tokens: int = 50) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_k=1,
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip()
    
    def evaluate_parity_ood(self, lengths: List[int], n_per_length: int = 20) -> EvalResult:
        """Test parity at various lengths."""
        results_by_length = {}
        
        for length in lengths:
            correct = 0
            for _ in range(n_per_length):
                bits = [random.randint(0, 1) for _ in range(length)]
                expected = str(sum(bits) % 2)
                prompt = f"parity({''.join(map(str, bits))}) ="
                
                response = self._generate(prompt)
                predicted = response.split()[0] if response else ""
                
                if predicted == expected:
                    correct += 1
            
            results_by_length[length] = correct / n_per_length
        
        return EvalResult(
            task="parity_ood",
            metric="accuracy_by_length",
            score=sum(results_by_length.values()) / len(results_by_length),
            n_examples=len(lengths) * n_per_length,
            details={"by_length": results_by_length},
        )
    
    def evaluate_addition_ood(self, digit_counts: List[int], n_per_count: int = 20) -> EvalResult:
        """Test addition at various digit counts."""
        results_by_digits = {}
        
        for n_digits in digit_counts:
            correct = 0
            for _ in range(n_per_count):
                a = random.randint(10**(n_digits-1), 10**n_digits - 1)
                b = random.randint(10**(n_digits-1), 10**n_digits - 1)
                expected = str(a + b)
                prompt = f"{a} + {b} ="
                
                response = self._generate(prompt)
                numbers = re.findall(r'\d+', response)
                predicted = numbers[0] if numbers else ""
                
                if predicted == expected:
                    correct += 1
            
            results_by_digits[n_digits] = correct / n_per_count
        
        return EvalResult(
            task="addition_ood",
            metric="accuracy_by_digits",
            score=sum(results_by_digits.values()) / len(results_by_digits),
            n_examples=len(digit_counts) * n_per_count,
            details={"by_digits": results_by_digits},
        )
    
    def run_all(self) -> List[EvalResult]:
        """Run OOD length tests."""
        results = []
        
        # Parity: trained on 8, test up to 64
        print("  Testing parity OOD...")
        parity_result = self.evaluate_parity_ood([8, 16, 24, 32, 48, 64], n_per_length=20)
        results.append(parity_result)
        print(f"    Overall: {parity_result.score*100:.1f}%")
        for length, acc in parity_result.details["by_length"].items():
            print(f"    len={length}: {acc*100:.1f}%")
        
        # Addition: trained on 1-4 digits, test up to 8
        print("  Testing addition OOD...")
        add_result = self.evaluate_addition_ood([2, 4, 5, 6, 7, 8], n_per_count=20)
        results.append(add_result)
        print(f"    Overall: {add_result.score*100:.1f}%")
        for digits, acc in add_result.details["by_digits"].items():
            print(f"    {digits}-digit: {acc*100:.1f}%")
        
        return results


class NeedleEvaluator:
    """
    Evaluate needle-in-haystack retrieval at various context lengths.
    Tests long-context memory vs KV cache pressure.
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.filler = [
            "The best way to learn is by doing.",
            "Ideas matter less than execution.",
            "Start small and iterate quickly.",
            "Focus on what matters most.",
            "Simple solutions beat complex ones.",
        ]
    
    def _build_haystack(self, context_length: int, needle: str, depth: float) -> Tuple[str, str]:
        """Build haystack with needle at specified depth."""
        needle_tokens = len(self.tokenizer.encode(needle))
        target_filler = context_length - needle_tokens - 50
        
        filler_text = ""
        while len(self.tokenizer.encode(filler_text)) < target_filler:
            filler_text += " " + random.choice(self.filler)
        
        # Trim to exact length
        filler_tokens = self.tokenizer.encode(filler_text)[:target_filler]
        filler_text = self.tokenizer.decode(filler_tokens)
        
        # Insert needle at depth
        insert_pos = int(len(filler_text) * depth)
        full_text = filler_text[:insert_pos] + " " + needle + " " + filler_text[insert_pos:]
        
        return full_text
    
    def evaluate_retrieval(
        self, 
        context_length: int, 
        n_depths: int = 5, 
        n_per_depth: int = 5
    ) -> EvalResult:
        """Test retrieval at various needle depths."""
        results_by_depth = {}
        
        for i in range(n_depths):
            depth = i / (n_depths - 1) if n_depths > 1 else 0.5
            correct = 0
            
            for _ in range(n_per_depth):
                secret = random.randint(1000, 9999)
                needle = f"The secret code is {secret}."
                
                haystack = self._build_haystack(context_length, needle, depth)
                prompt = haystack + " Question: What is the secret code? Answer:"
                
                # Truncate if needed
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                if input_ids.size(1) > self.model.config.max_seq_len:
                    input_ids = input_ids[:, :self.model.config.max_seq_len]
                
                input_ids = input_ids.to(self.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=10,
                        temperature=0.1,
                        top_k=1,
                    )
                
                response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                response = response[len(prompt):].strip() if len(prompt) < len(response) else ""
                
                numbers = re.findall(r'\d+', response)
                predicted = numbers[0] if numbers else ""
                
                if predicted == str(secret):
                    correct += 1
            
            results_by_depth[f"depth_{depth:.1f}"] = correct / n_per_depth
        
        avg_score = sum(results_by_depth.values()) / len(results_by_depth)
        
        return EvalResult(
            task=f"needle_{context_length}",
            metric="retrieval_accuracy",
            score=avg_score,
            n_examples=n_depths * n_per_depth,
            details={"by_depth": results_by_depth, "context_length": context_length},
        )
    
    def run_all(self, context_lengths: List[int] = [1024, 2048, 4096]) -> List[EvalResult]:
        """Test at multiple context lengths."""
        results = []
        
        for ctx_len in context_lengths:
            if ctx_len > self.model.config.max_seq_len:
                print(f"  Skipping {ctx_len} (exceeds max_seq_len {self.model.config.max_seq_len})")
                continue
            
            print(f"  Testing needle retrieval @ {ctx_len}...")
            result = self.evaluate_retrieval(ctx_len)
            results.append(result)
            print(f"    Accuracy: {result.score*100:.1f}%")
        
        return results


class PerplexityEvaluator:
    """Compute perplexity on held-out text."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate(self, texts: List[str], name: str = "test") -> EvalResult:
        """Compute perplexity on a list of texts."""
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.model.config.max_seq_len,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                if tokens.size(1) < 2:
                    continue
                
                labels = tokens.clone()
                _, loss, _ = self.model(tokens, labels=labels)
                
                n_tokens = tokens.size(1) - 1
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        
        return EvalResult(
            task=f"perplexity_{name}",
            metric="perplexity",
            score=ppl,
            n_examples=len(texts),
        )
    
    def evaluate_tinystories(self, n: int = 100) -> EvalResult:
        """Evaluate on TinyStories validation set."""
        try:
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split="validation")
            texts = [ds[i]["text"] for i in range(min(n, len(ds)))]
            return self.evaluate(texts, "tinystories")
        except Exception as e:
            print(f"    Could not load TinyStories: {e}")
            return EvalResult("perplexity_tinystories", "perplexity", float("inf"), 0)


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    from model import UnifiedTransformer
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    model = UnifiedTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config


def run_full_evaluation(model, tokenizer, device, args) -> Dict:
    """Run complete evaluation suite."""
    results = {}
    
    # Algorithmic
    print("\n[Algorithmic Exact-Match]")
    alg_eval = AlgorithmicEvaluator(model, tokenizer, device)
    alg_results = alg_eval.run_all(n_per_task=args.n_alg)
    results["algorithmic"] = [asdict(r) for r in alg_results]
    
    # OOD Length
    print("\n[OOD Length Generalization]")
    ood_eval = OODLengthEvaluator(model, tokenizer, device)
    ood_results = ood_eval.run_all()
    results["ood_length"] = [asdict(r) for r in ood_results]
    
    # Needle retrieval
    print("\n[Needle-in-Haystack Retrieval]")
    needle_eval = NeedleEvaluator(model, tokenizer, device)
    needle_results = needle_eval.run_all(context_lengths=[512, 1024, 2048])
    results["needle_retrieval"] = [asdict(r) for r in needle_results]
    
    # Perplexity
    print("\n[Perplexity]")
    ppl_eval = PerplexityEvaluator(model, tokenizer, device)
    ppl_result = ppl_eval.evaluate_tinystories(n=args.n_ppl)
    results["perplexity"] = [asdict(ppl_result)]
    print(f"  TinyStories PPL: {ppl_result.score:.2f}")
    
    return results


def print_summary(results: Dict):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Algorithmic
    alg = results.get("algorithmic", [])
    overall = next((r for r in alg if r["task"] == "overall"), None)
    if overall:
        print(f"\nAlgorithmic accuracy: {overall['score']*100:.1f}%")
    
    # OOD
    ood = results.get("ood_length", [])
    for r in ood:
        print(f"OOD {r['task']}: {r['score']*100:.1f}%")
    
    # Needle
    needle = results.get("needle_retrieval", [])
    for r in needle:
        ctx = r.get("details", {}).get("context_length", "?")
        print(f"Needle @{ctx}: {r['score']*100:.1f}%")
    
    # PPL
    ppl = results.get("perplexity", [])
    for r in ppl:
        print(f"Perplexity ({r['task']}): {r['score']:.2f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on ablation ladder")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_alg", type=int, default=100, help="Examples per algorithmic task")
    parser.add_argument("--n_ppl", type=int, default=100, help="Perplexity examples")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    print(f"Model variant: {config.variant}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run evaluation
    results = run_full_evaluation(model, tokenizer, device, args)
    results["config"] = {
        "variant": config.variant,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "max_seq_len": config.max_seq_len,
    }
    
    # Summary
    print_summary(results)
    
    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
