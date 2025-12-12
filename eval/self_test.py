"""Lightweight self checks for the algorithmic evaluation helpers."""
import json
from typing import Dict, List

import torch

from data import AlgorithmicGenerator
from eval import ood_grid
from eval.utils import build_dataset, evaluate_condition, seed_all


class AlwaysCorrectModel:
    def __init__(self, tokenizer, answer_map: Dict[str, str]):
        self.tokenizer = tokenizer
        self.answer_map = answer_map

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        decoded_prompts: List[str] = []
        for ids, mask in zip(input_ids, attention_mask):
            length = int(mask.sum().item())
            decoded_prompts.append(self.tokenizer.decode(ids[:length], skip_special_tokens=True))
        outputs = []
        for prompt in decoded_prompts:
            target = self.answer_map.get(prompt.strip(), "")
            generated = prompt + " " + target
            outputs.append(self.tokenizer.encode(generated, return_tensors="pt")[0])
        # pad to equal length
        max_len = max(len(o) for o in outputs)
        padded = []
        for o in outputs:
            if len(o) < max_len:
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                pad = torch.full((max_len - len(o),), pad_id, dtype=o.dtype)
                o = torch.cat([o, pad])
            padded.append(o)
        return torch.stack(padded)


def run_self_checks():
    from transformers import AutoTokenizer

    seed_all(42)
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Generator format smoke test
    ex = AlgorithmicGenerator.addition(rng=None)
    assert set(["input", "target", "task", "text"]) <= set(ex.keys())

    # Dyck balance check
    dyck_samples = build_dataset("dyck", 20, {"max_depth": 4, "length": 10}, seed=123)
    yes_count = sum(1 for s in dyck_samples if s["target"].lower() == "yes")
    no_count = sum(1 for s in dyck_samples if s["target"].lower() == "no")
    assert yes_count == no_count == 10

    # Always-correct stub
    cond = ood_grid.addition_grid()[0]
    dataset = build_dataset("addition", 8, cond.params, seed=999)
    answer_map = {ex["input"].strip(): ex["target"] for ex in dataset}
    model = AlwaysCorrectModel(tok, answer_map)
    res = evaluate_condition(
        model,
        tok,
        task="addition",
        condition="self_test",
        params=cond.params,
        n=8,
        device=torch.device("cpu"),
        max_new_tokens=32,
        seed=999,
        batch_size=4,
    )
    assert res.accuracy == 1.0, f"Expected perfect accuracy, got {res.accuracy}"
    print(json.dumps({"status": "ok", "samples": len(dataset)}, indent=2))


if __name__ == "__main__":
    run_self_checks()
