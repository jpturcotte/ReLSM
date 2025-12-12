"""Lightweight self checks for the algorithmic evaluation helpers."""
import json
from typing import Dict, List

import torch

from data import AlgorithmicGenerator
from eval import ood_grid
from utils import build_dataset, evaluate_condition, seed_all


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


class AlwaysWrongModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        decoded_prompts: List[str] = []
        for ids, mask in zip(input_ids, attention_mask):
            length = int(mask.sum().item())
            decoded_prompts.append(self.tokenizer.decode(ids[:length], skip_special_tokens=True))
        outputs = []
        for prompt in decoded_prompts:
            generated = prompt + " wrong"
            outputs.append(self.tokenizer.encode(generated, return_tensors="pt")[0])
        max_len = max(len(o) for o in outputs)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded = []
        for o in outputs:
            if len(o) < max_len:
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

    # Dyck grid condition smoke test with always-correct stub
    dyck_cond = ood_grid.dyck_grid()[0]
    dyck_dataset = build_dataset("dyck", dyck_cond.n, dyck_cond.params, seed=2024)
    dyck_answer_map = {ex["input"].strip(): ex["target"] for ex in dyck_dataset}
    dyck_model = AlwaysCorrectModel(tok, dyck_answer_map)
    dyck_res = evaluate_condition(
        dyck_model,
        tok,
        task="dyck",
        condition="dyck_iid",
        params=dyck_cond.params,
        n=dyck_cond.n,
        device=torch.device("cpu"),
        max_new_tokens=dyck_cond.max_new_tokens,
        seed=2024,
        batch_size=dyck_cond.n,
    )
    assert dyck_res.correct == dyck_cond.n and dyck_res.accuracy == 1.0

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

    # Always-wrong stub to ensure failures are recorded
    wrong_model = AlwaysWrongModel(tok)
    wrong_res = evaluate_condition(
        wrong_model,
        tok,
        task="addition",
        condition="self_test_wrong",
        params=cond.params,
        n=8,
        device=torch.device("cpu"),
        max_new_tokens=32,
        seed=111,
        batch_size=4,
    )
    assert wrong_res.correct == 0, f"Expected zero correct, got {wrong_res.correct}"
    assert wrong_res.examples, "Expected some failure examples to be collected"
    print(json.dumps({"status": "ok", "samples": len(dataset)}, indent=2))


if __name__ == "__main__":
    run_self_checks()
