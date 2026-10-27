"""Probe: what happens to token-table rows the training data never uses?

Context: docs/research/lexical-grokking-2026-10.md, section 4.1, and issue #186.
build_optimizer exempts tok_emb (tied with lm_head) from weight decay, to protect
tokens unseen in an algorithmic phase. This probe trains a tiny tied model on
tokens 0..49 of a 1,000-token vocabulary and measures rows 50..999.

Run from the repo root:  python docs/research/unseen_token_rows_probe.py
CPU, about one minute. Toy scale (d_model 64, 2 layers, 2,000 updates).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model import BaselineTransformer, TransformerConfig  # noqa: E402
from train import build_optimizer  # noqa: E402

V, SEEN, STEPS = 1000, 50, 2000


def unseen_stats(weight):
    rows = weight.detach()[SEEN:]
    unit = torch.nn.functional.normalize(rows, dim=1)
    off_diag = ~torch.eye(len(rows), dtype=torch.bool)
    return rows.norm(dim=1).mean().item(), (unit @ unit.T)[off_diag].mean().item()


def run(label, tie=True, decay_table=False, eps_table=None):
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=V, max_seq_len=16, d_model=64, n_layers=2,
                            n_heads=4, dropout=0.0, tie_weights=tie)
    model = BaselineTransformer(cfg)
    opt = build_optimizer(model, lr=1e-3, weight_decay=1.0)
    if decay_table or eps_table is not None:
        table = model.tok_emb.weight
        no_decay = opt.param_groups[1]
        no_decay["params"] = [p for p in no_decay["params"] if p is not table]
        group = {"params": [table], "weight_decay": 1.0 if decay_table else 0.0}
        if eps_table is not None:
            group["eps"] = eps_table
        opt.add_param_group(group)
    for _ in range(STEPS):
        x = torch.randint(0, SEEN, (32, 16))
        _, loss, _ = model(x, labels=x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    n_in, c_in = unseen_stats(model.tok_emb.weight)
    line = f"{label:<38} input rows: norm {n_in:6.3f}  mean cos {c_in:6.3f}"
    if not tie:
        n_out, c_out = unseen_stats(model.lm_head.weight)
        line += f" | output rows: norm {n_out:6.3f}  mean cos {c_out:6.3f}"
    print(line)


if __name__ == "__main__":
    print(f"init: row norm ~{0.02 * 64 ** 0.5:.3f}, mean cos ~0")
    run("tied, table exempt (repo default)")
    run("tied, table decayed (wd 1.0)", decay_table=True)
    run("tied, exempt, Adam eps 1e-4 on table", eps_table=1e-4)
    run("tied, exempt, Adam eps 1e-2 on table", eps_table=1e-2)
    run("untied, both exempt", tie=False)
