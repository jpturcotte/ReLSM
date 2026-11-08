"""Mirror test (H5 in hypotheses.md): split each run's train pairs by whether the mirror (b, a) is also in
train, and compare lse margins with the diagonal. One run per call; appends one JSON line to --out.

Usage: python mirror_test.py --out OUT.jsonl RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[4] / "tools"
sys.path.insert(0, str(WT))
import torch  # noqa: E402

import grok_mech_probe as mp  # noqa: E402


def stats(m, mask):
    v = m[mask]
    n = int(mask.sum())
    return {"n": n, "mean": float(v.mean()) if n else None,
            "se": float(v.std() / n ** 0.5) if n > 1 else None}


def one(run, ckpt="final"):
    model, args, update = mp.load(run, ckpt)
    x, y, cols, a, b, n, groups = mp.table(args)
    logits = mp.forward_logits(model, x)
    _, lse = mp.two_margins(logits[:, -1, cols], y)
    lse = lse.float()
    is_train = groups["diag_train"] | groups["off_train"]
    # the mirror (b, a) of each pair, looked up in the same pair table (ids are a*n + b over valid pairs)
    key = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(a.tolist(), b.tolist()))}
    mirror_idx = torch.tensor([key.get((int(j), int(i)), -1) for i, j in zip(a.tolist(), b.tolist())])
    has = mirror_idx >= 0
    mirror_train = torch.zeros_like(is_train)
    mirror_train[has] = is_train[mirror_idx[has]]
    off_tr = groups["off_train"]
    g = {"diag": groups["diag_train"],
         "mirror": off_tr & mirror_train,
         "lone": off_tr & ~mirror_train,
         "test_seen": ~is_train & (a != b) & mirror_train,  # held out, but the mirror is in train
         "test_unseen": groups["off_test"]}  # the probe's off_test: held out with the mirror held out too
    if getattr(args, "holdout_line", None) is not None:
        line = mp.gl.line_mask(a, b, n, args.holdout_line)
        for k in ("mirror", "lone", "test_seen", "test_unseen"):
            g[k] = g[k] & ~line
    s = {k: stats(lse, v) for k, v in g.items()}
    md = s["diag"]["mean"]
    d_mirror = s["mirror"]["mean"] - md if s["mirror"]["n"] else None
    d_lone = s["lone"]["mean"] - md if s["lone"]["n"] else None
    return {"run": str(run), "ckpt": ckpt, "update": update, "pairs": args.pairs, "operands": args.operands,
            "task": mp.gl.task_of(args) if not mp.gl.multi_task(args) else args.tasks[0],
            "groups": s, "D_mirror": d_mirror, "D_lone": d_lone,
            "R": d_lone / d_mirror if d_mirror and d_lone is not None else None,
            "mirror_minus_lone": (s["mirror"]["mean"] - s["lone"]["mean"]) if d_mirror is not None and d_lone is not None else None,
            "acc_lone": float((logits[:, -1, cols].argmax(1) == y)[g["lone"]].float().mean()) if s["lone"]["n"] else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--all_ckpts", action="store_true", help="every ckpt_*.pt in the run, then final")
    ap.add_argument("runs", nargs="+")
    o = ap.parse_args()
    for r in o.runs:
        r = Path(r)
        ckpts = (sorted(p.name for p in r.glob("ckpt_*.pt")) if o.all_ckpts else []) + ["final"]
        for c in ckpts:
            rec = one(r, c)
            with open(o.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec[k] for k in ("run", "ckpt", "D_mirror", "D_lone", "R", "mirror_minus_lone")}), flush=True)


if __name__ == "__main__":
    main()
