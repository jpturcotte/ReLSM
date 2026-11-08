"""Same-answer patching (H9 in hypotheses.md), final weights.

Recipients: diagonal train pairs (a, a); donors: off-diagonal train pairs with the same answer (up to 8 each).
Off-diagonal reference: up to 300 off-diagonal train recipients, up to 2 same-answer off-diagonal donors each
(not the recipient, not its mirror). Patches (final norm free):
  P_op0  operand part (a and b terms) of attn0's write at '=' from the donor; mlp0 onward recomputed
  P_size recipient's own operand part rescaled to the donor's RMS (size-only control)
  P_eq   the '=' residual after layer 0 from the donor
  P_ops  the residuals after layer 0 at positions a, op, b from the donor
  P_both all positions after layer 0 from the donor (must equal the donor)

Usage: python patch_test.py --device cuda --out OUT.jsonl RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[4] / "tools"
sys.path.insert(0, str(WT))
import torch  # noqa: E402

import grok_mech_probe as mp  # noqa: E402
from model import apply_rotary_emb  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from restore_test import peak_memory_gb  # noqa: E402

OPS = [0, 2]
K_DIAG, N_OFF, K_OFF = 8, 300, 2
PATCHES = ("P_op0", "P_size", "P_eq", "P_ops", "P_both")


def attn(blk, h, T, device):
    at = blk.attn
    B = h.shape[0]
    u = blk.attn_norm(h)
    q = at.q_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
    k = at.k_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
    v = at.v_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
    cos, sin = at.rotary(T, device)
    pos = torch.arange(T, device=device).unsqueeze(0)
    q, k = apply_rotary_emb(q, k, cos, sin, pos)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    pr = ((q @ k.transpose(-1, -2)) * at.scale).masked_fill(~causal, float("-inf")).softmax(-1)
    return pr, v


@torch.no_grad()
def forward(model, x, op_new=None, after0=None, mask=None):
    """op_new [B, d]: replaces the operand part of attn0's write at '='. after0 [B, T, d] with mask [T] bool:
    replaces the residual after layer 0 at the masked positions. Returns logits at '=' [B, V], the residual after
    layer 0 [B, T, d], and the operand part of attn0 at '=' [B, d] (before any replacement)."""
    T = x.shape[1]
    h = model.tok_emb(x)
    blk0, blk1 = model.layers[0], model.layers[1]
    pr, v = attn(blk0, h, T, x.device)
    B = h.shape[0]
    ao = blk0.attn.o_proj((pr @ v).transpose(1, 2).reshape(B, T, -1))
    po = torch.zeros_like(pr)
    po[:, :, T - 1, OPS] = pr[:, :, T - 1, OPS]
    op = blk0.attn.o_proj((po @ v).transpose(1, 2).reshape(B, T, -1))[:, -1]
    if op_new is not None:
        ao = ao.clone()
        ao[:, -1] = ao[:, -1] - op + op_new
    h = h + ao
    h = h + blk0.ff(blk0.ff_norm(h))
    h0 = h
    if after0 is not None:
        h = torch.where(mask.view(1, T, 1), after0, h)
    pr, v = attn(blk1, h, T, x.device)
    h = h + blk1.attn.o_proj((pr @ v).transpose(1, 2).reshape(B, T, -1))
    h = h + blk1.ff(blk1.ff_norm(h))
    logits = model.lm_head(model.norm(h[:, -1])).float()
    return logits, h0, op


def lse(logits, y, cols):
    return mp.two_margins(logits[:, cols], y)[1].float()


def rms(v):
    return v.float().pow(2).mean(-1, keepdim=True).sqrt()


def pairs(rec_idx, cand_mask, y, a, b, k, gen, exclude_mirror):
    R, D = [], []
    for i in rec_idx.tolist():
        m = cand_mask & (y == y[i])
        m[i] = False
        if exclude_mirror:
            m &= ~((a == b[i]) & (b == a[i]))
        c = m.nonzero().squeeze(1)
        if len(c) == 0:
            continue
        pick = c[torch.randperm(len(c), generator=gen)[:k]]
        R += [i] * len(pick)
        D += pick.tolist()
    return torch.tensor(R, dtype=torch.long), torch.tensor(D, dtype=torch.long)


def patched(model, x, R, D, h0, op, kind, chunk, T):
    out = []
    for s in range(0, len(R), chunk):
        r, d = R[s:s + chunk], D[s:s + chunk]
        xr = x[r]
        if kind == "P_op0":
            lg, _, _ = forward(model, xr, op_new=op[d])
        elif kind == "P_size":
            lg, _, _ = forward(model, xr, op_new=(op[r].float() * rms(op[d]) / rms(op[r])).to(op.dtype))
        else:
            mask = {"P_eq": [False] * (T - 1) + [True], "P_ops": [True] * (T - 1) + [False],
                    "P_both": [True] * T}[kind]
            lg, _, _ = forward(model, xr, after0=h0[d], mask=torch.tensor(mask, device=x.device))
        out.append(lg)
    return torch.cat(out)


def one(run, device, chunk):
    model, args, update = mp.load(run, "final")
    model.to(device)
    x, y, cols, a, b, n, groups = mp.table(args)
    T = x.shape[1]
    t0 = time.time()
    xd, yd, cd = x.to(device), y.to(device), cols.to(device)

    L, H0, OP, err_model = [], [], [], 0.0
    for s in range(0, len(x), chunk):
        lg, h0, op = forward(model, xd[s:s + chunk])
        with torch.no_grad():
            ref = model(xd[s:s + chunk])
        ref = ref[0] if isinstance(ref, tuple) else ref
        ref = ref[:, -1] if ref.dim() == 3 else ref
        err_model = max(err_model, float((ref.float() - lg).abs().max()))
        L.append(lg), H0.append(h0), OP.append(op)
    L, H0, OP = torch.cat(L), torch.cat(H0), torch.cat(OP)
    base = lse(L, yd, cd).cpu()

    gen = torch.Generator().manual_seed(0)
    dtr, otr = groups["diag_train"], groups["off_train"]
    sets = {}
    sets["diag"] = pairs(dtr.nonzero().squeeze(1), otr.clone(), y, a, b, K_DIAG, gen, False)
    off_rec = otr.nonzero().squeeze(1)
    off_rec = off_rec[torch.randperm(len(off_rec), generator=gen)[:N_OFF]]
    sets["off"] = pairs(off_rec, otr.clone(), y, a, b, K_OFF, gen, True)

    out, checks = {}, {"model_forward_err": err_model}
    for name, (R, D) in sets.items():
        res = {"n_pairs": len(R), "n_recipients": int(R.unique().numel()) if len(R) else 0}
        if len(R) == 0:
            out[name] = res
            continue
        Rd, Dd = R.to(device), D.to(device)
        rec, don = base[R], base[D]
        res["rec_margin"], res["don_margin"] = float(rec.mean()), float(don.mean())
        res["op_rms_ratio_don_over_rec"] = float((rms(OP[Dd]) / rms(OP[Rd])).mean())
        for kind in PATCHES:
            lg = patched(model, xd, Rd, Dd, H0, OP, kind, chunk, T)
            m = lse(lg, yd[Rd], cd).cpu()
            gap = float((don - rec).mean())
            res[kind] = {"margin": float(m.mean()), "delta": float((m - rec).mean()),
                         "closure": float((m - rec).mean()) / gap if abs(gap) > 1e-9 else None,
                         "acc": float((lg[:, cd].argmax(1) == yd[Rd]).float().mean())}
            if kind == "P_both":
                checks[f"both_equals_donor_{name}"] = float((lg - L[Dd]).abs().max())
        # self no-ops
        for kind in ("P_op0", "P_both"):
            lg = patched(model, xd, Rd, Rd, H0, OP, kind, chunk, T)
            checks[f"self_noop_{kind}_{name}"] = float((lg - L[Rd]).abs().max())
        out[name] = res

    peak_ram, peak_commit = peak_memory_gb()
    return {"run": str(run), "update": update, "operands": args.operands,
            "task": mp.gl.task_of(args) if not mp.gl.multi_task(args) else args.tasks[0],
            "checks": checks, "results": out, "seconds": time.time() - t0,
            "peak_ram_gb": peak_ram, "peak_commit_gb": peak_commit,
            "gpu_used_gb": (lambda f: (f[1] - f[0]) / 1e9)(torch.cuda.mem_get_info()) if device == "cuda" else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=mp.CHUNK)
    ap.add_argument("runs", nargs="+")
    o = ap.parse_args()
    done = set()
    if Path(o.out).exists():
        done = {json.loads(l)["run"] for l in open(o.out) if l.startswith("{")}
    for r in o.runs:
        if str(Path(r)) in done:
            continue
        rec = one(Path(r), o.device, o.chunk)
        with open(o.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        d = rec["results"]["diag"]
        print(json.dumps({"run": rec["run"], "checks": {k: f"{v:.1e}" for k, v in rec["checks"].items()},
                          "diag_pairs": d.get("n_pairs"), "rec": round(d.get("rec_margin", 0), 2),
                          "don": round(d.get("don_margin", 0), 2),
                          "closure": {k: round(d[k]["closure"], 3) for k in PATCHES if k in d and d[k]["closure"] is not None},
                          "off_delta": {k: round(rec["results"]["off"][k]["delta"], 3) for k in PATCHES
                                        if k in rec["results"]["off"]},
                          "ram": rec["peak_ram_gb"], "commit": rec["peak_commit_gb"], "gpu": rec["gpu_used_gb"],
                          "s": round(rec["seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
