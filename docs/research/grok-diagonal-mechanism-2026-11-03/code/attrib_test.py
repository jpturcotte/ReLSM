"""Route-resolved integrated-gradient attribution (H10 in hypotheses.md), final weights.

Recipient: a diagonal train pair (or an off-diagonal control); donor: a same-answer off-diagonal train pair.
The rest of the model after layer 0 is F(E, P): E = '=' state after layer 0 (residual, layer-1 query, self K/V);
P = layer-1 inputs at positions 0, 1, 2 (ops = 0 and 2, opr = 1). All move together on a straight line from
recipient to donor; attribution of a route = integral of grad_route F . delta_route (32-point Gauss-Legendre).
Choice A: P interpolated as pre-norm residuals; choice B: as post-norm inputs (= interpolating K and V).

Usage: python attrib_test.py --device cuda --out OUT.jsonl RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[4] / "tools"
sys.path.insert(0, str(WT))
import torch  # noqa: E402

import grok_mech_probe as mp  # noqa: E402
from model import apply_rotary_emb  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from restore_test import peak_memory_gb  # noqa: E402
from patch_test import attn, pairs  # noqa: E402

N_NODES = 32
K_DIAG, N_OFF, K_OFF = 2, 100, 2


@torch.no_grad()
def layer0(model, x):
    """Residual after layer 0 at all positions [B, T, d]."""
    T = x.shape[1]
    h = model.tok_emb(x)
    blk0 = model.layers[0]
    pr, v = attn(blk0, h, T, x.device)
    B = h.shape[0]
    h = h + blk0.attn.o_proj((pr @ v).transpose(1, 2).reshape(B, T, -1))
    return h + blk0.ff(blk0.ff_norm(h))


def tail(model, E, P, post, y, cols):
    """Margin of the true answer from E [B, d] and P [B, T-1, d] (pre- or post-norm prefix inputs)."""
    blk = model.layers[1]
    at = blk.attn
    B, Tm1, _ = P.shape
    T = Tm1 + 1
    U = torch.cat([P if post else blk.attn_norm(P), blk.attn_norm(E)[:, None]], 1)  # [B, T, d]
    q = at.q_proj(U).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
    k = at.k_proj(U).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
    v = at.v_proj(U).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
    cos, sin = at.rotary(T, E.device)
    q, k = apply_rotary_emb(q, k, cos, sin, torch.arange(T, device=E.device).unsqueeze(0))
    s = (q[:, :, -1:] @ k.transpose(-1, -2)) * at.scale  # the '=' query sees every position
    out = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, 1, -1)
    h = E + at.o_proj(out)[:, 0]
    h = h + blk.ff(blk.ff_norm(h))
    logits = model.lm_head(model.norm(h))
    return mp.two_margins(logits[:, cols], y)[1]


def attribute(model, h0, R, D, y, cols, post, n_nodes=N_NODES):
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    ts, ws = (nodes + 1) / 2, weights / 2
    blk = model.layers[1]
    with torch.no_grad():
        E0, E1 = h0[R, -1], h0[D, -1]
        P0, P1 = h0[R, :-1], h0[D, :-1]
        if post:
            P0, P1 = blk.attn_norm(P0), blk.attn_norm(P1)
        dE, dP = E1 - E0, P1 - P0
        yy = y[R]
        m0, m1 = tail(model, E0, P0, post, yy, cols), tail(model, E1, P1, post, yy, cols)
    aE = torch.zeros(len(R), dtype=torch.float64, device=h0.device)
    aO, aR = aE.clone(), aE.clone()
    for t, w in zip(ts, ws):
        E = (E0 + float(t) * dE).detach().requires_grad_(True)
        P = (P0 + float(t) * dP).detach().requires_grad_(True)
        with torch.enable_grad():
            m = tail(model, E, P, post, yy, cols)
            gE, gP = torch.autograd.grad(m.sum(), [E, P])
        aE += float(w) * (gE * dE).sum(-1).double()
        aO += float(w) * (gP[:, [0, 2]] * dP[:, [0, 2]]).sum((-1, -2)).double()
        aR += float(w) * (gP[:, 1] * dP[:, 1]).sum(-1).double()
    total = (m1 - m0).double()
    err = (aE + aO + aR - total).abs()
    routes = torch.stack([aE, aO, aR], 1)
    top = routes.argmax(1)  # largest signed route
    pos = routes.clamp(min=0)
    top_pos = torch.where(pos.max(1).values > 0, pos.argmax(1), torch.full_like(top, 3))  # 3 = no positive route
    return {"n_pairs": len(R), "rec_margin": float(m0.mean()), "don_margin": float(m1.mean()),
            "total": float(total.mean()), "E": float(aE.mean()), "ops": float(aO.mean()), "opr": float(aR.mean()),
            "abs_E": float(aE.abs().mean()), "abs_ops": float(aO.abs().mean()), "abs_opr": float(aR.abs().mean()),
            "pair_sum_abs_routes": float(routes.abs().sum(1).mean()), "pair_abs_total": float(total.abs().mean()),
            "completeness_mean": float(err.mean()), "completeness_max": float(err.max()),
            "top_route_frac": {k: float((top == i).float().mean()) for i, k in enumerate(("E", "ops", "opr"))},
            "top_positive_route_frac": {k: float((top_pos == i).float().mean())
                                        for i, k in enumerate(("E", "ops", "opr", "none"))},
            "endpoints": (m0, m1)}


def one(run, device, chunk):
    model, args, update = mp.load(run, "final")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x, y, cols, a, b, n, groups = mp.table(args)
    t0 = time.time()
    xd, yd, cd = x.to(device), y.to(device), cols.to(device)
    H0, base = [], []
    for s in range(0, len(x), chunk):
        H0.append(layer0(model, xd[s:s + chunk]))
        with torch.no_grad():
            out = model(xd[s:s + chunk])
        out = out[0] if isinstance(out, tuple) else out
        out = out[:, -1] if out.dim() == 3 else out
        base.append(mp.two_margins(out.float()[:, cd], yd[s:s + chunk])[1])
    H0, base = torch.cat(H0), torch.cat(base)

    gen = torch.Generator().manual_seed(0)
    dtr, otr = groups["diag_train"], groups["off_train"]
    sets = {"diag": pairs(dtr.nonzero().squeeze(1), otr.clone(), y, a, b, K_DIAG, gen, False)}
    off_rec = otr.nonzero().squeeze(1)
    off_rec = off_rec[torch.randperm(len(off_rec), generator=gen)[:N_OFF]]
    sets["off"] = pairs(off_rec, otr.clone(), y, a, b, K_OFF, gen, True)

    res, checks = {}, {}
    for name, (R, D) in sets.items():
        Rd, Dd = R.to(device), D.to(device)
        res[name] = {}
        for choice, post in (("A_prenorm", False), ("B_postnorm", True)):
            r = attribute(model, H0, Rd, Dd, yd, cd, post)
            m0, m1 = r.pop("endpoints")
            checks[f"endpoint_err_{name}_{choice}"] = float(max((m0 - base[Rd]).abs().max(), (m1 - base[Dd]).abs().max()))
            r64 = attribute(model, H0, Rd, Dd, yd, cd, post, n_nodes=64)
            r64.pop("endpoints")
            r["nodes64"] = {k: r64[k] for k in ("E", "ops", "opr", "completeness_mean", "completeness_max")}
            res[name][choice] = r
    peak_ram, peak_commit = peak_memory_gb()
    return {"run": str(run), "update": update, "operands": args.operands,
            "task": mp.gl.task_of(args) if not mp.gl.multi_task(args) else args.tasks[0],
            "checks": checks, "results": res, "seconds": time.time() - t0,
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
        summ = {s: {c: {k: round(v, 3) for k, v in d.items() if k in ("total", "E", "ops", "opr", "completeness_mean",
                                                                        "completeness_max")}
                    for c, d in rec["results"][s].items()} for s in rec["results"]}
        print(json.dumps({"run": rec["run"], "checks": {k: f"{v:.1e}" for k, v in rec["checks"].items()},
                          "summary": summ, "ram": rec["peak_ram_gb"], "commit": rec["peak_commit_gb"],
                          "gpu": rec["gpu_used_gb"], "s": round(rec["seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
