"""Residual tests (H7 in hypotheses.md), final weights, at '='.

H7a: each component's write (emb, attn0, mlp0, attn1, mlp1) into the final residual h: RMS on diagonal vs
     off-diagonal train pairs, and each component's share of the diagonal's excess squared norm
     (||h||^2 = sum_i c_i . h).
H7b: split h into the part visible to the centered answer readout A = W_U[cols] * gain (row space of A minus its
     class mean) and the invisible rest; strip the invisible part, recompute the final RMSNorm and score. Also the
     free-norm response to a layer-1 operand-score offset (+1, all heads, diagonal train pairs) with and without
     stripping.

Usage: python residual_test.py --device cuda --out OUT.jsonl RUN_DIR [RUN_DIR ...]
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

COMPS = ("emb", "attn0", "mlp0", "attn1", "mlp1")


@torch.no_grad()
def forward(model, x, offset=None, P=None):
    """Manual forward. offset: (layer, delta, select [B]) adds delta to both operand scores at '=' for all heads.
    P: [d, d] projector onto the visible subspace; if given, the final residual is projected before the norm.
    Returns logits at '=' [B, V], components at '=' {name: [B, d]} (fp32), and the final residual used [B, d]."""
    h = model.tok_emb(x)
    comps = {"emb": h[:, -1].float()}
    B, T, _ = h.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    for li, blk in enumerate(model.layers):
        at = blk.attn
        u = blk.attn_norm(h)
        q = at.q_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        k = at.k_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        v = at.v_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        cos, sin = at.rotary(T, x.device)
        q, k = apply_rotary_emb(q, k, cos, sin, pos)
        s = (q @ k.transpose(-1, -2)) * at.scale
        if offset is not None and offset[0] == li:
            _, delta, sel = offset
            s = s.clone()
            idx = sel.nonzero().squeeze(1)
            for p in (0, 2):
                s[idx, :, T - 1, p] += delta
        pr = s.masked_fill(~causal, float("-inf")).softmax(-1)
        ao = at.o_proj((pr @ v).transpose(1, 2).reshape(B, T, -1))
        h = h + ao
        comps[f"attn{li}"] = ao[:, -1].float()
        mo = blk.ff(blk.ff_norm(h))
        h = h + mo
        comps[f"mlp{li}"] = mo[:, -1].float()
    he = h[:, -1].float()
    if P is not None:
        he = he @ P
    r = he.pow(2).mean(-1).add(model.norm.eps).rsqrt()
    logits = model.lm_head((he * r.unsqueeze(-1)).type_as(h) * model.norm.weight)
    return logits, comps, he


def run_all(model, x, chunk, offset=None, P=None):
    L, C, H = [], {k: [] for k in COMPS}, []
    for i in range(0, len(x), chunk):
        s = slice(i, min(i + chunk, len(x)))
        off = None if offset is None else (offset[0], offset[1], offset[2][s])
        lg, comps, he = forward(model, x[s], off, P)
        L.append(lg)
        H.append(he)
        for k in COMPS:
            C[k].append(comps[k])
    return torch.cat(L), {k: torch.cat(v) for k, v in C.items()}, torch.cat(H)


def margins(logits, y, cols, groups):
    _, lse = mp.two_margins(logits[:, cols], y)
    out = {g: float(lse[groups[g]].mean()) for g in ("diag_train", "diag_test", "off_train", "off_test")}
    out["gap_train"] = out["off_train"] - out["diag_train"]
    out["gap_test"] = out["off_test"] - out["diag_test"]
    return out


def rms(v):
    return v.pow(2).mean(-1).sqrt()


def one(run, device, chunk):
    model, args, update = mp.load(run, "final")
    model.to(device)
    x, y, cols, a, b, n, groups = mp.table(args)
    x, y, cols = x.to(device), y.to(device), cols.to(device)
    groups = {k: v.to(device) for k, v in groups.items()}
    dtr, otr = groups["diag_train"], groups["off_train"]
    t0 = time.time()

    # visible subspace of the centered answer readout
    A = model.lm_head.weight[cols].float() * model.norm.weight.float()  # [C, d]
    Ac = A - A.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Ac, full_matrices=False)
    rank = int((S > S[0] * 1e-6).sum())
    V = Vh[:rank]
    P = V.T @ V  # [d, d]

    lg, comps, he = run_all(model, x, chunk)
    base = margins(lg, y, cols, groups)
    err_sum = float((sum(comps.values()) - he).abs().max())

    # H7a: component RMS and shares of the diagonal's excess squared norm
    hsq = he.pow(2).sum(-1)
    comp = {}
    for k in COMPS:
        c = comps[k]
        dot = (c * he).sum(-1)
        comp[k] = {"rms_diag": float(rms(c)[dtr].mean()), "rms_off": float(rms(c)[otr].mean()),
                   "dot_diag": float(dot[dtr].mean()), "dot_off": float(dot[otr].mean())}
        comp[k]["rms_ratio"] = comp[k]["rms_diag"] / comp[k]["rms_off"]
    excess = float(hsq[dtr].mean() - hsq[otr].mean())
    for k in COMPS:
        comp[k]["excess_share"] = (comp[k]["dot_diag"] - comp[k]["dot_off"]) / excess if excess else None
    # cosine between layer-0 attention output on (a, a) and the mean over pairs, as a coherence readout
    total = {"rms_diag": float(rms(he)[dtr].mean()), "rms_off": float(rms(he)[otr].mean()),
             "sq_diag": float(hsq[dtr].mean()), "sq_off": float(hsq[otr].mean()), "excess_sq": excess}

    # H7b: invisible energy, stripping
    hv = he @ P
    hi = he - hv
    inv_share = hi.pow(2).sum(-1) / hsq
    r0 = he.pow(2).mean(-1).add(model.norm.eps).rsqrt()
    r1 = hv.pow(2).mean(-1).add(model.norm.eps).rsqrt()
    lg_s, _, _ = run_all(model, x, chunk, P=P)
    stripped = margins(lg_s, y, cols, groups)
    err_vis = float(((hv - he) @ Ac.T).abs().max())  # centered pre-norm logits must not change

    # offset response, free norm, with and without stripping
    resp = {}
    for mode, PP, ref in (("plain", None, base), ("stripped", P, stripped)):
        lg_o, _, _ = run_all(model, x, chunk, offset=(1, 1.0, dtr), P=PP)
        m = margins(lg_o, y, cols, groups)
        resp[mode] = m["diag_train"] - ref["diag_train"]

    peak_ram, peak_commit = peak_memory_gb()
    return {"run": str(run), "update": update, "operands": args.operands,
            "task": mp.gl.task_of(args) if not mp.gl.multi_task(args) else args.tasks[0],
            "readout_rank": rank, "d": int(P.shape[0]),
            "checks": {"component_sum_err": err_sum, "visible_logit_change": err_vis},
            "components": comp, "total": total,
            "invisible_share": {"diag": float(inv_share[dtr].mean()), "off": float(inv_share[otr].mean())},
            "inv_rms": {"base_diag": float(r0[dtr].mean()), "base_off": float(r0[otr].mean()),
                        "stripped_diag": float(r1[dtr].mean()), "stripped_off": float(r1[otr].mean())},
            "margins": {"base": base, "stripped": stripped},
            "offset_response_free_norm": resp,
            "seconds": time.time() - t0, "peak_ram_gb": peak_ram, "peak_commit_gb": peak_commit}


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
        c = rec["components"]
        print(json.dumps({"run": rec["run"], "checks": rec["checks"], "rank": rec["readout_rank"],
                          "attn0_ratio": round(c["attn0"]["rms_ratio"], 3),
                          "shares": {k: round(c[k]["excess_share"], 2) for k in COMPS if c[k]["excess_share"] is not None},
                          "inv_share": rec["invisible_share"], "inv_rms": rec["inv_rms"],
                          "gap": [round(rec["margins"]["base"]["gap_train"], 3), round(rec["margins"]["stripped"]["gap_train"], 3)],
                          "offset": rec["offset_response_free_norm"], "ram": rec["peak_ram_gb"]}), flush=True)


if __name__ == "__main__":
    main()
