"""Operand-gain sweep (H8 in hypotheses.md), final weights, at '='.

Scale the layer-0 attention weights on the operand positions (a at 0, b at 2), all heads, at the '=' query by a
gain g (no renormalization). o_proj has no bias, so this scales exactly the operand part of attn0's write at '='.
Modes: replay (recompute downstream) and final-only (baseline final residual + (g-1) * operand part of attn0),
each with the final norm free (recomputed) or fixed (each example's baseline inverse RMS).

Usage: python gain_test.py --device cuda --out OUT.jsonl RUN_DIR [RUN_DIR ...]
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
GAINS = (0.5, 0.7071067811865476, 0.85, 1.0, 1.15, 1.4142135623730951)
OPS = (0, 2)  # operand positions in [a, op, b, =]


@torch.no_grad()
def forward(model, x, gain=1.0):
    """Manual forward with the layer-0 operand gain at '='. Returns the final residual at '=' [B, d] (fp32),
    the components at '=' {name: [B, d]}, the operand part of attn0 at '=' [B, d], and layer-0 operand mass."""
    h = model.tok_emb(x)
    comps = {"emb": h[:, -1].float()}
    B, T, _ = h.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    op_part, mass = None, None
    for li, blk in enumerate(model.layers):
        at = blk.attn
        u = blk.attn_norm(h)
        q = at.q_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        k = at.k_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        v = at.v_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        cos, sin = at.rotary(T, x.device)
        q, k = apply_rotary_emb(q, k, cos, sin, pos)
        s = (q @ k.transpose(-1, -2)) * at.scale
        pr = s.masked_fill(~causal, float("-inf")).softmax(-1)
        if li == 0:
            mass = pr[:, :, T - 1, list(OPS)].sum(-1).mean(-1).float()  # [B], before the gain
            if gain != 1.0:
                pr = pr.clone()
                pr[:, :, T - 1, list(OPS)] *= gain
            po = torch.zeros_like(pr)
            po[:, :, T - 1, list(OPS)] = pr[:, :, T - 1, list(OPS)]
            op_part = at.o_proj((po @ v).transpose(1, 2).reshape(B, T, -1))[:, -1].float()
        ao = at.o_proj((pr @ v).transpose(1, 2).reshape(B, T, -1))
        h = h + ao
        comps[f"attn{li}"] = ao[:, -1].float()
        mo = blk.ff(blk.ff_norm(h))
        h = h + mo
        comps[f"mlp{li}"] = mo[:, -1].float()
    return h[:, -1].float(), comps, op_part, mass


def inv_rms(model, he):
    return he.pow(2).mean(-1).add(model.norm.eps).rsqrt()


@torch.no_grad()
def readout(model, he, r):
    w = model.norm.weight
    return model.lm_head((he * r.unsqueeze(-1)).to(w.dtype) * w).float()


def run_all(model, x, chunk, gain):
    H, O, M, C = [], [], [], {k: [] for k in COMPS}
    for i in range(0, len(x), chunk):
        he, comps, op, mass = forward(model, x[i:i + chunk], gain)
        H.append(he)
        O.append(op)
        M.append(mass)
        for k in COMPS:
            C[k].append(comps[k])
    return torch.cat(H), {k: torch.cat(v) for k, v in C.items()}, torch.cat(O), torch.cat(M)


def score(logits, y, cols, groups):
    z = logits[:, cols]
    _, lse = mp.two_margins(z, y)
    acc = (z.argmax(1) == y)
    out = {}
    for g in ("diag_train", "diag_test", "off_train", "off_test"):
        out[g] = float(lse[groups[g]].mean())
        out["acc_" + g] = float(acc[groups[g]].float().mean())
    out["gap_train"] = out["off_train"] - out["diag_train"]
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

    he0, c0, op0, mass = run_all(model, x, chunk, 1.0)
    r0 = inv_rms(model, he0)
    base = score(readout(model, he0, r0), y, cols, groups)

    # check against the model's own forward
    err_model = 0.0
    for i in range(0, len(x), chunk):
        with torch.no_grad():
            out = model(x[i:i + chunk])
        out = out[0] if isinstance(out, tuple) else out
        out = out[:, -1] if out.dim() == 3 else out
        ref = readout(model, he0[i:i + chunk], r0[i:i + chunk])
        err_model = max(err_model, float((out.float() - ref).abs().max()))

    sweep, checks = {}, {"model_forward_err": err_model}
    for g in GAINS:
        he, c, _, _ = run_all(model, x, chunk, g)
        hf = he0 + (g - 1.0) * op0
        res = {"replay_free": score(readout(model, he, inv_rms(model, he)), y, cols, groups),
               "replay_fixed": score(readout(model, he, r0), y, cols, groups),
               "final_free": score(readout(model, hf, inv_rms(model, hf)), y, cols, groups),
               "final_fixed": score(readout(model, hf, r0), y, cols, groups)}
        res["comp_rms"] = {k: {"diag": float(rms(c[k])[dtr].mean()), "off": float(rms(c[k])[otr].mean())}
                           for k in COMPS}
        res["inv_rms"] = {"diag": float(inv_rms(model, he)[dtr].mean()), "off": float(inv_rms(model, he)[otr].mean())}
        key = f"{g:.3f}"
        sweep[key] = res
        checks[f"attn0_write_err_{key}"] = float((c["attn0"] - (c0["attn0"] + (g - 1.0) * op0)).abs().max())
        if g == 1.0:
            checks["replay_g1_err"] = float((he - he0).abs().max())
            checks["final_g1_err"] = float((hf - he0).abs().max())

    peak_ram, peak_commit = peak_memory_gb()
    return {"run": str(run), "update": update, "operands": args.operands,
            "task": mp.gl.task_of(args) if not mp.gl.multi_task(args) else args.tasks[0],
            "checks": checks, "base": base,
            "operand_mass_L0": {"diag": float(mass[dtr].mean()), "off": float(mass[otr].mean())},
            "op_part_rms": {"diag": float(rms(op0)[dtr].mean()), "off": float(rms(op0)[otr].mean())},
            "sweep": sweep, "seconds": time.time() - t0,
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
        s = rec["sweep"]["0.707"]
        gap = rec["base"]["gap_train"]
        print(json.dumps({"run": rec["run"], "checks": {k: f"{v:.1e}" for k, v in rec["checks"].items()},
                          "gap": round(gap, 3), "mass": rec["operand_mass_L0"], "op_rms": rec["op_part_rms"],
                          "g0.707": {m: [round(s[m]["diag_train"] - rec["base"]["diag_train"], 3),
                                         round(s[m]["off_train"] - rec["base"]["off_train"], 3)]
                                     for m in ("replay_fixed", "final_fixed", "replay_free", "final_free")},
                          "ram": rec["peak_ram_gb"], "commit": rec["peak_commit_gb"], "gpu": rec["gpu_used_gb"],
                          "s": round(rec["seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
