"""Restore tests v2 (H6c in hypotheses.md), after the Codex review of restore_test.py.

Fixes: chunking uses its argument; per-example norm targets are sliced per chunk and selected per example.
Checks (recorded per run): attention rows sum to 1 after any hook; self-target no-ops (mass and norm) give the
base logits; the unmodified manual forward matches model(x) example by example; norm-only swaps leave accuracy
unchanged.

Tests, all at query '=', final weights; lse margins on train groups:
  offset   layer-1 operand-score offset delta in {-1, -0.5, +0.5, +1} added to both operand positions (a/b ratio
           kept), for all heads and for each head alone, applied to diagonal train examples or to off-diagonal
           train examples, with the final-norm inverse RMS free or fixed at each example's own baseline value
  mass     the first version's tests (group-mean mass up on diagonals / down on off-diagonals), kept for reference
  norm     diagonals set to the off-diagonal mean inverse RMS; flattening control: diagonals set to their own mean

Usage: python restore_test2.py --device cuda --out OUT.jsonl RUN_DIR [RUN_DIR ...]
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

GROUPS = ("diag_train", "diag_test", "off_train", "off_test")
DELTAS = (-1.0, -0.5, 0.5, 1.0)
CHECK = {"max_row_sum_err": 0.0}


@torch.no_grad()
def forward(model, x, mass=None, inv=None, offset=None):
    """Manual forward with hooks; returns logits at '=' [B, V], layer-1 operand mass at '=' [B, H], and the
    final inverse RMS at '=' [B].
    mass:   (layer, target [H] or [B, H], select [B]) sets each head's operand mass on selected examples
    inv:    (target scalar or [B], select [B]) replaces the final-norm inverse RMS on selected examples
    offset: (layer, head or None for all, delta, select [B]) adds delta to the operand scores before softmax"""
    h = model.tok_emb(x)
    B, T, _ = h.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    m1 = None
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
            _, hd, delta, sel = offset
            s = s.clone()
            heads = slice(None) if hd is None else slice(hd, hd + 1)
            idx = sel.nonzero().squeeze(1)
            for p in (0, 2):
                s[idx, heads, T - 1, p] += delta
        pr = s.masked_fill(~causal, float("-inf")).softmax(-1)
        if mass is not None and mass[0] == li:
            pr = pr.clone()
            tgt, sel = mass[1], mass[2]
            row = pr[sel, :, T - 1, :]
            m = row[..., 0] + row[..., 2]
            assert bool(((m > 0) & (m < 1)).all()), "operand mass at a boundary"
            M = (tgt if tgt.dim() == 1 else tgt[sel]).expand_as(m)
            op = torch.zeros(T, dtype=torch.bool, device=x.device)
            op[[0, 2]] = True
            scale = torch.where(op, (M / m).unsqueeze(-1), ((1 - M) / (1 - m)).unsqueeze(-1))
            pr[sel, :, T - 1, :] = row * scale
        CHECK["max_row_sum_err"] = max(CHECK["max_row_sum_err"], float((pr[:, :, T - 1].sum(-1) - 1).abs().max()))
        if li == 1:
            m1 = pr[:, :, T - 1, 0] + pr[:, :, T - 1, 2]
        o = pr @ v
        h = h + at.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + blk.ff(blk.ff_norm(h))
    he = h[:, -1].float()
    r = he.pow(2).mean(-1).add(model.norm.eps).rsqrt()
    if inv is not None:
        tgt, sel = inv
        r = r.clone()
        r[sel] = tgt[sel] if tgt.dim() == 1 else tgt
    logits = model.lm_head((he * r.unsqueeze(-1)).type_as(h) * model.norm.weight)
    return logits, m1, r


def _slice(arg, s, kind):
    if arg is None:
        return None
    if kind == "mass":
        li, tgt, sel = arg
        return (li, tgt if tgt.dim() == 1 else tgt[s], sel[s])
    if kind == "inv":
        tgt, sel = arg
        return (tgt[s] if tgt.dim() == 1 else tgt, sel[s])
    li, hd, delta, sel = arg  # offset
    return (li, hd, delta, sel[s])


def run_all(model, x, chunk, **kw):
    out = []
    for i in range(0, len(x), chunk):
        s = slice(i, min(i + chunk, len(x)))
        out.append(forward(model, x[s], **{k: _slice(v, s, k) for k, v in kw.items()}))
    return tuple(torch.cat([o[j] for o in out]) for j in range(3))


def stats(logits, y, cols, groups):
    z = logits[:, cols]
    _, lse = mp.two_margins(z, y)
    acc = (z.argmax(1) == y).float()
    out = {g: float(lse[groups[g]].mean()) for g in GROUPS}
    out["gap_train"] = out["off_train"] - out["diag_train"]
    out["gap_test"] = out["off_test"] - out["diag_test"]
    out["acc"] = {g: float(acc[groups[g]].mean()) for g in GROUPS}
    return out


def one(run, device, chunk):
    model, args, update = mp.load(run, "final")
    model.to(device)
    x, y, cols, a, b, n, groups = mp.table(args)
    x, y, cols = x.to(device), y.to(device), cols.to(device)
    groups = {k: v.to(device) for k, v in groups.items()}
    dtr, otr = groups["diag_train"], groups["off_train"]
    diag = groups["diag_train"] | groups["diag_test"]
    off = groups["off_train"] | groups["off_test"]
    everyone = torch.ones(len(x), dtype=torch.bool, device=device)
    t0 = time.time()
    CHECK["max_row_sum_err"] = 0.0

    lg, m1, r = run_all(model, x, chunk)
    base = stats(lg, y, cols, groups)
    # check 1: the manual forward matches the model, example by example
    with torch.no_grad():
        ref = torch.cat([model(x[i:i + chunk])[0][:, -1] for i in range(0, len(x), chunk)])
    err_model = float((ref - lg).abs().max())
    # check 2: self-target no-ops
    lg_m, _, _ = run_all(model, x, chunk, mass=(1, m1, everyone))
    lg_r, _, _ = run_all(model, x, chunk, inv=(r, everyone))
    noop = {"mass": float((lg_m - lg).abs().max()), "norm": float((lg_r - lg).abs().max())}

    m_off, m_diag = m1[otr].mean(0), m1[dtr].mean(0)
    r_off, r_diag = r[otr].mean(), r[dtr].mean()
    rec = {"base": base}

    # first-version tests, now with the fixed code
    for name, kw in {"massL1_up_diag": dict(mass=(1, m_off, diag)),
                     "massL1_down_off": dict(mass=(1, m_diag, off)),
                     "norm_up_diag": dict(inv=(r_off, diag)),
                     "norm_down_off": dict(inv=(r_diag, off)),
                     "norm_flatten_diag": dict(inv=(r_diag, diag)),
                     "both_up_diag": dict(mass=(1, m_off, diag), inv=(r_off, diag))}.items():
        lg2, _, _ = run_all(model, x, chunk, **kw)
        rec[name] = stats(lg2, y, cols, groups)

    # operand-score offsets
    offs = []
    for target, sel in (("diag", dtr), ("off", otr)):
        for hd in (None, 0, 1, 2, 3):
            for delta in DELTAS:
                for norm in ("free", "fixed"):
                    kw = dict(offset=(1, hd, delta, sel))
                    if norm == "fixed":
                        kw["inv"] = (r, everyone)
                    lg2, m1b, _ = run_all(model, x, chunk, **kw)
                    st = stats(lg2, y, cols, groups)
                    g = "diag_train" if target == "diag" else "off_train"
                    offs.append({"target": target, "head": "all" if hd is None else hd, "delta": delta, "norm": norm,
                                 "d_margin": st[g] - base[g],
                                 "d_mass": float((m1b[sel] - m1[sel]).mean(0).mean()) if hd is None
                                 else float((m1b[sel, hd] - m1[sel, hd]).mean())})
    rec_acc_norm = {k: rec[k]["acc"] for k in ("norm_up_diag", "norm_down_off", "norm_flatten_diag")}
    peak_ram, peak_commit = peak_memory_gb()
    gpu = None
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        gpu = (total - free) / 2**30
    return {"run": str(run), "update": update,
            "task": mp.gl.task_of(args) if not mp.gl.multi_task(args) else args.tasks[0],
            "pairs": args.pairs, "operands": args.operands,
            "checks": {"max_abs_logit_err_vs_model": err_model, "noop_max_abs": noop,
                       "max_row_sum_err": CHECK["max_row_sum_err"],
                       "acc_base": base["acc"], "acc_norm_only": rec_acc_norm},
            "massL1_off": m_off.tolist(), "massL1_diag": m_diag.tolist(),
            "inv_rms_off": float(r_off), "inv_rms_diag": float(r_diag),
            "results": rec, "offsets": offs, "seconds": time.time() - t0,
            "peak_ram_gb": peak_ram, "peak_commit_gb": peak_commit, "gpu_used_gb": gpu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=mp.CHUNK)
    ap.add_argument("runs", nargs="+")
    o = ap.parse_args()
    done = set()
    if Path(o.out).exists():  # resumable: skip runs already recorded
        done = {json.loads(l)["run"] for l in open(o.out) if l.startswith("{")}
    for r in o.runs:
        if str(Path(r)) in done:
            continue
        rec = one(Path(r), o.device, o.chunk)
        with open(o.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        c = rec["checks"]
        allh = {(e["target"], e["norm"]): round(e["d_margin"], 3) for e in rec["offsets"] if e["head"] == "all" and e["delta"] == 1.0}
        print(json.dumps({"run": rec["run"], "err_vs_model": c["max_abs_logit_err_vs_model"], "noop": c["noop_max_abs"],
                          "row_err": c["max_row_sum_err"], "offset+1_all": {f"{k[0]}/{k[1]}": v for k, v in allh.items()},
                          "peak_ram_gb": rec["peak_ram_gb"], "s": round(rec["seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
