"""Restore tests (H6 in hypotheses.md): does the late diagonal gap come from layer-1 operand mass at '=' or
from the final-norm scale? One line of JSON per run to --out.

Interventions, all at the '=' position, final weights:
  massL1_up_diag    diagonal examples: each layer-1 head's operand mass set to that head's off-diagonal train
                    mean (operand ratio kept; op and '=' weights rescaled to fill the rest)
  massL1_down_off   control: off-diagonal examples set to that head's diagonal train mean mass
  norm_up_diag      diagonal examples: final-norm inverse RMS set to the off-diagonal train mean
  norm_down_off     control: off-diagonal examples set to the diagonal train mean
  both_up_diag      massL1_up_diag together with norm_up_diag (the norm target is the off-diagonal mean
                    from the unmodified pass)

Usage: python restore_test.py --device cuda --out OUT.jsonl RUN_DIR [RUN_DIR ...]
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

GROUPS = ("diag_train", "diag_test", "off_train", "off_test")


def peak_memory_gb():
    """This process's peak working set (RAM) and peak commit, in GB, from GetProcessMemoryInfo (Windows)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError(ctypes.get_last_error())
        return c.PeakWorkingSetSize / 2**30, c.PeakPagefileUsage / 2**30
    except Exception as e:  # not Windows, or the call failed
        print(f"peak memory readout failed: {e!r}", file=sys.stderr)
        return None, None


@torch.no_grad()
def forward(model, x, mass=None, inv=None):
    """The probe's manual forward with two hooks. mass: (layer, target [H] or [B, H], select [B]) sets each
    head's operand mass at '=' on selected examples. inv: (target scalar or [B], select [B]) replaces the
    final-norm inverse RMS at '=' on selected examples. Returns logits at '=' [B, V], the layer-1 operand
    mass at '=' [B, H], and the final inverse RMS at '=' [B]."""
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
        pr = ((q @ k.transpose(-1, -2)) * at.scale).masked_fill(~causal, float("-inf")).softmax(-1)
        if li == 1:
            m1 = pr[:, :, T - 1, 0] + pr[:, :, T - 1, 2]
        if mass is not None and mass[0] == li:
            pr = pr.clone()
            tgt, sel = mass[1], mass[2]
            row = pr[sel, :, T - 1, :]  # [b, H, T]
            m = row[..., 0] + row[..., 2]
            M = tgt if tgt.dim() == 1 else tgt[sel]
            M = M.expand_as(m)
            op = torch.zeros(T, dtype=torch.bool, device=x.device)
            op[[0, 2]] = True
            scale = torch.where(op, (M / m).unsqueeze(-1), ((1 - M) / (1 - m)).unsqueeze(-1))
            pr[sel, :, T - 1, :] = row * scale
        o = pr @ v
        ao = at.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + ao
        h = h + blk.ff(blk.ff_norm(h))
    he = h[:, -1].float()
    r = he.pow(2).mean(-1).add(model.norm.eps).rsqrt()
    if inv is not None:
        tgt, sel = inv
        r = r.clone()
        r[sel] = tgt
    logits = model.lm_head((he * r.unsqueeze(-1)).type_as(h) * model.norm.weight)
    return logits, m1, r


def run_all(model, x, chunk, **kw):
    out = [forward(model, x[s], **{k: _slice(v, s) for k, v in kw.items()}) for s in mp.chunks(len(x))]
    return tuple(torch.cat([o[i] for o in out]) for i in range(3))


def _slice(arg, s):
    """Cut the per-example parts of a hook argument to this chunk."""
    if arg is None:
        return None
    if len(arg) == 3:  # mass: (layer, target, select)
        li, tgt, sel = arg
        return (li, tgt if tgt.dim() == 1 else tgt[s], sel[s])
    tgt, sel = arg  # inv
    return (tgt, sel[s])


def margins(logits, y, cols, groups):
    _, lse = mp.two_margins(logits[:, cols], y)
    out = {g: float(lse[groups[g]].mean()) for g in GROUPS}
    out["gap_train"] = out["off_train"] - out["diag_train"]
    out["gap_test"] = out["off_test"] - out["diag_test"]
    return out


def one(run, device):
    model, args, update = mp.load(run, "final")
    model.to(device)
    x, y, cols, a, b, n, groups = mp.table(args)
    x, y, cols = x.to(device), y.to(device), cols.to(device)
    groups = {k: v.to(device) for k, v in groups.items()}
    diag = groups["diag_train"] | groups["diag_test"]
    off = groups["off_train"] | groups["off_test"]
    t0 = time.time()
    lg, m1, r = run_all(model, x, mp.CHUNK)
    base = margins(lg, y, cols, groups)
    m_off = m1[groups["off_train"]].mean(0)  # [H]
    m_diag = m1[groups["diag_train"]].mean(0)
    r_off = r[groups["off_train"]].mean()
    r_diag = r[groups["diag_train"]].mean()
    res = {"base": base}
    tests = {
        "massL1_up_diag": dict(mass=(1, m_off, diag)),
        "massL1_down_off": dict(mass=(1, m_diag, off)),
        "norm_up_diag": dict(inv=(r_off, diag)),
        "norm_down_off": dict(inv=(r_diag, off)),
        "both_up_diag": dict(mass=(1, m_off, diag), inv=(r_off, diag)),
    }
    for name, kw in tests.items():
        lg2, _, _ = run_all(model, x, mp.CHUNK, **kw)
        res[name] = margins(lg2, y, cols, groups)
    peak_vram = torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None  # torch's pool only
    if device == "cuda":  # the whole process on the GPU, CUDA context included
        free, total = torch.cuda.mem_get_info()
        res_vram_used_gb = (total - free) / 2**30
    else:
        res_vram_used_gb = None
    peak_ram, peak_commit = peak_memory_gb()
    return {"peak_ram_gb": peak_ram, "peak_commit_gb": peak_commit, "gpu_used_gb": res_vram_used_gb,"run": str(run), "update": update, "task": mp.gl.task_of(args) if not mp.gl.multi_task(args) else args.tasks[0],
            "pairs": args.pairs, "operands": args.operands,
            "massL1_off": m_off.tolist(), "massL1_diag": m_diag.tolist(),
            "inv_rms_off": float(r_off), "inv_rms_diag": float(r_diag),
            "results": res, "seconds": time.time() - t0, "peak_vram_gb": peak_vram}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("runs", nargs="+")
    o = ap.parse_args()
    for r in o.runs:
        rec = one(Path(r), o.device)
        with open(o.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        R = rec["results"]
        print(json.dumps({"run": rec["run"], "gap": R["base"]["gap_train"],
                          **{k: round(R[k]["gap_train"], 3) for k in R if k != "base"},
                          "peak_vram_gb": rec["peak_vram_gb"], "peak_ram_gb": rec["peak_ram_gb"],
                          "peak_commit_gb": rec["peak_commit_gb"], "gpu_used_gb": rec["gpu_used_gb"]}), flush=True)


if __name__ == "__main__":
    main()
