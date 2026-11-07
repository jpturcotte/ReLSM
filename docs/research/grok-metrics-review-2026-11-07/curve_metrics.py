"""Threshold-free curve metrics on the task x format phase-1 runs (grok_runs/taskfmt_2026-11-07).
Read-only: reads metrics.jsonl, final_errors.json and split_counts.json; writes JSON to argv[1] (or stdout).
Stdlib + numpy. No torch. Runs in a few seconds.

Curve: y(u) = test_acc_unseen_offdiag@task at each eval (every 100 updates, 0..25,000).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("GROK_RUNS", Path.home() / "grok_runs"))  # the study output folder
TF = ROOT / "taskfmt_2026-11-07"
FMT = ROOT / "format_2026-11-05"
H = 25000
SPLITS = json.load(open(TF / "split_counts" / "split_counts.json"))


def runs():
    for arm in ("TA", "D"):
        for s in (0, 1, 2):
            yield f"{arm}_add_s{s}", "add", s, FMT / "runs" / f"{arm}_s{s}"
    for task in ("mul", "div", "sub", "add2b", "sq", "sqab", "cubeab", "gcd"):
        for arm in ("TA", "D"):
            for s in (0, 1, 2):
                yield f"{arm}_{task}_s{s}", task, s, TF / "runs" / f"{arm}_{task}_s{s}"


def load(d, task):
    rows = [json.loads(l) for l in open(d / "metrics.jsonl", encoding="utf-8") if l.strip()]
    g = lambda k: np.array([r.get(f"{k}@{task}") if r.get(f"{k}@{task}") is not None else np.nan for r in rows], float)
    u = np.array([r["update"] for r in rows], float)
    return u, g("test_acc_unseen_offdiag"), g("train_acc"), g("test_loss_unseen"), \
        g("test_margin_mean_unseen_offdiag"), g("test_margin_p5_unseen_offdiag"), rows


def first(u, mask):
    i = np.flatnonzero(mask)
    return float(u[i[0]]) if i.size else None


def sustained(u, y, thr):
    """First eval from which y stays >= thr through the horizon."""
    bad = np.flatnonzero(y < thr)
    if bad.size == 0:
        return float(u[0])
    if bad[-1] == len(u) - 1:
        return None
    return float(u[bad[-1] + 1])


def smooth(y, k=5):
    """Centred running median over k evals (edges shrink)."""
    out = np.empty_like(y)
    h = k // 2
    for i in range(len(y)):
        out[i] = np.median(y[max(0, i - h): i + h + 1])
    return out


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid_fit(u, y, base):
    """y = base + (top - base) / (1 + exp(-(ln u - ln u0) / s)), top in [base, 1].
    Grid over ln u0 and s; top by least squares, clipped. Returns u0, width in decades (10->90 %), top, R^2."""
    m = (u > 0) & np.isfinite(y)
    x, yy = np.log(u[m]), y[m]
    best = None
    sst = float(((yy - yy.mean()) ** 2).sum()) or 1e-12
    for lu0 in np.linspace(np.log(100), np.log(10 * H), 100):
        for s in np.exp(np.linspace(np.log(0.02), np.log(3.0), 40)):
            f = 1 / (1 + np.exp(-(x - lu0) / s))
            den = float((f * f).sum())
            amp = float((f * (yy - base)).sum()) / den if den > 0 else 0.0
            amp = min(max(amp, 0.0), 1 - base)
            sse = float(((yy - base - amp * f) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, lu0, s, amp)
    sse, lu0, s, amp = best
    return {"u0": float(np.exp(lu0)), "width_dec": float(2 * math.log(9) * s / math.log(10)),
            "top": base + amp, "r2": 1 - sse / sst}


def horizon_trend(u, y, lo=20000, hi=H):
    """OLS of logit(y) and of y on u over [lo, hi]. Slopes per 1,000 updates; extrapolated 0.99 crossing.
    se is the plain OLS standard error; neighbouring evals are autocorrelated, so treat slope/se as a rough
    signal (we require slope > 3 se)."""
    m = (u >= lo) & (u <= hi) & np.isfinite(y)
    x, ly, yy = u[m] / 1000, logit(y[m]), y[m]
    b, a = np.polyfit(x, ly, 1)
    by, _ = np.polyfit(x, yy, 1)
    resid_sd = float(np.std(ly - (a + b * x), ddof=2))
    se_b = resid_sd / math.sqrt(float(((x - x.mean()) ** 2).sum()))
    at_h = a + b * hi / 1000
    cross = None
    if b > 0 and at_h < logit(np.array(0.99)):
        cross = hi + 1000 * (float(logit(np.array(0.99))) - at_h) / b
    return {"logit_slope_per_k": float(b), "logit_slope_se": se_b, "acc_slope_per_k": float(by),
            "extrap_cross_0.99": cross}


def beta_trend(u, y, lo, hi=H):
    """beta = slope of logit(y) on ln(u) over [lo, hi], evals thinned to every 500 updates (less autocorrelation).
    Near 1, logit(y) ~ -ln(error), so beta is the power-law decay rate of the error; near 0 it is the growth
    rate of y. One number for "still improving" at any level. x99: where the fitted line reaches logit(0.99)."""
    m = (u >= lo) & (u <= hi) & np.isfinite(y) & (u % 500 == 0)
    x, ly = np.log(u[m]), logit(y[m])
    b, a = np.polyfit(x, ly, 1)
    se = float(np.std(ly - (a + b * x), ddof=2) / math.sqrt(float(((x - x.mean()) ** 2).sum())))
    at = a + b * math.log(hi)
    x99 = None
    if b > 0 and at < logit(np.array(0.99)):
        e = (float(logit(np.array(0.99))) - at) / b
        x99 = float(hi * math.exp(e)) if e < math.log(1e7 / hi) else float("inf")  # inf: beyond 1e7 updates
    return {"beta": float(b), "beta_se": se, "x99": x99}


def err_powerlaw(u, y, lo=10000):
    """error = 1 - y ~ C * u^-alpha over [lo, H]; alpha and the projected update where error = 0.01."""
    m = (u >= lo) & np.isfinite(y) & (y < 1)
    if m.sum() < 5:
        return {"alpha": None, "proj_err_0.01": None}
    lx, le = np.log(u[m]), np.log(1 - y[m])
    b, a = np.polyfit(lx, le, 1)
    alpha = -float(b)
    ex = (math.log(0.01) - a) / b if b < 0 else None
    proj = float(math.exp(ex)) if ex is not None and ex < 50 else None  # None: no projection, or beyond e^50
    return {"alpha": alpha, "proj_err_0.01": proj}


def confusion_metrics(d, task, seed):
    """Balanced accuracy, macro-F1 and Cohen's kappa on the unseen off-diagonal set at update 25,000,
    rebuilt from the exact per-answer counts and the wrong rows. Returns None if counts don't match."""
    sc = SPLITS.get(f"{task}_s{seed}")
    if sc is None:
        return None
    n_true = {int(k): v for k, v in sc["hist"].items()}
    fe = json.load(open(d / "final_errors.json", encoding="utf-8"))
    cols = fe["columns"]
    ia, ib, ians, ipred = (cols.index(c) for c in ("a", "b", "answer", "pred"))
    its = cols.index("transpose_seen")
    wrong = [r for r in fe["test"] if r[ia] != r[ib] and not r[its]]
    n = sum(n_true.values())
    wrong_from, pred_as = {}, {}
    for r in wrong:
        wrong_from[r[ians]] = wrong_from.get(r[ians], 0) + 1
        pred_as[r[ipred]] = pred_as.get(r[ipred], 0) + 1
    classes = set(n_true) | set(pred_as)
    tp = {k: n_true.get(k, 0) - wrong_from.get(k, 0) for k in classes}
    n_pred = {k: tp[k] + pred_as.get(k, 0) for k in classes}
    acc = sum(tp.values()) / n
    recalls = [tp[k] / n_true[k] for k in n_true if n_true[k] > 0]
    bal = sum(recalls) / len(recalls)
    f1s = []
    for k in n_true:
        p = tp[k] / n_pred[k] if n_pred[k] else 0.0
        r = tp[k] / n_true[k]
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    pe = sum(n_true.get(k, 0) * n_pred.get(k, 0) for k in classes) / n / n
    kappa = (acc - pe) / (1 - pe)
    worst = sorted(((tp[k] / n_true[k], k, n_true[k]) for k in n_true if n_true[k] >= 5))[:3]
    return {"acc": acc, "balanced_acc": bal, "macro_f1": sum(f1s) / len(f1s), "kappa": kappa,
            "majority": sc.get("majority_baseline"), "n": n, "n_wrong": len(wrong),
            "worst_classes": [(k, round(rc, 3), nk) for rc, k, nk in worst]}


def analyse(name, task, seed, d, Hh=H, conf=True):
    """All metrics as if the run had stopped at update Hh (late window = last 1,000; trend = last 5,000)."""
    u, y, tr, tl, mm, mp5, rows = load(d, task)
    keep = u <= Hh
    u, y, tr, tl, mm, mp5, rows = u[keep], y[keep], tr[keep], tl[keep], mm[keep], mp5[keep], rows[:int(keep.sum())]
    base = SPLITS.get(f"{task}_s{seed}", {}).get("majority_baseline", 1 / 97)
    late = (u >= Hh - 1000) & (u <= Hh)
    L = float(np.nanmedian(y[late]))
    cc = (L - base) / (1 - base)
    m = u > 0
    auc_lin = float(np.trapezoid(y[m], u[m]) / (u[m][-1] - u[m][0]))
    lu = np.log10(u[m])
    auc_log = float(np.trapezoid(y[m], lu) / (lu[-1] - lu[0]))
    ys = smooth(y)
    tmem = first(u, tr >= 0.99)
    out = {"run": name, "task": task, "seed": seed, "base": base, "final_level": L, "final_cc": cc,
           "auc_lin": auc_lin, "auc_log": auc_log,
           "first_0.99": first(u, y >= 0.99), "sust_0.99": sustained(u, y, 0.99),
           "first_0.95": first(u, y >= 0.95), "sust_0.95": sustained(u, y, 0.95),
           "first_0.5": first(u, y >= 0.5), "tmem": tmem}
    # own-plateau fractions on the smoothed curve (only meaningful when the run moved off chance)
    for f in (0.1, 0.5, 0.9):
        lvl = base + f * (L - base)
        out[f"t{int(f*100)}_own"] = first(u, (ys >= lvl) & (u > 0)) if cc > 0.2 else None
    if out["t10_own"] and out["t90_own"]:
        out["width_dec_own"] = math.log10(out["t90_own"] / out["t10_own"])
    else:
        out["width_dec_own"] = None
    out["delay_ratio"] = (out["t50_own"] / tmem) if (out["t50_own"] and tmem) else None
    out["sig"] = sigmoid_fit(u, y, base)
    out["trend"] = horizon_trend(u, y, Hh - 5000, Hh)
    out["errpow"] = err_powerlaw(u, y, 0.4 * Hh)
    out["beta"] = beta_trend(u, y, 0.4 * Hh, Hh)
    lastk = (u >= Hh - 5000)
    ipk = int(np.nanargmax(tl))
    out["loss_hump"] = float(tl[ipk] / tl[0])  # peak unseen test loss over its value at init (ln 97 ~ 4.6)
    out["loss_hump_at"] = float(u[ipk])
    dy = np.diff(y[lastk])
    out["late_jitter_sd"] = float(np.std(dy))
    out["late_frac_ge_0.99"] = float(np.mean(y[lastk] >= 0.99))
    out["late_crossings_0.99"] = int(np.sum(np.diff((y[lastk] >= 0.99).astype(int)) != 0))
    out["late_range"] = [float(np.min(y[lastk])), float(np.max(y[lastk]))]
    out["test_loss_final"] = float(np.nanmedian(tl[late]))
    out["test_loss_min"] = float(np.nanmin(tl))
    out["test_loss_logslope_late"] = float(np.polyfit(u[lastk] / 1000, np.log(tl[lastk]), 1)[0])
    out["margin_mean_final"] = float(np.nanmedian(mm[late]))
    out["margin_p5_final"] = float(np.nanmedian(mp5[late]))
    out["margin_p5_slope_late"] = float(np.polyfit(u[lastk] / 1000, mp5[lastk], 1)[0])
    # norms at horizon (descriptive)
    out["norm_matrices_final"] = rows[-1].get("norm_matrices")
    out["conf"] = confusion_metrics(d, task, seed) if conf else None
    out["curve_5k"] = {int(k): round(float(y[u == k][0]), 4) for k in (1000, 2500, 5000, 10000, 15000, 20000, 25000)
                       if k <= Hh}
    out["state"], out["shape"] = classify(out)
    return out


def classify(o):
    """Post-hoc taxonomy (designed after seeing these runs; not pre-registered).
    Level and trend come from the last 1,000 / 5,000 updates; shape needs the run to have left chance."""
    L, cc, t = o["final_level"], o["final_cc"], o["trend"]
    b = o["beta"]
    rising = b["beta"] > max(3 * b["beta_se"], 0.3)  # logit gains > 0.3 per e-fold of updates, 3 se clear
    if L >= 0.99 and o["late_frac_ge_0.99"] >= 0.9:
        state = "converged"
    elif rising:
        state = "rising"
    elif L >= 0.97:
        state = "near-line plateau"
    elif cc < 0.05:
        state = "flat at chance"
    else:
        state = "stalled partial"
    shape = None
    if cc > 0.2 and o["t50_own"]:
        timing = "delayed" if (o["delay_ratio"] or 0) >= 2 else "concurrent"
        width = "sharp" if (o["width_dec_own"] or 9) < 0.35 else "gradual"
        shape = f"{timing}, {width}"
    return state, shape


if __name__ == "__main__":
    res = [analyse(*r) for r in runs()]
    for hh in (10000, 15000, 20000):  # truncation test: what each rule would have said at an earlier horizon
        for r, full in zip(runs(), res):
            o = analyse(*r, Hh=hh, conf=False)
            full[f"at_{hh}"] = {"state": o["state"], "shape": o["shape"], "first_0.99": o["first_0.99"],
                                "t50_own": o["t50_own"], "sig_u0": o["sig"]["u0"], "auc_log": o["auc_log"],
                                "extrap": o["trend"]["extrap_cross_0.99"], "proj": o["errpow"]["proj_err_0.01"],
                                "slope": o["trend"]["logit_slope_per_k"], "beta": o["beta"]["beta"],
                                "x99": o["beta"]["x99"], "L": o["final_level"]}
    txt = json.dumps(res, indent=1)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(txt, encoding="utf-8")
    else:
        print(txt)
