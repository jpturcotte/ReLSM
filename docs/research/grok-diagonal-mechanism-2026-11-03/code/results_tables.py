"""Turn the v2 probe records into plain tables (no interpretation).

Usage: python results_tables.py > results_tables.md
Reads v2/*.jsonl (probe records) and the E1 metrics.jsonl logs.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(os.environ.get("GROK_RUNS", "grok_runs"), "e1_2026-11-02")
ORDER = ["e1_add", "e1_mul", "e1_sub", "e1_div", "e1_s5",
         "diag_control", "diag_sepops", "add2b_add2b_shared", "add2b_add2b_sepops"]


def records():
    out = {}
    for p in glob.glob(os.path.join(HERE, "v2", "*.jsonl")):
        tag = os.path.basename(p)[:-6]
        rs = {}
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rs[r["ckpt"]] = r
        out[tag] = rs
    key = lambda t: (next(i for i, o in enumerate(ORDER) if t.startswith(o + "_s")), t)
    return dict(sorted(out.items(), key=lambda kv: key(kv[0])))


def f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def shrink(base, new):
    if base is None or new is None or abs(base) < 0.5:
        return "—"
    return f"{100 * (base - new) / base:.0f}%"


def row(cells):
    print("| " + " | ".join(str(c) for c in cells) + " |")


def head(cells):
    row(cells)
    print("|" + "---|" * len(cells))


def ckpt_order(rs):
    return sorted(rs, key=lambda c: rs[c]["update"])


def main():
    R = records()
    fin = {t: rs["final"] for t, rs in R.items()}

    print("# Probe results, v2 (numbers only)\n")
    print("All margins are recomputed in fp32 on CPU. `gap` = off-diagonal mean margin minus diagonal mean margin.")
    print("Primary margin `lse` = z_y − logsumexp(wrong logits), as grok_lab logs it; `top` = z_y − max wrong logit.")
    print("Groups: diag_train, diag_test (diagonal pairs (a, a)), off_train, off_test (held-out off-diagonal pairs).")
    print("`shrink` = (base gap − gap after the change) / base gap, shown only when |base| ≥ 0.5.")
    print("Final checkpoint unless stated. E1 = paired split, several-task code path, shared operand tokens.")
    print("diag_control = add, shared tokens, ordered split; diag_sepops = add, separate operand tokens.")
    print("add2b = a + 2b; shared vs separate operand tokens.\n")

    print("## T1. Gaps and accuracy (final)\n")
    head(["run", "update", "n diag train/test", "logged lse gap_train", "probe lse gap_train", "probe top gap_train",
          "lse gap_test", "acc diag_train", "acc diag_test", "acc off_test", "counts ok"])
    for t, r in fin.items():
        s, n = r["score"], r["n"]
        row([t, r["update"], f"{n['diag_train']}/{n['diag_test']}", f(r["logged"].get("gap_train") if r.get("logged") else None),
             f(s["lse"]["gap_train"]), f(s["top"]["gap_train"]), f(s["lse"]["gap_test"]),
             f(s["acc"]["diag_train"], 3), f(s["acc"]["diag_test"], 3), f(s["acc"]["off_test"], 4),
             r["count_check"]["ok"]])

    print("\n## T2. Operand classes as competitors (H4), final, train groups\n")
    head(["run", "lse gap_train", "gap, operand classes removed", "shrink", "gap, same number of random classes removed",
          "shrink", "runner-up is operand: diag_train", "off_train", "operand logit − mean wrong logit: diag_train", "off_train"])
    for t, r in fin.items():
        s = r["score"]
        b = s["lse"]["gap_train"]
        row([t, f(b), f(s["lse_no_operands"]["gap_train"]), shrink(b, s["lse_no_operands"]["gap_train"]),
             f(s["lse_no_random"]["gap_train"]), shrink(b, s["lse_no_random"]["gap_train"]),
             f(s["runner_is_operand"]["diag_train"], 3), f(s["runner_is_operand"]["off_train"], 3),
             f(s["operand_a_logit_above_mean_wrong"]["diag_train"]), f(s["operand_a_logit_above_mean_wrong"]["off_train"])])

    print("\n## T3. Interventions, final: lse gap_train after each (shrink in brackets)\n")
    print("ratio_Lk: at query '=' in layer k, each head's weights on the two operand positions set to m/2 each (m kept).")
    print("freeze_Lk: layer k's attention pattern replaced by its mean pattern over all inputs.")
    print("head_mean: one head's output at '=' replaced by its mean; the column shows the head with the largest shrink.\n")
    head(["run", "base", "ratio_L0", "ratio_L1", "freeze_L0", "freeze_L1", "freeze_L0+L1", "largest head_mean"])
    for t, r in fin.items():
        b = r["score"]["lse"]["gap_train"]
        iv = r["interventions"]
        cell = lambda k: f"{f(iv[k]['lse']['gap_train'])} ({shrink(b, iv[k]['lse']['gap_train'])})"
        hm = [(k, iv[k]["lse"]["gap_train"]) for k in iv if k.startswith("head_mean")]
        hk, hv = min(hm, key=lambda kv: kv[1]) if abs(b) >= 0.5 else (hm[0][0], hm[0][1])
        row([t, f(b), cell("ratio_L0"), cell("ratio_L1"), cell("freeze_L0"), cell("freeze_L1"), cell("freeze_L0+L1"),
             f"{hk.replace('head_mean.', '')}: {f(hv)} ({shrink(b, hv)})"])

    print("\n## T3b. Where the ratio interventions move the margin (lse, change vs base), final\n")
    head(["run", "ratio_L0 Δ diag_train", "ratio_L0 Δ off_train", "ratio_L0 max \\|Δ logit\\| on diagonals",
          "ratio_L1 Δ diag_train", "ratio_L1 Δ off_train", "ratio_L1 Δ off_test", "ratio_L1 Δ diag_test"])
    for t, r in fin.items():
        s, iv = r["score"]["lse"], r["interventions"]
        d = lambda k, g: iv[k]["lse"][g] - s[g]
        row([t, f(d("ratio_L0", "diag_train")), f(d("ratio_L0", "off_train")),
             f(iv["ratio_L0"].get("diag_max_abs_logit_change"), 5),
             f(d("ratio_L1", "diag_train")), f(d("ratio_L1", "off_train")), f(d("ratio_L1", "off_test")),
             f(d("ratio_L1", "diag_test"))])

    print("\n## T4. Exact margin split at '=' (top margin: correct vs runner-up direction), gap_train per component, final\n")
    print("Each component's write into the residual at '=', final-norm scale held fixed, projected on W_U[y] − W_U[runner-up].")
    print("Components sum to the top-margin gap (error column = largest absolute reconstruction error).\n")
    comps = ["emb", "attn0", "mlp0", "attn1", "mlp1"]
    head(["run", "top gap_train"] + comps + ["sum", "max err"])
    for t, r in fin.items():
        sp = r["split"]
        vals = [sp[c]["gap_train"] for c in comps]
        row([t, f(r["score"]["top"]["gap_train"])] + [f(v) for v in vals] + [f(sum(vals)), f"{r['split_max_abs_err']:.1e}"])

    print("\n## T4b. Per-head split, gap_train, final\n")
    hs = [f"attn{L}.h{h}" for L in (0, 1) for h in range(4)]
    head(["run"] + hs)
    for t, r in fin.items():
        row([t] + [f(r["split"][k]["gap_train"]) for k in hs])

    print("\n## T5. Attention at query '=' (final). Per layer: mean over heads, and the largest per-head value\n")
    print("ab_diff_std_off = std over off-diagonal pairs of w(a) − w(b). mass = w(a) + w(b). ratio = w(a) / mass.\n")
    head(["run", "L0 ab_diff_std mean / max", "L0 mass off / diag", "L0 ratio mean", "L0 ratio std mean / max",
          "L1 ab_diff_std mean / max", "L1 mass off / diag", "L1 ratio mean", "L1 ratio std mean / max"])
    for t, r in fin.items():
        cells = [t]
        for L in (0, 1):
            a = [r["attn"][f"L{L}.h{h}.qeq"] for h in range(4)]
            m = lambda k: sum(x[k] for x in a) / 4
            cells += [f"{f(m('ab_diff_std_off'), 3)} / {f(max(x['ab_diff_std_off'] for x in a), 3)}",
                      f"{f(m('mass_mean_off'), 2)} / {f(m('mass_mean_diag'), 2)}",
                      f(m("ratio_mean_off"), 3),
                      f"{f(m('ratio_std_off'), 3)} / {f(max(x['ratio_std_off'] for x in a), 3)}"]
        row(cells)

    print("\n## T6. E1 over training (probe checkpoints)\n")
    head(["run", "update", "acc off_train", "acc diag_train", "acc off_test", "lse gap_train", "lse gap_test",
          "ratio_L0 shrink", "ratio_L1 shrink", "freeze_L0+L1 shrink", "runner-up is operand (diag_train)",
          "split gap_train attn0 / mlp0 / attn1 / mlp1"])
    for t, rs in R.items():
        if not t.startswith("e1_"):
            continue
        for c in ckpt_order(rs):
            r = rs[c]
            s, iv, sp = r["score"], r["interventions"], r["split"]
            b = s["lse"]["gap_train"]
            row([t, r["update"], f(s["acc"]["off_train"], 3), f(s["acc"]["diag_train"], 3), f(s["acc"]["off_test"], 3),
                 f(b), f(s["lse"]["gap_test"]),
                 shrink(b, iv["ratio_L0"]["lse"]["gap_train"]), shrink(b, iv["ratio_L1"]["lse"]["gap_train"]),
                 shrink(b, iv["freeze_L0+L1"]["lse"]["gap_train"]), f(s["runner_is_operand"]["diag_train"], 3),
                 " / ".join(f(sp[k]["gap_train"]) for k in ("attn0", "mlp0", "attn1", "mlp1"))])

    print("\n## T7. E1 timing from the training logs (every eval, not just checkpoints)\n")
    print("Memorization at 500: train acc ≥ 0.9 there. Generalization start: first eval with off-diagonal test acc ≥ 0.5.")
    print("Gap = train_margin_mean_offdiag − train_margin_mean_diagonal (lse, logged in bf16 on the training device).\n")
    head(["run", "train acc @500", "train gap @500", "first update off-diag test acc ≥ 0.5", "train gap there",
          "first update train gap ≥ 1", "first update test acc ≥ 0.99", "train gap there", "max train gap (update)",
          "train gap, mean of last 10 %"])
    for d in sorted(glob.glob(os.path.join(E1, "*_s*"))):
        if not os.path.exists(os.path.join(d, "metrics.jsonl")):
            continue
        name = os.path.basename(d)
        t = name.split("_")[0]
        g = lambda r, k: r.get(f"{k}@{t}", r.get(k))
        rs = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
        rs = [r for r in rs if g(r, "test_acc") is not None and g(r, "train_margin_mean_offdiag") is not None]
        gap = lambda r: g(r, "train_margin_mean_offdiag") - g(r, "train_margin_mean_diagonal")
        at = lambda u: min(rs, key=lambda r: abs(r["update"] - u))
        gen = next((r for r in rs if g(r, "test_acc_unseen_offdiag") >= 0.5), None)
        g1 = next((r for r in rs if gap(r) >= 1), None)
        grok = next((r for r in rs if g(r, "test_acc") >= 0.99), None)
        mx = max(rs, key=gap)
        tail = rs[int(len(rs) * 0.9):]
        r500 = at(500)
        row([f"e1_{name}", f(g(r500, "train_acc"), 3), f(gap(r500)), gen["update"] if gen else "—",
             f(gap(gen)) if gen else "—", g1["update"] if g1 else "—", grok["update"] if grok else "—",
             f(gap(grok)) if grok else "—", f"{f(gap(mx))} ({mx['update']})", f(sum(map(gap, tail)) / len(tail))])


if __name__ == "__main__":
    main()
