"""Plain tables of the v2 restore tests (numbers only). Usage: python restore2_tables.py > restore2_tables.md"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, "..", "data", "restore_v2.jsonl")) if l.startswith("{")]
f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
name = lambda r: ("e1_" if "e1_" in r["run"] else "diag_") + os.path.basename(r["run"])  # noqa: E731


def off(r, target, head, delta, norm, key="d_margin"):
    return next(e[key] for e in r["offsets"] if e["target"] == target and e["head"] == head
                and e["delta"] == delta and e["norm"] == norm)


print("# Restore tests v2: tables (numbers only)\n")
print("Final weights. Margins are lse, train groups. `Δ` = change in the mean margin of the group the change was applied to.")
print("Runs: e1_add/e1_mul (paired split, shared tokens), diag_control (add, ordered split, shared tokens),")
print("diag_sepops (add, ordered split, separate operand tokens; its base gap is about 0).\n")

print("## T1. Checks\n")
print("| run | max \\|logit − model(x)\\| | no-op mass | no-op norm | max row-sum error | peak RAM GB | seconds |")
print("|---|---|---|---|---|---|---|")
for r in recs:
    c = r["checks"]
    print(f"| {name(r)} | {c['max_abs_logit_err_vs_model']:.1e} | {c['noop_max_abs']['mass']:.1e} | {c['noop_max_abs']['norm']:.1e} | "
          f"{c['max_row_sum_err']:.1e} | {f(r['peak_ram_gb'])} | {f(r['seconds'], 1)} |")

print("\n## T2. Layer-1 operand-score offset, all heads at once\n")
print("δ is added to the attention scores of both operand positions at query '=' in layer 1, on diagonal train examples")
print("(target diag) or on off-diagonal train examples (target off). norm free: the final RMSNorm recomputes; norm fixed:")
print("each example's final inverse RMS is held at its own baseline value. Δ mass = mean change of operand mass at δ = +1.\n")
print("| run | base diag_train | base off_train | target | norm | Δ at δ=−1 | δ=−0.5 | δ=+0.5 | δ=+1 | Δ mass at +1 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    b = r["results"]["base"]
    for t in ("diag", "off"):
        for nm in ("free", "fixed"):
            vals = [f(off(r, t, "all", d, nm)) for d in (-1.0, -0.5, 0.5, 1.0)]
            print(f"| {name(r)} | {f(b['diag_train'])} | {f(b['off_train'])} | {t} | {nm} | " + " | ".join(vals) +
                  f" | {f(off(r, t, 'all', 1.0, nm, 'd_mass'), 3)} |")

print("\n## T3. Offset δ = +1, one layer-1 head at a time, norm fixed\n")
print("| run | target | head 0 | head 1 | head 2 | head 3 | all heads |")
print("|---|---|---|---|---|---|---|")
for r in recs:
    for t in ("diag", "off"):
        print(f"| {name(r)} | {t} | " + " | ".join(f(off(r, t, h, 1.0, 'fixed')) for h in (0, 1, 2, 3, "all")) + " |")

print("\n## T4. Group-mean swaps (the first version's tests, rerun with the fixed code)\n")
print("mass up diag: diagonal examples get each layer-1 head's off-diagonal train mean operand mass (a/b ratio kept).")
print("mass down off: off-diagonal examples get the diagonal train mean. norm up diag: diagonal examples get the off-diagonal")
print("train mean final inverse RMS. norm down off: the reverse. norm flatten diag: diagonal examples get the diagonal mean.\n")
print("| run | base gap_train | mass up diag | mass down off | norm up diag | norm down off | both up diag | norm flatten diag: Δ diag_train | inv RMS diag / off | L1 mass diag / off (mean over heads) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    R, b = r["results"], r["results"]["base"]
    md = sum(r["massL1_diag"]) / 4
    mo = sum(r["massL1_off"]) / 4
    print(f"| {name(r)} | {f(b['gap_train'])} | " + " | ".join(f(R[k]['gap_train']) for k in
          ("massL1_up_diag", "massL1_down_off", "norm_up_diag", "norm_down_off", "both_up_diag")) +
          f" | {f(R['norm_flatten_diag']['diag_train'] - b['diag_train'])} | {f(r['inv_rms_diag'], 3)} / {f(r['inv_rms_off'], 3)} | {f(md, 3)} / {f(mo, 3)} |")

print("\n## T5. Accuracy under norm-only swaps (train groups)\n")
print("| run | base acc diag / off | norm up diag | norm down off | norm flatten diag |")
print("|---|---|---|---|---|")
for r in recs:
    a, c = r["checks"]["acc_base"], r["checks"]["acc_norm_only"]
    fmt = lambda d: f"{f(d['diag_train'], 3)} / {f(d['off_train'], 3)}"  # noqa: E731
    print(f"| {name(r)} | {fmt(a)} | {fmt(c['norm_up_diag'])} | {fmt(c['norm_down_off'])} | {fmt(c['norm_flatten_diag'])} |")
