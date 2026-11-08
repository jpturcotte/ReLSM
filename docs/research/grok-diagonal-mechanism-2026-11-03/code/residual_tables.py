"""Tables and registered verdicts for the residual tests (H7). Usage: python residual_tables.py > residual_results.md"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, "..", "data", "residual.jsonl")) if l.startswith("{")]
f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
name = lambda r: ("e1_" if "e1_" in r["run"] else "diag_") + os.path.basename(r["run"])  # noqa: E731
C = ("emb", "attn0", "mlp0", "attn1", "mlp1")
shared = [r for r in recs if r["operands"] == "shared"]
sep = [r for r in recs if r["operands"] != "shared"]

print("# Residual tests (H7) results\n")
print("Final residual at '=' (pre-norm, fp32), train groups. Checks: component sum error, and the change in centered")
print("pre-norm answer logits from stripping the invisible part (must be ~0).\n")
print("## T1. Component RMS ratio (diagonal / off-diagonal) and share of the diagonal's excess squared norm\n")
print("| run | " + " | ".join(f"{c} ratio" for c in C) + " | total ratio | " + " | ".join(f"{c} share" for c in C) + " | checks |")
print("|---|" + "---|" * (2 * len(C) + 2))
for r in recs:
    cp = r["components"]
    t = r["total"]
    print(f"| {name(r)} | " + " | ".join(f(cp[c]['rms_ratio'], 3) for c in C) + f" | {f(t['rms_diag'] / t['rms_off'], 3)} | " +
          " | ".join(f(cp[c]['excess_share']) for c in C) +
          f" | {r['checks']['component_sum_err']:.0e} / {r['checks']['visible_logit_change']:.0e} |")

print("\n## T2. Visible vs invisible energy, and stripping\n")
print(f"Readout rank {recs[0]['readout_rank']} of d = {recs[0]['d']}. scale gap = |1 − r_diag / r_off| (r = final inverse RMS).\n")
print("| run | invisible share diag / off | scale gap base | scale gap stripped | gap_train base | gap_train stripped | closed | offset +1 response (free norm): plain / stripped |")
print("|---|---|---|---|---|---|---|---|")
for r in recs:
    iv, s, m, o = r["invisible_share"], r["inv_rms"], r["margins"], r["offset_response_free_norm"]
    g0 = abs(1 - s["base_diag"] / s["base_off"])
    g1 = abs(1 - s["stripped_diag"] / s["stripped_off"])
    b, st = m["base"]["gap_train"], m["stripped"]["gap_train"]
    closed = f"{100 * (b - st) / b:.0f}%" if abs(b) >= 0.5 else "—"
    print(f"| {name(r)} | {f(iv['diag'], 3)} / {f(iv['off'], 3)} | {f(g0, 3)} | {f(g1, 3)} | {f(b)} | {f(st)} | {closed} | {f(o['plain'])} / {f(o['stripped'])} |")

print("\n## Verdicts against the registration\n")
r1 = [r["components"]["attn0"]["rms_ratio"] for r in shared]
s1 = [r["components"]["attn0"]["rms_ratio"] for r in sep]
print(f"- H7a (i) attn0 RMS ratio ≥ 1.25 in all 9 shared: {', '.join(f(x, 3) for x in r1)} → {all(x >= 1.25 for x in r1)}; "
      f"separate-token in [0.9, 1.1]: {', '.join(f(x, 3) for x in s1)} → {all(0.9 <= x <= 1.1 for x in s1)}")
r2 = [r["components"]["attn0"]["excess_share"] + r["components"]["mlp0"]["excess_share"] for r in shared]
print(f"- H7a (ii) attn0 + mlp0 share ≥ 0.5 in ≥ 7/9: {', '.join(f(x) for x in r2)} → {sum(x >= 0.5 for x in r2)}/9")
b1 = [(r["invisible_share"]["diag"], r["invisible_share"]["off"]) for r in shared]
print(f"- H7b (i) invisible share diag > off in ≥ 7/9: {sum(d > o for d, o in b1)}/9")
b2 = []
for r in shared:
    s = r["inv_rms"]
    b2.append(abs(1 - s["stripped_diag"] / s["stripped_off"]) <= 0.5 * abs(1 - s["base_diag"] / s["base_off"]))
print(f"- H7b (ii) scale gap at most half after stripping in ≥ 7/9: {sum(b2)}/9")
q = [r for r in shared if r["offset_response_free_norm"]["plain"] < -0.1]
b3 = [r["offset_response_free_norm"]["stripped"] >= 0.5 * r["offset_response_free_norm"]["plain"] for r in q]
need = min(7, len(q))
print(f"- H7b (iii) offset response at least 50% less negative after stripping: {sum(b3)}/{len(q)} qualifying runs (need {need})")
