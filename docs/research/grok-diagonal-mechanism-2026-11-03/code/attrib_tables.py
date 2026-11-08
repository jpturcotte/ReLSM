"""Tables and registered verdicts for H10 (attrib_test.py output). Usage: python -X utf8 attrib_tables.py IN.jsonl"""
import json
import sys
from collections import Counter

recs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
CH = ("A_prenorm", "B_postnorm")
RT = ("E", "ops", "opr")


def name(r):
    p = r["run"].replace("\\", "/").split("/")
    return ("diag_" if "diag_arms" in r["run"] else "e1_") + p[-1]


def label(d):
    """Registered per-run classification from the diagonal results under both choices."""
    A, B = d["A_prenorm"], d["B_postnorm"]
    sh = {c: {k: d[c][k] / d[c]["total"] for k in RT} for c in CH}
    if any(abs(A[k] - B[k]) > 0.5 for k in RT):
        return "coupled (A vs B > 0.5)", sh
    if any(sum(abs(d[c][k]) for k in RT) > 2 * abs(d[c]["total"]) for c in CH):
        return "coupled (cancellation)", sh
    if all(sh[c]["E"] >= 0.8 and sh[c]["ops"] <= 0.2 for c in CH):
        return "'=' account", sh
    if all(sh[c]["ops"] >= 0.8 and sh[c]["E"] <= 0.2 for c in CH):
        return "operand-read account", sh
    if all(sh[c]["opr"] >= 0.25 for c in CH):
        return "third route (op)", sh
    return "unresolved (none)", sh


print("# Route-resolved attribution (H10) results\n")
print("Routes: E = '=' state after layer 0 (residual, query, self K/V); ops = layer-1 inputs at operand positions;")
print("opr = layer-1 input at the op position. Values in nats, mean over (recipient, donor) pairs; total = donor − recipient margin.\n")

print("## T0. Checks\n")
print("| run | endpoint err (max over sets/choices) | completeness mean / max, diag A | diag B | off A | off B | diag pairs | off pairs |")
print("|---|---|---|---|---|---|---|---|")
for r in recs:
    R = r["results"]
    ep = max(r["checks"].values())
    cm = lambda s, c: f"{R[s][c]['completeness_mean']:.3f} / {R[s][c]['completeness_max']:.3f}"
    print(f"| {name(r)} | {ep:.1e} | {cm('diag', 'A_prenorm')} | {cm('diag', 'B_postnorm')} | {cm('off', 'A_prenorm')} | "
          f"{cm('off', 'B_postnorm')} | {R['diag']['A_prenorm']['n_pairs']} | {R['off']['A_prenorm']['n_pairs']} |")

for s, title in (("diag", "Diagonal recipients"), ("off", "Off-diagonal controls")):
    print(f"\n## {title}: route attribution (nats; share of total in brackets for diagonals)\n")
    print("| run | choice | recipient | donor | total | E | ops | opr | top route (E / ops / opr) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        for c in CH:
            d = r["results"][s][c]
            sh = (lambda k: f" ({d[k] / d['total']:+.0%})") if (s == "diag" and r["operands"] != "separate") else (lambda k: "")
            tf = d.get("top_positive_route_frac", d["top_route_frac"])
            print(f"| {name(r)} | {c[0]} | {d['rec_margin']:.2f} | {d['don_margin']:.2f} | {d['total']:+.2f} | "
                  f"{d['E']:+.2f}{sh('E')} | {d['ops']:+.2f}{sh('ops')} | {d['opr']:+.2f}{sh('opr')} | "
                  f"{tf['E']:.2f} / {tf['ops']:.2f} / {tf['opr']:.2f}" + (f" / none {tf['none']:.2f}" if "none" in tf else "") + " |")

if "pair_sum_abs_routes" in recs[0]["results"]["off"]["A_prenorm"]:
    print("\n## Per-pair cancellation: mean over pairs of Σ |route| vs mean |total| (diag / off)\n")
    print("| run | A diag | A off | B diag | B off |")
    print("|---|---|---|---|---|")
    for r in recs:
        R = r["results"]
        f = lambda s, c: f"{R[s][c]['pair_sum_abs_routes']:.2f} / {R[s][c]['pair_abs_total']:.2f}"
        print(f"| {name(r)} | {f('diag', 'A_prenorm')} | {f('off', 'A_prenorm')} | {f('diag', 'B_postnorm')} | {f('off', 'B_postnorm')} |")
    print("\n## Quadrature sensitivity: route means at 32 vs 64 nodes (diagonals), and 64-node completeness\n")
    print("| run | choice | E 32 / 64 | ops 32 / 64 | opr 32 / 64 | completeness 64 mean / max |")
    print("|---|---|---|---|---|---|")
    for r in recs:
        for c in CH:
            d = r["results"]["diag"][c]
            g = d["nodes64"]
            print(f"| {name(r)} | {c[0]} | {d['E']:+.3f} / {g['E']:+.3f} | {d['ops']:+.3f} / {g['ops']:+.3f} | "
                  f"{d['opr']:+.3f} / {g['opr']:+.3f} | {g['completeness_mean']:.4f} / {g['completeness_max']:.3f} |")
else:
    print("\n## Control-path damage: Σ |route means| on off-diagonal controls vs |total|\n")
    print("| run | A | B |")
    print("|---|---|---|")
    for r in recs:
        o = r["results"]["off"]
        print(f"| {name(r)} | " + " | ".join(f"{sum(abs(o[c][k]) for k in RT):.2f} / {abs(o[c]['total']):.2f}" for c in CH) + " |")

print("\n## Verdicts against the registration (9 shared-token runs)\n")
labels = {}
for r in recs:
    if r["operands"] == "separate":
        continue
    lab, sh = label(r["results"]["diag"])
    labels[name(r)] = lab
    shs = "; ".join(f"{c[0]}: E {sh[c]['E']:+.0%}, ops {sh[c]['ops']:+.0%}, opr {sh[c]['opr']:+.0%}" for c in CH)
    print(f"- {name(r)}: **{lab}** ({shs})")
cnt = Counter(labels.values())
win = [k for k, v in cnt.items() if v >= 7 and k in ("'=' account", "operand-read account", "third route (op)")]
print(f"\nCounts: {dict(cnt)}")
print(f"Overall: {win[0] if win else 'mixed'}")
mx = max(r["results"][s][c]["completeness_mean"] for r in recs for s in ("diag", "off") for c in CH)
print(f"Completeness check (run mean ≤ 0.05 nats): worst run mean {mx:.4f} → {'pass' if mx <= 0.05 else 'FAIL'}")
