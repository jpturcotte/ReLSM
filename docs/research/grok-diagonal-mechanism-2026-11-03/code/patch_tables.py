"""Tables and registered verdicts for H9 (patch_test.py output). Usage: python -X utf8 patch_tables.py IN.jsonl"""
import json
import sys

recs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
P = ("P_op0", "P_size", "P_eq", "P_ops", "P_both")


def name(r):
    p = r["run"].replace("\\", "/").split("/")
    return ("diag_" if "diag_arms" in r["run"] else "e1_") + p[-1]


shared = [r for r in recs if r["operands"] != "separate"]
sep = [r for r in recs if r["operands"] == "separate"]

print("# Same-answer patching (H9) results\n")
print("Recipients: diagonal train pairs; donors: off-diagonal train pairs with the same answer. Final norm free.")
print("closure = mean(patched − recipient) / mean(donor − recipient).\n")

print("## T0. Checks\n")
print("| run | model fwd err | P_both = donor (diag / off) | self no-op P_op0 / P_both (diag) | diag pairs (recipients) | off pairs |")
print("|---|---|---|---|---|---|")
for r in recs:
    c, d, o = r["checks"], r["results"]["diag"], r["results"]["off"]
    print(f"| {name(r)} | {c['model_forward_err']:.1e} | {c.get('both_equals_donor_diag', float('nan')):.1e} / "
          f"{c.get('both_equals_donor_off', float('nan')):.1e} | {c.get('self_noop_P_op0_diag', float('nan')):.1e} / "
          f"{c.get('self_noop_P_both_diag', float('nan')):.1e} | {d['n_pairs']} ({d['n_recipients']}) | {o['n_pairs']} |")

print("\n## T1. Diagonal recipients: closure (Δ margin in nats)\n")
print("| run | recipient | donor | operand RMS donor/recipient | " + " | ".join(P) + " |")
print("|---|---|---|---|" + "---|" * len(P))
rows = {}
for r in recs:
    d = r["results"]["diag"]
    cells = []
    for k in P:
        cl = d[k]["closure"]
        cells.append(f"{cl:+.0%} ({d[k]['delta']:+.2f})" if cl is not None and r["operands"] != "separate"
                     else f"— ({d[k]['delta']:+.2f})")
    rows[name(r)] = {k: d[k]["closure"] for k in P}
    print(f"| {name(r)} | {d['rec_margin']:.2f} | {d['don_margin']:.2f} | {d['op_rms_ratio_don_over_rec']:.3f} | "
          + " | ".join(cells) + " |")

print("\n## T2. Off-diagonal reference: mean Δ margin (recipient → same-answer off-diagonal donor)\n")
print("| run | recipient | donor | " + " | ".join(P) + " |")
print("|---|---|---|" + "---|" * len(P))
for r in recs:
    o = r["results"]["off"]
    print(f"| {name(r)} | {o['rec_margin']:.2f} | {o['don_margin']:.2f} | "
          + " | ".join(f"{o[k]['delta']:+.2f}" for k in P) + " |")

print("\n## T2b. Not registered (added after the smoke test): excess effect on diagonals over the off-diagonal reference\n")
print("(Δ diag − Δ off) / (donor − recipient gap on diagonals). Removes the mismatch cost that any partial patch between")
print("two different pairs pays. Positive = the patch helps diagonals more (or hurts them less) than off-diagonals.\n")
print("| run | " + " | ".join(P) + " |")
print("|---|" + "---|" * len(P))
for r in shared:
    d, o = r["results"]["diag"], r["results"]["off"]
    G = d["don_margin"] - d["rec_margin"]
    print(f"| {name(r)} | " + " | ".join(f"{(d[k]['delta'] - o[k]['delta']) / G:+.0%}" for k in P) + " |")

print("\n## T3. Accuracy of patched diagonals\n")
print("| run | " + " | ".join(P) + " |")
print("|---|" + "---|" * len(P))
for r in recs:
    d = r["results"]["diag"]
    print(f"| {name(r)} | " + " | ".join(f"{d[k]['acc']:.3f}" for k in P) + " |")

sh = [name(r) for r in shared]
c1 = sum(rows[n]["P_op0"] >= 0.40 for n in sh)
c2 = sum(rows[n]["P_op0"] - rows[n]["P_size"] >= 0.25 for n in sh)
c3 = sum(rows[n]["P_eq"] - rows[n]["P_ops"] >= 0.25 for n in sh)
alt = sum(rows[n]["P_ops"] >= 0.40 and rows[n]["P_op0"] <= 0.20 for n in sh)
print("\n## Verdicts against the registration (9 shared-token runs)\n")
def pcts(f):  # one signed percent per shared run; a helper because Python 3.11 can't nest f-string quotes
    return ", ".join(f"{f(rows[n]):+.0%}" for n in sh)


print(f"- C1 P_op0 closure ≥ 40 % in ≥ 7/9: {pcts(lambda r: r['P_op0'])} → {c1}/9")
print(f"- C2 P_op0 − P_size ≥ 25 points in ≥ 7/9: {pcts(lambda r: r['P_op0'] - r['P_size'])} → {c2}/9")
print(f"- C3 P_eq − P_ops ≥ 25 points in ≥ 7/9: {pcts(lambda r: r['P_eq'] - r['P_ops'])} → {c3}/9")
print(f"- Alternative (P_ops ≥ 40 % and P_op0 ≤ 20 %) in ≥ 7/9: {alt}/9")
print(f"- P_eq closures: {pcts(lambda r: r['P_eq'])}; P_ops closures: {pcts(lambda r: r['P_ops'])}")
