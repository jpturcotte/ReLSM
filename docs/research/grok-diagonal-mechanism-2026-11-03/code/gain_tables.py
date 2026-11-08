"""Tables and registered verdicts for H8 (gain_test.py output). Usage: python gain_tables.py IN.jsonl > OUT.md"""
import json
import sys

recs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
GK = ["0.500", "0.707", "0.850", "1.000", "1.150", "1.414"]
MODES = ["replay_fixed", "final_fixed", "replay_free", "final_free"]


def name(r):
    return r["run"].replace("\\", "/").rstrip("/").split("/")[-2].split("_")[0] + "_" + r["run"].replace("\\", "/").split("/")[-1] \
        if "diag_arms" in r["run"] else "e1_" + r["run"].replace("\\", "/").split("/")[-1]


def d(r, g, m, grp):
    return r["sweep"][g][m][grp] - r["base"][grp]


shared = [r for r in recs if r["operands"] != "separate"]
sep = [r for r in recs if r["operands"] == "separate"]

print("# Operand-gain sweep (H8) results\n")
print("Δ = change in mean lse margin vs g = 1, train groups. closure = Δ diag / baseline gap_train.\n")

print("## T0. Checks and baseline\n")
print("| run | model fwd err | max attn0 write err | g=1 errs | gap_train | L0 operand mass diag / off | operand write RMS diag / off (ratio) |")
print("|---|---|---|---|---|---|---|")
for r in recs:
    c = r["checks"]
    aw = max(v for k, v in c.items() if k.startswith("attn0_write_err"))
    o = r["op_part_rms"]
    print(f"| {name(r)} | {c['model_forward_err']:.1e} | {aw:.1e} | {c['replay_g1_err']:.0e} / {c['final_g1_err']:.0e} | "
          f"{r['base']['gap_train']:.2f} | {r['operand_mass_L0']['diag']:.3f} / {r['operand_mass_L0']['off']:.3f} | "
          f"{o['diag']:.3f} / {o['off']:.3f} ({o['diag'] / o['off']:.3f}) |")

for m in MODES:
    print(f"\n## Dose curve, {m}: Δ diag / Δ off\n")
    print("| run | " + " | ".join(f"g={g}" for g in GK if g != "1.000") + " |")
    print("|---|" + "---|" * (len(GK) - 1))
    for r in recs:
        cells = [f"{d(r, g, m, 'diag_train'):+.2f} / {d(r, g, m, 'off_train'):+.2f}" for g in GK if g != "1.000"]
        print(f"| {name(r)} | " + " | ".join(cells) + " |")

print("\n## g = 0.707 summary (norm fixed)\n")
print("| run | replay Δ diag | replay Δ off | final-only Δ diag | final-only Δ off | closure (replay) | replay − final (diag) | Δ diag − Δ off (replay) |")
print("|---|---|---|---|---|---|---|---|")
rows = {}
for r in recs:
    g = "0.707"
    rd, ro = d(r, g, "replay_fixed", "diag_train"), d(r, g, "replay_fixed", "off_train")
    fd, fo = d(r, g, "final_fixed", "diag_train"), d(r, g, "final_fixed", "off_train")
    gap = r["base"]["gap_train"]
    rows[name(r)] = (rd, ro, fd, fo, rd / gap, rd - fd, rd - ro)
    print(f"| {name(r)} | {rd:+.2f} | {ro:+.2f} | {fd:+.2f} | {fo:+.2f} | {rd / gap:+.0%} | {rd - fd:+.2f} | {rd - ro:+.2f} |")

print("\n## Accuracy (train) at g = 0.707, replay norm fixed vs base: diag / off\n")
print("| run | base | replay fixed | replay free |")
print("|---|---|---|---|")
for r in recs:
    b, s = r["base"], r["sweep"]["0.707"]
    print(f"| {name(r)} | {b['acc_diag_train']:.3f} / {b['acc_off_train']:.3f} | "
          f"{s['replay_fixed']['acc_diag_train']:.3f} / {s['replay_fixed']['acc_off_train']:.3f} | "
          f"{s['replay_free']['acc_diag_train']:.3f} / {s['replay_free']['acc_off_train']:.3f} |")

print("\n## Downstream RMS: diagonal at g = 0.707 vs off-diagonal at g = 1 (diag at g = 1 in brackets)\n")
print("| run | attn0 | mlp0 | attn1 | mlp1 | final inv RMS |")
print("|---|---|---|---|---|---|")
for r in recs:
    s, b = r["sweep"]["0.707"], r["sweep"]["1.000"]
    cells = [f"{s['comp_rms'][k]['diag']:.3f} vs {b['comp_rms'][k]['off']:.3f} [{b['comp_rms'][k]['diag']:.3f}]"
             for k in ("attn0", "mlp0", "attn1", "mlp1")]
    cells.append(f"{s['inv_rms']['diag']:.3f} vs {b['inv_rms']['off']:.3f} [{b['inv_rms']['diag']:.3f}]")
    print(f"| {name(r)} | " + " | ".join(cells) + " |")

sh = [name(r) for r in shared]
sp = [name(r) for r in sep]
d1 = sum(rows[n][4] >= 0.25 for n in sh)
alt = sum(rows[n][4] <= 0.10 for n in sh)
d2 = sum(rows[n][5] >= 0.5 for n in sh)
d3 = sum(rows[n][6] >= 0.5 for n in sh)
spok = sum(abs(rows[n][6]) <= 0.3 for n in sp)
print("\n## Verdicts against the registration (g = 0.707, replay, norm fixed)\n")
print(f"- D1 closure ≥ 25 % in ≥ 7/9: {', '.join(f'{rows[n][4]:+.0%}' for n in sh)} → {d1}/9")
print(f"- Alternative closure ≤ 10 % in ≥ 7/9: {alt}/9")
print(f"- D2 replay − final-only Δ diag ≥ 0.5 in ≥ 7/9: {', '.join(f'{rows[n][5]:+.2f}' for n in sh)} → {d2}/9")
print(f"- D3 Δ diag − Δ off ≥ 0.5 in ≥ 7/9: {', '.join(f'{rows[n][6]:+.2f}' for n in sh)} → {d3}/9")
print(f"- Separate-token |Δ diag − Δ off| ≤ 0.3 in 3/3: {', '.join(f'{rows[n][6]:+.2f}' for n in sp)} → {spok}/{len(sp)}")
