"""Score the mirror test (H5) against its registered predictions. Usage: python mirror_score.py > mirror_results.md"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["control", "sepops", "add2b_shared", "add2b_sepops", "e1_add"]
recs = {}
for p in glob.glob(os.path.join(HERE, "..", "data", "mirror", "*.jsonl")):
    lines = [json.loads(l) for l in open(p) if l.startswith("{")]
    if lines:
        recs[os.path.basename(p)[:-6]] = lines[-1]
recs = dict(sorted(recs.items(), key=lambda kv: (next(i for i, o in enumerate(ORDER) if kv[0].startswith(o)), kv[0])))


def f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def g(r, k):
    s = r["groups"][k]
    return f"{f(s['mean'])} ± {f(s['se'])} (n {s['n']})"


print("# Mirror test (H5) results\n")
print("lse margins of train pairs at the final checkpoint. mirror = off-diagonal train pair whose (b, a) is also in train;")
print("lone = off-diagonal train pair whose (b, a) is in test; diag = diagonal train pairs. ± is the standard error.")
print("D_mirror = M_mirror − M_diag, D_lone = M_lone − M_diag, R = D_lone / D_mirror.\n")
print("| run | diag | mirror | lone | D_mirror | D_lone | R | mirror − lone | acc lone | held-out, mirror in train | held-out, mirror held out |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k, r in recs.items():
    print(f"| {k} | {g(r, 'diag')} | {g(r, 'mirror')} | {g(r, 'lone')} | {f(r['D_mirror'])} | {f(r['D_lone'])} | "
          f"{f(r['R'])} | {f(r['mirror_minus_lone'])} | {f(r['acc_lone'], 3)} | {g(r, 'test_seen')} | {g(r, 'test_unseen')} |")

print("\n## Verdict against the registration\n")
ctl = [recs.get(f"control_s{s}") for s in range(3)]
if all(ctl):
    Rs = [r["R"] for r in ctl]
    if all(x <= 0.5 for x in Rs):
        v = "EXPOSURE (R ≤ 0.5 in all three seeds)"
    elif all(x >= 0.85 for x in Rs):
        v = "REPRESENTATION (R ≥ 0.85 in all three seeds)"
    else:
        v = "MIXED (neither threshold met in all seeds)"
    print(f"diag_control R per seed: {', '.join(f(x) for x in Rs)} → **{v}**")
else:
    print("diag_control incomplete: no verdict")
bad = [k for k, r in recs.items() if (k.startswith("sepops") or k.startswith("add2b_shared"))
       and r["mirror_minus_lone"] is not None and abs(r["mirror_minus_lone"]) >= 0.5]
print(f"\nControls with |mirror − lone| ≥ 0.5 (would flag a confound; registered controls are add2b shared and diag_sepops): "
      f"{', '.join(bad) if bad else 'none'}")
san = recs.get("e1_add_s0_sanity")
if san:
    print(f"\nSanity (E1 add s0, paired split): lone pairs n = {san['groups']['lone']['n']} (expected 0)")
