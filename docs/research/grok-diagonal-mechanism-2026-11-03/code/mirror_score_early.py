"""Score the mirror test over training (H5b). Usage: python mirror_score_early.py > mirror_early_results.md"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


runs = {}
for p in sorted(glob.glob(os.path.join(HERE, "early", "mirror", "*.jsonl"))):
    rs = {}
    for l in open(p):
        if l.startswith("{"):
            r = json.loads(l)
            rs[r["update"]] = r
    runs[os.path.basename(p)[:-6]] = [rs[u] for u in sorted(rs)]

print("# Mirror test over training (H5b), CPU reruns\n")
print("lse margins of train pairs. R = D_lone / D_mirror, scored only where D_mirror ≥ 1 (else in brackets).\n")
print("| run | update | train acc off (mirror / lone) | diag | mirror | lone | D_mirror | D_lone | R | mirror − lone |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, rs in runs.items():
    for r in rs:
        G = r["groups"]
        R = r["R"]
        rs_ = f(R) if r["D_mirror"] is not None and r["D_mirror"] >= 1 else f"({f(R)})"
        print(f"| {k} | {r['update']} | — / {f(r['acc_lone'], 3)} | {f(G['diag']['mean'])} ± {f(G['diag']['se'])} | "
              f"{f(G['mirror']['mean'])} | {f(G['lone']['mean'])} | {f(r['D_mirror'])} | {f(r['D_lone'])} | {rs_} | "
              f"{f(r['mirror_minus_lone'])} |")

print("\n## Verdict against the registration\n")
for s in range(3):
    rs = runs.get(f"control_s{s}", [])
    scored = [(r["update"], r["R"]) for r in rs if r["D_mirror"] is not None and r["D_mirror"] >= 1]
    early = [R for u, R in scored if u <= 1000]
    print(f"- control_s{s}: scored (update, R): {', '.join(f'({u}, {f(R)})' for u, R in scored) or 'none'}; "
          f"early ≤ 1000 all ≤ 0.5: {bool(early) and all(R <= 0.5 for R in early)}; "
          f"all ≥ 0.85: {bool(scored) and all(R >= 0.85 for _, R in scored)}")
bad = [(k, r["update"], r["mirror_minus_lone"]) for k, rs in runs.items() if k.startswith("sepops")
       for r in rs if r["mirror_minus_lone"] is not None and abs(r["mirror_minus_lone"]) >= 0.5]
print(f"\nsepops checkpoints with |mirror − lone| ≥ 0.5: {bad or 'none'}")
