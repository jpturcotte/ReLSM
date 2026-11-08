"""Score the restore tests (H6). Usage: python restore_score.py > restore_results.md"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, "restore", "all.jsonl")) if l.startswith("{")]
f = lambda x, d=2: f"{x:.{d}f}"  # noqa: E731
name = lambda r: ("e1_" if "e1_" in r["run"] else "diag_") + os.path.basename(r["run"])  # noqa: E731
TESTS = ["massL1_up_diag", "norm_up_diag", "both_up_diag", "massL1_down_off", "norm_down_off"]


def closes(r, t):
    b = r["results"]["base"]["gap_train"]
    return (b - r["results"][t]["gap_train"]) / b


print("# Restore tests (H6) results\n")
print("lse margins, final weights. Base and each test show gap_train; brackets give the share of the gap closed.\n")
print("| run | base gap | L1 mass up on diag | norm up on diag | both up on diag | L1 mass down on off | norm down on off |"
      " Δ diag_train (mass up / norm up) | Δ off_train (mass down / norm down) | inv RMS diag / off |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    R, b = r["results"], r["results"]["base"]
    cells = [f"{f(R[t]['gap_train'])} ({100 * closes(r, t):.0f}%)" if abs(b["gap_train"]) >= 0.5 else f(R[t]["gap_train"])
             for t in TESTS]
    dd = f"{f(R['massL1_up_diag']['diag_train'] - b['diag_train'])} / {f(R['norm_up_diag']['diag_train'] - b['diag_train'])}"
    do = f"{f(R['massL1_down_off']['off_train'] - b['off_train'])} / {f(R['norm_down_off']['off_train'] - b['off_train'])}"
    print(f"| {name(r)} | {f(b['gap_train'])} | " + " | ".join(cells) + f" | {dd} | {do} | "
          f"{f(r['inv_rms_diag'], 4)} / {f(r['inv_rms_off'], 4)} |")

print("\n## Verdicts against the registration\n")
main = [r for r in recs if "sepops" not in r["run"]]
for t, label in (("massL1_up_diag", "H6a (layer-1 operand mass)"), ("norm_up_diag", "H6b (final-norm scale)")):
    c = [closes(r, t) for r in main]
    v = "CAUSAL / MATTERS" if all(x >= 0.25 for x in c) else "NOT CAUSAL" if all(x < 0.10 for x in c) else "MIXED"
    print(f"- {label}: closes {', '.join(f'{100 * x:.0f}%' for x in c)} → **{v}**")
c = [closes(r, "both_up_diag") for r in main]
print(f"- both together: closes {', '.join(f'{100 * x:.0f}%' for x in c)}")
sep = [r for r in recs if "sepops" in r["run"]]
worst = max(abs(r["results"][t]["gap_train"] - r["results"]["base"]["gap_train"]) for r in sep for t in TESTS)
print(f"\nSanity: separate-token runs, largest |Δ gap| over all tests = {f(worst)} (registered bound 0.3)")
print("Sanity: base gaps vs the v2 probe are checked by hand in the summary.")
mem = recs[-1]
print(f"\nMemory for the whole job (12 checkpoints): peak RAM {f(mem['peak_ram_gb'])} GB, peak commit "
      f"{f(mem['peak_commit_gb'])} GB, GPU in use {f(mem['gpu_used_gb'])} GB")
