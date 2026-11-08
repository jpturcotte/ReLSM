"""Mechanism probes for grok_lab checkpoints: why are diagonal pairs (a, a) weak?

For each run directory (and each checkpoint in it), on the full table of the task (train and test pairs):
- recomputes the forward pass step by step and checks it matches the model's logits;
- checks the reconstructed split against the run record's counts, and reads the logged margins at the same
  update so the probe's numbers can be compared with the training logs;
- margins two ways: "lse" (correct logit minus logsumexp of the wrong ones, as grok_lab logs) and "top" (minus
  the strongest wrong one), per group, with the diagonal gap;
- competitors: how often the strongest wrong answer is an operand token, and the margins when the operand
  classes are removed from the competitors, against removing the same number of random wrong classes;
- margin split (top margin, exact): what each part writes into the final residual, projected onto the correct
  answer and onto the runner-up separately, with the final norm's scale held at its actual value. This is
  accounting, not cause: removing a part would change everything downstream;
- attention at '=' and at the second operand: weight on each position; the operand mass m = w_a + w_b and the
  ratio r = w_a / m, and how much each varies across off-diagonal pairs;
- interventions, each re-scored as above:
  - ratio_L0 (the decisive one): at '=' in layer 0 only, split each head's operand mass evenly (m/2, m/2),
    everything else unchanged. Diagonals are unchanged by construction (their two operand values are equal);
  - ratio_L1 (exploratory): the same in layer 1, where diagonal values are no longer identical;
  - freeze_L0 / freeze_L1 / freeze_L0+L1: a layer's whole attention pattern replaced by its mean (registered
    before the review; changes many things at once);
  - head_mean: one head's output at '=' replaced by its mean.

Groups: diag_train, diag_test, off_train, off_test (off_test = unseen off-diagonal: the swapped pair is not in
train either, as in grok_lab's test_*_unseen_offdiag). Gaps: train = off_train - diag_train, test = off_test -
diag_test. Operand classes: for every task here the answer alphabet is the operand alphabet, so the class of
operand a is index a among the answer columns.

Usage: python tools/grok_mech_probe.py [--ckpts] --out OUT.jsonl RUN_DIR [RUN_DIR ...]
The output is appended to and finished (run, checkpoint) pairs are skipped, so a stopped job can be restarted.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grok_lab as gl  # noqa: E402
from model import apply_rotary_emb  # noqa: E402  (grok_lab puts the repo root on the path)

GROUPS = ("diag_train", "diag_test", "off_train", "off_test")


def table(args):
    """Every valid pair of the run's (first) task: inputs, answers, answer columns, operands, alphabet size,
    groups."""
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if gl.multi_task(args):
        lay = gl.make_layout(args)
        name = args.tasks[0]
        spec = gl.TASK_SPECS[name]
        n = gl.alphabet_size(spec.alphabet, args.p)
        ids = torch.arange(n * n)
        a, b = ids // n, ids % n
        valid = spec.domain(a, b, args.p) if spec.domain is not None else torch.ones(n * n, dtype=torch.bool)
        train, _, _ = gl.make_split(n, args.task_frac.get(name, args.frac), split_seed, args.pairs, args.diagonals,
                                    None, valid if spec.domain is not None else None)
        ids = ids[valid]
        x, y = gl.encode_task(ids, 0, args, lay)
        cols = torch.tensor(lay.cols[0])
    else:
        n = args.p
        train, _, _ = gl.make_split(n, args.frac, split_seed, args.pairs, args.diagonals, args.holdout_line)
        ids = torch.arange(n * n)
        x, y = gl.encode(ids, n, args.operands, gl.task_of(args))
        cols = torch.arange(n + 2)
        if gl.separate_operands(args):
            cols = torch.cat([torch.arange(n), torch.tensor([2 * n, 2 * n + 1])])
    in_train = torch.zeros(n * n, dtype=torch.bool)
    in_train[train] = True
    a, b = ids // n, ids % n
    is_train = in_train[ids]
    seen = in_train[b * n + a]  # the swapped pair is in train
    diag = a == b
    off_test = ~diag & ~is_train & ~seen
    if getattr(args, "holdout_line", None) is not None:  # as grok_lab's evaluate: the held-out line is its own group
        off_test &= ~gl.line_mask(a, b, n, args.holdout_line)
    groups = {"diag_train": diag & is_train, "diag_test": diag & ~is_train,
              "off_train": ~diag & is_train, "off_test": off_test}
    return x, y, cols, a, b, n, groups


@torch.no_grad()
def forward(model, x, freeze=None, head_mean=None, ratio=None):
    """Manual forward. freeze: {layer: probs [H, T, T]} replaces that layer's attention pattern.
    head_mean: (layer, head, mean output [d]) replaces that head's output at '='. ratio: a layer in which each
    head's attention at '=' on the two operands is set to (m/2, m/2), keeping its operand mass m. Returns
    logits, the parts written into the residual, per-head outputs, attention probabilities, final residual."""
    h = model.tok_emb(x)
    parts, heads, probs = {"emb": h}, {}, []
    B, T, _ = h.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    pos = torch.arange(T).unsqueeze(0)
    for li, blk in enumerate(model.layers):
        at = blk.attn
        assert at.n_rep == 1, "GQA not handled"
        u = blk.attn_norm(h)
        q = at.q_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        k = at.k_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        v = at.v_proj(u).view(B, T, at.n_heads, at.d_head).transpose(1, 2)
        cos, sin = at.rotary(T, x.device)
        q, k = apply_rotary_emb(q, k, cos, sin, pos)
        s = (q @ k.transpose(-1, -2)) * at.scale
        pr = s.masked_fill(~causal, float("-inf")).softmax(-1)
        if freeze and li in freeze:
            pr = freeze[li].unsqueeze(0).expand_as(pr).clone()
        if ratio == li:
            pr = pr.clone()
            m = pr[:, :, T - 1, 0] + pr[:, :, T - 1, 2]
            pr[:, :, T - 1, 0] = m / 2
            pr[:, :, T - 1, 2] = m / 2
        probs.append(pr)
        o = pr @ v  # [B, H, T, dh]
        W = at.o_proj.weight  # [d, H*dh]
        ho = torch.stack([o[:, hh] @ W[:, hh * at.d_head:(hh + 1) * at.d_head].T for hh in range(at.n_heads)], 1)
        if head_mean is not None and head_mean[0] == li:
            ho[:, head_mean[1], T - 1] = head_mean[2]
        ao = ho.sum(1)
        if at.o_proj.bias is not None:
            ao = ao + at.o_proj.bias
        for hh in range(at.n_heads):
            heads[f"attn{li}.h{hh}"] = ho[:, hh]
        h = h + ao
        parts[f"attn{li}"] = ao
        mo = blk.ff(blk.ff_norm(h))
        h = h + mo
        parts[f"mlp{li}"] = mo
    logits = model.lm_head(model.norm(h))
    return logits, parts, heads, probs, h


def gm(v, groups):
    return {g: (v[m].float().mean().item() if m.any() else None) for g, m in groups.items()}


def with_gaps(stats):
    f = lambda a, b: None if stats[a] is None or stats[b] is None else stats[a] - stats[b]  # noqa: E731
    return {**stats, "gap_train": f("off_train", "diag_train"), "gap_test": f("off_test", "diag_test")}


def two_margins(z, y, drop=None):
    """top and lse margins of answer logits z [B, C]; drop: bool [B, C] of wrong classes left out."""
    w = z.double().scatter(1, y[:, None], float("-inf"))
    if drop is not None:
        w = w.masked_fill(drop, float("-inf"))
    zy = z.double().gather(1, y[:, None]).squeeze(1)
    return zy - w.max(1).values, zy - torch.logsumexp(w, 1)


def operand_drop(y, a, b, C):
    """Bool [B, C]: the operand classes a and b, except where an operand is the correct answer."""
    drop = torch.zeros(len(y), C, dtype=torch.bool)
    for o in (a, b):
        keep = o != y
        drop[keep.nonzero().squeeze(1), o[keep]] = True
    return drop


def random_drop(y, a, b, n, C, k, seed=0):
    """(drop, ok): drop is bool [B, C], k[i] random wrong classes per row, chosen among the alphabet's classes that
    are neither the answer nor an operand (the matched control for operand_drop). ok [B] is False where fewer than
    k[i] such classes exist (small alphabets, e.g. p = 3): that row has no matched control."""
    g = torch.Generator().manual_seed(seed)
    score = torch.rand(len(y), C, generator=g)
    score[:, n:] = -1.0  # never the op or '=' classes
    score.scatter_(1, y[:, None], -1.0)
    score.scatter_(1, a[:, None], -1.0)
    score.scatter_(1, b[:, None], -1.0)
    ok = (score >= 0).sum(1) >= k
    top = score.topk(min(2, C), dim=1).indices
    drop = torch.zeros(len(y), C, dtype=torch.bool)
    for j in range(min(2, C)):
        rows = ((k > j) & ok).nonzero().squeeze(1)
        drop[rows, top[rows, j]] = True
    return drop, ok


def score(logits, y, cols, a, b, n, groups):
    """Margins (top and lse), accuracy, competitor identity and operand-exclusion re-scoring, per group."""
    z = logits[:, -1, cols]
    C = z.shape[1]
    top, lse = two_margins(z, y)
    od = operand_drop(y, a, b, C)
    k = od.sum(1)
    top_xo, lse_xo = two_margins(z, y, od)
    rd, rd_ok = random_drop(y, a, b, n, C, k)
    top_xr, lse_xr = two_margins(z, y, rd)
    groups_r = {g: m & rd_ok for g, m in groups.items()}  # rows with a matched random control only
    w = z.scatter(1, y[:, None], float("-inf"))
    runner = w.argmax(1)
    runner_is_operand = ((runner == a) & (a != y)) | ((runner == b) & (b != y))
    has_rival_operand = k > 0
    zbar = w.masked_fill(torch.isinf(w), 0).sum(1) / (C - 1)  # mean wrong logit
    opz = torch.where(a != y, z.gather(1, a[:, None]).squeeze(1), torch.full_like(zbar, float("nan")))
    out = {"top": with_gaps(gm(top, groups)), "lse": with_gaps(gm(lse, groups)),
           "acc": gm(top > 0, groups),
           "top_no_operands": with_gaps(gm(top_xo, groups)), "lse_no_operands": with_gaps(gm(lse_xo, groups)),
           "top_no_random": with_gaps(gm(top_xr, groups_r)), "lse_no_random": with_gaps(gm(lse_xr, groups_r)),
           "runner_is_operand": gm(runner_is_operand.float(), {g: m & has_rival_operand for g, m in groups.items()}),
           "operand_a_logit_above_mean_wrong": gm(opz - zbar, {g: m & (a != y) for g, m in groups.items()})}
    return out


def margin_split(model, logits, parts, heads, resid, y, cols):
    """Exact split of the top margin at '=' into the parts, plus each part's push on the correct class and on
    the runner-up separately, and the final norm's scale."""
    z = logits[:, -1, cols]
    w = z.scatter(1, y[:, None], float("-inf"))
    runner = w.argmax(1)
    margin = z.gather(1, y[:, None]).squeeze(1) - w.max(1).values
    r = resid[:, -1]
    if isinstance(model.norm, torch.nn.Identity):
        inv_rms = torch.ones(len(r), 1)
        scale = torch.ones_like(r)
    else:
        inv_rms = r.pow(2).mean(-1, keepdim=True).add(model.norm.eps).rsqrt()
        scale = inv_rms * model.norm.weight
    Wu = model.lm_head.weight[cols]  # [C, d]
    wy, wr = Wu[y], Wu[runner]
    split, to_y, to_r = {}, {}, {}
    for name, c in {**parts, **heads}.items():
        cs = c[:, -1] * scale
        to_y[name] = (cs * wy).sum(-1)
        to_r[name] = (cs * wr).sum(-1)
        split[name] = to_y[name] - to_r[name]
    total = sum(split[k] for k in parts)
    return split, to_y, to_r, inv_rms.squeeze(1), (total - margin).abs().max().item()


def attn_stats(probs, groups):
    off = groups["off_train"] | groups["off_test"]
    dg = groups["diag_train"] | groups["diag_test"]
    out = {}
    for li, pr in enumerate(probs):
        for hh in range(pr.shape[1]):
            for qpos, qname in ((3, "eq"), (2, "b")):
                w = pr[:, hh, qpos, :qpos + 1]
                m = w[:, 0] + w[:, 2]
                rr = w[:, 0] / m.clamp_min(1e-12)
                out[f"L{li}.h{hh}.q{qname}"] = {
                    "off_mean": w[off].mean(0).tolist(), "diag_mean": w[dg].mean(0).tolist(),
                    "ab_diff_std_off": (w[:, 0] - w[:, 2])[off].std().item(),
                    "mass_mean_off": m[off].mean().item(), "mass_mean_diag": m[dg].mean().item(),
                    "mass_std_off": m[off].std().item(), "ratio_mean_off": rr[off].mean().item(),
                    "ratio_std_off": rr[off].std().item()}
    return out


def logged_at(run, args, update):
    """The logged margins at this update (bf16 in training, lse definition), for comparison with the probe."""
    t = args.tasks[0] if gl.multi_task(args) else None
    k = (lambda s: f"{s}@{t}") if t else (lambda s: s)
    row = None
    for line in open(run / "metrics.jsonl"):
        r = json.loads(line)
        if r.get("update") == update and k("test_margin_mean_diagonal") in r:
            row = r
    if row is None:
        return None
    g = {"diag_train": row.get(k("train_margin_mean_diagonal")), "diag_test": row.get(k("test_margin_mean_diagonal")),
         "off_train": row.get(k("train_margin_mean_offdiag")), "off_test": row.get(k("test_margin_mean_unseen_offdiag"))}
    return with_gaps(g)


def count_check(run, args, groups):
    rec = json.load(open(run / "run_record.json"))
    src = rec["tasks"][0] if gl.multi_task(args) and "tasks" in rec else rec
    mine = {"n_train": int((groups["diag_train"] | groups["off_train"]).sum()),
            "n_train_diagonal": int(groups["diag_train"].sum()), "n_test_diagonal": int(groups["diag_test"].sum())}
    theirs = {k: src.get(k) for k in mine}
    return {"ok": mine == theirs, "probe": mine, "record": theirs}


CHUNK = 2048  # pairs per forward: keeps the peak memory low (the s5 table has 14,400 pairs)


def chunks(B):
    return [slice(i, min(i + CHUNK, B)) for i in range(0, B, CHUNK)]


def forward_logits(model, x, **kw):
    return torch.cat([forward(model, x[s], **kw)[0] for s in chunks(len(x))])


def main_pass(model, x, y, cols):
    """The unmodified forward in chunks: logits, attention probabilities, each head's output at '=', the
    margin split, and the check against model(x). Only per-example results are kept between chunks."""
    logits, probs, head_eq, err_f, err_s = [], None, {}, 0.0, 0.0
    split, to_y, to_r, inv_rms = {}, {}, {}, []
    for s in chunks(len(x)):
        lg, parts, heads, pr, resid = forward(model, x[s])
        err_f = max(err_f, (model(x[s])[0] - lg).abs().max().item())
        sp, ty, tr, ir, e = margin_split(model, lg, parts, heads, resid, y[s], cols)
        err_s = max(err_s, e)
        logits.append(lg)
        probs = [p.clone() for p in pr] if probs is None else [torch.cat([q, p]) for q, p in zip(probs, pr)]
        for k, v in heads.items():
            head_eq.setdefault(k, []).append(v[:, -1])
        for d, src in ((split, sp), (to_y, ty), (to_r, tr)):
            for k, v in src.items():
                d.setdefault(k, []).append(v)
        inv_rms.append(ir)
        del parts, heads, resid
    cat = lambda d: {k: torch.cat(v) for k, v in d.items()}  # noqa: E731
    return (torch.cat(logits), probs, cat(head_eq), cat(split), cat(to_y), cat(to_r), torch.cat(inv_rms),
            err_f, err_s)


def probe(model, x, y, cols, a, b, n, groups):
    logits, probs, head_eq, split, to_y, to_r, inv_rms, err_f, err_s = main_pass(model, x, y, cols)
    rec = {"forward_max_abs_err": err_f, "split_max_abs_err": err_s}
    rec["score"] = score(logits, y, cols, a, b, n, groups)
    rec["split"] = {k: with_gaps(gm(v, groups)) for k, v in split.items()}
    rec["to_correct"] = {k: gm(v, groups) for k, v in to_y.items() if "." not in k}
    rec["to_runner"] = {k: gm(v, groups) for k, v in to_r.items() if "." not in k}
    rec["final_inv_rms"] = gm(inv_rms, groups)
    rec["attn"] = attn_stats(probs, groups)
    iv = {}
    for li in range(len(probs)):
        lg = forward_logits(model, x, ratio=li)
        iv[f"ratio_L{li}"] = score(lg, y, cols, a, b, n, groups)
        if li == 0:  # diagonals must be untouched: their two operand values are identical in layer 0
            dg = groups["diag_train"] | groups["diag_test"]
            iv["ratio_L0"]["diag_max_abs_logit_change"] = (lg[dg] - logits[dg]).abs().max().item()
        iv[f"freeze_L{li}"] = score(forward_logits(model, x, freeze={li: probs[li].mean(0)}), y, cols, a, b, n, groups)
    iv["freeze_L0+L1"] = score(forward_logits(model, x, freeze={li: p.mean(0) for li, p in enumerate(probs)}),
                               y, cols, a, b, n, groups)
    for name, v in head_eq.items():
        li, hh = int(name[4]), int(name.split(".h")[1])
        iv[f"head_mean.{name}"] = score(forward_logits(model, x, head_mean=(li, hh, v.mean(0))),
                                        y, cols, a, b, n, groups)
    rec["interventions"] = iv
    return rec


def load(run, ckpt):
    fin = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**fin["args"])
    args.dropout = 0.0
    model = gl.build_model(args)
    if ckpt == "final":
        sd, update = fin["model"], fin["update"]
    else:
        ck = torch.load(run / ckpt, map_location="cpu", weights_only=False)
        sd, update = ck["model"], ck["update"]
    model.load_state_dict(sd)
    model.eval()
    return model, args, update


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpts", action="store_true", help="also probe every ckpt_*.pt")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    done = set()  # resumable: a restarted job skips what is already in the output
    if Path(a.out).exists():
        for line in open(a.out):
            try:
                rec = json.loads(line)
                done.add((rec["run"], rec["ckpt"]))
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut short by a killed worker
    ends_clean = not Path(a.out).exists() or Path(a.out).read_bytes()[-1:] in (b"", b"\n")
    with open(a.out, "a") as f:
        if not ends_clean:
            f.write("\n")  # end a line cut short by a killed worker, so the next record starts fresh
        for r in map(Path, a.runs):
            names = (sorted(p.name for p in r.glob("ckpt_*.pt")) if a.ckpts else []) + ["final"]
            for ck in names:
                if (str(r), ck) in done:
                    continue
                model, args, update = load(r, ck)
                x, y, cols, oa, ob, n, groups = table(args)
                rec = {"run": str(r), "name": r.name, "ckpt": ck, "update": update,
                       "task": args.tasks[0] if gl.multi_task(args) else gl.task_of(args),
                       "operands": getattr(args, "operands", "shared"), "pairs": args.pairs,
                       "n": {g: int(m.sum()) for g, m in groups.items()},
                       "count_check": count_check(r, args, groups), "logged": logged_at(r, args, update),
                       **probe(model, x, y, cols, oa, ob, n, groups)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                s = rec["score"]
                print(f"{r.name:16s} {ck:16s} fwd_err {rec['forward_max_abs_err']:.1e} split_err "
                      f"{rec['split_max_abs_err']:.1e} counts_ok {rec['count_check']['ok']} "
                      f"gap_train lse {fmt_gap(s['lse']['gap_train'])} top {fmt_gap(s['top']['gap_train'])}", flush=True)


def fmt_gap(v):
    """A gap for the progress line; None when the run has no pairs in that group (e.g. --diagonals test)."""
    return "n/a" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    main()
