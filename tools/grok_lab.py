"""Standalone grokking harness: a + b mod p on BaselineTransformer, full batch, fp32.

PR 2 of docs/research/grokking-fix-plan-2026-10-28.md (arm A1-H of #249). It builds the repo's
model directly and uses nothing from data.py or train.py.

- Task: the enumerated a + b mod p table (or a + 2b with --task add2b, which is not commutative),
  one token per number. Input [a, op, b, =]; the
  answer is predicted at the last position. Vocabulary p + 2 (2p + 2 with --operands separate,
  where the second operand has its own tokens). --diagonals train|test moves every pair (a, a)
  to one side of the split; --holdout_line k moves the lines b = a +- k to test as its positive
  control. These exist to test why diagonals are weak: the model marks position only through
  RoPE, so two identical operand tokens pass identical value vectors. Under --operands separate
  the outputs keep the same p + 2 classes (the second operand's tokens are inputs only), and
  (a, b) and (b, a) share no tokens, so the transpose leak is gone: read that arm on the gap
  between diagonal and unseen off-diagonal pairs, not on accuracy against the control.
  Evals log margins per group: test_margin_mean_/p5_ for diagonal, unseen_offdiag and seen, and
  train_margin_mean_/p5_ for diagonal and offdiag. With --holdout_line, unseen_offdiag excludes
  the held-out lines too (test_acc_line and test_margin_mean_line report them).
- Split: seeded, at --frac. `ordered` splits the p^2 ordered pairs (Power 2022); `paired`
  keeps (a, b) and (b, a) on the same side. Each test pair is tagged "transpose seen" when
  (b, a) is in train; a diagonal pair (a, a) is its own transpose, so it is never seen.
  Diagonals are about 2% of the unseen test pairs (ordered split, frac 0.5; about 1% under
  paired, where every test pair is unseen), but they held 13 of the 15 late test errors in the
  long stack runs (2026-11-01), so evals also log test_acc_diagonal and test_acc_unseen_offdiag
  (unseen without the diagonals). test_acc_unseen still includes them, as before.
- Model: d 128, 2 layers, 4 heads, dropout 0, head tied to the embedding (ReLSM's model and the
  owner's primary arm). --untie gives a separate head (Power 2022, Nanda 2023); --no_final_norm
  removes the final RMSNorm. The head is set here explicitly, not taken from the model defaults.
- Optimizer: AdamW, betas (0.9, --beta2), --eps, weight decay --wd on the --wd_scope set.
  Linear warmup over --warmup updates, then constant LR; --lr_schedule wsd copies train.py's
  default instead (hold until 40% of the run, then linear decay to --min_lr at the last update;
  train.py's own LR is --lr 6e-4). Progress is read in updates; train.py reads it in answer
  tokens (#217), which for this task is close to proportional. No early stopping. By default: no
  clipping, full batch, fp32. To step toward train.py's setup one setting at a time: --clip,
  --batch (epoch-shuffled minibatches), --bf16 autocast (model and loss under bf16 autocast, as
  train.py runs on CUDA; autocast keeps the cross-entropy in fp32).
- Loss: cross-entropy at the answer position; --loss margin is Barron et al. Eq. 6,
  softplus(-m) with m = z_t - logsumexp_{j != t} z_j; --loss_fp64 casts logits first.
  --loss ce_rounded is cross-entropy with the softmax sum rounded once to --round_dtype: the
  low-precision cliff (bf16: margin 8 ln 2 = 5.55) on any device. It copies what PyTorch's CPU
  bf16 log_softmax kernel does; the CUDA kernel keeps the sum in fp32 and has no such cliff.

Outputs in --out: metrics.jsonl (append-only, one line per eval, also printed to stdout),
run_record.json, ckpt_<update>.pt (weights), final.pt (weights + optimizer) and final_errors.json
(every train and test pair the final model gets wrong, in eval mode). Files from an earlier run
in the same --out are moved to old_<time>/ first.

Reading metrics.jsonl: train_loss, test_loss and the margins are fp64 readouts of the logits
after update `update` (bf16-rounded logits under --bf16); train_obj_pre is the training
objective before that update. Under --batch, train_obj_* and the spike count follow single
minibatches, so they are noisier and not comparable with full-batch arms. Under --loss
ce_rounded, train_obj_* is quantized (per-example steps of log(1 + k 2^-7) for bf16) and reads
exactly 0 once every train example is past the cliff, while the fp64 train_loss stays nonzero.
Evals run in eval mode, so with --dropout > 0 the margin and zero-gradient readouts are of the
model without dropout, not of the gradient training saw. With --dropout > 0 each eval also
logs the same readouts from one train-mode pass over the train set (keys ending _trainmode;
None without dropout): one dropout draw per train example, which under full batch are exactly
the masks of the next update. That pass draws from a forked RNG, so it leaves the training
run's random stream unchanged.
margin_frac_cliff counts margins past the nominal fp32 cliff ln 2^24 = 16.6. The fp32 kernels
zero the target gradient somewhat earlier (about 14.5-16.6, depending on the kernel and on how
the other logits are spread), so for the share of examples that get no target gradient, quote
zero_tgt_grad_frac: it is measured with the arm's own loss on the arm's own device.

    python tools/grok_lab.py --out runs/grok/std_s0 --seed 0
"""
import argparse
import contextlib
import dataclasses
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import BaselineTransformer, TransformerConfig  # noqa: E402

CLIFF_FP32 = math.log(2.0 ** 24)  # 16.6: past this margin, fp32 cross-entropy's target gradient rounds to 0
CLIFF_BF16 = math.log(2.0 ** 8)  # 5.55: the same for --loss ce_rounded with bf16 (and CPU's bf16 cross-entropy)
ROUND_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def make_split(p, frac, seed, pairs="ordered", diagonals="split", holdout_line=None):
    """Return (train ids, test ids, seen) over pair ids a*p + b. seen[i]: (b, a) of test pair i is in train.
    diagonals: "split" leaves the pairs (a, a) where the seeded split put them; "train" or "test" then moves
    all p of them to that side (at frac 0.5 the train set grows or shrinks by about p / 2).
    holdout_line k: also move every pair on the lines b = a + k and b = a - k (mod p) to test, 2p pairs that
    are each other's transposes. It is the positive control for --diagonals test: a held-out line other than
    the diagonal, so a weak diagonal can be told apart from a weak held-out line."""
    g = torch.Generator().manual_seed(seed)
    n, n_train = p * p, int(round(frac * p * p))
    in_train = torch.zeros(n, dtype=torch.bool)
    if pairs == "ordered":
        in_train[torch.randperm(n, generator=g)[:n_train]] = True
    elif pairs == "paired":
        groups = [[a * p + b] if a == b else [a * p + b, b * p + a] for a in range(p) for b in range(a, p)]
        taken = 0
        for k in torch.randperm(len(groups), generator=g).tolist():
            if taken + len(groups[k]) <= n_train:
                in_train[groups[k]] = True
                taken += len(groups[k])
    else:
        raise ValueError(f"unknown --pairs {pairs!r}")
    if diagonals in ("train", "test"):
        in_train[torch.arange(p) * (p + 1)] = diagonals == "train"
    elif diagonals != "split":
        raise ValueError(f"unknown --diagonals {diagonals!r}")
    if holdout_line is not None:
        if not 0 < holdout_line < p or (2 * holdout_line) % p == 0:
            raise ValueError(f"--holdout_line {holdout_line} must be in 1..{p - 1} with 2k != 0 mod p")
        a = torch.arange(p)
        in_train[torch.cat([a * p + (a + holdout_line) % p, a * p + (a - holdout_line) % p])] = False
    train, test = in_train.nonzero().squeeze(1), (~in_train).nonzero().squeeze(1)
    seen = in_train[(test % p) * p + test // p]
    return train, test, seen


TASKS = {"add": (1, 1), "add2b": (1, 2)}  # answer = (ca * a + cb * b) mod p


def encode(ids, p, operands="shared", task="add"):
    """Pair ids -> inputs [a, op, b, =] and answers (a + b) mod p, or (a + 2b) mod p for task add2b (not
    commutative: (a, b) and (b, a) have different answers, so a symmetry of the input no longer matches one of
    the answer). shared: one token per number for both operands (op = p, '=' = p + 1). separate: the second
    operand uses its own tokens p + b, so (a, a) is no longer two identical tokens (op = 2p, '=' = 2p + 1);
    answers stay tokens 0..p-1."""
    a, b = ids // p, ids % p
    k = p if operands == "separate" else 0
    ca, cb = TASKS[task]
    x = torch.stack([a, torch.full_like(a, p + k), b + k, torch.full_like(a, p + k + 1)], dim=1)
    return x, (ca * a + cb * b) % p


def task_of(args):
    return getattr(args, "task", "add")  # run records from before the flag: add


def separate_operands(args):
    return getattr(args, "operands", "shared") == "separate"  # run records from before the flag: shared


def vocab_size(args):
    return 2 * args.p + 2 if separate_operands(args) else args.p + 2


def operands_of(x, args):
    """The numbers (a, b) behind encoded inputs x."""
    return x[:, 0], x[:, 2] - (args.p if separate_operands(args) else 0)


def build_model(args):
    cfg = TransformerConfig(vocab_size=vocab_size(args), max_seq_len=4, d_model=128, n_layers=2, n_heads=4,
                            dropout=args.dropout, tie_weights=not args.untie)
    with contextlib.redirect_stdout(sys.stderr):  # keep stdout pure JSONL
        model = BaselineTransformer(cfg)
    if args.no_final_norm:
        model.norm = nn.Identity()
    return model


def decays(name, scope):
    """Whether a parameter is in the weight-decay group."""
    low = name.lower()
    if scope == "all":  # Power 2022: every parameter, norm gains included
        return True
    if scope == "embeddings":  # every weight but the norm gains
        return "norm" not in low
    if scope == "repo":  # mirrors train.py build_optimizer's exemption (#186)
        return not (name.endswith(".bias") or "norm" in low or "tok_emb" in low or "lm_head" in low)
    raise ValueError(f"unknown --wd_scope {scope!r}")


def build_optimizer(model, args):
    """AdamW; group 0 is the decayed set. named_parameters() lists a tied table once."""
    named = list(model.named_parameters())
    groups = [{"params": [q for n, q in named if decays(n, args.wd_scope)], "weight_decay": args.wd}]
    exempt = [q for n, q in named if not decays(n, args.wd_scope)]
    if exempt:
        groups.append({"params": exempt, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, args.beta2), eps=args.eps)


WSD_STABLE_END = 0.4  # train.py get_lr: hold max LR until 40% of the run


def lr_at(u, args):
    """LR for update u (0-based). Progress is read before the update, as train.py reads
    tokens_seen. wsd copies train.py get_lr: max LR until 40%, then linear to --min_lr."""
    warm = min(1.0, (u + 1) / args.warmup) if args.warmup > 0 else 1.0
    if warm < 1.0 or args.lr_schedule == "constant":
        return args.lr * warm
    progress = u / args.updates
    if progress < WSD_STABLE_END:
        return args.lr
    return args.lr - (args.lr - args.min_lr) * min((progress - WSD_STABLE_END) / (1 - WSD_STABLE_END), 1.0)


def margins(z, y):
    """Per-example m = z_t - logsumexp_{j != t} z_j (the quantity the fp32 cliff applies to)."""
    zt = z.gather(1, y[:, None]).squeeze(1)
    return zt - torch.logsumexp(z.scatter(1, y[:, None], float("-inf")), dim=1)


def ce_rounded(z, y, dtype):
    """Cross-entropy with the softmax sum s = sum_j exp(z_j - max) rounded once to `dtype`, in
    both the loss and its gradient. Everything else runs in fp32 (fp64 under --loss_fp64).
    The target's gradient is 1/s' - 1 when it holds the max, so it is exactly 0 once s rounds to
    1, i.e. once the margin passes 8 ln 2 = 5.55 for bf16 (11 ln 2 = 7.6 for fp16). The other
    logits keep their gradient e_j / s'.
    This is what PyTorch's CPU bf16 log_softmax does, made device-independent."""
    z = z.to(torch.promote_types(z.dtype, torch.float32))
    shift = z.max(1, keepdim=True).values.detach()  # softmax is shift-invariant; no gradient through the max
    s = torch.exp(z - shift).sum(1, keepdim=True)
    # Round the value only. The cliff comes from s' = 1 with the detached shift (target gradient
    # 1/s' - 1). This form keeps the backward in fp32: a plain cast pair would round the upstream
    # gradient (1/N) to bf16, and the exact zero past the cliff would be lost for N = 4704.
    s = s + (s.to(dtype).to(s.dtype) - s).detach()
    log_p = z - shift - torch.log(s)
    return -log_p.gather(1, y[:, None]).mean()


def objective(logits, y, args):
    z = logits.double() if args.loss_fp64 else logits
    if args.loss == "ce":
        return F.cross_entropy(z, y)
    if args.loss == "ce_rounded":
        return ce_rounded(z, y, ROUND_DTYPES[args.round_dtype])
    return F.softplus(-margins(z, y)).mean()


def bf16_ctx(args, device):
    """--bf16 autocast: model and loss under bf16 autocast, as train.py runs on CUDA. Autocast
    runs cross-entropy in fp32, so the logits are bf16-rounded but the loss has only the fp32 cliff."""
    on = args.bf16 == "autocast"
    return torch.autocast(device.type, dtype=torch.bfloat16) if on else contextlib.nullcontext()


def answer_logits(model, x, args):
    """Logits at '=' over the answer classes. Under --operands separate the second operand's p tokens are
    dropped from the outputs, so the classes are the same p + 2 as with shared operands (answers 0..p-1, then
    op and '='): those rows of the tied table stay pure inputs, never pushed down as wrong answers."""
    with bf16_ctx(args, x.device):
        z = model(x)[0][:, -1, :]
    if separate_operands(args):
        p = args.p
        z = torch.cat([z[:, :p], z[:, 2 * p:]], dim=1)
    return z


def line_mask(xa, xb, p, k):
    """Pairs on the lines b = a + k or b = a - k (mod p)."""
    d = (xb - xa) % p
    return (d == k) | (d == (p - k) % p)


def loss_of(z, y, args):
    with bf16_ctx(args, z.device):
        return objective(z, y, args)


class Batches:
    """--batch: epoch shuffle without replacement, dropping the last partial batch (a shuffled DataLoader)."""

    def __init__(self, n, size, seed):
        if not 0 < size <= n:
            raise ValueError(f"--batch {size} must be in 1..{n}")
        self.n, self.size, self.g, self.order, self.pos = n, size, torch.Generator().manual_seed(seed + 1), None, n

    def next(self):
        if self.pos + self.size > self.n:
            self.order, self.pos = torch.randperm(self.n, generator=self.g), 0
        self.pos += self.size
        return self.order[self.pos - self.size:self.pos]


def global_norm(tensors):
    return math.sqrt(sum(float(t.detach().double().pow(2).sum()) for t in tensors))


def train_step(model, opt, x, y, args, lr, measure=False):
    """One update on (x, y), the full train set or a minibatch. Returns (loss before the update, measured stats or None)."""
    model.train()
    for group in opt.param_groups:
        group["lr"] = lr
    loss = loss_of(answer_logits(model, x, args), y, args)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if not measure:
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        return loss.detach(), None
    named = list(model.named_parameters())
    before = [q.detach().clone() for _, q in named]
    grad_norm = global_norm([q.grad for _, q in named if q.grad is not None])  # before clipping
    if args.clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    opt.step()
    deltas = [q.detach() - b for (_, q), b in zip(named, before)]
    rel = [global_norm([d]) / max(global_norm([b]), 1e-30) for d, b in zip(deltas, before)]
    k = max(range(len(rel)), key=rel.__getitem__)
    dnorm, pnorm = global_norm(deltas), global_norm(before)
    return loss.detach(), {"update_norm": dnorm, "param_norm_before": pnorm, "rel_update": dnorm / pnorm,
                           "grad_norm": grad_norm, "top_param": named[k][0], "top_param_rel_update": rel[k]}


MARGIN_KEYS = ("margin_min", "margin_mean", "margin_p5", "margin_frac_cliff", "margin_frac_bf16_cliff",
               "zero_tgt_grad_frac")


def margin_stats(z, y, args):
    """Train-set margin readouts from answer logits z (in the arm's precision)."""
    m = margins(z.double(), y)
    with torch.enable_grad():  # the target gradient the arm's own loss produces
        leaf = z.detach().requires_grad_()
        (g,) = torch.autograd.grad(loss_of(leaf, y, args), leaf)
    return dict(zip(MARGIN_KEYS, (m.min().item(), m.mean().item(), torch.quantile(m, 0.05).item(),
                                  (m > CLIFF_FP32).float().mean().item(), (m > CLIFF_BF16).float().mean().item(),
                                  (g.gather(1, y[:, None]) == 0).float().mean().item())))


def rng_devices(device):
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []


@torch.no_grad()
def evaluate(model, opt, data, args):
    model.eval()
    out = {}
    trainmode = None
    if args.dropout > 0:  # one pass with dropout on, as training sees it; forked RNG keeps the run's stream
        with torch.random.fork_rng(devices=rng_devices(data["train_x"].device)):
            model.train()
            try:
                trainmode = margin_stats(answer_logits(model, data["train_x"], args), data["train_y"], args)
            finally:
                model.eval()
    out.update({k + "_trainmode": None if trainmode is None else trainmode[k] for k in MARGIN_KEYS})
    for split in ("train", "test"):
        y = data[split + "_y"]
        z = answer_logits(model, data[split + "_x"], args)  # in the arm's precision
        correct = z.argmax(1) == y
        out[split + "_loss"] = F.cross_entropy(z.double(), y).item()  # fp64: no cliff in the readout
        out[split + "_acc"] = correct.float().mean().item()
        if split == "test":
            seen = data["test_seen"]
            out["test_acc_seen"] = correct[seen].float().mean().item() if seen.any() else None
            out["test_acc_unseen"] = correct[~seen].float().mean().item() if (~seen).any() else None
            out["test_loss_unseen"] = F.cross_entropy(z[~seen].double(), y[~seen]).item() if (~seen).any() else None
            x = data["test_x"]
            xa, xb = operands_of(x, args)
            diag = xa == xb  # (a, a): no transpose partner, so always inside "unseen"
            k = getattr(args, "holdout_line", None)
            line = line_mask(xa, xb, args.p, k) if k is not None else torch.zeros_like(diag)
            off = ~seen & ~diag & ~line  # the comparison group: random unseen pairs, no held-out family
            out["test_acc_diagonal"] = correct[diag].float().mean().item() if diag.any() else None
            out["test_acc_unseen_offdiag"] = correct[off].float().mean().item() if off.any() else None
            m = margins(z.double(), y)  # per group: a continuous readout, where accuracy counts a few pairs
            for g, mask in (("diagonal", diag), ("unseen_offdiag", off), ("seen", seen)):
                out[f"test_margin_mean_{g}"] = m[mask].mean().item() if mask.any() else None
                out[f"test_margin_p5_{g}"] = torch.quantile(m[mask], 0.05).item() if mask.any() else None
            if k is not None:
                out["test_acc_line"] = correct[line].float().mean().item()
                out["test_margin_mean_line"] = m[line].mean().item()
        else:
            out.update(margin_stats(z, y, args))
            xa, xb = operands_of(data["train_x"], args)  # memorized (a, a) vs other train pairs
            m = margins(z.double(), y)
            for g, mask in (("diagonal", xa == xb), ("offdiag", xa != xb)):
                out[f"train_margin_mean_{g}"] = m[mask].mean().item() if mask.any() else None
                out[f"train_margin_p5_{g}"] = torch.quantile(m[mask], 0.05).item() if mask.any() else None
    out["norm_total"] = global_norm(model.parameters())  # at init, mostly the RMSNorm gains (sqrt(640) = 25.3)
    out["norm_gains"] = global_norm([q for n, q in model.named_parameters() if "norm" in n])
    out["norm_matrices"] = global_norm([q for n, q in model.named_parameters() if "norm" not in n])
    out["norm_decayed"] = global_norm(opt.param_groups[0]["params"])
    out["norm_table"] = global_norm([model.tok_emb.weight])
    out["norm_head"] = global_norm([model.lm_head.weight])
    # with a final norm the logit scale is set by its gain and the head together
    out["norm_final_gain"] = global_norm([model.norm.weight]) if hasattr(model.norm, "weight") else None
    vs = [opt.state[q]["exp_avg_sq"].flatten() for gr in opt.param_groups for q in gr["params"] if q in opt.state]
    if vs:
        v = torch.cat(vs)
        out["adam_v_min"] = v.min().item()
        out["adam_v_median"] = v.median().item()
        out["adam_v_frac_sqrt_below_eps"] = (v.sqrt() < args.eps).float().mean().item()
    return out


WRONG_COLUMNS = ["a", "b", "answer", "pred", "transpose_seen", "margin"]


@torch.no_grad()
def wrong_pairs(model, data, args, update):
    """Every train and test pair the model gets wrong, in eval mode. transpose_seen is None for train pairs."""
    model.eval()
    out = {"update": update, "columns": WRONG_COLUMNS}
    for split in ("train", "test"):
        x, y = data[split + "_x"], data[split + "_y"]
        z = answer_logits(model, x, args)
        pred, m = z.argmax(1), margins(z.double(), y)
        bad = (pred != y).nonzero().squeeze(1)
        xa, xb = operands_of(x, args)
        seen = data["test_seen"][bad].tolist() if split == "test" else [None] * len(bad)
        out[split] = [[a, b, t, q, s, round(mm, 4)] for a, b, t, q, s, mm in
                      zip(xa[bad].tolist(), xb[bad].tolist(), y[bad].tolist(), pred[bad].tolist(), seen,
                          m[bad].tolist())]
    return out


class Spikes:
    """A spike starts when the training loss exceeds factor x max(running minimum, floor) and ends when it
    falls below sqrt(factor) x that base, so a slow tail near the line is not counted twice. The floor
    matters: fp32 cross-entropy is exactly 0 once every margin is past the cliff."""

    def __init__(self, factor, floor):
        self.factor, self.floor, self.min, self.count, self.active = factor, floor, math.inf, 0, False

    def feed(self, update, loss):
        event, base = None, max(self.min, self.floor)
        if not self.active and loss > self.factor * base:
            self.active, self.count, event = True, self.count + 1, [update, loss, self.min]
        elif self.active and loss <= math.sqrt(self.factor) * base:
            self.active = False
        self.min = min(self.min, loss)
        return event


def save_atomic(obj, path):
    """Write via a temporary file, so a killed run never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(obj, str):
        tmp.write_text(obj)
    else:
        torch.save(obj, tmp)
    tmp.replace(path)


def summarize(history, updates, level=0.99):
    """Grokking read from the whole trace, not the last line: an eval can land inside a loss spike.
    history: (update, unseen acc) or (update, unseen acc, unseen off-diagonal acc). "unseen" includes the
    diagonal pairs, which can lag far behind (with --diagonals test, by over 10,000 updates), so the
    off-diagonal timing is reported as well."""
    out = {}
    for col, name in ((1, "unseen"), (2, "unseen_offdiag")):
        ok = [(h[0], h[col]) for h in history if len(h) > col and h[col] is not None]
        if col == 2 and not any(len(h) > 2 for h in history):
            continue
        tail = [a for u, a in ok if u > 0.9 * updates]
        out[f"first_update_{name}_ge_0.99"] = next((u for u, a in ok if a >= level), None)
        out[f"frac_last10pct_evals_{name}_ge_0.99"] = sum(a >= level for a in tail) / len(tail) if tail else None
    return out


def git(*cmd):
    try:
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")  # true fp32; no TF32
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old = [f for f in out.iterdir() if f.is_file() and (f.suffix in (".jsonl", ".json", ".pt", ".tmp"))]
    if old:  # a restarted run never mixes with the files of an earlier one
        aside = out / time.strftime("old_%Y%m%d-%H%M%S")
        aside.mkdir()
        for f in old:
            f.replace(aside / f.name)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    train, test, seen = make_split(args.p, args.frac, split_seed, args.pairs, args.diagonals, args.holdout_line)
    (xtr, ytr), (xte, yte) = (encode(ids, args.p, args.operands, args.task) for ids in (train, test))
    missing = sorted(set(range(args.p)) - set(ytr.tolist()))
    if missing and not args.allow_missing_residues:
        raise ValueError(f"residues never seen as a train answer: {missing}")
    data = {k: v.to(device) for k, v in dict(train_x=xtr, train_y=ytr, test_x=xte, test_y=yte, test_seen=seen).items()}

    sampler = Batches(len(train), args.batch, args.seed) if args.batch else None  # validates --batch early
    torch.manual_seed(args.seed)
    model = build_model(args).to(device)
    opt = build_optimizer(model, args)
    exempt_names = [n for n, _ in model.named_parameters() if not decays(n, args.wd_scope)]
    record = {
        "args": vars(args), "argv": sys.argv, "commit": git("rev-parse", "HEAD"),
        "dirty": None if git("status", "--porcelain") is None else bool(git("status", "--porcelain")),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "seed": args.seed, "split_seed": split_seed, "threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "model_config": dataclasses.asdict(model.config), "init_std": 0.02,  # model.py _init_weights
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "reproducibility": "CPU runs with pinned threads repeat exactly (T6). CUDA runs are not bitwise "
                           "deterministic; same-seed GPU reruns can drift apart, most visibly after a loss spike.",
        "n_train": len(train), "n_test": len(test), "n_test_transpose_seen": int(seen.sum()),
        "n_test_unseen": int((~seen).sum()), "n_test_diagonal": int(((test // args.p) == (test % args.p)).sum()),
        "n_train_diagonal": int(((train // args.p) == (train % args.p)).sum()),
        "n_test_line": None if args.holdout_line is None else int(line_mask(test // args.p, test % args.p, args.p, args.holdout_line).sum()),
        # the same group as the test_*_unseen_offdiag metrics: unseen, not diagonal, not on the held-out line
        "n_test_unseen_offdiag": int((~seen & ((test // args.p) != (test % args.p)) & ~(
            line_mask(test // args.p, test % args.p, args.p, args.holdout_line) if args.holdout_line is not None
            else torch.zeros(len(test), dtype=torch.bool))).sum()),
        "missing_residues": missing, "examples_per_update": args.batch or len(train), "updates": args.updates,
        "full_batch": not args.batch,
        "n_params": sum(q.numel() for q in model.parameters()),
        "n_decayed": sum(q.numel() for q in opt.param_groups[0]["params"]), "exempt_params": exempt_names,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "running",
    }
    save_atomic(json.dumps(record, indent=1), out / "run_record.json")

    spikes, buf = Spikes(args.spike_factor, args.spike_floor), torch.zeros(args.eval_every, device=device)
    t0 = time.perf_counter()
    train_s, mark, last, history = 0.0, t0, {}, []
    with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:

        def log(update, extra):
            nonlocal mark, last
            t_eval = time.perf_counter()
            last = {"update": update, **extra, **evaluate(model, opt, data, args),
                    "train_s": train_s, "elapsed_s": time.perf_counter() - t0}
            line = json.dumps(last)
            f.write(line + "\n")
            f.flush()
            print(line, flush=True)
            mark = time.perf_counter()
            return t_eval

        try:
            log(0, {"lr": 0.0})
            for u in range(args.updates):
                lr = lr_at(u, args)
                at_log = (u + 1) % args.eval_every == 0 or u + 1 == args.updates
                if sampler is None:
                    bx, by = data["train_x"], data["train_y"]
                else:
                    idx = sampler.next().to(device)
                    bx, by = data["train_x"][idx], data["train_y"][idx]
                loss, stats = train_step(model, opt, bx, by, args, lr, at_log)
                buf[u % args.eval_every] = loss.float()
                if at_log:
                    losses = buf[: u % args.eval_every + 1].tolist()  # syncs the device
                    train_s += time.perf_counter() - mark
                    first = u + 1 - len(losses)
                    events = [e for i, v in enumerate(losses) if (e := spikes.feed(first + i, v))]
                    log(u + 1, {"lr": lr, "train_obj_pre": losses[-1], "train_obj_pre_max": max(losses),
                                **stats, "spikes": spikes.count, "spike_events": events,
                                "nonfinite": sum(not math.isfinite(v) for v in losses),
                                "train_obj_trace": losses})  # every update's loss since the last line
                    history.append((u + 1, last.get("test_acc_unseen"), last.get("test_acc_unseen_offdiag")))
                    if not all(math.isfinite(v) for v in losses):
                        raise FloatingPointError(f"non-finite training loss by update {u + 1}")
                if args.ckpt_every and (u + 1) % args.ckpt_every == 0 and u + 1 < args.updates:
                    save_atomic({"model": model.state_dict(), "update": u + 1}, out / f"ckpt_{u + 1:06d}.pt")
            wrong = wrong_pairs(model, data, args, args.updates)
            save_atomic(json.dumps(wrong), out / "final_errors.json")
            record["n_wrong_final"] = {"train": len(wrong["train"]), "test": len(wrong["test"]),
                                       "test_unseen": sum(not r[4] for r in wrong["test"]),
                                       "test_diagonal": sum(r[0] == r[1] for r in wrong["test"])}
            save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "update": args.updates,
                         "args": vars(args)}, out / "final.pt")
            record["status"] = "finished"
        except BaseException as e:
            record["status"] = f"failed: {type(e).__name__}: {e}"
            raise
        finally:
            record.update(ended=time.strftime("%Y-%m-%dT%H:%M:%S"), train_s=train_s,
                          elapsed_s=time.perf_counter() - t0, final=last, **summarize(history, args.updates))
            save_atomic(json.dumps(record, indent=1), out / "run_record.json")
    return last


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--pairs", choices=["ordered", "paired"], default="ordered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=None, help="default: --seed")
    ap.add_argument("--diagonals", choices=["split", "train", "test"], default="split",
                    help="where the pairs (a, a) go: as the seeded split puts them, or all in train or all in test")
    ap.add_argument("--holdout_line", type=int, default=None,
                    help="move the lines b = a +- k (mod p) to test: positive control for --diagonals test")
    ap.add_argument("--operands", choices=["shared", "separate"], default="shared",
                    help="separate: the second operand gets its own p tokens (vocabulary 2p + 2)")
    ap.add_argument("--task", choices=sorted(TASKS), default="add",
                    help="add: (a + b) mod p; add2b: (a + 2b) mod p, not commutative")
    ap.add_argument("--allow_missing_residues", action="store_true", help="for small smoke sets only")
    ap.add_argument("--untie", action="store_true", help="separate output head (default: tied to the embedding)")
    ap.add_argument("--no_final_norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--wd_scope", choices=["all", "embeddings", "repo"], default="all")
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--warmup", type=int, default=10, help="linear warmup, in updates")
    ap.add_argument("--lr_schedule", choices=["constant", "wsd"], default="constant",
                    help="wsd: train.py's default (max LR until 40%% of the run, then linear to --min_lr)")
    ap.add_argument("--min_lr", type=float, default=None, help="--lr_schedule wsd only; default 1.5e-4 (train.py's)")
    ap.add_argument("--loss", choices=["ce", "margin", "ce_rounded"], default="ce")
    ap.add_argument("--round_dtype", choices=sorted(ROUND_DTYPES), default=None,
                    help="--loss ce_rounded only: the dtype the softmax sum is rounded to "
                         "(default bfloat16, cliff 5.55; float16, cliff 7.6)")
    ap.add_argument("--loss_fp64", action="store_true",
                    help="cast logits to fp64 before the loss (fp32 cliff moves to 36.7; ce_rounded keeps its cliff)")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip before each step; 0 = none (train.py uses 1.0)")
    ap.add_argument("--batch", type=int, default=0, help="examples per update, epoch-shuffled; 0 = full batch")
    ap.add_argument("--bf16", choices=["off", "autocast"], default="off",
                    help="autocast: model and loss under bf16 autocast, as train.py runs on CUDA (no bf16 cliff; "
                         "use --loss ce_rounded for that)")
    ap.add_argument("--updates", type=int, default=100_000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=25_000, help="weights only; 0 = final checkpoint only")
    ap.add_argument("--spike_factor", type=float, default=10.0)
    ap.add_argument("--spike_floor", type=float, default=1e-3, help="losses below this never count as a spike's base")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    args = ap.parse_args(argv)
    if args.round_dtype is not None and args.loss != "ce_rounded":
        ap.error("--round_dtype applies only to --loss ce_rounded")
    if args.loss == "ce_rounded" and args.round_dtype is None:
        args.round_dtype = "bfloat16"
    if args.min_lr is not None and args.lr_schedule != "wsd":
        ap.error("--min_lr applies only to --lr_schedule wsd")
    if args.lr_schedule == "wsd" and args.min_lr is None:
        args.min_lr = 1.5e-4
    if args.lr_schedule == "wsd" and args.min_lr > args.lr:
        ap.error(f"--min_lr {args.min_lr} is above --lr {args.lr}: wsd would decay upward")
    if args.lr_schedule == "wsd" and args.warmup >= WSD_STABLE_END * args.updates:
        ap.error("--lr_schedule wsd needs --warmup below 40%% of --updates "
                 "(train.py would jump from the ramp into mid-decay, #173)")
    return args


if __name__ == "__main__":
    run(parse_args())
