"""Standalone grokking harness: a + b mod p on BaselineTransformer, full batch, fp32.

PR 2 of docs/research/grokking-fix-plan-2026-10-28.md (arm A1-H of #249). It builds the repo's
model directly and uses nothing from data.py or train.py.

- Task: the enumerated a + b mod p table, one token per number. Input [a, op, b, =]; the
  answer is predicted at the last position. Vocabulary p + 2.
- Split: seeded, at --frac. `ordered` splits the p^2 ordered pairs (Power 2022); `paired`
  keeps (a, b) and (b, a) on the same side. Each test pair is tagged "transpose seen" when
  (b, a) is in train; a diagonal pair (a, a) is its own transpose, so it is never seen.
- Model: d 128, 2 layers, 4 heads, dropout 0, head tied to the embedding (ReLSM's model and the
  owner's primary arm). --untie gives a separate head (Power 2022, Nanda 2023); --no_final_norm
  removes the final RMSNorm. The head is set here explicitly, not taken from the model defaults.
- Optimizer: AdamW, betas (0.9, --beta2), --eps, weight decay --wd on the --wd_scope set.
  Linear warmup over --warmup updates, then constant LR. No early stopping. By default: no
  clipping, full batch, fp32. To step toward train.py's setup one setting at a time: --clip,
  --batch (epoch-shuffled minibatches), --bf16 autocast (model and loss under bf16 autocast, as
  train.py runs on CUDA; autocast keeps the cross-entropy in fp32).
- Loss: cross-entropy at the answer position; --loss margin is Barron et al. Eq. 6,
  softplus(-m) with m = z_t - logsumexp_{j != t} z_j; --loss_fp64 casts logits first.
  --loss ce_rounded is cross-entropy with the softmax sum rounded once to --round_dtype: the
  low-precision cliff (bf16: margin 8 ln 2 = 5.55) on any device. It copies what PyTorch's CPU
  bf16 log_softmax kernel does; the CUDA kernel keeps the sum in fp32 and has no such cliff.

Outputs in --out: metrics.jsonl (append-only, one line per eval, also printed to stdout),
run_record.json, ckpt_<update>.pt (weights) and final.pt (weights + optimizer). Files from an
earlier run in the same --out are moved to old_<time>/ first.

Reading metrics.jsonl: train_loss, test_loss and the margins are fp64 readouts of the logits
after update `update` (bf16-rounded logits under --bf16); train_obj_pre is the training
objective before that update. Under --batch, train_obj_* and the spike count follow single
minibatches, so they are noisier and not comparable with full-batch arms. Under --loss
ce_rounded, train_obj_* is quantized (per-example steps of log(1 + k 2^-7) for bf16) and reads
exactly 0 once every train example is past the cliff, while the fp64 train_loss stays nonzero.
Evals run in eval mode, so with --dropout > 0 the margin and zero-gradient readouts are of the
model without dropout, not of the gradient training saw.

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


def make_split(p, frac, seed, pairs="ordered"):
    """Return (train ids, test ids, seen) over pair ids a*p + b. seen[i]: (b, a) of test pair i is in train."""
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
    train, test = in_train.nonzero().squeeze(1), (~in_train).nonzero().squeeze(1)
    seen = in_train[(test % p) * p + test // p]
    return train, test, seen


def encode(ids, p):
    """Pair ids -> inputs [a, op, b, =] (op = p, '=' = p + 1) and answers (a + b) mod p."""
    a, b = ids // p, ids % p
    x = torch.stack([a, torch.full_like(a, p), b, torch.full_like(a, p + 1)], dim=1)
    return x, (a + b) % p


def build_model(args):
    cfg = TransformerConfig(vocab_size=args.p + 2, max_seq_len=4, d_model=128, n_layers=2, n_heads=4,
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
    with bf16_ctx(args, x.device):
        return model(x)[0][:, -1, :]


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


@torch.no_grad()
def evaluate(model, opt, data, args):
    model.eval()
    out = {}
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
        else:
            m = margins(z.double(), y)
            out["margin_min"] = m.min().item()
            out["margin_mean"] = m.mean().item()
            out["margin_p5"] = torch.quantile(m, 0.05).item()
            out["margin_frac_cliff"] = (m > CLIFF_FP32).float().mean().item()
            out["margin_frac_bf16_cliff"] = (m > CLIFF_BF16).float().mean().item()
            with torch.enable_grad():  # the target gradient the arm's own loss produces
                leaf = z.detach().requires_grad_()
                (g,) = torch.autograd.grad(loss_of(leaf, y, args), leaf)
            out["zero_tgt_grad_frac"] = (g.gather(1, y[:, None]) == 0).float().mean().item()
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
    """Grokking read from the whole trace, not the last line: an eval can land inside a loss spike."""
    ok = [(u, a) for u, a in history if a is not None]
    first = next((u for u, a in ok if a >= level), None)
    tail = [a for u, a in ok if u > 0.9 * updates]
    return {"first_update_unseen_ge_0.99": first,
            "frac_last10pct_evals_unseen_ge_0.99": sum(a >= level for a in tail) / len(tail) if tail else None}


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
    train, test, seen = make_split(args.p, args.frac, split_seed, args.pairs)
    (xtr, ytr), (xte, yte) = encode(train, args.p), encode(test, args.p)
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
                lr = args.lr * min(1.0, (u + 1) / args.warmup) if args.warmup > 0 else args.lr
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
                    history.append((u + 1, last.get("test_acc_unseen")))
                    if not all(math.isfinite(v) for v in losses):
                        raise FloatingPointError(f"non-finite training loss by update {u + 1}")
                if args.ckpt_every and (u + 1) % args.ckpt_every == 0 and u + 1 < args.updates:
                    save_atomic({"model": model.state_dict(), "update": u + 1}, out / f"ckpt_{u + 1:06d}.pt")
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
    return args


if __name__ == "__main__":
    run(parse_args())
