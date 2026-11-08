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

Several tasks (--tasks, a prototype; design in docs/research/grok-multitask-design-2026-11-02.md):
- --tasks add,sub,... trains one model on several binary operations, input [a, op_t, b, =] with one op token
  per task. Tasks: add, add2b, sub, mul (nonzero operands: the group (Z/p)*), div (b != 0), sq (a^2 + b^2),
  sqab (a^2 + ab + b^2), cubeab (a^3 + ab), all mod p; gcd (of 1..p-1); s5 (composition in S5, 120 tokens of
  its own); add_dup (add under a second op token: the null partner for interference).
  Each task is scored over its own answer classes (its answer block, then every op token and '='). Under
  --answers shared, tasks on the same alphabet answer with its operand tokens; under per_task, each task gets
  output-only answer tokens. '--tasks add' is the same run as '--task add', bit for bit on CPU (tested).
- Splits come from one shared shuffle of the pairs (--task_split aligned, the default), so a pair held out of
  one task is held out of the others; independent gives each task its own. Per task: train fraction
  (--task_frac), loss weight (--task_weight, --task_weighting) and active range of updates (--task_start,
  --task_stop: sequential training and forgetting tests; --allow_idle permits updates with no task, which are
  weight decay only). Under --batch, minibatches are drawn from the active tasks' pooled examples, the order
  restarts when the active set changes, and each task's term is scaled by its share of the batch so the
  minibatch loss estimates the full-batch objective (multi_loss). --wd_until u turns weight decay off from
  update u (any run): the control that tells disuse plus decay from overwriting.
- Diagnostics, every eval: emb_fourier_* and emb_logfourier_* (any run with number tokens; _b for the second
  operand's block) give the number embedding's power by frequency along a and along log a, with the top-5
  share and frequencies: which structure (add-like or mul-like) the table holds, and whether a stopped task's
  frequencies shrink or get replaced. With two or more tasks, grad_cos_all@i|j, grad_cos_trunk@i|j and
  grad_norm_task@t measure conflict between tasks directly. --ckpt_at keeps a few checkpoints for analysis.
- Under --tasks the train-mode dropout readout runs per task, so its masks are not the next update's (that
  holds only for a single task under full batch).
- Every evaluate() readout is logged per task with the suffix @task, and pooled (example-weighted) for the
  accuracies and losses; the norms and Adam readouts are logged once. final_errors.json gains a task column,
  and run_record.json a "tasks" list and per-task grokking times (first_update_*@task) plus the forgetting
  readouts min_test_acc_after_first_ge_0.99@task and min_test_acc_last10pct@task.

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
import itertools
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


def make_split(p, frac, seed, pairs="ordered", diagonals="split", holdout_line=None, valid=None):
    """Return (train ids, test ids, seen) over pair ids a*p + b. seen[i]: (b, a) of test pair i is in train.
    diagonals: "split" leaves the pairs (a, a) where the seeded split put them; "train" or "test" then moves
    all p of them to that side (at frac 0.5 the train set grows or shrinks by about p / 2).
    holdout_line k: also move every pair on the lines b = a + k and b = a - k (mod p) to test, 2p pairs that
    are each other's transposes. It is the positive control for --diagonals test: a held-out line other than
    the diagonal, so a weak diagonal can be told apart from a weak held-out line.
    valid: optional bool mask over the p*p ids, the task's domain (b != 0 for a / b). Only valid pairs are
    split, and frac is a fraction of them. The shuffle is always over all p*p pairs (or all pair groups) and
    the domain is applied afterwards, so two tasks with different domains and the same seed still split
    almost the same pairs (--task_split aligned for add and mul). With every pair valid the split is the same
    as before."""
    g = torch.Generator().manual_seed(seed)
    n = p * p
    ok = torch.ones(n, dtype=torch.bool) if valid is None else valid
    n_train = int(round(frac * int(ok.sum())))
    in_train = torch.zeros(n, dtype=torch.bool)
    if pairs == "ordered":
        perm = torch.randperm(n, generator=g)
        in_train[perm[ok[perm]][:n_train]] = True  # the first n_train domain pairs in the shared shuffle
    elif pairs == "paired":
        groups = [[a * p + b] if a == b else [a * p + b, b * p + a] for a in range(p) for b in range(a, p)]
        taken = 0
        for k in torch.randperm(len(groups), generator=g).tolist():
            kept = groups[k] if valid is None else [i for i in groups[k] if ok[i]]
            if kept and taken + len(kept) <= n_train:
                in_train[kept] = True
                taken += len(kept)
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
    in_train &= ok  # --diagonals train may have marked a pair outside the domain
    train, test = in_train.nonzero().squeeze(1), (~in_train & ok).nonzero().squeeze(1)
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


# --- Several tasks in one model (--tasks) -------------------------------------------------------------
# Each task is a binary operation on an alphabet ("num": the p residues; "s5": the 120 permutations of 5
# items), encoded [a, op_t, b, =] with one op token per task. Answers are indices into the task's alphabet.

def _inverses(p):
    return torch.tensor([pow(i, p - 2, p) if i else 0 for i in range(p)])


_S5 = []


def s5_table():
    """120 x 120 composition table: entry [f, g] is the index of f o g (apply g first), permutations of
    range(5) in itertools order (index 0 is the identity)."""
    if not _S5:
        perms = list(itertools.permutations(range(5)))
        index = {q: i for i, q in enumerate(perms)}
        _S5.append(torch.tensor([[index[tuple(f[g[k]] for k in range(5))] for g in perms] for f in perms]))
    return _S5[0]


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    alphabet: str  # "num" or "s5"
    commutative: bool
    answer: object  # (a, b, p) -> answer index in the alphabet
    domain: object = None  # (a, b, p) -> bool mask of the pairs the task is defined on; None: all pairs
    needs_prime: bool = False


TASK_SPECS = {
    "add": TaskSpec("num", True, lambda a, b, p: (a + b) % p),
    # the same function under a second op token: the null partner for interference ("sharing costs this much
    # even when the two tasks agree"). With --task_split aligned it gets add's split.
    "add_dup": TaskSpec("num", True, lambda a, b, p: (a + b) % p),
    "add2b": TaskSpec("num", False, lambda a, b, p: (a + 2 * b) % p),
    "sub": TaskSpec("num", False, lambda a, b, p: (a - b) % p),
    # nonzero operands only: the cyclic group (Z/p)*, so 0 is never an answer and no answer class dominates
    "mul": TaskSpec("num", True, lambda a, b, p: (a * b) % p, lambda a, b, p: (a != 0) & (b != 0), True),
    "div": TaskSpec("num", False, lambda a, b, p: (a * _inverses(p)[b]) % p, lambda a, b, p: b != 0, True),
    "sq": TaskSpec("num", True, lambda a, b, p: (a * a + b * b) % p),
    "sqab": TaskSpec("num", True, lambda a, b, p: (a * a + a * b + b * b) % p),
    "cubeab": TaskSpec("num", False, lambda a, b, p: (a * a * a + a * b) % p),
    "s5": TaskSpec("s5", False, lambda a, b, p: s5_table()[a, b]),
    # not modular: gcd of 1..p-1. Large values occur almost only on the diagonal (gcd(a, a) = a), so the
    # start-up check needs --diagonals train or --allow_missing_residues
    "gcd": TaskSpec("num", True, lambda a, b, p: torch.gcd(a, b), lambda a, b, p: (a != 0) & (b != 0)),
}
SPLIT_SEED_STRIDE = 7919  # --task_split independent: task i splits with split_seed + i * this


def alphabet_size(alphabet, p):
    return p if alphabet == "num" else 120


def multi_task(args):
    return bool(getattr(args, "tasks", None))


@dataclasses.dataclass
class Layout:
    """Token ids under --tasks. Blocks in order: each alphabet's operand tokens ("num" first), the second
    operand's tokens under --operands separate, one answer block per task under --answers per_task, one op
    token per task, then '='. For a single add task this is the same vocabulary as --task add."""
    first: dict
    second: dict
    answer: list
    op: list
    eq: int
    vocab: int
    cols: list  # per task: the answer classes, its answer block then every op token and '='


def make_layout(args):
    names, p = args.tasks, args.p
    alphas = [a for a in ("num", "s5") if any(TASK_SPECS[n].alphabet == a for n in names)]
    off, first = 0, {}
    for a in alphas:
        first[a], off = off, off + alphabet_size(a, p)
    second = dict(first)
    if separate_operands(args):
        for a in alphas:
            second[a], off = off, off + alphabet_size(a, p)
    answer = []
    for n in names:
        a = TASK_SPECS[n].alphabet
        if args.answers == "per_task":
            answer.append(off)
            off += alphabet_size(a, p)
        else:
            answer.append(first[a])
    op = list(range(off, off + len(names)))
    eq = off + len(names)
    cols = [list(range(s, s + alphabet_size(TASK_SPECS[n].alphabet, p))) + op + [eq] for n, s in zip(names, answer)]
    return Layout(first, second, answer, op, eq, eq + 1, cols)


def encode_task(ids, i, args, lay):
    """Pair ids a*n + b of task i -> inputs [a, op_i, b, =] and answers (indices into the task's answer classes)."""
    name = args.tasks[i]
    spec = TASK_SPECS[name]
    n = alphabet_size(spec.alphabet, args.p)
    a, b = ids // n, ids % n
    x = torch.stack([lay.first[spec.alphabet] + a, torch.full_like(a, lay.op[i]),
                     lay.second[spec.alphabet] + b, torch.full_like(a, lay.eq)], dim=1)
    return x, spec.answer(a, b, args.p)


def vocab_size(args):
    if multi_task(args):
        return make_layout(args).vocab
    return 2 * args.p + 2 if separate_operands(args) else args.p + 2


def operands_of(x, args):
    """The numbers (a, b) behind encoded inputs x."""
    off = getattr(args, "operand_offsets", None)  # a task's view under --tasks
    if off is not None:
        return x[:, 0] - off[0], x[:, 2] - off[1]
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
    cols = getattr(args, "answer_cols", None)  # a task's view under --tasks
    if cols is not None:
        return z[:, cols]
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


def train_step(model, opt, x, y, args, lr, measure=False, loss_fn=None):
    """One update on (x, y), the full train set or a minibatch. Returns (loss before the update, measured stats or None).
    loss_fn: a no-argument function that returns the loss instead (the --tasks objective)."""
    model.train()
    for group in opt.param_groups:
        group["lr"] = lr
    loss = loss_fn() if loss_fn is not None else loss_of(answer_logits(model, x, args), y, args)
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
def evaluate(model, opt, data, args, model_readouts=True):
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
    if model_readouts:
        out.update(model_stats(model, opt, args))
    return out


def model_stats(model, opt, args):
    """Model-wide readouts: norms and Adam's second moments (logged once per eval, whatever the tasks)."""
    out = {}
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
    for suffix, start in number_blocks(args):
        out.update({k + suffix: v for k, v in embedding_spectra(model.tok_emb.weight, args.p, start).items()})
    return out


# --- Diagnostics: what the number embedding encodes, and whether tasks pull the same way ----------------

def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def primitive_root(p):
    """Smallest generator of (Z/p)*, p prime (5 for p = 97; 1 for p = 2, where (Z/2)* = {1})."""
    if p == 2:
        return 1
    m, qs, d = p - 1, set(), 2
    while d * d <= m:
        while m % d == 0:
            qs.add(d)
            m //= d
        d += 1
    if m > 1:
        qs.add(m)
    return next(g for g in range(2, p) if all(pow(g, (p - 1) // q, p) != 1 for q in qs))


def number_blocks(args):
    """(key suffix, first row) of the number-token blocks in the table: the first operand's ("") and, under
    --operands separate, the second operand's ("_b"). None when no task uses numbers (s5 alone)."""
    if multi_task(args):
        if not any(TASK_SPECS[n].alphabet == "num" for n in args.tasks):
            return []
        lay = make_layout(args)
        first, second = lay.first["num"], lay.second["num"]
    else:
        first, second = 0, (args.p if separate_operands(args) else 0)
    return [("", first)] + ([("_b", second)] if second != first else [])


TOP_FREQS = 5


def _fold_power(rows):
    """Power per frequency k = 1..n//2 of rows (n x d) along n, with k and n - k folded together (DC dropped)."""
    pw = torch.fft.fft(rows.double(), dim=0).abs().pow(2).sum(1).tolist()  # one device sync
    n = rows.shape[0]
    return [float(pw[k] + (pw[n - k] if n - k != k else 0.0)) for k in range(1, n // 2 + 1)]


def embedding_spectra(weight, p, start):
    """How the p number embeddings (rows start..start+p-1) vary with the number, in two bases:
    fourier: along a = 0..p-1. A grokked a + b (Nanda 2023) puts nearly all of it in a few frequencies.
    logfourier: along j, for a = g^j (g a primitive root, a != 0; p prime only). (Z/p)* is cyclic, so a
      Fourier solution for a * b lives here instead.
    Per basis: the power at each frequency (spectrum, k = 1..), the share in the top 5 (top5_frac; a random
    table gives about 5/48 at p = 97) and those 5 frequencies (top5, largest first). Comparing the two bases
    over time shows which structure a task built, whether two tasks compete for the same table, and, after a
    task stops training, whether its frequencies shrink together (decay) or get replaced (overwriting)."""
    rows = weight.detach()[start:start + p]
    bases = {"fourier": rows}
    if is_prime(p) and p > 2:  # (Z/2)* has one element, so there is no log-Fourier spectrum
        g = primitive_root(p)
        bases["logfourier"] = rows[torch.tensor([pow(g, j, p) for j in range(p - 1)], device=rows.device)]
    out = {}
    for name, r in bases.items():
        spec = _fold_power(r)
        top = sorted(range(len(spec)), key=lambda k: -spec[k])[:TOP_FREQS]
        total = sum(spec)
        out[f"emb_{name}_spectrum"] = spec
        out[f"emb_{name}_top5"] = [k + 1 for k in top]
        out[f"emb_{name}_top5_frac"] = sum(spec[k] for k in top) / total if total > 0 else None
    return out


def task_grad_cosines(model, views, args):
    """For every pair of tasks: the cosine between their full-train-set gradients (the arm's own loss, eval
    mode, so no dropout), over all parameters (grad_cos_all@i|j) and over the trunk without the embedding
    table and head (grad_cos_trunk@i|j); and each task's gradient norm (grad_norm_task@t). Negative cosine is
    direct conflict; near zero, the tasks use separate directions. A task past the fp32 cliff has a near-zero
    gradient (see grad_norm_task), and its cosine then says little. Gradients come from autograd.grad, so the
    .grad buffers, the optimizer and the RNG are untouched: training is unchanged."""
    if len(views) < 2:
        return {}
    named = list(model.named_parameters())
    params = [q for _, q in named]
    trunk = [k for k, (n, _) in enumerate(named) if "tok_emb" not in n and "lm_head" not in n]
    model.eval()
    grads = []
    with torch.enable_grad():
        for v in views:
            loss = loss_of(answer_logits(model, v.data["train_x"], v.args), v.data["train_y"], args)
            gs = torch.autograd.grad(loss, params, allow_unused=True)
            grads.append([torch.zeros_like(q) if g is None else g.detach() for g, q in zip(gs, params)])
    out = {}
    for v, g in zip(views, grads):
        out[f"grad_norm_task@{v.name}"] = global_norm(g)
    for i, j in itertools.combinations(range(len(views)), 2):
        for scope, idx in (("all", range(len(params))), ("trunk", trunk)):
            a = torch.cat([grads[i][k].flatten().double() for k in idx])
            b = torch.cat([grads[j][k].flatten().double() for k in idx])
            den = float(a.norm() * b.norm())
            out[f"grad_cos_{scope}@{views[i].name}|{views[j].name}"] = float(a @ b) / den if den > 0 else None
    return out


WRONG_COLUMNS =["a", "b", "answer", "pred", "transpose_seen", "margin"]


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


@dataclasses.dataclass
class TaskView:
    """One task under --tasks: its own split, data and an args copy that evaluate() and wrong_pairs() read."""
    name: str
    index: int
    args: argparse.Namespace
    data: dict
    lo: int  # its train examples are rows lo..hi-1 of the pooled train set
    hi: int
    record: dict


def setup_tasks(args, device):
    """Splits and encodes every task. Returns (pooled data, views, layout). Task i splits with split_seed
    (--task_split aligned, the default: one shared shuffle, so tasks agree on which pairs are test pairs, up to
    a few pairs at the edge when domains differ) or split_seed + i * SPLIT_SEED_STRIDE (independent). Task 0
    always gets the same split as a single-task run."""
    lay = make_layout(args)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    views, xs, ys = [], [], []
    for i, name in enumerate(args.tasks):
        spec = TASK_SPECS[name]
        n = alphabet_size(spec.alphabet, args.p)
        ids = torch.arange(n * n)
        a, b = ids // n, ids % n
        valid = spec.domain(a, b, args.p) if spec.domain is not None else None
        seed_i = split_seed + (0 if args.task_split == "aligned" else SPLIT_SEED_STRIDE * i)
        frac = args.task_frac.get(name, args.frac)
        train, test, seen = make_split(n, frac, seed_i, args.pairs, args.diagonals, None, valid)
        (xtr, ytr), (xte, yte) = (encode_task(t, i, args, lay) for t in (train, test))
        dom = slice(None) if valid is None else valid
        missing = sorted(set(spec.answer(a[dom], b[dom], args.p).tolist()) - set(ytr.tolist()))
        if missing and not args.allow_missing_residues:
            raise ValueError(f"task {name}: answers never seen in train: {missing}")
        diag_te, diag_tr = (test // n) == (test % n), (train // n) == (train % n)
        lo = sum(len(x) for x in xs)
        xs.append(xtr)
        ys.append(ytr)
        view_args = argparse.Namespace(**{**vars(args), "p": n, "holdout_line": None,
                                          "answer_cols": torch.tensor(lay.cols[i], device=device),
                                          "operand_offsets": (lay.first[spec.alphabet], lay.second[spec.alphabet])})
        data = {k: v.to(device) for k, v in dict(train_x=xtr, train_y=ytr, test_x=xte, test_y=yte, test_seen=seen).items()}
        record = {"task": name, "alphabet": spec.alphabet, "commutative": spec.commutative, "op_token": lay.op[i],
                  "answer_tokens": [lay.answer[i], lay.answer[i] + n - 1], "n_classes": len(lay.cols[i]),
                  "frac": frac, "split_seed": seed_i, "n_train": len(train), "n_test": len(test),
                  "n_test_transpose_seen": int(seen.sum()), "n_test_unseen": int((~seen).sum()),
                  "n_test_diagonal": int(diag_te.sum()), "n_train_diagonal": int(diag_tr.sum()),
                  "n_test_unseen_offdiag": int((~seen & ~diag_te).sum()), "missing_residues": missing,
                  "start": args.task_start.get(name, 0), "stop": args.task_stop.get(name),
                  "weight": args.task_weight.get(name, 1.0)}
        views.append(TaskView(name, i, view_args, data, lo, lo + len(train), record))
    pooled = {"train_x": torch.cat(xs).to(device), "train_y": torch.cat(ys).to(device)}
    return pooled, views, lay


class TaskSchedule:
    """Which tasks train at update u (--task_start / --task_stop), and the batch: every active task's train
    examples (full batch) or epoch-shuffled minibatches over them. The minibatch order restarts whenever the
    active set changes; while it does not change (always, without a schedule) it is the single-task order.
    With --allow_idle an update may have no active task: it then takes a zero-gradient step, so AdamW applies
    its weight decay and the dying Adam momentum only (the pure-decay control for forgetting)."""

    def __init__(self, views, args, device):
        self.views, self.args, self.device = views, args, device
        self.start = [v.record["start"] for v in views]
        self.stop = [v.record["stop"] if v.record["stop"] is not None else math.inf for v in views]
        self.task_of_row = torch.cat([torch.full((v.hi - v.lo,), v.index) for v in views])
        self.key, self.phase = None, 0

    def active(self, u):
        return tuple(i for i in range(len(self.views)) if self.start[i] <= u < self.stop[i])

    def boundaries(self, updates):
        return sorted(u for u in {0, *self.start, *self.stop} if u < updates)

    def check(self, updates):
        """Every update trains at least one task (unless --allow_idle), and --batch fits every phase."""
        for u in self.boundaries(updates):
            act = self.active(u)
            if not act and not getattr(self.args, "allow_idle", False):
                raise ValueError(f"no task is active at update {u} (--allow_idle permits it)")
            if act and self.args.batch:
                Batches(sum(self.views[i].hi - self.views[i].lo for i in act), self.args.batch, 0)

    def _rebuild(self, act):
        self.key = act
        self.phase += 1
        if not act:  # idle
            self.sampler, self.parts, self.pool_dev, self.everything = None, [], None, False
            return
        self.pool = torch.cat([torch.arange(self.views[i].lo, self.views[i].hi) for i in act])
        self.everything = len(act) == len(self.views)
        self.sampler = Batches(len(self.pool), self.args.batch, self.args.seed + 1000 * (self.phase - 1)) if self.args.batch else None
        if self.sampler is None:  # full batch: the task parts are fixed until the active set changes
            self.parts = self._parts(self.pool)
            self.pool_dev = None if self.everything else self.pool.to(self.device)

    def _parts(self, rows):
        t = self.task_of_row[rows]
        out = []
        for i in self.key:
            pos = (t == i).nonzero().squeeze(1)
            if len(pos):
                out.append((i, pos.to(self.device), len(pos)))
        return out

    def batch(self, u, data):
        """(x, y, parts) for update u; parts lists (task index, positions in the batch, count)."""
        act = self.active(u)
        if act != self.key:
            self._rebuild(act)
        if not act:
            return None, None, []
        if self.sampler is None:
            if self.pool_dev is None:
                return data["train_x"], data["train_y"], self.parts
            return data["train_x"][self.pool_dev], data["train_y"][self.pool_dev], self.parts
        rows = self.pool[self.sampler.next()]
        idx = rows.to(self.device)
        return data["train_x"][idx], data["train_y"][idx], self._parts(rows)


def multi_loss(model, x, y, parts, views, args, active):
    """The --tasks objective: each task's loss over its own answer classes, combined with fixed weights over the
    active tasks. The full-batch objective is
      equal:    sum_i (u_i / U) mean_i,             U = sum of u over the active tasks
      examples: sum_i u_i n_i mean_i / sum_j u_j n_j  (the mean over all examples when every u is 1)
    with u_i = --task_weight and n_i the task's train size. A minibatch of B rows drawn from the N pooled
    active rows, c_i of them from task i, is scaled so its expected value is that objective: task i's mean
    gets the extra factor c_i N / (B n_i). A task with few rows in a batch therefore counts little, and an
    absent task adds nothing; the weights are never renormalized over the tasks a batch happens to hold.
    Full batch (c_i = n_i, B = N) and a single task give factor 1 exactly. No active task: a zero loss."""
    if not parts:  # --allow_idle: zero gradient everywhere, so the step is weight decay (and old momentum)
        return sum(q.sum() for q in model.parameters()) * 0.0
    with bf16_ctx(args, x.device):
        full = model(x)[0][:, -1, :]
    n = {i: views[i].hi - views[i].lo for i in active}
    u = {i: views[i].record["weight"] for i in active}
    big_n, b = sum(n.values()), len(x)
    total = None
    for i, pos, c in parts:
        if args.task_weighting == "equal":
            coef = (u[i] / sum(u.values())) * ((c * big_n) / (b * n[i]))
        else:
            coef = u[i] * c * big_n / (b * sum(u[j] * n[j] for j in active))
        term = coef * loss_of(full[pos][:, views[i].args.answer_cols], y[pos], args)
        total = term if total is None else total + term
    return total


def _wmean(pairs):
    pairs = [(v, c) for v, c in pairs if v is not None and c]
    return sum(v * c for v, c in pairs) / sum(c for _, c in pairs) if pairs else None


def evaluate_tasks(model, opt, views, args):
    """evaluate() for each task (keys suffixed @task), the model-wide readouts once, and pooled accuracies
    and losses over all tasks (example-weighted, whatever --task_weighting is)."""
    out, per = {}, []
    for v in views:
        r = evaluate(model, opt, v.data, v.args, model_readouts=False)
        per.append(r)
        out.update({f"{k}@{v.name}": val for k, val in r.items()})
    n = lambda key: [v.record[key] for v in views]  # noqa: E731
    pooled = {"train_loss": "n_train", "train_acc": "n_train", "test_loss": "n_test", "test_acc": "n_test",
              "test_acc_unseen": "n_test_unseen", "test_loss_unseen": "n_test_unseen",
              "test_acc_diagonal": "n_test_diagonal", "test_acc_unseen_offdiag": "n_test_unseen_offdiag"}
    for key, count in pooled.items():
        out[key] = _wmean(zip([r[key] for r in per], n(count)))
    out.update(model_stats(model, opt, args))
    out.update(task_grad_cosines(model, views, args))
    return out


def summarize_tasks(task_hist, updates, level=0.99):
    """Per task: summarize() on its unseen / unseen off-diagonal accuracy, the first eval with all its test
    pairs at >= level (for a non-commutative task the transpose is no leak, so this is its grokking time), the
    lowest test accuracy after that, and the lowest over the last 10% of evals. A single low eval after
    grokking can be a loss spike; a low last-10% minimum, or a run of low evals in metrics.jsonl, is forgetting."""
    out = {}
    for name, hist in task_hist.items():
        out.update({f"{k}@{name}": v for k, v in summarize([h[:3] for h in hist], updates, level).items()})
        first = next((u for u, *_, acc in hist if acc is not None and acc >= level), None)
        out[f"first_update_test_ge_{level}@{name}"] = first
        out[f"min_test_acc_after_first_ge_{level}@{name}"] = (
            None if first is None else min(acc for u, *_, acc in hist if u >= first))
        tail = [acc for u, *_, acc in hist if u > 0.9 * updates and acc is not None]
        out[f"min_test_acc_last10pct@{name}"] = min(tail) if tail else None
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
    views, schedule, sampler = None, None, None
    if multi_task(args):
        data, views, lay = setup_tasks(args, device)
        schedule = TaskSchedule(views, args, device)
        schedule.check(args.updates)  # validates --batch and the schedule early
        tot = lambda key: sum(v.record[key] for v in views)  # noqa: E731
        counts = {"n_train": len(data["train_x"]), "n_test": tot("n_test"),
                  "n_test_transpose_seen": tot("n_test_transpose_seen"), "n_test_unseen": tot("n_test_unseen"),
                  "n_test_diagonal": tot("n_test_diagonal"), "n_train_diagonal": tot("n_train_diagonal"),
                  "n_test_line": None, "n_test_unseen_offdiag": tot("n_test_unseen_offdiag"),
                  "missing_residues": {v.name: v.record["missing_residues"] for v in views},
                  "tasks": [v.record for v in views],
                  "vocab_layout": {k: getattr(lay, k) for k in ("first", "second", "answer", "op", "eq", "vocab")}}
    else:
        train, test, seen = make_split(args.p, args.frac, split_seed, args.pairs, args.diagonals, args.holdout_line)
        (xtr, ytr), (xte, yte) = (encode(ids, args.p, args.operands, args.task) for ids in (train, test))
        missing = sorted(set(range(args.p)) - set(ytr.tolist()))
        if missing and not args.allow_missing_residues:
            raise ValueError(f"residues never seen as a train answer: {missing}")
        data = {k: v.to(device) for k, v in dict(train_x=xtr, train_y=ytr, test_x=xte, test_y=yte, test_seen=seen).items()}
        sampler = Batches(len(train), args.batch, args.seed) if args.batch else None  # validates --batch early
        counts = {
            "n_train": len(train), "n_test": len(test), "n_test_transpose_seen": int(seen.sum()),
            "n_test_unseen": int((~seen).sum()), "n_test_diagonal": int(((test // args.p) == (test % args.p)).sum()),
            "n_train_diagonal": int(((train // args.p) == (train % args.p)).sum()),
            "n_test_line": None if args.holdout_line is None else int(line_mask(test // args.p, test % args.p, args.p, args.holdout_line).sum()),
            # the same group as the test_*_unseen_offdiag metrics: unseen, not diagonal, not on the held-out line
            "n_test_unseen_offdiag": int((~seen & ((test // args.p) != (test % args.p)) & ~(
                line_mask(test // args.p, test % args.p, args.p, args.holdout_line) if args.holdout_line is not None
                else torch.zeros(len(test), dtype=torch.bool))).sum()),
            "missing_residues": missing}

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
        **counts,
        # under --tasks with --task_start / --task_stop: all tasks' examples, the most an update can hold
        "examples_per_update": args.batch or counts["n_train"], "updates": args.updates,
        "full_batch": not args.batch,
        "n_params": sum(q.numel() for q in model.parameters()),
        "n_decayed": sum(q.numel() for q in opt.param_groups[0]["params"]), "exempt_params": exempt_names,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "running",
    }
    save_atomic(json.dumps(record, indent=1), out / "run_record.json")

    spikes, buf = Spikes(args.spike_factor, args.spike_floor), torch.zeros(args.eval_every, device=device)
    t0 = time.perf_counter()
    train_s, mark, last, history = 0.0, t0, {}, []
    task_hist = {v.name: [] for v in views} if views else {}
    with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:

        def log(update, extra):
            nonlocal mark, last
            t_eval = time.perf_counter()
            readout = evaluate_tasks(model, opt, views, args) if views else evaluate(model, opt, data, args)
            last = {"update": update, **extra, **readout,
                    "train_s": train_s, "elapsed_s": time.perf_counter() - t0}
            line = json.dumps(last)
            f.write(line + "\n")
            f.flush()
            print(line, flush=True)
            mark = time.perf_counter()
            return t_eval

        try:
            log(0, {"lr": 0.0, **({"active_tasks": [views[i].name for i in schedule.active(0)]} if views else {})})
            wd_until = getattr(args, "wd_until", None)
            for u in range(args.updates):
                lr = lr_at(u, args)
                if wd_until is not None and u == wd_until:  # --wd_until: no weight decay from this update on
                    for group in opt.param_groups:
                        group["weight_decay"] = 0.0
                at_log = (u + 1) % args.eval_every == 0 or u + 1 == args.updates
                loss_fn = None
                if views:
                    bx, by, parts = schedule.batch(u, data)
                    loss_fn = lambda: multi_loss(model, bx, by, parts, views, args, schedule.key)  # noqa: E731
                elif sampler is None:
                    bx, by = data["train_x"], data["train_y"]
                else:
                    idx = sampler.next().to(device)
                    bx, by = data["train_x"][idx], data["train_y"][idx]
                loss, stats = train_step(model, opt, bx, by, args, lr, at_log, loss_fn)
                buf[u % args.eval_every] = loss.float()
                if at_log:
                    losses = buf[: u % args.eval_every + 1].tolist()  # syncs the device
                    train_s += time.perf_counter() - mark
                    first = u + 1 - len(losses)
                    events = [e for i, v in enumerate(losses) if (e := spikes.feed(first + i, v))]
                    active = {"active_tasks": [views[i].name for i in schedule.active(u)]} if views else {}
                    log(u + 1, {"lr": lr, "train_obj_pre": losses[-1], "train_obj_pre_max": max(losses),
                                **stats, "spikes": spikes.count, "spike_events": events,
                                "nonfinite": sum(not math.isfinite(v) for v in losses),
                                "train_obj_trace": losses, **active})  # every update's loss since the last line
                    history.append((u + 1, last.get("test_acc_unseen"), last.get("test_acc_unseen_offdiag")))
                    for name, hist in task_hist.items():
                        hist.append((u + 1, last[f"test_acc_unseen@{name}"], last[f"test_acc_unseen_offdiag@{name}"],
                                     last[f"test_acc@{name}"]))
                    if not all(math.isfinite(v) for v in losses):
                        raise FloatingPointError(f"non-finite training loss by update {u + 1}")
                if (args.ckpt_every and (u + 1) % args.ckpt_every == 0 and u + 1 < args.updates) or \
                        u + 1 in getattr(args, "ckpt_at", ()):
                    save_atomic({"model": model.state_dict(), "update": u + 1}, out / f"ckpt_{u + 1:06d}.pt")
            if views:  # one table, with the task in an extra column
                wrong = {"update": args.updates, "columns": WRONG_COLUMNS + ["task"], "train": [], "test": []}
                for v in views:
                    w = wrong_pairs(model, v.data, v.args, args.updates)
                    for split in ("train", "test"):
                        wrong[split] += [r + [v.name] for r in w[split]]
            else:
                wrong = wrong_pairs(model, data, args, args.updates)
            save_atomic(json.dumps(wrong), out / "final_errors.json")
            tally = lambda tr, te: {"train": len(tr), "test": len(te), "test_unseen": sum(not r[4] for r in te),  # noqa: E731
                                    "test_diagonal": sum(r[0] == r[1] for r in te)}
            record["n_wrong_final"] = tally(wrong["train"], wrong["test"])
            if views:
                record["n_wrong_final"]["by_task"] = {
                    v.name: tally([r for r in wrong["train"] if r[6] == v.name], [r for r in wrong["test"] if r[6] == v.name])
                    for v in views}
            save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "update": args.updates,
                         "args": vars(args)}, out / "final.pt")
            record["status"] = "finished"
        except BaseException as e:
            record["status"] = f"failed: {type(e).__name__}: {e}"
            raise
        finally:
            record.update(ended=time.strftime("%Y-%m-%dT%H:%M:%S"), train_s=train_s,
                          elapsed_s=time.perf_counter() - t0, final=last, **summarize(history, args.updates),
                          **summarize_tasks(task_hist, args.updates))
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
    ap.add_argument("--tasks", default=None,
                    help=f"comma list from {','.join(TASK_SPECS)}: several tasks in one model, one op token each "
                         "(replaces --task; '--tasks add' is the same run as '--task add')")
    ap.add_argument("--answers", choices=["shared", "per_task"], default="shared",
                    help="--tasks only. shared: tasks on the same alphabet answer with its operand tokens; "
                         "per_task: each task gets its own output-only answer tokens")
    ap.add_argument("--task_split", choices=["aligned", "independent"], default="aligned",
                    help="--tasks only. aligned (default): every task splits from one shuffle of the pairs, so a test "
                         "pair of one task is (almost always) a test pair of the others; independent: own shuffle each")
    ap.add_argument("--allow_idle", action="store_true",
                    help="--tasks only: let the schedule leave updates with no active task (zero-gradient steps: "
                         "weight decay only, the pure-decay control for forgetting)")
    ap.add_argument("--task_weighting", choices=["equal", "examples"], default="equal",
                    help="--tasks only. equal: each task's mean loss counts the same; examples: mean over all examples")
    for flag, what in (("task_frac", "train fraction (default --frac)"),
                       ("task_weight", "relative loss weight (default 1)"),
                       ("task_start", "first update it trains (default 0)"),
                       ("task_stop", "update it stops training (default never)")):
        ap.add_argument("--" + flag, default="", help=f"--tasks only: name=value list such as sub=0.3,add=0.5; {what}")
    ap.add_argument("--allow_missing_residues", action="store_true", help="for small smoke sets only")
    ap.add_argument("--untie", action="store_true", help="separate output head (default: tied to the embedding)")
    ap.add_argument("--no_final_norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--wd_scope", choices=["all", "embeddings", "repo"], default="all")
    ap.add_argument("--wd_until", type=int, default=None,
                    help="weight decay stops (set to 0) from this update on; default: never")
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
    ap.add_argument("--ckpt_at", default="",
                    help="comma list of updates to also save weights at (about 2 MB each), e.g. 1000,5000,10000: "
                         "sparse checkpoints around grokking or a task switch, for analysis after the run")
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
    parse_tasks(ap, args)
    return args


def parse_tasks(ap, args):
    """--tasks and its options: names checked, name=value lists turned into dicts."""
    per_task = ("task_frac", "task_weight", "task_start", "task_stop")
    if args.wd_until is not None and not 0 < args.wd_until < args.updates:
        ap.error("--wd_until must be inside the run (1 .. --updates - 1)")
    try:
        args.ckpt_at = sorted({int(s) for s in args.ckpt_at.split(",") if s.strip()})
    except ValueError:
        ap.error(f"--ckpt_at {args.ckpt_at!r}: expected a comma list of updates")
    if any(not 0 < u < args.updates for u in args.ckpt_at):
        ap.error("--ckpt_at updates must be inside the run (1 .. --updates - 1; final.pt holds the last)")
    if not args.tasks:
        args.tasks = None
        if args.answers != "shared" or any(getattr(args, k) for k in per_task) or args.task_split != "aligned" \
                or args.task_weighting != "equal" or args.allow_idle:
            ap.error("--answers, --task_split, --task_weighting, --allow_idle and --task_* apply only with --tasks")
        for k in per_task:
            setattr(args, k, {})
        return
    names = [s.strip() for s in args.tasks.split(",") if s.strip()]
    unknown = [n for n in names if n not in TASK_SPECS]
    if unknown or not names or len(set(names)) != len(names):
        ap.error(f"--tasks {args.tasks!r}: names from {sorted(TASK_SPECS)}, each once")
    if args.task != "add":
        ap.error("--tasks replaces --task")
    if args.holdout_line is not None:
        ap.error("--holdout_line is not supported with --tasks")
    if any(TASK_SPECS[n].needs_prime for n in names) and any(args.p % d == 0 for d in range(2, int(args.p ** 0.5) + 1)):
        ap.error(f"mul and div need a prime --p, not {args.p}")
    args.tasks = names
    for k in per_task:
        cast = int if k in ("task_start", "task_stop") else float
        parsed = {}
        for item in filter(None, (s.strip() for s in getattr(args, k).split(","))):
            name, _, value = item.partition("=")
            if name not in names or not value:
                ap.error(f"--{k} {item!r}: expected name=value with a name from --tasks")
            parsed[name] = cast(value)
        setattr(args, k, parsed)
    if any(not 0 < f <= 1 for f in args.task_frac.values()) or any(w <= 0 for w in args.task_weight.values()):
        ap.error("--task_frac must be in (0, 1] and --task_weight above 0")
    for n in names:
        start, stop = args.task_start.get(n, 0), args.task_stop.get(n)
        if start < 0 or (stop is not None and stop <= start):
            ap.error(f"task {n}: needs 0 <= start < stop")


if __name__ == "__main__":
    run(parse_args())
