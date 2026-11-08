"""Bridge between tools/grok_lab.py and train.py: the grok_lab a + b mod p table, posed through train.py's own
tokenizer, mod_add templates, BOS/EOS, label masking, loss, batching, schedule, optimizer and autocast.

Used by train.py when --mod_table_p is set (spec: grok_runs/checks_2026-11-03/objective_audit.md §4). Tasks
(--mod_table_tasks): add, a + b mod p over all pairs; mul, a * b mod p over nonzero operands (the group (Z/p)*,
as grok_lab's mul), under 4 templates worded like mod_add's. Both split with grok_lab.make_split on the same seed
(grok_lab's --task_split aligned). --mod_table_start task=u keeps a task out of training until update u.
  ModTableDataset  every train pair of grok_lab.make_split(p, frac, seed, pairs), under all 4 mod_add templates,
                   encoded by FixedAlgorithmicDataset._encode_example (byte-identical to the --fixed_data path).
  TableScorer      teacher-forces all p*p pairs x 4 templates in eval mode and reads, at each answer position
                   (the answer digits and EOS): argmax correct, the margin z_t - logsumexp_{j != t} z_j over the
                   full vocab and over the 11 answer ids only (digits + EOS), and whether the target's fp32
                   gradient term rounds to zero (margin > ln 2^24). Pair-level: exact = every position right.
                   Groups: train/test x diagonal/off-diagonal, and single-digit diagonals (a < 10).
"""
import importlib.util
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import ConcatDataset, DataLoader

from data import FixedAlgorithmicDataset, MOD_ADD_TEMPLATES, make_collate_fn

CLIFF_FP32 = math.log(2.0 ** 24)
_GL = None

# mod_mul's templates: mod_add's four, with the operation words changed and nothing else
MOD_MUL_TEMPLATES = (
    "Compute ({a} * {b}) mod {mod}. Answer:",
    "What is ({a} * {b}) modulo {mod}? Answer:",
    "Multiply {a} and {b}, then take mod {mod}. Answer:",
    "({a} times {b}) % {mod} equals. Answer:",
)
# task -> (templates, answer, domain or None, the example's task label)
TABLE_TASKS = {
    "add": (MOD_ADD_TEMPLATES, lambda a, b, p: (a + b) % p, None, "mod_add"),
    "mul": (MOD_MUL_TEMPLATES, lambda a, b, p: (a * b) % p, lambda a, b, p: (a != 0) & (b != 0), "mod_mul"),
}
# The other grok_lab number tasks (task x format study, 2026-11-07), same functions and domains as grok_lab's
# TASK_SPECS. Templates carry no digit other than the modulus (operations in words), so the only digit runs in a
# sequence are the operands, the modulus and the answer.
TABLE_TASKS.update({
    "sub": (("Compute ({a} - {b}) mod {mod}. Answer:", "What is ({a} - {b}) modulo {mod}? Answer:",
             "Subtract {b} from {a}, then take mod {mod}. Answer:", "({a} minus {b}) % {mod} equals. Answer:"),
            lambda a, b, p: (a - b) % p, None, "mod_sub"),
    "div": (("Compute ({a} / {b}) mod {mod}. Answer:", "What is ({a} / {b}) modulo {mod}? Answer:",
             "Divide {a} by {b}, then take mod {mod}. Answer:", "({a} divided by {b}) % {mod} equals. Answer:"),
            lambda a, b, p: (a * pow(int(b), p - 2, p)) % p, lambda a, b, p: b != 0, "mod_div"),
    "add2b": (("Compute ({a} plus twice {b}) mod {mod}. Answer:", "What is ({a} plus twice {b}) modulo {mod}? Answer:",
               "Add {a} and twice {b}, then take mod {mod}. Answer:", "({a} plus double {b}) % {mod} equals. Answer:"),
              lambda a, b, p: (a + 2 * b) % p, None, "mod_add2b"),
    "sq": (("Compute ({a} squared plus {b} squared) mod {mod}. Answer:",
            "What is ({a} squared plus {b} squared) modulo {mod}? Answer:",
            "Square {a} and {b}, add them, then take mod {mod}. Answer:",
            "({a} squared plus {b} squared) % {mod} equals. Answer:"),
           lambda a, b, p: (a * a + b * b) % p, None, "mod_sq"),
    "sqab": (("Compute ({a} squared plus {a} times {b} plus {b} squared) mod {mod}. Answer:",
              "What is ({a} squared plus {a} times {b} plus {b} squared) modulo {mod}? Answer:",
              "Square {a}, add {a} times {b} and {b} squared, then take mod {mod}. Answer:",
              "({a} squared plus {a} times {b} plus {b} squared) % {mod} equals. Answer:"),
             lambda a, b, p: (a * a + a * b + b * b) % p, None, "mod_sqab"),
    "cubeab": (("Compute ({a} cubed plus {a} times {b}) mod {mod}. Answer:",
                "What is ({a} cubed plus {a} times {b}) modulo {mod}? Answer:",
                "Cube {a}, add {a} times {b}, then take mod {mod}. Answer:",
                "({a} cubed plus {a} times {b}) % {mod} equals. Answer:"),
               lambda a, b, p: (a * a * a + a * b) % p, None, "mod_cubeab"),
    "gcd": (("Compute the gcd of {a} and {b}. Answer:", "What is gcd({a}, {b})? Answer:",
             "Find the greatest common divisor of {a} and {b}. Answer:", "The gcd of {a} and {b} equals. Answer:"),
            lambda a, b, p: math.gcd(int(a), int(b)), lambda a, b, p: (a != 0) & (b != 0), "gcd"),
})
# grok_lab's --diagonals per task: gcd's large values occur almost only on the diagonal, so it trains them all
TASK_DIAGONALS = {"gcd": "train"}
# One token per number (--mod_table_numbers / --mod_table_answer token): llemma's byte-fallback pieces, which no
# ASCII template, digit or modulus ever produces (checked by the gate). Operands and answers get separate banks,
# as grok_lab's --answers per_task: operand n -> id 3 + n (<0x00>..<0x60>), answer n -> id 100 + n (<0x61>..).
SUR_OPERAND0, SUR_ANSWER0 = 3, 100
PRIMITIVE_ROOT = {97: 5}
# as grok_lab's needs_prime: div's pow(b, p - 2, p) is an inverse only mod a prime, and mul's nonzero domain is a
# group (no answer 0) only then
PRIME_TASKS = ("mul", "div")


def check_modulus(task, p):
    if task in PRIME_TASKS and (p < 2 or any(p % d == 0 for d in range(2, math.isqrt(p) + 1))):
        raise ValueError(f"--mod_table_p: {task} needs a prime modulus, not {p}")


def parse_tasks(spec):
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TABLE_TASKS for t in tasks):
        raise ValueError(f"--mod_table_tasks {spec!r}: a comma list of distinct tasks from {sorted(TABLE_TASKS)}")
    return tasks


def parse_starts(spec, tasks):
    """'mul=2500,...' -> {task: first update it trains on}; unlisted tasks start at 0."""
    starts = {t: 0 for t in tasks}
    for item in [x for x in (spec or "").split(",") if x.strip()]:
        name, _, u = item.partition("=")
        name = name.strip()
        if name not in starts or not u.strip().isdigit():
            raise ValueError(f"--mod_table_start {spec!r}: entries task=update for tasks in {tasks}")
        starts[name] = int(u)
    if min(starts.values()) != 0:
        raise ValueError("--mod_table_start: at least one task must train from update 0")
    return starts


def domain_mask(p, task):
    dom = TABLE_TASKS[task][2]
    if dom is None:
        return None
    i = torch.arange(p * p)
    return dom(i // p, i % p, p)


def make_table_loader(datasets, per_task_batch, seed, num_workers, collate_fn):
    """The training loader over the active tasks: an epoch shuffle of their pooled sequences (grok_lab's pooled
    sampler), per_task_batch sequences per active task per micro-batch, last partial batch dropped. A single task
    gives exactly the loader the one-task bridge used (b1/b2), so its stream is unchanged."""
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(ds, batch_size=per_task_batch * len(datasets), shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
                      generator=torch.Generator().manual_seed(seed))


def grok_lab():
    """tools/grok_lab.py, loaded by path (tools/ is not a package)."""
    global _GL
    if _GL is None:
        spec = importlib.util.spec_from_file_location("grok_lab", Path(__file__).with_name("grok_lab.py"))
        _GL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GL)
    return _GL


def table_split(p, frac, seed, pairs, task="add"):
    """(train ids, test ids, seen) over pair ids a*p + b, exactly grok_lab's split (mul: its domain mask)."""
    if task in TASK_DIAGONALS:
        return grok_lab().make_split(p, frac, seed, pairs, TASK_DIAGONALS[task], valid=domain_mask(p, task))
    return grok_lab().make_split(p, frac, seed, pairs, valid=domain_mask(p, task))


def display_map(task, p, relabel, seed=0):
    """--mod_table_relabel: the number written for each value (operands and answer alike), or None for identity.
    log  (mul): n -> its discrete log base the primitive root, 0..p-2. The relation shown becomes addition
         mod p - 1 on the logs: a bijection of the same table, digit-aligned with addition.
    perm: n -> a fixed seeded random bijection of the task's values onto the same coordinate set as log for
         mul (0..p-2) and onto 0..p-1 otherwise (random.Random(seed).shuffle; stdlib, deterministic).
    Neither supports div: its operand a and its answer include 0, which has no discrete log.
    Scoring is on the displayed answer, which is a bijective image of the true one."""
    import random
    if relabel == "none":
        return None
    values = list(range(1, p)) if task in ("mul", "div") else list(range(p))
    if relabel == "log":
        if task != "mul" or p not in PRIMITIVE_ROOT:  # div: a = 0 (answer 0) has no log
            raise ValueError(f"--mod_table_relabel log: mul with a known primitive root, not {task} mod {p}")
        g = PRIMITIVE_ROOT[p]
        return {pow(g, k, p): k for k in range(p - 1)}
    if relabel == "perm":
        if task == "div":
            raise ValueError("--mod_table_relabel perm: div's operand a includes 0; not supported")
        target = list(range(len(values)))
        random.Random(seed).shuffle(target)
        return dict(zip(values, target))
    raise ValueError(f"unknown --mod_table_relabel {relabel!r}")


def example_for(a, b, p, t, task="add", disp=None):
    templates, answer, _, label = TABLE_TASKS[task]
    result = answer(a, b, p)
    if disp is not None:
        a, b, result = disp[a], disp[b], disp[result]
    prompt = templates[t].format(a=a, b=b, mod=p)
    return {"text": f"{prompt}{result}", "input": prompt, "target": str(result), "task": label}


def placeholder_order(template):
    import string
    return [f for _, f, _, _ in string.Formatter().parse(template) if f]


def render_tokens(payload, ex_values, digit_ids, numbers, answer, eos_id):
    """--mod_table_numbers / --mod_table_answer token: replace each operand's digit run by its one surrogate id
    and/or the answer's digit run by its one answer id, in the BOS + prompt + answer + EOS payload. ex_values: the
    expected digit strings in order (placeholders a, b, mod as they occur in the template, then the answer) and
    which of them are operands. Raises if the sequence's digit runs are not exactly those strings."""
    ids = payload["input_ids"].tolist()
    labels = payload["labels"].tolist()
    prompt_len = sum(1 for x in labels if x == -100)
    runs, k = [], 0
    while k < len(ids):
        if ids[k] in digit_ids:
            s = k
            while k < len(ids) and ids[k] in digit_ids:
                k += 1
            runs.append((s, k))
        else:
            k += 1
    expect = ex_values["strings"]
    got = ["".join(str(digit_ids[x]) for x in ids[s:e]) for s, e in runs]
    if got != expect:
        raise ValueError(f"mod table render: digit runs {got} != expected {expect}")
    if runs[-1][0] != prompt_len or ids[-1] != eos_id or runs[-1][1] != len(ids) - 1:
        raise ValueError("mod table render: the last digit run is not the answer before EOS")
    out, prev = [], 0
    for idx, (s, e) in enumerate(runs):
        out += ids[prev:s]
        kind = ex_values["kinds"][idx]
        if kind == "operand" and numbers == "token":
            out.append(SUR_OPERAND0 + int(expect[idx]))
        elif kind == "answer" and answer == "token":
            out.append(SUR_ANSWER0 + int(expect[idx]))
        else:
            out += ids[s:e]
        if kind == "answer":
            new_prompt_len = len(out) - (1 if answer == "token" else e - s)
        prev = e
    out += ids[prev:]
    input_ids = torch.tensor(out, dtype=torch.long)
    lab = input_ids.clone()
    lab[:new_prompt_len] = -100
    res = dict(payload)
    res["input_ids"], res["labels"] = input_ids, lab
    if "input_len_tokens" in res:
        res["input_len_tokens"] = new_prompt_len - 1  # without BOS, as the encoder's add_special_tokens=False count
        res["target_len_tokens"] = len(out) - 1 - new_prompt_len
    return res


class ModTableDataset(FixedAlgorithmicDataset):
    """Every train pair of one task under every template (4 x n_train sequences). Same payload as --fixed_data."""

    def __init__(self, tokenizer, p=97, frac=0.5, split_seed=0, pairs="paired", max_seq_len=32,
                 include_lengths=True, task="add", numbers="digits", answer_format="digits", relabel="none",
                 relabel_seed=0):
        # FixedAlgorithmicDataset's state, without its random example generation
        self.tokenizer, self.max_seq_len, self.include_lengths = tokenizer, max_seq_len, include_lengths
        self.task = task
        check_modulus(task, p)
        # number rendering and relabelling (task x format study); the defaults are the original path, untouched
        if numbers not in ("digits", "token") or answer_format not in ("digits", "token"):
            raise ValueError(f"mod table: numbers/answer must be digits or token, not {numbers!r}/{answer_format!r}")
        if relabel != "none" and (numbers, answer_format) != ("digits", "digits"):
            raise ValueError("mod table: --mod_table_relabel applies to digit strings only")
        self.numbers, self.answer_format, self.relabel, self.relabel_seed = numbers, answer_format, relabel, relabel_seed
        self.disp = display_map(task, p, relabel, relabel_seed)
        self.digit_ids = {tokenizer.encode(d, add_special_tokens=False)[-1]: d for d in "0123456789"}
        self.templates, answer, _, label = TABLE_TASKS[task]
        self.tasks, self.seed = [label], split_seed
        self.p, self.frac, self.pairs = p, frac, pairs
        self.train_ids, self.test_ids, self.test_seen = table_split(p, frac, split_seed, pairs, task)
        dom = domain_mask(p, task)
        self.domain = torch.ones(p * p, dtype=torch.bool) if dom is None else dom
        answers = {int(answer(i // p, i % p, p)) for i in self.train_ids.tolist()}
        every = {int(answer(i // p, i % p, p)) for i in self.domain.nonzero().squeeze(1).tolist()}
        missing = sorted(every - answers)
        if missing:
            raise ValueError(f"mod table: answers never seen in train: {missing}")
        self.examples, self.index = [], []
        for i in self.train_ids.tolist():
            a, b = divmod(i, p)
            for t in range(len(self.templates)):
                self.examples.append(self.encode(a, b, t))
                self.index.append((a, b, t))
        self.num_examples = len(self.examples)

    def encode(self, a, b, t):
        """The payload of pair (a, b) under template t: the original path's, then the number rendering."""
        p, tokenizer = self.p, self.tokenizer
        ex = self._encode_example(example_for(a, b, p, t, self.task, self.disp))
        ids = ex["input_ids"]
        if (int(ids.numel()) >= self.max_seq_len or int(ids[-1]) != tokenizer.eos_token_id
                or int((ex["labels"] != -100).sum()) < 2):
            raise ValueError(f"mod table: sequence for ({a}, {b}, template {t}) is truncated or has "
                             f"no answer; raise --alg_seq_len")
        if (self.numbers, self.answer_format) == ("digits", "digits"):
            return ex
        da, db = (a, b) if self.disp is None else (self.disp[a], self.disp[b])
        vals = {"a": str(da), "b": str(db), "mod": str(p)}
        order = placeholder_order(self.templates[t])
        ans = TABLE_TASKS[self.task][1](a, b, p)
        strings = [vals[f] for f in order] + [str(ans if self.disp is None else self.disp[ans])]
        kinds = ["operand" if f in ("a", "b") else "mod" for f in order] + ["answer"]
        return render_tokens(ex, {"strings": strings, "kinds": kinds}, self.digit_ids, self.numbers,
                             self.answer_format, tokenizer.eos_token_id)

    def split_record(self):
        p = self.p
        tr, te = self.train_ids, self.test_ids
        return {"task": self.task, "p": p, "frac": self.frac, "pairs": self.pairs, "split_seed": self.seed,
                "n_train_pairs": len(tr), "n_test_pairs": len(te), "n_train_sequences": self.num_examples,
                "n_test_transpose_seen": int(self.test_seen.sum()),
                "n_train_diagonal": int(((tr // p) == (tr % p)).sum()),
                "n_test_diagonal": int(((te // p) == (te % p)).sum()),
                "n_domain_pairs": int(self.domain.sum()), "templates": list(self.templates),
                "numbers": self.numbers, "answer_format": self.answer_format, "relabel": self.relabel,
                "relabel_seed": self.relabel_seed if self.relabel == "perm" else None}


def move_aside(output_dir):
    """Move everything an earlier attempt left in output_dir (scores, record, metrics, checkpoints, summary) into
    old_<time>/, so a restarted run never mixes with it. Called before train.py's logger opens any file."""
    out = Path(output_dir)
    if not out.exists():
        return
    old = [f for f in out.iterdir() if not f.name.startswith("old_")]
    if old:
        aside = out / time.strftime("old_%Y%m%d-%H%M%S")
        aside.mkdir(parents=True, exist_ok=True)
        for f in old:
            f.replace(aside / f.name)


# Exact resume of a bridge run (train.py --mod_table_resume auto, --mod_table_init_from). resume.pt holds everything
# the loop carries from one iteration to the next; see train.py's save_resume().
RESUME_FILE = "resume.pt"
RESUME_FORMAT = 1
RESUME_FREE_ARGS = frozenset({"mod_table_resume_every"})  # may differ between a run and its resume: no effect on training
EXTEND_FREE_ARGS = RESUME_FREE_ARGS | {"total_tokens", "alg_tokens", "output_dir", "mod_table_init_from"}
# bridge_record fields that must match between the checkpointed run and the one continuing it: same code, split,
# tokenizer, shapes, library versions, device and optimizer
RESUME_SAME_RECORD = ("file_sha256", "split_sha256", "tokenizer", "tokenizer_vocab_sha256", "tokenizer_len",
                      "tokenizer_files", "eos_id", "bos_id", "pad_id", "param_shapes", "versions", "device_name",
                      "autocast_dtype", "n_params", "tied", "dropout", "optimizer", "tasks", "starts", "splits")


def atomic_write_text(path, text):
    """Write text to path through a fsync'd temporary file and os.replace (no half-written file after a crash)."""
    import os
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path, data):
    import os
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj, path, before_replace=None):
    """torch.save through a fsync'd temporary file and os.replace; before_replace() runs between the two (a test
    hook). A crash at any point leaves either the previous file or the new one, never a partial one."""
    import os
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if before_replace is not None:
        before_replace()
    os.replace(tmp, path)


def load_resume(path):
    # weights_only=True: resume.pt holds only tensors and plain Python values (no config, Namespace or numpy objects;
    # #138), so it loads in any fresh process without add_safe_globals
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != RESUME_FORMAT:
        raise ValueError(f"{path}: not a format-{RESUME_FORMAT} resume file")
    return state


# --mod_table_resume_fallback previous: a second, verified copy (resume.prev.pt), each payload with a sidecar
# (NAME.sha256: JSON with its size and sha256). A slot counts only when its payload matches its sidecar. Publishing
# keeps one verified pair at every crash point (process kill; power loss is best effort on Windows, whose os.replace
# does not write through):
#   1. current pair verifies -> copy current to resume.prev.pt (tmp, fsync, replace), then its sidecar
#   2. new payload -> resume.pt.tmp (fsync), loaded back with weights_only=True, hashed; then replace resume.pt,
#      then its sidecar
# killed in 1: current verifies; killed in 2 before its sidecar: the backup verifies. Right after a fallback, the
# rejected current is never copied over the good backup (rotate=False on that first save).
RESUME_PREV = "resume.prev.pt"


def sidecar(path):
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _present(path):
    """True/False for a regular file being there; SystemExit for anything that is not a plain yes or no (a directory,
    a permission or sharing error: not damage). os.stat, not Path.exists(), which reads some errors as False."""
    import os
    import stat
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return False
    except OSError as e:
        raise SystemExit(f"--mod_table_resume: cannot stat {path} ({type(e).__name__}: {e}); not damage, so no "
                         f"fallback") from e
    if not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"--mod_table_resume: {path} is not a regular file; not damage, so no fallback")
    return True


def check_pair(path):
    """None when path and its sidecar agree (size and sha256), else the reason it does not count (damage: the caller
    may try the other slot). A file that cannot be read for another reason (permission, sharing, any other OSError)
    is not damage: SystemExit."""
    import re
    path = Path(path)
    if not _present(path):
        return "missing"
    if not _present(sidecar(path)):
        return "no sidecar"
    try:
        text = sidecar(path).read_text(encoding="utf-8")
        try:
            meta = json.loads(text)
        except ValueError:
            return "unreadable sidecar"
        size, sha = (meta.get("size"), meta.get("sha256")) if isinstance(meta, dict) else (None, None)
        if type(size) is not int or size < 0 or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            return "malformed sidecar"
        actual = path.stat().st_size
        if actual != size:
            return f"size {actual} != sidecar {size}"
        if file_sha256(path) != sha:
            return "sha256 differs from its sidecar"
        return None
    except UnicodeDecodeError:
        return "unreadable sidecar"
    except OSError as e:
        raise SystemExit(f"--mod_table_resume: cannot read {path} or its sidecar ({type(e).__name__}: {e}); "
                         f"not damage, so no fallback") from e


def select_resume(output_dir):
    """--mod_table_resume_fallback previous at startup: (state, chosen, rejected) from resume.pt, else from
    resume.prev.pt; None for a fresh run. Only a failing pair (missing payload, missing or bad sidecar, size or sha256
    mismatch) falls back. A pair that verifies is the bytes publish_resume wrote and read back, so if it then fails to
    load, the cause is not damage (library version, memory, an unsupported global): stop, never roll back. The
    caller's argument/fingerprint checks then apply to the chosen file and stop on a mismatch. SystemExit when neither
    is usable, or when both payloads are gone but a sidecar or resume_log.jsonl shows checkpoints were published."""
    output_dir = Path(output_dir)
    cur, prev = output_dir / RESUME_FILE, output_dir / RESUME_PREV
    if not _present(cur) and not _present(prev):
        evidence = [q.name for q in (sidecar(cur), sidecar(prev), output_dir / "resume_log.jsonl") if _present(q)]
        if evidence:
            raise SystemExit(f"--mod_table_resume: {output_dir} has no resume.pt or resume.prev.pt, but {evidence} "
                             f"show checkpoints were published: not starting this run afresh (move them aside to start afresh)")
        return None  # nothing was ever committed (a first save cut off before its replace counts as nothing)
    rejected = []
    for q in (cur, prev):
        why = check_pair(q)
        if why is not None:
            rejected.append({"file": q.name, "reason": why})
            continue
        try:
            state = load_resume(q)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            raise SystemExit(f"--mod_table_resume: {q.name} verifies against its sidecar but does not load "
                             f"({type(e).__name__}: {str(e)[:300]}); not damage, so no fallback") from e
        return state, q.name, rejected
    raise SystemExit(f"--mod_table_resume: no usable checkpoint in {output_dir}: "
                     + "; ".join(f"{r['file']}: {r['reason']}" for r in rejected)
                     + " (move them aside to start afresh)")


def publish_resume(obj, output_dir, rotate=True, kill=None):
    """Write obj as resume.pt under the protocol above. kill(point) is the test hook: "rotate" (backup payload
    replaced, its sidecar not yet), "save" (new payload staged and verified, not yet replaced), "replaced" (resume.pt
    replaced, its sidecar not yet)."""
    import os
    import shutil
    output_dir = Path(output_dir)
    cur, prev = output_dir / RESUME_FILE, output_dir / RESUME_PREV
    kill = kill or (lambda point: None)
    if rotate and check_pair(cur) is None:
        tmp = prev.with_name(prev.name + ".tmp")
        shutil.copyfile(cur, tmp)
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, prev)
        kill("rotate")
        atomic_write_text(sidecar(prev), sidecar(cur).read_text(encoding="utf-8"))
    tmp = cur.with_name(cur.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    back = load_resume(tmp)  # the payload as written loads under weights_only=True, before it counts
    if back.get("micro_steps") != obj.get("micro_steps") or back.get("kind") != obj.get("kind"):
        raise RuntimeError(f"{tmp}: read-back differs from what was written")
    del back
    meta = json.dumps({"sha256": file_sha256(tmp), "size": tmp.stat().st_size, "kind": obj.get("kind"),
                       "micro_steps": obj.get("micro_steps")})
    kill("save")
    os.replace(tmp, cur)
    kill("replaced")
    atomic_write_text(sidecar(cur), meta)


def stale_resume_temps(output_dir):
    """Temporary files a cut-off write leaves (never part of a committed pair)."""
    output_dir = Path(output_dir)
    names = [RESUME_FILE, RESUME_PREV]
    return [p for n in names for p in (output_dir / (n + ".tmp"), sidecar(output_dir / n).with_name(
        sidecar(output_dir / n).name + ".tmp")) if p.exists()]


def plain(obj):
    """obj as JSON-plain values (dicts, lists, str, int, float, bool, None): what resume.pt may hold besides tensors.
    Raises TypeError on anything else (a tensor or numpy value would otherwise sneak in)."""
    return json.loads(json.dumps(obj))


def numpy_rng_state():
    import numpy as np
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"name": str(name), "keys": [int(k) for k in keys], "pos": int(pos), "has_gauss": int(has_gauss),
            "cached": float(cached)}


def set_numpy_rng_state(st):
    import numpy as np
    np.random.set_state((st["name"], np.array(st["keys"], dtype=np.uint32), st["pos"], st["has_gauss"], st["cached"]))


def bytes_tensor(data):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone() if data else torch.zeros(0, dtype=torch.uint8)


def arg_differences(saved, current, free):
    """[(name, saved, current)] for every argument that differs, except those in free."""
    keys = sorted(set(saved) | set(current))
    return [(k, saved.get(k, "<absent>"), current.get(k, "<absent>")) for k in keys
            if k not in free and saved.get(k, "<absent>") != current.get(k, "<absent>")]


def record_differences(saved, fresh):
    return [k for k in RESUME_SAME_RECORD if saved.get(k) != fresh.get(k)]


def tensor_sha256(*tensors):
    import hashlib
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def code_tree_sha256(root):
    """One hash over every .py file under root (train.py's directory: the code that runs, eval_hub.py and eval/
    included), by relative path."""
    import hashlib
    root = Path(root)
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if "__pycache__" in rel or rel.startswith("old_"):
            continue
        h.update(rel.encode() + b"\0" + hashlib.sha256(f.read_bytes()).hexdigest().encode())
    return h.hexdigest()


def dataset_sha256(ds):
    """One hash over a ModTableDataset's encoded training sequences (catches any tokenizer or rendering change)."""
    import hashlib
    h = hashlib.sha256()
    for ex in ds.examples:
        h.update(tensor_sha256(ex["input_ids"], ex["labels"]).encode())
    return h.hexdigest()


def scorer_sha256(scorer):
    """One hash over a TableScorer's encoded scoring inputs, target positions, targets and answer ids: catches a
    tokenizer or rendering change that touches only held-out pairs, which dataset_sha256 cannot see."""
    import hashlib
    h = hashlib.sha256(tensor_sha256(scorer.answer_ids).encode())
    for b in scorer.batches:
        h.update(tensor_sha256(*b).encode())
    return h.hexdigest()


def tokenizer_sha256(tokenizer):
    """The instantiated tokenizer: its class, special tokens, Python-side settings (init kwargs from
    tokenizer_config.json, and the live length, padding, truncation and decoding attributes) and, for a fast tokenizer,
    the full serialized backend (normalizer, pre-tokenizer, model, post-processor, added tokens); else its vocabulary
    and added tokens."""
    import hashlib
    backend = getattr(tokenizer, "backend_tokenizer", None)
    body = backend.to_str() if backend is not None else json.dumps(
        [sorted(tokenizer.get_vocab().items()), sorted(tokenizer.get_added_vocab().items())])
    live = {a: getattr(tokenizer, a, None) for a in ("model_max_length", "padding_side", "truncation_side",
                                                       "clean_up_tokenization_spaces", "split_special_tokens",
                                                       "add_bos_token", "add_eos_token", "add_prefix_space")}
    # json keeps primitive types apart (False, "False" and 0 differ); anything else is tagged with its type name
    by_key = lambda d: sorted(d.items(), key=lambda kv: str(kv[0]))  # noqa: E731
    head = json.dumps([type(tokenizer).__name__, str(tokenizer.special_tokens_map),
                       by_key(getattr(tokenizer, "init_kwargs", {})), by_key(live)],
                      default=lambda o: [type(o).__name__, str(o)])
    return hashlib.sha256((head + "\0" + body).encode()).hexdigest()


def batch_sha256(batch):
    return tensor_sha256(batch["input_ids"], batch["labels"])


def state_sha256(model, optimizer):
    """Hashes of the model's and the optimizer's tensors (for comparing runs without loading their files)."""
    import hashlib
    sd = model.state_dict()
    m = hashlib.sha256()
    for k in sorted(sd):
        m.update(k.encode())
        m.update(tensor_sha256(sd[k]).encode())
    o = hashlib.sha256()
    osd = optimizer.state_dict()
    for pid in sorted(osd["state"]):
        for k in sorted(osd["state"][pid]):
            v = osd["state"][pid][k]
            o.update(f"{pid}/{k}".encode())
            o.update((tensor_sha256(v) if torch.is_tensor(v) else repr(v)).encode())
    o.update(json.dumps([{k: v for k, v in g.items() if k != "params"} for g in osd["param_groups"]],
                        sort_keys=True, default=str).encode())
    return {"model": m.hexdigest(), "optimizer": o.hexdigest()}


def _git(*cmd):
    import subprocess
    try:
        return subprocess.run(["git", *cmd], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(tokenizer, dataset, model=None):
    """Code version, file hashes, tokenizer identity and split hash, for the run record (#187, #189). A frozen copy
    (outside git) carries FREEZE.json from make_plan.py --freeze: the source commit, dirty-diff hash, file hashes."""
    import hashlib
    import platform
    root = Path(__file__).resolve().parents[1]
    files = {}
    for rel in ("train.py", "data.py", "model.py", "utils.py", "tools/grok_bridge.py", "tools/grok_lab.py"):
        try:
            files[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            files[rel] = None
    status = _git("status", "--porcelain")
    h = hashlib.sha256()
    for t in (dataset.train_ids, dataset.test_ids):
        h.update(t.numpy().tobytes())
    vocab = sorted(tokenizer.get_vocab().items())
    tok_files = {}
    for attr in ("vocab_file", "tokenizer_file"):
        f = getattr(tokenizer, attr, None)
        if f and Path(f).exists():
            tok_files[attr] = {"path": str(f), "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()}
    freeze = None
    if (root / "FREEZE.json").exists():
        freeze = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:
        tf_version = None
    rec = {"commit": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
           "freeze": freeze, "file_sha256": files, "tokenizer": getattr(tokenizer, "name_or_path", None),
           "tokenizer_vocab_size": tokenizer.vocab_size, "tokenizer_len": len(tokenizer),
           "tokenizer_vocab_sha256": hashlib.sha256(json.dumps(vocab).encode()).hexdigest(),
           "tokenizer_files": tok_files, "eos_id": tokenizer.eos_token_id, "bos_id": tokenizer.bos_token_id,
           "pad_id": tokenizer.pad_token_id, "split_sha256": h.hexdigest(),
           "versions": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "transformers": tf_version}}
    if model is not None:
        rec["param_shapes"] = {n: list(q.shape) for n, q in model.named_parameters()}
    return rec


@contextmanager
def head_off(model):
    """Run model(...) up to the final norm: lm_head swapped for Identity (the tie is untouched)."""
    head = model.lm_head
    model.lm_head = nn.Identity()
    try:
        yield head
    finally:
        model.lm_head = head


class TableScorer:
    """Scores all p*p pairs x 4 templates. One JSONL row per call; failing rows on the final call."""

    GROUPS = ("train_diag", "train_off", "test_diag", "test_off", "test_diag_1digit", "test_off_seen",
              "test_off_unseen")

    def __init__(self, tokenizer, dataset, device, ctx, out_path, batch_size=512):
        self.ds, self.device, self.ctx, self.out_path = dataset, device, ctx, Path(out_path)
        self.batch_size = batch_size
        p = dataset.p
        digit_ids = []
        for d in "0123456789":
            ids = tokenizer.encode(d, add_special_tokens=False)
            digit_ids.append(ids[-1])
        if getattr(dataset, "answer_format", "digits") == "token":  # one answer id per residue, then EOS
            digit_ids = list(range(SUR_ANSWER0, SUR_ANSWER0 + p))
        self.answer_ids = torch.tensor(sorted(set(digit_ids)) + [tokenizer.eos_token_id], device=device)
        collate = make_collate_fn(tokenizer)
        in_train = torch.zeros(p * p, dtype=torch.bool)
        in_train[dataset.train_ids] = True
        seen = torch.zeros(p * p, dtype=torch.bool)
        seen[dataset.test_ids[dataset.test_seen]] = True
        rows, meta = [], []
        task = getattr(dataset, "task", "add")
        for i in dataset.domain.nonzero().squeeze(1).tolist():  # add: every pair, in the one-task order
            a, b = divmod(i, p)
            for t in range(len(dataset.templates)):
                rows.append(dataset.encode(a, b, t) if hasattr(dataset, "encode")
                            else dataset._encode_example(example_for(a, b, p, t, task)))
                meta.append((a, b, t, bool(in_train[i]), bool(seen[i])))
        self.meta = meta
        self.batches = []
        for s in range(0, len(rows), batch_size):
            batch = collate(rows[s:s + batch_size])
            ids, labels = batch["input_ids"], batch["labels"]
            tgt = labels[:, 1:]
            pos = (tgt != -100).nonzero()  # (row, position j): logits at j predict labels[j + 1]
            self.batches.append((ids.to(device), pos[:, 0].to(device), pos[:, 1].to(device),
                                 tgt[pos[:, 0], pos[:, 1]].to(device)))
        bad = set(torch.cat([b[3] for b in self.batches]).unique().tolist()) - set(self.answer_ids.tolist())
        if bad:
            raise ValueError(f"mod table scorer: target ids outside digits + EOS: {sorted(bad)}")
        n = len(meta)
        g = {k: torch.zeros(n, dtype=torch.bool) for k in self.GROUPS}
        for k, (a, b, t, tr, sn) in enumerate(meta):
            diag = a == b
            if tr:
                g["train_diag" if diag else "train_off"][k] = True
            else:
                g["test_diag" if diag else "test_off"][k] = True
                if diag and a < 10:
                    g["test_diag_1digit"][k] = True
                if not diag:
                    g["test_off_seen" if sn else "test_off_unseen"][k] = True
        self.groups = {k: v.to(device) for k, v in g.items()}
        self.pair_of_row = torch.tensor([a * p + b for a, b, *_ in meta], device=device)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def per_row(self, model):
        """Per (pair, template) row: exact, min margin (full vocab), first-digit margin, min 11-class margin,
        cliff-proxy counts. Model left in its previous train/eval mode."""
        was_training = model.training
        model.eval()
        out = {k: [] for k in ("exact", "min_margin", "first_margin", "min_margin11", "cliff_any", "cliff_n",
                               "n_pos")}
        try:
            for ids, r, j, t in self.batches:
                with self.ctx, head_off(model) as head:
                    h, _, _ = model(ids)
                    z = head(h[r, j]).float()  # (positions, vocab) in fp32
                zt = z.gather(1, t[:, None]).squeeze(1)
                zo = z.clone()
                zo.scatter_(1, t[:, None], float("-inf"))
                margin = zt - torch.logsumexp(zo, dim=1)
                z11 = z[:, self.answer_ids]
                t11 = (self.answer_ids[None, :] == t[:, None]).float().argmax(1)
                z11o = z11.clone()
                z11o.scatter_(1, t11[:, None], float("-inf"))
                margin11 = z11.gather(1, t11[:, None]).squeeze(1) - torch.logsumexp(z11o, dim=1)
                correct = z.argmax(1) == t
                nrow = int(ids.shape[0])
                first = torch.ones_like(r, dtype=torch.bool)
                first[1:] = r[1:] != r[:-1]
                for key, vals, red in (("exact", correct.float(), "amin"), ("min_margin", margin, "amin"),
                                       ("min_margin11", margin11, "amin"),
                                       ("cliff_any", (margin > CLIFF_FP32).float(), "amax"),
                                       ("cliff_n", (margin > CLIFF_FP32).float(), "sum"),
                                       ("n_pos", torch.ones_like(margin), "sum")):
                    init = {"amin": float("inf"), "amax": float("-inf"), "sum": 0.0}[red]
                    buf = torch.full((nrow,), init, device=z.device)
                    buf.scatter_reduce_(0, r, vals, reduce=red, include_self=True)
                    out[key].append(buf)
                fm = torch.zeros(nrow, device=z.device)
                fm[r[first]] = margin[first]
                out["first_margin"].append(fm)
        finally:
            model.train(was_training)
        return {k: torch.cat(v) for k, v in out.items()}

    def summarize(self, rows):
        res = {}
        exact = rows["exact"] > 0.5
        for gname, m in self.groups.items():
            if not bool(m.any()):
                continue
            res[f"acc_{gname}"] = float(exact[m].float().mean())
            for key in ("min_margin", "first_margin", "min_margin11"):
                res[f"{key}_mean_{gname}"] = float(rows[key][m].mean())
            # a threshold proxy, not a measured gradient: past the fp32 cliff, the CE target gradient underflows.
            # cliff_seq = share of sequences with any answer position past it; cliff_pos = share of positions.
            res[f"cliff_seq_frac_{gname}"] = float(rows["cliff_any"][m].mean())
            res[f"cliff_pos_frac_{gname}"] = float(rows["cliff_n"][m].sum() / rows["n_pos"][m].sum())
            # strict: a pair counts only if all 4 templates are exact
            pairs = self.pair_of_row[m]
            # float, not bool: CUDA's scatter_reduce has no bool kernel (the CPU one does)
            allok = torch.ones(int(pairs.max()) + 1, dtype=torch.float32, device=pairs.device)
            allok.scatter_reduce_(0, pairs, exact[m].float(), reduce="amin", include_self=True)
            res[f"acc_all4_{gname}"] = float(allok[pairs.unique()].mean())
        res["acc_test"] = float(exact[self.groups["test_diag"] | self.groups["test_off"]].float().mean())
        res["acc_train"] = float(exact[self.groups["train_diag"] | self.groups["train_off"]].float().mean())
        # gaps (off-diagonal minus diagonal mean min-margin), full vocab and 11-class
        for side in ("train", "test"):
            for key in ("min_margin", "min_margin11"):
                if f"{key}_mean_{side}_diag" not in res:  # gcd trains every diagonal pair: no test diagonal
                    continue
                off, dg = res[f"{key}_mean_{side}_off"], res[f"{key}_mean_{side}_diag"]
                res[f"gap_{key}_{side}"] = off - dg
                res[f"relgap_{key}_{side}"] = (off - dg) / off if off else None
        return res

    @staticmethod
    @torch.no_grad()
    def layer_norms(model):
        """Per layer: the MLP's norm-gain norm (ff_norm.weight; G0 = sqrt(d) at init) and the Frobenius norm of each
        MLP and attention matrix -- the readouts of the grok_lab MLP-disengagement studies."""
        out = {}
        for i, blk in enumerate(getattr(model, "layers", [])):
            for name, mod in (("ff_norm", getattr(blk, "ff_norm", None)),):
                if mod is not None and getattr(mod, "weight", None) is not None:
                    out[f"g_ff_norm@{i}"] = float(mod.weight.float().norm())
            for pname, q in blk.named_parameters():
                if q.dim() == 2 and (pname.startswith("ff.") or pname.startswith("attn.")):
                    out[f"wnorm_{pname}@{i}"] = float(q.float().norm())
        return out

    @torch.no_grad()
    def verify_head(self, model, n_batches=4):
        """The scorer's shortcut (final-norm states, real head applied to gathered positions) against the full
        forward's logits at the same positions, under the same ctx. Returns max |diff| and argmax agreement."""
        was_training = model.training
        model.eval()
        worst, agree, n, bitwise = 0.0, 0, 0, True
        try:
            for ids, r, j, t in self.batches[:n_batches]:
                with self.ctx:
                    full, _, _ = model(ids)
                    zf = full[r, j].float()
                with self.ctx, head_off(model) as head:
                    h, _, _ = model(ids)
                    zg = head(h[r, j]).float()
                worst = max(worst, float((zf - zg).abs().max()))
                bitwise = bitwise and bool(torch.equal(zf, zg))
                agree += int((zf.argmax(1) == zg.argmax(1)).sum())
                n += int(zf.shape[0])
                # free this batch's full-vocab logits before the next forward: otherwise two batches are alive
                # at once (~2 GB each at 512 x 32 x 32k fp32); outputs unchanged (memspike_2026-11-07)
                del full, zf, h, zg
        finally:
            model.train(was_training)
        return {"head_check_max_abs": worst, "head_check_argmax_agree": agree / max(n, 1),
                "head_check_bitwise": bitwise, "head_check_positions": n}

    def __call__(self, model, update, tokens_seen, final=False, extra=None):
        t0 = time.perf_counter()
        rows = self.per_row(model)
        rec = {"update": int(update), "tokens_seen": int(tokens_seen), **self.summarize(rows),
               **self.layer_norms(model), **(extra or {})}
        if final:
            rec.update(self.verify_head(model))
        rec["score_s"] = time.perf_counter() - t0
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        if final:
            wrong = (rows["exact"] < 0.5).nonzero().squeeze(1).tolist()
            fail = [list(self.meta[k]) + [float(rows["min_margin"][k]), float(rows["first_margin"][k])]
                    for k in wrong]
            # final_errors.json for add (the one-task name), final_errors_<task>.json otherwise
            name = self.out_path.stem.replace("table_scores", "final_errors") + ".json"
            with open(self.out_path.with_name(name), "w", encoding="utf-8") as f:
                json.dump({"update": int(update), "columns": ["a", "b", "template", "in_train", "transpose_seen",
                                                              "min_margin", "first_margin"], "rows": fail}, f)
        return rec
