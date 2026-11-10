"""Exact resume for train.py's main path (--resume_from): #138, #154, #150.

A rotating checkpoint (checkpoint_step_N.pt) written at the end of an iteration holds, under "resume", everything the
training loop carries into the next iteration: data positions, every RNG, pending gradients, Grokfast EMA, loop and
logger state, the best model as it was then. A resume from it continues as if the run had never stopped ("exact").
Any other resume (an older checkpoint, best/final/interrupt files, changed arguments, code or libraries, data loader
workers) restores what the file has but starts the data loaders afresh; it runs only with --resume_inexact
("inexact"). Every resume is recorded: in metrics.json's hyperparameter row and summary.json ("resumes"), and every
attempt in resume_log.jsonl.

This module holds the parts that need no training state: argument canonicalization, the run fingerprint, file
publishing with sha256 sidecars, the attempt journal, legacy metrics history and test hooks. train.py does the rest.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

RESUME_FORMAT = 1
# arguments that may differ between a run and its resume: they do not change what is trained
FREE_ARGS = frozenset({"resume_from", "resume_inexact", "output_dir"})
JOURNAL = "resume_log.jsonl"
# fingerprint fields recorded but not compared (#160 made every draw independent of the hash seed)
RECORDED_ONLY = frozenset({"pythonhashseed", "hash_randomization", "ignore_environment", "cuda_initialized_before"})


class ResumeRefused(SystemExit):
    """A resume that must stop before any output file changes."""


# ---------------------------------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------------------------------

def canonical_args(args) -> Any:
    """The normalizations train() used to make to args in place, in one idempotent place, so the arguments saved in
    a checkpoint and the ones compared on resume are the same values. Called by main() and at the top of train()."""
    if getattr(args, "mix_band_tokens", None) is None:
        args.mix_band_tokens = int(0.5 * args.alg_tokens)
    if getattr(args, "disable_dynamic_task_weighting", False):
        args.task_weighting = "uniform"
    for name, default in (("resume_inexact", False), ("cuda_deterministic", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def plain(obj: Any) -> Any:
    """obj as JSON-plain values (no tensors, numpy values or objects): raises TypeError on anything else."""
    return json.loads(json.dumps(obj))


def arg_differences(saved: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    keys = sorted(set(saved) | set(current))
    return [f"--{k}: {saved.get(k, '<absent>')!r} -> {current.get(k, '<absent>')!r}" for k in keys
            if k not in FREE_ARGS and saved.get(k, "<absent>") != current.get(k, "<absent>")]


# ---------------------------------------------------------------------------------------------------------------
# CUDA numerical policy (--cuda_deterministic) and the run fingerprint
# ---------------------------------------------------------------------------------------------------------------

def apply_cuda_policy() -> None:
    """--cuda_deterministic: one fixed numerical policy, set before any device probe. Strict determinism (errors, never
    warn_only), no TF32, no reduced-precision reductions or FP16 accumulation, math-only SDPA."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if _has_fp16_accumulation():  # torch >= 2.7; older torch has no such setting
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)


def _has_fp16_accumulation() -> bool:
    return hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")


def policy_value_ok(have: Any, want: Any) -> bool:
    """A read-back setting meets the exact policy: equal, or absent from this torch (None) where the policy wants it
    off. A setting torch does not have cannot be on."""
    return have == want or (want is False and have is None)


def numerical_policy() -> Dict[str, Any]:
    """The numerical settings torch reports now (read back, not assumed). Plain values only."""
    m = torch.backends.cuda.matmul
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(m.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "fp16_reduced_precision_reduction": bool(m.allow_fp16_reduced_precision_reduction),
        "bf16_reduced_precision_reduction": bool(m.allow_bf16_reduced_precision_reduction),
        "fp16_accumulation": bool(m.allow_fp16_accumulation) if _has_fp16_accumulation() else None,
        "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "cudnn_sdp": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
        "math_sdp_reduced_precision": bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()),
        # the effective preference, read back (its setter is unsupported on Windows; never called here)
        "preferred_blas_library": str(getattr(torch.backends.cuda.preferred_blas_library(), "name",
                                              torch.backends.cuda.preferred_blas_library())),
        "env_cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "env_torch_blas_prefer_cublaslt": os.environ.get("TORCH_BLAS_PREFER_CUBLASLT"),
        "num_threads": int(torch.get_num_threads()),
    }


CUDA_EXACT_POLICY = {
    "deterministic_algorithms": True, "deterministic_warn_only": False, "cudnn_benchmark": False,
    "cudnn_deterministic": True, "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
    "float32_matmul_precision": "highest", "fp16_reduced_precision_reduction": False,
    "bf16_reduced_precision_reduction": False, "fp16_accumulation": False, "flash_sdp": False,
    "mem_efficient_sdp": False, "cudnn_sdp": False, "math_sdp": True, "math_sdp_reduced_precision": False,
    "env_cublas_workspace_config": ":4096:8",
}


def prepare_numerics(cuda_deterministic: bool) -> Dict[str, Any]:
    """Step 1 of train() (design H4), before any device probe: record whether CUDA was already initialized (this does
    not initialize it); with --cuda_deterministic, check the launch environment, apply the policy and read it back
    at once, stopping if torch did not take a setting. Returns {"cuda_initialized_before", "policy"}."""
    before = bool(torch.cuda.is_initialized())
    policy = None
    if cuda_deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise SystemExit("--cuda_deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment, "
                             "set before Python starts")
        apply_cuda_policy()
        policy = numerical_policy()
        off = {k: (policy.get(k), v) for k, v in CUDA_EXACT_POLICY.items() if not policy_value_ok(policy.get(k), v)}
        if off:
            raise SystemExit(f"--cuda_deterministic: torch did not take these settings (now, wanted): {off}")
    return {"cuda_initialized_before": before, "policy": policy}


def _version(dist: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def tokenizer_is_deterministic(tokenizer) -> Optional[str]:
    """None when encoding is deterministic, else why not (BPE dropout, sentencepiece sampling)."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            model = json.loads(backend.to_str()).get("model", {})
        except Exception:
            return "tokenizer backend could not be read"
        if model.get("dropout") not in (None, 0, 0.0):
            return f"tokenizer uses BPE dropout {model.get('dropout')}"
    sp = getattr(tokenizer, "sp_model_kwargs", None) or {}
    if sp.get("enable_sampling"):
        return "tokenizer samples (sentencepiece enable_sampling)"
    return None


def run_fingerprint(code_root: Path, tokenizer, device: torch.device, train_dtype: Optional[torch.dtype],
                    eval_dtype: Optional[torch.dtype], cuda_initialized_before: bool) -> Dict[str, Any]:
    """What must be the same for a resume to be exact: code, libraries, tokenizer, device and numerical policy."""
    from tools.grok_bridge import code_tree_sha256, tokenizer_sha256

    fp: Dict[str, Any] = {
        "code": code_tree_sha256(code_root),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": _version("numpy"),
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "tokenizers": _version("tokenizers"),
        "tokenizer": tokenizer_sha256(tokenizer),
        "device_type": device.type,
        "train_autocast_dtype": str(train_dtype) if train_dtype is not None else None,
        "eval_autocast_dtype": str(eval_dtype) if eval_dtype is not None else None,
        "policy": numerical_policy(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "hash_randomization": int(sys.flags.hash_randomization),
        "ignore_environment": int(sys.flags.ignore_environment),
        "cuda_initialized_before": bool(cuda_initialized_before),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        fp["cuda"] = {
            "name": str(props.name),
            "uuid": str(getattr(props, "uuid", "")),
            "capability": [int(props.major), int(props.minor)],
            "multi_processor_count": int(props.multi_processor_count),
            "max_threads_per_multi_processor": int(getattr(props, "max_threads_per_multi_processor", 0)),
            "total_memory": int(props.total_memory),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": int(torch.backends.cudnn.version() or 0),
            "driver": _cuda_driver_version(),
        }
    return plain(fp)


def _cuda_driver_version() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def fingerprint_differences(saved: Dict[str, Any], now: Dict[str, Any], prefix: str = "") -> List[str]:
    diffs: List[str] = []
    for k in sorted(set(saved) | set(now)):
        if not prefix and k in RECORDED_ONLY:
            continue
        a, b = saved.get(k, "<absent>"), now.get(k, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            diffs.extend(fingerprint_differences(a, b, prefix=f"{prefix}{k}."))
        elif a != b:
            diffs.append(f"{prefix}{k}: {a!r} -> {b!r}")
    return diffs


def cuda_exact_problems(fp: Dict[str, Any], cuda_deterministic: bool) -> List[str]:
    """Why a CUDA run's checkpoints cannot be exact (empty on CPU)."""
    if fp.get("device_type") != "cuda":
        return []
    why = []
    if not cuda_deterministic:
        why.append("CUDA run without --cuda_deterministic")
    if fp.get("cuda_initialized_before"):
        why.append("CUDA was initialized before train() set the numerical policy")
    pol = fp.get("policy", {})
    for k, v in CUDA_EXACT_POLICY.items():
        if not policy_value_ok(pol.get(k), v):
            why.append(f"CUDA policy {k} is {pol.get(k)!r}, exact needs {v!r}")
    return why


# ---------------------------------------------------------------------------------------------------------------
# test hooks (env-gated; read only when set)
# ---------------------------------------------------------------------------------------------------------------

class TestHooks:
    """RELSM_TEST_KILL_STEP="MICROSTEP:POINT" ends the process with os._exit(86) when the run reaches POINT at that
    micro-step (POINT "interrupt" raises KeyboardInterrupt instead). RELSM_TEST_KILL_MARKER=PATH makes it one-shot
    and records where it fired. Points: step (after a micro-step's backward/optimizer step), log (a log row saved),
    eval (IID eval logged, before train-eval), end (end of the iteration), interrupt, and file operations named
    ROLE:OP, e.g. ckpt:save (staged, before its os.replace), ckpt:sidecar_removed, ckpt:payload (replaced, before
    its sidecar), ckpt:prune#K, best:payload, final:payload, summary:write, rs:summary, rs:move#K, rs:logger,
    rs:best (startup of a resume). Separate from the bridge's RELSM_TEST_KILL.
    RELSM_TEST_BEST_ACC=X makes the best-model check see accuracy X * micro-step (the logged metrics keep the measured
    accuracy), so a short test run writes best_model.pt at its first IID evaluation and again at each later one."""

    def __init__(self) -> None:
        spec = os.environ.get("RELSM_TEST_KILL_STEP", "")
        step, _, point = spec.partition(":")
        self.step = int(step) if step.strip() else None
        self.point = point or "end"
        self.marker = os.environ.get("RELSM_TEST_KILL_MARKER")
        self.batch_log = os.environ.get("RELSM_TEST_BATCH_LOG")
        self.digest = os.environ.get("RELSM_TEST_DIGEST")
        self.expect_device = os.environ.get("RELSM_TEST_EXPECT_DEVICE")
        best = os.environ.get("RELSM_TEST_BEST_ACC", "").strip()
        self.best_acc = (lambda step, x=float(best): x * step) if best else None

    def __call__(self, step: int, point: str, path: Optional[Path] = None) -> None:
        if self.step is None or step != self.step or point != self.point:
            return
        if self.marker:
            if Path(self.marker).exists():
                return
            Path(self.marker).write_text(json.dumps({"step": step, "point": point,
                                                      "path": None if path is None else str(path)}) + "\n",
                                         encoding="utf-8")
        if point == "interrupt":
            print(f"[test] RELSM_TEST_KILL_STEP: KeyboardInterrupt at micro-step {step}", flush=True)
            raise KeyboardInterrupt
        print(f"[test] RELSM_TEST_KILL_STEP: os._exit at micro-step {step} ({point})", flush=True)
        os._exit(86)

    def log_batch(self, micro_step: int, batch: Dict[str, Any]) -> None:
        if not self.batch_log:
            return
        import hashlib

        h = hashlib.sha256()
        for key in sorted(batch):
            v = batch[key]
            h.update(key.encode() + b"\0")
            if torch.is_tensor(v):
                t = v.detach().cpu().contiguous()
                h.update(str((t.dtype, tuple(t.shape))).encode())
                h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
            else:
                h.update(json.dumps(v, default=str).encode())
        with open(self.batch_log, "a", encoding="utf-8") as f:
            f.write(f"{micro_step} {h.hexdigest()}\n")


# ---------------------------------------------------------------------------------------------------------------
# files: atomic writes, sha256 sidecars
# ---------------------------------------------------------------------------------------------------------------

def sidecar(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return  # directories cannot be opened for fsync on Windows
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


REPLACE_TRIES = 5
REPLACE_FIRST_SLEEP_S = 0.0625


def replace_retrying(src: Path, dst: Path, tries: int = REPLACE_TRIES,
                     first_sleep_s: float = REPLACE_FIRST_SLEEP_S) -> None:
    """os.replace(src, dst), retried on PermissionError. On Windows the replace fails while another process has dst
    (or src) open: antivirus, the search indexer, a tool reading the output folder. Such a lock lasts moments, so
    this tries up to `tries` times, sleeping first_sleep_s and doubling between tries (0.0625 + 0.125 + 0.25 + 0.5 s,
    under a second, by default), then raises the last error so a lasting failure still stops the run."""
    delay = first_sleep_s
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def atomic_write_bytes(path: Path, data: bytes, before_replace: Optional[Callable[[], None]] = None) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if before_replace is not None:
        before_replace()
    replace_retrying(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def serialize(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def publish(path: Path, step: int, kill: Callable[[str], None], role: str, data: Optional[bytes] = None,
            obj: Any = None) -> Dict[str, Any]:
    """Write a checkpoint (its bytes, or obj through torch.save) to path, then its sidecar. A name that already has a
    sidecar (a rewound run writing the same step again) loses it first, so a sidecar never stands beside a payload it
    does not describe. Kill points: ROLE:sidecar_removed, ROLE:save (staged, before the replace), ROLE:payload
    (replaced, before the new sidecar)."""
    import hashlib

    path = Path(path)
    sc = sidecar(path)
    if sc.exists():
        sc.unlink()
        kill(f"{role}:sidecar_removed")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        if data is not None:
            f.write(data)
        else:
            torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if data is not None:
        meta = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "step": int(step)}
    else:
        meta = {"sha256": file_sha256(tmp), "size": tmp.stat().st_size, "step": int(step)}
    kill(f"{role}:save")
    replace_retrying(tmp, path)
    _fsync_dir(path.parent)
    kill(f"{role}:payload")
    atomic_write_text(sc, json.dumps(meta))
    return meta


def check_sidecar(path: Path, sha256: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """(why, sidecar): why is None when path has a sidecar that matches it, "no sidecar" when it has none, else what
    is wrong. sha256: the file's, when already computed."""
    path = Path(path)
    sc = sidecar(path)
    if not sc.exists():
        return "no sidecar", None
    try:
        meta = json.loads(sc.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return "unreadable sidecar", None
    if not isinstance(meta, dict) or type(meta.get("size")) is not int or not isinstance(meta.get("sha256"), str):
        return "malformed sidecar", None
    size = path.stat().st_size
    if size != meta["size"]:
        return f"size {size} differs from its sidecar's {meta['size']}", meta
    if (sha256 or file_sha256(path)) != meta["sha256"]:
        return "sha256 differs from its sidecar", meta
    return None, meta


def preflight(src: Path, output_dir: Path) -> Dict[str, Any]:
    """Everything --resume_from checks before train() builds anything: the file, its sidecar, a safe load, the
    attempt journal. Read-only; ResumeRefused on any failure."""
    src = Path(src)
    if not src.is_file():
        raise ResumeRefused(f"--resume_from: {src} is not a file")
    sha = file_sha256(src)
    why, meta = check_sidecar(src, sha)
    if why == "no sidecar":
        print(f"[resume] {src.name} has no sidecar (written before exact resume, or killed before its sidecar)")
    elif why is not None:
        raise ResumeRefused(f"--resume_from: {src}: {why}. A named checkpoint is never swapped for another file; "
                            f"pick another checkpoint or move this one aside")
    from utils import load_checkpoint

    try:
        ckpt = load_checkpoint(src, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any load failure is a refusal with its reason
        raise ResumeRefused(f"--resume_from: {src} could not be loaded with weights_only=True "
                            f"({type(exc).__name__}: {exc})") from None
    if meta is not None and int(meta.get("step", -1)) != int(ckpt.get("step", -2)):
        raise ResumeRefused(f"--resume_from: {src}: its sidecar names step {meta.get('step')}, the file step "
                            f"{ckpt.get('step')}")
    r = ckpt.get("resume")
    if r is not None and (not isinstance(r, dict) or r.get("format") != RESUME_FORMAT):
        fmt = r.get("format") if isinstance(r, dict) else type(r).__name__
        raise ResumeRefused(f"--resume_from: {src}: unknown resume format {fmt!r}")
    if ckpt.get("run_id") is not None and not isinstance(ckpt["run_id"], str):
        raise ResumeRefused(f"--resume_from: {src}: run_id is not a string")
    journal = Path(output_dir) / JOURNAL
    records, tail = read_journal(journal)
    return {"src": src, "sha256": sha, "sidecar": meta, "ckpt": ckpt, "journal": journal,
            "journal_records": len(records), "journal_tail": tail}


# ---------------------------------------------------------------------------------------------------------------
# attempt journal (resume_log.jsonl)
# ---------------------------------------------------------------------------------------------------------------

def read_journal(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """(records, tail): every complete line must parse as one JSON object, else ResumeRefused (a corrupt journal stops
    before anything is written). tail is None, "frame" (the last line is one complete object without its newline: it
    counts and gets its newline) or "torn" (a cut-off last line: kept aside and cut off before the next append).
    Read-only."""
    path = Path(path)
    if not path.exists():
        return [], None
    data = path.read_bytes()
    lines = data.split(b"\n")
    body, last = lines[:-1], lines[-1]
    records = []
    for i, line in enumerate(body):
        try:
            rec = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ResumeRefused(f"--resume_from: {path} line {i + 1} is not JSON; the attempt journal is corrupt "
                                f"(move it aside to continue)") from None
        if not isinstance(rec, dict):
            raise ResumeRefused(f"--resume_from: {path} line {i + 1} is not a JSON object")
        records.append(rec)
    if not last:
        return records, None
    try:
        rec = json.loads(last.decode("utf-8"))
        if isinstance(rec, dict):
            records.append(rec)
            return records, "frame"
    except (ValueError, UnicodeDecodeError):
        pass
    return records, "torn"


def repair_journal(path: Path, tail: Optional[str]) -> None:
    """Make the journal end on a record boundary (after validation only)."""
    path = Path(path)
    if tail == "frame":
        with open(path, "ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
    elif tail == "torn":
        data = path.read_bytes()
        cut = data.rfind(b"\n") + 1
        n = 1
        while path.with_name(f"{path.name}.torn.{n}").exists():
            n += 1
        atomic_write_bytes(path.with_name(f"{path.name}.torn.{n}"), data[cut:])
        with open(path, "r+b") as f:
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())


def append_journal(path: Path, record: Dict[str, Any]) -> None:
    """One record as one line: a single write, flush and fsync."""
    line = (json.dumps(plain(record)) + "\n").encode("utf-8")
    with open(path, "ab") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------------------------------------------
# metrics.json history
# ---------------------------------------------------------------------------------------------------------------

TOKEN_KEY = "train/throughput/tokens_seen"
# task-curriculum checkpoint fields derived from other settings, or defaults with no argument of their own
CURRICULUM_DERIVED = frozenset({"init_difficulty", "ema_decay", "step_size"})
# checkpoint curriculum_config / task_curriculum_config field -> hyperparameter-row argument
CURRICULUM_ARG = {"min_task_evals": "curriculum_min_task_evals"}


def read_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """metrics.json as {"hyperparameters", "records", "task_accuracy": {target: {task: [entries]}}}, None if absent or
    unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    out: Dict[str, Any] = {"hyperparameters": None, "records": [],
                           "task_accuracy": {"eval": {}, "eval_ood": {}, "train_eval": {}}}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.pop("type", None)
            if kind == "hyperparameters":
                out["hyperparameters"] = row.get("data")
            elif kind == "record":
                out["records"].append(row)
            elif kind == "task_accuracy":
                target = row.pop("target", "eval")
                task = row.pop("task")
                out["task_accuracy"].setdefault(target, {}).setdefault(task, []).append(row)
    except (ValueError, KeyError, UnicodeDecodeError):
        return None
    return out


def legacy_history(metrics: Optional[Dict[str, Any]], checkpoint: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """History for a resume from a checkpoint without logger state (written before exact resume): the rows of the
    source directory's metrics.json up to the checkpoint step, if that file is compatible with the checkpoint.
    Compatibility, not proof: an old checkpoint carries no run identity, so adopted history is "unverified".
    Returns (history or None, why)."""
    if metrics is None or not isinstance(metrics.get("hyperparameters"), dict):
        return None, "no readable metrics.json with a hyperparameter row"
    row = metrics["hyperparameters"]
    step = int(checkpoint.get("step", 0))
    tokens = int(checkpoint.get("curriculum_tokens_seen", checkpoint.get("tokens_seen", 0)))
    for section in ("curriculum_config", "task_curriculum_config"):
        for k, v in (checkpoint.get(section) or {}).items():
            if section == "task_curriculum_config" and k in CURRICULUM_DERIVED:
                continue
            arg = CURRICULUM_ARG.get(k, k)
            if arg not in row:
                continue
            a = row[arg]
            same = (float(a) == float(v)) if isinstance(v, (int, float)) and isinstance(a, (int, float)) else a == v
            if not same:
                return None, f"metrics.json was written with {arg}={a!r}, the checkpoint has {v!r}"
    later_tokens = None
    for r in metrics["records"]:
        t = r.get(TOKEN_KEY)
        if t is None:
            continue
        if r.get("step", 0) <= step and t > tokens:
            return None, f"metrics.json row at step {r.get('step')} has more tokens than the checkpoint"
        if r.get("step", 0) > step and later_tokens is None:
            later_tokens = t
    if later_tokens is not None and later_tokens <= tokens:
        return None, "metrics.json rows after the checkpoint do not follow it"
    keep = lambda entries: [e for e in entries if e.get("step", 0) <= step]  # noqa: E731
    return {
        "hyperparameters": dict(row),
        "records": keep(metrics["records"]),
        "task_accuracies": {k: keep(v) for k, v in metrics["task_accuracy"].get("eval", {}).items()},
        "ood_task_accuracies": {k: keep(v) for k, v in metrics["task_accuracy"].get("eval_ood", {}).items()},
        "train_eval_task_accuracies": {k: keep(v) for k, v in metrics["task_accuracy"].get("train_eval", {}).items()},
    }, "compatible"


def next_free(path: Path, stem_suffix: str) -> Path:
    n = 1
    while path.with_name(f"{path.name}.{stem_suffix}.{n}").exists():
        n += 1
    return path.with_name(f"{path.name}.{stem_suffix}.{n}")


def new_run_id() -> str:
    return str(uuid.uuid4())  # os.urandom: draws from no RNG the run uses


def attempt_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------------------------------------------

def python_rng_state() -> Any:
    return random.getstate()


def set_python_rng_state(state: Any) -> None:
    version, internal, gauss = state
    random.setstate((version, tuple(internal), gauss))


def non_persistent_buffers(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Buffers that state_dict() leaves out (RoPE inv_freq and its cos/sin caches, which can grow), by name."""
    persistent = set(model.state_dict().keys())
    return {name: buf.detach().cpu().clone() for name, buf in model.named_buffers() if name not in persistent}


def set_non_persistent_buffers(model: torch.nn.Module, buffers: Dict[str, torch.Tensor]) -> None:
    current = dict(model.named_buffers())
    persistent = set(model.state_dict().keys())
    want = {n for n in current if n not in persistent}
    if set(buffers) != want:
        raise ValueError(f"non-persistent buffers differ: saved {sorted(buffers)[:5]}, model {sorted(want)[:5]}")
    for name, value in buffers.items():
        owner_name, _, attr = name.rpartition(".")
        owner = model.get_submodule(owner_name) if owner_name else model
        old = getattr(owner, attr)
        owner.register_buffer(attr, value.to(device=old.device, dtype=old.dtype).clone(), persistent=False)


def tensor_digest(t: torch.Tensor) -> str:
    from tools.grok_bridge import tensor_sha256

    return tensor_sha256(t)


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone() if data else torch.zeros(0, dtype=torch.uint8)


def validate_exact_state(ckpt: Dict[str, Any], r: Dict[str, Any]) -> None:
    """A checkpoint with resume state must be complete and self-consistent before anything is restored or written
    (ValueError: the resume stops, nothing changed). What it must hold follows from the run that wrote it (its own
    arguments and fingerprint), not from this run's. Legacy files (no resume state) keep their optional fields."""
    if bool(r["exact"]) == bool(r["why_inexact"]):
        raise ValueError(f"its exact flag ({r['exact']!r}) and its reasons ({r['why_inexact']!r}) disagree")
    if int(r["micro_steps"]) != int(ckpt.get("step", -1)):
        raise ValueError("its resume state and its step disagree")
    src_args = r["args"]
    had_curriculum = bool(src_args.get("use_task_curriculum"))
    need = ["model_state_dict", "optimizer_state_dict", "scaler_state_dict", "curriculum_rng_state",
            "curriculum_tokens_seen", "tokens_seen", "best_alg_acc"]
    if had_curriculum:
        need.append("curriculum_state")
        if src_args.get("task_curriculum_strategy") == "dag":
            need.append("dag_state")
    if had_curriculum != (r.get("snapshots") is not None):
        raise ValueError("its curriculum snapshots and its own arguments disagree")
    missing = [k for k in need if ckpt.get(k) is None]
    if missing:
        raise ValueError(f"it lacks {', '.join(missing)}")
    # (its per-parameter state is legitimately empty before the first optimizer step)
    if not isinstance(ckpt["optimizer_state_dict"].get("state"), dict) or not ckpt["optimizer_state_dict"].get(
            "param_groups"):
        raise ValueError("its optimizer state is malformed (no state dict or no parameter groups)")
    # the sampler's position is saved only when it can be captured (not with worker loaders): its absence makes the
    # checkpoint inexact, which the exact flag already says; an exact checkpoint without it is malformed
    if r["exact"] and r.get("sampler") is None:
        raise ValueError("it is marked exact but holds no data sampler position")
    counters = {"step tokens_seen": ckpt["tokens_seen"], "resume tokens_seen": r["tokens_seen"],
                "curriculum_tokens_seen": ckpt["curriculum_tokens_seen"]}
    if r.get("sampler") is not None:
        counters["sampler tokens_seen"] = r["sampler"].get("tokens_seen")
    if len({None if v is None else int(v) for v in counters.values()}) != 1:
        raise ValueError(f"its token counters disagree: {counters}")
    if float(r["loop"]["best_alg_acc"]) != float(ckpt["best_alg_acc"]):
        raise ValueError("its best accuracy and its loop state disagree")
    # a CUDA run's checkpoints carry its CUDA stream, bound to the device in its fingerprint; a CPU run's carry none
    fp = r["fingerprint"]
    cuda = r["rng"].get("cuda")
    if fp.get("device_type") == "cuda":
        if cuda is None:
            raise ValueError("it was written on CUDA but holds no CUDA RNG state")
        if str(cuda.get("uuid")) != str((fp.get("cuda") or {}).get("uuid")):
            raise ValueError("its CUDA RNG state belongs to another device than its fingerprint's")
    elif cuda is not None:
        raise ValueError(f"it was written on {fp.get('device_type')!r} but holds a CUDA RNG state")
    validate_rng(r["rng"])


def validate_rng(rng: Dict[str, Any], device: Optional[torch.device] = None) -> None:
    """Every saved RNG state must load into a fresh generator of its kind (ValueError otherwise), so a malformed
    state stops the resume before the startup file steps, not after them. The CUDA state loads into a temporary
    generator on device when it is a CUDA device (it is restored there); elsewhere only its shape is checked."""
    import numpy as np

    try:
        torch.Generator().set_state(rng["torch"])
        version, internal, gauss = rng["python"]
        random.Random().setstate((version, tuple(internal), gauss))
        st = rng["numpy"]
        np.random.RandomState().set_state((st["name"], np.array(st["keys"], dtype=np.uint32), st["pos"],
                                           st["has_gauss"], st["cached"]))
        cuda = rng.get("cuda")
        if cuda is not None:
            if not (torch.is_tensor(cuda["state"]) and cuda["state"].dtype == torch.uint8 and cuda["state"].dim() == 1
                    and cuda["state"].numel() > 0):
                raise ValueError("CUDA state is not a nonempty uint8 vector")
            int(cuda["index"])
            str(cuda["uuid"])
            if device is not None and device.type == "cuda":
                torch.Generator(device=device).set_state(cuda["state"])
    except Exception as exc:  # noqa: BLE001 - any way a state fails to load (numpy raises IndexError too) is malformed
        raise ValueError(f"its RNG state is malformed ({type(exc).__name__}: {exc})") from None


def recover_legacy_best(src: Path, checkpoint: Dict[str, Any],
                        run_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """The best model at a checkpoint without resume.best (legacy, best, final or interrupt file): None with "none"
    when there was no best; the source directory's best_model.pt, re-serialized in the safe format, when its step is
    at or before the checkpoint's and its accuracy equals the checkpoint's best_alg_acc ("recovered"); else
    {"unknown": True} ("unknown")."""
    from model import config_to_payload
    from utils import load_checkpoint

    best_acc = float(checkpoint.get("best_alg_acc", 0.0) or 0.0)
    if best_acc <= 0.0:
        return None, "none"
    path = Path(src).parent / "best_model.pt"
    if not path.is_file():
        return {"unknown": True}, "unknown"
    try:
        best = load_checkpoint(path, map_location="cpu")
    except Exception:  # noqa: BLE001 - an unreadable best file is an unknown best
        return {"unknown": True}, "unknown"
    acc = best.get("alg_accuracy", best.get("best_alg_acc"))
    if int(best.get("step", 1 << 62)) > int(checkpoint.get("step", 0)) or acc is None or float(acc) != best_acc:
        return {"unknown": True}, "unknown"
    if best.get("config") is not None:
        best["config"] = config_to_payload(best["config"])
    if run_id is not None and not isinstance(best.get("run_id"), str):
        best["run_id"] = str(run_id)  # the continuation's id (H3); the original file is left as it is
    data = serialize(best)
    import hashlib

    return {"bytes": bytes_tensor(data), "sha256": hashlib.sha256(data).hexdigest(),
            "step": int(best.get("step", 0))}, "recovered"


def checkpoint_step_of(name: str) -> Optional[int]:
    """N for checkpoint_step_N.pt / checkpoint_step_N_TAG.pt, else None."""
    import re

    m = re.fullmatch(r"checkpoint_step_(\d+)(_[A-Za-z0-9]+)?\.pt", name)
    return int(m.group(1)) if m else None


def _digestable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        from tools.grok_bridge import tensor_sha256

        return "tensor:" + tensor_sha256(obj)
    if isinstance(obj, dict):
        return {str(k): _digestable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_digestable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return f"float:{obj!r}"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return f"{type(obj).__name__}:{obj!r}"


def digest_of(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(_digestable(obj), sort_keys=True).encode()).hexdigest()


def write_digest(path: Path, model, optimizer, grokfast_ema, scaler, device: torch.device,
                 autocast_dtype: Optional[torch.dtype], micro_steps: int, tokens_seen: int) -> None:
    """RELSM_TEST_DIGEST: digests of the state an exact resume must reproduce, at the end of train()."""
    import numpy as np

    names = {id(p): n for n, p in model.named_parameters()}
    out = {
        "micro_steps": int(micro_steps),
        "tokens_seen": int(tokens_seen),
        "device": str(device) if device.type != "cuda" else f"cuda:{torch.cuda.current_device()}",
        "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
        "model": digest_of(model.state_dict()),
        "optimizer": digest_of(optimizer.state_dict()),
        "optimizer_bindings": [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups],
        "grokfast_ema": digest_of(dict(grokfast_ema)),
        "buffers": digest_of(non_persistent_buffers(model)),
        "grads": digest_of({n: p.grad for n, p in model.named_parameters()}),
        "scaler": digest_of(scaler.state_dict()),
        "rng_python": digest_of(random.getstate()),
        "rng_numpy": digest_of([np.random.get_state()[0], np.random.get_state()[1].tolist(),
                                *np.random.get_state()[2:]]),
        "rng_torch": digest_of(torch.get_rng_state()),
        "rng_cuda": digest_of(torch.cuda.get_rng_state(device)) if device.type == "cuda" else None,
    }
    atomic_write_text(Path(path), json.dumps(out, indent=1, sort_keys=True))
