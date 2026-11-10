"""The run record (#187): what a checkpoint needs to explain itself, as plain values.

Every checkpoint train.py writes (checkpoint_step_N.pt, its _interrupt variant, best_model.pt, final_model.pt) carries
it under "run_record": the command line and the resolved arguments, the git commit and whether the tree was dirty,
the run fingerprint (code hash, library versions, tokenizer, dtypes, numerical policy, GPU), the platform, the start
time, values train() resolves from the arguments (AMP dtype, warmup, train-eval mode, parameter counts), the optimizer
setup, and, on a resume, the file it continued from (with that file's own record) and every earlier file in the chain.
It holds only JSON values (dicts, lists, strings, numbers, booleans, None; a NaN or inf argument stays a float), so a
checkpoint still loads with torch.load(weights_only=True). A resume from a rotating checkpoint compares its "resume"
key, not this record; a resume from a file without exact-resume state (best, final, interrupt) compares only this
record's args, listing and printing each changed argument (train.py). Files written before this change have no
"run_record"; readers use .get("run_record").
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

RECORD_FORMAT = 1
GIT_TIMEOUT_S = 20


def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _git(root: Path, *argv: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *argv], capture_output=True, check=True,
                          timeout=GIT_TIMEOUT_S).stdout


def git_state(root: Path) -> Dict[str, Any]:
    """The commit checked out in root and whether the tree differs from it. Every value is None (with "error" saying
    why) when git or the repository is unavailable. dirty: a tracked file differs from the commit (staged or not);
    diff_sha256: sha256 of `git diff HEAD --binary` when dirty; untracked_entries: the entries `git status` lists as
    untracked (neither tracked nor ignored), a directory with no tracked file counting once, however many files it
    holds (listing every file could take long in a checkout with large untracked folders)."""
    state: Dict[str, Any] = {"commit": None, "dirty": None, "diff_sha256": None, "untracked_entries": None,
                             "error": None}
    try:
        state["commit"] = _git(root, "rev-parse", "HEAD").decode().strip()
        lines = _git(root, "status", "--porcelain=v1", "--untracked-files=normal").decode(errors="replace").splitlines()
        state["dirty"] = any(ln and not ln.startswith("??") for ln in lines)
        state["untracked_entries"] = sum(ln.startswith("??") for ln in lines)
        if state["dirty"]:
            state["diff_sha256"] = hashlib.sha256(_git(root, "diff", "HEAD", "--binary")).hexdigest()
    except Exception as exc:  # no git, not a repository, a timeout: the record says so instead of failing the run
        msg = exc.stderr.decode(errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else ""
        state.update(commit=None, dirty=None, diff_sha256=None, untracked_entries=None,
                     error=f"{type(exc).__name__}: {msg or exc}"[:300])
    return state


def environment() -> Dict[str, Any]:
    """Where the run ran (the library versions are in the fingerprint)."""
    import torch

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "python_implementation": platform.python_implementation(),
        "python_build": sys.version,
        "torch_cuda": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version() or 0) if torch.backends.cudnn.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "num_threads": int(torch.get_num_threads()),
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
                                               "OMP_NUM_THREADS")},
        "pid": os.getpid(),
    }


def optimizer_setup(optimizer) -> Dict[str, Any]:
    """The optimizer as built: class, betas, eps and each group's size (#246, #218, #186). Per-group lr is not kept:
    the schedule rewrites it every step."""
    groups = []
    for g in optimizer.param_groups:
        groups.append({
            "weight_decay": float(g.get("weight_decay", 0.0)),
            "n_tensors": len(g["params"]),
            "n_params": int(sum(p.numel() for p in g["params"])),
        })
    g0 = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = g0.get("betas")
    return {
        "class": type(optimizer).__name__,
        "betas": [float(b) for b in betas] if betas is not None else None,
        "eps": float(g0["eps"]) if g0.get("eps") is not None else None,
        "groups": groups,
    }


def optimizer_steps(optimizer) -> Optional[int]:
    """Optimizer updates applied so far, read from AdamW's per-parameter step counters (0 before the first update;
    None for an optimizer that keeps no step counter). Skipped fp16 updates are not counted, as AdamW does not."""
    steps = [int(s["step"]) for s in optimizer.state.values() if isinstance(s, dict) and "step" in s]
    if steps:
        return max(steps)
    return 0 if all(not s for s in optimizer.state.values()) else None


def model_counts(model) -> Dict[str, Any]:
    """Parameter counts as measured on the built model (tied weights counted once)."""
    params = list(model.parameters())  # parameters() yields a shared tensor once
    tied = None
    lm_head, tok_emb = getattr(model, "lm_head", None), getattr(model, "tok_emb", None)
    if lm_head is not None and tok_emb is not None:
        tied = getattr(lm_head, "weight", None) is getattr(tok_emb, "weight", None)
    return {
        "n_params_unique": int(sum(p.numel() for p in params)),
        "n_params_trainable": int(sum(p.numel() for p in params if p.requires_grad)),
        "n_tensors": len(params),
        "tied_embeddings": tied,
    }


def lineage(src: Path, sha256: Optional[str], kind: str, step: Optional[int],
            record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """{"parent", "ancestors"} for a run that resumed from src: parent is that file (path, sha256, resume kind, step)
    with its own run record (None for a file written before #187); ancestors are the files that record names in turn,
    latest first, each with its record. The chain is kept flat, each record stripped of its own parent and ancestors,
    so it grows by one entry per resume and never nests."""
    record = dict(record) if isinstance(record, dict) else None
    older = []
    if record is not None:
        if isinstance(record.get("parent"), dict):
            older = [record["parent"], *list(record.get("ancestors") or [])]
        record.pop("parent", None)
        record.pop("ancestors", None)
    parent = {"file": str(src), "name": Path(src).name, "sha256": sha256, "kind": kind,
              "step": int(step) if step is not None else None, "record": record}
    return _plain({"parent": parent, "ancestors": older})


def build(*, run_id: str, args: Dict[str, Any], fingerprint: Optional[Dict[str, Any]], git: Dict[str, Any],
          resolved: Dict[str, Any], optimizer: Dict[str, Any], resumed: Optional[Dict[str, Any]],
          argv=None, started: Optional[str] = None, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The record, once per process (the run start); resumed is lineage(...) or None for a fresh run. Save-time
    values are added by at_save."""
    resumed = resumed or {"parent": None, "ancestors": []}
    return _plain({
        "format": RECORD_FORMAT,
        "run_id": run_id,
        "started": started or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": list(sys.argv if argv is None else argv),
        "args": args,
        "seed": args.get("seed"),  # top level, for eval_hub (#227)
        "git": git,
        "fingerprint": fingerprint,
        "environment": environment() if env is None else env,
        "resolved": resolved,
        "optimizer": optimizer,
        "parent": resumed["parent"],
        "ancestors": resumed["ancestors"],
    })


def at_save(record: Dict[str, Any], optimizer, micro_steps: int, tokens_seen: int) -> Dict[str, Any]:
    """record plus what changes from one save to the next."""
    return {**record, "saved": {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "micro_steps": int(micro_steps),
                                "optimizer_steps": optimizer_steps(optimizer), "tokens_seen": int(tokens_seen)}}
