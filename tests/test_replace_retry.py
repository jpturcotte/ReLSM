"""Atomic writes (metrics.json, checkpoints, summary.json, the bridge's files) survive a brief Windows file lock on
the final os.replace."""

import ast
import json
import os
from pathlib import Path

import pytest
import torch

import exact_resume as er
from train import MetricsLogger


def _flaky_replace(monkeypatch, failures):
    """Patch os.replace to raise PermissionError `failures` times (None: always), then behave normally."""
    real_replace = os.replace
    calls = []

    def fake(src, dst):
        calls.append((src, dst))
        if failures is None or len(calls) <= failures:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fake)
    sleeps = []
    monkeypatch.setattr(er.time, "sleep", sleeps.append)
    return calls, sleeps


def test_metrics_write_lands_after_two_permission_errors(tmp_path, monkeypatch):
    logger = MetricsLogger(tmp_path)
    logger.log(step=7, phase="train", loss=1.5)
    calls, sleeps = _flaky_replace(monkeypatch, failures=2)

    logger.save()

    assert len(calls) == 3
    assert sleeps == [er.REPLACE_FIRST_SLEEP_S, er.REPLACE_FIRST_SLEEP_S * 2]
    rows = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert {"type": "record", "step": 7, "phase": "train", "loss": 1.5} in rows
    assert not (tmp_path / "metrics.json.tmp").exists()


def test_metrics_write_raises_when_the_lock_lasts(tmp_path, monkeypatch):
    logger = MetricsLogger(tmp_path)
    before = (tmp_path / "metrics.json").read_text()
    logger.log(step=7, phase="train", loss=1.5)
    calls, sleeps = _flaky_replace(monkeypatch, failures=None)

    with pytest.raises(PermissionError):
        logger.save()

    assert len(calls) == er.REPLACE_TRIES
    assert len(sleeps) == er.REPLACE_TRIES - 1
    assert sum(sleeps) < 1.0
    assert (tmp_path / "metrics.json").read_text() == before


def test_atomic_write_bytes_retries(tmp_path, monkeypatch):
    calls, _ = _flaky_replace(monkeypatch, failures=2)
    er.atomic_write_bytes(tmp_path / "summary.json", b"{}")
    assert len(calls) == 3
    assert (tmp_path / "summary.json").read_bytes() == b"{}"


def test_other_errors_are_not_retried(tmp_path, monkeypatch):
    calls = []

    def fake(src, dst):
        calls.append(src)
        raise FileNotFoundError(src)

    monkeypatch.setattr(os, "replace", fake)
    with pytest.raises(FileNotFoundError):
        er.replace_retrying(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 1


windows_only = pytest.mark.skipif(os.name != "nt", reason="an open file blocks os.replace onto it only on Windows")


def _release_on_first_lock(monkeypatch, held):
    """Wrap os.replace so a real replace that meets the real lock (PermissionError) closes the held handle before the
    error goes on: the lock is released by the failed try itself, not a timer, so the retry is what must succeed.
    Returns each real replace as (outcome, destination name), outcome "locked" or "ok"."""
    real_replace = os.replace
    outcomes = []

    def wrapped(src, dst):
        try:
            real_replace(src, dst)
        except PermissionError:
            outcomes.append(("locked", Path(dst).name))
            held.close()
            raise
        outcomes.append(("ok", Path(dst).name))

    monkeypatch.setattr(os, "replace", wrapped)
    return outcomes


@windows_only
def test_metrics_write_waits_out_a_real_windows_lock(tmp_path, monkeypatch):
    logger = MetricsLogger(tmp_path)
    logger.log(step=7, phase="train", loss=1.5)
    held = open(tmp_path / "metrics.json", "rb")  # Python opens without FILE_SHARE_DELETE, so a replace onto it fails
    try:
        outcomes = _release_on_first_lock(monkeypatch, held)
        logger.save()
    finally:
        held.close()
    assert outcomes == [("locked", "metrics.json"), ("ok", "metrics.json")]
    rows = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert {"type": "record", "step": 7, "phase": "train", "loss": 1.5} in rows


# tools/grok_bridge.py (the --mod_table_p bridge) writes through the same helper


def _resume_obj(micro_steps):
    from tools import grok_bridge as gb

    return {"format": gb.RESUME_FORMAT, "kind": "test", "micro_steps": micro_steps, "w": torch.zeros(3)}


def test_grok_bridge_writes_retry(tmp_path, monkeypatch):
    from tools import grok_bridge as gb

    calls, _ = _flaky_replace(monkeypatch, failures=2)
    gb.atomic_write_text(tmp_path / "bridge_record.json", "{}")
    assert len(calls) == 3 and (tmp_path / "bridge_record.json").read_text(encoding="utf-8") == "{}"

    calls.clear()
    gb.atomic_write_bytes(tmp_path / "table.bin", b"\x01\x02")
    assert len(calls) == 3 and (tmp_path / "table.bin").read_bytes() == b"\x01\x02"

    calls.clear()
    gb.atomic_torch_save({"a": 1}, tmp_path / "state.pt")
    assert len(calls) == 3 and torch.load(tmp_path / "state.pt", weights_only=True) == {"a": 1}


def test_grok_bridge_publish_resume_raises_when_the_lock_lasts(tmp_path, monkeypatch):
    from tools import grok_bridge as gb

    gb.publish_resume(_resume_obj(1), tmp_path)
    before = (tmp_path / gb.RESUME_FILE).read_bytes()
    calls, _ = _flaky_replace(monkeypatch, failures=None)
    with pytest.raises(PermissionError):
        gb.publish_resume(_resume_obj(2), tmp_path, rotate=False)
    assert len(calls) == er.REPLACE_TRIES
    assert (tmp_path / gb.RESUME_FILE).read_bytes() == before


@windows_only
def test_grok_bridge_publish_resume_waits_out_a_real_windows_lock(tmp_path, monkeypatch):
    from tools import grok_bridge as gb

    gb.publish_resume(_resume_obj(1), tmp_path)
    cur = tmp_path / gb.RESUME_FILE
    held = open(cur, "rb")
    try:
        outcomes = _release_on_first_lock(monkeypatch, held)
        gb.publish_resume(_resume_obj(2), tmp_path)  # rotates resume.pt to resume.prev.pt, then replaces resume.pt
    finally:
        held.close()
    # resume.prev.pt is not held, so its replace goes through; the one onto the held resume.pt is retried
    onto_cur = [outcome for outcome, name in outcomes if name == gb.RESUME_FILE]
    assert onto_cur == ["locked", "ok"] and [o for o, _ in outcomes].count("locked") == 1
    assert gb.check_pair(cur) is None and gb.load_resume(cur)["micro_steps"] == 2
    prev = tmp_path / gb.RESUME_PREV
    assert gb.check_pair(prev) is None and gb.load_resume(prev)["micro_steps"] == 1


def _is_move(node):
    """A rename call: os.replace / os.rename, shutil.move, or a pathlib Path.replace / Path.rename. str.replace takes
    at least two arguments, so a one-argument .replace(x) or .rename(x) can only be a path move."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return f.id in ("replace", "rename", "move")  # from os import replace, from shutil import move
    if not isinstance(f, ast.Attribute):
        return False
    if isinstance(f.value, ast.Name) and (f.value.id, f.attr) in {("os", "replace"), ("os", "rename"), ("shutil", "move")}:
        return True
    return f.attr in ("replace", "rename") and len(node.args) == 1 and not node.keywords


def test_no_bare_rename_left():
    # every rename in these files goes through replace_retrying (atomic writes, and the moves aside on resume and on
    # a fresh bridge attempt); a new one fails here. Comments and docstrings that mention os.replace do not count.
    repo = Path(__file__).resolve().parents[1]
    for rel in ("train.py", "exact_resume.py", "tools/grok_bridge.py"):
        tree = ast.parse((repo / rel).read_text(encoding="utf-8"))
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "replace_retrying":
                allowed.update(id(n) for n in ast.walk(node))
        bare = [node.lineno for node in ast.walk(tree) if _is_move(node) and id(node) not in allowed]
        assert not bare, f"{rel}: a rename outside replace_retrying at lines {bare}; use exact_resume.replace_retrying"


def test_the_rename_guard_sees_every_form():
    src = """
import os, shutil
from os import replace
os.replace(a, b); os.rename(a, b); shutil.move(a, b); replace(a, b); p.replace(q); p.rename(q)
s.replace("x", "y"); s.replace("x", "y", 1)
"""
    assert sum(_is_move(n) for n in ast.walk(ast.parse(src))) == 6
