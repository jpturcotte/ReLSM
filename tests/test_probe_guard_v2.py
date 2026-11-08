"""CPU tests for tools/probe_guard_v2.py (no torch).

Most tests drive the real plan loop against FakeSystem: scripted free RAM, commit and disk, and
fake process trees whose memory ramps up while they "run". A few Windows-only tests start small
plain-Python children to check the Job Object measurement on the real machine.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_guard_v2 as pg  # noqa: E402

GB = pg.GB
MB = 2 ** 20
RUNNER_GREP = ("shed", "GIVING UP", "FAILED", "REFUSED", "GUARD", "low disk")  # run_study.run_one()


# ---------------------------------------------------------------- fakes

class FakeProc:
    """One job's tree: a venv-style launcher (4 MB) at pid, the interpreter at pid+1 (memory
    ramps to ws_gb / commit_gb over ramp_s), an optional child at pid+2 and an optional stray at
    pid+3 that outlives the root by stray_s."""

    def __init__(self, fs, pid, spec):
        self.peak_cache = {}
        self.fs, self.pid, self.spec, self.t0 = fs, pid, spec, fs.t
        self.returncode = None
        self.killed_at = None

    def end(self):
        return self.t0 + self.spec.get("duration", 30)

    def _cut(self, t_end):
        return t_end if self.killed_at is None else min(t_end, self.killed_at)

    def intervals(self):
        s = self.spec
        out = {self.pid: (self.t0, self._cut(self.end())), self.pid + 1: (self.t0, self._cut(self.end()))}
        if "child_gb" in s:
            a = self.t0 + s.get("child_start", 5)
            out[self.pid + 2] = (a, self._cut(a + s.get("child_s", 10)))
        if "stray_s" in s:
            out[self.pid + 3] = (self.t0 + 1, self._cut(self.end() + s["stray_s"]))
        return out

    def alive(self, pid, t=None):
        t = self.fs.t if t is None else t
        iv = self.intervals().get(pid)
        return iv is not None and iv[0] <= t < iv[1]

    def mem(self, pid, t=None):
        """(ws, private) in bytes at time t."""
        t = self.fs.t if t is None else t
        if not self.alive(pid, t):
            return 0, 0
        s = self.spec
        if pid == self.pid + 3 and "stray_gb" in s:
            return int(s["stray_gb"] * GB), int(s["stray_gb"] * GB)
        if pid == self.pid or pid == self.pid + 3:
            return 4 * MB, 2 * MB
        if pid == self.pid + 1:
            if "spike_gb" in s and t >= self.t0 + s["duration"] - s.get("spike_s", 1.0):
                return int(s["spike_gb"] * GB), int(s["spike_gb"] * GB)  # just before exit
            f = min(1.0, (t - self.t0) / max(s.get("ramp_s", 10), 1e-9))
            return int(s.get("ws_gb", 1.0) * f * GB), int(s.get("commit_gb", 1.0) * f * GB)
        return int(s["child_gb"] * GB), int(s["child_gb"] * GB)

    def poll(self):
        if self.returncode is None:
            if self.killed_at is not None:
                self.returncode = 1
            elif self.fs.t >= self.end():
                self.returncode = self.spec.get("code", 0)
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is None:
            if timeout is not None and self.end() > self.fs.t + timeout:
                self.fs.t += timeout
                raise subprocess.TimeoutExpired("fake", timeout)
            self.fs.t = max(self.fs.t, self.end())
        return self.poll()

    def kill(self):
        if self.killed_at is None and not self.spec.get("unkillable"):
            self.killed_at = self.fs.t


class FakeJob:
    def __init__(self, proc):
        self.proc = proc
        self.peak_private = 0
        self.closed = False


class FakeSystem:
    def __init__(self, specs, ram=4.0, commit=20.0, disk=50.0, sys_disk=None, total_ram=16.0):
        self.specs = specs
        self.t = 0.0
        self.ram = ram if callable(ram) else (lambda t, v=ram: v)
        self.commit = commit if callable(commit) else (lambda t, v=commit: v)
        self.disk = disk if callable(disk) else (lambda t, v=disk: v)
        sd = disk if sys_disk is None else sys_disk
        self.sys_disk = sd if callable(sd) else (lambda t, v=sd: v)
        self.total_ram = total_ram
        self.procs = []
        self.jobs = []
        self.next_pid = 100
        self.spawned = []  # (name, time)
        self.resources_fail_at = None
        self.open_fails = False

    # time
    def now(self):
        return self.t

    def sleep(self, dt):
        steps = max(1, int(round(dt / 0.05)))  # the Job Object peak commit is exact
        for _ in range(steps):
            self.t = round(self.t + dt / steps, 9)  # no float drift across many steps
            for j in self.jobs:
                if not j.closed:
                    j.peak_private = max(j.peak_private, sum(j.proc.mem(p)[1] for p in j.proc.intervals()))

    # resources
    def resources(self, path):
        if self.resources_fail_at is not None and self.t >= self.resources_fail_at:
            raise OSError("fake resource query failure")
        ws = sum(p.mem(pid)[0] for p in self.procs for pid in p.intervals())
        private = sum(p.mem(pid)[1] for p in self.procs for pid in p.intervals())
        return {"ram_gb": self.ram(self.t) - ws / GB, "commit_gb": self.commit(self.t) - private / GB,
                "commit_limit_gb": 30.0, "total_ram_gb": self.total_ram,
                "disk_gb": self.disk(self.t), "sys_disk_gb": self.sys_disk(self.t)}

    # jobs
    def spawn_in_job(self, argv, env, out_path, err_path, commit_limit_bytes=0):
        name = os.path.basename(out_path)[:-len(".jsonl")]
        spec = self.specs[name]
        if spec.get("spawn_error"):
            raise OSError("fake spawn failure")
        Path(out_path).write_text("")
        proc = FakeProc(self, self.next_pid, spec)
        self.next_pid += 10
        self.procs.append(proc)
        job = FakeJob(proc)
        self.jobs.append(job)
        self.spawned.append((name, self.t))
        return proc, job

    def open_job_process(self, pid, job):
        if self.open_fails:
            raise RuntimeError("fake open failure")
        return (job, pid) if job.proc.alive(pid) else None

    def close(self, h):
        pass

    def process_running(self, h):
        return h is not None and h[0].proc.alive(h[1])

    def process_memory(self, h):
        job, pid = h
        if not job.proc.alive(pid):  # what Windows reports after exit
            return {"ws": 0, "peak_ws": 4 * MB, "private": 0}
        bad = job.proc.spec.get("mem_unreadable_after")
        if bad is not None and self.t >= job.proc.t0 + bad:
            return None
        ws, private = job.proc.mem(pid)
        p = job.proc
        # the OS peak: every 0.05 s since the process started (kept per pid, so it stays cheap)
        i, peak = p.peak_cache.get(pid, (0, 0))
        n = int((self.t - p.t0) / 0.05) + 1
        for k in range(i, n):
            peak = max(peak, p.mem(pid, p.t0 + k * 0.05)[0])
        p.peak_cache[pid] = (max(i, n), peak)
        return {"ws": ws, "peak_ws": max(peak, ws), "private": private}

    def job_pids(self, job):
        bad = job.proc.spec.get("pids_unreadable_after")
        if bad is not None and self.t >= job.proc.t0 + bad:
            return None
        return [pid for pid in job.proc.intervals() if job.proc.alive(pid)]

    def job_total_processes(self, job):
        return sum(1 for a, b in job.proc.intervals().values() if a <= self.t)

    def job_peak_commit(self, job):
        return job.peak_private

    def job_terminate(self, job):
        job.proc.kill()

    def job_close(self, job):
        job.closed = True
        job.proc.kill()  # KILL_ON_JOB_CLOSE



def make_cfg(tmp_path, **over):
    cfg = pg.load_config({"GUARD_MEM_HISTORY": str(tmp_path / "hist.jsonl")})
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def run(tmp_path, plan, fs, **over):
    cfg = make_cfg(tmp_path, **over)
    hist = pg.History(cfg.GUARD_MEM_HISTORY)
    logs = []
    code = pg.run_plan(plan, str(tmp_path / "out"), fs, cfg, hist, logs.append)
    return code, logs, hist


def other_process_can_lock(path):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import probe_guard_v2 as pg; "
            "lk = pg.PlanLock(sys.argv[2]); ok = lk.acquire(); print(int(ok)); ok and lk.release()")
    out = subprocess.run([sys.executable, "-c", code, str(Path(pg.__file__).parent), str(path)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == "1"


def job(name, **kw):
    return {"name": name, "cmd": ["fake_job.py", "--out", name], **kw}


def has(logs, text):
    return any(text in line for line in logs)


# ---------------------------------------------------------------- estimates

def hist_with(tmp_path, rows):
    p = tmp_path / "h.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return pg.History(str(p))


def row(key, ws, commit=2.0, outcome="ok", complete=True, family="fake_job.py"):
    return {"key": key, "family": family, "outcome": outcome, "complete": complete,
            "peak_ws_gb": ws, "peak_commit_gb": commit}


def test_estimate_order_default_family_key(tmp_path):
    cfg = make_cfg(tmp_path)
    j = job("a")
    e = pg.estimate(j, hist_with(tmp_path, []), cfg)
    assert (e["ram_gb"], e["ram_source"]) == (1.3, "default")
    assert (e["commit_gb"], e["commit_source"]) == (3.0, "default")
    e = pg.estimate(j, hist_with(tmp_path, [row("fake_job.py|other", 2.0)]), cfg)
    assert e["ram_gb"] == pytest.approx(2.2) and e["ram_source"] == "family:fake_job.py"
    e = pg.estimate(j, hist_with(tmp_path, [row("fake_job.py|other", 2.0), row("fake_job.py|a", 1.0)]), cfg)
    assert e["ram_gb"] == pytest.approx(1.1) and e["ram_source"] == "history:fake_job.py|a"


def test_estimate_uses_max_of_last_n(tmp_path):
    cfg = make_cfg(tmp_path, HISTORY_N=3)
    rows = [row("fake_job.py|a", v) for v in (9.0, 1.0, 2.0, 1.5)]  # 9.0 is older than the last 3
    assert pg.estimate(job("a"), hist_with(tmp_path, rows), cfg)["ram_gb"] == pytest.approx(2.2)


def test_lower_bounds_only_raise(tmp_path):
    cfg = make_cfg(tmp_path)
    k = "fake_job.py|a"
    # A startup crash (tiny peak) does not lower the default...
    e = pg.estimate(job("a"), hist_with(tmp_path, [row(k, 0.05, outcome="failed")]), cfg)
    assert e["ram_gb"] == 1.3
    # ... and does not evict a full measurement.
    rows = [row(k, 3.0)] + [row(k, 0.05, outcome="failed")] * 6
    assert pg.estimate(job("a"), hist_with(tmp_path, rows), cfg)["ram_gb"] == pytest.approx(3.3)
    # A shed at 2.5 GB raises the default.
    e = pg.estimate(job("a"), hist_with(tmp_path, [row(k, 2.5, outcome="shed")]), cfg)
    assert e["ram_gb"] == pytest.approx(2.75) and e["ram_source"].startswith("lower-bound")
    # A finished but partly measured run is a lower bound too, so it can't hide family history.
    rows = [row("fake_job.py|b", 2.0), row(k, 0.5, complete=False)]
    e = pg.estimate(job("a"), hist_with(tmp_path, rows), cfg)
    assert e["ram_source"] == "family:fake_job.py"


def test_declared_need_vs_history(tmp_path):
    cfg = make_cfg(tmp_path)
    h = hist_with(tmp_path, [row("fake_job.py|a", 4.0)])
    e = pg.estimate(job("a", ram_gb=5.0), h, cfg)
    assert (e["ram_gb"], e["ram_source"], e["warnings"]) == (5.0, "declared", [])
    e = pg.estimate(job("a", ram_gb=1.3), h, cfg)  # below the measured need: measured wins
    assert e["ram_gb"] == pytest.approx(4.4) and "using 4.40 GB" in e["warnings"][0]
    e = pg.estimate(job("a", ram_gb=1.3, mem_override=True), h, cfg)
    assert e["ram_gb"] == 1.3 and "mem_override" in e["warnings"][0]
    e = pg.estimate(job("a", ram_gb=1.3), hist_with(tmp_path, []), cfg)
    assert (e["ram_gb"], e["ram_source"]) == (1.3, "declared")


def test_history_skips_bad_rows(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps(row("fake_job.py|a", 2.0)) + "\n"
                 + json.dumps(row("fake_job.py|a", None)) + "\n"
                 + json.dumps(row("fake_job.py|a", "big")) + "\n"
                 + "[1, 2]\n" + '{"key": "fake_job.py|a", "peak_ws')  # partial last line
    h = pg.History(str(p))
    assert h.peak("peak_ws_gb", 5, key="fake_job.py|a") == 2.0


def test_key_and_family():
    assert pg.job_key({"name": "x", "cmd": ["tools/grok_lab.py"]}) == "grok_lab.py|x"
    assert pg.job_key({"name": "x", "cmd": [], "mem_key": "k"}) == "k"
    assert pg.job_family({"name": "x", "cmd": ["-m", "pkg.mod"]}) == "-m pkg.mod"
    assert pg.job_family({"name": "x", "cmd": ["C:\\a\\train.py", "--x"]}) == "train.py"


def test_config_validation():
    with pytest.raises(ValueError):
        pg.load_config({"ADMIT_TIMEOUT": "maybe"})
    assert pg.load_config({"MIN_RAM_GB": "0.4"}).v1_names_set == ["MIN_RAM_GB"]


# ---------------------------------------------------------------- admission

def test_waits_for_room_then_starts(tmp_path):
    fs = FakeSystem({"a": {"duration": 30, "ws_gb": 1.0}}, ram=lambda t: 1.0 if t < 100 else 3.0)
    code, logs, hist = run(tmp_path, [job("a", ram_gb=1.3)], fs)
    assert code == 0
    assert has(logs, "waiting for memory: a needs free ram >= 1.70")
    (name, t_start), = fs.spawned
    assert 100 <= t_start <= 106  # two checks in a row after room appears
    r = hist.rows[-1]
    assert r["outcome"] == "ok" and r["ram_before_gb"] == pytest.approx(3.0, abs=0.01)
    assert r["ram_min_gb"] < 2.1 and r["need_source"] == "declared / default"


def test_one_tick_spike_does_not_admit(tmp_path):
    fs = FakeSystem({"a": {"duration": 10}},
                    ram=lambda t: 3.0 if 50 <= t < 51 or t >= 200 else 1.0)
    code, logs, _ = run(tmp_path, [job("a", ram_gb=1.3)], fs)
    assert code == 0 and fs.spawned[0][1] >= 200


def test_admission_timeout_fail(tmp_path):
    fs = FakeSystem({"a": {}}, ram=1.0)
    code, logs, hist = run(tmp_path, [job("a", ram_gb=1.3)], fs, ADMIT_WAIT_S=300)
    assert code == 8 and fs.spawned == [] and hist.rows == []
    assert has(logs, "GIVING UP on a: not admitted in 30")


def test_admission_timeout_start(tmp_path):
    fs = FakeSystem({"a": {"duration": 10, "ws_gb": 0.2}}, ram=1.0)
    code, logs, _ = run(tmp_path, [job("a", ram_gb=1.3)], fs, ADMIT_WAIT_S=300, ADMIT_TIMEOUT="start")
    assert code == 0 and has(logs, "GUARD ADMIT_TIMEOUT: starting a")
    assert 300 <= fs.spawned[0][1] <= 304


def test_admission_timeout_start_still_needs_commit(tmp_path):
    fs = FakeSystem({"a": {}}, ram=1.0, commit=4.0)
    code, logs, _ = run(tmp_path, [job("a", ram_gb=1.3)], fs, ADMIT_WAIT_S=300, ADMIT_TIMEOUT="start")
    assert code == 8 and fs.spawned == []


def test_need_larger_than_machine_given_up_at_once(tmp_path):
    fs = FakeSystem({"a": {}}, total_ram=8.0)
    code, logs, _ = run(tmp_path, [job("a", ram_gb=9.0)], fs)
    assert code == 8 and fs.t < 10 and has(logs, "more than this machine has")


def test_pagefile_headroom_blocks_admission(tmp_path):
    # Page-file drive has 11 GB free: 11 - 3 GB commit need < 9 GB floor.
    fs = FakeSystem({"a": {"duration": 10}}, disk=50.0, sys_disk=lambda t: 11.0 if t < 100 else 20.0)
    code, logs, _ = run(tmp_path, [job("a")], fs)
    assert code == 0 and fs.spawned[0][1] >= 100


def test_outstanding_reservation(tmp_path):
    # Two jobs of 2 GB, 4 GB free, MAX_PROCS=2. Just after the first starts free RAM still reads
    # ~4 GB; only the outstanding reservation stops the second from starting next to it.
    specs = {"a": {"duration": 60, "ws_gb": 2.0, "ramp_s": 40}, "b": {"duration": 10, "ws_gb": 2.0}}
    fs = FakeSystem(specs, ram=4.0)
    code, logs, _ = run(tmp_path, [job("a", ram_gb=2.0), job("b", ram_gb=2.0)], fs, MAX_PROCS=2)
    assert code == 0
    (_, ta), (_, tb) = fs.spawned
    assert tb >= ta + 60


def test_cooldown_and_start_gap(tmp_path):
    specs = {"a": {"duration": 2, "ws_gb": 0.1}, "b": {"duration": 2, "ws_gb": 0.1}}
    fs = FakeSystem(specs, ram=8.0)
    code, _, _ = run(tmp_path, [job("a"), job("b")], fs, MAX_PROCS=2)
    (_, ta), (_, tb) = fs.spawned
    assert code == 0 and tb - ta >= 20


# ---------------------------------------------------------------- emergencies

def test_short_ram_dip_does_not_shed(tmp_path):
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 1.0}},
                    ram=lambda t: 0.5 if 60 <= t < 90 else 4.0)  # free = 0.5 - 1.0 < 0 for 30 s
    code, logs, hist = run(tmp_path, [job("a")], fs)
    assert code == 0 and not has(logs, "shedding") and len(fs.spawned) == 1
    assert hist.rows[-1]["ram_min_gb"] < 0.15


def test_sustained_low_ram_sheds_and_relearns(tmp_path):
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}},
                    ram=lambda t: 2.1 if t < 150 else 6.0)  # job alone leaves ~0.1 GB free
    code, logs, hist = run(tmp_path, [job("a", mem_key="k")], fs)
    assert code == 0
    assert has(logs, "GUARD: free RAM < EMERG_RAM_GB 0.15 for 6") and has(logs, "shedding newest job a")
    assert has(logs, "plan job a was shed; queued again (1/3)")
    shed_row, ok_row = hist.rows
    assert shed_row["outcome"] == "shed" and shed_row["peak_ws_gb"] >= 2.0
    # The retry waited for room for what the shed attempt reached, not the 1.3 GB default.
    assert has(logs, "next: a needs ram 2.2") and "lower-bound:k" in ok_row["need_source"]
    assert ok_row["outcome"] == "ok" and fs.spawned[1][1] >= 150


def test_commit_emergency_sheds_at_once(tmp_path):
    fs = FakeSystem({"a": {"duration": 100, "commit_gb": 1.0, "ramp_s": 1}},
                    commit=lambda t: 20.0 if t < 40 else 2.0)
    code, logs, hist = run(tmp_path, [job("a")], fs, MAX_FAILS=0)
    assert has(logs, "free commit 1.00 GB < EMERG_COMMIT_GB 1.5")
    assert code == 5 and has(logs, "was shed 1 times; GIVING UP")
    assert 40 <= fs.t < 50 + 30  # shed within a tick (plus the kill wait)


def test_repeated_sheds_give_up(tmp_path):
    fs = FakeSystem({"a": {"duration": 500, "ws_gb": 0.3}}, ram=0.3)
    code, logs, hist = run(tmp_path, [job("a", ram_gb=0.01, mem_override=True)], fs, ADMIT_RAM_FLOOR_GB=0.0)
    assert code == 5 and len(fs.spawned) == 4 and has(logs, "was shed 4 times; GIVING UP on it")
    assert [r["outcome"] for r in hist.rows] == ["shed"] * 4


def test_low_disk_kills_and_stops(tmp_path):
    fs = FakeSystem({"a": {"duration": 500}}, sys_disk=lambda t: 50.0 if t < 30 else 5.0)
    code, logs, hist = run(tmp_path, [job("a")], fs)
    assert code == 3 and has(logs, "GUARD: low disk") and hist.rows[-1]["outcome"] == "killed-disk"
    assert all(j.proc.killed_at is not None for j in fs.jobs)


# ---------------------------------------------------------------- measurement

def test_measures_the_whole_tree(tmp_path):
    spec = {"duration": 40, "ws_gb": 1.5, "commit_gb": 1.2, "child_gb": 0.5, "child_start": 4, "child_s": 10}
    fs = FakeSystem({"a": spec}, ram=6.0)
    code, logs, hist = run(tmp_path, [job("a")], fs)
    r = hist.rows[-1]
    assert code == 0 and r["complete"] is True and r["nprocs_total"] == r["nprocs_sampled"] == 3
    # launcher 4 MB + interpreter 1.5 GB + child 0.5 GB (it had exited long before the end)
    assert r["peak_ws_gb"] == pytest.approx(2.0 + 4 / 1024, abs=0.01)
    assert r["peak_commit_gb"] == pytest.approx(1.7, abs=0.02)
    assert r["ram_before_gb"] == pytest.approx(6.0, abs=0.01) and r["ram_min_gb"] < 4.1
    assert r["ram_after_gb"] == pytest.approx(6.0, abs=0.01)
    assert has(logs, "GUARD MEM a ok: peak_ws_gb=2.00") and has(logs, "procs 3/3")


def test_child_between_samples_marks_partial(tmp_path):
    spec = {"duration": 40, "ws_gb": 1.0, "child_gb": 3.0, "child_start": 5.1, "child_s": 0.1}
    fs = FakeSystem({"a": spec}, ram=8.0)
    code, logs, hist = run(tmp_path, [job("a")], fs)
    r = hist.rows[-1]
    assert r["complete"] is False and r["nprocs_total"] == 3 and r["nprocs_sampled"] == 2
    assert r["peak_commit_gb"] >= 3.5  # the Job Object still saw the child's commit
    assert has(logs, "(partial measurement)")
    # ...so it is only a lower bound for the next estimate.
    assert hist.peak("peak_ws_gb", 5, key=r["key"]) is None


def test_strays_are_ended(tmp_path):
    fs = FakeSystem({"a": {"duration": 20, "stray_s": 300}}, ram=6.0)
    code, logs, hist = run(tmp_path, [job("a")], fs)
    assert code == 0 and has(logs, "GUARD WARNING: process(es) of a still running")
    r = hist.rows[-1]
    assert fs.t < 60 and r["outcome"] == "ok" and r["forced_end"] is True and r["complete"] is False


def test_stray_that_exits_in_time_is_waited_for(tmp_path):
    fs = FakeSystem({"a": {"duration": 20, "stray_s": 4}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a")], fs)
    assert code == 0 and not has(logs, "still running")


# ---------------------------------------------------------------- outcomes and plumbing

def test_failed_job(tmp_path):
    fs = FakeSystem({"a": {"duration": 5, "code": 1}, "b": {"duration": 5}}, ram=6.0)
    code, logs, hist = run(tmp_path, [job("a"), job("b")], fs)
    assert code == 5 and has(logs, "plan job a FAILED (exit code 1)") and has(logs, "plan job b finished (1/2)")
    assert [r["outcome"] for r in hist.rows] == ["failed", "ok"]


def test_spawn_error(tmp_path):
    fs = FakeSystem({"a": {"spawn_error": True}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a")], fs)
    assert code == 5 and has(logs, "plan job a FAILED to start")


def test_job_timeout(tmp_path):
    fs = FakeSystem({"a": {"duration": 1000}}, ram=6.0)
    code, logs, hist = run(tmp_path, [job("a")], fs, JOB_TIMEOUT_S=100)
    assert code == 5 and hist.rows[-1]["outcome"] == "timeout" and fs.t < 200


def test_graceful_stop(tmp_path):
    out = tmp_path / "out"
    specs = {"a": {"duration": 20}, "b": {"duration": 20}}

    class StopAfterFirst(FakeSystem):
        def spawn_in_job(self, *a, **k):
            res = super().spawn_in_job(*a, **k)
            (out / "plan.STOP").write_text("")
            return res

    fs = StopAfterFirst(specs, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a"), job("b")], fs)
    assert code == 2 and [n for n, _ in fs.spawned] == ["a"] and has(logs, "plan job a finished")


def test_old_outputs_moved_aside_and_old_stop_cleared(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.jsonl").write_text("old\n")
    (out / "plan.STOP").write_text("")
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a")], fs)
    assert code == 0 and (out / "a.jsonl").read_text() == ""
    assert len(list(out.glob("a.jsonl.*.old"))) == 1


def test_lock_refused_while_held(tmp_path):
    # The holder is another process: POSIX lockf locks belong to a process, so a second lock taken in this
    # process would succeed and test nothing.
    out = tmp_path / "out"
    out.mkdir()
    code = ("import os, sys; sys.path.insert(0, sys.argv[1]); import probe_guard_v2 as pg; "
            "lk = pg.PlanLock(sys.argv[2]); assert lk.acquire(); print('held', os.getpid(), flush=True); "
            "sys.stdin.readline(); lk.release()")
    holder = subprocess.Popen([sys.executable, "-c", code, str(Path(pg.__file__).parent), str(out / "plan.lock")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        word, pid = holder.stdout.readline().split()  # its own pid: a venv python.exe is a launcher
        assert word == "held"
        fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
        code, logs, _ = run(tmp_path, [job("a")], fs)
        assert code == 6 and has(logs, "REFUSED: another guard holds") and fs.spawned == []
        assert has(logs, f'"pid": {pid}')  # who holds it is readable
    finally:
        holder.communicate("\n", timeout=10)  # the holder releases its lock and exits
    code, logs, _ = run(tmp_path, [job("a")], fs)
    assert code == 0


def test_lock_file_left_behind_is_not_a_lock(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.lock").write_text("42 7\n")  # a v1 lock, or one from a guard that died
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a")], fs)
    assert code == 0


def test_v1_settings_warned(tmp_path):
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a")], fs, v1_names_set=["PER_PROC_RAM_GB"])
    assert code == 0 and has(logs, "GUARD WARNING: PER_PROC_RAM_GB is a v1 setting")


def test_plan_validation(tmp_path):
    fs = FakeSystem({}, ram=6.0)
    for bad in ([], [{"name": "a"}], [job("a"), job("a")]):
        with pytest.raises(ValueError):
            run(tmp_path, bad, fs)


def test_important_lines_reach_the_runner_log(tmp_path):
    """run_study.run_one() copies guard lines containing one of RUNNER_GREP into its own log."""
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}, "b": {}},
                    ram=lambda t: 2.1 if t < 150 else 6.0)
    _, logs, _ = run(tmp_path, [job("a"), job("b", ram_gb=50.0)], fs, v1_names_set=["MIN_RAM_GB"])
    for marker in ("GUARD MEM", "shedding", "queued again", "GIVING UP", "v1 setting"):
        line = next(x for x in logs if marker in x)
        assert any(k in line for k in RUNNER_GREP), line


def test_invalid_numbers_rejected(tmp_path):
    for bad in (job("a", ram_gb=-5), job("a", commit_gb=float("nan")), job("a", ram_gb="2"),
                job("a", mem_override="yes"), job("a", env=["X=1"])):
        with pytest.raises(ValueError):
            pg.validate_plan([bad])
    for env in ({"ADMIT_WAIT_S": "nan"}, {"MIN_DISK_GB": "-1"}, {"TICK_S": "inf"}):
        with pytest.raises(ValueError):
            pg.load_config(env)
    h = hist_with(tmp_path, [row("fake_job.py|a", float("inf")), row("fake_job.py|a", -1.0)])
    assert h.peak("peak_ws_gb", 5, key="fake_job.py|a") is None


def test_lower_bounds_are_not_forgotten(tmp_path):
    cfg = make_cfg(tmp_path)
    k = "fake_job.py|a"
    rows = [row(k, 4.0, outcome="shed")] + [row(k, 0.05, outcome="failed")] * 6
    assert pg.estimate(job("a"), hist_with(tmp_path, rows), cfg)["ram_gb"] == pytest.approx(4.4)


def test_history_append_after_partial_line_and_write_failure(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps(row("k", 1.0)) + "\n" + '{"key": "k", "peak_ws')
    h = pg.History(str(p))
    h.add(row("k", 2.0))
    assert pg.History(str(p)).peak("peak_ws_gb", 5, key="k") == 2.0
    (tmp_path / "afile").write_text("")
    bad = pg.History(str(tmp_path / "afile" / "h.jsonl"))  # parent is a file: every write fails
    with pytest.raises(OSError):
        bad.add(row("k", 3.0))
    assert bad.peak("peak_ws_gb", 5, key="k") == 3.0  # this run still learns from it


def test_guard_failure_ends_running_jobs(tmp_path):
    fs = FakeSystem({"a": {"duration": 500}}, ram=6.0)
    fs.resources_fail_at = 50
    with pytest.raises(OSError):
        run(tmp_path, [job("a")], fs)
    assert fs.jobs[0].proc.killed_at is not None and fs.jobs[0].closed
    assert other_process_can_lock(tmp_path / "out" / "plan.lock")  # the lock was released too


def test_meter_failure_after_spawn_ends_the_job(tmp_path):
    fs = FakeSystem({"a": {"duration": 500}}, ram=6.0)
    fs.open_fails = True
    with pytest.raises(RuntimeError):
        run(tmp_path, [job("a")], fs)
    assert fs.jobs[0].proc.killed_at is not None and fs.jobs[0].closed


def test_unkillable_job_stops_everything(tmp_path):
    fs = FakeSystem({"a": {"duration": 1000, "unkillable": True}, "b": {"duration": 5}}, ram=6.0)
    code, logs, hist = run(tmp_path, [job("a"), job("b")], fs, JOB_TIMEOUT_S=100, ADMIT_CONFIRM=1)
    assert code == 4 and [n for n, _ in fs.spawned] == ["a"]
    assert has(logs, "GUARD ERROR: could not end the processes of a")


def test_unkillable_job_on_low_disk_is_exit_4(tmp_path):
    fs = FakeSystem({"a": {"duration": 1000, "unkillable": True}}, disk=lambda t: 50.0 if t < 30 else 1.0)
    code, logs, _ = run(tmp_path, [job("a")], fs)
    assert code == 4 and has(logs, "GUARD: low disk")


def test_unreadable_job_list_is_not_empty(tmp_path):
    # The root exits at 30 s, but from 20 s on the job list can not be read: the guard must not
    # call the job finished (or start another); it ends with exit 4.
    fs = FakeSystem({"a": {"duration": 30, "pids_unreadable_after": 20}, "b": {"duration": 5}}, ram=6.0)
    code, logs, hist = run(tmp_path, [job("a"), job("b")], fs)
    assert code == 4 and [n for n, _ in fs.spawned] == ["a"]
    assert hist.rows[0]["complete"] is False and hist.rows[0]["read_errors"] > 0


def test_emergency_checks_continue_while_strays_drain(tmp_path):
    fs = FakeSystem({"a": {"duration": 20, "stray_s": 300}}, commit=lambda t: 20.0 if t < 22 else 1.0)
    code, logs, _ = run(tmp_path, [job("a")], fs, STRAY_WAIT_S=60, MAX_FAILS=0)
    assert has(logs, "shedding newest job a") and fs.t < 60


def test_cooldown_after_shed(tmp_path):
    fs = FakeSystem({"a": {"duration": 500, "ws_gb": 0.3}}, ram=0.3)
    run(tmp_path, [job("a", ram_gb=0.01, mem_override=True)], fs, ADMIT_RAM_FLOOR_GB=0.0)
    starts = [t for _, t in fs.spawned]
    assert len(starts) == 4
    assert all(b - a >= 60 + 90 for a, b in zip(starts, starts[1:]))  # 60 s low RAM, then cooldown


def test_unreadable_process_memory_makes_row_partial(tmp_path):
    fs = FakeSystem({"a": {"duration": 30, "ws_gb": 2.0, "mem_unreadable_after": 5}}, ram=6.0)
    code, _, hist = run(tmp_path, [job("a")], fs)
    r = hist.rows[-1]
    assert code == 0 and r["read_errors"] > 0 and r["complete"] is False
    assert hist.peak("peak_ws_gb", 5, key=r["key"]) is None  # not a full measurement


def test_declared_below_a_small_lower_bound_needs_override(tmp_path):
    cfg = make_cfg(tmp_path)
    h = hist_with(tmp_path, [row("fake_job.py|a", 1.0, commit=1.0, outcome="shed")])
    e = pg.estimate(job("a", ram_gb=0.1), h, cfg)  # shed at 1.0 GB, below the 1.3 default
    assert e["ram_gb"] == 1.3 and e["warnings"]
    e = pg.estimate(job("a", ram_gb=0.1, mem_override=True), h, cfg)
    assert e["ram_gb"] == 0.1


def test_fresh_numbers_after_a_stray_kill(tmp_path):
    # MAX_PROCS=2. A's stray holds 6 GB of commit; base commit drops at 31 s, just before the
    # stray is ended at 32 s. Deciding on the pre-kill reading would shed healthy job b.
    specs = {"a": {"duration": 20, "commit_gb": 1.0, "ramp_s": 1, "stray_s": 300, "stray_gb": 6.0},
             "b": {"duration": 100, "commit_gb": 0.5, "ramp_s": 1}}
    fs = FakeSystem(specs, ram=20.0, commit=lambda t: 9.0 if t < 31 else 7.5)
    code, logs, _ = run(tmp_path, [job("a"), job("b", commit_gb=0.1, mem_override=True)], fs,
                        MAX_PROCS=2, MIN_COMMIT_GB=0.0, STRAY_WAIT_S=9)
    assert [n for n, _ in fs.spawned] == ["a", "b"]
    assert has(logs, "process(es) of a still running") and not has(logs, "shedding")
    assert code == 0


def test_admission_deadline_holds_with_slow_confirm(tmp_path):
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a")], fs, ADMIT_CONFIRM=100, ADMIT_WAIT_S=4)
    assert code == 0 and fs.spawned[0][1] <= 6


def test_cleanup_survives_a_broken_log(tmp_path):
    fs = FakeSystem({"a": {"duration": 500, "ws_gb": 2.0}}, ram=lambda t: 2.1 if t < 150 else 6.0)
    cfg = make_cfg(tmp_path)

    def log(msg):
        if "shedding" in msg or "GUARD ERROR" in msg:
            raise RuntimeError("log broke")

    with pytest.raises(RuntimeError):
        pg.run_plan([job("a")], str(tmp_path / "out"), fs, cfg, pg.History(cfg.GUARD_MEM_HISTORY), log)
    assert all(j.proc.killed_at is not None and j.closed for j in fs.jobs)


def test_cleanup_continues_past_a_failing_job(tmp_path):
    specs = {"a": {"duration": 500, "ws_gb": 0.1}, "b": {"duration": 500, "ws_gb": 0.1}}
    fs = FakeSystem(specs, ram=8.0)
    fs.resources_fail_at = 100
    real_terminate = fs.job_terminate

    def terminate(job):
        if job is fs.jobs[0]:
            raise RuntimeError("terminate broke")
        real_terminate(job)

    fs.job_terminate = terminate
    with pytest.raises(OSError):
        run(tmp_path, [job("a"), job("b")], fs, MAX_PROCS=2)
    assert len(fs.jobs) == 2 and all(j.closed for j in fs.jobs)  # close ends a's tree too
    assert all(j.proc.killed_at is not None for j in fs.jobs)


def test_lock_released_if_writing_its_text_fails(tmp_path, monkeypatch):
    path = str(tmp_path / "plan.lock")
    lock = pg.PlanLock(path)
    real_write = pg.os.write
    seen = []

    def failing_write(fd, b):
        seen.append(fd)
        raise OSError("disk full")

    monkeypatch.setattr(pg.os, "write", failing_write)
    with pytest.raises(OSError):
        lock.acquire()
    monkeypatch.setattr(pg.os, "write", real_write)
    assert lock.fd is None and seen
    with pytest.raises(OSError):
        os.fstat(seen[0])  # the descriptor itself was closed, not just forgotten
    assert other_process_can_lock(path)


def test_kill_wait_is_measured_in_elapsed_time(tmp_path):
    fs = FakeSystem({"a": {"duration": 1000, "unkillable": True}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a")], fs, JOB_TIMEOUT_S=100, KILL_WAIT_S=30)
    # timeout at ~100 s, then at most one KILL_WAIT_S for the tree and the root together
    assert code == 4 and fs.t <= fs.spawned[0][1] + 100 + 2 + 30 + 1


# ---------------------------------------------------------------- chains and retry: none

def test_shed_chain_member_fails_whole_chain_and_moves_state(tmp_path):
    state = tmp_path / "chain_state"
    state.mkdir()
    (state / "resume.pt").write_text("half-written")
    specs = {"c1": {"duration": 30, "ws_gb": 0.1}, "c2": {"duration": 200, "ws_gb": 2.0},
             "c3": {"duration": 30}}
    fs = FakeSystem(specs, ram=lambda t: 2.1 if t < 150 else 6.0)
    plan = [job(n, chain="c", state_dir=str(state)) for n in ("c1", "c2", "c3")]
    code, logs, hist = run(tmp_path, plan, fs)
    assert code == 5 and [n for n, _ in fs.spawned] == ["c1", "c2"]  # c2 not retried, c3 not run
    assert has(logs, "plan job c2 was shed; GIVING UP on it") and has(logs, "GIVING UP on chain c")
    assert not state.exists() and len(list(tmp_path.glob("chain_state.guard-killed-*"))) == 1
    assert [r["outcome"] for r in hist.rows] == ["ok", "shed"]


def test_retry_none_job_is_not_retried(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}, "b": {"duration": 5}},
                    ram=lambda t: 2.1 if t < 150 else 6.0)
    code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state)), job("b")], fs)
    assert code == 5 and [n for n, _ in fs.spawned] == ["a", "b"]  # b is not in a chain: it runs
    assert not state.exists() and has(logs, "retry none")


def test_failed_chain_member_skips_the_rest_but_keeps_state(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"c1": {"duration": 5, "code": 1}, "c2": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("c1", chain="c", state_dir=str(state)), job("c2", chain="c")], fs)
    assert code == 5 and [n for n, _ in fs.spawned] == ["c1"] and state.exists()  # evidence kept
    assert has(logs, "GIVING UP on chain c: c1 failed (exit code 1)")


def test_chain_members_never_overlap(tmp_path):
    specs = {"c1": {"duration": 60, "ws_gb": 0.1}, "c2": {"duration": 10, "ws_gb": 0.1}}
    fs = FakeSystem(specs, ram=8.0)
    code, _, _ = run(tmp_path, [job("c1", chain="c"), job("c2", chain="c")], fs, MAX_PROCS=2)
    (_, t1), (_, t2) = fs.spawned
    assert code == 0 and t2 >= t1 + 60


def test_chain_that_runs_cleanly(tmp_path):
    fs = FakeSystem({n: {"duration": 5} for n in ("c1", "c2", "c3")}, ram=6.0)
    code, logs, _ = run(tmp_path, [job(n, chain="c") for n in ("c1", "c2", "c3")], fs)
    assert code == 0 and [n for n, _ in fs.spawned] == ["c1", "c2", "c3"]


def test_chain_and_retry_validation():
    for bad in (job("a", retry="always"), job("a", chain=""), job("a", chain="c", retry="shed"),
                job("a", state_dir=3)):
        with pytest.raises(ValueError):
            pg.validate_plan([bad])
    pg.validate_plan([job("a", chain="c", retry="none", state_dir="x")])


def test_chain_member_not_admitted_stops_the_chain(tmp_path):
    fs = FakeSystem({"c1": {}, "c2": {"duration": 5}}, ram=1.0)
    code, logs, _ = run(tmp_path, [job("c1", chain="c", ram_gb=1.3), job("c2", chain="c", ram_gb=0.1)], fs,
                        ADMIT_WAIT_S=100)
    assert code == 8 and fs.spawned == [] and has(logs, "GIVING UP on chain c: c1 was not admitted")


def test_forced_stray_fails_a_chain_member(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"c1": {"duration": 20, "stray_s": 300}, "c2": {"duration": 5}}, ram=6.0)
    code, logs, hist = run(tmp_path, [job("c1", chain="c", state_dir=str(state)), job("c2", chain="c")], fs)
    assert code == 5 and [n for n, _ in fs.spawned] == ["c1"] and not state.exists()
    assert has(logs, "plan job c1 FAILED: processes it left running were ended")


def test_plain_job_state_moved_aside_before_retry(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}}, ram=lambda t: 2.1 if t < 150 else 6.0)
    code, logs, _ = run(tmp_path, [job("a", state_dir=str(state) + os.sep)], fs)  # trailing separator
    assert code == 0 and len(fs.spawned) == 2 and has(logs, "queued again (1/3)")
    moved = list(tmp_path.glob("s.guard-killed-*"))
    assert len(moved) == 1 and moved[0].parent == tmp_path  # a sibling, not inside the folder


def test_state_that_cannot_be_moved_blocks_the_retry(tmp_path, monkeypatch):
    state = tmp_path / "s"
    state.mkdir()

    def no_rename(a, b):
        raise OSError("in use")

    monkeypatch.setattr(pg.os, "rename", no_rename)
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}}, ram=lambda t: 2.1 if t < 150 else 6.0)
    code, logs, _ = run(tmp_path, [job("a", state_dir=str(state))], fs)
    assert code == 5 and len(fs.spawned) == 1 and has(logs, "its state_dir could not be moved aside")
    assert state.exists() and (tmp_path / "s.DO_NOT_RESUME").exists()


def test_low_disk_moves_a_killed_chain_state_aside(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"c1": {"duration": 500}, "c2": {}}, sys_disk=lambda t: 50.0 if t < 30 else 5.0)
    code, _, _ = run(tmp_path, [job("c1", chain="c", state_dir=str(state)), job("c2", chain="c")], fs)
    assert code == 3 and not state.exists() and len(list(tmp_path.glob("s.guard-killed-*"))) == 1


def test_guard_failure_moves_state_aside(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"a": {"duration": 500}}, ram=6.0)
    fs.resources_fail_at = 50
    with pytest.raises(OSError):
        run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
    assert not state.exists()


def test_state_dir_overlapping_out_dir_is_refused(tmp_path):
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    for d in (tmp_path / "out", tmp_path, Path(tmp_path.anchor)):
        with pytest.raises(ValueError):
            run(tmp_path, [job("a", state_dir=str(d))], fs)
    code, _, _ = run(tmp_path, [job("a", state_dir=str(tmp_path / "out" / "inner"))], fs)
    assert code == 0  # a folder inside OUT_DIR is fine


def test_marked_state_is_refused_before_anything_starts(tmp_path, monkeypatch):
    state = tmp_path / "s"
    state.mkdir()
    monkeypatch.setattr(pg.os, "rename", lambda a, b: (_ for _ in ()).throw(OSError("in use")))
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}}, ram=lambda t: 2.1 if t < 150 else 6.0)
    run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
    assert (tmp_path / "s.DO_NOT_RESUME").exists()
    fs2 = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs2)
    assert code == 7 and fs2.spawned == [] and has(logs, "REFUSED: state the guard could not move aside")


def test_broken_log_after_a_shed_still_moves_state_aside(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"a": {"duration": 500, "ws_gb": 2.0}}, ram=lambda t: 2.1 if t < 150 else 6.0)
    cfg = make_cfg(tmp_path)

    def log(msg):
        if "was shed" in msg or "GUARD MEM" in msg or "moved" in msg or "GUARD ERROR" in msg:
            raise RuntimeError("log broke")

    with pytest.raises(RuntimeError):
        pg.run_plan([job("a", retry="none", state_dir=str(state))], str(tmp_path / "out"), fs, cfg,
                    pg.History(cfg.GUARD_MEM_HISTORY), log)
    assert not state.exists() and len(list(tmp_path.glob("s.guard-killed-*"))) == 1
    assert all(j.proc.killed_at is not None and j.closed for j in fs.jobs)


def test_unmovable_state_is_refused_even_without_the_marker(tmp_path, monkeypatch):
    state = tmp_path / "s"
    state.mkdir()
    monkeypatch.setattr(pg.os, "rename", lambda a, b: (_ for _ in ()).throw(OSError("in use")))
    monkeypatch.setattr(pg, "_write_marker", lambda p, t: (_ for _ in ()).throw(OSError("denied")))
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}}, ram=lambda t: 2.1 if t < 150 else 6.0)
    code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
    assert code == 5 and not (tmp_path / "s.DO_NOT_RESUME").exists()
    assert has(logs, "recorded in") and (tmp_path / "blocked_state.jsonl").exists()
    fs2 = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs2)
    assert code == 7 and fs2.spawned == [] and has(logs, "REFUSED: state the guard could not move")
    monkeypatch.undo()
    old_id = pg._folder_id(str(state))
    state.rmdir()  # a person cleared it
    state.mkdir()
    fs3 = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs3)
    if pg._same_folder(old_id, pg._folder_id(str(state))):
        # No creation time and the inode number was reused (can happen on Linux): the new
        # folder can't be told apart, so it stays blocked until the record's line is removed.
        assert code == 7
        (tmp_path / "blocked_state.jsonl").write_text("")
        code, _, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs3)
    assert code == 0 and len(fs3.spawned) == 1  # a fresh folder at the same path is fine


def test_folder_identity_with_and_without_creation_time():
    assert pg._same_folder([1, 2, 30], [1, 2, 30])
    assert not pg._same_folder([1, 2, 30], [1, 2, 31])   # same id, made again
    assert pg._same_folder([1, 2, None], [1, 2, None])   # no creation time: id decides
    assert not pg._same_folder([1, 2, None], [1, 3, None])
    assert pg._same_folder([1, 2], [1, 2, 30])           # records from before creation times
    assert not pg._same_folder([1, 2, 30], [1, 0, 30])   # no file id: never a match by id


def test_state_dir_plan_refused_when_the_record_cannot_be_written(tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a", state_dir=str(tmp_path / "s"))], fs,
                        GUARD_MEM_HISTORY=str(blocker / "sub" / "hist.jsonl"))
    assert code == 7 and fs.spawned == [] and has(logs, "can't be written")


def _shed_with_unmovable_state(tmp_path, monkeypatch, state):
    monkeypatch.setattr(pg.os, "rename", lambda a, b: (_ for _ in ()).throw(OSError("in use")))
    monkeypatch.setattr(pg, "_write_marker", lambda p, t: (_ for _ in ()).throw(OSError("denied")))
    fs = FakeSystem({"a": {"duration": 200, "ws_gb": 2.0}}, ram=lambda t: 2.1 if t < 150 else 6.0)
    code, _, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
    assert code == 5 and not (tmp_path / "s.DO_NOT_RESUME").exists()
    monkeypatch.undo()


def test_block_matches_the_same_folder_spelled_differently(tmp_path, monkeypatch):
    state = tmp_path / "s"
    state.mkdir()
    _shed_with_unmovable_state(tmp_path, monkeypatch, state)
    other = str(state).swapcase() if sys.platform == "win32" else str(tmp_path / "." / "s")
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a", retry="none", state_dir=other)], fs)
    assert code == 7 and fs.spawned == []


def test_torn_record_line_does_not_hide_a_later_block(tmp_path, monkeypatch):
    rec = tmp_path / "blocked_state.jsonl"
    rec.write_bytes(b'{"path": "C:/x", "id"')  # an earlier write cut short
    state = tmp_path / "s"
    state.mkdir()
    cfg = make_cfg(tmp_path)
    assert pg.block_state(cfg, str(state), {"job": "a"})
    lines = rec.read_bytes().splitlines()
    assert len(lines) == 2 and pg.json.loads(lines[1])["path"] == str(state)
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a", state_dir=str(state))], fs)
    assert code == 7 and fs.spawned == [] and has(logs, "line 1 can't be read")
    rec.write_bytes(b"\n".join(lines[1:]) + b"\n")  # a person fixed the damaged line
    code, logs, _ = run(tmp_path, [job("a", state_dir=str(state))], fs)
    assert code == 7 and has(logs, "REFUSED: state the guard could not move")


def test_damaged_record_does_not_block_plans_without_state(tmp_path):
    (tmp_path / "blocked_state.jsonl").write_bytes(b"garbage")
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a")], fs)
    assert code == 0


def test_short_spike_just_before_exit_is_measured(tmp_path):
    spec = {"duration": 59.3, "ws_gb": 3.0, "ramp_s": 5, "spike_gb": 6.0, "spike_s": 0.6}
    fs = FakeSystem({"a": dict(spec)}, ram=12.0)
    code, _, hist = run(tmp_path, [job("a")], fs)
    row = [r for r in hist.rows if r["name"] == "a"][-1]
    assert code == 0 and row["peak_ws_gb"] >= 6.0
    # reading only every TICK_S misses it: this is the case the finer reads are for
    (tmp_path / "hist.jsonl").unlink()
    fs = FakeSystem({"a": dict(spec)}, ram=12.0)
    code, _, hist = run(tmp_path, [job("a")], fs, MEAS_TICK_S=2.0)
    row = [r for r in hist.rows if r["name"] == "a"][-1]
    assert code == 0 and row["peak_ws_gb"] < 6.0


def test_meas_tick_must_be_positive():
    with pytest.raises(ValueError):
        pg.load_config({"MEAS_TICK_S": "0"})


def test_waiting_job_uses_a_need_measured_while_it_waited(tmp_path):
    # MAX_PROCS=2. b (same family as a) waits while a runs. a then records a 4 GB peak, so
    # b's family need becomes 4.4 GB. With 4.5 GB free, b must keep waiting, not start on the
    # 1.3 GB default it was first given.
    specs = {"a": {"duration": 30, "ws_gb": 4.0, "commit_gb": 1.0, "ramp_s": 2},
             "b": {"duration": 30, "ws_gb": 4.0, "commit_gb": 1.0, "ramp_s": 2}}
    fs = FakeSystem(specs, ram=4.5)
    code, logs, _ = run(tmp_path, [job("a"), job("b")], fs, MAX_PROCS=2, ADMIT_WAIT_S=120)
    assert [n for n, _ in fs.spawned] == ["a"] and code == 8
    assert has(logs, "b's need changed")


_DIE_MID_RUN = """
import os, sys, pathlib
sys.path.insert(0, sys.argv[1])
import test_probe_guard_v2 as T
tmp, state = pathlib.Path(sys.argv[2]), sys.argv[3]
fs = T.FakeSystem({"a": {"duration": 500}}, ram=6.0)
real_sleep = fs.sleep
def sleep(dt):
    real_sleep(dt)
    if fs.t > 20:
        os._exit(9)  # killed: no Python cleanup runs, as with a forced end of the guard's process
fs.sleep = sleep
T.run(tmp, [T.job("a", retry="none", state_dir=state)], fs)
"""


def test_state_of_a_killed_guard_is_moved_aside_by_the_next_guard(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    out = subprocess.run([sys.executable, "-c", _DIE_MID_RUN, str(Path(__file__).parent), str(tmp_path),
                          str(state)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 9, out.stderr
    assert state.exists() and len(list((tmp_path / "active").glob("*.json"))) == 1
    fs = FakeSystem({"b": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("b")], fs)
    assert code == 0 and has(logs, "a was running when its guard died")
    assert not state.exists() and len(list(tmp_path.glob("s.guard-killed-*"))) == 1
    assert list((tmp_path / "active").glob("*.json")) == []


def test_a_live_guards_running_state_is_left_alone(tmp_path):
    other = tmp_path / "other_out"
    other.mkdir()
    state = tmp_path / "s"
    state.mkdir()
    code = ("import os, sys; sys.path.insert(0, sys.argv[1]); import probe_guard_v2 as pg; "
            "lk = pg.PlanLock(sys.argv[2]); assert lk.acquire(); print('held', flush=True); "
            "sys.stdin.readline(); lk.release()")
    holder = subprocess.Popen([sys.executable, "-c", code, str(Path(pg.__file__).parent),
                               str(other / "plan.lock")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        (tmp_path / "active").mkdir()
        rec = tmp_path / "active" / "1-1-a.json"
        rec.write_text(json.dumps({"job": "a", "dirs": [str(state)], "lock": str(other / "plan.lock"),
                                   "pid": 1, "time": "t"}))
        fs = FakeSystem({"b": {"duration": 5}}, ram=6.0)
        code, logs, _ = run(tmp_path, [job("b")], fs)
        assert code == 0 and state.exists() and rec.exists() and not has(logs, "guard died")
    finally:
        holder.communicate("\n", timeout=10)
    fs = FakeSystem({"b": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("b")], fs)  # its guard is gone now
    assert code == 0 and not state.exists() and not rec.exists()


def test_a_finished_run_leaves_no_record(tmp_path):
    state = tmp_path / "s"
    state.mkdir()
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, _, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
    assert code == 0 and state.exists() and list((tmp_path / "active").glob("*.json")) == []


def _dead_guard_record(tmp_path, state):
    (tmp_path / "active").mkdir(exist_ok=True)
    rec = tmp_path / "active" / "1-1-a.json"
    rec.write_text(json.dumps({"job": "a", "dirs": [str(state)], "lock": str(tmp_path / "gone" / "plan.lock"),
                               "pid": 1, "time": "t"}))
    return rec


def test_unreadable_run_record_is_kept_and_blocks_state(tmp_path, monkeypatch):
    state = tmp_path / "s"
    state.mkdir()
    rec = _dead_guard_record(tmp_path, state)
    real_open = open

    def flaky_open(path, *a, **kw):
        if str(path) == str(rec):
            raise OSError("read error")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", flaky_open)
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
    assert code == 7 and fs.spawned == [] and rec.exists() and state.exists()
    assert has(logs, "run records can't be read")
    monkeypatch.undo()
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)  # readable again
    assert code == 0 and not rec.exists() and not state.exists() and has(logs, "guard died")


def test_state_recovery_could_not_handle_is_refused(tmp_path, monkeypatch):
    state = tmp_path / "s"
    state.mkdir()
    rec = _dead_guard_record(tmp_path, state)
    monkeypatch.setattr(pg.os, "rename", lambda a, b: (_ for _ in ()).throw(OSError("in use")))
    monkeypatch.setattr(pg, "_write_marker", lambda p, t: (_ for _ in ()).throw(OSError("denied")))
    monkeypatch.setattr(pg, "block_state", lambda cfg, d, row: False)
    fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
    code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
    assert code == 7 and fs.spawned == [] and rec.exists()
    assert has(logs, "could be neither moved aside nor recorded")


def test_state_in_use_by_a_live_guard_is_refused(tmp_path):
    other = tmp_path / "other_out"
    other.mkdir()
    state = tmp_path / "s"
    state.mkdir()
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import probe_guard_v2 as pg; "
            "lk = pg.PlanLock(sys.argv[2]); assert lk.acquire(); print('held', flush=True); "
            "sys.stdin.readline(); lk.release()")
    holder = subprocess.Popen([sys.executable, "-c", code, str(Path(pg.__file__).parent),
                               str(other / "plan.lock")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        (tmp_path / "active").mkdir()
        (tmp_path / "active" / "1-1-a.json").write_text(json.dumps(
            {"job": "a", "dirs": [str(state)], "lock": str(other / "plan.lock"), "pid": 1, "time": "t"}))
        fs = FakeSystem({"a": {"duration": 5}}, ram=6.0)
        code, logs, _ = run(tmp_path, [job("a", retry="none", state_dir=str(state))], fs)
        assert code == 7 and fs.spawned == [] and has(logs, "used by a job of another running guard")
    finally:
        holder.communicate("\n", timeout=10)


# ---------------------------------------------------------------- the real machine (Windows only)

win = pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")

CHILD = ("import subprocess, sys, time\n"
         "x = b'\\x01' * (150 << 20)\n"
         "subprocess.run([sys.executable, '-c', \"y = b'\\\\x01' * (100 << 20); import time; time.sleep(2.5)\"])\n"
         "time.sleep(0.5)\n")


@win
def test_real_job_object_measures_tree(tmp_path):
    w = pg.WinSystem()
    proc, jobobj = w.spawn_in_job([sys.executable, "-c", CHILD], dict(os.environ),
                                  str(tmp_path / "o.jsonl"), str(tmp_path / "o.err"))
    meter = pg.TreeMeter(w, proc.pid, jobobj)
    try:
        deadline = time.monotonic() + 60
        while proc.poll() is None and time.monotonic() < deadline:
            meter.sample()
            time.sleep(0.2)
        while meter.pids() and time.monotonic() < deadline:
            time.sleep(0.1)
        res = meter.result()
    finally:
        meter.close()  # kill on close: nothing survives a failed test
    assert proc.returncode == 0, (tmp_path / "o.err").read_text()
    assert res["complete"] and res["nprocs_total"] >= 2
    assert res["peak_ws_gb"] >= 0.24 and res["peak_commit_gb"] >= 0.24


@win
def test_real_kill_on_close(tmp_path):
    w = pg.WinSystem()
    proc, jobobj = w.spawn_in_job([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ),
                                  str(tmp_path / "o.jsonl"), str(tmp_path / "o.err"))
    meter = pg.TreeMeter(w, proc.pid, jobobj)
    time.sleep(1.0)
    pids = meter.pids()
    assert pids
    meter.close()  # as if the guard died: the job and its tree end
    proc.wait(timeout=10)
    time.sleep(0.5)
    for p in pids:
        h = w.k.OpenProcess(w.QUERY, False, p)
        if h:
            try:
                assert not w.process_running(h)
            finally:
                w.close(h)


@win
def test_real_plan_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path, ADMIT_RAM_FLOOR_GB=0.0, MIN_COMMIT_GB=0.0, MIN_DISK_GB=0.5, EMERG_COMMIT_GB=0.0,
                   EMERG_RAM_GB=0.0, START_GAP_S=0.0, TICK_S=0.5, STRAY_WAIT_S=5.0)
    hist = pg.History(cfg.GUARD_MEM_HISTORY)
    plan = [{"name": "tiny", "cmd": ["-c", CHILD], "ram_gb": 0.01, "commit_gb": 0.01}]
    logs = []
    code = pg.run_plan(plan, str(tmp_path / "out"), pg.WinSystem(), cfg, hist, logs.append)
    assert code == 0, logs
    r = hist.rows[-1]
    assert r["outcome"] == "ok" and r["complete"] and r["peak_ws_gb"] >= 0.24, r
    assert json.loads((tmp_path / "hist.jsonl").read_text().splitlines()[-1])["name"] == "tiny"


@win
def test_real_lock_released_when_holder_dies(tmp_path):
    lock = tmp_path / "plan.lock"
    code = ("import sys, time; sys.path.insert(0, sys.argv[1]); import probe_guard_v2 as pg; "
            "assert pg.PlanLock(sys.argv[2]).acquire(); print('held', flush=True); time.sleep(60)")
    holder = subprocess.Popen([sys.executable, "-c", code, str(Path(pg.__file__).parent), str(lock)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert not pg.PlanLock(str(lock)).acquire()
    finally:
        holder.kill()  # the holder dies without releasing anything
        holder.wait(timeout=10)
    time.sleep(0.5)
    mine = pg.PlanLock(str(lock))
    assert mine.acquire()
    mine.release()


class _K32Proxy:
    """Wraps WinSystem.k so one Windows call can be made to fail."""

    def __init__(self, k, **overrides):
        self._k, self._o = k, overrides

    def __getattr__(self, name):
        return self._o[name] if name in self._o else getattr(self._k, name)


def _spy_popen(monkeypatch):
    started = []
    real_popen = pg.subprocess.Popen

    def spy(*a, **k):
        p = real_popen(*a, **k)
        started.append(p)
        return p

    monkeypatch.setattr(pg.subprocess, "Popen", spy)
    return started


@win
def test_real_failed_assignment_leaves_nothing(tmp_path, monkeypatch):
    w = pg.WinSystem()
    started = _spy_popen(monkeypatch)
    w.k = _K32Proxy(w.k, AssignProcessToJobObject=lambda job, h: 0)
    with pytest.raises(OSError, match="AssignProcessToJobObject"):
        w.spawn_in_job([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ),
                       str(tmp_path / "o.jsonl"), str(tmp_path / "o.err"))
    assert len(started) == 1 and started[0].poll() is not None  # the suspended process was ended


@win
def test_real_resume_must_really_resume(tmp_path, monkeypatch):
    w = pg.WinSystem()
    started = _spy_popen(monkeypatch)
    w.k = _K32Proxy(w.k, ResumeThread=lambda th: 2)  # "suspended twice": it would stay suspended
    with pytest.raises(OSError, match="could not resume"):
        w.spawn_in_job([sys.executable, "-c", "pass"], dict(os.environ),
                       str(tmp_path / "o.jsonl"), str(tmp_path / "o.err"))
    assert len(started) == 1 and started[0].poll() is not None


@win
def test_real_resources_are_sane():
    r = pg.WinSystem().resources(os.getcwd())
    assert 0 < r["ram_gb"] < r["total_ram_gb"] and 0 < r["commit_gb"] < r["commit_limit_gb"]
