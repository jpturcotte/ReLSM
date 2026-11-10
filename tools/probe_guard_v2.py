"""Memory guard v2 for torch jobs on a shared Windows machine.

Each job gets its own RAM and commit need (declared in the plan, or measured in earlier runs),
waits for room before it starts, is measured while it runs (whole process tree, through a
Windows Job Object), and is killed ("shed") only in an emergency. Design and the reasons for
each number: docs/research/probe_guard_v2_design.md.

Commands (run with the repo venv python):
  plan PLAN.json OUT_DIR   Run the plan's jobs. Entry fields: name, cmd (list, run as
                           [python] + cmd), env (optional); optional ram_gb, commit_gb
                           (declared needs), mem_override (true: let a declared need go below
                           the measured one), mem_key, mem_family (history keys),
                           retry ("shed" default, or "none": a shed fails the job),
                           chain (jobs sharing state: run in plan order, never at once;
                           a member that fails or is shed ends the chain), state_dir
                           (moved aside when the guard kills the job or its chain).
                           stdout -> OUT_DIR/<name>.jsonl, stderr -> OUT_DIR/<name>.err.
  estimate PLAN.json       Print each job's RAM and commit need and where it comes from.
  check                    Print free RAM, commit and disk.
Graceful stop: create OUT_DIR/plan.STOP. Running jobs finish; nothing new starts.

Exit codes: 0 all finished; 2 graceful stop; 3 low disk; 4 a job's processes could not be
ended (or their state could not be read); 5 a job failed, timed out or was given up after
repeated sheds; 6 another guard holds OUT_DIR/plan.lock; 7 a state_dir is marked DO_NOT_RESUME;
8 a job was not admitted within ADMIT_WAIT_S (5 wins when both happen).
"""
import collections
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types

GB = 2 ** 30
PY = sys.executable

SETTINGS = {  # name: (type, default); all read from the environment
    "MAX_PROCS": (int, 1),
    "MIN_DISK_GB": (float, 9.0),
    "MIN_COMMIT_GB": (float, 3.0),
    "ADMIT_RAM_FLOOR_GB": (float, 0.4),
    "ADMIT_CONFIRM": (int, 2),
    "ADMIT_WAIT_S": (float, 1800.0),
    "ADMIT_TIMEOUT": (str, "fail"),
    "EMERG_COMMIT_GB": (float, 1.5),
    "EMERG_RAM_GB": (float, 0.15),
    "EMERG_RAM_HOLD_S": (float, 60.0),
    "DEFAULT_RAM_GB": (float, 1.3),
    "DEFAULT_COMMIT_GB": (float, 3.0),
    "NEED_MARGIN": (float, 0.10),
    "HISTORY_N": (int, 5),
    "GUARD_MEM_HISTORY": (str, "~/.relsm_guard/mem_history.jsonl"),
    "PAGEFILE_DRIVE": (str, os.environ.get("SystemDrive", "C:") + "\\"),
    "JOB_COMMIT_LIMIT_GB": (float, 0.0),   # 0 = no per-job commit ceiling
    "JOB_TIMEOUT_S": (float, 0.0),         # 0 = no time limit for a running job
    "MAX_FAILS": (int, 3),
    "COOLDOWN_S": (float, 90.0),
    "START_GAP_S": (float, 20.0),
    "TICK_S": (float, 2.0),
    "MEAS_TICK_S": (float, 0.25),          # memory reads between decisions
    "WAIT_LOG_S": (float, 60.0),
    "STRAY_WAIT_S": (float, 10.0),
    "KILL_WAIT_S": (float, 30.0),
}
V1_ONLY = ("PER_PROC_GB", "PER_PROC_RAM_GB", "MIN_RAM_GB")  # v1 names v2 does not read


def _finite_nonneg(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


def load_config(env=None):
    env = os.environ if env is None else env
    cfg = types.SimpleNamespace()
    for name, (typ, default) in SETTINGS.items():
        v = typ(env.get(name, default))
        if typ in (int, float) and not _finite_nonneg(v):
            raise ValueError(f"{name} must be a finite number >= 0, not {v!r}")
        setattr(cfg, name, v)
    if cfg.ADMIT_TIMEOUT not in ("fail", "start"):
        raise ValueError(f"ADMIT_TIMEOUT must be fail or start, not {cfg.ADMIT_TIMEOUT!r}")
    if (cfg.MAX_PROCS < 1 or cfg.ADMIT_CONFIRM < 1 or cfg.HISTORY_N < 1 or cfg.TICK_S <= 0
            or cfg.MEAS_TICK_S <= 0):
        raise ValueError("MAX_PROCS, ADMIT_CONFIRM, HISTORY_N must be >= 1; TICK_S and "
                         "MEAS_TICK_S > 0")
    cfg.GUARD_MEM_HISTORY = os.path.expanduser(cfg.GUARD_MEM_HISTORY)
    cfg.v1_names_set = [n for n in V1_ONLY if n in env]
    return cfg


def make_logger(stream=None):
    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, file=stream or sys.stdout, flush=True)
    return log


def fmt(r):
    return (f"ram_gb={r['ram_gb']:.2f} commit_gb={r['commit_gb']:.1f} disk_gb={r['disk_gb']:.1f} "
            f"pagefile_disk_gb={r['sys_disk_gb']:.1f}")


def admission_text(r, need, out_ram, out_commit, since_start, since_shed, rules_ok, cfg):
    """Log text for a job that is not admitted: (what blocks it, every rule's threshold).

    `rules_ok` holds the loop's own ram/commit/disk/gaps results, so the text names the rule
    that failed; the disk and gap parts repeat the loop's expressions term for term. Text
    only: admission is decided in the loop. "running jobs" is the part of the running jobs'
    need they do not hold yet (always 0 with MAX_PROCS=1)."""
    drive = (cfg.PAGEFILE_DRIVE or "").rstrip("\\/") or "page-file drive"

    def minus(free, out, need_gb, kind):
        held = f" - running jobs {out:.2f}" if out > 0 else ""
        return f"free {free:.2f}{held} - {kind} need {need_gb:.2f}"

    why = []
    if not rules_ok["ram"]:
        why.append(f"ram: {minus(r['ram_gb'], out_ram, need['ram_gb'], 'ram')} "
                   f"< floor {cfg.ADMIT_RAM_FLOOR_GB:.2f}")
    if not rules_ok["commit"]:
        why.append(f"commit: {minus(r['commit_gb'], out_commit, need['commit_gb'], 'commit')} "
                   f"< floor {cfg.MIN_COMMIT_GB:.2f}")
    if not rules_ok["disk"]:
        if not r["disk_gb"] >= cfg.MIN_DISK_GB:
            why.append(f"disk: out_dir free {r['disk_gb']:.2f} < floor {cfg.MIN_DISK_GB:.2f}")
        if not r["sys_disk_gb"] - out_commit - need["commit_gb"] >= cfg.MIN_DISK_GB:
            why.append(f"disk: {drive} {minus(r['sys_disk_gb'], out_commit, need['commit_gb'], 'commit')} "
                       f"< floor {cfg.MIN_DISK_GB:.2f}")
    if not rules_ok["gaps"]:
        if not since_start >= cfg.START_GAP_S:
            why.append(f"start gap: {since_start:.0f} s since the last start < {cfg.START_GAP_S:.0f} s")
        if not since_shed >= cfg.COOLDOWN_S:
            why.append(f"cooldown: {since_shed:.0f} s since a job was killed for memory "
                       f"< {cfg.COOLDOWN_S:.0f} s")
    needs = (f"needs free ram >= {out_ram + need['ram_gb'] + cfg.ADMIT_RAM_FLOOR_GB:.2f} GB, "
             f"free commit >= {out_commit + need['commit_gb'] + cfg.MIN_COMMIT_GB:.2f} GB, "
             f"out_dir free >= {cfg.MIN_DISK_GB:.2f} GB, "
             f"{drive} free >= {out_commit + need['commit_gb'] + cfg.MIN_DISK_GB:.2f} GB, "
             f"start gap {cfg.START_GAP_S:.0f} s, cooldown {cfg.COOLDOWN_S:.0f} s")
    return "; ".join(why) or "unknown rule", needs


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line from a crash
                if isinstance(row, dict):
                    rows.append(row)
    return rows


# ---------------------------------------------------------------- needs from history

class History:
    """Append-only JSONL of measured runs, one row per job run (any outcome)."""

    def __init__(self, path):
        self.path = path
        self.rows = read_jsonl(path)

    def add(self, row):
        """Keep the row in memory first (a retry in this run learns from it even if the disk
        write fails), then append it on its own line and fsync."""
        self.rows.append(row)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":  # a crash left a partial line: don't glue onto it
                    fh.write(b"\n")
            fh.write((json.dumps(row) + "\n").encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())

    def peak(self, field, n, key=None, family=None, complete=True):
        """Largest `field` among matching rows, or None. complete=True: the last n finished
        runs whose measurement saw every process (full measurements). complete=False: every
        other row of any age (failed, shed, timed out, partly measured): lower bounds, kept
        until the key changes. A missing or malformed number is skipped, never read as 0."""
        vals = []
        for r in self.rows:
            if ((r.get("outcome") == "ok" and r.get("complete") is True) != complete
                    or (key is not None and r.get("key") != key)
                    or (family is not None and r.get("family") != family)):
                continue
            v = r.get(field)
            if _finite_nonneg(v):
                vals.append(v)
        if complete:
            vals = vals[-n:]
        return max(vals) if vals else None


def job_family(job):
    if job.get("mem_family"):
        return str(job["mem_family"])
    cmd = [str(c) for c in job.get("cmd", [])]
    if len(cmd) >= 2 and cmd[0] == "-m":
        return "-m " + cmd[1]
    for c in cmd:
        if c.endswith(".py"):
            return os.path.basename(c.replace("\\", "/"))
    return None


def job_key(job):
    """mem_key if given; else family|name (names alone repeat across studies)."""
    return str(job.get("mem_key") or f"{job_family(job)}|{job['name']}")


def estimate(job, hist, cfg):
    """{'ram_gb', 'commit_gb', 'ram_source', 'commit_source', 'warnings'} for one job.
    Base: full measurements for the key, else for the family, else the default. Lower bounds
    (failed, shed or partly measured runs of this key) can only raise it. A declared value is
    used when it is at least the measured need, or when the job sets mem_override."""
    key, fam = job_key(job), job_family(job)
    m = 1 + cfg.NEED_MARGIN
    out = {"warnings": []}
    for kind, field, declared, default in (("ram", "peak_ws_gb", "ram_gb", cfg.DEFAULT_RAM_GB),
                                           ("commit", "peak_commit_gb", "commit_gb", cfg.DEFAULT_COMMIT_GB)):
        by_key = hist.peak(field, cfg.HISTORY_N, key=key)
        by_fam = hist.peak(field, cfg.HISTORY_N, family=fam) if fam else None
        lower = hist.peak(field, cfg.HISTORY_N, key=key, complete=False)
        # Without a declaration: the full measurement (else the default), raised by any lower bound.
        if by_key is not None:
            val, src = by_key * m, f"history:{key}"
        elif by_fam is not None:
            val, src = by_fam * m, f"family:{fam}"
        else:
            val, src = default, "default"
        if lower is not None and lower * m > val:
            val, src = lower * m, f"lower-bound:{key}"
        # A declaration may go below what was measured (full or lower bound, even one below the
        # default) only with mem_override.
        evidence = [x * m for x in (by_key if by_key is not None else by_fam, lower) if x is not None]
        if job.get(declared) is not None:
            d = float(job[declared])
            seen = max(evidence) if evidence else None
            if seen is None or d >= seen or job.get("mem_override") is True:
                if seen is not None and d < seen:
                    out["warnings"].append(f"{job['name']}: declared {declared}={d:.2f} is below the "
                                           f"measured {seen:.2f} GB; mem_override keeps the declared value")
                val, src = d, "declared"
            else:
                out["warnings"].append(f"{job['name']}: declared {declared}={d:.2f} is below the measured "
                                       f"{seen:.2f} GB; using {val:.2f} GB (set mem_override to keep it)")
        out[f"{kind}_gb"], out[f"{kind}_source"] = round(val, 3), src
    if out["ram_source"] in ("default",) or out["ram_source"].startswith("lower-bound"):
        runs = [r for r in hist.rows if r.get("key") == key]
        if len(runs) >= 2 and not any(r.get("outcome") == "ok" and r.get("complete") is True for r in runs):
            last = next((r for r in reversed(runs) if r.get("partial_why")), runs[-1])
            out["warnings"].append(f"{job['name']}: {len(runs)} runs recorded but none fully measured "
                                   f"(last: {'; '.join(last.get('partial_why') or ['measured before v2'])}); "
                                   f"its RAM need is {out['ram_source']}, not measured")
    return out


# ---------------------------------------------------------------- measuring a job's tree

# Between two reads, processes can start and end unseen (a venv launcher's quick child, a git
# call), a process read last time can grow before it ends, and processes read only at one end
# can overlap. Each stretch between two reads gets two upper bounds on the RAM in use, and
# the smaller one counts:
#   fill bound: the peaks so far of every process read at either end, plus, for each process
#     never read or ended since its last read, MISSED_START_GB and FILL_GB_PER_S times how long
#     it could have gone unread (a process can't touch new memory faster; measured 3.3-4.3 GB/s
#     on this machine, doubled);
#   commit bound: the Job Object's exact peak commit so far, which counts every process read or
#     not, plus the shared pages of the processes read and MISSED_START_GB for each one never
#     read (RAM in use is private memory, which commit covers, plus shared pages).
FILL_GB_PER_S = 8.0
MISSED_START_GB = 0.05
MEASURE_VERSION = 2  # rows from v2 onward: peak_ws_gb is the bounded concurrent peak, not a sum


class TreeMeter:
    """Measures one job's process tree through its Job Object (every process the job starts
    joins it). Peaks are read only while a process is alive: after exit Windows reports a bogus
    ~4 MB peak working set. The Job Object keeps its own exact peak commit.
    pids() returns None when the Job Object can not be read: callers treat that as "not
    known to be empty", never as empty."""

    def __init__(self, system, pid, job, t_start=None):
        self.sys, self.pid, self.job = system, pid, job
        self.handles = {}       # pid -> handle of a process verified to be in the job (or None)
        self.peak_ws = {}       # pid -> largest PeakWorkingSetSize read while alive (bytes)
        self.ws_now = 0         # current tree working set (bytes)
        self.private_now = 0    # current tree private commit (bytes)
        self.ws_sampled_peak = 0
        self.private_sampled_peak = 0
        self.ws_conc_peak = 0   # largest bound on the RAM in use at once (see FILL_GB_PER_S)
        # the last read, or the moment before the job was launched: processes can run before
        # this meter exists, and the first stretch must cover them
        self.t_prev = system.now() if t_start is None else t_start
        self.read_prev = {}     # pid -> (peak_ws so far, shared pages) at the last read
        self.total_prev = 0     # process count read just before the last read's list
        self.listed = set()     # every pid ever listed in the job
        self.unread = {}        # pid listed but never read -> (times listed, earliest it could have started)
        self.unread_long = set()  # listed at two reads or more, never read
        self.missed = 0         # processes never listed: started and ended between two reads
        self.samples = 0
        self.read_errors = 0
        self.forced = False     # processes had to be ended after the job's root exited

    def pids(self):
        if not self.job:
            return []
        p = self.sys.job_pids(self.job)
        if p is None:
            self.read_errors += 1
        return p

    def sample(self):
        before = self.sys.job_total_processes(self.job) if self.job else None
        live = self.pids()
        if live is None:
            return self.ws_now  # keep the last numbers; this sample says nothing
        ws = private = new = 0
        cur = {}
        for pid in live:
            if pid not in self.listed:
                self.listed.add(pid)
                new += 1
            if self.handles.get(pid) is None:
                # Membership is checked on the open handle, so a pid reused between the list
                # and the open can not add a stranger's memory. A failed open is retried on
                # the next sample while the pid is still listed.
                self.handles[pid] = self.sys.open_job_process(pid, self.job)
            h = self.handles[pid]
            if h is None:
                self._not_read(pid)
                continue  # exited before it could be opened, or not openable: never sampled
            mem = self.sys.process_memory(h)
            if not self.sys.process_running(h):
                self._not_read(pid)
                continue  # exited before or during the read: its numbers are not real
            if mem is None:
                self.read_errors += 1  # alive but unreadable: this measurement is partial
                self._not_read(pid)
                continue
            ws += mem["ws"]
            private += mem["private"]
            self.peak_ws[pid] = max(self.peak_ws.get(pid, 0), mem["peak_ws"])
            # Shared pages are the working set minus its resident private part. Private commit
            # can't stand in for that part (it counts memory not yet touched), so without the
            # resident private size the whole working set counts as possibly shared.
            pws = mem.get("private_ws")
            cur[pid] = (self.peak_ws[pid], mem["ws"] - min(pws, mem["ws"]) if pws is not None else mem["ws"])
            self.unread.pop(pid, None)
        self.samples += 1
        self._bound(cur, set(live), new, before)
        self.ws_now, self.private_now = ws, private
        self.ws_sampled_peak = max(self.ws_sampled_peak, ws)
        self.private_sampled_peak = max(self.private_sampled_peak, private)
        return ws

    def _not_read(self, pid):
        if pid in self.peak_ws:
            return  # read before: it ended, which _bound covers
        n, t = self.unread.get(pid, (0, self.t_prev))
        self.unread[pid] = (n + 1, t)
        if n + 1 >= 2:
            self.unread_long.add(pid)

    def _bound(self, cur, live, new, before):
        """Raise ws_conc_peak to cover this read and the stretch since the last one."""
        now = self.sys.now()
        after = self.sys.job_total_processes(self.job) if self.job else None
        commit = self.sys.job_peak_commit(self.job) if self.job else None
        gap = max(0.0, now - self.t_prev)
        # A process's peak so far also covers a spike between reads while it was read at both
        # ends; processes alive together are added, ones that ran one after another are not.
        conc = sum(p for p, _ in cur.values())
        # Started since the count before the last list, and not listed now or before: started
        # and ended unseen. Counts are read before the last list and after this one, so this
        # can overcount by a process that started during a read, never undercount.
        if self.job and (before is None or after is None):
            self.read_errors += 1  # a stretch without a count: missed processes are unknown
        missed = max(0, after - self.total_prev - new) if after is not None else 0
        ended = [pid for pid in self.read_prev if pid not in cur]
        gone = [pid for pid in self.unread if pid not in live]
        if missed or ended or gone:
            unread_s = (missed + len(ended)) * gap + sum(now - self.unread.pop(p)[1] for p in gone)
            # a process read at both ends keeps its newer peak and the larger shared count:
            # shared pages seen at the last read may have been held into the stretch
            union = dict(self.read_prev)
            for pid, (p, sh) in cur.items():
                union[pid] = (p, max(sh, union[pid][1])) if pid in union else (p, sh)
            fill = (sum(p for p, _ in union.values())
                    + (unread_s * FILL_GB_PER_S + (missed + len(gone)) * MISSED_START_GB) * GB)
            bound = fill
            if commit:
                shared = sum(sh for _, sh in union.values())
                bound = min(fill, commit + shared + (missed + len(gone)) * MISSED_START_GB * GB)
            conc = max(conc, bound)
        self.missed += missed
        self.ws_conc_peak = max(self.ws_conc_peak, conc)
        self.t_prev, self.read_prev = now, cur
        if before is not None:
            self.total_prev = before

    def kill(self, deadline):
        """End every process in the job. True only once the job is known to be empty; gives
        up at `deadline` (a system.now() time)."""
        if not self.job:
            return True
        self.sys.job_terminate(self.job)
        while True:
            p = self.pids()
            if p == []:
                return True
            if self.sys.now() >= deadline:
                return False
            self.sys.sleep(0.5)

    def result(self):
        total = self.sys.job_total_processes(self.job) if self.job else None
        peak_commit = self.sys.job_peak_commit(self.job) if self.job else None
        sampled = len(self.peak_ws)
        missed = max(0, total - sampled) if total else None
        why = []
        if not total:
            why.append("process count unknown")
        if not peak_commit:
            why.append("no peak commit")
        if self.forced:
            why.append("processes had to be ended")
        if self.read_errors:
            why.append(f"{self.read_errors} read error(s)")
        if self.unread_long:
            # listed at two reads or more but never read: alive too long for the bound on
            # processes that start and end between reads
            why.append(f"{len(self.unread_long)} process(es) alive but never read")
        return {
            "measure": MEASURE_VERSION,
            "peak_ws_gb": round(self.ws_conc_peak / GB, 3) if self.peak_ws else None,
            "peak_ws_sum_gb": round(sum(self.peak_ws.values()) / GB, 3) if self.peak_ws else None,
            "peak_ws_sampled_gb": round(self.ws_sampled_peak / GB, 3) if self.peak_ws else None,
            "peak_private_sampled_gb": round(self.private_sampled_peak / GB, 3) if self.peak_ws else None,
            "peak_commit_gb": round(peak_commit / GB, 3) if peak_commit else None,
            "nprocs_total": total,
            "nprocs_sampled": sampled,
            "nprocs_missed": missed,
            "samples": self.samples,
            "read_errors": self.read_errors,
            "forced_end": self.forced,
            "complete": not why,
            "partial_why": why,
        }

    def close(self):
        try:
            for h in self.handles.values():
                if h is not None:
                    self.sys.close(h)
        finally:
            self.handles = {}
            if self.job:
                job, self.job = self.job, None
                self.sys.job_close(job)  # KILL_ON_JOB_CLOSE ends anything still in it


# ---------------------------------------------------------------- the Windows system

if os.name == "nt":
    from ctypes import wintypes as W

    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    class _PerfInfo(ctypes.Structure):  # PERFORMANCE_INFORMATION (system-wide commit, in pages)
        _fields_ = [("cb", W.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            ("HandleCount", W.DWORD), ("ProcessCount", W.DWORD), ("ThreadCount", W.DWORD)]

    class _MemCounters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX2 (newer Windows; else _EX)
        _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
            "PeakPagefileUsage", "PrivateUsage", "PrivateWorkingSetSize")] + [
            ("SharedCommitUsage", ctypes.c_uint64)]

    _MEM_EX_SIZE = _MemCounters.PrivateWorkingSetSize.offset  # PROCESS_MEMORY_COUNTERS_EX

    class _BasicLimit(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

    class _ExtLimit(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [("Basic", _BasicLimit), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _Accounting(ctypes.Structure):  # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", W.DWORD), ("TotalProcesses", W.DWORD),
                    ("ActiveProcesses", W.DWORD), ("TotalTerminatedProcesses", W.DWORD)]

    class _ThreadEntry(ctypes.Structure):  # THREADENTRY32
        _fields_ = [("dwSize", W.DWORD), ("cntUsage", W.DWORD), ("th32ThreadID", W.DWORD),
                    ("th32OwnerProcessID", W.DWORD), ("tpBasePri", W.LONG), ("tpDeltaPri", W.LONG),
                    ("dwFlags", W.DWORD)]

    def _pid_list_type(n):  # JOBOBJECT_BASIC_PROCESS_ID_LIST with room for n ids
        class _PidList(ctypes.Structure):
            _fields_ = [("NumberOfAssignedProcesses", W.DWORD), ("NumberOfProcessIdsInList", W.DWORD),
                        ("ProcessIdList", ctypes.c_size_t * n)]
        return _PidList


class WinSystem:
    """The real machine. Everything the tests replace is here, so the guard logic never calls
    Windows directly."""
    STILL_ACTIVE = 259
    QUERY, VM_READ, TERMINATE, SET_QUOTA, SYNC = 0x1000, 0x0010, 0x0001, 0x0100, 0x100000
    CREATE_SUSPENDED = 0x4
    LIMIT_JOB_MEMORY, LIMIT_KILL_ON_JOB_CLOSE = 0x200, 0x2000
    ERROR_MORE_DATA = 234

    def __init__(self, pagefile_drive=None):
        self.pagefile_drive = pagefile_drive
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.restype = W.HANDLE
        k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        k.OpenThread.restype = W.HANDLE
        k.OpenThread.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        k.ResumeThread.argtypes = [W.HANDLE]
        k.ResumeThread.restype = W.DWORD
        k.CloseHandle.argtypes = [W.HANDLE]
        k.GetExitCodeProcess.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
        k.K32GetProcessMemoryInfo.argtypes = [W.HANDLE, ctypes.c_void_p, W.DWORD]
        k.K32GetPerformanceInfo.argtypes = [ctypes.c_void_p, W.DWORD]
        k.CreateJobObjectW.restype = W.HANDLE
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
        k.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        k.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)]
        k.QueryInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD,
                                                ctypes.c_void_p]
        k.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
        k.CreateToolhelp32Snapshot.restype = W.HANDLE
        k.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
        k.Thread32First.argtypes = [W.HANDLE, ctypes.c_void_p]
        k.Thread32Next.argtypes = [W.HANDLE, ctypes.c_void_p]
        k.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
        self.k = k

    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    def resources(self, path):
        m = _MemStatus()
        m.dwLength = ctypes.sizeof(_MemStatus)
        if not self.k.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise OSError("GlobalMemoryStatusEx failed")
        p = _PerfInfo()
        p.cb = ctypes.sizeof(_PerfInfo)
        if not self.k.K32GetPerformanceInfo(ctypes.byref(p), p.cb):
            raise OSError("GetPerformanceInfo failed")
        page = p.PageSize
        return {"ram_gb": m.ullAvailPhys / GB,
                "commit_gb": (p.CommitLimit - p.CommitTotal) * page / GB,   # system-wide
                "commit_limit_gb": p.CommitLimit * page / GB,
                "total_ram_gb": m.ullTotalPhys / GB,
                "disk_gb": shutil.disk_usage(path).free / GB,
                "sys_disk_gb": shutil.disk_usage(self.pagefile_drive or path).free / GB}

    # -- starting a job inside a fresh Job Object, with no window where it runs outside it

    def spawn_in_job(self, argv, env, out_path, err_path, commit_limit_bytes=0):
        """Start argv suspended, put it in a new Job Object (kill on close, optional commit
        ceiling), then resume it. Returns (Popen, job). On any failure the suspended process
        (which never ran) is ended, the job handle closed, and OSError raised."""
        job = self.k.CreateJobObjectW(None, None)
        if not job:
            raise OSError(f"CreateJobObject failed (error {ctypes.get_last_error()})")
        proc = None
        try:
            e = _ExtLimit()
            e.Basic.LimitFlags = self.LIMIT_KILL_ON_JOB_CLOSE
            if commit_limit_bytes:
                e.Basic.LimitFlags |= self.LIMIT_JOB_MEMORY
                e.JobMemoryLimit = int(commit_limit_bytes)
            if not self.k.SetInformationJobObject(job, 9, ctypes.byref(e), ctypes.sizeof(e)):
                raise OSError(f"SetInformationJobObject failed (error {ctypes.get_last_error()})")
            with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w") as err:
                proc = subprocess.Popen(argv, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                        creationflags=self.CREATE_SUSPENDED)
            h = self.k.OpenProcess(self.QUERY | self.TERMINATE | self.SET_QUOTA, False, proc.pid)
            if not h:
                raise OSError(f"OpenProcess on the new job failed (error {ctypes.get_last_error()})")
            try:
                if not self.k.AssignProcessToJobObject(job, h):
                    raise OSError(f"AssignProcessToJobObject failed (error {ctypes.get_last_error()})")
            finally:
                self.k.CloseHandle(h)
            self._resume(proc.pid)
            return proc, job
        except BaseException:
            try:
                if proc is not None:
                    proc.kill()
                    proc.wait(timeout=10)
            finally:
                self.k.CloseHandle(job)
            raise

    def _resume(self, pid):
        """Resume the new process. CREATE_SUSPENDED suspends only its primary thread (Popen
        does not keep that thread's handle), so every thread of the pid gets one ResumeThread:
        the primary returns 1 (suspended once, now running); a thread something else injected
        may already run (0). Raises unless at least one thread went from 1 to running."""
        snap = self.k.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
        if not snap or snap == ctypes.c_void_p(-1).value:
            raise OSError(f"thread snapshot failed (error {ctypes.get_last_error()})")
        counts = []
        try:
            t = _ThreadEntry()
            t.dwSize = ctypes.sizeof(_ThreadEntry)
            ok = self.k.Thread32First(snap, ctypes.byref(t))
            while ok:
                if t.th32OwnerProcessID == pid:
                    th = self.k.OpenThread(0x0002, False, t.th32ThreadID)  # THREAD_SUSPEND_RESUME
                    if not th:
                        raise OSError(f"OpenThread failed (error {ctypes.get_last_error()})")
                    try:
                        counts.append(self.k.ResumeThread(th))
                    finally:
                        self.k.CloseHandle(th)
                ok = self.k.Thread32Next(snap, ctypes.byref(t))
        finally:
            self.k.CloseHandle(snap)
        if 1 not in counts:
            raise OSError(f"could not resume the new job (previous suspend counts {counts})")

    # -- reading the job

    def open_job_process(self, pid, job):
        h = self.k.OpenProcess(self.QUERY | self.VM_READ | self.SYNC, False, pid)
        if not h:  # VM_READ can be refused; the limited right is enough on Windows 8.1+
            h = self.k.OpenProcess(self.QUERY | self.SYNC, False, pid)
        if not h:
            return None
        inside = W.BOOL()
        if not self.k.IsProcessInJob(h, job, ctypes.byref(inside)) or not inside.value:
            self.k.CloseHandle(h)
            return None
        return h

    def close(self, h):
        self.k.CloseHandle(h)

    def process_running(self, h):
        if h is None:
            return False
        code = W.DWORD()
        return bool(self.k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == self.STILL_ACTIVE

    def process_memory(self, h):
        m = _MemCounters()
        m.cb = ctypes.sizeof(_MemCounters)
        if self.k.K32GetProcessMemoryInfo(h, ctypes.byref(m), m.cb):
            return {"ws": m.WorkingSetSize, "peak_ws": m.PeakWorkingSetSize, "private": m.PrivateUsage,
                    "private_ws": m.PrivateWorkingSetSize}
        m.cb = _MEM_EX_SIZE  # older Windows: no resident private size
        if not self.k.K32GetProcessMemoryInfo(h, ctypes.byref(m), m.cb):
            return None
        return {"ws": m.WorkingSetSize, "peak_ws": m.PeakWorkingSetSize, "private": m.PrivateUsage}

    def job_pids(self, job):
        """Live pids in the job, or None if the job can not be read. The buffer grows until
        the whole list fits."""
        n = 256
        for _ in range(8):
            ids = _pid_list_type(n)()
            ok = self.k.QueryInformationJobObject(job, 3, ctypes.byref(ids), ctypes.sizeof(ids), None)
            err = ctypes.get_last_error()
            if ok and ids.NumberOfProcessIdsInList >= ids.NumberOfAssignedProcesses:
                return [int(p) for p in ids.ProcessIdList[:ids.NumberOfProcessIdsInList]]
            if not ok and err != self.ERROR_MORE_DATA:
                return None
            n = max(2 * n, ids.NumberOfAssignedProcesses + 16)
        return None

    def job_total_processes(self, job):
        a = _Accounting()
        if not self.k.QueryInformationJobObject(job, 1, ctypes.byref(a), ctypes.sizeof(a), None):
            return None
        return a.TotalProcesses

    def job_peak_commit(self, job):
        e = _ExtLimit()
        if not self.k.QueryInformationJobObject(job, 9, ctypes.byref(e), ctypes.sizeof(e), None):
            return None
        return e.PeakJobMemoryUsed

    def job_terminate(self, job):
        return bool(self.k.TerminateJobObject(job, 1))  # the caller checks the job is empty

    def job_close(self, job):
        self.k.CloseHandle(job)


# ---------------------------------------------------------------- lock

class PlanLock:
    """One guard per OUT_DIR. The operating system holds the lock (a byte-range lock on
    OUT_DIR/plan.lock) and drops it when the holder exits or dies, so there is no stale lock
    to judge and no window where two guards both take one over. The file itself stays; it
    shows the holder's pid for people."""
    OFFSET = 1 << 20  # lock a byte past the text, so others can still read who holds it

    def __init__(self, path):
        self.path, self.fd = path, None

    def acquire(self, note=True):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            os.lseek(fd, self.OFFSET, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, self.OFFSET)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd  # from here on, release() can undo it
        if not note:
            return True
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            text = json.dumps({"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
            os.write(fd, (text + "\n").ljust(256).encode("utf-8"))  # overwrite older, longer text
        except BaseException:
            self.release()
            raise
        return True

    def holder(self):
        try:
            return open(self.path, encoding="utf-8").read(256).strip() or "unknown"
        except OSError:
            return "unknown"

    def release(self):
        if self.fd is None:
            return
        try:
            os.lseek(self.fd, self.OFFSET, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.lockf(self.fd, fcntl.LOCK_UN, 1, self.OFFSET)
        except OSError:
            pass  # closing the file drops the lock anyway
        finally:
            os.close(self.fd)
            self.fd = None


# ---------------------------------------------------------------- plan mode

class Run:
    def __init__(self, entry, proc, meter, need, t0, r):
        self.entry, self.proc, self.meter, self.need, self.t0 = entry, proc, meter, need, t0
        self.before = r
        self.ram_min, self.commit_min = r["ram_gb"], r["commit_gb"]
        self.commit_limit_max = r.get("commit_limit_gb", 0.0)
        self.root_exit_t = None  # when the started process exited; the run ends when its job is empty
        self.active, self.active_dirs = None, []  # its record of held state (see active_dir)

    @property
    def name(self):
        return self.entry["job"]["name"]

    def outstanding(self):
        """The part of this job's need its tree does not hold yet (current usage, not peaks)."""
        return (max(0.0, self.need["ram_gb"] - self.meter.ws_now / GB),
                max(0.0, self.need["commit_gb"] - self.meter.private_now / GB))


def validate_plan(plan):
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a non-empty list of jobs")
    names = set()
    for j in plan:
        if not isinstance(j, dict) or not j.get("name") or not isinstance(j.get("cmd"), list):
            raise ValueError(f"each job needs a name and a cmd list: {j!r}")
        if j["name"] in names:
            raise ValueError(f"duplicate job name {j['name']!r}")
        names.add(j["name"])
        for f in ("ram_gb", "commit_gb"):
            if j.get(f) is not None and not _finite_nonneg(j[f]):
                raise ValueError(f"job {j['name']!r}: {f} must be a finite number >= 0, not {j[f]!r}")
        if "mem_override" in j and not isinstance(j["mem_override"], bool):
            raise ValueError(f"job {j['name']!r}: mem_override must be true or false")
        if "env" in j and not isinstance(j["env"], dict):
            raise ValueError(f"job {j['name']!r}: env must be an object")
        if j.get("retry", "shed") not in ("shed", "none"):
            raise ValueError(f"job {j['name']!r}: retry must be 'shed' or 'none'")
        if "chain" in j and (not isinstance(j["chain"], str) or not j["chain"]):
            raise ValueError(f"job {j['name']!r}: chain must be a non-empty string")
        if j.get("chain") and j.get("retry") == "shed":
            raise ValueError(f"job {j['name']!r}: a chain member can't be retried on its own "
                             f"(drop retry, or drop chain)")
        if "state_dir" in j and (not isinstance(j["state_dir"], str) or not j["state_dir"]):
            raise ValueError(f"job {j['name']!r}: state_dir must be a non-empty path")


def _norm(path):
    return os.path.normpath(os.path.abspath(path))


def check_state_dirs(plan, out_dir):
    """Normalise every state_dir and refuse ones the guard could not safely move aside: a
    drive root, OUT_DIR itself, or a folder that contains OUT_DIR (the guard's lock, logs and
    the jobs' outputs live there). Returns the plan with normalised paths."""
    od = os.path.normcase(_norm(out_dir))
    out = []
    for j in plan:
        if j.get("state_dir"):
            sd = _norm(j["state_dir"])
            if os.path.dirname(sd) == sd:
                raise ValueError(f"job {j['name']!r}: state_dir {sd} is a drive root")
            n = os.path.normcase(sd)
            if od == n or od.startswith(n.rstrip(os.sep) + os.sep):
                raise ValueError(f"job {j['name']!r}: state_dir {sd} is or contains the guard's OUT_DIR")
            j = {**j, "state_dir": sd}
        out.append(j)
    return out


def blocked_file(cfg):
    """The guard's own record of state folders it could not move aside. It sits next to the
    memory history, in a folder the guard writes anyway, so it does not depend on being able
    to write next to a job's state_dir."""
    return os.path.join(os.path.dirname(cfg.GUARD_MEM_HISTORY) or ".", "blocked_state.jsonl")


def _folder_id(d):
    """[device, file id, creation time or None]. NTFS ids are not reused for a new folder;
    POSIX inode numbers can be, so the creation time is added where the OS reports one."""
    try:
        st = os.stat(d)
    except OSError:
        return None
    born = getattr(st, "st_birthtime_ns", None)
    if born is None and os.name == "nt":
        born = st.st_ctime_ns  # creation time on Windows
    return [st.st_dev, st.st_ino, born]


def _same_folder(rid, fid):
    if not rid or not fid or not fid[1] or list(rid[:2]) != fid[:2]:
        return False
    return len(rid) < 3 or rid[2] is None or fid[2] is None or rid[2] == fid[2]


def _write_marker(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def block_state(cfg, d, row):
    """Record that folder d must not be resumed from. True if the record was written."""
    try:
        with open(blocked_file(cfg), "ab") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell():  # a torn last line from an earlier write must not swallow this one
                with open(blocked_file(cfg), "rb") as rf:
                    rf.seek(-1, os.SEEK_END)
                    torn = rf.read(1) != b"\n"
                if torn:
                    fh.write(b"\n")
            line = json.dumps({**row, "path": d, "key": os.path.normcase(d), "id": _folder_id(d)})
            fh.write(line.encode("utf-8") + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True
    except OSError:
        return False


def blocked_states(cfg, dirs):
    """The folders in dirs that must not be resumed from: a <d>.DO_NOT_RESUME file sits next
    to it, or the guard's record names it and it is still the same folder (moving or deleting
    the folder clears the record; a fresh folder at the same path has a new id). Raises
    ValueError if the record has a line that can't be read, since a block may be hidden in it."""
    rows = []
    try:
        with open(blocked_file(cfg), "rb") as fh:
            for n, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                    if not isinstance(row, dict):
                        raise ValueError("not an object")
                except ValueError:
                    raise ValueError(f"{blocked_file(cfg)} line {n} can't be read; check which folder "
                                     f"it meant, then fix or remove that line") from None
                rows.append(row)
    except (FileNotFoundError, NotADirectoryError):
        pass  # no record can have been written there
    except OSError as e:
        raise ValueError(f"{blocked_file(cfg)} can't be read ({e})") from None
    out = []
    for d in dirs:
        if os.path.exists(d + ".DO_NOT_RESUME"):
            out.append(d)
            continue
        fid = _folder_id(d)
        if fid is None:
            continue
        key = os.path.normcase(d)
        for r in rows:
            rid = r.get("id")
            same_id = _same_folder(rid, fid)
            same_path = os.path.normcase(r.get("key") or r.get("path") or "") == key
            if same_id or (same_path and (not rid or not fid[1])):
                out.append(d)
                break
    return out


def move_state_aside(cfg, dirs, name, why, log):
    """Move each existing folder in dirs aside (<d>.guard-killed-<stamp>), so no later run
    resumes from half-written state. A folder that can't be moved is recorded as blocked
    (blocked_state.jsonl, and <d>.DO_NOT_RESUME when possible). `log` must not raise.
    Returns (every folder moved, folders neither moved nor recorded)."""
    moved_all, unhandled = True, []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for d in sorted(x for x in set(dirs) if x):
        if not os.path.exists(d):
            continue
        dest, n = f"{d}.guard-killed-{stamp}", 1
        while os.path.exists(dest):
            n += 1
            dest = f"{d}.guard-killed-{stamp}-{n}"
        try:
            os.rename(d, dest)
        except OSError as e:
            moved_all = False
            when = time.strftime('%Y-%m-%dT%H:%M:%S')
            recorded = block_state(cfg, d, {"time": when, "job": name, "why": why, "error": str(e)})
            try:
                _write_marker(d + ".DO_NOT_RESUME", f"{when} {name} {why}; the guard "
                              f"could not move this folder aside: {e}\n")
                marker = True
            except OSError:
                marker = False
            wrote = [x for x, done in ((blocked_file(cfg), recorded),
                                       (d + ".DO_NOT_RESUME", marker)) if done]
            if wrote:
                log(f"GUARD ERROR: could not move {d} aside ({e}); recorded in {wrote}; "
                    f"the guard will refuse plans that use it")
            else:
                unhandled.append(d)
                log(f"GUARD ERROR: could not move {d} aside ({e}) and could not record it "
                    f"anywhere; DO NOT resume from it")
        else:
            log(f"GUARD: moved half-written state aside ({name} {why}): {d} -> {dest}")
    return moved_all, unhandled


def active_dir(cfg):
    """Records of runs that hold state, one file per run, written before the run starts and
    removed once it has ended and its state was dealt with. A record left behind means its
    guard was killed (Windows then ends the job's tree, perhaps mid-write)."""
    return os.path.join(os.path.dirname(cfg.GUARD_MEM_HISTORY) or ".", "active")


def note_active(cfg, lock_path, name, dirs):
    os.makedirs(active_dir(cfg), exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]
    path = os.path.join(active_dir(cfg), f"{os.getpid()}-{time.time_ns()}-{safe}.json")
    row = {"job": name, "dirs": sorted(dirs), "lock": os.path.abspath(lock_path), "pid": os.getpid(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


def clear_active(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def recover_dead_runs(cfg, own_lock_path, log):
    """Deal with runs whose guard died without cleaning up. A record's guard is alive while
    it holds its plan.lock; if that lock can be taken (or it is this guard's own lock, which
    only a dead earlier guard could have used), the run's state is moved aside or blocked
    and the record removed. Records of live guards are left alone.
    Returns {"unresolved": folders that could be neither moved nor recorded, "in_use": folders
    of runs whose guard is alive, "unreadable": records that could not be read (kept)}; a plan
    must not use any of these folders (or any folder at all, while a record is unreadable)."""
    out = {"unresolved": set(), "in_use": set(), "unreadable": []}
    d = active_dir(cfg)
    try:
        names = sorted(os.listdir(d))
    except (FileNotFoundError, NotADirectoryError):
        return out  # no record can have been written there
    except OSError as e:
        log(f"GUARD ERROR: can't list the run records in {d} ({e})")
        out["unreadable"].append(d)
        return out
    own = os.path.normcase(os.path.abspath(own_lock_path))
    for f in names:
        if not f.endswith(".json"):
            continue
        path = os.path.join(d, f)
        try:
            with open(path, encoding="utf-8") as fh:
                row = json.loads(fh.read())
            lock_path, dirs, name = row["lock"], list(row["dirs"]), str(row["job"])
        except FileNotFoundError:
            continue  # another guard dealt with it
        except (OSError, ValueError, KeyError, TypeError) as e:
            # Its folders are unknown, so it is kept and no state may be used until it is read
            # or a person removes it.
            log(f"GUARD ERROR: can't read the run record {path} ({e}); keeping it")
            out["unreadable"].append(path)
            continue
        probe = None
        if os.path.normcase(os.path.abspath(lock_path)) != own and os.path.exists(lock_path):
            probe = PlanLock(lock_path)
            try:
                if not probe.acquire(note=False):
                    out["in_use"].update(dirs)  # its guard is alive
                    continue
            except OSError as e:
                log(f"GUARD WARNING: could not check whether the guard of {path} is alive ({e}); "
                    f"leaving it")
                out["in_use"].update(dirs)
                continue
        try:
            if not os.path.exists(path):
                continue  # another guard dealt with it before we took the lock
            log(f"GUARD: {name} was running when its guard died ({row.get('time', '?')}, pid "
                f"{row.get('pid', '?')}); dealing with its state")
            _, unhandled = move_state_aside(cfg, dirs, name, "was running when its guard died", log)
            if unhandled:
                out["unresolved"].update(unhandled)  # the record stays; the next guard tries again
            else:
                clear_active(path)
        finally:
            if probe is not None:
                probe.release()
    return out


def run_plan(plan, out_dir, system, cfg, hist, log):
    validate_plan(plan)
    plan = check_state_dirs(plan, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    lock = PlanLock(os.path.join(out_dir, "plan.lock"))
    if not lock.acquire():
        log(f"REFUSED: another guard holds {lock.path} ({lock.holder()}); stop it first")
        return 6
    try:
        rec = recover_dead_runs(cfg, lock.path, log)  # before anything could resume from that state
        dirs = sorted({j["state_dir"] for j in plan if j.get("state_dir")})
        if dirs and rec["unreadable"]:
            log(f"REFUSED: the plan has state_dir jobs and run records can't be read: "
                f"{rec['unreadable']}; check which state they meant, then fix or remove them")
            return 7
        keys = {os.path.normcase(x): x for x in dirs}
        unresolved = sorted(keys[k] for k in keys if k in {os.path.normcase(x) for x in rec["unresolved"]})
        if unresolved:
            log(f"REFUSED: state left by a guard that died could be neither moved aside nor recorded: "
                f"{unresolved}; move or delete each folder first")
            return 7
        busy = sorted(keys[k] for k in keys if k in {os.path.normcase(x) for x in rec["in_use"]})
        if busy:
            log(f"REFUSED: state used by a job of another running guard: {busy}")
            return 7
        try:
            marked = blocked_states(cfg, dirs) if dirs else []
        except ValueError as e:
            log(f"REFUSED: the plan has state_dir jobs and the guard's record of blocked state is damaged: {e}")
            return 7
        if marked:
            log(f"REFUSED: state the guard could not move aside after killing a job is still in place: "
                f"{marked}; move or delete each folder (and any .DO_NOT_RESUME file next to it) first")
            return 7
        if dirs:
            try:  # the records of unmovable state and running jobs must be writable up front
                for f in (blocked_file(cfg), os.path.join(active_dir(cfg), "probe.tmp")):
                    os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
                    with open(f, "a", encoding="utf-8"):
                        pass
                os.remove(os.path.join(active_dir(cfg), "probe.tmp"))
            except OSError as e:
                log(f"REFUSED: the plan has state_dir jobs but the guard's records in "
                    f"{os.path.dirname(blocked_file(cfg))} can't be written ({e})")
                return 7
        stop_file = os.path.join(out_dir, "plan.STOP")
        if os.path.exists(stop_file):
            os.remove(stop_file)  # an old STOP file would otherwise end this run at once
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for job in plan:  # move old outputs aside so a stale file is never read as this run's
            for ext in (".jsonl", ".err"):
                p = os.path.join(out_dir, job["name"] + ext)
                if os.path.exists(p):
                    os.replace(p, p + f".{stamp}.old")
        for n in cfg.v1_names_set:
            log(f"GUARD WARNING: {n} is a v1 setting; v2 ignores it (see the design note)")
        return _loop(plan, out_dir, stop_file, system, cfg, hist, log)
    finally:
        lock.release()


def _loop(plan, out_dir, stop_file, system, cfg, hist, log):
    plan_entries = [{"job": j, "sheds": 0} for j in plan]
    pending = collections.deque(plan_entries)
    running = []  # newest last; a run stays here until its whole tree has ended
    finished, failed, not_admitted = [], [], []
    last_start = last_shed = last_wait_log = -1e18
    ram_low_since = wait_since = None
    confirm = 0
    head_need = None
    lock_path = os.path.join(out_dir, "plan.lock")
    unhandled = set()  # state folders that could be neither moved aside nor recorded

    def nap(dt):
        """Wait dt, reading each running job's memory every MEAS_TICK_S meanwhile. Peaks can
        only be read while a process is alive, and some jobs spike for a second or two just
        before they exit, so a read every TICK_S would miss them."""
        end = system.now() + dt
        while True:
            left = end - system.now()
            if left <= 1e-9:
                return
            system.sleep(min(cfg.MEAS_TICK_S, left))
            for run in running:
                run.meter.sample()

    def safe_log(msg):
        try:
            log(msg)
        except Exception:
            pass

    def holds_state(job):
        return bool(job.get("state_dir") or job.get("chain") or job.get("retry") == "none")

    def quarantine(job, why):
        """The guard killed this job (or ended it in a way that may cut a write short): move
        each existing state_dir of the job, or of every member of its chain, aside, so no later
        run resumes from half-written state. If a folder can't be moved, leave a
        <state_dir>.DO_NOT_RESUME file next to it. True if every folder was moved."""
        ok, left = move_state_aside(cfg, state_dirs(job), job["name"], why, safe_log)
        unhandled.update(left)
        return ok

    def state_dirs(job):
        chain = job.get("chain")
        dirs = {job.get("state_dir")} | {e["job"].get("state_dir") for e in plan_entries
                                         if chain and e["job"].get("chain") == chain}
        return sorted(x for x in dirs if x)

    def retire(run):
        """Drop an ended run. Its active record goes too, unless some of its state could be
        neither moved nor recorded (the next guard then tries again)."""
        running.remove(run)
        if run.active and not unhandled.intersection(run.active_dirs):
            clear_active(run.active)

    def end_chain(job, why, guard_killed, bucket):
        """This job ended badly (or never started). Its chain can't go on: the remaining
        members go to `bucket` (failed, or not_admitted). If the guard killed the job, its
        state is moved aside (see quarantine)."""
        nonlocal pending
        chain = job.get("chain")
        members = [e["job"] for e in pending if chain and e["job"].get("chain") == chain]
        if members:
            pending = collections.deque(e for e in pending if e["job"].get("chain") != chain)
            bucket.extend(m["name"] for m in members)
            log(f"GIVING UP on chain {chain}: {job['name']} {why}; not running {[m['name'] for m in members]}")
        return quarantine(job, why) if guard_killed else True

    def chain_busy(job):
        chain = job.get("chain")
        return bool(chain) and any(x.entry["job"].get("chain") == chain for x in running)

    def record(run, outcome, code):
        res = run.meter.result()
        run.meter.close()
        after = system.resources(out_dir)
        job = run.entry["job"]
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": job["name"], "key": job_key(job),
               "family": job_family(job), "outcome": outcome, "exit_code": code,
               "duration_s": round(system.now() - run.t0, 1), **res,
               "ram_before_gb": round(run.before["ram_gb"], 3), "ram_min_gb": round(run.ram_min, 3),
               "ram_after_gb": round(after["ram_gb"], 3),
               "commit_before_gb": round(run.before["commit_gb"], 3),
               "commit_min_gb": round(run.commit_min, 3), "commit_after_gb": round(after["commit_gb"], 3),
               "commit_limit_growth_gb": round(max(run.commit_limit_max, after.get("commit_limit_gb", 0.0))
                                               - run.before.get("commit_limit_gb", 0.0), 3),
               "need_ram_gb": run.need["ram_gb"], "need_commit_gb": run.need["commit_gb"],
               "need_source": f"{run.need['ram_source']} / {run.need['commit_source']}",
               "out_dir": os.path.abspath(out_dir)}
        try:
            hist.add(row)
        except OSError as e:
            log(f"GUARD WARNING: could not write the memory history {hist.path}: {e}")
        g = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        log(f"GUARD MEM {run.name} {outcome}: peak_ws_gb={g(res['peak_ws_gb'])} "
            f"peak_commit_gb={g(res['peak_commit_gb'])} procs {res['nprocs_sampled']}/{res['nprocs_total']}"
            f"{'' if res['complete'] else ' (partial measurement)'} | need ram {run.need['ram_gb']:.2f} "
            f"commit {run.need['commit_gb']:.2f} | free ram before {run.before['ram_gb']:.2f} min "
            f"{run.ram_min:.2f} after {after['ram_gb']:.2f} | {row['duration_s']:.0f} s")

    def kill_run(run, outcome):
        """Kill a run's whole tree, move its state aside, record it, then drop it from
        `running` (it stays there, and so in the cleanup's reach, until all that is done).
        Returns (tree ended, state moved aside or none to move)."""
        deadline = system.now() + cfg.KILL_WAIT_S
        ok = run.meter.kill(deadline)
        try:
            run.proc.wait(timeout=max(0.1, deadline - system.now()))
        except subprocess.TimeoutExpired:
            ok = False
        moved = quarantine(run.entry["job"], outcome)  # before anything that could fail
        if not ok:
            log(f"GUARD ERROR: could not end the processes of {run.name}: {run.meter.pids()}")
        record(run, outcome, run.proc.returncode)
        retire(run)
        return ok, moved

    def stop_all(outcome):
        ok = True
        for run in reversed(list(running)):
            ok = kill_run(run, outcome)[0] and ok
        return ok

    def _start(job, entry, need, r):
        env = {**os.environ, **{k: str(v) for k, v in job.get("env", {}).items()}}
        dirs = state_dirs(job) if holds_state(job) else []
        active = None
        try:
            if dirs:  # written before launch, so a guard killed later leaves it behind
                active = note_active(cfg, lock_path, job["name"], dirs)
            t_launch = system.now()
            proc, jobobj = system.spawn_in_job([PY] + [str(c) for c in job["cmd"]], env,
                                               os.path.join(out_dir, job["name"] + ".jsonl"),
                                               os.path.join(out_dir, job["name"] + ".err"),
                                               int(cfg.JOB_COMMIT_LIMIT_GB * GB))
        except OSError as e:
            if active:
                clear_active(active)
            failed.append(job["name"])
            log(f"plan job {job['name']} FAILED to start: {e}")
            end_chain(job, "failed to start", False, failed)
            return
        try:
            meter = TreeMeter(system, proc.pid, jobobj, t_launch)
            meter.sample()
            run = Run(entry, proc, meter, need, system.now(), r)
            run.active, run.active_dirs = active, dirs
        except BaseException:
            try:
                system.job_terminate(jobobj)
            finally:
                system.job_close(jobobj)
            raise
        running.append(run)
        log(f"start {job['name']} pid {proc.pid} (need ram {need['ram_gb']:.2f} GB, "
            f"commit {need['commit_gb']:.2f} GB) | {fmt(r)}")

    try:
        while True:
            now = system.now()
            r = system.resources(out_dir)
            for run in running:
                run.meter.sample()
                run.ram_min = min(run.ram_min, r["ram_gb"])
                run.commit_min = min(run.commit_min, r["commit_gb"])
                run.commit_limit_max = max(run.commit_limit_max, r.get("commit_limit_gb", 0.0))

            # A run ends when its started process has exited AND its job is empty. Processes
            # still running STRAY_WAIT_S later are ended; that measurement is then partial.
            ended = False
            for run in list(running):
                if run.proc.poll() is None:
                    continue
                if run.root_exit_t is None:
                    run.root_exit_t = now
                left = run.meter.pids()
                if left != []:
                    if now - run.root_exit_t < cfg.STRAY_WAIT_S:
                        continue
                    log(f"GUARD WARNING: process(es) of {run.name} still running (or unreadable) "
                        f"{now - run.root_exit_t:.0f} s after it exited; ending them: {left}")
                    run.meter.forced = True
                    ended_ok = run.meter.kill(system.now() + cfg.KILL_WAIT_S)
                    if holds_state(run.entry["job"]):
                        quarantine(run.entry["job"], "left processes that had to be ended")
                    if not ended_ok:
                        log(f"GUARD ERROR: could not end the processes of {run.name}: {run.meter.pids()}")
                        failed.append(run.name)
                        record(run, "stuck", run.proc.returncode)
                        retire(run)
                        log("GUARD ERROR: stopping; a job's processes are still alive or unreadable")
                        stop_all("killed-stuck")
                        return 4
                code = run.proc.returncode
                record(run, "ok" if code == 0 else "failed", code)
                retire(run)
                ended = True
                if run.meter.forced and holds_state(run.entry["job"]):
                    failed.append(run.name)
                    log(f"plan job {run.name} FAILED: processes it left running were ended, so its "
                        f"state may be cut short")
                    end_chain(run.entry["job"], "left processes that had to be ended", False, failed)
                elif code == 0:
                    finished.append(run.name)
                    log(f"plan job {run.name} finished ({len(finished)}/{len(plan)})")
                else:
                    failed.append(run.name)
                    log(f"plan job {run.name} FAILED (exit code {code}); its output is partial, see "
                        f"{os.path.join(out_dir, run.name + '.err')}")
                    end_chain(run.entry["job"], f"failed (exit code {code})", False, failed)
            if ended:  # a tree just let go of its memory: decide on numbers read after that
                now, r = system.now(), system.resources(out_dir)

            if min(r["disk_gb"], r["sys_disk_gb"]) < cfg.MIN_DISK_GB:
                log(f"GUARD: low disk ({fmt(r)}); killing {len(running)} job(s) and stopping")
                return 3 if stop_all("killed-disk") else 4

            if cfg.JOB_TIMEOUT_S > 0:
                late = [x for x in running if x.root_exit_t is None and now - x.t0 > cfg.JOB_TIMEOUT_S]
                for run in late:
                    log(f"GUARD: {run.name} has run for > {cfg.JOB_TIMEOUT_S:.0f} s; killing it")
                    failed.append(run.name)
                    if not kill_run(run, "timeout")[0]:
                        stop_all("killed-stuck")
                        return 4
                    end_chain(run.entry["job"], "timed out", False, failed)
                if late:
                    nap(cfg.TICK_S)
                    continue  # decide on fresh numbers, not the ones from before the kill

            ram_low_since = (ram_low_since if ram_low_since is not None else now) \
                if (r["ram_gb"] < cfg.EMERG_RAM_GB and running) else None
            reason = None
            if running and r["commit_gb"] < cfg.EMERG_COMMIT_GB:
                reason = f"free commit {r['commit_gb']:.2f} GB < EMERG_COMMIT_GB {cfg.EMERG_COMMIT_GB}"
            elif ram_low_since is not None and now - ram_low_since >= cfg.EMERG_RAM_HOLD_S:
                reason = f"free RAM < EMERG_RAM_GB {cfg.EMERG_RAM_GB} for {now - ram_low_since:.0f} s"
            if reason:
                run = running[-1]
                log(f"GUARD: {reason} ({fmt(r)}); shedding newest job {run.name} pid {run.proc.pid}; "
                    f"no new job for {cfg.COOLDOWN_S:.0f} s")
                ok, moved = kill_run(run, "shed")
                last_shed, ram_low_since = system.now(), None
                entry = run.entry
                entry["sheds"] += 1
                if not ok:
                    failed.append(run.name)
                    stop_all("killed-stuck")
                    return 4
                job = entry["job"]
                if job.get("retry") == "none" or job.get("chain"):
                    failed.append(run.name)
                    log(f"plan job {run.name} was shed; GIVING UP on it (it may not be retried on "
                        f"its own: {'chain ' + job['chain'] if job.get('chain') else 'retry none'})")
                    end_chain(job, "was shed", False, failed)
                elif entry["sheds"] > cfg.MAX_FAILS:
                    failed.append(run.name)
                    log(f"plan job {run.name} was shed {entry['sheds']} times; GIVING UP on it")
                elif not moved:
                    failed.append(run.name)
                    log(f"plan job {run.name} was shed; GIVING UP on it (its state_dir could not be "
                        f"moved aside, so a retry would resume from half-written state)")
                else:
                    pending.appendleft(entry)
                    log(f"plan job {run.name} was shed; queued again ({entry['sheds']}/{cfg.MAX_FAILS})")
                wait_since, confirm = None, 0
                nap(cfg.TICK_S)
                continue

            stop = os.path.exists(stop_file)
            if stop and not running:
                log("graceful stop complete")
                return 2
            if not pending and not running:
                break

            if pending and not stop and len(running) < cfg.MAX_PROCS and not chain_busy(pending[0]["job"]):
                entry = pending[0]
                job = entry["job"]
                # Estimate on every check: a job that ends meanwhile can add measurements that
                # change this one's need (same key or family).
                fresh = estimate(job, hist, cfg)
                if wait_since is None:  # a new head of the queue (or a retry)
                    wait_since, confirm, last_wait_log = now, 0, now
                    head_need = fresh
                    for w in head_need["warnings"]:
                        log(f"GUARD WARNING: {w}")
                    log(f"next: {job['name']} needs ram {head_need['ram_gb']:.2f} GB "
                        f"({head_need['ram_source']}), commit {head_need['commit_gb']:.2f} GB "
                        f"({head_need['commit_source']})")
                elif (fresh["ram_gb"], fresh["commit_gb"]) != (head_need["ram_gb"], head_need["commit_gb"]):
                    confirm = 0  # room must hold for the new need, from scratch
                    head_need = fresh
                    for w in head_need["warnings"]:
                        log(f"GUARD WARNING: {w}")
                    log(f"{job['name']}'s need changed: ram {head_need['ram_gb']:.2f} GB "
                        f"({head_need['ram_source']}), commit {head_need['commit_gb']:.2f} GB "
                        f"({head_need['commit_source']})")
                need = head_need
                out_ram = sum(x.outstanding()[0] for x in running)
                out_commit = sum(x.outstanding()[1] for x in running)
                ram_ok = r["ram_gb"] - out_ram - need["ram_gb"] >= cfg.ADMIT_RAM_FLOOR_GB
                commit_ok = r["commit_gb"] - out_commit - need["commit_gb"] >= cfg.MIN_COMMIT_GB
                # The page file may grow by up to the jobs' commit; keep that much disk on top.
                disk_ok = (r["disk_gb"] >= cfg.MIN_DISK_GB
                           and r["sys_disk_gb"] - out_commit - need["commit_gb"] >= cfg.MIN_DISK_GB)
                gaps_ok = now - last_start >= cfg.START_GAP_S and now - last_shed >= cfg.COOLDOWN_S
                room = ram_ok and commit_ok and disk_ok and gaps_ok
                rules_ok = {"ram": ram_ok, "commit": commit_ok, "disk": disk_ok, "gaps": gaps_ok}  # for logs
                timed_out = now - wait_since >= cfg.ADMIT_WAIT_S
                start_why = None
                if need["ram_gb"] + cfg.ADMIT_RAM_FLOOR_GB > r.get("total_ram_gb", float("inf")):
                    pending.popleft()
                    not_admitted.append(job["name"])
                    log(f"GIVING UP on {job['name']}: it needs {need['ram_gb']:.2f} GB RAM, more than "
                        f"this machine has ({r['total_ram_gb']:.1f} GB)")
                    end_chain(job, "was not admitted", False, not_admitted)
                    wait_since = None
                else:
                    confirm = confirm + 1 if room else 0
                    if room and (confirm >= cfg.ADMIT_CONFIRM or timed_out):
                        start_why = "room"  # at the deadline, room now is enough
                    elif timed_out:
                        if cfg.ADMIT_TIMEOUT == "start" and commit_ok and disk_ok and gaps_ok:
                            start_why = "timeout"
                            log(f"GUARD ADMIT_TIMEOUT: starting {job['name']} under RAM pressure after "
                                f"{now - wait_since:.0f} s ({fmt(r)})")
                        elif cfg.ADMIT_TIMEOUT == "fail" or not (commit_ok and disk_ok):
                            pending.popleft()
                            not_admitted.append(job["name"])
                            why, needs = admission_text(r, need, out_ram, out_commit, now - last_start,
                                                        now - last_shed, rules_ok, cfg)
                            log(f"GIVING UP on {job['name']}: not admitted in {now - wait_since:.0f} s, "
                                f"blocked by {why} ({needs} | {fmt(r)})")
                            end_chain(job, "was not admitted", False, not_admitted)
                            wait_since = None
                        # else: start mode, waiting only out the start gap / cooldown (<= COOLDOWN_S)
                    elif not room and now - last_wait_log >= cfg.WAIT_LOG_S:
                        last_wait_log = now
                        why, needs = admission_text(r, need, out_ram, out_commit, now - last_start,
                                                    now - last_shed, rules_ok, cfg)
                        log(f"waiting for room: {job['name']} blocked by {why} | {needs} | {fmt(r)} | "
                            f"waited {now - wait_since:.0f}/{cfg.ADMIT_WAIT_S:.0f} s")
                if start_why:
                    pending.popleft()
                    _start(job, entry, need, r)
                    wait_since, confirm = None, 0
                    last_start = system.now()
            else:
                wait_since, confirm = None, 0
            nap(cfg.TICK_S)
    except BaseException:
        # Never leave a job running behind a guard that is about to release its lock. Each
        # step is isolated, so one failure (or a broken log) can't skip the other jobs.
        for run in list(running):
            try:
                run.meter.kill(system.now() + cfg.KILL_WAIT_S)
            except BaseException:
                pass
            try:
                run.meter.close()  # KILL_ON_JOB_CLOSE ends the tree even if the kill failed
            except BaseException:
                pass
            try:
                quarantine(run.entry["job"], "was ended because the guard failed")
                if run.active and not unhandled.intersection(run.active_dirs):
                    clear_active(run.active)
            except BaseException:
                pass
        try:
            log("GUARD ERROR: the guard failed; its running jobs were ended")
        except BaseException:
            pass
        raise

    if failed:
        log(f"FINISHED WITH FAILURES: {failed}" + (f"; not admitted: {not_admitted}" if not_admitted else ""))
        return 5
    if not_admitted:
        log(f"NOT ADMITTED in time: {not_admitted}")
        return 8
    log("ALL_DONE")
    return 0


# ---------------------------------------------------------------- CLI

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    log = make_logger()
    if not argv or argv[0] not in ("plan", "estimate", "check") or (argv[0] == "plan" and len(argv) < 3) \
            or (argv[0] == "estimate" and len(argv) < 2):
        print(__doc__)
        return 1
    cfg = load_config()
    if argv[0] == "check":
        print(fmt(WinSystem(cfg.PAGEFILE_DRIVE).resources(os.getcwd())))
        return 0
    plan = json.load(open(argv[1], encoding="utf-8"))
    hist = History(cfg.GUARD_MEM_HISTORY)
    if argv[0] == "estimate":
        validate_plan(plan)
        for job in plan:
            e = estimate(job, hist, cfg)
            print(f"{job['name']}: ram {e['ram_gb']:.2f} GB ({e['ram_source']}), "
                  f"commit {e['commit_gb']:.2f} GB ({e['commit_source']})")
            for w in e["warnings"]:
                print(f"  WARNING: {w}")
        return 0
    return run_plan(plan, argv[2], WinSystem(cfg.PAGEFILE_DRIVE), cfg, hist, log)


if __name__ == "__main__":
    sys.exit(main())
