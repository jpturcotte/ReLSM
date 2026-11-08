# probe_guard v2: per-job memory, admission by waiting, shedding as a last resort

Status: built in `tools/probe_guard_v2.py`, tested in `tests/test_probe_guard_v2.py` (CPU,
no torch; seven Windows-only tests use small plain-Python children). **Not in use yet.** Phase 2
of the task-format study hash-locks its runner (`run_study.py`), which calls the v1 guard by
absolute path. Neither file changes. A new study switches over after phase 2 ends (see
"Switching over").

## Why

v1 (`docs/research/probe_guard.py` on the lexical-grokking branch) reserves a fixed amount
for every job: it starts a job only when free RAM >= `MIN_RAM_GB + PER_PROC_RAM_GB`
(0.4 + 1.3 GB today), and it kills ("sheds") the running job as soon as one 5 s sample of
free RAM falls below `MIN_RAM_GB` (0.4 GB). Every shed costs a full rerun.

The guard logs from 2026-11-04 to 2026-11-07 show 11 sheds. In all of them:

- free RAM went from 1.7-4.8 GB at start to 0.1-0.4 GB. The drop is the job's own. Measured
  on 2026-11-07, CPU `train.py` runs with a lookup-table head sit at about 3.0 GB private
  while training. At the very end, a check of the head holds two batches of full-vocabulary
  logits at once: about 6.0 GB private for about 2 s, and free RAM about 4.5 GB below where
  it was when the run started. One shed attempt went 3.55 -> 1.9 -> 0.11 GB free; a healthy
  retry went 4.44 -> 1.28 -> 5.7. So these jobs need about 4.5 GB of free RAM, not the
  1.3 GB that v1 reserves, and the same command survives or is shed depending on how much
  RAM was free when it started. v2 still sizes a job from its own process tree, not from
  free-RAM drops, because other programs on a shared machine move that number too.
- free commit at the moment of the shed was 6.6-17.2 GB, far above `MIN_COMMIT_GB` (3).
  None was close to the failure that started the guard (2026-10-26: commit exhausted, page
  file grew to 38 GB, disk full). Low free RAM with commit to spare means Windows is
  trimming working sets and paging: slower, not broken.

Measured on this machine while writing v2: the venv `python.exe` is a launcher that starts
the real interpreter as a child process. The process the guard starts holds about 4 MB; the
work happens in its child. Any per-job number must cover the whole process tree.

## What v2 does

### 1. Each job's need comes from its measured history or a declaration

For RAM (`peak_ws_gb`) and commit (`peak_commit_gb`) separately:

1. **Measured base.** Full measurements are finished runs whose measurement covers the tree
   (see "When a run counts as fully measured" below). Take the largest peak among the last `HISTORY_N` (5) full measurements for the
   job's key, else for its family, times `1 + NEED_MARGIN` (1.10).
2. **Lower bounds.** Rows for the key from failed, shed or timed-out runs, and finished runs
   that were only partly measured, show at least that much was needed. All of them count,
   however old (a new `mem_key` starts afresh). They can raise the need and never lower it,
   so a lone startup crash (tiny peak) leaves the default in place. Rows written before
   `measure: 2` keep their `peak_ws_gb` as a lower bound, though it was then a sum of
   lifetime peaks: it can be several times what was held at once, but it can also be one
   process's real spike that no read saw, and nothing in the row tells the two apart. A
   study that wants to drop such rows starts a new `GUARD_MEM_HISTORY` file.
   A job with two or more recorded runs and no full measurement gets a `GUARD WARNING` that
   names the last reason, so a need stuck on the default is visible in the log.
3. **Declared.** Plan fields `ram_gb`, `commit_gb`. A declared value is used when it is at
   least everything measured for the job (full measurements and lower bounds, even a lower
   bound below the default). If it is lower, the undeclared estimate wins, with a warning,
   unless the job sets `mem_override: true`.
4. **Default.** `DEFAULT_RAM_GB` (1.3) and `DEFAULT_COMMIT_GB` (3.0), today's settings.

Keys: `mem_key` if given, else `<family>|<name>`. Family: `mem_family` if given, else the
file name of the first `.py` in `cmd` (or `-m <module>`). Job names repeat across studies, so
a study with different shapes under the same names should set `mem_key` (or its own
`GUARD_MEM_HISTORY`). "Max of the last 5, plus 10%" is a heuristic, not a guarantee.

A missing, negative or non-finite number in the history is skipped, never read as 0. A
history row is kept in memory before it is written, so a retry in the same run learns from it
even if the write fails. A partial last line left by a crash does not swallow the next row.
Plan numbers (`ram_gb`, `commit_gb`), `mem_override` and every setting are checked before
anything starts: negative, non-finite or wrongly typed values are refused. The need is
estimated afresh on every admission check, so a retry after a shed uses the shed run's row,
and a job waiting at the head of the queue picks up measurements that a job of its family
recorded meanwhile. When the need changes, the log says so and the confirmation count
starts again.

### 2. A job starts only when there is room for it, and waits (with a limit) otherwise

The next job is admitted when all of these hold on `ADMIT_CONFIRM` (2) checks in a row:

    free RAM    - outstanding RAM    - need RAM    >= ADMIT_RAM_FLOOR_GB   (0.4)
    free commit - outstanding commit - need commit >= MIN_COMMIT_GB        (3)
    free disk (output drive)                       >= MIN_DISK_GB          (9)
    free disk (page-file drive) - outstanding commit - need commit >= MIN_DISK_GB

plus at most `MAX_PROCS` running, `START_GAP_S` since the last start and `COOLDOWN_S` since
the last shed. "Outstanding" is, for each job already running, the part of its need that
its tree does not hold yet: `max(0, need - current usage)`, using current usage, never past
peaks. A job that has just started has not taken its memory yet, and free RAM does not show
it. With `MAX_PROCS=1` nothing else runs at admission, so it is 0.

The page-file rule leaves room for the page file to grow by the job's whole commit need
without taking the disk under the floor.

Two checks in a row stop a momentary rise in free RAM (Chrome freeing a tab) from admitting
a job that the next sample would not. They do not reserve anything against other programs.

While it waits, the guard logs the numbers once a minute. The wait clock (monotonic) starts
when the job is next in line and a slot is free. At `ADMIT_WAIT_S` (1800 s), room on that one
check is enough to start (the `ADMIT_CONFIRM` streak is not required at the deadline).
Without room:

- `ADMIT_TIMEOUT=fail` (default): the job is not started, the log says
  `GIVING UP on <name>: not admitted in ...`, and the plan exits with code 8. Nothing runs
  under pressure, nothing is killed, no output is written.
- `ADMIT_TIMEOUT=start`: the job starts anyway if the commit and disk rules hold (only the
  RAM rule is waived), logged as `GUARD ADMIT_TIMEOUT: starting <name> under RAM pressure`.
  If commit or disk is short at that point, the job is given up as with `fail`, so the
  wait always ends. If only the start gap or the cooldown is still running, it waits that
  out (at most `COOLDOWN_S`).

A job whose RAM need plus the floor exceeds total physical RAM is given up at once.

### 3. Every job runs inside its own Job Object and is measured

Starting a job:

1. create a Job Object with `KILL_ON_JOB_CLOSE` (and, if `JOB_COMMIT_LIMIT_GB` > 0, a
   tree-wide commit ceiling);
2. start the process **suspended**;
3. put it in the Job Object;
4. resume its thread.

The process runs no code before it is in the job, so the launcher can't start the
interpreter outside it. `CREATE_SUSPENDED` suspends the primary thread, and Popen does not
keep that thread's handle, so each thread of the new pid gets one `ResumeThread`. Starting
counts as done once a thread went from suspended-once to running (a thread something else
injected may already be running). If any step fails, the suspended process is ended (it never ran),
the job handle is closed, and the job is recorded as failed to start. One gap remains: if
the guard itself is killed in the instant between creating the process and assigning it, a
suspended process (one that never ran) is left behind. Its command line names the job's
output directory, so the runner's orphan check finds it and refuses to start. Every process the job starts joins the same Job
Object. Nested jobs are supported on Windows 8 and later, so this works even if the guard
itself runs inside a job. The job handle is not inheritable, and the guard keeps it until
the job's final numbers are read. If the guard dies, Windows closes the handle and ends the
whole tree, so no orphan is left behind. For a job that holds state, that kill can cut a
write short; see "Chains and jobs that must not be retried" for how the next guard deals with it.

Every `MEAS_TICK_S` (0.25 s) the guard reads the Job Object's process list, opens each new process,
checks on the open handle that it really is in the job (so a reused pid can't add a
stranger), and reads `K32GetProcessMemoryInfo` with `PROCESS_MEMORY_COUNTERS_EX2`: working
set, peak working set, private commit and resident private size (`PrivateWorkingSetSize`).
Where Windows rejects `_EX2` (older versions), it reads `PROCESS_MEMORY_COUNTERS_EX`
without the resident private size, and the bound below then counts each process's whole
working set as possibly shared, so `peak_ws_gb` comes out higher. A failed open is retried
while the pid is still listed. A read that fails while the process is alive counts as a read error. These reads
run between the decisions, which stay every `TICK_S` (2 s): peaks can only be read while a
process is alive, and the jobs in "Why" spike for about 2 s right before they exit, which a
read every 2 s can miss. Recorded per job:

| field | meaning |
|---|---|
| `peak_ws_gb` | the RAM need: the largest bound on the RAM the tree held at once (see "How the RAM peak is bounded") |
| `peak_ws_sum_gb` | sum of every process's lifetime peak (the pre-`measure: 2` `peak_ws_gb`); kept to compare |
| `peak_ws_sampled_gb` | largest sum of current `WorkingSetSize` seen in one sample |
| `peak_private_sampled_gb` | largest sum of current private commit seen in one sample |
| `peak_commit_gb` | the Job Object's `PeakJobMemoryUsed`: exact tree-wide peak private commit (the commit need) |
| `nprocs_total`, `nprocs_sampled`, `nprocs_missed` | processes the job ever had (Job Object accounting), how many were read alive, how many never were (most of them start and end between two reads) |
| `complete`, `partial_why` | whether the run counts as fully measured, and if not, why |
| `measure` | 2 for rows with the measurement above (rows without it summed lifetime peaks) |
| `ram_before_gb`, `ram_min_gb`, `ram_after_gb` | free RAM at admission, lowest while running, right after the tree ended |
| `commit_before_gb`, `commit_min_gb`, `commit_after_gb` | same for free commit |
| `commit_limit_growth_gb` | how much the system commit limit grew while it ran (the page file grew) |
| `read_errors`, `forced_end` | Job Object or process reads that failed while it was alive; strays had to be ended (either makes the row partial) |
| `need_ram_gb`, `need_commit_gb`, `need_source` | the estimate used to admit it |
| `outcome`, `exit_code`, `duration_s`, `samples` | ok / failed / shed / timeout / killed-disk, and so on |

Each row is appended and fsync'd to `GUARD_MEM_HISTORY`
(default `~/.relsm_guard/mem_history.jsonl`, outside every study folder so later studies
read it). A `GUARD MEM` summary line goes to the guard log.

A job is done when its started process has exited **and** its Job Object is empty. Until
then the run keeps its slot, keeps being sampled, and the emergency rules keep applying.
Processes still running `STRAY_WAIT_S` (10 s) after the root exits are ended, with a
warning. That run's measurement is then marked partial (`forced_end: true`), because work
was cut short. A Job Object that can't be read never counts as empty. If a tree can't be
ended, or its state can't be read, within `KILL_WAIT_S` (30 s), the guard ends everything
else and stops with exit 4 (also on the low-disk path) rather than start another job next to
it.

Waits for a kill are deadlines on the monotonic clock (one `KILL_WAIT_S` for the tree and
its root together). A run keeps its place in the running list until it is fully killed and
recorded. Whenever a job ends, free RAM, commit and disk are read again before any other
decision in that tick, so a release is never followed by a shed based on the earlier reading.

If the guard itself fails (an exception), it ends and closes every running job before it
releases its lock. Each job's cleanup is separate, and none depends on logging, so one
failure can't skip the others. If the guard is killed, closing the Job Object handles ends
the jobs (`KILL_ON_JOB_CLOSE`).

Limits of the measurement:

- Peaks must be read while the process is alive. After exit, Windows reports a peak working
  set of about 4 MB and a peak commit of 0 (measured). Commit is not affected: the Job
  Object keeps its own peak. Growth after a process's last read is covered by the bound below.
- A process that lives less than 0.25 s may never be read. Real jobs start such processes
  all the time: a venv `python.exe` launcher whose child exits at once, a `git` call, a
  `python -c` helper. The bound below covers them too.
- `peak_ws_gb` is meant as an upper bound on what the tree held at once, but it rests on
  three assumptions: a process can't touch new memory faster than `FILL_GB_PER_S`; a
  process that was never read has at most `MISSED_START_GB` of shared pages; and a process
  read before maps no new shared pages after its last read (see below). It includes
  shared, file-backed pages (DLLs, CUDA libraries), which mostly don't take free RAM, so it
  can overstate the draw on free RAM. Processes that ran one after another are not added
  together.
  `ram_before_gb - ram_min_gb` is logged next to it (other programs move that number too),
  so a later study can judge it. Until a job family has a few measured runs, declare its
  need, both RAM and commit; for the jobs in "Why", `ram_gb: 4.5` and `commit_gb: 6.6`
  (their 6.0 GB private peak plus the 10% margin; commit alone at the 3 GB default would
  admit them with too little room for the final check).
- `peak_commit_gb` covers the job's tree only, not commit that other programs take meanwhile.

#### How the RAM peak is bounded

At each read the guard adds up the peaks so far (`PeakWorkingSetSize`) of the processes it
read. A process read at both ends of a stretch is covered by its peak so far. Three things
can happen between two reads that one read can't see: a process starts and ends unseen
(counted from the Job Object's process total, read before each list and after it, so the
count can be one too high but never too low); a process read last time grows and ends; and
processes seen only at one end each overlap. For such a stretch the guard takes two upper
bounds on the RAM in use and keeps the smaller one:

- **fill bound**: the peaks of every process read at either end, plus, for each process
  never read or ended since its last read, `MISSED_START_GB` (0.05) and `FILL_GB_PER_S` (8)
  times how long it could have gone unread. Measured on this machine (2026-11-08), a new
  process fills fresh memory at 3.3-4.3 GB/s; 8 is about twice that.
- **commit bound**: the Job Object's exact peak commit so far, which counts every process
  read or not, plus the shared pages of the processes read (working set minus its resident
  private part, `PrivateWorkingSetSize`; private commit can't stand in for that part, since
  it counts memory not touched yet; where Windows doesn't report it, the whole working set),
  plus `MISSED_START_GB` for each process never read. RAM in use is private memory, which
  commit covers, plus shared pages.

The commit bound takes the shared pages of the processes read as they were at the two reads,
the larger of the two for a process read at both. It
does not allow for shared pages a process maps after its last read and before it ends (a
file mapped in its last 0.25 s). That is a deliberate choice: such pages are file-backed, so
Windows can drop them under pressure without writing to the page file, while allowing for
them would add up to 2 GB for every process that ends, so needs would sit far above what
jobs hold. Private memory growing in that window is covered, by the exact peak commit.

The first stretch starts at the time read just before the job is launched, which the guard
passes to the job's meter. The meter itself is made only after the job has been resumed, so
a stretch starting there would miss processes that ran and ended in between.

A stretch whose process count could not be read counts as a read error, so the run is
partial: processes that started and ended in it are unknown.

`peak_ws_gb` is the largest of these over the run. Most of the time the commit bound is the
smaller one, so `peak_ws_gb` comes out near the peak commit plus shared pages. The fill
bound counts when the peak commit is loose, for example after a process that committed far
more than it touched.

#### When a run counts as fully measured

A run is fully measured (`complete: true`) when the process count and the Job Object's peak
commit were read, no reads failed, no processes had to be ended, and no process stayed
listed for two reads or more without being read (that one was alive too long for the bound
above). Processes that started and ended between reads no longer make a run partial: the
bound covers them.

Measured on this machine (2026-11-08), through the real guard: jobs that start quick
helpers (`python -c pass` four times, six `git --version` calls, a short child) never read
1 to 5 of their processes. All were fully measured, and the second run's estimate came from
history. Their `peak_ws_gb` was 0.01-0.16 GB above the peak commit. Six 250 MB workers run
one after another gave 0.28 GB, where the old sum gave 1.54 GB.

Before this rule, a run counted as fully measured only if every process was read. On the
real machine no run ever was: all 31 rows of the first studies were partial, from 1 missed
process out of 3 (a comparison script) to 45 out of 179 (a Codex session). So every job got
the default need. And `peak_ws_gb` summed lifetime peaks, up to 10 GB for a job that held
1.8 GB at once.

### 4. Shedding only in an emergency

While jobs run:

- free disk on the output drive or the page-file drive < `MIN_DISK_GB`: kill every job and
  stop (exit 3), as in v1.
- free commit (system-wide, from `GetPerformanceInfo`: commit limit minus committed) <
  `EMERG_COMMIT_GB` (1.5): shed the newest job at once. Commit exhaustion is
  the failure that grows the page file and fails allocations.
- free RAM < `EMERG_RAM_GB` (0.15) **continuously** for `EMERG_RAM_HOLD_S` (60 s): shed the
  newest job. Short dips are paging, not danger.
- optional `JOB_TIMEOUT_S` (> 0): a job running longer is killed and counted as failed.

A shed kills the job's whole Job Object, waits until it is empty, and writes its measurement
row (`outcome: shed`). A plain job goes back to the head of the queue at most `MAX_FAILS`
times, then is given up (`GIVING UP`, exit 5). Jobs that must not be retried on their own
are handled differently (see "Chains and jobs that must not be retried"). No new job starts for `COOLDOWN_S`. Because the
shed row is a lower bound, the retry waits for at least the room the shed attempt reached.

The emergency rules react to the whole machine, not just the guard's jobs. The 60 s hold on
low RAM exists because memory spikes that last a few seconds, the job's own or another
program's, mean paging, not danger. The shed attempt in "Why" (0.11 GB free for seconds,
commit to spare) would not have been shed under these rules. Waiting for room before the
start still matters more for those jobs: their spike comes at the very end, so a shed there
loses the whole run, and a retry runs after the final checkpoint was written. Declare them
with `retry: "none"` and their `state_dir`, or as a chain (next section).

These checks are best effort. They poll every 2 s, so a large allocation can still take
commit between two checks, and a system-managed page file can grow (taking disk) before free
commit looks low. Two rules limit that: the page-file headroom at admission, and the disk
check on the page-file drive. `JOB_COMMIT_LIMIT_GB` is a hard ceiling for one job's tree (an
allocation beyond it fails inside the job). It is off by default, because a ceiling set from
a too-low estimate would fail a healthy job. Other programs on the machine are outside the
guard's control.

### 5. Chains and jobs that must not be retried

A shed-and-retry reruns the same command on whatever the killed run left behind. That is
only safe for a job that starts from nothing. On 2026-11-07 the v1 guard shed a chained
resume-test job during its final write-up. The retry resumed from the checkpoint the killed
run had already written, so the job's resume history gained an unplanned resume. The
trained weights were bit-identical to the reference, but the run failed the test anyway, and
the whole chain had to be rerun from scratch.

Plan fields for this:

- `retry: "none"`: a shed (or timeout) fails the job instead of queueing it again.
- `chain: "<name>"`: jobs that share state (each picks up the previous one's checkpoint).
  Members run in plan order and never at the same time, even with `MAX_PROCS` > 1. A member
  is never retried on its own (`retry: "shed"` on a member is refused). The remaining
  members are not run, and the log says `GIVING UP on chain <name>`, if a member:
  - is shed, times out, or fails;
  - fails to start (exit 5);
  - is not admitted in time (exit 8).
- `state_dir: "<path>"`: a folder the job resumes from. Whenever the guard ends a job that
  may have been mid-write, each existing `state_dir` of the job, or of every member of its
  chain, is renamed to `<path>.guard-killed-<stamp>` next to it. That covers a shed, a
  timeout, the low-disk stop, a job that could not be ended, the guard's own failure, and
  ending processes a job left running. A plain retried job's folder is moved before the
  retry starts. If a folder can't be moved (files still open), the guard writes
  `<path>.DO_NOT_RESUME` next to it and does not retry that job. When a job fails on its
  own, its state is left in place as evidence.

The move happens straight after the kill, before the guard logs or records anything, so a
failing log or history write cannot skip it; the run stays in the guard's cleanup list until
the move is done. A folder that can't be moved is also recorded in the guard's own
`blocked_state.jsonl` (next to `GUARD_MEM_HISTORY`, with the folder's file id), so the block
holds even when nothing can be written next to the folder. A plan is refused before anything
starts (`REFUSED`, exit 7) if one of its `state_dir`s has a `.DO_NOT_RESUME` file next to it
or is the same folder the record names (matched by file id, so a differently cased or
spelled path to it is caught too). Moving or deleting the folder clears the record; a fresh
folder at that path has a new id (on Windows; on Linux a new folder can reuse the old inode
number and no creation time is reported, so it stays blocked until its line in
`blocked_state.jsonl` is removed). The guard ends a torn last line before appending; a line
it can't read makes it refuse every plan with a `state_dir` until a person fixes that line.
A plan with any `state_dir` is also refused (exit 7) when that record can't be written, so
the block is in place before any job could need it.

A job that declares `chain`, `retry: "none"` or `state_dir`, exits 0, but leaves processes the
guard has to end after `STRAY_WAIT_S` counts as failed: its last write may have been cut short.

The guard itself can be ended without running any cleanup: its process tree killed by
whatever launched it, or the terminal closed. Windows then ends its jobs (kill on close),
perhaps mid-write, and nothing moves their state aside. So before starting a job with
state folders, the guard writes a record of it to `active/` next to `GUARD_MEM_HISTORY`:
the job, its folders (and its chain's), and the `plan.lock` of the guard that started it.
The record is removed once the run has ended and its state was dealt with. Every guard,
right after taking its own lock and before checking or admitting anything, reads the
records left behind. A record whose `plan.lock` is still held belongs to a live guard and is
left alone. Otherwise its guard died: the folders are moved aside (or blocked, as above),
with the log line `... was running when its guard died`, and the record is removed. A plan
with a `state_dir` is refused (exit 7) if `active/` can't be written, if it uses a folder that
a live guard's record names, if it uses a folder recovery could neither move nor record (the
record then stays for the next guard), or while any record can't be read (its folders are
unknown, so the record is kept until it can be read or a person removes it). A guard killed after
writing a record but before the job started leaves a record too; its state is then moved
aside although nothing wrote to it, which costs a rerun but never a resume from bad state.

`state_dir` is normalised to an absolute path. A drive root, the guard's `OUT_DIR`, or a
folder that contains `OUT_DIR` is refused before anything starts, because moving it would
move the guard's own lock and logs. The guard can't know which folders a job writes, so
`state_dir` must be declared. For a runner that reruns a failed chain, starting clean means
rerunning every member from the first. Clearing a `.DO_NOT_RESUME` file is left to a person
(or the runner) who has moved or deleted the folder.

## What stays the same as v1

Plan mode: `python tools/probe_guard_v2.py plan PLAN.json OUT_DIR`, the same plan entries
(`name`, `cmd`, `env`), and the same outputs (`OUT_DIR/<name>.jsonl` and `.err`, older ones
moved aside). Graceful stop through `OUT_DIR/plan.STOP` works as before. The runner's grep
keywords still appear in the log (`shed`, `GIVING UP`, `FAILED`, `REFUSED`, `GUARD`, `low
disk`); every new important line starts with `GUARD` (`GUARD MEM`, `GUARD WARNING`,
`GUARD ADMIT_TIMEOUT`), so the existing grep copies them. Exit codes 0, 2, 3, 5, 6 mean what
they meant in v1; 4 now means "a job's processes could not be ended"; 7 and 8 are new (7: a
`state_dir` is marked `.DO_NOT_RESUME`; 8: a job was not admitted in time, and 5 wins if a
job also failed). Two helper commands: `check` (current
resources) and `estimate PLAN.json` (each job's need and where it came from, without running
anything).

Changed: the lock. The operating system holds it: a byte-range lock on `OUT_DIR/plan.lock`,
dropped by Windows when the holder exits or dies. There is no stale lock to judge, so two
guards can't both take one over. The file stays behind; its text shows the holder's pid. If
writing that text fails, the lock is released before the error is raised. A
leftover file, from v1 or from a guard that died, is not a lock. The lock covers one
`OUT_DIR`; keeping to one guard on the whole machine stays the runner's job (its `GPU.lock`).

Not carried over: v1's `queue` mode (long-lived workers run many jobs per process, so a
per-job peak is not defined, and no current runner uses it) and `stop --kill`.

## Settings (environment variables)

| name | default | role |
|---|---|---|
| `MAX_PROCS` | 1 | jobs running at once |
| `MIN_DISK_GB` | 9 | admission floor (both drives) and stop-everything floor |
| `PAGEFILE_DRIVE` | `%SystemDrive%\` | drive whose free space the page file grows into |
| `MIN_COMMIT_GB` | 3 | admission floor for commit after the need |
| `ADMIT_RAM_FLOOR_GB` | 0.4 | admission floor for RAM after the need |
| `ADMIT_CONFIRM` | 2 | checks in a row the room must hold |
| `ADMIT_WAIT_S` | 1800 | longest wait for room |
| `ADMIT_TIMEOUT` | fail | `fail` or `start` |
| `EMERG_COMMIT_GB` | 1.5 | shed at once below this free commit |
| `EMERG_RAM_GB` | 0.15 | shed below this free RAM ... |
| `EMERG_RAM_HOLD_S` | 60 | ... held this long |
| `JOB_COMMIT_LIMIT_GB` | 0 (off) | hard commit ceiling for one job's tree |
| `JOB_TIMEOUT_S` | 0 (off) | kill a job that runs longer |
| `DEFAULT_RAM_GB` | 1.3 | need when nothing is declared or measured |
| `DEFAULT_COMMIT_GB` | 3.0 | same for commit |
| `NEED_MARGIN` | 0.10 | added to measured peaks |
| `HISTORY_N` | 5 | recent runs a measured need looks at |
| `GUARD_MEM_HISTORY` | `~/.relsm_guard/mem_history.jsonl` | measurement log |
| `MAX_FAILS` | 3 | sheds before giving up on a job |
| `COOLDOWN_S` | 90 | no start for this long after a shed |
| `START_GAP_S` | 20 | at least this long between starts |
| `TICK_S` | 2 | decision interval |
| `MEAS_TICK_S` | 0.25 | memory reads of running jobs between decisions |
| `STRAY_WAIT_S`, `KILL_WAIT_S` | 10, 30 | wait for a tree to empty after the root exits / after a kill |

`PER_PROC_GB`, `PER_PROC_RAM_GB` and `MIN_RAM_GB` from v1 are not read. Their roles are now
`DEFAULT_COMMIT_GB`, `DEFAULT_RAM_GB` and `ADMIT_RAM_FLOOR_GB`. If any is set, v2 logs a
`GUARD WARNING`, so a copied v1 environment gets noticed.

## Switching over (after phase 2 ends, in a new study only)

The hash-locked runner and v1 guard stay as they are. In the new study:

1. Copy the runner to a new file; do not edit the hash-locked one.
2. In the copy, point `GUARD` at `tools/probe_guard_v2.py` of a checkout that has it, and set
   the v2 names in `GUARD_ENV`. Copy the optional fields (`ram_gb`, `commit_gb`,
   `mem_override`, `mem_key`, `mem_family`, `retry`, `chain`, `state_dir`) into the plan
   along with `name`, `cmd` and `env`. Today's runner sends one job per guard call. To get
   the chain rule, it must send a chain's members in one plan, or else track chains itself
   and pass `retry: "none"` plus `state_dir` on every member.
3. Handle the guard's exit code **before** the validator and retry accounting. Today's runner
   increments `reruns.json` before it calls the guard, so a job that was never started (exit 8),
   refused (6) or stopped (2) would use up its one rerun. Count a rerun only when the guard
   started the job, treat 8 as "not run, try in the next session", and stop the session on
   3, 4 and 7, not only on 3 (7 needs a person to look at the marked folder).
4. If the runner has a time budget, pass the time left as `ADMIT_WAIT_S` for each job, so
   waiting for room cannot run past the budget.
5. The first runs of each job family use the defaults. Run `estimate PLAN.json` before a long
   study to see which jobs still have no measured need.

## Open questions for the owner

- `ADMIT_TIMEOUT` default: `fail` never runs a job under pressure but can leave jobs for the
  next session on a busy machine; `start` runs them and relies on the emergency rules.
- If `peak_ws_gb` proves much larger than what a job really draws (it counts shared DLL
  pages, and its bound can sit near the peak commit), jobs will wait longer than they need to. The log also records the drop in free RAM. Admitting on that drop
  instead would be unsafe, because other programs on the machine move it too.
