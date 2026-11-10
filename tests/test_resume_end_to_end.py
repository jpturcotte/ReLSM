"""End-to-end resume equivalence for train.py: a run killed and resumed ends bit-identical to one that never stopped.

Two paths, each a tiny CPU config run in fresh subprocesses with one thread:
- main path (--resume_from, docs/resume.md): killed mid-accumulation (RELSM_TEST_KILL_STEP 9:step), resumed from
  checkpoint_step_6 (itself mid-accumulation), killed again between checkpoint_step_12's payload and its sidecar
  (12:ckpt:payload), resumed from that checkpoint (no sidecar);
- bridge path (--mod_table_p with --mod_table_resume auto, rerun in place): killed after update 5's optimizer step
  (RELSM_TEST_KILL 5:step), rerun from update 3's resume.pt, killed between update 6's resume.pt and its sidecar
  (6:replaced), rerun to the end from resume.prev.pt (--mod_table_resume_fallback previous; that resume.pt fails its
  check against the sidecar it still has from update 3).

Each chain is compared with its uninterrupted reference: the end-of-run digests (weights, optimizer, EMA, buffers,
gradients, scaler, RNGs; RELSM_TEST_DIGEST), the per-micro-batch hashes (RELSM_TEST_BATCH_LOG), metrics.json's rows
(floats bitwise, timing fields excluded), summary.json, and final_model.pt and best_model.pt field by field (best
accuracy, curriculum RNG and counters; run_id excluded); the bridge also its table_scores.jsonl, bridge_record.json and
final resume.pt (loader position, curriculum RNG, logging window). Each reference must reach its token budget, and each
resume must come from the intended file (the resume entries; the bridge's resume_log.jsonl and file checks between its
legs). The resume checkpoints fall inside a logging window, so the window and the rewound metrics rows are restored,
not just reset. One more case: a resume with a changed --max_lr is refused and leaves the directory byte-identical.

The jobs run as four sequences (the reference runs, and each chain's legs in order), one after another by default, so
one train.py process runs at a time (about 2 GB of RAM; four at once peaked near 5 GB). RELSM_E2E_PARALLEL=N runs up
to N sequences at once. Every subprocess pays train.py's import time. The checkpoints peak at about 0.8 GB on disk
and are deleted when the module ends.
"""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import exact_resume as er  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TIMEOUT_S = 900
KILLED_RC = 86  # os._exit code of both kill hooks
# wall-clock fields: metrics.json's throughput and VRAM, table_scores.jsonl's scorer time
TIMING = {"train/throughput/tokens_per_sec", "eval/throughput/eval_duration", "train/throughput/vram_gb", "score_s"}
# summary.json: wall-clock fields, and the resume list (the reference has none)
SUMMARY_SKIP = {"total_time_hours", "avg_tokens_per_sec", "peak_vram_gb", "resumes"}
MAIN_TOKENS, BRIDGE_TOKENS = 800, 1700

# checkpoints at micro-steps 6, 12, 18 (mid-accumulation), logged every 4: each resume lands inside a logging window
MAIN_ARGS = [
    "--model_size", "grok", "--variant", "baseline", "--num_workers", "0", "--seed", "0", "--tie_embeddings",
    "--grad_accum_steps", "4", "--eval_interval", "6", "--eval_checkpoints", "2", "--log_interval", "4",
    "--eval_samples", "2", "--eval_sample_count_per_task", "1", "--dropout", "0.1", "--grokfast",
    "--alg_batch_size", "8", "--alg_seq_len", "32", "--lang_seq_len", "64", "--lang_batch_size", "2",
    "--difficulty_schedule", "smooth", "--alg_examples", "64",
    "--alg_tokens", str(MAIN_TOKENS), "--total_tokens", str(MAIN_TOKENS), "--train_eval_mode", "none",
]
# resume.pt every 3 updates (micro-step 12, 24, ...), logged every 5 micro-steps: inside a logging window too
BRIDGE_ARGS = [
    "--model_size", "grok", "--variant", "baseline", "--ffn_rule", "llama", "--alg_tasks", "mod_add",
    "--fixed_data", "--mod_table_p", "11", "--mod_table_frac", "0.5", "--mod_table_pairs", "paired",
    "--mod_table_tasks", "add", "--alg_seq_len", "32", "--alg_batch_size", "16", "--grad_accum_steps", "4",
    "--num_workers", "0", "--max_lr", "6e-4", "--weight_decay", "1.0", "--grad_clip", "1.0",
    "--alg_tokens", str(BRIDGE_TOKENS), "--total_tokens", str(BRIDGE_TOKENS), "--mix_band_tokens", "0",
    "--lexical_frac_phase1", "0", "--no-eval_algorithmic", "--train_eval_mode", "none",
    "--eval_interval", "100000000", "--eval_checkpoints", "1",
    "--log_interval", "5", "--mod_table_eval_every", "5", "--tie_embeddings", "--dropout", "0.1",
    "--lr_schedule", "constant", "--warmup_tokens", "500", "--seed", "0",
    "--mod_table_resume", "auto", "--mod_table_resume_every", "3", "--mod_table_resume_fallback", "previous",
]


def _tokenizer_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    from utils import DEFAULT_TOKENIZER_NAME  # a broken project import fails collection, never skips

    return any(isinstance(try_to_load_from_cache(DEFAULT_TOKENIZER_NAME, f), str)
               for f in ("tokenizer.json", "tokenizer.model"))


pytestmark = pytest.mark.skipif(not _tokenizer_cached(), reason="train.py's default tokenizer is not in the local "
                                "Hugging Face cache (these runs are offline)")


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELSM_TEST_")}
    env.update(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.update({k: str(v) for k, v in extra.items()})
    return env


class Job:
    def __init__(self, name, root, args, out, kill=None, resume_from=None, trace=True):
        """kill: (env var, spec) for one one-shot kill; trace: write this job's batch log and digest."""
        self.name, self.root = name, root
        self.log, self.marker = root / f"{name}.log", root / f"{name}.killed"
        self.batches, self.digest = root / f"{name}.batches", root / f"{name}.digest.json"
        extra = {}
        if trace:
            extra.update(RELSM_TEST_BATCH_LOG=self.batches, RELSM_TEST_DIGEST=self.digest)
        if kill:
            extra.update({kill[0]: kill[1], "RELSM_TEST_KILL_MARKER": self.marker})
        self.cmd = [sys.executable, str(REPO / "train.py"), *args, "--output_dir", str(out)]
        if resume_from:
            self.cmd += ["--resume_from", str(resume_from)]
        self.env = _env(**extra)
        self.rc = None

    def start(self):
        self._f = open(self.log, "w", encoding="utf-8")
        self._p = subprocess.Popen(self.cmd, cwd=REPO, env=self.env, stdout=self._f, stderr=subprocess.STDOUT)
        return self

    def wait(self):
        try:
            self.rc = self._p.wait(timeout=TIMEOUT_S)
        finally:
            if self._p.poll() is None:
                self._p.kill()
                self._p.wait()  # reaped, its files closed, before the log closes and teardown deletes checkpoints
            self._f.close()
        return self

    def tail(self, n=40):
        return "\n".join(self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _sequence(*steps):
    """Run steps (a Job, or a callable for work between jobs) one after another."""
    def run():
        for step in steps:
            if isinstance(step, Job):
                step.start().wait()
            else:
                step()
    return run


def _tree(d):
    """Every file under d, by relative path, with the sha256 of its bytes."""
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume_e2e")
    d = {"main_ref": root / "main_ref", "main": root / "main_chain", "bridge_ref": root / "bridge_ref",
         "bridge": root / "bridge_chain"}
    kill_main, kill_bridge = "RELSM_TEST_KILL_STEP", "RELSM_TEST_KILL"
    jobs = {
        "main_ref": Job("main_ref", root, MAIN_ARGS, d["main_ref"]),
        "main_1": Job("main_1", root, MAIN_ARGS, d["main"], kill=(kill_main, "9:step")),
        "main_2": Job("main_2", root, MAIN_ARGS, d["main"], kill=(kill_main, "12:ckpt:payload"),
                      resume_from=d["main"] / "checkpoint_step_6.pt"),
        "main_3": Job("main_3", root, MAIN_ARGS, d["main"], resume_from=d["main"] / "checkpoint_step_12.pt"),
        "bridge_ref": Job("bridge_ref", root, BRIDGE_ARGS, d["bridge_ref"]),
        "bridge_1": Job("bridge_1", root, BRIDGE_ARGS, d["bridge"], kill=(kill_bridge, "5:step")),
        "bridge_2": Job("bridge_2", root, BRIDGE_ARGS, d["bridge"], kill=(kill_bridge, "6:replaced")),
        "bridge_3": Job("bridge_3", root, BRIDGE_ARGS, d["bridge"]),
        # on the finished reference itself (the reference ran with the default --max_lr 6e-4): a resume that went
        # ahead would move its outputs, which the refusal test sees
        "refuse": Job("refuse", root, MAIN_ARGS + ["--max_lr", "1e-3"], d["main_ref"],
                      resume_from=d["main_ref"] / "checkpoint_step_18.pt", trace=False),
    }
    before, between = {}, {}

    def snapshot_reference():
        before["refuse"] = _tree(d["main_ref"]) if jobs["main_ref"].rc == 0 else None

    def record_main_run_id():
        if (d["main"] / "checkpoint_step_6.pt").exists():
            between["main_run_id"] = _load(d["main"] / "checkpoint_step_6.pt")["run_id"]

    def check_bridge_pair():
        # after the 6:replaced kill: resume.pt (update 6) beside update 3's sidecar fails, resume.prev.pt verifies
        from tools.grok_bridge import check_pair

        between["bridge"] = {q: check_pair(d["bridge"] / q) for q in ("resume.pt", "resume.prev.pt")}

    sequences = [
        _sequence(jobs["main_ref"], snapshot_reference, jobs["refuse"]),
        _sequence(jobs["main_1"], record_main_run_id, jobs["main_2"], jobs["main_3"]),
        _sequence(jobs["bridge_ref"]),
        _sequence(jobs["bridge_1"], jobs["bridge_2"], check_bridge_pair, jobs["bridge_3"]),
    ]
    try:
        workers = max(1, min(int(os.environ.get("RELSM_E2E_PARALLEL", "1")), len(sequences)))
        with ThreadPoolExecutor(workers) as pool:
            for f in [pool.submit(s) for s in sequences]:
                f.result()
        yield {"root": root, "jobs": jobs, "dirs": d, "refuse_before": before.get("refuse"), "between": between}
    finally:
        # the checkpoints are about 0.8 GB per session and pytest keeps its last few temp dirs: delete them, keep the
        # logs, batch logs, digests and metrics for a failure
        for p in root.rglob("*"):
            if p.is_file() and ".pt" in p.name and not p.name.endswith(".sha256"):
                p.unlink()


def _rc(job, want):
    assert job.rc == want, f"{job.name}: exit code {job.rc}, expected {want}\n{job.tail()}"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _trace(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m, h = line.split()
        out.append((int(m), h))
    return out


def _rows(out_dir, name="metrics.json"):
    """A JSON-lines file's rows other than the hyperparameter row, timing fields dropped, as exact JSON text."""
    rows = [json.loads(line) for line in (Path(out_dir) / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [json.dumps({k: v for k, v in r.items() if k not in TIMING}) for r in rows
            if r.get("type") != "hyperparameters"]


def _first_row_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            dx, dy = json.loads(x), json.loads(y)
            keys = sorted(k for k in set(dx) | set(dy) if json.dumps(dx.get(k)) != json.dumps(dy.get(k)))
            return f"row {i} (step {dx.get('step')}): {keys[:8]}"
    return None if len(a) == len(b) else f"{len(a)} rows vs {len(b)}"


def _field_digests(obj, skip=()):
    """A loaded payload as {field: digest} (tensors by dtype, shape and bytes); skip: field names, or "a.b" for a
    field inside the dict field a."""
    out = {}
    for k, v in obj.items():
        if k in skip:
            continue
        inner = {s.split(".", 1)[1] for s in skip if s.startswith(f"{k}.")}
        if inner and isinstance(v, dict):
            v = {ik: iv for ik, iv in v.items() if ik not in inner}
        out[k] = er.digest_of(v)
    return out


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _same_fields(name, a, b, skip=()):
    a, b = _field_digests(a, skip), _field_digests(b, skip)
    assert a == b, f"{name} differs in " + str(sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))


def _same_file_state(ref_path, chain_path, skip=(), required=True):
    """Every field but skip is equal. required=False: both files may be absent (a best model may never be chosen)."""
    assert ref_path.exists() == chain_path.exists(), f"{ref_path.name}: present {ref_path.exists()} vs " \
                                                     f"{chain_path.exists()}"
    assert ref_path.exists() or not required, f"{ref_path.name} is missing"
    if ref_path.exists():
        _same_fields(ref_path.name, _load(ref_path), _load(chain_path), skip)


def _same_final_resume(ref_dir, chain_dir, digest):
    """The bridge's final resume.pt (what --mod_table_init_from extends from): written at the end of the run, and
    equal to the reference's in every field, the embedded copies included. Wall-clock values are dropped; args and
    the bridge record are compared as a resume compares them (output_dir and the free arguments may differ)."""
    from tools.grok_bridge import RESUME_FREE_ARGS, arg_differences, record_differences

    a, b = _load(ref_dir / "resume.pt"), _load(chain_dir / "resume.pt")
    for s in (a, b):
        assert (s["kind"], s["micro_steps"], s["tokens_seen"]) == ("final", digest["micro_steps"],
                                                                  digest["tokens_seen"])
    assert arg_differences(a["args"], b["args"], RESUME_FREE_ARGS | {"output_dir"}) == []
    assert record_differences(a["bridge_record"], b["bridge_record"]) == []

    def logger(s):
        lg = dict(s["logger"])
        lg["records"] = [{k: v for k, v in r.items() if k not in TIMING} for r in lg["records"]]
        assert lg["records"], "the embedded metrics history is empty"
        return lg

    def table_files(s):
        out = {}
        for name, t in s["table_files"].items():
            lines = t.numpy().tobytes().decode("utf-8").splitlines()
            out[name] = [{k: v for k, v in json.loads(ln).items() if k not in TIMING} for ln in lines if ln.strip()]
        assert any(out.values()), "the embedded table scores are empty"
        return out

    _same_fields("resume.pt", {"logger": logger(a), "table_files": table_files(a)},
                 {"logger": logger(b), "table_files": table_files(b)})
    _same_fields("resume.pt", a, b, skip=("saved_at", "args", "bridge_record", "logger", "table_files",
                                           "loop.train_seconds", "loop.checkpoint_paths", "run_record"))
    # its run record (#187): the reference's, and the chain's names the resume.prev.pt it reran from
    _same_records("resume.pt", a["run_record"], b["run_record"], free=RESUME_FREE_ARGS)
    assert _lineage(a["run_record"]) == [] and _lineage(b["run_record"]) == [("bridge-resume", "resume.prev.pt")]


def _same_records(name, a, b, free=()):
    """Two run records (#187) say the same about the run: arguments (but output_dir, resume_from and free), code,
    commit, environment, resolved values, optimizer and progress at the save. Wall-clock, argv, lineage differ."""
    assert isinstance(a, dict) and isinstance(b, dict), f"{name}: run_record missing ({type(a)}, {type(b)})"
    diffs = er.arg_differences(a["args"], b["args"])
    assert not [x for x in diffs if x.split(":")[0].lstrip("-") not in free], f"{name}: {diffs}"
    for k in ("format", "git", "fingerprint", "resolved", "optimizer", "seed"):
        assert a[k] == b[k], f"{name}: run_record.{k} differs"
    for k in ("micro_steps", "optimizer_steps", "tokens_seen"):
        assert a["saved"][k] == b["saved"][k], f"{name}: run_record.saved.{k} differs"


def _lineage(rec):
    """[(kind, file name), ...] of the files a record's run resumed from, latest first; every one carries a record."""
    chain = [e for e in [rec["parent"], *rec["ancestors"]] if e is not None]
    assert all(isinstance(e["record"], dict) and len(e["sha256"] or "") == 64 for e in chain), chain
    return [(e["kind"], e["name"]) for e in chain]


def _same_run_record(ref_path, chain_path, parents, free=()):
    """The chain's file carries the reference's run record (_same_records) and names the files the chain resumed
    from (parents, latest first); the reference resumed from none."""
    a, b = _load(ref_path)["run_record"], _load(chain_path)["run_record"]
    _same_records(chain_path.name, a, b, free)
    assert _lineage(a) == [] and _lineage(b) == parents, (_lineage(a), _lineage(b))


def _same_summary(ref_dir, chain_dir):
    a, b = ({k: v for k, v in _json(x / "summary.json").items() if k not in SUMMARY_SKIP} for x in (ref_dir, chain_dir))
    differ = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k)) != json.dumps(b.get(k)))
    assert not differ, f"summary.json differs in {differ}"


def _check_chain(runs, ref_name, legs, kills, resume_starts, tokens):
    """legs: the chain's jobs in order; kills: the expected marker of each killed leg; resume_starts: the first
    micro-step each later leg trains (the one after the state it continued from); tokens: the run's budget."""
    jobs = runs["jobs"]
    ref = jobs[ref_name]
    _rc(ref, 0)
    for leg, want in zip(legs[:-1], kills):
        _rc(jobs[leg], KILLED_RC)
        assert want in jobs[leg].marker.read_text(encoding="utf-8"), jobs[leg].tail()
    _rc(jobs[legs[-1]], 0)

    ref_trace = _trace(ref.batches)
    ref_by_step = dict(ref_trace)
    assert [m for m, _ in ref_trace] == list(range(1, len(ref_trace) + 1))
    merged = {}
    for leg, start in zip(legs, [1, *resume_starts]):
        t = _trace(jobs[leg].batches)
        assert t and t[0][0] == start, f"{leg} starts at micro-step {t[0][0] if t else None}, expected {start}"
        for m, h in t:
            assert ref_by_step.get(m) == h, f"{leg}: batch at micro-step {m} differs from the reference"
            merged[m] = h
    assert merged == ref_by_step

    a, b = _json(ref.digest), _json(jobs[legs[-1]].digest)
    assert a == b, "end-of-run state differs: " + str(sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))
    # the reference itself ran to its budget (a shared early stop would leave both equally short)
    assert a["tokens_seen"] >= tokens and a["micro_steps"] == len(ref_trace), (a["tokens_seen"], a["micro_steps"])


def test_main_path_killed_and_resumed_matches_an_uninterrupted_run(runs):
    _check_chain(runs, "main_ref", ["main_1", "main_2", "main_3"],
                 kills=['"step": 9, "point": "step"', '"step": 12, "point": "ckpt:payload"'],
                 resume_starts=[7, 13], tokens=MAIN_TOKENS)
    d = runs["dirs"]
    ref_rows, chain_rows = _rows(d["main_ref"]), _rows(d["main"])
    assert chain_rows == ref_rows, _first_row_diff(ref_rows, chain_rows)
    kinds = [json.loads(r)["type"] for r in ref_rows]
    assert kinds.count("record") >= 4 and "task_accuracy" in kinds, kinds
    _same_summary(d["main_ref"], d["main"])
    _same_file_state(d["main_ref"] / "final_model.pt", d["main"] / "final_model.pt", skip=("run_id", "run_record"))
    _same_file_state(d["main_ref"] / "best_model.pt", d["main"] / "best_model.pt", skip=("run_id", "run_record"),
                     required=False)
    _same_run_record(d["main_ref"] / "final_model.pt", d["main"] / "final_model.pt",
                     [("exact", "checkpoint_step_12.pt"), ("exact", "checkpoint_step_6.pt")])
    # one run identity across the chain (the first leg's), not the reference's
    hyp = json.loads((d["main"] / "metrics.json").read_text(encoding="utf-8").splitlines()[0])
    run_id = runs["between"].get("main_run_id")
    assert run_id and run_id != _load(d["main_ref"] / "final_model.pt")["run_id"], runs["between"]
    assert (hyp["data"]["run_id"], _load(d["main"] / "final_model.pt")["run_id"]) == (run_id, run_id)
    resumes = hyp["data"]["resumes"]
    assert [(r["kind"], r["step"], Path(r["from"]).name) for r in resumes] == [
        ("exact", 6, "checkpoint_step_6.pt"), ("exact", 12, "checkpoint_step_12.pt")], resumes


def test_bridge_path_killed_and_rerun_matches_an_uninterrupted_run(runs):
    from tools.grok_bridge import RESUME_FREE_ARGS

    # resume.pt is written every 3 updates (12 micro-steps). Kill at update 5: the rerun continues after update 3.
    # Kill at update 6 between resume.pt and its sidecar: that resume.pt fails its check (its sidecar is still update
    # 3's), so the rerun falls back to resume.prev.pt (update 3 again; --mod_table_resume_fallback previous)
    pair = runs["between"].get("bridge") or {}
    assert pair.get("resume.prev.pt", "absent") is None, pair
    assert str(pair.get("resume.pt")).startswith("size "), pair  # "size N != sidecar M"
    _check_chain(runs, "bridge_ref", ["bridge_1", "bridge_2", "bridge_3"],
                 kills=["update 5 (step)", "update 6 (replaced)"], resume_starts=[13, 13], tokens=BRIDGE_TOKENS)
    d = runs["dirs"]
    log = [json.loads(line) for line in (d["bridge"] / "resume_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(e["update"], e["checkpoint"]["file"], e["checkpoint"]["fallback"], e["checkpoint"]["rejected"])
            for e in log] == [(3, "resume.pt", False, []),
                              (3, "resume.prev.pt", True, [{"file": "resume.pt", "reason": pair["resume.pt"]}])], log
    ref_rows, chain_rows = _rows(d["bridge_ref"]), _rows(d["bridge"])
    assert chain_rows == ref_rows, _first_row_diff(ref_rows, chain_rows)
    assert len(ref_rows) >= 4, ref_rows
    ref_scores, chain_scores = _rows(d["bridge_ref"], "table_scores.jsonl"), _rows(d["bridge"], "table_scores.jsonl")
    assert chain_scores == ref_scores, _first_row_diff(ref_scores, chain_scores)
    assert len(ref_scores) >= 3, ref_scores
    _same_summary(d["bridge_ref"], d["bridge"])
    _same_file_state(d["bridge_ref"] / "final_model.pt", d["bridge"] / "final_model.pt",
                     skip=("run_id", "run_record"))
    # the last leg reran from resume.prev.pt (update 3), written by the first leg, which started afresh
    _same_run_record(d["bridge_ref"] / "final_model.pt", d["bridge"] / "final_model.pt",
                     [("bridge-resume", "resume.prev.pt")], free=RESUME_FREE_ARGS)
    # the final resume.pt: what the loop carries (loader position, curriculum RNG, logging window, best accuracy)
    _same_final_resume(d["bridge_ref"], d["bridge"], _json(runs["jobs"]["bridge_ref"].digest))
    rec = {k: _json(d[k] / "bridge_record.json") for k in ("bridge_ref", "bridge")}
    keys = ("status", "updates", "micro_steps", "tokens_seen", "joins", "final_state_sha256")
    assert rec["bridge_ref"]["status"] == "finished", rec["bridge_ref"].get("status")
    assert [rec["bridge"].get(k) for k in keys] == [rec["bridge_ref"].get(k) for k in keys]


def test_a_changed_max_lr_is_refused_and_changes_nothing(runs):
    job, before = runs["jobs"]["refuse"], runs["refuse_before"]
    assert before is not None, "the reference run failed, so there was nothing to refuse"
    assert job.rc not in (None, 0, KILLED_RC), f"refuse: exit code {job.rc}\n{job.tail()}"
    assert "max_lr" in job.log.read_text(encoding="utf-8", errors="replace"), job.tail()
    assert _tree(runs["dirs"]["main_ref"]) == before
