"""The run record every checkpoint carries (#187): its contents, and that each checkpoint kind loads with
torch.load(weights_only=True) and holds it.

The unit tests need no training run. The run tests start train.py on a tiny CPU config in fresh subprocesses (as
tests/test_resume_end_to_end.py does): one run interrupted part-way (rotating checkpoints, the interrupt
checkpoint, best_model.pt, final_model.pt), then a resume from a copy of its final model with the record removed (a
file written before #187), itself interrupted after one micro-step; a relaunch from the record's command line, whose
first batches match; and resumes of best_model.pt with a changed --grad_accum_steps, which name the flag.
"""
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import run_record as rr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HAVE_GIT = shutil.which("git") is not None


def roundtrip(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


def _git(cwd, *argv):
    subprocess.run(["git", "-C", str(cwd), *argv], check=True, capture_output=True)


# ---------------------------------------------------------------------------------------------------------------
# git state
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_GIT, reason="git is not installed")
def test_git_state_reads_commit_dirty_diff_and_untracked(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()

    clean = rr.git_state(tmp_path)
    assert clean == {"commit": head, "dirty": False, "diff_sha256": None, "untracked_entries": 0, "error": None}

    (tmp_path / "b.txt").write_text("new\n")  # untracked only: not dirty
    st = rr.git_state(tmp_path)
    assert st["dirty"] is False and st["untracked_entries"] == 1
    (tmp_path / "out").mkdir()  # an untracked directory is one entry, however many files it holds
    for k in range(3):
        (tmp_path / "out" / f"f{k}.bin").write_bytes(b"x")
    assert rr.git_state(tmp_path)["untracked_entries"] == 2

    (tmp_path / "a.py").write_text("x = 2\n")
    dirty = rr.git_state(tmp_path)
    assert dirty["commit"] == head and dirty["dirty"] is True and len(dirty["diff_sha256"]) == 64
    (tmp_path / "a.py").write_text("x = 3\n")
    assert rr.git_state(tmp_path)["diff_sha256"] != dirty["diff_sha256"]  # the hash follows the diff

    _git(tmp_path, "add", "a.py")  # staged changes are dirty too
    assert rr.git_state(tmp_path)["dirty"] is True


@pytest.mark.skipif(not HAVE_GIT, reason="git is not installed")
def test_git_state_outside_a_repository_says_why(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    st = rr.git_state(tmp_path)
    assert (st["commit"], st["dirty"], st["diff_sha256"], st["untracked_entries"]) == (None, None, None, None)
    assert st["error"] and "CalledProcessError" in st["error"]


def test_git_state_without_git_does_not_fail(tmp_path, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(rr.subprocess, "run", no_git)
    st = rr.git_state(tmp_path)
    assert st["commit"] is None and st["error"].startswith("FileNotFoundError")


# ---------------------------------------------------------------------------------------------------------------
# contents
# ---------------------------------------------------------------------------------------------------------------

class Tiny(torch.nn.Module):
    def __init__(self, tie=True):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(10, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 10, bias=False)
        if tie:
            self.lm_head.weight = self.tok_emb.weight


def _adamw(model):
    decay = [model.tok_emb.weight]
    no_decay = [model.norm.weight, model.norm.bias]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=1e-3, betas=(0.9, 0.95), eps=1e-8)


def test_optimizer_setup_and_step_count():
    model = Tiny()
    opt = _adamw(model)
    setup = rr.optimizer_setup(opt)
    assert setup == {"class": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8,
                     "groups": [{"weight_decay": 0.1, "n_tensors": 1, "n_params": 40},
                                {"weight_decay": 0.0, "n_tensors": 2, "n_params": 8}]}
    assert rr.optimizer_steps(opt) == 0
    for k in (1, 2):
        model.lm_head(model.norm(model.tok_emb(torch.tensor([1, 2])))).sum().backward()
        opt.step()
        opt.zero_grad()
        assert rr.optimizer_steps(opt) == k
    fresh = _adamw(model)
    fresh.load_state_dict(roundtrip(opt.state_dict()))
    assert rr.optimizer_steps(fresh) == 2  # after a state_dict reload too


def test_model_counts_count_tied_weights_once():
    assert rr.model_counts(Tiny(tie=True)) == {"n_params_unique": 48, "n_params_trainable": 48, "n_tensors": 3,
                                               "tied_embeddings": True}
    assert rr.model_counts(Tiny(tie=False))["n_params_unique"] == 88
    assert rr.model_counts(Tiny(tie=False))["tied_embeddings"] is False


def _record(**over):
    model = Tiny()
    kw = dict(run_id="rid", args={"seed": 7, "grad_accum_steps": 4, "lr_schedule": "wsd", "warmup_tokens": None},
              fingerprint={"code": "c" * 64, "torch": torch.__version__}, git={"commit": "a" * 40, "dirty": False},
              resolved={**rr.model_counts(model), "train_autocast_dtype": None}, optimizer=rr.optimizer_setup(
                  _adamw(model)), resumed=None, argv=["train.py", "--seed", "7"])
    kw.update(over)
    return rr.build(**kw), model


def test_record_holds_the_run_description_and_survives_a_weights_only_load():
    record, model = _record()
    assert record["format"] == rr.RECORD_FORMAT and record["run_id"] == "rid"
    assert record["argv"] == ["train.py", "--seed", "7"] and record["seed"] == 7
    assert record["args"]["lr_schedule"] == "wsd" and record["args"]["warmup_tokens"] is None
    assert record["git"]["commit"] == "a" * 40 and record["fingerprint"]["code"] == "c" * 64
    env = record["environment"]
    assert env["platform"] and env["host"] is not None and env["torch_cuda"] == torch.version.cuda
    assert set(env["env"]) >= {"CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED"}
    assert record["started"] and record["parent"] is None and record["ancestors"] == []

    saved = rr.at_save(record, _adamw(model), micro_steps=12, tokens_seen=345)
    assert saved["saved"]["micro_steps"] == 12 and saved["saved"]["tokens_seen"] == 345
    assert saved["saved"]["optimizer_steps"] == 0 and "saved" not in record  # the start record is not changed
    payload = {"model_state_dict": model.state_dict(), "run_record": saved}
    back = roundtrip(payload)  # weights_only=True
    assert back["run_record"] == saved
    assert json.loads(json.dumps(back["run_record"])) == saved  # JSON values only


def test_record_refuses_values_that_are_not_plain():
    with pytest.raises(TypeError):
        _record(args={"output_dir": Path("x")})
    with pytest.raises(TypeError):
        _record(resolved={"t": torch.zeros(1)})


def test_lineage_is_flat_latest_first():
    first, _ = _record(run_id="r1")
    l1 = rr.lineage(Path("out") / "checkpoint_step_6.pt", "f" * 64, "exact", 6,
                    rr.at_save(first, _adamw(Tiny()), 6, 100))
    second, _ = _record(run_id="r1", resumed=l1)
    assert second["parent"]["name"] == "checkpoint_step_6.pt" and second["ancestors"] == []
    l2 = rr.lineage(Path("out") / "checkpoint_step_12.pt", None, "inexact", 12, rr.at_save(second, _adamw(Tiny()),
                                                                                          12, 200))
    third, _ = _record(run_id="r1", resumed=l2)
    p, anc = third["parent"], third["ancestors"]
    assert (p["name"], p["kind"], p["step"], p["sha256"]) == ("checkpoint_step_12.pt", "inexact", 12, None)
    assert [a["name"] for a in anc] == ["checkpoint_step_6.pt"] and anc[0]["sha256"] == "f" * 64
    # every record in the chain is stripped of its own lineage, so nothing nests
    for e in (p, *anc):
        assert "parent" not in e["record"] and "ancestors" not in e["record"] and e["record"]["saved"]
    assert p["record"]["saved"]["micro_steps"] == 12 and anc[0]["record"]["saved"]["micro_steps"] == 6
    legacy = rr.lineage(Path("old.pt"), None, "inexact", 3, None)  # a file written before #187
    assert legacy == {"parent": {"file": "old.pt", "name": "old.pt", "sha256": None, "kind": "inexact", "step": 3,
                                 "record": None}, "ancestors": []}
    roundtrip({"run_record": third})


def test_a_long_resume_chain_grows_by_one_entry_and_does_not_nest():
    rec, _ = _record()
    # 400 resumes: records nested parent -> record -> parent would be 1200 levels deep, past json's recursion limit
    for k in range(1, 401):
        rec = rr.build(**{**_kw_of(rec), "resumed": rr.lineage(Path(f"c{k}.pt"), None, "exact", k, rec)})
    assert len(rec["ancestors"]) == 399 and rec["parent"]["name"] == "c400.pt"
    assert [a["step"] for a in rec["ancestors"][:3]] == [399, 398, 397] and rec["ancestors"][-1]["step"] == 1
    assert all("parent" not in a["record"] and "ancestors" not in a["record"] for a in rec["ancestors"])
    roundtrip({"run_record": rec})


def _kw_of(rec):
    return dict(run_id=rec["run_id"], args=rec["args"], fingerprint=rec["fingerprint"], git=rec["git"],
                resolved=rec["resolved"], optimizer=rec["optimizer"], argv=rec["argv"], env=rec["environment"])


# ---------------------------------------------------------------------------------------------------------------
# train.py: every checkpoint kind carries the record
# ---------------------------------------------------------------------------------------------------------------

def _tokenizer_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    from utils import DEFAULT_TOKENIZER_NAME

    return any(isinstance(try_to_load_from_cache(DEFAULT_TOKENIZER_NAME, f), str)
               for f in ("tokenizer.json", "tokenizer.model"))


needs_tokenizer = pytest.mark.skipif(not _tokenizer_cached(), reason="train.py's default tokenizer is not in the local "
                                     "Hugging Face cache (these runs are offline)")
ACCUM, INTERRUPT_AT = 4, 60
# evaluations and rotating checkpoints every 20 micro-steps (best_model.pt at each, through RELSM_TEST_BEST_ACC), an
# interrupt at micro-step 60, a budget never reached
RUN_ARGS = [
    "--model_size", "grok", "--variant", "baseline", "--num_workers", "0", "--seed", "0", "--tie_embeddings",
    "--alg_tasks", "parity", "--grad_accum_steps", str(ACCUM), "--eval_interval", "20", "--eval_checkpoints", "2",
    "--log_interval", "10", "--eval_samples", "16", "--eval_sample_count_per_task", "16",
    "--alg_batch_size", "64", "--alg_seq_len", "32", "--lang_seq_len", "64", "--lang_batch_size", "2",
    "--alg_examples", "256", "--alg_tokens", "1000000", "--total_tokens", "1000000", "--train_eval_mode", "none",
    "--lr_schedule", "constant", "--warmup_tokens", "200", "--max_lr", "3e-3",
]


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELSM_TEST_")}
    env.update(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _train(log, *argv, **env):
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, str(REPO / "train.py"), *argv], cwd=REPO, env=_env(**env), stdout=f,
                           stderr=subprocess.STDOUT, timeout=900)
    return p.returncode


def _tail(log, n=40):
    return "\n".join(Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("run_record")
    out = root / "run"
    rc = _train(root / "run.log", *RUN_ARGS, "--output_dir", str(out),
                RELSM_TEST_KILL_STEP=f"{INTERRUPT_AT}:interrupt", RELSM_TEST_KILL_MARKER=root / "run.killed",
                RELSM_TEST_BEST_ACC="0.001",  # a short run scores 0: the hook makes each evaluation a new best
                RELSM_TEST_BATCH_LOG=root / "run.batches")
    try:
        yield {"root": root, "out": out, "rc": rc, "log": root / "run.log"}
    finally:
        for p in root.rglob("*.pt"):
            p.unlink()


def _expected_commit():
    if not HAVE_GIT:
        return None
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def _check_record(path, payload):
    from tools.grok_bridge import code_tree_sha256

    rec = payload.get("run_record")
    assert isinstance(rec, dict), f"{path.name} has no run_record"
    assert rec["format"] == rr.RECORD_FORMAT and rec["run_id"] == payload["run_id"]
    assert "--seed" in rec["argv"] and rec["args"]["seed"] == 0 and rec["seed"] == 0
    assert rec["args"]["grad_accum_steps"] == ACCUM and rec["args"]["output_dir"] == str(path.parent)
    assert rec["fingerprint"]["code"] == code_tree_sha256(REPO)
    assert rec["fingerprint"]["torch"] == torch.__version__ and rec["fingerprint"]["device_type"] == "cpu"
    commit = _expected_commit()  # None when the tests run from a copy that is not a git checkout
    assert rec["git"]["commit"] == commit
    if commit is not None:
        assert rec["git"]["error"] is None and isinstance(rec["git"]["dirty"], bool)
    else:
        assert rec["git"]["error"] and rec["git"]["dirty"] is None
    res = rec["resolved"]
    assert res["device"] == "cpu" and res["train_eval_mode"] == "none" and res["grad_scaler"] is False
    assert res["tied_embeddings"] is True and res["n_params_unique"] > 0 and res["warmup_tokens"] == 200
    assert res["lexical_frac_phase1_effective"] == 0.0 and rec["args"]["lexical_frac_phase1"] == 0.05  # #187 audit
    assert res["model_config"] == payload["config"] and "halt_threshold" in res["model_config"]["fields"]
    assert rec["optimizer"]["class"] == "AdamW" and rec["optimizer"]["betas"] == [0.9, 0.95]
    assert sum(g["n_params"] for g in rec["optimizer"]["groups"]) == res["n_params_trainable"]
    saved = rec["saved"]
    assert (saved["micro_steps"], saved["tokens_seen"]) == (payload["step"], payload["tokens_seen"])
    assert saved["optimizer_steps"] == payload["step"] // ACCUM  # every update applied (fp32 on CPU: none skipped)
    json.dumps(rec)
    return rec


@needs_tokenizer
def test_every_checkpoint_kind_carries_the_record(run):
    assert run["rc"] == 0, _tail(run["log"])
    out = run["out"]
    files = {p.name for p in out.glob("*.pt")}
    interrupt_name = f"checkpoint_step_{INTERRUPT_AT}_interrupt.pt"
    assert {"checkpoint_step_40.pt", interrupt_name, "best_model.pt", "final_model.pt"} <= files, files
    starts = set()
    for name in sorted(files):
        payload = torch.load(out / name, map_location="cpu", weights_only=True)
        rec = _check_record(out / name, payload)
        assert rec["parent"] is None and rec["ancestors"] == []
        starts.add((rec["started"], rec["git"]["commit"], json.dumps(rec["fingerprint"], sort_keys=True)))
    assert len(starts) == 1  # one record per process, taken at the start
    best = torch.load(out / "best_model.pt", map_location="cpu", weights_only=True)
    assert best["step"] == 40 and best["run_record"]["saved"]["micro_steps"] == 40
    interrupt = torch.load(out / interrupt_name, map_location="cpu", weights_only=True)
    assert "resume" not in interrupt and interrupt["run_record"]["saved"]["micro_steps"] == INTERRUPT_AT
    # the best model embedded in a rotating checkpoint (its bytes) carries the record as well
    ckpt = torch.load(out / "checkpoint_step_40.pt", map_location="cpu", weights_only=True)
    embedded = ckpt["resume"]["best"]
    assert isinstance(embedded, dict) and "bytes" in embedded, embedded
    inner = torch.load(io.BytesIO(embedded["bytes"].numpy().tobytes()), map_location="cpu", weights_only=True)
    _check_record(out / "best_model.pt", inner)


@needs_tokenizer
def test_a_checkpoint_without_the_record_still_loads_and_resumes(run):
    from utils import load_checkpoint

    assert run["rc"] == 0, _tail(run["log"])
    legacy_dir, out = run["root"] / "legacy", run["root"] / "resumed"
    legacy_dir.mkdir()
    payload = torch.load(run["out"] / "final_model.pt", map_location="cpu", weights_only=True)
    del payload["run_record"]
    legacy = legacy_dir / "final_model.pt"
    torch.save(payload, legacy)  # no sidecar either, as before #138
    assert "run_record" not in load_checkpoint(legacy, map_location="cpu")
    # an inexact resume, interrupted after one micro-step: its final model names the legacy file as its parent
    rc = _train(run["root"] / "resumed.log", *RUN_ARGS, "--output_dir", str(out), "--resume_from", str(legacy),
                "--resume_inexact", RELSM_TEST_KILL_STEP=f"{INTERRUPT_AT + 1}:interrupt",
                RELSM_TEST_KILL_MARKER=run["root"] / "resumed.killed")
    assert rc == 0, _tail(run["root"] / "resumed.log")
    final = torch.load(out / "final_model.pt", map_location="cpu", weights_only=True)
    rec = _check_record(out / "final_model.pt", final)
    assert rec["saved"]["micro_steps"] == INTERRUPT_AT + 1 and "--resume_from" in rec["argv"]
    p = rec["parent"]
    assert (p["file"], p["name"], p["kind"], p["step"], p["record"]) == (str(legacy), "final_model.pt", "inexact",
                                                                          payload["step"], None)
    assert len(p["sha256"]) == 64 and rec["ancestors"] == []


def _batches(path):
    return [line.split() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@needs_tokenizer
def test_relaunching_from_the_record_reproduces_the_first_batches(run):
    import exact_resume as er

    assert run["rc"] == 0, _tail(run["log"])
    rec = torch.load(run["out"] / "final_model.pt", map_location="cpu", weights_only=True)["run_record"]
    argv = list(rec["argv"][1:])  # argv[0] is train.py
    out = run["root"] / "relaunch"
    argv[argv.index("--output_dir") + 1] = str(out)
    first = 8
    rc = _train(run["root"] / "relaunch.log", *argv, RELSM_TEST_KILL_STEP=f"{first}:interrupt",
                RELSM_TEST_KILL_MARKER=run["root"] / "relaunch.killed", RELSM_TEST_BEST_ACC="0.001",
                RELSM_TEST_BATCH_LOG=run["root"] / "relaunch.batches")
    assert rc == 0, _tail(run["root"] / "relaunch.log")
    a, b = _batches(run["root"] / "run.batches"), _batches(run["root"] / "relaunch.batches")
    assert len(b) == first and b == a[:first], (a[:first], b)
    again = torch.load(out / "final_model.pt", map_location="cpu", weights_only=True)["run_record"]
    assert er.arg_differences(rec["args"], again["args"]) == []  # output_dir is free
    assert again["run_id"] != rec["run_id"] and again["parent"] is None


@needs_tokenizer
def test_resuming_a_best_model_with_a_changed_flag_names_it(run):
    assert run["rc"] == 0, _tail(run["log"])
    best = run["out"] / "best_model.pt"
    step = torch.load(best, map_location="cpu", weights_only=True)["step"]
    argv = [*RUN_ARGS, "--resume_from", str(best)]
    argv[argv.index("--grad_accum_steps") + 1] = "2"
    # without --resume_inexact: refused, the changed flag among the reasons, nothing written
    out = run["root"] / "changed_refused"
    rc = _train(run["root"] / "changed_refused.log", *argv, "--output_dir", str(out))
    text = Path(run["root"] / "changed_refused.log").read_text(encoding="utf-8", errors="replace")
    assert rc != 0 and "argument --grad_accum_steps: 4 -> 2" in text, _tail(run["root"] / "changed_refused.log")
    assert not out.exists() or not list(out.glob("*.pt"))
    # with it: the warning names the flag, and the recorded resume lists it
    out = run["root"] / "changed"
    rc = _train(run["root"] / "changed.log", *argv, "--resume_inexact", "--output_dir", str(out),
                RELSM_TEST_KILL_STEP=f"{step + 1}:interrupt", RELSM_TEST_KILL_MARKER=run["root"] / "changed.killed")
    text = Path(run["root"] / "changed.log").read_text(encoding="utf-8", errors="replace")
    assert rc == 0, _tail(run["root"] / "changed.log")
    assert "[resume] WARNING: arguments differ from the checkpoint's: --grad_accum_steps: 4 -> 2" in text
    hyp = json.loads((out / "metrics.json").read_text(encoding="utf-8").splitlines()[0])["data"]
    entry = hyp["resumes"][-1]
    assert entry["kind"] == "inexact" and "argument --grad_accum_steps: 4 -> 2" in entry["why_inexact"], entry
    rec = torch.load(out / "final_model.pt", map_location="cpu", weights_only=True)["run_record"]
    assert rec["args"]["grad_accum_steps"] == 2 and rec["parent"]["name"] == "best_model.pt"
    assert rec["parent"]["record"]["args"]["grad_accum_steps"] == 4
