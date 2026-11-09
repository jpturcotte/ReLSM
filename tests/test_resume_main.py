"""Exact resume for train.py's --resume_from (#138, #154, #150): the parts that need no training run.

The end-to-end checks (interrupted + resumed == uninterrupted, in fresh processes) are run by the study harness
described in docs/resume.md; these tests cover each piece on its own.
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import data  # noqa: E402
import exact_resume as er  # noqa: E402
from data import (  # noqa: E402
    AlgorithmicDataset, CurriculumSampler, LanguageDataConfig, LanguageDataset, PermutationBatchSampler,
    make_collate_fn,
)
import model as _model  # noqa: E402
from model import TransformerConfig, config_from_payload, config_to_payload  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _same_model_module(monkeypatch):
    # another test re-imports model; pickling TransformerConfig needs the module this file imported
    monkeypatch.setitem(sys.modules, "model", _model)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(ch) % 200) for ch in text]


class TinySeqDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        t = torch.full((3,), idx, dtype=torch.long)
        return {"input_ids": t.clone(), "labels": t.clone()}


def canon(obj):
    """Batches as plain, comparable values (tensors with dtype and shape)."""
    if torch.is_tensor(obj):
        return ("tensor", str(obj.dtype), tuple(obj.shape), obj.tolist())
    if isinstance(obj, dict):
        return {k: canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canon(v) for v in obj]
    return obj


def roundtrip(obj):
    """obj through torch.save and a weights_only load, as a checkpoint carries it."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


# ---------------------------------------------------------------------------------------------------------------
# #138: checkpoints load with weights_only=True
# ---------------------------------------------------------------------------------------------------------------

def test_config_payload_roundtrip_and_rejections():
    cfg = TransformerConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=16)
    payload = config_to_payload(cfg)
    assert payload["schema"] == 1
    assert config_from_payload(roundtrip(payload)) == cfg
    assert config_from_payload(cfg) is cfg  # an old pickle's object, as is
    with pytest.raises(ValueError, match="schema"):
        config_from_payload({"schema": 2, "fields": payload["fields"]})
    with pytest.raises(ValueError, match="does not know"):
        config_from_payload({"schema": 1, "fields": {**payload["fields"], "new_field": 1}})


def test_load_checkpoint_new_and_old_files_with_a_scoped_allowlist(tmp_path):
    from utils import load_checkpoint

    cfg = TransformerConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=16)
    new, old = tmp_path / "new.pt", tmp_path / "old.pt"
    torch.save({"config": config_to_payload(cfg), "step": 3}, new)
    torch.save({"config": cfg, "step": 4}, old)  # written before the payload format: a pickled object
    torch.load(new, weights_only=True)  # no allowlist needed at all
    assert load_checkpoint(new)["config"] == cfg
    assert load_checkpoint(old)["config"] == cfg
    # the allowlist is scoped to the load: in a fresh process, the process-wide list stays without TransformerConfig
    code = ("import torch; from utils import load_checkpoint; from model import TransformerConfig; "
            f"load_checkpoint({str(old)!r}); print(TransformerConfig in torch.serialization.get_safe_globals())")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


# ---------------------------------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------------------------------

def test_canonical_args_is_idempotent():
    args = SimpleNamespace(mix_band_tokens=None, alg_tokens=1000, disable_dynamic_task_weighting=True,
                           task_weighting="dynamic")
    er.canonical_args(args)
    once = dict(vars(args))
    er.canonical_args(args)
    assert vars(args) == once
    assert args.mix_band_tokens == 500 and args.task_weighting == "uniform"
    assert args.resume_inexact is False and args.cuda_deterministic is False


def test_arg_differences_ignore_free_args():
    saved = {"seed": 1, "output_dir": "a", "resume_from": None, "lr": 0.1}
    now = {"seed": 1, "output_dir": "b", "resume_from": "x.pt", "resume_inexact": True, "lr": 0.2}
    assert er.arg_differences(saved, now) == ["--lr: 0.1 -> 0.2"]


# ---------------------------------------------------------------------------------------------------------------
# #154: data positions
# ---------------------------------------------------------------------------------------------------------------

def _stock(n, b):
    return torch.utils.data.DataLoader(TinySeqDataset(n), batch_size=b, shuffle=True)


def _perm(n, b):
    return torch.utils.data.DataLoader(TinySeqDataset(n), batch_sampler=PermutationBatchSampler(n, b))


def _trace(loader, epochs):
    """Every batch over several epochs, with the global torch RNG state after iter() and after each next()."""
    out = []
    for _ in range(epochs):
        it = iter(loader)
        out.append(("iter", torch.get_rng_state().tolist()))
        for batch in it:
            out.append((canon(batch), torch.get_rng_state().tolist()))
    return out


def test_permutation_sampler_draws_exactly_as_shuffle_true():
    torch.manual_seed(123)
    ref = _trace(_stock(10, 3), epochs=3)
    torch.manual_seed(123)
    new = _trace(_perm(10, 3), epochs=3)
    assert new == ref
    assert len(ref) == 3 * (1 + 4)  # three epochs of four batches (the last one partial)


def test_permutation_sampler_position_roundtrip_mid_epoch():
    torch.manual_seed(7)
    loader = _perm(10, 3)
    it = iter(loader)
    head = [canon(next(it)) for _ in range(2)]
    state = roundtrip(loader.batch_sampler.last_iter.state_dict())
    rng = torch.get_rng_state()
    rest = [canon(b) for b in it]

    other = _perm(10, 3)
    other.batch_sampler._pending_state = state
    it2 = iter(other)
    assert other.batch_sampler._pending_state is None
    torch.set_rng_state(rng)
    assert [canon(b) for b in it2] == rest
    assert len(head) + len(rest) == 4


def test_permutation_sampler_refuses_a_bad_state():
    sampler = PermutationBatchSampler(5, 2)
    good = {"kind": "permutation", "n": 5, "batch_size": 2, "perm": torch.tensor([0, 1, 2, 3, 4]), "pos": 2}
    sampler._pending_state = {**good, "perm": torch.tensor([0, 0, 2, 3, 4])}
    with pytest.raises(ValueError, match="not a permutation"):
        iter(sampler)
    sampler._pending_state = {**good, "batch_size": 3}
    with pytest.raises(ValueError, match="batch size"):
        iter(sampler)
    sampler._pending_state = {**good, "perm": None}
    with pytest.raises(ValueError, match="without a permutation"):
        iter(sampler)


def _alg_dataset(n=40):
    return AlgorithmicDataset(tokenizer=CharTokenizer(), num_examples=n, max_seq_len=64, base_batch_size=4,
                              tasks=["addition", "copy"], seed=0, include_lengths=True)


def test_algorithmic_iterator_position_roundtrip():
    ref = [canon(b) for b in _alg_dataset()]
    assert len(ref) > 4

    ds = _alg_dataset()
    it = iter(ds)
    for _ in range(3):
        next(it)
    state, counters = roundtrip(it.state_dict()), roundtrip(ds.exact_dataset_state())

    ds2 = _alg_dataset()
    ds2._pending_state = state
    it2 = iter(ds2)
    ds2.load_exact_dataset_state(counters)
    assert [canon(b) for b in it2] == ref[3:]


def test_algorithmic_iterator_roundtrip_before_the_first_draw():
    ref = [canon(b) for b in _alg_dataset()]
    ds = _alg_dataset()
    state = roundtrip(iter(ds).state_dict())
    ds2 = _alg_dataset()
    ds2._pending_state = state
    assert [canon(b) for b in iter(ds2)] == ref


def _lang_stub(docs_by_source):
    def stub(config, *args, **kwargs):
        docs = docs_by_source.get(config.name)
        if docs is None:
            return data._EmptyLanguageSource(config.name, "failed to load: OSError: gone", failed=True)
        return iter([{"text": t, "source": config.name} for t in docs])
    return stub


def _lang_dataset():
    configs = {
        "a": LanguageDataConfig(name="src/a", subset=None, split="train", text_field="text", weight=0.6,
                                max_samples=None),
        "b": LanguageDataConfig(name="src/b", subset=None, split="train", text_field="text", weight=0.4,
                                max_samples=None),
    }
    return LanguageDataset(CharTokenizer(), configs=configs, max_seq_len=32)


def test_language_iterator_fast_forward_checks_routing_and_content(monkeypatch):
    docs = {"src/a": [f"alpha {i}" for i in range(4)], "src/b": None}  # b fails: a recorded pop
    monkeypatch.setattr(data, "load_language_dataset", _lang_stub(docs))
    ds = _lang_dataset()
    it = iter(ds)
    ref = [canon(next(it)) for _ in range(11)]  # past two rebuilds
    events = list(it.events)
    assert any(e[0] == "pop" for e in events) and any(e[0] == "rebuild" for e in events)

    it = iter(_lang_dataset())
    for _ in range(6):
        next(it)
    state = roundtrip(it.state_dict())
    ds2 = _lang_dataset()
    ds2._pending_state = state
    it2 = iter(ds2)
    assert [canon(next(it2)) for _ in range(5)] == ref[6:]

    # the source changed under the same name: refused, not silently different
    monkeypatch.setattr(data, "load_language_dataset", _lang_stub({"src/a": [f"beta {i}" for i in range(4)],
                                                                   "src/b": None}))
    ds3 = _lang_dataset()
    ds3._pending_state = state
    with pytest.raises(ValueError, match="last example differs"):
        iter(ds3)
    # a source that failed before and works now: the routing differs
    monkeypatch.setattr(data, "load_language_dataset", _lang_stub({"src/a": docs["src/a"], "src/b": ["b0"] * 9}))
    ds4 = _lang_dataset()
    ds4._pending_state = state
    with pytest.raises(ValueError):
        iter(ds4)


def _sampler(alias: bool, fixed: bool):
    collate = make_collate_fn(CharTokenizer())
    if fixed:
        alg = torch.utils.data.DataLoader(TinySeqDataset(7), batch_sampler=PermutationBatchSampler(7, 2))
    else:
        alg = torch.utils.data.DataLoader(_alg_dataset(24), batch_size=None, collate_fn=collate)
    lang = alg if alias else torch.utils.data.DataLoader(TinySeqDataset(5),
                                                         batch_sampler=PermutationBatchSampler(5, 2))
    return CurriculumSampler(alg_loader=alg, lang_loader=lang, total_tokens=400, alg_tokens=200,
                             mix_band_tokens=60, persistent_alg_frac=0.2, lexical_frac_phase1=0.0, seed=3)


@pytest.mark.parametrize("alias,fixed", [(True, False), (False, False), (True, True), (False, True)])
def test_curriculum_sampler_exact_state_roundtrip(alias, fixed):
    torch.manual_seed(5)
    s = _sampler(alias, fixed)
    ref = [canon(s.next_batch()) for _ in range(30)]  # across rollovers and the phase change
    ref_tokens = s.tokens_seen

    torch.manual_seed(5)
    s = _sampler(alias, fixed)
    for _ in range(12):
        s.next_batch()
    state, why = s.exact_state()
    assert why == [] and state["alias"] is alias
    state = roundtrip(state)
    counters = roundtrip(s.alg_loader.dataset.exact_dataset_state()) if not fixed else None
    rng = torch.get_rng_state()

    torch.manual_seed(999)  # a fresh process builds with other draws first
    s2 = _sampler(alias, fixed)
    s2.restore_exact_state(state)
    if counters is not None:
        s2.alg_loader.dataset.load_exact_dataset_state(counters)
    torch.set_rng_state(rng)  # the global RNGs last
    assert [canon(s2.next_batch()) for _ in range(18)] == ref[12:]
    assert s2.tokens_seen == ref_tokens


def test_exact_state_names_workers():
    s = _sampler(alias=False, fixed=True)
    s.alg_loader.num_workers = 2  # as if built with workers: the position lives in worker processes
    state, why = s.exact_state()
    assert state is None and any("num_workers 2" in w for w in why)


# ---------------------------------------------------------------------------------------------------------------
# files: publishing with sidecars, the attempt journal
# ---------------------------------------------------------------------------------------------------------------

def test_publish_writes_payload_then_sidecar_and_check_sidecar_catches_changes(tmp_path):
    path = tmp_path / "checkpoint_step_10.pt"
    seen = []
    meta = er.publish(path, 10, kill=seen.append, role="ckpt", obj={"step": 10, "w": torch.ones(3)})
    assert seen == ["ckpt:save", "ckpt:payload"]
    assert er.check_sidecar(path)[0] is None and meta["step"] == 10
    data_ = bytearray(path.read_bytes())
    data_[-1] ^= 1
    path.write_bytes(bytes(data_))
    assert er.check_sidecar(path)[0] == "sha256 differs from its sidecar"
    assert er.check_sidecar(tmp_path / "none.pt")[0] == "no sidecar"


def test_publish_over_a_name_with_a_sidecar_removes_it_first(tmp_path):
    path = tmp_path / "checkpoint_step_10.pt"
    er.publish(path, 10, kill=lambda p: None, role="ckpt", obj={"v": 1})

    class Killed(Exception):
        pass

    def kill(point):
        if point == "ckpt:save":
            raise Killed

    with pytest.raises(Killed):
        er.publish(path, 10, kill=kill, role="ckpt", obj={"v": 2})
    # killed while staged: the old payload is whole and no sidecar claims it is the new one
    assert not er.sidecar(path).exists()
    assert torch.load(path, weights_only=True) == {"v": 1}


def test_journal_frames_a_complete_tail_and_sets_aside_a_torn_one(tmp_path):
    j = tmp_path / er.JOURNAL
    er.append_journal(j, {"n": 1})
    with open(j, "ab") as f:
        f.write(b'{"n": 2}')  # a complete record without its newline
    records, tail = er.read_journal(j)
    assert [r["n"] for r in records] == [1, 2] and tail == "frame"
    er.repair_journal(j, tail)
    er.append_journal(j, {"n": 3})
    assert [r["n"] for r in er.read_journal(j)[0]] == [1, 2, 3]

    with open(j, "ab") as f:
        f.write(b'{"n": 4, "cut')  # torn
    records, tail = er.read_journal(j)
    assert [r["n"] for r in records] == [1, 2, 3] and tail == "torn"
    er.repair_journal(j, tail)
    assert (tmp_path / f"{er.JOURNAL}.torn.1").read_bytes() == b'{"n": 4, "cut'
    er.append_journal(j, {"n": 5})
    assert er.read_journal(j) == ([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 5}], None)


def test_journal_with_a_corrupt_line_is_refused_read_only(tmp_path):
    j = tmp_path / er.JOURNAL
    j.write_bytes(b'{"n": 1}\nnot json\n{"n": 3}\n')
    before = j.read_bytes()
    with pytest.raises(er.ResumeRefused, match="line 2"):
        er.read_journal(j)
    assert j.read_bytes() == before


def test_checkpoint_step_of():
    assert er.checkpoint_step_of("checkpoint_step_30.pt") == 30
    assert er.checkpoint_step_of("checkpoint_step_14_interrupt.pt") == 14
    assert er.checkpoint_step_of("checkpoint_step_30.pt.sha256") is None
    assert er.checkpoint_step_of("best_model.pt") is None


# ---------------------------------------------------------------------------------------------------------------
# #150: legacy history
# ---------------------------------------------------------------------------------------------------------------

def _metrics(tmp_path, hyper, records):
    path = tmp_path / "metrics.json"
    lines = [json.dumps({"type": "hyperparameters", "data": hyper})]
    lines += [json.dumps({"type": "record", **r}) for r in records]
    lines.append(json.dumps({"type": "task_accuracy", "target": "eval", "task": "copy", "step": 10,
                             "eval/acc/exact_match": 0.5}))
    lines.append(json.dumps({"type": "task_accuracy", "target": "eval", "task": "copy", "step": 30,
                             "eval/acc/exact_match": 0.6}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return er.read_metrics(path)


T = er.TOKEN_KEY


def test_legacy_history_keeps_rows_up_to_the_checkpoint(tmp_path):
    m = _metrics(tmp_path, {"alg_tokens": 100, "curriculum_min_task_evals": 1},
                 [{"step": 10, "phase": "algorithmic", T: 50}, {"step": 20, "phase": "algorithmic", T: 100},
                  {"step": 20, "phase": "eval", "eval/acc/exact_match": 0.5},  # no token field: by step alone
                  {"step": 30, "phase": "algorithmic", T: 150}])
    ckpt = {"step": 20, "curriculum_tokens_seen": 100, "curriculum_config": {"alg_tokens": 100},
            "task_curriculum_config": {"min_task_evals": 1.0, "ema_decay": 0.9}}
    hist, why = er.legacy_history(m, ckpt)
    assert why == "compatible"
    assert [(r["step"], r["phase"]) for r in hist["records"]] == [(10, "algorithmic"), (20, "algorithmic"),
                                                                  (20, "eval")]
    assert [e["step"] for e in hist["task_accuracies"]["copy"]] == [10]


def test_legacy_history_refuses_another_runs_rows(tmp_path):
    m = _metrics(tmp_path, {"alg_tokens": 999}, [{"step": 10, "phase": "algorithmic", T: 50}])
    hist, why = er.legacy_history(m, {"step": 20, "tokens_seen": 100, "curriculum_config": {"alg_tokens": 100}})
    assert hist is None and "alg_tokens" in why
    m = _metrics(tmp_path, {"alg_tokens": 100}, [{"step": 10, "phase": "algorithmic", T: 500}])
    hist, why = er.legacy_history(m, {"step": 20, "tokens_seen": 100, "curriculum_config": {"alg_tokens": 100}})
    assert hist is None and "more tokens" in why
    m = _metrics(tmp_path, {"alg_tokens": 100}, [{"step": 30, "phase": "algorithmic", T: 90}])
    hist, why = er.legacy_history(m, {"step": 20, "tokens_seen": 100, "curriculum_config": {"alg_tokens": 100}})
    assert hist is None and "do not follow" in why


# ---------------------------------------------------------------------------------------------------------------
# what must match for an exact resume
# ---------------------------------------------------------------------------------------------------------------

def _cuda_fp():
    return {"device_type": "cuda", "torch": "2.9.1", "pythonhashseed": "0", "cuda_initialized_before": False,
            "policy": {**er.CUDA_EXACT_POLICY, "preferred_blas_library": "Cublas",
                       "env_torch_blas_prefer_cublaslt": None, "num_threads": 2},
            "cuda": {"name": "GPU", "uuid": "GPU-1", "capability": [8, 9]}}


def test_cuda_eligibility_checks_every_policy_field():
    assert er.cuda_exact_problems(_cuda_fp(), cuda_deterministic=True) == []
    assert er.cuda_exact_problems(_cuda_fp(), cuda_deterministic=False) == ["CUDA run without --cuda_deterministic"]
    for key, value in er.CUDA_EXACT_POLICY.items():
        fp = _cuda_fp()
        fp["policy"][key] = "changed" if isinstance(value, str) else (not value)
        assert any(key in w for w in er.cuda_exact_problems(fp, cuda_deterministic=True)), key
    assert er.cuda_exact_problems({"device_type": "cpu"}, cuda_deterministic=False) == []


def test_fingerprint_differences_compare_nested_fields_and_skip_recorded_ones():
    a, b = _cuda_fp(), _cuda_fp()
    b["pythonhashseed"] = "2"  # recorded only since #160
    b["cuda_initialized_before"] = True
    assert er.fingerprint_differences(a, b) == []
    b["cuda"]["uuid"] = "GPU-2"
    b["policy"]["env_torch_blas_prefer_cublaslt"] = "1"
    assert er.fingerprint_differences(a, b) == ["cuda.uuid: 'GPU-1' -> 'GPU-2'",
                                                "policy.env_torch_blas_prefer_cublaslt: None -> '1'"]


def test_numerical_policy_is_plain_json():
    json.dumps(er.numerical_policy())


def test_tokenizer_determinism_gate():
    assert er.tokenizer_is_deterministic(CharTokenizer()) is None

    class Backend:
        def __init__(self, dropout):
            self.dropout = dropout

        def to_str(self):
            return json.dumps({"model": {"type": "BPE", "dropout": self.dropout}})

    assert er.tokenizer_is_deterministic(SimpleNamespace(backend_tokenizer=Backend(None))) is None
    assert "dropout" in er.tokenizer_is_deterministic(SimpleNamespace(backend_tokenizer=Backend(0.1)))
    assert "samples" in er.tokenizer_is_deterministic(SimpleNamespace(sp_model_kwargs={"enable_sampling": True}))


def test_dag_state_serializes_the_same_under_any_hash_seed():
    code = (
        "import json; from curriculum import DagUnlockState\n"
        "d = DagUnlockState(roots=['copy','reverse','compare'], prereqs={'successor':['copy','reverse','compare'],"
        "'chain':['successor']}, thresholds={'successor':{'copy':0.9,'reverse':0.9,'compare':0.9},"
        "'chain':{'successor':0.9}})\n"
        "d.update_from_ema({'copy':1.0,'reverse':1.0,'compare':1.0,'successor':0.1,'chain':0.0})\n"
        "print(json.dumps(d.state_dict()))\n"
    )
    outs = []
    for seed in ("0", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                                   text=True, check=True).stdout)
    assert outs[0] == outs[1]


# ---------------------------------------------------------------------------------------------------------------
# model state outside state_dict()
# ---------------------------------------------------------------------------------------------------------------

def test_non_persistent_buffers_roundtrip_after_rope_growth():
    from model import BaselineTransformer

    cfg = TransformerConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=16)
    torch.manual_seed(0)
    m = BaselineTransformer(cfg)
    m.eval()
    x = torch.randint(0, 50, (1, 24))
    with torch.no_grad():
        m(x)  # 24 > 16: the RoPE cos/sin caches grow
    saved = roundtrip(er.non_persistent_buffers(m))
    assert any(t.shape[0] == 24 for t in saved.values())

    torch.manual_seed(1)
    m2 = BaselineTransformer(cfg)
    m2.load_state_dict(m.state_dict())
    m2.eval()
    er.set_non_persistent_buffers(m2, saved)
    for name, buf in m2.named_buffers():
        if name in saved:
            assert torch.equal(buf, saved[name])
    y = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        assert torch.equal(m(y)[0], m2(y)[0])


# ---------------------------------------------------------------------------------------------------------------
# the best model of a file without resume.best
# ---------------------------------------------------------------------------------------------------------------

def test_recover_legacy_best(tmp_path):
    cfg = TransformerConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=16)
    src = tmp_path / "checkpoint_step_30.pt"
    assert er.recover_legacy_best(src, {"step": 30, "best_alg_acc": 0.0}) == (None, "none")
    assert er.recover_legacy_best(src, {"step": 30, "best_alg_acc": 0.5})[1] == "unknown"  # no file
    torch.save({"config": cfg, "step": 10, "alg_accuracy": 0.5, "w": torch.ones(2)}, tmp_path / "best_model.pt")
    best, how = er.recover_legacy_best(src, {"step": 30, "best_alg_acc": 0.5})
    assert how == "recovered" and best["step"] == 10
    inner = torch.load(io.BytesIO(best["bytes"].numpy().tobytes()), weights_only=True)  # safe format now
    assert config_from_payload(inner["config"]) == cfg
    assert er.recover_legacy_best(src, {"step": 5, "best_alg_acc": 0.5})[1] == "unknown"  # best after it
    assert er.recover_legacy_best(src, {"step": 30, "best_alg_acc": 0.75})[1] == "unknown"  # another best
    best, how = er.recover_legacy_best(src, {"step": 30, "best_alg_acc": 0.5}, run_id="continuation-id")
    inner = torch.load(io.BytesIO(best["bytes"].numpy().tobytes()), weights_only=True)
    assert how == "recovered" and inner["run_id"] == "continuation-id"  # the continuation's id (H3)


# ---------------------------------------------------------------------------------------------------------------
# an exact checkpoint must be complete and consistent before anything is restored (Codex code review 1)
# ---------------------------------------------------------------------------------------------------------------

def _exact_fixture():
    import random as _random

    from tools.grok_bridge import numpy_rng_state

    ckpt = {"model_state_dict": {"w": torch.zeros(1)}, "step": 12, "tokens_seen": 300,
            "optimizer_state_dict": {"state": {}, "param_groups": [{"params": [0]}]}, "scaler_state_dict": {},
            "curriculum_rng_state": _random.getstate(), "curriculum_tokens_seen": 300, "best_alg_acc": 0.0,
            "curriculum_state": {"x": 1}, "dag_state": {"y": 1}}
    r = {"exact": True, "why_inexact": [], "micro_steps": 12, "tokens_seen": 300, "sampler": {"tokens_seen": 300},
         "loop": {"best_alg_acc": 0.0}, "snapshots": {"difficulty": {}},
         "args": {"use_task_curriculum": True, "task_curriculum_strategy": "dag"}, "fingerprint": {"device_type": "cpu"},
         "rng": {"torch": torch.get_rng_state(), "cuda": None, "python": _random.getstate(),
                 "numpy": numpy_rng_state()}}
    return ckpt, r


def test_validate_exact_state_accepts_a_complete_checkpoint():
    ckpt, r = roundtrip(_exact_fixture())
    er.validate_exact_state(ckpt, r)
    # a run without the task curriculum (its own arguments say so) needs no curriculum or DAG state
    ckpt.pop("curriculum_state"), ckpt.pop("dag_state")
    r.update(snapshots=None, args={"use_task_curriculum": False})
    er.validate_exact_state(ckpt, r)


def test_validate_exact_state_accepts_an_inexact_checkpoint_without_a_sampler_position():
    # worker loaders: no sampler position, so the checkpoint is inexact and --resume_inexact can still take it
    ckpt, r = roundtrip(_exact_fixture())
    r.update(exact=False, why_inexact=["alg loader has num_workers 2"], sampler=None)
    er.validate_exact_state(ckpt, r)


@pytest.mark.parametrize("damage, match", [
    (lambda c, r: r.update(exact=False), "exact flag"),
    (lambda c, r: r.update(why_inexact=["x"]), "exact flag"),
    (lambda c, r: c.pop("optimizer_state_dict"), "optimizer_state_dict"),
    (lambda c, r: c.pop("curriculum_state"), "curriculum_state"),
    (lambda c, r: c.pop("dag_state"), "dag_state"),
    (lambda c, r: c.update(scaler_state_dict=None), "scaler_state_dict"),
    (lambda c, r: c["optimizer_state_dict"].update(param_groups=[]), "optimizer state is malformed"),
    (lambda c, r: (c.update(tokens_seen=10), r.update(tokens_seen=100), r["sampler"].update(tokens_seen=999)),
     "token counters disagree"),
    (lambda c, r: c.update(curriculum_tokens_seen=301), "token counters disagree"),
    (lambda c, r: r.update(sampler=None), "no data sampler position"),
    (lambda c, r: r["sampler"].update(tokens_seen=299), "token counters disagree"),
    (lambda c, r: r.update(micro_steps=13), "step disagree"),
    (lambda c, r: r["loop"].update(best_alg_acc=0.5), "best accuracy"),
    (lambda c, r: r["rng"].update(torch=torch.zeros(3, dtype=torch.uint8)), "RNG state is malformed"),
    (lambda c, r: r["rng"].update(python=(3, (1, 2), None)), "RNG state is malformed"),
    (lambda c, r: r["rng"]["numpy"].update(keys=[1, 2]), "RNG state is malformed"),
    (lambda c, r: (r.update(fingerprint={"device_type": "cuda", "cuda": {"uuid": "u"}}),
                   r["rng"].update(cuda={"state": torch.zeros(0, dtype=torch.uint8), "index": 0, "uuid": "u"})),
     "RNG state is malformed"),
    # the CUDA stream must be there for a CUDA run, bound to its device, and absent for a CPU run (Codex recheck 1)
    (lambda c, r: r.update(fingerprint={"device_type": "cuda", "cuda": {"uuid": "u"}}), "holds no CUDA RNG state"),
    (lambda c, r: (r.update(fingerprint={"device_type": "cuda", "cuda": {"uuid": "u"}}),
                   r["rng"].update(cuda={"state": torch.ones(8, dtype=torch.uint8), "index": 0, "uuid": "v"})),
     "another device"),
    (lambda c, r: r["rng"].update(cuda={"state": torch.ones(8, dtype=torch.uint8), "index": 0, "uuid": "u"}),
     "holds a CUDA RNG state"),
    # what must be there follows from the writing run's own arguments, not this run's (Codex recheck 12)
    (lambda c, r: r.update(snapshots=None), "curriculum snapshots and its own arguments"),
    (lambda c, r: r.update(args={"use_task_curriculum": False}), "curriculum snapshots and its own arguments"),
])
def test_validate_exact_state_refuses_incomplete_or_inconsistent_state(damage, match):
    ckpt, r = roundtrip(_exact_fixture())
    damage(ckpt, r)
    with pytest.raises(ValueError, match=match):
        er.validate_exact_state(ckpt, r)


# ---------------------------------------------------------------------------------------------------------------
# a local language source is exact only as a plain directory of data files (Codex code review 2)
# ---------------------------------------------------------------------------------------------------------------

def _lang_cfg(path):
    return {"local": LanguageDataConfig(name=str(path), subset=None, split="train", text_field="text", weight=1.0,
                                        max_samples=None)}


def test_language_manifest_plain_directory_is_exact(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    manifest, why = data.language_manifest(_lang_cfg(tmp_path))
    assert why == [] and [f for f, _ in manifest[0]["files"]] == ["a.jsonl"]


@pytest.mark.parametrize("name", ["README.md", "readme.txt", "loader.py", "sub/config.yaml", "x.yml"])
def test_language_manifest_redirecting_files_make_it_inexact(tmp_path, name):
    (tmp_path / "a.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    f = tmp_path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("---\nconfigs:\n- config_name: default\n  data_files: ../outside/train.jsonl\n---\n",
                 encoding="utf-8")
    _, why = data.language_manifest(_lang_cfg(tmp_path))
    assert len(why) == 1 and "outside the hashed directory" in why[0] and name in why[0]


def test_language_manifest_links_out_make_it_inexact(tmp_path):
    root, outside = tmp_path / "corpus", tmp_path / "outside.jsonl"
    root.mkdir()
    outside.write_text('{"text": "y"}\n', encoding="utf-8")
    try:
        os.symlink(outside, root / "b.jsonl")
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create symlinks")
    _, why = data.language_manifest(_lang_cfg(root))
    assert any("links that resolve outside" in w for w in why)


# ---------------------------------------------------------------------------------------------------------------
# the scoped allowlist never removes the caller's own entry (Codex code review 5)
# ---------------------------------------------------------------------------------------------------------------

def test_load_checkpoint_keeps_a_callers_allowlist_entry(tmp_path):
    cfg = TransformerConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=16)
    old = tmp_path / "old.pt"
    torch.save({"config": cfg, "step": 4}, old)
    code = ("import torch; from utils import load_checkpoint; from model import TransformerConfig; "
            "torch.serialization.add_safe_globals([TransformerConfig]); "
            f"load_checkpoint({str(old)!r}); print(TransformerConfig in torch.serialization.get_safe_globals())")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "True"


# ---------------------------------------------------------------------------------------------------------------
# design H4: the CUDA policy is applied and read back before the first device probe (Codex code review 6)
# ---------------------------------------------------------------------------------------------------------------

def test_prepare_numerics_applies_and_reads_back_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(er, "apply_cuda_policy", lambda: calls.append("apply"))
    monkeypatch.setattr(er, "numerical_policy", lambda: calls.append("readback") or dict(er.CUDA_EXACT_POLICY))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: calls.append("is_initialized") or False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    out = er.prepare_numerics(True)
    assert calls == ["is_initialized", "apply", "readback"] and out["cuda_initialized_before"] is False
    calls.clear()
    assert er.prepare_numerics(False)["policy"] is None and calls == ["is_initialized"]


def test_prepare_numerics_stops_on_a_bad_environment_or_a_setting_torch_did_not_take(monkeypatch):
    monkeypatch.setattr(er, "apply_cuda_policy", lambda: None)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(SystemExit, match="CUBLAS_WORKSPACE_CONFIG"):
        er.prepare_numerics(True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(er, "numerical_policy", lambda: {**er.CUDA_EXACT_POLICY, "matmul_allow_tf32": True})
    with pytest.raises(SystemExit, match="matmul_allow_tf32"):
        er.prepare_numerics(True)


def test_train_prepares_numerics_before_the_first_device_probe():
    src = (REPO / "train.py").read_text(encoding="utf-8")
    body = src[src.index("def train(args):"):]
    assert body.index("er.prepare_numerics(") < body.index("torch.cuda.is_available()")


# ---------------------------------------------------------------------------------------------------------------
# torch before 2.7 has no matmul.allow_fp16_accumulation (PR #297 review): the policy is read and set without it
# ---------------------------------------------------------------------------------------------------------------

class _MatmulBefore27:
    """torch 2.5/2.6's cuBLASModule: three settings; any other name raises AttributeError on get and on set."""
    NAMES = ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")

    def __init__(self):
        object.__setattr__(self, "values", {n: True for n in self.NAMES})

    def __getattr__(self, name):
        if name in self.NAMES:
            return self.values[name]
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name not in self.NAMES:
            raise AttributeError("Unknown attribute " + name)
        self.values[name] = value


def test_policy_value_ok_takes_an_absent_setting_as_off_only():
    assert er.policy_value_ok(False, False) and er.policy_value_ok(None, False)
    assert not er.policy_value_ok(None, True) and not er.policy_value_ok(None, ":4096:8")
    assert not er.policy_value_ok(True, False)


def test_numerical_policy_without_fp16_accumulation_reads_none(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda, "matmul", _MatmulBefore27())
    pol = er.numerical_policy()
    assert pol["fp16_accumulation"] is None and pol["matmul_allow_tf32"] is True
    json.dumps(pol)


def test_numerical_policy_with_fp16_accumulation_reads_it():
    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        pytest.skip("this torch has no allow_fp16_accumulation")
    assert er.numerical_policy()["fp16_accumulation"] is bool(torch.backends.cuda.matmul.allow_fp16_accumulation)


def test_apply_cuda_policy_without_fp16_accumulation_sets_the_rest(monkeypatch):
    mm = _MatmulBefore27()
    monkeypatch.setattr(torch.backends.cuda, "matmul", mm)
    calls = []
    # every global setter is recorded instead of called, so the test process keeps its numerics
    for mod, name in ((torch, "use_deterministic_algorithms"), (torch, "set_float32_matmul_precision"),
                      (torch.backends.cuda, "enable_flash_sdp"), (torch.backends.cuda, "enable_mem_efficient_sdp"),
                      (torch.backends.cuda, "enable_cudnn_sdp"), (torch.backends.cuda, "enable_math_sdp"),
                      (torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp")):
        monkeypatch.setattr(mod, name, lambda *a, _n=name: calls.append((_n, a)))
    for name in ("benchmark", "deterministic", "allow_tf32"):
        monkeypatch.setattr(torch.backends.cudnn, name, getattr(torch.backends.cudnn, name))  # restored after
    er.apply_cuda_policy()
    assert mm.values == {n: False for n in _MatmulBefore27.NAMES}
    assert ("use_deterministic_algorithms", (True,)) in calls and ("enable_math_sdp", (True,)) in calls


def test_cuda_exact_problems_accept_an_absent_fp16_accumulation_but_not_one_on():
    fp = _cuda_fp()
    fp["policy"]["fp16_accumulation"] = None
    assert er.cuda_exact_problems(fp, cuda_deterministic=True) == []
    fp["policy"]["fp16_accumulation"] = True
    assert er.cuda_exact_problems(fp, cuda_deterministic=True) == [
        "CUDA policy fp16_accumulation is True, exact needs False"]


def test_prepare_numerics_accepts_a_torch_without_fp16_accumulation(monkeypatch):
    monkeypatch.setattr(er, "apply_cuda_policy", lambda: None)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(er, "numerical_policy", lambda: {**er.CUDA_EXACT_POLICY, "fp16_accumulation": None})
    assert er.prepare_numerics(True)["policy"]["fp16_accumulation"] is None


def test_an_unknown_best_sets_aside_a_best_model_already_there():
    # the restore step in train()'s resume commit: only an embedded best is published; no best yet and an unknown
    # one both move an existing best_model.pt and its sidecar aside (checked end to end by the study harness)
    src = (REPO / "train.py").read_text(encoding="utf-8")
    body = src[src.index('best_path = output_dir / "best_model.pt"'):src.index('kill("rs:best")')]
    assert "elif best is None" not in body
    tail = body[body.index('elif isinstance(best, dict) and "bytes" in best:'):]
    assert "else:" in tail and "aside(er.sidecar(best_path))" in tail and "aside(best_path)" in tail
