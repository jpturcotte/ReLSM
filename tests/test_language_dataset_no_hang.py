"""LanguageDataset must stop with a clear error, not rebuild forever, when no source yields (#159 point 3)."""
import threading
from types import SimpleNamespace

import pytest

try:  # noqa: SIM105 - require torch for these tests
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - torch not installed in lean envs
    pytest.skip("Skipping language dataset tests because torch is not installed", allow_module_level=True)

import data
from data import LanguageDataConfig, LanguageDataset

TIMEOUT_S = 30.0
MAX_LOADS = 300  # 100 rebuilds of 3 sources; the fixed code loads each source at most three times per test


class RebuildLoop(BaseException):
    """Raised by a stub once the loader is called far more often than any finite pass needs.

    It is a BaseException, so the `except Exception` in the loader and in __iter__ does not swallow it. The
    thread then ends with the test instead of rebuilding on in the background, which floods pytest's output
    capture (and crashed pytest on Windows).
    """


def _capped(fn, cap=MAX_LOADS):
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > cap:
            raise RebuildLoop(f"loader called {calls['n']} times: LanguageDataset is rebuilding in a loop")
        return fn(*args, **kwargs)

    return wrapped


class CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def _configs():
    return {
        "a": LanguageDataConfig(name="src/a", subset=None, split="train", text_field="text", weight=0.5,
                                max_samples=None),
        "b": LanguageDataConfig(name="src/b", subset="x", split="train", text_field="text", weight=0.3,
                                max_samples=None),
        "c": LanguageDataConfig(name="src/c", subset=None, split="train", text_field="text", weight=0.2,
                                max_samples=None),
    }


def _dataset():
    return LanguageDataset(CharTokenizer(), configs=_configs(), max_seq_len=64)


def _bounded(fn, timeout=TIMEOUT_S):
    """Run fn in a daemon thread; fail (instead of hanging the suite) if it does not finish in time."""
    out = {}

    def target():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed back to the test
            out["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"LanguageDataset did not stop within {timeout} s (rebuild loop)"
    if isinstance(out.get("error"), RebuildLoop):
        pytest.fail(str(out["error"]))
    return out


def _texts(example_batches):
    return ["".join(chr(int(i)) for i in ex["input_ids"]) for ex in example_batches]


def test_all_sources_empty_raises_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(data, "load_language_dataset", _capped(lambda config, *a, **k: iter(())))
    out = _bounded(lambda: next(iter(_dataset())))
    assert isinstance(out.get("error"), RuntimeError), out
    msg = str(out["error"])
    assert "first pass" in msg
    for name in ("src/a", "src/b", "src/c"):
        assert f"{name}: stream ended with no example" in msg


def test_all_sources_fail_to_load_reports_each_reason(monkeypatch, capsys):
    datasets = pytest.importorskip("datasets")

    def failing_load_dataset(name, *args, **kwargs):
        raise ConnectionError(f"offline: {name}")

    monkeypatch.setattr(datasets, "load_dataset", _capped(failing_load_dataset))
    out = _bounded(lambda: next(iter(_dataset())))
    assert isinstance(out.get("error"), RuntimeError), out
    msg = str(out["error"])
    for name in ("src/a", "src/b", "src/c"):
        assert f"{name}: failed to load: ConnectionError: offline: {name}" in msg
    printed = capsys.readouterr().out
    for name in ("src/a", "src/b", "src/c"):
        assert printed.count(f"Warning: language source {name} failed to load") == 1


def test_one_working_source_gives_same_examples(monkeypatch, capsys):
    good = [f"doc{i}" for i in range(5)]

    def stub(config, *args, **kwargs):
        if config.name == "src/b":
            return iter([{"text": t, "source": config.name} for t in good])
        if config.name == "src/c":
            return data._EmptyLanguageSource(config.name, "failed to load: OSError: gone", failed=True)
        return iter(())

    monkeypatch.setattr(data, "load_language_dataset", _capped(stub))
    it = iter(_dataset())
    out = _bounded(lambda: [next(it) for _ in range(12)])
    assert "error" not in out, out
    # The one working source in its own order, again after each rebuild (as before the fix).
    assert _texts(out["value"]) == good + good + good[:2]
    for ex in out["value"]:
        assert torch.equal(ex["input_ids"], ex["labels"])
    printed = capsys.readouterr().out
    assert printed.count("Warning: language source src/c failed to load: OSError: gone") == 1


def test_sources_that_stop_after_a_good_pass_raise_on_the_empty_pass(monkeypatch):
    calls = {"n": 0}

    def stub(config, *args, **kwargs):
        calls["n"] += 1
        if config.name == "src/a" and calls["n"] <= 3:  # first build only
            return iter([{"text": "only once", "source": config.name}])
        return iter(())

    monkeypatch.setattr(data, "load_language_dataset", _capped(stub))
    it = iter(_dataset())
    first = _bounded(lambda: next(it))
    assert _texts([first["value"]]) == ["only once"]
    out = _bounded(lambda: next(it))
    assert isinstance(out.get("error"), RuntimeError), out
    assert "pass after rebuild 1" in str(out["error"])


def test_worker_with_only_skipped_sources_stops_quietly(monkeypatch):
    monkeypatch.setattr(data, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=2))
    monkeypatch.setattr(
        data,
        "load_language_dataset",
        _capped(
            lambda config, *a, **k: data._EmptyLanguageSource(
                config.name, "skipped on worker 1 (no sharding support)", failed=False
            )
        ),
    )
    out = _bounded(lambda: list(_dataset()))
    assert out == {"value": []}
