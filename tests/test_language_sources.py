"""The language sources: FineWiki Simple English replaces the dead `wikipedia` script config (#159 point 4)."""
import pytest

try:  # noqa: SIM105 - require torch for these tests (data.py imports it)
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - torch not installed in lean envs
    pytest.skip("Skipping language source tests because torch is not installed", allow_module_level=True)

import data
from data import LANGUAGE_CONFIGS, load_language_dataset

FINEWIKI_REV = "8bd13e72e6a002407649b3e898535f42ceb1aeb9"


class FakeStream:
    """Mimics a streaming HF dataset made of `files` files: shard() gives each worker a contiguous block of files
    (as `split_shard_indices_by_worker` does) and raises IndexError for a worker left with none, as datasets 4.4.1
    does (`_merge_gen_kwargs([])`)."""

    def __init__(self, items, files, log):
        self.items, self.num_shards, self.log = list(items), files, log

    def shard(self, num_shards, index, contiguous=True):
        self.log.append(("shard", num_shards, index, contiguous))
        div, mod = divmod(self.num_shards, num_shards)
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        if end == start:
            raise IndexError("list index out of range")
        per_file = len(self.items) // self.num_shards
        return FakeStream(self.items[start * per_file:end * per_file], end - start, self.log)

    def shuffle(self, seed, buffer_size):
        return self

    def take(self, n):
        self.log.append(("take", n))
        return FakeStream(self.items[:n], self.num_shards, self.log)

    def __iter__(self):
        return iter(self.items)


@pytest.fixture
def fake_hub(monkeypatch):
    """Patches datasets.load_dataset. Returns (calls, files): set files[name] for a source's file count."""
    datasets = pytest.importorskip("datasets")
    calls, files = [], {}

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        name = args[0]
        rows = [{"text": f"{name} article {i} " + "w" * 60} for i in range(8)]
        return FakeStream(rows, files.get(name, 4), calls)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return calls, files


def _loads(calls):
    return [c for c in calls if isinstance(c[0], tuple)]


def test_source_list_weights_and_caps():
    assert list(LANGUAGE_CONFIGS) == ["tinystories", "wikipedia", "minipile"]
    got = {k: (c.name, c.subset, c.split, c.text_field, c.weight, c.max_samples, c.revision)
           for k, c in LANGUAGE_CONFIGS.items()}
    assert got == {
        "tinystories": ("roneneldan/TinyStories", None, "train", "text", 0.5, 500_000, None),
        "wikipedia": ("HuggingFaceFW/finewiki", "simple", "train", "text", 0.3, 200_000, FINEWIKI_REV),
        "minipile": ("JeanKaddour/minipile", None, "train", "text", 0.2, 300_000, None),
    }


def test_load_calls_pin_finewiki_and_pass_no_remote_code(fake_hub):
    calls, _ = fake_hub
    for config in LANGUAGE_CONFIGS.values():
        examples = list(load_language_dataset(config, None, 64, log=False))
        assert len(examples) == 8
        assert all(ex["source"] == config.name for ex in examples)
    assert _loads(calls) == [
        (("roneneldan/TinyStories",), {"split": "train", "streaming": True}),
        (("HuggingFaceFW/finewiki", "simple"), {"split": "train", "streaming": True, "revision": FINEWIKI_REV}),
        (("JeanKaddour/minipile",), {"split": "train", "streaming": True}),
    ]


def test_unpinned_config_gets_no_revision_argument(fake_hub):
    calls, _ = fake_hub
    config = data.LanguageDataConfig(name="org/x", subset="cfg", split="train", text_field="text", weight=1.0,
                                     max_samples=None)
    list(load_language_dataset(config, None, 64, log=False))
    assert _loads(calls) == [(("org/x", "cfg"), {"split": "train", "streaming": True})]


def test_single_file_source_with_more_workers_falls_back_instead_of_crashing(fake_hub):
    calls, files = fake_hub
    cfg = LANGUAGE_CONFIGS["wikipedia"]
    files[cfg.name] = 1  # FineWiki simple is one parquet file
    for worker in (1, 2, 3):
        got = list(load_language_dataset(cfg, None, 64, log=False, num_shards=4, shard_index=worker,
                                         max_samples_override=50_000))
        assert got == []
    worker0 = list(load_language_dataset(cfg, None, 64, log=False, num_shards=4, shard_index=0,
                                         max_samples_override=50_000))
    assert len(worker0) == 8  # worker 0 streams the whole source ...
    assert ("take", cfg.max_samples) in calls  # ... with the full cap, not the per-worker share
    assert not any(c[0] == "shard" for c in calls)


def test_multi_file_source_is_still_sharded_per_worker(fake_hub):
    calls, files = fake_hub
    cfg = LANGUAGE_CONFIGS["tinystories"]
    files[cfg.name] = 4
    got = list(load_language_dataset(cfg, None, 64, log=False, num_shards=4, shard_index=2,
                                     max_samples_override=125_000))
    assert len(got) == 2
    assert ("shard", 4, 2, True) in calls
    assert ("take", 125_000) in calls
