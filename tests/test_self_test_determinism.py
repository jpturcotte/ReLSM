"""The self-test's determinism check ignores wall-clock fields and nothing else (#177)."""

import copy
import json

import pytest

torch = pytest.importorskip("torch")

import eval.self_test as st
from eval.self_test import (
    VOLATILE_FIELDS,
    assert_deterministic,
    determinism_view,
    weights_sha256,
)


def _tiny_model():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    model.register_buffer("step", torch.tensor(7))  # 0-dim tensor
    model.register_buffer("table", torch.arange(6.0), persistent=False)  # like the RoPE tables
    return model


def _payload(timestamp="2026-11-10T12:00:00", elapsed=0.25):
    return {
        "meta": {
            "timestamp": timestamp,
            "seed": 42,
            "suite": "self_test",
            "decoding": {"max_new_tokens": 8, "do_sample": False},
        },
        "self_test": {
            "status": "ok",
            "samples": [
                {"prompt": "Hello, world!", "completion": "abc", "output_token_ids": [5, 6, 7]},
                {"prompt": "2 + 2 =", "completion": "4", "output_token_ids": [9]},
            ],
            "tokens_generated": 4,
            "elapsed_seconds": elapsed,
        },
    }


def _set(payload, path, value):
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


# --- the comparison -------------------------------------------------------------------------------


def test_volatile_fields_are_only_the_two_clock_fields():
    assert set(VOLATILE_FIELDS) == {("meta", "timestamp"), ("self_test", "elapsed_seconds")}


def test_equal_apart_from_timestamps_passes():
    h = weights_sha256(_tiny_model())
    first = determinism_view(_payload("2026-11-10T12:00:00", 0.25), h)
    second = determinism_view(_payload("2026-11-10T12:00:09", 3.75), h)
    assert_deterministic(first, second)


def test_changed_weight_fails():
    model = _tiny_model()
    first = determinism_view(_payload(), weights_sha256(model))
    with torch.no_grad():
        model.weight[0, 0] += 1e-6
    second = determinism_view(_payload(), weights_sha256(model))
    with pytest.raises(AssertionError, match="weights_sha256"):
        assert_deterministic(first, second)


@pytest.mark.parametrize(
    "path, value",
    [
        (("self_test", "samples", 1, "completion"), "5"),
        # same normalized completion, different generated tokens
        (("self_test", "samples", 0, "output_token_ids"), [5, 6, 8]),
        (("self_test", "tokens_generated"), 5),
        (("self_test", "status"), "error"),
        (("meta", "seed"), 43),
        (("meta", "decoding", "do_sample"), True),
    ],
)
def test_changed_output_or_setting_fails(path, value):
    h = weights_sha256(_tiny_model())
    changed = _payload()
    _set(changed, path, value)
    where = ".".join(str(k) for k in path)
    with pytest.raises(AssertionError, match=where.replace(".", r"\.")):
        assert_deterministic(determinism_view(_payload(), h), determinism_view(changed, h))


def test_extra_field_fails():
    h = weights_sha256(_tiny_model())
    changed = _payload()
    changed["self_test"]["extra"] = 1
    with pytest.raises(AssertionError, match=r"self_test\.extra"):
        assert_deterministic(determinism_view(_payload(), h), determinism_view(changed, h))


def test_view_does_not_mutate_payload():
    payload = _payload()
    before = copy.deepcopy(payload)
    determinism_view(payload, "x")
    assert payload == before


# --- the weights hash ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["step", "table"])
def test_changed_buffer_changes_hash(name):
    model = _tiny_model()
    before = weights_sha256(model)
    getattr(model, name).add_(1)
    assert weights_sha256(model) != before


def test_hash_sees_dtype_and_shape():
    a = torch.nn.Linear(2, 2)
    assert weights_sha256(a) != weights_sha256(copy.deepcopy(a).to(torch.float64))
    c, d = torch.nn.Module(), torch.nn.Module()
    c.register_buffer("w", torch.zeros(4))
    d.register_buffer("w", torch.zeros(2, 2))
    assert weights_sha256(c) != weights_sha256(d)


def test_hash_handles_bf16_bool_and_noncontiguous():
    base = torch.arange(6.0).reshape(2, 3)
    a, b = torch.nn.Module(), torch.nn.Module()
    a.register_buffer("w", base.t())  # non-contiguous view
    b.register_buffer("w", base.t().contiguous())
    assert not a.w.is_contiguous()
    assert weights_sha256(a) == weights_sha256(b)  # same values, same hash
    m = torch.nn.Module()
    m.register_buffer("h", torch.ones(3, dtype=torch.bfloat16))
    m.register_buffer("mask", torch.tensor([True, False]))
    before = weights_sha256(m)
    m.mask[1] = True
    after_mask = weights_sha256(m)
    assert after_mask != before
    m.h[0] = 2.0
    assert weights_sha256(m) != after_mask


# --- the self-test end to end, with a small tokenizer ----------------------------------------------


class CharTokenizer:
    """Characters as token ids 2..127; 0 is pad, 1 is eos."""

    vocab_size = 128
    pad_token_id = 0
    eos_token_id = 1
    name_or_path = "char-test"

    def __call__(self, text, **_):
        return {"input_ids": torch.tensor([[ord(c) % 126 + 2 for c in text]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        return "".join(chr(t - 2) for t in ids if t >= 2)


def test_self_checks_pass_end_to_end(capsys):
    st.run_self_checks(tokenizer=CharTokenizer())
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {"status": "ok"}


def test_self_checks_fail_on_weight_change_bf16_cannot_see(monkeypatch):
    """The second load gets one fp32 weight moved by one ulp: invisible after the bf16 cast, so the
    outputs can stay equal, but the hash is taken as loaded and must catch it."""

    real_load = st.load_checkpoint
    loads = []

    def load(path, *a, **k):
        ckpt = real_load(path, *a, **k)
        loads.append(path)
        if len(loads) == 2:
            t = next(v for v in ckpt["model_state_dict"].values() if v.is_floating_point())
            flat = t.view(-1)
            for i in range(flat.numel()):
                moved = torch.nextafter(flat[i], torch.tensor(float("inf")))
                if moved.to(torch.bfloat16) == flat[i].to(torch.bfloat16):
                    flat[i] = moved
                    break
            else:
                pytest.fail("no weight where one ulp is invisible in bf16")
        return ckpt

    monkeypatch.setattr(st, "load_checkpoint", load)
    with pytest.raises(AssertionError) as exc:
        st.run_self_checks(tokenizer=CharTokenizer())
    assert len(loads) == 2
    # the outputs were equal (bf16 saw the same weights); only the loaded-weights hash differs
    assert str(exc.value).endswith("differing fields: weights_sha256")
