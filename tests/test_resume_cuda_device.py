"""exact_resume.validate_rng on a real CUDA device (the CUDA RNG checks of an exact or inexact resume).

torch's CUDA generator takes a 16-byte state (seed + offset) or an 8-byte seed-only one; validate_rng loads a saved
state into a temporary generator on the device, the same set_state the restore uses, so it refuses exactly the
states the restore could not load. Skipped without a CUDA device.
"""
import random

import pytest
import torch

import exact_resume as er
from tools.grok_bridge import numpy_rng_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _rng(cuda_state):
    return {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": numpy_rng_state(),
            "cuda": {"state": cuda_state, "index": 0, "uuid": "GPU-test"}}


def test_cuda_is_the_device_under_test():
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1


def test_a_real_cuda_state_loads_and_the_global_stream_is_untouched():
    dev = torch.device("cuda:0")
    torch.rand(3, device=dev)  # move the global stream off its seed
    before = torch.cuda.get_rng_state(dev)
    assert before.numel() == 16  # seed + offset
    er.validate_rng(_rng(before.clone()), dev)
    assert torch.equal(torch.cuda.get_rng_state(dev), before)


def test_an_8_byte_state_is_a_seed_only_state_torch_accepts():
    # why v1's case did not raise: 8 bytes are read as the seed (offset 0), a valid state
    g = torch.Generator(device="cuda:0")
    g.set_state(torch.ones(8, dtype=torch.uint8))
    assert g.initial_seed() == int.from_bytes(b"\x01" * 8, "little")
    er.validate_rng(_rng(torch.ones(8, dtype=torch.uint8)), torch.device("cuda:0"))


@pytest.mark.parametrize("n", [5, 24])
def test_a_state_torch_cannot_load_is_refused_on_the_device(n):
    dev = torch.device("cuda:0")
    before = torch.cuda.get_rng_state(dev)
    with pytest.raises(RuntimeError):
        torch.Generator(device=dev).set_state(torch.ones(n, dtype=torch.uint8))  # torch itself refuses it
    with pytest.raises(ValueError, match="malformed"):
        er.validate_rng(_rng(torch.ones(n, dtype=torch.uint8)), dev)
    assert torch.equal(torch.cuda.get_rng_state(dev), before)


def test_off_the_device_only_the_shape_is_checked():
    # the boundary the docs state: a CPU process cannot load a CUDA state, so it checks the shape only
    er.validate_rng(_rng(torch.ones(5, dtype=torch.uint8)), torch.device("cpu"))
    with pytest.raises(ValueError, match="malformed"):
        er.validate_rng(_rng(torch.zeros(0, dtype=torch.uint8)), torch.device("cpu"))
