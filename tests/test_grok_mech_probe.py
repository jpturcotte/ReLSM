"""CPU tests for tools/grok_mech_probe.py's matched random control (random_drop)."""
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import grok_mech_probe as mp  # noqa: E402


def all_pairs(p):
    a, b = (torch.tensor(v) for v in zip(*itertools.product(range(p), repeat=2)))
    return (a + b) % p, a, b


def test_random_control_never_drops_the_answer_or_an_operand():
    for p in (2, 3, 5, 23):
        y, a, b = all_pairs(p)
        C = p + 2  # the alphabet, then op and '='
        k = mp.operand_drop(y, a, b, C).sum(1)
        drop, ok = mp.random_drop(y, a, b, p, C, k)
        rows = torch.arange(len(y))
        assert not drop[rows, y].any()
        assert not (drop[rows, a] & (a != y)).any() and not (drop[rows, b] & (b != y)).any()
        assert not drop[:, p:].any()  # never the op or '=' classes
        # a row with a matched control drops exactly k[i] classes; one without drops none
        assert (drop.sum(1)[ok] == k[ok]).all() and (drop.sum(1)[~ok] == 0).all()
        if p == 23:
            assert ok.all()  # the studies' alphabets always have a matched control
        if p == 3:
            # (a, b) = (1, 2): answer 0, two rival operands, and no other numeric class left
            i = int(((a == 1) & (b == 2)).nonzero())
            assert not ok[i]
