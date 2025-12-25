import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from curriculum import DagUnlockState


def _make_state():
    dag_roots = ["successor", "copy", "reverse"]
    dag_prereqs = {
        "chain": ["copy"],
        "parity": ["copy", "reverse"],
        "addition": ["chain", "parity"],
        "multiplication": ["addition", "chain"],
        "mod_add": ["addition", "parity"],
        "compare": ["addition"],
        "dyck": ["copy", "reverse"],
    }
    dag_thresholds = {
        "chain": {"copy": 0.99},
        "parity": {"copy": 0.99, "reverse": 0.98},
        "addition": {"chain": 0.95, "parity": 0.95},
        "multiplication": {"addition": 0.97, "chain": 0.95},
        "mod_add": {"addition": 0.97, "parity": 0.95},
        "compare": {"addition": 0.97},
        "dyck": {"copy": 0.99, "reverse": 0.98},
    }
    return DagUnlockState(
        roots=dag_roots,
        prereqs=dag_prereqs,
        thresholds=dag_thresholds,
        patience_evals=4,
        ramp_evals=3,
        replay_ratio=0.2,
        replay_ratio_backslide=0.35,
        unlock_margin=0.01,
        lock_margin=0.03,
        frontier_recent_evals=4,
        mastery_margin=0.02,
    )


def test_dag_unlock_and_ramp():
    state = _make_state()
    assert state.get_gate("copy") == pytest.approx(1.0)
    assert state.get_gate("reverse") == pytest.approx(1.0)
    assert state.get_gate("successor") == pytest.approx(1.0)
    assert state.get_gate("chain") == pytest.approx(0.0)

    ema = {"copy": 1.0, "reverse": 0.99}
    for _ in range(3):
        state.update_from_ema(ema)
        assert state.get_gate("chain") == pytest.approx(0.0)

    state.update_from_ema(ema)
    chain_gate = state.get_gate("chain")
    assert 0.0 < chain_gate < 1.0

    frontier = state.compute_frontier(ema)
    assert "chain" in frontier
    assert "copy" in frontier

    state.update_from_ema(ema)
    assert state.get_gate("chain") == pytest.approx(chain_gate + 1.0 / 3.0)

    state.update_from_ema(ema)
    assert state.get_gate("chain") == pytest.approx(1.0)
    assert state.get_gate("parity") > 0.0


def test_dag_unlock_addition_and_backslide():
    state = _make_state()
    ema = {"copy": 1.0, "reverse": 0.99}
    for _ in range(6):
        state.update_from_ema(ema)

    ema_add = {"chain": 0.96, "parity": 0.96, "copy": 1.0, "reverse": 0.99}
    for _ in range(4):
        state.update_from_ema(ema_add)

    assert state.get_gate("addition") > 0.0
    frontier = state.compute_frontier(ema_add)
    assert "addition" in frontier

    ema_backslide = {
        "chain": 0.95,
        "parity": 0.90,
        "addition": 0.97,
        "copy": 1.0,
        "reverse": 0.99,
    }
    state.update_from_ema(ema_backslide)
    assert state.paused is True
    assert state.compute_replay_ratio(ema_backslide) == pytest.approx(0.35)
