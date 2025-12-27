import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from curriculum import DagUnlockState


def _make_state():
    dag_roots = ["copy", "reverse", "compare"]
    dag_prereqs = {
        "successor": ["copy", "reverse", "compare"],
        "parity": ["copy", "reverse"],
        "chain": ["successor"],
        "addition": ["chain", "parity"],
        "mod_add": ["addition"],
        "multiplication": ["addition", "reverse"],
        "dyck": ["multiplication"],
    }
    dag_thresholds = {
        "successor": {"copy": 0.99, "reverse": 0.98, "compare": 0.98},
        "parity": {"copy": 0.99, "reverse": 0.98},
        "chain": {"successor": 0.99},
        "addition": {"chain": 0.95, "parity": 0.95},
        "mod_add": {"addition": 0.97},
        "multiplication": {"addition": 0.97, "reverse": 0.98},
        "dyck": {"multiplication": 0.97},
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
    assert state.get_gate("compare") == pytest.approx(1.0)
    assert state.get_gate("successor") == pytest.approx(0.0)
    assert state.get_gate("chain") == pytest.approx(0.0)

    ema = {"copy": 1.0, "reverse": 0.99, "compare": 0.99}
    for _ in range(3):
        state.update_from_ema(ema)
        assert state.get_gate("successor") == pytest.approx(0.0)

    state.update_from_ema(ema)
    successor_gate = state.get_gate("successor")
    assert 0.0 < successor_gate < 1.0

    frontier = state.compute_frontier(ema)
    assert "successor" in frontier
    assert "copy" in frontier

    state.update_from_ema(ema)
    assert state.get_gate("successor") == pytest.approx(successor_gate + 1.0 / 3.0)

    state.update_from_ema(ema)
    state.update_from_ema(ema)
    assert state.get_gate("successor") == pytest.approx(1.0)

    ema_chain = {"copy": 1.0, "reverse": 0.99, "compare": 0.99, "successor": 0.99}
    for _ in range(4):
        state.update_from_ema(ema_chain)

    assert state.get_gate("chain") > 0.0
    assert state.get_gate("parity") > 0.0


def test_dag_unlock_addition_and_backslide():
    state = _make_state()
    ema_roots = {"copy": 1.0, "reverse": 0.99, "compare": 0.99}
    for _ in range(6):
        state.update_from_ema(ema_roots)

    ema_chain = {
        "successor": 0.99,
        "copy": 1.0,
        "reverse": 0.99,
        "compare": 0.99,
    }
    for _ in range(4):
        state.update_from_ema(ema_chain)

    ema_add = {
        "chain": 0.96,
        "parity": 0.96,
        "successor": 0.99,
        "copy": 1.0,
        "reverse": 0.99,
        "compare": 0.99,
    }
    for _ in range(4):
        state.update_from_ema(ema_add)

    assert state.get_gate("addition") > 0.0
    frontier = state.compute_frontier(ema_add)
    assert "addition" in frontier

    ema_backslide = {
        "chain": 0.95,
        "parity": 0.90,
        "addition": 0.97,
        "successor": 0.99,
        "copy": 1.0,
        "reverse": 0.99,
        "compare": 0.99,
    }
    state.update_from_ema(ema_backslide)
    assert state.paused is True
    assert state.compute_replay_ratio(ema_backslide) == pytest.approx(0.35)


def test_dag_unlock_state_roundtrip():
    state = _make_state()
    ema_roots = {"copy": 1.0, "reverse": 0.99, "compare": 0.99}
    for _ in range(5):
        state.update_from_ema(ema_roots)

    ema_chain = {
        "successor": 0.99,
        "copy": 1.0,
        "reverse": 0.99,
        "compare": 0.99,
    }
    for _ in range(4):
        state.update_from_ema(ema_chain)

    ema_add = {
        "chain": 0.96,
        "parity": 0.96,
        "successor": 0.99,
        "copy": 1.0,
        "reverse": 0.99,
        "compare": 0.99,
    }
    for _ in range(4):
        state.update_from_ema(ema_add)

    payload = state.state_dict()
    restored = _make_state()
    restored.load_state_dict(payload)

    assert restored.eval_index == state.eval_index
    assert restored.paused == state.paused
    assert restored.get_gate_snapshot() == state.get_gate_snapshot()
    assert restored.streak == state.streak
    assert restored.unlocked_eval_index == state.unlocked_eval_index
