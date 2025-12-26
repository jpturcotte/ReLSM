"""Task-level curriculum state for algorithmic training."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Optional, Sequence, Set


class TaskCurriculumState:
    """Track per-task difficulty and EMA metrics in shared memory."""

    def __init__(
        self,
        manager,
        tasks: Optional[Iterable[str]] = None,
        *,
        init_difficulty: float = 0.2,
        min_difficulty: float = 0.0,
        ema_decay: float = 0.98,
        step_size: float = 0.05,
        min_task_evals: int = 5,
    ) -> None:
        self._manager = manager
        self._state = manager.dict() if manager is not None else {}
        # Keep all difficulty values at or above the configured floor.
        self.min_difficulty = min_difficulty
        self.init_difficulty = max(init_difficulty, min_difficulty)
        self.ema_decay = ema_decay
        self.step_size = step_size
        self.min_task_evals = min_task_evals
        if tasks is not None:
            for task in tasks:
                self._ensure_task(task)

    def _ensure_task(self, task: str) -> None:
        if task in self._state:
            return
        payload = {
            "difficulty": float(self.init_difficulty),
            "ema_acc": 0.0,
            "ema_loss": 0.0,
            "best_ema_loss": math.inf,
            "last_update_step": 0,
            "step_count": 0,
        }
        if self._manager is not None:
            task_state = self._manager.dict(payload)
        else:
            task_state = dict(payload)
        self._state[task] = task_state

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_manager"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update_metrics(
        self,
        task: str,
        accuracy: float,
        loss: Optional[float],
        step: int,
    ) -> Dict[str, float]:
        self._ensure_task(task)
        task_state = self._state[task]
        decay = self.ema_decay

        ema_acc = float(task_state.get("ema_acc", 0.0))
        step_count = int(task_state.get("step_count", 0))
        if step_count == 0 or ema_acc == 0.0:
            ema_acc = float(accuracy)
        else:
            ema_acc = decay * ema_acc + (1.0 - decay) * float(accuracy)
        task_state["ema_acc"] = ema_acc

        if loss is not None:
            ema_loss = float(task_state.get("ema_loss", 0.0))
            ema_loss = decay * ema_loss + (1.0 - decay) * float(loss)
            task_state["ema_loss"] = ema_loss
            best_ema_loss = float(task_state.get("best_ema_loss", math.inf))
            if ema_loss < best_ema_loss:
                task_state["best_ema_loss"] = ema_loss

        task_state["step_count"] = step_count + 1
        task_state["last_seen_step"] = int(step)
        return {
            "ema_acc": ema_acc,
            "ema_loss": float(task_state.get("ema_loss", 0.0)),
        }

    def override_difficulty(self, task: str, new_difficulty: float) -> None:
        """Force a difficulty update and reset EMA to a safe passing value."""
        self._ensure_task(task)
        task_state = self._state[task]
        task_state["difficulty"] = float(max(new_difficulty, self.min_difficulty))
        task_state["ema_acc"] = 0.85

    def state_dict(self) -> Dict[str, Dict[str, float]]:
        return {task: dict(values) for task, values in self._state.items()}

    def load_state_dict(self, state_dict: Dict[str, Dict[str, float]]) -> None:
        for task, values in state_dict.items():
            self._ensure_task(task)
            self._state[task].update(values)

    def step_curriculum(
        self,
        task: str,
        step: int,
        cooldown: int,
        *,
        acc_threshold_high: float = 0.85,
        acc_threshold_low: float = 0.55,
    ) -> float:
        self._ensure_task(task)
        task_state = self._state[task]
        if int(task_state.get("step_count", 0)) < self.min_task_evals:
            return float(task_state.get("difficulty", self.init_difficulty))
        last_update_step = int(task_state.get("last_update_step", 0))
        if step - last_update_step < cooldown:
            return float(task_state.get("difficulty", self.init_difficulty))

        difficulty = float(task_state.get("difficulty", self.init_difficulty))
        ema_acc = float(task_state.get("ema_acc", 0.0))
        if ema_acc > acc_threshold_high:
            difficulty += self.step_size
        elif ema_acc < acc_threshold_low:
            difficulty -= self.step_size
        else:
            return difficulty

        difficulty = min(max(difficulty, self.min_difficulty), 1.0)
        task_state["difficulty"] = difficulty
        task_state["last_update_step"] = int(step)
        return difficulty

    def get_difficulty(self, task: str, jitter_prob: float = 0.1) -> float:
        self._ensure_task(task)
        task_state = self._state[task]
        difficulty = float(task_state.get("difficulty", self.init_difficulty))
        if jitter_prob > 0.0 and random.random() < jitter_prob:
            return random.uniform(0.0, difficulty)
        return difficulty

    def get_sampling_weight(self, task: str, min_weight: float = 0.05) -> float:
        """
        Return sampling weight inversely proportional to competence.
        High error -> High weight (sample more often).
        High accuracy -> Low weight (maintenance mode).
        """
        self._ensure_task(task)
        state = self._state[task]
        ema_acc = float(state.get("ema_acc", 0.0))
        return max(min_weight, 1.0 - ema_acc)

    def get_difficulty_snapshot(self) -> Dict[str, float]:
        """Return a plain dict snapshot for worker-local caching."""
        return {
            task: self.get_task_state(task)["difficulty"] for task in self._state
        }

    def get_sampling_weights_snapshot(self) -> Dict[str, float]:
        """Return a plain dict of sampling weights."""
        return {task: self.get_sampling_weight(task) for task in self._state}

    def get_task_state(self, task: str) -> Dict[str, float]:
        self._ensure_task(task)
        task_state = self._state[task]
        return {
            "difficulty": float(
                max(task_state.get("difficulty", self.init_difficulty), self.min_difficulty)
            ),
            "ema_acc": float(task_state.get("ema_acc", 0.0)),
            "ema_loss": float(task_state.get("ema_loss", 0.0)),
            "best_ema_loss": float(task_state.get("best_ema_loss", math.inf)),
            "last_update_step": int(task_state.get("last_update_step", 0)),
            "step_count": int(task_state.get("step_count", 0)),
        }

    def get_mean_difficulty(self, tasks: Iterable[str]) -> float:
        values = [self.get_task_state(task)["difficulty"] for task in tasks]
        if not values:
            return 0.0
        return sum(values) / len(values)


class DagUnlockState:
    """Track DAG-based staged unlock state using EMA accuracy thresholds."""

    def __init__(
        self,
        *,
        roots: Sequence[str],
        prereqs: Dict[str, Sequence[str]],
        thresholds: Dict[str, Dict[str, float]],
        patience_evals: int = 4,
        ramp_evals: int = 3,
        replay_ratio: float = 0.2,
        replay_ratio_backslide: float = 0.35,
        unlock_margin: float = 0.01,
        lock_margin: float = 0.03,
        frontier_recent_evals: int = 4,
        mastery_margin: float = 0.02,
        mastery_baseline: float = 0.99,
    ) -> None:
        self.roots = set(roots)
        self.prereqs = {task: list(reqs) for task, reqs in prereqs.items()}
        self.thresholds = {task: dict(values) for task, values in thresholds.items()}
        self.patience_evals = max(int(patience_evals), 1)
        self.ramp_evals = max(int(ramp_evals), 1)
        self.replay_ratio = float(replay_ratio)
        self.replay_ratio_backslide = float(replay_ratio_backslide)
        self.unlock_margin = float(unlock_margin)
        self.lock_margin = float(lock_margin)
        self.frontier_recent_evals = max(int(frontier_recent_evals), 1)
        self.mastery_margin = float(mastery_margin)
        self.mastery_baseline = float(mastery_baseline)
        self.eval_index = 0
        self.paused = False
        self._backslide = False

        tasks: Set[str] = set(self.roots)
        tasks.update(self.prereqs.keys())
        for reqs in self.prereqs.values():
            tasks.update(reqs)
        tasks.update(self.thresholds.keys())
        self.tasks = sorted(tasks)

        self.gate: Dict[str, float] = {task: 0.0 for task in self.tasks}
        self.streak: Dict[str, int] = {task: 0 for task in self.tasks}
        self.unlocked_eval_index: Dict[str, int] = {}
        for root in self.roots:
            self.gate[root] = 1.0
            self.unlocked_eval_index[root] = 0

    def state_dict(self) -> Dict[str, object]:
        return {
            "roots": list(self.roots),
            "prereqs": {task: list(reqs) for task, reqs in self.prereqs.items()},
            "thresholds": {task: dict(values) for task, values in self.thresholds.items()},
            "patience_evals": int(self.patience_evals),
            "ramp_evals": int(self.ramp_evals),
            "replay_ratio": float(self.replay_ratio),
            "replay_ratio_backslide": float(self.replay_ratio_backslide),
            "unlock_margin": float(self.unlock_margin),
            "lock_margin": float(self.lock_margin),
            "frontier_recent_evals": int(self.frontier_recent_evals),
            "mastery_margin": float(self.mastery_margin),
            "mastery_baseline": float(self.mastery_baseline),
            "eval_index": int(self.eval_index),
            "paused": bool(self.paused),
            "backslide": bool(self._backslide),
            "gate": {task: float(value) for task, value in self.gate.items()},
            "streak": {task: int(value) for task, value in self.streak.items()},
            "unlocked_eval_index": {
                task: int(value) for task, value in self.unlocked_eval_index.items()
            },
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not state:
            return
        if "roots" in state:
            self.roots = set(state["roots"])
        if "prereqs" in state:
            self.prereqs = {
                task: list(reqs) for task, reqs in state["prereqs"].items()
            }
        if "thresholds" in state:
            self.thresholds = {
                task: dict(values) for task, values in state["thresholds"].items()
            }
        if "patience_evals" in state:
            self.patience_evals = max(int(state["patience_evals"]), 1)
        if "ramp_evals" in state:
            self.ramp_evals = max(int(state["ramp_evals"]), 1)
        if "replay_ratio" in state:
            self.replay_ratio = float(state["replay_ratio"])
        if "replay_ratio_backslide" in state:
            self.replay_ratio_backslide = float(state["replay_ratio_backslide"])
        if "unlock_margin" in state:
            self.unlock_margin = float(state["unlock_margin"])
        if "lock_margin" in state:
            self.lock_margin = float(state["lock_margin"])
        if "frontier_recent_evals" in state:
            self.frontier_recent_evals = max(int(state["frontier_recent_evals"]), 1)
        if "mastery_margin" in state:
            self.mastery_margin = float(state["mastery_margin"])
        if "mastery_baseline" in state:
            self.mastery_baseline = float(state["mastery_baseline"])
        if "eval_index" in state:
            self.eval_index = int(state["eval_index"])
        if "paused" in state:
            self.paused = bool(state["paused"])
        if "backslide" in state:
            self._backslide = bool(state["backslide"])

        tasks: Set[str] = set(self.roots)
        tasks.update(self.prereqs.keys())
        for reqs in self.prereqs.values():
            tasks.update(reqs)
        tasks.update(self.thresholds.keys())
        self.tasks = sorted(tasks)

        if "gate" in state:
            self.gate = {task: float(value) for task, value in state["gate"].items()}
        if "streak" in state:
            self.streak = {task: int(value) for task, value in state["streak"].items()}
        if "unlocked_eval_index" in state:
            self.unlocked_eval_index = {
                task: int(value)
                for task, value in state["unlocked_eval_index"].items()
            }

    def _prereqs_satisfied(self, task: str, ema_acc: Dict[str, float]) -> bool:
        thresholds = self.thresholds.get(task, {})
        if not thresholds:
            return False
        for prereq, minimum in thresholds.items():
            score = ema_acc.get(prereq, 0.0)
            if score < minimum - self.unlock_margin:
                return False
        return True

    def update_from_ema(self, ema_acc: Dict[str, float]) -> None:
        self.eval_index += 1
        for root in self.roots:
            self.gate[root] = 1.0
            self.unlocked_eval_index.setdefault(root, 0)

        self._backslide = False
        for task, gate in self.gate.items():
            if gate <= 0.0 or task in self.roots:
                continue
            thresholds = self.thresholds.get(task, {})
            for prereq, minimum in thresholds.items():
                score = ema_acc.get(prereq, 0.0)
                if score < minimum - self.lock_margin:
                    self._backslide = True
                    break
            if self._backslide:
                break

        self.paused = self._backslide
        newly_unlocked: Set[str] = set()

        if not self.paused:
            for task in self.tasks:
                if task in self.roots:
                    continue
                if self.gate.get(task, 0.0) > 0.0:
                    continue
                if self._prereqs_satisfied(task, ema_acc):
                    self.streak[task] = self.streak.get(task, 0) + 1
                else:
                    self.streak[task] = 0
                if self.streak[task] >= self.patience_evals:
                    initial_gate = 1.0 if self.ramp_evals <= 1 else 1.0 / self.ramp_evals
                    self.gate[task] = initial_gate
                    self.unlocked_eval_index.setdefault(task, self.eval_index)
                    newly_unlocked.add(task)

            if self.ramp_evals > 1:
                ramp = 1.0 / self.ramp_evals
                for task, gate in list(self.gate.items()):
                    if task in newly_unlocked or task in self.roots:
                        continue
                    if 0.0 < gate < 1.0:
                        self.gate[task] = min(1.0, gate + ramp)

    def get_gate(self, task: str) -> float:
        return float(self.gate.get(task, 0.0))

    def get_gate_snapshot(self) -> Dict[str, float]:
        return {task: float(value) for task, value in self.gate.items()}

    def is_unlocked(self, task: str) -> bool:
        return self.get_gate(task) > 0.0

    def compute_frontier(self, ema_acc: Dict[str, float]) -> Set[str]:
        frontier: Set[str] = set()
        for task, gate in self.gate.items():
            if gate <= 0.0:
                continue
            unlocked_index = self.unlocked_eval_index.get(task)
            if unlocked_index is None:
                continue
            if self.eval_index - unlocked_index <= self.frontier_recent_evals:
                frontier.add(task)
                for prereq in self.prereqs.get(task, []):
                    threshold = self.thresholds.get(task, {}).get(
                        prereq, self.mastery_baseline
                    )
                    if ema_acc.get(prereq, 0.0) < threshold + self.mastery_margin:
                        frontier.add(prereq)
        return frontier

    def compute_replay_ratio(self, ema_acc: Dict[str, float]) -> float:
        if self.paused or self._backslide:
            return self.replay_ratio_backslide
        return self.replay_ratio
