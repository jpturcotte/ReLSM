"""Task-level curriculum state for algorithmic training."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Optional


class TaskCurriculumState:
    """Track per-task difficulty and EMA metrics in shared memory."""

    def __init__(
        self,
        manager,
        tasks: Optional[Iterable[str]] = None,
        *,
        init_difficulty: float = 0.2,
        ema_decay: float = 0.98,
        step_size: float = 0.05,
    ) -> None:
        self._manager = manager
        self._state = manager.dict()
        self.init_difficulty = init_difficulty
        self.ema_decay = ema_decay
        self.step_size = step_size
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
        ema_acc = decay * ema_acc + (1.0 - decay) * float(accuracy)
        task_state["ema_acc"] = ema_acc

        if loss is not None:
            ema_loss = float(task_state.get("ema_loss", 0.0))
            ema_loss = decay * ema_loss + (1.0 - decay) * float(loss)
            task_state["ema_loss"] = ema_loss
            best_ema_loss = float(task_state.get("best_ema_loss", math.inf))
            if ema_loss < best_ema_loss:
                task_state["best_ema_loss"] = ema_loss

        task_state["step_count"] = int(task_state.get("step_count", 0)) + 1
        task_state["last_seen_step"] = int(step)
        return {
            "ema_acc": ema_acc,
            "ema_loss": float(task_state.get("ema_loss", 0.0)),
        }

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

        difficulty = min(max(difficulty, 0.0), 1.0)
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

    def get_task_state(self, task: str) -> Dict[str, float]:
        self._ensure_task(task)
        task_state = self._state[task]
        return {
            "difficulty": float(task_state.get("difficulty", self.init_difficulty)),
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
