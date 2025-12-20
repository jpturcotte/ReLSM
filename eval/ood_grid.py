"""Canonical IID and OOD evaluation grids for algorithmic tasks."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from data import TASK_DIFFICULTY_CONFIG

OOD_GRID_VERSION = "2024-09-05"


@dataclass
class Condition:
    task: str
    name: str
    n: int
    params: Dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 32


def build_iid_conditions(n_per_task: int = 200) -> List[Condition]:
    """IID conditions match training distribution."""
    conditions: List[Condition] = []
    for task, config in TASK_DIFFICULTY_CONFIG.items():
        final_params = config["phases"][-1][2]
        conditions.append(
            Condition(
                task=task,
                name="iid",
                n=n_per_task,
                params=final_params,
            )
        )
    return conditions


def build_ood_conditions(n_per_condition: int = 100) -> List[Condition]:
    """OOD conditions test length/difficulty generalization."""
    conditions: List[Condition] = []

    for task in ["addition", "successor"]:
        config = TASK_DIFFICULTY_CONFIG[task]
        ood_start = config["ood_start"]
        for digits in [ood_start, ood_start + 2, ood_start + 4]:
            conditions.append(
                Condition(
                    task=task,
                    name=f"ood_{digits}digit",
                    n=n_per_condition,
                    params={"digits": digits},
                )
            )

    for digits in [4, 5]:
        conditions.append(
            Condition(
                task="multiplication",
                name=f"ood_{digits}digit",
                n=n_per_condition,
                params={"digits": digits},
            )
        )

    for task in ["copy", "reverse", "parity"]:
        for length in [16, 20, 24]:
            conditions.append(
                Condition(
                    task=task,
                    name=f"ood_len{length}",
                    n=n_per_condition,
                    params={"length": length},
                )
            )

    for depth in [5, 6]:
        conditions.append(
            Condition(
                task="dyck",
                name=f"ood_depth{depth}",
                n=n_per_condition,
                params={"max_depth": depth},
            )
        )

    for mod in [2003, 4999]:
        conditions.append(
            Condition(
                task="mod_add",
                name=f"ood_mod{mod}",
                n=n_per_condition,
                params={"mod": mod},
            )
        )

    return conditions


IID_CONDITIONS = build_iid_conditions()
OOD_CONDITIONS = build_ood_conditions()
ALL_CONDITIONS = IID_CONDITIONS + OOD_CONDITIONS


def build_grid(tasks: Optional[List[str]] = None) -> List[Condition]:
    """Build evaluation grid filtered to requested tasks."""
    if tasks is None:
        return list(ALL_CONDITIONS)
    selected = set(tasks)
    return [condition for condition in ALL_CONDITIONS if condition.task in selected]
