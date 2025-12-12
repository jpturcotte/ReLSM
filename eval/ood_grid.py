"""Canonical IID and OOD evaluation grids for algorithmic tasks."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Condition:
    task: str
    name: str
    n: int
    params: Dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 32


def addition_grid() -> List[Condition]:
    iid = Condition(
        task="addition",
        name="iid",
        n=200,
        params={"digits": 4, "digit_bounds": (0, 9)},
        max_new_tokens=32,
    )
    ood_length = [
        Condition(
            task="addition",
            name=f"ood_length_{d}",
            n=200,
            params={"digits": d, "digit_bounds": (0, 9)},
            max_new_tokens=32,
        )
        for d in [5, 6, 8, 10, 12]
    ]
    ood_carry = Condition(
        task="addition",
        name="ood_carry",
        n=200,
        params={"digits": 4, "digit_bounds": (7, 9)},
        max_new_tokens=32,
    )
    ood_both = [
        Condition(
            task="addition",
            name=f"ood_both_{d}",
            n=200,
            params={"digits": d, "digit_bounds": (7, 9)},
            max_new_tokens=32,
        )
        for d in [6, 8, 10]
    ]
    return [iid] + ood_length + [ood_carry] + ood_both


def dyck_grid() -> List[Condition]:
    iid = Condition(
        task="dyck",
        name="iid",
        n=200,
        params={"max_depth": 4, "length": 18},
        max_new_tokens=8,
    )
    ood_length = [
        Condition(
            task="dyck",
            name=f"ood_length_{l}",
            n=200,
            params={"length": l, "max_depth": 4},
            max_new_tokens=8,
        )
        for l in [32, 48, 64]
    ]
    ood_depth = [
        Condition(
            task="dyck",
            name=f"ood_depth_{d}",
            n=200,
            params={"max_depth": d, "length": 18},
            max_new_tokens=8,
        )
        for d in [6, 8, 12]
    ]
    ood_both = [
        Condition(
            task="dyck",
            name="ood_both_len48_depth6",
            n=200,
            params={"length": 48, "max_depth": 6},
            max_new_tokens=8,
        ),
        Condition(
            task="dyck",
            name="ood_both_len64_depth8",
            n=200,
            params={"length": 64, "max_depth": 8},
            max_new_tokens=8,
        ),
    ]
    return [iid] + ood_length + ood_depth + ood_both


def chain_grid() -> List[Condition]:
    iid = Condition(
        task="chain",
        name="iid",
        n=200,
        params={"n_ops": 6, "operand_high": 20, "allow_negative": False},
        max_new_tokens=64,
    )
    ood_ops = [
        Condition(
            task="chain",
            name=f"ood_ops_{o}",
            n=200,
            params={"n_ops": o, "operand_high": 20, "allow_negative": False},
            max_new_tokens=64,
        )
        for o in [8, 10, 12]
    ]
    ood_mag = [
        Condition(
            task="chain",
            name=f"ood_magnitude_{m}",
            n=200,
            params={"n_ops": 6, "operand_high": m, "allow_negative": False},
            max_new_tokens=64,
        )
        for m in [50, 100]
    ]
    ood_neg = Condition(
        task="chain",
        name="ood_neg",
        n=200,
        params={"n_ops": 6, "operand_high": 20, "allow_negative": True},
        max_new_tokens=64,
    )
    ood_both = [
        Condition(
            task="chain",
            name=f"ood_both_ops{o}_mag50_neg",
            n=200,
            params={"n_ops": o, "operand_high": 50, "allow_negative": True},
            max_new_tokens=64,
        )
        for o in [10, 12]
    ]
    return [iid] + ood_ops + ood_mag + [ood_neg] + ood_both


def simple_task_grid() -> Dict[str, List[Condition]]:
    return {
        "parity": [
            Condition("parity", "iid", 200, {"length": 8}, max_new_tokens=8),
            *[
                Condition(
                    "parity",
                    f"ood_len_{l}",
                    200,
                    {"length": l},
                    max_new_tokens=8,
                )
                for l in [16, 32, 64]
            ],
        ],
        "copy": [
            Condition("copy", "iid", 200, {"length": 8}, max_new_tokens=64),
            *[
                Condition("copy", f"ood_len_{l}", 200, {"length": l}, max_new_tokens=64)
                for l in [16, 32]
            ],
        ],
        "reverse": [
            Condition("reverse", "iid", 200, {"length": 8}, max_new_tokens=64),
            *[
                Condition("reverse", f"ood_len_{l}", 200, {"length": l}, max_new_tokens=64)
                for l in [16, 32]
            ],
        ],
        "multiplication": [
            Condition("multiplication", "iid", 200, {"max_val": 99}, max_new_tokens=32),
            Condition("multiplication", "ood_digits_3", 200, {"max_val": 999}, max_new_tokens=32),
            Condition("multiplication", "ood_digits_4", 200, {"max_val": 9999}, max_new_tokens=32),
        ],
        "mod_add": [
            Condition("mod_add", "iid", 200, {"mod": 97}, max_new_tokens=16),
            Condition("mod_add", "ood_mod_1000", 200, {"mod": 1000}, max_new_tokens=16),
            Condition("mod_add", "ood_mod_10000", 200, {"mod": 10000}, max_new_tokens=16),
        ],
        "compare": [
            Condition("compare", "iid", 200, {"max_val": 999}, max_new_tokens=8),
            Condition("compare", "ood_digits_5", 200, {"max_val": 99999}, max_new_tokens=8),
            Condition("compare", "ood_digits_6", 200, {"max_val": 999999}, max_new_tokens=8),
        ],
        "successor": [
            Condition("successor", "iid", 200, {"max_val": 999}, max_new_tokens=8),
            Condition("successor", "ood_digits_5", 200, {"max_val": 99999}, max_new_tokens=8),
            Condition("successor", "ood_digits_6", 200, {"max_val": 999999}, max_new_tokens=8),
        ],
    }


def build_grid(tasks: Optional[List[str]] = None) -> List[Condition]:
    tasks = tasks or [
        "addition",
        "dyck",
        "chain",
        "parity",
        "copy",
        "reverse",
        "multiplication",
        "mod_add",
        "compare",
        "successor",
    ]
    selected = set(tasks)
    grid: List[Condition] = []

    if "addition" in selected:
        grid.extend(addition_grid())
    if "dyck" in selected:
        grid.extend(dyck_grid())
    if "chain" in selected:
        grid.extend(chain_grid())

    simple = simple_task_grid()
    for task, conditions in simple.items():
        if task in selected:
            grid.extend(conditions)

    return grid
