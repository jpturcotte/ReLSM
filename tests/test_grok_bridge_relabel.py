"""The bridge's table checks. --mod_table_p: mul and div need a prime modulus (div's targets are inverses only
then). --mod_table_relabel: every (task, relabel) display_map accepts must encode the task's whole domain, the way
the scorer does; the combinations it cannot encode (div: operand and answer 0 have no discrete log) are refused."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.grok_bridge import PRIMITIVE_ROOT, TABLE_TASKS, check_modulus, display_map, example_for  # noqa: E402

P = 97


@pytest.mark.parametrize("task", ["mul", "div"])
@pytest.mark.parametrize("p", [0, 1, 4, 9, 91])
def test_mul_and_div_refuse_a_composite_modulus(task, p):
    with pytest.raises(ValueError):
        check_modulus(task, p)


@pytest.mark.parametrize("p", [2, 3, 23, 97])
def test_every_task_accepts_a_prime_modulus(p):
    for task in TABLE_TASKS:
        check_modulus(task, p)


def test_other_tasks_accept_any_modulus():
    for task in TABLE_TASKS:
        if task not in ("mul", "div"):
            check_modulus(task, 9)


def test_div_targets_are_inverses_mod_a_prime():
    answer = TABLE_TASKS["div"][1]
    for b in range(1, P):
        assert answer(1, b, P) * b % P == 1


def domain_pairs(task, p=P):
    dom = TABLE_TASKS[task][2]
    return [(a, b) for a in range(p) for b in range(p) if dom is None or bool(dom(a, b, p))]


@pytest.mark.parametrize("relabel", ["log", "perm"])
def test_div_is_refused(relabel):
    with pytest.raises(ValueError):
        display_map("div", P, relabel)


def test_log_needs_mul_and_a_known_root():
    with pytest.raises(ValueError):
        display_map("add", P, "log")
    with pytest.raises(ValueError):
        display_map("mul", 89, "log")
    assert P in PRIMITIVE_ROOT


def test_log_turns_mul_into_addition():
    disp = display_map("mul", P, "log")
    assert sorted(disp) == list(range(1, P)) and sorted(disp.values()) == list(range(P - 1))
    for a, b in domain_pairs("mul"):
        assert disp[a * b % P] == (disp[a] + disp[b]) % (P - 1)


@pytest.mark.parametrize("task,relabel",
                         [("mul", "log")] + [(t, "perm") for t in sorted(TABLE_TASKS) if t != "div"])
def test_every_accepted_relabel_encodes_its_whole_domain(task, relabel):
    disp = display_map(task, P, relabel, seed=20261107)
    assert sorted(disp.values()) == list(range(len(disp)))  # a bijection onto 0..n-1
    for a, b in domain_pairs(task):
        ex = example_for(a, b, P, 0, task, disp)  # KeyError here is the bug this guards
        assert ex["target"] == str(disp[TABLE_TASKS[task][1](a, b, P)])
