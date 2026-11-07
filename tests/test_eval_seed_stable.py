"""Eval sets must not depend on the per-process hash salt (#160).

The cross-process checks build the eval grid in two fresh interpreters with
different PYTHONHASHSEED values and compare the results.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from utils import stable_seed

REPO_ROOT = Path(__file__).resolve().parents[1]

# 1. Every IID and OOD condition, built with the seeds eval_hub.run_algorithmic_suite uses.
# 2. A few conditions run through the real evaluate_condition (generation stubbed out) and
#    eval_hub's results serializer, so the digest that reaches the results JSON is checked too.
# The two control values show the salt really differs between the processes (not vacuous).
_CHILD = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import torch
import utils
from eval import ood_grid
from eval_hub import _algorithmic_results_to_dict
from utils import build_dataset, dataset_sha256, evaluate_condition, stable_seed

seed = 1234

def cond_seed_for(cond):
    cond_seed_bytes = hashlib.sha256(f"{cond.task}:{cond.name}:{seed}".encode()).digest()
    return seed + int.from_bytes(cond_seed_bytes[:4], "little")

digests = {}
sizes = {}
for cond in ood_grid.build_grid(include_iid=True, include_ood=True):
    difficulty = cond.params.get("difficulty", 0.5)
    dataset = build_dataset(cond.task, cond.n, cond.params, cond_seed_for(cond), difficulty=difficulty)
    digests[f"{cond.task}/{cond.name}"] = dataset_sha256(dataset)
    sizes[f"{cond.task}/{cond.name}"] = len(dataset)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]


def empty_generate_batch(model, tokenizer, prompts, **_):
    k = len(prompts)
    info = {"raw_predictions": [""] * k, "pred_token_ids": [[] for _ in range(k)],
            "stop_reasons": ["eos"] * k, "eos_emitted": [True] * k, "first_token_is_eos": [True] * k}
    return [""] * k, [0] * k, 0.001, info


utils.generate_batch = empty_generate_batch
wired = {}
picked = [c for c in ood_grid.build_grid(include_iid=True, include_ood=True)
          if (c.task, c.name) in {("addition", "iid"), ("dyck", "iid"), ("copy", "iid")}]
picked.append(ood_grid.OOD_CONDITIONS[0])
for cond in picked:
    n = 16
    difficulty = cond.params.get("difficulty", 0.5)
    res = evaluate_condition(None, CharTokenizer(), task=cond.task, condition=cond.name, params=cond.params,
                             n=n, device=torch.device("cpu"), max_new_tokens=8, seed=cond_seed_for(cond),
                             difficulty=difficulty)
    row = _algorithmic_results_to_dict([res])["conditions"][0]
    direct = dataset_sha256(build_dataset(cond.task, n, cond.params, cond_seed_for(cond), difficulty=difficulty))
    wired[f"{cond.task}/{cond.name}"] = [row["dataset_sha256"], direct]

print(json.dumps({
    "digests": digests,
    "sizes": sizes,
    "wired": wired,
    "train_sample_seed": {t: stable_seed(42, t) for t in ("addition", "parity", "copy")},
    "control_str_hash": hash("addition"),
    "control_old_example_seed": hash((1234, "addition", 0)) & 0x7FFFFFFF,
}))
"""


def _run_child(hash_seed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_eval_grid_identical_across_hash_seeds():
    first = _run_child("1")
    second = _run_child("2")

    # The two processes really do have different salts: the old seeding
    # expression gives different values in each.
    assert first["control_str_hash"] != second["control_str_hash"]
    assert first["control_old_example_seed"] != second["control_old_example_seed"]

    assert first["digests"], "grid produced no conditions"
    assert all(size > 0 for size in first["sizes"].values()), first["sizes"]
    assert first["digests"] == second["digests"]
    assert first["train_sample_seed"] == second["train_sample_seed"]

    # The digest evaluate_condition stores is the one eval_hub writes, it hashes the
    # examples actually built, and it is the same in both processes.
    assert len(first["wired"]) == 4
    for name, (in_json, direct) in first["wired"].items():
        assert len(in_json) == 64 and in_json == direct, name
    assert first["wired"] == second["wired"]


def _repo_sources():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] == "tests" or any(p.startswith(".") or p in ("__pycache__", "runs") for p in rel.parts):
            continue
        yield rel, path


def test_no_builtin_hash_in_repo_code():
    # hash() of a str is salted per process; anything seeded or ordered by it
    # differs between runs (#160). Use utils.stable_seed instead.
    offenders = []
    sources = list(_repo_sources())
    assert any(str(rel) == "train.py" for rel, _ in sources)
    for rel, path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_stable_seed_pinned_value():
    # Pins the derivation: changing it changes every eval set, so it must be deliberate.
    assert stable_seed(42, "addition", 0) == 1089842462


def test_stable_seed_canonicalises_parts():
    np = pytest.importorskip("numpy")
    assert stable_seed(np.int64(42), "addition", 0) == stable_seed(42, "addition", 0)

    class OddStr(str):
        __repr__ = object.__repr__  # repr would hold a memory address

    assert stable_seed(42, OddStr("addition"), 0) == stable_seed(42, "addition", 0)
    assert stable_seed(1, "1") != stable_seed("1", 1)
    assert 0 <= stable_seed(-7, "é") <= 0x7FFFFFFF


@pytest.mark.parametrize("bad", [1.5, None, True, (1, 2)])
def test_stable_seed_rejects_other_types(bad):
    with pytest.raises(TypeError):
        stable_seed(bad, "addition")
