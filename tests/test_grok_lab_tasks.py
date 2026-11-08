"""CPU tests for the --tasks prototype in tools/grok_lab.py (several tasks in one model).

The anchor is test_tasks_add_is_the_single_task_run: '--tasks add' must reproduce '--task add' bit for bit,
so every multi-task number is read on the same scale as the single-task arms. Threads are pinned to 1.
"""
import itertools
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import grok_lab as gl  # noqa: E402

SHORT = ["--updates", "30", "--eval_every", "10", "--ckpt_every", "0"]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def args_for(tmp_path, *extra):
    return gl.parse_args(["--out", str(tmp_path), "--threads", "1", "--device", "cpu", *extra])


def lines(out):
    return [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("extra", [[], ["--operands", "separate"], ["--batch", "64", "--clip", "1"],
                                   ["--dropout", "0.2", "--diagonals", "test"], ["--pairs", "paired"]])
def test_tasks_add_is_the_single_task_run(tmp_path, extra):
    common = ["--p", "23", *SHORT, *extra]
    gl.run(args_for(tmp_path / "single", *common))
    gl.run(args_for(tmp_path / "multi", "--tasks", "add", *common))
    w1 = torch.load(tmp_path / "single" / "final.pt")["model"]
    w2 = torch.load(tmp_path / "multi" / "final.pt")["model"]
    assert w1.keys() == w2.keys() and all(torch.equal(w1[k], w2[k]) for k in w1)
    pooled = {"train_loss", "train_acc", "test_loss", "test_acc", "test_acc_unseen", "test_loss_unseen",
              "test_acc_diagonal", "test_acc_unseen_offdiag", "test_loss_unseen_offdiag"}
    for s, m in zip(lines(tmp_path / "single"), lines(tmp_path / "multi")):
        for k, v in s.items():
            if k in ("train_s", "elapsed_s"):
                continue
            if k in pooled:  # example-weighted mean over one task: the same number up to rounding
                assert m[k + "@add"] == v, k
                assert (m[k] is None) if v is None else m[k] == pytest.approx(v, rel=1e-12, abs=1e-15), k
            elif k in m:  # training readouts and the model-wide norms
                assert m[k] == v, k
            else:
                assert m[k + "@add"] == v, k
    r1 = json.loads((tmp_path / "single" / "run_record.json").read_text())
    r2 = json.loads((tmp_path / "multi" / "run_record.json").read_text())
    for k in ("n_train", "n_test", "n_test_unseen", "n_test_diagonal", "n_test_unseen_offdiag", "model_config",
              "first_update_unseen_ge_0.99", "n_wrong_final"):
        if k == "n_wrong_final":
            assert {kk: vv for kk, vv in r2[k].items() if kk != "by_task"} == r1[k]
        else:
            assert r1[k] == r2[k], k
    e1 = json.loads((tmp_path / "single" / "final_errors.json").read_text())
    e2 = json.loads((tmp_path / "multi" / "final_errors.json").read_text())
    assert e2["test"] == [r + ["add"] for r in e1["test"]] and e2["train"] == [r + ["add"] for r in e1["train"]]


def test_task_answers_brute_force():
    p = 7
    inv = {b: pow(b, -1, p) for b in range(1, p)}
    want = {"add": lambda a, b: (a + b) % p, "add_dup": lambda a, b: (a + b) % p,
            "add2b": lambda a, b: (a + 2 * b) % p, "sub": lambda a, b: (a - b) % p,
            "mul": lambda a, b: a * b % p, "div": lambda a, b: a * inv[b] % p, "sq": lambda a, b: (a * a + b * b) % p,
            "sqab": lambda a, b: (a * a + a * b + b * b) % p, "cubeab": lambda a, b: (a ** 3 + a * b) % p,
            "gcd": math.gcd}
    a, b = torch.arange(p).repeat_interleave(p), torch.arange(p).repeat(p)
    for name, f in want.items():
        spec = gl.TASK_SPECS[name]
        dom = spec.domain(a, b, p) if spec.domain else torch.ones(p * p, dtype=torch.bool)
        got = spec.answer(a[dom], b[dom], p).tolist()
        assert got == [f(x, y) for x, y in zip(a[dom].tolist(), b[dom].tolist())], name
        sym = spec.answer(b[dom], a[dom], p).tolist() == got
        assert sym == spec.commutative or name in ("div",), name  # div's domain is not symmetric
    assert gl.TASK_SPECS["mul"].domain(a, b, p).sum() == (p - 1) ** 2
    perms = list(itertools.permutations(range(5)))
    t = gl.s5_table()
    assert t.shape == (120, 120) and perms[0] == (0, 1, 2, 3, 4)
    for f, g in [(1, 7), (33, 119), (64, 5)]:
        assert perms[t[f, g]] == tuple(perms[f][perms[g][k]] for k in range(5))  # (f o g)(k) = f(g(k))
    assert torch.equal(t[0], torch.arange(120)) and torch.equal(t[:, 0], torch.arange(120))  # identity
    assert all(sorted(row.tolist()) == list(range(120)) for row in t)  # Latin square: a group table
    gen = torch.Generator().manual_seed(0)
    x, y, z = torch.randint(0, 120, (3, 200), generator=gen)
    assert torch.equal(t[t[x, y], z], t[x, t[y, z]])  # associative
    assert not torch.equal(t, t.T)  # S5 is not commutative


def test_layout(tmp_path):
    p = 23
    lay = gl.make_layout(args_for(tmp_path, "--p", str(p), "--tasks", "add,sub"))
    assert (lay.op, lay.eq, lay.vocab) == ([23, 24], 25, 26)
    assert lay.cols[0] == lay.cols[1] == list(range(23)) + [23, 24, 25]  # shared answers
    lay = gl.make_layout(args_for(tmp_path, "--p", str(p), "--tasks", "add,sub", "--answers", "per_task"))
    assert lay.answer == [23, 46] and (lay.op, lay.eq, lay.vocab) == ([69, 70], 71, 72)
    assert lay.cols[1] == list(range(46, 69)) + [69, 70, 71] and not set(range(23)) & set(lay.cols[0])
    a = args_for(tmp_path, "--p", str(p), "--tasks", "s5,add", "--operands", "separate")
    lay = gl.make_layout(a)
    assert lay.first == {"num": 0, "s5": 23} and lay.second == {"num": 143, "s5": 166}
    assert (lay.op, lay.eq) == ([286, 287], 288) and gl.vocab_size(a) == 289
    assert lay.cols[0][:120] == list(range(23, 143)) and len(lay.cols[0]) == 123
    x, y = gl.encode_task(torch.tensor([5 * 120 + 7]), 0, a, lay)
    assert x.tolist() == [[28, 286, 173, 288]] and y.tolist() == [gl.s5_table()[5, 7].item()]
    for extra, v in (([], p + 2), (["--operands", "separate"], 2 * p + 2)):  # one add task: the old vocabulary
        assert gl.vocab_size(args_for(tmp_path, "--p", str(p), "--tasks", "add", *extra)) == v


@pytest.mark.parametrize("pairs", ["ordered", "paired"])
def test_split_respects_the_domain(pairs):
    p = 23
    ids = torch.arange(p * p)
    valid = ids % p != 0  # div: b != 0
    train, test, seen = gl.make_split(p, 0.5, 0, pairs, valid=valid)
    tr, te = set(train.tolist()), set(test.tolist())
    assert tr.isdisjoint(te) and tr | te == set(ids[valid].tolist())
    assert abs(len(tr) - round(0.5 * int(valid.sum()))) <= 1
    for i, s in zip(test.tolist(), seen.tolist()):
        a, b = divmod(i, p)
        assert s == (b * p + a in tr)
    if pairs == "paired":  # a pair and its transpose stay together whenever both are in the domain
        assert all((i % p) * p + i // p in tr for i in tr if (i // p) != 0)
    for side in ("train", "test"):  # --diagonals never adds (0, 0), which is outside the domain
        tr2, te2, _ = gl.make_split(p, 0.5, 0, pairs, side, valid=valid)
        assert 0 not in set(tr2.tolist()) | set(te2.tolist())
    for a, b in zip(gl.make_split(p, 0.5, 3, pairs), gl.make_split(p, 0.5, 3, pairs, valid=torch.ones(p * p, dtype=torch.bool))):
        assert torch.equal(a, b)  # an all-valid domain is the old split


def test_task_splits_independent_or_aligned(tmp_path):
    p = 23
    assert args_for(tmp_path, "--p", str(p), "--tasks", "add").task_split == "aligned"  # the default
    for mode in ("independent", "aligned"):
        a = args_for(tmp_path, "--p", str(p), "--tasks", "add,sub,add_dup", "--task_split", mode)
        _, views, lay = gl.setup_tasks(a, torch.device("cpu"))
        ids = [set((v.data["train_x"][:, 0] * p + v.data["train_x"][:, 2]).tolist()) for v in views]
        assert ids[0] == set(gl.make_split(p, 0.5, 0)[0].tolist())  # task 0: the single-task split
        assert (ids[0] == ids[1] == ids[2]) == (mode == "aligned")
        assert views[1].record["split_seed"] == (0 if mode == "aligned" else gl.SPLIT_SEED_STRIDE)
        assert lay.op == [p, p + 1, p + 2] and torch.equal(views[0].data["train_y"], views[2].data["train_y"]) == (
            mode == "aligned")  # add_dup: add's table under its own op token


@pytest.mark.parametrize("pairs", ["ordered", "paired"])
def test_aligned_split_holds_across_domains(pairs):
    """add (all pairs) and mul (a, b != 0) from the same seed: mul's test pairs are add's test pairs, up to a
    few at the edge of the shared shuffle. Independent seeds leak about half of them into add's train set."""
    p = 23
    ids = torch.arange(p * p)
    mul_dom = (ids // p != 0) & (ids % p != 0)
    add_train = set(gl.make_split(p, 0.5, 0, pairs)[0].tolist())
    leak = lambda seed: len(set(gl.make_split(p, 0.5, seed, pairs, valid=mul_dom)[1].tolist()) & add_train)  # noqa: E731
    n_test = len(gl.make_split(p, 0.5, 0, pairs, valid=mul_dom)[1])
    assert leak(0) <= 0.1 * n_test  # aligned: a few pairs at most
    assert leak(gl.SPLIT_SEED_STRIDE) >= 0.3 * n_test  # independent: about half


def test_multi_run_logs_each_task_and_pools(tmp_path):
    p = 23
    last = gl.run(args_for(tmp_path, "--p", str(p), "--tasks", "add,div,s5", "--task_frac", "s5=0.3", *SHORT))
    rec = json.loads((tmp_path / "run_record.json").read_text())
    tasks = {t["task"]: t for t in rec["tasks"]}
    assert tasks["div"]["n_train"] + tasks["div"]["n_test"] == p * (p - 1)
    assert tasks["s5"]["n_train"] == round(0.3 * 14400) and tasks["s5"]["n_classes"] == 120 + 4
    assert rec["n_train"] == sum(t["n_train"] for t in rec["tasks"]) == rec["examples_per_update"]
    assert rec["model_config"]["vocab_size"] == p + 120 + 4
    for name in tasks:
        assert 0 <= last[f"test_acc@{name}"] <= 1 and math.isfinite(last[f"train_margin_mean_offdiag@{name}"])
        assert f"first_update_test_ge_0.99@{name}" in rec and f"first_update_unseen_ge_0.99@{name}" in rec
    pooled = sum(last[f"test_acc@{n}"] * t["n_test"] for n, t in tasks.items()) / rec["n_test"]
    assert last["test_acc"] == pytest.approx(pooled, rel=1e-12)
    assert "norm_total" in last and "norm_total@add" not in last
    err = json.loads((tmp_path / "final_errors.json").read_text())
    assert err["columns"][-1] == "task"
    s5 = gl.s5_table()
    for a, b, t, q, s, m, name in err["train"] + err["test"]:
        want = {"add": lambda: (a + b) % p, "div": lambda: a * pow(b, -1, p) % p, "s5": lambda: s5[a, b].item()}[name]()
        assert t == want and q != t
    by = rec["n_wrong_final"]["by_task"]
    assert sum(v["test"] for v in by.values()) == rec["n_wrong_final"]["test"] == len(err["test"])
    assert by["div"]["test"] == round((1 - last["test_acc@div"]) * tasks["div"]["n_test"])


@pytest.mark.parametrize("batch", [None, "64"])
def test_schedule_starts_and_stops_tasks(tmp_path, monkeypatch, batch):
    seen = []
    real = gl.multi_loss

    def spy(model, x, y, parts, views, args, active):
        seen.append(([i for i, _, _ in parts], sum(c for _, _, c in parts), len(x)))
        assert set(i for i, _, _ in parts) <= set(active)
        return real(model, x, y, parts, views, args, active)

    monkeypatch.setattr(gl, "multi_loss", spy)
    gl.run(args_for(tmp_path, "--p", "23", "--tasks", "add,sub", "--task_start", "sub=10", "--task_stop", "add=20",
                    *SHORT, *(["--batch", batch] if batch else [])))
    assert len(seen) == 30
    for u, (tasks, n_parts, n_x) in enumerate(seen):
        want = [0] if u < 10 else [0, 1] if u < 20 else [1]
        assert tasks == want or (batch and set(tasks) <= set(want)), (u, tasks)
        assert n_parts == n_x == (64 if batch else 264 * len(want))  # round(0.5 * 529) = 264 train pairs per task
    assert [line["active_tasks"] for line in lines(tmp_path)] == [["add"], ["add"], ["add", "sub"], ["sub"]]


def test_multi_loss_weighting(tmp_path):
    p = 23
    for weighting in ("equal", "examples"):
        a = args_for(tmp_path, "--p", str(p), "--tasks", "add,mul", "--task_weighting", weighting,
                     "--task_weight", "mul=3")
        data, views, _ = gl.setup_tasks(a, torch.device("cpu"))
        n0, n1 = views[0].hi - views[0].lo, views[1].hi - views[1].lo
        assert (n0, n1) == (264, 242)  # round(0.5 * 529), round(0.5 * 22^2)
        torch.manual_seed(0)
        model = gl.build_model(a)

        def task_losses(x, y, k0):
            return [gl.loss_of(gl.answer_logits(model, x[s], v.args), y[s], a).item()
                    for s, v in ((slice(0, k0), views[0]), (slice(k0, len(x)), views[1]))]

        # full batch: the objective itself
        x, y = data["train_x"], data["train_y"]
        full = gl.multi_loss(model, x, y, [(0, torch.arange(n0), n0), (1, torch.arange(n0, n0 + n1), n1)],
                             views, a, (0, 1)).item()
        l0, l1 = task_losses(x, y, n0)
        want = (l0 + 3 * l1) / 4 if weighting == "equal" else (n0 * l0 + 3 * n1 * l1) / (n0 + 3 * n1)
        assert full == pytest.approx(want, rel=1e-6)
        # a minibatch with 50 add rows and 20 mul rows: each task's mean scaled by c_i N / (B n_i)
        rows = torch.cat([torch.arange(0, 50), torch.arange(views[1].lo, views[1].lo + 20)])
        x, y = data["train_x"][rows], data["train_y"][rows]
        got = gl.multi_loss(model, x, y, [(0, torch.arange(50), 50), (1, torch.arange(50, 70), 20)],
                            views, a, (0, 1)).item()
        m0, m1 = task_losses(x, y, 50)
        big_n, b = n0 + n1, 70
        if weighting == "equal":
            want = 0.25 * (50 * big_n) / (b * n0) * m0 + 0.75 * (20 * big_n) / (b * n1) * m1
        else:
            want = (50 * m0 + 3 * 20 * m1) * big_n / (b * (n0 + 3 * n1))
        assert got == pytest.approx(want, rel=1e-6)
        # a task absent from the batch is not renormalized away: add alone keeps its own weight
        only = gl.multi_loss(model, x[:50], y[:50], [(0, torch.arange(50), 50)], views, a, (0, 1)).item()
        scale = 0.25 * big_n / n0 if weighting == "equal" else big_n / (n0 + 3 * n1)
        assert only == pytest.approx(scale * m0, rel=1e-6)


def test_idle_phase_is_decay_only(tmp_path):
    a = ["--p", "23", "--tasks", "add", "--task_stop", "add=10", *SHORT]
    with pytest.raises(ValueError, match="no task is active"):
        gl.run(args_for(tmp_path / "refused", *a))
    gl.run(args_for(tmp_path, *a, "--allow_idle"))
    log = lines(tmp_path)
    assert [line["active_tasks"] for line in log] == [["add"], ["add"], [], []]
    assert all(v == 0.0 for line in log[2:] for v in line["train_obj_trace"])  # zero loss
    assert all(line["grad_norm"] == 0.0 and line["rel_update"] > 0 for line in log[2:])  # no gradient, still moving


def test_wd_until_turns_decay_off(tmp_path):
    gl.run(args_for(tmp_path, "--p", "23", "--wd_until", "10", *SHORT))
    groups = torch.load(tmp_path / "final.pt")["optimizer"]["param_groups"]
    assert all(g["weight_decay"] == 0.0 for g in groups)
    assert args_for(tmp_path).wd_until is None
    for bad in ("0", "30"):
        with pytest.raises(SystemExit):
            args_for(tmp_path, "--wd_until", bad, *SHORT)


def test_per_task_answers_are_output_only_tokens(tmp_path):
    p = 23
    a = args_for(tmp_path, "--p", str(p), "--tasks", "add", "--answers", "per_task")
    data, views, lay = gl.setup_tasks(a, torch.device("cpu"))
    torch.manual_seed(0)
    model = gl.build_model(a)
    z = gl.answer_logits(model, views[0].data["train_x"], views[0].args)
    assert z.shape[1] == p + 2 and views[0].args.answer_cols.tolist()[:p] == list(range(p, 2 * p))
    gl.loss_of(z, views[0].data["train_y"], a).backward()
    g = model.tok_emb.weight.grad
    assert g[p:2 * p].abs().sum() > 0  # answer rows get gradient only through the tied head
    last = gl.run(args_for(tmp_path, "--p", str(p), "--tasks", "add", "--answers", "per_task", *SHORT))
    assert math.isfinite(last["test_loss@add"])


def test_primitive_root():
    for p, g in ((97, 5), (23, 5), (7, 3)):
        assert gl.primitive_root(p) == g
        assert sorted(pow(g, j, p) for j in range(p - 1)) == list(range(1, p))  # generates (Z/p)*


def test_embedding_spectra_find_planted_structure():
    p, d = 23, 8
    a = torch.arange(p, dtype=torch.float64)
    w = torch.zeros(p + 2, d)
    w[:p, 0], w[:p, 1] = torch.cos(2 * math.pi * 5 * a / p), torch.sin(2 * math.pi * 5 * a / p)  # add-like, k = 5
    s = gl.embedding_spectra(w, p, 0)
    assert s["emb_fourier_top5"][0] == 5 and s["emb_fourier_top5_frac"] == pytest.approx(1.0)
    assert len(s["emb_fourier_spectrum"]) == 11 and len(s["emb_logfourier_spectrum"]) == 11  # k = 1..11
    g = gl.primitive_root(p)
    w = torch.zeros(p + 2, d)
    for j in range(p - 1):  # mul-like: periodic in log a with frequency 3; row 0 left at zero
        w[pow(g, j, p), 0] = math.cos(2 * math.pi * 3 * j / (p - 1))
        w[pow(g, j, p), 1] = math.sin(2 * math.pi * 3 * j / (p - 1))
    s = gl.embedding_spectra(w, p, 0)
    assert s["emb_logfourier_top5"][0] == 3 and s["emb_logfourier_top5_frac"] == pytest.approx(1.0)
    assert s["emb_fourier_top5_frac"] < 0.9  # the same table is not concentrated along a
    shifted = torch.cat([torch.randn(p, d), w[:p]])  # a block that starts at row p
    assert gl.embedding_spectra(shifted, p, p)["emb_logfourier_top5"][0] == 3
    assert "emb_logfourier_spectrum" not in gl.embedding_spectra(torch.randn(26, d), 25, 0)  # p not prime
    assert gl.primitive_root(2) == 1  # (Z/2)* = {1}
    s2 = gl.embedding_spectra(torch.randn(4, d), 2, 0)  # --p 2: no log-Fourier basis, and no crash
    assert "emb_logfourier_spectrum" not in s2 and "emb_fourier_spectrum" in s2


@pytest.mark.parametrize("extra", [[], ["--operands", "separate"], ["--tasks", "s5"]])
def test_spectra_are_logged(tmp_path, extra):
    last = gl.run(args_for(tmp_path, "--p", "23", *SHORT, *extra))
    if "s5" in extra:  # no number tokens
        assert not any(k.startswith("emb_") for k in last)
        return
    assert 0 < last["emb_fourier_top5_frac"] <= 1 and len(last["emb_logfourier_top5"]) == 5
    assert ("emb_fourier_top5_frac_b" in last) == ("separate" in extra)


def test_task_grad_cosines(tmp_path):
    p = 23
    last = gl.run(args_for(tmp_path, "--p", str(p), "--tasks", "add,sub,mul", *SHORT))
    for pair in ("add|sub", "add|mul", "sub|mul"):
        for scope in ("all", "trunk"):
            assert -1 <= last[f"grad_cos_{scope}@{pair}"] <= 1
    assert all(last[f"grad_norm_task@{t}"] > 0 for t in ("add", "sub", "mul"))
    single = gl.run(args_for(tmp_path / "single", "--p", str(p), "--tasks", "add", *SHORT))
    assert not any(k.startswith("grad_cos") for k in single)
    # against a direct computation on a fresh model
    a = args_for(tmp_path, "--p", str(p), "--tasks", "add,sub")
    _, views, _ = gl.setup_tasks(a, torch.device("cpu"))
    torch.manual_seed(0)
    model = gl.build_model(a)
    got = gl.task_grad_cosines(model, views, a)
    g = []
    for v in views:
        model.zero_grad()
        gl.loss_of(gl.answer_logits(model, v.data["train_x"], v.args), v.data["train_y"], a).backward()
        g.append(torch.cat([q.grad.flatten().double() for q in model.parameters()]))
    assert got["grad_cos_all@add|sub"] == pytest.approx(float(g[0] @ g[1] / (g[0].norm() * g[1].norm())), rel=1e-6)


def test_grad_cosine_readout_does_not_change_training(tmp_path, monkeypatch):
    common = ["--p", "23", "--tasks", "add,sub", *SHORT, "--dropout", "0.1", "--batch", "64"]
    gl.run(args_for(tmp_path / "on", *common))
    monkeypatch.setattr(gl, "task_grad_cosines", lambda *a: {})
    gl.run(args_for(tmp_path / "off", *common))
    w1, w2 = (torch.load(tmp_path / k / "final.pt")["model"] for k in ("on", "off"))
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


def test_ckpt_at(tmp_path):
    gl.run(args_for(tmp_path, "--p", "23", *SHORT, "--ckpt_at", "10,25"))
    assert sorted(f.name for f in tmp_path.glob("ckpt_*.pt")) == ["ckpt_000010.pt", "ckpt_000025.pt"]
    assert torch.load(tmp_path / "ckpt_000025.pt")["update"] == 25
    for bad in ("0", "30", "x"):
        with pytest.raises(SystemExit):
            args_for(tmp_path, *SHORT, "--ckpt_at", bad)


def test_forgetting_summary():
    hist = {"add": [(100, 0.5, 0.5, 0.5), (200, 1.0, 1.0, 0.995), (300, 0.7, 0.7, 0.6), (400, 1.0, 1.0, 1.0)]}
    s = gl.summarize_tasks(hist, 400)
    assert s["first_update_test_ge_0.99@add"] == 200 and s["min_test_acc_after_first_ge_0.99@add"] == 0.6
    assert s["first_update_unseen_ge_0.99@add"] == 200
    assert s["min_test_acc_last10pct@add"] == 1.0  # the dip at 300 was before the last 10%
    assert gl.summarize_tasks({"sub": [(100, 0.1, 0.1, 0.1)]}, 400)["min_test_acc_after_first_ge_0.99@sub"] is None


@pytest.mark.parametrize("bad", [["--tasks", "add,nope"], ["--tasks", "add,add"], ["--tasks", "add", "--task", "add2b"],
                                 ["--tasks", "add", "--holdout_line", "1"], ["--answers", "per_task"],
                                 ["--task_frac", "add=0.3"], ["--tasks", "add", "--task_frac", "sub=0.3"],
                                 ["--tasks", "mul", "--p", "24"], ["--tasks", "add,sub", "--task_stop", "add=0"],
                                 ["--tasks", "add", "--task_frac", "add=1.5"], ["--task_split", "independent"],
                                 ["--allow_idle"]])
def test_bad_task_flags(tmp_path, bad):
    with pytest.raises(SystemExit):
        args_for(tmp_path, *bad)


def test_gcd_needs_its_diagonals_in_train(tmp_path):
    common = ["--p", "23", "--tasks", "gcd", "--updates", "10", "--eval_every", "10", "--ckpt_every", "0"]
    with pytest.raises(ValueError, match="never seen in train"):  # gcd 22 occurs only at (22, 22)
        gl.run(args_for(tmp_path / "a", *common, "--diagonals", "test"))
    gl.run(args_for(tmp_path / "b", *common, "--diagonals", "train"))
    rec = json.loads((tmp_path / "b" / "run_record.json").read_text())
    assert rec["tasks"][0]["n_train_diagonal"] == 22 and rec["tasks"][0]["missing_residues"] == []


def test_schedule_with_no_active_task_refuses_to_start(tmp_path):
    with pytest.raises(ValueError, match="no task is active"):
        gl.run(args_for(tmp_path, "--p", "23", "--tasks", "add,sub", "--task_start", "sub=20", "--task_stop", "add=10",
                        *SHORT))


@pytest.mark.parametrize("tasks,extra", [("add,gcd", ["--diagonals", "train"]), ("add,mul", [])])
def test_class_metrics_logged_per_task_match_final_errors(tmp_path, tasks, extra):
    """The per-eval class readouts at the last eval, rebuilt from final_errors.json the way the 2026-11-07
    metrics review rebuilt them (curve_metrics.confusion_metrics), plus the pooled off-diagonal loss. gcd needs
    its diagonals in train; add,mul keeps test diagonals, so the diagonal exclusion is exercised too."""
    p = 23
    common = ["--p", str(p), "--tasks", tasks, *extra, *SHORT]
    gl.run(args_for(tmp_path, *common))
    rows, rec = lines(tmp_path), json.loads((tmp_path / "run_record.json").read_text())
    names = tasks.split(",")
    evals = [r for r in rows if f"test_acc@{names[-1]}" in r]
    keys = [f"test_{k}_unseen_offdiag" for k in gl.CLASS_KEYS] + ["test_loss_unseen_offdiag"]
    assert len(evals) == 4 and all(f"{k}@{t}" in r for r in evals for k in keys for t in names)
    assert all("test_loss_unseen_offdiag" in r for r in evals)  # pooled
    if not extra:  # test diagonals present: the off-diagonal set is a strict subset of unseen
        assert rec["n_test_unseen_offdiag"] < rec["n_test_unseen"]
    last = evals[-1]
    err = json.loads((tmp_path / "final_errors.json").read_text())
    c = {k: err["columns"].index(k) for k in ("a", "b", "answer", "pred", "transpose_seen", "task")}
    _, views, _ = gl.setup_tasks(args_for(tmp_path / "x", *common), torch.device("cpu"))
    for v in views:
        name, commutative = v.name, gl.TASK_SPECS[v.name].commutative
        xa, xb = gl.operands_of(v.data["test_x"], v.args)
        off = ~v.data["test_seen"] & (xa != xb)
        assert v.record["n_test_unseen_offdiag"] == int(off.sum())
        y = v.data["test_y"][off].tolist()
        n_true = {k: y.count(k) for k in set(y)}
        wrong = [r for r in err["test"] if r[c["task"]] == name and r[c["a"]] != r[c["b"]] and not r[c["transpose_seen"]]]
        tp = dict(n_true)
        n_pred = dict(n_true)
        for r in wrong:
            tp[r[c["answer"]]] -= 1
            n_pred[r[c["answer"]]] -= 1
            n_pred[r[c["pred"]]] = n_pred.get(r[c["pred"]], 0) + 1
        n = len(y)
        bal = sum(tp[k] / n_true[k] for k in n_true) / len(n_true)
        acc = sum(tp.values()) / n
        pe = sum(n_true[k] * n_pred.get(k, 0) for k in n_true) / n / n
        assert last[f"test_bal_acc_unseen_offdiag@{name}"] == pytest.approx(bal, rel=1e-12)
        assert last[f"test_acc_unseen_offdiag@{name}"] == pytest.approx(acc, abs=1e-6)
        assert last[f"test_kappa_unseen_offdiag@{name}"] == pytest.approx((acc - pe) / (1 - pe), rel=1e-12)
        assert last[f"test_majority_baseline_unseen_offdiag@{name}"] == pytest.approx(max(n_true.values()) / n)
        assert last[f"test_pred_hist_unseen_offdiag@{name}"] == {str(k): v for k, v in sorted(n_pred.items()) if v}
        big = sorted((tp[k] / n_true[k], k, n_true[k]) for k in n_true if n_true[k] >= 5)
        got = tuple(last[f"test_{k}_unseen_offdiag@{name}"] for k in ("worst_class_recall", "worst_class", "worst_class_n"))
        assert got == (tuple(big[0]) if big else (None, None, None))
    pooled = sum(last[f"test_loss_unseen_offdiag@{t['task']}"] * t["n_test_unseen_offdiag"] for t in rec["tasks"])
    assert last["test_loss_unseen_offdiag"] == pytest.approx(pooled / rec["n_test_unseen_offdiag"], rel=1e-12)
    if "gcd" in names:
        assert last["test_majority_baseline_unseen_offdiag@gcd"] > 0.2  # skewed, as in the review
