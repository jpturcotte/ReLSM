"""CPU tests for tools/grok_lab.py, the standalone grokking harness (PR 2 of the fix plan).

T1 split, T2 decay groups and decoupled decay, T3 harness overfit smoke (not #256, which tests
train.py), T5 measured update norm, T6 same seed -> same metrics. Threads are pinned to 1.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import grok_lab as gl  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def args_for(tmp_path, *extra):
    return gl.parse_args(["--out", str(tmp_path), "--threads", "1", "--device", "cpu", *extra])


# T1 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("pairs", ["ordered", "paired"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t1_split_covers_table_disjoint_and_tags_transposes(pairs, seed):
    p = 97
    train, test, seen = gl.make_split(p, 0.5, seed, pairs)
    tr, te = set(train.tolist()), set(test.tolist())
    assert tr.isdisjoint(te)
    assert tr | te == set(range(p * p))
    assert abs(len(tr) - 4704) <= 1 and (pairs == "paired" or len(tr) == 4704)
    for i, s in zip(test.tolist(), seen.tolist()):  # brute-force transpose tag
        a, b = divmod(i, p)
        assert s == (b * p + a in tr)
    if pairs == "paired":
        assert all((i % p) * p + i // p in tr for i in tr)
        assert not seen.any()
    else:
        assert 0.4 < seen.float().mean().item() < 0.6  # about half the test pairs leak via the transpose
    _, y = gl.encode(train, p)
    assert set(y.tolist()) == set(range(p))  # every residue is a train answer


def test_t1_encode_and_split_seeded():
    x, y = gl.encode(torch.tensor([0, 5 * 97 + 96]), 97)
    assert x.tolist() == [[0, 97, 0, 98], [5, 97, 96, 98]] and y.tolist() == [0, 4]
    a, b = gl.make_split(97, 0.5, 7)[0], gl.make_split(97, 0.5, 7)[0]
    assert torch.equal(a, b) and not torch.equal(a, gl.make_split(97, 0.5, 8)[0])


def test_t1_missing_residue_refuses_to_start(tmp_path):
    with pytest.raises(ValueError, match="residues never seen"):
        gl.run(args_for(tmp_path, "--frac", "0.002", "--updates", "1"))


# T2 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("tie", [True, False])
def test_t2_scope_all_decays_every_parameter(tmp_path, tie):
    args = args_for(tmp_path, *([] if tie else ["--untie"]))
    model = gl.build_model(args)
    opt = gl.build_optimizer(model, args)
    n_model = sum(q.numel() for q in model.parameters())
    assert len(opt.param_groups) == 1
    assert sum(q.numel() for q in opt.param_groups[0]["params"]) == n_model
    assert opt.param_groups[0]["weight_decay"] == 1.0
    assert opt.defaults["betas"] == (0.9, 0.98) and opt.defaults["eps"] == 1e-8
    assert (model.lm_head.weight is model.tok_emb.weight) == tie


@pytest.mark.parametrize("tie", [True, False])
def test_t2_repo_and_embeddings_scopes(tmp_path, tie):
    args = args_for(tmp_path, "--wd_scope", "repo", *([] if tie else ["--untie"]))
    model = gl.build_model(args)
    norms = {n for n, _ in model.named_parameters() if n.endswith("norm.weight")}
    assert len(norms) == 5  # 2 per block + the final norm
    exempt = {n for n, _ in model.named_parameters() if not gl.decays(n, "repo")}
    # a tied table is listed once, as tok_emb.weight
    assert exempt == norms | ({"tok_emb.weight"} if tie else {"tok_emb.weight", "lm_head.weight"})
    assert {n for n, _ in model.named_parameters() if not gl.decays(n, "embeddings")} == norms
    no_final = gl.build_model(args_for(tmp_path, "--no_final_norm"))
    assert "norm.weight" not in dict(no_final.named_parameters())


def test_t2_zero_gradient_update_is_pure_decoupled_decay(tmp_path):
    args = args_for(tmp_path, "--lr", "1e-3", "--wd", "1.0")
    model = gl.build_model(args)
    opt = gl.build_optimizer(model, args)
    before = [q.detach().clone() for q in model.parameters()]
    for q in model.parameters():
        q.grad = torch.zeros_like(q)
    opt.step()
    for q, b in zip(model.parameters(), before):
        assert torch.equal(q.detach(), b.mul(1 - 1e-3 * 1.0))


# T3 ---------------------------------------------------------------------------------------

def test_t3_harness_overfits_512_pairs(tmp_path):
    """Harness smoke test, not #256 (that one tests data.py and train.py, in PR 6)."""
    t0 = time.perf_counter()
    last = gl.run(args_for(tmp_path, "--frac", str(512 / 97 ** 2), "--wd", "0", "--updates", "500",
                           "--eval_every", "500", "--ckpt_every", "0", "--allow_missing_residues",
                           "--threads", str(min(8, os.cpu_count() or 1))))
    elapsed = time.perf_counter() - t0
    rec = json.loads((tmp_path / "run_record.json").read_text())
    assert rec["n_train"] == 512 and rec["status"] == "finished"
    assert last["train_loss"] < 0.1 and last["train_acc"] >= 0.99, last
    assert elapsed < 60, f"{elapsed:.1f} s"


# T5 ---------------------------------------------------------------------------------------

def test_t5_logged_update_norm_is_measured(tmp_path):
    args = args_for(tmp_path, "--p", "23")
    train, _, _ = gl.make_split(23, 0.5, 0)
    x, y = gl.encode(train, 23)
    torch.manual_seed(0)
    model = gl.build_model(args)
    opt = gl.build_optimizer(model, args)
    for _ in range(5):
        gl.train_step(model, opt, x, y, args, 1e-3)
    before = [q.detach().clone() for q in model.parameters()]
    _, st = gl.train_step(model, opt, x, y, args, 1e-3, measure=True)
    dnorm, pnorm = st["update_norm"], st["param_norm_before"]
    assert st["grad_norm"] > 0 and st["top_param"] in dict(model.named_parameters())
    true = math.sqrt(sum(float((q.detach() - b).double().pow(2).sum()) for q, b in zip(model.parameters(), before)))
    assert true > 0 and abs(dnorm - true) <= 0.01 * true
    assert abs(pnorm - math.sqrt(sum(float(b.double().pow(2).sum()) for b in before))) <= 1e-6 * pnorm


def test_margin_loss_matches_ce_and_keeps_the_target_gradient(tmp_path):
    z = torch.tensor([[0.0, 1.0, 2.0], [30.0, 0.0, 0.0]])
    y = torch.tensor([2, 0])
    ce, margin = args_for(tmp_path), args_for(tmp_path, "--loss", "margin")
    assert torch.allclose(gl.objective(z, y, margin), gl.objective(z, y, ce), rtol=1e-5)
    assert gl.margins(z, y)[1].item() == pytest.approx(30 - math.log(2), abs=1e-5)
    grads = {}
    for name, a in (("ce", ce), ("margin", margin)):
        leaf = z.clone().requires_grad_()
        (grads[name],) = torch.autograd.grad(gl.objective(leaf, y, a), leaf)
    assert grads["ce"][1, 0].item() == 0.0  # past the fp32 cliff
    assert grads["margin"][1, 0].item() < 0.0


def test_spike_counter_survives_an_exact_zero_loss():
    s = gl.Spikes(10.0, 1e-3)
    losses = [4.6, 1.0, 0.0, 1e-6, 0.0, 0.02, 0.5, 0.005, 3e-3, 0.2, 0.009, 0.0105, 0.002, 0.011]
    events = [s.feed(i, v) for i, v in enumerate(losses)]
    # 0.005 keeps the first spike open; the tail 0.009 -> 0.0105 is still the second spike
    assert [e[0] for e in events if e] == [5, 9, 13] and s.count == 3


def test_clip_bounds_the_gradient_and_logs_the_raw_norm(tmp_path):
    args = args_for(tmp_path, "--p", "23", "--clip", "1e-3")
    x, y = gl.encode(gl.make_split(23, 0.5, 0)[0], 23)
    model = gl.build_model(args)
    opt = gl.build_optimizer(model, args)
    _, st = gl.train_step(model, opt, x, y, args, 1e-3, measure=True)
    assert st["grad_norm"] > 1e-2  # raw norm, before clipping
    assert gl.global_norm([q.grad for q in model.parameters()]) <= 1e-3 * (1 + 1e-5)
    gl.train_step(model, opt, x, y, args, 1e-3)  # the unmeasured path most updates take
    assert gl.global_norm([q.grad for q in model.parameters()]) <= 1e-3 * (1 + 1e-5)


def test_batches_cover_each_example_once_per_epoch():
    b = gl.Batches(10, 3, seed=0)
    epoch = [b.next().tolist() for _ in range(3)]  # 9 of 10, the last partial batch dropped
    assert all(len(e) == 3 for e in epoch) and len({i for e in epoch for i in e}) == 9
    assert b.next().tolist() + b.next().tolist() + b.next().tolist() != sum(epoch, [])  # reshuffled
    exact = gl.Batches(9, 3, seed=0)  # n divisible by the batch: no example dropped
    for _ in range(4):  # four epochs of three batches
        first = [exact.next().tolist() for _ in range(3)]
        assert sorted(sum(first, [])) == list(range(9))
    with pytest.raises(ValueError):
        gl.Batches(10, 11, seed=0)


def _loss_and_grad(z, a):
    leaf = z.clone().requires_grad_()
    loss = gl.loss_of(leaf, torch.tensor([0] * len(z)), a)
    (g,) = torch.autograd.grad(loss, leaf)
    return loss.item(), g


def test_ce_rounded_puts_the_bf16_cliff_in_the_loss_and_the_target_gradient(tmp_path):
    a = args_for(tmp_path, "--loss", "ce_rounded")
    # [m, 0, 0] has margin m - ln 2 and sum s = 1 + 2 e^-m. bf16 spacing at 1 is 2^-7.
    loss, g = _loss_and_grad(torch.tensor([[6.0, 0.0, 0.0]]), a)  # margin 5.3: s rounds up to 1 + 2^-7
    s = 1 + 2.0 ** -7
    assert loss == pytest.approx(math.log(s), rel=1e-4)  # fp32: 1 ulp of 1 is 1e-5 of the result
    assert g[0, 0].item() == pytest.approx(1 / s - 1, rel=1e-4)  # what CPU's bf16 kernel gives (-0.00775)
    loss, g = _loss_and_grad(torch.tensor([[9.0, 0.0, 0.0]]), a)  # margin 8.3: s rounds to 1
    assert loss == 0.0 and g[0, 0].item() == 0.0
    assert g[0, 1].item() == pytest.approx(math.exp(-9), rel=1e-5)  # the other logits keep e_j / s'
    loss, g = _loss_and_grad(torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), a)  # far from the cliff, and a miss
    ce, g_ce = _loss_and_grad(torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), args_for(tmp_path))
    assert loss == pytest.approx(ce, rel=1e-2) and torch.allclose(g, g_ce, rtol=1e-2, atol=1e-4)


def test_ce_rounded_fp16_and_autocast(tmp_path):
    a16 = args_for(tmp_path, "--loss", "ce_rounded", "--round_dtype", "float16")
    # margin 6.3 < 7.6: s = 1.00182 rounds up two fp16 ulps to 1 + 2^-9
    assert _loss_and_grad(torch.tensor([[7.0, 0.0, 0.0]]), a16)[1][0, 0].item() == pytest.approx(1 / (1 + 2.0 ** -9) - 1, rel=1e-4)
    with pytest.raises(SystemExit):
        args_for(tmp_path, "--round_dtype", "float16")  # without --loss ce_rounded
    assert _loss_and_grad(torch.tensor([[9.0, 0.0, 0.0]]), a16)[1][0, 0].item() == 0.0  # margin 8.3 > 7.6
    # train.py's regime: bf16 logits, cross-entropy under autocast (run in fp32): no cliff at 8.3
    z = torch.tensor([[9.0, 0.0, 0.0]]).bfloat16()
    assert _loss_and_grad(z, args_for(tmp_path, "--bf16", "autocast"))[1][0, 0].item() < 0
    # the rounded loss keeps its cliff when it runs inside autocast on bf16 logits
    g = _loss_and_grad(z, args_for(tmp_path, "--bf16", "autocast", "--loss", "ce_rounded"))[1]
    assert g[0, 0].item() == 0.0 and g.dtype == torch.bfloat16


@pytest.mark.parametrize("extra", [["--bf16", "autocast"], ["--loss", "ce_rounded"],
                                   ["--bf16", "autocast", "--loss", "ce_rounded"], ["--batch", "64", "--clip", "1"]])
def test_train_py_like_flags_run(tmp_path, extra):
    last = gl.run(args_for(tmp_path, "--p", "23", "--updates", "20", "--eval_every", "10", "--ckpt_every", "0", *extra))
    assert last["update"] == 20 and math.isfinite(last["train_loss"])
    rec = json.loads((tmp_path / "run_record.json").read_text())
    assert rec["examples_per_update"] == (64 if "--batch" in extra else rec["n_train"])


def test_summary_reads_the_trace_not_the_last_line():
    hist = [(100, 0.1), (200, 0.995), (900, 0.999), (950, 0.04), (1000, 0.999)]  # a spike at 950
    s = gl.summarize(hist, 1000)
    assert s["first_update_unseen_ge_0.99"] == 200
    assert s["frac_last10pct_evals_unseen_ge_0.99"] == pytest.approx(1 / 2)
    assert gl.summarize([(100, None)], 1000)["first_update_unseen_ge_0.99"] is None


# T6 ---------------------------------------------------------------------------------------

def test_t6_same_seed_same_metrics(tmp_path):
    runs = []
    for k in range(2):
        out = tmp_path / str(k)
        gl.run(args_for(out, "--p", "23", "--updates", "30", "--eval_every", "10", "--ckpt_every", "20"))
        lines = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]
        for rec in lines:
            rec.pop("train_s"), rec.pop("elapsed_s")
        runs.append(lines)
    assert len(runs[0]) == 4 and runs[0] == runs[1]
    assert runs[0][-1]["update"] == 30 and runs[0][-1]["rel_update"] > 0
    rec = json.loads((tmp_path / "0" / "run_record.json").read_text())
    assert len(runs[0][1]["train_obj_trace"]) == 10
    for key in ("args", "commit", "dirty", "torch", "device", "seed", "threads", "examples_per_update", "updates",
                "model_config", "first_update_unseen_ge_0.99", "frac_last10pct_evals_unseen_ge_0.99"):
        assert key in rec
    assert rec["threads"] == 1 and rec["examples_per_update"] == rec["n_train"] and rec["updates"] == 30
    assert rec["commit"] is None or len(rec["commit"]) == 40
    assert (tmp_path / "0" / "ckpt_000020.pt").exists() and (tmp_path / "0" / "final.pt").exists()
