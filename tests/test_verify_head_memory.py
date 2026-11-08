"""Regression test for the end-of-run RAM spike (memspike_2026-11-07/FINDINGS.md): TableScorer.verify_head must not
hold one batch's full-vocab logits while the next batch's full forward runs.

The seam is verify_head itself: a tiny model whose full forward (lm_head in place) registers its output in a weak set,
and which records, at the start of each full forward, how many earlier full outputs are still alive.

As a pytest test, it imports tools/grok_bridge.py from the repo. As a guard job, `--code DIR` picks a frozen copy and
`--out FILE` writes the verdict: python test_verify_head_memory.py --code DIR --out FILE.
"""
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace


def _check(grok_bridge):
    import contextlib

    import torch
    import torch.nn as nn

    alive = weakref.WeakSet()
    seen = []

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 8)
            self.lm_head = nn.Linear(8, 50, bias=False)

        def forward(self, ids):
            full = not isinstance(self.lm_head, nn.Identity)
            if full:
                seen.append(len(alive))  # earlier full outputs still referenced when this forward starts
            out = self.lm_head(self.emb(ids))
            if full:
                alive.add(out)
            return out, None, None

    torch.manual_seed(0)
    batches = []
    for _ in range(4):
        ids = torch.randint(0, 50, (6, 5))
        r = torch.arange(6).repeat_interleave(2)
        j = torch.tensor([1, 3] * 6)
        batches.append((ids, r, j, ids[r, j + 1]))
    scorer = SimpleNamespace(batches=batches, ctx=contextlib.nullcontext())
    res = grok_bridge.TableScorer.verify_head(scorer, Tiny())
    return {"alive_at_each_full_forward": seen, "head_check": res,
            "pass": len(seen) == 4 and max(seen) == 0 and res["head_check_argmax_agree"] == 1.0}


def test_verify_head_frees_each_batch():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import grok_bridge
    out = _check(grok_bridge)
    assert out["pass"], out


if __name__ == "__main__":
    code = Path(sys.argv[sys.argv.index("--code") + 1])
    sys.path.insert(0, str(code))
    from tools import grok_bridge  # noqa: E402

    assert Path(grok_bridge.__file__).resolve().is_relative_to(code.resolve()), grok_bridge.__file__
    out = _check(grok_bridge)
    out["code"] = str(code)
    Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))
