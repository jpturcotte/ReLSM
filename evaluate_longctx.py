"""Long-context needle-in-haystack evaluation.

Usage:
    python evaluate_longctx.py --checkpoint path/to/model.pt --ctx 4096 16384

Evaluates retrieval accuracy at multiple context lengths and needle depths.
"""

import argparse
from typing import List

import torch

from utils import NeedleInHaystackGenerator, load_model_and_tokenizer, prepare_tokenizer, select_device


@torch.no_grad()
def evaluate_ctx(model, tokenizer, device, ctx_len: int, n_per_depth: int, depths: List[float]):
    generator = NeedleInHaystackGenerator(tokenizer, context_length=ctx_len)
    correct = 0
    total = 0
    by_depth = {}

    for depth in depths:
        depth_correct = 0
        for _ in range(n_per_depth):
            ex = generator.generate(needle_depth=depth)
            input_ids = ex["input_ids"].to(device).unsqueeze(0)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.1,
                top_k=1,
            )

            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if "Answer:" in decoded:
                tail = decoded.split("Answer:")[-1].strip().split()
            else:
                tail = decoded.split()
            predicted = tail[0] if tail else ""

            if predicted == ex["answer"]:
                depth_correct += 1
                correct += 1
            total += 1
        by_depth[depth] = depth_correct / max(n_per_depth, 1)

    return {"context_length": ctx_len, "accuracy": correct / max(total, 1), "by_depth": by_depth}


def main():
    parser = argparse.ArgumentParser(description="Needle-in-haystack long-context evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ctx", type=int, nargs="*", default=[4096, 16384], help="Context lengths to test")
    parser.add_argument("--samples", type=int, default=20, help="Examples per depth")
    parser.add_argument("--depths", type=float, nargs="*", default=[0.25, 0.5, 0.75, 0.9])

    args = parser.parse_args()

    device = select_device(args.device)

    tokenizer = prepare_tokenizer(args.tokenizer)

    print(f"Loading model from {args.checkpoint} on {device}...")
    model, config, _ = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)
    print(f"Variant: {config.variant}, max_seq_len: {config.max_seq_len}")

    for ctx_len in args.ctx:
        if ctx_len > model.config.max_seq_len:
            print(f"Skipping ctx={ctx_len} (max_seq_len={model.config.max_seq_len})")
            continue
        stats = evaluate_ctx(model, tokenizer, device, ctx_len, args.samples, args.depths)
        print(f"\nContext {ctx_len}: accuracy {stats['accuracy']*100:.1f}%")
        for depth, acc in stats["by_depth"].items():
            print(f"  depth={depth:.2f}: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
