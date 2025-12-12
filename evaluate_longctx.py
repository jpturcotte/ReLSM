"""Needle-in-haystack retrieval evaluation and CLI wrapper."""

import argparse
from pathlib import Path
from typing import Dict, Sequence

import torch

from utils import (
    NeedleInHaystackGenerator,
    area_under_depth_curve,
    gather_metadata,
    get_eval_generation_kwargs,
    load_model_and_tokenizer,
    prepare_tokenizer,
    save_json,
    seed_all,
    select_device,
)


@torch.no_grad()
def run_longctx_eval(
    model,
    tokenizer,
    device: torch.device,
    ctx_lengths: Sequence[int],
    depths: Sequence[float],
    samples_per_depth: int,
    seed: int,
    max_new_tokens: int = 10,
) -> Dict:
    """Run long-context needle retrieval and return structured metrics."""

    seed_all(seed)
    results = []
    gen_kwargs = get_eval_generation_kwargs(
        tokenizer=tokenizer, max_new_tokens=max_new_tokens
    )

    for ctx_len in ctx_lengths:
        if hasattr(model, "config") and ctx_len > getattr(
            model.config, "max_seq_len", ctx_len
        ):
            continue
        generator = NeedleInHaystackGenerator(tokenizer, context_length=ctx_len)
        by_depth = {}
        correct = 0
        total = 0
        for depth in depths:
            depth_correct = 0
            for _ in range(samples_per_depth):
                example = generator.generate(needle_depth=depth)
                input_ids = example["input_ids"].to(device).unsqueeze(0)
                outputs = model.generate(input_ids, **gen_kwargs)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Answer:" in decoded:
                    tail = decoded.split("Answer:")[-1].strip().split()
                else:
                    tail = decoded.split()
                predicted = tail[0] if tail else ""
                if predicted == example["answer"]:
                    depth_correct += 1
                    correct += 1
                total += 1
            by_depth[depth] = depth_correct / max(samples_per_depth, 1)
        retrieval_acc = correct / max(total, 1)
        results.append(
            {
                "context_length": ctx_len,
                "metrics": {
                    "retrieval_accuracy": retrieval_acc,
                    "by_depth": by_depth,
                    "area_under_depth_curve": area_under_depth_curve(by_depth),
                },
                "config": {
                    "depths": list(depths),
                    "samples_per_depth": samples_per_depth,
                    "max_new_tokens": max_new_tokens,
                },
            }
        )

    auc = 0.0
    if results:
        auc = sum(r["metrics"]["area_under_depth_curve"] for r in results) / len(results)

    return {"per_context": results, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Needle-in-haystack long-context evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ctx", type=int, nargs="*", default=[4096, 16384], help="Context lengths to test")
    parser.add_argument("--samples", type=int, default=20, help="Examples per depth")
    parser.add_argument("--depths", type=float, nargs="*", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = select_device(args.device)

    tokenizer = prepare_tokenizer(args.tokenizer)

    print(f"Loading model from {args.checkpoint} on {device}...")
    model, config, _ = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)
    print(f"Variant: {config.variant}, max_seq_len: {config.max_seq_len}")

    results = run_longctx_eval(
        model,
        tokenizer,
        device,
        ctx_lengths=args.ctx,
        depths=args.depths,
        samples_per_depth=args.samples,
        seed=args.seed,
    )

    metadata = gather_metadata(
        checkpoint=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=device,
        model_config=config,
        generation_kwargs=get_eval_generation_kwargs(tokenizer=tokenizer, max_new_tokens=10),
    )
    payload = {"metadata": metadata, "results": {"longctx": results}}
    if args.output:
        save_json(payload, Path(args.output))
        print(f"Saved results to {args.output}")

    for ctx in results["per_context"]:
        print(
            f"ctx={ctx['context_length']}: acc={ctx['metrics']['retrieval_accuracy']*100:.1f}% (AUC={ctx['metrics']['area_under_depth_curve']*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
