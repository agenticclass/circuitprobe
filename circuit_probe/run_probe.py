#!/usr/bin/env python3
"""
Run CircuitProbe on a model and report predicted reasoning circuit locations.

Usage:
    python run_probe.py --model <model_name_or_path> [options]

Examples:
    # Small model on CPU (for development)
    python run_probe.py --model Qwen/Qwen2.5-0.5B

    # Larger model with reduced precision
    python run_probe.py --model Qwen/Qwen2.5-7B --dtype float16

    # Load from local path
    python run_probe.py --model ./models/qwen-7b --n-examples 50
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit_probe import ActivationCollector, CircuitScorer, ContrastiveCollector, save_stats
from calibration_data import get_calibration_set, get_contrastive_sets


def parse_args():
    parser = argparse.ArgumentParser(description="Run CircuitProbe on a transformer model")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Model precision (float16/bfloat16 saves memory)")
    parser.add_argument("--n-examples", type=int, default=50,
                        help="Number of calibration examples (more = better stats, slower)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max sequence length for tokenization")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[3, 4, 5],
                        help="Block sizes to consider")
    parser.add_argument("--method", default="composite",
                        choices=["anomaly", "boundary", "composite", "contrastive"],
                        help="Scoring method")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Ground truth circuit as 'start,end' (e.g., '6,11') for validation")
    parser.add_argument("--output-dir", default="../results",
                        help="Directory to save results")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top predictions to show")
    return parser.parse_args()


def load_model(model_name, dtype_str):
    """Load model and tokenizer with memory-efficient settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]

    print(f"Loading model: {model_name} (dtype={dtype_str})")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use eager attention to get attention weights for entropy computation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except (ValueError, TypeError):
        # Some models don't support attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.eval()

    t1 = time.time()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded in {t1-t0:.1f}s ({param_count:.0f}M parameters)")

    return model, tokenizer


def main():
    args = parse_args()

    # Load model
    model, tokenizer = load_model(args.model, args.dtype)

    # Get calibration data
    n_reasoning = args.n_examples // 2
    n_general = args.n_examples - n_reasoning
    texts, labels = get_calibration_set(n_reasoning=n_reasoning, n_general=n_general)
    print(f"Calibration set: {len(texts)} examples ({n_reasoning} reasoning, {n_general} general)")

    # Collect activation statistics
    print("\nCollecting activation statistics...")
    t0 = time.time()

    contrast_scores = None
    if args.method == "contrastive":
        reasoning_texts, general_texts = get_contrastive_sets()
        cc = ContrastiveCollector(model, tokenizer)
        r_stats, g_stats, contrast_scores = cc.collect_contrastive(
            reasoning_texts, general_texts, max_length=args.max_length
        )
        stats = r_stats  # Use reasoning stats as the primary stats
    else:
        collector = ActivationCollector(model, tokenizer)
        stats = collector.collect_stats(texts, max_length=args.max_length)

    t1 = time.time()
    print(f"Collection took {t1-t0:.1f}s")

    # Score candidate blocks
    print("\nScoring candidate blocks...")
    scorer = CircuitScorer(stats, contrast_scores=contrast_scores)
    scores = scorer.score_blocks(
        block_sizes=tuple(args.block_sizes),
        method=args.method,
    )

    # Print results
    scorer.print_layer_stats()
    scorer.print_rankings(scores, top_k=args.top_k)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.replace("/", "_")
    stats_path = os.path.join(args.output_dir, f"stats_{model_short}.json")
    save_stats(stats, stats_path)

    # Save scores
    import json
    scores_path = os.path.join(args.output_dir, f"scores_{model_short}.json")
    scores_data = [
        {"start": s.start, "end": s.end, "score": float(s.score),
         "metrics": {k: float(v) for k, v in s.metrics.items()}}
        for s in scores[:20]
    ]
    with open(scores_path, "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"\nSaved scores to {scores_path}")

    print(f"\nCircuitProbe prediction: layers [{scores[0].start}-{scores[0].end})")
    print(f"Total time: {time.time()-t0:.1f}s on CPU")

    # Ground truth validation
    if args.ground_truth:
        gt_start, gt_end = map(int, args.ground_truth.split(","))
        print(f"\n{'='*60}")
        print(f"GROUND TRUTH VALIDATION")
        print(f"Ground truth circuit: [{gt_start}-{gt_end})")
        print(f"Top prediction:       [{scores[0].start}-{scores[0].end})")

        # Check if ground truth appears in top-k
        for i, s in enumerate(scores):
            if s.start == gt_start and s.end == gt_end:
                print(f"Ground truth found at rank {i+1}")
                break
            # Near match (within 2 layers)
            if abs(s.start - gt_start) <= 2 and abs(s.end - gt_end) <= 2:
                print(f"Near match at rank {i+1}: [{s.start}-{s.end})")
                break
        else:
            print(f"Ground truth NOT in top {args.top_k}")

        # Check overlap with top predictions
        for i, s in enumerate(scores[:5]):
            overlap_start = max(s.start, gt_start)
            overlap_end = min(s.end, gt_end)
            overlap = max(0, overlap_end - overlap_start)
            gt_size = gt_end - gt_start
            pred_size = s.end - s.start
            iou = overlap / (gt_size + pred_size - overlap) if (gt_size + pred_size - overlap) > 0 else 0
            print(f"  Rank {i+1} [{s.start}-{s.end}): IoU={iou:.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
