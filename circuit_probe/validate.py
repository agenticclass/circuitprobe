#!/usr/bin/env python3
"""
Validate CircuitProbe predictions against ground truth sweep data.

Runs all scoring methods and reports accuracy metrics for the paper.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit_probe import ActivationCollector, CircuitScorer, ContrastiveCollector, save_stats
from calibration_data import get_calibration_set, get_contrastive_sets


# Ground truth circuits from sweep data
GROUND_TRUTH = {
    # Model: [(start, end, description)]
    "phi-4": [
        (6, 11, "primary circuit, +7.6% math, 100% reasoning"),
        (33, 39, "secondary circuit, 100% reasoning, -0.6% math"),
    ],
    "devstral-24b": [
        (12, 15, "layers 12-14, +54% logical deduction (50-sample)"),
    ],
    "qwen2.5-32b": [
        (7, 10, "layers 7-9, +23% reasoning (published by circuit finder)"),
    ],
}


def load_model(model_name, dtype_str="float16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    print(f"Loading {model_name} ({dtype_str})...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
    model.eval()

    print(f"  Loaded in {time.time()-t0:.1f}s, {model.config.num_hidden_layers} layers")
    return model, tokenizer


def run_all_methods(model, tokenizer, max_length=128, n_examples=50):
    """Run all scoring methods and return results."""
    methods = ["anomaly", "boundary", "composite", "contrastive"]
    results = {}

    # Collect stats for non-contrastive methods
    print("\nCollecting activation stats (mixed calibration)...")
    t0 = time.time()
    texts, labels = get_calibration_set(n_reasoning=n_examples//2, n_general=n_examples - n_examples//2)
    collector = ActivationCollector(model, tokenizer)
    stats = collector.collect_stats(texts, max_length=max_length)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Score with non-contrastive methods
    for method in ["anomaly", "boundary", "composite", "stability"]:
        scorer = CircuitScorer(stats)
        scores = scorer.score_blocks(block_sizes=(3, 4, 5), method=method)
        results[method] = scores

    # Contrastive method
    print("\nCollecting contrastive stats...")
    t0 = time.time()
    reasoning_texts, general_texts = get_contrastive_sets()
    cc = ContrastiveCollector(model, tokenizer)
    r_stats, g_stats, contrast = cc.collect_contrastive(reasoning_texts, general_texts, max_length=max_length)
    print(f"  Done in {time.time()-t0:.1f}s")

    scorer = CircuitScorer(r_stats, contrast_scores=contrast)
    scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="contrastive")
    results["contrastive"] = scores

    return results, stats


def evaluate_against_ground_truth(results, gt_circuits, n_layers):
    """Evaluate all methods against ground truth circuits."""
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS (ground truth: {[f'[{s}-{e})' for s,e,_ in gt_circuits]})")
    print(f"{'='*80}")

    primary_gt = gt_circuits[0]  # Primary circuit
    gt_s, gt_e, gt_desc = primary_gt

    print(f"\nPrimary ground truth: [{gt_s}-{gt_e}) — {gt_desc}")
    print(f"\n{'Method':<15} {'Top-1':<15} {'Exact?':<8} {'Near?':<8} {'GT Rank':<10} {'IoU':<8}")
    print("-" * 70)

    for method, scores in results.items():
        top1 = scores[0]

        # Exact match
        exact = top1.start == gt_s and top1.end == gt_e

        # Near match (within 2 layers)
        near = abs(top1.start - gt_s) <= 2 and abs(top1.end - gt_e) <= 2

        # Find GT rank
        gt_rank = "N/A"
        for i, s in enumerate(scores):
            if s.start == gt_s and s.end == gt_e:
                gt_rank = str(i + 1)
                break

        # IoU of top-1 with GT
        overlap_start = max(top1.start, gt_s)
        overlap_end = min(top1.end, gt_e)
        overlap = max(0, overlap_end - overlap_start)
        union = (top1.end - top1.start) + (gt_e - gt_s) - overlap
        iou = overlap / union if union > 0 else 0

        exact_str = "YES" if exact else "no"
        near_str = "YES" if near else "no"

        print(f"{method:<15} [{top1.start}-{top1.end}){'':<7} {exact_str:<8} {near_str:<8} {gt_rank:<10} {iou:<8.2f}")

    # Also check top-5 for each method
    print(f"\nTop-5 predictions per method:")
    for method, scores in results.items():
        preds = ", ".join(f"[{s.start}-{s.end})" for s in scores[:5])
        print(f"  {method:<15} {preds}")

    # Check if any secondary circuits are found
    if len(gt_circuits) > 1:
        print(f"\nSecondary circuits:")
        for gt_s2, gt_e2, gt_desc2 in gt_circuits[1:]:
            print(f"  [{gt_s2}-{gt_e2}) — {gt_desc2}")
            for method, scores in results.items():
                for i, s in enumerate(scores[:20]):
                    if abs(s.start - gt_s2) <= 2 and abs(s.end - gt_e2) <= 2:
                        print(f"    {method}: near match at rank {i+1} [{s.start}-{s.end})")
                        break


def main():
    parser = argparse.ArgumentParser(description="Validate CircuitProbe")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--ground-truth", type=str, required=True,
                        help="Comma-separated ground truth circuits: 'start1:end1,start2:end2'")
    parser.add_argument("--output-dir", default="../results")
    args = parser.parse_args()

    # Parse ground truth
    gt_circuits = []
    for circuit_str in args.ground_truth.split(","):
        s, e = map(int, circuit_str.split(":"))
        gt_circuits.append((s, e, ""))

    # Load model
    model, tokenizer = load_model(args.model, args.dtype)
    n_layers = model.config.num_hidden_layers

    # Run all methods
    results, stats = run_all_methods(model, tokenizer, args.max_length, args.n_examples)

    # Evaluate
    evaluate_against_ground_truth(results, gt_circuits, n_layers)

    # Save stats
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.replace("/", "_")
    save_stats(stats, os.path.join(args.output_dir, f"stats_{model_short}.json"))

    # Save all scores
    all_scores = {}
    for method, scores in results.items():
        all_scores[method] = [
            {"start": s.start, "end": s.end, "score": float(s.score),
             "metrics": {k: float(v) for k, v in s.metrics.items()}}
            for s in scores[:30]
        ]
    with open(os.path.join(args.output_dir, f"validation_{model_short}.json"), "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
