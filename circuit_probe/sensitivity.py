#!/usr/bin/env python3
"""
Calibration set sensitivity analysis (Appendix B).

Tests how CircuitProbe predictions change with:
1. Different calibration set sizes (10, 25, 50, 100)
2. Different calibration compositions (reasoning-only, general-only, mixed)
3. Different random subsets of the same size
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit_probe import ActivationCollector, CircuitScorer, save_stats
from calibration_data import REASONING_EXAMPLES, GENERAL_EXAMPLES


def run_sensitivity(model_name, dtype_str="float32", max_length=128, output_dir="../results"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers")

    all_texts = REASONING_EXAMPLES + GENERAL_EXAMPLES
    results = {}

    # --- Experiment 1: Calibration set size ---
    print("\n=== Experiment 1: Calibration Set Size ===")
    size_results = {}
    for n in [10, 20, 30, 50]:
        if n > len(all_texts):
            continue
        texts = all_texts[:n]
        print(f"\n  n={n}:")

        collector = ActivationCollector(model, tokenizer)
        t0 = time.time()
        stats = collector.collect_stats(texts, max_length=max_length)
        elapsed = time.time() - t0

        scorer = CircuitScorer(stats)
        stab_scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="stability")
        anom_scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="anomaly")

        stab_top3 = [(s.start, s.end) for s in stab_scores[:3]]
        anom_top3 = [(s.start, s.end) for s in anom_scores[:3]]
        print(f"    Stability top-3: {stab_top3} ({elapsed:.1f}s)")
        print(f"    Anomaly top-3: {anom_top3}")

        size_results[n] = {
            "stability_top3": stab_top3,
            "anomaly_top3": anom_top3,
            "time": elapsed,
        }
    results["size"] = size_results

    # --- Experiment 2: Calibration composition ---
    print("\n=== Experiment 2: Calibration Composition ===")
    comp_results = {}
    compositions = {
        "reasoning_only": REASONING_EXAMPLES[:25],
        "general_only": GENERAL_EXAMPLES[:25],
        "mixed_50_50": REASONING_EXAMPLES[:12] + GENERAL_EXAMPLES[:13],
        "mixed_75_25": REASONING_EXAMPLES[:19] + GENERAL_EXAMPLES[:6],
    }

    for comp_name, texts in compositions.items():
        print(f"\n  {comp_name} ({len(texts)} examples):")

        collector = ActivationCollector(model, tokenizer)
        stats = collector.collect_stats(texts, max_length=max_length)

        scorer = CircuitScorer(stats)
        stab_scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="stability")
        anom_scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="anomaly")

        stab_top3 = [(s.start, s.end) for s in stab_scores[:3]]
        anom_top3 = [(s.start, s.end) for s in anom_scores[:3]]
        print(f"    Stability top-3: {stab_top3}")
        print(f"    Anomaly top-3: {anom_top3}")

        comp_results[comp_name] = {
            "stability_top3": stab_top3,
            "anomaly_top3": anom_top3,
        }
    results["composition"] = comp_results

    # --- Experiment 3: Random subset stability ---
    print("\n=== Experiment 3: Random Subset Stability ===")
    random.seed(42)
    subset_results = []
    n_subsets = 5
    subset_size = min(20, len(all_texts))

    for trial in range(n_subsets):
        subset = random.sample(all_texts, subset_size)

        collector = ActivationCollector(model, tokenizer)
        stats = collector.collect_stats(subset, max_length=max_length)

        scorer = CircuitScorer(stats)
        stab_scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="stability")
        stab_top1 = (stab_scores[0].start, stab_scores[0].end)
        subset_results.append(stab_top1)
        print(f"  Trial {trial+1}: Stability top-1 = [{stab_top1[0]}-{stab_top1[1]})")

    results["random_subsets"] = subset_results

    # Measure agreement
    starts = [r[0] for r in subset_results]
    print(f"\n  Top-1 start positions: {starts}")
    print(f"  Range: {max(starts) - min(starts)} layers")
    print(f"  Std: {np.std(starts):.2f}")
    if max(starts) - min(starts) <= 3:
        print(f"  Conclusion: STABLE across random subsets")
    else:
        print(f"  Conclusion: UNSTABLE — predictions vary with subset selection")

    # Save
    model_short = model_name.replace("/", "_")
    with open(os.path.join(output_dir, f"sensitivity_{model_short}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to {output_dir}/sensitivity_{model_short}.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output-dir", default="../results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_sensitivity(args.model, args.dtype, output_dir=args.output_dir)
