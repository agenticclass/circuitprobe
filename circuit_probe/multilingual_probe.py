#!/usr/bin/env python3
"""
Multilingual CircuitProbe analysis.

Tests whether reasoning circuit locations shift when the model processes
inputs in different languages.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit_probe import ActivationCollector, CircuitScorer, save_stats
from calibration_data import get_multilingual_sets


def run_multilingual(model_name, dtype_str="float16", max_length=128, output_dir="../results"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    print(f"Loading {model_name} ({dtype_str})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers")

    lang_sets = get_multilingual_sets()
    results = {}

    for lang, texts in lang_sets.items():
        print(f"\n{'='*50}")
        print(f"Language: {lang} ({len(texts)} examples)")
        print(f"{'='*50}")

        collector = ActivationCollector(model, tokenizer)
        t0 = time.time()
        stats = collector.collect_stats(texts, max_length=max_length)
        elapsed = time.time() - t0
        print(f"Collection: {elapsed:.1f}s")

        # Save per-language stats
        model_short = model_name.replace("/", "_")
        save_stats(stats, os.path.join(output_dir, f"stats_{model_short}_{lang}.json"))

        # Score with stability and anomaly
        scorer = CircuitScorer(stats)

        lang_results = {}
        for method in ["stability", "anomaly"]:
            scores = scorer.score_blocks(block_sizes=(3, 4, 5), method=method)
            top5 = [(s.start, s.end, float(s.score)) for s in scores[:5]]
            lang_results[method] = top5
            print(f"\n  {method.upper()} top 3:")
            for i, (s, e, sc) in enumerate(top5[:3]):
                print(f"    {i+1}. [{s}-{e}) score={sc:.4f}")

        results[lang] = lang_results

    # Comparison table
    print(f"\n\n{'='*70}")
    print("MULTILINGUAL COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Language':<12} {'Stability Top-1':<18} {'Stability Top-3':<35} {'Anomaly Top-1':<18}")
    print("-" * 85)

    for lang, lr in results.items():
        stab_top1 = f"[{lr['stability'][0][0]}-{lr['stability'][0][1]})"
        stab_top3 = ", ".join(f"[{s}-{e})" for s, e, _ in lr['stability'][:3])
        anom_top1 = f"[{lr['anomaly'][0][0]}-{lr['anomaly'][0][1]})"
        print(f"{lang:<12} {stab_top1:<18} {stab_top3:<35} {anom_top1:<18}")

    # Measure stability of predictions across languages
    stab_positions = [lr['stability'][0][0] for lr in results.values()]
    anom_positions = [lr['anomaly'][0][0] for lr in results.values()]

    print(f"\nStability circuit start positions: {stab_positions}")
    print(f"  Range: {max(stab_positions) - min(stab_positions)} layers")
    print(f"  Std: {np.std(stab_positions):.2f}")

    print(f"\nAnomaly circuit start positions: {anom_positions}")
    print(f"  Range: {max(anom_positions) - min(anom_positions)} layers")
    print(f"  Std: {np.std(anom_positions):.2f}")

    shift = max(stab_positions) - min(stab_positions)
    if shift <= 2:
        print(f"\nConclusion: Stability circuits are STABLE across languages (shift <= 2 layers)")
    else:
        print(f"\nConclusion: Stability circuits SHIFT across languages (shift = {shift} layers)")

    # Save full results
    with open(os.path.join(output_dir, f"multilingual_{model_short}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", default="../results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_multilingual(args.model, args.dtype, args.max_length, args.output_dir)
