#!/usr/bin/env python3
"""Generate per-layer activation stat plots for the paper."""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuit_probe import load_stats


MODELS = {
    "Phi-4 (40L)": {
        "path": "../results/stats_microsoft_phi-4.json",
        "gt_primary": (6, 11),
        "gt_secondary": (33, 39),
    },
    "Qwen2.5-7B (28L)": {
        "path": "../results/stats_Qwen_Qwen2.5-7B.json",
        "gt_primary": (6, 11),
        "gt_secondary": None,
    },
    "Mistral-7B (32L)": {
        "path": "../results/stats_mistralai_Mistral-7B-v0.3.json",
        "gt_primary": None,
        "gt_secondary": None,
    },
    "TinyLlama-1.1B (22L)": {
        "path": "../results/stats_TinyLlama-1.1B.json",
        "gt_primary": None,
        "gt_secondary": None,
    },
}


def plot_repr_change_and_derivative(output_path="../results/figures"):
    """Plot representation change and its derivative across layers for all models."""
    os.makedirs(output_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Layer Representation Change Across Architectures", fontsize=14, fontweight='bold')

    for idx, (name, info) in enumerate(MODELS.items()):
        if not os.path.exists(info["path"]):
            continue

        stats = load_stats(info["path"])
        n = len(stats)
        layers = list(range(n))

        changes = [s.representation_change if not np.isnan(s.representation_change) else 0 for s in stats]

        # Derivative (absolute)
        deltas = [abs(changes[i] - changes[i-1]) if i > 0 else 0 for i in range(n)]

        ax = axes[idx // 2][idx % 2]

        # Plot representation change
        ax.bar(layers, changes, alpha=0.6, color='steelblue', label='Repr. Change')

        # Highlight ground truth circuits
        if info["gt_primary"]:
            s, e = info["gt_primary"]
            ax.axvspan(s, e, alpha=0.2, color='green', label=f'Primary GT [{s}-{e})')
        if info["gt_secondary"]:
            s, e = info["gt_secondary"]
            ax.axvspan(s, e, alpha=0.2, color='orange', label=f'Secondary GT [{s}-{e})')

        # Overlay derivative on secondary axis
        ax2 = ax.twinx()
        ax2.plot(layers[1:], deltas[1:], 'r-', alpha=0.7, linewidth=1.5, label='|Derivative|')
        ax2.set_ylabel('|Δ Repr. Change|', color='red', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Repr. Change Magnitude')
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "repr_change_all_models.png"), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}/repr_change_all_models.png")


def plot_stability_zone(output_path="../results/figures"):
    """Plot the stability zone detection metrics."""
    os.makedirs(output_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stability Zone Detection: Derivative of Representation Change", fontsize=14, fontweight='bold')

    for idx, (name, info) in enumerate(MODELS.items()):
        if not os.path.exists(info["path"]):
            continue

        stats = load_stats(info["path"])
        n = len(stats)

        changes = [s.representation_change if not np.isnan(s.representation_change) else 0 for s in stats]
        deltas = [abs(changes[i] - changes[i-1]) if i > 0 else 0 for i in range(n)]

        # Variance growth rate
        variances = [s.cross_example_variance if not np.isnan(s.cross_example_variance) else 0 for s in stats]
        var_growth = [variances[i] / max(variances[i-1], 1e-10) if i > 0 else 1 for i in range(n)]

        ax = axes[idx // 2][idx % 2]

        # Derivative (lower = more stable)
        bars = ax.bar(range(n), deltas, alpha=0.7, color='coral', label='|Δ Repr. Change|')

        # Color stability zone bars green
        # Find the stability zone (lowest derivative region of 5+ consecutive layers)
        window = 5
        min_sum = float('inf')
        min_start = 0
        for i in range(2, n - window):
            s = sum(deltas[i:i+window])
            if s < min_sum:
                min_sum = s
                min_start = i
        for i in range(min_start, min_start + window):
            if i < len(bars):
                bars[i].set_color('green')
                bars[i].set_alpha(0.9)

        # Highlight ground truth
        if info["gt_primary"]:
            s, e = info["gt_primary"]
            ax.axvspan(s, e, alpha=0.15, color='blue', label=f'GT [{s}-{e})')

        ax.set_title(f"{name} — Stability Zone: [{min_start}-{min_start+window})", fontsize=11)
        ax.set_xlabel('Layer')
        ax.set_ylabel('|Δ Repr. Change|')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "stability_zones.png"), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}/stability_zones.png")


def plot_variance_growth(output_path="../results/figures"):
    """Plot variance growth rates to show the stability zone."""
    os.makedirs(output_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Example Variance Growth Rate Across Layers", fontsize=14, fontweight='bold')

    for idx, (name, info) in enumerate(MODELS.items()):
        if not os.path.exists(info["path"]):
            continue

        stats = load_stats(info["path"])
        n = len(stats)
        variances = [s.cross_example_variance if not np.isnan(s.cross_example_variance) else 1e-10 for s in stats]
        var_growth = [variances[i] / max(variances[i-1], 1e-10) if i > 0 else 1 for i in range(n)]

        ax = axes[idx // 2][idx % 2]
        ax.plot(range(n), var_growth, 'b-o', markersize=3, label='Var Growth Rate')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        if info["gt_primary"]:
            s, e = info["gt_primary"]
            ax.axvspan(s, e, alpha=0.15, color='green', label=f'GT [{s}-{e})')

        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Variance Growth Rate (Var[i]/Var[i-1])')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "variance_growth.png"), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}/variance_growth.png")


if __name__ == "__main__":
    plot_repr_change_and_derivative()
    plot_stability_zone()
    plot_variance_growth()
    print("\nAll figures generated.")
