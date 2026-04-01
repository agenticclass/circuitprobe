"""
CircuitProbe: Lightweight probing for reasoning circuits in transformers.

This module collects per-layer activation statistics from a transformer model
and computes a circuit criticality score for each candidate layer block.

Designed to work on CPU with limited RAM by processing layers incrementally
and using model sharding when necessary.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class LayerStats:
    """Activation statistics for a single layer, aggregated across calibration examples."""
    layer_idx: int
    # Mean L2 norm of (output - input) across examples
    representation_change: float = 0.0
    # Cosine similarity between this layer's output and the next layer's output
    forward_similarity: float = 0.0
    # Cosine similarity between this layer's input and output
    self_similarity: float = 0.0
    # Mean attention entropy across all heads (if attention weights available)
    attention_entropy: float = 0.0
    # Ratio of output norm to input norm
    norm_growth: float = 0.0
    # Variance of outputs across different examples
    cross_example_variance: float = 0.0
    # Effective rank of the representation change (how many dimensions are used)
    representation_rank: float = 0.0


@dataclass
class BlockScore:
    """Circuit criticality score for a candidate layer block."""
    start: int
    end: int
    score: float = 0.0
    # Individual metric contributions for analysis
    metrics: dict = field(default_factory=dict)


class ActivationCollector:
    """
    Hooks into a transformer model to collect per-layer activation statistics.

    Works with HuggingFace transformers models. Processes examples one at a time
    to minimize memory usage on CPU.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        # Find the transformer layers
        self.layers = self._find_layers()
        self.n_layers = len(self.layers)
        self.param_count = sum(p.numel() for p in model.parameters())
        self.use_attention = self.param_count < 2e9  # Only for < 2B models
        print(f"Found {self.n_layers} transformer layers ({self.param_count/1e9:.1f}B params, attn={'ON' if self.use_attention else 'OFF'})")

        # Storage for activation hooks
        self._layer_inputs = {}
        self._layer_outputs = {}
        self._attention_weights = {}
        self._hooks = []

    def _find_layers(self):
        """Find the transformer block layers in the model."""
        # Try common attribute names for the layer list
        for attr_path in [
            "model.layers",           # Llama, Mistral, Qwen
            "transformer.h",          # GPT-2, GPT-Neo
            "model.decoder.layers",   # OPT
            "gpt_neox.layers",        # GPT-NeoX, Pythia
        ]:
            obj = self.model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue

        raise ValueError(
            "Could not find transformer layers. "
            "Supported architectures: Llama, Mistral, Qwen, GPT-2, OPT, GPT-NeoX"
        )

    def _register_hooks(self):
        """Register forward hooks on all layers to capture inputs and outputs."""
        self._clear_hooks()

        for idx, layer in enumerate(self.layers):
            # Capture input and output of each layer
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    # input is a tuple, first element is the hidden state
                    if isinstance(input, tuple):
                        self._layer_inputs[layer_idx] = input[0].detach()
                    else:
                        self._layer_inputs[layer_idx] = input.detach()

                    # output can be a tuple (hidden_state, attention_weights, ...)
                    if isinstance(output, tuple):
                        self._layer_outputs[layer_idx] = output[0].detach()
                        # Some models return attention weights as second element
                        if len(output) > 1 and output[1] is not None:
                            self._attention_weights[layer_idx] = output[1].detach()
                    else:
                        self._layer_outputs[layer_idx] = output.detach()

                return hook_fn

            hook = layer.register_forward_hook(make_hook(idx))
            self._hooks.append(hook)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._layer_inputs = {}
        self._layer_outputs = {}
        self._attention_weights = {}

    def collect_stats(self, calibration_texts, max_length=512):
        """
        Run calibration texts through the model and collect per-layer statistics.

        Args:
            calibration_texts: List of strings to use as calibration data.
            max_length: Maximum sequence length for tokenization.

        Returns:
            List of LayerStats, one per layer.
        """
        self._register_hooks()

        n_examples = len(calibration_texts)

        # Accumulators for each metric per layer
        repr_changes = [[] for _ in range(self.n_layers)]
        self_sims = [[] for _ in range(self.n_layers)]
        norm_growths = [[] for _ in range(self.n_layers)]
        attn_entropies = [[] for _ in range(self.n_layers)]

        # For cross-example variance, we store mean output vectors
        output_means = [[] for _ in range(self.n_layers)]

        # For representation rank, we collect change vectors
        repr_change_vectors = [[] for _ in range(self.n_layers)]

        print(f"Processing {n_examples} calibration examples...")

        for i, text in enumerate(calibration_texts):
            if (i + 1) % 10 == 0:
                print(f"  Example {i + 1}/{n_examples}")

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )

            # Forward pass (no gradient needed)
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs, output_attentions=self.use_attention)
                    if self.use_attention and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        for layer_idx, attn in enumerate(outputs.attentions):
                            self._attention_weights[layer_idx] = attn.detach()
                except TypeError:
                    # Some models don't support output_attentions
                    self.model(**inputs)

            # Compute per-layer metrics for this example
            for idx in range(self.n_layers):
                inp = self._layer_inputs.get(idx)
                out = self._layer_outputs.get(idx)

                if inp is None or out is None:
                    continue

                # Flatten to (seq_len, hidden_dim) and take mean across positions
                inp_mean = inp.squeeze(0).mean(dim=0)  # (hidden_dim,)
                out_mean = out.squeeze(0).mean(dim=0)   # (hidden_dim,)

                # 1. Representation change magnitude
                change = (out_mean - inp_mean).float()
                repr_changes[idx].append(change.norm().item())

                # Store change vector for rank computation (keep small subset)
                if i < 50:  # Limit storage for rank computation
                    repr_change_vectors[idx].append(change.numpy())

                # 2. Self-similarity (cosine between layer input and output)
                cos_sim = torch.nn.functional.cosine_similarity(
                    inp_mean.float().unsqueeze(0),
                    out_mean.float().unsqueeze(0)
                ).item()
                self_sims[idx].append(cos_sim)

                # 3. Norm growth ratio
                inp_norm = inp_mean.float().norm().item()
                out_norm = out_mean.float().norm().item()
                if inp_norm > 1e-8:
                    norm_growths[idx].append(out_norm / inp_norm)

                # 4. Attention entropy (if available)
                attn = self._attention_weights.get(idx)
                if attn is not None:
                    # attn shape: (batch, num_heads, seq_len, seq_len)
                    # Compute entropy per head, then average
                    attn_probs = attn.squeeze(0).float()  # (heads, seq, seq)
                    # Clamp to avoid log(0)
                    attn_probs = attn_probs.clamp(min=1e-10)
                    entropy = -(attn_probs * attn_probs.log()).sum(dim=-1)  # (heads, seq)
                    mean_entropy = entropy.mean().item()
                    attn_entropies[idx].append(mean_entropy)

                # 5. Store output mean for cross-example variance
                output_means[idx].append(out_mean.float().numpy())

            # Clear stored activations to free memory
            self._layer_inputs = {}
            self._layer_outputs = {}
            self._attention_weights = {}

        self._clear_hooks()

        # Aggregate into LayerStats
        stats = []
        for idx in range(self.n_layers):
            ls = LayerStats(layer_idx=idx)

            if repr_changes[idx]:
                ls.representation_change = np.mean(repr_changes[idx])

            if self_sims[idx]:
                ls.self_similarity = np.mean(self_sims[idx])

            if norm_growths[idx]:
                ls.norm_growth = np.mean(norm_growths[idx])

            if attn_entropies[idx]:
                ls.attention_entropy = np.mean(attn_entropies[idx])

            # Cross-example variance
            if output_means[idx]:
                stacked = np.stack(output_means[idx])  # (n_examples, hidden_dim)
                ls.cross_example_variance = np.mean(np.var(stacked, axis=0))

            # Representation rank (approximate via singular values)
            if repr_change_vectors[idx] and len(repr_change_vectors[idx]) >= 5:
                change_matrix = np.stack(repr_change_vectors[idx])  # (n, hidden_dim)
                try:
                    svs = np.linalg.svd(change_matrix, compute_uv=False)
                    # Effective rank: exp(entropy of normalized singular values)
                    svs_norm = svs / (svs.sum() + 1e-10)
                    svs_norm = svs_norm[svs_norm > 1e-10]
                    ls.representation_rank = np.exp(-np.sum(svs_norm * np.log(svs_norm)))
                except np.linalg.LinAlgError:
                    ls.representation_rank = 0.0

            stats.append(ls)

        # Compute forward similarity (needs adjacent layer outputs)
        # Re-run one example to get all layer outputs simultaneously
        if calibration_texts:
            inputs = self.tokenizer(
                calibration_texts[0],
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )
            self._register_hooks()
            with torch.no_grad():
                self.model(**inputs)

            for idx in range(self.n_layers - 1):
                out_curr = self._layer_outputs.get(idx)
                out_next = self._layer_outputs.get(idx + 1)
                if out_curr is not None and out_next is not None:
                    curr_mean = out_curr.squeeze(0).mean(dim=0).float()
                    next_mean = out_next.squeeze(0).mean(dim=0).float()
                    sim = torch.nn.functional.cosine_similarity(
                        curr_mean.unsqueeze(0), next_mean.unsqueeze(0)
                    ).item()
                    stats[idx].forward_similarity = sim

            self._clear_hooks()

        return stats


class ContrastiveCollector:
    """
    Collects activation statistics separately for reasoning and general inputs,
    then computes per-layer contrast scores.
    """

    def __init__(self, model, tokenizer):
        self.collector = ActivationCollector(model, tokenizer)

    def collect_contrastive(self, reasoning_texts, general_texts, max_length=256):
        """
        Collect stats for both sets and compute per-layer contrast.

        Returns:
            Tuple of (reasoning_stats, general_stats, contrast_scores)
            where contrast_scores[i] measures how differently layer i behaves
            on reasoning vs general inputs.
        """
        print("Collecting reasoning stats...")
        r_stats = self.collector.collect_stats(reasoning_texts, max_length)

        print("Collecting general stats...")
        g_stats = self.collector.collect_stats(general_texts, max_length)

        # Compute per-layer contrast
        contrast = []
        for r, g in zip(r_stats, g_stats):
            # Normalized difference for each metric
            diffs = []

            # Representation change contrast
            denom = max(abs(r.representation_change), abs(g.representation_change), 1e-10)
            diffs.append(abs(r.representation_change - g.representation_change) / denom)

            # Self-similarity contrast
            diffs.append(abs(r.self_similarity - g.self_similarity))

            # Norm growth contrast
            denom = max(abs(r.norm_growth), abs(g.norm_growth), 1e-10)
            diffs.append(abs(r.norm_growth - g.norm_growth) / denom)

            # Cross-example variance contrast
            denom = max(abs(r.cross_example_variance), abs(g.cross_example_variance), 1e-10)
            diffs.append(abs(r.cross_example_variance - g.cross_example_variance) / denom)

            # Attention entropy contrast
            if r.attention_entropy > 0 and g.attention_entropy > 0:
                denom = max(abs(r.attention_entropy), abs(g.attention_entropy), 1e-10)
                diffs.append(abs(r.attention_entropy - g.attention_entropy) / denom)

            contrast.append(np.mean(diffs))

        return r_stats, g_stats, contrast


class CircuitScorer:
    """
    Computes circuit criticality scores for candidate layer blocks
    using collected activation statistics.
    """

    def __init__(self, layer_stats, contrast_scores=None):
        self.stats = layer_stats
        self.n_layers = len(layer_stats)
        self.contrast = contrast_scores  # Optional contrastive scores

    def score_blocks(self, block_sizes=(3, 4, 5), method="composite"):
        """
        Score all candidate blocks.

        Args:
            block_sizes: Tuple of block sizes to consider.
            method: Scoring method ('anomaly', 'boundary', 'composite', 'contrastive', 'stability').

        Returns:
            List of BlockScore sorted by score descending.
        """
        scores = []

        for k in block_sizes:
            for start in range(self.n_layers - k + 1):
                end = start + k

                if method == "anomaly":
                    score = self._anomaly_score(start, end)
                elif method == "boundary":
                    score = self._boundary_score(start, end)
                elif method == "contrastive":
                    score = self._contrastive_score(start, end)
                elif method == "stability":
                    score = self._stability_score(start, end)
                elif method == "composite":
                    score = self._composite_score(start, end)
                else:
                    score = self._composite_score(start, end)

                scores.append(score)

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def _anomaly_score(self, start, end):
        """
        Score based on how much the block deviates from the model-wide baseline.
        Reasoning circuits should appear as statistical outliers.
        """
        block_stats = self.stats[start:end]
        all_stats = self.stats

        metrics = {}
        total_score = 0.0

        # Representation change: are these layers doing more work than average?
        block_change = np.mean([s.representation_change for s in block_stats])
        all_change = np.mean([s.representation_change for s in all_stats])
        std_change = np.std([s.representation_change for s in all_stats]) + 1e-10
        z_change = (block_change - all_change) / std_change
        metrics["repr_change_z"] = z_change
        total_score += z_change

        # Self-similarity: lower self-similarity means more transformation
        block_sim = np.mean([s.self_similarity for s in block_stats])
        all_sim = np.mean([s.self_similarity for s in all_stats])
        std_sim = np.std([s.self_similarity for s in all_stats]) + 1e-10
        z_sim = -(block_sim - all_sim) / std_sim  # Negative because lower sim = more work
        metrics["self_sim_z"] = z_sim
        total_score += z_sim

        # Cross-example variance: reasoning layers should be more input-dependent
        block_var = np.mean([s.cross_example_variance for s in block_stats])
        all_var = np.mean([s.cross_example_variance for s in all_stats])
        std_var = np.std([s.cross_example_variance for s in all_stats]) + 1e-10
        z_var = (block_var - all_var) / std_var
        metrics["variance_z"] = z_var
        total_score += z_var

        # Representation rank: higher rank means more diverse transformations
        block_rank = np.mean([s.representation_rank for s in block_stats])
        all_rank = np.mean([s.representation_rank for s in all_stats])
        std_rank = np.std([s.representation_rank for s in all_stats]) + 1e-10
        z_rank = (block_rank - all_rank) / std_rank
        metrics["rank_z"] = z_rank
        total_score += z_rank

        return BlockScore(start=start, end=end, score=total_score, metrics=metrics)

    def _boundary_score(self, start, end):
        """
        Score based on boundary contrast: a reasoning circuit should be
        internally coherent but distinct from its neighbors.
        """
        block_stats = self.stats[start:end]
        metrics = {}
        total_score = 0.0

        # Internal coherence: high forward similarity within the block
        if end - start > 1:
            internal_sims = [self.stats[i].forward_similarity for i in range(start, end - 1)]
            internal_coherence = np.mean(internal_sims) if internal_sims else 0.0
        else:
            internal_coherence = 0.0
        metrics["internal_coherence"] = internal_coherence
        total_score += internal_coherence

        # Boundary contrast: low similarity at the edges
        left_contrast = 0.0
        if start > 0:
            left_contrast = 1.0 - self.stats[start - 1].forward_similarity

        right_contrast = 0.0
        if end < self.n_layers:
            right_contrast = 1.0 - self.stats[end - 1].forward_similarity

        boundary = (left_contrast + right_contrast) / 2
        metrics["boundary_contrast"] = boundary
        total_score += boundary

        return BlockScore(start=start, end=end, score=total_score, metrics=metrics)

    def _stability_score(self, start, end):
        """
        Score based on the 'stability zone' hypothesis: reasoning circuits are
        where the representation change gradient flattens out — the transition
        from chaotic early processing to steady-state.

        This metric looks for blocks where:
        1. Representation change is STABLE (low gradient between adjacent layers)
        2. Variance growth is moderate (not minimal like embedding layers, not maximal like late layers)
        3. The block sits at the boundary between high-change and steady-change regions
        """
        block_stats = self.stats[start:end]
        all_stats = self.stats
        metrics = {}
        total_score = 0.0

        # 1. Stability: low absolute derivative of representation change
        changes = [s.representation_change if not np.isnan(s.representation_change) else 0.0 for s in all_stats]
        if end < len(changes):
            block_deltas = [abs(changes[i+1] - changes[i]) for i in range(start, min(end, len(changes)-1))]
            all_deltas = [abs(changes[i+1] - changes[i]) for i in range(len(changes)-1)]
            mean_delta = np.mean(block_deltas) if block_deltas else 0
            all_mean_delta = np.mean(all_deltas)
            std_delta = np.std(all_deltas) + 1e-10
            # Lower delta = more stable = higher score
            z_stability = -(mean_delta - all_mean_delta) / std_delta
            metrics["stability_z"] = z_stability
            total_score += z_stability * 1.5  # Weight stability highly

        # 2. Moderate variance growth (not too low, not too high)
        variances = [s.cross_example_variance if not np.isnan(s.cross_example_variance) else 0.0 for s in all_stats]
        if end < len(variances) and start > 0:
            block_var_growth = [variances[i] / max(variances[i-1], 1e-10) for i in range(max(start,1), end)]
            all_var_growth = [variances[i] / max(variances[i-1], 1e-10) for i in range(1, len(variances))]
            # Target: moderate growth (around median)
            median_growth = np.median(all_var_growth)
            block_mean_growth = np.mean(block_var_growth) if block_var_growth else 0
            deviation = abs(block_mean_growth - median_growth)
            all_deviations = [abs(g - median_growth) for g in all_var_growth]
            std_dev = np.std(all_deviations) + 1e-10
            z_moderate = -(deviation - np.mean(all_deviations)) / std_dev
            metrics["moderate_var_z"] = z_moderate
            total_score += z_moderate

        # 3. Transition detection: block is right after a high-change region
        if start > 2:
            pre_block_change = np.mean(changes[max(0,start-3):start])
            block_change = np.mean([s.representation_change for s in block_stats])
            if pre_block_change > 0:
                transition_ratio = block_change / pre_block_change
                # A strong drop (ratio < 0.5) suggests transition from chaos to stability
                transition_score = max(0, 1.0 - transition_ratio)
                metrics["transition_score"] = transition_score
                total_score += transition_score

        # 4. Representation rank: reasoning circuits should have moderate-high rank
        block_rank = np.mean([s.representation_rank for s in block_stats])
        all_rank = np.mean([s.representation_rank for s in all_stats])
        std_rank = np.std([s.representation_rank for s in all_stats]) + 1e-10
        z_rank = (block_rank - all_rank) / std_rank
        metrics["rank_z"] = z_rank
        total_score += z_rank * 0.5

        return BlockScore(start=start, end=end, score=total_score, metrics=metrics)

    def _contrastive_score(self, start, end):
        """
        Score based on how differently the block behaves on reasoning vs general inputs.
        Requires contrast_scores from ContrastiveCollector.
        """
        if self.contrast is None:
            # Fall back to anomaly if no contrastive data
            return self._anomaly_score(start, end)

        block_contrast = np.mean(self.contrast[start:end])
        all_contrast = np.mean(self.contrast)
        std_contrast = np.std(self.contrast) + 1e-10
        z_contrast = (block_contrast - all_contrast) / std_contrast

        # Also incorporate anomaly signal
        anomaly = self._anomaly_score(start, end)

        combined = 0.5 * z_contrast + 0.5 * anomaly.score
        metrics = {**anomaly.metrics, "contrastive_z": z_contrast, "contrastive_raw": block_contrast}

        return BlockScore(start=start, end=end, score=combined, metrics=metrics)

    def _composite_score(self, start, end):
        """
        Combined score using stability + anomaly signals.
        Takes the max of normalized stability and anomaly scores to
        capture both early (stability-detected) and late (anomaly-detected) circuits.
        """
        stability = self._stability_score(start, end)
        anomaly = self._anomaly_score(start, end)

        # We use both scores — take the max contribution
        # This lets stability dominate for early layers and anomaly for late layers
        metrics = {}
        metrics["stability_score"] = stability.score
        metrics["anomaly_score"] = anomaly.score
        metrics.update({f"stab_{k}": v for k, v in stability.metrics.items()})
        metrics.update({f"anom_{k}": v for k, v in anomaly.metrics.items()})

        # Combined: weighted sum where both signals contribute
        combined = 0.5 * stability.score + 0.5 * anomaly.score

        return BlockScore(start=start, end=end, score=combined, metrics=metrics)

    def print_rankings(self, scores, top_k=10):
        """Print the top-k candidate blocks with their scores."""
        print(f"\nTop {top_k} predicted reasoning circuit locations:")
        print(f"{'Rank':<6}{'Layers':<15}{'Score':<10}{'Key Metrics'}")
        print("-" * 70)

        for i, s in enumerate(scores[:top_k]):
            key_metrics = ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(s.metrics.items())
                if not k.endswith("_component")
            )
            print(f"{i+1:<6}[{s.start}-{s.end}){'':<7}{s.score:<10.4f}{key_metrics}")

    def print_layer_stats(self):
        """Print per-layer statistics for visualization."""
        print(f"\nPer-layer activation statistics ({self.n_layers} layers):")
        print(f"{'Layer':<7}{'ReprChg':<10}{'SelfSim':<10}{'FwdSim':<10}"
              f"{'NormGr':<10}{'Variance':<12}{'Rank':<10}{'AttnEnt':<10}")
        print("-" * 85)

        for s in self.stats:
            print(f"{s.layer_idx:<7}"
                  f"{s.representation_change:<10.4f}"
                  f"{s.self_similarity:<10.4f}"
                  f"{s.forward_similarity:<10.4f}"
                  f"{s.norm_growth:<10.4f}"
                  f"{s.cross_example_variance:<12.6f}"
                  f"{s.representation_rank:<10.2f}"
                  f"{s.attention_entropy:<10.4f}")


def save_stats(stats, path):
    """Save layer stats to JSON for later analysis."""
    data = []
    for s in stats:
        data.append({
            "layer_idx": s.layer_idx,
            "representation_change": float(s.representation_change),
            "self_similarity": float(s.self_similarity),
            "forward_similarity": float(s.forward_similarity),
            "norm_growth": float(s.norm_growth),
            "cross_example_variance": float(s.cross_example_variance),
            "representation_rank": float(s.representation_rank),
            "attention_entropy": float(s.attention_entropy),
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved stats to {path}")


def load_stats(path):
    """Load layer stats from JSON."""
    with open(path) as f:
        data = json.load(f)
    stats = []
    for d in data:
        ls = LayerStats(**d)
        stats.append(ls)
    return stats
