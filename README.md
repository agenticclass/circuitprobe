# CircuitProbe

Lightweight probing for reasoning circuits in transformer language models.

CircuitProbe predicts reasoning circuit locations from activation statistics in under 5 minutes on CPU, compared to 7-25 GPU-hours for brute-force sweeps.

## What are reasoning circuits?

Transformer models contain localized "reasoning circuits": contiguous blocks of 3-5 layers that, when duplicated at inference time, improve reasoning performance without any training. CircuitProbe finds these circuits thousands of times faster than exhaustive search.

## Two types of circuits

We discovered that reasoning circuits come in two types:

- **Stability circuits** (early layers, 10-25% depth): detected via the derivative of representation change
- **Magnitude circuits** (late layers, 85-100% depth): detected via anomaly scoring

This pattern holds across 9 models spanning 6 architectures.

## Quick start

```python
from circuit_probe.circuit_probe import ActivationCollector, CircuitScorer, save_stats
from circuit_probe.calibration_data import get_calibration_set
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load any transformer model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Get calibration data (10+ examples of any text)
texts, labels = get_calibration_set(n_reasoning=10, n_general=10)

# Collect activation statistics
collector = ActivationCollector(model, tokenizer)
stats = collector.collect_stats(texts, max_length=128)

# Score candidate blocks
scorer = CircuitScorer(stats)
stability_scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="stability")
anomaly_scores = scorer.score_blocks(block_sizes=(3, 4, 5), method="anomaly")

# Top predictions
print("Stability circuit:", stability_scores[0].start, "-", stability_scores[0].end)
print("Magnitude circuit:", anomaly_scores[0].start, "-", anomaly_scores[0].end)
```

## Results

| Model | Params | Stability | Magnitude | Validated? |
|-------|--------|-----------|-----------|------------|
| Phi-4 | 14B | [5-10) | [37-40) | Yes (sweep GT) |
| Qwen2.5-7B | 7.6B | [6-11) | [24-27) | Yes (GGUF, top-1 best) |
| Qwen3-8B | 8B | [8-11) | [33-36) | Probed |
| Mistral-Instruct | 7B | [3-6) | [29-32) | Yes (GGUF, +41%) |
| TinyLlama | 1.1B | [3-6) | [19-22) | Yes (GPU 250q) |

## Small model scaling

Layer duplication is a free scaling technique for small models:

| Model | Params | Baseline | With Circuit | Change |
|-------|--------|----------|-------------|--------|
| Qwen2.5-0.5B | 0.5B | 40.0% | 42.4% | +2.4% |
| Qwen2.5-3B | 3B | 49.2% | 54.0% | +4.8% |
| TinyLlama | 1.1B | 16.4% | 26.4% | +10.0% |

## Requirements

```
torch
transformers
numpy
```

## Citation

```bibtex
@article{panuganti2026circuitprobe,
  title={CircuitProbe: Predicting Reasoning Circuits in Transformers via Stability Zone Detection},
  author={Panuganti, Rajkiran},
  year={2026}
}
```
