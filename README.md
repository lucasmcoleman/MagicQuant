# MagicQuant

**Evolutionary Tensor Search for Optimal LLM GGUF Hybrid Quantization**

A Python implementation of the MagicQuant framework — an evolutionary search algorithm that discovers optimal per-group quantization configurations for LLM GGUF files. Instead of applying one quantization scheme globally, MagicQuant assigns different schemes to different tensor groups (embeddings, attention, FFN) based on measured sensitivity, producing models that break the standard size/quality/speed Pareto frontier.

## Origin & Credit

This project is an implementation of the **MagicQuant** methodology created by [**magiccodingman**](https://github.com/magiccodingman). The original research, empirical findings, and the "MXFP4 Anomaly" discovery are documented in the [MagicQuant Wiki](https://github.com/magiccodingman/MagicQuant-Wiki). Published hybrid models are available on HuggingFace:

- [Magic Quant Collection](https://huggingface.co/collections/magiccodingman/magic-quant) — Verified best-of-the-best hybrid quants
- [MXFP4 Hybrid GGUF Collection](https://huggingface.co/collections/magiccodingman/mxfp4-hybrid-gguf) — Experimental MXFP4 hybrids

The core insight from the original research: the vast majority of parameters in a transformer (FFN layers) can tolerate aggressive MXFP4 compression, while a small set of "brain" layers (embeddings, attention output, LM head) need protection at higher precision. This "Carbon Fiber Body, Ferrari Engine" pattern consistently produces models that are smaller than Q4 but retain Q5/Q6 quality.

## How It Works

```
BF16 Source Model
     |
     v
[1. Sensitivity Probing] — Quantize one group at a time, measure PPL impact
     |
     v
[2. Evolutionary Search] — Discover optimal hybrid configs per compression tier
     |                      Protector/Crusher mutations, epsilon-greedy exploration
     v
[3. Tiered Generation]  — Output best Q4, Q5, Q6 hybrid GGUFs
     |                     Each with per-tensor quantization via GGUF writer
     v
  Q4: E:BF16 H:Q8 Q:Q6K K:Q8 O:BF16 U:MXFP4 D:MXFP4  (24 GB from 60 GB)
  Q5: E:BF16 H:BF16 Q:Q8 K:BF16 O:BF16 U:MXFP4 D:MXFP4 (29 GB from 60 GB)
  Q6: E:BF16 H:BF16 Q:Q6K K:Q8 O:BF16 U:BF16 D:Q8       (44 GB from 60 GB)
```

### Tensor Groups

| Group | Role | Typical Sensitivity | Default Scheme |
|-------|------|-------------------|----------------|
| **E** | Token Embeddings | Very High | BF16 |
| **H** | LM Head | Very High | BF16 |
| **O** | Attention Output | High | Q8_0 / BF16 |
| **Q** | Attention Query | Moderate | Q6_K / IQ4_NL |
| **K** | Attention Key/Value | Moderate | Q8_0 / Q6_K |
| **U** | FFN Up/Gate | Low (robust) | **MXFP4** |
| **D** | FFN Down | Low (robust) | **MXFP4** |
| **X** | MoE Experts | Very Low | MXFP4 |
| **R** | MoE Router | High | Q8_0 |

### Supported Quantization Schemes

| Scheme | Type | bpw | Noise | Best For |
|--------|------|-----|-------|----------|
| BF16 | Float | 16.0 | 0.0 | Brain layers (E, H, O) |
| Q8_0 | Integer | 8.5 | 1.0 | Near-lossless protection |
| Q6_K | K-quant | 6.56 | 2.2 | High-quality attention |
| Q5_K | K-quant | 5.5 | 3.0 | Balanced attention |
| IQ4_NL | Non-linear | 4.5 | 3.8 | Best ~4-bit quality |
| **MXFP4** | **FP4 (E2M1)** | **4.25** | **4.0** | **FFN / MoE experts** |
| Q4_K_M | K-quant | 4.5 | 4.5 | Fallback integer 4-bit |

MXFP4 implements the [OCP MX Microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) FP4 format (E2M1 values with shared E8M0 exponent). Its non-uniform quantization levels (0, 0.5, 1, 1.5, 2, 3, 4, 6) are denser near zero, naturally matching the Gaussian-like weight distribution of transformers — producing lower noise than integer Q4 at better compression.

## Installation

```bash
git clone https://github.com/lucasmcoleman/MagicQuant.git
cd MagicQuant
pip install -e .
```

Requires Python 3.9+ and NumPy. Optional: [llama.cpp](https://github.com/ggerganov/llama.cpp) for real perplexity measurement during probing.

## Usage

### Full Pipeline

```bash
# 1. Analyze model structure
magicquant analyze model-bf16.gguf

# 2. Run sensitivity probing (uses heuristics if llama.cpp not available)
magicquant probe model-bf16.gguf --output-dir ./output

# 3. Run evolutionary search
magicquant search model-bf16.gguf --output-dir ./output --generations 50

# 4. Generate best Q4, Q5, Q6 hybrid GGUFs
magicquant generate model-bf16.gguf --output-dir ./output --tiers Q4,Q5,Q6
```

### Manual Hybrid from YAML Config

```yaml
# config.yaml
model:
  name: Qwen3-30B-A3B
  source: ./Qwen3-30B-A3B-BF16.gguf
quantization:
  base: MXFP4_MOE
  groups:
    E: BF16
    H: BF16
    O: Q8_0
    Q: IQ4_NL
    K: IQ4_NL
```

```bash
magicquant hybrid config.yaml --output-dir ./output
```

### Python API

```python
from magicquant.gguf.writer import create_hybrid_gguf

create_hybrid_gguf(
    output_path="model-hybrid.gguf",
    base_model_path="model-bf16.gguf",
    quant_config={
        "base": "MXFP4_MOE",
        "groups": {
            "E": "BF16",
            "H": "BF16",
            "O": "Q8_0",
            "Q": "IQ4_NL",
            "K": "IQ4_NL",
        }
    }
)
```

## Architecture

```
magicquant/
  gguf/
    reader.py          — GGUF binary parser
    writer.py          — Hybrid GGUF writer (two-pass streaming)
    tensor_groups.py   — Tensor group classification (E, H, Q, K, O, U, D, X, R)
  quant/
    schemes.py         — Quantization scheme definitions
    converters.py      — Vectorized ggml block encoders (single source of truth)
  evolution/
    probing.py         — Sensitivity measurement (real or heuristic)
    predictor.py       — Loss/size/speed prediction with collapse penalties
    survival.py        — Evolutionary search with Protector/Crusher mutations
  utils/
    llamacpp.py        — llama.cpp integration for perplexity measurement
    naming.py          — Hybrid model filename generation
  orchestrator.py      — Pipeline coordination
  __main__.py          — CLI entry point
```

## License

This implementation is provided as-is. The MagicQuant methodology and research are credited to [magiccodingman](https://github.com/magiccodingman).
