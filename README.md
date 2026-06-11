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

> **Note on BF16:** Groups designated `BF16` (brain layers E/H/O) are stored on
> disk as **F16**, not BF16. llama.cpp's compute graph has incomplete BF16
> support, so the writer downgrades to F16 (and logs a one-time warning).
> Values outside the F16 range (|x| > 65504, or subnormals) may become Inf/0.

## Installation

```bash
git clone https://github.com/lucasmcoleman/MagicQuant.git
cd MagicQuant
pip install -e .
```

Requires Python 3.9+ and NumPy. Optional: [llama.cpp](https://github.com/ggerganov/llama.cpp) for real perplexity measurement during probing.

For **Quantization-Aware Training** (`magicquant qat`) install the optional
`[qat]` extra, which pulls the heavy training stack (the core install stays
torch-free):

```bash
pip install -e ".[qat]"   # torch, transformers, peft, trl, datasets
```

> `torch` must be the build that works on your hardware (the ROCm wheel on AMD,
> CUDA on NVIDIA, or CPU). Install the matching wheel first if the default index
> doesn't provide it.

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

### Quantization-Aware Training (QAT-LoRA)

`magicquant qat` fine-tunes a model to be robust to a chosen per-group hybrid
config *before* it ships as that hybrid. It freezes the base model,
fake-quantizes it to the search's per-group schemes in the forward pass (a
differentiable per-scheme fake-quant with a straight-through estimator, validated
against libggml), and trains LoRA adapters that compensate. Largest benefit at the
aggressive tiers (Q2/Q3/MXFP4). Requires the `[qat]` extra (see Installation).

```bash
# Train adapters robust to the Q4 tier from a prior search's results.
magicquant qat ./my-model \
    --config ./output/search_results.json \
    --tier Q4 \
    --dataset data/chat.jsonl \
    --out ./output/qat_adapters \
    --lora-r 32 --lora-alpha 64 --epochs 1 --lr 2e-4
```

The per-group hybrid config comes from a prior `search`'s `search_results.json`
(`--config` + `--tier`); the dataset is a chat JSONL (`{"messages": [...]}` per
line) trained with completion-only loss. Adapters + a `qat_meta.json` (base model,
scheme-by-group, hyperparams, config hash) are written to `--out`. Merge the
adapters, then pack the exact hybrid with `magicquant generate`. In Foundry this
is surfaced as the **QAT** pipeline stage (toggle + config + live logs).

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
  qat/                 — Quantization-Aware Training (QAT-LoRA); needs the [qat] extra
    fake_quant.py      — Differentiable per-scheme fake-quant + STE (vs libggml)
    wrap.py            — QATLinear (fake-quants merged base+LoRA) + wrap_model
    names.py           — HF module -> GGUF tensor name mapping (reuses source.py)
    config.py          — load_hybrid_config: search_results.json -> {group: ggml_type}
    train.py           — run_qat: the QAT-LoRA loop (completion-only) + adapters
    validate.py        — perplexity comparison (QAT hybrid vs plain hybrid)
  utils/
    llamacpp.py        — llama.cpp integration for perplexity measurement
    naming.py          — Hybrid model filename generation
  orchestrator.py      — Pipeline coordination
  __main__.py          — CLI entry point
```

## Configuration via Environment

The `search` command (and `search --dry-run`) reads settings from environment
variables with the `MAGICQUANT_` prefix or a `.env` file; explicit CLI flags
override them. Defaults below are the single source of truth (`config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MAGICQUANT_SOURCE_MODEL_PATH` | (required) | Path to source model |
| `MAGICQUANT_OUTPUT_DIR` | `./output` | Output directory |
| `MAGICQUANT_LLAMACPP_PATH` | auto-detect | Path to llama.cpp |
| `MAGICQUANT_TARGET_BASE_QUANT` | `MXFP4_MOE` | Base quantization scheme |
| `MAGICQUANT_SEARCH_GENERATIONS` | `30` | Generations per round |
| `MAGICQUANT_POPULATION_SIZE` | `80` | Candidates per generation |
| `MAGICQUANT_MEASUREMENT_ROUNDS` | `3` | Build-measure-learn cycles |
| `MAGICQUANT_TIERS` | `["Q4","Q5","Q6"]` | Compression tiers |

## Docker

```bash
docker build -t magicquant:latest .
docker run --rm -v ./output:/app/output magicquant:latest search /data/model.gguf
```

## Development

```bash
pip install -e ".[dev]"
make test    # Run pytest suite
make lint    # Syntax check
make clean   # Remove build artifacts
```

## Known Limitations

- K-quant encoders use simple min/max with RMSE optimization, not llama.cpp's full importance-matrix-weighted quantization
- Tokenizer reading only handles BPE (tokenizer.json); SentencePiece (.model) is not supported
- Source models must be BF16/F16/F32 — pre-quantized sources are rejected with a clear error

## License

MIT. The MagicQuant methodology and research are credited to [magiccodingman](https://github.com/magiccodingman).
