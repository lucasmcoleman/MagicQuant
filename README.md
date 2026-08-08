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
[1. Sensitivity Probing] — Quantize one group at a time, measure KL divergence
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

**A tier name is a size band, not a recipe.** Note above that the "Q4" config
contains no Q4 tensors at all, and the "Q6" contains BF16 and Q8. A tier is
whatever mix of schemes landed in that size band with the lowest measured loss.
Grading a build by whether it "contains N-bit tensors" is a category error;
size and measured quality are the only criteria. Bands are defined in
`quant/tiers.py` as ratios to the BF16 baseline, and `tools/reselect_tiers.py`
re-derives a finished run's ladder from its stored measurements.

Probes score by **KL divergence against saved reference logits**, not
perplexity. Perplexity cannot resolve a single-group probe: the change is far
below run-to-run noise, so probes came back undifferentiated and normalization
then flattened them to zero — on one MoE model the expert group, 93% of all
parameters, ended up with sensitivity weight exactly 0.000. KL measures the
same perturbation at roughly 79 sigma instead of 0.55.

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

The default search pool (what `search` samples from with no extra flags):

| Scheme | Type | bpw | Noise (heuristic) | Best For |
|--------|------|-----|-------|----------|
| BF16 | Float | 16.0 | 0.0 | Brain layers (E, H, O) |
| Q8_0 | Integer | 8.5 | 1.0 | Near-lossless protection |
| Q6_K | K-quant | 6.56 | 2.2 | High-quality attention |
| Q5_K | K-quant | 5.5 | 3.0 | Balanced attention |
| IQ4_NL | Non-linear | 4.5 | 3.8 | Best ~4-bit quality **with an imatrix** — dropped from the pool without one (see Known Limitations) |
| **MXFP4** | **FP4 (E2M1)** | **4.25** | **4.0** | **FFN / MoE experts** |
| Q4_K_M | K-quant | 4.5 | 4.5 | Fallback integer 4-bit |
| Q3_K | K-quant | 3.44 | 8.0 | Aggressive attention / robust-group compression |
| Q2_K | K-quant | 2.625 | 15.0 | Most aggressive robust-group compression |

MXFP4 implements the [OCP MX Microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) FP4 format (E2M1 values with shared E8M0 exponent). Its non-uniform quantization levels (0, 0.5, 1, 1.5, 2, 3, 4, 6) are denser near zero, naturally matching the Gaussian-like weight distribution of transformers — producing lower noise than integer Q4 at better compression.

`quant/schemes.py`'s registry holds 23 schemes total; the 9 above are the
default pool. The other 14 are real, working schemes but are **not**
uniformly "supported" in the same sense — each is gated behind an explicit
opt-in or a hardware/calibration requirement:

- **`IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ1_M`, `IQ1_S`**
  — opt-in via `enable_iq` (search's `--enable-iq`). `IQ2_XS`, `IQ2_XXS`, and
  `IQ1_S` additionally *require* an imatrix to encode at all.
- **`Q4_0`, `Q4_1`** — legacy block=32 quants, v2-only (`--target-profile
  q4nx`); excluded from v1's random-config sampling to keep the seed-pinned
  search fixture stable.
- **`ROCMFP3`, `ROCMFP4`, `ROCMFP6`, `ROCMFP8`** — AMD-native fork schemes,
  opt-in via `enable_rocmfpx`. Encoding requires a
  [ROCmFPX](https://github.com/ciru-ai/ROCmFPX) build of libggml, and the
  resulting GGUF loads only on that fork, not stock llama.cpp.

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

### Importance Matrix (imatrix)

`search`/`generate` already capture and apply an importance-weighted imatrix
automatically by default, from a bundled calibration corpus (see Known
Limitations below) — most users never need to run this directly.
`magicquant imatrix` is the manual/advanced entry point: capture one against
your own corpus once and reuse the result, or produce a file for the Python
API's `create_hybrid_gguf(..., imatrix=...)`.

```bash
magicquant imatrix model-bf16.gguf \
    -f my-corpus.txt \
    -o model-bf16.imatrix.gguf \
    --chunks 200 --ctx-size 512
```

- `model` (positional) — GGUF model to instrument.
- `-f`/`--corpus` (required) — plain-text calibration corpus (e.g. a
  wikitext-2 **train** split). Must be a different file than whatever corpus
  you score perplexity against — calibrating and evaluating on the same text
  makes every measured loss optimistic with nothing in the output revealing
  it.
- `-o`/`--output` — output imatrix GGUF (default: `<model>.imatrix.gguf`).
- `--chunks` — max corpus chunks to process (default `-1` = all).
- `--ctx-size` — chunk length in tokens (default `512`).

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
adapters with `magicquant qat-merge`, then pack the exact hybrid with
`magicquant generate`. In Foundry this is surfaced as the **QAT** pipeline
stage (toggle + config + live logs).

```bash
# Merge the trained adapters into the base model's safetensors (streamed
# shard-by-shard; never materializes the full model in memory).
magicquant qat-merge ./my-model \
    --adapters ./output/qat_adapters \
    --out ./output/qat_merged
```

- `base_model` (positional) — HF model id or local path to the base model
  whose safetensors get merged (the same model QAT was run against).
- `--adapters` (required) — adapter directory written by `magicquant qat`
  (needs `adapter_model.safetensors` + `qat_meta.json`).
- `--out` (required) — output directory for the merged safetensors model.

Use this rather than a generic PEFT merge — MagicQuant's adapter key naming
doesn't match PEFT's convention, and a generic merge has silently produced an
unmodified copy of the base model in the past. `qat-merge` fails loudly
instead of degrading to a no-op.

> **Budget builds interoperate here too.** `search --algo v2 --budget-gb <N>`
> now also merges a `BUDGET-<N>GiB` pseudo-tier into `search_results.json`,
> carrying a per-group projection alongside the exact per-tensor allocation.
> So `--tier BUDGET-<N>GiB` works with QAT unchanged, and Foundry's ROCmFPX
> `mq-budget` mode consumes the same block — no separate interchange format.
> The merge is additive: an existing file's Q4/Q5/Q6 tiers are left untouched,
> and a legacy file without a version stamp does not gain one.

**Validated result.** In a confound-controlled run on Qwen2.5-0.5B base with an
aggressive Q4_K-attention/MXFP4-FFN hybrid — bf16 PPL 16.35, plain quant 19.54
(+3.19 damage), quant+QAT 15.13, and a bf16+identical-LoRA control 13.46 — the
quant-vs-bf16 gap shrank from +3.19 to +1.67 once the LoRA's own domain adaptation
is held fixed on both arms. **QAT recovered 47.5% of the quantization loss beyond
plain LoRA domain-adaptation.** Recovery scales with quant aggressiveness, and the
final GGUF pack is exact-ggml (byte-identical to llama.cpp) even though training
uses a faithful torch fake-quant. See [`docs/qat.md`](docs/qat.md) for the full
methodology, the fake-quant/STE design, multimodal/bf16 support, and caveats.

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

### Model Card

`magicquant card` writes a HuggingFace-ready model card summarizing a
finished search's tier results:

```bash
magicquant card --output-dir ./output --model model-bf16.gguf --base-model org/Model-Name
```

- `--output-dir` — directory containing `search_results.json` (default `./output`).
- `--model` — source model path, used to derive the card title.
- `--base-model` — base model name to show on the card.
- `--upload` — upload the generated card to HuggingFace (requires
  `huggingface_hub`, `pip install 'magicquant[hf]'`); needs `--repo`.
- `--repo` — target HF repo id (`owner/name`) for `--upload`.

## Architecture

```
magicquant/
  gguf/
    reader.py          — GGUF binary parser
    writer.py          — Hybrid GGUF writer (two-pass streaming)
    source.py          — ModelSource: GGUF / safetensors / LoRA-merged input abstraction
    tensor_groups.py   — Tensor group classification (E, H, Q, K, O, U, D, S, N, V, X, R)
  quant/
    schemes.py         — Quantization scheme registry (single source of truth for scheme metadata)
    converters.py      — encode_to_ggml_bytes() dispatch (single encoder source of truth)
    ggml_binding.py    — ctypes binding to libggml (byte-identical to llama-quantize)
    ggml_facts.py      — Block/type-size tables derived from ggml_binding
    tiers.py           — Tier (size-band) boundary classification
    calibration.py     — Optional calibrated noise/speed-multiplier overrides
  evolution/
    probing.py         — Sensitivity measurement (real or heuristic)
    predictor.py       — Loss/size/speed prediction with collapse penalties
    survival.py        — Evolutionary search with Protector/Crusher mutations
  v2/                  — Budget search (--algo v2, NON-DEFAULT; see docs/redesign.md)
    sensitivity.py     — Per-tensor x per-scheme distortion table (imatrix-weighted)
    calibrate.py       — Group-probe kappa fitting (amplification factors)
    allocate.py        — Multiple-choice knapsack (Lagrangian greedy + polish)
    resolve.py         — Per-tensor scheme resolution -> writer override map
    interchange.py     — search_results.json BUDGET-<N>GiB pseudo-tier interop
    outcome.py         — Frontier/measurement bookkeeping
    search.py          — v2 orchestration entry point
  qat/                 — Quantization-Aware Training (QAT-LoRA); needs the [qat] extra
    fake_quant.py      — Differentiable per-scheme fake-quant + STE (vs libggml)
    wrap.py            — QATLinear (fake-quants merged base+LoRA) + wrap_model
    expert_wrap.py     — Fused 3-D MoE expert QAT (per-expert LoRA via parametrization)
    names.py           — HF module -> GGUF tensor name mapping (reuses source.py)
    config.py          — load_hybrid_config / load_tensor_config (search_results.json -> schemes)
    diskmap.py         — Base-weight map for adapter-key reconciliation preflight
    merge.py           — Streaming on-disk base+adapter merge; backs `magicquant qat-merge`
    train.py           — run_qat: the QAT-LoRA loop (completion-only) + adapters
    validate.py        — perplexity comparison (QAT hybrid vs plain hybrid)
  utils/
    llamacpp.py        — llama.cpp integration for perplexity measurement
    naming.py          — Hybrid model filename generation
    measurement.py     — Shared measurement/statistics helpers
    model_card.py      — HuggingFace model card generation; backs `magicquant card`
  imatrix.py           — Importance-matrix capture via llama-imatrix; backs `magicquant imatrix`
  incumbents.py        — Best-known-config tracking across search rounds
  pareto.py            — Pareto-frontier utilities
  config.py            — Settings from env vars / .env (single source of truth for defaults)
  logging.py           — structlog configuration
  orchestrator.py      — Pipeline coordination
  __main__.py          — CLI entry point
```

`v2/` is the 2026-07 budget-search redesign (`--algo v2 --budget-gb`); the v1
evolutionary path above remains the default. `qat/_ggml_ref.py` (a private
reference helper, not part of the public API) is omitted from this tree.

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
docker build -f docker/Dockerfile -t magicquant:latest .
docker run --rm -v ./output:/app/output magicquant:latest search /data/model.gguf
```

## Development

```bash
pip install -e ".[dev]"
make test    # Run pytest suite
make lint    # Ruff lint (magicquant/ + tests/)
make clean   # Remove build artifacts
```

## Known Limitations

- Tokenizer reading only handles BPE (tokenizer.json); SentencePiece (.model) is not supported
- Source models must be BF16/F16/F32 — pre-quantized sources are rejected with a clear error
- `IQ4_NL` is excluded from the search pool when no importance matrix is
  available. Uncalibrated it lost all 11 measured comparisons across two 27B
  models (3-20x worse than same-bpw MXFP4/Q4_K_M) despite *better* isolated
  weight-reconstruction error, because its lookup table places levels to
  minimise unweighted error. With an imatrix it is sampled normally.

> Encoding is byte-identical to llama.cpp — MagicQuant calls libggml directly
> rather than reimplementing the encoders, so there is no MSE quality gap.
> Importance-matrix weighting is supported and **on by default**: the bundled
> calibration corpus is ~1 MB spanning 18 languages plus code, math and agentic
> prompts, capture is bounded to 200 chunks, and the result is cached per model.
> The corpus is verified disjoint from the perplexity eval corpus, and
> calibration is refused outright if the two ever resolve to the same file —
> scoring a run on the text it was calibrated against would make every measured
> loss optimistic with nothing in the output revealing it.

## License

MIT. The MagicQuant methodology and research are credited to [magiccodingman](https://github.com/magiccodingman).
