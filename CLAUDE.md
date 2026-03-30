# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MagicQuant creates hybrid GGUF files with per-tensor-group quantization. Different tensor groups (embeddings, attention, FFN) get different quantization schemes based on sensitivity. MXFP4 (ggml type 39) is the primary compression scheme for robust layers (FFN, SSM). The methodology is from [magiccodingman](https://github.com/magiccodingman/MagicQuant-Wiki).

## Commands

```bash
pip install -e .                    # Install (editable)
pip install -e ".[yaml,dev]"        # With optional deps
magicquant analyze model.gguf       # Inspect tensor groups
magicquant search model.gguf --rounds 3  # Measured evolutionary search
magicquant generate model.gguf --tiers Q4,Q5,Q6  # Generate hybrids
```

No test suite exists yet. Validate with end-to-end runs on real models.

## Architecture

The pipeline has two paths:

**Direct quantization** (the common case): `create_hybrid_gguf()` reads a source model (GGUF or safetensors), classifies each tensor into a group, applies the configured quantization scheme per group, and writes a valid GGUF. The writer uses two-pass streaming (Pass 1: compute offsets, Pass 2: read+encode+write per tensor via a background thread).

**Evolutionary search**: The orchestrator runs sensitivity probing → evolutionary search → tiered generation. With `--rounds N`, it enters the Predict→Build→Measure→Learn loop where candidate GGUFs are actually built and measured with llama-perplexity, and residuals feed back into the predictor.

### Key data flow

```
Source (GGUF/safetensors/LoRA) → ModelSource abstraction (source.py)
  → GGUFWriter (writer.py) calls encode_to_ggml_bytes() per tensor (converters.py)
  → Output GGUF with mixed ggml types
```

### Source abstraction (source.py)

`open_model_source(path)` auto-detects format and returns a `ModelSource`:
- `GGUFSource` — reads from .gguf files
- `SafetensorsSource` — reads HF safetensors directories, maps tensor names to GGUF convention, reads config.json for metadata, reads tokenizer.json for vocabulary
- `LoRAMergedSource` — wraps a base source and applies LoRA deltas on-the-fly during `read_tensor_f32()`

### Quantization (converters.py)

Single source of truth for all ggml block encoding. `encode_to_ggml_bytes(weights, ggml_type_name)` is the public API. Encoders are vectorized with numpy. K-quant encoders (Q6_K, Q5_K, Q4_K) use RMSE-optimized scale selection (7 candidates per sub-block). The MXFP4 encoder matches llama.cpp's `quantize_row_mxfp4_ref` exactly (split-half nibble packing, doubled kvalues table, E8M0 exponent formula).

### Tensor groups (tensor_groups.py)

Groups: E (embeddings), H (head/MTP), Q (query), K (key/value), O (attention output), U (FFN up/gate), D (FFN down), S (SSM/linear attention), N (norms), V (vision), X (MoE experts), R (MoE router). Classification is regex-based on GGUF tensor names.

## Critical Invariants

- **MXFP4 is ggml type 39** — native llama.cpp support. Never use a custom type ID.
- **converters.py is the single encoder source** — writer.py must not contain quantization logic.
- **Shapes are stored row-major** in ModelSource (same as GGUFReader convention). The writer reverses to ggml on-disk order. SafetensorsSource must NOT pre-reverse.
- **BF16 uses round-to-nearest-even**, not truncation.
- **Source models must be BF16/F16/F32** — the writer raises ValueError on pre-quantized sources.
- **LlamaCppTools is lazy** — the orchestrator works without llama.cpp installed (prediction-only mode).

## HF Tensor Name Mapping

SafetensorsSource strips `model.language_model.` prefix for multimodal models, then matches against `_HF_TO_GGUF_PATTERNS`. The arch mapping in `_build_gguf_metadata_from_config` is synced from llama.cpp's `convert_hf_to_gguf.py` and handles nested `text_config` for composite models (Qwen3.5, etc.).

## Known Limitations

- K-quant encoders use simple min/max with RMSE optimization, not llama.cpp's full importance-matrix-weighted quantization. Quality gap is ~10-27% MSE vs llama.cpp native.
- Tokenizer reading only handles BPE (tokenizer.json). SentencePiece (.model) requires protobuf and is not implemented.
- The evolutionary search mostly rediscovers obvious configs (BF16 brain + MXFP4 FFN) for dense models. Its value comes from the measured search loop and MoE models with larger search spaces.
