# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MagicQuant creates hybrid GGUF files with per-tensor-group quantization. Different tensor groups (embeddings, attention, FFN) get different quantization schemes based on sensitivity. MXFP4 (ggml type 39) is the primary compression scheme for robust layers (FFN, SSM). The methodology is from [magiccodingman](https://github.com/magiccodingman/MagicQuant-Wiki).

## Commands

```bash
pip install -e .                    # Install (editable)
pip install -e ".[yaml,dev]"        # With optional deps (yaml + pytest + gguf)
pip install -e ".[qat]"             # QAT training stack (torch/transformers/peft/trl/datasets)
magicquant analyze model.gguf       # Inspect tensor groups
magicquant search model.gguf --rounds 3  # Measured evolutionary search
magicquant generate model.gguf --tiers Q4,Q5,Q6  # Generate hybrids
magicquant qat ./model --config search_results.json --tier Q4 --dataset chat.jsonl --out adapters/  # QAT-LoRA
```

### Tests

```bash
pip install -e ".[dev]"             # pulls pytest + gguf
python -m pytest tests/ -q          # unit + writer + search + integration
```

- `tests/test_quantization_guards.py` — dtype guards + encoder output sizes (libggml).
- `tests/test_tensor_groups.py` — MoE/dense/SSM classification (locks the `*_exps` fix).
- `tests/test_tiers.py` — tier boundaries (leaf `magicquant.quant.tiers.classify_tier`).
- `tests/test_writer.py` — crash-safety (.partial + os.replace), UNKNOWN hard error,
  metadata serialization, BF16->F16 warning, end-to-end read->write->reopen.
- `tests/integration/test_encoder_parity.py` — byte-for-byte vs `llama-quantize`
  (skips when `gguf` or `llama-quantize` is unavailable).
- `tests/test_refactor_regression.py` — seed-pinned evolutionary-search golden fixture;
  REGENERATE `tests/fixtures/refactor_regression_seed42.json` whenever noise_factors,
  group sets, the sensitive-group floor clamp, or early-stop change behavior.

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

### Quantization (converters.py + ggml_binding.py)

`encode_to_ggml_bytes(weights, ggml_type_name, imatrix=None)` is the public API.
Float passthroughs (BF16/F16/F32) are native numpy; ALL quantized types route
through `magicquant.quant.ggml_binding.ggml_encode`, a ctypes binding to
`libggml` that calls `ggml_quantize_chunk` — the SAME function `llama-quantize`
uses. Output is therefore **byte-identical** to llama.cpp (proven by
`tests/integration/test_encoder_parity.py`). There are NO numpy K-quant
encoders anymore (the May 2026 refactor deleted them).

`ggml_binding._verify_type_ids` runs at first-encode and raises if the loaded
libggml renumbers/​resizes any type (drift guard). The block/type-size tables in
converters.py are derived from `ggml_binding._GGML_BLOCK_SIZE/_GGML_TYPE_SIZE`
(single source of truth) so they cannot drift.

### Tensor groups (tensor_groups.py)

Groups: E (embeddings), H (head/MTP), Q (query), K (key/value), O (attention output), U (FFN up/gate), D (FFN down), S (SSM/linear attention), N (norms), V (vision), X (MoE experts), R (MoE router). Classification is regex-based on GGUF tensor names.

### QAT (magicquant/qat/, optional `[qat]` extra)

Quantization-Aware Training (QAT-LoRA): fine-tune a model to be robust to a chosen
per-group hybrid config before it ships as that hybrid. `magicquant qat <model>
--config search_results.json --tier Q4 --dataset chat.jsonl --out adapters/` runs
`qat.train.run_qat(cfg)`, which freezes the base, fake-quantizes it to the
per-group schemes in the forward (`qat.fake_quant.fake_quant` — a differentiable
per-scheme quant→dequant with a straight-through estimator, validated against
libggml within a tolerance, NOT byte-exact), wraps routed `nn.Linear`s with
`qat.wrap.QATLinear` (fake-quants the merged base+LoRA each step) via
`wrap_model` + `TensorGroupClassifier`, and trains LoRA adapters with
completion-only loss. The per-group config is loaded by `qat.config.load_hybrid_config`
(search_results.json tier → `{group: ggml_type_name}`); HF→GGUF name mapping reuses
`gguf/source.py`'s `_HF_TO_GGUF_PATTERNS` via `qat.names.hf_to_ggml_name`. Heavy
training deps (torch/transformers/peft/trl/datasets) live in the `[qat]` extra;
the package `__init__` keeps `run_qat` lazily imported (from `qat.train`) so the
light surface only needs torch. Surfaced as Foundry's **QAT** pipeline stage.

## Critical Invariants

- **MXFP4 is ggml type 39** — native llama.cpp support. Never use a custom type ID.
- **converters.py is the single encoder source** — writer.py must not contain quantization logic.
- **Shapes are stored row-major** in ModelSource (same as GGUFReader convention). The writer reverses to ggml on-disk order. SafetensorsSource must NOT pre-reverse.
- **BF16 uses round-to-nearest-even**, not truncation. NOTE: BF16-designated
  groups are written to disk as **F16** (llama.cpp's compute graph has
  incomplete BF16 support); the writer logs a one-time warning. Out-of-F16-range
  values may become Inf/0.
- **Source models must be BF16/F16/F32** — the writer raises ValueError on pre-quantized sources.
- **LlamaCppTools is lazy** — the orchestrator works without llama.cpp installed (prediction-only mode).

## HF Tensor Name Mapping

SafetensorsSource strips `model.language_model.` prefix for multimodal models, then matches against `_HF_TO_GGUF_PATTERNS`. The arch mapping in `_build_gguf_metadata_from_config` is synced from llama.cpp's `convert_hf_to_gguf.py` and handles nested `text_config` for composite models (Qwen3.5, etc.).

## Known Limitations

- Quantized encoding is byte-identical to llama.cpp (it calls libggml directly),
  so there is NO MSE quality gap. imatrix-weighted quantization is plumbed through
  `encode_to_ggml_bytes(..., imatrix=)` but imatrix CAPTURE (PR4) is not yet
  implemented; K-quant/IQ encoding currently runs unweighted.
- Tokenizer reading only handles BPE (tokenizer.json). SentencePiece (.model) requires protobuf and is not implemented.
- The evolutionary search mostly rediscovers obvious configs (BF16 brain + MXFP4 FFN) for dense models. Its value comes from the measured search loop and MoE models with larger search spaces.
