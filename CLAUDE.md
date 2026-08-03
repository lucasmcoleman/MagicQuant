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

## v2 budget search (`--algo v2`, docs/redesign.md)

`magicquant search <model> --algo v2 --budget-gb <B>` runs the 2026-07
algorithmic redesign (`magicquant/v2/`): per-tensor × per-scheme distortion
table (imatrix-weighted encode+decode through libggml — needs the new
`ggml_binding.ggml_decode`; CPU-only, cached), optional chunk-capped group
probes fitting per-group amplification κ, then an exact multiple-choice
knapsack (`v2/allocate.py`: convex-hull Lagrangian greedy + polish) that
allocates per-TENSOR schemes to the byte budget and emits the whole
predicted quality-size frontier in one solve. Only 2–3 frontier anchors get
full-corpus perplexity verification. Outputs `v2_results.json` +
`frontier.json` + the budget GGUF (built via the writer's per-tensor
`"tensors"` override key). Failure doctrine: measurements either succeed or
are recorded as failures (`v2_results.json "failures"`); probes never fall
back to fabricated heuristics (strict; `--allow-partial-probes` for imputed-
median κ, loudly tagged). `--target-profile q4nx` restricts choices to
Q4_0/Q4_1/Q8_0/MXFP4 so the output packs losslessly for the FLM NPU
converter. Q4_0/Q4_1 exist in the registry for this but are excluded from
v1 sampling (`LEGACY_Q4_SCHEME_NAMES`), keeping the seed-pinned fixture
stable. The v1 evolutionary path is the untouched default; the one
deliberate v1 behavior change from this work: `run_measured_search` now
constructs its `SensitivityProber` with `strict=True`, so a failed probe
raises `ProbeMeasurementError` instead of silently poisoning the run with
heuristic sensitivities (prediction-only search keeps the fallback).

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

**Validated.** In a confound-controlled run on Qwen2.5-0.5B base with an
aggressive Q4_K-attention/MXFP4-FFN hybrid: bf16 PPL 16.35, plain quant 19.54
(+3.19 damage), quant+QAT 15.13, and a bf16+identical-LoRA control 13.46. Holding
the LoRA's domain adaptation fixed on both arms, the quant-vs-bf16 gap shrank from
+3.19 to +1.67 — **QAT recovered 47.5% of the quantization loss beyond plain LoRA
domain-adaptation** (torch fake-quant space; 45–66% across tiers in the recovery
curve). The same design repeated end-to-end through the real libggml pack +
llama-perplexity gives **38.1% recovery in the shipped GGUF** (B 12.56 / Q 14.21 /
QT 11.94 / BT 10.92) — the drop vs fake-quant space is the measured cost of the
train/ship approximation. The GGUF pack of the final model is exact-ggml
(`merge_qat_adapters` → `magicquant generate`, byte-identical to llama.cpp) while
training uses the faithful-but-approximate torch fake-quant. Multimodal (Gemma-3/4)
and bf16 bases are supported (QAT targets the text decoder). Full write-up:
`docs/qat.md`.

## ROCmFPX fork schemes (opt-in, fork-only)

The AMD-native ROCmFPX types (`ROCMFP3/4/6/8`, ggml ids 100–104 from the
[ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) llama.cpp fork) are
registered in `quant/schemes.py` (category `rocmfpx`) but **excluded from the
default search** — their per-class sampling mass is zero and
`_generate_random_config` drops them unless `EvolutionarySurvivor(...,
enable_rocmfpx=True)` (surfaced as `run_full_search/run_measured_search(...,
enable_rocmfpx=True)`, and Foundry's `--magicquant-rocmfpx`). Enabling adds
ROCmFPX seeds + sampling mass so the search compares them head-to-head.

Encoding requires a ROCmFPX build's libggml: `ggml_binding` probes support by
NAME via `ggml_type_from_name` (never passing an out-of-range enum id to a
stock lib), exposes `handle.rocmfpx_supported` / `handle.supports(type)`, and
`encode()` raises a clear error for a fork type on a stock lib. Point
`MAGICQUANT_LIBGGML_DIR` at `…/ROCmFPX/build-strix-rocmfp4/bin`. GGUFs
containing these types load only on the fork, not stock llama.cpp.

## Critical Invariants

- **A tier is a SIZE BAND, not a recipe.** A "Q5" is whatever mix of schemes
  landed in the Q5 band with the lowest measured loss; it may contain zero
  Q5_K tensors. Bands come from `quant/tiers.py` (`classify_tier`,
  `TIER_BOUNDARIES`) as a ratio to the BF16 baseline. Never grade a build by
  which schemes it contains — size and measured quality are the only criteria.
  Four published models once shipped a uniform Q6_K labelled "Q5" because the
  v1 Q5 band ran to ratio 0.45; `tools/reselect_tiers.py` re-derives a finished
  run's ladder from stored measurements to catch this.
- **Three distinct imatrix relationships**, and conflating them breaks things:
  `requires_imatrix` (cannot encode without one — IQ1/IQ2 family, never
  sampled), `uses_imatrix` (consumes one if offered — true of every K-quant,
  all fine without), and `IMATRIX_DEPENDENT_SCHEME_NAMES` (encodes fine but only
  COMPETITIVE with one). IQ4_NL is the third case: without calibration it lost
  all 11 measured comparisons across two 27B models, 3-20x worse than same-bpw
  MXFP4/Q4_K_M, despite *better* isolated weight-reconstruction error (0.051 vs
  MXFP4's 0.101 relative RMS) — its lookup table places levels to minimise
  UNWEIGHTED error, optimising a metric perplexity does not care about.
  Filtering on `uses_imatrix` would wrongly drop Q4_K_M/Q5_K/Q6_K too.
- **The neighbour walk SKIPS imatrix-dependent schemes, it does not stop at
  them.** IQ4_NL sits mid-chain (Q5_K <-> IQ4_NL <-> MXFP4_MOE), so the
  end-of-chain treatment used for `requires_imatrix` would strand Q5_K with no
  downgrade and MXFP4_MOE with no upgrade.
- **The calibration corpus must never be the perplexity eval corpus.**
  Calibrating on the text a run is scored against makes every measured loss
  optimistic with nothing in the output showing it. `enable_imatrix` refuses
  when both resolve to the same file; `tools/build_calib_corpus.py` asserts
  disjointness when rebuilding (currently 0.00000% shared 8-grams).
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
  so there is NO MSE quality gap. **imatrix support (M4) is implemented**:
  `magicquant imatrix model.gguf -f corpus` captures via `llama-imatrix`
  (`magicquant/imatrix.py`), `create_hybrid_gguf(..., imatrix=path_or_dict)`
  threads per-tensor importance vectors to the encoder with the true row width,
  and Pass 1 hard-errors if a target type REQUIRES an imatrix (IQ1/IQ2 family,
  once PR3 registers them) but none was provided. Weighting is USED by the
  K-quants (Q2_K–Q6_K) and IQ4_NL; **MXFP4 and Q8_0 ignore it by ggml design**
  (absmax/E8M0 scaling has no importance input). A row whose width isn't a
  multiple of the requested K-quant's 256-block falls back to a block-32
  quant (MXFP4 for low-bit targets, Q8_0 for high-bit) rather than F32; F32
  is used only for SSM/linear-attention operands (group `S`, which llama.cpp
  requires in F32) or rows that aren't even 32-divisible
  (`writer._block32_fallback`). Either way the fallback target already
  ignores imatrix by ggml design, so these tensors are unaffected regardless
  of which fallback type they land on. Limits: per-expert MoE
  imatrix slices are rejected (clear error, drop the imatrix for `*_exps`
  tensors); orchestrator/search auto-capture is deferred to PR4-full with PR3.
- Tokenizer reading only handles BPE (tokenizer.json). SentencePiece (.model) requires protobuf and is not implemented.
- The evolutionary search mostly rediscovers obvious configs (BF16 brain + MXFP4 FFN) for dense models. Its value comes from the measured search loop and MoE models with larger search spaces.
