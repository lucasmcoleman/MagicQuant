# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MagicQuant creates hybrid GGUF files with per-tensor-group quantization. Different tensor groups (embeddings, attention, FFN) get different quantization schemes based on sensitivity. MXFP4 (ggml type 39) is the primary compression scheme for robust layers (FFN, SSM). The methodology is from [magiccodingman](https://github.com/magiccodingman/MagicQuant-Wiki).

## Commands

```bash
pip install -e .                    # Install (editable)
pip install -e ".[yaml,dev]"        # With optional deps (yaml + pytest + gguf)
pip install -e ".[qat]"             # QAT training stack (torch/transformers/peft/accelerate)
magicquant analyze model.gguf       # Inspect tensor groups
magicquant search model.gguf --rounds 3  # Measured evolutionary search
magicquant search model.gguf --rounds 0  # Prediction-only (no llama.cpp needed)
magicquant generate model.gguf --tiers Q4,Q5,Q6  # Generate hybrids
magicquant qat ./model --config search_results.json --tier Q4 --dataset chat.jsonl --out adapters/  # QAT-LoRA
```

### Tests

Two virtualenvs on the dev box; NEVER use bare `python`/`pytest` (PATH resolves to
an unrelated shim without pytest — it exits 0 having run nothing):

```bash
.venv/bin/python -m pytest tests/ -q            # main suite (~1090 tests, ~15s)
.venv/bin/python -m pytest tests/test_writer.py::test_name -q   # single test
.venv-qat/bin/python -m pytest tests/ -q        # torch env: runs the ~210 QAT tests too
.venv/bin/ruff check --select F magicquant/ tools/ tests/   # BLOCKING in CI — keep at zero
```

Torch-gated tests (`test_qat_*`, `test_fake_quant`, LoRA differential) skip in
`.venv` and run in `.venv-qat`. Any change under `magicquant/qat/` must be
verified in `.venv-qat`. If `.venv-qat` shows 3 failures in
`test_probe_resolution.py::TestPredictorTracking`, that is a scipy-absent
environment gap, not a code bug (`pip install scipy` there greens it).

Key guard tests to know about:
- `tests/test_refactor_regression.py` — seed-pinned evolutionary-search golden fixture;
  REGENERATE `tests/fixtures/refactor_regression_seed42.json` whenever noise_factors,
  group sets, the sensitive-group floor clamp, or early-stop change behavior.
- `tests/integration/test_encoder_parity.py` — byte-for-byte vs `llama-quantize`
  (skips when `gguf` or `llama-quantize` is unavailable).
- `tests/test_v2_search_characterization.py` — end-to-end pins on `run_budget_search`,
  including the EXACT top-level key set of `v2_results.json` (an additive key there is
  a deliberate test failure — decide, don't just append).
- `tests/test_writer_gguf_constants.py` — drift tripwires for every table derived from
  the installed `gguf` package (`_GGUF_TYPE_*`, `_ftype_map`).
- `tests/test_naming.py` — round-trips `naming.config_key` through
  `reselect_tiers._parse_key` (persisted-format contract).
- `tests/test_orchestrator_measurement.py` has a test asserting OSError ESCAPES the
  llama.cpp subprocess triage — never widen that exception contract.

## v2 budget search (`--algo v2`, docs/redesign.md)

`magicquant search <model> --algo v2 --budget-gb <B>` runs the 2026-07
algorithmic redesign (`magicquant/v2/`): per-tensor × per-scheme distortion
table (imatrix-weighted encode+decode through libggml — needs
`ggml_binding.ggml_decode`; CPU-only, cached), optional chunk-capped group
probes fitting per-group amplification κ, then an exact multiple-choice
knapsack (`v2/allocate.py`) that allocates per-TENSOR schemes to the byte
budget and emits the whole predicted quality-size frontier in one solve. Only
2–3 frontier anchors get full-corpus perplexity verification. Outputs
`v2_results.json` + `frontier.json` + the budget GGUF (via the writer's
per-tensor `"tensors"` override key). Failure doctrine: measurements either
succeed or are recorded as failures; probes never fall back to fabricated
heuristics (strict; `--allow-partial-probes` for imputed-median κ, loudly
tagged). `--target-profile q4nx` restricts choices to Q4_0/Q4_1/Q8_0/MXFP4
for the FLM NPU converter. `--enable-iq` adds the IQ family (minus sub-2-bit)
to the choice set; imatrix-requiring members drop out automatically when no
imatrix is active; the q4nx profile and explicit scheme overrides are never
touched by it. The v1 evolutionary path is the untouched default; v1's
`run_measured_search` constructs its `SensitivityProber` with `strict=True`
(failed probe raises `ProbeMeasurementError`; prediction-only keeps the
heuristic fallback).

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

New-architecture ingestion: the GGUF-source path needs no per-arch lists (tensor-group
classification is suffix/pattern-based and generalizes — proven day-one on muse-glimmer).
The safetensors path gates on `arch_map` (model_type → GGUF arch) and
`_HF_TO_GGUF_PATTERNS`, both deliberately mirrored from llama.cpp's converter — an
unknown model_type FAILS LOUDLY rather than guessing (the qwen3_5 wrong-namespace
incident is why). `tools/draft_upstream_sync.py` + the Upstream Sync PR workflow draft
the mirror updates when upstream moves; merging the drafted PR is a human decision
because each arch_map entry is a verified claim, not a table row.

### Quantization (converters.py + ggml_binding.py)

`encode_to_ggml_bytes(weights, ggml_type_name, imatrix=None, n_per_row=None)` is the
public API. Float passthroughs (BF16/F16/F32) are native numpy; ALL quantized types
route through `magicquant.quant.ggml_binding.ggml_encode`, a ctypes binding to
`libggml` that calls `ggml_quantize_chunk` — the SAME function `llama-quantize`
uses. Output is therefore **byte-identical** to llama.cpp (proven by
`tests/integration/test_encoder_parity.py`). There are NO numpy K-quant
encoders (the May 2026 refactor deleted them).

Single-source-of-truth facts live in `quant/ggml_facts.py`: type ids, block/type
sizes, and `expected_size()` (the one encoded-size formula — converters and the
binding delegate to it). `ggml_binding._verify_type_ids` raises at first encode if
the loaded libggml renumbers/resizes any type; a warning-level cross-check also
compares each scheme's static `requires_imatrix` against the live lib.

### Tensor groups (tensor_groups.py)

Groups: E (embeddings), H (head/MTP), Q (query), K (key/value), O (attention output), U (FFN up/gate), D (FFN down), S (SSM/linear attention), N (norms), V (vision), X (MoE experts), R (MoE router). Classification is regex-based on GGUF tensor names; explicit patterns are checked before suffix heuristics, first match wins.

### QAT (magicquant/qat/, optional `[qat]` extra)

Quantization-Aware Training (QAT-LoRA): fine-tune a model to be robust to a chosen
per-group hybrid config before it ships as that hybrid. `magicquant qat <model>
--config search_results.json --tier Q4 --dataset chat.jsonl --out adapters/` runs
`qat.train.run_qat(cfg)`, which freezes the base, fake-quantizes it to the
per-group schemes in the forward (`qat.fake_quant.fake_quant` — a differentiable
per-scheme quant→dequant with a straight-through estimator, validated against
libggml within a tolerance, NOT byte-exact), wraps routed `nn.Linear`s with
`qat.wrap.QATLinear` via `wrap_model` + `TensorGroupClassifier`, and trains LoRA
adapters with completion-only loss. Heavy deps (torch/transformers/peft/accelerate)
live in the `[qat]` extra; `run_qat` is lazily imported so the light surface only
needs torch. Validated: 38.1% quantization-loss recovery in the shipped GGUF
(details: `docs/qat.md`). Resume identity: `_config_hash` covers model, schemes
(group and tensor), base LoRA rank/alpha, and expert config — a mismatched
checkpoint refuses to resume (`--no-resume` to start fresh).

transformers-version compat seams (transformers is FLOORED, not pinned — CI's
qat-tests job installs latest on purpose, so upstream API removals surface there
first): `_from_pretrained` (dtype= vs torch_dtype=), `_build_training_args`
(warmup_ratio vs float warmup_steps), `_ids_from_chat_template` (BatchEncoding vs
list). When a new drift class appears, add the same try/TypeError-fallback shape
and a signature-fake test for both paths.

## ROCmFPX fork schemes (opt-in, fork-only)

The AMD-native ROCmFPX types (`ROCMFP3/4/6/8`, ggml ids 100–104 from the
[ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) llama.cpp fork) are
registered in `quant/schemes.py` (category `rocmfpx`) but **excluded from the
default search** unless `enable_rocmfpx=True`. Encoding requires a ROCmFPX
build's libggml: `ggml_binding` probes support by NAME via
`ggml_type_from_name` (never passing an out-of-range enum id to a stock lib)
and exposes `handle.rocmfpx_supported` / `handle.supports(type)` (intentional
public surface). Point `MAGICQUANT_LIBGGML_DIR` at a fork build's bin dir.
GGUFs containing these types load only on the fork, not stock llama.cpp.

## Critical Invariants

- **A tier is a SIZE BAND, not a recipe.** A "Q5" is whatever mix of schemes
  landed in the Q5 band with the lowest measured loss; it may contain zero
  Q5_K tensors. Bands come from `quant/tiers.py` as a ratio to the BF16
  baseline. Never grade a build by which schemes it contains — size and
  measured quality are the only criteria. `tools/reselect_tiers.py`
  re-derives a finished run's ladder from stored measurements.
- **Three distinct imatrix relationships**, and conflating them breaks things:
  `requires_imatrix` (cannot encode without one — IQ1/IQ2 family),
  `uses_imatrix` (consumes one if offered — every K-quant and the IQ family,
  all fine without), and `IMATRIX_DEPENDENT_SCHEME_NAMES` (encodes fine but
  only COMPETITIVE with one — IQ4_NL lost all 11 measured comparisons without
  calibration despite better isolated reconstruction error). Filtering on
  `uses_imatrix` would wrongly drop Q4_K_M/Q5_K/Q6_K too.
- **The neighbour walk SKIPS imatrix-dependent schemes, it does not stop at
  them** (IQ4_NL sits mid-chain between Q5_K and MXFP4_MOE).
- **The calibration corpus must never be the perplexity eval corpus.**
  `enable_imatrix` refuses when both resolve to the same file;
  `tools/build_calib_corpus.py` asserts disjointness.
- **MXFP4 is ggml type 39** — native llama.cpp support. Never a custom type ID.
- **converters.py is the single encoder source** — writer.py must not contain quantization logic.
- **Shapes are stored row-major** in ModelSource (GGUFReader convention). The writer reverses to ggml on-disk order. SafetensorsSource must NOT pre-reverse.
- **BF16 uses round-to-nearest-even**, not truncation. BF16-designated groups
  are written to disk as **F16** (llama.cpp compute-graph limitation); the
  writer logs a one-time warning.
- **Source models must be BF16/F16/F32** — the writer raises on pre-quantized sources.
- **The writer never copies `split.*` KV keys** (`_SPLIT_KV_KEYS`): GGUFReader
  flattens KVs to plain ints, the writer would re-type them u32, and llama.cpp
  strictly type-checks `split.count` as u16 — a copied key makes every emitted
  artifact unloadable. Single-file artifacts must not carry split metadata at all.
- **Persisted-format contracts**: `naming.config_key`'s `group:scheme|...` string
  is the measurement key in `search_results.json`/checkpoints and is parsed back
  by external tools — never change its format. `_serialize_measurement`
  reproduces each artifact's exact historical key ORDER (contractual); new
  fields append at the tail of BOTH orders. `tiered_survivors` and `tiered`
  stay separate keys (external consumers read them with opposite precedence).
- **`measurement_invalid` entries are inert by filter, everywhere.** Every
  consumer of `self._measured` filters them (`.get()` truthiness, never
  `is True`). Timeout disclosures (`measurement_timeout`/`timeout_leg`) and
  scored-without-KL markers (`kl_timeout`) ride on that contract.
- **Measured searches fail fast on an incompatible llama.cpp build**: before any
  measurement, the resolved build's libllama is byte-scanned for the source
  GGUF's arch literal (`LlamaBinaryArchError` at t+0 instead of a baseline
  failure 40 minutes in). Definitive absence fails; undeterminable proceeds.
  `MAGICQUANT_SKIP_ARCH_CHECK=1` is the escape hatch. The resolved binary is
  persisted (`llamacpp_binary`) and a resume with a different binary is a
  condition mismatch.
- **LlamaCppTools is lazy** — the orchestrator works without llama.cpp installed
  (prediction-only mode). Tool discovery order (dirs-outer/names-inner in
  `_find_tool_in_dirs`) is load-bearing: a legacy root binary wins over build/bin.
- **Measurement subprocess timeouts are size-aware**, never flat: 4 MB/s floor
  scaled by artifact bytes, 2× for KL legs, `MAGICQUANT_SUBPROCESS_TIMEOUT`
  overrides the base. The triage helper catches EXACTLY
  CalledProcessError/TimeoutExpired — OSError must propagate (pinned by test).

## Environment variables

Runtime settings flow through `config.py` (`MagicQuantSettings`, pydantic, env
prefix `MAGICQUANT_`, `.env` supported) — CLI flags override env, env overrides
defaults; that file is the full registry. Operational knobs read directly from
the environment (not settings): `MAGICQUANT_PPL_CHUNKS` (cap perplexity/KL
chunk count), `MAGICQUANT_NGL` / `MAGICQUANT_THREADS` (llama.cpp subprocess
flags), `MAGICQUANT_SUBPROCESS_TIMEOUT`, `MAGICQUANT_SKIP_ARCH_CHECK`,
`MAGICQUANT_LIBGGML_DIR` (fork libggml), `MAGICQUANT_ALLOW_UNVALIDATED_ARCH`
and `MAGICQUANT_ALLOW_DEQUANT_SOURCE` (escape hatches for the loud gates —
each logs what it is bypassing).

## CI, releases, upstream sync

- `ci.yml`: test matrix floor is **Python 3.10** — no 3.11+-only stdlib
  (`tomllib` needs the `tomli` fallback pattern). The `qat-tests` job installs
  **latest** transformers deliberately (drift detection); when it breaks, adapt
  via the compat-seam pattern, don't pin. `ruff check --select F` is BLOCKING
  (keep F findings at zero); full ruff stays advisory.
- Releases are tag-triggered (`release.yml`): the human cuts `[Unreleased]`
  into a dated CHANGELOG section, bumps `pyproject.toml`, tags `vX.Y.Z`, and
  pushes; CI validates via `tools/release_check.py` (version equality, strict
  dated section, Unreleased-above, duplicate detection, 120k notes cap), then
  builds and publishes the GitHub Release. No PyPI. CHANGELOG entries follow
  Keep a Changelog and record files touched + how verified, per commit.
- `upstream-watch.yml` reports NEW llama.cpp drift (delta vs
  `tools/upstream_baseline.json`); `upstream-sync-pr.yml` drafts one rolling
  PR of mechanical arch_map/pattern candidates (`tools/draft_upstream_sync.py`)
  with a manual checklist for everything below high confidence. Merging that
  PR is deliberately human.
- Historical audit docs live in `docs/history/`; `AUDIT_2026-07-01.md` at the
  repo root is the live findings tracker. `docs/cleanup-2026-08-plan.md` is the
  audit trail for the 2026-08 cleanup pass.

## HF Tensor Name Mapping

SafetensorsSource strips `model.language_model.` prefix for multimodal models, then matches against `_HF_TO_GGUF_PATTERNS`. The arch mapping in `_build_gguf_metadata_from_config` is synced from llama.cpp's `convert_hf_to_gguf.py` and handles nested `text_config` for composite models (Qwen3.5, etc.); transformers-5 nested `rope_parameters` is handled generically. Vision-encoder and MTP tensors are skipped (they belong in a separate mmproj GGUF).

## Known Limitations

- imatrix support is implemented (`magicquant imatrix model.gguf -f corpus`,
  threading per-tensor importance with true row width). Weighting is USED by
  the K-quants and the IQ family; **MXFP4/ROCmFPX/float/legacy Q8_0 ignore it
  by ggml design**. Non-256-divisible rows fall back to a block-32 quant
  (MXFP4 low-bit / Q8_0 high-bit) rather than F32; F32 only for SSM group `S`
  or non-32-divisible rows. Per-expert MoE imatrix slices are rejected with a
  clear error.
- Tokenizer reading only handles BPE (tokenizer.json). SentencePiece (.model) requires protobuf and is not implemented.
- Multi-part (`-00001-of-N`) GGUF sources are not stitched — merge with
  `llama-gguf-split --merge` first.
- The evolutionary search mostly rediscovers obvious configs (BF16 brain + MXFP4 FFN) for dense models. Its value comes from the measured search loop and MoE models with larger search spaces.
