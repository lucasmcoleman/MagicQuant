# MagicQuant Encoder Expansion — Design Spec

**Date:** 2026-05-04
**Status:** Approved (pending implementation plan)
**Owner:** Lucas Coleman
**Author of design:** brainstorming session

## Goals

1. Make the **Q2 and Q3 tier bands** in MagicQuant actually produce output (today they're empty because no scheme below ~4.25 bpw is registered).
2. Expand the search space to **all standard llama.cpp block formats** so MagicQuant's evolutionary search can find optimal hybrids across the full quality/size frontier — not just within the existing 7 schemes.
3. Eliminate the **~10–27% MSE gap** between MagicQuant's encoders and llama.cpp's, noted in `CLAUDE.md`.
4. Reduce per-scheme maintenance burden — adding a new scheme today touches 7+ places across 4 files.

## Non-goals

- Changing the orchestrator's evolutionary search algorithm itself.
- Changing Foundry's UI flow (apart from the already-shipped Q2 tier picker).
- Implementing `Q4_K_S`/`Q4_K_M`/`Q4_K_L` as separate schemes — these are llama.cpp **model-level recipes** that mix block formats, exactly what MagicQuant's search already does. Adding them as schemes is redundant.

## Final scope decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | C — full K-quant + IQ-quant set (16 new schemes) + retrofit existing 7 | Tool's premise is finding optimal hybrids; partial palettes undermine that |
| Verification | A — byte-for-byte parity vs `llama-quantize` | Trivially achievable under ctypes pivot (same underlying C function); simpler and stricter than dequant-within-tolerance |
| Architecture | A — refactor first as separate PR, then add encoders | Refactor has independent value; cleaner per-encoder PRs |
| Calibration | A — empirical bench, one-time | 2 hr upfront, amortized across all future runs |
| Phasing | B — K-quant batch → legacy Q-quants → IQ-quants → imatrix | Tier-floor fix ships fastest; IQ complexity isolated |
| Approach | D — ctypes binding to libggml | 3x faster than native impl, byte-parity by construction |
| libggml availability | Hard-required via `llama-cpp-python` install dep | Single code path, all schemes always work |

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ MagicQuant                                                     │
│                                                                │
│  ┌──────────────────┐   ┌──────────────────────────────────┐   │
│  │ schemes.py       │   │ ggml_binding.py  (NEW)           │   │
│  │  (single source  │←──│  - load_libggml() (discovery)    │   │
│  │   of truth for   │   │  - ggml_encode(weights, type,    │   │
│  │   all scheme     │   │      imatrix=None) → bytes       │   │
│  │   metadata)      │   │  - GGML_TYPE_IDS                 │   │
│  └─────▲────────────┘   └────────┬─────────────────────────┘   │
│        │ iterate registry        │ ctypes call                 │
│        │                         ▼                             │
│  ┌─────┴────────┐         ┌──────────────────┐                 │
│  │ predictor    │         │ converters.py    │                 │
│  │ survival     │         │  encode_to_ggml_ │                 │
│  │ probing      │         │  bytes() dispatch│                 │
│  │ orchestrator │         │  to ggml_encode  │                 │
│  └──────────────┘         └──────────────────┘                 │
└────────────────────────────────────────────────────────────────┘
                  │
                  │ ctypes
                  ▼
┌────────────────────────────────────────────────────────────────┐
│ libggml-base.so + libggml-cpu.so                               │
│   ggml_quantize_chunk(type, src, dst, start, nrows,            │
│                       n_per_row, imatrix)                      │
└────────────────────────────────────────────────────────────────┘
```

**Module responsibilities:**

- `magicquant/quant/schemes.py` — canonical registry. `QuantizationScheme` carries all attributes other modules need (`noise_factor`, `bits_per_weight`, `speed_multiplier`, `category`, `upgrade_neighbor`, `downgrade_neighbor`, `min_for_group_class`, `ggml_type_id`, `requires_imatrix`).
- `magicquant/quant/ggml_binding.py` — new ~180-line ctypes wrapper. Discovers libggml at runtime, exposes a single `ggml_encode()` function.
- `magicquant/quant/converters.py` — shrinks dramatically. Float-format encoders (BF16/F16/F32) stay native. Quantized formats route to `ggml_encode()`.
- `magicquant/evolution/predictor.py`, `survival.py`, `probing.py` — read scheme attributes from the registry instead of holding parallel dicts.
- `magicquant/orchestrator.py` — base-quant ranking pulls from registry.

## Library discovery

```
1. MAGICQUANT_LIBGGML_DIR env var (explicit user override)
2. Existing system llama.cpp build:
   - ~/llama.cpp/build/bin/
   - ~/llama.cpp-build/build/bin/
   - /usr/local/lib/
   - /home/linuxbrew/.linuxbrew/lib/
   - $LLAMACPP_PATH/build/bin/
3. llama-cpp-python bundled libs (always available since it's a hard dep)
```

`pyproject.toml` adds `llama-cpp-python>=0.3.0` as a hard install dependency. The wheel ships `libggml-base.so` + `libggml-cpu.so` for all major platforms; on platforms without prebuilt wheels, pip builds from source.

`ctypes.CDLL` loads with `RTLD_GLOBAL` so cross-library symbol resolution works (libggml-cpu calls into libggml-base for IQ codebooks).

A startup sanity check verifies `ggml_type_size(type_id)` returns the expected block size per `GGML_TYPE_SIZE`. If mismatched (e.g., ggml renumbered types in a future release), the binding refuses to start with a clear error pointing at `GGML_TYPE_IDS` in `ggml_binding.py`.

## Refactor (PR0)

**Extended `QuantizationScheme`:**

```python
@dataclass(frozen=True)
class QuantizationScheme:
    name: str                             # MagicQuant identifier
    ggml_type_name: str                   # ggml block type
    ggml_type_id: int                     # numeric ggml type enum
    bits_per_weight: float
    noise_factor: float                   # calibrated, lower = better quality
    speed_multiplier: float
    category: Literal["k_quant", "iq_quant", "legacy_q", "float", "mxfp4"]
    is_moe_optimized: bool = False
    requires_imatrix: bool = False
    min_for_group_class: dict[str, str] = ...
    upgrade_neighbor: Optional[str] = None
    downgrade_neighbor: Optional[str] = None
```

**Consumer-side reads from registry** — `predictor.py`, `survival.py`, `probing.py`, `orchestrator.py:585` lose their hardcoded scheme dicts:

```python
# predictor.py
def _noise_factor(self, scheme_name: str) -> float:
    return get_scheme_by_name(scheme_name).noise_factor

# survival.py
SCHEME_QUALITY_ORDER = sorted(get_all_schemes(), key=lambda s: s.noise_factor)
def _upgrade(s: str) -> str:
    return get_scheme_by_name(s).upgrade_neighbor or s

# orchestrator.py:585
key=lambda s: get_scheme_by_name(s).bits_per_weight
```

**Random-config weight arrays** in `survival.py:_generate_random_config` rewrite to be category-indexed (forward-compatible to new schemes), not positional.

**Behavior change:** zero. A regression test `tests/test_refactor_regression.py` pins `seed=42` and asserts `EvolutionarySurvivor.run_evolution()` produces an identical candidate sequence post-refactor.

## ctypes binding (`ggml_binding.py`)

```python
import ctypes
import os
from pathlib import Path
from typing import Optional
import numpy as np

GGML_TYPE_IDS = {
    "F32":     0,  "F16":     1,
    "Q4_0":    2,  "Q4_1":    3,
    "Q5_0":    6,  "Q5_1":    7,
    "Q8_0":    8,  "Q8_1":    9,
    "Q2_K":   10,  "Q3_K":   11,  "Q4_K":   12,  "Q5_K":   13,  "Q6_K":   14,
    "Q8_K":   15,
    "IQ2_XXS": 16, "IQ2_XS":  17, "IQ3_XXS": 18, "IQ1_S":   19,
    "IQ4_NL":  20, "IQ3_S":   21, "IQ2_S":   22, "IQ4_XS":  23,
    "BF16":   30,
    "IQ1_M":  29,
    "MXFP4":  39,
}

class LibggmlNotFound(RuntimeError): ...

def _discover_libggml() -> tuple[Path, Path]: ...

class _LibggmlHandle:
    def __init__(self):
        base_path, cpu_path = _discover_libggml()
        self._base = ctypes.CDLL(str(base_path), mode=ctypes.RTLD_GLOBAL)
        self._cpu = ctypes.CDLL(str(cpu_path), mode=ctypes.RTLD_GLOBAL)
        self._setup_signatures()
        self._verify_type_ids()  # startup sanity check
        self._base.ggml_quantize_init(ctypes.c_int(-1))

    def encode(self, weights, ggml_type, imatrix=None) -> bytes:
        flat = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
        type_id = GGML_TYPE_IDS[ggml_type]
        out_size = ggml_tensor_data_size(ggml_type, flat.size)
        dst = (ctypes.c_uint8 * out_size)()
        imat_ptr = (np.ascontiguousarray(imatrix, dtype=np.float32)
                    .ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    if imatrix is not None else None)
        actual = self._base.ggml_quantize_chunk(
            type_id,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(dst, ctypes.c_void_p),
            0, 1, flat.size,
            imat_ptr,
        )
        if actual != out_size:
            raise RuntimeError(f"...byte count mismatch...")
        return bytes(dst)

_HANDLE: Optional[_LibggmlHandle] = None

def get_handle() -> _LibggmlHandle: ...

def ggml_encode(weights, ggml_type, imatrix=None) -> bytes:
    return get_handle().encode(weights, ggml_type, imatrix)
```

**Process-wide singleton.** `_setup_signatures()` and `ggml_quantize_init()` happen once. ggml's IQ codebook tables initialize on first use.

**Memory safety.** Input buffer is `ascontiguousarray` so ctypes' `data_as` gives a stable pointer. Output is a stack-allocated ctypes array; `bytes()` copies before return.

## Encoder dispatch (`converters.py` after retrofit)

`converters.py` shrinks from ~960 lines to ~250 lines. Quantized formats route to `ggml_encode`; float passthroughs stay native:

```python
def encode_to_ggml_bytes(weights, ggml_type_name, imatrix=None) -> bytes:
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError("...floating-point...")
    flat = weights.astype(np.float32).flatten()
    if ggml_type_name == "BF16":  return _encode_f32_to_bf16(flat)
    if ggml_type_name == "F16":   return _encode_f32_to_f16(flat)
    if ggml_type_name == "F32":   return _encode_f32_to_f32(flat)
    if ggml_type_name not in GGML_TYPE_IDS:
        raise ValueError(f"No ggml encoder for type '{ggml_type_name}'")
    return ggml_encode(flat, ggml_type_name, imatrix=imatrix)
```

**Deletions:** all `_encode_ggml_*` functions (Q8_0, Q4_0, Q6_K, Q5_K, Q4_K, IQ4_NL, MXFP4, ~600 lines), helpers (`_pack_k4k5_scales`, `_pack_6bit/5bit/4bit`, `_optimize_symmetric_scale`, `_optimize_asymmetric_scale`, `_pad_to`, ~200 lines), `Quantizer` class quantize/dequantize branches except float passthroughs (~100 lines), `_GGML_ENCODERS` dispatch dict.

`Quantizer.quantize_weights()` callers that needed the numpy-level quantize-and-return path switch to a wrapper that calls `ggml_encode` then `ggml_dequantize_row_<type>` (also bound via ctypes).

## Calibration bench

**`tools/calibrate_noise_factors.py` (new):**

```
1. Load BF16 reference model (default: Llama-3.2-1B-Instruct, ~2.5 GB).
2. Load calibration corpus (default: wikitext-2-raw/wiki.test.raw).
3. Compute baseline perplexity at BF16. Cache.
4. For each scheme in get_all_schemes():
   a. Generate uniform config: {group: scheme.name for all groups}
   b. Build hybrid GGUF via existing create_hybrid_gguf()
   c. Run llama-perplexity on calibration corpus
   d. Record (name, ppl, ppl_loss, ppl_ratio)
   e. Delete temp GGUF
5. Normalize: Q8_0 = noise_factor 1.0; others scaled by ppl_loss ratio.
6. Write tools/calibration_results.json (committed to repo).
```

**Integration:** values from JSON are pasted into `schemes.py` constructors directly (no module-load file IO). Comments reference the calibration source so future contributors can re-bench.

**Compute budget:** ~24 perplexity runs × ~3–5 min each ≈ 1.5–2 hr. One-time cost during PR1.

**Edge cases:** schemes that fail to quantize a particular tensor get a heuristic placeholder. Schemes with non-finite perplexity get capped at 50.0.

**Re-runnable:** `python tools/calibrate_noise_factors.py --model PATH --output tools/calibration_results.json`. Deterministic given same model + corpus.

## Testing strategy

### Layer 1 — Unit tests (`tests/test_quantization_guards.py`, extended)

Existing tests stay. New parametrized cases for output sizes across all 23 schemes. Add:
- `test_libggml_discovery_finds_libs()`
- `test_libggml_discovery_raises_when_missing()`
- `test_ggml_type_ids_match_runtime_sizes()` — catches ID drift if ggml renumbers types

### Layer 2 — Encoder parity harness (`tests/integration/test_encoder_parity.py`, NEW)

```python
@pytest.fixture(scope="module")
def reference_tensor():
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.02, size=(2048, 2048)).astype(np.float32)

@pytest.mark.parametrize("scheme", ALL_QUANTIZED_SCHEMES)
def test_encoder_byte_for_byte_matches_llama_quantize(scheme, reference_tensor, tmp_path):
    """Build tiny F32 GGUF; quantize via MagicQuant's ctypes binding and via
    llama-quantize subprocess; assert output bytes are identical.

    This is the primary correctness test. Both code paths call the same
    underlying ggml C function, so any mismatch indicates a binding bug
    (wrong type_id, wrong nrows/n_per_row, non-contiguous memory, etc.)."""

@pytest.mark.parametrize("scheme", ALL_QUANTIZED_SCHEMES)
def test_encoder_round_trip_within_format_tolerance(scheme, reference_tensor):
    """Encode via ctypes → dequantize via ggml's dequantize_row_<type> →
    assert max-error <= scheme's format-defined tolerance (~1.5x format LSB).

    Sanity check that the format produces meaningful approximations of the
    input. Catches issues like all-zeros output, NaN propagation, or
    extreme quantization noise that the byte-parity test wouldn't surface
    (since it just verifies we agree with llama-quantize, which could in
    principle also be broken in the same way)."""
```

The harness needs a tiny 1-tensor F32 GGUF fixture, generated once with the `gguf` package and committed under `tests/fixtures/`.

**Note on byte parity:** because MagicQuant and `llama-quantize` both invoke the same `ggml_quantize_chunk` function from the same compiled `libggml-cpu.so`, byte-identical output is the natural expectation, not a stretch goal. The only way they diverge is via different code paths or arguments — exactly what we want the test to catch.

### Layer 3 — Smoke (`tests/integration/test_smoke_full_pipeline.py`, NEW)

```python
def test_q2_tier_actually_produces_output(tiny_bf16_model_fixture):
    """Full search → Q2 tier → GGUF generation. Asserts Q2 band populated and
    produced GGUF loads in libggml."""
```

### Layer 4 — Install-path tests (manual, README-documented)

```bash
python -m venv /tmp/fresh-venv
source /tmp/fresh-venv/bin/activate
pip install -e .
python -c "from magicquant.quant.ggml_binding import ggml_encode; \
           import numpy as np; \
           print(len(ggml_encode(np.random.randn(256).astype(np.float32), 'Q2_K')))"
# Expect: 84 bytes, no error
```

### Refactor regression (PR0)

`tests/test_refactor_regression.py` — pinned `seed=42`, asserts evolutionary search produces identical candidate sequence pre/post refactor.

### CI integration

Layers 1, 2, 3 run on every PR. Layer 4 is a checklist in the PR template. CI installs `llama-cpp-python` as part of test setup.

## Phased PR plan

### PR0 — Refactor: centralize scheme metadata
- Extend `QuantizationScheme`; populate for existing 7 schemes.
- Convert `predictor.py`, `survival.py`, `probing.py`, `orchestrator.py:585` to read from registry.
- Rewrite random-config weights to category-indexed.
- Add `tests/test_refactor_regression.py` with pinned-RNG snapshot.
- **Behavior change: zero.** All existing tests pass identically.
- Coding: ~20 min. Compute: ~5 min for tests.

### PR1 — libggml binding + K-quant batch + retrofit + calibration
- New `magicquant/quant/ggml_binding.py` (~180 lines).
- `pyproject.toml`: add `llama-cpp-python>=0.3.0` hard dep.
- `converters.py`: delete pure-Python `_encode_ggml_*` and helpers; replace with dispatch to `ggml_encode()`.
- Register **Q2_K, Q3_K** as new schemes.
- Retrofit Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, MXFP4, Q4_0 to ctypes path.
- Run calibration bench; commit `tools/calibration_results.json`; paste noise factors into `schemes.py`.
- Add Layer-2 + Layer-3 tests.
- **Tier band coverage after PR1:**
  - **Q3 band reliably populates.** Q3_K (3.44 bpw → ratio 0.215) lands in Q3 band; uniform-Q3_K configs and Q3_K-dominant hybrids land in the 0.16–0.22 range.
  - **Q2 band still effectively unreachable.** Q2_K alone (2.625 bpw → ratio 0.164) sits just outside the Q2 boundary (≤ 0.16). Any hybrid with non-Q2_K precision in the brain/sensitive groups pushes the ratio further up. **Full Q2 band coverage requires sub-Q2_K schemes from PR3 (IQ-quants).**
  - PR1 makes Q2_K available as a scheme — so configs in the upper-Q3-band can use it — but the Q2 band itself stays empty until PR3.
- Remove CLAUDE.md "10-27% MSE gap" caveat.
- Coding: ~30 min. Compute: ~1.5–2 hr (calibration bench).

### PR2 — Legacy Q-quants
- Register Q4_0 (already encoded but not registered as a scheme), Q4_1, Q5_0, Q5_1.
- Add to predictor/survival category-weighted tables.
- Add ggml_type_id mappings.
- Add encoder-parity tests.
- Coding: ~10 min. Compute: ~5 min for tests.

### PR3 — IQ-quant batch
- Register IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_S, IQ3_M, IQ4_XS.
- Confirm `ggml_quantize_init(-1)` is invoked at handle creation (initializes IQ codebooks).
- Extend `upgrade_neighbor` / `downgrade_neighbor` chains:
  `Q2_K → IQ2_S → IQ2_XS → IQ2_XXS → IQ1_M → IQ1_S` (downgrade chain)
  reverse for upgrade.
- Update random-config weights for `iq_quant` category.
- Sensitivity floors: embeddings/head shouldn't go below Q5_K — IQ-quants for "robust" groups only.
- Add encoder-parity tests for each.
- **Q2 band starts populating** with IQ2_*/IQ1_* available — search can now build configs with average bpw < 2.56.
- Smoke test: Q2-tier GGUF must have perplexity within 1.5x of baseline.
- Coding: ~25 min. Compute: ~10 min for tests.

### PR4 — Importance-matrix support
- New `magicquant/imatrix.py` — capture activation magnitudes per tensor by running a calibration corpus through the source model. Initially a subprocess wrapper around `llama-imatrix`.
- Wire `imatrix: Optional[np.ndarray]` parameter through `create_hybrid_gguf()` → `encode_to_ggml_bytes()` → `ggml_encode()`.
- Schemes where `ggml_quantize_requires_imatrix(type_id) == True`: orchestrator captures imatrix once at start of run, passes to encoders.
- Foundry UI: optional "Calibration dataset" input in the MagicQuant config panel.
- Foundry's `core/pipeline.py` gets `imatrix_dataset` field on `MagicQuantConfig`.
- Quality bench: re-run encoder-parity tests with imatrix; confirm IQ2/IQ3 outputs improve measurably.
- Coding: ~30 min. Compute: ~30 min for imatrix capture + tests.

### Total realistic timeline
- **~2 hr coding time** across all 5 PRs.
- **~2.5–3 hr compute waits**, dominated by the calibration bench in PR1.
- **End-to-end wall-clock: ~4–5 hr**, reducible to **~3 hr** if calibration is run in background while drafting PR2/PR3 code.

### Branch strategy
Each PR branches off `master` after the previous merges. No long-lived feature branches. Each PR is independently revertable until the next one builds on it.

## Migration / rollout

**Existing-user impact:**

| Audience | Impact |
|---|---|
| Existing Foundry users running `pip install --upgrade` | pip pulls `llama-cpp-python` automatically. ~150 MB disk. Existing scheme selections continue to work, now byte-identical to llama.cpp |
| Fresh Foundry installs from GitHub | `pip install -e .` succeeds. libggml bundled. First run "just works" |
| Power users with pre-existing local llama.cpp | Discovery prefers system paths. Their existing libggml is used, no double-install |

**Backward compatibility:**
- `encode_to_ggml_bytes(weights, ggml_type_name)` signature stays compatible (new optional `imatrix=None`).
- All 7 existing scheme names continue valid.
- `search_results.json`/`tiered_survivors.json` shape unchanged.
- Foundry's UI continues to read MagicQuant outputs the same way.

**Rollback risk:**
- Pre-merge encoder-parity tests catch byte-layout / quality regressions.
- Calibration JSON is committed alongside the noise-factor edits — reverting PR1 reverts everything coherently.
- Hot fix path for a future ggml ABI break: pin `llama-cpp-python` to a known-good version.

**ggml ABI drift:** startup sanity check in `_LibggmlHandle.__init__` catches type-ID renumbering; refuses to start with a clear error.

**Documentation updates:**
- `MagicQuant/README.md` — note `llama-cpp-python` is now a hard dep; explain discovery order.
- `MagicQuant/CLAUDE.md` — remove "10-27% MSE gap" caveat; update architecture description; note `ggml_binding.py`.
- `Foundry/FOUNDRY_MAP.md` — Stage 3 now requires libggml at runtime.
- `Foundry/CHANGELOG.md` and `MagicQuant/CHANGELOG.md` — entries per PR.

## Future work (not in scope for this project)

### Per-architecture noise-factor calibration tables

The calibration bench in PR1 calibrates `noise_factor` values against a single reference model (default Llama-3.2-1B). Cross-architecture generalization is a known limitation — Qwen3.5, MoE, and Mistral models may show systematic prediction bias.

**When to revisit:** if the search shows systematic misbehavior on non-Llama architectures, or when starting a major project that targets a specific architecture (Qwen, Mistral, MoE).

**Approach:**
- Run `tools/calibrate_noise_factors.py --model <representative-bf16-model> --output tools/calibration_<arch>.json` for each major architecture family.
- Extend `schemes.py` to load arch-specific noise factors based on detected architecture (e.g., from the source model's `config.json` `model_type` field).
- Add fallback to default Llama calibration when arch is unrecognized.

### Native imatrix capture (replacing llama-imatrix subprocess)

PR4 wraps `llama-imatrix` as a subprocess for the initial implementation. A future PR could implement imatrix capture natively via PyTorch hooks during a forward pass on the calibration corpus, eliminating the subprocess hop.

### Integrate Q4_K_S / Q4_K_M / Q4_K_L recipe presets as starter populations

The K-quant `_S/_M/_L` variants are llama.cpp's curated mixing recipes (e.g., Q4_K_M = Q4_K throughout + Q6_K for `output.weight`). These could be added to `survival.py:_initialize_population` as additional named seeds, giving the search good starting points that match llama.cpp's standard outputs.

## Acceptance criteria

The project is complete when:

1. PR0–PR4 are all merged to `master` on both `MagicQuant` and (for PR4) `Foundry`.
2. The Q3 tier band reliably produces output GGUFs after PR1 merges.
3. The Q2 tier band reliably produces output GGUFs after PR3 merges.
4. All 7 existing schemes produce byte-identical output to `llama-quantize` (eliminating the existing MSE gap).
5. All 16 new schemes pass the encoder-parity harness against `llama-quantize`.
6. `pip install -e .` from a fresh clone of Foundry succeeds and produces a working quantization run with no manual configuration.
7. `tools/calibration_results.json` is committed; `noise_factor` values in `schemes.py` reference it.
8. `MagicQuant/CLAUDE.md` no longer mentions the "10-27% MSE gap."
