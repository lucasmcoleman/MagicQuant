> **SUPERSEDED — pre-libggml binding (April 2026).** The audited commit
> predates the May 2026 ctypes refactor that replaced all numpy K-quant
> encoders with a direct `libggml` binding (byte-identical to llama.cpp). The
> encoder CRITICAL/HIGH items here (collapse penalty, K-quant clamping, MSE
> gap, three tier systems, dtype guard/timeout, file-size baseline) are all
> fixed at HEAD. See `AUDIT_2026-06-09.md` / `AUDIT_FIXPLAN_2026-06-09.md` for
> the current state. Kept for historical reference only.

# MagicQuant Audit Report

**Date:** 2026-04-03
**Auditor:** Claude Opus 4.6 (1M context)
**Commit state:** master (up to date) + 1 uncommitted change in `magicquant/gguf/writer.py`
**Scope:** Full codebase audit of all 13 key modules

---

## Executive Summary

MagicQuant is a well-structured Python package for creating hybrid GGUF files with per-tensor-group quantization. The codebase is generally clean, with a clear separation of concerns between the GGUF reader/writer, quantization encoders, evolutionary search, and orchestration layers. However, the audit found **1 critical bug**, **5 high-severity issues**, **14 medium-severity issues**, and **10 low-severity issues** across the categories of code quality, fragility/reliability, architecture, and packaging.

---

## CRITICAL Issues

### C-1. No dtype guard before quantization encoding

**File:** `/server/programming/MagicQuant/magicquant/gguf/writer.py`, line 205-208
**File:** `/server/programming/MagicQuant/magicquant/quant/converters.py`, line 928-948

The pipeline reads tensors via `source.read_tensor_f32(name)` (writer.py:205) and passes the result directly to `encode_to_ggml_bytes(f32, target)` (writer.py:208). While `encode_to_ggml_bytes` does call `weights.astype(np.float32)` at line 947 of converters.py, the _read_encode_worker function at writer.py:207 also has a fallback path `f32.view(np.uint8).tobytes()` (writer.py:210) that executes when `can_decode` is False. This path does no dtype validation at all.

The core problem: `read_tensor_f32()` nominally returns float32, but there is no assertion or guard at any call site. Specifically:

1. **GGUFSource.read_tensor_f32** (source.py:122-142) returns `None` for quantized types, but returns raw `np.frombuffer` data for F32/F16/BF16. If the stored ggml_type enum is incorrect or a new type is added, the wrong decoding path executes silently.

2. **SafetensorsSource.read_tensor_f32** (source.py:742-767) has a catch-all `else` branch at line 767: `np.frombuffer(buf, dtype=np_dtype).astype(np.float32)` -- this handles I8/I16/I32/I64/F64 types, which are integer or double tensors not appropriate for quantization, yet they get silently cast to float32 and quantized.

3. **LoRAMergedSource.read_tensor_f32** (source.py:900-919) reshapes the merged delta and adds it to base_f32, but if the shapes are mismatched due to LoRA configuration issues, numpy will raise a confusing error rather than a clear diagnostic.

4. **encode_to_ggml_bytes** (converters.py:928-948) does `weights.astype(np.float32).flatten()` which silently succeeds on any numeric numpy array, but would produce garbage if the input were already quantized integer blocks being cast as if they were floating-point values.

**Impact:** Silently corrupt output GGUF. A user who feeds an unexpected source format would get a model that loads but produces nonsense inference results, with no error or warning.

**Recommendation:** Add an explicit `assert f32.dtype == np.float32` guard in `_read_encode_worker` after `source.read_tensor_f32()` returns (writer.py:206), and add a similar guard at the entry of `encode_to_ggml_bytes`. Log a clear error if dtype is wrong.

---

## HIGH Severity Issues

### H-1. subprocess.run without timeout for quantize_model

**File:** `/server/programming/MagicQuant/magicquant/utils/llamacpp.py`, lines 173-178

The `quantize_model()` method calls `subprocess.run()` with no `timeout` parameter. A hung llama-quantize process would block MagicQuant forever. In contrast, `calculate_perplexity()` at line 231 correctly uses `timeout=600`.

```python
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    check=True
    # missing: timeout=...
)
```

**Impact:** MagicQuant can hang indefinitely during model quantization.

### H-2. subprocess.run without timeout for _find_llamacpp

**File:** `/server/programming/MagicQuant/magicquant/utils/llamacpp.py`, lines 54-60

The `which`/`where` subprocess call in `_find_llamacpp()` has no timeout. While this is unlikely to hang in practice, it violates defensive coding principles for external process calls.

### H-3. Broad except blocks swallow errors silently

Multiple locations catch `Exception` broadly, often printing a message but not logging or re-raising, which hides root causes during debugging:

| File | Line | Context |
|------|------|---------|
| `orchestrator.py` | 67 | `llama_tools` property -- prints WARNING but returns None, masking the actual error |
| `orchestrator.py` | 332 | `_build_candidate` -- prints "Build failed" but swallows the traceback |
| `orchestrator.py` | 508 | `generate_hybrid_model` -- prints "Failed" but loses traceback |
| `probing.py` | 228 | `_real_probe` -- falls back to heuristic silently on any exception |
| `__main__.py` | 86 | `cmd_probe` -- catches all exceptions from LlamaCppTools init with no logging |

**Impact:** When things go wrong (corrupt files, I/O errors, numpy shape mismatches), users see generic messages with no diagnostic information. Debugging requires adding print statements.

### H-4. Mutable default argument `adapter_path=None` in method signature

**File:** `/server/programming/MagicQuant/magicquant/gguf/writer.py`, lines 259, 264
**File:** `/server/programming/MagicQuant/magicquant/gguf/writer.py`, line 557

The `create_hybrid_gguf` method and convenience function use `adapter_path: str = None` without an `Optional` type annotation. While `None` is immutable and this is technically safe, the type annotation is incorrect -- it should be `Optional[str] = None`. This is a consistency issue rather than a runtime bug, but it causes type checkers to report errors.

### H-5. Worker thread exception can leave partial/corrupt output file

**File:** `/server/programming/MagicQuant/magicquant/gguf/writer.py`, lines 486-533

When `_read_encode_worker` raises an exception, the main loop at line 500-501 drains the queue and re-raises. However, the output file has already been partially written. There is no cleanup to remove the corrupt partial GGUF file. If the process is interrupted between the exception and the caller handling it, a partial file remains on disk that could be mistaken for a valid model.

**Impact:** Partial GGUF files on disk after errors, which could be loaded by llama.cpp and produce silent corruption.

---

## MEDIUM Severity Issues

### M-1. Logging only configured in writer.py

**File:** `/server/programming/MagicQuant/magicquant/gguf/writer.py`, lines 24, 27

Only `writer.py` imports and configures a logger. All other modules (orchestrator.py, source.py, converters.py, probing.py, survival.py, predictor.py, llamacpp.py) use `print()` for all output. This means:
- No log levels (DEBUG/INFO/WARNING/ERROR)
- No ability to redirect output to files
- No structured logging for production use
- Verbose output cannot be silenced per-module

### M-2. String concatenation for paths instead of pathlib

76 uses of `os.path.join` / `os.path` across the codebase, but only 2 files import `pathlib.Path` (orchestrator.py, llamacpp.py), and even those mix `os.path` calls with Path objects. Key examples:

| File | Line(s) | Pattern |
|------|---------|---------|
| `source.py` | 456, 491, 534, 614-628 | Extensive `os.path.join` chains |
| `probing.py` | 179, 182 | `os.path.join(probe_dir, ...)` |
| `writer.py` | 442 | `os.path.dirname(os.path.abspath(...))` |
| `__main__.py` | 95, 119, 199 | `os.path.join(args.output_dir, ...)` |
| `llamacpp.py` | 42-48, 70-74 | Hardcoded path search lists with `os.path` |

### M-3. Hardcoded values throughout

| File | Line | Value | Description |
|------|------|-------|-------------|
| `orchestrator.py` | 140 | `["E", "H", "Q", "K", "O", "U", "D"]` | Groups list duplicated (also at lines 435, probing.py:315, __main__.py:98, survival.py:45) |
| `orchestrator.py` | 169 | `360` | Hardcoded baseline_tps with no explanation |
| `orchestrator.py` | 127 | `5.0` | Default baseline PPL fallback |
| `llamacpp.py` | 42-48 | `"C:/llama.cpp"`, `"C:/Program Files/llama.cpp"` | Windows-specific hardcoded paths |
| `llamacpp.py` | 19 | `512` | Default ctx_size |
| `predictor.py` | 178-181 | `0.04, 0.12, 0.31...` | Hardcoded parameter distribution weights |
| `survival.py` | 91 | `0.55, 0.40, 0.28` | Tier boundary magic numbers (duplicated in orchestrator.py:594-600 with slightly different values) |
| `probing.py` | 252-262 | `2.0, 1.8, 1.6...` | Hardcoded heuristic sensitivity multipliers |

### M-4. Tier boundary inconsistency between orchestrator and evolution

**File:** `/server/programming/MagicQuant/magicquant/orchestrator.py`, lines 590-600
**File:** `/server/programming/MagicQuant/magicquant/evolution/survival.py`, lines 228-248
**File:** `/server/programming/MagicQuant/magicquant/evolution/predictor.py`, lines 296-304

Three separate tier classification implementations exist with different boundary values:

| Location | Q6 | Q5 | Q4 | Q3 |
|----------|----|----|----|----|
| orchestrator._classify_tier | >0.55 | >0.40 | >0.28 | else |
| survival._classify_into_tiers | >0.55 | >0.40 | >0.28 | else |
| predictor.TierClassifier | 0.65-0.80 | 0.50-0.65 | 0.35-0.50 | 0.20-0.35 |

The predictor's `TierClassifier` is never used by any other module and has significantly different boundaries. This class appears to be dead code.

### M-5. Duplicated code: run_measured_search and run_full_search share ~50 lines

**File:** `/server/programming/MagicQuant/magicquant/orchestrator.py`, lines 76-286 vs 395-478

Both methods contain near-identical blocks for:
- Sensitivity probing setup (lines 132-156 vs 428-451)
- Model group detection (lines 140-152 vs 435-447)
- Predictor initialization (lines 164-170 vs 453-459)

### M-6. GGUFReader.get_parameter_count only counts 2D+ tensors

**File:** `/server/programming/MagicQuant/magicquant/gguf/reader.py`, lines 237-246

The method skips 1D tensors (`if len(shape) >= 2`), which excludes bias vectors and normalization weights. While these are a small fraction of total parameters, the count is inaccurate and could mislead size estimates that depend on it.

### M-7. _save_results writes tiered data twice under different keys

**File:** `/server/programming/MagicQuant/magicquant/orchestrator.py`, lines 351-389

The `_save_results` method writes almost identical data under both `"tiered_survivors"` (line 366) and `"tiered"` (line 375) keys. The only difference is that `"tiered"` uses `.get()` for optional fields. This is confusing and wastes space.

### M-8. parse_name in naming.py is broken for multi-hyphen model names

**File:** `/server/programming/MagicQuant/magicquant/utils/naming.py`, lines 91-146

The `parse_name()` function splits on `-` and tries to parse override blocks, but the logic for determining where the model name ends and the base quantization begins is fragile. For a name like `Qwen3-4B-Instruct-MXFP4_MOE-EH-BF16.gguf`, it incorrectly identifies `BF16` as the base_quant (line 118: `base_quant = parts[-1]`). The override detection logic at line 128 checks for a pattern that never matches because the block was already split on `-`.

### M-9. calculate_expected_size is a no-op

**File:** `/server/programming/MagicQuant/magicquant/utils/naming.py`, lines 208-230

The function `calculate_expected_size` takes a `base_quant_bits` parameter, computes `total_params`, then immediately returns `total_params * (base_quant_bits / 16.0)` which simplifies to just `base_model_size`. The override parameter is accepted but never used. This function always returns the input model size unchanged.

### M-10. No validation of quant_config at entry

**File:** `/server/programming/MagicQuant/magicquant/gguf/writer.py`, lines 259-265

`create_hybrid_gguf` accepts a `quant_config: Dict` with no validation of its structure. If `groups` is missing, it defaults to `{}` silently. If `base` contains a typo (e.g., `"MXP4_MOE"`), it falls through to the default `"Q4_0"` at line 323 without warning:

```python
target_ggml_name = scheme_map.get(scheme, "Q4_0")  # silent fallback
```

**Impact:** Typos in configuration silently produce wrong quantization.

### M-11. File handle leak in GGUFSource.read_tensor_f32

**File:** `/server/programming/MagicQuant/magicquant/gguf/source.py`, lines 132-142

Each call to `read_tensor_f32` opens a new file handle with `open(self._path, "rb")`. While the `with` block ensures closure, this opens/closes the file once per tensor (hundreds of times for a large model). There is no caching or memory-mapping, unlike `SafetensorsSource` which uses `mmap`.

### M-12. QUANT_TYPE_MAP comment is incorrect

**File:** `/server/programming/MagicQuant/magicquant/utils/llamacpp.py`, line 272

The comment says `"MXFP4_MOE": "MXFP4"  # MagicQuant custom type (not native llama.cpp)` but per the CLAUDE.md and writer.py, MXFP4 IS native llama.cpp (ggml type 39). The comment is stale.

### M-13. run_full_search and run_measured_search both open and close model source

**File:** `/server/programming/MagicQuant/magicquant/orchestrator.py`, lines 143-147, 438-442

Both methods independently open a model source just to get tensor names for group detection:

```python
_src = open_model_source(self.source_model_path)
try:
    tensor_names = _src.get_tensor_names()
finally:
    _src.close()
```

This is duplicated and opens the source model twice (once here, once inside `create_hybrid_gguf`).

### M-14. EvolutionarySurvivor.run_evolution uses O(n^2) dedup check

**File:** `/server/programming/MagicQuant/magicquant/evolution/survival.py`, lines 121-123

```python
config_key = str(sorted(winner['config'].items()))
if config_key not in [str(sorted(c['config'].items())) for c in best_configs]:
```

This regenerates the string key for every existing config on every check, making it O(n*m) where n is winners and m is best_configs. Should use a set for O(1) lookup.

---

## LOW Severity Issues

### L-1. Missing type annotations on cross-module functions

| File | Line | Function |
|------|------|----------|
| `writer.py` | 124 | `_write_string(f, s: str)` -- `f` is untyped |
| `writer.py` | 130 | `_write_metadata_value(f, value: Any)` -- `f` is untyped |
| `writer.py` | 190 | `_read_encode_worker(source, entries, result_queue)` -- all params untyped |
| `orchestrator.py` | 351 | `_save_results(self, all_configs, tiered)` -- params untyped |
| `source.py` | 315 | Return type of `_build_gguf_metadata_from_config` not annotated |
| `source.py` | 445 | Return type of `_build_tokenizer_metadata` not annotated |
| `converters.py` | 341 | `_pad_to` return type not annotated |

### L-2. Unused import: numpy in schemes.py

**File:** `/server/programming/MagicQuant/magicquant/quant/schemes.py`, line 11

`import numpy as np` is imported but never used in this module.

### L-3. Inline import of json inside function body

**File:** `/server/programming/MagicQuant/magicquant/gguf/writer.py`, line 412

`import json` is imported inside `create_hybrid_gguf` (also `import queue as _queue_mod` at line 494). These should be at module level.

### L-4. GGUFReader.close() is a no-op

**File:** `/server/programming/MagicQuant/magicquant/gguf/reader.py`, lines 191-193

The `close()` method does nothing (`pass`), but the class is used as a context manager. While the file is opened and closed within `open()` using a `with` block, the context manager pattern creates a false expectation that resources are held.

### L-5. TierClassifier in predictor.py is dead code

**File:** `/server/programming/MagicQuant/magicquant/evolution/predictor.py`, lines 293-315

The `TierClassifier` class is defined but never imported or used anywhere in the codebase. Its tier boundaries also conflict with the boundaries used by the orchestrator and evolution modules (see M-4).

### L-6. Magic number 500_000_000 in converters.py

**File:** `/server/programming/MagicQuant/magicquant/quant/converters.py`, lines 382, 444

The memory threshold `500_000_000` (500 MB) for switching to chunked processing is a magic number with no named constant or comment explaining why 500 MB was chosen.

### L-7. Quantizer class in converters.py is never used by the writer

**File:** `/server/programming/MagicQuant/magicquant/quant/converters.py`, lines 75-330

The `Quantizer` class (lines 75-330) provides numpy-level quantization returning arrays + metadata dicts, but the GGUF writer exclusively uses `encode_to_ggml_bytes()`. The Quantizer class is exported via `__init__.py` but may only be useful for testing/simulation. Its continued presence increases maintenance burden (250+ lines).

### L-8. Inconsistent default output directory

The default output directory is `"./output"` in `__main__.py` (lines 314, 331, 392) and `"./output"` in `orchestrator.py` (line 638). This is consistent, but uses a relative path that depends on the working directory, which can be surprising.

### L-9. get_model_architecture fallback inference is fragile

**File:** `/server/programming/MagicQuant/magicquant/gguf/reader.py`, lines 214-234

The fallback logic checks for `'transformer'` or `'model'` in tensor names, but `'model'` would match essentially any LLaMA-family model, making the "inference" always return `'llama'` as a default. This is effectively dead code since the primary metadata key check almost always succeeds for valid GGUF files.

### L-10. HybridValidator in survival.py has a single static method

**File:** `/server/programming/MagicQuant/magicquant/evolution/survival.py`, lines 363-378

The `HybridValidator` class contains only one static method (`validate_config`) that checks a single condition. It could be a standalone function. The class is also not used anywhere in the codebase -- it appears to be scaffolding for future validation logic.

---

## Packaging Issues

### P-1. Dependencies too minimal (MEDIUM)

**File:** `/server/programming/MagicQuant/pyproject.toml`, lines 12-14

Only `numpy>=1.21.0` is listed as a dependency. While the package works with just numpy for core functionality, the following are effectively required but undeclared:
- No `mmap` dependency declared (used by SafetensorsSource -- though this is a stdlib module, so this is fine)
- PyYAML is optional-only but the `hybrid` command hard-fails without it
- No pin on upper numpy version (numpy 2.0 changed some behaviors)

### P-2. No test suite (MEDIUM)

The pyproject.toml lists `pytest` as a dev dependency, but no tests exist (`tests/` directory is excluded in setuptools config but there appears to be no test directory at all). The CLAUDE.md explicitly states "No test suite exists yet."

### P-3. Version only defined in pyproject.toml and __init__.py (LOW)

**File:** `/server/programming/MagicQuant/magicquant/__init__.py`, line 8
**File:** `/server/programming/MagicQuant/pyproject.toml`, line 6

The version `"0.1.0"` is defined in both locations but not programmatically synced. They could drift.

---

## Summary Table

| Severity | Count | Key Themes |
|----------|-------|------------|
| CRITICAL | 1 | No dtype guard on quantization input |
| HIGH | 5 | Missing timeouts, broad exception handling, partial file cleanup |
| MEDIUM | 14 | Missing logging, hardcoded values, duplicated code, stale/dead code |
| LOW | 10 | Missing type annotations, unused imports, dead code, magic numbers |
| PACKAGING | 3 | Minimal deps, no tests, version sync |
| **TOTAL** | **33** | |

---

## Priority Recommendations

1. **Immediate:** Add dtype assertion guard in `_read_encode_worker` and `encode_to_ggml_bytes` (C-1)
2. **Immediate:** Add timeout to `quantize_model()` subprocess call (H-1)
3. **Immediate:** Clean up partial output file on worker thread exception (H-5)
4. **Short-term:** Replace bare `except Exception` blocks with specific exceptions + logging (H-3)
5. **Short-term:** Add structured logging to all modules (M-1)
6. **Short-term:** Validate quant_config at entry and warn on unknown scheme names (M-10)
7. **Medium-term:** Unify tier classification into a single source of truth (M-4)
8. **Medium-term:** Extract duplicated orchestrator code into shared helpers (M-5)
9. **Medium-term:** Add a basic test suite covering the encoder round-trip and writer pipeline (P-2)
