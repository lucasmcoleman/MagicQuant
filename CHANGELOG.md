# Changelog

## [0.2.0] - 2026-04-03 — Production Hardening

### Bug Fixes
- **CRITICAL**: Added dtype validation guard in `encode_to_ggml_bytes()` — rejects non-floating-point tensors with descriptive error instead of silently producing corrupt output
- **CRITICAL**: Added dtype guard in GGUF writer worker thread before encoding dispatch
- Fixed `general.file_type` metadata mapping — was hardcoded to `1` for all branches, now uses a proper lookup table mapping quantization schemes to llama_ftype enum values

### Added
- `magicquant/config.py` — Pydantic-settings `BaseSettings` (`MagicQuantSettings`) for configuration via environment variables (`MAGICQUANT_` prefix) and `.env` files. Includes `validate_paths()` for startup checks and typed properties for `output_path` / `source_path`.
- `magicquant/logging.py` — Structured logging via structlog with `configure_logging()` (console or JSON output) and `get_logger()` factory. Consistent schema: `event`, `stage`, `tensor_name`, `progress`, etc.
- Tenacity retry wrappers on `calculate_perplexity()` subprocess call — `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))` handles transient GPU OOM and file-lock failures.
- `tests/test_quantization_guards.py` — 18 tests covering dtype validation, encoder output sizes, and BF16 round-trip correctness.
- `--dry-run` flag on `magicquant search` command — validates configuration, opens source model, verifies tensor count and architecture, checks llama.cpp availability, and reports a summary without running the search.

### Changed
- **Structured logging in orchestrator**: All `print()` statements in `orchestrator.py` replaced with `structlog` logger calls (`log.info`, `log.warning`, `log.error`). Every event includes a `stage` field for filtering (init, baseline, probing, measurement, build, generate, results).
- **Subprocess hardening** (`llamacpp.py`):
  - All `subprocess.run()` calls have explicit `timeout` parameters (30s for `which`/`where`, 600s for perplexity, 1800s for quantization).
  - No `shell=True` usage (verified by audit).
  - Proper error handling for both `CalledProcessError` and `TimeoutExpired` on every call.
  - Extracted `_parse_perplexity_output()` as a standalone function for testability.
- **Pathlib conversion**: Replaced `os.path.join()`, `os.path.exists()`, `os.path.isfile()`, `os.path.getsize()`, `os.makedirs()`, and `os.remove()` with `pathlib.Path` operations in `orchestrator.py`, `writer.py`, `llamacpp.py`, and `__main__.py`. Public APIs still accept `str` paths.
- **Dependencies**: Added `pydantic-settings>=2.0.0`, `structlog>=24.0.0`, `tenacity>=8.2.0`, `python-dotenv>=1.0.0` to core deps. Added `pytest-asyncio>=0.21.0` to dev deps.
