# Repository Guidelines

## Project Structure & Module Organization

MagicQuant builds hybrid GGUF models using per-group quantization. Source lives in `magicquant/`: `gguf/` handles model I/O, `quant/` contains schemes and libggml bindings, `evolution/` and `v2/` implement search, and `qat/` provides optional training. CLI entry points live in `__main__.py`; configuration lives in `config.py`. Tests and fixtures are under `tests/`, including `tests/integration/`. Calibration text is in `magicquant/data/`; documentation, maintenance scripts, and container definitions live in `docs/`, `tools/`, and `docker/`.

## Build, Test, and Development Commands

Use Python 3.10+ and a `.venv` virtual environment. Use explicit environment executables: this development machine's bare `python`/`pytest` can resolve to an unrelated shim.

- `.venv/bin/python -m pip install -e ".[dev]"` — install editable source and development dependencies.
- `.venv/bin/magicquant analyze model.gguf` — inspect a local model's tensor groups.
- `.venv/bin/python -m pytest tests/ -q` — run the main suite.
- `.venv/bin/ruff check --select F magicquant/ tools/ tests/` — run CI's required Pyflakes checks; full Ruff lint remains advisory.
- `.venv/bin/ruff format magicquant/ tests/` — format Python code; keep changes scoped to your work.
- `make docker-build` — build the `magicquant:latest` container image.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Follow nearby type annotations and docstrings. Preserve Python 3.10 compatibility. Keep quantization facts centralized in `magicquant/quant/ggml_facts.py`.

## Testing Guidelines

Use pytest files named `test_<feature>.py` and functions named `test_<behavior>`. Add focused regression tests for behavior changes and reuse `tests/fixtures/`. No numeric coverage threshold is configured. Use `-m "not slow and not gpu"` to exclude marked tests. Encoder parity tests require libggml and `llama-quantize`; unavailable binaries cause skips. Verify QAT changes with `.venv-qat/bin/python -m pytest tests/ -q` after installing the `qat` and `dev` extras there. CI exercises Python 3.10 and 3.12.

## Commit & Pull Request Guidelines

Follow observed subjects such as `fix(gguf): preserve metadata types` or `docs: clarify setup`. Keep commits focused. Update `CHANGELOG.md` using Keep a Changelog categories, recording affected files and verification. PR descriptions should explain behavior changes, link relevant issues, and report validation commands/results and skipped checks.

## Configuration & Artifacts

Use `MAGICQUANT_` environment settings or a local `.env`; CLI flags override environment settings. Keep secrets, model weights, generated GGUFs, and `output/` artifacts out of commits.
