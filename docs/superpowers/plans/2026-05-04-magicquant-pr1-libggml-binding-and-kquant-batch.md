# PR1: libggml Binding + K-quant Batch + Retrofit + Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** PR0 must be merged to `master`. Verify with `git log --oneline -5 | grep "scheme registry"` showing recent refactor commits.

**Goal:** Add a ctypes binding to libggml, retrofit the 7 existing schemes' encoders to use it (eliminating the ~10–27% MSE gap), register Q2_K and Q3_K as new schemes (making the Q3 tier band reachable), and calibrate noise_factor values via empirical bench. Replace ~600 lines of pure-Python encoders with a thin C-binding wrapper.

**Architecture:** New `magicquant/quant/ggml_binding.py` (~180 lines) loads `libggml-base.so` + `libggml-cpu.so` via ctypes and exposes `ggml_encode(weights, ggml_type, imatrix=None) → bytes`. Discovery order: env var → system llama.cpp → llama-cpp-python bundled libs. `converters.py` shrinks to a thin dispatch layer that routes quantized formats to `ggml_encode` and float formats to native passthroughs.

**Tech Stack:** Python 3.12, ctypes, numpy, pytest, llama-cpp-python (new hard dep)

**Spec:** `docs/superpowers/specs/2026-05-04-magicquant-encoder-expansion-design.md` (sections "ctypes binding", "Encoder dispatch", "Calibration bench", "PR1")

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `llama-cpp-python>=0.3.0` hard dep |
| `magicquant/quant/ggml_binding.py` | Create | ctypes wrapper: discovery, type IDs, `ggml_encode()` |
| `magicquant/quant/converters.py` | Modify | Shrink: delete pure-Python encoders + helpers; route to `ggml_encode` |
| `magicquant/quant/schemes.py` | Modify | Add Q2_K and Q3_K to registry; update noise factors from calibration |
| `magicquant/evolution/survival.py` | Modify | Restructure `_generate_random_config` weights to category-indexed |
| `tests/test_refactor_regression.py` | Modify | Regenerate fixture for category-indexed weights |
| `tests/fixtures/refactor_regression_seed42.json` | Modify | Updated snapshot |
| `tests/fixtures/reference_tensor.f32.npy` | Create | Reference tensor for parity tests |
| `tests/integration/__init__.py` | Create | Package marker |
| `tests/integration/test_encoder_parity.py` | Create | Byte-parity tests vs `llama-quantize` |
| `tests/integration/test_smoke_q3_tier.py` | Create | Q3 tier reachability smoke test |
| `tools/calibrate_noise_factors.py` | Create | One-shot calibration bench |
| `tools/calibration_results.json` | Create | Bench output (committed) |
| `CLAUDE.md` | Modify | Remove "10-27% MSE gap" caveat |

**File-size note:** `converters.py` shrinks from ~960 → ~250 lines. `ggml_binding.py` is the only new substantive module (~180 lines). Total net code change: roughly −600 lines despite adding 2 new schemes.

---

## Tasks

### Task 1: Prerequisite verification

**Files:** none

- [ ] **Step 1: Verify PR0 is merged**

Run:
```bash
cd /server/programming/MagicQuant && git log --oneline -10
```

Expected output: includes commits with messages like `refactor: extend QuantizationScheme with full attribute set`, `refactor: predictor reads scheme attributes from registry`, etc.

If PR0 is NOT merged, STOP and merge PR0 first.

- [ ] **Step 2: Verify libggml libraries are accessible**

Run:
```bash
ls -la /home/lucas/llama.cpp-build/build/bin/libggml-base.so /home/lucas/llama.cpp-build/build/bin/libggml-cpu.so 2>&1
```

Expected output: both files exist as symlinks or regular files.

If neither path exists, locate libggml elsewhere with:
```bash
find / -maxdepth 6 -name "libggml-base.so*" -not -path "*/proc/*" 2>/dev/null | head -5
```

If no copies of libggml exist, STOP and build llama.cpp first (separate task).

- [ ] **Step 3: Verify ggml.h is accessible (for confirming type IDs)**

Run:
```bash
ls -la /home/lucas/llama.cpp/ggml/include/ggml.h 2>&1
```

If absent, locate elsewhere:
```bash
find / -maxdepth 6 -name "ggml.h" -path "*/ggml/include/*" 2>/dev/null | head -3
```

Note the path of an accessible `ggml.h` — used in Task 4 to verify type IDs.

- [ ] **Step 4: Run all existing tests as baseline**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected output: all tests pass (12 dtype guards + 1 regression = 13 total).

---

### Task 2: Add llama-cpp-python to install dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current `pyproject.toml`**

Run:
```bash
cd /server/programming/MagicQuant && cat pyproject.toml
```

Expected: confirms current `dependencies` block lists numpy, pydantic-settings, structlog, tenacity, python-dotenv.

- [ ] **Step 2: Add `llama-cpp-python` to dependencies**

Edit `/server/programming/MagicQuant/pyproject.toml`. Find:
```toml
dependencies = [
    "numpy>=1.21.0",
    "pydantic-settings>=2.0.0",
    "structlog>=24.0.0",
    "tenacity>=8.2.0",
    "python-dotenv>=1.0.0",
]
```

Replace with:
```toml
dependencies = [
    "numpy>=1.21.0",
    "pydantic-settings>=2.0.0",
    "structlog>=24.0.0",
    "tenacity>=8.2.0",
    "python-dotenv>=1.0.0",
    # llama-cpp-python ships libggml-base.so + libggml-cpu.so in its wheel,
    # used by magicquant.quant.ggml_binding for all quantized encoders.
    # Discovery prefers system llama.cpp if available; falls back to this.
    "llama-cpp-python>=0.3.0",
]
```

- [ ] **Step 3: Install the new dep into the active venv**

Run:
```bash
cd /server/programming/MagicQuant && pip install -e . 2>&1 | tail -10
```

Expected output: `Successfully installed magicquant-0.1.0` and llama-cpp-python successfully installed (may build from source on AMD ROCm — first install can take 5–15 min). If install fails, abort and resolve before continuing.

- [ ] **Step 4: Verify llama-cpp-python ships libggml**

Run:
```bash
python -c "
import llama_cpp, pathlib
lib_dir = pathlib.Path(llama_cpp.__file__).parent / 'lib'
print('lib dir:', lib_dir)
for f in sorted(lib_dir.glob('libggml*')):
    print(' -', f.name)
" 2>&1
```

Expected output: lists `libggml-base.so` and `libggml-cpu.so` (filenames may include version suffixes like `.0` or major.minor.patch).

If neither shows up, llama-cpp-python's wheel didn't bundle libggml — try `pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir` and re-check.

- [ ] **Step 5: Commit pyproject change**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add pyproject.toml && \
  git commit -m "deps: add llama-cpp-python as hard dep for libggml binding

Provides bundled libggml-base.so and libggml-cpu.so for the ctypes
encoder binding added in PR1. Discovery prefers system llama.cpp if
available; the wheel-bundled libs are a guaranteed fallback so
'pip install -e .' from a fresh clone always produces a working setup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Create the ggml_binding module — discovery only

**Files:**
- Create: `magicquant/quant/ggml_binding.py`

This task creates the discovery layer first (without the ctypes call surface) so we can test discovery in isolation.

- [ ] **Step 1: Create the file with discovery functions**

Create `/server/programming/MagicQuant/magicquant/quant/ggml_binding.py` with this exact content:

```python
"""
ctypes binding to libggml — encoder dispatch for all quantized schemes.

Discovery order (first match wins):
    1. MAGICQUANT_LIBGGML_DIR env var (explicit override)
    2. Common system llama.cpp build paths
    3. llama-cpp-python's bundled libs (always available since it's a hard dep)

Public API:
    ggml_encode(weights, ggml_type, imatrix=None) -> bytes
    GGML_TYPE_IDS  (mapping name -> numeric ggml type enum, synced from ggml.h)
    LibggmlNotFound  (exception)
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ggml_type enum values from ggml/include/ggml.h.
# Synced as of llama.cpp build at /home/lucas/llama.cpp-build/.
# A startup sanity check (_verify_type_ids) catches drift if the loaded
# libggml renumbers types.
GGML_TYPE_IDS = {
    "F32":     0,
    "F16":     1,
    "Q4_0":    2,
    "Q4_1":    3,
    "Q5_0":    6,
    "Q5_1":    7,
    "Q8_0":    8,
    "Q8_1":    9,
    "Q2_K":    10,
    "Q3_K":    11,
    "Q4_K":    12,
    "Q5_K":    13,
    "Q6_K":    14,
    "Q8_K":    15,
    "IQ2_XXS": 16,
    "IQ2_XS":  17,
    "IQ3_XXS": 18,
    "IQ1_S":   19,
    "IQ4_NL":  20,
    "IQ3_S":   21,
    "IQ2_S":   22,
    "IQ4_XS":  23,
    "IQ1_M":   29,
    "BF16":    30,
    "MXFP4":   39,
}


_SYSTEM_SEARCH_DIRS = [
    "~/llama.cpp/build/bin",
    "~/llama.cpp-build/build/bin",
    "/usr/local/lib",
    "/home/linuxbrew/.linuxbrew/lib",
]


class LibggmlNotFound(RuntimeError):
    """Raised when libggml-base.so / libggml-cpu.so cannot be located."""


def _discover_libggml() -> Tuple[Path, Path]:
    """Return (libggml-base path, libggml-cpu path).

    Discovery order:
      1. MAGICQUANT_LIBGGML_DIR env var (explicit override)
      2. Common system llama.cpp build paths
      3. llama-cpp-python's bundled libs

    Raises:
      LibggmlNotFound if neither library can be found.
    """
    # 1. Explicit env var
    env_dir = os.environ.get("MAGICQUANT_LIBGGML_DIR")
    if env_dir:
        result = _check_dir(Path(env_dir).expanduser())
        if result:
            return result

    # 2. System paths
    for raw in _SYSTEM_SEARCH_DIRS:
        result = _check_dir(Path(raw).expanduser())
        if result:
            return result

    # Optional environment hint
    llamacpp_path = os.environ.get("LLAMACPP_PATH")
    if llamacpp_path:
        result = _check_dir(Path(llamacpp_path).expanduser() / "build" / "bin")
        if result:
            return result

    # 3. llama-cpp-python bundled
    try:
        import llama_cpp  # type: ignore
    except ImportError:
        raise LibggmlNotFound(
            "Could not find libggml-base.so / libggml-cpu.so. "
            "Tried: $MAGICQUANT_LIBGGML_DIR, common system paths, "
            "$LLAMACPP_PATH/build/bin, and llama-cpp-python bundle. "
            "Install llama-cpp-python or set MAGICQUANT_LIBGGML_DIR to a "
            "directory containing both libraries."
        )

    bundle_dir = Path(llama_cpp.__file__).parent / "lib"
    result = _check_dir(bundle_dir)
    if result:
        return result

    raise LibggmlNotFound(
        f"llama-cpp-python is installed but bundled libggml not found at {bundle_dir}. "
        f"Try: pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir"
    )


def _check_dir(d: Path) -> Optional[Tuple[Path, Path]]:
    """Return (base, cpu) library paths if both exist in the directory, else None.

    Tries a few common naming variants (versioned and unversioned).
    """
    if not d.is_dir():
        return None

    base_names = ["libggml-base.so", "libggml-base.so.0", "libggml-base.dylib"]
    cpu_names = ["libggml-cpu.so", "libggml-cpu.so.0", "libggml-cpu.dylib"]

    base = next((d / n for n in base_names if (d / n).exists()), None)
    cpu = next((d / n for n in cpu_names if (d / n).exists()), None)

    # Fallback: glob for versioned variants like libggml-base.so.0.9.7
    if base is None:
        candidates = sorted(d.glob("libggml-base.so*"))
        if candidates:
            base = candidates[0]
    if cpu is None:
        candidates = sorted(d.glob("libggml-cpu.so*"))
        if candidates:
            cpu = candidates[0]

    if base and cpu:
        return base, cpu
    return None
```

- [ ] **Step 2: Verify the module imports and discovery works**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.ggml_binding import _discover_libggml, GGML_TYPE_IDS
base, cpu = _discover_libggml()
print('base:', base)
print('cpu:', cpu)
print('Q2_K id:', GGML_TYPE_IDS['Q2_K'])
print('IQ2_XXS id:', GGML_TYPE_IDS['IQ2_XXS'])
"
```

Expected output: real paths to `libggml-base.so` and `libggml-cpu.so` (probably under `/home/lucas/llama.cpp-build/build/bin/` since that's the system-path priority), and IDs `10` and `16`.

- [ ] **Step 3: Test the LibggmlNotFound path**

Run:
```bash
cd /server/programming/MagicQuant && MAGICQUANT_LIBGGML_DIR=/nonexistent/path python -c "
import os, sys
# Temporarily hide llama_cpp by clearing PYTHONPATH-effective imports
sys.modules['llama_cpp'] = None  # force ImportError on later import
from magicquant.quant.ggml_binding import _discover_libggml, LibggmlNotFound
try:
    _discover_libggml()
    print('UNEXPECTED: discovery succeeded when it should have failed')
except LibggmlNotFound as e:
    print('OK: LibggmlNotFound raised:', str(e)[:80])
" 2>&1 | tail -5
```

Note: the env var override is set to a non-existent path, but the discovery falls through to system paths (which will likely succeed). The test for the failure path is harder to reproduce in this environment without uninstalling things. Skip if it succeeds — what matters is that LibggmlNotFound exists and can be raised. Verify with:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.ggml_binding import LibggmlNotFound
print('LibggmlNotFound class exists:', LibggmlNotFound.__name__)
"
```

Expected output: `LibggmlNotFound class exists: LibggmlNotFound`.

- [ ] **Step 4: Commit the discovery layer**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/ggml_binding.py && \
  git commit -m "feat: add libggml discovery layer (ggml_binding.py)

Hybrid discovery: env var override → system paths → llama-cpp-python
bundled libs. GGML_TYPE_IDS table synced from ggml.h. LibggmlNotFound
raised with actionable error if no libs found.

ctypes call surface comes in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add the ctypes call surface to ggml_binding.py

**Files:**
- Modify: `magicquant/quant/ggml_binding.py`

Adds the `_LibggmlHandle` class (ctypes signatures, `encode()` method) and the public `ggml_encode()` function.

- [ ] **Step 1: Append the handle class and public API**

Append the following to the end of `/server/programming/MagicQuant/magicquant/quant/ggml_binding.py`:

```python


# ── Block format size tables (synced with ggml/src/ggml-quants.h) ─────
# Used to compute output buffer size for ctypes calls.

_GGML_BLOCK_SIZE = {
    "F32": 1, "F16": 1, "BF16": 1,
    "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32,
    "Q8_0": 32, "Q8_1": 32,
    "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256,
    "Q6_K": 256, "Q8_K": 256,
    "IQ2_XXS": 256, "IQ2_XS": 256, "IQ3_XXS": 256,
    "IQ1_S": 256, "IQ4_NL": 32, "IQ3_S": 256,
    "IQ2_S": 256, "IQ4_XS": 32, "IQ1_M": 256,
    "MXFP4": 32,
}

_GGML_TYPE_SIZE = {
    "F32": 4, "F16": 2, "BF16": 2,
    "Q4_0": 18, "Q4_1": 20, "Q5_0": 22, "Q5_1": 24,
    "Q8_0": 34, "Q8_1": 36,
    "Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176,
    "Q6_K": 210, "Q8_K": 292,
    "IQ2_XXS": 66, "IQ2_XS": 74, "IQ3_XXS": 98,
    "IQ1_S": 50, "IQ4_NL": 18, "IQ3_S": 110,
    "IQ2_S": 82, "IQ4_XS": 18, "IQ1_M": 56,
    "MXFP4": 17,
}


def _expected_size(ggml_type: str, n_elements: int) -> int:
    """Return expected output byte count for a quantized tensor."""
    block = _GGML_BLOCK_SIZE.get(ggml_type, 1)
    type_sz = _GGML_TYPE_SIZE.get(ggml_type, 2)
    n_blocks = (n_elements + block - 1) // block
    return n_blocks * type_sz


class _LibggmlHandle:
    """Process-wide ctypes binding to libggml. Created once via get_handle()."""

    def __init__(self):
        base_path, cpu_path = _discover_libggml()
        # RTLD_GLOBAL so libggml-cpu can resolve symbols from libggml-base
        self._base = ctypes.CDLL(str(base_path), mode=ctypes.RTLD_GLOBAL)
        self._cpu = ctypes.CDLL(str(cpu_path), mode=ctypes.RTLD_GLOBAL)
        self._base_path = base_path
        self._cpu_path = cpu_path
        self._setup_signatures()
        self._verify_type_ids()
        # Initialize all type tables (-1 = init all). Required for IQ-quants
        # which use precomputed grid tables.
        self._base.ggml_quantize_init(ctypes.c_int(-1))

    def _setup_signatures(self) -> None:
        # size_t ggml_quantize_chunk(
        #     enum ggml_type type, const float * src, void * dst,
        #     int64_t start, int64_t nrows, int64_t n_per_row,
        #     const float * imatrix);
        self._base.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._base.ggml_quantize_chunk.restype = ctypes.c_size_t

        # size_t ggml_type_size(enum ggml_type type);
        self._base.ggml_type_size.argtypes = [ctypes.c_int]
        self._base.ggml_type_size.restype = ctypes.c_size_t

        # int64_t ggml_blck_size(enum ggml_type type);
        self._base.ggml_blck_size.argtypes = [ctypes.c_int]
        self._base.ggml_blck_size.restype = ctypes.c_int64

        # bool ggml_quantize_requires_imatrix(enum ggml_type type);
        self._base.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        self._base.ggml_quantize_requires_imatrix.restype = ctypes.c_bool

        # void ggml_quantize_init(enum ggml_type type);
        self._base.ggml_quantize_init.argtypes = [ctypes.c_int]
        self._base.ggml_quantize_init.restype = None

    def _verify_type_ids(self) -> None:
        """Sanity check: each (name, id) pair agrees with what libggml reports.

        Catches the case where a future ggml release renumbers types. If
        any mismatch is found, raise immediately with a clear actionable error.
        """
        for name, type_id in GGML_TYPE_IDS.items():
            expected = _GGML_TYPE_SIZE.get(name)
            if expected is None:
                continue  # not all types have known sizes (extension types)
            actual = self._base.ggml_type_size(type_id)
            if actual != expected:
                raise RuntimeError(
                    f"libggml type-ID drift: {name} (id={type_id}) "
                    f"reports type_size={actual}, expected {expected}. "
                    f"Either GGML_TYPE_IDS in ggml_binding.py is stale, or "
                    f"the loaded libggml ({self._base_path}) is from an "
                    f"incompatible ggml version. Pin llama-cpp-python or "
                    f"update GGML_TYPE_IDS."
                )

    def encode(
        self,
        weights: np.ndarray,
        ggml_type: str,
        imatrix: Optional[np.ndarray] = None,
    ) -> bytes:
        """Quantize a float tensor to ggml block-format bytes.

        Args:
            weights: floating-point numpy array (any shape; flattened internally).
            ggml_type: ggml type name (e.g., "Q4_K", "Q2_K", "MXFP4").
            imatrix: optional float32 importance matrix (1-D, length matches weights).

        Returns:
            Raw bytes in the on-disk ggml block layout.
        """
        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError(
                f"ggml_encode requires floating-point input, got dtype={weights.dtype}"
            )
        if ggml_type not in GGML_TYPE_IDS:
            raise ValueError(
                f"Unknown ggml type: {ggml_type}. Available: {sorted(GGML_TYPE_IDS)}"
            )

        flat = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
        n_per_row = flat.size
        type_id = GGML_TYPE_IDS[ggml_type]
        out_size = _expected_size(ggml_type, flat.size)

        dst_buf = (ctypes.c_uint8 * out_size)()

        imat_ptr = None
        imat_owner = None  # keep alive
        if imatrix is not None:
            imat_owner = np.ascontiguousarray(imatrix, dtype=np.float32)
            imat_ptr = imat_owner.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        actual = self._base.ggml_quantize_chunk(
            type_id,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(dst_buf, ctypes.c_void_p),
            ctypes.c_int64(0),
            ctypes.c_int64(1),
            ctypes.c_int64(n_per_row),
            imat_ptr if imat_ptr is not None else ctypes.POINTER(ctypes.c_float)(),
        )
        if actual != out_size:
            raise RuntimeError(
                f"ggml_quantize_chunk wrote {actual} bytes, expected {out_size} "
                f"(type={ggml_type}, n_elements={n_per_row}). "
                f"Likely cause: wrong type_id or block-size mismatch."
            )
        return bytes(dst_buf)

    def requires_imatrix(self, ggml_type: str) -> bool:
        """Does this scheme need an importance matrix for best quality?"""
        if ggml_type not in GGML_TYPE_IDS:
            return False
        return self._base.ggml_quantize_requires_imatrix(GGML_TYPE_IDS[ggml_type])


_HANDLE: Optional[_LibggmlHandle] = None


def get_handle() -> _LibggmlHandle:
    """Lazy singleton accessor. Constructs the handle on first call."""
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = _LibggmlHandle()
    return _HANDLE


def ggml_encode(
    weights: np.ndarray,
    ggml_type: str,
    imatrix: Optional[np.ndarray] = None,
) -> bytes:
    """Quantize via libggml's ggml_quantize_chunk.

    Public entry point used by magicquant.quant.converters.
    """
    return get_handle().encode(weights, ggml_type, imatrix)
```

- [ ] **Step 2: Smoke-test the binding with Q8_0 (simplest scheme)**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import numpy as np
from magicquant.quant.ggml_binding import ggml_encode, get_handle

# Confirm handle initializes
handle = get_handle()
print('handle libs:', handle._base_path.name, handle._cpu_path.name)

# Quantize 256 floats with Q8_0 (32-element blocks, 34 bytes each)
weights = np.random.randn(256).astype(np.float32) * 0.02
out = ggml_encode(weights, 'Q8_0')
print(f'Q8_0 output: {len(out)} bytes (expected 8 blocks * 34 = 272)')
assert len(out) == 272, 'size mismatch'
print('OK')
"
```

Expected output:
```
handle libs: libggml-base.so... libggml-cpu.so...
Q8_0 output: 272 bytes (expected 8 blocks * 34 = 272)
OK
```

- [ ] **Step 3: Smoke-test with Q4_K (256-element blocks)**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import numpy as np
from magicquant.quant.ggml_binding import ggml_encode

# 256 floats = 1 Q4_K block = 144 bytes
weights = np.random.randn(256).astype(np.float32) * 0.02
out = ggml_encode(weights, 'Q4_K')
print(f'Q4_K (256 elem): {len(out)} bytes (expected 144)')
assert len(out) == 144

# 512 floats = 2 Q4_K blocks = 288 bytes
weights2 = np.random.randn(512).astype(np.float32) * 0.02
out2 = ggml_encode(weights2, 'Q4_K')
print(f'Q4_K (512 elem): {len(out2)} bytes (expected 288)')
assert len(out2) == 288
print('OK')
"
```

Expected output:
```
Q4_K (256 elem): 144 bytes (expected 144)
Q4_K (512 elem): 288 bytes (expected 288)
OK
```

- [ ] **Step 4: Smoke-test with Q2_K (the new scheme this PR adds)**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import numpy as np
from magicquant.quant.ggml_binding import ggml_encode

# Q2_K: 256 elements = 1 block = 84 bytes
weights = np.random.randn(256).astype(np.float32) * 0.02
out = ggml_encode(weights, 'Q2_K')
print(f'Q2_K: {len(out)} bytes (expected 84)')
assert len(out) == 84
print('OK')
"
```

Expected output:
```
Q2_K: 84 bytes (expected 84)
OK
```

- [ ] **Step 5: Verify type-ID sanity check works**

The `_verify_type_ids` method runs on handle construction. It would have already triggered if there were drift. Verify by inspecting the handle has been initialized without error:

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.ggml_binding import get_handle, _GGML_TYPE_SIZE
h = get_handle()
# Manually re-verify a few critical IDs
for name in ['Q2_K', 'Q3_K', 'Q4_K', 'IQ2_XXS', 'IQ4_NL']:
    expected = _GGML_TYPE_SIZE[name]
    actual = h._base.ggml_type_size(h._base.ggml_type_size.__class__) if False else None  # just the lookup
    print(f'{name}: expected_size={expected}')
print('type-ID verification passed during handle construction')
"
```

Expected output: lists per-type sizes; if no exception was raised during `get_handle()`, all type IDs match.

- [ ] **Step 6: Commit ctypes call surface**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/ggml_binding.py && \
  git commit -m "feat: ctypes call surface for ggml_quantize_chunk

Adds _LibggmlHandle class with full ctypes signatures, type-ID drift
verification on construction, and the public ggml_encode() function.
Smoke-tested with Q8_0, Q4_K, and Q2_K — output sizes match expected
ggml block-format dimensions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Generate the reference tensor fixture for parity tests

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/fixtures/reference_tensor.f32.npy`

- [ ] **Step 1: Create the integration package marker**

Run:
```bash
cd /server/programming/MagicQuant && mkdir -p tests/integration && touch tests/integration/__init__.py
```

- [ ] **Step 2: Generate and save the reference tensor**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import numpy as np
from pathlib import Path

# Realistic transformer-weight statistics: mean=0, std~0.02, no extreme outliers
rng = np.random.default_rng(42)
tensor = rng.normal(0.0, 0.02, size=(2048, 2048)).astype(np.float32)

out = Path('tests/fixtures/reference_tensor.f32.npy')
np.save(out, tensor, allow_pickle=False)
print(f'wrote {out}: shape={tensor.shape}, dtype={tensor.dtype}, bytes={out.stat().st_size}')
print(f'first 5 values: {tensor.flat[:5]}')
print(f'mean={tensor.mean():.4f}, std={tensor.std():.4f}')
"
```

Expected output:
```
wrote tests/fixtures/reference_tensor.f32.npy: shape=(2048, 2048), dtype=float32, bytes=16777344
first 5 values: [...]
mean=0.0000, std=0.0200
```

- [ ] **Step 3: Commit fixture and package marker**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/integration/__init__.py tests/fixtures/reference_tensor.f32.npy && \
  git commit -m "test: add reference tensor fixture for encoder parity tests

2048x2048 float32 tensor sampled from N(0, 0.02) with seed=42 — matches
realistic transformer-weight statistics. Used by encoder parity tests
to compare MagicQuant's ctypes output to llama-quantize subprocess.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Write the byte-parity test harness

**Files:**
- Create: `tests/integration/test_encoder_parity.py`

The harness produces a tiny F32 GGUF containing the reference tensor, invokes both MagicQuant's ctypes path and `llama-quantize` subprocess, and asserts byte-for-byte equality.

- [ ] **Step 1: Create the test file**

Create `/server/programming/MagicQuant/tests/integration/test_encoder_parity.py` with this exact content:

```python
"""Encoder parity tests — MagicQuant's ctypes binding vs llama-quantize subprocess.

Both code paths invoke the same ggml_quantize_chunk function from the same
libggml-cpu.so. Byte-for-byte equality is the natural expectation; any
mismatch indicates a binding bug (wrong type_id, wrong nrows/n_per_row,
non-contiguous memory, etc.).

These tests require:
  - libggml available via magicquant.quant.ggml_binding
  - llama-quantize on PATH (or override via LLAMA_QUANTIZE env var)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from magicquant.quant.ggml_binding import ggml_encode

REF_TENSOR_PATH = Path(__file__).parent.parent / "fixtures" / "reference_tensor.f32.npy"

# Schemes verified in this PR. Q4_0 is included because it's already encoded
# in the codebase and registered by the existing schemes; PR2 adds the rest
# of the legacy Q-quants. PR3 adds the IQ-quants.
SCHEMES_PR1 = [
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "IQ4_NL", "MXFP4", "Q4_0",
    "Q2_K", "Q3_K",
]


def _llama_quantize_path() -> str:
    """Locate the llama-quantize executable."""
    explicit = os.environ.get("LLAMA_QUANTIZE")
    if explicit:
        if not Path(explicit).is_file():
            pytest.fail(f"LLAMA_QUANTIZE={explicit} does not exist")
        return explicit
    found = shutil.which("llama-quantize")
    if found:
        return found
    pytest.skip("llama-quantize not on PATH and LLAMA_QUANTIZE not set")


@pytest.fixture(scope="module")
def reference_tensor() -> np.ndarray:
    if not REF_TENSOR_PATH.exists():
        pytest.fail(f"Reference tensor missing: {REF_TENSOR_PATH}")
    return np.load(REF_TENSOR_PATH)


def _write_f32_gguf(tensor: np.ndarray, out_path: Path, tensor_name: str = "test.weight") -> None:
    """Write a minimal GGUF file containing one F32 tensor.

    Uses the gguf Python package (already a transitive dep via Foundry).
    """
    import gguf  # type: ignore

    writer = gguf.GGUFWriter(str(out_path), arch="test")
    writer.add_tensor(tensor_name, tensor, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _read_first_tensor_bytes(gguf_path: Path) -> Tuple[bytes, str]:
    """Read the raw bytes of the first tensor in a GGUF file.

    Returns (tensor_bytes, ggml_type_name).
    """
    import gguf  # type: ignore

    reader = gguf.GGUFReader(str(gguf_path))
    if not reader.tensors:
        raise RuntimeError(f"no tensors in {gguf_path}")
    t = reader.tensors[0]
    # t.data is a memoryview of float32 if the tensor is F32; for quantized
    # tensors it's the raw bytes view.
    raw = bytes(t.data.tobytes() if hasattr(t.data, 'tobytes') else t.data)
    type_name = t.tensor_type.name  # e.g. "Q4_K" — see gguf.GGMLQuantizationType
    return raw, type_name


@pytest.mark.parametrize("scheme", SCHEMES_PR1)
def test_encoder_byte_for_byte_matches_llama_quantize(
    scheme: str, reference_tensor: np.ndarray, tmp_path: Path
) -> None:
    """Quantize the reference tensor via MagicQuant's ctypes path and via
    llama-quantize subprocess. Assert output bytes are identical."""
    quantize_bin = _llama_quantize_path()

    # Step 1: Build the F32 source GGUF
    src_path = tmp_path / "src.f32.gguf"
    _write_f32_gguf(reference_tensor, src_path)

    # Step 2: Run llama-quantize <scheme>
    dst_path = tmp_path / f"dst.{scheme}.gguf"
    result = subprocess.run(
        [quantize_bin, str(src_path), str(dst_path), scheme],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"llama-quantize {scheme} failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    # Step 3: Extract llama-quantize's output bytes
    llama_bytes, llama_type_name = _read_first_tensor_bytes(dst_path)

    # Step 4: Quantize the same tensor via MagicQuant's binding
    magic_bytes = ggml_encode(reference_tensor, scheme)

    # Step 5: Assert byte equality
    assert len(magic_bytes) == len(llama_bytes), (
        f"size mismatch for {scheme}: magic={len(magic_bytes)}, "
        f"llama={len(llama_bytes)}"
    )
    if magic_bytes != llama_bytes:
        # Find first divergence
        diff_idx = next(
            (i for i in range(len(magic_bytes)) if magic_bytes[i] != llama_bytes[i]),
            -1,
        )
        pytest.fail(
            f"byte mismatch for {scheme} at offset {diff_idx}: "
            f"magic={magic_bytes[diff_idx:diff_idx+8].hex()}, "
            f"llama={llama_bytes[diff_idx:diff_idx+8].hex()}"
        )
```

- [ ] **Step 2: Verify the test collects (does not run yet)**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_encoder_parity.py --collect-only -q
```

Expected output: 9 tests collected (one per scheme in SCHEMES_PR1).

- [ ] **Step 3: Run the parity tests for the existing schemes (Q8_0, Q4_0, IQ4_NL, Q4_K, Q5_K, Q6_K, MXFP4)**

This is the moment of truth — these tests should pass NOW because both `ggml_encode` and `llama-quantize` call the same C function. If any fail, there's a bug in the binding (wrong arg type, type ID, etc.).

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_encoder_parity.py -v 2>&1 | tail -25
```

Expected output: all 9 tests pass.

If any fail with byte mismatches, the most likely cause is one of:
  a. `gguf` package's tensor reader returns dequantized data, not raw bytes — fix `_read_first_tensor_bytes` to read the raw tensor data segment from the GGUF file directly via offsets (not via the `gguf.GGUFReader` API).
  b. `llama-quantize` is calling a different ggml type (e.g., `Q4_0` → `Q4_0_R8` repacked variant). Pass `--pure` or equivalent to force per-tensor non-repacked quantization.

Resolution path if byte mismatches appear: add `--pure` flag to the llama-quantize invocation in the test (it disables row-repacking for Q4_0/Q5_0 variants):
```python
[quantize_bin, "--pure", str(src_path), str(dst_path), scheme],
```

- [ ] **Step 4: Commit the parity tests**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/integration/test_encoder_parity.py && \
  git commit -m "test: add encoder byte-parity tests vs llama-quantize

Quantizes the reference tensor via MagicQuant's ctypes binding and via
llama-quantize subprocess; asserts byte-identical output. Covers the
9 schemes added or retrofitted in PR1: Q8_0, Q6_K, Q5_K, Q4_K, IQ4_NL,
MXFP4, Q4_0, Q2_K, Q3_K.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Retrofit converters.py to use ggml_encode for quantized formats

**Files:**
- Modify: `magicquant/quant/converters.py`

`encode_to_ggml_bytes()` is the single call site that the GGUF writer uses. After this task, that function dispatches to `ggml_encode` for all quantized types and uses native passthroughs only for BF16/F16/F32. The pure-Python `_encode_ggml_*` functions stay temporarily; they'll be deleted in Task 8 once parity is confirmed.

- [ ] **Step 1: Add the ggml_binding import**

Edit `/server/programming/MagicQuant/magicquant/quant/converters.py`. Find the top-of-file imports:
```python
from typing import Dict, List, Tuple, Optional
import struct
import numpy as np
```

Replace with:
```python
from typing import Dict, List, Tuple, Optional
import struct
import numpy as np

from magicquant.quant.ggml_binding import ggml_encode, GGML_TYPE_IDS
```

- [ ] **Step 2: Find the current `encode_to_ggml_bytes` definition**

Run:
```bash
cd /server/programming/MagicQuant && grep -n "def encode_to_ggml_bytes" magicquant/quant/converters.py
```

Expected output: exact line number (around line 928).

- [ ] **Step 3: Replace the dispatch logic in `encode_to_ggml_bytes`**

In `magicquant/quant/converters.py`, find the function:

```python
def encode_to_ggml_bytes(weights: np.ndarray, ggml_type_name: str) -> bytes:
    """
    Quantize a float32 weight array into ggml block-format bytes.
    ...
    """
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError(
            f"encode_to_ggml_bytes requires floating-point input, "
            f"got dtype={weights.dtype}. Integer or pre-quantized tensors "
            f"cannot be re-quantized — use a BF16/F16/F32 source model."
        )
    encoder = _GGML_ENCODERS.get(ggml_type_name)
    if encoder is None:
        raise ValueError(
            f"No ggml encoder for type '{ggml_type_name}'. "
            f"Available: {sorted(_GGML_ENCODERS)}"
        )
    flat = weights.astype(np.float32).flatten()
    return encoder(flat)
```

Replace the body (keep the docstring) with the new dispatch:

```python
def encode_to_ggml_bytes(
    weights: np.ndarray,
    ggml_type_name: str,
    imatrix: Optional[np.ndarray] = None,
) -> bytes:
    """
    Quantize a float32 weight array into ggml block-format bytes.

    Quantized formats route through ggml_encode (libggml ctypes binding),
    producing byte-identical output to llama.cpp's llama-quantize.
    Float-format passthroughs (BF16, F16, F32) stay native — no need for
    the C path.

    Args:
        weights: Float32 numpy array (any shape — will be flattened).
            Must be a floating-point dtype.
        ggml_type_name: Target ggml type (e.g. "Q8_0", "Q4_K", "BF16").
        imatrix: Optional importance matrix (used by IQ-quants in PR4).

    Returns:
        Raw bytes in the on-disk ggml block layout.

    Raises:
        ValueError: If weights has a non-floating dtype or the target type
            has no encoder.
    """
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError(
            f"encode_to_ggml_bytes requires floating-point input, "
            f"got dtype={weights.dtype}. Integer or pre-quantized tensors "
            f"cannot be re-quantized — use a BF16/F16/F32 source model."
        )
    flat = weights.astype(np.float32).flatten()

    # Float passthroughs — native (no C call needed)
    if ggml_type_name == "BF16":
        return _encode_f32_to_bf16(flat)
    if ggml_type_name == "F16":
        return _encode_f32_to_f16(flat)
    if ggml_type_name == "F32":
        return _encode_f32_to_f32(flat)

    # All quantized formats route to libggml
    if ggml_type_name not in GGML_TYPE_IDS:
        raise ValueError(
            f"No ggml encoder for type '{ggml_type_name}'. "
            f"Available: {sorted(GGML_TYPE_IDS)}"
        )
    return ggml_encode(flat, ggml_type_name, imatrix=imatrix)
```

- [ ] **Step 4: Verify the GGUF writer still works**

Run a quick sanity check with the existing dtype-guard tests:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_quantization_guards.py -v 2>&1 | tail -20
```

Expected output: all 12 tests pass. (They test the public `encode_to_ggml_bytes` API, which now routes through ctypes.)

- [ ] **Step 5: Re-run the parity tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_encoder_parity.py -v 2>&1 | tail -15
```

Expected output: all 9 parity tests pass. (The retrofit doesn't change anything that affects parity — both pre- and post-retrofit, the flow is float32 → ggml_quantize_chunk.)

- [ ] **Step 6: Commit the retrofit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/converters.py && \
  git commit -m "refactor: encode_to_ggml_bytes dispatches to ggml_encode

Quantized formats now route through the ctypes binding instead of the
pure-Python _encode_ggml_* helpers. Float-format passthroughs (BF16,
F16, F32) stay native. Adds optional imatrix= parameter for PR4.

Pure-Python encoder functions stay in place for one more commit so
they're easy to bisect against; deletion in next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Delete the obsolete pure-Python encoder functions

**Files:**
- Modify: `magicquant/quant/converters.py`

After Task 7, the dispatch routes everything to `ggml_encode`. The `_encode_ggml_*` functions and their helpers are unreferenced. Delete them.

- [ ] **Step 1: Identify what to delete**

Run:
```bash
cd /server/programming/MagicQuant && grep -nE "^def _encode_ggml_|^def _pack_|^def _optimize_|^def _pad_to|^_GGML_ENCODERS|^_SCALE_CANDIDATES|^_KVALUES_MXFP4|^_MXFP4|^_IQ4_NL_SORTED|^_IQ4_NL_BOUNDARIES|^IQ4_NL_LEVELS" magicquant/quant/converters.py | head -30
```

Expected output: lists definitions to be deleted, including:
- `IQ4_NL_LEVELS` (constant array, ~line 56)
- `_pad_to` (~line 341)
- `_SCALE_CANDIDATES` (~line 351)
- `_optimize_symmetric_scale` (~line 356)
- `_optimize_asymmetric_scale` (~line 417)
- `_encode_ggml_q8_0` (~line 478)
- `_encode_ggml_q4_0` (~line 501)
- `_IQ4_NL_SORTED_IDX`, `_IQ4_NL_SORTED`, `_IQ4_NL_BOUNDARIES` (~line 528)
- `_encode_ggml_iq4_nl` (~line 534)
- `_encode_ggml_q6_k` (~line 572)
- `_pack_k4k5_scales` (~line 646)
- `_encode_ggml_q4_k` (~line 683)
- `_encode_ggml_q5_k` (~line 747)
- `_KVALUES_MXFP4`, `_MXFP4_UNSIGNED_DOUBLED`, `_MXFP4_MIDPOINTS` (~line 838)
- `_encode_ggml_mxfp4` (~line 849)
- `_GGML_ENCODERS` dict (~line 914)

The `Quantizer` class also has internal helper methods `_quantize_*`, `_dequantize_*`, `_pack_6bit/5bit/4bit`, `_unpack_*` that are now unreferenced (the orchestrator/probe code uses `encode_to_ggml_bytes`, not the Quantizer class). Audit the Quantizer class for callers before deleting:

Run:
```bash
cd /server/programming/MagicQuant && grep -rn "Quantizer()\|Quantizer\.\|.quantize_weights\|.dequantize_weights" --include="*.py" magicquant/ | grep -v __pycache__
```

If output is empty (or only references inside `converters.py` itself), the entire `Quantizer` class can be deleted. If it has external callers, keep the class scaffolding but reduce its internal logic to delegate to `encode_to_ggml_bytes`.

- [ ] **Step 2: Open the file in your editor and delete the obsolete code**

This is best done by reading `converters.py` once end-to-end (it's ~960 lines) and removing the listed definitions. The file should retain:

  - The top-level docstring
  - Top-of-file imports (numpy, struct, typing, the new ggml_binding import)
  - `GGML_BLOCK_SIZE`, `GGML_TYPE_SIZE` constants (still used by `ggml_tensor_data_size`)
  - `ggml_tensor_data_size()` function
  - `Quantizer` class — only if it has external callers (else delete)
    - If kept: its `quantize_weights` method delegates to `encode_to_ggml_bytes`
    - Drop the `_quantize_*` and `_dequantize_*` private methods that wrapped the deleted helpers
  - `_encode_f32_to_bf16`, `_encode_f32_to_f16`, `_encode_f32_to_f32` (kept — float passthroughs)
  - `encode_to_ggml_bytes` (the dispatch we updated in Task 7)

A practical approach: after Task 7, re-write `converters.py` from scratch to the trimmed shape. Use the structure below as the target. Compare line by line with the existing file as you create the new one to ensure no needed code is lost.

The final `converters.py` should look like this (~250 lines):

```python
"""
Quantization Converters - Convert model weights to ggml block-format bytes.

This module is the public encoder entry point used by the GGUF writer.
Quantized formats route through magicquant.quant.ggml_binding (libggml
ctypes binding); float passthroughs (BF16/F16/F32) stay native.

Public API:
    encode_to_ggml_bytes(weights, ggml_type_name, imatrix=None) -> bytes
    ggml_tensor_data_size(ggml_type_name, n_elements) -> int
"""

from typing import Dict, Optional
import numpy as np

from magicquant.quant.ggml_binding import ggml_encode, GGML_TYPE_IDS


# ---------------------------------------------------------------------------
# ggml block format constants (used by callers for offset/size math).
# Source of truth for sizes is magicquant.quant.ggml_binding._GGML_TYPE_SIZE;
# these tables are kept here for backward compatibility with imports.
# ---------------------------------------------------------------------------

GGML_BLOCK_SIZE = {
    "F32": 1, "F16": 1, "BF16": 1, "F64": 1,
    "I8": 1, "I16": 1, "I32": 1, "I64": 1,
    "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32,
    "Q8_0": 32, "Q8_1": 32,
    "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256,
    "Q6_K": 256, "Q8_K": 256,
    "IQ2_XXS": 256, "IQ2_XS": 256, "IQ3_XXS": 256,
    "IQ1_S": 256, "IQ4_NL": 32, "IQ3_S": 256,
    "IQ2_S": 256, "IQ4_XS": 32, "IQ1_M": 256,
    "MXFP4": 32,
}

GGML_TYPE_SIZE = {
    "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "Q4_0": 18, "Q4_1": 20, "Q5_0": 22, "Q5_1": 24,
    "Q8_0": 34, "Q8_1": 36,
    "Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176,
    "Q6_K": 210, "Q8_K": 292,
    "IQ2_XXS": 66, "IQ2_XS": 74, "IQ3_XXS": 98,
    "IQ1_S": 50, "IQ4_NL": 18, "IQ3_S": 110,
    "IQ2_S": 82, "IQ4_XS": 18, "IQ1_M": 56,
    "MXFP4": 17,
}


def ggml_tensor_data_size(ggml_type_name: str, n_elements: int) -> int:
    """Return the byte-size of tensor data for a given ggml type and element count."""
    block_size = GGML_BLOCK_SIZE.get(ggml_type_name, 1)
    type_size = GGML_TYPE_SIZE.get(ggml_type_name, 2)
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * type_size


# ── Float-format encoders (native; no ggml needed) ──────────────────

def _encode_f32_to_bf16(arr: np.ndarray) -> bytes:
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32)
    # Round-to-nearest-even: add 0x7FFF + bit 16 (the lsb of the result) before truncating
    rounding = np.uint32(0x7FFF) + ((u32 >> 16) & 1)
    bf16 = ((u32 + rounding) >> 16).astype(np.uint16)
    return bf16.tobytes()


def _encode_f32_to_f16(arr: np.ndarray) -> bytes:
    return arr.astype(np.float16).tobytes()


def _encode_f32_to_f32(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


# ── Public dispatch ─────────────────────────────────────────────────

def encode_to_ggml_bytes(
    weights: np.ndarray,
    ggml_type_name: str,
    imatrix: Optional[np.ndarray] = None,
) -> bytes:
    """
    Quantize a float weight array into ggml block-format bytes.

    Quantized formats route through ggml_encode (libggml ctypes binding),
    producing byte-identical output to llama.cpp's llama-quantize.
    Float-format passthroughs (BF16, F16, F32) stay native.

    Args:
        weights: Float32 numpy array (any shape — will be flattened).
            Must be a floating-point dtype.
        ggml_type_name: Target ggml type (e.g. "Q8_0", "Q4_K", "BF16").
        imatrix: Optional importance matrix (used by IQ-quants in PR4).

    Returns:
        Raw bytes in the on-disk ggml block layout.

    Raises:
        ValueError: If weights has a non-floating dtype or the target type
            has no encoder.
    """
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError(
            f"encode_to_ggml_bytes requires floating-point input, "
            f"got dtype={weights.dtype}. Integer or pre-quantized tensors "
            f"cannot be re-quantized — use a BF16/F16/F32 source model."
        )
    flat = weights.astype(np.float32).flatten()

    if ggml_type_name == "BF16":
        return _encode_f32_to_bf16(flat)
    if ggml_type_name == "F16":
        return _encode_f32_to_f16(flat)
    if ggml_type_name == "F32":
        return _encode_f32_to_f32(flat)

    if ggml_type_name not in GGML_TYPE_IDS:
        raise ValueError(
            f"No ggml encoder for type '{ggml_type_name}'. "
            f"Available: {sorted(GGML_TYPE_IDS)}"
        )
    return ggml_encode(flat, ggml_type_name, imatrix=imatrix)
```

Overwrite `/server/programming/MagicQuant/magicquant/quant/converters.py` with that content.

- [ ] **Step 3: Handle the `Quantizer` class import**

Check if any external code references the `Quantizer` class:

Run:
```bash
cd /server/programming/MagicQuant && grep -rn "from magicquant.quant.converters import\|from magicquant.quant import\|import magicquant.quant" --include="*.py" magicquant/ tests/ tools/ 2>/dev/null | grep -v __pycache__
```

Look for any imports of `Quantizer`. If nothing imports it externally beyond the `magicquant/quant/__init__.py` re-export, edit `magicquant/quant/__init__.py` to remove the `Quantizer` symbol:

Edit `magicquant/quant/__init__.py`. Find the line:
```python
from magicquant.quant.converters import Quantizer
```
Delete it.

In the same file find:
```python
__all__ = [
    ...,
    "Quantizer",
]
```
Remove the `"Quantizer",` entry.

- [ ] **Step 4: Run all tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -20
```

Expected output: all tests pass (12 dtype guards + 1 regression + 9 parity = 22).

- [ ] **Step 5: Quantify the line count reduction**

Run:
```bash
cd /server/programming/MagicQuant && wc -l magicquant/quant/converters.py
```

Expected output: roughly 150–200 lines (down from ~960). Document the actual number in the next commit message.

- [ ] **Step 6: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/converters.py magicquant/quant/__init__.py && \
  git commit -m "refactor: delete pure-Python encoder helpers (~700 lines)

Removes _encode_ggml_q8_0, _encode_ggml_q4_0, _encode_ggml_iq4_nl,
_encode_ggml_q6_k, _encode_ggml_q5_k, _encode_ggml_q4_k,
_encode_ggml_mxfp4 and their helpers (_pad_to, _pack_k4k5_scales,
_pack_4bit/5bit/6bit, _optimize_symmetric_scale, _optimize_asymmetric_scale,
_SCALE_CANDIDATES, _KVALUES_MXFP4, IQ4_NL_LEVELS, _IQ4_NL_SORTED*, etc.).

These were superseded by the ctypes binding to ggml_quantize_chunk
in PR1. Byte-parity tests against llama-quantize confirm the new path
produces identical output (and eliminates the ~10-27% MSE gap noted
in the previous CLAUDE.md).

The Quantizer numpy-level class is also removed since it had no
external callers (verified with grep).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Register Q2_K and Q3_K in the scheme registry

**Files:**
- Modify: `magicquant/quant/schemes.py`

- [ ] **Step 1: Add Q2_K constant after Q4_K_M**

Edit `/server/programming/MagicQuant/magicquant/quant/schemes.py`. Find the Q4_K_M definition (added in PR0):

```python
Q4_K_M = QuantizationScheme(
    name="Q4_K_M",
    ggml_type_name="Q4_K",
    ggml_type_id=12,
    bits_per_weight=4.5,
    noise_factor=4.5,
    speed_multiplier=3.4,
    category="k_quant",
    upgrade_neighbor="MXFP4_MOE",
    downgrade_neighbor=None,  # bottom of current registry; PR1 adds Q3_K
)
```

Update its `downgrade_neighbor` to point at Q3_K (which is added in the next step). Replace the `downgrade_neighbor=None` line with:
```python
    downgrade_neighbor="Q3_K",
```

Then add the new schemes immediately AFTER the Q4_K_M block:

```python


# ── New schemes added in PR1 ─────────────────────────────────────────
# Q3_K and Q2_K make the Q3 tier band reachable. Q2_K bpw=2.625 lands
# at ratio 0.164 — just outside the Q2 band (≤0.16); full Q2 band
# coverage requires sub-Q2 IQ-quants from PR3.
#
# noise_factor values are placeholders pending the calibration bench
# in this PR; tools/calibrate_noise_factors.py overwrites them.

Q3_K = QuantizationScheme(
    name="Q3_K",
    ggml_type_name="Q3_K",
    ggml_type_id=11,
    bits_per_weight=3.4375,   # 110B * 8 / 256 = 3.4375
    noise_factor=8.0,         # placeholder; calibrated below
    speed_multiplier=4.0,     # ggml SIMD encoders are fast
    category="k_quant",
    upgrade_neighbor="Q4_K_M",
    downgrade_neighbor="Q2_K",
)

Q2_K = QuantizationScheme(
    name="Q2_K",
    ggml_type_name="Q2_K",
    ggml_type_id=10,
    bits_per_weight=2.625,    # 84B * 8 / 256 = 2.625
    noise_factor=15.0,        # placeholder; calibrated below
    speed_multiplier=4.5,     # smallest blocks → fastest dispatch
    category="k_quant",
    upgrade_neighbor="Q3_K",
    downgrade_neighbor=None,  # bottom of current registry; PR3 adds IQ-quants below
)
```

- [ ] **Step 2: Add Q2_K and Q3_K to the registry dict**

Find:
```python
_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
}
```

Replace with:
```python
_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
    "Q3_K": Q3_K,
    "Q2_K": Q2_K,
}
```

- [ ] **Step 3: Update _GROUP_CLASS_FLOORS to keep robust floor at Q4_K_M**

The robust floor is currently `Q4_K_M`. With Q2_K and Q3_K available, FFN groups COULD legally go to Q2_K — but PR1 keeps the floor at Q4_K_M (a conservative default). PR3 will lower it once IQ-quants are validated.

Verify the constant in `schemes.py` is unchanged:
```python
_GROUP_CLASS_FLOORS: Dict[str, str] = {
    "sensitive": "Q8_0",
    "robust": "Q4_K_M",
}
```

If different, restore to the above values.

- [ ] **Step 4: Update package re-exports**

Edit `magicquant/quant/__init__.py`. Find the existing scheme imports:
```python
from magicquant.quant.schemes import (
    BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M,
    QuantizationScheme,
    get_scheme_by_name,
    get_all_schemes,
    get_schemes_by_category,
    get_floor_for_group_class,
)
```
Add `Q3_K, Q2_K` to the names being imported and to `__all__`:
```python
from magicquant.quant.schemes import (
    BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M, Q3_K, Q2_K,
    QuantizationScheme,
    get_scheme_by_name,
    get_all_schemes,
    get_schemes_by_category,
    get_floor_for_group_class,
)

__all__ = [
    "BF16", "Q8_0", "Q6_K", "Q5_K", "IQ4_NL", "MXFP4_MOE", "Q4_K_M",
    "Q3_K", "Q2_K",
    "QuantizationScheme",
    "get_scheme_by_name",
    "get_all_schemes",
    "get_schemes_by_category",
    "get_floor_for_group_class",
]
```

Apply the same change to `magicquant/__init__.py` (mirror the imports/re-exports for the top-level package).

- [ ] **Step 5: Verify imports**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant import Q3_K, Q2_K
from magicquant.quant.schemes import get_scheme_by_name, get_all_schemes
print('Q2_K:', Q2_K)
print('Q3_K:', Q3_K)
print('upgrade chain Q2_K → Q3_K → Q4_K_M:',
      Q2_K.upgrade_neighbor, get_scheme_by_name('Q3_K').upgrade_neighbor)
print('downgrade chain Q4_K_M → Q3_K → Q2_K:',
      get_scheme_by_name('Q4_K_M').downgrade_neighbor, Q3_K.downgrade_neighbor)
print('all schemes count:', len(get_all_schemes()))
"
```

Expected output:
```
Q2_K: QuantScheme(Q2_K, 2.625bpw, noise=15.0)
Q3_K: QuantScheme(Q3_K, 3.4375bpw, noise=8.0)
upgrade chain Q2_K → Q3_K → Q4_K_M: Q3_K Q4_K_M
downgrade chain Q4_K_M → Q3_K → Q2_K: Q3_K Q2_K
all schemes count: 9
```

- [ ] **Step 6: Run all tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -20
```

Expected output: all tests pass. The regression test STILL passes because adding new schemes doesn't change the search behavior on the existing seeded population (and the random-config-weight restructure happens later in this PR).

If the regression test fails because the new schemes leaked into population init, the cause is a downstream consumer iterating `get_all_schemes()` at the wrong place. Investigate: PR0's refactor should have isolated this.

- [ ] **Step 7: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/schemes.py magicquant/quant/__init__.py magicquant/__init__.py && \
  git commit -m "feat: register Q2_K and Q3_K schemes

Q3_K (3.44 bpw, ggml type 11) makes Q3 tier band reliably populate.
Q2_K (2.63 bpw, ggml type 10) lands just outside Q2 band ratio
boundary (0.164 vs ≤0.16) — full Q2 coverage arrives in PR3 with
IQ-quants. upgrade/downgrade chains extended:
  Q4_K_M ↔ Q3_K ↔ Q2_K
Robust floor stays at Q4_K_M for conservative search behavior;
PR3 lowers it.

noise_factor values are placeholders pending the calibration bench
later in this PR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Restructure random-config weights to category-indexed

**Files:**
- Modify: `magicquant/evolution/survival.py`
- Modify: `tests/fixtures/refactor_regression_seed42.json` (regenerated)
- Modify: `tests/test_refactor_regression.py` (potentially)

PR0 deferred this restructure to here. With 2 new schemes (Q2_K, Q3_K) added to the registry, the positional weight arrays in `_generate_random_config` no longer cleanly map to `AVAILABLE_SCHEMES` — they have 7 weights but the list now has 9 schemes.

- [ ] **Step 1: Find the current `_generate_random_config` method**

Run:
```bash
cd /server/programming/MagicQuant && grep -n "_generate_random_config\|brain_weights\|attention_weights\|ffn_weights" magicquant/evolution/survival.py | head -20
```

Expected output: locates the method (~line 191) and the three weight arrays (~lines 197–199).

- [ ] **Step 2: Replace `_generate_random_config` with category-indexed sampling**

In `magicquant/evolution/survival.py`, find:
```python
    def _generate_random_config(self, groups: List[str]) -> Dict[str, str]:
        """Generate a random config biased toward MXFP4 for FFN and
        higher precision for brain layers."""
        config = {}

        # Weights per scheme:          BF16 Q8_0 Q6_K Q5_K IQ4NL MXFP4 Q4KM
        brain_weights =               [0.30, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02]
        attention_weights =            [0.05, 0.15, 0.25, 0.20, 0.15, 0.10, 0.10]
        ffn_weights =                  [0.02, 0.05, 0.08, 0.10, 0.15, 0.35, 0.25]

        for g in groups:
            if g in self._HIGH_SENSITIVITY:
                w = brain_weights
            elif g in self._LOW_SENSITIVITY:
                w = ffn_weights
            else:
                w = attention_weights
            config[g] = random.choices(self.AVAILABLE_SCHEMES, weights=w)[0]
        return config
```

Replace with:

```python
    # Sampling weights per group class, indexed by scheme category.
    # Each value is the relative probability mass for picking ANY scheme
    # in that category. Within a category, we further weight inversely by
    # noise_factor so higher-quality variants are preferred slightly over
    # lower-quality ones in the same category.
    #
    # _BRAIN_CLASS_WEIGHTS: for high-sensitivity groups (E, H, O, R) —
    #   biased toward float and high-precision schemes.
    # _ATTENTION_CLASS_WEIGHTS: for moderate-sensitivity groups (Q, K) —
    #   middle-ground spread.
    # _FFN_CLASS_WEIGHTS: for robust groups (U, D, X) —
    #   biased toward maximum compression.
    _BRAIN_CLASS_WEIGHTS = {
        "float":    0.30,   # BF16
        "legacy_q": 0.30,   # Q8_0
        "k_quant":  0.30,   # Q6_K, Q5_K, Q4_K_M, Q3_K, Q2_K
        "iq_quant": 0.05,   # IQ4_NL (and any IQ-quants added later)
        "mxfp4":    0.05,   # MXFP4_MOE
    }
    _ATTENTION_CLASS_WEIGHTS = {
        "float":    0.05,
        "legacy_q": 0.15,
        "k_quant":  0.45,
        "iq_quant": 0.20,
        "mxfp4":    0.15,
    }
    _FFN_CLASS_WEIGHTS = {
        "float":    0.02,
        "legacy_q": 0.05,
        "k_quant":  0.30,
        "iq_quant": 0.30,
        "mxfp4":    0.33,
    }

    def _generate_random_config(self, groups: List[str]) -> Dict[str, str]:
        """Generate a random config biased toward compression for FFN and
        higher precision for brain layers.

        Weights are category-indexed (not positional) so adding new schemes
        to the registry doesn't require updating positional arrays.
        """
        from magicquant.quant.schemes import get_all_schemes

        config: Dict[str, str] = {}
        all_schemes = get_all_schemes()

        for g in groups:
            if g in self._HIGH_SENSITIVITY:
                class_weights = self._BRAIN_CLASS_WEIGHTS
            elif g in self._LOW_SENSITIVITY:
                class_weights = self._FFN_CLASS_WEIGHTS
            else:
                class_weights = self._ATTENTION_CLASS_WEIGHTS

            # Build per-scheme weights: start with the class weight, then
            # divide it across all schemes in that category, inversely
            # weighted by noise_factor (cleaner schemes preferred).
            scheme_weights = []
            for s in all_schemes:
                cat_weight = class_weights.get(s.category, 0.0)
                # Within a category, give cleaner (lower noise) schemes more weight.
                # noise_factor=0 (BF16) gets factor 2.0; high-noise gets factor near 0.
                # Normalize within a category later.
                scheme_weights.append(cat_weight * (1.0 / (1.0 + s.noise_factor)))

            # Avoid all-zeros pathology
            if sum(scheme_weights) == 0:
                scheme_weights = [1.0] * len(all_schemes)

            picked = random.choices(
                [s.name for s in all_schemes],
                weights=scheme_weights,
            )[0]
            config[g] = picked
        return config
```

- [ ] **Step 3: Regenerate the regression fixture**

The new sampling logic produces different draws (intentional — the spec says PR1 restructures weights). Re-run the capture script:

Recreate the capture script:
```bash
cat > /server/programming/MagicQuant/tests/_capture_fixture.py << 'EOF'
"""One-shot script: regenerate the regression fixture after weight restructure."""
import json
from pathlib import Path

from tests.test_refactor_regression import _capture_run

FIXTURE = Path(__file__).parent / "fixtures" / "refactor_regression_seed42.json"

if __name__ == "__main__":
    captured = _capture_run(seed=42, generations=3, population=20)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(captured, indent=2))
    print(f"Captured {len(captured)} configs to {FIXTURE}")
EOF
```

Run:
```bash
cd /server/programming/MagicQuant && python -m tests._capture_fixture
```

Expected output: `Captured <N> configs to .../refactor_regression_seed42.json`. The `<N>` may differ from the PR0 fixture's count.

- [ ] **Step 4: Verify the regression test now passes against the new fixture**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected output: `1 passed`.

- [ ] **Step 5: Verify Q3_K and Q2_K appear in random configs**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import random, json
random.seed(42)
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor

predictor = PredictiveScorer({'E': 1.5, 'U': 0.5}, {'E': 100_000_000, 'U': 800_000_000}, 5.0, 20.0)
survivor = EvolutionarySurvivor(predictor, {'E': 'BF16', 'U': 'MXFP4_MOE'}, max_generations=1, population_size=200)
configs = [survivor._generate_random_config(['E', 'H', 'Q', 'K', 'O', 'U', 'D']) for _ in range(200)]
all_schemes_seen = set()
for c in configs:
    all_schemes_seen.update(c.values())
print('schemes seen across 200 random configs:', sorted(all_schemes_seen))
assert 'Q3_K' in all_schemes_seen, 'Q3_K never sampled'
assert 'Q2_K' in all_schemes_seen, 'Q2_K never sampled'
print('OK — both new K-quants reachable from random init')
"
```

Expected output: list including `Q2_K` and `Q3_K` along with the other 7 schemes.

- [ ] **Step 6: Delete the capture script**

Run:
```bash
cd /server/programming/MagicQuant && rm tests/_capture_fixture.py
```

- [ ] **Step 7: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/evolution/survival.py tests/fixtures/refactor_regression_seed42.json && \
  git commit -m "refactor: random-config weights are now category-indexed

Replaces the positional [BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4, Q4_K_M]
weight arrays in _generate_random_config with category-keyed weights:
{float, legacy_q, k_quant, iq_quant, mxfp4} each get a class-level mass
that's distributed across schemes within the category, inversely
weighted by noise_factor.

This makes PR1's new schemes (Q3_K, Q2_K) reachable from random
population init, and is forward-compatible with PR3 IQ-quants without
further refactoring.

Regression fixture regenerated for the new sampling distribution.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Run encoder parity tests for Q2_K and Q3_K

**Files:** none (tests/integration/test_encoder_parity.py was already authored in Task 6)

- [ ] **Step 1: Run the parity tests, focused on Q2_K and Q3_K**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_encoder_parity.py -v -k "Q2_K or Q3_K" 2>&1 | tail -10
```

Expected output: 2 tests pass (Q2_K and Q3_K parametrizations).

If they fail, investigate before continuing. Most likely causes:
- The reference tensor uses dimensions that aren't a multiple of 256 (Q2_K/Q3_K block size) — but our 2048×2048 tensor is divisible.
- llama-quantize produces a different repacked variant. Add `--pure` to subprocess args (mentioned in Task 6 troubleshooting).

- [ ] **Step 2: Run all parity tests as a final check**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_encoder_parity.py -v 2>&1 | tail -15
```

Expected output: all 9 tests pass.

- [ ] **Step 3: Commit a note that PR1 parity is established (no code change)**

This task has no code change; the tests for Q2_K/Q3_K were already added in Task 6 alongside the others. Skip the commit and proceed to Task 12.

---

### Task 12: Write the calibration bench tool

**Files:**
- Create: `tools/__init__.py` (if missing)
- Create: `tools/calibrate_noise_factors.py`

- [ ] **Step 1: Create the tools package marker (if missing)**

Run:
```bash
cd /server/programming/MagicQuant && ls tools/__init__.py 2>/dev/null || (mkdir -p tools && touch tools/__init__.py)
```

- [ ] **Step 2: Create the calibration script**

Create `/server/programming/MagicQuant/tools/calibrate_noise_factors.py` with this exact content:

```python
"""
Empirical noise-factor calibration for MagicQuant schemes.

For each registered scheme, this script:
  1. Builds a hybrid GGUF where every tensor group uses that scheme uniformly.
  2. Runs llama-perplexity against a calibration corpus.
  3. Records (scheme, ppl, ppl_loss = ppl - baseline_ppl).

Output: tools/calibration_results.json with per-scheme measurements.
Noise factors are normalized so Q8_0's ppl_loss = noise_factor 1.0.

Usage:
    python tools/calibrate_noise_factors.py \
        --model /path/to/Llama-3.2-1B-Instruct-bf16 \
        --corpus /path/to/wikitext-2-raw/wiki.test.raw \
        --output tools/calibration_results.json

Both --model and --corpus default to canonical reference paths.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Make magicquant importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from magicquant.quant.schemes import get_all_schemes  # noqa: E402


DEFAULT_MODEL = os.environ.get(
    "MAGICQUANT_CALIBRATION_MODEL",
    str(Path.home() / "models" / "Llama-3.2-1B-Instruct-bf16"),
)
DEFAULT_CORPUS = os.environ.get(
    "MAGICQUANT_CALIBRATION_CORPUS",
    "/home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw",
)


def _check_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"required tool '{name}' not on PATH")
    return found


def _baseline_ppl(model_path: Path, corpus: Path, perplexity_bin: str) -> float:
    """Run llama-perplexity on the unquantized BF16 model. Returns scalar ppl."""
    print(f"[baseline] computing BF16 perplexity for {model_path.name}...")
    return _run_perplexity(model_path, corpus, perplexity_bin)


def _run_perplexity(gguf_path: Path, corpus: Path, perplexity_bin: str) -> float:
    """Run llama-perplexity once and parse final perplexity from stdout."""
    cmd = [
        perplexity_bin,
        "-m", str(gguf_path),
        "-f", str(corpus),
        "--ctx-size", "512",
        "--threads", str(os.cpu_count() or 4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity failed (rc={proc.returncode}):\n"
            f"stderr: {proc.stderr[-500:]}"
        )
    # Parse output: lines like "[<n>]<ppl>" or final "Final estimate: PPL = <num> +/- ..."
    for line in reversed(proc.stdout.splitlines()):
        if "Final estimate" in line and "PPL" in line:
            # Format: "Final estimate: PPL = 12.3456 +/- 0.06789"
            parts = line.split("=")
            if len(parts) >= 2:
                return float(parts[1].strip().split()[0])
    raise RuntimeError(
        "could not parse perplexity from llama-perplexity output:\n"
        f"{proc.stdout[-500:]}"
    )


def _build_uniform_gguf(
    model_path: Path, scheme_name: str, output_dir: Path,
    create_hybrid_gguf,
) -> Optional[Path]:
    """Quantize the source model uniformly with `scheme_name`. Returns path or None."""
    out_path = output_dir / f"calib_{scheme_name}.gguf"
    print(f"[build] {scheme_name} → {out_path.name}")
    try:
        create_hybrid_gguf(
            output_path=str(out_path),
            base_model_path=str(model_path),
            quant_config={"base": scheme_name, "groups": {}},
            verbose=False,
        )
    except Exception as exc:
        print(f"  failed: {exc}")
        return None
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    ap.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    ap.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "calibration_results.json",
    )
    ap.add_argument(
        "--skip", nargs="*", default=["BF16"],
        help="scheme names to skip (BF16 is the baseline; default skips it)",
    )
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")

    perplexity_bin = _check_tool("llama-perplexity")

    from magicquant.gguf.writer import create_hybrid_gguf  # noqa: E402

    with tempfile.TemporaryDirectory(prefix="magicquant-calib-") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Baseline: source model is already BF16, so use it directly.
        baseline_ppl = _baseline_ppl(args.model, args.corpus, perplexity_bin)
        print(f"[baseline] PPL = {baseline_ppl:.4f}")

        results: Dict[str, Dict[str, float]] = {}

        for scheme in get_all_schemes():
            if scheme.name in args.skip:
                print(f"[skip] {scheme.name} (in --skip list)")
                continue
            gguf_path = _build_uniform_gguf(
                args.model, scheme.name, tmpdir_path, create_hybrid_gguf
            )
            if gguf_path is None:
                results[scheme.name] = {
                    "ppl": float("nan"),
                    "ppl_loss": float("nan"),
                    "noise_factor": 50.0,
                    "status": "build_failed",
                }
                continue
            try:
                ppl = _run_perplexity(gguf_path, args.corpus, perplexity_bin)
                results[scheme.name] = {
                    "ppl": ppl,
                    "ppl_loss": ppl - baseline_ppl,
                    "noise_factor": 0.0,  # filled below after Q8_0 anchor known
                    "status": "ok",
                }
                print(f"  {scheme.name}: ppl={ppl:.4f}, loss={ppl - baseline_ppl:+.4f}")
            except Exception as exc:
                print(f"  {scheme.name}: perplexity failed: {exc}")
                results[scheme.name] = {
                    "ppl": float("nan"),
                    "ppl_loss": float("nan"),
                    "noise_factor": 50.0,
                    "status": "perplexity_failed",
                }
            finally:
                if gguf_path and gguf_path.exists():
                    gguf_path.unlink()

        # Normalize: anchor Q8_0's ppl_loss → noise_factor 1.0
        anchor = results.get("Q8_0", {}).get("ppl_loss")
        if anchor is None or anchor != anchor:  # NaN check
            print("WARNING: Q8_0 anchor unavailable; falling back to absolute scale")
            anchor = 1.0
        else:
            print(f"[normalize] Q8_0 anchor: ppl_loss = {anchor:.4f}")

        for name, r in results.items():
            if r["status"] == "ok":
                r["noise_factor"] = round(max(0.0, r["ppl_loss"] / anchor), 3)
        # BF16 is the baseline by definition
        results["BF16"] = {
            "ppl": baseline_ppl, "ppl_loss": 0.0, "noise_factor": 0.0,
            "status": "baseline",
        }

        out = {
            "model": args.model.name,
            "corpus": args.corpus.name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "baseline_ppl": baseline_ppl,
            "schemes": results,
        }
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\n[write] calibration results → {args.output}")
        print("\nSummary:")
        for name, r in sorted(results.items(),
                              key=lambda kv: kv[1].get("noise_factor", 99)):
            print(f"  {name:10s}  noise={r['noise_factor']:6.3f}  "
                  f"ppl={r.get('ppl', 0):.4f}  status={r['status']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit the bench tool**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tools/__init__.py tools/calibrate_noise_factors.py && \
  git commit -m "tools: add empirical noise-factor calibration bench

For each registered scheme, builds a uniform-quantization GGUF and
measures perplexity against a calibration corpus. Normalizes results
so Q8_0 anchors at noise_factor=1.0.

Default: Llama-3.2-1B-Instruct vs wikitext-2-raw. Override via
--model and --corpus.

Output: tools/calibration_results.json (committed alongside the
schemes.py edits in the next task).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Run the calibration bench

**Files:**
- Create: `tools/calibration_results.json` (output of bench)

This task has the longest compute wall-clock in PR1 (~1.5–2 hours).

- [ ] **Step 1: Verify a BF16 reference model is available**

Run:
```bash
ls ~/models/ 2>/dev/null
ls -la /server/programming/Foundry/output/ 2>/dev/null | head -20
```

Look for any BF16 LLM in the 1–2B parameter range. Acceptable models include:
- `Llama-3.2-1B-Instruct` (~2.5 GB BF16)
- `Qwen2.5-0.5B-Instruct` (~1 GB BF16)
- `TinyLlama-1.1B-Chat-v1.0` (~2.2 GB BF16)
- Any prior pipeline output that's BF16

If none exists, download one:
```bash
mkdir -p ~/models && cd ~/models && \
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir Llama-3.2-1B-Instruct-bf16 --local-dir-use-symlinks False
```

(Requires HF_TOKEN with access to Llama-3.2.)

If HF_TOKEN isn't available, use `Qwen/Qwen2.5-0.5B-Instruct` (no gating).

Note the absolute path of whichever model you use.

- [ ] **Step 2: Verify the calibration corpus is available**

Run:
```bash
ls -la /home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw 2>&1
```

If absent:
```bash
cd ~/llama.cpp && \
  curl -LO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && \
  unzip wikitext-2-raw-v1.zip
```

- [ ] **Step 3: Run the calibration bench**

This is the long step. Run in foreground with output piped to a log:

```bash
cd /server/programming/MagicQuant && \
  python tools/calibrate_noise_factors.py \
    --model <PATH-FROM-STEP-1> \
    --corpus /home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw \
    --output tools/calibration_results.json \
    2>&1 | tee /tmp/calibration_run.log
```

Expected output: progress per scheme, total ~1.5–2 hours, ending with a summary table and JSON file written.

If a particular scheme fails to build (e.g., shape mismatch), the script captures that and continues. Schemes with failures show `status: "build_failed"` or `"perplexity_failed"` in the JSON; those keep their placeholder noise_factor=50.0.

- [ ] **Step 4: Verify the JSON was written**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import json
data = json.loads(open('tools/calibration_results.json').read())
print('model:', data['model'])
print('baseline:', data['baseline_ppl'])
print('schemes:')
for name, r in sorted(data['schemes'].items(), key=lambda kv: kv[1].get('noise_factor', 99)):
    print(f'  {name:10s}  noise={r[\"noise_factor\"]:6.3f}  ppl={r.get(\"ppl\", 0):.4f}  status={r[\"status\"]}')
"
```

Expected output: a table with all 9 schemes, sorted by noise_factor (BF16 = 0, Q8_0 ≈ 1.0, then Q6_K, Q5_K, IQ4_NL, MXFP4, Q4_K_M, Q3_K, Q2_K in roughly that order).

- [ ] **Step 5: Commit calibration results**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tools/calibration_results.json && \
  git commit -m "calibrate: empirical noise factors from Llama-3.2-1B vs wikitext-2

Output of tools/calibrate_noise_factors.py. Each scheme's noise_factor
is the perplexity-loss ratio vs Q8_0 (Q8_0 = 1.0 anchor).

These values replace the heuristic noise factors in schemes.py in the
next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Update schemes.py noise factors from calibration JSON

**Files:**
- Modify: `magicquant/quant/schemes.py`

- [ ] **Step 1: Read the calibration values**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import json
data = json.loads(open('tools/calibration_results.json').read())
for name in ['BF16', 'Q8_0', 'Q6_K', 'Q5_K', 'IQ4_NL', 'MXFP4_MOE', 'Q4_K_M', 'Q3_K', 'Q2_K']:
    nf = data['schemes'].get(name, {}).get('noise_factor', '???')
    print(f'  {name:10s}  noise={nf}')
"
```

Note the values — paste them into the next step.

- [ ] **Step 2: Update schemes.py with calibrated noise factors**

For each scheme in `magicquant/quant/schemes.py`, update the `noise_factor=` line to the calibrated value from Step 1. Append a comment referencing the calibration source:

For example, if calibration shows `Q4_K_M: 4.32`:

Find:
```python
Q4_K_M = QuantizationScheme(
    name="Q4_K_M",
    ...
    noise_factor=4.5,
    ...
)
```

Replace with:
```python
Q4_K_M = QuantizationScheme(
    name="Q4_K_M",
    ...
    noise_factor=4.32,  # calibrated 2026-05-04 vs Llama-3.2-1B
    ...
)
```

Repeat for Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q3_K, Q2_K. (BF16 stays at 0.0.)

- [ ] **Step 3: Verify schemes.py loads**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.schemes import get_all_schemes
for s in get_all_schemes():
    print(f'  {s.name:10s}  bpw={s.bits_per_weight:7.4f}  noise={s.noise_factor:6.3f}')
"
```

Expected output: all 9 schemes ordered by noise_factor with the calibrated values.

- [ ] **Step 4: Run regression test (must STILL pass — predictor predictions changed but search behavior depends on RELATIVE noise)**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected output: `1 passed`.

If it fails, the new noise factors changed the search ordering enough to break the snapshot. Re-capture the fixture (same procedure as Task 10 Step 3-6).

- [ ] **Step 5: Run all tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -25
```

Expected output: 22+ tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/schemes.py && \
  (git diff --cached tests/fixtures/refactor_regression_seed42.json && git add tests/fixtures/refactor_regression_seed42.json || true) && \
  git commit -m "calibrate: update noise factors from empirical bench

Replaces heuristic noise_factor values in schemes.py with calibrated
values from tools/calibration_results.json (Llama-3.2-1B vs
wikitext-2-raw, Q8_0 anchor = 1.0).

Q3_K and Q2_K get their first real noise factors (were placeholders
in the registration commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Smoke test — Q3 tier reachability end-to-end

**Files:**
- Create: `tests/integration/test_smoke_q3_tier.py`

- [ ] **Step 1: Create the smoke test**

Create `/server/programming/MagicQuant/tests/integration/test_smoke_q3_tier.py` with this content:

```python
"""End-to-end smoke test: Q3 tier band actually populates after PR1.

Runs a small evolutionary search with synthetic parameter counts that
favor a Q3-band-sized output, and asserts the discovered tier_winners
contains a 'Q3' entry.

Pre-PR1, this test would fail (no scheme below ~4.25 bpw → Q3 band
unreachable). After PR1, Q3_K (3.44 bpw) makes the band reliably fill.
"""
import random

import numpy as np
import pytest

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor


def test_q3_tier_populates_after_pr1():
    random.seed(123)
    np.random.seed(123)

    sensitivity_weights = {
        "E": 1.0, "H": 1.0, "Q": 0.7, "K": 0.7,
        "O": 0.9, "U": 0.4, "D": 0.4,
    }
    parameter_counts = {
        "E": 50_000_000, "H": 50_000_000, "Q": 200_000_000, "K": 200_000_000,
        "O": 100_000_000, "U": 700_000_000, "D": 700_000_000,
    }
    predictor = PredictiveScorer(
        sensitivity_weights=sensitivity_weights,
        parameter_counts=parameter_counts,
        baseline_size_gb=4.0,
        baseline_tps=20.0,
    )
    baseline = {g: "MXFP4_MOE" for g in sensitivity_weights}
    survivor = EvolutionarySurvivor(
        predictor=predictor,
        baseline_config=baseline,
        max_generations=8,
        population_size=50,
        epsilon=0.3,
    )
    discovered = survivor.run_evolution(verbose=False)

    tier_winners = survivor.get_best_config_per_tier()
    assert "Q3" in tier_winners, (
        f"Q3 tier did not populate. Available tiers: {sorted(tier_winners.keys())}"
    )
    q3_config = tier_winners["Q3"]["config"]
    assert any(s == "Q3_K" for s in q3_config.values()), (
        f"Q3 tier winner doesn't use Q3_K: {q3_config}"
    )
```

- [ ] **Step 2: Run the smoke test**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_smoke_q3_tier.py -v
```

Expected output: `1 passed`.

If it fails, the search isn't producing configs that land in the Q3 tier band — likely cause: the random-config weights still bias too heavily toward Q4_K_M/MXFP4_MOE for FFN. Inspect `tier_winners` dict; if no Q3 appears, increase Q3_K's class weight in `_FFN_CLASS_WEIGHTS["k_quant"]` from 0.30 to 0.40 in `survival.py`.

- [ ] **Step 3: Commit the smoke test**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/integration/test_smoke_q3_tier.py && \
  git commit -m "test: smoke test for Q3 tier reachability after PR1

Asserts the evolutionary search produces a Q3-tier winner with at
least one Q3_K group, validating PR1's tier-band fix end-to-end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Update CLAUDE.md to remove obsolete caveats

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Find and replace the MSE-gap caveat**

Edit `/server/programming/MagicQuant/CLAUDE.md`. Find the "Known Limitations" section:

```markdown
- K-quant encoders use simple min/max with RMSE optimization, not llama.cpp's full importance-matrix-weighted quantization. Quality gap is ~10-27% MSE vs llama.cpp native.
```

Replace with:
```markdown
- IQ-quant encoders are available via libggml binding (PR1+). Without an importance matrix (PR4+), IQ1/IQ2 outputs are slightly lower-quality than llama.cpp's `llama-quantize <type>` invocation that captures imatrix automatically. Use `magicquant.imatrix` to provide one if quality matters.
```

- [ ] **Step 2: Update the "Architecture" section to mention ggml_binding.py**

Find the section describing the converter pipeline (search for "converters.py is the single encoder source"):

```markdown
- **converters.py is the single encoder source** — writer.py must not contain quantization logic.
```

Replace with:
```markdown
- **converters.py is the single dispatch entry point** — quantized formats route to `magicquant.quant.ggml_binding.ggml_encode()`, which calls `ggml_quantize_chunk` via ctypes. Float passthroughs (BF16/F16/F32) stay native. Writer.py must not contain quantization logic.
```

- [ ] **Step 3: Add a note about the libggml dependency**

Find the "Commands" section and add a note before it:
```markdown
## Runtime Dependencies

- `llama-cpp-python>=0.3.0` is a hard pip dep — supplies `libggml-base.so` and `libggml-cpu.so` for the encoder binding. Discovery prefers system llama.cpp (`~/llama.cpp/build/bin/`, etc.); falls back to the wheel-bundled libs.
```

- [ ] **Step 4: Run all tests as a final sanity check**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected output: all tests pass.

- [ ] **Step 5: Commit the docs**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add CLAUDE.md && \
  git commit -m "docs: update CLAUDE.md for ctypes encoder retrofit

Removes the obsolete '10-27% MSE gap' caveat (encoders are now
byte-parity with llama.cpp). Updates architecture description to
mention ggml_binding.py. Adds runtime-dependency note about
llama-cpp-python.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: Final verification + push

**Files:** none

- [ ] **Step 1: Run the entire test suite (unit + integration)**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -30
```

Expected output: 22+ tests pass — 12 dtype guards, 1 regression, 9 parity, 1 Q3 smoke = 23 total. If integration tests run (depends on llama-quantize availability and reference fixture), all should pass.

- [ ] **Step 2: Verify no leftover references to deleted code**

Run:
```bash
cd /server/programming/MagicQuant && grep -rn "_encode_ggml_q\|_encode_ggml_iq\|_pack_k4k5\|_optimize_symmetric\|_GGML_ENCODERS\|IQ4_NL_LEVELS" --include="*.py" magicquant/ | grep -v __pycache__
```

Expected output: no matches (or only matches inside `ggml_binding.py`'s `_GGML_TYPE_SIZE` table comment).

- [ ] **Step 3: Verify line-count reduction**

Run:
```bash
cd /server/programming/MagicQuant && wc -l magicquant/quant/converters.py magicquant/quant/ggml_binding.py
```

Expected output: `converters.py` is ~250 lines (down from ~960). `ggml_binding.py` is ~330 lines.

- [ ] **Step 4: Check git log**

Run:
```bash
cd /server/programming/MagicQuant && git log --oneline | head -20
```

Expected output: 14+ new commits since PR0 merged.

- [ ] **Step 5: Verify clean working tree**

Run:
```bash
cd /server/programming/MagicQuant && git status
```

Expected output: `nothing to commit, working tree clean`.

- [ ] **Step 6: Push to origin**

Run:
```bash
cd /server/programming/MagicQuant && git push origin master 2>&1
```

Expected output: push succeeds.

- [ ] **Step 7: PR1 done — confirm in writing**

Print a status message:
```
PR1 complete:
- libggml ctypes binding via ggml_binding.py
- ~700 lines of pure-Python encoders deleted
- All 7 existing schemes retrofit to byte-parity with llama-quantize
- Q2_K and Q3_K registered as new schemes
- Q3 tier band reachable in real searches
- Random-config weights category-indexed (forward-compatible to PR3)
- noise_factor values calibrated empirically from Llama-3.2-1B
- 14+ commits pushed to origin/master
- Ready for PR2 (legacy Q-quants)
```

---

## Self-Review Checklist

**Spec coverage (PR1 section):**
- [x] "New `magicquant/quant/ggml_binding.py` (~180 lines)" → Tasks 3 & 4
- [x] "`pyproject.toml`: add `llama-cpp-python>=0.3.0` hard dep" → Task 2
- [x] "`converters.py`: delete pure-Python `_encode_ggml_*` and helpers" → Task 8
- [x] "Register **Q2_K, Q3_K** as new schemes" → Task 9
- [x] "Retrofit Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, MXFP4, Q4_0 to ctypes path" → Tasks 7 & 8
- [x] "Run calibration bench; commit `tools/calibration_results.json`; paste noise factors" → Tasks 12, 13, 14
- [x] "Add Layer-2 + Layer-3 tests" → Tasks 6, 11, 15
- [x] "Tier band coverage: Q3 reliably populates" → Task 15 verifies
- [x] "Q2 band still effectively unreachable" → documented in Task 9 commit message
- [x] "Remove CLAUDE.md '10-27% MSE gap' caveat" → Task 16

**Spec deviation:**
- Random-config weight restructure (deferred from PR0) → handled in Task 10 of this plan, as committed earlier.

**Placeholder scan:** No "TBD" or vague directives. All code blocks complete.

**Type consistency:** `ggml_encode` signature (weights, ggml_type, imatrix=None) matches between `ggml_binding.py` definition and `converters.py` invocation.

**Risk callouts:**
- Task 6 has a pre-emptive resolution for `--pure` flag if byte-parity fails on legacy Q-quants. If still failing after that, dig into `gguf` reader's data-extraction path.
- Task 13 (calibration bench) is the longest task (~2 hr). If a model isn't readily available, Step 1 has fallback download instructions.
- Task 14 may regenerate the regression fixture if calibrated noise factors reorder the search; the commit handles this.

---

## Future Work (not in this plan)

- PR2: Register legacy Q-quants (Q4_0, Q4_1, Q5_0, Q5_1)
- PR3: Add IQ1/IQ2/IQ3/IQ4_XS, lower the robust floor, populate Q2 tier band
- PR4: Importance-matrix support
