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
    "IQ2_S": 256, "IQ4_XS": 256, "IQ1_M": 256,
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
    "IQ2_S": 82, "IQ4_XS": 136, "IQ1_M": 56,
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
