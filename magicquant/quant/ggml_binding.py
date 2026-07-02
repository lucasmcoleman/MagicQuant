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

# ROCmFPX fork types (https://github.com/ciru-ai/ROCmFPX — a llama.cpp fork).
# IDs from the fork's ggml/include/ggml.h; NOT present in stock ggml, where
# these enum values are past GGML_TYPE_COUNT. The binding therefore never
# passes these IDs to a loaded libggml unless the fork-support probe
# (``_probe_rocmfpx``) has confirmed the lib knows them — an out-of-range
# enum would index past ggml's type_traits array.
ROCMFPX_TYPE_IDS = {
    "Q4_0_ROCMFP4":      100,
    "Q4_0_ROCMFP4_FAST": 101,
    "Q6_0_ROCMFPX":      102,
    "Q8_0_ROCMFPX":      103,
    "Q3_0_ROCMFPX":      104,
}
ROCMFPX_TYPE_NAMES = frozenset(ROCMFPX_TYPE_IDS)
GGML_TYPE_IDS.update(ROCMFPX_TYPE_IDS)

# Map our type-name keys to the lowercase strings the fork registers in its
# ggml type_traits table (used by the ggml_type_from_name support probe).
_ROCMFPX_REGISTERED_NAME = {
    "Q4_0_ROCMFP4":      "q4_0_rocmfp4",
    "Q4_0_ROCMFP4_FAST": "q4_0_rocmfp4_fast",
    "Q6_0_ROCMFPX":      "q6_0_rocmfpx",
    "Q8_0_ROCMFPX":      "q8_0_rocmfpx",
    "Q3_0_ROCMFPX":      "q3_0_rocmfpx",
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
    # ROCmFPX fork types — all 32-element blocks (ggml/rocmfp4/rocmfp4.h,
    # ggml/rocmfpx/rocmfpx.h in the fork).
    "Q4_0_ROCMFP4": 32, "Q4_0_ROCMFP4_FAST": 32,
    "Q3_0_ROCMFPX": 32, "Q6_0_ROCMFPX": 32, "Q8_0_ROCMFPX": 32,
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
    # ROCmFPX fork block byte sizes, verified against the fork's block structs:
    # rocmfp4 = 16 qs + 2 scale; fast = 16 + 1; fp3 = 12 + 2; fp6 = 24 + 2;
    # fp8 = 32 + 1. Cross-checked at runtime by _verify_type_ids when the
    # loaded libggml supports them.
    "Q4_0_ROCMFP4": 18, "Q4_0_ROCMFP4_FAST": 17,
    "Q3_0_ROCMFPX": 14, "Q6_0_ROCMFPX": 26, "Q8_0_ROCMFPX": 33,
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
        self.rocmfpx_supported: frozenset = self._probe_rocmfpx()
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

        # enum ggml_type ggml_type_from_name(const char * name);
        # Present on the ROCmFPX fork; may be absent on older stock builds.
        self._has_type_from_name = hasattr(self._base, "ggml_type_from_name")
        if self._has_type_from_name:
            self._base.ggml_type_from_name.argtypes = [ctypes.c_char_p]
            self._base.ggml_type_from_name.restype = ctypes.c_int

    def _probe_rocmfpx(self) -> frozenset:
        """Return the set of ROCmFPX type NAMES the loaded libggml supports.

        Probes by NAME via ggml_type_from_name (which returns GGML_TYPE_COUNT
        for unknown names) — never by passing a possibly-out-of-range type ID
        to a size/name lookup, which would index past a stock lib's
        type_traits array. Empty set on a stock (non-fork) libggml.
        """
        if not self._has_type_from_name:
            return frozenset()
        supported = set()
        for name, reg in _ROCMFPX_REGISTERED_NAME.items():
            got = self._base.ggml_type_from_name(reg.encode("ascii"))
            if got == ROCMFPX_TYPE_IDS[name]:
                supported.add(name)
        return frozenset(supported)

    def _verify_type_ids(self) -> None:
        """Sanity check: each (name, id) pair agrees with what libggml reports.

        Catches the case where a future ggml release renumbers types. If
        any mismatch is found, raise immediately with a clear actionable error.
        Fork-only ROCmFPX types are verified only when the loaded lib supports
        them (else skipped — a stock lib legitimately doesn't know them, and
        the ID is out of range so we must not call ggml_type_size on it).
        """
        for name, type_id in GGML_TYPE_IDS.items():
            expected = _GGML_TYPE_SIZE.get(name)
            if expected is None:
                continue  # not all types have known sizes (extension types)
            if name in ROCMFPX_TYPE_NAMES and name not in self.rocmfpx_supported:
                continue  # fork type, loaded lib doesn't have it — don't probe its ID
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

    def supports(self, ggml_type: str) -> bool:
        """True if the loaded libggml can encode this type.

        Non-fork types are assumed supported (stock ggml has them all);
        ROCmFPX fork types require the fork's libggml (probed at load).
        """
        if ggml_type in ROCMFPX_TYPE_NAMES:
            return ggml_type in self.rocmfpx_supported
        return ggml_type in GGML_TYPE_IDS

    def encode(
        self,
        weights: np.ndarray,
        ggml_type: str,
        imatrix: Optional[np.ndarray] = None,
        n_per_row: Optional[int] = None,
    ) -> bytes:
        """Quantize a float tensor to ggml block-format bytes.

        Args:
            weights: floating-point numpy array (any shape; flattened internally).
            ggml_type: ggml type name (e.g., "Q4_K", "Q2_K", "MXFP4").
            imatrix: optional float32 importance vector — one entry per input
                column of the tensor (length must equal ``n_per_row``).
            n_per_row: the tensor's true row width. REQUIRED with ``imatrix``
                (importance is applied per column, so row structure matters);
                ignored otherwise — the unweighted path quantizes the flat
                buffer as a single row, which is byte-identical because ggml
                blocks never span row boundaries.

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
        if ggml_type in ROCMFPX_TYPE_NAMES and ggml_type not in self.rocmfpx_supported:
            raise ValueError(
                f"'{ggml_type}' is a ROCmFPX fork type, but the loaded libggml "
                f"({self._base_path}) does not support it. Point "
                f"MAGICQUANT_LIBGGML_DIR at a ROCmFPX build "
                f"(e.g. ~/ROCmFPX/build-strix-rocmfp4/bin)."
            )

        flat = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)

        if imatrix is not None:
            if n_per_row is None:
                raise ValueError(
                    "imatrix-weighted encoding requires n_per_row (the "
                    "tensor's row width); importance is applied per column."
                )
            if n_per_row <= 0 or flat.size % n_per_row != 0:
                raise ValueError(
                    f"tensor size {flat.size} is not a multiple of "
                    f"n_per_row={n_per_row}"
                )
            imat_check = np.asarray(imatrix).reshape(-1)
            if imat_check.size != n_per_row:
                raise ValueError(
                    f"imatrix length {imat_check.size} != row width "
                    f"{n_per_row}. Each weight tensor needs its own "
                    f"importance vector with one entry per input column. "
                    f"(Per-expert MoE imatrix slices are not supported yet — "
                    f"drop the imatrix for this tensor.)"
                )
            nrows = flat.size // n_per_row
        else:
            # Historical fast path: one row spanning the whole buffer.
            n_per_row = flat.size
            nrows = 1

        type_id = GGML_TYPE_IDS[ggml_type]
        out_size = _expected_size(ggml_type, flat.size)

        # Bound the output allocation so a corrupt/absurd element count can't
        # request a multi-terabyte ctypes buffer (raise cleanly instead of
        # crashing the interpreter). 16 GiB is generous for any single tensor.
        _MAX_OUT_BYTES = 16 * 1024 ** 3
        if out_size <= 0 or out_size > _MAX_OUT_BYTES:
            raise ValueError(
                f"Refusing to encode: computed output size {out_size} bytes "
                f"(type={ggml_type}, n_elements={n_per_row}) is out of bounds "
                f"(0, {_MAX_OUT_BYTES}]. Tensor element count is likely corrupt."
            )

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
            ctypes.c_int64(nrows),
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
        if ggml_type in ROCMFPX_TYPE_NAMES and ggml_type not in self.rocmfpx_supported:
            return False
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
    n_per_row: Optional[int] = None,
) -> bytes:
    """Quantize via libggml's ggml_quantize_chunk.

    Public entry point used by magicquant.quant.converters.
    """
    return get_handle().encode(weights, ggml_type, imatrix, n_per_row=n_per_row)
