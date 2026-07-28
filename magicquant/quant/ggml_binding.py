"""
ctypes binding to libggml — encoder dispatch for all quantized schemes.

Discovery order (first match wins):
    1. MAGICQUANT_LIBGGML_DIR env var (explicit override)
    2. Common system llama.cpp build paths
    3. llama-cpp-python's bundled libs (always available since it's a hard dep)

Public API:
    ggml_encode(weights, ggml_type, imatrix=None) -> bytes
    ggml_decode(data, ggml_type, n_elements) -> np.ndarray (float32)
    supports_decode(ggml_type) -> bool
    GGML_TYPE_IDS  (mapping name -> numeric ggml type enum, synced from ggml.h)
    LibggmlNotFound  (exception)
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from magicquant.quant import ggml_facts

logger = logging.getLogger(__name__)

# ggml_type enum values. Derived from magicquant.quant.ggml_facts, which
# builds the stock entries from the installed `gguf` package (llama.cpp's
# own pure-python package) and overlays the ROCmFPX fork-only registry — see
# that module's docstring for why (single source of truth; the previous
# hand-copied table here duplicated converters.py's and could drift from it,
# which is exactly what happened with IQ4_XS at one point).
# GGML_TYPE_IDS name kept for backward compatibility (external references).
# A startup sanity check (_verify_type_ids) catches drift if the loaded
# libggml renumbers types.
GGML_TYPE_IDS = dict(ggml_facts.NAME_TO_ID)

# ROCmFPX fork types (https://github.com/ciru-ai/ROCmFPX — a llama.cpp fork).
# IDs from the fork's ggml/include/ggml.h; NOT present in stock ggml, where
# these enum values are past GGML_TYPE_COUNT. The binding therefore never
# passes these IDs to a loaded libggml unless the fork-support probe
# (``_probe_rocmfpx``) has confirmed the lib knows them — an out-of-range
# enum would index past ggml's type_traits array.
#
# Aliases of ggml_facts.FORK_TYPES / .ROCMFPX_TYPE_NAMES — kept as names
# here since this module (and its tests) reference them directly; the
# fork facts themselves live in exactly one place (ggml_facts.FORK_TYPES).
ROCMFPX_TYPE_IDS = {name: info["id"] for name, info in ggml_facts.FORK_TYPES.items()}
ROCMFPX_TYPE_NAMES = ggml_facts.ROCMFPX_TYPE_NAMES

# Map our type-name keys to the lowercase strings the fork registers in its
# ggml type_traits table (used by the ggml_type_from_name support probe).
_ROCMFPX_REGISTERED_NAME = {
    name: info["registered_name"] for name, info in ggml_facts.FORK_TYPES.items()
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


# ── Block format size tables ────────────────────────────────────────
# Used to compute output buffer size for ctypes calls. Derived from
# magicquant.quant.ggml_facts (stock from the `gguf` package, fork overlay
# from ggml_facts.FORK_TYPES) — see that module's docstring. Names kept
# here as aliases since this module's tests reference them directly.
_GGML_BLOCK_SIZE = dict(ggml_facts.BLOCK_SIZE)
_GGML_TYPE_SIZE = dict(ggml_facts.TYPE_SIZE)


def _expected_size(ggml_type: str, n_elements: int) -> int:
    """Return expected output byte count for a quantized tensor."""
    block = _GGML_BLOCK_SIZE.get(ggml_type, 1)
    type_sz = _GGML_TYPE_SIZE.get(ggml_type, 2)
    n_blocks = (n_elements + block - 1) // block
    return n_blocks * type_sz


# ── Dequantize row symbol table (synced with ggml/src/ggml-quants.h /
# ggml-cpu, plus the ROCmFPX fork's rocmfp4.h / rocmfpx.h) ─────────────────
#
# All symbols share the C signature `void f(const void *x, float *y,
# int64_t k)`, where k is the ELEMENT count (not block count) — blocks are
# contiguous so a single call handles a whole block-aligned tensor. Symbol
# names verified present via `nm -D` on
# /home/lucas/ROCmFPX/build-strix-rocmfp4/bin/libggml-base.so; stock builds
# have the non-ROCmFPX entries.
_DEQUANT_SYMBOLS = {
    "Q8_0": "dequantize_row_q8_0", "Q6_K": "dequantize_row_q6_K",
    "Q5_K": "dequantize_row_q5_K", "Q4_K": "dequantize_row_q4_K",
    "Q3_K": "dequantize_row_q3_K", "Q2_K": "dequantize_row_q2_K",
    "IQ4_NL": "dequantize_row_iq4_nl", "IQ4_XS": "dequantize_row_iq4_xs",
    "MXFP4": "dequantize_row_mxfp4",
    "Q4_0": "dequantize_row_q4_0", "Q4_1": "dequantize_row_q4_1",
    "Q5_0": "dequantize_row_q5_0", "Q5_1": "dequantize_row_q5_1",
    "Q4_0_ROCMFP4": "rocmfp4_dequantize_row_q4_0",
    "Q4_0_ROCMFP4_FAST": "rocmfp4_dequantize_row_q4_0_fast",
    "Q6_0_ROCMFPX": "rocmfpx_dequantize_row_fp6",
    "Q8_0_ROCMFPX": "rocmfpx_dequantize_row_fp8",
    "Q3_0_ROCMFPX": "rocmfpx_dequantize_row_fp3",
}


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
        self._cross_check_block_type_sizes()
        self._dequant_fn_cache: dict = {}
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

        Scoped to LOAD-BEARING types only: the ones magicquant.quant.schemes
        actually dispatches to ggml_quantize_chunk as a real quantization
        target, plus F32/F16/BF16 (writer.py uses these for real offset math
        via forced-F32/BF16->F16 paths even though they aren't scheme encode
        targets -- see the load_bearing union below), plus supported ROCmFPX
        fork types.

        HISTORICAL NOTE on Q8_1 (id=9), why it was excluded from this scope
        and no longer needs to be: this scoping was originally forced by a
        confirmed, narrow discrepancy -- the installed `gguf` pip package
        (0.18.0) reports Q8_1 as 40 bytes/block (a stale
        "float d; float s; qs[32]" formula), while every real libggml build
        checked (four independent builds, cross-probed 2026-07-28) reports
        36 bytes/block (the current "ggml_fp16_t d; ggml_fp16_t s; qs[32]"
        struct). Q8_1 is an ephemeral CPU dot-product intermediate type,
        never written into an on-disk GGUF and not offered by any
        MagicQuant scheme, so at the time this exclusion was written, one
        stale gguf-py constant for a type nothing here ever quantizes into
        was kept from blocking every other type's construction-time safety
        net by excluding Q8_1 from load_bearing rather than overriding the
        upstream fact.

        ggml_facts.py has SINCE added a documented override
        (``TYPE_SIZE["Q8_1"] = 36``) correcting this at the source, so the
        value this method would see for Q8_1 is now the real-libggml-
        verified 36, not gguf-py's stale 40 -- the original reason for
        excluding it no longer applies, but Q8_1 remains excluded here
        anyway because it's still not a real quantization target for any
        scheme (the scoping's actual, ongoing rationale). The discrepancy
        this exclusion once had to route around is no longer swallowed
        upstream of this method either way: it's still surfaced, non-fatally,
        by _cross_check_block_type_sizes below (which scans every type in
        GGML_TYPE_IDS, scoped or not) -- though post-override that check now
        finds agreement (36 == 36) rather than a mismatch. Anything actually
        dispatched to ggml_quantize_chunk with a wrong TYPE_SIZE would
        additionally be caught the moment it's used for real, by encode()'s
        own actual-bytes-written-vs-expected-size check.

        See tests/test_ggml_facts_snapshot.py's
        test_q8_1_known_upstream_staleness, which pins both the corrected
        exported value (36) and the still-stale raw gguf-py constant (40),
        and fails the moment upstream fixes the latter -- the signal to
        revisit this note and the override together.
        """
        from magicquant.quant.schemes import get_all_schemes

        load_bearing = (
            {s.ggml_type_name for s in get_all_schemes()}
            | ROCMFPX_TYPE_NAMES
            # F32/F16/BF16: not scheme encode targets, but writer.py relies
            # on correct sizes for these in its own offset math (forced-F32
            # SSM/group-S fallback, BF16->F16 casts) -- see docstring above.
            | {"F32", "F16", "BF16"}
        )
        for name, type_id in GGML_TYPE_IDS.items():
            if name not in load_bearing:
                continue
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

    def _cross_check_block_type_sizes(self) -> None:
        """One-time, LOG-ONLY cross-check of BLOCK_SIZE/TYPE_SIZE against
        what the loaded libggml itself reports via ggml_blck_size() /
        ggml_type_size().

        This is deliberately softer than ``_verify_type_ids`` above: that
        check already RAISES on a type_size mismatch (it guards the live
        encode/decode call path against silently corrupting output), so it
        stays a hard gate. This helper additionally covers BLOCK_SIZE
        (never cross-checked against the lib before ggml_facts existed) and
        logs loudly on any mismatch rather than crashing — so importing
        ggml_facts' block-size table for a type this particular build
        happens to disagree with (e.g. a genuinely optional/reserved type)
        surfaces as a visible warning a human can investigate, not a
        process-ending exception on every caller.

        Stock types are always checked; ROCmFPX fork types are checked only
        when ``rocmfpx_supported`` confirms the loaded lib actually knows
        them (an unsupported fork ID is out of range for a stock lib's
        type_traits array and must never be probed).
        """
        for name, type_id in GGML_TYPE_IDS.items():
            if name in ROCMFPX_TYPE_NAMES and name not in self.rocmfpx_supported:
                continue  # fork type the loaded lib doesn't have — id out of range
            expected_block = _GGML_BLOCK_SIZE.get(name)
            expected_size = _GGML_TYPE_SIZE.get(name)
            if expected_block is None and expected_size is None:
                continue
            try:
                actual_block = int(self._base.ggml_blck_size(type_id))
                actual_size = int(self._base.ggml_type_size(type_id))
            except Exception as exc:  # defensive: a symbol/ABI surprise must not crash import
                logger.warning(
                    "ggml_facts cross-check: could not query libggml for "
                    "'%s' (id=%d): %s", name, type_id, exc,
                )
                continue
            if expected_block is not None and actual_block != expected_block:
                logger.warning(
                    "ggml_facts cross-check: '%s' (id=%d) block_size "
                    "mismatch -- ggml_facts says %d, loaded libggml (%s) "
                    "says %d. Investigate before trusting offset math for "
                    "this type.",
                    name, type_id, expected_block, self._base_path, actual_block,
                )
            if expected_size is not None and actual_size != expected_size:
                logger.warning(
                    "ggml_facts cross-check: '%s' (id=%d) type_size "
                    "mismatch -- ggml_facts says %d, loaded libggml (%s) "
                    "says %d. Investigate before trusting offset math for "
                    "this type.",
                    name, type_id, expected_size, self._base_path, actual_size,
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
            imatrix: optional float32 importance vector. Either a plain
                ``[n_per_row]`` vector shared across every row (the normal
                dense-tensor case), or a per-expert ``[n_experts * n_per_row]``
                vector, expert-major (one ``n_per_row``-length slice per
                expert) — the layout ``magicquant.imatrix.load_imatrix``
                produces for a stacked MoE ``_exps`` tensor, since llama-imatrix
                tracks activation importance separately per expert (routing
                sends each token to only a few experts, so a shared vector
                would silently average unrelated experts' statistics
                together). Length must be a multiple of ``n_per_row``.
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

        n_slices = 1
        imat_check = None
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
            nrows = flat.size // n_per_row
            imat_check = np.ascontiguousarray(imatrix, dtype=np.float32).reshape(-1)
            if imat_check.size == 0 or imat_check.size % n_per_row != 0:
                raise ValueError(
                    f"imatrix length {imat_check.size} is not a multiple of "
                    f"row width {n_per_row}. Each weight tensor needs one "
                    f"([n_per_row]) or, for stacked MoE experts, several "
                    f"([n_experts * n_per_row], expert-major) importance "
                    f"vectors."
                )
            n_slices = imat_check.size // n_per_row
            if n_slices > 1 and nrows % n_slices != 0:
                raise ValueError(
                    f"imatrix has {n_slices} per-expert slices but the "
                    f"tensor's {nrows} rows don't divide evenly by that — "
                    f"shape mismatch between the captured imatrix and this "
                    f"weight tensor."
                )
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

        if n_slices <= 1:
            imat_ptr = None
            if imat_check is not None:
                imat_ptr = imat_check.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

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

        # Per-expert MoE tensor: quantize each expert's slice separately with
        # its own imatrix slice, mirroring llama.cpp's own handling (see
        # llama-quant.cpp's per-i03 loop, which offsets src/dst/imatrix by
        # i03*nelements_matrix / i03*row_size*nrows / i03*n_per_row — the
        # same per-expert-imatrix scheme, confirmed against a real gpt-oss-20b
        # imatrix capture during development).
        rows_per_slice = nrows // n_slices
        elems_per_slice = rows_per_slice * n_per_row
        slice_bytes = _expected_size(ggml_type, elems_per_slice)
        dst_base = ctypes.addressof(dst_buf)
        total_written = 0
        for slice_idx in range(n_slices):
            src_slice = flat[slice_idx * elems_per_slice: (slice_idx + 1) * elems_per_slice]
            imat_slice = imat_check[slice_idx * n_per_row: (slice_idx + 1) * n_per_row]
            dst_ptr = ctypes.c_void_p(dst_base + slice_idx * slice_bytes)

            actual = self._base.ggml_quantize_chunk(
                type_id,
                src_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dst_ptr,
                ctypes.c_int64(0),
                ctypes.c_int64(rows_per_slice),
                ctypes.c_int64(n_per_row),
                imat_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if actual != slice_bytes:
                raise RuntimeError(
                    f"ggml_quantize_chunk wrote {actual} bytes for expert "
                    f"slice {slice_idx}/{n_slices}, expected {slice_bytes} "
                    f"(type={ggml_type}). Likely cause: wrong type_id or "
                    f"block-size mismatch."
                )
            total_written += actual

        if total_written != out_size:
            raise RuntimeError(
                f"per-expert encode wrote {total_written} bytes total, "
                f"expected {out_size} (type={ggml_type}, n_slices={n_slices})."
            )
        return bytes(dst_buf)

    def requires_imatrix(self, ggml_type: str) -> bool:
        """Does this scheme need an importance matrix for best quality?"""
        if ggml_type in ROCMFPX_TYPE_NAMES and ggml_type not in self.rocmfpx_supported:
            return False
        if ggml_type not in GGML_TYPE_IDS:
            return False
        return self._base.ggml_quantize_requires_imatrix(GGML_TYPE_IDS[ggml_type])

    def _dequant_fn(self, ggml_type: str):
        """Lazily dlsym and cache the `dequantize_row_*` symbol for a type.

        Tries libggml-base first, then libggml-cpu (dequantize kernels can
        live in either depending on the build). Returns None if the symbol
        is absent from both — callers use this as the "can decode" probe.
        """
        if ggml_type in self._dequant_fn_cache:
            return self._dequant_fn_cache[ggml_type]

        fn = None
        name = _DEQUANT_SYMBOLS.get(ggml_type)
        if name is not None:
            for lib in (self._base, self._cpu):
                try:
                    fn = getattr(lib, name)
                    break
                except AttributeError:
                    continue
            if fn is not None:
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int64,
                ]
                fn.restype = None

        self._dequant_fn_cache[ggml_type] = fn
        return fn

    def supports_decode(self, ggml_type: str) -> bool:
        """True if the loaded libggml can dequantize this type.

        F32/F16/BF16 are native numpy conversions, so always supported.
        ROCmFPX fork types additionally require the fork's libggml (probed
        at load, same gate as ``supports``/``encode``).
        """
        if ggml_type in ("F32", "F16", "BF16"):
            return True
        if ggml_type in ROCMFPX_TYPE_NAMES:
            return ggml_type in self.rocmfpx_supported and self._dequant_fn(ggml_type) is not None
        return self._dequant_fn(ggml_type) is not None

    def decode(self, data: bytes, ggml_type: str, n_elements: int) -> np.ndarray:
        """Dequantize raw ggml block-format bytes back to a float32 array.

        Args:
            data: raw bytes in the on-disk ggml block layout (as produced
                by ``encode``/``ggml_quantize_chunk``).
            ggml_type: ggml type name (e.g., "Q4_K", "Q2_K", "MXFP4").
            n_elements: number of scalar elements the tensor holds (NOT
                block count).

        Returns:
            float32 numpy array of shape (n_elements,).

        Unlike ``encode``, decoding is stateless per block — there is no
        imatrix or per-expert slicing to thread through, since dequantizing
        a block never depends on anything outside that block.
        """
        if ggml_type == "F32":
            expected = 4 * n_elements
            if len(data) != expected:
                raise ValueError(
                    f"F32 decode: buffer is {len(data)} bytes, expected "
                    f"{expected} (n_elements={n_elements})"
                )
            return np.frombuffer(data, dtype=np.float32)

        if ggml_type == "F16":
            expected = 2 * n_elements
            if len(data) != expected:
                raise ValueError(
                    f"F16 decode: buffer is {len(data)} bytes, expected "
                    f"{expected} (n_elements={n_elements})"
                )
            return np.frombuffer(data, dtype=np.float16).astype(np.float32)

        if ggml_type == "BF16":
            expected = 2 * n_elements
            if len(data) != expected:
                raise ValueError(
                    f"BF16 decode: buffer is {len(data)} bytes, expected "
                    f"{expected} (n_elements={n_elements})"
                )
            u16 = np.frombuffer(data, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            return u32.view(np.float32)

        if ggml_type not in GGML_TYPE_IDS:
            raise ValueError(
                f"Unknown ggml type: {ggml_type}. Available: {sorted(GGML_TYPE_IDS)}"
            )

        if not self.supports_decode(ggml_type):
            raise RuntimeError(
                f"'{ggml_type}' cannot be dequantized by the loaded libggml "
                f"({self._base_path}): no dequantize_row_* symbol found in "
                f"libggml-base or libggml-cpu (or, for a ROCmFPX fork type, "
                f"the loaded lib does not support it). Point "
                f"MAGICQUANT_LIBGGML_DIR at a build that provides it "
                f"(e.g. ~/ROCmFPX/build-strix-rocmfp4/bin for fork types)."
            )

        expected = _expected_size(ggml_type, n_elements)
        if len(data) != expected:
            raise ValueError(
                f"{ggml_type} decode: buffer is {len(data)} bytes, expected "
                f"{expected} bytes for n_elements={n_elements}"
            )

        fn = self._dequant_fn(ggml_type)
        src = np.frombuffer(data, dtype=np.uint8)
        out = np.empty(n_elements, dtype=np.float32)
        fn(
            src.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(n_elements),
        )
        return out


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


def ggml_decode(data: bytes, ggml_type: str, n_elements: int) -> np.ndarray:
    """Dequantize via libggml's dequantize_row_* kernels.

    Public entry point mirroring ggml_encode; returns a float32 ndarray of
    shape (n_elements,).
    """
    return get_handle().decode(data, ggml_type, n_elements)


def supports_decode(ggml_type: str) -> bool:
    """True if the loaded libggml can dequantize this type."""
    return get_handle().supports_decode(ggml_type)
