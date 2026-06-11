"""libggml quantize+dequant reference — USED ONLY BY TESTS.

The fake-quant kernels in ``fake_quant.py`` are GPU-fast torch approximations of
each ggml scheme. Their correctness contract is "stay within a tolerance of what
the real libggml quantizer produces on the same weights". This module provides
that ground truth: it runs the weights through the exact same encoder the GGUF
writer ships with (``ggml_encode`` -> ``ggml_quantize_chunk``), then dequantizes
back to f32 using libggml's own ``dequantize_row_*`` kernels.

It is intentionally NOT imported by any production code path — the trainer never
touches libggml; only the test suite uses this to validate the torch kernels.
"""

from __future__ import annotations

import ctypes

import numpy as np

from magicquant.quant.ggml_binding import get_handle, ggml_encode


# ggml exposes per-type dequant kernels with the C signature
#     void dequantize_row_<type>(const block_t * x, float * y, int64_t k)
# These are exported (T) symbols in libggml-base.so. Map the ggml type name to
# the exported symbol. Float passthroughs are handled separately below.
_DEQUANT_SYMBOL = {
    "Q4_0": "dequantize_row_q4_0",
    "Q4_1": "dequantize_row_q4_1",
    "Q5_0": "dequantize_row_q5_0",
    "Q5_1": "dequantize_row_q5_1",
    "Q8_0": "dequantize_row_q8_0",
    "Q2_K": "dequantize_row_q2_K",
    "Q3_K": "dequantize_row_q3_K",
    "Q4_K": "dequantize_row_q4_K",
    "Q5_K": "dequantize_row_q5_K",
    "Q6_K": "dequantize_row_q6_K",
    "IQ4_NL": "dequantize_row_iq4_nl",
    "IQ4_XS": "dequantize_row_iq4_xs",
    "MXFP4": "dequantize_row_mxfp4",
}

# Cache bound ctypes function objects keyed by ggml type name.
_BOUND: dict = {}


def _get_dequant_fn(ggml_type: str):
    """Return a ctypes-bound dequantize_row_<type> for the given ggml type."""
    if ggml_type in _BOUND:
        return _BOUND[ggml_type]
    sym = _DEQUANT_SYMBOL.get(ggml_type)
    if sym is None:
        raise ValueError(f"No libggml dequant symbol known for ggml type {ggml_type!r}")
    base = get_handle()._base
    fn = getattr(base, sym)
    # void dequantize_row_X(const void * x, float * y, int64_t k)
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    fn.restype = None
    _BOUND[ggml_type] = fn
    return fn


def _bf16_roundtrip(w: np.ndarray) -> np.ndarray:
    """f32 -> bf16 (round-to-nearest-even) -> f32, matching converters.py."""
    f32 = np.ascontiguousarray(w, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((u32 >> 16) & 1)
    bf16_bits = ((u32 + rounding) >> 16).astype(np.uint16)
    # widen back: bf16 is the high 16 bits of the f32
    up = (bf16_bits.astype(np.uint32) << 16)
    return up.view(np.float32).reshape(w.shape)


def _f16_roundtrip(w: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(w, dtype=np.float32).astype(np.float16).astype(np.float32).reshape(w.shape)


def ggml_quant_dequant(w_np: np.ndarray, ggml_type: str) -> np.ndarray:
    """Quantize ``w_np`` with libggml, then dequantize back to f32.

    Returns an array with the same shape as the input. This is the "ship
    reference" the fake-quant kernels are validated against.

    Quantization is row-wise in ggml (the encoder's ``n_per_row`` is the full
    flattened length here), and the K-quant / MXFP4 / Q8_0 block structure tiles
    along that flat axis, so a flat encode+dequant reproduces exactly what the
    writer would store for a 2-D weight (block boundaries are identical).
    """
    w = np.ascontiguousarray(w_np, dtype=np.float32)
    shape = w.shape
    n = w.size

    if ggml_type == "F32":
        return w.copy()
    if ggml_type == "BF16":
        return _bf16_roundtrip(w)
    if ggml_type == "F16":
        return _f16_roundtrip(w)

    # Quantized: encode via libggml, then dequant via libggml's own kernel.
    qbytes = ggml_encode(w.reshape(-1), ggml_type)
    src = (ctypes.c_uint8 * len(qbytes)).from_buffer_copy(qbytes)
    out = (ctypes.c_float * n)()
    fn = _get_dequant_fn(ggml_type)
    fn(ctypes.cast(src, ctypes.c_void_p), out, ctypes.c_int64(n))
    deq = np.frombuffer(out, dtype=np.float32, count=n).copy()
    return deq.reshape(shape)
