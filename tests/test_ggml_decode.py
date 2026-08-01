"""Round-trip tests for the libggml dequantization path (ggml_decode).

Mirrors the encode-side test style (test_quantization_guards.py,
test_rocmfpx_schemes.py): skip the whole module when libggml can't be
located, and additionally skip individual quantized-type cases when the
bound libggml doesn't export that type's dequantize_row_* symbol (e.g. a
stock build vs. the ROCmFPX fork).

Run with a ROCmFPX build (also exercises the fork's own dequant kernels):
    MAGICQUANT_LIBGGML_DIR=~/ROCmFPX/build-strix-rocmfp4/bin \
        .venv/bin/python -m pytest tests/test_ggml_decode.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from magicquant.quant.ggml_binding import (
    LibggmlNotFound,
    get_handle,
    ggml_decode,
    ggml_encode,
    supports_decode,
)


@pytest.fixture(autouse=True)
def _require_libggml():
    """Skip every test in this module if libggml can't be discovered.

    Same try/except-and-skip shape used by test_rocmfpx_schemes.py's
    live-encode check.
    """
    try:
        get_handle()
    except LibggmlNotFound as e:
        pytest.skip(f"libggml not available: {e}")


# Per-type relative-RMSE bound (rmse(x, decode(encode(x))) / std(x)).
_RMSE_BOUNDS = {
    "Q8_0": 0.01,
    "Q6_K": 0.02,
    "Q5_K": 0.04,
    "Q4_K": 0.12,
    "IQ4_NL": 0.12,
    "Q4_0": 0.15,
    "MXFP4": 0.20,
}


def _relative_rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)) / np.std(x))


def _test_tensor() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 256)).astype(np.float32)


# ── quantized round trip ─────────────────────────────────────────────────

@pytest.mark.parametrize("ggml_type", sorted(_RMSE_BOUNDS))
def test_quantized_roundtrip_relative_rmse(ggml_type):
    handle = get_handle()
    if not handle.supports_decode(ggml_type):
        pytest.skip(f"{ggml_type} decode not supported by loaded libggml")

    x = _test_tensor()
    n = x.size
    blob = ggml_encode(x, ggml_type)
    y = ggml_decode(blob, ggml_type, n)

    assert y.shape == (n,)
    assert y.dtype == np.float32

    rmse = _relative_rmse(x.reshape(-1), y)
    # Guard against decode silently returning the (bit-identical) input
    # buffer instead of actually dequantizing.
    assert rmse > 0.0, (
        f"{ggml_type}: decode error is exactly zero — suspicious, decode "
        f"may be echoing the input rather than dequantizing"
    )
    assert rmse < _RMSE_BOUNDS[ggml_type], (
        f"{ggml_type}: relative rmse {rmse:.4f} exceeds bound "
        f"{_RMSE_BOUNDS[ggml_type]}"
    )


def test_rmse_ordering_by_precision():
    """Higher-bitrate schemes must reconstruct more faithfully."""
    x = _test_tensor()
    n = x.size
    rmses = {}
    for ggml_type in ("Q4_K", "Q6_K", "Q8_0"):
        blob = ggml_encode(x, ggml_type)
        y = ggml_decode(blob, ggml_type, n)
        rmses[ggml_type] = _relative_rmse(x.reshape(-1), y)

    assert rmses["Q4_K"] > rmses["Q6_K"] > rmses["Q8_0"]


# ── float passthrough paths ──────────────────────────────────────────────

def test_f32_roundtrip_exact():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(256).astype(np.float32)
    y = ggml_decode(x.tobytes(), "F32", x.size)
    assert y.dtype == np.float32
    np.testing.assert_array_equal(x, y)


def test_f16_roundtrip_close():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(256).astype(np.float32)
    blob = ggml_encode(x, "F16")
    y = ggml_decode(blob, "F16", x.size)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, x, atol=1e-2)


def test_bf16_roundtrip_close():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(256).astype(np.float32)
    blob = ggml_encode(x, "BF16")
    y = ggml_decode(blob, "BF16", x.size)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, x, atol=1e-2)


# ── guards ────────────────────────────────────────────────────────────────

def test_wrong_buffer_size_raises_value_error():
    x = np.zeros(256, dtype=np.float32)
    blob = ggml_encode(x, "Q8_0")
    with pytest.raises(ValueError):
        ggml_decode(blob[:-1], "Q8_0", 256)


def test_f32_wrong_buffer_size_raises_value_error():
    with pytest.raises(ValueError):
        ggml_decode(b"\x00" * 10, "F32", 256)


def test_unknown_type_raises_value_error():
    with pytest.raises(ValueError):
        ggml_decode(b"\x00" * 256, "NOT_A_REAL_TYPE", 256)


def test_supports_decode_true_for_float_types():
    assert supports_decode("F32")
    assert supports_decode("F16")
    assert supports_decode("BF16")
