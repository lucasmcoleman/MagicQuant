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
