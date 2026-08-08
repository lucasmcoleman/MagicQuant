"""
Quantization Converters - Convert model weights to ggml block-format bytes.

This module is the public encoder entry point used by the GGUF writer.
Quantized formats route through magicquant.quant.ggml_binding (libggml
ctypes binding); float passthroughs (BF16/F16/F32) stay native.

Public API:
    encode_to_ggml_bytes(weights, ggml_type_name, imatrix=None, n_per_row=None) -> bytes
    ggml_tensor_data_size(ggml_type_name, n_elements) -> int
"""

from typing import Optional
import numpy as np

from magicquant.quant.ggml_binding import ggml_encode, GGML_TYPE_IDS
from magicquant.quant import ggml_facts


# ---------------------------------------------------------------------------
# ggml block format constants (used by callers for offset/size math).
#
# Single source of truth: magicquant.quant.ggml_facts, which derives stock
# entries from the installed `gguf` package rather than a hand-copied table
# (the previous duplicate hand-maintained tables had IQ4_XS wrong: block=32/
# size=18 — those are IQ4_NL's values — while the binding correctly had
# block=256/size=136, causing corrupt Pass-1 offsets the moment PR3
# registered IQ4_XS). The integer/float passthrough types (F64/I8/I16/I32/
# I64) that used to need a manual overlay here are now part of ggml_facts'
# stock table too, since the `gguf` package publishes them directly.
# ---------------------------------------------------------------------------

GGML_BLOCK_SIZE = dict(ggml_facts.BLOCK_SIZE)
GGML_TYPE_SIZE = dict(ggml_facts.TYPE_SIZE)


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
    n_per_row: Optional[int] = None,
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
        imatrix: Optional per-column importance matrix. Consumed by the
            K-quants and the IQ family; ignored by MXFP4/ROCmFPX/float/
            legacy Q8_0 (ggml design — absmax/E8M0 scaling has no
            importance input).
        n_per_row: The tensor's row width. Conditionally required: only
            when imatrix is given AND weights is 1-D (weights.ndim < 2)
            — otherwise it's inferred from weights.shape[-1]. Ignored
            when imatrix is None.

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
    flat = weights.astype(np.float32).ravel()

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

    # Imatrix weighting is per input column, so ggml_quantize_chunk must see
    # the tensor's true row width (the unweighted path flattens to one row,
    # which is byte-identical because blocks never span row boundaries).
    # Sources hand the writer FLAT buffers, so the writer passes n_per_row
    # explicitly from its Pass-1 shape metadata; direct callers with shaped
    # arrays get it inferred from the trailing dimension.
    if imatrix is not None and n_per_row is None:
        if weights.ndim < 2:
            raise ValueError(
                f"imatrix-weighted encoding needs the tensor's row width: "
                f"pass n_per_row= or a 2-D+ weights array (got shape "
                f"{weights.shape})."
            )
        n_per_row = int(weights.shape[-1])
    return ggml_encode(flat, ggml_type_name, imatrix=imatrix, n_per_row=n_per_row)
