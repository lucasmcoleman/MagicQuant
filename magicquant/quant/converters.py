"""
Quantization Converters - Convert model weights to different quantization schemes.

This module is the single source of truth for all quantization operations in
MagicQuant.  It contains:

1. ``Quantizer`` — The original numpy-level quantizer (returns arrays + metadata
   dicts, useful for prediction / simulation / dequantization).

2. **ggml block encoders** — Pure-Python encoders that produce byte buffers in
   the exact binary block layouts that llama.cpp / ggml expects.  These are used
   by the GGUF writer to create files that llama.cpp can load directly.

3. ``encode_to_ggml_bytes()`` / ``ggml_tensor_data_size()`` — The public API
   that the GGUF writer calls.
"""

from typing import Dict, List, Tuple, Optional
import struct
import numpy as np


# ---------------------------------------------------------------------------
# ggml block format constants
# ---------------------------------------------------------------------------

# Elements per block for each quantised ggml type
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
    "MXFP4": 32,  # OCP MX FP4 (E2M1 + shared E8M0 exponent)
}

# Bytes consumed per block
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
    "MXFP4": 17,  # 1B E8M0 shared exponent + 16B packed E2M1 nibbles
}

# IQ4_NL non-linear quantisation levels (from ggml)
IQ4_NL_LEVELS = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10,
     1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


def ggml_tensor_data_size(ggml_type_name: str, n_elements: int) -> int:
    """Return the byte-size of tensor data for a given ggml type and element count."""
    block_size = GGML_BLOCK_SIZE.get(ggml_type_name, 1)
    type_size = GGML_TYPE_SIZE.get(ggml_type_name, 2)
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * type_size


# =====================================================================
# Existing Quantizer class (numpy-level, for prediction / simulation)
# =====================================================================

class Quantizer:
    """
    Converts floating-point weights to various quantization formats.

    Each quantization scheme has its own algorithm for compressing weights
    while minimizing precision loss.
    """

    def __init__(self):
        self.quantile_cache: Dict[str, float] = {}

    def quantize_weights(
        self,
        weights: np.ndarray,
        scheme_name: str,
        group_name: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Quantize a weight tensor to the specified scheme.

        Args:
            weights: Original float32/float16 weights (numpy array)
            scheme_name: Target quantization scheme (e.g., "Q8_0", "MXFP4_MOE")
            group_name: Optional name of tensor group for specialized handling

        Returns:
            Tuple of (quantized_weights, metadata_dict)
        """
        from magicquant.quant.schemes import get_scheme_by_name

        scheme = get_scheme_by_name(scheme_name)

        # Route to appropriate quantization method
        if scheme_name == "BF16":
            return self._quantize_bf16(weights)
        elif scheme_name == "Q8_0":
            return self._quantize_q8_0(weights)
        elif scheme_name == "Q6_K":
            return self._quantize_q6_k(weights)
        elif scheme_name == "Q5_K":
            return self._quantize_q5_k(weights)
        elif scheme_name == "Q4_K_M":
            return self._quantize_q4_k_m(weights)
        elif scheme_name == "IQ4_NL":
            return self._quantize_iq4_nl(weights)
        elif scheme_name == "MXFP4_MOE":
            return self._quantize_mxfp4_moe(weights, group_name)
        else:
            raise ValueError(f"Unknown quantization scheme: {scheme_name}")

    def _quantize_bf16(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        return weights.astype(np.float32), {"bits": 16}

    def _quantize_q8_0(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        original_shape = weights.shape
        w_flat = weights.astype(np.float32).flatten()
        abs_max = np.max(np.abs(w_flat))
        if abs_max < 1e-9:
            return np.zeros_like(weights, dtype=np.uint8), {
                "scale": 0.0, "bits": 8, "range": [-127, 127]
            }
        scale = abs_max / 127.5
        q = np.round(w_flat / scale).clip(-127, 127).astype(np.int8)
        return q.reshape(original_shape), {
            "scale": float(scale), "bits": 8,
            "range": [-127, 127], "original_max": float(abs_max)
        }

    def _quantize_q6_k(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        w_flat = weights.astype(np.float32).flatten()
        block_size = 32
        num_blocks = len(w_flat) // block_size
        quantized, metadata = [], {"block_size": block_size, "scales": [], "bits": 6}
        for i in range(num_blocks):
            block = w_flat[i * block_size:(i + 1) * block_size]
            abs_max = np.max(np.abs(block))
            if abs_max < 1e-9:
                quantized.extend([0] * block_size); metadata["scales"].append(0.0); continue
            scale = abs_max / 31.0; metadata["scales"].append(float(scale))
            quantized.extend(np.round(block / scale).clip(-31, 31).astype(np.int8).tolist())
        packed = self._pack_6bit(quantized, num_blocks)
        return packed, metadata

    def _quantize_q5_k(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        w_flat = weights.astype(np.float32).flatten()
        block_size = 32
        num_blocks = len(w_flat) // block_size
        quantized, metadata = [], {"block_size": block_size, "scales": [], "bits": 5}
        for i in range(num_blocks):
            block = w_flat[i * block_size:(i + 1) * block_size]
            abs_max = np.max(np.abs(block))
            if abs_max < 1e-9:
                quantized.extend([0] * block_size); metadata["scales"].append(0.0); continue
            scale = abs_max / 15.0; metadata["scales"].append(float(scale))
            quantized.extend(np.round(block / scale).clip(-15, 15).astype(np.int8).tolist())
        packed = self._pack_5bit(quantized, num_blocks)
        return packed, metadata

    def _quantize_q4_k_m(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        w_flat = weights.astype(np.float32).flatten()
        block_size = 32
        num_blocks = len(w_flat) // block_size
        quantized, metadata = [], {"block_size": block_size, "scales": [], "bits": 4}
        for i in range(num_blocks):
            block = w_flat[i * block_size:(i + 1) * block_size]
            abs_max = np.max(np.abs(block))
            if abs_max < 1e-9:
                quantized.extend([0] * block_size); metadata["scales"].append(0.0); continue
            scale = abs_max / 7.5; metadata["scales"].append(float(scale))
            quantized.extend(np.round(block / scale).clip(-8, 7).astype(np.int8).tolist())
        packed = self._pack_4bit(quantized, num_blocks)
        return packed, metadata

    def _quantize_iq4_nl(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        w_flat = weights.astype(np.float32).flatten()
        block_size = 32
        num_blocks = len(w_flat) // block_size
        quantized, metadata = [], {"block_size": block_size, "scales": [], "non_linear": True, "bits": 4}
        for i in range(num_blocks):
            block = w_flat[i * block_size:(i + 1) * block_size]
            abs_max = np.max(np.abs(block))
            if abs_max < 1e-9:
                quantized.extend([0] * block_size); metadata["scales"].append(0.0); continue
            scale = abs_max / 7.5; metadata["scales"].append(float(scale))
            quantized.extend(np.round(block / scale).clip(-8, 7).astype(np.int8).tolist())
        packed = self._pack_4bit(quantized, num_blocks)
        return packed, metadata

    def _quantize_mxfp4_moe(self, weights: np.ndarray, group_name: Optional[str]) -> Tuple[np.ndarray, Dict]:
        w_flat = weights.astype(np.float32).flatten()
        block_size = 16
        num_blocks = len(w_flat) // block_size
        quantized, metadata = [], {"block_size": block_size, "scales": [], "is_mxfp4": True, "bits": 4}
        for i in range(num_blocks):
            block = w_flat[i * block_size:(i + 1) * block_size]
            abs_max = np.max(np.abs(block))
            if abs_max < 1e-9:
                quantized.extend([0] * block_size); metadata["scales"].append(0.0); continue
            scale = abs_max / 7.0; metadata["scales"].append(float(scale))
            quantized.extend(np.round(block / scale).clip(-7, 6).astype(np.int8).tolist())
        packed = self._pack_4bit(quantized, num_blocks)
        return packed, metadata

    # -- bit packing helpers (for numpy-level quantizer) --

    def _pack_6bit(self, values: List[int], num_blocks: int) -> np.ndarray:
        result = []
        for i in range(num_blocks):
            block = values[i * 32: (i + 1) * 32]
            for j in range(0, len(block), 4):
                if j + 3 < len(block):
                    v1, v2, v3, v4 = block[j:j+4]
                    result.extend([
                        ((v1 & 0x3F) | ((v2 & 0x0F) << 6)) & 0xFF,
                        (((v2 & 0x30) >> 4) | ((v3 & 0x3F) << 2)) & 0xFF,
                        (((v3 & 0x03) >> 2) | ((v4 & 0x3F) << 4)) & 0xFF,
                    ])
        return np.array(result, dtype=np.uint8)

    def _pack_5bit(self, values: List[int], num_blocks: int) -> np.ndarray:
        result = []
        for i in range(num_blocks):
            block = values[i * 32: (i + 1) * 32]
            for j in range(0, len(block) - 1, 8):
                if j + 7 < len(block):
                    v = block[j:j+8]
                    result.extend([
                        (v[0] & 0x1F) | ((v[1] & 0x1F) << 5),
                        ((v[1] & 0xE0) >> 3) | ((v[2] & 0x1F) << 2) | ((v[3] & 0x03) << 7),
                        ((v[3] & 0x1C) >> 2) | ((v[4] & 0x1F) << 4),
                        ((v[4] & 0xE0) >> 5) | ((v[5] & 0x1F) << 1) | ((v[6] & 0x1F) << 6),
                        ((v[6] & 0xE0) >> 3) | ((v[7] & 0x1F) << 2),
                    ])
        return np.array(result, dtype=np.uint8)

    def _pack_4bit(self, values: List[int], num_blocks: int) -> np.ndarray:
        result = []
        for i in range(num_blocks):
            block = values[i * 32: min((i + 1) * 32, len(values))]
            for j in range(0, len(block), 2):
                if j + 1 < len(block):
                    result.append((block[j] & 0x0F) | ((block[j+1] & 0x0F) << 4))
                else:
                    result.append(block[j] & 0x0F)
        return np.array(result, dtype=np.uint8)

    # -- dequantization --

    def dequantize_weights(self, quantized: np.ndarray, metadata: Dict, scheme_name: str) -> np.ndarray:
        if scheme_name == "Q8_0":
            return self._dequantize_q8_0(quantized, metadata)
        elif scheme_name == "Q6_K":
            return self._dequantize_q6_k(quantized, metadata)
        elif scheme_name == "Q5_K":
            return self._dequantize_q5_k(quantized, metadata)
        elif scheme_name == "Q4_K_M":
            return self._dequantize_q4_k_m(quantized, metadata)
        else:
            raise ValueError(f"Dequantization not implemented for: {scheme_name}")

    def _dequantize_q8_0(self, quantized, metadata):
        scale = metadata.get("scale", 1.0)
        if scale == 0.0:
            return np.zeros_like(quantized, dtype=np.float32)
        return quantized.astype(np.int8).astype(np.float32) * scale

    def _dequantize_q6_k(self, quantized, metadata):
        block_size = metadata.get("block_size", 32)
        scales = metadata.get("scales", [])
        unpacked = self._unpack_6bit(quantized, len(scales) * block_size)
        result = []
        for i, scale in enumerate(scales):
            s, e = i * block_size, (i + 1) * block_size
            result.extend(unpacked[s:e] * scale if scale > 0 else [0.0] * block_size)
        return np.array(result[:len(quantized) // 6 * block_size], dtype=np.float32)

    def _dequantize_q5_k(self, quantized, metadata):
        block_size = metadata.get("block_size", 32)
        scales = metadata.get("scales", [])
        unpacked = self._unpack_5bit(quantized, len(scales) * block_size)
        result = []
        for i, scale in enumerate(scales):
            s, e = i * block_size, (i + 1) * block_size
            result.extend(unpacked[s:e] * scale if scale > 0 else [0.0] * block_size)
        return np.array(result[:len(quantized) // 5 * block_size], dtype=np.float32)

    def _dequantize_q4_k_m(self, quantized, metadata):
        block_size = metadata.get("block_size", 32)
        scales = metadata.get("scales", [])
        unpacked = self._unpack_4bit(quantized, len(scales) * block_size)
        result = []
        for i, scale in enumerate(scales):
            s, e = i * block_size, (i + 1) * block_size
            result.extend(unpacked[s:e] * scale if scale > 0 else [0.0] * block_size)
        return np.array(result[:len(quantized) // 4 * block_size], dtype=np.float32)

    def _unpack_6bit(self, data, target_len):
        result = []
        for b in data:
            result.append(b & 0x3F)
            result.append((b >> 6) & 0x3F)
        return result[:target_len]

    def _unpack_5bit(self, data, target_len):
        result = []
        for b in data:
            result.append(b & 0x1F)
            result.append((b >> 5) & 0x1F)
        return result[:target_len]

    def _unpack_4bit(self, data, target_len):
        result = []
        for b in data:
            result.append(b & 0x0F)
            result.append((b >> 4) & 0x0F)
        return result[:target_len]


# =====================================================================
# ggml block-format encoders
# =====================================================================
# These produce raw byte buffers matching the on-disk ggml block layout
# that llama.cpp reads.  Each function takes a flat float32 numpy array
# (already padded to the block size if needed) and returns ``bytes``.
# =====================================================================

def _pad_to(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Pad *arr* to a multiple of *block_size* with zeros."""
    rem = len(arr) % block_size
    if rem:
        arr = np.concatenate([arr, np.zeros(block_size - rem, dtype=arr.dtype)])
    return arr


# Scale multipliers for RMSE optimization.  We try each candidate and
# keep the one with the lowest mean-squared error per sub-block.
_SCALE_CANDIDATES = np.array(
    [0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12], dtype=np.float32
)


def _optimize_symmetric_scale(
    sub_blocks: np.ndarray,
    naive_scales: np.ndarray,
    max_q: int,
    min_q: int = None,
) -> np.ndarray:
    """
    Find the scale that minimizes MSE for symmetric quantization.

    Args:
        sub_blocks: (n_blocks, n_sub, sub_size) — the float32 values
        naive_scales: (n_blocks, n_sub) — initial scale estimates
        max_q: Maximum quantized value (e.g., 32 for Q6_K signed -32..31)
        min_q: Minimum quantized value (default: -max_q)

    Returns:
        Optimized scales: (n_blocks, n_sub)
    """
    if min_q is None:
        min_q = -max_q
    n_cand = len(_SCALE_CANDIDATES)
    # candidate_scales: (n_blocks, n_sub, n_cand)
    candidates = naive_scales[:, :, None] * _SCALE_CANDIDATES[None, None, :]
    # Avoid division by zero
    inv_cand = np.where(candidates > 0, 1.0 / candidates, 0.0)
    # Quantize with each candidate: (n_blocks, n_sub, sub_size, n_cand)
    q = np.round(
        sub_blocks[:, :, :, None] * inv_cand[:, :, None, :]
    ).clip(min_q, max_q)
    # Dequantize
    deq = q * candidates[:, :, None, :]
    # MSE per candidate: (n_blocks, n_sub, n_cand)
    mse = np.mean((sub_blocks[:, :, :, None] - deq) ** 2, axis=2)
    # Pick best candidate per sub-block
    best_idx = np.argmin(mse, axis=2)  # (n_blocks, n_sub)
    # Gather the best scale
    n_b, n_s = naive_scales.shape
    return candidates[
        np.arange(n_b)[:, None],
        np.arange(n_s)[None, :],
        best_idx,
    ]


def _optimize_asymmetric_scale(
    sub_blocks: np.ndarray,
    naive_scales: np.ndarray,
    offsets: np.ndarray,
    max_q: int,
) -> np.ndarray:
    """
    Find the scale that minimizes MSE for asymmetric quantization.

    The offset (min) is fixed by the format; only the scale is optimized.

    Args:
        sub_blocks: (n_blocks, n_sub, sub_size) — float32 values
        naive_scales: (n_blocks, n_sub) — initial scale estimates
        offsets: (n_blocks, n_sub) — min offsets (subtracted before quantizing)
        max_q: Maximum quantized value (e.g., 15 for Q4_K, 31 for Q5_K)

    Returns:
        Optimized scales: (n_blocks, n_sub)
    """
    n_cand = len(_SCALE_CANDIDATES)
    candidates = naive_scales[:, :, None] * _SCALE_CANDIDATES[None, None, :]
    inv_cand = np.where(candidates > 0, 1.0 / candidates, 0.0)
    # shifted = val + offset
    shifted = sub_blocks + offsets[:, :, None]  # (n_blocks, n_sub, sub_size)
    # Quantize: q = round(shifted / scale), clamped to [0, max_q]
    q = np.round(
        shifted[:, :, :, None] * inv_cand[:, :, None, :]
    ).clip(0, max_q)
    # Dequantize: val = q * scale - offset
    deq = q * candidates[:, :, None, :] - offsets[:, :, None, None]
    mse = np.mean((sub_blocks[:, :, :, None] - deq) ** 2, axis=2)
    best_idx = np.argmin(mse, axis=2)
    n_b, n_s = naive_scales.shape
    return candidates[
        np.arange(n_b)[:, None],
        np.arange(n_s)[None, :],
        best_idx,
    ]


# ── Q8_0: 34 bytes per 32-element block ─────────────────────────────

def _encode_ggml_q8_0(flat: np.ndarray) -> bytes:
    """Block layout: f16 scale (2 B) + 32 x int8 quants (32 B) = 34 B."""
    flat = _pad_to(flat, 32)
    n_blocks = len(flat) // 32
    blocks = flat.reshape(n_blocks, 32)
    amax = np.max(np.abs(blocks), axis=1)
    d = np.where(amax > 0, amax / 127.0, 0.0)
    d_f16 = d.astype(np.float16)
    inv_d = np.where(d_f16 != 0, 1.0 / d_f16.astype(np.float32), 0.0)
    quants = np.round(blocks * inv_d[:, None]).clip(-128, 127).astype(np.int8)

    # Vectorized packing: interleave f16 scale bytes with quant bytes
    # Each block: 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes
    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)  # (n_blocks, 2)
    q_bytes = quants.view(np.uint8)  # (n_blocks, 32)
    result = np.empty((n_blocks, 34), dtype=np.uint8)
    result[:, :2] = d_bytes
    result[:, 2:] = q_bytes
    return result.tobytes()


# ── Q4_0: 18 bytes per 32-element block ─────────────────────────────

def _encode_ggml_q4_0(flat: np.ndarray) -> bytes:
    """Block layout: f16 scale (2 B) + 16 B nibbles = 18 B."""
    flat = _pad_to(flat, 32)
    n_blocks = len(flat) // 32
    blocks = flat.reshape(n_blocks, 32)
    amax = np.max(np.abs(blocks), axis=1)
    d = np.where(amax > 0, amax / 8.0, 0.0)
    d_f16 = d.astype(np.float16)
    inv_d = np.where(d_f16 != 0, 1.0 / d_f16.astype(np.float32), 0.0)
    # Unsigned 4-bit with zero-point at 8
    q = np.round(blocks * inv_d[:, None] + 8.0).clip(0, 15).astype(np.uint8)

    # Vectorized nibble packing: pair even/odd elements
    # q shape: (n_blocks, 32) -> take even indices and odd indices
    packed = (q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)  # (n_blocks, 16)
    packed = packed.astype(np.uint8)

    # Interleave f16 scale bytes with packed nibble bytes
    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)
    result = np.empty((n_blocks, 18), dtype=np.uint8)
    result[:, :2] = d_bytes
    result[:, 2:] = packed
    return result.tobytes()


# ── IQ4_NL: 18 bytes per 32-element block (non-linear levels) ──────

# Pre-sorted IQ4_NL levels and their original indices for searchsorted
_IQ4_NL_SORTED_IDX = np.argsort(IQ4_NL_LEVELS)
_IQ4_NL_SORTED = IQ4_NL_LEVELS[_IQ4_NL_SORTED_IDX]
# Midpoints between consecutive sorted levels for searchsorted boundaries
_IQ4_NL_BOUNDARIES = 0.5 * (_IQ4_NL_SORTED[:-1] + _IQ4_NL_SORTED[1:])

def _encode_ggml_iq4_nl(flat: np.ndarray) -> bytes:
    """Same block layout as Q4_0, but uses the IQ4_NL lookup table."""
    flat = _pad_to(flat, 32)
    n_blocks = len(flat) // 32
    blocks = flat.reshape(n_blocks, 32)
    amax = np.max(np.abs(blocks), axis=1)
    d = np.where(amax > 0, amax / 127.0, 0.0)
    d_f16 = d.astype(np.float16)
    inv_d = np.where(d_f16 != 0, 1.0 / d_f16.astype(np.float32), 0.0)
    # Scale weights, then find closest IQ4_NL level index for each
    scaled = blocks * inv_d[:, None]  # (n_blocks, 32)

    # Use searchsorted on pre-computed boundaries to find nearest level
    # This avoids allocating a (n_blocks, 32, 16) temporary array
    sorted_idx = np.searchsorted(_IQ4_NL_BOUNDARIES, scaled.ravel())
    # sorted_idx is an index into the sorted levels; map back to original indices
    indices = _IQ4_NL_SORTED_IDX[sorted_idx].reshape(n_blocks, 32).astype(np.uint8)

    # Vectorized nibble packing
    packed = (indices[:, 0::2] & 0x0F) | ((indices[:, 1::2] & 0x0F) << 4)
    packed = packed.astype(np.uint8)

    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)
    result = np.empty((n_blocks, 18), dtype=np.uint8)
    result[:, :2] = d_bytes
    result[:, 2:] = packed
    return result.tobytes()


# ── Q6_K: 210 bytes per 256-element super-block ─────────────────────
#
# Block layout (from ggml struct block_q6_K):
#   uint8_t ql[128]    — low 4 bits of 6-bit quants (interleaved)
#   uint8_t qh[64]     — high 2 bits of 6-bit quants (interleaved)
#   int8_t  scales[16]  — per-sub-block scales
#   ggml_fp16_t d       — master scale
#   Total: 128 + 64 + 16 + 2 = 210

def _encode_ggml_q6_k(flat: np.ndarray) -> bytes:
    flat = _pad_to(flat, 256)
    n_blocks = len(flat) // 256
    blocks = flat.reshape(n_blocks, 256)

    # 16 sub-blocks of 16 elements — compute per-sub-block scales
    # Reshape to (n_blocks, 16, 16) for vectorized sub-block operations
    sub_blocks = blocks.reshape(n_blocks, 16, 16)
    sub_amax = np.max(np.abs(sub_blocks), axis=2)  # (n_blocks, 16)
    naive_scales = np.where(sub_amax > 0, sub_amax / 32.0, 0.0)  # (n_blocks, 16)
    # RMSE-optimized scale selection
    sub_scales = _optimize_symmetric_scale(sub_blocks, naive_scales, max_q=31)

    # Master scale per super-block
    max_ss = np.max(sub_scales, axis=1)  # (n_blocks,)
    d = np.where(max_ss > 0, max_ss / 127.0, 0.0)  # (n_blocks,)
    d_f16 = d.astype(np.float16)  # (n_blocks,)
    d_val = d_f16.astype(np.float32)  # (n_blocks,)

    # Quantise sub-block scales to int8
    inv_d = np.where(d_val > 0, 1.0 / d_val, 0.0)  # (n_blocks,)
    qscales = np.round(sub_scales * inv_d[:, None]).clip(-128, 127).astype(np.int8)  # (n_blocks, 16)

    # Quantise each element to 6-bit unsigned (0..63, representing -32..31)
    # Effective scale per sub-block element: d_val * qscales
    eff = d_val[:, None] * qscales.astype(np.float32)  # (n_blocks, 16)
    inv_eff = np.where(eff != 0, 1.0 / eff, 0.0)  # (n_blocks, 16)
    # Broadcast: sub_blocks (n_blocks, 16, 16) * inv_eff (n_blocks, 16, 1)
    L_signed = np.round(sub_blocks * inv_eff[:, :, None]).clip(-32, 31).astype(np.int8)
    # Where eff == 0, result should be 32 (the zero-point)
    L = (L_signed.astype(np.int16) + 32).astype(np.uint8)  # (n_blocks, 16, 16)
    L = L.reshape(n_blocks, 256)

    # Pack into ql (128 B) and qh (64 B) using ggml interleaved layout
    # Reshape L to (n_blocks, 2, 4, 32) where dim1=half (0..1), dim2=quarter (0..3)
    # Original layout: L[off + l], L[off + l + 32], L[off + l + 64], L[off + l + 96]
    # with off = half * 128, for l in 0..31
    Lr = L.reshape(n_blocks, 2, 4, 32)  # [block, half, quarter, position]

    # ql: 128 bytes = 2 halves * 64 bytes each
    # ql[ql_off + l]      = (L[off+l] & 0x0F) | ((L[off+l+32] & 0x0F) << 4)   => quarters 0,1
    # ql[ql_off + l + 32] = (L[off+l+64] & 0x0F) | ((L[off+l+96] & 0x0F) << 4) => quarters 2,3
    ql = np.empty((n_blocks, 2, 2, 32), dtype=np.uint8)
    ql[:, :, 0, :] = (Lr[:, :, 0, :] & 0x0F) | ((Lr[:, :, 1, :] & 0x0F) << 4)
    ql[:, :, 1, :] = (Lr[:, :, 2, :] & 0x0F) | ((Lr[:, :, 3, :] & 0x0F) << 4)
    ql = ql.reshape(n_blocks, 128)

    # qh: 64 bytes = 2 halves * 32 bytes each
    # qh[qh_off + l] = ((L[off+l]>>4)&3) | ((L[off+l+32]>>4)&3)<<2 |
    #                   ((L[off+l+64]>>4)&3)<<4 | ((L[off+l+96]>>4)&3)<<6
    qh = (
        ((Lr[:, :, 0, :] >> 4) & 0x03)       |
        (((Lr[:, :, 1, :] >> 4) & 0x03) << 2) |
        (((Lr[:, :, 2, :] >> 4) & 0x03) << 4) |
        (((Lr[:, :, 3, :] >> 4) & 0x03) << 6)
    ).astype(np.uint8)  # (n_blocks, 2, 32)
    qh = qh.reshape(n_blocks, 64)

    # Assemble output: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes per block
    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)
    scales_bytes = qscales.view(np.uint8).reshape(n_blocks, 16)
    result = np.empty((n_blocks, 210), dtype=np.uint8)
    result[:, :128] = ql
    result[:, 128:192] = qh
    result[:, 192:208] = scales_bytes
    result[:, 208:210] = d_bytes
    return result.tobytes()


# ── K-quant scale packing (shared by Q4_K and Q5_K) ─────────────────
#
# 8 sub-block scales (6-bit) and 8 sub-block mins (6-bit) are packed
# into 12 bytes.  The layout is defined by ggml's get_scale_min_k4().

def _pack_k4k5_scales(scales: np.ndarray, mins: np.ndarray) -> bytes:
    """Pack 8 x 6-bit scales and 8 x 6-bit mins into 12 bytes.

    Works for both single-block (1-D arrays of length 8) and batched
    inputs (2-D arrays of shape (n_blocks, 8)).  For batched inputs
    returns a uint8 array of shape (n_blocks, 12).
    """
    if scales.ndim == 1:
        # Single-block path (kept for backward compat)
        p = bytearray(12)
        for j in range(4):
            p[j]     = (int(scales[j]) & 0x3F) | (((int(scales[j + 4]) >> 4) & 0x03) << 6)
            p[j + 4] = (int(mins[j])   & 0x3F) | (((int(mins[j + 4])   >> 4) & 0x03) << 6)
            p[j + 8] = (int(scales[j + 4]) & 0x0F) | ((int(mins[j + 4]) & 0x0F) << 4)
        return bytes(p)

    # Batched vectorized path: scales/mins are (n_blocks, 8) uint8
    n = scales.shape[0]
    sc = scales.astype(np.uint8)
    mn = mins.astype(np.uint8)

    p = np.empty((n, 12), dtype=np.uint8)
    # p[:, 0:4] = (sc[:, 0:4] & 0x3F) | (((sc[:, 4:8] >> 4) & 0x03) << 6)
    p[:, 0:4] = (sc[:, 0:4] & 0x3F) | (((sc[:, 4:8] >> 4) & 0x03) << 6)
    # p[:, 4:8] = (mn[:, 0:4] & 0x3F) | (((mn[:, 4:8] >> 4) & 0x03) << 6)
    p[:, 4:8] = (mn[:, 0:4] & 0x3F) | (((mn[:, 4:8] >> 4) & 0x03) << 6)
    # p[:, 8:12] = (sc[:, 4:8] & 0x0F) | ((mn[:, 4:8] & 0x0F) << 4)
    p[:, 8:12] = (sc[:, 4:8] & 0x0F) | ((mn[:, 4:8] & 0x0F) << 4)
    return p


# ── Q4_K: 144 bytes per 256-element super-block ─────────────────────
#
# Block layout:
#   ggml_fp16_t d (2 B) + ggml_fp16_t dmin (2 B) + scales (12 B) +
#   qs (128 B) = 144

def _encode_ggml_q4_k(flat: np.ndarray) -> bytes:
    flat = _pad_to(flat, 256)
    n_blocks = len(flat) // 256
    blocks = flat.reshape(n_blocks, 256)

    # 8 sub-blocks of 32 elements — asymmetric quantisation (min/max)
    sub = blocks.reshape(n_blocks, 8, 32)  # (n_blocks, 8, 32)
    sub_min = np.min(sub, axis=2)   # (n_blocks, 8)
    sub_max = np.max(sub, axis=2)   # (n_blocks, 8)
    sub_mins = np.where(sub_min < 0, -sub_min, 0.0).astype(np.float32)  # (n_blocks, 8)
    rng = sub_max + sub_mins  # (n_blocks, 8)
    naive_scales = np.where(rng > 0, rng / 15.0, 0.0).astype(np.float32)
    # RMSE-optimized scale selection
    sub_scales = _optimize_asymmetric_scale(sub, naive_scales, sub_mins, max_q=15)

    # Master scales
    max_scale = np.max(sub_scales, axis=1)  # (n_blocks,)
    max_min = np.max(sub_mins, axis=1)      # (n_blocks,)
    d = np.where(max_scale > 0, max_scale / 63.0, 0.0)
    dmin = np.where(max_min > 0, max_min / 63.0, 0.0)
    d_f16 = d.astype(np.float16)
    dmin_f16 = dmin.astype(np.float16)
    d_val = d_f16.astype(np.float32)
    dmin_val = dmin_f16.astype(np.float32)

    # Quantise sub-block scales and mins to 6-bit (0..63)
    inv_d = np.where(d_val > 0, 1.0 / d_val, 0.0)
    inv_dmin = np.where(dmin_val > 0, 1.0 / dmin_val, 0.0)
    qs_sc = np.round(sub_scales * inv_d[:, None]).clip(0, 63).astype(np.uint8)  # (n_blocks, 8)
    qs_mn = np.round(sub_mins * inv_dmin[:, None]).clip(0, 63).astype(np.uint8)  # (n_blocks, 8)

    # Pack scales using batched helper
    scales_packed = _pack_k4k5_scales(qs_sc, qs_mn)  # (n_blocks, 12)

    # Quantise values: q = round((val + dmin*m) / (d*sc)), 0..15
    eff_d = d_val[:, None] * qs_sc.astype(np.float32)  # (n_blocks, 8)
    eff_m = dmin_val[:, None] * qs_mn.astype(np.float32)  # (n_blocks, 8)
    inv_eff_d = np.where(eff_d > 0, 1.0 / eff_d, 0.0)  # (n_blocks, 8)
    # sub is (n_blocks, 8, 32), eff_m is (n_blocks, 8) -> broadcast
    q = np.round((sub + eff_m[:, :, None]) * inv_eff_d[:, :, None]).clip(0, 15).astype(np.uint8)
    # (n_blocks, 8, 32)

    # Nibble packing: pair even/odd within each sub-block
    packed = (q[:, :, 0::2] & 0x0F) | ((q[:, :, 1::2] & 0x0F) << 4)  # (n_blocks, 8, 16)
    packed = packed.astype(np.uint8).reshape(n_blocks, 128)

    # Assemble output: d(2) + dmin(2) + scales(12) + qs(128) = 144
    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)
    dmin_bytes = dmin_f16.view(np.uint8).reshape(n_blocks, 2)
    result = np.empty((n_blocks, 144), dtype=np.uint8)
    result[:, 0:2] = d_bytes
    result[:, 2:4] = dmin_bytes
    result[:, 4:16] = scales_packed
    result[:, 16:144] = packed
    return result.tobytes()


# ── Q5_K: 176 bytes per 256-element super-block ─────────────────────
#
# Block layout:
#   d (2 B) + dmin (2 B) + scales (12 B) + qh (32 B) + qs (128 B) = 176
#
# 5-bit quants: low 4 bits in qs (nibble-packed), high bit in qh (bit-packed).

def _encode_ggml_q5_k(flat: np.ndarray) -> bytes:
    flat = _pad_to(flat, 256)
    n_blocks = len(flat) // 256
    blocks = flat.reshape(n_blocks, 256)

    # 8 sub-blocks of 32 elements — asymmetric quantisation (min/max)
    sub = blocks.reshape(n_blocks, 8, 32)  # (n_blocks, 8, 32)
    sub_min_vals = np.min(sub, axis=2)   # (n_blocks, 8)
    sub_max_vals = np.max(sub, axis=2)   # (n_blocks, 8)
    sub_mins = np.where(sub_min_vals < 0, -sub_min_vals, 0.0).astype(np.float32)
    rng = sub_max_vals + sub_mins
    naive_scales = np.where(rng > 0, rng / 31.0, 0.0).astype(np.float32)
    # RMSE-optimized scale selection
    sub_scales = _optimize_asymmetric_scale(sub, naive_scales, sub_mins, max_q=31)

    # Master scales
    max_scale = np.max(sub_scales, axis=1)  # (n_blocks,)
    max_min = np.max(sub_mins, axis=1)      # (n_blocks,)
    d = np.where(max_scale > 0, max_scale / 63.0, 0.0)
    dmin = np.where(max_min > 0, max_min / 63.0, 0.0)
    d_f16 = d.astype(np.float16)
    dmin_f16 = dmin.astype(np.float16)
    d_val = d_f16.astype(np.float32)
    dmin_val = dmin_f16.astype(np.float32)

    # Quantise sub-block scales and mins to 6-bit (0..63)
    inv_d = np.where(d_val > 0, 1.0 / d_val, 0.0)
    inv_dmin = np.where(dmin_val > 0, 1.0 / dmin_val, 0.0)
    qs_sc = np.round(sub_scales * inv_d[:, None]).clip(0, 63).astype(np.uint8)
    qs_mn = np.round(sub_mins * inv_dmin[:, None]).clip(0, 63).astype(np.uint8)

    # Pack scales using batched helper
    scales_packed = _pack_k4k5_scales(qs_sc, qs_mn)  # (n_blocks, 12)

    # Quantise values to 5-bit unsigned (0..31)
    eff_d = d_val[:, None] * qs_sc.astype(np.float32)  # (n_blocks, 8)
    eff_m = dmin_val[:, None] * qs_mn.astype(np.float32)  # (n_blocks, 8)
    inv_eff_d = np.where(eff_d > 0, 1.0 / eff_d, 0.0)
    L = np.round((sub + eff_m[:, :, None]) * inv_eff_d[:, :, None]).clip(0, 31).astype(np.uint8)
    # (n_blocks, 8, 32)
    L = L.reshape(n_blocks, 256)

    # Pack low 4 bits into qs (128 B, nibble-packed)
    qs = ((L[:, 0::2] & 0x0F) | ((L[:, 1::2] & 0x0F) << 4)).astype(np.uint8)  # (n_blocks, 128)

    # Pack high bit (bit 4) into qh (32 B, bit-packed)
    # L has values 0..31 so bit 4 is the high bit
    high_bits = ((L >> 4) & 0x01).astype(np.uint8)  # (n_blocks, 256) — 0 or 1
    # Pack 256 bits into 32 bytes using reshape + bitwise ops
    # Reshape to (n_blocks, 32, 8) — 32 output bytes, 8 bits each
    hb = high_bits.reshape(n_blocks, 32, 8)
    # Multiply by bit positions [1, 2, 4, 8, 16, 32, 64, 128] and sum
    bit_positions = (1 << np.arange(8, dtype=np.uint16))  # [1,2,4,8,16,32,64,128]
    qh = np.sum(hb.astype(np.uint16) * bit_positions[None, None, :], axis=2).astype(np.uint8)  # (n_blocks, 32)

    # Assemble output: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176
    d_bytes = d_f16.view(np.uint8).reshape(n_blocks, 2)
    dmin_bytes = dmin_f16.view(np.uint8).reshape(n_blocks, 2)
    result = np.empty((n_blocks, 176), dtype=np.uint8)
    result[:, 0:2] = d_bytes
    result[:, 2:4] = dmin_bytes
    result[:, 4:16] = scales_packed
    result[:, 16:48] = qh
    result[:, 48:176] = qs
    return result.tobytes()


# ── Float format encoders ────────────────────────────────────────────

# ── MXFP4: 17 bytes per 32-element block (OCP MX E2M1 + shared E8M0) ─
#
# Block layout:
#   uint8_t shared_exp  — E8M0 shared exponent: actual scale = 2^(exp − 127)
#   uint8_t qs[16]      — 32 × 4-bit E2M1 values packed as nibbles
#   Total: 1 + 16 = 17 bytes
#
# E2M1 FP4 representable unsigned values:
#   index  bits   value
#   0      000    0.0     (zero)
#   1      001    0.5     (subnormal: 0.1₂ × 2^0)
#   2      010    1.0     (1.0₂ × 2^0)
#   3      011    1.5     (1.1₂ × 2^0)
#   4      100    2.0     (1.0₂ × 2^1)
#   5      101    3.0     (1.1₂ × 2^1)
#   6      110    4.0     (1.0₂ × 2^2)
#   7      111    6.0     (1.1₂ × 2^2)
# With sign bit (bit 3): 16 codes total, ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}

_E2M1_UNSIGNED = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           dtype=np.float32)
# Midpoints between consecutive unsigned E2M1 levels, for nearest-level lookup
_E2M1_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                            dtype=np.float32)


def _encode_ggml_mxfp4(flat: np.ndarray) -> bytes:
    """MXFP4 (OCP MX FP4): E2M1 values with per-block E8M0 shared exponent."""
    flat = _pad_to(flat, 32)
    n_blocks = len(flat) // 32
    blocks = flat.reshape(n_blocks, 32)

    # Compute shared exponent per block.
    # The max representable magnitude with E2M1 is 6.0 × 2^(exp−127).
    # So exp = ceil(log2(max_abs / 6)) + 127, clamped to [0, 254].
    # (exp=255 is reserved for special values in some specs; we avoid it.)
    amax = np.max(np.abs(blocks), axis=1)  # (n_blocks,)
    # Avoid log2(0)
    safe_amax = np.where(amax > 0, amax, 1.0)
    raw_exp = np.ceil(np.log2(safe_amax / 6.0)) + 127.0
    shared_exp = np.where(amax > 0, raw_exp, 0.0).clip(0, 254).astype(np.uint8)

    # Effective scale per block: 2^(shared_exp − 127)
    scale = np.ldexp(np.ones(n_blocks, dtype=np.float64),
                     shared_exp.astype(np.int32) - 127).astype(np.float32)

    # Scale elements into the E2M1 range [-6, +6]
    inv_scale = np.where(scale > 0, 1.0 / scale, 0.0)
    scaled = blocks * inv_scale[:, None]  # (n_blocks, 32)

    # Quantize: find nearest unsigned E2M1 level, then apply sign
    signs = (scaled < 0).astype(np.uint8)              # (n_blocks, 32)
    abs_scaled = np.abs(scaled)
    # searchsorted on midpoints gives the index of the nearest E2M1 level
    unsigned_idx = np.searchsorted(_E2M1_MIDPOINTS, abs_scaled).astype(np.uint8)
    # unsigned_idx is 0..7, sign is 0 or 1
    codes = unsigned_idx | (signs << 3)  # 4-bit code: bit3=sign, bits0-2=level

    # Pack 32 nibbles into 16 bytes per block
    lo = codes[:, 0::2] & 0x0F
    hi = codes[:, 1::2] & 0x0F
    packed = (lo | (hi << 4)).astype(np.uint8)  # (n_blocks, 16)

    # Assemble blocks: 1B exponent + 16B nibbles = 17B
    result = np.empty((n_blocks, 17), dtype=np.uint8)
    result[:, 0] = shared_exp
    result[:, 1:] = packed
    return result.tobytes()


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


# ── Dispatch ─────────────────────────────────────────────────────────

_GGML_ENCODERS = {
    "Q8_0":   _encode_ggml_q8_0,
    "Q4_0":   _encode_ggml_q4_0,
    "IQ4_NL": _encode_ggml_iq4_nl,
    "Q6_K":   _encode_ggml_q6_k,
    "Q5_K":   _encode_ggml_q5_k,
    "Q4_K":   _encode_ggml_q4_k,
    "MXFP4":  _encode_ggml_mxfp4,
    "BF16":   _encode_f32_to_bf16,
    "F16":    _encode_f32_to_f16,
    "F32":    _encode_f32_to_f32,
}


def encode_to_ggml_bytes(weights: np.ndarray, ggml_type_name: str) -> bytes:
    """
    Quantize a float32 weight array into ggml block-format bytes.

    This is the public entry point that the GGUF writer calls.

    Args:
        weights: Float32 numpy array (any shape — will be flattened).
        ggml_type_name: Target ggml type (e.g. "Q8_0", "Q4_K", "BF16").

    Returns:
        Raw bytes in the on-disk ggml block layout.
    """
    encoder = _GGML_ENCODERS.get(ggml_type_name)
    if encoder is None:
        raise ValueError(
            f"No ggml encoder for type '{ggml_type_name}'. "
            f"Available: {sorted(_GGML_ENCODERS)}"
        )
    flat = weights.astype(np.float32).flatten()
    return encoder(flat)
