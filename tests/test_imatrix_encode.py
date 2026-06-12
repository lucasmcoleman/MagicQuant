"""Imatrix-weighted encoding through the libggml binding.

The imatrix is a per-input-column weighting, so ``ggml_quantize_chunk`` must be
called with the tensor's TRUE row width — not the flatten-to-one-row shortcut the
unweighted path uses (harmless there: blocks never span row boundaries, so one
row of N*W elements quantizes byte-identically to N rows of W).
"""
import numpy as np
import pytest

from magicquant.quant.converters import encode_to_ggml_bytes

ROWS, COLS = 4, 256  # one K-quant superblock per row


def _weights():
    rng = np.random.default_rng(42)
    return rng.standard_normal((ROWS, COLS)).astype(np.float32)


def _imatrix():
    # Strongly uneven importance so weighted quantization must choose
    # different scales than unweighted.
    imat = np.full(COLS, 0.01, dtype=np.float32)
    imat[:32] = 100.0
    return imat


def test_weighted_output_differs_from_unweighted():
    w = _weights()
    plain = encode_to_ggml_bytes(w, "Q4_K")
    weighted = encode_to_ggml_bytes(w, "Q4_K", imatrix=_imatrix())
    assert len(plain) == len(weighted)
    assert plain != weighted, (
        "imatrix did not change the encoding — it is not being threaded "
        "through to ggml_quantize_chunk"
    )


def test_weighted_output_is_deterministic():
    w = _weights()
    a = encode_to_ggml_bytes(w, "Q4_K", imatrix=_imatrix())
    b = encode_to_ggml_bytes(w, "Q4_K", imatrix=_imatrix())
    assert a == b


def test_wrong_imatrix_length_raises():
    w = _weights()
    with pytest.raises(ValueError, match="imatrix"):
        encode_to_ggml_bytes(w, "Q4_K", imatrix=np.ones(COLS // 2, dtype=np.float32))


def test_imatrix_with_1d_weights_raises():
    # A 1-D tensor has no row structure to weight against; require 2-D+.
    w = _weights().reshape(-1)
    with pytest.raises(ValueError, match="row"):
        encode_to_ggml_bytes(w, "Q4_K", imatrix=_imatrix())


def test_unweighted_path_unchanged_by_shape():
    """Regression: without imatrix, 2-D and flattened 1-D encode identically
    (the historical flatten-to-one-row behavior must not change)."""
    w = _weights()
    assert encode_to_ggml_bytes(w, "Q4_K") == encode_to_ggml_bytes(w.reshape(-1), "Q4_K")


def test_float_passthrough_ignores_imatrix():
    # F32/F16/BF16 are not quantized; imatrix must be silently irrelevant.
    w = _weights()
    assert encode_to_ggml_bytes(w, "F32", imatrix=_imatrix()) == \
        encode_to_ggml_bytes(w, "F32")
