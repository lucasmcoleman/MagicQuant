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


# ── Per-expert MoE imatrix (stacked ``_exps`` tensors) ──────────────────────
#
# llama-imatrix tracks importance separately per expert for a stacked
# ``_exps`` tensor (confirmed empirically against a real gpt-oss-20b capture:
# in_sum2 comes back shaped [n_experts, n_per_row], not [n_per_row], and
# several experts were left at count=0 with a small corpus since routing
# sends each token to only a few experts). magicquant.imatrix.load_imatrix
# flattens that into one [n_experts * n_per_row] vector, expert-major; encode()
# must detect the multi-slice length and quantize each expert with its own
# slice, mirroring llama.cpp's own per-i03 loop in llama-quant.cpp.

N_EXPERTS = 3


def _expert_weights():
    rng = np.random.default_rng(7)
    return rng.standard_normal((N_EXPERTS, ROWS, COLS)).astype(np.float32)


def _per_expert_imatrix():
    # Deliberately distinct per expert so a shared/blended vector would
    # produce different bytes than the correct per-expert quantization.
    imat = np.zeros((N_EXPERTS, COLS), dtype=np.float32)
    imat[0] = 0.01
    imat[1] = 100.0
    imat[1][:32] = 0.01
    imat[2] = 50.0
    return imat.reshape(-1)


def test_per_expert_imatrix_differs_from_plain():
    w = _expert_weights()
    plain = encode_to_ggml_bytes(w, "Q4_K")
    weighted = encode_to_ggml_bytes(w, "Q4_K", imatrix=_per_expert_imatrix(), n_per_row=COLS)
    assert len(plain) == len(weighted)
    assert plain != weighted


def test_per_expert_imatrix_matches_quantizing_each_expert_alone():
    """Ground truth: quantizing the whole stacked tensor with a per-expert
    imatrix must byte-for-byte equal quantizing each expert's slice alone
    with its own imatrix and concatenating -- proving experts don't leak
    into each other's quantization."""
    w = _expert_weights()
    imat = _per_expert_imatrix().reshape(N_EXPERTS, COLS)

    combined = encode_to_ggml_bytes(w, "Q4_K", imatrix=imat.reshape(-1), n_per_row=COLS)

    per_expert_alone = b"".join(
        encode_to_ggml_bytes(w[e], "Q4_K", imatrix=imat[e], n_per_row=COLS)
        for e in range(N_EXPERTS)
    )
    assert combined == per_expert_alone


def test_per_expert_imatrix_order_matters():
    # Swapping which slice goes to which expert must change the output --
    # otherwise experts are silently sharing/averaging importance data.
    w = _expert_weights()
    imat = _per_expert_imatrix().reshape(N_EXPERTS, COLS)
    normal = encode_to_ggml_bytes(w, "Q4_K", imatrix=imat.reshape(-1), n_per_row=COLS)
    swapped = encode_to_ggml_bytes(
        w, "Q4_K", imatrix=imat[[1, 0, 2]].reshape(-1), n_per_row=COLS
    )
    assert normal != swapped


def test_per_expert_imatrix_row_count_must_divide_evenly():
    w = _expert_weights()
    # 5 slices, but the tensor only has N_EXPERTS*ROWS = 12 rows -- 12 % 5 != 0.
    bad_imat = np.ones(COLS * 5, dtype=np.float32)
    with pytest.raises(ValueError, match="don't divide evenly"):
        encode_to_ggml_bytes(w, "Q4_K", imatrix=bad_imat, n_per_row=COLS)


def test_shared_dense_imatrix_still_works_on_multi_expert_shape():
    # A plain [n_per_row] vector (n_slices == 1) must still broadcast across
    # every row of a multi-expert tensor, same as any other dense tensor.
    w = _expert_weights()
    shared = _imatrix()
    out = encode_to_ggml_bytes(w, "Q4_K", imatrix=shared, n_per_row=COLS)
    expected = encode_to_ggml_bytes(w.reshape(-1, COLS), "Q4_K", imatrix=shared, n_per_row=COLS)
    assert out == expected
