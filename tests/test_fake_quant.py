"""Differentiable fake-quant kernels validated against the real libggml round-trip.

The correctness contract: each scheme's torch fake-quant dequantized output is
close to what shipping the same weights through libggml (ggml_encode -> dequant)
actually produces. The fake-quant is a GPU-fast faithful approximation, NOT a
byte-exact reimplementation — STE makes gradient exactness moot, so we assert a
relative-error tolerance per scheme rather than equality.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from magicquant.qat.fake_quant import fake_quant, FakeQuantSTE, SCHEME_FAKE_QUANT
from magicquant.qat._ggml_ref import ggml_quant_dequant


# Per-scheme tolerance for the libggml round-trip match (mean relative error).
# Floats are tight; 8-bit loose; 4-bit (Q4_K) and MXFP4 wider. (Q6_K/Q5_K in
# Task 3.)
_TOLERANCE = {
    "BF16": 0.02,
    "F16": 0.02,
    "F32": 0.001,
    "Q8_0": 0.05,
    "MXFP4": 0.08,
    "Q4_K": 0.08,
}


def _ggml_roundtrip(w_np, ggml_type):
    """Quantize with libggml then dequantize back to f32 — the ship reference."""
    return ggml_quant_dequant(w_np.astype(np.float32), ggml_type)


@pytest.mark.parametrize(
    "ggml_type", ["BF16", "F16", "F32", "Q8_0", "MXFP4", "Q4_K"]
)
def test_fake_quant_matches_libggml(ggml_type):
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    fq = fake_quant(w, ggml_type).detach().cpu().numpy()
    ref = _ggml_roundtrip(w.cpu().numpy(), ggml_type)
    # close to the real ggml round-trip (faithful, not byte-exact)
    rel = np.abs(fq - ref).mean() / (np.abs(ref).mean() + 1e-8)
    tol = _TOLERANCE[ggml_type]
    assert rel < tol, f"{ggml_type} fake-quant deviates {rel:.4f} from libggml (tol={tol})"


@pytest.mark.parametrize("ggml_type", ["Q8_0", "MXFP4", "Q4_K"])
def test_fake_quant_idempotent(ggml_type):
    torch.manual_seed(1)
    w = torch.randn(128, 128)
    once = fake_quant(w, ggml_type)
    twice = fake_quant(once, ggml_type)
    assert torch.allclose(once, twice, atol=1e-4), f"{ggml_type} not idempotent"


def test_ste_gradient_passes_through():
    w = torch.randn(64, 64, requires_grad=True)
    out = fake_quant(w, "Q8_0")
    out.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert torch.allclose(w.grad, torch.ones_like(w.grad), atol=1e-5)  # STE = identity


def test_bf16_passthrough_is_near_identity():
    w = torch.randn(32, 32)
    assert torch.allclose(fake_quant(w, "BF16"), w.bfloat16().float(), atol=1e-2)


def test_unmapped_scheme_falls_back_to_bf16():
    """A scheme with no kernel falls back to BF16 passthrough with a warning."""
    w = torch.randn(32, 32)
    with pytest.warns(UserWarning):
        out = fake_quant(w, "IQ2_XXS")  # not in v1 registry
    assert torch.allclose(out, w.bfloat16().float(), atol=1e-2)


def test_registry_has_task2_schemes():
    for name in ["BF16", "F16", "F32", "Q8_0", "MXFP4", "Q4_K"]:
        assert name in SCHEME_FAKE_QUANT, f"{name} missing from SCHEME_FAKE_QUANT"


def test_fakequantste_backward_is_identity_for_arbitrary_grad():
    """STE backward returns the upstream gradient unchanged (and None for fn)."""
    w = torch.randn(16, 16, requires_grad=True)
    out = fake_quant(w, "Q4_K")
    g = torch.randn(16, 16)
    out.backward(g)
    assert torch.allclose(w.grad, g, atol=1e-6)


def test_mxfp4_values_land_on_e2m1_grid():
    """Every MXFP4 dequant value is a grid entry scaled by a power of two."""
    torch.manual_seed(2)
    w = torch.randn(4, 32)  # one block per row
    out = fake_quant(w, "MXFP4").cpu().numpy()
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for row in out:
        nz = row[np.abs(row) > 0]
        for v in nz:
            mag = abs(v)
            # value / 2**k must equal a grid magnitude for some integer k
            ratios = mag / grid[1:]  # exclude zero
            log2r = np.log2(ratios)
            assert np.any(np.abs(log2r - np.round(log2r)) < 1e-5), (
                f"MXFP4 value {v} not on E2M1 grid * 2^k"
            )
