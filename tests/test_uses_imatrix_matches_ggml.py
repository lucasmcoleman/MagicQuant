"""``uses_imatrix`` must match what the encoder actually does.

The registry carries three distinct imatrix relationships (see CLAUDE.md):
``requires_imatrix`` (cannot encode without one), ``uses_imatrix`` (consumes
one if offered), and ``IMATRIX_DEPENDENT_SCHEME_NAMES`` (encodes fine but is
only competitive with one). Only the first had a live cross-check --
``ggml_binding._cross_check_requires_imatrix`` validates it against
``ggml_quantize_requires_imatrix``. ``uses_imatrix`` had none, and it drifted:
Q4_0 and Q4_1 sat at ``False`` for months while ggml was running them through
the weighted ``make_qx_quants`` / ``make_qkx3_quants`` path all along, and the
only test on them pinned the wrong value.

Asserting it against a constant is what let that happen, so this asserts it
against behaviour instead: encode the same weights with and without an
imatrix and check the bytes differ exactly when ``uses_imatrix`` says they
should. Since the encoder is byte-identical to llama.cpp (it calls
``ggml_quantize_chunk`` directly), this is a direct read of ggml's own
answer.

SAFETY: a ``requires_imatrix`` type encoded without one does not raise a
Python exception -- ``ggml.c`` hits ``GGML_ASSERT(imatrix != NULL)`` and
aborts the whole process (SIGABRT), taking pytest with it. Those schemes are
skipped here, not caught.
"""

import numpy as np
import pytest

from magicquant.quant.converters import encode_to_ggml_bytes
from magicquant.quant.ggml_binding import get_handle
from magicquant.quant.schemes import get_all_schemes

# 256-divisible so every K-quant encodes natively rather than tripping a
# block-size fallback into some other type.
_N_PER_ROW = 512
_N_ROWS = 4


def _weights() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((_N_ROWS, _N_PER_ROW)).astype(np.float32)


def _imatrix() -> np.ndarray:
    # Strictly positive and clearly non-uniform: a flat importance vector
    # would leave some weighted encoders bit-identical to the unweighted
    # ones and make this test vacuously pass.
    rng = np.random.default_rng(7)
    return (np.abs(rng.standard_normal(_N_PER_ROW)) + 0.1).astype(np.float32)


def _encodable_schemes():
    handle = get_handle()
    out = []
    for scheme in sorted(get_all_schemes(), key=lambda s: s.name):
        if scheme.requires_imatrix:
            continue  # cannot be encoded without one; see SAFETY above
        try:
            if not handle.supports(scheme.ggml_type_name):
                continue  # e.g. ROCmFPX types on a stock libggml
        except Exception:
            continue
        out.append(scheme)
    return out


_SCHEMES = _encodable_schemes()


def test_there_is_something_to_check():
    """Guard the guard: if support probing silently returned nothing, every
    parametrized case below would vanish and the suite would still be green."""
    names = {s.name for s in _SCHEMES}
    assert {"Q4_0", "Q4_1", "Q4_K_M", "Q6_K", "Q8_0", "MXFP4_MOE"} <= names, names
    assert any(s.uses_imatrix for s in _SCHEMES)
    assert any(not s.uses_imatrix for s in _SCHEMES)


@pytest.mark.parametrize("scheme", _SCHEMES, ids=lambda s: s.name)
def test_uses_imatrix_matches_encoder_behaviour(scheme):
    weights = _weights()
    plain = encode_to_ggml_bytes(
        weights, scheme.ggml_type_name, imatrix=None, n_per_row=_N_PER_ROW
    )
    weighted = encode_to_ggml_bytes(
        weights, scheme.ggml_type_name, imatrix=_imatrix(), n_per_row=_N_PER_ROW
    )

    assert len(plain) == len(weighted), (
        f"{scheme.name}: an imatrix must not change the encoded size"
    )

    differs = plain != weighted
    if scheme.uses_imatrix:
        assert differs, (
            f"{scheme.name} is registered uses_imatrix=True but ggml produced "
            "byte-identical output with and without an imatrix -- the registry "
            "is over-crediting it. Check the quantize_* dispatch in "
            "ggml-quants.c: a function can accept quant_weights, branch on it, "
            "and still call the unweighted _ref encoder in both arms (Q2_0 and "
            "Q1_0 do exactly that)."
        )
    else:
        assert not differs, (
            f"{scheme.name} is registered uses_imatrix=False but ggml produced "
            "different bytes with an imatrix -- the registry is under-crediting "
            "it, the way Q4_0/Q4_1 were until 2026-08. Flip it to True."
        )


def test_legacy_q4_specifically_consumes_the_imatrix():
    """Pinned separately from the parametrized sweep: this is the exact bug
    the sweep was written for, and it should fail loudly by name if the
    registry regresses, not just as one anonymous parametrized case."""
    for name in ("Q4_0", "Q4_1"):
        weights = _weights()
        plain = encode_to_ggml_bytes(weights, name, imatrix=None, n_per_row=_N_PER_ROW)
        weighted = encode_to_ggml_bytes(
            weights, name, imatrix=_imatrix(), n_per_row=_N_PER_ROW
        )
        assert plain != weighted, f"{name} ignored the imatrix"
