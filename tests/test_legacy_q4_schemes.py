"""Q4_0/Q4_1 legacy (v2-only) scheme support.

Mirrors tests/test_iq_schemes.py and tests/test_rocmfpx_schemes.py's
structure for an opt-in family: registered in the scheme registry (so v2's
per-tensor scheme-override allocation in writer.py's "tensors" key can
reference them by name), but excluded unconditionally from v1's
random-config sampling pool (survival.py's _generate_random_config) so the
default evolutionary search -- and its seed-pinned regression fixture --
stays byte-identical.
"""
import random

import numpy as np
import pytest

from magicquant.quant.schemes import (
    get_scheme_by_name, get_all_schemes, LEGACY_Q4_SCHEME_NAMES,
)
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor


# ── registry ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,ggml_type_name,bpw", [
    ("Q4_0", "Q4_0", 4.5),
    ("Q4_1", "Q4_1", 5.0),
])
def test_legacy_q4_scheme_metadata(name, ggml_type_name, bpw):
    s = get_scheme_by_name(name)
    assert s.ggml_type_name == ggml_type_name
    assert s.bits_per_weight == bpw
    assert s.category == "legacy_q"
    # "legacy" is about the block layout, not about imatrix support. Both of
    # these run ggml's weighted make_qx_quants / make_qkx3_quants path when an
    # imatrix is supplied -- see the note above the Q4_0 entry in schemes.py,
    # and tests/test_uses_imatrix_matches_ggml.py, which asserts this against
    # the live library instead of against a constant.
    assert s.uses_imatrix is True
    assert s.requires_imatrix is False


def test_legacy_q4_scheme_names_constant_matches_registry_entries():
    assert set(LEGACY_Q4_SCHEME_NAMES) == {"Q4_0", "Q4_1"}
    reg_names = {s.name for s in get_all_schemes()}
    assert LEGACY_Q4_SCHEME_NAMES <= reg_names


def test_legacy_q4_upgrade_neighbor_is_none():
    assert get_scheme_by_name("Q4_0").upgrade_neighbor is None
    assert get_scheme_by_name("Q4_1").upgrade_neighbor is None


def test_no_existing_scheme_points_at_legacy_q4():
    """No pre-existing scheme's upgrade_neighbor/downgrade_neighbor may
    reference Q4_0/Q4_1 -- the v1 mutation neighbor-walk (Protector/Crusher)
    must never be able to land on them."""
    for s in get_all_schemes():
        if s.name in LEGACY_Q4_SCHEME_NAMES:
            continue
        assert s.upgrade_neighbor not in LEGACY_Q4_SCHEME_NAMES, (
            f"{s.name}.upgrade_neighbor points at a legacy Q4 scheme"
        )
        assert s.downgrade_neighbor not in LEGACY_Q4_SCHEME_NAMES, (
            f"{s.name}.downgrade_neighbor points at a legacy Q4 scheme"
        )


# ── v1 sampling exclusion ────────────────────────────────────────────────────

def _survivor(**kwargs):
    predictor = PredictiveScorer(sensitivity_weights={}, baseline_size_gb=5.0)
    return EvolutionarySurvivor(
        predictor=predictor,
        baseline_config={},
        max_generations=1,
        population_size=10,
        **kwargs,
    )


def test_generate_random_config_never_yields_legacy_q4():
    random.seed(7)
    survivor = _survivor()
    groups = ['E', 'H', 'Q', 'K', 'O', 'U', 'D']

    seen = set()
    for _ in range(300):
        cfg = survivor._generate_random_config(groups)
        seen.update(cfg.values())

    assert not (seen & LEGACY_Q4_SCHEME_NAMES), (
        f"legacy Q4 scheme(s) leaked into v1 random-config sampling: "
        f"{seen & LEGACY_Q4_SCHEME_NAMES}"
    )


def test_generate_random_config_never_yields_legacy_q4_rocmfpx_and_iq_enabled():
    """Same exclusion must hold even with the other opt-in families enabled
    -- the drop is unconditional, not gated behind enable_rocmfpx/enable_iq."""
    random.seed(7)
    survivor = _survivor(enable_rocmfpx=True, enable_iq=True)
    groups = ['E', 'H', 'Q', 'K', 'O', 'U', 'D']

    seen = set()
    for _ in range(300):
        cfg = survivor._generate_random_config(groups)
        seen.update(cfg.values())

    assert not (seen & LEGACY_Q4_SCHEME_NAMES)


# ── encode path sanity (real libggml required; skipped otherwise) ──────────

def _libggml_available():
    try:
        from magicquant.quant.ggml_binding import get_handle
        get_handle()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _libggml_available(), reason="libggml not available")
@pytest.mark.parametrize("name,block,type_size", [
    ("Q4_0", 32, 18),
    ("Q4_1", 32, 20),
])
def test_encode_to_ggml_bytes_produces_expected_byte_length(name, block, type_size):
    from magicquant.quant.converters import encode_to_ggml_bytes, GGML_BLOCK_SIZE, GGML_TYPE_SIZE

    # These byte sizes come from ggml itself (block=32 legacy quants), and
    # converters.py's tables are derived from the ggml_binding block/size
    # tables -- confirm Q4_0/Q4_1 are already present there (they predate
    # this scheme-registry addition; the registry entries above just make
    # them reachable by MagicQuant scheme name).
    assert GGML_BLOCK_SIZE[name] == block
    assert GGML_TYPE_SIZE[name] == type_size

    rows, cols = 2, 256
    w = np.random.randn(rows, cols).astype(np.float32)
    blob = encode_to_ggml_bytes(w, name)

    n_elements = rows * cols
    n_blocks = (n_elements + block - 1) // block
    expected = n_blocks * type_size
    assert len(blob) == expected
