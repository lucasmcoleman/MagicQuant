"""Sub-4-bit IQ-quant support (registry + imatrix gate + search gating).

Mirrors tests/test_rocmfpx_schemes.py's structure for the new opt-in IQ
family (IQ4_XS/IQ3_S/IQ3_XXS/IQ2_S/IQ2_XS/IQ2_XXS/IQ1_M/IQ1_S): registered in
the scheme registry but excluded from the default evolutionary search unless
enable_iq=True, and schemes the library flags as requires_imatrix are NEVER
sampled (the search threads no imatrix).
"""
import random

import numpy as np
import pytest

from magicquant.quant.schemes import (
    get_scheme_by_name, get_all_schemes, IQ_SCHEME_NAMES,
)
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor


# ── registry ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,type_id,bpw", [
    ("IQ4_XS", 23, 4.25),
    ("IQ3_S", 21, 3.4375),
    ("IQ3_XXS", 18, 3.0625),
    ("IQ2_S", 22, 2.5625),
    ("IQ2_XS", 17, 2.3125),
    ("IQ2_XXS", 16, 2.0625),
    ("IQ1_M", 29, 1.75),
    ("IQ1_S", 19, 1.5625),
])
def test_iq_scheme_metadata(name, type_id, bpw):
    s = get_scheme_by_name(name)
    assert s.ggml_type_id == type_id
    assert s.bits_per_weight == bpw
    assert s.category == "iq_quant"
    assert s.ggml_type_name == name


def test_iq_scheme_names_constant_matches_expected_set():
    expected = {
        "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S",
        "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
    }
    assert set(IQ_SCHEME_NAMES) == expected
    # IQ4_NL is an existing default-pool scheme, not part of the new opt-in
    # family — must never be swept into IQ_SCHEME_NAMES.
    assert "IQ4_NL" not in IQ_SCHEME_NAMES


def test_iq_scheme_names_subset_of_registry_iq_quant_category():
    reg_iq = {s.name for s in get_all_schemes() if s.category == "iq_quant"}
    # IQ4_NL plus the 8 new schemes.
    assert reg_iq == IQ_SCHEME_NAMES | {"IQ4_NL"}


# ── search gating ────────────────────────────────────────────────────────────

def _survivor(enable_iq, **kwargs):
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X"]
    predictor = PredictiveScorer(
        sensitivity_weights={g: 1.0 for g in groups},
        parameter_counts={g: 100_000_000 for g in groups},
        baseline_size_gb=5.0,
        baseline_tps=100.0,
    )
    return EvolutionarySurvivor(
        predictor=predictor,
        baseline_config={"E": "BF16", "H": "BF16"},
        max_generations=6,
        population_size=60,
        epsilon=0.2,
        enable_iq=enable_iq,
        **kwargs,
    )


def _all_schemes_used(configs):
    used = set()
    for c in configs:
        used.update(c["config"].values())
    return used


def _requires_imatrix_names():
    return {s.name for s in get_all_schemes() if s.requires_imatrix}


def test_default_search_never_samples_iq_schemes():
    random.seed(13)
    np.random.seed(13)
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X"]
    configs = _survivor(enable_iq=False).run_evolution(groups=groups, verbose=False)
    used = _all_schemes_used(configs)
    assert not (used & IQ_SCHEME_NAMES), (
        "IQ schemes leaked into a default (disabled) search"
    )
    assert not (used & _requires_imatrix_names()), (
        "a requires_imatrix scheme leaked into a default search"
    )


def test_enabled_search_samples_non_imatrix_iq_schemes():
    random.seed(13)
    np.random.seed(13)
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X"]
    configs = _survivor(enable_iq=True).run_evolution(groups=groups, verbose=False)
    used = _all_schemes_used(configs)
    non_imatrix_iq = IQ_SCHEME_NAMES - _requires_imatrix_names()
    assert used & non_imatrix_iq, (
        "no non-imatrix IQ scheme appeared despite enable_iq=True"
    )
    # requires_imatrix schemes must NEVER appear, enable_iq or not.
    assert not (used & _requires_imatrix_names()), (
        "a requires_imatrix scheme leaked into an enable_iq=True search"
    )


def test_requires_imatrix_schemes_never_sampled_regardless_of_enable_iq():
    imatrix_names = _requires_imatrix_names()
    assert imatrix_names, "expected at least one requires_imatrix scheme in the registry"
    for enable_iq in (False, True):
        random.seed(99)
        np.random.seed(99)
        groups = ["E", "H", "Q", "K", "O", "U", "D", "X"]
        configs = _survivor(enable_iq=enable_iq).run_evolution(groups=groups, verbose=False)
        used = _all_schemes_used(configs)
        assert not (used & imatrix_names), (
            f"requires_imatrix scheme leaked with enable_iq={enable_iq}"
        )


# ── live encode round trip (real libggml required; skipped otherwise) ────────
#
# Run in a SUBPROCESS, mirroring tests/test_rocmfpx_schemes.py's pattern: keep
# any libggml ctypes loading isolated from other test imports in this process.
# Only encode schemes the library itself reports as NOT requiring an imatrix
# (encoding an imatrix-required type without one may error or produce a
# degenerate result).

import os
import subprocess
import sys
import textwrap

_CANDIDATE_NAMES = [
    "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
]

_ENCODE_CHILD = textwrap.dedent("""
    import sys, numpy as np
    from magicquant.quant.ggml_binding import get_handle, _GGML_BLOCK_SIZE, _GGML_TYPE_SIZE
    h = get_handle()
    names = %r
    w = np.random.randn(256 * 20).astype("float32")
    results = []
    for name in names:
        if h.requires_imatrix(name):
            continue
        blk = _GGML_BLOCK_SIZE[name]
        size = _GGML_TYPE_SIZE[name]
        n_blocks = (w.size + blk - 1) // blk
        expected = n_blocks * size
        n = len(h.encode(w, name))
        assert n == expected, f"{name}: {n} != {expected}"
        results.append(name)
    print("OK:" + ",".join(results))
""" % (_CANDIDATE_NAMES,))


def _libggml_available():
    try:
        from magicquant.quant.ggml_binding import get_handle
        get_handle()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _libggml_available(), reason="libggml not available")
def test_live_encode_matches_block_size_subprocess():
    proc = subprocess.run(
        [sys.executable, "-c", _ENCODE_CHILD],
        env=dict(os.environ), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    out = proc.stdout.strip().splitlines()[-1]
    assert out.startswith("OK:"), f"unexpected child output: {out!r}\n{proc.stderr}"
    encoded = set(out[len("OK:"):].split(",")) if out != "OK:" else set()
    # At least the schemes we've verified as non-imatrix must have encoded.
    assert encoded, "no IQ scheme was encoded (all reported as requires_imatrix?)"
