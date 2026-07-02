"""Scheme registry hygiene tests (M2 scaffolding).

Until tools/calibrate_noise_factors.py is run, noise_factors are heuristic.
But the source must no longer claim PR1 will overwrite them or label any value
a 'placeholder' — those stale claims were removed. This guard fails if they
creep back in.
"""
from pathlib import Path

import magicquant.quant.schemes as schemes
from magicquant.quant.schemes import get_all_schemes, get_scheme_by_name


def test_no_placeholder_or_pr1_claims():
    src = Path(schemes.__file__).read_text()
    lowered = src.lower()
    assert "placeholder" not in lowered, "schemes.py still calls a value a placeholder"
    assert "pr1 will replace" not in lowered
    assert "pr1 adds the ctypes" not in lowered


def test_noise_factors_strictly_increase_toward_compression():
    """Sanity: every scheme has a well-formed, non-negative noise_factor,
    and the known anchor ordering holds.

    NOTE: this used to assert `noises == sorted(noises)` where `noises` was
    pulled straight from `get_all_schemes()` — but that function already
    returns schemes sorted by noise_factor, so the assertion was
    tautologically true no matter what the actual values were. Instead we
    check real invariants: every noise_factor is a finite non-negative
    number, and a handful of schemes whose relative quality ordering is
    well known (BF16 < Q8_0 < Q4_K_M < Q2_K) are ordered correctly.
    """
    import math

    for scheme in get_all_schemes():
        assert isinstance(scheme.noise_factor, (int, float))
        assert math.isfinite(scheme.noise_factor)
        assert scheme.noise_factor >= 0.0, (
            f"{scheme.name} has a negative noise_factor: {scheme.noise_factor}"
        )

    bf16 = get_scheme_by_name("BF16").noise_factor
    q8_0 = get_scheme_by_name("Q8_0").noise_factor
    q4_k_m = get_scheme_by_name("Q4_K_M").noise_factor
    q2_k = get_scheme_by_name("Q2_K").noise_factor

    assert bf16 < q8_0 < q4_k_m < q2_k

    # The cleanest scheme (BF16) has zero noise.
    assert bf16 == 0.0
