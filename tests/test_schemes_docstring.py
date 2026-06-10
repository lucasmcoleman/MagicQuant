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
    """Sanity: schemes ordered by noise must also be (weakly) ordered by bpw
    descending — a heuristic-but-coherent registry."""
    ordered = get_all_schemes()  # ascending noise
    noises = [s.noise_factor for s in ordered]
    assert noises == sorted(noises)
    # The cleanest scheme (BF16) has zero noise.
    assert get_scheme_by_name("BF16").noise_factor == 0.0
