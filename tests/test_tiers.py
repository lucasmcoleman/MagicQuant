"""Tier-classification tests (L8).

The leaf module ``magicquant.quant.tiers.classify_tier`` must return the same
label as ``MagicQuantOrchestrator._classify_tier`` across the full ratio grid,
since the orchestrator now delegates to the leaf module (breaking the prior
circular import).
"""
import pytest

from magicquant.quant.schemes import get_all_schemes
from magicquant.quant.tiers import (
    CURRENT_TIER_SCHEME_VERSION,
    LEGACY_TIER_SCHEME_VERSION,
    TIER_BOUNDARIES_V1,
    classify_tier,
    describe_tier_band,
    tier_scheme_version,
)


# v2 boundaries (2026-07 fix): Q6 (0.375,0.46], Q5 (0.328,0.375],
# Q4 (0.242,0.328], Q3 (0.178,0.242]; <=0.178 Q2; >0.46 Q8.
@pytest.mark.parametrize("ratio,expected", [
    (0.10, "Q2"),
    (0.178, "Q2"),  # boundary: <= 0.178 is Q2
    (0.179, "Q3"),
    (0.242, "Q3"),  # boundary: <= 0.242 is Q3
    (0.243, "Q4"),
    (0.328, "Q4"),  # boundary: <= 0.328 is Q4
    (0.329, "Q5"),
    (0.375, "Q5"),  # boundary: <= 0.375 is Q5
    (0.376, "Q6"),
    (0.46, "Q6"),   # boundary: <= 0.46 is Q6
    (0.47, "Q8"),   # > 0.46 is Q8
    (1.00, "Q8"),
])
def test_classify_tier_boundaries(ratio, expected):
    # baseline=1.0 so size_gb == ratio exactly (no float multiplication drift
    # at the boundary values).
    assert classify_tier(ratio, 1.0) == expected


def test_classify_tier_zero_baseline():
    assert classify_tier(5.0, 0.0) == "Q4"
    assert classify_tier(5.0, -1.0) == "Q4"


def test_matches_orchestrator_classify_tier():
    """Leaf classify_tier must agree with the orchestrator's delegating method
    across a fine grid that crosses every boundary."""
    from magicquant.orchestrator import MagicQuantOrchestrator

    baseline = 8.0
    for i in range(0, 200):
        ratio = i / 200.0  # 0.0 .. ~1.0
        size = ratio * baseline
        assert classify_tier(size, baseline) == \
            MagicQuantOrchestrator._classify_tier(size, baseline), (
                f"mismatch at ratio={ratio}"
            )


# ── Registry verification (the actual point of the 2026-07 boundary fix) ────
#
# TIER_SCHEME_VERSION 2 was derived BY computing every registry scheme's
# bits_per_weight/16.0 ratio and placing boundaries so each canonical
# scheme's own uniform build classifies under its own name. This test is
# the live proof of that -- it must keep passing (or be deliberately
# re-tuned, per the tiers.py module docstring) whenever the scheme registry
# changes.
_EXPECTED_TIER = {
    "BF16": "Q8", "Q8_0": "Q8", "ROCMFP8": "Q8",
    "Q6_K": "Q6", "ROCMFP6": "Q6",
    # Q5_0 = 5.5/16 = 0.34375 and Q5_1 = 6.0/16 = 0.375 both land in Q5's
    # band (0.328, 0.375] -- Q5_1 sits exactly on the closed upper bound, so
    # Q6's `0.375 < 0.375` is False and it classifies Q5.
    "Q5_K": "Q5", "Q5_0": "Q5", "Q5_1": "Q5",
    "Q4_1": "Q4", "Q4_K_M": "Q4", "IQ4_NL": "Q4", "Q4_0": "Q4",
    "ROCMFP4": "Q4", "MXFP4_MOE": "Q4", "IQ4_XS": "Q4", "NVFP4": "Q4",
    "ROCMFP3": "Q3", "Q3_K": "Q3", "IQ3_S": "Q3", "IQ3_XXS": "Q3",
    "Q2_K": "Q2", "IQ2_S": "Q2", "IQ2_XS": "Q2", "IQ2_XXS": "Q2",
    "IQ1_M": "Q2", "IQ1_S": "Q2",
}


@pytest.mark.parametrize("scheme", get_all_schemes(), ids=lambda s: s.name)
def test_registry_schemes_classify_correctly(scheme):
    """A uniform build of every named registry scheme lands in the
    correspondingly-named tier -- the whole point of the v2 boundary fix
    (v1 put uniform Q6_K and Q8_0 in the wrong band; see tiers.py docstring).
    """
    expected = _EXPECTED_TIER.get(scheme.name)
    if expected is None:
        pytest.skip(f"{scheme.name} has no documented expected tier")
    ratio = scheme.bits_per_weight / 16.0
    assert classify_tier(ratio, 1.0) == expected, (
        f"{scheme.name} (bpw={scheme.bits_per_weight}, ratio={ratio:.4f}) "
        f"classified as {classify_tier(ratio, 1.0)!r}, expected {expected!r}"
    )


def test_v1_boundaries_would_misclassify_q6_k_and_q8_0():
    """Documents the actual bug the v2 fix corrects: under the OLD (v1)
    boundaries, uniform Q6_K (ratio 0.4102) fell in the "Q5" band and real
    Q8_0 (ratio 0.5312) fell in the "Q6" band. Demonstrates the failure
    directly against the preserved historical boundary table rather than
    asserting it only in prose."""
    def classify_v1(ratio):
        for tier, low, high in TIER_BOUNDARIES_V1:
            if low < ratio <= high:
                return tier
        if ratio <= 0.16:
            return "Q2"
        return "Q8"

    assert classify_v1(6.5625 / 16.0) == "Q5"   # Q6_K misclassified as "Q5"
    assert classify_v1(8.5 / 16.0) == "Q6"      # Q8_0 misclassified as "Q6"
    # v2 gets both right (already covered by test_registry_schemes_classify_correctly).
    assert classify_tier(6.5625, 16.0) == "Q6"
    assert classify_tier(8.5, 16.0) == "Q8"


# ── tier_scheme_version compatibility read path ──────────────────────────

def test_tier_scheme_version_reads_stamped_value():
    assert tier_scheme_version({"tier_scheme_version": 2}) == 2
    assert tier_scheme_version({"tier_scheme_version": CURRENT_TIER_SCHEME_VERSION}) \
        == CURRENT_TIER_SCHEME_VERSION


def test_tier_scheme_version_defaults_legacy_when_absent():
    """A search_results.json written before this versioning existed has no
    'tier_scheme_version' key at all -- must read as legacy (v1), not crash
    and not silently claim to be current."""
    assert tier_scheme_version({}) == LEGACY_TIER_SCHEME_VERSION
    assert tier_scheme_version({"baseline_ppl": 6.78}) == LEGACY_TIER_SCHEME_VERSION


def test_tier_scheme_version_tolerates_garbage():
    assert tier_scheme_version({"tier_scheme_version": None}) == LEGACY_TIER_SCHEME_VERSION
    assert tier_scheme_version({"tier_scheme_version": "2"}) == LEGACY_TIER_SCHEME_VERSION
    assert tier_scheme_version({"tier_scheme_version": 0}) == LEGACY_TIER_SCHEME_VERSION
    assert tier_scheme_version({"tier_scheme_version": -1}) == LEGACY_TIER_SCHEME_VERSION


def test_tier_scheme_version_guards_bool_against_int_subclass():
    """``isinstance(True, int)`` is True in Python, so a stray
    ``{"tier_scheme_version": True}`` must not fall through the isinstance
    check and be returned as-is (a bool, not a real version number) --
    it must read as LEGACY like any other garbage value.

    Proven to fail pre-fix: the old body was
    ``if not isinstance(version, int) or version <= 0: return LEGACY...;
    return version`` -- for ``version = True``, ``isinstance(True, int)`` is
    True and ``True <= 0`` is False, so the guard clause never triggers and
    the function falls through to ``return version``, returning the bool
    ``True`` itself (which compares equal to ``1`` but is not a real int
    version number) instead of guarding it out like every other garbage
    literal.
    """
    assert tier_scheme_version({"tier_scheme_version": True}) == LEGACY_TIER_SCHEME_VERSION
    result = tier_scheme_version({"tier_scheme_version": True})
    assert type(result) is int and type(result) is not bool


# ── describe_tier_band ────────────────────────────────────────────────────

def test_describe_tier_band_matches_active_boundaries():
    assert describe_tier_band("Q5") == "(0.328, 0.375]"
    assert describe_tier_band("Q6") == "(0.375, 0.46]"
    assert describe_tier_band("Q4") == "(0.242, 0.328]"
    assert describe_tier_band("Q3") == "(0.178, 0.242]"
    assert describe_tier_band("Q2") == "(0, 0.178]"
    assert describe_tier_band("Q8") == "(0.46, 1.0]"
    assert describe_tier_band("nonsense") == "unknown tier"
