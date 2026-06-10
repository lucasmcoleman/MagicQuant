"""Tier-classification tests (L8).

The leaf module ``magicquant.quant.tiers.classify_tier`` must return the same
label as ``MagicQuantOrchestrator._classify_tier`` across the full ratio grid,
since the orchestrator now delegates to the leaf module (breaking the prior
circular import).
"""
import pytest

from magicquant.quant.tiers import classify_tier


@pytest.mark.parametrize("ratio,expected", [
    (0.10, "Q2"),
    (0.16, "Q2"),   # boundary: <= 0.16 is Q2
    (0.17, "Q3"),
    (0.22, "Q3"),   # boundary: <= 0.22 is Q3
    (0.23, "Q4"),
    (0.33, "Q4"),   # boundary: <= 0.33 is Q4
    (0.40, "Q5"),
    (0.45, "Q5"),   # boundary: <= 0.45 is Q5
    (0.50, "Q6"),
    (0.65, "Q6"),   # boundary: <= 0.65 is Q6
    (0.70, "Q8"),   # > 0.65 is Q8
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
