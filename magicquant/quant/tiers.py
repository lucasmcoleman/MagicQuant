"""
Tier classification — leaf module with the canonical size-ratio boundaries.

A model's compression tier (Q8/Q6/Q5/Q4/Q3/Q2) is derived purely from the
ratio of its predicted/measured size to the BF16 baseline size. This logic
used to live on ``MagicQuantOrchestrator._classify_tier``, which forced the
leaf evolution modules (survival.py, predictor.py) to do function-local
imports of the top-level orchestrator to dodge a circular import.

Putting the pure arithmetic here — with no upward dependencies — lets every
module import ``classify_tier`` directly. ``MagicQuantOrchestrator._classify_tier``
now delegates here so a single set of boundaries is used everywhere.
"""

# Tier boundaries as (lower_exclusive, upper_inclusive) on size_gb / baseline_gb.
# Tighter boundaries: Q6 targets 45-65% of BF16 (not open-ended); configs
# above 65% are over-protected and wasteful and land in Q8.
TIER_BOUNDARIES = [
    ("Q6", 0.45, 0.65),
    ("Q5", 0.33, 0.45),
    ("Q4", 0.22, 0.33),
    ("Q3", 0.16, 0.22),
]

# Tier assigned when baseline_gb is unusable (<= 0).
DEFAULT_TIER = "Q4"


def classify_tier(size_gb: float, baseline_gb: float) -> str:
    """Classify a model size into a compression tier by ratio to baseline.

    Args:
        size_gb: The model size in GB (predicted or measured).
        baseline_gb: The BF16 baseline size in GB.

    Returns:
        One of "Q8", "Q6", "Q5", "Q4", "Q3", "Q2".
    """
    if baseline_gb <= 0:
        return DEFAULT_TIER
    ratio = size_gb / baseline_gb
    for tier, low, high in TIER_BOUNDARIES:
        if low < ratio <= high:
            return tier
    if ratio <= 0.16:
        return "Q2"
    return "Q8"  # ratio > 0.65 — barely compressed, separate tier
