"""F1 regression: _generate_random_config must realize each category's
*documented* sampling mass (the _FFN/_ATTENTION/_BRAIN_CLASS_WEIGHTS
tables), not a mass that scales with how many schemes share the category.

Uses FFN (robust, no sensitive-floor clamp) and attention groups so the
measured shares reflect the raw per-category weight tables directly.
"""
import random
from collections import Counter

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.quant.schemes import get_scheme_by_name

TOLERANCE = 0.05
N_SAMPLES = 4000


def _survivor(**kwargs):
    predictor = PredictiveScorer(sensitivity_weights={}, baseline_size_gb=5.0)
    return EvolutionarySurvivor(
        predictor=predictor,
        baseline_config={},
        max_generations=1,
        population_size=10,
        **kwargs,
    )


def _sample_category_shares(survivor, group, n=N_SAMPLES):
    counts = Counter()
    for _ in range(n):
        cfg = survivor._generate_random_config([group])
        scheme = get_scheme_by_name(cfg[group])
        counts[scheme.category] += 1
    return {cat: n_hits / n for cat, n_hits in counts.items()}


def test_ffn_category_mass_matches_documented_weights():
    random.seed(42)
    survivor = _survivor()
    shares = _sample_category_shares(survivor, "D")  # FFN, no floor clamp

    for category, expected in EvolutionarySurvivor._FFN_CLASS_WEIGHTS.items():
        observed = shares.get(category, 0.0)
        assert abs(observed - expected) <= TOLERANCE, (
            f"FFN category {category!r}: expected ~{expected:.2f}, got {observed:.2f}"
        )


def test_attention_category_mass_matches_documented_weights():
    random.seed(42)
    survivor = _survivor()
    shares = _sample_category_shares(survivor, "Q")  # attention, no floor clamp

    for category, expected in EvolutionarySurvivor._ATTENTION_CLASS_WEIGHTS.items():
        observed = shares.get(category, 0.0)
        assert abs(observed - expected) <= TOLERANCE, (
            f"attention category {category!r}: expected ~{expected:.2f}, got {observed:.2f}"
        )


def test_mxfp4_is_not_starved_relative_to_kquant_in_ffn():
    """The audit's headline symptom: mxfp4 (1 scheme, 33% documented) must
    not end up sampled *less* than k_quant (5 schemes, 30% documented)."""
    random.seed(7)
    survivor = _survivor()
    shares = _sample_category_shares(survivor, "U")

    assert shares.get("mxfp4", 0.0) > shares.get("k_quant", 0.0) * 0.8
