"""F1 regression: _generate_random_config must realize each category's
*documented* sampling mass (the _FFN/_ATTENTION/_BRAIN_CLASS_WEIGHTS
tables), not a mass that scales with how many schemes share the category.

Uses FFN (robust, no sensitive-floor clamp) and attention groups so the
measured shares reflect the raw per-category weight tables directly.

Built with has_imatrix=True so the FULL pool is in play: the documented
weight tables describe a pool that includes IQ4_NL (the only default-pool
member of category 'iq_quant'). Without an imatrix that scheme is gated out
(see IMATRIX_DEPENDENT_SCHEME_NAMES) and its mass renormalizes proportionally
across the surviving categories -- asserted separately below, since that
renormalization is the intended behaviour and not a drift in the tables.
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
    survivor = _survivor(has_imatrix=True)
    shares = _sample_category_shares(survivor, "D")  # FFN, no floor clamp

    for category, expected in EvolutionarySurvivor._FFN_CLASS_WEIGHTS.items():
        observed = shares.get(category, 0.0)
        assert abs(observed - expected) <= TOLERANCE, (
            f"FFN category {category!r}: expected ~{expected:.2f}, got {observed:.2f}"
        )


def test_attention_category_mass_matches_documented_weights():
    random.seed(42)
    survivor = _survivor(has_imatrix=True)
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


def test_iq_category_mass_renormalizes_when_gated_out():
    """Without an imatrix, iq_quant's mass must vanish and redistribute
    PROPORTIONALLY across the remaining categories -- not silently reshape
    the relative ordering of what's left."""
    random.seed(42)
    shares = _sample_category_shares(_survivor(has_imatrix=False), "D")
    assert shares.get("iq_quant", 0.0) == 0.0, (
        f"iq_quant sampled without an imatrix: {shares.get('iq_quant')}"
    )
    weights = EvolutionarySurvivor._FFN_CLASS_WEIGHTS
    lost = weights.get("iq_quant", 0.0)
    assert lost > 0, "test assumes iq_quant carries FFN mass in the full pool"
    for category, documented in weights.items():
        if category == "iq_quant":
            continue
        expected = documented / (1.0 - lost)     # proportional renormalization
        observed = shares.get(category, 0.0)
        assert abs(observed - expected) <= TOLERANCE, (
            f"FFN category {category!r} after gating iq_quant: expected "
            f"~{expected:.2f} (={documented:.2f}/{1-lost:.2f}), got {observed:.2f}"
        )
