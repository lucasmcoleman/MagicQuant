"""Sensitive-group floor enforcement tests (L6).

_generate_random_config must never assign a high-sensitivity "brain" group
(E, H, O, R) a scheme below the sensitive floor (Q8_0, 8.5 bpw).
"""
import random

from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.quant.schemes import get_scheme_by_name, get_floor_for_group_class


def _survivor():
    predictor = PredictiveScorer(sensitivity_weights={}, baseline_size_gb=5.0)
    return EvolutionarySurvivor(
        predictor=predictor,
        baseline_config={},
        max_generations=1,
        population_size=10,
    )


def test_random_config_respects_sensitive_floor():
    random.seed(123)
    survivor = _survivor()
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X", "R"]
    floor_bpw = get_scheme_by_name(get_floor_for_group_class("sensitive")).bits_per_weight

    for _ in range(500):
        cfg = survivor._generate_random_config(groups)
        for g in EvolutionarySurvivor._HIGH_SENSITIVITY:
            if g in cfg:
                bpw = get_scheme_by_name(cfg[g]).bits_per_weight
                assert bpw >= floor_bpw, (
                    f"group {g} got {cfg[g]} ({bpw} bpw) below floor {floor_bpw}"
                )


def test_robust_groups_unconstrained_by_sensitive_floor():
    """Robust groups (U, D, X) may still be sampled below the sensitive floor."""
    random.seed(123)
    survivor = _survivor()
    groups = ["U", "D", "X"]
    floor_bpw = get_scheme_by_name(get_floor_for_group_class("sensitive")).bits_per_weight

    saw_below = False
    for _ in range(500):
        cfg = survivor._generate_random_config(groups)
        for g in ("U", "D", "X"):
            if get_scheme_by_name(cfg[g]).bits_per_weight < floor_bpw:
                saw_below = True
    assert saw_below, "robust groups should be able to go below the sensitive floor"
