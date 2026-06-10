"""Search group-coverage tests (H2).

Evolutionary search must vary X/R/S groups when they are present, not leave
them pinned to the base quant. Previously run_evolution was called without
groups so it fell back to DEFAULT_GROUPS=['E','H','Q','K','O','U','D'] and X/R/S
never varied.
"""
import random

import numpy as np
import pytest

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor


def _make_survivor(groups):
    counts = {g: 100_000_000 for g in groups}
    counts["X"] = 800_000_000 if "X" in groups else counts.get("X", 0)
    predictor = PredictiveScorer(
        sensitivity_weights={g: 1.0 for g in groups},
        parameter_counts={g: counts.get(g, 100_000_000) for g in groups},
        baseline_size_gb=5.0,
        baseline_tps=100.0,
    )
    return EvolutionarySurvivor(
        predictor=predictor,
        baseline_config={"E": "BF16", "H": "BF16"},
        max_generations=5,
        population_size=40,
        epsilon=0.2,
    )


def test_x_group_present_and_varied():
    random.seed(7)
    np.random.seed(7)
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X", "R"]
    survivor = _make_survivor(groups)
    discovered = survivor.run_evolution(groups=groups, verbose=False)

    assert discovered, "search produced no configs"

    # X must appear in every discovered config.
    for cfg in discovered:
        assert "X" in cfg["config"], f"config missing X: {cfg['config']}"

    # X must take more than one distinct scheme across the discovered set.
    x_schemes = {cfg["config"]["X"] for cfg in discovered}
    assert len(x_schemes) > 1, (
        f"X group never varied — only saw {x_schemes}. Search is ignoring X."
    )


def test_default_groups_when_none_passed():
    """Calling run_evolution() with no groups uses DEFAULT_GROUPS (no X)."""
    random.seed(7)
    np.random.seed(7)
    groups = EvolutionarySurvivor.DEFAULT_GROUPS
    survivor = _make_survivor(list(groups) + ["X"])
    discovered = survivor.run_evolution(verbose=False)
    assert discovered
    for cfg in discovered:
        assert "X" not in cfg["config"]
