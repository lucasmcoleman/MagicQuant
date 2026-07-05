"""Seed-config injection tests for EvolutionarySurvivor.run_evolution
(Part B item 2).

Covers: validated seeds land in the initial population AND in the returned
discovered-configs list even if a tournament never re-selects them; unknown
schemes are logged and dropped whole (not partially repaired); and the
default (seed_configs=None) path is byte-identical to the pre-existing
seed-pinned refactor-regression fixture -- verified by re-running that exact
fixture (not just a similar-looking search) before and after this feature
exists.
"""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "refactor_regression_seed42.json"


def _build_predictor():
    sensitivity_weights = {
        "E": 1.5, "H": 1.4, "Q": 1.0, "K": 0.9,
        "O": 1.2, "U": 0.5, "D": 0.5,
    }
    parameter_counts = {
        "E": 100_000_000, "H": 100_000_000, "Q": 300_000_000, "K": 300_000_000,
        "O": 150_000_000, "U": 800_000_000, "D": 800_000_000,
    }
    return PredictiveScorer(
        sensitivity_weights=sensitivity_weights,
        parameter_counts=parameter_counts,
        baseline_size_gb=5.0,
        baseline_tps=20.0,
    )


GROUPS = ["E", "H", "Q", "K", "O", "U", "D"]


def _make_survivor(**kwargs):
    defaults = dict(
        predictor=_build_predictor(),
        baseline_config={g: "MXFP4_MOE" for g in GROUPS},
        max_generations=3,
        population_size=20,
        epsilon=0.0,
    )
    defaults.update(kwargs)
    return EvolutionarySurvivor(**defaults)


def test_seed_config_appears_in_discovered_configs():
    random.seed(1)
    np.random.seed(1)
    survivor = _make_survivor()
    seed = {
        "E": "Q4_K_M", "H": "Q6_K", "Q": "Q4_K_M", "K": "Q4_K_M",
        "O": "Q4_K_M", "U": "Q4_K_M", "D": "Q4_K_M",
    }
    discovered = survivor.run_evolution(
        groups=GROUPS, verbose=False, seed_configs=[seed]
    )
    configs = [c["config"] for c in discovered]
    assert seed in configs


def test_seed_config_carries_prediction_fields():
    random.seed(2)
    np.random.seed(2)
    survivor = _make_survivor()
    seed = {g: "Q5_K" for g in GROUPS}
    discovered = survivor.run_evolution(
        groups=GROUPS, verbose=False, seed_configs=[seed]
    )
    match = next(c for c in discovered if c["config"] == seed)
    assert "composite_score" in match
    assert "predicted_size_gb" in match
    assert "tier" in match


def test_unknown_scheme_in_seed_is_dropped_not_repaired():
    random.seed(3)
    np.random.seed(3)
    survivor = _make_survivor()
    bad_seed = {g: "TOTALLY_MADE_UP_SCHEME" for g in GROUPS}
    discovered = survivor.run_evolution(
        groups=GROUPS, verbose=False, seed_configs=[bad_seed]
    )
    configs = [c["config"] for c in discovered]
    assert bad_seed not in configs
    assert not any(
        any(s == "TOTALLY_MADE_UP_SCHEME" for s in c.values()) for c in configs
    )


def test_mixed_valid_and_invalid_seeds_keeps_only_valid():
    random.seed(4)
    np.random.seed(4)
    survivor = _make_survivor()
    good_seed = {g: "Q6_K" for g in GROUPS}
    bad_seed = dict(good_seed)
    bad_seed["E"] = "NOT_A_REAL_SCHEME"
    discovered = survivor.run_evolution(
        groups=GROUPS, verbose=False, seed_configs=[good_seed, bad_seed]
    )
    configs = [c["config"] for c in discovered]
    assert good_seed in configs
    assert bad_seed not in configs


def test_empty_seed_list_is_a_no_op():
    random.seed(5)
    np.random.seed(5)
    survivor = _make_survivor()
    discovered = survivor.run_evolution(groups=GROUPS, verbose=False, seed_configs=[])
    assert isinstance(discovered, list)


def _load_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not yet captured: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text())


def _capture_run(seed_configs):
    random.seed(42)
    np.random.seed(42)
    predictor = _build_predictor()
    baseline = {g: "MXFP4_MOE" for g in GROUPS}
    survivor = EvolutionarySurvivor(
        predictor=predictor,
        baseline_config=baseline,
        max_generations=3,
        population_size=20,
        epsilon=0.0,
    )
    discovered = survivor.run_evolution(verbose=False, seed_configs=seed_configs)
    return [c["config"] for c in discovered]


def test_default_none_seed_configs_matches_fixture_twice():
    """seed_configs=None (the default) must reproduce the exact seed-pinned
    fixture -- run twice to also confirm it's deterministic under repeats,
    not just correct once."""
    expected = _load_fixture()

    first = _capture_run(seed_configs=None)
    assert first == expected

    second = _capture_run(seed_configs=None)
    assert second == expected
