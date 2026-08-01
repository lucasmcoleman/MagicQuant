"""Refactor regression test for PR0.

Pins random.seed and numpy.random seed, runs a small evolutionary search,
captures the candidate sequence, and asserts it matches a stored fixture.

This test must pass identically before and after the scheme-registry refactor.

The fixture (tests/fixtures/refactor_regression_seed42.json) was
DELIBERATELY regenerated for the 2026-07 TIER_SCHEME_VERSION 2 fix
(magicquant.quant.tiers.TIER_BOUNDARIES): EvolutionarySurvivor's per-
generation tournament selection classifies the population into tiers via
``classify_tier`` (see survival.py's ``_classify_into_tiers``), so shifting
the boundaries changes which candidates compete against which within a
generation -- and therefore the whole downstream trajectory, even under an
identical RNG seed. This is intentional, expected drift from that fix, not
an unrelated behavior change the fixture is supposed to catch -- re-verify
manually (inspect the captured configs for sanity: real scheme names, no
crash/NaN artifacts) before regenerating this fixture again for any FUTURE
change, exactly as was done here.
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
    """Build a deterministic predictor with synthetic sensitivity weights."""
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


def _capture_run(seed: int = 42, generations: int = 3, population: int = 20):
    """Run a small deterministic evolution and capture the discovered configs.

    Returns a list of config dicts (group → scheme), in discovery order.
    """
    random.seed(seed)
    np.random.seed(seed)

    predictor = _build_predictor()
    baseline = {g: "MXFP4_MOE" for g in ["E", "H", "Q", "K", "O", "U", "D"]}
    survivor = EvolutionarySurvivor(
        predictor=predictor,
        baseline_config=baseline,
        max_generations=generations,
        population_size=population,
        epsilon=0.0,  # disable epsilon-greedy randomness for determinism
    )
    discovered = survivor.run_evolution(verbose=False)
    return [c["config"] for c in discovered]


def _load_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not yet captured: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text())


def test_evolution_seed42_matches_fixture():
    """Search behavior must be identical pre- and post-refactor."""
    captured = _capture_run(seed=42, generations=3, population=20)
    expected = _load_fixture()
    assert captured == expected, (
        "Refactor changed search behavior. "
        f"Captured {len(captured)} configs, expected {len(expected)}. "
        f"First diff at index {next((i for i, (a, b) in enumerate(zip(captured, expected)) if a != b), 'end')}"
    )
