"""Early-stopping tests (L7).

run_evolution gains an optional ``patience`` that stops the search once the
best composite_score plateaus. The default (patience=None) preserves the
historical full-budget behavior so the seed-pinned refactor-regression fixture
is unaffected.
"""
import random

import numpy as np

from magicquant.evolution.survival import EvolutionarySurvivor


class _ConstantPredictor:
    """Predictor stub that returns the same scores for every config and counts
    how many times it is consulted."""

    def __init__(self):
        self.calls = 0
        self.baseline_size_gb = 5.0
        self.baseline_tps = 20.0
        self.sensitivity_weights = {
            "E": 1.5, "H": 1.4, "Q": 1.0, "K": 0.9,
            "O": 1.2, "U": 0.5, "D": 0.5,
        }

    def score_hybrid(self, config, **kwargs):
        self.calls += 1
        # Constant scores -> best composite_score never improves between
        # generations, so patience must trigger.
        return {
            'predicted_loss': 1.0,
            'predicted_size_gb': 2.5,
            'predicted_tps': 10.0,
            'loss_score': 0.8,
            'size_score': 0.5,
            'tps_score': 0.5,
            'composite_score': 0.65,
        }


def _make_survivor(predictor, max_generations, population):
    baseline = {g: "MXFP4_MOE" for g in ["E", "H", "Q", "K", "O", "U", "D"]}
    return EvolutionarySurvivor(
        predictor=predictor,
        baseline_config=baseline,
        max_generations=max_generations,
        population_size=population,
        epsilon=0.0,  # deterministic
    )


def test_patience_stops_early():
    """With patience=2 and a constant-score predictor, the search stops well
    before the full generation budget, so far fewer predictions run than
    max_generations * population."""
    random.seed(0)
    np.random.seed(0)

    pred = _ConstantPredictor()
    max_generations = 50
    population = 20
    survivor = _make_survivor(pred, max_generations, population)

    survivor.run_evolution(verbose=False, patience=2)

    # Plateau is immediate (constant scores), so it should stop a couple of
    # generations in — strictly fewer predictions than the full budget.
    assert pred.calls < max_generations * population
    # Sanity: it ran at least a few generations before stopping.
    assert pred.calls >= population


def test_patience_none_runs_full_budget():
    """patience=None (default) runs every generation: predictions roughly
    match the full max_generations * population budget (and definitely far
    more than the early-stop case)."""
    random.seed(0)
    np.random.seed(0)

    pred = _ConstantPredictor()
    max_generations = 6
    population = 20
    survivor = _make_survivor(pred, max_generations, population)

    survivor.run_evolution(verbose=False)  # patience defaults to None

    # Every generation predicts the full population (the first generation seeds
    # the population, later generations refill to population_size).
    assert pred.calls == max_generations * population


def test_patience_does_not_change_default_results():
    """Calling run_evolution with patience=None yields the identical discovered
    config sequence as omitting patience entirely (default-off guarantee)."""
    def _run(**kw):
        random.seed(42)
        np.random.seed(42)
        pred = _ConstantPredictor()
        survivor = _make_survivor(pred, max_generations=4, population=15)
        return [c['config'] for c in survivor.run_evolution(verbose=False, **kw)]

    assert _run() == _run(patience=None)
