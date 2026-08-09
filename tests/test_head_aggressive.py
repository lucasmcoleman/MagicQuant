"""Tests for the opt-in head_aggressive dial (LANE 2 / PART A).

EvolutionarySurvivor.__init__ gains a head_aggressive: bool = False knob
that, when True, reweights random-config sampling for the 'H' group ONLY
toward the smaller K-quants (Q6_K/Q5_K) and Q8_0, away from BF16 --
output.weight streams in full every generated token (unlike the
row-gathered token_embd), so a BF16 head is a per-token bandwidth tax the
PPL-only objective never sees. It's a bias (reweighted categories), not a
hard exclusion: BF16 stays reachable, just unlikely.

Covers: aggressive draws for H are predominantly non-float; default draws
(head_aggressive unset, i.e. False) are byte-identical to the pre-existing
behavior; every group other than H is completely unaffected by the flag,
seed-for-seed; and the seed-pinned refactor-regression fixture (which
constructs EvolutionarySurvivor without this param) is unchanged.
"""
import random
from collections import Counter

import numpy as np

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor

GROUPS = ["E", "H", "Q", "K", "O", "U", "D"]


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


def test_head_aggressive_defaults_to_false():
    survivor = _make_survivor()
    assert survivor.head_aggressive is False


def test_head_aggressive_draws_are_predominantly_non_float():
    random.seed(11)
    survivor = _make_survivor(head_aggressive=True)
    counts = Counter()
    n = 4000
    for _ in range(n):
        cfg = survivor._generate_random_config(["H"])
        counts[cfg["H"]] += 1

    non_float = n - counts["BF16"]
    assert non_float / n > 0.9, f"Expected predominantly non-float H draws, got {counts}"
    # The dial's documented target band should dominate the non-float mass.
    target_band = counts["Q6_K"] + counts["Q5_K"] + counts["Q8_0"]
    assert target_band / n > 0.85, f"Expected Q6_K/Q5_K/Q8_0 to dominate, got {counts}"


def test_head_aggressive_false_matches_historical_distribution():
    """With the dial off (explicit False), H sampling must match the
    pre-existing brain-class distribution: the sensitive-group floor clamps
    every sub-Q8_0 pick back to Q8_0, so only BF16 (~30%) and Q8_0 (~70%)
    should ever appear."""
    random.seed(11)
    survivor = _make_survivor(head_aggressive=False)
    counts = Counter()
    n = 4000
    for _ in range(n):
        cfg = survivor._generate_random_config(["H"])
        counts[cfg["H"]] += 1

    assert set(counts) <= {"BF16", "Q8_0"}
    assert 0.20 < counts["BF16"] / n < 0.40


def test_head_aggressive_leaves_other_groups_byte_identical():
    """The bias must touch H's sampling ONLY -- every other group's pick
    must be identical whether head_aggressive is True or False, given the
    same RNG seed (random.choices always consumes exactly one draw per
    group regardless of the weights passed in, so this is deterministic)."""
    random.seed(123)
    survivor_default = _make_survivor(head_aggressive=False)
    cfg_default = survivor_default._generate_random_config(GROUPS)

    random.seed(123)
    survivor_aggressive = _make_survivor(head_aggressive=True)
    cfg_aggressive = survivor_aggressive._generate_random_config(GROUPS)

    for g in GROUPS:
        if g == "H":
            continue
        assert cfg_default[g] == cfg_aggressive[g], (
            f"Group {g} diverged: {cfg_default[g]!r} vs {cfg_aggressive[g]!r}"
        )


def test_head_aggressive_threads_through_run_evolution_without_error():
    """Smoke-test the full run_evolution path (not just _generate_random_config
    directly) with head_aggressive=True."""
    random.seed(5)
    np.random.seed(5)
    survivor = _make_survivor(head_aggressive=True)
    discovered = survivor.run_evolution(groups=GROUPS, verbose=False)
    assert len(discovered) > 0
    for cfg in discovered:
        assert "H" in cfg["config"]
