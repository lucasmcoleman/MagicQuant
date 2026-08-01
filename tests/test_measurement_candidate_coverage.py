"""Per-tier measurement-candidate coverage tests (Part B item 4).

``MagicQuantOrchestrator._select_measurement_candidates`` must guarantee at
least one candidate per discovered tier band survives into the returned
list -- tier winners are measured unconditionally, never truncated away by
a small ``n`` (candidates_per_round) and never crowded out by epsilon-greedy
random picks appended after them.
"""
from magicquant.orchestrator import MagicQuantOrchestrator


def _orch():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._measured = {}
    return orch


def test_tier_winners_survive_even_when_more_tiers_than_n(monkeypatch):
    """Six distinct tiers discovered, but n (candidates_per_round) is only
    2 -- every tier's sole (and therefore winning) candidate must still be
    returned, not just the first two by tier-iteration order."""
    orch = _orch()
    # baseline_gb=10.0; sizes chosen to land one config in each of Q8, Q6,
    # Q5, Q4, Q3, Q2 (see magicquant.quant.tiers.TIER_BOUNDARIES -- 2026-07
    # fix, TIER_SCHEME_VERSION 2: Q6 (0.375,0.46], Q5 (0.328,0.375],
    # Q4 (0.242,0.328], Q3 (0.178,0.242]).
    sizes = {
        "Q8": 8.0, "Q6": 4.2, "Q5": 3.5, "Q4": 2.8, "Q3": 2.0, "Q2": 1.0,
    }
    configs = [
        {"config": {"E": f"cfg_{tier}"}, "predicted_size_gb": size, "composite_score": 1.0}
        for tier, size in sizes.items()
    ]

    result = orch._select_measurement_candidates(configs, baseline_gb=10.0, n=2)
    result_configs = [c["config"] for c in result]

    for cfg in configs:
        assert cfg["config"] in result_configs, (
            f"tier winner {cfg['config']} was dropped by a small n"
        )


def test_epsilon_picks_never_crowd_out_the_single_tier_winner():
    """All candidates land in the same tier (Q6); one is the clear winner
    by composite_score, the rest are noise. Even with many noise
    candidates and a tight n, the real winner must be included, and total
    picks must not exceed n."""
    orch = _orch()
    winner = {"config": {"E": "winner"}, "predicted_size_gb": 6.0, "composite_score": 5.0}
    noise = [
        {"config": {"E": f"noise{i}"}, "predicted_size_gb": 6.0, "composite_score": 0.1}
        for i in range(20)
    ]

    result = orch._select_measurement_candidates([winner] + noise, baseline_gb=10.0, n=3)
    result_configs = [c["config"] for c in result]

    assert winner["config"] in result_configs
    assert len(result) == 3


def test_already_measured_tier_winner_does_not_block_epsilon_budget():
    """A tier winner that's already been measured in a prior round doesn't
    need a rebuild, but its slot shouldn't be wasted -- the full epsilon
    budget still gets spent on the remaining pool."""
    orch = _orch()
    winner = {"config": {"E": "already_done"}, "predicted_size_gb": 6.0, "composite_score": 5.0}
    orch._measured[MagicQuantOrchestrator._config_key(winner["config"])] = {
        "config": winner["config"],
    }
    noise = [
        {"config": {"E": f"noise{i}"}, "predicted_size_gb": 6.0, "composite_score": 0.1}
        for i in range(5)
    ]

    result = orch._select_measurement_candidates([winner] + noise, baseline_gb=10.0, n=2)
    result_configs = [c["config"] for c in result]

    assert winner["config"] not in result_configs  # already measured, no rebuild
    assert len(result) == 2  # full budget spent on noise, not shrunk by the skip


def test_no_candidates_returns_empty():
    orch = _orch()
    assert orch._select_measurement_candidates([], baseline_gb=10.0, n=4) == []
