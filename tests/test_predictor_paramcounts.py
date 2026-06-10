"""Predictor parameter-count threading tests (M1).

When real per-group parameter_counts are supplied, predict_size must reflect
the actual distribution — for a MoE where experts (X) hold the bulk of the
weights, changing X's scheme must dominate the predicted size.
"""
from magicquant.evolution.predictor import PredictiveScorer


def _moe_param_counts():
    # Experts (X) hold ~85% of the model's weights — typical MoE.
    return {
        "E": 50_000_000, "H": 50_000_000, "Q": 40_000_000, "K": 40_000_000,
        "O": 30_000_000, "U": 20_000_000, "D": 20_000_000,
        "X": 850_000_000, "R": 5_000_000,
    }


def test_predict_size_scales_with_x_share():
    """Setting the experts group X to a heavier scheme (Q6_K) vs MXFP4 must
    raise the predicted size substantially, proportional to X's param share."""
    counts = _moe_param_counts()
    scorer = PredictiveScorer(
        sensitivity_weights={},
        parameter_counts=counts,
        baseline_size_gb=10.0,
        baseline_tps=100.0,
    )
    groups = list(counts.keys())

    cfg_light = {g: "MXFP4_MOE" for g in groups}   # X at 4.25 bpw
    cfg_heavy = dict(cfg_light)
    cfg_heavy["X"] = "Q6_K"                          # X at 6.5625 bpw

    size_light = scorer.predict_size(cfg_light)
    size_heavy = scorer.predict_size(cfg_heavy)

    assert size_heavy > size_light
    # X is ~80% of params; bumping it from 4.25 to 6.5625 bpw should move the
    # predicted size by a meaningful fraction (much more than the 0.05 default
    # would have allowed).
    assert (size_heavy - size_light) / size_light > 0.10


def test_predict_size_uses_real_counts_not_default():
    """A model whose X share differs from the 0.05 fallback must yield a
    different prediction when real counts are passed vs not."""
    counts = _moe_param_counts()
    groups = list(counts.keys())
    cfg = {g: "MXFP4_MOE" for g in groups}
    cfg["X"] = "Q8_0"

    with_counts = PredictiveScorer(
        sensitivity_weights={}, parameter_counts=counts,
        baseline_size_gb=10.0, baseline_tps=100.0,
    ).predict_size(cfg)

    without_counts = PredictiveScorer(
        sensitivity_weights={}, parameter_counts=None,
        baseline_size_gb=10.0, baseline_tps=100.0,
    ).predict_size(cfg)

    # With X dominating, Q8_0 on X should make the real-count prediction
    # noticeably larger than the fallback (which spreads X at only 0.05).
    assert with_counts != without_counts


def test_moe_fallback_distribution_weights_experts():
    """When no counts are passed but X is in the group list, the fallback
    distribution should put most mass on X."""
    scorer = PredictiveScorer(
        sensitivity_weights={}, parameter_counts=None,
        baseline_size_gb=10.0, baseline_tps=100.0,
    )
    dist = scorer._compute_param_dist(["E", "H", "Q", "K", "O", "U", "D", "X", "R"])
    assert dist["X"] > 0.4
    assert dist["X"] == max(dist.values())
