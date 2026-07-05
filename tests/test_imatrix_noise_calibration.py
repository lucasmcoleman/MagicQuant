"""Imatrix-aware noise factor tests.

Covers `magicquant.quant.schemes.effective_noise_factor` (the pure scaling
function) and `PredictiveScorer`'s `imatrix_active` threading (the consumer
that actually changes predicted loss during search).

Ground truth motivating this: the completed Qwopus3.6-27B measured search
(see tests/fixtures/qwopus_search_results_2026-07-04/) showed MXFP4's ggml
encoder ignores an active imatrix (byte-identical output, verified
2026-07-04) while k_quant/iq_quant encoders consume it -- so a search with
an imatrix active should discount noise for the latter, not the former.
"""
import pytest

from magicquant.quant.schemes import (
    IMATRIX_NOISE_SCALE,
    BF16,
    IQ4_NL,
    MXFP4_MOE,
    Q4_K_M,
    Q8_0,
    ROCMFP4,
    effective_noise_factor,
    get_scheme_by_name,
)
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.quant import calibration


@pytest.fixture(autouse=True)
def _no_calibration_file(tmp_path, monkeypatch):
    """Force `calibration.calibrated_noise_factor` to always miss, so these
    tests exercise the registry/effective_noise_factor path deterministically
    regardless of whether a real tools/calibration_results.json happens to
    exist on disk or what a previous test left cached."""
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", missing)
    calibration._reset_cache()
    yield
    calibration._reset_cache()


# ── effective_noise_factor ───────────────────────────────────────────

def test_imatrix_inactive_returns_static_noise_factor():
    """imatrix_active=False must return the scheme's noise_factor untouched,
    regardless of whether the scheme consumes an imatrix."""
    assert effective_noise_factor(Q4_K_M, imatrix_active=False) == Q4_K_M.noise_factor
    assert effective_noise_factor(MXFP4_MOE, imatrix_active=False) == MXFP4_MOE.noise_factor


def test_imatrix_active_discounts_consuming_scheme():
    """A k_quant/iq_quant scheme (uses_imatrix=True) gets the discount when
    an imatrix is active."""
    assert Q4_K_M.uses_imatrix is True
    expected = Q4_K_M.noise_factor * IMATRIX_NOISE_SCALE
    assert effective_noise_factor(Q4_K_M, imatrix_active=True) == pytest.approx(expected)
    assert effective_noise_factor(Q4_K_M, imatrix_active=True) < Q4_K_M.noise_factor


def test_imatrix_active_does_not_discount_non_consuming_scheme():
    """MXFP4/rocmfpx/float/legacy_q schemes (uses_imatrix=False) are
    unaffected by an active imatrix -- their ggml encoders ignore it."""
    for scheme in (MXFP4_MOE, ROCMFP4, BF16, Q8_0):
        assert scheme.uses_imatrix is False, scheme.name
        assert effective_noise_factor(scheme, imatrix_active=True) == scheme.noise_factor


def test_base_noise_factor_override_still_gated_by_uses_imatrix():
    """A caller-supplied base_noise_factor (e.g. a calibrated value) is
    scaled the same way a registry value would be -- gated on uses_imatrix,
    not on which value it's scaling."""
    calibrated = 2.0
    assert effective_noise_factor(
        Q4_K_M, imatrix_active=True, base_noise_factor=calibrated
    ) == pytest.approx(calibrated * IMATRIX_NOISE_SCALE)
    assert effective_noise_factor(
        MXFP4_MOE, imatrix_active=True, base_noise_factor=calibrated
    ) == calibrated


def test_every_k_quant_and_iq_quant_scheme_uses_imatrix():
    from magicquant.quant.schemes import get_schemes_by_category
    for scheme in get_schemes_by_category("k_quant") + get_schemes_by_category("iq_quant"):
        assert scheme.uses_imatrix is True, f"{scheme.name} should consume an imatrix"


def test_every_mxfp4_rocmfpx_float_scheme_does_not_use_imatrix():
    from magicquant.quant.schemes import get_schemes_by_category
    for category in ("mxfp4", "rocmfpx", "float"):
        for scheme in get_schemes_by_category(category):
            assert scheme.uses_imatrix is False, f"{scheme.name} should not consume an imatrix"


def test_legacy_q_q8_0_does_not_use_imatrix():
    """Q8_0's ggml quantize function explicitly discards quant_weights --
    verified against ggml/src/ggml-quants.c (`(void)quant_weights;`)."""
    assert Q8_0.uses_imatrix is False


# ── PredictiveScorer.imatrix_active ──────────────────────────────────

def _scorer(imatrix_active: bool) -> PredictiveScorer:
    return PredictiveScorer(
        sensitivity_weights={"Q": 1.0, "U": 1.0},
        imatrix_active=imatrix_active,
    )


def test_predictive_scorer_defaults_imatrix_inactive():
    scorer = PredictiveScorer(sensitivity_weights={})
    assert scorer.imatrix_active is False


def test_imatrix_active_lowers_predicted_loss_for_consuming_config():
    """A config using only imatrix-consuming schemes must predict a lower
    loss with imatrix_active=True than with it False."""
    config = {"Q": "Q4_K_M", "U": "IQ4_NL"}
    off = _scorer(False).predict_loss(config)
    on = _scorer(True).predict_loss(config)
    assert on < off


def test_imatrix_active_does_not_change_predicted_loss_for_mxfp4_only_config():
    """A config using only non-consuming schemes (MXFP4) must predict the
    SAME loss whether or not imatrix is active."""
    config = {"Q": "MXFP4_MOE", "U": "MXFP4_MOE"}
    off = _scorer(False).predict_loss(config)
    on = _scorer(True).predict_loss(config)
    assert on == pytest.approx(off)


def test_imatrix_active_partial_config_only_discounts_consuming_groups():
    """Mixed config: only the imatrix-consuming group's contribution should
    shrink; the MXFP4 group's contribution is identical either way."""
    scorer_off = _scorer(False)
    scorer_on = _scorer(True)
    config = {"Q": "Q4_K_M", "U": "MXFP4_MOE"}

    off = scorer_off.predict_loss(config)
    on = scorer_on.predict_loss(config)
    assert on < off

    # The delta must equal exactly the Q group's discount (sens_weight=1.0
    # for both groups here, so no cross-group scaling to account for).
    q_noise = get_scheme_by_name("Q4_K_M").noise_factor
    expected_delta = q_noise - q_noise * IMATRIX_NOISE_SCALE
    assert (off - on) == pytest.approx(expected_delta)


def test_unknown_scheme_falls_back_unscaled_regardless_of_imatrix():
    scorer_off = _scorer(False)
    scorer_on = _scorer(True)
    config = {"Q": "NOT_A_REAL_SCHEME"}
    assert scorer_off.predict_loss(config) == scorer_on.predict_loss(config)
