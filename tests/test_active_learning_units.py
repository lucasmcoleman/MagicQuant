"""predict_loss must always speak noise units, measured or not.

The defect this pins (found 2026-08-14, confirmed by tracing):

    orchestrator  residual = measured_loss - predicted_loss
                  ...measured_loss is (ppl-baseline)/baseline, a RELATIVE
                  FRACTION (~0.005); predict_loss is a sum of
                  sensitivity_weight x noise_factor, NOISE UNITS (~2.0)
    predictor     residual_cache[key] = residual
                  predict_loss: total_loss += residual_cache[key]
                  -> predicted + (measured - predicted) == measured,
                     in the WRONG UNITS

Downstream, score_hybrid's `loss_score = 1 - predicted_loss / 5.0` mapped
~0.005 to 0.999 and ~2.0 to 0.60, and _tournament_selection sorts every
candidate in a tier on the resulting composite and keeps the top 3. A measured
config therefore carried ~+0.20 composite -- on a 0-1 scale whose entire
precision term spans 0.50 -- purely from the unit mismatch. It won its tier
almost regardless of merit, and the next generation was built from its
mutants, so exploration collapsed after round 1.

The fix converts the measurement INTO noise units via a scale fitted from the
(predicted, measured) pairs, so the residual is a real calibration correction
and the two populations stay comparable.
"""

import pytest

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor


def _scorer():
    return PredictiveScorer(
        {"E": 0.4, "U": 0.3, "D": 0.3},
        parameter_counts={"E": 1000, "U": 1000, "D": 1000},
        baseline_size_gb=10.0,
    )


CFG_A = {"E": "Q8_0", "U": "Q4_K_M", "D": "Q4_K_M"}
CFG_B = {"E": "Q6_K", "U": "Q3_K", "D": "Q3_K"}
CFG_C = {"E": "Q5_K", "U": "MXFP4_MOE", "D": "MXFP4_MOE"}


# ── the invariant ───────────────────────────────────────────────────────────

def test_predict_loss_stays_in_noise_units_after_measurement():
    """The core invariant. A measured config's predicted_loss must remain on
    the same scale as an unmeasured one -- it must NOT become the raw
    relative-PPL measurement."""
    s = _scorer()
    unmeasured = s.predict_loss(CFG_B)

    # Two measurements so a scale can be fitted, with realistic magnitudes:
    # relative-PPL fractions, ~400x smaller than the noise-unit predictions.
    s.record_measurement(CFG_A, 0.0032)
    s.record_measurement(CFG_C, 0.0051)

    measured = s.predict_loss(CFG_A)
    # Still noise-scale, not fraction-scale.
    assert measured > 0.1, (
        f"predict_loss returned {measured} for a measured config -- that is "
        "relative-PPL scale, not noise units. The residual is mixing units."
    )
    assert 0.2 * unmeasured < measured < 5 * unmeasured, (
        "a measured config must stay within the same order of magnitude as an "
        "unmeasured one; anything else re-introduces the tournament artifact"
    )


def test_no_correction_until_a_scale_can_be_fitted():
    """One pair cannot fit a slope. An unfitted scale means NO correction --
    'not enough signal' is the honest answer, never a guess."""
    s = _scorer()
    before = s.predict_loss(CFG_A)
    s.record_measurement(CFG_A, 0.0032)
    assert s._loss_scale is None
    assert s.residual_for(CFG_A) is None
    assert s.predict_loss(CFG_A) == pytest.approx(before)


def test_nonpositive_measurements_do_not_fit_the_scale():
    """A measurement at or below baseline is inside the noise floor and
    carries no calibration signal, so it must not steer the fit."""
    s = _scorer()
    s.record_measurement(CFG_A, -0.0041)   # below baseline
    s.record_measurement(CFG_B, 0.0)       # exactly baseline
    assert s._loss_scale is None, "non-positive measurements must not fit a scale"


def test_scale_is_fitted_and_residual_is_a_real_correction():
    s = _scorer()
    s.record_measurement(CFG_A, 0.0032)
    s.record_measurement(CFG_C, 0.0051)
    assert s._loss_scale is not None and s._loss_scale > 0
    # The residual is a deviation from the fitted line, not the whole
    # prediction -- so it must be SMALL relative to the prediction itself.
    raw = s._predict_loss_uncorrected(CFG_A)
    resid = s.residual_for(CFG_A)
    assert abs(resid) < abs(raw), (
        "a residual as large as the prediction means the correction is "
        "swallowing the model, which is the old unit-mismatch behaviour"
    )


# ── the consequence the defect actually had ─────────────────────────────────

def test_measured_config_does_not_get_a_free_composite_bonus():
    """The tournament reads composite_score and keeps the top 3 per tier,
    mixing measured and unmeasured candidates. Under the defect a measured
    config gained ~+0.20 composite from units alone. Here the same config's
    composite must barely move on being measured -- a measurement should
    refine a score, not transform its scale."""
    s = _scorer()
    before = s.score_hybrid(CFG_A)["composite_score"]

    s.record_measurement(CFG_A, 0.0032)
    s.record_measurement(CFG_C, 0.0051)
    after = s.score_hybrid(CFG_A)["composite_score"]

    assert abs(after - before) < 0.10, (
        f"composite moved {abs(after - before):.3f} on measurement alone. The "
        "unit mismatch gave ~+0.20 here, which let any measured config win its "
        "tier regardless of merit."
    )


def test_tournament_does_not_systematically_prefer_measured_candidates():
    """End to end through the real selector: a measured MEDIOCRE config must
    not beat an unmeasured GOOD one."""
    s = _scorer()
    good, mediocre = CFG_A, CFG_B          # CFG_A is the higher-precision mix
    s.record_measurement(mediocre, 0.0049)
    s.record_measurement(CFG_C, 0.0051)

    surv = EvolutionarySurvivor(predictor=s, baseline_config={"E": "BF16"})
    candidates = [
        {"config": mediocre, "composite_score": s.score_hybrid(mediocre)["composite_score"]},
        {"config": good, "composite_score": s.score_hybrid(good)["composite_score"]},
    ]
    winners = surv._tournament_selection({"Q4": candidates})
    assert winners[0]["config"] == good, (
        "the measured-but-worse config won -- the unit artifact is back"
    )


def test_calibration_uses_the_uncorrected_prediction():
    """The fit must be against the model's raw output. Fitting against the
    corrected value would feed the correction back into its own calibration."""
    s = _scorer()
    raw_before = s._predict_loss_uncorrected(CFG_A)
    s.record_measurement(CFG_A, 0.0032)
    s.record_measurement(CFG_C, 0.0051)
    stored_pred, stored_meas = s._measurement_pairs[s._make_config_key(CFG_A)]
    assert stored_pred == pytest.approx(raw_before)
    assert stored_meas == pytest.approx(0.0032)


# ── item 1: ambiguous source directories fail closed ────────────────────────

def test_open_model_source_refuses_a_directory_with_several_ggufs(tmp_path):
    """It used to take gguf_files[0] from an UNORDERED os.listdir. Observed
    2026-08-13: a run directory holding model-bf16.gguf (417 tensors) beside
    model-bf16-nomtp.gguf (401, MTP removed) resolved to the MTP-FREE variant.
    Quantizing the wrong model is undetectable downstream -- the file is
    valid, the tensor count plausible, the artifact ships."""
    from magicquant.gguf.source import open_model_source

    for n in ("model-bf16.gguf", "model-bf16-nomtp.gguf"):
        (tmp_path / n).write_bytes(b"GGUF" + b"\0" * 32)

    with pytest.raises(ValueError) as exc:
        open_model_source(str(tmp_path))
    msg = str(exc.value)
    # Both candidates must be named -- the caller knows which they meant.
    assert "model-bf16.gguf" in msg and "model-bf16-nomtp.gguf" in msg
    assert "llama-gguf-split --merge" in msg, "should point at the shard case too"


def test_open_model_source_still_accepts_a_single_gguf_directory(tmp_path):
    from magicquant.gguf.source import open_model_source
    (tmp_path / "only.gguf").write_bytes(b"GGUF" + b"\0" * 32)
    # Exactly one candidate is unambiguous, so this must resolve. GGUFSource
    # construction is lazy (it does not parse until opened), so a successful
    # return here is the correct outcome -- the guard must not fire.
    src = open_model_source(str(tmp_path))
    assert src is not None


# ── item 2: the distortion cache key ────────────────────────────────────────

def test_cache_key_ignores_the_model_PATH_but_not_its_CONTENT(tmp_path):
    """Path-keying meant a directory rename cost a full ~90 min recompute for
    a file whose bytes never changed."""
    from magicquant.v2.sensitivity import _model_identity

    a, b = tmp_path / "before.gguf", tmp_path / "after.gguf"
    payload = b"GGUF" + bytes(range(256)) * 8
    a.write_bytes(payload)
    b.write_bytes(payload)

    ida, idb = _model_identity(a), _model_identity(b)
    assert "path" not in ida, "the absolute path must not be part of identity"
    assert ida["head_sha256"] == idb["head_sha256"]
    assert ida["size"] == idb["size"]

    other = tmp_path / "different.gguf"
    other.write_bytes(b"GGUF" + bytes(range(255, -1, -1)) * 8)
    assert _model_identity(other)["head_sha256"] != ida["head_sha256"], (
        "size+mtime alone would collide here; the content hash is what "
        "separates two different models of equal size"
    )


def test_cache_key_carries_a_resolution_version():
    """TABLE_VERSION covers the table's SCHEMA. The table's CONTENTS are
    encode->decode error at each tensor's RESOLVED type, so a resolution-logic
    change invalidates them while the key still matches -- which is exactly
    what the 2026-08-13 weight-suffix gate did to ssm_a/ssm_d."""
    from magicquant.v2 import sensitivity as sens

    assert isinstance(sens.RESOLUTION_VERSION, int)
    k1 = sens._cache_key({"size": 1}, ["Q8_0"], {}, None)
    old = sens.RESOLUTION_VERSION
    try:
        sens.RESOLUTION_VERSION = old + 1
        k2 = sens._cache_key({"size": 1}, ["Q8_0"], {}, None)
    finally:
        sens.RESOLUTION_VERSION = old
    assert k1 != k2, "bumping RESOLUTION_VERSION must invalidate the cache"
