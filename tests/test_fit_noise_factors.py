"""Tests for tools/fit_noise_factors.py.

Round-trips the least-squares fit against a synthetic search_results.json
built from KNOWN ground-truth noise factors (so the fit should recover them
closely), then smoke-tests the same code path against the real measured
search_results.json from the completed Qwopus3.6-27B run (checked into
tests/fixtures/qwopus_search_results_2026-07-04/) to confirm it runs
end-to-end on real data without raising.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.fit_noise_factors import (  # noqa: E402
    build_calibration_envelope,
    fit_noise_factors,
    load_fit_inputs,
)

REAL_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "qwopus_search_results_2026-07-04"


def _write_search_results(tmp_path, configs_and_losses, sensitivity_weights):
    """Write a minimal search_results.json + sibling sensitivity.json."""
    measurements = {
        f"cfg{i}": {"config": cfg, "measured_loss": loss}
        for i, (cfg, loss) in enumerate(configs_and_losses)
    }
    (tmp_path / "search_results.json").write_text(json.dumps({
        "baseline_ppl": 5.0,
        "measurements": measurements,
    }))
    (tmp_path / "sensitivity.json").write_text(json.dumps({
        "normalized_weights": sensitivity_weights,
    }))
    return tmp_path / "search_results.json"


# ── Synthetic round-trip ──────────────────────────────────────────────

GROUND_TRUTH_NOISE = {"Q4_K_M": 4.5, "IQ4_NL": 3.8, "MXFP4_MOE": 4.0}
SENS_WEIGHTS = {"Q": 0.3, "U": 0.3, "D": 0.4}
COLLAPSE_BETA = 0.02


def _synthetic_measured_loss(config, sens_weights):
    """Compute a loss EXACTLY matching PredictiveScorer.predict_loss's
    formula (minus residual_cache, which is irrelevant here) from
    GROUND_TRUTH_NOISE, so the fit should recover those values."""
    total = sum(
        sens_weights.get(g, 1.0 / len(config)) * GROUND_TRUTH_NOISE.get(scheme, 0.0)
        for g, scheme in config.items()
    )
    compressed_sensitive = sum(
        1 for g, scheme in config.items()
        if g in ("E", "H", "O", "R") and scheme != "BF16"
    )
    if compressed_sensitive > 0:
        total += COLLAPSE_BETA * compressed_sensitive
    return total


def test_round_trip_recovers_known_noise_factors(tmp_path):
    configs = [
        {"Q": "Q4_K_M", "U": "IQ4_NL", "D": "MXFP4_MOE"},
        {"Q": "IQ4_NL", "U": "MXFP4_MOE", "D": "Q4_K_M"},
        {"Q": "MXFP4_MOE", "U": "Q4_K_M", "D": "IQ4_NL"},
        {"Q": "Q4_K_M", "U": "Q4_K_M", "D": "MXFP4_MOE"},
        {"Q": "IQ4_NL", "U": "IQ4_NL", "D": "IQ4_NL"},
        {"Q": "MXFP4_MOE", "U": "MXFP4_MOE", "D": "MXFP4_MOE"},
    ]
    configs_and_losses = [
        (cfg, _synthetic_measured_loss(cfg, SENS_WEIGHTS)) for cfg in configs
    ]
    path = _write_search_results(tmp_path, configs_and_losses, SENS_WEIGHTS)

    inputs, warnings = load_fit_inputs([path])
    assert warnings == []
    assert len(inputs) == len(configs)

    fitted = fit_noise_factors(inputs)
    assert set(fitted) == set(GROUND_TRUTH_NOISE)
    for scheme, truth in GROUND_TRUTH_NOISE.items():
        assert fitted[scheme]["noise_factor"] == pytest.approx(truth, abs=1e-3), (
            f"{scheme}: fitted {fitted[scheme]['noise_factor']} vs truth {truth}"
        )


def test_calibration_envelope_matches_loader_shape(tmp_path):
    """The output envelope must be exactly what
    magicquant.quant.calibration reads: {"schemes": {name: {"noise_factor":
    ...}}}."""
    from magicquant.quant import calibration

    configs_and_losses = [
        ({"Q": "Q4_K_M", "U": "MXFP4_MOE"}, _synthetic_measured_loss(
            {"Q": "Q4_K_M", "U": "MXFP4_MOE"}, SENS_WEIGHTS)),
        ({"Q": "MXFP4_MOE", "U": "Q4_K_M"}, _synthetic_measured_loss(
            {"Q": "MXFP4_MOE", "U": "Q4_K_M"}, SENS_WEIGHTS)),
    ]
    path = _write_search_results(tmp_path, configs_and_losses, SENS_WEIGHTS)
    inputs, _ = load_fit_inputs([path])
    fitted = fit_noise_factors(inputs)
    envelope = build_calibration_envelope(fitted, [str(path)])

    calib_path = tmp_path / "calibration_results.json"
    calib_path.write_text(json.dumps(envelope))
    calibration._reset_cache()
    try:
        import unittest.mock as mock
        with mock.patch.object(calibration, "_CALIBRATION_PATH", calib_path):
            calibration._reset_cache()
            assert calibration.calibrated_noise_factor("Q4_K_M") == fitted["Q4_K_M"]["noise_factor"]
            assert calibration.calibrated_noise_factor("BF16") == 0.0
    finally:
        calibration._reset_cache()


def test_missing_sibling_sensitivity_file_is_skipped_with_warning(tmp_path):
    (tmp_path / "search_results.json").write_text(json.dumps({
        "measurements": {
            "cfg0": {"config": {"Q": "Q4_K_M"}, "measured_loss": 0.1},
        },
    }))
    # No sibling sensitivity.json written.
    inputs, warnings = load_fit_inputs([tmp_path / "search_results.json"])
    assert inputs == []
    assert len(warnings) == 1
    assert "sensitivity.json" in warnings[0]


def test_empty_inputs_returns_empty_fit():
    assert fit_noise_factors([]) == {}


# ── Real-data smoke test ──────────────────────────────────────────────

def test_smoke_on_real_qwopus_search_results():
    """Must run end-to-end on the real measured search_results.json without
    raising, and produce a fitted value for every non-BF16 scheme that
    appears in the real data."""
    real_path = REAL_FIXTURE_DIR / "search_results.json"
    if not real_path.exists():
        pytest.skip(f"real fixture not present: {real_path}")

    inputs, warnings = load_fit_inputs([real_path])
    assert warnings == []
    assert len(inputs) == 7  # the 7 measured configs from that run

    fitted = fit_noise_factors(inputs)
    assert fitted  # non-empty
    for info in fitted.values():
        assert info["noise_factor"] >= 0.0
        assert info["n_observations"] >= 1

    # Sanity: schemes actually used in that run's configs are all present.
    real_data = json.loads(real_path.read_text())
    used_schemes = {
        scheme
        for entry in real_data["measurements"].values()
        for scheme in entry["config"].values()
        if scheme != "BF16"
    }
    assert used_schemes == set(fitted)
