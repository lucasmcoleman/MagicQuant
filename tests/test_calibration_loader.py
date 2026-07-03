"""Calibration loader tests.

`magicquant/quant/calibration.py` optionally overrides the static
noise_factor registry (`magicquant/quant/schemes.py`) with empirically
measured values from `tools/calibration_results.json`, when that file
exists. These tests verify:

  - Absent file -> `calibrated_noise_factor` returns None, and the predictor
    falls back to the registry value exactly as before this feature existed.
  - Present (valid) file -> the measured value is used, both directly and
    via `PredictiveScorer._noise_factor_for`.
  - Malformed file -> `calibrated_noise_factor` returns None, no exception.
"""
import json

from magicquant.quant import calibration
from magicquant.quant.schemes import get_scheme_by_name
from magicquant.evolution.predictor import PredictiveScorer


def test_no_calibration_file_returns_none(tmp_path, monkeypatch):
    # Point the loader at a path that doesn't exist, regardless of whether
    # a real tools/calibration_results.json happens to be present on disk.
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", missing)
    calibration._reset_cache()

    assert calibration.calibrated_noise_factor("Q8_0") is None

    scorer = PredictiveScorer(sensitivity_weights={})
    registry_value = get_scheme_by_name("Q8_0").noise_factor
    assert scorer._noise_factor_for("Q8_0") == registry_value


def test_calibration_file_overrides_registry(tmp_path, monkeypatch):
    calib_path = tmp_path / "calibration_results.json"
    calib_path.write_text(json.dumps({
        "Q8_0": {"ppl": 5.5, "ppl_loss": 0.5, "noise_factor": 1.0},
        "Q4_K_M": {"ppl": 7.1, "ppl_loss": 2.1, "noise_factor": 4.6},
    }))
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", calib_path)
    calibration._reset_cache()

    assert calibration.calibrated_noise_factor("Q4_K_M") == 4.6

    scorer = PredictiveScorer(sensitivity_weights={})
    assert scorer._noise_factor_for("Q4_K_M") == 4.6
    # Sanity: this differs from the static registry value, proving the
    # calibrated value actually won out rather than coincidentally matching.
    assert get_scheme_by_name("Q4_K_M").noise_factor != 4.6


def test_calibration_file_missing_scheme_returns_none(tmp_path, monkeypatch):
    calib_path = tmp_path / "calibration_results.json"
    calib_path.write_text(json.dumps({
        "Q8_0": {"ppl": 5.5, "ppl_loss": 0.5, "noise_factor": 1.0},
    }))
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", calib_path)
    calibration._reset_cache()

    assert calibration.calibrated_noise_factor("Q2_K") is None


def test_malformed_json_returns_none(tmp_path, monkeypatch):
    calib_path = tmp_path / "calibration_results.json"
    calib_path.write_text("{not valid json,,,")
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", calib_path)
    calibration._reset_cache()

    assert calibration.calibrated_noise_factor("Q8_0") is None


def test_calibration_file_nested_schemes_envelope_is_read(tmp_path, monkeypatch):
    """Round-trip against the EXACT shape `tools/calibrate_noise_factors.py`
    writes: {"model":..., "corpus":..., "date":..., "baseline_ppl":...,
    "schemes": {name: {...}}} — not the flat fixture shape used elsewhere
    in this file. Regression for F2: the loader must unwrap "schemes"."""
    calib_path = tmp_path / "calibration_results.json"
    calib_path.write_text(json.dumps({
        "model": "some/model",
        "corpus": "wikitext-2",
        "date": "2026-07-01",
        "baseline_ppl": 5.2,
        "schemes": {
            "Q8_0": {"ppl": 5.5, "ppl_loss": 0.5, "noise_factor": 1.0, "status": "ok"},
            "Q4_K_M": {"ppl": 7.1, "ppl_loss": 2.1, "noise_factor": 4.6, "status": "ok"},
            "BF16": {"ppl": 5.2, "ppl_loss": 0.0, "noise_factor": 0.0, "status": "baseline"},
        },
    }))
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", calib_path)
    calibration._reset_cache()

    assert calibration.calibrated_noise_factor("Q4_K_M") == 4.6
    assert calibration.calibrated_noise_factor("Q8_0") == 1.0

    scorer = PredictiveScorer(sensitivity_weights={})
    assert scorer._noise_factor_for("Q4_K_M") == 4.6
    assert get_scheme_by_name("Q4_K_M").noise_factor != 4.6


def test_cache_is_populated_after_first_load(tmp_path, monkeypatch):
    calib_path = tmp_path / "calibration_results.json"
    calib_path.write_text(json.dumps({
        "Q8_0": {"ppl": 5.5, "ppl_loss": 0.5, "noise_factor": 1.0},
    }))
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", calib_path)
    calibration._reset_cache()

    assert calibration.calibrated_noise_factor("Q8_0") == 1.0

    # Delete the file; the cached value should still be served without
    # re-reading from disk.
    calib_path.unlink()
    assert calibration.calibrated_noise_factor("Q8_0") == 1.0

    calibration._reset_cache()
