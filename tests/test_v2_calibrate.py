"""kappa fitting: slice-baseline use, censoring of below-resolution probes,
pseudo-key exclusion."""

from magicquant.v2.calibrate import CENSOR_FRAC, fit_kappa
from magicquant.v2.outcome import MeasurementOutcome as MO


def test_censoring_floors_below_resolution_groups():
    outcomes = {
        "__slice_baseline__": MO.success(10.0),
        "D": MO.success(10.6),   # rel 0.06
        "O": MO.success(10.2),   # rel 0.02
        "E": MO.success(9.99),   # rel -0.001 -> below resolution
    }
    eps = {"D": 30.0, "O": 2.0, "E": 600.0}
    kappa, prov = fit_kappa(outcomes, eps, baseline_ppl=999.0)
    assert "__slice_baseline__" not in kappa
    assert prov == {"D": "measured", "O": "measured", "E": "measured-censored"}
    # floor = CENSOR_FRAC * median(positive rels [0.02, 0.06]) = 0.25 * 0.06
    assert abs(kappa["E"] - (CENSOR_FRAC * 0.06) / 600.0) < 1e-12
    # cleanly measured groups are untouched
    assert abs(kappa["D"] - 0.06 / 30.0) < 1e-12
    assert abs(kappa["O"] - 0.02 / 2.0) < 1e-12


def test_slice_baseline_preferred_over_full_baseline():
    outcomes = {
        "__slice_baseline__": MO.success(20.0),  # capped-slice baseline
        "D": MO.success(21.0),                   # rel vs slice = 0.05
    }
    kappa, prov = fit_kappa(outcomes, {"D": 10.0}, baseline_ppl=10.0)
    assert abs(kappa["D"] - 0.05 / 10.0) < 1e-12


def test_failed_probe_imputes_median_when_allowed():
    outcomes = {
        "__slice_baseline__": MO.success(10.0),
        "D": MO.success(10.6),
        "K": MO.failure("boom"),
    }
    kappa, prov = fit_kappa(outcomes, {"D": 30.0, "K": 5.0}, baseline_ppl=10.0)
    assert prov["K"] == "imputed-median"
    assert kappa["K"] == kappa["D"]


def test_no_allocatable_mass_group():
    outcomes = {"__slice_baseline__": MO.success(10.0), "N": MO.success(10.1)}
    kappa, prov = fit_kappa(outcomes, {"N": 0.0}, baseline_ppl=10.0)
    assert kappa["N"] == 0.0
    assert prov["N"] == "no-allocatable-mass"
