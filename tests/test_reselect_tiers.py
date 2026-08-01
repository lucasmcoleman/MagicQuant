"""Retroactive tier re-selection over a finished run's measurements.

``tools/reselect_tiers.py`` re-derives a run's ladder from the candidates it
already measured, so a selection bug can be corrected without re-measuring.
It is the instrument that decides which published artifacts are wrong, so the
three judgements it makes -- baseline recovery, band assignment, and which
candidates are usable at all -- are pinned here.

The regression it exists to catch: v1's ``Q5`` band ``(0.33, 0.45]`` held both
uniform Q5_K (ratio 0.3441) and uniform Q6_K (0.4102), and v1 picked within a
band by loss alone, so Q6_K won every Q5 slot on every run.
"""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "reselect_tiers.py"
_spec = importlib.util.spec_from_file_location("reselect_tiers", TOOL_PATH)
reselect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reselect)


GROUPS = ["D", "E", "H", "K", "O", "Q", "R", "U", "X"]

# Fractions of a notional BF16 model's weights per group. Arbitrary but
# fixed, so synthetic sizes are exactly reproducible.
GROUP_FRACTION = {
    "D": 0.30, "E": 0.05, "H": 0.05, "K": 0.05, "O": 0.10,
    "Q": 0.10, "R": 0.01, "U": 0.24, "X": 0.10,
}
BASELINE_GB = 100.0
# Tensors no scheme touches (norms etc.) -- the fit's constant term.
FIXED_OVERHEAD_GB = 0.5


def _size_of(config):
    """Exact size the linear model should recover for a config."""
    total = sum(
        GROUP_FRACTION[g] * BASELINE_GB * reselect._bpw(scheme) / 16.0
        for g, scheme in config.items()
    )
    return total + FIXED_OVERHEAD_GB


def _key(config):
    return "|".join(f"{g}:{config[g]}" for g in sorted(config))


def _uniform(scheme):
    return {g: scheme for g in GROUPS}


def _run(entries, baseline_ppl=10.0):
    """Build a search_results.json payload from (config, ppl) pairs."""
    measurements = {}
    for config, ppl in entries:
        measurements[_key(config)] = {
            "config": config,
            "ppl": ppl,
            "measured_loss": (ppl - baseline_ppl) / baseline_ppl,
            "size_gb": _size_of(config),
        }
    return {"baseline_ppl": baseline_ppl, "measurements": measurements}


def _write(tmp_path, payload):
    path = tmp_path / "search_results.json"
    path.write_text(json.dumps(payload))
    return path


# A ladder mirroring the real runs: uniform builds at each precision, where
# quality improves monotonically with size.
LADDER = [
    (_uniform("Q4_K_M"), 10.25),
    (_uniform("Q5_K"), 10.14),
    (_uniform("Q6_K"), 10.05),
    (_uniform("Q8_0"), 10.02),
]

# Recovering 9 group sizes plus a constant needs at least 10 distinct
# configs, so pad the ladder with single-group perturbations. They are all
# given clearly worse perplexity than any uniform build, so whichever band
# each lands in, it never displaces that band's rightful winner.
_FILLERS = [
    {**_uniform("Q4_K_M"), "X": "Q8_0"},
    {**_uniform("Q4_K_M"), "D": "BF16"},
    {**_uniform("Q5_K"), "U": "Q8_0"},
    {**_uniform("Q5_K"), "E": "Q4_K_M"},
    {**_uniform("Q6_K"), "K": "Q4_K_M"},
    {**_uniform("Q6_K"), "O": "BF16"},
    {**_uniform("Q8_0"), "Q": "Q4_K_M"},
    {**_uniform("Q8_0"), "H": "Q5_K"},
]


def _full_ladder():
    """The uniform ladder plus enough distinct fillers to fit the size model."""
    return LADDER + [(config, 10.90 + i * 0.01) for i, config in enumerate(_FILLERS)]


def test_recovers_baseline_size_exactly():
    """Per-group sizes are solvable from enough candidates, so BF16 is too."""
    candidates = [(_size_of(c), c) for c, _ in LADDER] + [
        (_size_of(c), c)
        for c in (
            {**_uniform("Q4_K_M"), "X": "Q8_0"},
            {**_uniform("Q6_K"), "D": "Q4_K_M"},
            {**_uniform("Q5_K"), "U": "BF16"},
            {**_uniform("Q8_0"), "E": "Q4_K_M"},
            {**_uniform("Q4_K_M"), "O": "BF16", "K": "Q6_K"},
            {**_uniform("Q6_K"), "Q": "Q4_K_M", "H": "BF16"},
        )
    ]
    baseline, residual = reselect.recover_baseline_gb(candidates)
    assert residual < 1e-9
    assert baseline == pytest.approx(BASELINE_GB + FIXED_OVERHEAD_GB, rel=1e-9)


def test_underdetermined_system_is_rejected():
    """Fewer measurements than unknowns must fail loudly, not guess."""
    candidates = [(_size_of(c), c) for c, _ in LADDER]
    with pytest.raises(reselect.BaselineFitError, match="cannot determine"):
        reselect.recover_baseline_gb(candidates)


def test_inconsistent_sizes_are_rejected():
    """A size that no per-group model explains means the run is untrustworthy."""
    candidates = [(_size_of(c), c) for c, _ in LADDER] * 3
    candidates[0] = (candidates[0][0] * 1.5, candidates[0][1])
    with pytest.raises(reselect.BaselineFitError, match="fits poorly"):
        reselect.recover_baseline_gb(candidates)


def test_uniform_q5k_and_q6k_land_in_distinct_bands(tmp_path):
    """The v1 collision: both used to be 'Q5', and Q6_K always won it."""
    payload = _run(_full_ladder())
    report = reselect.analyze(_write(tmp_path, payload), max_residual=1e-6)
    assert report["error"] is None

    by_size = {round(r["size_gb"], 4): r["tier"] for r in report["candidates"]}
    assert by_size[round(_size_of(_uniform("Q5_K")), 4)] == "Q5"
    assert by_size[round(_size_of(_uniform("Q6_K")), 4)] == "Q6"

    # Each band's winner is the uniform build that band is named for.
    assert report["corrected"]["Q5"]["config"] == _uniform("Q5_K")
    assert report["corrected"]["Q6"]["config"] == _uniform("Q6_K")
    assert report["corrected"]["Q4"]["config"] == _uniform("Q4_K_M")


def test_shipping_a_q6k_build_as_q5_is_flagged(tmp_path):
    """Exactly what every pre-v2 run did."""
    payload = _run(_full_ladder())
    q6k = payload["measurements"][_key(_uniform("Q6_K"))]
    payload["tiered"] = {"Q5": dict(q6k)}
    report = reselect.analyze(_write(tmp_path, payload), max_residual=1e-6)

    (shipped,) = report["shipped"]
    assert shipped["shipped_as"] == "Q5"
    assert shipped["actual_tier"] == "Q6"
    assert "MISLABELED" in shipped["flags"]


def test_bigger_and_worse_is_flagged_dominated(tmp_path):
    """A tier no one should ever ship: beaten on size *and* quality."""
    payload = _run(_full_ladder())
    # A large build that measures worse than the smaller Q6_K.
    bloated = {**_uniform("Q8_0"), "D": "BF16"}
    payload["measurements"][_key(bloated)] = {
        "config": bloated,
        "ppl": 10.40,
        "measured_loss": 0.040,
        "size_gb": _size_of(bloated),
    }
    payload["tiered"] = {"Q8": payload["measurements"][_key(bloated)]}
    report = reselect.analyze(_write(tmp_path, payload), max_residual=1e-6)

    (shipped,) = report["shipped"]
    assert "DOMINATED" in shipped["flags"]
    assert shipped["dominated_by_size_gb"] < shipped["size_gb"]
    assert shipped["dominated_by_loss"] < shipped["loss"]


def test_sub_baseline_perplexity_is_discarded(tmp_path):
    """The pre-a6f8dd0 parser could log a timing as a PPL; such an entry
    would then win its band via ``min()``. It must not survive selection."""
    payload = _run(_full_ladder())
    bogus = {**_uniform("Q8_0"), "X": "BF16"}
    payload["measurements"][_key(bogus)] = {
        "config": bogus,
        "ppl": 2.70,                # baseline is 10.0
        "measured_loss": -0.73,
        "size_gb": _size_of(bogus),
    }
    payload["tiered"] = {"Q8": payload["measurements"][_key(bogus)]}
    report = reselect.analyze(_write(tmp_path, payload), max_residual=1e-6)

    assert report["n_implausible"] == 1
    assert "IMPLAUSIBLE" in report["shipped"][0]["flags"]
    # It is excluded from the ladder rather than winning the Q8 band.
    assert report["corrected"]["Q8"]["config"] != bogus
    assert report["corrected"]["Q8"]["loss"] > 0
    # ...and never counts as a frontier point.
    assert all(not r["pareto"] for r in report["candidates"] if r["implausible"])


def test_noise_sized_dip_below_baseline_is_kept(tmp_path):
    """Real runs jitter a little under baseline; only large dips are broken."""
    payload = _run(_full_ladder())
    jittery = {**_uniform("Q8_0"), "R": "BF16"}
    payload["measurements"][_key(jittery)] = {
        "config": jittery,
        "ppl": 9.95,               # -0.5%, well inside the 5% tolerance
        "measured_loss": -0.005,
        "size_gb": _size_of(jittery),
    }
    report = reselect.analyze(_write(tmp_path, payload), max_residual=1e-6)
    assert report["n_implausible"] == 0


def test_band_with_no_candidate_is_reported_empty(tmp_path):
    """Nothing to ship there without a new pack + measure."""
    payload = _run(_full_ladder())
    report = reselect.analyze(_write(tmp_path, payload), max_residual=1e-6)
    assert "Q2" in report["empty_bands"]
    assert "Q5" not in report["empty_bands"]


def test_missing_measurements_reported_not_raised(tmp_path):
    report = reselect.analyze(_write(tmp_path, {"measurements": {}}), max_residual=1e-6)
    assert report["error"] == "no usable measurements"
