"""Cross-run noise calibration tests (LANE B / PART 2).

Covers:
  - MagicQuantOrchestrator._write_noise_calibration: fits per-scheme noise
    factors from THIS run's measurements + sensitivity weights (reusing
    tools/fit_noise_factors.py's least-squares fit directly) and writes
    <output_dir>/noise_calibration.json in the nested envelope
    magicquant.quant.calibration reads. Opt-in via
    run_measured_search(write_calibration=True); off by default.
  - PredictiveScorer.calibration_source: an optional override path so a run
    can LOAD a specific calibration file instead of the fixed
    tools/calibration_results.json. "" (default) preserves the historical
    fixed-path lookup exactly.
  - Orchestrator wiring: run_measured_search/run_full_search forward
    calibration_source into the PredictiveScorer they construct, and
    write_calibration/calibration_source default to off/"" (unchanged
    historical behavior).

All new knobs are opt-in and default to False/"" -- required for the
seed-pinned refactor-regression fixture (tests/test_refactor_regression.py),
which is exercised (twice, as separate processes) alongside this suite.
"""
from pathlib import Path
import json

import pytest

import magicquant.gguf.source as source_mod
import magicquant.orchestrator as orch_mod
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.orchestrator import MagicQuantOrchestrator
from magicquant.quant import calibration


@pytest.fixture(autouse=True)
def _isolate_default_calibration_file(tmp_path, monkeypatch):
    """Force the DEFAULT calibration path (`_CALIBRATION_PATH`) to miss, so
    these tests are deterministic regardless of whether a real
    tools/calibration_results.json happens to exist on disk, and regardless
    of what a previous test left cached. `calibration_source` overrides are
    a SEPARATE cache (`_source_cache`) and are unaffected by this."""
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(calibration, "_CALIBRATION_PATH", missing)
    calibration._reset_cache()
    yield
    calibration._reset_cache()


# ── Synthetic ground truth (mirrors tests/test_fit_noise_factors.py) ────

GROUND_TRUTH_NOISE = {"Q4_K_M": 4.5, "IQ4_NL": 3.8, "MXFP4_MOE": 4.0}
SENS_WEIGHTS = {"Q": 0.3, "U": 0.3, "D": 0.4}
COLLAPSE_BETA = 0.02

SYNTHETIC_CONFIGS = [
    {"Q": "Q4_K_M", "U": "IQ4_NL", "D": "MXFP4_MOE"},
    {"Q": "IQ4_NL", "U": "MXFP4_MOE", "D": "Q4_K_M"},
    {"Q": "MXFP4_MOE", "U": "Q4_K_M", "D": "IQ4_NL"},
    {"Q": "Q4_K_M", "U": "Q4_K_M", "D": "MXFP4_MOE"},
    {"Q": "IQ4_NL", "U": "IQ4_NL", "D": "IQ4_NL"},
    {"Q": "MXFP4_MOE", "U": "MXFP4_MOE", "D": "MXFP4_MOE"},
]


def _synthetic_measured_loss(config, sens_weights=SENS_WEIGHTS):
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


def _make_bare_orchestrator(tmp_path):
    """A real MagicQuantOrchestrator, source path never touched (only
    output_dir needs to exist), with sensitivity_weights/_measured populated
    directly -- exercises _write_noise_calibration without a full
    run_measured_search."""
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    orch.sensitivity_weights = dict(SENS_WEIGHTS)
    orch._measured = {
        f"cfg{i}": {"config": cfg, "measured_loss": _synthetic_measured_loss(cfg)}
        for i, cfg in enumerate(SYNTHETIC_CONFIGS)
    }
    return orch


# ── _write_noise_calibration: round-trip ─────────────────────────────────


def test_write_noise_calibration_round_trip_recovers_known_noise_factors(tmp_path):
    orch = _make_bare_orchestrator(tmp_path)
    orch._write_noise_calibration()

    calib_path = orch.output_dir / "noise_calibration.json"
    assert calib_path.exists()

    envelope = json.loads(calib_path.read_text())
    assert set(envelope["schemes"]) >= set(GROUND_TRUTH_NOISE)

    for scheme, truth in GROUND_TRUTH_NOISE.items():
        loaded = calibration.calibrated_noise_factor(scheme, str(calib_path))
        assert loaded == pytest.approx(truth, abs=1e-3), (
            f"{scheme}: loaded {loaded} vs ground truth {truth}"
        )


def test_write_noise_calibration_matches_loader_shape(tmp_path):
    """The written envelope must be exactly what
    magicquant.quant.calibration reads: {"schemes": {name: {"noise_factor":
    ...}}}."""
    orch = _make_bare_orchestrator(tmp_path)
    orch._write_noise_calibration()
    calib_path = orch.output_dir / "noise_calibration.json"

    envelope = json.loads(calib_path.read_text())
    assert isinstance(envelope["schemes"], dict)
    for name, info in envelope["schemes"].items():
        if name == "BF16":
            continue
        assert "noise_factor" in info
        assert isinstance(info["noise_factor"], (int, float))


def test_write_noise_calibration_no_measurements_skips_gracefully(tmp_path):
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    orch.sensitivity_weights = {}
    orch._measured = {}

    orch._write_noise_calibration()  # must not raise

    assert not (orch.output_dir / "noise_calibration.json").exists()


def test_write_noise_calibration_survives_missing_repo_root_on_sys_path(
    tmp_path, monkeypatch
):
    """Regression: the Foundry pipeline stage imports `magicquant` via
    PYTHONPATH without the repo root on sys.path, so the plain
    `from tools.fit_noise_factors import ...` raised ModuleNotFoundError and
    write_calibration silently no-opped (run 3, 2026-07-06). The fallback
    must locate `tools/` next to the package and still write the file."""
    import sys

    repo_root = str(Path(orch_mod.__file__).resolve().parents[1])
    # Simulate the stage context: repo root absent, `tools` not yet imported,
    # and no PEP 660 editable finder mapping `tools` (Foundry's venv maps only
    # `magicquant`; this repo's own editable install maps both, which would
    # mask the bug).
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if str(Path(p or ".").resolve()) != repo_root]
    )
    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            f
            for f in sys.meta_path
            # PEP 660 finders may be registered as classes, so check the
            # object's own __module__/__name__, not type(f).
            if "editable" not in str(getattr(f, "__module__", "")).lower()
            and "editable" not in str(getattr(f, "__name__", "")).lower()
        ],
    )
    for mod in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
        monkeypatch.delitem(sys.modules, mod)

    orch = _make_bare_orchestrator(tmp_path)
    orch._write_noise_calibration()

    assert (orch.output_dir / "noise_calibration.json").exists()


def test_write_noise_calibration_excludes_measurement_invalid_entries(tmp_path):
    """MAJOR 3 regression: _write_noise_calibration used to build FitInput
    rows filtering only on ``measured_loss is not None``, so a
    measurement_invalid (physically-impossible, e.g. NaN-driven) reading
    got fitted into noise_calibration.json -- the same poisoning the
    measurement loop's active-learning feed already avoids for the
    predictor, made persistent here instead."""
    orch = _make_bare_orchestrator(tmp_path)
    # Poison the fit with an incident-shaped impossible reading.
    orch._measured["poisoned"] = {
        "config": {"Q": "Q4_K_M", "U": "IQ4_NL", "D": "MXFP4_MOE"},
        "measured_loss": -0.9225,
        "measurement_invalid": True,
    }
    orch._write_noise_calibration()

    calib_path = orch.output_dir / "noise_calibration.json"
    envelope = json.loads(calib_path.read_text())
    for scheme, truth in GROUND_TRUTH_NOISE.items():
        loaded = calibration.calibrated_noise_factor(scheme, str(calib_path))
        assert loaded == pytest.approx(truth, abs=1e-3), (
            f"{scheme}: fit was poisoned by a measurement_invalid entry -- "
            f"loaded {loaded} vs ground truth {truth}"
        )


def test_write_noise_calibration_ignores_entries_without_measured_loss(tmp_path):
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    orch.sensitivity_weights = dict(SENS_WEIGHTS)
    orch._measured = {
        "unmeasured": {"config": {"Q": "Q4_K_M"}, "measured_loss": None},
    }

    orch._write_noise_calibration()

    assert not (orch.output_dir / "noise_calibration.json").exists()


# ── PredictiveScorer.calibration_source ──────────────────────────────────


def _write_calibration_file(tmp_path, name, entries):
    path = tmp_path / name
    path.write_text(json.dumps({"schemes": entries}))
    return path


def test_calibration_source_default_matches_no_calibration_source(tmp_path):
    """calibration_source="" (the default) must behave identically to
    never passing it -- both fall back to the (isolated-missing) default
    _CALIBRATION_PATH lookup."""
    default_scorer = PredictiveScorer(sensitivity_weights={})
    explicit_empty_scorer = PredictiveScorer(sensitivity_weights={}, calibration_source="")

    for scheme in ("Q8_0", "Q4_K_M", "MXFP4_MOE"):
        assert (
            default_scorer._noise_factor_for(scheme)
            == explicit_empty_scorer._noise_factor_for(scheme)
        )
        assert (
            default_scorer._speed_for(scheme)
            == explicit_empty_scorer._speed_for(scheme)
        )


def test_calibration_source_overrides_noise_factor(tmp_path):
    calib_path = _write_calibration_file(
        tmp_path, "custom_calibration.json",
        {"Q4_K_M": {"noise_factor": 1.23}},
    )
    scorer = PredictiveScorer(sensitivity_weights={}, calibration_source=str(calib_path))

    assert scorer._noise_factor_for("Q4_K_M") == 1.23
    from magicquant.quant.schemes import get_scheme_by_name
    assert get_scheme_by_name("Q4_K_M").noise_factor != 1.23


def test_calibration_source_overrides_speed_multiplier(tmp_path):
    calib_path = _write_calibration_file(
        tmp_path, "custom_calibration.json",
        {"IQ4_NL": {"speed_multiplier": 2.5}},
    )
    scorer = PredictiveScorer(sensitivity_weights={}, calibration_source=str(calib_path))

    assert scorer._speed_for("IQ4_NL") == 2.5


def test_calibration_source_shifts_predicted_loss(tmp_path):
    """A config's predicted_loss must shift when calibration_source overrides
    one of its schemes' noise_factor, relative to the (isolated) default
    lookup for the same scorer configuration."""
    config = {"Q": "Q4_K_M", "U": "MXFP4_MOE"}
    sensitivity_weights = {"Q": 0.5, "U": 0.5}

    baseline_scorer = PredictiveScorer(sensitivity_weights=sensitivity_weights)
    baseline_loss = baseline_scorer.predict_loss(config)

    calib_path = _write_calibration_file(
        tmp_path, "custom_calibration.json",
        {"Q4_K_M": {"noise_factor": 0.1}},
    )
    overridden_scorer = PredictiveScorer(
        sensitivity_weights=sensitivity_weights, calibration_source=str(calib_path)
    )
    overridden_loss = overridden_scorer.predict_loss(config)

    assert overridden_loss != pytest.approx(baseline_loss)


def test_calibration_source_cache_does_not_clobber_default_cache(tmp_path):
    """A calibration_source override for one scorer must not leak into (or
    be shadowed by) the default-path lookup another scorer still uses."""
    calib_path = _write_calibration_file(
        tmp_path, "custom_calibration.json",
        {"Q8_0": {"noise_factor": 9.99}},
    )
    overridden_scorer = PredictiveScorer(sensitivity_weights={}, calibration_source=str(calib_path))
    default_scorer = PredictiveScorer(sensitivity_weights={})

    assert overridden_scorer._noise_factor_for("Q8_0") == 9.99
    from magicquant.quant.schemes import get_scheme_by_name
    assert default_scorer._noise_factor_for("Q8_0") == get_scheme_by_name("Q8_0").noise_factor


# ── Orchestrator wiring ───────────────────────────────────────────────────

_TENSOR_NAMES = [
    "token_embd.weight",
    "output.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
]


class _FakeSource:
    def get_tensor_names(self):
        return list(_TENSOR_NAMES)

    def get_all_tensors_info(self):
        return [{"name": n, "shape": [4, 4]} for n in _TENSOR_NAMES]

    def close(self):
        pass


class _FakeLlamaTools:
    def __init__(self):
        self.ctx_size = 512

    def calculate_perplexity(self, path, verbose=False, **kw):
        return 5.0

    def _resolve_data_file(self, data_file=None):
        return "/fake/corpus.txt"


def _make_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    orch._llama_tools = _FakeLlamaTools()

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    counter = {"n": 0}

    def fake_build_candidate(config, name, base_quant):
        counter["n"] += 1
        p = candidates_dir / f"{name}_{counter['n']}.gguf"
        p.write_bytes(b"0" * 1024)
        return str(p)

    monkeypatch.setattr(orch, "_build_candidate", fake_build_candidate)
    return orch


class _CapturingPredictiveScorer(PredictiveScorer):
    """Real PredictiveScorer, recording every constructor call's kwargs so
    orchestrator-level forwarding of calibration_source can be asserted."""

    captured: list = []

    def __init__(self, **kwargs):
        type(self).captured.append(kwargs)
        super().__init__(**kwargs)


@pytest.fixture(autouse=True)
def _reset_captured_predictor():
    _CapturingPredictiveScorer.captured = []
    yield
    _CapturingPredictiveScorer.captured = []


def test_run_measured_search_forwards_calibration_source(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch_mod, "PredictiveScorer", _CapturingPredictiveScorer)
    calib_path = tmp_path / "my_calibration.json"
    calib_path.write_text(json.dumps({"schemes": {}}))

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        calibration_source=str(calib_path),
    )

    assert _CapturingPredictiveScorer.captured
    assert _CapturingPredictiveScorer.captured[0]["calibration_source"] == str(calib_path)


def test_run_measured_search_default_calibration_source_is_empty_string(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch_mod, "PredictiveScorer", _CapturingPredictiveScorer)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
    )

    assert _CapturingPredictiveScorer.captured[0]["calibration_source"] == ""


def test_run_full_search_forwards_calibration_source(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch_mod, "PredictiveScorer", _CapturingPredictiveScorer)
    calib_path = tmp_path / "my_calibration.json"
    calib_path.write_text(json.dumps({"schemes": {}}))

    orch.run_full_search(
        max_generations=2, population_size=8, verbose=False,
        calibration_source=str(calib_path),
    )

    assert _CapturingPredictiveScorer.captured[0]["calibration_source"] == str(calib_path)


# ── Orchestrator wiring: write_calibration end-to-end ────────────────────


def test_run_measured_search_default_does_not_write_calibration(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
    )

    assert not (orch.output_dir / "noise_calibration.json").exists()


def test_run_measured_search_write_calibration_true_writes_file(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        write_calibration=True,
    )

    calib_path = orch.output_dir / "noise_calibration.json"
    assert calib_path.exists()
    envelope = json.loads(calib_path.read_text())
    assert "schemes" in envelope

    used_schemes = {
        scheme
        for info in orch._measured.values()
        for scheme in info["config"].values()
        if scheme != "BF16"
    }
    assert used_schemes <= set(envelope["schemes"])
