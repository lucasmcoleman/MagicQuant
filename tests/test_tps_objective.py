"""Tunable tps-aware search-objective tests (LANE B / PART 1).

Covers:
  - PredictiveScorer.score_hybrid's opt-in ``use_bytes_tps``: a
    deterministic, bandwidth-bound proxy (tps_score = min(1,
    baseline_size_gb / predicted_size)) that replaces the noisy per-scheme
    speed_multiplier path when explicitly requested. Off by default --
    byte-identical composite_score to the historical formula.
  - EvolutionarySurvivor's opt-in ``objective_weights``/``use_bytes_tps``,
    threaded through to ``score_hybrid`` every generation (see
    ``_predict_population``).
  - MagicQuantOrchestrator._build_objective_weights: renormalizes
    precision:size to fill the remainder after reserving ``speed_weight``,
    keeping their default 0.50:0.35 ratio.
  - Orchestrator wiring: run_measured_search/run_full_search forward
    speed_weight/use_bytes_tps into the EvolutionarySurvivor they
    construct.

All new knobs are opt-in and default to None/False, preserving the
historical call shape exactly -- required for the seed-pinned
refactor-regression fixture (tests/test_refactor_regression.py), which is
exercised (twice, as separate processes) alongside this suite.
"""
import pytest

import magicquant.gguf.source as source_mod
import magicquant.orchestrator as orch_mod
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.orchestrator import MagicQuantOrchestrator


def _scorer(**kw):
    defaults = dict(
        sensitivity_weights={
            "E": 1.0, "H": 1.0, "Q": 0.8, "K": 0.8, "O": 0.9, "U": 0.4, "D": 0.4,
        },
        parameter_counts={
            "E": 100_000_000, "H": 100_000_000, "Q": 300_000_000, "K": 300_000_000,
            "O": 150_000_000, "U": 800_000_000, "D": 800_000_000,
        },
        baseline_size_gb=10.0,
    )
    defaults.update(kw)
    return PredictiveScorer(**defaults)


_GROUPS = ["E", "H", "Q", "K", "O", "U", "D"]

CONFIGS = [
    {"E": "BF16", "H": "BF16", "Q": "Q6_K", "K": "Q6_K", "O": "Q8_0",
     "U": "MXFP4_MOE", "D": "MXFP4_MOE"},
    {"E": "BF16", "H": "BF16", "Q": "IQ4_NL", "K": "IQ4_NL", "O": "Q8_0",
     "U": "MXFP4_MOE", "D": "MXFP4_MOE"},
    {"E": "Q8_0", "H": "Q8_0", "Q": "Q4_K_M", "K": "Q4_K_M", "O": "Q6_K",
     "U": "MXFP4_MOE", "D": "Q3_K"},
    {g: "Q6_K" for g in _GROUPS},
    {g: "MXFP4_MOE" for g in _GROUPS},
]


# ── score_hybrid: use_bytes_tps ──────────────────────────────────────────


def test_use_bytes_tps_matches_formula_exactly():
    """tps_score under use_bytes_tps=True must exactly equal
    min(1, baseline_size_gb / max(predicted_size, eps)), computed
    independently via predict_size -- not the predict_tps path."""
    scorer = _scorer()
    eps = scorer._BYTES_TPS_EPS
    for cfg in CONFIGS:
        predicted_size = scorer.predict_size(cfg)
        expected = min(1.0, scorer.baseline_size_gb / max(predicted_size, eps))
        result = scorer.score_hybrid(cfg, use_bytes_tps=True)
        assert result["tps_score"] == pytest.approx(expected)


def test_use_bytes_tps_zero_baseline_gives_zero_tps_score():
    """baseline_size_gb=0 routes predict_size to its 1.0 fallback
    (_estimate_simple_size), so the bytes-based ratio is 0/1.0 = 0 --
    deterministic, unlike the predict_tps fallback path it replaces."""
    scorer = _scorer(baseline_size_gb=0)
    result = scorer.score_hybrid(CONFIGS[0], use_bytes_tps=True)
    assert result["tps_score"] == 0.0


def test_use_bytes_tps_differs_from_default_speed_scoring():
    """With baseline_tps=0 (so the default path uses the discriminating
    predict_tps/4.0 branch, not the saturated baseline_tps>0 branch, see
    score_hybrid), the two tps_score paths give genuinely different values
    for the same config -- use_bytes_tps ignores speed_multiplier entirely
    in favor of a pure size ratio."""
    scorer = _scorer(baseline_tps=0)
    cfg = CONFIGS[0]
    default_tps_score = scorer.score_hybrid(cfg)["tps_score"]
    bytes_tps_score = scorer.score_hybrid(cfg, use_bytes_tps=True)["tps_score"]
    assert default_tps_score != pytest.approx(bytes_tps_score)


# ── score_hybrid: default call is byte-identical to the historical formula ──


def _historical_score_hybrid(scorer, group_schemes,
                              precision_weight=0.50, size_weight=0.35,
                              speed_weight=0.15):
    """Recomputes score_hybrid's pre-use_bytes_tps formula independently, so
    the current default call path can be checked byte-identical against it."""
    predicted_loss = scorer.predict_loss(group_schemes)
    predicted_size = scorer.predict_size(group_schemes)
    predicted_tps = scorer.predict_tps(group_schemes)

    loss_score = max(0, 1 - predicted_loss / 5.0)
    if scorer.baseline_size_gb > 0:
        size_score = max(0.0, 1.0 - predicted_size / scorer.baseline_size_gb)
    else:
        size_score = max(0.0, 1.0 - predicted_size)

    if scorer.baseline_tps > 0:
        tps_score = min(1, predicted_tps / scorer.baseline_tps)
    elif predicted_tps > 0:
        tps_score = min(1, predicted_tps / 4.0)
    else:
        tps_score = 0.0

    return (
        precision_weight * loss_score
        + size_weight * size_score
        + speed_weight * tps_score
    )


@pytest.mark.parametrize("baseline_tps", [0, 20.0, 360])
def test_default_composite_score_byte_identical_to_historical_formula(baseline_tps):
    scorer = _scorer(baseline_tps=baseline_tps)
    for cfg in CONFIGS:
        expected = _historical_score_hybrid(scorer, cfg)
        actual = scorer.score_hybrid(cfg)["composite_score"]
        assert actual == expected


# ── tunable objective_weights cause a real ranking reversal ──────────────


def test_higher_speed_weight_outranks_smaller_faster_config():
    """With baseline_tps unset (0, the class default), predict_tps's
    per-scheme speed_multiplier meaningfully discriminates configs (see
    score_hybrid's ``elif predicted_tps > 0`` branch) -- unlike the
    baseline_tps>0 branch, which saturates to 1.0 for every config since
    every registry speed_multiplier is >= 1.0 (BF16's own floor)."""
    scorer = _scorer(baseline_tps=0)

    config_precise_big = {  # near-BF16 quality, much bigger file
        "E": "BF16", "H": "BF16", "Q": "Q8_0", "K": "Q8_0",
        "O": "Q8_0", "U": "Q8_0", "D": "Q8_0",
    }
    config_aggressive_small = {  # noisier, much smaller/faster
        "E": "BF16", "H": "BF16", "Q": "MXFP4_MOE", "K": "MXFP4_MOE",
        "O": "Q8_0", "U": "MXFP4_MOE", "D": "MXFP4_MOE",
    }

    default_big = scorer.score_hybrid(config_precise_big)["composite_score"]
    default_small = scorer.score_hybrid(config_aggressive_small)["composite_score"]
    assert default_big > default_small, (
        "expected default weights to favor the more precise/bigger config"
    )

    speed_weights = MagicQuantOrchestrator._build_objective_weights(0.85)
    speed_big = scorer.score_hybrid(config_precise_big, *speed_weights)["composite_score"]
    speed_small = scorer.score_hybrid(config_aggressive_small, *speed_weights)["composite_score"]
    assert speed_small > speed_big, (
        "expected a high speed_weight to flip the ranking toward the "
        "smaller/faster config"
    )


# ── EvolutionarySurvivor: objective_weights/use_bytes_tps forwarding ─────


class _RecordingPredictor:
    """Stands in for PredictiveScorer -- records exactly what kwargs
    _predict_population passed to score_hybrid, without any real scoring
    math (so plumbing is tested in isolation from the formula itself)."""

    def __init__(self):
        self.calls = []

    def score_hybrid(self, config, **kwargs):
        self.calls.append(kwargs)
        return {
            "predicted_loss": 0.0, "predicted_size_gb": 1.0, "predicted_tps": 1.0,
            "loss_score": 1.0, "size_score": 1.0, "tps_score": 1.0,
            "composite_score": 1.0,
        }


def _survivor(predictor, **kw):
    return EvolutionarySurvivor(predictor=predictor, baseline_config={}, **kw)


def test_predict_population_default_calls_score_hybrid_with_no_extra_args():
    predictor = _RecordingPredictor()
    survivor = _survivor(predictor)
    survivor._predict_population([{"config": {"E": "BF16"}}])
    assert predictor.calls == [{}]


def test_predict_population_forwards_objective_weights():
    predictor = _RecordingPredictor()
    survivor = _survivor(predictor, objective_weights=(0.3, 0.2, 0.5))
    survivor._predict_population([{"config": {"E": "BF16"}}])
    assert predictor.calls == [{
        "precision_weight": 0.3, "size_weight": 0.2, "speed_weight": 0.5,
        "use_bytes_tps": False,
    }]


def test_predict_population_forwards_use_bytes_tps_alone():
    """use_bytes_tps=True with objective_weights left at None still routes
    through score_hybrid's default 0.50/0.35/0.15 weights -- only
    use_bytes_tps changes."""
    predictor = _RecordingPredictor()
    survivor = _survivor(predictor, use_bytes_tps=True)
    survivor._predict_population([{"config": {"E": "BF16"}}])
    assert predictor.calls == [{
        "precision_weight": 0.50, "size_weight": 0.35, "speed_weight": 0.15,
        "use_bytes_tps": True,
    }]


def test_objective_weights_and_use_bytes_tps_default_to_none_false():
    survivor = _survivor(_RecordingPredictor())
    assert survivor.objective_weights is None
    assert survivor.use_bytes_tps is False


# ── MagicQuantOrchestrator._build_objective_weights ──────────────────────


def test_build_objective_weights_none_when_speed_weight_unset():
    assert MagicQuantOrchestrator._build_objective_weights(None) is None


def test_build_objective_weights_renormalizes_precision_and_size():
    precision, size, speed = MagicQuantOrchestrator._build_objective_weights(0.40)
    assert speed == 0.40
    assert precision == pytest.approx(0.36, abs=0.01)
    assert size == pytest.approx(0.24, abs=0.01)
    assert precision + size + speed == pytest.approx(1.0)
    # Ratio between precision and size is preserved at the default 0.50:0.35.
    assert precision / size == pytest.approx(0.50 / 0.35)


@pytest.mark.parametrize("speed_weight", [0.0, 0.15, 0.5, 0.9, 1.0])
def test_build_objective_weights_always_sums_to_one(speed_weight):
    weights = MagicQuantOrchestrator._build_objective_weights(speed_weight)
    assert sum(weights) == pytest.approx(1.0)


def test_build_objective_weights_zero_speed_matches_default_ratio():
    """speed_weight=0.0 (explicitly reserving nothing for speed) renormalizes
    precision/size to consume the FULL remainder (1.0) while preserving
    their 0.50:0.35 ratio -- distinct from speed_weight=None, which skips
    objective_weights entirely and leaves score_hybrid's own 0.50/0.35/0.15
    defaults (which reserve 0.15 for speed) in effect."""
    precision, size, speed = MagicQuantOrchestrator._build_objective_weights(0.0)
    assert speed == 0.0
    assert precision / size == pytest.approx(0.50 / 0.35)
    assert precision + size == pytest.approx(1.0)


# ── Orchestrator wiring: forwards speed_weight/use_bytes_tps to the survivor ──


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


class _CapturingSurvivor(EvolutionarySurvivor):
    """Real EvolutionarySurvivor, just recording every constructor call's
    kwargs so orchestrator-level forwarding can be asserted without a
    hand-rolled stand-in that might silently drift from the real
    constructor's signature."""

    captured: list = []

    def __init__(self, **kwargs):
        type(self).captured.append(kwargs)
        super().__init__(**kwargs)


@pytest.fixture(autouse=True)
def _reset_captured():
    _CapturingSurvivor.captured = []
    yield
    _CapturingSurvivor.captured = []


def test_run_measured_search_forwards_speed_knobs_to_survivor(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch_mod, "EvolutionarySurvivor", _CapturingSurvivor)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        speed_weight=0.4, use_bytes_tps=True,
    )

    assert _CapturingSurvivor.captured, "survivor was never constructed"
    kwargs = _CapturingSurvivor.captured[0]
    assert kwargs["objective_weights"] == MagicQuantOrchestrator._build_objective_weights(0.4)
    assert kwargs["use_bytes_tps"] is True


def test_run_measured_search_default_forwards_none_and_false(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch_mod, "EvolutionarySurvivor", _CapturingSurvivor)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
    )

    kwargs = _CapturingSurvivor.captured[0]
    assert kwargs["objective_weights"] is None
    assert kwargs["use_bytes_tps"] is False


def test_run_full_search_forwards_speed_knobs_to_survivor(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch_mod, "EvolutionarySurvivor", _CapturingSurvivor)

    orch.run_full_search(
        max_generations=2, population_size=8, verbose=False,
        speed_weight=0.4, use_bytes_tps=True,
    )

    assert _CapturingSurvivor.captured, "survivor was never constructed"
    kwargs = _CapturingSurvivor.captured[0]
    assert kwargs["objective_weights"] == MagicQuantOrchestrator._build_objective_weights(0.4)
    assert kwargs["use_bytes_tps"] is True


def test_run_full_search_default_forwards_none_and_false(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch_mod, "EvolutionarySurvivor", _CapturingSurvivor)

    orch.run_full_search(max_generations=2, population_size=8, verbose=False)

    kwargs = _CapturingSurvivor.captured[0]
    assert kwargs["objective_weights"] is None
    assert kwargs["use_bytes_tps"] is False
