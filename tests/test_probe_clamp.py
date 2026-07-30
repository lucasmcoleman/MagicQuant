"""Regression tests for FIX 2(a)/(b): probing.py must never silently treat a
physically impossible measurement (quantized probe PPL below baseline) as a
genuine zero-sensitivity result, and must expose when the whole search ran
without real sensitivity signal.

Incident context: a measured search recorded sensitivity probes of 2.6-2.8
against a baseline of 34.8363 (all NaN-cascade artifacts -- see
tests/test_perplexity_parse.py for the parser-side root cause). The
unguarded ``max(0.0, ppl - baseline) / baseline`` clamped 8/9 of these to
exactly 0.0 sensitivity with no log line and no record of *why*, while the
probe entry still carried ``measured: true`` and ``probing_provenance``
still said "measured".
"""
import logging

import pytest

from magicquant.evolution.probing import SensitivityProber


class _GroupAwareCalc:
    """Fake perplexity calculator: returns a per-group PPL keyed by which
    probe GGUF filename (``probe_<group>.gguf``) it's asked to measure --
    mirrors _real_probe's real file-naming scheme."""

    def __init__(self, ppl_by_group, default=40.0):
        self.ppl_by_group = ppl_by_group
        self.default = default

    def calculate_perplexity(self, model_path, verbose=False):
        for group, ppl in self.ppl_by_group.items():
            if f"probe_{group}.gguf" in model_path:
                return ppl
        return self.default


class _FakeReader:
    def open(self): pass
    def close(self): pass
    def get_tensor_names(self): return ["blk.0.attn_q.weight"]


def _patch_probe_build(monkeypatch):
    monkeypatch.setattr("magicquant.gguf.reader.GGUFReader", lambda *a, **k: _FakeReader())
    monkeypatch.setattr("magicquant.gguf.writer.create_hybrid_gguf", lambda **k: None)


def _prober(tmp_path, calc, baseline=34.8363, **kwargs):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return SensitivityProber(
        base_model_path=str(model),
        baseline_perplexity=baseline,
        perplexity_calculator=calc,
        output_dir=str(tmp_path / "_probes"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# FIX 2(a): a probe below baseline is clamped AND flagged, with a WARNING.
# ---------------------------------------------------------------------------

def test_probe_below_baseline_is_clamped_and_flagged(tmp_path, monkeypatch, caplog):
    _patch_probe_build(monkeypatch)
    # Incident-shaped values: bogus probes of 2.6-2.8 against 34.8363.
    calc = _GroupAwareCalc({"E": 2.74, "H": 2.6})
    prober = _prober(tmp_path, calc)

    with caplog.at_level(logging.WARNING, logger="magicquant.evolution.probing"):
        prober.probe_all_groups(groups=["E", "H"], aggressive_scheme="Q4_K_M", verbose=False)

    for group in ("E", "H"):
        entry = next(p for p in prober.probe_models if p["group"] == group)
        assert entry["clamped"] is True, f"group {group} should be flagged clamped"
        assert entry["sensitivity"] == 0.0

    assert any("clamp" in rec.message.lower() for rec in caplog.records), (
        "expected a WARNING naming the clamped group(s)"
    )


def test_probe_above_baseline_is_not_clamped(tmp_path, monkeypatch):
    """Sanity check: a REAL (above-baseline) probe must never be flagged."""
    _patch_probe_build(monkeypatch)
    calc = _GroupAwareCalc({"E": 40.0})
    prober = _prober(tmp_path, calc)
    prober.probe_all_groups(groups=["E"], aggressive_scheme="Q4_K_M", verbose=False)
    entry = prober.probe_models[0]
    assert entry["clamped"] is False
    assert entry["sensitivity"] > 0.0


def test_majority_clamped_downgrades_provenance_to_suspect(tmp_path, monkeypatch):
    _patch_probe_build(monkeypatch)
    # 2 of 3 groups clamped -- more than half.
    calc = _GroupAwareCalc({"E": 2.74, "H": 2.6, "Q": 40.0})
    prober = _prober(tmp_path, calc)
    prober.probe_all_groups(groups=["E", "H", "Q"], aggressive_scheme="Q4_K_M", verbose=False)
    assert prober.probing_provenance == "suspect", (
        "majority-clamped run must downgrade provenance away from 'measured'"
    )


def test_minority_clamped_does_not_downgrade_provenance(tmp_path, monkeypatch):
    _patch_probe_build(monkeypatch)
    # Only 1 of 4 groups clamped -- not a majority.
    calc = _GroupAwareCalc({"E": 2.74}, default=40.0)
    prober = _prober(tmp_path, calc)
    prober.probe_all_groups(groups=["E", "H", "Q", "K"], aggressive_scheme="Q4_K_M", verbose=False)
    assert prober.probing_provenance == "measured"


def test_clamp_eps_uses_baseline_ppl_err_when_reachable(tmp_path, monkeypatch):
    """A probe at 90 against baseline 100 is below the flat 2% default
    threshold (98) and would clamp -- but with a reported baseline error of
    20, eps becomes 20% (threshold 80), and 90 no longer clamps."""
    _patch_probe_build(monkeypatch)
    calc = _GroupAwareCalc({"E": 90.0})
    prober = _prober(tmp_path, calc, baseline=100.0, baseline_ppl_err=20.0)
    prober.probe_all_groups(groups=["E"], aggressive_scheme="Q4_K_M", verbose=False)
    entry = prober.probe_models[0]
    assert entry["clamped"] is False


# ---------------------------------------------------------------------------
# FIX 2(b): get_normalized_weights must expose the degenerate (no-signal)
# state, not just silently return uniform weights.
# ---------------------------------------------------------------------------

def test_get_normalized_weights_degenerate_flag_and_warning(caplog):
    prober = SensitivityProber(
        base_model_path="missing.gguf", baseline_perplexity=10.0,
        perplexity_calculator=None,
    )
    prober.sensitivity_results = {"E": 0.0, "H": 0.0, "Q": 0.0}

    with caplog.at_level(logging.WARNING, logger="magicquant.evolution.probing"):
        weights = prober.get_normalized_weights()

    assert prober.weights_degenerate is True
    assert weights == pytest.approx({"E": 1 / 3, "H": 1 / 3, "Q": 1 / 3})
    assert any("without sensitivity signal" in rec.message.lower() for rec in caplog.records)


def test_get_normalized_weights_not_degenerate_with_real_signal():
    prober = SensitivityProber(
        base_model_path="missing.gguf", baseline_perplexity=10.0,
        perplexity_calculator=None,
    )
    prober.sensitivity_results = {"E": 0.4, "H": 0.0}
    weights = prober.get_normalized_weights()
    assert prober.weights_degenerate is False
    assert weights == pytest.approx({"E": 1.0, "H": 0.0})


def test_save_results_records_weights_degenerate(tmp_path):
    import json

    prober = SensitivityProber(
        base_model_path="missing.gguf", baseline_perplexity=10.0,
        perplexity_calculator=None,
    )
    prober.sensitivity_results = {"E": 0.0, "H": 0.0}
    out = tmp_path / "sensitivity.json"
    prober.save_results(str(out))
    data = json.loads(out.read_text())
    assert data["weights_degenerate"] is True
