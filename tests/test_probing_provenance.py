"""Sensitivity-probing provenance: probe_all_groups must stamp whether the
sensitivity weights it produced came from real llama-perplexity measurements
("measured"), a mix ("partial"), or ALL heuristic fallback ("heuristic") --
and warn loudly in the all-heuristic case, since the evolutionary search
would otherwise silently run entirely on static empirical guesses instead of
this model's actual quantization behavior.

Ground truth for why this matters: in the completed Qwopus3.6-27B run, every
one of the 7 real sensitivity probes crashed llama-perplexity (the SSM-conv
BF16 bug this same lane fixes in the writer) and silently fell back to
heuristic -- the whole evolutionary search ran on guessed weights with no
record of it. See magicquant.evolution.probing.SensitivityProber.
"""
import json
import logging

from magicquant.evolution.probing import SensitivityProber


class _FakeReader:
    def open(self): pass
    def close(self): pass
    def get_tensor_names(self): return ["blk.0.attn_q.weight"]


def _patch_writer_success(monkeypatch):
    monkeypatch.setattr("magicquant.gguf.reader.GGUFReader", lambda *a, **k: _FakeReader())
    monkeypatch.setattr("magicquant.gguf.writer.create_hybrid_gguf", lambda *a, **k: None)


def _prober(tmp_path, calculator):
    # base_model_path must exist so _probe_single_group takes the real-probe
    # branch whenever a calculator is supplied.
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return SensitivityProber(
        base_model_path=str(model),
        baseline_perplexity=5.0,
        perplexity_calculator=calculator,
        output_dir=str(tmp_path / "_probes"),
    )


class _AlwaysMeasuredCalc:
    def calculate_perplexity(self, *a, **k):
        return 5.5


class _FlakyCalc:
    """First probe measures fine; every subsequent one "fails" (returns
    None, which _real_probe treats as a measurement failure -> heuristic)."""

    def __init__(self):
        self.calls = 0

    def calculate_perplexity(self, *a, **k):
        self.calls += 1
        return 5.5 if self.calls == 1 else None


def test_fresh_prober_provenance_is_unknown(tmp_path):
    prober = _prober(tmp_path, calculator=None)
    assert prober.probing_provenance == "unknown"


def test_all_measured_sets_measured_provenance(tmp_path, monkeypatch, caplog):
    _patch_writer_success(monkeypatch)
    prober = _prober(tmp_path, _AlwaysMeasuredCalc())

    with caplog.at_level(logging.WARNING, logger="magicquant.evolution.probing"):
        prober.probe_all_groups(groups=["Q", "K"], verbose=False)

    assert prober.probing_provenance == "measured"
    assert all(p["measured"] for p in prober.probe_models)
    assert not any("ZERO real measurements" in r.message for r in caplog.records)


def test_no_calculator_is_all_heuristic_and_warns(tmp_path, caplog):
    prober = _prober(tmp_path, calculator=None)

    with caplog.at_level(logging.WARNING, logger="magicquant.evolution.probing"):
        prober.probe_all_groups(groups=["Q", "K"], verbose=False)

    assert prober.probing_provenance == "heuristic"
    assert not any(p["measured"] for p in prober.probe_models)
    assert any("ZERO real measurements" in r.message for r in caplog.records)


def test_partial_measurement_sets_partial_provenance(tmp_path, monkeypatch, caplog):
    _patch_writer_success(monkeypatch)
    prober = _prober(tmp_path, _FlakyCalc())

    with caplog.at_level(logging.WARNING, logger="magicquant.evolution.probing"):
        prober.probe_all_groups(groups=["Q", "K"], verbose=False)

    assert prober.probing_provenance == "partial"
    measured_flags = [p["measured"] for p in prober.probe_models]
    assert measured_flags.count(True) == 1
    assert measured_flags.count(False) == 1
    # Partial is not the "every probe failed" case -- no loud warning.
    assert not any("ZERO real measurements" in r.message for r in caplog.records)


def test_save_results_persists_probing_provenance(tmp_path):
    prober = _prober(tmp_path, calculator=None)
    prober.probe_all_groups(groups=["Q"], verbose=False)

    out = tmp_path / "sensitivity.json"
    prober.save_results(str(out))

    data = json.loads(out.read_text())
    assert data["probing_provenance"] == "heuristic"
    assert data["probes"][0]["measured"] is False
