"""Probe error-handling tests (L13).

A writer contract error (ValueError) inside _real_probe must propagate and be
logged with a traceback, not be silently masked by a fabricated heuristic PPL.
Other (measurement) failures still fall back to heuristic, but are logged.
"""
import logging

import pytest

from magicquant.evolution.probing import SensitivityProber


class _FakeCalc:
    def calculate_perplexity(self, *a, **k):
        return 5.0


def _prober(tmp_path):
    # base_model_path must be an existing file so _real_probe is taken.
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return SensitivityProber(
        base_model_path=str(model),
        baseline_perplexity=5.0,
        perplexity_calculator=_FakeCalc(),
        output_dir=str(tmp_path / "_probes"),
    )


def test_value_error_propagates(tmp_path, monkeypatch, caplog):
    prober = _prober(tmp_path)

    # Make the reader step succeed but create_hybrid_gguf raise ValueError.
    import magicquant.evolution.probing as probing_mod

    class _FakeReader:
        def open(self): pass
        def close(self): pass
        def get_tensor_names(self): return ["blk.0.attn_q.weight"]

    monkeypatch.setattr("magicquant.gguf.reader.GGUFReader", lambda *a, **k: _FakeReader())

    def _raise(*a, **k):
        raise ValueError("source is already quantized")
    monkeypatch.setattr("magicquant.gguf.writer.create_hybrid_gguf", _raise)

    with caplog.at_level(logging.ERROR, logger="magicquant.evolution.probing"):
        with pytest.raises(ValueError, match="already quantized"):
            prober._real_probe("Q", "Q4_K_M", "BF16", verbose=False)

    assert any("writer contract error" in r.message for r in caplog.records)


def test_runtime_error_falls_back_to_heuristic(tmp_path, monkeypatch):
    prober = _prober(tmp_path)

    class _FakeReader:
        def open(self): pass
        def close(self): pass
        def get_tensor_names(self): return ["blk.0.attn_q.weight"]

    monkeypatch.setattr("magicquant.gguf.reader.GGUFReader", lambda *a, **k: _FakeReader())

    def _raise(*a, **k):
        raise RuntimeError("subprocess died")
    monkeypatch.setattr("magicquant.gguf.writer.create_hybrid_gguf", _raise)

    # Should NOT raise — falls back to a heuristic float, flagged unmeasured.
    ppl, measured = prober._real_probe("Q", "Q4_K_M", "BF16", verbose=False)
    assert isinstance(ppl, float)
    assert measured is False
