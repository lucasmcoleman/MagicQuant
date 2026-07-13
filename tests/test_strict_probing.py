"""The silent-degradation kill: a failed probe in strict mode raises
ProbeMeasurementError instead of substituting fabricated heuristics, and
the measured search runs its prober strict. Non-strict (prediction-only)
behavior is unchanged."""

import numpy as np
import pytest

import magicquant.gguf.writer as writer_mod
from magicquant.evolution.probing import ProbeMeasurementError, SensitivityProber


class _NoneCalculator:
    """Measurement always 'succeeds' subprocess-wise but yields no PPL."""

    def __init__(self):
        self.calls = 0

    def calculate_perplexity(self, path, verbose=True, **kw):
        self.calls += 1
        return None


class _RaisingCalculator:
    def calculate_perplexity(self, path, verbose=True, **kw):
        raise RuntimeError("injected transient GPU failure")


@pytest.fixture()
def fake_build(monkeypatch, tmp_path):
    """Make create_hybrid_gguf a cheap no-op that produces a file, and give
    the prober a real source path so _real_probe is taken."""
    src = tmp_path / "model.gguf"
    src.write_bytes(b"GGUF-stub")

    def _fake_create(output_path, base_model_path, quant_config, verbose=False,
                     **kw):
        from pathlib import Path
        Path(output_path).write_bytes(b"probe-stub")
        return output_path

    monkeypatch.setattr(writer_mod, "create_hybrid_gguf", _fake_create)

    # _real_probe also opens the source with GGUFReader to enumerate groups;
    # stub that out too.
    import magicquant.gguf.reader as reader_mod

    class _FakeReader:
        def __init__(self, path):
            pass

        def open(self):
            pass

        def get_tensor_names(self):
            return ["token_embd.weight", "blk.0.ffn_down.weight"]

        def close(self):
            pass

    monkeypatch.setattr(reader_mod, "GGUFReader", _FakeReader)
    return str(src)


def _prober(src, calc, strict):
    return SensitivityProber(
        base_model_path=src,
        baseline_perplexity=10.0,
        perplexity_calculator=calc,
        output_dir=None,
        strict=strict,
    )


def test_strict_raises_on_unmeasurable_probe(fake_build, tmp_path):
    calc = _NoneCalculator()
    prober = _prober(fake_build, calc, strict=True)
    prober.output_dir = str(tmp_path / "probes")
    with pytest.raises(ProbeMeasurementError):
        prober.probe_all_groups(groups=["E"], verbose=False)
    assert calc.calls == 2  # one retry before raising


def test_strict_raises_on_probe_exception(fake_build, tmp_path):
    prober = _prober(fake_build, _RaisingCalculator(), strict=True)
    prober.output_dir = str(tmp_path / "probes")
    with pytest.raises(ProbeMeasurementError):
        prober.probe_all_groups(groups=["E"], verbose=False)


def test_nonstrict_keeps_heuristic_fallback(fake_build, tmp_path):
    """Historical prediction-only behavior unchanged: fabricated estimate,
    measured=False, provenance 'heuristic'."""
    prober = _prober(fake_build, _NoneCalculator(), strict=False)
    prober.output_dir = str(tmp_path / "probes")
    results = prober.probe_all_groups(groups=["E"], verbose=False)
    assert "E" in results
    assert prober.probing_provenance == "heuristic"
    assert prober.probe_models[0]["measured"] is False


def test_measured_search_constructs_strict_prober(monkeypatch, tmp_path):
    """run_measured_search must pass strict=True to its prober."""
    import magicquant.orchestrator as orch_mod
    from magicquant.orchestrator import MagicQuantOrchestrator

    captured = {}

    class _Sentinel(Exception):
        pass

    class _SpyProber:
        def __init__(self, *a, **kw):
            captured.update(kw)
            raise _Sentinel()

    monkeypatch.setattr(orch_mod, "SensitivityProber", _SpyProber)

    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF-stub")

    orch = MagicQuantOrchestrator(
        source_model_path=str(src), output_dir=str(tmp_path / "out")
    )

    class _FakeTools:
        ppl_chunks = None
        ctx_size = 512

        def calculate_perplexity(self, *a, **k):
            return 10.0

        def _resolve_data_file(self, *_a):
            return None

    orch._llama_tools = _FakeTools()

    # Group detection opens the source — stub it.
    import magicquant.gguf.source as source_mod

    class _FakeSource:
        def get_tensor_names(self):
            return ["token_embd.weight"]

        def close(self):
            pass

    monkeypatch.setattr(
        source_mod, "open_model_source", lambda *a, **k: _FakeSource()
    )

    with pytest.raises(_Sentinel):
        orch.run_measured_search(
            measurement_rounds=1, verbose=False, resume=False,
        )
    assert captured.get("strict") is True
