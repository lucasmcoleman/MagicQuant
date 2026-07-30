"""Regression tests for MAJOR 4: SensitivityProber's "suspect" provenance
and weights_degenerate flag were write-only -- stamped into
sensitivity.json/search_results.json for a human to notice later, but
nothing outside tests actually gated on them. A search positively
identified as signal-less (more than half its measured probes physically
impossible, or every group's sensitivity <=0) still completed and shipped
tiers exactly like a healthy measured run.

MagicQuantOrchestrator._enforce_probing_signal_gate makes both signals
load-bearing: refuse to proceed by default; MAGICQUANT_ALLOW_DEGENERATE_PROBING=1
opts back into the historical silent-proceed behavior.
"""
import pytest

from magicquant.orchestrator import MagicQuantOrchestrator


def _bare_orchestrator(probing_provenance, weights_degenerate):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.probing_provenance = probing_provenance
    orch.weights_degenerate = weights_degenerate
    return orch


# ── Default behavior: refuse loudly ──────────────────────────────────────


def test_suspect_provenance_raises_by_default():
    orch = _bare_orchestrator("suspect", False)
    with pytest.raises(RuntimeError, match="no reliable signal"):
        orch._enforce_probing_signal_gate()


def test_degenerate_weights_raises_by_default():
    orch = _bare_orchestrator("measured", True)
    with pytest.raises(RuntimeError, match="no reliable signal"):
        orch._enforce_probing_signal_gate()


def test_both_degenerate_signals_raises_by_default():
    orch = _bare_orchestrator("suspect", True)
    with pytest.raises(RuntimeError):
        orch._enforce_probing_signal_gate()


# ── Healthy / documented-heuristic provenance must NOT be gated ──────────


@pytest.mark.parametrize("provenance", ["measured", "partial", "heuristic", "unknown"])
def test_non_degenerate_provenance_does_not_raise(provenance):
    orch = _bare_orchestrator(provenance, False)
    orch._enforce_probing_signal_gate()  # must not raise


# ── Explicit override ────────────────────────────────────────────────────


def test_override_env_var_allows_suspect_to_proceed(monkeypatch, capsys):
    # MagicQuant logs via structlog's PrintLoggerFactory (prints straight to
    # stdout), not stdlib logging -- caplog can't see it, so assert on
    # captured stdout instead.
    monkeypatch.setenv("MAGICQUANT_ALLOW_DEGENERATE_PROBING", "1")
    orch = _bare_orchestrator("suspect", False)
    orch._enforce_probing_signal_gate()  # must not raise

    out = capsys.readouterr().out
    assert "no reliable signal" in out, (
        "override path must still log a WARNING naming the degradation"
    )


def test_override_env_var_allows_degenerate_weights_to_proceed(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_ALLOW_DEGENERATE_PROBING", "1")
    orch = _bare_orchestrator("measured", True)
    orch._enforce_probing_signal_gate()  # must not raise


def test_override_env_var_wrong_value_still_raises(monkeypatch):
    """Only the exact string "1" opts in -- guards against a truthy-but-
    wrong value (e.g. "true", "yes") silently doing nothing and the caller
    mistakenly believing they'd overridden the gate."""
    monkeypatch.setenv("MAGICQUANT_ALLOW_DEGENERATE_PROBING", "true")
    orch = _bare_orchestrator("suspect", False)
    with pytest.raises(RuntimeError):
        orch._enforce_probing_signal_gate()


def test_override_env_var_absent_still_raises(monkeypatch):
    monkeypatch.delenv("MAGICQUANT_ALLOW_DEGENERATE_PROBING", raising=False)
    orch = _bare_orchestrator("suspect", False)
    with pytest.raises(RuntimeError):
        orch._enforce_probing_signal_gate()


# ── Checkpoint round-trip: weights_degenerate must survive resume too ────


def test_weights_degenerate_persisted_and_restored_across_checkpoint(tmp_path):
    """Mirrors the BLOCKER fix's shape for a different field: a resumed
    run must see the SAME weights_degenerate signal a fresh run would have,
    or a degenerate-but-resumed search could sail through the gate that a
    fresh run of the same search would have refused."""
    from pathlib import Path as _P

    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._search_seed = 42
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch.weights_degenerate = True
    orch.sensitivity_weights = {"E": 1.0}
    orch._kl_base_logits_path = None
    orch._kl_corpus_path = None
    orch.source_model_path = str(tmp_path / "m.gguf")
    _P(orch.source_model_path).write_bytes(b"g")
    orch._llama_tools = None
    orch._llamacpp_path = None
    orch._imatrix = None
    orch._measured = {}

    ckpt_path = tmp_path / "ckpt.json"
    orch._write_measured_checkpoint(ckpt_path)

    import json
    checkpoint = json.loads(ckpt_path.read_text())
    assert checkpoint["weights_degenerate"] is True

    orch2 = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch2.probing_provenance = checkpoint["probing_provenance"]
    orch2.weights_degenerate = checkpoint.get("weights_degenerate", False)
    with pytest.raises(RuntimeError):
        orch2._enforce_probing_signal_gate()
