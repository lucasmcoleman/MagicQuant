"""Regression test for the MINOR fix to
MagicQuantOrchestrator._safe_resolve_corpus: it used to swallow ALL
exceptions -- including the pin-violation RuntimeError
LlamaCppTools._resolve_data_file raises when the auto-resolved corpus
disagrees with what it pinned earlier in the run -- turning the corpus into
None and silently voiding this (and every future) resume's mismatch check.
"""
import magicquant.orchestrator as orch_mod
from magicquant.orchestrator import MagicQuantOrchestrator


class _FlippingLlamaTools:
    """First call succeeds; every call after ``flip()`` raises the same
    RuntimeError LlamaCppTools._resolve_data_file raises on a pin
    violation."""

    def __init__(self, first_value):
        self._value = first_value
        self._raise = False

    def flip(self):
        self._raise = True

    def _resolve_data_file(self, data_file=None):
        if self._raise:
            raise RuntimeError(
                "PPL corpus resolution changed mid-run: this LlamaCppTools "
                "instance pinned '/a.txt' at first use, but a later "
                "auto-resolution now produces '/b.txt'."
            )
        return self._value


def _bare_orchestrator(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._llamacpp_path = None
    orch._last_resolved_corpus = None
    return orch


def test_pin_violation_preserves_last_known_good_value_instead_of_none(tmp_path, capsys):
    orch = _bare_orchestrator(tmp_path)
    tools = _FlippingLlamaTools("/real/corpus.txt")
    orch._llama_tools = tools

    first = orch._safe_resolve_corpus()
    assert first == "/real/corpus.txt"

    tools.flip()  # simulate the pin violation on a later call
    second = orch._safe_resolve_corpus()

    assert second == "/real/corpus.txt", (
        "a pin-violation RuntimeError must not silently turn the corpus "
        "into None -- it must preserve the last known-good value"
    )


def test_pin_violation_logs_error_not_silence(tmp_path, capsys):
    orch = _bare_orchestrator(tmp_path)
    tools = _FlippingLlamaTools("/real/corpus.txt")
    orch._llama_tools = tools
    orch._safe_resolve_corpus()  # prime _last_resolved_corpus

    tools.flip()
    orch._safe_resolve_corpus()

    out = capsys.readouterr().out
    assert "pin violation" in out.lower() or "corpus resolution failed" in out.lower()


def test_no_llama_tools_returns_none_without_raising(tmp_path, monkeypatch):
    # Force the llama_tools property to report "unavailable" deterministically
    # (rather than relying on real LlamaCppTools construction failing, which
    # depends on this box's environment).
    monkeypatch.setattr(orch_mod.MagicQuantOrchestrator, "llama_tools", property(lambda self: None))
    orch = _bare_orchestrator(tmp_path)
    orch._llama_tools = None  # llama.cpp unavailable
    assert orch._safe_resolve_corpus() is None


def test_healthy_repeated_calls_return_same_value(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    orch._llama_tools = _FlippingLlamaTools("/real/corpus.txt")
    assert orch._safe_resolve_corpus() == "/real/corpus.txt"
    assert orch._safe_resolve_corpus() == "/real/corpus.txt"
