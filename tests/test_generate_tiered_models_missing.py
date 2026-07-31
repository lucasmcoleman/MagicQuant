"""Regression tests: generate_tiered_models must treat an EXPLICITLY
REQUESTED tier that has no config as a loud, surfaced gap -- not a silent
INFO no-op.

Incident context: the v2 tier-boundary fix (magicquant.quant.tiers,
2026-07) narrowed several bands (e.g. Q5 from v1's (0.33, 0.45] down to
(0.328, 0.375]). Foundry's default tiers=["Q4", "Q5", "Q6"] can now come
back with a tier missing where it never would have under v1 -- and before
this fix that only ever produced a `log.info("No config for tier,
skipping", ...)` line, invisible to anything that doesn't tail the run's
logs (e.g. the publish path deciding what to disclose on a model card).

Note: orchestrator.py logs via magicquant.logging's structlog-based `log`
(unlike e.g. probing.py/writer.py, which use plain stdlib `logging` and can
be asserted on via pytest's `caplog`). structlog is left unconfigured in
tests (nothing calls magicquant.logging.configure_logging), so its default
backing is a PrintLogger, NOT stdlib logging -- caplog never sees it. These
tests monkeypatch the module-level `log` object directly instead.
"""
import json

import pytest

import magicquant.orchestrator as orch_mod
from magicquant.orchestrator import MagicQuantOrchestrator
from magicquant.quant.tiers import describe_tier_band


class _FakeLog:
    """Records (level, event, kwargs) for every structlog-style call
    (log.info(event, **kw) / log.warning(event, **kw) / ...)."""

    def __init__(self):
        self.calls = []

    def _record(self, level):
        def _call(event, **kw):
            self.calls.append((level, event, kw))
        return _call

    def __getattr__(self, level):
        return self._record(level)


@pytest.fixture
def fake_log(monkeypatch):
    fl = _FakeLog()
    monkeypatch.setattr(orch_mod, "log", fl)
    return fl


def _make_orch(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.generate_hybrid_model = lambda **kw: str(tmp_path / "fake.gguf")
    return orch


def _tiered_with_only(*tiers):
    return {
        t: {"config": {"E": t}, "ppl": 5.0, "measured_loss": 0.01}
        for t in tiers
    }


def test_missing_requested_tier_logs_warning_not_info(tmp_path, fake_log):
    """Proven to fail pre-fix: the old code only had
    ``log.info("No config for tier, skipping", ...)`` for a missing tier,
    so no WARNING (or higher) call was ever made for a gap in the caller's
    explicitly requested tier list.
    """
    orch = _make_orch(tmp_path)
    tiered = _tiered_with_only("Q4", "Q6")  # Q5 missing
    orch.generate_tiered_models(
        tiered=tiered, model_name_prefix="Model",
        tiers=["Q4", "Q5", "Q6"],
    )
    warning_calls = [c for c in fake_log.calls if c[0] == "warning"]
    assert any(c[2].get("tier") == "Q5" for c in warning_calls), (
        f"expected a WARNING naming the missing 'Q5' tier, got: {fake_log.calls}"
    )


def test_missing_requested_tier_names_the_band(tmp_path, fake_log):
    orch = _make_orch(tmp_path)
    tiered = _tiered_with_only("Q4", "Q6")  # Q5 missing
    orch.generate_tiered_models(
        tiered=tiered, model_name_prefix="Model",
        tiers=["Q4", "Q5", "Q6"],
    )
    level, event, kw = next(c for c in fake_log.calls if c[0] == "warning")
    assert kw.get("band") == describe_tier_band("Q5") == "(0.328, 0.375]"


def test_missing_requested_tier_does_not_hard_fail(tmp_path):
    """A genuinely empty band is legitimate -- the other requested tiers
    must still generate."""
    orch = _make_orch(tmp_path)
    tiered = _tiered_with_only("Q4", "Q6")  # Q5 missing
    generated = orch.generate_tiered_models(
        tiered=tiered, model_name_prefix="Model",
        tiers=["Q4", "Q5", "Q6"],
    )
    assert len(generated) == 2  # Q4 and Q6 still generated


def test_missing_requested_tier_surfaced_in_search_results_json(tmp_path):
    """Proven to fail pre-fix: search_results.json never gained any field
    recording a requested-but-empty tier -- a downstream reader (e.g. the
    publish path) had no way to see the gap short of re-deriving it from
    logs."""
    orch = _make_orch(tmp_path)
    # Simulate the search_results.json that _save_results would already
    # have written earlier in the run, before generate_tiered_models runs.
    results_path = tmp_path / "search_results.json"
    results_path.write_text(json.dumps({"baseline_ppl": 6.78, "tiered": {}}))

    tiered = _tiered_with_only("Q4", "Q6")  # Q5 missing
    orch.generate_tiered_models(
        tiered=tiered, model_name_prefix="Model",
        tiers=["Q4", "Q5", "Q6"],
    )

    written = json.loads(results_path.read_text())
    assert written["requested_tiers_missing"] == [
        {"tier": "Q5", "band": describe_tier_band("Q5")}
    ]
    # The rest of the previously-written file must survive untouched.
    assert written["baseline_ppl"] == 6.78


def test_no_missing_tiers_does_not_touch_search_results_json(tmp_path):
    """When every requested tier had a config, search_results.json is left
    exactly as-is (no spurious empty 'requested_tiers_missing' key, no
    rewrite at all)."""
    orch = _make_orch(tmp_path)
    results_path = tmp_path / "search_results.json"
    original = json.dumps({"baseline_ppl": 6.78, "tiered": {}})
    results_path.write_text(original)

    tiered = _tiered_with_only("Q4", "Q5", "Q6")
    orch.generate_tiered_models(
        tiered=tiered, model_name_prefix="Model",
        tiers=["Q4", "Q5", "Q6"],
    )

    assert results_path.read_text() == original


def test_missing_tier_surfacing_is_best_effort_when_file_absent(tmp_path):
    """generate_tiered_models can be invoked standalone against a directory
    with no search_results.json (see magicquant/__main__.py's `generate`
    subcommand) -- recording the gap must degrade gracefully, not raise,
    since the GGUF files for the tiers that DID resolve were already
    generated successfully."""
    orch = _make_orch(tmp_path)  # no search_results.json written
    tiered = _tiered_with_only("Q4", "Q6")
    generated = orch.generate_tiered_models(
        tiered=tiered, model_name_prefix="Model",
        tiers=["Q4", "Q5", "Q6"],
    )
    assert len(generated) == 2
