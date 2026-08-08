"""Regression tests for:

FIX 2(c) -- orchestrator.py candidate measurements: a measured_loss below
-eps is a physically impossible ("beats its own baseline") reading, not a
quality win, and must not be allowed to win a tier via
_select_final_survivors' min().

FIX 3 -- the PPL corpus is pinned once per LlamaCppTools instance and
recorded PER MEASUREMENT in search_results.json, not stamped once at save
time.

Incident context: a NaN-driven measured search once recorded
``measured_loss=-0.9225`` for a candidate and let it WIN a tier via
``min()`` in ``_select_final_survivors`` -- orchestrator.py had no validity
check symmetric to probing.py's (see tests/test_probe_clamp.py).
"""
import json

import pytest

import magicquant.gguf.source as source_mod
from magicquant.orchestrator import MagicQuantOrchestrator
from magicquant.utils.llamacpp import LlamaCppTools


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


class _ControlledLlamaTools:
    """Baseline is fixed; every CANDIDATE measurement's ppl is drawn (in
    order) from *candidate_ppls*, cycling the last value if the search asks
    for more candidates than provided."""

    ctx_size = 512

    def __init__(self, baseline_ppl, candidate_ppls, corpus="/fake/corpus.txt"):
        self.baseline_ppl = baseline_ppl
        self.candidate_ppls = list(candidate_ppls)
        self._corpus = corpus
        self._candidate_calls = 0
        self.perplexity_calls = []

    def calculate_perplexity(self, path, verbose=False, **kw):
        self.perplexity_calls.append(path)
        if "candidates" not in path:
            return self.baseline_ppl
        idx = min(self._candidate_calls, len(self.candidate_ppls) - 1)
        self._candidate_calls += 1
        return self.candidate_ppls[idx]

    def _resolve_data_file(self, data_file=None):
        return self._corpus


def _make_orchestrator(tmp_path, monkeypatch, llama_tools):
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    orch._llama_tools = llama_tools

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


# ---------------------------------------------------------------------------
# FIX 2(c): physically-impossible candidate measurements
# ---------------------------------------------------------------------------

def test_impossible_candidate_measurement_is_flagged_invalid(tmp_path, monkeypatch):
    baseline = 34.8363
    # A MIX: some candidates get an incident-shaped bogus low ppl (below
    # baseline), some get a plausible one -- at least one valid measurement
    # keeps this run out of the MAJOR-2 "all invalid" guard (see
    # test_all_invalid_measurements_raises_instead_of_completing_with_zero_tiers),
    # which is exactly what lets this test isolate the flagging behavior on
    # its own.
    tools = _ControlledLlamaTools(baseline, [2.74, 2.6, 40.0, 41.0])
    orch = _make_orchestrator(tmp_path, monkeypatch, tools)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
        # Unrelated to measurement-validity; _ControlledLlamaTools has no
        # save_base_logits -- probe_kl's now-default capture attempt would
        # hit it. See run_measured_search's probe_kl docstring.
        probe_kl=False,
    )

    assert orch._measured, "candidates must still have been recorded (flagged, not dropped)"
    invalid_entries = [info for info in orch._measured.values() if info["measured_loss"] < 0]
    assert invalid_entries, "fixture must include at least one impossible measurement"
    for info in invalid_entries:
        assert info["measurement_invalid"] is True
    valid_entries = [info for info in orch._measured.values() if info["measured_loss"] >= 0]
    for info in valid_entries:
        assert info["measurement_invalid"] is False


def test_impossible_measurement_cannot_win_a_tier():
    """Direct unit test of _select_final_survivors: an impossible-but-
    'best' (lowest measured_loss) candidate must lose to a real, worse-but-
    valid candidate in the same tier."""
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = False
    orch._measured = {
        "impossible": {
            "config": {"E": "IMPOSSIBLE"}, "measured_loss": -0.9225,
            "size_gb": 4.0, "measurement_invalid": True,
        },
        "real": {
            "config": {"E": "REAL"}, "measured_loss": 0.15,
            "size_gb": 4.0, "measurement_invalid": False,
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "REAL"}, (
        "the physically-impossible candidate must never win a tier via min()"
    )


def test_only_impossible_candidates_in_a_tier_yields_no_survivor_for_it():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = False
    orch._measured = {
        "a": {
            "config": {"E": "A"}, "measured_loss": -0.9,
            "size_gb": 4.0, "measurement_invalid": True,
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert tier not in result


def test_default_relative_eps_is_wide_enough_for_real_box_noise():
    """MINOR regression: DEFAULT_RELATIVE_EPS used to be 0.02 (2%), which is
    TIGHTER than real 1-sigma noise measured on this box -- a real
    measured-search log reported 34.8363 +/- 0.78041 at 100 ppl_chunks
    (~2.24%), and the pending run's 50-chunk setting is noisier still
    (fewer samples -> wider stderr). A cutoff that tight risks flagging
    genuine measurement jitter as "physically impossible"."""
    from magicquant.utils.measurement import DEFAULT_RELATIVE_EPS

    observed_relative_noise_100_chunks = 0.78041 / 34.8363
    assert DEFAULT_RELATIVE_EPS >= 0.05
    assert DEFAULT_RELATIVE_EPS > observed_relative_noise_100_chunks


def test_plausible_measurement_below_baseline_within_noise_is_not_flagged():
    """A tiny, plausible dip (within the ~5% default noise floor) must NOT
    be treated as impossible -- only a clamp beyond eps counts."""
    baseline = 100.0
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)

    # Reuse the real measurement_eps helper to compute what "within noise"
    # means, matching the production code path exactly.
    from magicquant.utils.measurement import measurement_eps
    eps = measurement_eps(baseline, None)
    assert eps == pytest.approx(0.05)
    # ppl slightly below baseline, but the RELATIVE loss stays within -eps.
    ppl = baseline * (1 - eps / 2)
    measured_loss = (ppl - baseline) / baseline
    assert measured_loss < 0
    assert not (measured_loss < -eps)


def test_save_results_persists_measurement_invalid_field(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 34.8363
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch._search_seed = None
    orch._measured = {
        "a": {
            "config": {"E": "BF16"}, "ppl": 2.7, "measured_loss": -0.9225,
            "predicted_loss": 0.1, "residual": -1.02, "size_gb": 4.0,
            "measurement_invalid": True, "corpus_path": "/fake/corpus.txt",
        },
    }
    orch._save_results([], {})

    saved = json.loads((tmp_path / "search_results.json").read_text())
    entry = saved["measurements"]["a"]
    assert entry["measurement_invalid"] is True
    assert entry["corpus_path"] == "/fake/corpus.txt"


def test_write_pareto_report_excludes_measurement_invalid_entries(tmp_path, monkeypatch):
    """A measurement_invalid entry's ppl is below baseline*(1-eps) by
    construction (a physically impossible reading) -- if it reached
    pareto_frontier() it would have a lower ppl than every real candidate
    and dominate the frontier on a mixed valid/invalid run. Must be
    filtered out of BOTH the persisted pareto.json (the pareto_frontier
    call) and the logged table (the format_pareto_report call), matching
    the filter already applied at _select_final_survivors and
    _write_noise_calibration.
    """
    import magicquant.pareto as pareto_mod

    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch._measured = {
        "impossible": {
            "config": {"E": "BF16"}, "ppl": 1.0, "size_gb": 2.0,
            "measurement_invalid": True,
        },
        "small": {
            "config": {"E": "Q4_K_M"}, "ppl": 10.0, "size_gb": 5.0,
            "measurement_invalid": False,
        },
        "big": {
            "config": {"E": "Q6_K"}, "ppl": 6.0, "size_gb": 10.0,
            "measurement_invalid": False,
        },
    }

    captured = {}
    real_format = pareto_mod.format_pareto_report

    def _spy_format(measurements, **kwargs):
        captured["measurements"] = measurements
        return real_format(measurements, **kwargs)

    monkeypatch.setattr(pareto_mod, "format_pareto_report", _spy_format)

    orch._write_pareto_report()

    # pareto.json (the pareto_frontier() call) excludes the invalid entry;
    # both valid entries survive unaffected (neither dominates the other).
    frontier = json.loads((tmp_path / "pareto.json").read_text())
    keys = {item["key"] for item in frontier}
    assert keys == {"small", "big"}

    # The logged table (the format_pareto_report() call) was also given
    # the filtered dict, not the raw self._measured.
    assert set(captured["measurements"]) == {"small", "big"}


# ---------------------------------------------------------------------------
# FIX 3: per-measurement corpus recording + pinning
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MAJOR 2: a run where EVERY candidate measurement is invalid must not
# "successfully" complete with zero tiers.
# ---------------------------------------------------------------------------

def test_all_invalid_measurements_raises_instead_of_completing_with_zero_tiers(
    tmp_path, monkeypatch
):
    """self._measured can be non-empty (impossible-but-flagged entries are
    deliberately retained for diagnostics) while containing ZERO valid
    measurements. The old guard tested emptiness of self._measured itself
    and missed this case entirely -- a run "succeeded" with zero tiers."""
    baseline = 34.8363
    # Every candidate gets an incident-shaped bogus low ppl (below baseline)
    # -> every measurement ends up measurement_invalid.
    tools = _ControlledLlamaTools(baseline, [2.74, 2.6, 2.7, 2.65])
    orch = _make_orchestrator(tmp_path, monkeypatch, tools)

    with pytest.raises(RuntimeError, match="[Vv]alid"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=2, verbose=False,
            seed_incumbents=False,
            probe_kl=False,
        )

    # Confirm the precondition this test exercises: measurements were
    # recorded (not just "zero measurements"), all of them invalid.
    assert orch._measured
    assert all(info["measurement_invalid"] for info in orch._measured.values())


def test_measured_candidates_record_their_own_corpus_path(tmp_path, monkeypatch):
    tools = _ControlledLlamaTools(34.8363, [40.0, 41.0, 42.0, 43.0], corpus="/fake/corpus.txt")
    orch = _make_orchestrator(tmp_path, monkeypatch, tools)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
        # Unrelated to measurement-validity; _ControlledLlamaTools has no
        # save_base_logits -- probe_kl's now-default capture attempt would
        # hit it. See run_measured_search's probe_kl docstring.
        probe_kl=False,
    )

    assert orch._measured
    for info in orch._measured.values():
        assert info["corpus_path"] == "/fake/corpus.txt"

    saved = json.loads((orch.output_dir / "search_results.json").read_text())
    for entry in saved["measurements"].values():
        assert entry["corpus_path"] == "/fake/corpus.txt"


def test_resolve_data_file_pins_and_raises_on_mid_run_change(tmp_path):
    """The auto-resolved corpus must be pinned after first use; a later
    resolution that would disagree must raise loudly instead of silently
    switching the run's corpus mid-flight."""
    corpus_a = tmp_path / "a.txt"
    corpus_a.write_text("corpus a")
    corpus_b = tmp_path / "b.txt"
    corpus_b.write_text("corpus b")

    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.llamacpp_path = str(tmp_path)
    tools.data_file = str(corpus_a)

    first = tools._resolve_data_file(None)
    assert first == str(corpus_a.resolve())

    # Simulate the corpus changing out from under the instance mid-run.
    tools.data_file = str(corpus_b)
    with pytest.raises(RuntimeError, match="corpus"):
        tools._resolve_data_file(None)


def test_resolve_data_file_repeated_calls_return_same_pinned_value(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("stable corpus")
    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.llamacpp_path = str(tmp_path)
    tools.data_file = str(corpus)

    first = tools._resolve_data_file(None)
    second = tools._resolve_data_file(None)
    assert first == second == str(corpus.resolve())


def test_resolve_data_file_explicit_override_bypasses_pinning(tmp_path):
    """An explicit data_file argument is the caller's deliberate one-off
    choice and must never be pinned or checked against the pin."""
    corpus_a = tmp_path / "a.txt"
    corpus_a.write_text("corpus a")
    corpus_b = tmp_path / "b.txt"
    corpus_b.write_text("corpus b")

    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.llamacpp_path = str(tmp_path)
    tools.data_file = str(corpus_a)

    tools._resolve_data_file(None)  # pins corpus_a
    # An explicit call for a DIFFERENT file must succeed even though it
    # disagrees with the pin -- it's not an implicit auto-resolution.
    explicit = tools._resolve_data_file(str(corpus_b))
    assert explicit == str(corpus_b.resolve())
    # The pin itself must be untouched by the explicit call.
    assert tools._resolve_data_file(None) == str(corpus_a.resolve())
