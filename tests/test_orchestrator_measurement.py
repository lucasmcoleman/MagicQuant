"""Orchestrator wiring for the deferred hardware-gated items: imatrix reuse,
KL-divergence, and speed-bench measurement inside run_measured_search's
build/measure loop, plus run_full_search's imatrix symmetry.

Real EvolutionarySurvivor/PredictiveScorer/SensitivityProber run unmocked
(same objects test_refactor_regression.py exercises); only the I/O boundary
(model source, llama.cpp tools, candidate GGUF building) is faked, so this
exercises the actual wiring rather than a hand-rolled stand-in.
"""
import json
import subprocess

import pytest

import magicquant.gguf.source as source_mod
import magicquant.orchestrator as orch_mod
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


class _FakeLlamaTools:
    """Stands in for LlamaCppTools -- no real llama.cpp binary involved."""

    def __init__(self):
        self.ctx_size = 512
        self.kl_calls = []
        self.bench_calls = []
        self.perplexity_calls = []
        # calculate_kl_divergence's canned result -- override per-test to
        # exercise the fused-ppl / missing-ppl / fallback paths.
        self.kl_result = {"mean_kl": 0.01}

    def calculate_perplexity(self, path, verbose=False, **kw):
        self.perplexity_calls.append(path)
        return 5.0

    def _resolve_data_file(self, data_file=None):
        return "/fake/corpus.txt"

    def save_base_logits(self, base_model_path, corpus_path, out_logits_path, **kw):
        from pathlib import Path
        Path(out_logits_path).write_text("fake logits" * 1000)
        # This pass's own "Final estimate: PPL" -- distinct from the
        # standalone calculate_perplexity's 5.0 so fusion-vs-standalone is
        # unambiguous in assertions.
        return 5.0

    def calculate_kl_divergence(self, quant_model_path, base_logits_path, corpus_path, **kw):
        self.kl_calls.append((quant_model_path, base_logits_path, corpus_path))
        return self.kl_result

    def bench(self, model_path, **kw):
        self.bench_calls.append(model_path)
        return {"pp_tps": 100.0, "tg_tps": 50.0}


class _TimeoutAwareFakeLlamaTools(_FakeLlamaTools):
    """Reproduces LlamaCppTools._run_subprocess_or_none's REAL contract for
    a subprocess timeout: caught internally, surfaced as a return of None
    PLUS ``self._last_subprocess_failure = {"kind": "timeout", ...}`` --
    never raised up to the orchestrator (see that method's docstring). A
    fake that instead literally raised subprocess.TimeoutExpired would be
    unfaithful to production, where TimeoutExpired never escapes
    _run_subprocess_or_none at all.
    """

    def __init__(self, kl_times_out=False, ppl_times_out=False):
        super().__init__()
        self._kl_times_out = kl_times_out
        self._ppl_times_out = ppl_times_out
        self._last_subprocess_failure = None

    def calculate_kl_divergence(self, quant_model_path, base_logits_path, corpus_path, **kw):
        self.kl_calls.append((quant_model_path, base_logits_path, corpus_path))
        if self._kl_times_out:
            self._last_subprocess_failure = {
                "kind": "timeout", "label": "KL divergence calculation",
            }
            return None
        self._last_subprocess_failure = None
        return self.kl_result

    def calculate_perplexity(self, path, verbose=False, **kw):
        self.perplexity_calls.append(path)
        if self._ppl_times_out:
            self._last_subprocess_failure = {
                "kind": "timeout", "label": "Perplexity calculation",
            }
            return None
        self._last_subprocess_failure = None
        return 5.0


class _MixedTimeoutFakeLlamaTools(_FakeLlamaTools):
    """The FIRST candidate-level calculate_perplexity call succeeds
    normally (a real, fully-scored measurement, "residual" and all);
    every call after that times out on the PPL leg (KL stays disabled by
    the caller, so only the PPL leg matters) -- a MIXED round: one
    normally-scored entry alongside one-or-more measurement-timeout
    disclosure entries that have no "residual" at all. This is the exact
    self._measured shape that broke the unguarded "Round summary"
    ``abs(m["residual"]) for m in self._measured.values()`` sum (Q1
    regression, Opus review 2026-08-10)."""

    def __init__(self):
        super().__init__()
        self._call_n = 0
        self._last_subprocess_failure = None

    def calculate_perplexity(self, path, verbose=False, **kw):
        self.perplexity_calls.append(path)
        self._call_n += 1
        if self._call_n == 1:
            self._last_subprocess_failure = None
            return 5.0
        self._last_subprocess_failure = {
            "kind": "timeout", "label": "Perplexity calculation",
        }
        return None


def _make_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    fake_tools = _FakeLlamaTools()
    orch._llama_tools = fake_tools

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    counter = {"n": 0}

    def fake_build_candidate(config, name, base_quant):
        counter["n"] += 1
        p = candidates_dir / f"{name}_{counter['n']}.gguf"
        p.write_bytes(b"0" * 1024)
        return str(p)

    monkeypatch.setattr(orch, "_build_candidate", fake_build_candidate)
    return orch, fake_tools


def test_measured_search_records_kl_and_bench_per_candidate(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    all_configs, tiered = orch.run_measured_search(
        search_generations=2,
        population_size=8,
        measurement_rounds=1,
        candidates_per_round=2,
        verbose=False,
        enable_kl=True,
        enable_speed_bench=True,
    )

    assert orch._measured, "expected at least one measured candidate"
    for info in orch._measured.values():
        assert info["kl"] == {"mean_kl": 0.01}
        assert info["bench"] == {"pp_tps": 100.0, "tg_tps": 50.0}
    assert fake_tools.kl_calls
    assert fake_tools.bench_calls

    results = json.loads((orch.output_dir / "search_results.json").read_text())
    saved = next(iter(results["measurements"].values()))
    assert saved["kl"] == {"mean_kl": 0.01}
    assert saved["bench"] == {"pp_tps": 100.0, "tg_tps": 50.0}


def test_measured_search_default_has_no_kl_or_bench(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
    )

    assert orch._measured
    for info in orch._measured.values():
        assert "kl" not in info
        assert "bench" not in info


# ── Feature 1+2: PPL+KL fusion (candidate) and baseline+logits fusion ──────


def test_measured_search_full_fusion_skips_standalone_perplexity_entirely(tmp_path, monkeypatch):
    """When enable_kl succeeds end-to-end -- save_base_logits returns a ppl,
    and every candidate's calculate_kl_divergence result carries "ppl" --
    calculate_perplexity must never be called at all: not for the baseline
    (fused via save_base_logits) and not per-candidate (fused via
    calculate_kl_divergence), collapsing 2 llama-perplexity invocations per
    measurement into 1."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.kl_result = {"mean_kl": 0.01, "ppl": 6.0, "ppl_err": 0.1}

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        enable_kl=True,
    )

    assert orch.baseline_ppl == pytest.approx(5.0)  # from save_base_logits, not calculate_perplexity
    assert orch.baseline_provenance == "measured"
    assert orch._measured
    for info in orch._measured.values():
        assert info["ppl"] == pytest.approx(6.0)  # from the fused KL call
        assert info["kl"]["ppl"] == pytest.approx(6.0)
    assert fake_tools.perplexity_calls == [], "calculate_perplexity must never be called when fully fused"
    assert fake_tools.kl_calls  # the fused call did happen


def test_measured_search_kl_missing_ppl_falls_back_to_calculate_perplexity(tmp_path, monkeypatch):
    """KL succeeds but its result lacks "ppl" (e.g. an older/unexpected
    output format) -- must fall back to a standalone calculate_perplexity
    call per candidate, while the baseline (independent of per-candidate KL
    content) still fuses via save_base_logits."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.kl_result = {"mean_kl": 0.01}  # no "ppl"

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        enable_kl=True,
    )

    assert orch.baseline_ppl == pytest.approx(5.0)
    assert orch._measured
    assert fake_tools.perplexity_calls, "fallback calculate_perplexity must have run per candidate"
    for info in orch._measured.values():
        assert info["ppl"] == pytest.approx(5.0)  # from the calculate_perplexity fallback
        assert info["kl"] == {"mean_kl": 0.01}


def test_measured_search_kl_call_raising_falls_back_and_records_no_kl(tmp_path, monkeypatch):
    """calculate_kl_divergence raising outright (e.g. OSError from a
    missing/wrong-arch binary) must fall back to calculate_perplexity for
    ppl and record no "kl" field -- the existing "KL failure must not
    abort/win" guarantee, now expressed through the fused code path."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    def raise_oserror(*a, **k):
        raise FileNotFoundError("llama-perplexity: exec format error")

    monkeypatch.setattr(fake_tools, "calculate_kl_divergence", raise_oserror)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        enable_kl=True,
    )

    assert orch._measured
    assert fake_tools.perplexity_calls
    for info in orch._measured.values():
        assert info["ppl"] == pytest.approx(5.0)
        assert "kl" not in info


def test_measured_search_save_base_logits_failure_falls_back_to_standalone_baseline(tmp_path, monkeypatch):
    """If save_base_logits fails (returns None), the fused baseline attempt
    must fall back to the historical standalone baseline pass, and KL
    scoring must be disabled entirely for the run (no per-candidate KL
    calls, matching the pre-fusion "disabling KL-divergence scoring"
    warning path)."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(fake_tools, "save_base_logits", lambda *a, **k: None)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        enable_kl=True,
    )

    assert orch.baseline_ppl == pytest.approx(5.0)
    assert orch.baseline_provenance == "measured"
    assert orch.source_model_path in fake_tools.perplexity_calls, (
        "standalone baseline fallback must have called calculate_perplexity on the source model"
    )
    assert orch._kl_base_logits_path is None
    assert not fake_tools.kl_calls
    for info in orch._measured.values():
        assert "kl" not in info


# ── probe_kl vs enable_kl split (KL probe-scoring vs KL candidate objective,
# ── independent knobs since probe_kl defaults True) ─────────────────────────


def test_probe_kl_default_wires_kl_base_logits_to_prober_without_objective_blend(
    tmp_path, monkeypatch
):
    """probe_kl defaults True: base-logits capture succeeds and reaches the
    sensitivity prober (Step 2), but the candidate OBJECTIVE stays PPL-only
    because enable_kl was never passed -- self._kl_weight stays 0.0. The two
    knobs must not leak into each other."""
    import magicquant.orchestrator as orch_mod

    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    captured_prober_kwargs = {}
    real_prober_cls = orch_mod.SensitivityProber

    class _SpyProber(real_prober_cls):
        def __init__(self, **kwargs):
            captured_prober_kwargs.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(orch_mod, "SensitivityProber", _SpyProber)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    # Capture happened (probe_kl's default request) and produced a real path...
    assert orch._kl_base_logits_path is not None
    assert orch._kl_capture_requested is True
    assert orch._kl_capture_failed is False
    # ... which reached the prober ...
    assert captured_prober_kwargs.get("kl_base_logits_path") == orch._kl_base_logits_path
    # ... but the candidate objective never blended KL: enable_kl was off.
    assert orch._kl_weight == 0.0
    for info in orch._measured.values():
        assert "kl" not in info, "per-candidate KL objective measurement must not run"


def test_enable_kl_still_blends_objective_alongside_default_probe_kl(tmp_path, monkeypatch):
    """enable_kl=True keeps blending kl_weight into the candidate objective
    exactly as before, unaffected by probe_kl now defaulting on alongside
    it -- the two knobs share one capture pass but stay functionally
    independent."""
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
        enable_kl=True, kl_weight=0.3,
    )

    assert orch._kl_weight == 0.3
    assert orch._measured
    for info in orch._measured.values():
        assert info["kl"] == {"mean_kl": 0.01}, (
            "enable_kl's per-candidate KL objective measurement must still run"
        )


def test_probe_kl_only_capture_failure_warns_and_falls_back_to_ppl_probes(
    tmp_path, monkeypatch, capsys
):
    """probe_kl is requested (the default) but base-logits capture is
    impossible (no calibration corpus resolved) -- must log a warning
    naming probe_kl and complete the search via raw-PPL probe scoring, NOT
    raise. This is the historical "enable_kl requested but no calibration
    corpus resolved" warning's sibling for the now-default probe_kl path;
    unlike an explicit enable_kl=True capture failure, nothing here claims
    an objective blend that then silently fails to happen."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(fake_tools, "_resolve_data_file", lambda *a, **k: None)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )  # must not raise

    assert orch._kl_base_logits_path is None
    assert orch._kl_capture_requested is True
    assert orch._kl_capture_failed is True
    assert orch._kl_weight == 0.0
    assert orch._measured, "search must still complete via raw-PPL probe fallback"

    out = capsys.readouterr().out
    assert "probe_kl" in out
    assert "fall back" in out.lower()


# ── Feature 3: one-ahead build/measure overlap ──────────────────────────────


def test_measured_search_overlaps_build_with_measurement(tmp_path, monkeypatch):
    """While candidate i's (GPU-bound) measurement runs, candidate i+1's
    (CPU-bound) build must already be underway on a background thread --
    not waiting for i's measurement to finish first."""
    import threading
    import time as _time

    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )

    events = []
    lock = threading.Lock()

    def log_event(name):
        with lock:
            events.append(name)

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    counter = {"n": 0}

    def fake_build_candidate(config, name, base_quant):
        log_event(f"build_start:{name}")
        counter["n"] += 1
        p = candidates_dir / f"{name}_{counter['n']}.gguf"
        p.write_bytes(b"0" * 1024)
        log_event(f"build_end:{name}")
        return str(p)

    monkeypatch.setattr(orch, "_build_candidate", fake_build_candidate)

    class _FakeLlamaToolsOverlap:
        ctx_size = 512

        def calculate_perplexity(self, path, verbose=False, **kw):
            # The baseline pass (Step 1c) also calls calculate_perplexity,
            # over the source model rather than a candidate GGUF -- exclude
            # it from the logged events so the overlap assertions below only
            # reason about per-candidate measurements.
            is_candidate = str(candidates_dir) in path
            if is_candidate:
                log_event(f"measure_start:{path}")
            # Measurement (GPU-bound subprocess) is much slower than a build
            # (CPU-bound, near-instant fake) -- gives the prefetch a real
            # window to demonstrably start/finish inside it.
            _time.sleep(0.2)
            if is_candidate:
                log_event(f"measure_end:{path}")
            return 5.0

        def _resolve_data_file(self, data_file=None):
            return "/fake/corpus.txt"

    orch._llama_tools = _FakeLlamaToolsOverlap()

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=3, verbose=False,
        seed_incumbents=False,
        # _FakeLlamaToolsOverlap has no save_base_logits -- probe_kl's
        # now-default capture attempt would hit it; unrelated to this test.
        probe_kl=False,
    )

    assert len(orch._measured) >= 2, "need at least 2 real candidates to observe overlap"

    build_starts = [e for e in events if e.startswith("build_start:")]
    build_ends = [e for e in events if e.startswith("build_end:")]
    measure_starts = [e for e in events if e.startswith("measure_start:")]
    measure_ends = [e for e in events if e.startswith("measure_end:")]
    assert len(build_starts) >= 2
    assert len(measure_ends) >= 2

    # Core overlap assertion: the SECOND build (the one-ahead prefetch for
    # candidate 2) must both start AND finish before the FIRST measurement
    # completes -- proving it ran concurrently with that measurement rather
    # than serially after it.
    idx_measure_end_1 = events.index(measure_ends[0])
    idx_build_start_2 = events.index(build_starts[1])
    idx_build_end_2 = events.index(build_ends[1])
    assert idx_build_start_2 < idx_measure_end_1
    assert idx_build_end_2 < idx_measure_end_1

    # Results ordering identical to serial: measurements are still consumed
    # on the main thread in candidate order (1, 2, 3, ...), never reordered.
    assert measure_starts == sorted(measure_starts, key=events.index)
    assert [e.split(":", 1)[1] for e in measure_starts] == sorted(
        (e.split(":", 1)[1] for e in measure_starts), key=lambda p: events.index(f"measure_start:{p}")
    )


def test_measured_search_overlap_build_failure_for_one_candidate_only_skips_that_one(tmp_path, monkeypatch):
    """A build failure for any one candidate (whether it happened to be
    prefetched ahead or not) must surface exactly like the historical
    serial loop: logged and skipped, without killing the rest of the
    search."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    real_build = orch._build_candidate

    def flaky_build(config, name, base_quant):
        if "candidate2" in name:
            return None
        return real_build(config, name, base_quant)

    monkeypatch.setattr(orch, "_build_candidate", flaky_build)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=3, verbose=False,
        seed_incumbents=False,
    )

    # The other candidates must still have been measured despite candidate
    # 2's build failing.
    assert len(orch._measured) >= 2


def test_measured_search_overlap_cleans_up_prefetched_build_on_exception(tmp_path, monkeypatch):
    """If something raises while processing candidate i (after its build
    already prefetched candidate i+1's), the in-flight/finished prefetch
    build for i+1 must be joined and its candidate GGUF deleted -- the
    candidates dir must never leak a build that was never consumed.

    (Candidate i's OWN file may still leak in this scenario -- its cleanup
    line never runs when an exception fires before reaching it, exactly
    like the pre-existing serial loop. Only the ONE-AHEAD prefetch's
    cleanup is this feature's contract.)
    """
    from magicquant.evolution.predictor import PredictiveScorer

    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    built_paths = []
    real_build = orch._build_candidate

    def tracking_build(config, name, base_quant):
        p = real_build(config, name, base_quant)
        built_paths.append(p)
        return p

    monkeypatch.setattr(orch, "_build_candidate", tracking_build)

    calls = {"n": 0}
    original_record_residual = PredictiveScorer.record_residual

    def flaky_record_residual(self, config, residual):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated mid-processing failure")
        return original_record_residual(self, config, residual)

    monkeypatch.setattr(PredictiveScorer, "record_residual", flaky_record_residual)

    with pytest.raises(RuntimeError, match="simulated mid-processing failure"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=3, verbose=False,
            seed_incumbents=False,
        )

    # The contract is LEAK-FREE, not "the prefetch physically ran": if the
    # exception reaches the finally before the single worker dequeues the
    # prefetch job, cancel() legitimately wins and no second build happens
    # (Opus review reproduced that ordering ~1 in 8 suite runs). Either way,
    # no unconsumed candidate GGUF may remain on disk.
    assert built_paths, "candidate i itself must have been built"
    leaked = [
        f for f in (tmp_path / "candidates").glob("*.gguf")
        if str(f) != built_paths[0]  # candidate i's own file may remain (pre-existing serial behavior)
    ]
    assert not leaked, f"prefetched (never-consumed) build(s) leaked: {leaked}"


def test_enable_imatrix_reaches_candidate_builds(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    fake_imatrix = {"blk.0.attn_q.weight": object()}
    monkeypatch.setattr(
        "magicquant.imatrix.ensure_imatrix", lambda *a, **k: fake_imatrix
    )

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        use_imatrix=True,
    )

    assert orch._imatrix is fake_imatrix


def test_run_full_search_use_imatrix_is_symmetric(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    fake_imatrix = {"blk.0.attn_q.weight": object()}
    monkeypatch.setattr(
        "magicquant.imatrix.ensure_imatrix", lambda *a, **k: fake_imatrix
    )

    orch.run_full_search(
        max_generations=2, population_size=8, verbose=False, use_imatrix=True,
    )

    assert orch._imatrix is fake_imatrix


def test_selection_score_plain_measured_loss_without_kl():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    info = {"measured_loss": 0.2}
    assert orch._selection_score(info) == 0.2


def test_selection_score_blends_kl_when_present():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.5
    info = {"measured_loss": 0.2, "kl": {"mean_kl": -0.1}}
    assert orch._selection_score(info) == 0.2 + 0.5 * abs(-0.1)


def test_selection_score_ignores_kl_without_mean_kl_key():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.5
    info = {"measured_loss": 0.2, "kl": {"mean_kl": None}}
    assert orch._selection_score(info) == 0.2


def test_select_final_survivors_kl_blend_changes_tier_winner():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 10.0  # dominate the ranking to make the effect obvious
    orch._measured = {
        "a": {
            "config": {"E": "BF16"}, "measured_loss": 0.10,
            "size_gb": 4.0, "kl": {"mean_kl": 0.5},
        },
        "b": {
            "config": {"E": "Q4_K"}, "measured_loss": 0.12,
            "size_gb": 4.0, "kl": {"mean_kl": 0.01},
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "Q4_K"}


def test_select_final_survivors_failed_kl_cannot_beat_worst_real_kl():
    # A candidate whose KL measurement failed ("kl" missing/None) must never
    # look better than the worst candidate that actually measured KL in the
    # same tier -- otherwise a measurement failure is rewarded over real
    # (if poor) data.
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 1.0
    orch._measured = {
        "failed": {
            "config": {"E": "FAILED"}, "measured_loss": 0.05, "size_gb": 4.0,
        },
        "good": {
            "config": {"E": "GOOD"}, "measured_loss": 0.05, "size_gb": 4.0,
            "kl": {"mean_kl": 0.01},
        },
        "bad": {
            "config": {"E": "BAD"}, "measured_loss": 0.05, "size_gb": 4.0,
            "kl": {"mean_kl": 0.5},
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "GOOD"}


def test_select_final_survivors_no_kl_data_falls_back_to_measured_loss():
    # When nothing in the tier has KL data (enable_kl=False, the default),
    # selection must be unaffected -- exactly the pre-KL behavior.
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._measured = {
        "a": {"config": {"E": "WORSE"}, "measured_loss": 0.20, "size_gb": 4.0},
        "b": {"config": {"E": "BETTER"}, "measured_loss": 0.05, "size_gb": 4.0},
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "BETTER"}


def test_measured_search_survives_kl_and_bench_raising_oserror(tmp_path, monkeypatch):
    # A missing/wrong-arch binary raises OSError/FileNotFoundError from
    # subprocess.run, which calculate_kl_divergence/bench don't catch
    # internally -- the orchestrator's measured-search loop must swallow it
    # per-candidate rather than aborting the whole search.
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    def raise_oserror(*a, **k):
        raise FileNotFoundError("llama-bench: exec format error")

    monkeypatch.setattr(fake_tools, "calculate_kl_divergence", raise_oserror)
    monkeypatch.setattr(fake_tools, "bench", raise_oserror)

    all_configs, tiered = orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        enable_kl=True, enable_speed_bench=True,
    )

    assert orch._measured, "search must still produce measured candidates"
    for info in orch._measured.values():
        assert "kl" not in info
        assert "bench" not in info


# ── Measurement-timeout recording (2026-08 field report) ────────────────────


def test_measured_search_records_kl_timeout_when_ppl_still_succeeds(tmp_path, monkeypatch):
    """The KL leg times out but the PPL fallback succeeds -- the candidate
    must still be scored (it has a real measured_loss and competes for its
    tier normally), but the record must show it ran WITHOUT the KL term
    its siblings carry (field report: previously invisible)."""
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch._llama_tools = _TimeoutAwareFakeLlamaTools(kl_times_out=True, ppl_times_out=False)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        enable_kl=True,
    )

    assert orch._measured, "PPL succeeded -- candidates must still be scored"
    for info in orch._measured.values():
        assert info.get("kl_timeout") is True
        assert "kl" not in info
        assert info.get("measured_loss") is not None
        assert not info.get("measurement_invalid")

    results = json.loads((orch.output_dir / "search_results.json").read_text())
    saved = next(iter(results["measurements"].values()))
    assert saved["kl_timeout"] is True
    # Tail-appended, per _serialize_measurement's key-order contract.
    keys = list(saved.keys())
    assert keys.index("kl_timeout") > keys.index("measurement_invalid")


def test_measured_search_records_measurement_timeout_disclosure_when_both_legs_time_out(
    tmp_path, monkeypatch,
):
    """Both the KL leg and the PPL fallback time out -- zero measurements
    are produced (matches the field report exactly: 4h of healthy compute,
    nothing measured), so run_measured_search must still raise the
    existing "zero VALID measurements" guard (F3) rather than reporting
    success. But self._measured must NOT stay empty: a minimal disclosure
    entry (measurement_invalid=True, so it is excluded from tier
    competition and from the VALID count -- inert to selection) is what
    distinguishes "measured and lost" from "never attempted"."""
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch._llama_tools = _TimeoutAwareFakeLlamaTools(kl_times_out=True, ppl_times_out=True)

    with pytest.raises(RuntimeError, match="zero VALID"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=2, verbose=False,
            enable_kl=True,
        )

    assert orch._measured, "timeout disclosure entries must still be recorded"
    for info in orch._measured.values():
        assert info["measurement_invalid"] is True
        assert info["measurement_timeout"] is True
        assert info["timeout_leg"] == "ppl"  # the PPL fallback is always the deciding leg
        assert info["kl_timeout"] is True  # the KL leg independently timed out too
        assert info["config"]
        assert info.get("ppl") is None

    # Persisted to the checkpoint (so a subsequent kill+resume doesn't
    # re-attempt a candidate already known to time out at this size) even
    # though the run as a whole raised before ever reaching _save_results.
    checkpoint = json.loads(orch._measured_checkpoint_path().read_text())
    entry = next(iter(checkpoint["measured"].values()))
    assert entry["measurement_timeout"] is True
    assert entry["timeout_leg"] == "ppl"
    assert entry["kl_timeout"] is True
    # Tail-appended, per the checkpoint site's key-order contract.
    keys = list(entry.keys())
    assert keys[-3:] == ["kl_timeout", "measurement_timeout", "timeout_leg"]


def test_measured_search_no_leaked_timeout_across_candidates(tmp_path, monkeypatch):
    """Q2 (Opus review, 2026-08-10): end-to-end reproduction of the
    _last_subprocess_failure leak through a real run_measured_search
    round. Uses a REAL LlamaCppTools instance (not the hand-rolled
    _FakeLlamaTools/_TimeoutAwareFakeLlamaTools doubles) because the bug
    lives inside calculate_perplexity's own control flow -- a fake
    reimplementing calculate_perplexity from scratch could not exercise
    the early-return path that causes the leak.

    "Candidate 1" genuinely times out (a mocked _run_perplexity_subprocess
    raises TimeoutExpired for real) and correctly gets a disclosure entry.
    "Candidate 2" is engineered to hit calculate_perplexity's own
    corpus-resolution early return (`if resolved_data_file is None: return
    None`) -- its OWN failure has nothing to do with a timeout, and
    _run_perplexity_subprocess is never even called for it. Pre-fix, it
    would still read candidate 1's STALE _last_subprocess_failure and get
    mislabeled+blacklisted as a timeout too; fixed, it gets no disclosure
    entry at all (matching a config that was never measured).
    """
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")

    real_tools = LlamaCppTools.__new__(LlamaCppTools)
    real_tools.perplexity_tool = "/bin/true"
    real_tools.bench_tool = None
    real_tools.ctx_size = 512
    real_tools.ngl = None
    real_tools.threads = None
    real_tools.ppl_chunks = None
    real_tools.data_file = str(corpus)
    real_tools._pinned_corpus = None
    orch._llama_tools = real_tools

    candidate_calls = {"n": 0}
    real_calculate_perplexity = LlamaCppTools.calculate_perplexity

    def controlled_calculate_perplexity(model_path, verbose=True, data_file=None, ctx_size=None):
        if model_path == orch.source_model_path:
            # Baseline: always resolves and measures normally.
            return real_calculate_perplexity(
                real_tools, model_path, verbose=verbose,
                data_file=data_file, ctx_size=ctx_size,
            )
        candidate_calls["n"] += 1
        if candidate_calls["n"] != 2:
            # Candidate 1 (and any candidate beyond 2): normal corpus
            # resolution -- whether the underlying subprocess then
            # succeeds or times out is controlled by fake_run_subprocess
            # below, keyed on candidate_calls["n"].
            return real_calculate_perplexity(
                real_tools, model_path, verbose=verbose,
                data_file=data_file, ctx_size=ctx_size,
            )
        # Candidate 2: force the early return specifically -- corpus is
        # unresolvable for THIS call only (e.g. a transient vanished-
        # corpus condition), never reaching _run_perplexity_subprocess.
        orig_resolve = real_tools._resolve_data_file
        real_tools._resolve_data_file = lambda data_file=None: None
        try:
            return real_calculate_perplexity(
                real_tools, model_path, verbose=verbose,
                data_file=data_file, ctx_size=ctx_size,
            )
        finally:
            real_tools._resolve_data_file = orig_resolve

    real_tools.calculate_perplexity = controlled_calculate_perplexity

    def fake_run_subprocess(cmd, timeout):
        model_path = cmd[cmd.index("-m") + 1]
        if model_path == orch.source_model_path:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="", stderr="Final estimate: PPL = 5.0 +/- 0.1",
            )
        if candidate_calls["n"] == 1:
            # Candidate 1's real, genuine timeout.
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        # Any later candidate that actually reaches the subprocess
        # (everyone except candidate 2, which is diverted above) measures
        # normally.
        return subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr="Final estimate: PPL = 6.0 +/- 0.1",
        )

    monkeypatch.setattr(real_tools, "_run_perplexity_subprocess", fake_run_subprocess)

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    build_counter = {"n": 0}

    def fake_build_candidate(config, name, base_quant):
        build_counter["n"] += 1
        p = candidates_dir / f"{name}_{build_counter['n']}.gguf"
        p.write_bytes(b"0" * 1024)
        return str(p)

    monkeypatch.setattr(orch, "_build_candidate", fake_build_candidate)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        probe_kl=False,  # isolate to the PPL leg only, per the bug's own repro
    )

    assert candidate_calls["n"] >= 2, "test setup requires at least 2 measured candidates"

    disclosure_entries = [
        info for info in orch._measured.values()
        if info.get("measurement_invalid") and info.get("measurement_timeout")
    ]
    # Exactly one real timeout occurred (candidate 1) -- candidate 2's
    # early return must NOT have produced a second (mislabeled) one.
    assert len(disclosure_entries) == 1
    assert disclosure_entries[0]["timeout_leg"] == "ppl"


def test_measured_search_no_disclosure_entry_for_non_timeout_ppl_failure(tmp_path, monkeypatch):
    """A plain (non-timeout) measurement failure -- e.g. unparseable
    output or a CalledProcessError -- must keep the EXACT pre-fix
    behavior: no self._measured entry at all, not a disclosure entry. The
    new disclosure mechanism is scoped to genuine subprocess TIMEOUTS
    only.

    Q3 (Opus review, 2026-08-10): the fake explicitly sets
    ``_last_subprocess_failure = {"kind": "error", ...}`` on BOTH legs
    (KL via ``enable_kl=True``, and the PPL fallback) rather than just
    leaving the attribute unset. An unset attribute would make
    ``kl_timed_out``/``ppl_timed_out`` False via the ``bool(None)``
    short-circuit alone, never actually exercising the
    ``kind == "timeout"`` comparison against a REAL (non-timeout) failure
    dict -- this pins that the comparison itself, not just attribute
    absence, correctly reads False for ``kind == "error"``.
    """
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    def flaky_perplexity(path, verbose=False, **kw):
        # Baseline is fused via save_base_logits (enable_kl=True triggers
        # the Step 1b fusion attempt, unaffected by this override), so
        # every call reaching this fake is a CANDIDATE call -- all fail,
        # non-timeout.
        fake_tools._last_subprocess_failure = {
            "kind": "error", "label": "Perplexity calculation",
        }
        return None

    def flaky_kl(quant_model_path, base_logits_path, corpus_path, **kw):
        fake_tools._last_subprocess_failure = {
            "kind": "error", "label": "KL divergence calculation",
        }
        return None

    monkeypatch.setattr(fake_tools, "calculate_perplexity", flaky_perplexity)
    monkeypatch.setattr(fake_tools, "calculate_kl_divergence", flaky_kl)

    with pytest.raises(RuntimeError, match="zero VALID"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=2, verbose=False,
            enable_kl=True,
        )

    assert orch._measured == {}, (
        "a non-timeout failure on both legs must leave no trace, matching pre-fix behavior"
    )


def test_round_summary_logged_with_mixed_scored_and_disclosure_entries(tmp_path, monkeypatch):
    """Q1 (Opus review, 2026-08-10): the round-summary log used to index
    every self._measured entry's "residual" unconditionally
    (``abs(m["residual"]) for m in self._measured.values()``) -- a
    measurement-timeout disclosure entry has no "residual" at all, so a
    MIXED round (one normally-scored candidate + one-or-more disclosure
    entries) with verbose=True used to KeyError inside the round-summary
    log call itself, an untested path (reviewer reverted the filtering
    fix and the suite stayed green). Captures the log via the established
    _FakeLog pattern (see its own docstring: structlog's real backing is
    a bare PrintLogger, invisible to caplog) to confirm the line is both
    emitted (not silently swallowed by an exception) and correct
    (mean_abs_residual reflects only the one scored entry's residual).
    """
    fake_log = _FakeLog()
    monkeypatch.setattr(orch_mod, "log", fake_log)

    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch._llama_tools = _MixedTimeoutFakeLlamaTools()

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=True,
    )

    scored = [
        info for info in orch._measured.values()
        if not info.get("measurement_invalid")
    ]
    disclosures = [
        info for info in orch._measured.values()
        if info.get("measurement_invalid") and info.get("measurement_timeout")
    ]
    assert len(scored) == 1, "test setup expects exactly one normally-scored candidate"
    assert disclosures, "test setup expects at least one timeout disclosure entry"

    round_summary_calls = [
        (event, kw) for level, event, kw in fake_log.calls if event == "Round summary"
    ]
    assert len(round_summary_calls) == 1, "the round-summary log call must not have been skipped/crashed"
    _, kw = round_summary_calls[0]
    assert kw["total_measurements"] == len(orch._measured)
    assert kw["mean_abs_residual"] == pytest.approx(round(abs(scored[0]["residual"]), 4))


def test_round_summary_mean_abs_residual_none_when_all_disclosure_entries(tmp_path, monkeypatch):
    """Q1 sibling: when EVERY entry in a verbose round is a measurement-
    timeout disclosure (no residuals at all), the round-summary line must
    still be logged -- not silently suppressed by an ``if residuals:``
    guard -- with mean_abs_residual=None, so round-level visibility
    doesn't drop in exactly the failure case this whole fix exists to
    surface. The run overall still raises the pre-existing zero-VALID-
    measurements guard (F3) since nothing was actually measured, but the
    round-summary log line fires (inside the round loop) before that
    guard is ever reached (after the loop)."""
    fake_log = _FakeLog()
    monkeypatch.setattr(orch_mod, "log", fake_log)

    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch._llama_tools = _TimeoutAwareFakeLlamaTools(kl_times_out=True, ppl_times_out=True)

    with pytest.raises(RuntimeError, match="zero VALID"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=2, verbose=True,
            enable_kl=True,
        )

    round_summary_calls = [
        (event, kw) for level, event, kw in fake_log.calls if event == "Round summary"
    ]
    assert len(round_summary_calls) == 1, (
        "the round-summary line must still fire even when every entry is a disclosure"
    )
    _, kw = round_summary_calls[0]
    assert kw["mean_abs_residual"] is None
    assert kw["total_measurements"] == len(orch._measured)


def test_serialize_measurement_appends_timeout_fields_at_tail_both_sites():
    v = {
        "config": {"E": "BF16"}, "ppl": 5.5, "measured_loss": 0.1,
        "predicted_loss": 0.09, "residual": 0.01, "path": "/tmp/c.gguf",
        "size_gb": 4.0, "kl": {"mean_kl": 0.02}, "bench": {"pp_tps": 90.0},
        "incumbent": "Q4", "corpus_path": "/fake/corpus.txt",
        "measurement_invalid": False,
        "kl_timeout": True, "measurement_timeout": False, "timeout_leg": None,
    }

    save_results_entry = MagicQuantOrchestrator._serialize_measurement(v, include_path=False)
    assert list(save_results_entry.keys()) == [
        "config", "ppl", "measured_loss", "predicted_loss", "residual",
        "size_gb", "kl", "bench", "incumbent", "corpus_path",
        "measurement_invalid", "kl_timeout", "measurement_timeout", "timeout_leg",
    ]
    assert save_results_entry["kl_timeout"] is True

    checkpoint_entry = MagicQuantOrchestrator._serialize_measurement(v, include_path=True)
    assert list(checkpoint_entry.keys()) == [
        "config", "ppl", "measured_loss", "predicted_loss", "residual",
        "path", "size_gb", "kl", "bench", "incumbent", "measurement_invalid",
        "corpus_path", "kl_timeout", "measurement_timeout", "timeout_leg",
    ]
    assert checkpoint_entry["kl_timeout"] is True


def test_serialize_measurement_timeout_fields_default_when_absent():
    """A pre-fix entry (no kl_timeout/measurement_timeout/timeout_leg at
    all) must serialize with the new fields defaulting to falsy/None, not
    KeyError -- the backward-compatibility guarantee for an OLDER
    checkpoint/search_results.json resuming under the fixed code."""
    v = {"config": {"E": "BF16"}, "measured_loss": 0.1, "size_gb": 4.0}
    entry = MagicQuantOrchestrator._serialize_measurement(v, include_path=False)
    assert entry["kl_timeout"] is False
    assert entry["measurement_timeout"] is False
    assert entry["timeout_leg"] is None


def test_timeout_disclosure_fields_survive_checkpoint_round_trip(tmp_path):
    """kl_timeout/measurement_timeout/timeout_leg must survive a checkpoint
    write+load round trip -- otherwise a resumed run silently loses the
    disclosure and re-attempts a candidate already known to time out."""
    from pathlib import Path

    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._search_seed = 42
    orch.baseline_ppl = 34.8363
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch.sensitivity_weights = {"E": 0.5}
    orch._kl_base_logits_path = None
    orch._kl_corpus_path = None
    orch.source_model_path = str(tmp_path / "m.gguf")
    Path(orch.source_model_path).write_bytes(b"g")
    orch._llama_tools = None
    orch._llamacpp_path = None
    orch._imatrix = None
    orch._measured = {
        "timed_out": {
            "config": {"E": "TIMEOUT"},
            "measurement_invalid": True,
            "measurement_timeout": True,
            "timeout_leg": "ppl",
            "kl_timeout": True,
        },
    }

    ckpt_path = tmp_path / "ckpt.json"
    orch._write_measured_checkpoint(ckpt_path)
    checkpoint = json.loads(ckpt_path.read_text())

    # Simulate exactly what run_measured_search's resume path does with a
    # loaded checkpoint (orchestrator.py's `for key, entry in
    # checkpoint.get("measured", {}).items(): self._measured[key] =
    # dict(entry)`).
    orch2 = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch2._measured = {}
    for key, entry in checkpoint.get("measured", {}).items():
        orch2._measured[key] = dict(entry)

    resumed = orch2._measured["timed_out"]
    assert resumed["measurement_timeout"] is True
    assert resumed["timeout_leg"] == "ppl"
    assert resumed["kl_timeout"] is True
    assert resumed["measurement_invalid"] is True


def test_pre_fix_checkpoint_without_timeout_fields_resumes_with_defaults(tmp_path):
    """An OLDER checkpoint written before this fix has no
    kl_timeout/measurement_timeout/timeout_leg keys at all -- resume must
    not KeyError, and re-serializing a resumed entry must default the
    missing fields to falsy/None rather than crashing."""
    ckpt_path = tmp_path / "ckpt.json"
    ckpt_path.write_text(json.dumps({
        "version": 2,
        "seed": 42,
        "source_model": {"kind": "gguf", "path": "m.gguf"},
        "measurement_conditions": {},
        "baseline_ppl": 34.8363,
        "baseline_provenance": "measured",
        "sensitivity_weights": {"E": 0.5},
        "probing_provenance": "measured",
        "weights_degenerate": False,
        "kl": {"enabled": False, "base_logits_path": None, "corpus_path": None},
        "imatrix": {"active": False, "n_tensors": None},
        "measured": {
            "old": {
                "config": {"E": "OLD"}, "ppl": 5.0, "measured_loss": 0.1,
                "predicted_loss": 0.09, "residual": 0.01, "size_gb": 4.0,
                "kl": None, "bench": None, "incumbent": None,
                "measurement_invalid": False, "corpus_path": "/fake/corpus.txt",
            },
        },
    }))
    checkpoint = json.loads(ckpt_path.read_text())

    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._measured = {}
    for key, entry in checkpoint.get("measured", {}).items():
        orch._measured[key] = dict(entry)

    resumed = orch._measured["old"]
    assert resumed.get("kl_timeout") is None
    assert resumed.get("measurement_timeout") is None
    assert resumed.get("timeout_leg") is None

    reserialized = MagicQuantOrchestrator._serialize_measurement(resumed, include_path=True)
    assert reserialized["kl_timeout"] is False
    assert reserialized["measurement_timeout"] is False
    assert reserialized["timeout_leg"] is None


def test_save_results_persists_kl_and_bench_fields(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch._search_seed = None
    orch._measured = {
        "a": {
            "config": {"E": "BF16"}, "ppl": 5.5, "measured_loss": 0.1,
            "predicted_loss": 0.09, "residual": 0.01, "size_gb": 4.0,
            "kl": {"mean_kl": 0.02}, "bench": {"pp_tps": 90.0},
        },
    }
    orch._save_results([], {})

    saved = json.loads((tmp_path / "search_results.json").read_text())
    entry = saved["measurements"]["a"]
    assert entry["kl"] == {"mean_kl": 0.02}
    assert entry["bench"] == {"pp_tps": 90.0}


def test_save_results_stamps_current_tier_scheme_version(tmp_path):
    from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION

    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch._search_seed = None
    orch._measured = {}
    orch._save_results([], {})

    saved = json.loads((tmp_path / "search_results.json").read_text())
    assert saved["tier_scheme_version"] == CURRENT_TIER_SCHEME_VERSION


def test_save_results_records_incumbent_vs_evolved_source_per_tier_winner(tmp_path):
    """The ALSO ask: today (pre-fix) it's invisible whether a tier winner
    came from magicquant.incumbents' seeded llama.cpp mixture or the
    evolutionary search itself -- across four real models the Q4/Q5 tiers
    were repeatedly won by the incumbent seed with the search contributing
    nothing, unnoticed for lack of exactly this field."""
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch._search_seed = None
    orch._measured = {}
    tiered = {
        "Q4": {"config": {"E": "Q4_K_M"}, "measured_loss": 0.02,
               "size_gb": 4.0, "incumbent": "Q4"},
        "Q6": {"config": {"E": "IQ4_XS"}, "measured_loss": 0.01,
               "size_gb": 6.0},  # no "incumbent" key -> evolved
    }
    orch._save_results([], tiered)
    saved = json.loads((tmp_path / "search_results.json").read_text())

    assert saved["tiered"]["Q4"]["source"] == "incumbent"
    assert saved["tiered"]["Q6"]["source"] == "evolved"
    assert saved["tiered_survivors"]["Q4"]["source"] == "incumbent"
    assert saved["tiered_survivors"]["Q6"]["source"] == "evolved"


def test_measured_search_raises_when_every_candidate_build_fails(tmp_path, monkeypatch):
    """F3: a measured search where every build/measure fails must not report
    success -- self._measured stays empty, so run_measured_search must raise
    instead of falling through to _select_final_survivors/_save_results.

    (MAJOR 2 reworded the guard's message to "zero VALID measurements" --
    same guard, now also covering the all-measurement_invalid case; see
    tests/test_measurement_validity.py::
    test_all_invalid_measurements_raises_instead_of_completing_with_zero_tiers.)
    """
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch, "_build_candidate", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="zero VALID"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=2, verbose=False,
        )

    assert not (orch.output_dir / "search_results.json").exists(), (
        "must not write search_results.json for an all-failed run"
    )


def test_measured_search_raises_when_every_perplexity_measurement_fails(tmp_path, monkeypatch):
    """Same guard, but the build succeeds and perplexity measurement is what
    fails for every candidate (calculate_perplexity returns None)."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    calls = {"n": 0}

    def flaky_perplexity(path, verbose=False, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return 5.0  # baseline succeeds
        return None  # every candidate measurement fails

    monkeypatch.setattr(fake_tools, "calculate_perplexity", flaky_perplexity)

    with pytest.raises(RuntimeError, match="zero VALID"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=2, verbose=False,
            # probe_kl's now-default base-logits fusion would otherwise
            # obtain the baseline via save_base_logits (always 5.0 in this
            # fixture) instead of this test's own flaky_perplexity, shifting
            # which call is "call 1" and defeating the fixture below.
            probe_kl=False,
        )


def test_save_results_defaults_kl_and_bench_to_none_when_absent(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "prediction-only"
    orch.probing_provenance = "unknown"
    orch._search_seed = None
    orch._measured = {
        "a": {"config": {"E": "BF16"}, "predicted_loss": 0.09, "size_gb": 4.0},
    }
    orch._save_results([], {})

    saved = json.loads((tmp_path / "search_results.json").read_text())
    entry = saved["measurements"]["a"]
    assert entry["kl"] is None
    assert entry["bench"] is None


def test_measurement_chunks_applied_to_llama_tools(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        measurement_chunks=7,
    )

    assert fake_tools.ppl_chunks == 7


def test_measurement_chunks_none_leaves_ppl_chunks_untouched(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.ppl_chunks = "untouched"

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
    )

    assert fake_tools.ppl_chunks == "untouched"


def test_run_full_search_measurement_chunks_is_symmetric(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_full_search(
        max_generations=2, population_size=8, verbose=False,
        measurement_chunks=9,
    )

    assert fake_tools.ppl_chunks == 9


# ── _save_results "measurement" metadata block ──────────────────────────────


def test_save_results_measurement_metadata_full(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.ppl_chunks = 10
    orch.output_dir.mkdir(parents=True, exist_ok=True)
    (orch.output_dir / "sensitivity.json").write_text(
        json.dumps({"probing_provenance": "measured"})
    )
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "measured"
    orch._search_seed = None
    orch._imatrix = {"a": object(), "b": object()}
    orch._kl_base_logits_path = str(tmp_path / "_kl_base_logits.kld")
    orch._kl_weight = 0.3
    orch._measured = {}

    orch._save_results([], {})

    saved = json.loads((orch.output_dir / "search_results.json").read_text())
    meta = saved["measurement"]
    assert meta["chunks"] == 10
    assert meta["ctx_size"] == fake_tools.ctx_size
    assert meta["corpus"] == "/fake/corpus.txt"
    assert meta["imatrix_active"] is True
    assert meta["imatrix_n_tensors"] == 2
    assert meta["kl_enabled"] is True
    assert meta["kl_weight"] == 0.3
    assert meta["probing_provenance"] == "measured"


def test_save_results_measurement_metadata_defaults_for_bare_state(tmp_path):
    # Older/bare orchestrator state (no _llama_tools, no _imatrix, no KL
    # attributes at all -- as produced by __new__ in several tests above)
    # must not crash _save_results.
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "prediction-only"
    orch._search_seed = None
    orch._measured = {}

    orch._save_results([], {})

    saved = json.loads((tmp_path / "search_results.json").read_text())
    meta = saved["measurement"]
    assert meta["chunks"] is None
    assert meta["ctx_size"] is None
    assert meta["corpus"] is None
    assert meta["imatrix_active"] is False
    assert meta["imatrix_n_tensors"] is None
    assert meta["kl_enabled"] is False
    assert meta["kl_weight"] == 0.0
    assert meta["probing_provenance"] is None


def test_enable_imatrix_uses_sibling_of_perplexity_tool(tmp_path, monkeypatch):
    """enable_imatrix must aim llama-imatrix at the SAME llama.cpp build as
    the perplexity tool -- ensure_imatrix's PATH fallback can resolve to a
    different (e.g. stock brew) build that can't load arches only the
    configured fork supports."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    bin_dir = tmp_path / "fork" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "llama-imatrix").write_text("")
    fake_tools.perplexity_tool = str(bin_dir / "llama-perplexity")

    captured = {}

    def fake_ensure(source, corpus_path=None, **kwargs):
        captured.update(kwargs)
        return {"t": object()}

    monkeypatch.setattr("magicquant.imatrix.ensure_imatrix", fake_ensure)
    assert orch.enable_imatrix() is True
    assert captured["imatrix_bin"] == str(bin_dir / "llama-imatrix")


def test_enable_imatrix_falls_back_to_path_lookup_without_sibling(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.perplexity_tool = str(tmp_path / "nowhere" / "llama-perplexity")

    captured = {}

    def fake_ensure(source, corpus_path=None, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("magicquant.imatrix.ensure_imatrix", fake_ensure)
    orch.enable_imatrix()
    assert "imatrix_bin" not in captured


# ── build/measure overlap memory gate (OOM fix, 2026-07-05) ──────────────────

def _bare_orch_for_overlap(tmp_path, src_size):
    from magicquant.orchestrator import MagicQuantOrchestrator
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    src = tmp_path / "model-bf16.gguf"
    src.write_bytes(b"0" * src_size)
    orch.source_model_path = str(src)
    return orch


def test_overlap_env_force_off(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGICQUANT_OVERLAP_BUILDS", "0")
    orch = _bare_orch_for_overlap(tmp_path, 1024)
    assert orch._should_overlap_builds() is False


def test_overlap_env_force_on_beats_memory_heuristic(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGICQUANT_OVERLAP_BUILDS", "1")
    orch = _bare_orch_for_overlap(tmp_path, 10_000)
    monkeypatch.setattr(orch, "_available_ram_bytes", lambda: 1000)  # tiny RAM
    assert orch._should_overlap_builds() is True


def test_overlap_auto_disables_when_source_large(tmp_path, monkeypatch):
    monkeypatch.delenv("MAGICQUANT_OVERLAP_BUILDS", raising=False)
    orch = _bare_orch_for_overlap(tmp_path, 51 * 1024)  # ~51 "GB" scaled
    monkeypatch.setattr(orch.__class__, "_available_ram_bytes",
                        staticmethod(lambda: 79 * 1024))  # 51 > 79*0.35
    assert orch._should_overlap_builds() is False


def test_overlap_auto_enables_when_source_small(tmp_path, monkeypatch):
    monkeypatch.delenv("MAGICQUANT_OVERLAP_BUILDS", raising=False)
    orch = _bare_orch_for_overlap(tmp_path, 1 * 1024)
    monkeypatch.setattr(orch.__class__, "_available_ram_bytes",
                        staticmethod(lambda: 79 * 1024))
    assert orch._should_overlap_builds() is True


def test_overlap_auto_keeps_overlap_when_ram_unreadable(tmp_path, monkeypatch):
    monkeypatch.delenv("MAGICQUANT_OVERLAP_BUILDS", raising=False)
    orch = _bare_orch_for_overlap(tmp_path, 51 * 1024)
    monkeypatch.setattr(orch.__class__, "_available_ram_bytes",
                        staticmethod(lambda: None))
    assert orch._should_overlap_builds() is True


# ── predictor-tracking diagnostic wiring (E8) ────────────────────────────────
#
# magicquant.utils.measurement.predictor_rank_correlation/predictor_is_tracking
# were built and unit-tested (tests/test_probe_resolution.py::TestPredictorTracking)
# but never called by the search they were designed to guard. These tests
# check the wiring: _log_predictor_tracking (called once, end-of-run, from
# run_measured_search) computes the verdict cumulatively over self._measured,
# excludes measurement_invalid and incumbent-seeded (measured_loss=None)
# entries, logs it, and _save_results persists it under the additive
# "predictor_tracking" key.

def _bare_orch_for_tracking(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 34.8363
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch._search_seed = None
    return orch


class _FakeLog:
    """Records (level, event, kwargs) for every structlog-style call
    (log.info(event, **kw) / log.warning(event, **kw) / ...). orchestrator.py
    logs via magicquant.logging's structlog-based `log`, which is left
    unconfigured in tests (nothing calls configure_logging) so its default
    backing is a bare PrintLogger -- caplog never sees it, and
    structlog.testing.capture_logs() is unreliable across a full test-suite
    run because cache_logger_on_first_use=True (set by any earlier test that
    DOES call configure_logging, e.g. a CLI test) permanently binds the
    module-level `log` object's processor chain on its first real call,
    ahead of any later capture_logs() context. Monkeypatching the module
    attribute directly (the same pattern as
    tests/test_generate_tiered_models_missing.py) sidesteps all of that."""

    def __init__(self):
        self.calls = []

    def _record(self, level):
        def _call(event, **kw):
            self.calls.append((level, event, kw))
        return _call

    def __getattr__(self, level):
        return self._record(level)


# The exact pairs Laguna-S recorded (also used verbatim in
# tests/test_probe_resolution.py::TestPredictorTracking) -- Kendall tau over
# them is -0.0426, i.e. NOT tracking. Reusing the real incident's numbers
# rather than synthetic ones keeps this test tied to the failure it guards.
_LAGUNA_S_PREDICTED = [
    4.253018, 2.966267, 2.28, 1.439624, 0.491492, 0.491492, 0.0,
    0.913953, 1.385445, 1.385445, 1.385445, 3.064993, 0.446342,
    1.033281, 0.857834, 0.446342,
]
_LAGUNA_S_MEASURED = [
    0.025496, 0.013969, 0.004845, 0.047605, 0.045422, 0.020864,
    0.025925, 0.244713, 0.046464, 0.051313, 0.240506, 0.21743,
    0.047818, 0.043852, 0.241443, 0.025642,
]


def test_predictor_tracking_verdict_logged_and_persisted(tmp_path, monkeypatch):
    # This test asserts a REAL computed tau, so it needs scipy for real
    # (unlike test_predictor_tracking_unknown_when_scipy_absent below, which
    # deliberately fakes scipy's absence). scipy is a [dev]-only dependency
    # (pyproject.toml), and .venv-qat -- the QAT extra's own venv -- doesn't
    # have it; skip rather than fail there, matching how
    # tests/test_probe_resolution.py::TestPredictorTracking is documented as
    # an accepted environment gap in that venv rather than something this
    # test should also newly fail.
    pytest.importorskip("scipy")

    fake_log = _FakeLog()
    monkeypatch.setattr(orch_mod, "log", fake_log)

    orch = _bare_orch_for_tracking(tmp_path)
    orch._measured = {
        f"laguna-{i}": {
            "config": {"E": "BF16"},
            "predicted_loss": p, "measured_loss": m,
            "ppl": 10.0, "size_gb": 4.0,
        }
        for i, (p, m) in enumerate(zip(_LAGUNA_S_PREDICTED, _LAGUNA_S_MEASURED))
    }
    # A measurement_invalid entry with a wildly different pair -- if this
    # leaked into the correlation it would change tau; it must not.
    orch._measured["poisoned"] = {
        "config": {"E": "BF16"}, "predicted_loss": 99.0, "measured_loss": -0.9225,
        "ppl": 2.7, "size_gb": 4.0, "measurement_invalid": True,
    }
    # An incumbent-seeded entry: predicted_loss present, measured_loss is
    # None (never measured) -- must also be excluded, not treated as 0/NaN.
    orch._measured["incumbent-seed"] = {
        "config": {"E": "Q4_K_M"}, "predicted_loss": 0.5, "measured_loss": None,
        "ppl": None, "size_gb": 4.0,
    }

    orch._log_predictor_tracking()

    assert orch._predictor_tracking["is_tracking"] is False
    assert orch._predictor_tracking["tau"] == pytest.approx(-0.0426, abs=1e-3)
    assert orch._predictor_tracking["n_pairs"] == 16

    # (a) verdict is logged -- NOT tracking is a warning, per the LB
    # constraint that only False (not None/"unknown") is evidence of a
    # broken run.
    warnings = [(event, kw) for level, event, kw in fake_log.calls if level == "warning"]
    assert any("not tracking" in event.lower() for event, kw in warnings)

    # (b) verdict is persisted in _save_results' output under the additive
    # "predictor_tracking" key, without disturbing any existing key.
    orch._save_results([], {})
    saved = json.loads((tmp_path / "search_results.json").read_text())
    assert saved["predictor_tracking"]["is_tracking"] is False
    assert saved["predictor_tracking"]["tau"] == pytest.approx(-0.0426, abs=1e-3)
    assert saved["predictor_tracking"]["n_pairs"] == 16
    # Pre-existing keys are untouched.
    assert saved["baseline_ppl"] == 34.8363
    assert "measurements" in saved and len(saved["measurements"]) == 18


def test_predictor_tracking_unknown_when_scipy_absent(tmp_path, monkeypatch):
    """.venv-qat has no scipy -- predictor_rank_correlation's ImportError
    branch must degrade to (None, None) ("unknown"), not crash, and must not
    be logged as "not tracking" (that would be the "measured nothing
    reported as measured zero" defect reproduced in the reporting layer)."""
    import builtins

    real_import = builtins.__import__

    def _no_scipy(name, *args, **kwargs):
        if name == "scipy.stats" or name.startswith("scipy"):
            raise ImportError("simulated scipy-absent environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_scipy)

    fake_log = _FakeLog()
    monkeypatch.setattr(orch_mod, "log", fake_log)

    orch = _bare_orch_for_tracking(tmp_path)
    orch._measured = {
        f"c{i}": {
            "config": {"E": "BF16"}, "predicted_loss": p, "measured_loss": m,
            "ppl": 10.0, "size_gb": 4.0,
        }
        for i, (p, m) in enumerate(zip(_LAGUNA_S_PREDICTED, _LAGUNA_S_MEASURED))
    }

    orch._log_predictor_tracking()  # must not raise

    assert orch._predictor_tracking == {
        "is_tracking": None, "tau": None, "n_pairs": 16,
    }
    warnings = [(event, kw) for level, event, kw in fake_log.calls if level == "warning"]
    assert not any("not tracking" in event.lower() for event, kw in warnings)

    orch._save_results([], {})
    saved = json.loads((tmp_path / "search_results.json").read_text())
    assert saved["predictor_tracking"] == {
        "is_tracking": None, "tau": None, "n_pairs": 16,
    }


def test_predictor_tracking_absent_defaults_to_none_in_save_results(tmp_path):
    """A bare orchestrator that calls _save_results directly (as
    test_measurement_validity.py's fixtures do) without ever calling
    _log_predictor_tracking -- e.g. run_full_search's prediction-only path,
    which has no measurements to check -- must not KeyError or fabricate a
    verdict; the additive key is simply None."""
    orch = _bare_orch_for_tracking(tmp_path)
    orch._measured = {}
    orch._save_results([], {})
    saved = json.loads((tmp_path / "search_results.json").read_text())
    assert saved["predictor_tracking"] is None
