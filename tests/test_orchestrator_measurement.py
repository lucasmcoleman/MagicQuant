"""Orchestrator wiring for the deferred hardware-gated items: imatrix reuse,
KL-divergence, and speed-bench measurement inside run_measured_search's
build/measure loop, plus run_full_search's imatrix symmetry.

Real EvolutionarySurvivor/PredictiveScorer/SensitivityProber run unmocked
(same objects test_refactor_regression.py exercises); only the I/O boundary
(model source, llama.cpp tools, candidate GGUF building) is faked, so this
exercises the actual wiring rather than a hand-rolled stand-in.
"""
import json

import pytest

import magicquant.gguf.source as source_mod
from magicquant.orchestrator import MagicQuantOrchestrator


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

    def calculate_perplexity(self, path, verbose=False, **kw):
        return 5.0

    def _resolve_data_file(self, data_file=None):
        return "/fake/corpus.txt"

    def save_base_logits(self, base_model_path, corpus_path, out_logits_path, **kw):
        from pathlib import Path
        Path(out_logits_path).write_text("fake logits" * 1000)
        return True

    def calculate_kl_divergence(self, quant_model_path, base_logits_path, corpus_path, **kw):
        self.kl_calls.append((quant_model_path, base_logits_path, corpus_path))
        return {"mean_kl": 0.01}

    def bench(self, model_path, **kw):
        self.bench_calls.append(model_path)
        return {"pp_tps": 100.0, "tg_tps": 50.0}


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


def test_save_results_persists_kl_and_bench_fields(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "measured"
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


def test_measured_search_raises_when_every_candidate_build_fails(tmp_path, monkeypatch):
    """F3: a measured search where every build/measure fails must not report
    success -- self._measured stays empty, so run_measured_search must raise
    instead of falling through to _select_final_survivors/_save_results."""
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(orch, "_build_candidate", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="zero successful"):
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

    with pytest.raises(RuntimeError, match="zero successful"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=2, verbose=False,
        )


def test_save_results_defaults_kl_and_bench_to_none_when_absent(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "prediction-only"
    orch._search_seed = None
    orch._measured = {
        "a": {"config": {"E": "BF16"}, "predicted_loss": 0.09, "size_gb": 4.0},
    }
    orch._save_results([], {})

    saved = json.loads((tmp_path / "search_results.json").read_text())
    entry = saved["measurements"]["a"]
    assert entry["kl"] is None
    assert entry["bench"] is None
