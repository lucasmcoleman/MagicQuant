"""Tests for the opt-in speed-aware final-survivor selection (LANE 2 / PART B).

_select_final_survivors currently picks the lowest KL-blended measured_loss
per tier. run_measured_search gains speed_aware: bool = False and
speed_epsilon: float = 0.005 params; when speed_aware is True and at least
one candidate in a tier carries "bench" data (populated only by the
existing enable_speed_bench path, {"pp_ts": float, "tg_ts": float}),
_select_final_survivors re-ranks WITHIN the epsilon-relative measured_loss
band around the tier's best by measured tg throughput. This is a pure
post-hoc re-ranking of already-recorded measurements -- no new GPU calls.

Covers: the pure _speed_aware_pick helper directly; _select_final_survivors
wiring (within-epsilon re-rank, outside-epsilon quality wins, no-bench-data
fallback, speed_aware=False / missing-attribute default-off); and full
run_measured_search plumbing end-to-end against a faked llama.cpp boundary
(same faking style as test_orchestrator_measurement.py).
"""
import pytest

import magicquant.gguf.source as source_mod
from magicquant.orchestrator import MagicQuantOrchestrator


# ----------------------------------------------------------------------
# Pure _speed_aware_pick helper
# ----------------------------------------------------------------------

def test_speed_aware_pick_reranks_within_epsilon():
    quality_best = {"measured_loss": 0.10, "bench": {"tg_ts": 20.0}}
    faster = {"measured_loss": 0.1004, "bench": {"tg_ts": 80.0}}  # +0.4% -- within 0.5%
    result = MagicQuantOrchestrator._speed_aware_pick(
        [quality_best, faster], quality_best, epsilon=0.005
    )
    assert result is faster


def test_speed_aware_pick_ignores_candidate_outside_epsilon():
    quality_best = {"measured_loss": 0.10, "bench": {"tg_ts": 20.0}}
    outside = {"measured_loss": 0.11, "bench": {"tg_ts": 999.0}}  # +10% -- outside 0.5%
    result = MagicQuantOrchestrator._speed_aware_pick(
        [quality_best, outside], quality_best, epsilon=0.005
    )
    assert result is quality_best


def test_speed_aware_pick_falls_back_when_no_candidate_has_bench():
    quality_best = {"measured_loss": 0.10}
    other = {"measured_loss": 0.1001}
    result = MagicQuantOrchestrator._speed_aware_pick(
        [quality_best, other], quality_best, epsilon=0.005
    )
    assert result is quality_best


def test_speed_aware_pick_only_considers_within_epsilon_bench_having_candidates():
    quality_best = {"measured_loss": 0.10, "bench": {"tg_ts": 20.0}}
    no_bench_but_within_epsilon = {"measured_loss": 0.1001}
    result = MagicQuantOrchestrator._speed_aware_pick(
        [quality_best, no_bench_but_within_epsilon], quality_best, epsilon=0.005
    )
    assert result is quality_best


def test_speed_aware_pick_exact_tie_prefers_higher_tg():
    a = {"measured_loss": 0.10, "bench": {"tg_ts": 20.0}}
    b = {"measured_loss": 0.10, "bench": {"tg_ts": 20.1}}
    result = MagicQuantOrchestrator._speed_aware_pick([a, b], a, epsilon=0.005)
    assert result is b


# ----------------------------------------------------------------------
# _select_final_survivors wiring
# ----------------------------------------------------------------------

def test_select_final_survivors_speed_aware_reranks_within_epsilon():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = True
    orch._speed_epsilon = 0.005
    orch._measured = {
        "a": {
            "config": {"E": "SLOW"}, "measured_loss": 0.10,
            "size_gb": 4.0, "bench": {"tg_ts": 20.0},
        },
        "b": {
            "config": {"E": "FAST"}, "measured_loss": 0.1004,
            "size_gb": 4.0, "bench": {"tg_ts": 80.0},
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "FAST"}


def test_select_final_survivors_speed_aware_respects_epsilon_boundary():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = True
    orch._speed_epsilon = 0.005
    orch._measured = {
        "a": {
            "config": {"E": "QUALITY"}, "measured_loss": 0.10,
            "size_gb": 4.0, "bench": {"tg_ts": 20.0},
        },
        "b": {
            "config": {"E": "OUTSIDE"}, "measured_loss": 0.11,
            "size_gb": 4.0, "bench": {"tg_ts": 999.0},
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "QUALITY"}


def test_select_final_survivors_speed_aware_no_bench_data_falls_back_to_loss():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = True
    orch._speed_epsilon = 0.005
    orch._measured = {
        "a": {"config": {"E": "WORSE"}, "measured_loss": 0.20, "size_gb": 4.0},
        "b": {"config": {"E": "BETTER"}, "measured_loss": 0.05, "size_gb": 4.0},
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "BETTER"}


def test_select_final_survivors_speed_aware_false_ignores_bench_data():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = False
    orch._speed_epsilon = 0.005
    orch._measured = {
        "a": {
            "config": {"E": "BEST_LOSS_SLOW"}, "measured_loss": 0.10,
            "size_gb": 4.0, "bench": {"tg_ts": 1.0},
        },
        "b": {
            "config": {"E": "WORSE_LOSS_FAST"}, "measured_loss": 0.1004,
            "size_gb": 4.0, "bench": {"tg_ts": 999.0},
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "BEST_LOSS_SLOW"}


def test_select_final_survivors_missing_speed_attrs_defaults_to_off():
    """An orchestrator instance predating this feature (no _speed_aware/
    _speed_epsilon attributes at all, e.g. built via __new__ by an older
    caller) must behave exactly as before -- getattr defaults, no
    AttributeError."""
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._measured = {
        "a": {
            "config": {"E": "BEST_LOSS_SLOW"}, "measured_loss": 0.10,
            "size_gb": 4.0, "bench": {"tg_ts": 1.0},
        },
        "b": {
            "config": {"E": "WORSE_LOSS_FAST"}, "measured_loss": 0.1004,
            "size_gb": 4.0, "bench": {"tg_ts": 999.0},
        },
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "BEST_LOSS_SLOW"}


# ----------------------------------------------------------------------
# Full run_measured_search wiring (faked llama.cpp boundary)
# ----------------------------------------------------------------------

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
    """Stands in for LlamaCppTools. Every measurement (baseline AND every
    candidate) returns the SAME perplexity, so every candidate's
    measured_loss is exactly 0 -- an exact tie, well within any epsilon --
    and bench() hands out tg_ts values from a fixed sequence so
    speed_aware's tiebreak is the only thing that can distinguish winners.
    """

    def __init__(self, ppl=5.0, tg_values=None):
        self.ctx_size = 512
        self._ppl = ppl
        self._tg_values = list(tg_values or [])
        self._tg_idx = 0
        self.bench_calls = []

    def calculate_perplexity(self, path, verbose=False, **kw):
        return self._ppl

    def _resolve_data_file(self, data_file=None):
        return "/fake/corpus.txt"

    def bench(self, model_path, **kw):
        self.bench_calls.append(model_path)
        idx = min(self._tg_idx, len(self._tg_values) - 1) if self._tg_values else None
        tg = self._tg_values[idx] if idx is not None else 10.0
        self._tg_idx += 1
        return {"pp_ts": 100.0, "tg_ts": tg}


def _make_orchestrator(tmp_path, monkeypatch, ppl=5.0, tg_values=None):
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    fake_tools = _FakeLlamaTools(ppl=ppl, tg_values=tg_values)
    orch._llama_tools = fake_tools

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    counter = {"n": 0}

    def fake_build_candidate(config, name, base_quant):
        counter["n"] += 1
        p = candidates_dir / f"{name}_{counter['n']}.gguf"
        p.write_bytes(b"0" * 1024)  # identical size -> every candidate lands in one tier
        return str(p)

    monkeypatch.setattr(orch, "_build_candidate", fake_build_candidate)
    return orch, fake_tools


def test_run_measured_search_threads_speed_aware_params(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch.run_measured_search(
        search_generations=1, population_size=4, measurement_rounds=1,
        candidates_per_round=1, verbose=False, seed_incumbents=False,
        speed_aware=True, speed_epsilon=0.02,
    )
    assert orch._speed_aware is True
    assert orch._speed_epsilon == 0.02


def test_run_measured_search_defaults_speed_aware_off(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch.run_measured_search(
        search_generations=1, population_size=4, measurement_rounds=1,
        candidates_per_round=1, verbose=False, seed_incumbents=False,
    )
    assert orch._speed_aware is False
    assert orch._speed_epsilon == 0.005


def test_run_measured_search_speed_aware_end_to_end_prefers_faster_tied_candidate(
    tmp_path, monkeypatch
):
    # Three candidates get measured (round 1); every one ties on
    # measured_loss (ppl == baseline_ppl == 5.0 for all of them). The
    # SECOND one measured gets the highest tg -- distinguishing this from
    # the plain (first-tied-wins) fallback below.
    orch, fake_tools = _make_orchestrator(
        tmp_path, monkeypatch, ppl=5.0, tg_values=[10.0, 500.0, 20.0]
    )
    all_configs, tiered = orch.run_measured_search(
        search_generations=2, population_size=8, measurement_rounds=1,
        candidates_per_round=3, verbose=False, seed=1, seed_incumbents=False,
        enable_speed_bench=True, speed_aware=True,
    )

    assert orch._measured, "expected at least one measured candidate"
    assert all(abs(info["measured_loss"]) < 1e-9 for info in orch._measured.values())
    assert len(tiered) == 1, "fake GGUFs are all the same size -- expect one tier"

    best_by_tg = max(orch._measured.values(), key=lambda i: i["bench"]["tg_ts"])
    winner = next(iter(tiered.values()))
    assert winner["config"] == best_by_tg["config"]


def test_run_measured_search_speed_aware_false_keeps_plain_tiebreak(tmp_path, monkeypatch):
    # Same setup as above, but speed_aware=False: an exact measured_loss tie
    # must NOT be broken by tg -- the historical min()-over-ties behavior
    # (first-measured wins) is unaffected by bench data merely being present.
    orch, fake_tools = _make_orchestrator(
        tmp_path, monkeypatch, ppl=5.0, tg_values=[10.0, 500.0, 20.0]
    )
    all_configs, tiered = orch.run_measured_search(
        search_generations=2, population_size=8, measurement_rounds=1,
        candidates_per_round=3, verbose=False, seed=1, seed_incumbents=False,
        enable_speed_bench=True, speed_aware=False,
    )

    assert orch._measured
    assert len(tiered) == 1
    first_measured = next(iter(orch._measured.values()))
    winner = next(iter(tiered.values()))
    assert winner["config"] == first_measured["config"]
