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
        [quality_best, faster], quality_best, epsilon=0.005,
        score_of=lambda c: c['measured_loss'], speed_metric="bench"
    )
    assert result is faster


def test_speed_aware_pick_ignores_candidate_outside_epsilon():
    quality_best = {"measured_loss": 0.10, "bench": {"tg_ts": 20.0}}
    outside = {"measured_loss": 0.11, "bench": {"tg_ts": 999.0}}  # +10% -- outside 0.5%
    result = MagicQuantOrchestrator._speed_aware_pick(
        [quality_best, outside], quality_best, epsilon=0.005,
        score_of=lambda c: c['measured_loss'], speed_metric="bench"
    )
    assert result is quality_best


def test_speed_aware_pick_falls_back_when_no_candidate_has_bench():
    quality_best = {"measured_loss": 0.10}
    other = {"measured_loss": 0.1001}
    result = MagicQuantOrchestrator._speed_aware_pick(
        [quality_best, other], quality_best, epsilon=0.005,
        score_of=lambda c: c['measured_loss'], speed_metric="bench"
    )
    assert result is quality_best


def test_speed_aware_pick_only_considers_within_epsilon_bench_having_candidates():
    quality_best = {"measured_loss": 0.10, "bench": {"tg_ts": 20.0}}
    no_bench_but_within_epsilon = {"measured_loss": 0.1001}
    result = MagicQuantOrchestrator._speed_aware_pick(
        [quality_best, no_bench_but_within_epsilon], quality_best, epsilon=0.005,
        score_of=lambda c: c['measured_loss'], speed_metric="bench"
    )
    assert result is quality_best


def test_speed_aware_pick_exact_tie_prefers_higher_tg():
    a = {"measured_loss": 0.10, "bench": {"tg_ts": 20.0}}
    b = {"measured_loss": 0.10, "bench": {"tg_ts": 20.1}}
    result = MagicQuantOrchestrator._speed_aware_pick([a, b], a, epsilon=0.005,
        score_of=lambda c: c['measured_loss'], speed_metric="bench")
    assert result is b


# ----------------------------------------------------------------------
# _select_final_survivors wiring
# ----------------------------------------------------------------------

def test_select_final_survivors_speed_aware_reranks_within_epsilon():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = True
    orch._speed_metric = "bench"
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
    orch._speed_metric = "bench"
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
    orch._speed_metric = "bench"
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
        speed_aware=True, speed_metric="bench", speed_epsilon=0.02,
    )
    assert orch._speed_aware is True
    assert orch._speed_epsilon == 0.02


def test_run_measured_search_defaults_speed_aware_on(tmp_path, monkeypatch):
    """2026-07 fix: speed_aware defaults ON (a real ThinkingCap run shipped a
    larger, worse-raw-PPL "Q5" winner purely via a KL tiebreak -- selection
    was size-blind by default). speed_epsilon defaults to None, resolved
    lazily by _select_final_survivors via measurement.measurement_eps(...)
    rather than a fixed float. Pass speed_aware=False to restore the old
    bare argmin-over-KL-blended-score behavior."""
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch.run_measured_search(
        search_generations=1, population_size=4, measurement_rounds=1,
        candidates_per_round=1, verbose=False, seed_incumbents=False,
    )
    assert orch._speed_aware is True
    assert orch._speed_epsilon is None


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
        enable_speed_bench=True, speed_aware=True, speed_metric="bench",
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


def test_speed_aware_respects_kl_guard(monkeypatch):
    """enable_kl + speed_aware together: a candidate whose KL FAILED (no "kl")
    but whose raw PPL is within epsilon and whose tg is fastest must NOT be
    picked -- the speed band uses the KL-guarded score, so the failed
    candidate is pushed out of the band by the worst-KL penalty (adversarial
    review 2026-07-05)."""
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 1.0
    orch._speed_aware = True
    orch._speed_epsilon = 0.02  # generous band so raw-loss alone would admit it
    orch._measured = {
        # real-KL winner, modest tg
        "good": {"config": {"E": "GOOD"}, "measured_loss": 0.100, "size_gb": 4.0,
                 "kl": {"mean_kl": 0.01}, "bench": {"tg_ts": 20.0}},
        # KL FAILED (no "kl"), raw loss within epsilon, FASTEST tg -- must be rejected
        "failed": {"config": {"E": "FAILED"}, "measured_loss": 0.101, "size_gb": 4.0,
                   "bench": {"tg_ts": 999.0}},
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "GOOD"}, "KL-failed fast candidate wrongly won the speed band"


# ── deterministic bytes metric (the noise-robust default) ────────────────────

def test_speed_aware_pick_bytes_default_prefers_smaller():
    # No bench data at all; bytes mode ranks within-epsilon band by size_gb.
    a = {"measured_loss": 0.10, "size_gb": 5.0}
    b = {"measured_loss": 0.1004, "size_gb": 4.0}  # within epsilon, smaller
    result = MagicQuantOrchestrator._speed_aware_pick(
        [a, b], a, epsilon=0.005, score_of=lambda c: c["measured_loss"]
    )
    assert result is b


def test_speed_aware_pick_bytes_ignores_outside_epsilon():
    a = {"measured_loss": 0.10, "size_gb": 5.0}
    smaller_but_worse = {"measured_loss": 0.11, "size_gb": 1.0}  # outside epsilon
    result = MagicQuantOrchestrator._speed_aware_pick(
        [a, smaller_but_worse], a, epsilon=0.005, score_of=lambda c: c["measured_loss"]
    )
    assert result is a


def test_speed_aware_bytes_needs_no_bench_data_unlike_bench_mode():
    # bytes mode works with zero bench data; bench mode would no-op here.
    a = {"measured_loss": 0.10, "size_gb": 5.0}
    b = {"measured_loss": 0.1002, "size_gb": 3.0}
    assert MagicQuantOrchestrator._speed_aware_pick(
        [a, b], a, epsilon=0.005, score_of=lambda c: c["measured_loss"],
        speed_metric="bytes") is b
    assert MagicQuantOrchestrator._speed_aware_pick(
        [a, b], a, epsilon=0.005, score_of=lambda c: c["measured_loss"],
        speed_metric="bench") is a  # no bench -> quality_best


def test_speed_aware_bytes_honors_kl_guard():
    # KL-failed candidate can't win the bytes tiebreak when winner is KL-confirmed.
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 1.0
    orch._speed_aware = True
    orch._speed_metric = "bytes"
    orch._speed_epsilon = 0.02
    orch._measured = {
        "good": {"config": {"E": "GOOD"}, "measured_loss": 0.100, "size_gb": 5.0,
                 "kl": {"mean_kl": 0.01}},
        "failed_smaller": {"config": {"E": "FAILED"}, "measured_loss": 0.101, "size_gb": 2.0},
    }
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(5.0, 8.0)
    assert result[tier]["config"] == {"E": "GOOD"}


# ── Real-run regression: ThinkingCap-Qwen3.6-27B measured search ────────────
#
# output/ThinkingCap-Qwen3.6-27B/magicquant/search_results.json (a real
# measured run, kl_weight=0.1): the shipped "Q5" tier winner was the uniform
# Q6_K config -- BOTH larger (20.89 GB vs 17.68 GB) AND worse raw PPL
# (6.827036 vs 6.826419) than the uniform Q5_K config it beat. It won only
# because the KL tiebreak (kl_weight * |mean_kl|) tipped the combined score:
# Q6_K's mean_kl (0.010119) was lower than Q5_K's (0.0185), enough to
# outweigh Q6_K's slightly WORSE raw measured_loss once kl_weight=0.1 was
# applied -- even though the raw PPL gap between them (~0.009%) was two
# orders of magnitude below the run's own reported measurement error
# (ppl_err ~0.107, i.e. ~1.6% of baseline_ppl=6.7803).
_THINKINGCAP_BASELINE_PPL = 6.7803
_THINKINGCAP_Q5_K = {
    "config": {"D": "Q5_K", "E": "Q5_K", "H": "Q6_K", "K": "Q5_K",
               "O": "Q5_K", "Q": "Q5_K", "S": "Q5_K", "U": "Q5_K"},
    "measured_loss": 0.006801911419848551,
    "size_gb": 17.675239741802216,
    "kl": {"mean_kl": 0.0185, "max_kl": 18.072206, "p90_kl": 0.020889,
           "ppl": 6.826419, "ppl_err": 0.107295},
}
_THINKINGCAP_Q6_K = {
    "config": {"D": "Q6_K", "E": "Q6_K", "H": "Q6_K", "K": "Q6_K",
               "O": "Q6_K", "Q": "Q6_K", "S": "Q6_K", "U": "Q6_K"},
    "measured_loss": 0.006892910343200038,
    "size_gb": 20.890495479106903,
    "kl": {"mean_kl": 0.010119, "max_kl": 16.915789, "p90_kl": 0.01204,
           "ppl": 6.827036, "ppl_err": 0.10734},
}


def test_thinkingcap_smaller_better_ppl_candidate_wins_default():
    """Regression for the real ThinkingCap bug: with speed_aware ON
    (default), the smaller AND better-raw-PPL Q5_K config must win its tier
    over the larger, worse-raw-PPL Q6_K config that a bare KL-blended argmin
    picked instead.

    Proven to fail pre-fix: before this fix, _select_final_survivors' bare
    ``min(candidates, key=_score)`` (KL-blended) picked the Q6_K config --
    the exact real-world failure this test guards against.
    """
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.1
    orch._speed_aware = True
    orch._speed_epsilon = None
    orch.baseline_ppl = _THINKINGCAP_BASELINE_PPL
    orch._measured = {
        "q5k": dict(_THINKINGCAP_Q5_K),
        "q6k": dict(_THINKINGCAP_Q6_K),
    }
    # Force both real candidates into ONE tier band -- independent of
    # whatever TIER_BOUNDARIES currently says about their actual sizes
    # (post-Fix-B they'd classify into DIFFERENT tiers; this test is about
    # WITHIN-tier selection, so it pins them together deliberately).
    orch._classify_tier = staticmethod(lambda size_gb, baseline_gb: "Q5")
    result = orch._select_final_survivors(baseline_gb=50.0)
    assert result["Q5"]["config"] == _THINKINGCAP_Q5_K["config"], (
        "the larger, worse-raw-PPL Q6_K candidate won again -- "
        f"got {result['Q5']['config']}"
    )


def test_thinkingcap_bare_argmin_reproduces_the_original_bug():
    """With speed_aware=False (the escape hatch), the ORIGINAL KL-blended
    bare-argmin behavior is restored exactly -- including its bug. Documents
    that speed_aware=False is a genuine behavioral restoration, not just a
    epsilon=0 special case."""
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.1
    orch._speed_aware = False
    orch.baseline_ppl = _THINKINGCAP_BASELINE_PPL
    orch._measured = {
        "q5k": dict(_THINKINGCAP_Q5_K),
        "q6k": dict(_THINKINGCAP_Q6_K),
    }
    orch._classify_tier = staticmethod(lambda size_gb, baseline_gb: "Q5")
    result = orch._select_final_survivors(baseline_gb=50.0)
    assert result["Q5"]["config"] == _THINKINGCAP_Q6_K["config"]


def test_select_final_survivors_default_speed_metric_is_bytes():
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0
    orch._speed_aware = True
    orch._speed_epsilon = 0.005  # note: no _speed_metric set -> getattr default "bytes"
    orch._measured = {
        "big_fast": {"config": {"E": "BIG"}, "measured_loss": 0.10, "size_gb": 5.0,
                     "bench": {"tg_ts": 999.0}},
        "small": {"config": {"E": "SMALL"}, "measured_loss": 0.1004, "size_gb": 4.0,
                  "bench": {"tg_ts": 1.0}},
    }
    # default bytes: picks SMALL (fewer bytes), ignoring the noisy high tg_ts.
    result = orch._select_final_survivors(baseline_gb=8.0)
    tier = orch._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "SMALL"}


# ── speed_epsilon must actually thread the run's own reported ppl_err ──────
#
# _select_final_survivors resolves speed_epsilon (when the instance's
# _speed_epsilon is None, the documented default) via
# magicquant.utils.measurement.measurement_eps(baseline_ppl, reported_err).
# Before this fix, the call site passed NO reported_err at all -- always
# measurement_eps(baseline_ppl), which per measurement.py's signature always
# falls back to the flat DEFAULT_RELATIVE_EPS=0.05 regardless of how
# precise (or imprecise) this run's own measurements actually were. This
# threads the tier quality-winner's fused-KL-pass ppl_err through, exactly
# like the per-candidate measurement_invalid check already does.
def test_select_final_survivors_threads_reported_ppl_err_into_speed_epsilon():
    """A run with a genuinely large reported measurement error (ppl_err) must
    widen the speed-aware epsilon band beyond the flat 0.05 default -- i.e.
    the 3-sigma path in measurement_eps must actually get exercised using
    THIS run's own numbers, not silently ignored.

    Proven to fail pre-fix: the old call site was
    ``speed_epsilon = measurement_eps(baseline_ppl)`` with no reported_err,
    which is pinned at the flat 0.05 default no matter what. baseline_ppl=10,
    A's measured_loss=0.02 (quality_best, kl.ppl_err=1.0 -- a genuinely
    noisy run, 3*1.0/10=0.3 relative), B's measured_loss=0.024 (+20%
    relative to A, size 8 < A's size 10). Under the flat 0.05 default the
    threshold is only 0.02*1.05=0.021 -- B (0.024) falls OUTSIDE the band,
    so the (larger) quality-best A wins. Under the correctly-threaded
    3-sigma epsilon (0.3), the threshold is 0.02*1.3=0.026 -- B falls
    INSIDE the band and, being smaller, wins the bytes-based speed-aware
    pick instead.
    """
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._kl_weight = 0.0  # keep _score == raw measured_loss, no KL blending
    orch._speed_aware = True
    orch._speed_epsilon = None  # resolve lazily via measurement_eps(...)
    orch.baseline_ppl = 10.0
    orch._measured = {
        "a": {
            "config": {"E": "A"}, "measured_loss": 0.02, "size_gb": 10.0,
            "kl": {"mean_kl": 0.001, "ppl_err": 1.0},
        },
        "b": {
            "config": {"E": "B"}, "measured_loss": 0.024, "size_gb": 8.0,
            "kl": {"mean_kl": 0.001, "ppl_err": 1.0},
        },
    }
    orch._classify_tier = staticmethod(lambda size_gb, baseline_gb: "Q5")
    result = orch._select_final_survivors(baseline_gb=50.0)
    assert result["Q5"]["config"] == {"E": "B"}, (
        "expected the smaller in-band candidate B to win once this run's "
        "own reported ppl_err (0.3 relative, well above the flat 0.05 "
        "default) is actually threaded into the speed-aware epsilon -- "
        f"got {result['Q5']['config']}"
    )
