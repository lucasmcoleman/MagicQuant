"""Measured-search checkpoint/resume tests (Part F).

Uses the same fake-tools pattern as test_orchestrator_measurement.py: real
EvolutionarySurvivor/PredictiveScorer/SensitivityProber run unmocked, only
the I/O boundary (model source, llama.cpp tools, candidate GGUF building) is
faked.
"""
import json
from pathlib import Path

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
    """Stands in for LlamaCppTools -- no real llama.cpp binary involved.

    Distinguishes a baseline call from a candidate-measurement call by
    comparing the path against the source model path (not "the first call
    ever"), since a resumed run may never call calculate_perplexity for the
    baseline at all -- treating "first call" as baseline would mislabel a
    resumed run's first (real) candidate measurement.
    """

    def __init__(self, source_model_path):
        self.ctx_size = 512
        self.source_model_path = source_model_path
        self.ppl_calls = 0
        self.baseline_calls = 0
        self.candidate_calls = 0
        self._kill_after = None  # raise after N successful candidate measurements

    def calculate_perplexity(self, path, verbose=False, **kw):
        self.ppl_calls += 1
        if path == self.source_model_path:
            self.baseline_calls += 1
            return 5.0
        if self._kill_after is not None and self.candidate_calls >= self._kill_after:
            raise RuntimeError("simulated kill mid-measurement")
        self.candidate_calls += 1
        return 5.0 + 0.01 * self.candidate_calls  # distinct per-candidate ppl

    def _resolve_data_file(self, data_file=None):
        return "/fake/corpus.txt"

    def save_base_logits(self, base_model_path, corpus_path, out_logits_path, **kw):
        from pathlib import Path
        Path(out_logits_path).write_text("fake logits" * 1000)
        return 5.0  # this pass's own "Final estimate: PPL" (fused baseline)


def _make_orchestrator(tmp_path, monkeypatch, source_name="nonexistent.gguf"):
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / source_name),
        output_dir=str(tmp_path / "out"),
    )
    fake_tools = _FakeLlamaTools(orch.source_model_path)
    orch._llama_tools = fake_tools

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    counter = {"n": 0}

    def fake_build_candidate(config, name, base_quant):
        counter["n"] += 1
        p = candidates_dir / f"{name}_{counter['n']}.gguf"
        p.write_bytes(b"0" * 1024)
        return str(p)

    monkeypatch.setattr(orch, "_build_candidate", fake_build_candidate)
    return orch, fake_tools


def _checkpoint_path(orch):
    return orch.output_dir / "_measured_checkpoint.json"


# ── Happy path: checkpoint deleted on successful completion ──────────────


def test_checkpoint_deleted_on_successful_completion(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    assert not _checkpoint_path(orch).exists()


def test_checkpoint_written_during_a_run(tmp_path, monkeypatch):
    """Sanity check: the checkpoint file exists WHILE _write_measured_checkpoint
    has been called at least once, by inspecting its final (pre-delete)
    content shape via a monkeypatched os.replace that captures the write."""
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    captured = {}
    import magicquant.orchestrator as orch_mod

    original_replace = orch_mod.os.replace

    def spy_replace(src, dst):
        from pathlib import Path
        captured["last"] = json.loads(Path(src).read_text())
        return original_replace(src, dst)

    monkeypatch.setattr(orch_mod.os, "replace", spy_replace)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    assert captured, "checkpoint was never written"
    assert "baseline_ppl" in captured["last"]
    assert "measured" in captured["last"]


# ── Kill-and-resume ───────────────────────────────────────────────────────


def test_kill_after_n_measurements_then_resume_skips_baseline_and_measured(
    tmp_path, monkeypatch
):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools._kill_after = 2  # allow baseline + 2 candidate measurements

    with pytest.raises(RuntimeError, match="simulated kill"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=5, verbose=False,
            seed_incumbents=False, seed=42,
        )

    assert _checkpoint_path(orch).exists()
    checkpoint = json.loads(_checkpoint_path(orch).read_text())
    n_measured_at_kill = len(checkpoint["measured"])
    assert n_measured_at_kill == 2

    # Resume: fresh orchestrator, same output_dir/source, no more killing.
    orch2, fake_tools2 = _make_orchestrator(tmp_path, monkeypatch)

    orch2.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=5, verbose=False,
        seed_incumbents=False, seed=42,
    )

    assert not _checkpoint_path(orch2).exists()  # completed successfully
    # Baseline must NOT have been re-measured on resume.
    assert fake_tools2.baseline_calls == 0
    # The 2 checkpointed candidates were restored verbatim, not rebuilt --
    # a rebuild would re-measure them with fake_tools2's OWN fresh
    # candidate_calls counter and produce a different "ppl" value.
    for key, restored_entry in checkpoint["measured"].items():
        assert key in orch2._measured
        assert orch2._measured[key]["ppl"] == restored_entry["ppl"]
    # The search continued past the resume point -- more than just the 2
    # restored candidates ended up measured by the time this run finished.
    assert len(orch2._measured) > n_measured_at_kill


def test_resumed_run_does_not_recall_perplexity_for_baseline(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools._kill_after = 1

    with pytest.raises(RuntimeError):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=5, verbose=False,
            seed_incumbents=False, seed=42,
        )

    orch2, fake_tools2 = _make_orchestrator(tmp_path, monkeypatch)
    orch2.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=5, verbose=False,
        seed_incumbents=False, seed=42,
    )

    # On resume the baseline is restored from the checkpoint --
    # calculate_perplexity must never be called with the source model path.
    assert fake_tools2.baseline_calls == 0


# ── Mismatch forces fresh ─────────────────────────────────────────────────


def test_seed_mismatch_forces_fresh_run(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False, seed=1,
    )
    # Successful run deletes its own checkpoint -- write one back manually to
    # simulate "a checkpoint exists from a run with a different seed".
    fake_checkpoint = {
        "version": 1,
        "seed": 999,
        "source_model": orch._source_identity(),
        "measurement_conditions": orch._current_measurement_conditions(),
        "baseline_ppl": 1.23,
        "baseline_provenance": "measured",
        "sensitivity_weights": {"E": 1.0},
        "probing_provenance": "measured",
        "kl": {"enabled": False, "base_logits_path": None, "corpus_path": None},
        "imatrix": {"active": False, "n_tensors": None},
        "measured": {"fake:key": {"config": {"E": "BF16"}, "ppl": 1.0}},
    }
    _checkpoint_path(orch).write_text(json.dumps(fake_checkpoint))

    orch2, fake_tools2 = _make_orchestrator(tmp_path, monkeypatch)
    orch2.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False, seed=1,
    )

    # Fresh baseline was measured for real (not the fabricated 1.23).
    assert orch2.baseline_ppl == 5.0
    assert "fake:key" not in orch2._measured


def test_source_identity_mismatch_forces_fresh_run(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    fake_checkpoint = {
        "version": 1,
        "seed": None,
        "source_model": {"path": "/completely/different/model.gguf", "size": 1, "mtime": 1.0},
        "measurement_conditions": orch._current_measurement_conditions(),
        "baseline_ppl": 1.23,
        "baseline_provenance": "measured",
        "sensitivity_weights": {"E": 1.0},
        "probing_provenance": "measured",
        "kl": {"enabled": False, "base_logits_path": None, "corpus_path": None},
        "imatrix": {"active": False, "n_tensors": None},
        "measured": {},
    }
    _checkpoint_path(orch).write_text(json.dumps(fake_checkpoint))

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    assert orch.baseline_ppl == 5.0


def test_measurement_conditions_mismatch_forces_fresh_run(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    fake_checkpoint = {
        "version": 1,
        "seed": None,
        "source_model": orch._source_identity(),
        "measurement_conditions": {"chunks": 999, "ctx_size": 512, "corpus": "/fake/corpus.txt"},
        "baseline_ppl": 1.23,
        "baseline_provenance": "measured",
        "sensitivity_weights": {"E": 1.0},
        "probing_provenance": "measured",
        "kl": {"enabled": False, "base_logits_path": None, "corpus_path": None},
        "imatrix": {"active": False, "n_tensors": None},
        "measured": {},
    }
    _checkpoint_path(orch).write_text(json.dumps(fake_checkpoint))

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    assert orch.baseline_ppl == 5.0


def test_resume_false_always_runs_fresh_even_with_matching_checkpoint(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )
    # Manually recreate a matching checkpoint (the successful run above
    # deleted its own).
    matching = {
        "version": 1,
        "seed": None,
        "source_model": orch._source_identity(),
        "measurement_conditions": orch._current_measurement_conditions(),
        "baseline_ppl": 1.23,
        "baseline_provenance": "measured",
        "sensitivity_weights": orch.sensitivity_weights,
        "probing_provenance": orch.probing_provenance,
        "kl": {"enabled": False, "base_logits_path": None, "corpus_path": None},
        "imatrix": {"active": False, "n_tensors": None},
        "measured": {},
    }
    _checkpoint_path(orch).write_text(json.dumps(matching))

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False, resume=False,
    )

    # resume=False must ignore even a matching checkpoint and remeasure.
    assert orch.baseline_ppl == 5.0


# ── Corrupted checkpoint ──────────────────────────────────────────────────


def test_corrupted_checkpoint_json_runs_fresh_not_crash(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    _checkpoint_path(orch).parent.mkdir(parents=True, exist_ok=True)
    _checkpoint_path(orch).write_text("{not valid json::::")

    all_configs, tiered = orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    assert orch.baseline_ppl == 5.0
    assert orch._measured
    assert not _checkpoint_path(orch).exists()  # completed + cleaned up


# ── KL base-logits reuse on resume ─────────────────────────────────────────


def test_kl_base_logits_reused_when_file_still_exists(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    save_calls = {"n": 0}

    def fake_save_base_logits(base_model_path, corpus_path, out_logits_path, **kw):
        save_calls["n"] += 1
        from pathlib import Path
        Path(out_logits_path).write_text("logits")
        return 5.0  # this pass's own "Final estimate: PPL" (fused baseline)

    monkeypatch.setattr(fake_tools, "save_base_logits", fake_save_base_logits)
    fake_tools._kill_after = 1

    with pytest.raises(RuntimeError):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=5, verbose=False,
            seed_incumbents=False, enable_kl=True,
        )
    assert save_calls["n"] == 1
    # The baseline was fused from THIS save_base_logits pass, not a separate
    # standalone calculate_perplexity call on the source model.
    assert orch.baseline_ppl == pytest.approx(5.0)
    assert fake_tools.baseline_calls == 0

    orch2, fake_tools2 = _make_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(fake_tools2, "save_base_logits", fake_save_base_logits)

    orch2.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=5, verbose=False,
        seed_incumbents=False, enable_kl=True,
    )

    # Base logits were reused from the checkpoint, not regenerated.
    assert save_calls["n"] == 1
    # Baseline was restored from the checkpoint (itself fused on the first
    # run), not re-measured via calculate_perplexity at all.
    assert orch2.baseline_ppl == pytest.approx(5.0)
    assert fake_tools2.baseline_calls == 0


def test_checkpoint_tolerates_numpy_typed_measurements(tmp_path):
    import numpy as np
    from pathlib import Path as _P
    from magicquant.orchestrator import MagicQuantOrchestrator

    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._search_seed = 42
    orch.baseline_ppl = 5.0
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch.sensitivity_weights = {"E": 0.5}
    orch._kl_base_logits_path = None
    orch._kl_corpus_path = None
    orch.source_model_path = str(tmp_path / "m.gguf")
    _P(orch.source_model_path).write_bytes(b"g")
    orch._llama_tools = None
    orch._llamacpp_path = None
    orch._imatrix = None
    orch._measured = {
        "a": {"config": {"E": "BF16"}, "ppl": np.float32(5.5),
              "measured_loss": np.float64(0.1),
              "kl": {"mean_kl": np.float32(0.01)},
              "bench": {"tg": np.int64(9)}},
    }
    out = tmp_path / "ckpt.json"
    orch._write_measured_checkpoint(out)
    import json as _json
    data = _json.loads(out.read_text())
    assert data["measured"]["a"]["kl"]["mean_kl"] == pytest.approx(0.01)


# ── BLOCKER: measurement_invalid / corpus_path must survive a checkpoint
# round-trip, or the resume-boundary defeats the impossible-measurement
# guard entirely ────────────────────────────────────────────────────────


def test_measurement_invalid_and_corpus_path_survive_checkpoint_round_trip(tmp_path):
    """_write_measured_checkpoint writes each measurement via an explicit
    key whitelist. If that whitelist is never extended with
    ``measurement_invalid``/``corpus_path``, a resumed run loses both
    fields -- info.get("measurement_invalid") comes back None (falsy) and
    a physically-impossible candidate that was correctly flagged before the
    kill can win a tier again after resume. This must not regress."""
    from magicquant.orchestrator import MagicQuantOrchestrator

    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch._search_seed = 42
    orch.baseline_ppl = 34.8363
    orch.baseline_provenance = "measured"
    orch.probing_provenance = "measured"
    orch.sensitivity_weights = {"E": 0.5}
    orch._kl_base_logits_path = None
    orch._kl_corpus_path = None
    orch.source_model_path = str(tmp_path / "m.gguf")
    from pathlib import Path as _P
    _P(orch.source_model_path).write_bytes(b"g")
    orch._llama_tools = None
    orch._llamacpp_path = None
    orch._imatrix = None
    orch._measured = {
        "impossible": {
            "config": {"E": "IMPOSSIBLE"},
            # Impossibly low measured_loss -- the "best" number in the tier
            # by min(), which is exactly why it must stay excluded.
            "measured_loss": -0.9225,
            "size_gb": 4.0,
            "measurement_invalid": True,
            "corpus_path": "/fake/corpus.txt",
        },
        "real": {
            "config": {"E": "REAL"},
            "measured_loss": 0.15,
            "size_gb": 4.0,
            "measurement_invalid": False,
            "corpus_path": "/fake/corpus.txt",
        },
    }

    ckpt_path = tmp_path / "ckpt.json"
    orch._write_measured_checkpoint(ckpt_path)
    checkpoint = json.loads(ckpt_path.read_text())

    # Simulate exactly what run_measured_search's resume path does with a
    # loaded checkpoint (orchestrator.py ~line 424-425).
    orch2 = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch2._kl_weight = 0.0
    orch2._speed_aware = False
    orch2._measured = {}
    for key, entry in checkpoint.get("measured", {}).items():
        orch2._measured[key] = dict(entry)

    assert orch2._measured["impossible"]["measurement_invalid"] is True, (
        "measurement_invalid must survive a checkpoint write+load round trip"
    )
    assert orch2._measured["impossible"]["corpus_path"] == "/fake/corpus.txt", (
        "corpus_path must survive a checkpoint write+load round trip"
    )

    result = orch2._select_final_survivors(baseline_gb=8.0)
    tier = orch2._classify_tier(4.0, 8.0)
    assert result[tier]["config"] == {"E": "REAL"}, (
        "a candidate flagged measurement_invalid before a kill must still "
        "be excluded from tier selection after a checkpoint resume"
    )


# ── BLOCKER (F1): the KL base-logits file (_kl_base_logits.kld) is huge --
# ── roughly chunks * ctx_size * vocab_size * 2 bytes, verified 69 GB for a
# ── 27B model at 100 chunks -- and with probe_kl defaulting True, every
# ── measured run now creates one. It must be deleted once the run
# ── completes successfully, but kept when a run fails/is killed (so a
# ── resume doesn't have to pay the ~18-min recapture again), and kept
# ── regardless of outcome when the caller explicitly opts to retain it.
# ─────────────────────────────────────────────────────────────────────────


def test_kl_base_logits_deleted_on_successful_completion(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    # Precondition: probe_kl defaults True and the fake tools implement
    # save_base_logits, so a real capture must have happened -- otherwise
    # this test would trivially pass with nothing ever created.
    assert orch._kl_base_logits_path is not None
    assert not Path(orch._kl_base_logits_path).exists(), (
        "the KL base-logits file must be deleted after a successful run"
    )


def test_keep_kl_base_logits_flag_retains_file_after_success(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
        keep_kl_base_logits=True,
    )

    assert orch._kl_base_logits_path is not None
    assert Path(orch._kl_base_logits_path).exists(), (
        "keep_kl_base_logits=True must retain the file even after success"
    )


def test_keep_kl_logits_env_var_retains_file_after_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGICQUANT_KEEP_KL_LOGITS", "1")
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    assert orch._kl_base_logits_path is not None
    assert Path(orch._kl_base_logits_path).exists(), (
        "MAGICQUANT_KEEP_KL_LOGITS=1 must retain the file even after success"
    )


def test_kl_base_logits_retained_when_run_fails_mid_measurement(tmp_path, monkeypatch):
    """A killed/failed run must leave the KL base-logits file in place --
    it's the exact thing a resume needs to skip the expensive recapture."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools._kill_after = 1

    with pytest.raises(RuntimeError, match="simulated kill"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=5, verbose=False,
            seed_incumbents=False, seed=42,
        )

    assert orch._kl_base_logits_path is not None
    assert Path(orch._kl_base_logits_path).exists(), (
        "a failed/killed run must retain the KL base-logits file for resume"
    )


# ── BLOCKER (F2): a checkpoint recorded under a PPL-only objective must not
# ── silently half-resume into a KL-blended run, but a checkpoint written
# ── before enable_kl/kl_weight existed as conditions must still resume a
# ── same-objective (PPL-only) run -- every pre-fix checkpoint was PPL-only.
# ─────────────────────────────────────────────────────────────────────────


_BASE_CONDITIONS = {"chunks": None, "ctx_size": 512, "corpus": "/fake/corpus.txt"}


def test_conditions_match_old_checkpoint_ppl_only_config_resumes():
    """Combination 1: old checkpoint (no enable_kl/kl_weight keys at all --
    the pre-fix shape) + a fresh enable_kl=False config -- must resume."""
    stored = dict(_BASE_CONDITIONS)
    current = {**_BASE_CONDITIONS, "enable_kl": False, "kl_weight": 0.0}
    assert MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_old_checkpoint_kl_blended_config_rejected():
    """Combination 2: the same old (pre-fix) checkpoint + a fresh
    enable_kl=True config -- must be rejected. This is the actual bug F2
    fixes: without this, the checkpoint above would ALSO have resumed here,
    silently mixing PPL-only measurements into a run everything else
    believes is KL-blended."""
    stored = dict(_BASE_CONDITIONS)
    current = {**_BASE_CONDITIONS, "enable_kl": True, "kl_weight": 0.1}
    assert not MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_new_checkpoint_same_ppl_only_objective_resumes():
    """Combination 3: a post-fix checkpoint recorded with enable_kl=False +
    a fresh enable_kl=False config (same objective) -- must resume."""
    stored = {**_BASE_CONDITIONS, "enable_kl": False, "kl_weight": 0.0}
    current = dict(stored)
    assert MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_new_checkpoint_same_kl_blended_objective_resumes():
    """Combination 4: a post-fix checkpoint recorded with enable_kl=True +
    a fresh run with the SAME enable_kl=True/kl_weight (same objective) --
    must resume."""
    stored = {**_BASE_CONDITIONS, "enable_kl": True, "kl_weight": 0.3}
    current = dict(stored)
    assert MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_kl_weight_change_with_kl_off_is_ignored():
    """kl_weight only participates in the comparison when the CURRENT run
    has enable_kl on -- a stored weight that differs while blending is off
    changes nothing real (run_measured_search forces kl_weight to 0.0
    whenever enable_kl is False) and must not force an unnecessary
    re-measurement."""
    stored = {**_BASE_CONDITIONS, "enable_kl": False, "kl_weight": 0.5}
    current = {**_BASE_CONDITIONS, "enable_kl": False, "kl_weight": 0.0}
    assert MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_kl_weight_change_with_kl_on_is_rejected():
    """A genuine kl_weight change while blending is ON for both sides IS a
    real objective change (it alters _select_final_survivors' ranking) and
    must be rejected."""
    stored = {**_BASE_CONDITIONS, "enable_kl": True, "kl_weight": 0.5}
    current = {**_BASE_CONDITIONS, "enable_kl": True, "kl_weight": 0.1}
    assert not MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_old_style_checkpoint_without_kl_keys_resumes_under_ppl_only_run(
    tmp_path, monkeypatch
):
    """Integration-level check for combination 1: a checkpoint dict shaped
    like one written before this fix (no enable_kl/kl_weight keys in
    measurement_conditions at all) must still resume a fresh enable_kl=False
    (default) run -- not force a re-measurement from scratch."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools._kill_after = 2

    with pytest.raises(RuntimeError, match="simulated kill"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=5, verbose=False,
            seed_incumbents=False, seed=42,
        )

    # Strip the new keys from the checkpoint the run above actually wrote,
    # simulating the pre-fix shape.
    ckpt_path = _checkpoint_path(orch)
    checkpoint = json.loads(ckpt_path.read_text())
    checkpoint["measurement_conditions"].pop("enable_kl", None)
    checkpoint["measurement_conditions"].pop("kl_weight", None)
    ckpt_path.write_text(json.dumps(checkpoint))

    orch2, fake_tools2 = _make_orchestrator(tmp_path, monkeypatch)
    orch2.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=5, verbose=False,
        seed_incumbents=False, seed=42,
    )

    assert fake_tools2.baseline_calls == 0, (
        "an old-shaped PPL-only checkpoint must still resume a PPL-only run"
    )


def test_old_style_checkpoint_without_kl_keys_rejected_under_kl_blended_run(
    tmp_path, monkeypatch
):
    """Integration-level check for combination 2: the same pre-fix
    checkpoint shape must be REJECTED when the new run wants enable_kl=True
    -- resuming would silently mix PPL-only measurements into a run
    everything else believes is KL-blended."""
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False, seed=7,
    )
    # Successful run deleted its own checkpoint -- rebuild one shaped like a
    # pre-fix (no enable_kl/kl_weight keys) checkpoint.
    conditions = orch._current_measurement_conditions()
    conditions.pop("enable_kl", None)
    conditions.pop("kl_weight", None)
    fake_checkpoint = {
        "version": 2,
        "seed": 7,
        "source_model": orch._source_identity(),
        "measurement_conditions": conditions,
        "baseline_ppl": 1.23,
        "baseline_provenance": "measured",
        "sensitivity_weights": {"E": 1.0},
        "probing_provenance": "measured",
        "kl": {"enabled": False, "base_logits_path": None, "corpus_path": None},
        "imatrix": {"active": False, "n_tensors": None},
        "measured": {"fake:key": {"config": {"E": "BF16"}, "ppl": 1.0}},
    }
    _checkpoint_path(orch).write_text(json.dumps(fake_checkpoint))

    orch2, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch2.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False, seed=7,
        enable_kl=True,
    )

    # Rejected -- fresh baseline was measured for real, the fake
    # checkpoint's bogus measurement was never restored.
    assert orch2.baseline_ppl == 5.0
    assert "fake:key" not in orch2._measured


# ── Fail-fast llama.cpp arch check: instrument persistence + checkpoint ──
# ── compat (2026-08 multi-build-coexistence fix) ──────────────────────────
# See magicquant/utils/llamacpp.py's binary_supports_arch. The resolved
# perplexity-tool path and arch-check verdict are persisted (additively, at
# the tail) into BOTH search_results.json and the measured checkpoint; a
# checkpoint missing "llamacpp_binary" (written before this fix) is legacy-
# compatible, but a PRESENT-and-different one is a real measurement-
# instrument change and must force a fresh run, same as any other
# measurement-condition mismatch.


def test_results_json_has_llamacpp_binary_and_arch_check_keys(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.perplexity_tool = "/fake/llama-perplexity"

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
    )

    results = json.loads((orch.output_dir / "search_results.json").read_text())
    assert results["llamacpp_binary"] == "/fake/llama-perplexity"
    assert results["llamacpp_arch_check"] in ("supported", "unknown", "skipped")
    # Additive, tail-appended -- same key-order contract as
    # _serialize_measurement's per-measurement fields.
    assert list(results.keys())[-2:] == ["llamacpp_binary", "llamacpp_arch_check"]


def test_checkpoint_has_llamacpp_binary_and_arch_check_keys(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.perplexity_tool = "/fake/llama-perplexity"
    fake_tools._kill_after = 1

    with pytest.raises(RuntimeError, match="simulated kill"):
        orch.run_measured_search(
            search_generations=2, population_size=8,
            measurement_rounds=1, candidates_per_round=5, verbose=False,
            seed_incumbents=False, seed=3,
        )

    checkpoint = json.loads(_checkpoint_path(orch).read_text())
    assert checkpoint["llamacpp_binary"] == "/fake/llama-perplexity"
    assert checkpoint["llamacpp_arch_check"] in ("supported", "unknown", "skipped")
    assert list(checkpoint.keys())[-2:] == ["llamacpp_binary", "llamacpp_arch_check"]


def _base_fake_checkpoint(orch):
    """Minimal well-formed checkpoint dict matching orch's own identity/
    conditions -- deliberately WITHOUT an "llamacpp_binary" key, i.e. the
    legacy (pre-this-fix) shape. Callers add "llamacpp_binary" themselves
    to test the present-and-matching / present-and-different cases."""
    return {
        "version": 2,
        "seed": None,
        "source_model": orch._source_identity(),
        "measurement_conditions": orch._current_measurement_conditions(),
        "baseline_ppl": 1.23,
        "baseline_provenance": "measured",
        "sensitivity_weights": {},
        "probing_provenance": "measured",
        "kl": {"enabled": False, "base_logits_path": None, "corpus_path": None},
        "imatrix": {"active": False, "n_tensors": None},
        "measured": {},
    }


def test_load_matching_checkpoint_accepts_missing_llamacpp_binary_key(tmp_path, monkeypatch):
    """A checkpoint written before this fix has no llamacpp_binary key at
    all -- must still be treated as compatible (legacy)."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.perplexity_tool = "/build-A/llama-perplexity"
    checkpoint = _base_fake_checkpoint(orch)
    assert "llamacpp_binary" not in checkpoint  # sanity: this IS the legacy shape
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._load_matching_checkpoint(path, verbose=False) is not None


def test_load_matching_checkpoint_accepts_matching_llamacpp_binary(tmp_path, monkeypatch):
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.perplexity_tool = "/build-A/llama-perplexity"
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["llamacpp_binary"] = "/build-A/llama-perplexity"
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._load_matching_checkpoint(path, verbose=False) is not None


def test_load_matching_checkpoint_rejects_present_different_llamacpp_binary(tmp_path, monkeypatch):
    """Different instrument = different measurements: a checkpoint recorded
    under a DIFFERENT llama.cpp binary must not resume, even though every
    other condition (chunks/ctx_size/corpus/seed/source) matches."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.perplexity_tool = "/build-A/llama-perplexity"
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["llamacpp_binary"] = "/build-B/llama-perplexity"
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._load_matching_checkpoint(path, verbose=False) is None


def test_load_matching_checkpoint_accepts_symlink_spelling_of_same_llamacpp_binary(
    tmp_path, monkeypatch
):
    """Opus review: the comparison must go through os.path.realpath() on
    BOTH sides so a relative-vs-absolute or symlinked spelling of the SAME
    binary doesn't spuriously reject a valid checkpoint -- that would throw
    away exactly the hours the checkpoint protects. A real symlink here
    (not just string normalization) proves actual filesystem resolution,
    not merely path-string cleanup."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)

    real_bin_dir = tmp_path / "real_build" / "bin"
    real_bin_dir.mkdir(parents=True)
    real_bin = real_bin_dir / "llama-perplexity"
    real_bin.write_bytes(b"fake binary bytes")

    link_dir = tmp_path / "pinned_link"
    link_dir.symlink_to(real_bin_dir, target_is_directory=True)
    linked_bin = link_dir / "llama-perplexity"

    # The checkpoint recorded the SYMLINK spelling; this run resolves the
    # REAL path -- same binary, different spelling.
    fake_tools.perplexity_tool = str(real_bin)
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["llamacpp_binary"] = str(linked_bin)
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._load_matching_checkpoint(path, verbose=False) is not None


def test_load_matching_checkpoint_rejects_stored_binary_when_current_is_unresolvable(
    tmp_path, monkeypatch
):
    """current_binary=None (no resolvable perplexity_tool at all this run)
    is DELIBERATELY still a mismatch against a stored path -- this run
    can't confirm it's even using a binary, let alone the same one."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    assert not hasattr(fake_tools, "perplexity_tool")  # sanity: unresolvable
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["llamacpp_binary"] = "/build-A/llama-perplexity"
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._load_matching_checkpoint(path, verbose=False) is None


# ── PR6 review F1: the imatrix_id backward-compat default must derive from
# ── the checkpoint's own top-level "imatrix" block, not hardcode inactive --
# ── an imatrix-active pre-fix checkpoint resuming into a use_imatrix=False
# ── run must be REJECTED, not silently matched on a false "both inactive".
# ─────────────────────────────────────────────────────────────────────────


def test_load_matching_checkpoint_rejects_imatrix_active_prefix_checkpoint_under_no_imatrix_run(
    tmp_path, monkeypatch
):
    """The actual F1 bug: a checkpoint written before "imatrix_id" existed
    in measurement_conditions, but whose top-level "imatrix" block (recorded
    since 2026-07-05, i.e. already present on every pre-fix checkpoint this
    branch could ever meet) says imatrix WAS active for its candidates --
    must NOT resume a fresh run with no imatrix. Before the fix, the missing
    "imatrix_id" key defaulted to {"active": False}, which spuriously
    matched a current {"active": False} and resumed, silently blending
    imatrix-weighted candidate measurements with freshly-taken unweighted
    ones in a single ranking."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["measurement_conditions"].pop("imatrix_id", None)
    checkpoint["imatrix"] = {"active": True, "n_tensors": 5}
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._imatrix is None  # sanity: current run genuinely has no imatrix
    assert orch._load_matching_checkpoint(path, verbose=False) is None


def test_load_matching_checkpoint_rejects_imatrix_active_prefix_checkpoint_even_under_imatrix_run(
    tmp_path, monkeypatch
):
    """Same pre-fix shape, but the current run DOES have an imatrix active
    too -- still rejected, since the stored checkpoint cannot prove it was
    the SAME imatrix (no hash was captured before the imatrix_id key
    existed). Not trusted rather than partially trusted, in both
    directions."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    orch._imatrix = {"blk.0.attn_q.weight": [1.0, 2.0, 3.0]}
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["measurement_conditions"].pop("imatrix_id", None)
    checkpoint["imatrix"] = {"active": True, "n_tensors": 5}
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._load_matching_checkpoint(path, verbose=False) is None


def test_load_matching_checkpoint_resumes_prefix_checkpoint_with_no_imatrix_record_at_all(
    tmp_path, monkeypatch
):
    """The backward-compat path F1 must NOT break: a checkpoint with no
    "imatrix_id" key AND no imatrix record at all (either no top-level
    "imatrix" key, simulating a checkpoint older than 2026-07-05, or an
    explicit {"active": False}) genuinely has no imatrix evidence either
    way -- it must still resume a fresh no-imatrix run, exactly as before
    this fix."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["measurement_conditions"].pop("imatrix_id", None)
    del checkpoint["imatrix"]  # older than the "imatrix" block itself
    path = _checkpoint_path(orch)
    path.write_text(json.dumps(checkpoint))

    assert orch._imatrix is None
    assert orch._load_matching_checkpoint(path, verbose=False) is not None


def test_mismatched_llamacpp_binary_forces_fresh_measured_search(tmp_path, monkeypatch):
    """Integration-level companion to the direct _load_matching_checkpoint
    tests above: end to end through run_measured_search, a checkpoint
    recorded under a different llama.cpp binary must not silently resurrect
    a stale measurement, and the baseline gets remeasured for real."""
    orch, fake_tools = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools.perplexity_tool = "/build-A/llama-perplexity"
    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False, seed=9,
    )
    # Successful run deleted its own checkpoint -- rebuild one recorded
    # under a DIFFERENT llama.cpp binary (e.g. a later run submission
    # resolving to the fork build instead of the pinned one).
    checkpoint = _base_fake_checkpoint(orch)
    checkpoint["seed"] = 9
    checkpoint["measured"] = {"fake:key": {"config": {"E": "BF16"}, "ppl": 1.0}}
    checkpoint["llamacpp_binary"] = "/build-B/llama-perplexity"
    _checkpoint_path(orch).write_text(json.dumps(checkpoint))

    orch2, fake_tools2 = _make_orchestrator(tmp_path, monkeypatch)
    fake_tools2.perplexity_tool = "/build-A/llama-perplexity"  # differs from stored build-B
    orch2.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False, seed=9,
    )

    assert orch2.baseline_ppl == 5.0
    assert "fake:key" not in orch2._measured
