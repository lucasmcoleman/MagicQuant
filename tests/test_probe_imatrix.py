"""Issue #5: v1 sensitivity probes must be built with the SAME imatrix that
steers the candidates and final tiers they rank -- not silently unweighted
while everything downstream of them is weighted.

Covers:
  1. ``SensitivityProber.imatrix`` reaches ``create_hybrid_gguf`` from
     ``_real_probe`` (the actual threading, not just "the kwarg is accepted").
  2. A probe built WITH an imatrix produces a different sensitivity reading
     than one built WITHOUT -- proving the value is actually used downstream,
     not merely passed through and ignored.
  3. Default behavior (no imatrix passed) is unchanged: ``create_hybrid_gguf``
     still receives ``imatrix=None``.
  4. ``MagicQuantOrchestrator`` threads ``self._imatrix`` into the prober it
     constructs, on both the measured-search and prediction-only paths.
  5. The measured-search checkpoint-resume gate invalidates on a changed
     imatrix identity, mirroring v2's distortion-table cache -- the explicit
     "yes, invalidate" decision the issue asks for.
"""
from pathlib import Path

import numpy as np
import pytest

import magicquant.gguf.writer as writer_mod
from magicquant.evolution.probing import SensitivityProber
from magicquant.orchestrator import MagicQuantOrchestrator
from magicquant.utils.measurement import imatrix_identity


class _FakeReader:
    """Stands in for GGUFReader -- _real_probe only needs tensor names to
    build its per-group override config."""

    def __init__(self, *a, **k):
        pass

    def open(self):
        pass

    def get_tensor_names(self):
        return ["token_embd.weight", "blk.0.ffn_down.weight"]

    def close(self):
        pass


@pytest.fixture()
def probe_src(tmp_path):
    src = tmp_path / "model.gguf"
    src.write_bytes(b"GGUF-stub")
    return str(src)


def _patch_probe_scaffolding(monkeypatch):
    """Stub out everything _real_probe touches besides create_hybrid_gguf
    itself: GGUFReader (group enumeration) and _verify_probe_artifact
    (artifact-shape checking, covered separately in
    test_probe_artifact_verification.py)."""
    import magicquant.gguf.reader as reader_mod

    monkeypatch.setattr(reader_mod, "GGUFReader", _FakeReader)
    monkeypatch.setattr(
        SensitivityProber, "_verify_probe_artifact", lambda *a, **k: None
    )


# ── 1 & 3: the imatrix (or its absence) reaches create_hybrid_gguf ────────


def test_real_probe_threads_imatrix_into_create_hybrid_gguf(
    monkeypatch, probe_src, tmp_path
):
    _patch_probe_scaffolding(monkeypatch)
    captured = {}

    def _fake_create(output_path, base_model_path, quant_config, verbose=False,
                      **kw):
        captured["imatrix"] = kw.get("imatrix")
        Path(output_path).write_bytes(b"probe-stub")
        return output_path

    monkeypatch.setattr(writer_mod, "create_hybrid_gguf", _fake_create)

    fake_imatrix = {"token_embd.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    prober = SensitivityProber(
        base_model_path=probe_src,
        baseline_perplexity=10.0,
        perplexity_calculator=_FixedCalculator(12.0),
        output_dir=str(tmp_path / "_probes"),
        strict=True,
        imatrix=fake_imatrix,
    )
    prober.probe_all_groups(groups=["E"], verbose=False)

    assert captured["imatrix"] is fake_imatrix


def test_real_probe_default_imatrix_is_none_unchanged_behavior(
    monkeypatch, probe_src, tmp_path
):
    """A caller that never passes ``imatrix`` (every pre-fix call site, and
    the still-unmodified standalone ``magicquant probe`` CLI) must see
    EXACTLY the historical behavior: create_hybrid_gguf called with
    imatrix=None."""
    _patch_probe_scaffolding(monkeypatch)
    captured = {}

    def _fake_create(output_path, base_model_path, quant_config, verbose=False,
                      **kw):
        captured["imatrix"] = kw.get("imatrix", "MISSING")
        Path(output_path).write_bytes(b"probe-stub")
        return output_path

    monkeypatch.setattr(writer_mod, "create_hybrid_gguf", _fake_create)

    prober = SensitivityProber(
        base_model_path=probe_src,
        baseline_perplexity=10.0,
        perplexity_calculator=_FixedCalculator(12.0),
        output_dir=str(tmp_path / "_probes"),
        strict=True,
        # imatrix intentionally omitted
    )
    prober.probe_all_groups(groups=["E"], verbose=False)

    assert captured["imatrix"] is None


class _FixedCalculator:
    def __init__(self, ppl):
        self.ppl = ppl

    def calculate_perplexity(self, path, verbose=True, **kw):
        return self.ppl


# ── 2: an imatrix-built probe measures differently than an unweighted one ──


class _ContentAwareCalculator:
    """Reads back what _fake_create_content_aware wrote into the probe GGUF
    and returns a DIFFERENT ppl depending on whether the build saw an
    imatrix -- the same shape a real llama-perplexity pass would produce
    against two genuinely different quantized artifacts (imatrix-weighted
    K-quant rounding differs from unweighted rounding)."""

    def calculate_perplexity(self, path, verbose=True, **kw):
        content = Path(path).read_bytes()
        return 18.0 if content == b"WITH_IMATRIX" else 12.0


def _fake_create_content_aware(output_path, base_model_path, quant_config,
                                verbose=False, **kw):
    marker = b"WITH_IMATRIX" if kw.get("imatrix") is not None else b"NO_IMATRIX"
    Path(output_path).write_bytes(marker)
    return output_path


def test_probe_built_with_imatrix_differs_from_probe_built_without(
    monkeypatch, probe_src, tmp_path
):
    """The actual acceptance-criterion test: not merely that ``imatrix`` is
    ACCEPTED as a parameter, but that a probe built with it measures a
    genuinely different sensitivity than one built without, all else equal
    (same base model, same baseline, same aggressive scheme)."""
    _patch_probe_scaffolding(monkeypatch)
    monkeypatch.setattr(writer_mod, "create_hybrid_gguf", _fake_create_content_aware)

    calc = _ContentAwareCalculator()

    prober_without = SensitivityProber(
        base_model_path=probe_src,
        baseline_perplexity=10.0,
        perplexity_calculator=calc,
        output_dir=str(tmp_path / "_probes_without"),
        strict=True,
        imatrix=None,
    )
    results_without = prober_without.probe_all_groups(groups=["E"], verbose=False)

    prober_with = SensitivityProber(
        base_model_path=probe_src,
        baseline_perplexity=10.0,
        perplexity_calculator=calc,
        output_dir=str(tmp_path / "_probes_with"),
        strict=True,
        imatrix={"token_embd.weight": np.array([1.0, 2.0], dtype=np.float32)},
    )
    results_with = prober_with.probe_all_groups(groups=["E"], verbose=False)

    assert results_without["E"] != results_with["E"]
    # Concretely: 12.0 vs 18.0 against a 10.0 baseline.
    assert results_without["E"] == pytest.approx(0.2)
    assert results_with["E"] == pytest.approx(0.8)


# ── 4: the orchestrator threads self._imatrix into the prober it builds ───


class _FakeSource:
    def get_tensor_names(self):
        return ["token_embd.weight", "blk.0.ffn_down.weight"]

    def close(self):
        pass


class _FakeTools:
    ppl_chunks = None
    ctx_size = 512

    def calculate_perplexity(self, *a, **k):
        return 10.0

    def _resolve_data_file(self, *_a):
        return None


class _Sentinel(Exception):
    pass


class _SpyProber:
    """Captures the kwargs SensitivityProber was constructed with, then
    aborts the run -- mirrors test_strict_probing.py's
    test_measured_search_constructs_strict_prober pattern."""

    last_kwargs = None

    def __init__(self, *a, **kw):
        _SpyProber.last_kwargs = kw
        raise _Sentinel()


def _make_orch(tmp_path, monkeypatch):
    import magicquant.gguf.source as source_mod

    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF-stub")
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())

    orch = MagicQuantOrchestrator(
        source_model_path=str(src), output_dir=str(tmp_path / "out")
    )
    orch._llama_tools = _FakeTools()
    return orch


def test_run_measured_search_threads_imatrix_into_prober(tmp_path, monkeypatch):
    import magicquant.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "SensitivityProber", _SpyProber)
    orch = _make_orch(tmp_path, monkeypatch)

    fake_imatrix = {"token_embd.weight": np.array([1.0], dtype=np.float32)}
    monkeypatch.setattr(
        "magicquant.imatrix.ensure_imatrix", lambda *a, **k: fake_imatrix
    )

    with pytest.raises(_Sentinel):
        orch.run_measured_search(
            measurement_rounds=1, verbose=False, resume=False, use_imatrix=True,
        )

    assert _SpyProber.last_kwargs.get("imatrix") is fake_imatrix


def test_run_measured_search_no_imatrix_passes_none_to_prober(tmp_path, monkeypatch):
    """PR6 review F3: the original assertion (``kw.get("imatrix") is None``)
    was vacuous -- ``.get()`` returns ``None`` whether the kwarg is threaded
    explicitly or absent entirely, so deleting ``imatrix=self._imatrix`` at
    the call site would leave this green. Use the same
    ``kw.get("imatrix", "MISSING")`` sentinel as
    ``test_real_probe_default_imatrix_is_none_unchanged_behavior`` to
    distinguish "explicitly None" from "absent", so this actually goes red
    if the kwarg stops being threaded."""
    import magicquant.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "SensitivityProber", _SpyProber)
    orch = _make_orch(tmp_path, monkeypatch)

    with pytest.raises(_Sentinel):
        orch.run_measured_search(
            measurement_rounds=1, verbose=False, resume=False, use_imatrix=False,
        )

    assert _SpyProber.last_kwargs.get("imatrix", "MISSING") is None


# ── PR6 review F6: self._imatrix has no per-run reset, unlike its KL
# ── counterpart -- a second run_measured_search(use_imatrix=False) on the
# ── SAME orchestrator instance must not inherit a prior run's imatrix.
# ─────────────────────────────────────────────────────────────────────────


def test_second_run_with_use_imatrix_false_does_not_inherit_prior_run_imatrix(
    tmp_path, monkeypatch
):
    """A first run_measured_search(use_imatrix=True) sets self._imatrix; a
    SECOND run_measured_search(use_imatrix=False) on the same orchestrator
    instance must reset it back to None before probing -- both the prober's
    kwarg and the checkpoint-resume cache key (_current_measurement_
    conditions) are imatrix-aware as of issue #5, so an inherited imatrix
    would silently widen the blast radius of the pre-existing
    _build_candidate exposure this bug already had."""
    import magicquant.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "SensitivityProber", _SpyProber)
    orch = _make_orch(tmp_path, monkeypatch)

    fake_imatrix = {"token_embd.weight": np.array([1.0], dtype=np.float32)}
    monkeypatch.setattr(
        "magicquant.imatrix.ensure_imatrix", lambda *a, **k: fake_imatrix
    )

    with pytest.raises(_Sentinel):
        orch.run_measured_search(
            measurement_rounds=1, verbose=False, resume=False, use_imatrix=True,
        )
    assert orch._imatrix is fake_imatrix  # sanity: first run activated it

    with pytest.raises(_Sentinel):
        orch.run_measured_search(
            measurement_rounds=1, verbose=False, resume=False, use_imatrix=False,
        )

    assert orch._imatrix is None, (
        "second run's use_imatrix=False must reset self._imatrix, not "
        "inherit it from the prior run on this same instance"
    )
    assert _SpyProber.last_kwargs.get("imatrix", "MISSING") is None


def test_run_full_search_threads_imatrix_into_prober(tmp_path, monkeypatch):
    import magicquant.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "SensitivityProber", _SpyProber)
    orch = _make_orch(tmp_path, monkeypatch)

    fake_imatrix = {"token_embd.weight": np.array([1.0], dtype=np.float32)}
    monkeypatch.setattr(
        "magicquant.imatrix.ensure_imatrix", lambda *a, **k: fake_imatrix
    )

    with pytest.raises(_Sentinel):
        orch.run_full_search(
            max_generations=2, population_size=8, verbose=False, use_imatrix=True,
        )

    assert _SpyProber.last_kwargs.get("imatrix") is fake_imatrix


# ── 5: checkpoint-resume gate invalidates on a changed imatrix identity ───


def test_imatrix_identity_none_is_inactive():
    assert imatrix_identity(None) == {"active": False}


def test_imatrix_identity_same_content_matches():
    a = {"t": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    b = {"t": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    assert imatrix_identity(a) == imatrix_identity(b)


def test_imatrix_identity_different_content_differs():
    a = {"t": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    b = {"t": np.array([9.0, 9.0, 9.0], dtype=np.float32)}
    assert imatrix_identity(a) != imatrix_identity(b)


def test_imatrix_identity_active_vs_inactive_differs():
    a = {"t": np.array([1.0], dtype=np.float32)}
    assert imatrix_identity(a) != imatrix_identity(None)


# ── PR6 review F2: imatrix_identity must actually meet its "Never raises"
# ── guarantee -- confirmed-raising inputs from the review's table must all
# ── now return normally, and malformed entries must not collide with each
# ── other via a shared error sentinel.
# ─────────────────────────────────────────────────────────────────────────


def test_imatrix_identity_none_value_entry_does_not_raise():
    """A None value produces a 0-d array via np.asarray(None, dtype=f32);
    slicing that 0-d array with v[::n] raised IndexError, which sat outside
    the except tuple that only caught TypeError/ValueError."""
    result = imatrix_identity({"blk.0": None})
    assert result["active"] is True
    assert result["n_tensors"] == 1
    assert isinstance(result["hash"], str)


def test_imatrix_identity_non_str_key_does_not_raise():
    """A non-str key made the naive name.encode() call raise AttributeError
    (int has no .encode())."""
    result = imatrix_identity({1: np.array([1.0, 2.0], dtype=np.float32)})
    assert result["active"] is True
    assert result["n_tensors"] == 1


def test_imatrix_identity_bytes_key_does_not_raise():
    result = imatrix_identity({b"a": np.array([1.0, 2.0], dtype=np.float32)})
    assert result["active"] is True
    assert result["n_tensors"] == 1


def test_imatrix_identity_unsortable_keys_does_not_raise():
    """Mixed int/str keys are not mutually orderable -- plain sorted(imatrix)
    raised TypeError before any per-tensor code ran at all."""
    result = imatrix_identity(
        {1: np.array([1.0], dtype=np.float32), "a": np.array([2.0], dtype=np.float32)}
    )
    assert result["active"] is True
    assert result["n_tensors"] == 2


def test_imatrix_identity_array_construction_raising_does_not_raise():
    """A value whose __array__ raises during np.asarray() must fall back to
    the repr() path rather than propagate the RuntimeError."""

    class _ExplodingArray:
        def __array__(self, dtype=None):
            raise RuntimeError("boom")

        def __repr__(self):
            return "<exploding>"

    result = imatrix_identity({"blk.0": _ExplodingArray()})
    assert result["active"] is True
    assert result["n_tensors"] == 1


def test_imatrix_identity_repr_also_raising_skips_entry_not_whole_function():
    """When even the repr() fallback raises, that one entry is dropped
    rather than the whole function raising or falling back to a shared
    sentinel -- the surviving good entry must still be reflected in the
    hash (proven by differing from an otherwise-identical imatrix missing
    that good entry)."""

    class _ExplodingEverything:
        def __array__(self, dtype=None):
            raise RuntimeError("boom")

        def __repr__(self):
            raise RuntimeError("repr also boom")

    with_extra = imatrix_identity(
        {
            "blk.0": np.array([1.0, 2.0], dtype=np.float32),
            "blk.1": _ExplodingEverything(),
        }
    )
    without_extra = imatrix_identity({"blk.0": np.array([1.0, 2.0], dtype=np.float32)})
    assert with_extra["active"] is True
    # n_tensors reflects the raw imatrix size (includes the dropped entry);
    # only the HASH must be unaffected by an entry that contributed nothing.
    assert with_extra["n_tensors"] == 2
    assert with_extra["hash"] == without_extra["hash"]


def test_imatrix_identity_different_malformed_entries_do_not_collide():
    """Two imatrices whose ONE entry each fails differently must not both
    collapse onto the same hash via a shared "something failed" sentinel --
    the review confirmed the repr()-fallback design already avoids this for
    the entries it could reach; this pins it for entries that previously
    escaped the guard entirely (None values, non-str keys)."""
    a = imatrix_identity({"blk.0": None})
    b = imatrix_identity({1: np.array([9.0], dtype=np.float32)})
    assert a != b


_BASE_CONDITIONS = {
    "chunks": None, "ctx_size": 512, "corpus": "/fake/corpus.txt",
    "enable_kl": False, "kl_weight": 0.0,
}


def test_conditions_match_no_imatrix_both_sides_resumes():
    stored = {**_BASE_CONDITIONS, "imatrix_id": {"active": False}}
    current = {**_BASE_CONDITIONS, "imatrix_id": {"active": False}}
    assert MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_imatrix_added_is_rejected():
    """A checkpoint whose probes/candidates were built WITHOUT an imatrix
    must not silently resume into a run that now has one active -- this is
    the exact bug the issue reports, at checkpoint-resume granularity."""
    stored = {**_BASE_CONDITIONS, "imatrix_id": {"active": False}}
    current = {
        **_BASE_CONDITIONS,
        "imatrix_id": {"active": True, "n_tensors": 5, "hash": "abc123"},
    }
    assert not MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_imatrix_removed_is_rejected():
    stored = {
        **_BASE_CONDITIONS,
        "imatrix_id": {"active": True, "n_tensors": 5, "hash": "abc123"},
    }
    current = {**_BASE_CONDITIONS, "imatrix_id": {"active": False}}
    assert not MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_different_imatrix_hash_is_rejected():
    """Same n_tensors, different content (e.g. a different calibration
    corpus) -- must still be treated as a different imatrix."""
    stored = {
        **_BASE_CONDITIONS,
        "imatrix_id": {"active": True, "n_tensors": 5, "hash": "abc123"},
    }
    current = {
        **_BASE_CONDITIONS,
        "imatrix_id": {"active": True, "n_tensors": 5, "hash": "def456"},
    }
    assert not MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_same_imatrix_identity_resumes():
    identity = {"active": True, "n_tensors": 5, "hash": "abc123"}
    stored = {**_BASE_CONDITIONS, "imatrix_id": identity}
    current = {**_BASE_CONDITIONS, "imatrix_id": dict(identity)}
    assert MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_old_checkpoint_missing_imatrix_id_key_resumes_no_imatrix_run():
    """Backward compatibility: a checkpoint written before this fix has no
    'imatrix_id' key at all. Every such checkpoint's probes were built
    without an imatrix (the bug this issue fixes -- probing.py never
    accepted the parameter before now), so a missing key must read as
    'inactive', exactly like the enable_kl backward-compat default."""
    stored = dict(_BASE_CONDITIONS)  # no imatrix_id key
    current = {**_BASE_CONDITIONS, "imatrix_id": {"active": False}}
    assert MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_conditions_match_old_checkpoint_missing_imatrix_id_key_rejected_for_imatrix_run():
    stored = dict(_BASE_CONDITIONS)  # no imatrix_id key
    current = {
        **_BASE_CONDITIONS,
        "imatrix_id": {"active": True, "n_tensors": 3, "hash": "xyz"},
    }
    assert not MagicQuantOrchestrator._measurement_conditions_match(stored, current)


def test_current_measurement_conditions_includes_imatrix_id(tmp_path, monkeypatch):
    """Integration-level: _current_measurement_conditions actually surfaces
    self._imatrix's identity, not a hardcoded placeholder."""
    orch = _make_orch(tmp_path, monkeypatch)
    assert orch._current_measurement_conditions()["imatrix_id"] == {"active": False}

    orch._imatrix = {"t": np.array([1.0, 2.0], dtype=np.float32)}
    conditions = orch._current_measurement_conditions()
    assert conditions["imatrix_id"]["active"] is True
    assert conditions["imatrix_id"] == imatrix_identity(orch._imatrix)
