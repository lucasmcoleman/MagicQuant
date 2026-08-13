"""Structurally-unquantizable groups must be SKIPPED, not raised on.

Ground truth: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (arch
nemotron_h_moe) dies in sensitivity probing on group R (Router) --
``blk.*.ffn_gate_inp.weight`` (expert gating, on llama.cpp's never-quantize
list) and ``blk.*.exp_probs_b.bias`` (1-D). 100% of the group is forced to
F32 by writer policy regardless of scheme, so a probe of it is numerically
identical to the baseline on every scheme, on every run -- exactly the
signal ``_verify_probe_artifact`` exists to catch as a bug, except here it
isn't one.

``_detect_fixed_groups`` (magicquant/evolution/probing.py) tells the two
apart by checking whether EVERY tensor in the group is unquantizable for a
KNOWN, policy-based reason (``_is_never_quantized``, 1-D, non-32-divisible
row, or an F32-required SSM operand -- see ``_tensor_fixed_reason``). The
critical property under test is that this is a conjunction, not just
"quantized == 0": a group with even one legitimate candidate that came back
untouched must still raise ``ProbeMeasurementError`` exactly as before --
that IS the original bug this module's guard exists to catch.
"""
import logging

import pytest

from magicquant.evolution.probing import ProbeMeasurementError, SensitivityProber
from magicquant.utils.measurement import PROBE_FIXED


class _FakeCalc:
    def calculate_perplexity(self, *a, **k):
        return 5.0


def _prober(tmp_path, **kwargs):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return SensitivityProber(
        base_model_path=str(model),
        baseline_perplexity=5.0,
        perplexity_calculator=_FakeCalc(),
        output_dir=str(tmp_path / "_probes"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# A group that is 100% never-quantize-by-name / 1-D is skipped, not raised.
# ---------------------------------------------------------------------------

class _RouterReader:
    """Mirrors group R on the real Nemotron model: one ffn_gate_inp.weight
    (2-D, name-matched) and one exp_probs_b.bias (1-D) per layer."""

    _INFO = {
        "blk.0.ffn_gate_inp.weight": {"n_dims": 2, "shape": [128, 2688]},
        "blk.0.exp_probs_b.bias": {"n_dims": 1, "shape": [128]},
        "blk.1.ffn_gate_inp.weight": {"n_dims": 2, "shape": [128, 2688]},
        "blk.1.exp_probs_b.bias": {"n_dims": 1, "shape": [128]},
    }

    def open(self):
        pass

    def close(self):
        pass

    def get_tensor_names(self):
        return list(self._INFO)

    def get_tensor_info(self, name):
        info = self._INFO.get(name)
        return dict(info, name=name) if info else None


def test_structurally_fixed_group_is_skipped_not_raised(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        "magicquant.gguf.reader.GGUFReader", lambda *a, **k: _RouterReader()
    )
    prober = _prober(tmp_path)

    with caplog.at_level(logging.INFO, logger="magicquant.evolution.probing"):
        results = prober.probe_all_groups(groups=["R"], verbose=False)

    assert results["R"] == 0.0
    assert prober.sensitivity_results["R"] == 0.0
    assert prober.resolutions["R"] == PROBE_FIXED

    assert "R" in prober.fixed_groups
    assert prober.fixed_groups["R"]["tensor_count"] == 4
    reason = prober.fixed_groups["R"]["reason"]
    assert "ffn_gate_inp.weight" in reason
    assert "1-D" in reason

    entry = next(p for p in prober.probe_models if p["group"] == "R")
    assert entry["fixed"] is True
    assert entry["measured"] is False
    assert entry["clamped"] is False
    assert entry["fixed_reason"] == reason

    assert any(
        "never-quantizable" in r.message and "skipping probe" in r.message
        for r in caplog.records
    )


def test_fixed_group_does_not_affect_provenance_of_other_groups(tmp_path, monkeypatch):
    """A fixed group must not drag a fully-measured run's provenance down
    from 'measured' to 'partial', and must not count against
    resolved_mass_fraction -- it never needed resolving in the first place.
    """
    monkeypatch.setattr(
        "magicquant.gguf.reader.GGUFReader", lambda *a, **k: _RouterReader()
    )

    # Only "R" is ever classified by _RouterReader's tensor names, so "Q"
    # has no tensors and _real_probe would ordinarily choke on "no tensors
    # classified into that group". Stub the build/verify boundary so the
    # non-fixed group measures cleanly instead -- this test is about
    # provenance/coverage bookkeeping, not the real writer.
    monkeypatch.setattr("magicquant.gguf.writer.create_hybrid_gguf", lambda **k: None)
    monkeypatch.setattr(
        "magicquant.evolution.probing.SensitivityProber._verify_probe_artifact",
        lambda *a, **k: None,
    )

    class _AboveBaselineCalc:
        def calculate_perplexity(self, *a, **k):
            return 6.0  # baseline 5.0 -- clearly resolved, not clamped

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    prober = SensitivityProber(
        base_model_path=str(model),
        baseline_perplexity=5.0,
        perplexity_calculator=_AboveBaselineCalc(),
        output_dir=str(tmp_path / "_probes"),
        parameter_counts={"R": 30_000_000, "Q": 500_000},
    )

    prober.probe_all_groups(groups=["R", "Q"], verbose=False)

    assert prober.probing_provenance == "measured"
    assert prober.resolved_mass_fraction == pytest.approx(1.0)

    # get_normalized_weights already treats "sensitivity 0.0" uniformly
    # regardless of WHY it's zero -- no change needed there (see CLAUDE
    # session notes); a fixed group's 0.0 washes out exactly like any
    # other zero-sensitivity group once something else has real signal.
    weights = prober.get_normalized_weights()
    assert weights["R"] == 0.0
    assert weights["Q"] == pytest.approx(1.0)
    assert prober.weights_degenerate is False


# ---------------------------------------------------------------------------
# The anti-bug guard: a genuine candidate that came back untouched still
# raises. This is the important test -- it proves the new skip path cannot
# be mistaken for the original silent-degradation bug.
# ---------------------------------------------------------------------------

class _OrdinaryReader:
    """A single, perfectly ordinary 2-D, 32-/256-divisible weight -- not
    never-quantize-by-name, not 1-D, not an SSM operand. A legitimate
    quantization candidate that _detect_fixed_groups must NOT mark fixed."""

    def open(self):
        pass

    def close(self):
        pass

    def get_tensor_names(self):
        return ["blk.0.ffn_down.weight"]

    def get_tensor_info(self, name):
        return {"name": name, "n_dims": 2, "shape": [256, 256]}


class _UpstreamTensor:
    def __init__(self, name, type_name):
        self.name = name
        self.tensor_type = type("TT", (), {"name": type_name})()


class _UpstreamUntouchedReader:
    """Stand-in for gguf.GGUFReader reading the built probe artifact: the
    writer silently left the group's only tensor at the keep scheme."""

    def __init__(self, path):
        self.tensors = [_UpstreamTensor("blk.0.ffn_down.weight", "Q8_0")]


def test_genuine_untouched_group_still_raises(tmp_path, monkeypatch):
    import gguf

    monkeypatch.setattr(
        "magicquant.gguf.reader.GGUFReader", lambda *a, **k: _OrdinaryReader()
    )
    monkeypatch.setattr("magicquant.gguf.writer.create_hybrid_gguf", lambda **k: None)
    monkeypatch.setattr(gguf, "GGUFReader", _UpstreamUntouchedReader)

    prober = _prober(tmp_path)

    # Not fixed: the pre-check must fall through to the real probe path.
    assert prober._detect_fixed_groups(["D"]) == {}

    with pytest.raises(ProbeMeasurementError, match="still at full precision"):
        prober.probe_all_groups(groups=["D"], aggressive_scheme="Q4_K_M", verbose=False)


# ---------------------------------------------------------------------------
# Downstream: a fixed group is excluded from the evolutionary search's
# mutable-group set, so no candidate ever gets a scheme choice for it.
# ---------------------------------------------------------------------------

def test_fixed_group_excluded_from_mutable_search_groups(tmp_path):
    from magicquant.orchestrator import MagicQuantOrchestrator

    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    orch._search_groups = ["E", "H", "Q", "K", "O", "U", "D", "X", "R"]
    orch.fixed_groups = {"R": {"reason": "ffn_gate_inp.weight / 1-D", "tensor_count": 48}}

    orch._exclude_fixed_groups(verbose=False)

    assert "R" not in orch._search_groups
    assert set(orch._search_groups) == {"E", "H", "Q", "K", "O", "U", "D", "X"}


def test_no_fixed_groups_leaves_search_groups_untouched(tmp_path):
    from magicquant.orchestrator import MagicQuantOrchestrator

    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    original = ["E", "H", "Q", "K", "O", "U", "D"]
    orch._search_groups = list(original)
    orch.fixed_groups = {}

    orch._exclude_fixed_groups(verbose=False)

    assert orch._search_groups == original
