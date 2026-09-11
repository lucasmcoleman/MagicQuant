"""Every tensor must be budgeted; calibration must use separate text."""

from types import SimpleNamespace

import pytest

from magicquant.orchestrator import MagicQuantOrchestrator
from magicquant.v2.search import _build_units


def _table(choices, *, fixed=False):
    return {"tensors": {
        "blk.0.ffn_down.weight": {"group": "D", "fixed": fixed, "choices": choices},
        "token_embd.weight": {"group": "E", "choices": {
            "BF16": {"actual": "F16", "bytes": 512, "werr": 0.001},
        }},
    }}


def test_floor_cannot_silently_remove_tensor_from_budget():
    table = _table({"Q4_K_M": {"actual": "Q4_K", "bytes": 144, "werr": 1.0}})
    with pytest.raises(ValueError, match="ffn_down.*no admissible choices"):
        _build_units(table, {}, {"D": "BF16"})


def test_failed_distortion_cannot_silently_remove_tensor_from_budget():
    table = _table({"Q8_0": {"actual": "Q8_0", "bytes": 272, "werr": None}})
    with pytest.raises(ValueError, match="ffn_down.*no admissible choices"):
        _build_units(table, {}, {})


def test_unknown_fixed_tensor_size_is_rejected():
    table = _table({"UNKNOWN(99)": {
        "actual": "UNKNOWN(99)", "bytes": None, "werr": 0.0,
    }}, fixed=True)
    with pytest.raises(ValueError, match="ffn_down.*serialized size is unknown"):
        _build_units(table, {}, {})


def test_valid_alternative_preserves_full_tensor_coverage():
    table = _table({
        "Q8_0": {"actual": "Q8_0", "bytes": 272, "werr": None},
        "BF16": {"actual": "F16", "bytes": 512, "werr": 0.001},
    })
    units = _build_units(table, {}, {"D": "BF16"})
    assert {u.name for u in units} == set(table["tensors"])
    assert sum(u.choices[0].bytes for u in units) == 1024


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("mode", ["default", "explicit", "symlink", "hardlink"])
def test_v1_overlap_skips_capture_and_clears_existing_imatrix(monkeypatch, tmp_path, mode, pinned):
    corpus = tmp_path / "evaluation.txt"
    corpus.write_text("evaluation only")
    calibration = corpus
    if mode in ("symlink", "hardlink"):
        calibration = tmp_path / "calibration.txt"
        if mode == "symlink":
            calibration.symlink_to(corpus)
        else:
            calibration.hardlink_to(corpus)
    monkeypatch.setattr("magicquant.imatrix.DEFAULT_CORPUS_PATH", corpus)
    monkeypatch.setattr("magicquant.imatrix.resolve_imatrix_bin", lambda _: None)
    calls = []
    monkeypatch.setattr("magicquant.imatrix.ensure_imatrix", lambda *a, **k: calls.append(k))
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.source_model_path = "source.gguf"
    orch._imatrix = {"stale": object()}
    orch._llama_tools = SimpleNamespace(
        _pinned_corpus=str(corpus) if pinned else None,
        _resolve_data_file=lambda _: str(corpus),
    )
    assert orch.enable_imatrix(None if mode == "default" else str(calibration)) is False
    assert orch._imatrix is None
    assert calls == []


def test_v1_distinct_default_corpus_still_captures(monkeypatch, tmp_path):
    evaluation = tmp_path / "evaluation.txt"
    evaluation.write_text("evaluation only")
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("calibration only")
    monkeypatch.setattr("magicquant.imatrix.DEFAULT_CORPUS_PATH", calibration)
    monkeypatch.setattr("magicquant.imatrix.resolve_imatrix_bin", lambda _: None)
    result = {"weight": object()}
    monkeypatch.setattr("magicquant.imatrix.ensure_imatrix", lambda *a, **k: result)
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.source_model_path = "source.gguf"
    orch._imatrix = None
    orch._llama_tools = SimpleNamespace(_resolve_data_file=lambda _: str(evaluation))
    assert orch.enable_imatrix() is True
    assert orch._imatrix is result
