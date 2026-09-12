"""CPU-only regressions for distortion accounting and calibration validity."""

from types import SimpleNamespace

import numpy as np
import pytest

from magicquant.v2.calibrate import fit_kappa
from magicquant.v2.outcome import MeasurementOutcome as MO
from magicquant.v2.search import V2Config, _calibrate_kappa, _resolve_imatrix_and_schemes
from magicquant.v2.sensitivity import compute_distortion_table
from tests.test_writer import StubSource


def _float_entry(monkeypatch, tmp_path, weights, *, sample_rows=None, imatrix=None, metadata=None):
    name = "blk.0.ffn_down_exps.weight" if weights.ndim == 3 else "blk.0.ffn_down.weight"
    source = StubSource([(name, weights, weights.shape)], metadata=metadata)
    monkeypatch.setattr("magicquant.gguf.source.open_model_source", lambda _: source)
    path = tmp_path / "source.gguf"
    path.write_bytes(b"synthetic source")
    table = compute_distortion_table(
        str(path), ["BF16"], sample_rows=sample_rows,
        imatrix={name: imatrix} if imatrix is not None else None,
    )
    return table["tensors"][name]


def test_sampled_float_error_estimates_whole_tensor(monkeypatch, tmp_path):
    # Identical rows make the expected scaling exact, with no sampling noise.
    weights = np.tile(np.linspace(0.1, 1.1, 32, dtype=np.float32), (16, 1))
    full = _float_entry(monkeypatch, tmp_path, weights)
    sampled = _float_entry(monkeypatch, tmp_path, weights, sample_rows=2)
    assert sampled["choices"]["BF16"]["werr"] == pytest.approx(
        full["choices"]["BF16"]["werr"]
    )


@pytest.mark.parametrize("sample_rows", [None, 1])
def test_expert_imatrix_weights_float_error_and_norm(monkeypatch, tmp_path, sample_rows):
    weights = np.full((2, 4, 32), 1.0003, dtype=np.float32)
    imatrix = np.concatenate([np.ones(32), np.full(32, 100.0)]).astype(np.float32)
    entry = _float_entry(
        monkeypatch, tmp_path, weights, imatrix=imatrix, sample_rows=sample_rows,
    )
    weighting = imatrix.reshape(2, 1, 32)
    rounded = weights.astype(np.float16).astype(np.float32)
    expected_error = np.sum(np.square(weights - rounded) * weighting, dtype=np.float64)
    expected_norm = np.sum(np.square(weights) * weighting, dtype=np.float64)
    assert entry["choices"]["BF16"]["werr"] == pytest.approx(expected_error)
    assert entry["wnorm"] == pytest.approx(expected_norm)


@pytest.mark.parametrize("sample_rows", [0, -1])
def test_nonpositive_sample_count_rejected(monkeypatch, tmp_path, sample_rows):
    with pytest.raises(ValueError, match="sample_rows.*positive"):
        _float_entry(
            monkeypatch, tmp_path, np.ones((4, 32), dtype=np.float32),
            sample_rows=sample_rows,
        )


def test_failed_cumulative_reference_does_not_invent_single_group_measurements():
    outcomes = {
        "__slice_baseline__": MO.success(10.0),
        "__base_aggressive__": MO.failure("reference failed"),
        "D": MO.success(12.0),
        "E": MO.success(11.0),
        "N": MO.success(10.0),
    }
    kappa, provenance = fit_kappa(outcomes, {"D": 2.0, "E": 4.0, "N": 0.0}, 10.0)
    assert provenance == {
        "D": "imputed-median", "E": "imputed-median", "N": "no-allocatable-mass",
    }
    assert kappa == {"D": 1.0, "E": 1.0, "N": 0.0}


def test_failed_cumulative_reference_is_recorded_in_search_failures(monkeypatch, tmp_path):
    outcomes = {
        "__slice_baseline__": MO.success(10.0),
        "__base_aggressive__": MO.failure("reference failed"),
        "D": MO.success(12.0),
    }
    monkeypatch.setattr("magicquant.v2.search.run_group_probes", lambda *a, **k: outcomes)
    cfg = V2Config("source.gguf", str(tmp_path), 1.0, allow_partial_probes=True)
    table = {"tensors": {"weight": {
        "group": "D", "choices": {"Q4_K_M": {"werr": 2.0}},
    }}}
    _, provenance, _, _, failures = _calibrate_kappa(None, cfg, table, None, 10.0, tmp_path)
    assert provenance["D"] == "imputed-median"
    assert failures[0]["group"] == "__base_aggressive__"
    assert failures[0]["error"] == "reference failed"


@pytest.mark.parametrize("mode", ["explicit", "default", "symlink", "hardlink"])
def test_imatrix_corpus_overlap_rejected_before_capture(monkeypatch, tmp_path, mode):
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
    calls = []
    monkeypatch.setattr("magicquant.imatrix.ensure_imatrix", lambda *a, **k: calls.append(k))
    cfg = V2Config(
        "source.gguf", str(tmp_path), 1.0, schemes=["BF16"],
        imatrix_corpus=None if mode == "default" else str(calibration),
    )
    tools = SimpleNamespace(_resolve_data_file=lambda _: str(corpus))
    with pytest.raises(ValueError, match="same file.*perplexity evaluation"):
        _resolve_imatrix_and_schemes(tools, cfg)
    assert calls == []


def test_distinct_corpora_allow_capture(monkeypatch, tmp_path):
    evaluation = tmp_path / "eval.txt"
    evaluation.write_text("evaluation only")
    calibration = tmp_path / "calib.txt"
    calibration.write_text("calibration only")
    captured = {}

    def ensure(path, corpus_path=None, **kwargs):
        captured["corpus"] = corpus_path
        return {"weight": np.ones(32, dtype=np.float32)}

    monkeypatch.setattr("magicquant.imatrix.ensure_imatrix", ensure)
    monkeypatch.setattr("magicquant.imatrix.resolve_imatrix_bin", lambda _: None)
    cfg = V2Config(
        "source.gguf", str(tmp_path), 1.0, schemes=["BF16"],
        imatrix_corpus=str(calibration),
    )
    tools = SimpleNamespace(_resolve_data_file=lambda _: str(evaluation))
    imatrix, schemes = _resolve_imatrix_and_schemes(tools, cfg)
    assert imatrix is not None
    assert schemes == ["BF16"]
    assert captured["corpus"] == str(calibration)
