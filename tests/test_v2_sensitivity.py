"""Distortion table: per-scheme error ordering, imatrix weighting effect,
fixed-unit handling, cache roundtrip. Skips without libggml."""

import numpy as np
import pytest

import magicquant.gguf.source as source_mod
from magicquant.v2.sensitivity import compute_distortion_table

from tests.test_writer import StubSource


@pytest.fixture(autouse=True)
def _need_libggml():
    try:
        from magicquant.quant.ggml_binding import get_handle
        get_handle()
    except Exception:
        pytest.skip("libggml unavailable")


@pytest.fixture()
def stub_model(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    tensors = [
        ("blk.0.ffn_down.weight", rng.standard_normal(256 * 256).astype(np.float32), (256, 256)),
        ("blk.0.attn_q.weight", rng.standard_normal(128 * 256).astype(np.float32), (128, 256)),
        ("blk.0.attn_norm.weight", rng.standard_normal(256).astype(np.float32), (256,)),
    ]
    src = StubSource(tensors)
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
    model_path = tmp_path / "stub.gguf"
    model_path.write_bytes(b"GGUF-stub")
    return str(model_path)


SCHEMES = ["Q8_0", "Q6_K", "Q4_K_M", "MXFP4_MOE", "BF16"]


def test_error_ordering_and_fixed_units(stub_model):
    table = compute_distortion_table(stub_model, SCHEMES, imatrix=None)
    t = table["tensors"]["blk.0.ffn_down.weight"]
    assert not t["fixed"]
    e8 = t["choices"]["Q8_0"]["werr"]
    e6 = t["choices"]["Q6_K"]["werr"]
    e4 = t["choices"]["Q4_K_M"]["werr"]
    assert 0 < e8 < e6 < e4, (e8, e6, e4)
    # BF16 resolves to F16 on disk with small-but-nonzero error
    bf = t["choices"]["BF16"]
    assert bf["actual"] == "F16"
    assert 0 < bf["werr"] < e8
    # sizes: fewer bits -> fewer bytes
    assert t["choices"]["Q4_K_M"]["bytes"] < t["choices"]["Q6_K"]["bytes"] < t["choices"]["Q8_0"]["bytes"]
    # 1-D norm is a fixed F32 unit
    norm = table["tensors"]["blk.0.attn_norm.weight"]
    assert norm["fixed"]
    assert list(norm["choices"]) == ["F32"]
    assert norm["choices"]["F32"]["werr"] == 0.0


def test_imatrix_weighting_changes_error(stub_model):
    cols = 256
    hot = np.ones(cols, dtype=np.float32)
    hot[:32] = 1000.0  # first 32 input columns matter 1000x more
    imatrix = {"blk.0.ffn_down.weight": hot}
    plain = compute_distortion_table(stub_model, ["Q4_K_M"], imatrix=None)
    weighted = compute_distortion_table(stub_model, ["Q4_K_M"], imatrix=imatrix)
    ep = plain["tensors"]["blk.0.ffn_down.weight"]["choices"]["Q4_K_M"]["werr"]
    ew = weighted["tensors"]["blk.0.ffn_down.weight"]["choices"]["Q4_K_M"]["werr"]
    assert ew != ep
    assert ew > ep  # hot columns dominate the weighted sum
    assert weighted["meta"]["imatrix"]["active"] is True
    # tensor without an imatrix entry is computed unweighted, not dropped
    assert weighted["tensors"]["blk.0.attn_q.weight"]["choices"]["Q4_K_M"]["werr"] > 0


def test_cache_roundtrip(stub_model, tmp_path):
    cache = tmp_path / "cache"
    t1 = compute_distortion_table(
        stub_model, ["Q8_0"], imatrix=None, cache_dir=str(cache)
    )
    files = list(cache.glob("distortion_*.json"))
    assert len(files) == 1
    t2 = compute_distortion_table(
        stub_model, ["Q8_0"], imatrix=None, cache_dir=str(cache)
    )
    assert t2["tensors"]["blk.0.ffn_down.weight"]["choices"]["Q8_0"]["werr"] == (
        t1["tensors"]["blk.0.ffn_down.weight"]["choices"]["Q8_0"]["werr"]
    )


def test_row_sampling_estimator_close(stub_model):
    full = compute_distortion_table(stub_model, ["Q8_0"], imatrix=None)
    sampled = compute_distortion_table(
        stub_model, ["Q8_0"], imatrix=None, sample_rows=64
    )
    ef = full["tensors"]["blk.0.ffn_down.weight"]["choices"]["Q8_0"]["werr"]
    es = sampled["tensors"]["blk.0.ffn_down.weight"]["choices"]["Q8_0"]["werr"]
    assert abs(es - ef) / ef < 0.25  # unbiased strided estimate, iid data
