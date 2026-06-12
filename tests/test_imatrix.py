"""Tests for magicquant.imatrix — capture + load of llama-imatrix importance data.

llama-imatrix (modern llama.cpp) writes a GGUF whose tensors come in pairs:
``<weight_name>.in_sum2`` (sum of squared activations per input column, length =
the weight's row width) and ``<weight_name>.counts`` (activation count, length 1).
The per-column importance vector handed to ``ggml_quantize_chunk`` is
``in_sum2 / counts``.

The synthetic fixtures here are written with the ``gguf`` pip package (already a
dev dependency for the writer round-trip tests); loading uses MagicQuant's own
reader, so the load path under test is exactly the production one.
"""
import os
import shutil

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from magicquant.imatrix import capture_imatrix, load_imatrix


def _write_imatrix_gguf(path, pairs, *, general_type="imatrix"):
    """Write a minimal imatrix-shaped GGUF. pairs: {weight_name: (sum2, count)}."""
    w = gguf.GGUFWriter(str(path), arch="imatrix")
    if general_type is not None:
        w.add_type(general_type)
    for name, (sum2, count) in pairs.items():
        sum2 = np.asarray(sum2, dtype=np.float32)
        w.add_tensor(f"{name}.in_sum2", sum2)
        w.add_tensor(f"{name}.counts",
                     np.asarray([count], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_load_imatrix_divides_sum2_by_counts(tmp_path):
    sum2 = np.array([4.0, 8.0, 12.0, 16.0], dtype=np.float32)
    path = _write_imatrix_gguf(tmp_path / "im.gguf",
                               {"blk.0.attn_q.weight": (sum2, 4.0)})
    result = load_imatrix(path)
    assert set(result) == {"blk.0.attn_q.weight"}
    vec = result["blk.0.attn_q.weight"]
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec, [1.0, 2.0, 3.0, 4.0])


def test_load_imatrix_multiple_weights(tmp_path):
    path = _write_imatrix_gguf(tmp_path / "im.gguf", {
        "blk.0.ffn_down.weight": (np.full(8, 2.0, dtype=np.float32), 2.0),
        "blk.1.attn_k.weight": (np.full(4, 9.0, dtype=np.float32), 3.0),
    })
    result = load_imatrix(path)
    assert set(result) == {"blk.0.ffn_down.weight", "blk.1.attn_k.weight"}
    np.testing.assert_allclose(result["blk.0.ffn_down.weight"], np.full(8, 1.0))
    np.testing.assert_allclose(result["blk.1.attn_k.weight"], np.full(4, 3.0))


def test_load_imatrix_rejects_file_without_imatrix_tensors(tmp_path):
    # A GGUF with ordinary tensors (no .in_sum2 pairs) is not an imatrix file.
    w = gguf.GGUFWriter(str(tmp_path / "model.gguf"), arch="llama")
    w.add_tensor("blk.0.attn_q.weight", np.zeros(16, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="imatrix"):
        load_imatrix(tmp_path / "model.gguf")


def test_load_imatrix_missing_counts_raises(tmp_path):
    w = gguf.GGUFWriter(str(tmp_path / "im.gguf"), arch="imatrix")
    w.add_tensor("blk.0.attn_q.weight.in_sum2",
                 np.ones(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="counts"):
        load_imatrix(tmp_path / "im.gguf")


def test_capture_imatrix_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="llama-imatrix"):
        capture_imatrix("model.gguf", "corpus.txt", tmp_path / "out.gguf")


@pytest.mark.skipif(
    not (shutil.which("llama-imatrix")
         and os.environ.get("MAGICQUANT_TEST_MODEL_GGUF")
         and os.environ.get("MAGICQUANT_TEST_CORPUS")),
    reason="needs llama-imatrix + MAGICQUANT_TEST_MODEL_GGUF + MAGICQUANT_TEST_CORPUS",
)
def test_capture_imatrix_end_to_end(tmp_path):
    out = capture_imatrix(
        os.environ["MAGICQUANT_TEST_MODEL_GGUF"],
        os.environ["MAGICQUANT_TEST_CORPUS"],
        tmp_path / "captured.gguf",
        chunks=1,
    )
    result = load_imatrix(out)
    assert result, "capture produced no importance vectors"
    for name, vec in result.items():
        assert vec.ndim == 1 and vec.size > 0, name
        assert np.isfinite(vec).all(), name
