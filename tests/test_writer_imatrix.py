"""Writer-level imatrix support: threading + the requires-imatrix gate.

[M4-imatrix-not-threaded] The encoder leaf accepted ``imatrix=`` since PR1 but
the writer never passed it, so all K-quant/IQ encoding ran unweighted, and an
imatrix-REQUIRING type (IQ1/IQ2, once registered) would have produced unusable
output silently. The writer now:

  - accepts ``imatrix={gguf_tensor_name: importance_vector}`` and hands each
    tensor its vector at encode time;
  - hard-errors in Pass 1 (before any bytes are written) if a target type
    requires an imatrix and none was provided for that tensor.
"""
import numpy as np
import pytest

import magicquant.gguf.source as source_mod
import magicquant.gguf.writer as writer_mod
from magicquant.gguf.writer import create_hybrid_gguf

from tests.test_writer import StubSource, _f32_tensor


@pytest.fixture
def stub_source(monkeypatch):
    src = StubSource([
        _f32_tensor("blk.0.attn_q.weight", (32, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
    ])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
    return src


def _build(out_path, **kwargs):
    return create_hybrid_gguf(
        output_path=str(out_path),
        base_model_path="ignored",
        quant_config={"base": "Q4_K_M", "groups": {}},
        verbose=False,
        **kwargs,
    )


def test_imatrix_is_threaded_to_encoder(stub_source, tmp_path, monkeypatch):
    seen = {}
    real_encode = writer_mod.encode_to_ggml_bytes

    def spy(weights, ggml_type_name, imatrix=None, n_per_row=None):
        seen[len(seen)] = (ggml_type_name, imatrix, n_per_row)
        return real_encode(weights, ggml_type_name, imatrix=imatrix,
                           n_per_row=n_per_row)

    monkeypatch.setattr(writer_mod, "encode_to_ggml_bytes", spy)

    imat = {
        "blk.0.attn_q.weight": np.ones(256, dtype=np.float32),
        "blk.0.ffn_down.weight": np.full(256, 2.0, dtype=np.float32),
    }
    _build(tmp_path / "out.gguf", imatrix=imat)

    weighted = [(t, npr) for (t, im, npr) in seen.values() if im is not None]
    assert len(weighted) == 2, f"expected 2 weighted encodes, saw {len(weighted)}"
    # The writer must hand the encoder each tensor's true row width.
    assert all(npr == 256 for (_t, npr) in weighted), weighted


def test_imatrix_changes_output_bytes(stub_source, tmp_path):
    _build(tmp_path / "plain.gguf")
    imat = np.full(256, 0.01, dtype=np.float32)
    imat[:32] = 100.0
    _build(tmp_path / "weighted.gguf", imatrix={
        "blk.0.attn_q.weight": imat,
        "blk.0.ffn_down.weight": imat,
    })
    plain = (tmp_path / "plain.gguf").read_bytes()
    weighted = (tmp_path / "weighted.gguf").read_bytes()
    assert plain != weighted


def test_missing_imatrix_for_required_type_raises(stub_source, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(writer_mod, "_requires_imatrix", lambda t: True)
    with pytest.raises(ValueError, match="blk.0.attn_q.weight"):
        _build(tmp_path / "out.gguf")
    assert not (tmp_path / "out.gguf").exists()
    assert not (tmp_path / "out.gguf.partial").exists()


def test_partial_imatrix_for_required_type_raises(stub_source, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(writer_mod, "_requires_imatrix", lambda t: True)
    with pytest.raises(ValueError, match="blk.0.ffn_down.weight"):
        _build(tmp_path / "out.gguf", imatrix={
            "blk.0.attn_q.weight": np.ones(256, dtype=np.float32),
        })


def test_required_type_with_full_imatrix_succeeds(stub_source, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(writer_mod, "_requires_imatrix", lambda t: True)
    out = _build(tmp_path / "out.gguf", imatrix={
        "blk.0.attn_q.weight": np.ones(256, dtype=np.float32),
        "blk.0.ffn_down.weight": np.ones(256, dtype=np.float32),
    })
    assert (tmp_path / "out.gguf").exists(), out


def test_float_targets_skip_the_gate(stub_source, tmp_path, monkeypatch):
    # All-F32 config: gate must not even be consulted (no libggml needed).
    def boom(t):
        raise AssertionError("gate consulted for float passthrough")
    monkeypatch.setattr(writer_mod, "_requires_imatrix", boom)
    create_hybrid_gguf(
        output_path=str(tmp_path / "f32.gguf"),
        base_model_path="ignored",
        quant_config={"base": "F32", "groups": {}},
        verbose=False,
    )
    assert (tmp_path / "f32.gguf").exists()
