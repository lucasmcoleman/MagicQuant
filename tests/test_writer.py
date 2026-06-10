"""Writer correctness tests: crash-safety (M3), UNKNOWN handling (M8),
metadata serialization (L3/L4/L10), BF16 downgrade warning (L5), and an
end-to-end read->write->reopen smoke test.
"""
import io
import struct

import numpy as np
import pytest

from magicquant.gguf.source import ModelSource
from magicquant.gguf.writer import (
    GGUFWriter,
    create_hybrid_gguf,
    _write_metadata_value,
    _GGUF_TYPE_UINT32,
    _GGUF_TYPE_INT64,
    _GGUF_TYPE_FLOAT32,
    _GGUF_TYPE_ARRAY,
)


# ---------------------------------------------------------------------------
# Stub source
# ---------------------------------------------------------------------------

class StubSource(ModelSource):
    """In-memory F32 source with a few tensors. Optionally raises / returns a
    bad dtype on a chosen tensor to exercise crash-safety, and can report an
    UNKNOWN source type for a chosen tensor.
    """

    def __init__(self, tensors, *, raise_on=None, bad_dtype_on=None,
                 unknown_on=None, metadata=None):
        # tensors: list of (name, np.ndarray-f32, shape)
        self._tensors = tensors
        self._raise_on = raise_on
        self._bad_dtype_on = bad_dtype_on
        self._unknown_on = unknown_on
        self._metadata = metadata or {"general.architecture": "llama"}

    def get_metadata(self):
        return dict(self._metadata)

    def get_tensor_names(self):
        return [n for (n, _a, _s) in self._tensors]

    def get_all_tensors_info(self):
        infos = []
        for (name, arr, shape) in self._tensors:
            infos.append({
                "name": name,
                "n_dims": len(shape),
                "shape": list(shape),
                "data_type": 0,  # F32
            })
        return infos

    def read_tensor_f32(self, tensor_name):
        if tensor_name == self._raise_on:
            raise RuntimeError(f"injected read failure on {tensor_name}")
        for (name, arr, shape) in self._tensors:
            if name == tensor_name:
                if tensor_name == self._bad_dtype_on:
                    return arr.astype(np.int32)  # wrong dtype
                return arr.astype(np.float32)
        return None

    def get_source_type_name(self, tensor_name):
        if tensor_name == self._unknown_on:
            return "UNKNOWN(99)"
        return "F32"


def _f32_tensor(name, shape):
    n = 1
    for d in shape:
        n *= d
    return (name, np.random.randn(n).astype(np.float32), shape)


# ---------------------------------------------------------------------------
# M3: crash-safe writer
# ---------------------------------------------------------------------------

def test_writer_no_partial_left_on_worker_exception(tmp_path):
    src = StubSource([
        _f32_tensor("blk.0.attn_q.weight", (32, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
    ], raise_on="blk.0.ffn_down.weight")

    out = str(tmp_path / "out.gguf")
    import magicquant.gguf.source as source_mod
    orig = source_mod.open_model_source
    source_mod.open_model_source = lambda *a, **k: src
    try:
        with pytest.raises(Exception):
            create_hybrid_gguf(
                output_path=out,
                base_model_path="ignored",
                quant_config={"base": "Q8_0", "groups": {}},
                verbose=False,
            )
    finally:
        source_mod.open_model_source = orig

    assert not (tmp_path / "out.gguf").exists(), "final file must not exist"
    assert not (tmp_path / "out.gguf.partial").exists(), "no .partial left behind"


def test_writer_bad_dtype_raises_and_no_file(tmp_path):
    src = StubSource([
        _f32_tensor("blk.0.attn_q.weight", (32, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
    ], bad_dtype_on="blk.0.ffn_down.weight")

    out = str(tmp_path / "out.gguf")
    import magicquant.gguf.source as source_mod
    orig = source_mod.open_model_source
    source_mod.open_model_source = lambda *a, **k: src
    try:
        with pytest.raises(ValueError, match="floating-point"):
            create_hybrid_gguf(
                output_path=out,
                base_model_path="ignored",
                quant_config={"base": "Q8_0", "groups": {}},
                verbose=False,
            )
    finally:
        source_mod.open_model_source = orig

    assert not (tmp_path / "out.gguf").exists()
    assert not (tmp_path / "out.gguf.partial").exists()


# ---------------------------------------------------------------------------
# M8: UNKNOWN source type is a hard error
# ---------------------------------------------------------------------------

def test_unknown_source_type_raises(tmp_path):
    src = StubSource([
        _f32_tensor("blk.0.attn_q.weight", (32, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
    ], unknown_on="blk.0.ffn_down.weight")

    out = str(tmp_path / "out.gguf")
    import magicquant.gguf.source as source_mod
    orig = source_mod.open_model_source
    source_mod.open_model_source = lambda *a, **k: src
    try:
        with pytest.raises(ValueError) as exc:
            create_hybrid_gguf(
                output_path=out,
                base_model_path="ignored",
                quant_config={"base": "Q8_0", "groups": {}},
                verbose=False,
            )
        assert "blk.0.ffn_down.weight" in str(exc.value)
        assert "UNKNOWN" in str(exc.value)
    finally:
        source_mod.open_model_source = orig

    assert not (tmp_path / "out.gguf").exists()
    assert not (tmp_path / "out.gguf.partial").exists()


# ---------------------------------------------------------------------------
# L3 / L4: metadata serialization
# ---------------------------------------------------------------------------

def test_np_int64_writes_uint32_tag():
    buf = io.BytesIO()
    _write_metadata_value(buf, np.int64(32))
    tag = struct.unpack("<I", buf.getvalue()[:4])[0]
    assert tag == _GGUF_TYPE_UINT32


def test_np_float32_writes_float32_tag():
    buf = io.BytesIO()
    _write_metadata_value(buf, np.float32(1.5))
    tag = struct.unpack("<I", buf.getvalue()[:4])[0]
    assert tag == _GGUF_TYPE_FLOAT32


def test_negative_int_writes_int64():
    buf = io.BytesIO()
    _write_metadata_value(buf, -5)
    tag = struct.unpack("<I", buf.getvalue()[:4])[0]
    assert tag == _GGUF_TYPE_INT64


def test_int_array_large_value_no_struct_error():
    """An array containing 2**31 must not raise struct.error (L4)."""
    buf = io.BytesIO()
    _write_metadata_value(buf, [1, 2, 2 ** 31])  # exceeds int32 max
    data = buf.getvalue()
    outer_tag = struct.unpack("<I", data[:4])[0]
    assert outer_tag == _GGUF_TYPE_ARRAY
    elem_tag = struct.unpack("<I", data[4:8])[0]
    # 2**31 fits in uint32, so UINT32 tag is expected
    assert elem_tag == _GGUF_TYPE_UINT32


# ---------------------------------------------------------------------------
# L10: file_type enum + L5 BF16 warning + end-to-end smoke
# ---------------------------------------------------------------------------

def _build_and_reopen(tmp_path, group_schemes, base="Q8_0"):
    src = StubSource([
        _f32_tensor("token_embd.weight", (256, 256)),
        _f32_tensor("blk.0.attn_q.weight", (256, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
        _f32_tensor("output.weight", (256, 256)),
    ])
    out = str(tmp_path / "out.gguf")
    import magicquant.gguf.source as source_mod
    orig = source_mod.open_model_source
    source_mod.open_model_source = lambda *a, **k: src
    try:
        create_hybrid_gguf(
            output_path=out,
            base_model_path="ignored",
            quant_config={"base": base, "groups": group_schemes},
            verbose=False,
        )
    finally:
        source_mod.open_model_source = orig

    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(out)
    reader.open()
    return reader


def test_file_type_q6k_is_18(tmp_path):
    reader = _build_and_reopen(tmp_path, {g: "Q6_K" for g in ["E", "H", "Q", "D"]}, base="Q6_K")
    try:
        meta = reader.get_metadata()
        assert meta["general.file_type"] == 18  # LLAMA_FTYPE Q6_K
    finally:
        reader.close()


def test_end_to_end_offsets_aligned_and_monotonic(tmp_path):
    reader = _build_and_reopen(tmp_path, {"Q": "Q8_0", "D": "Q4_K_M"})
    try:
        infos = reader.get_all_tensors_info()
        assert len(infos) == 4
        offsets = [t["offset"] for t in infos]
        assert offsets == sorted(offsets), "offsets must be monotonic"
        for off in offsets:
            assert off % 32 == 0, f"offset {off} not 32-aligned"
    finally:
        reader.close()


def test_bf16_group_written_as_f16_with_warning(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="magicquant.gguf.writer")
    reader = _build_and_reopen(tmp_path, {"Q": "BF16"})
    try:
        info = reader.get_tensor_info("blk.0.attn_q.weight")
        assert info["data_type"] == 1  # F16 id
    finally:
        reader.close()
    assert any("BF16" in rec.message and "F16" in rec.message
               for rec in caplog.records), "expected a one-time BF16->F16 warning"
