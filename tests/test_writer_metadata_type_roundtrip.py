"""Regression: copied GGUF metadata arrays/scalars must round-trip at their
EXACT source element type -- signed integer types must not come back as
their unsigned cousins (and vice versa), regardless of the values' magnitude.

Empirically confirmed (2026-08-20): every standard MagicQuant K-quant hybrid
render wrote ``tokenizer.ggml.token_type`` -- INT32 (5) on every real GGUF
source -- out as UINT32 (4). Mainline llama.cpp's vocab loader hard-requires
GGUF_TYPE_INT32 for this key and refuses to load the file ("invalid gguf
type for tokenizer.ggml.token_type"; real user report, HF discussion
lmcoleman/Qwen3.8-27B-MagicQuant-GGUF#1, current llama.cpp Windows CUDA
release b-series). uint32 was observed in Qwen3.8/Nemotron/ThinkingCap/
Muse-Glimmer renders; int32 in their BF16 sources AND in ROCmFPX renders
(a different writer, outside this repo -- Foundry's ROCmFPX MQ-hybrid mode
never touches magicquant.gguf.writer._write_metadata_value at all, so it
was never exposed to this bug).

Root cause: magicquant.gguf.reader.GGUFReader._read_value discarded the
on-disk GGUF element type entirely, returning plain Python ints/lists.
magicquant.gguf.writer._write_metadata_value then re-derived an output type
from the Python value's magnitude alone, preferring UINT32 for any
non-negative int array -- which cannot distinguish a signed array whose
values happen to be small and non-negative (token_type: 0-6) from a
genuinely unsigned one.

Fix: GGUFReader now tags every integer scalar/array KV with the exact
on-disk GGUF type it was parsed as (GGUFTypedInt/GGUFTypedArray, both
transparent subclasses of int/list). _write_metadata_value uses that
recorded type verbatim when present, and only falls back to the historical
magnitude-based inference for genuinely untyped values (e.g. freshly
constructed Python lists that were never read from a GGUF file).
"""
import io
import struct

import numpy as np
import pytest

from magicquant.gguf.reader import GGUFReader, GGUFTypedArray, GGUFTypedInt
from magicquant.gguf.writer import (
    create_hybrid_gguf,
    _write_metadata_value,
    _GGUF_TYPE_UINT8,
    _GGUF_TYPE_INT8,
    _GGUF_TYPE_UINT16,
    _GGUF_TYPE_INT16,
    _GGUF_TYPE_UINT32,
    _GGUF_TYPE_INT32,
    _GGUF_TYPE_UINT64,
    _GGUF_TYPE_INT64,
    _GGUF_TYPE_ARRAY,
)

gguf = pytest.importorskip("gguf")


# ---------------------------------------------------------------------------
# Low-level unit tests: _write_metadata_value against reader-tagged values
# directly, no file I/O. These pin the exact byte layout emitted.
# ---------------------------------------------------------------------------

def _read_array_header(data: bytes):
    """Unpack (outer_tag, elem_tag, count) from a serialized array value."""
    outer_tag = struct.unpack("<I", data[0:4])[0]
    elem_tag = struct.unpack("<I", data[4:8])[0]
    count = struct.unpack("<Q", data[8:16])[0]
    return outer_tag, elem_tag, count


@pytest.mark.parametrize(
    "gguf_type,fmt,values",
    [
        (_GGUF_TYPE_UINT8, "B", [0, 1, 255]),
        (_GGUF_TYPE_INT8, "b", [-5, 0, 127]),
        (_GGUF_TYPE_UINT16, "H", [0, 1, 65535]),
        (_GGUF_TYPE_INT16, "h", [-100, 0, 300]),
        (_GGUF_TYPE_UINT32, "I", [0, 1, 4294967295]),
        # The empirically-confirmed shape of the bug: a signed INT32 array
        # whose values are all small and non-negative -- exactly
        # tokenizer.ggml.token_type's shape (llama.cpp token-type enum
        # values 0-6). Magnitude-based inference always preferred UINT32
        # for arrays like this.
        (_GGUF_TYPE_INT32, "i", [1, 2, 3, 4, 5, 6, 1]),
        (_GGUF_TYPE_UINT64, "Q", [0, 1, 2**33]),
        (_GGUF_TYPE_INT64, "q", [-1, 0, 2**33]),
    ],
)
def test_typed_array_preserves_exact_element_type_and_values(gguf_type, fmt, values):
    buf = io.BytesIO()
    _write_metadata_value(buf, GGUFTypedArray(values, gguf_type))
    data = buf.getvalue()

    outer_tag, elem_tag, count = _read_array_header(data)
    assert outer_tag == _GGUF_TYPE_ARRAY
    assert elem_tag == gguf_type, (
        f"expected element type {gguf_type}, wrote {elem_tag} -- "
        f"signedness/width was lost"
    )
    assert count == len(values)

    itemsize = struct.calcsize("<" + fmt)
    payload = data[16:]
    decoded = [
        struct.unpack_from("<" + fmt, payload, i * itemsize)[0]
        for i in range(count)
    ]
    assert decoded == values


def test_token_type_array_specifically_stays_int32():
    """The exact reported key: tokenizer.ggml.token_type, INT32 source,
    values in llama.cpp's token-type enum range (0-6)."""
    buf = io.BytesIO()
    _write_metadata_value(
        buf, GGUFTypedArray([1, 2, 3, 4, 5, 6, 1], _GGUF_TYPE_INT32),
    )
    data = buf.getvalue()
    outer_tag, elem_tag, count = _read_array_header(data)
    assert outer_tag == _GGUF_TYPE_ARRAY
    assert elem_tag == _GGUF_TYPE_INT32, (
        "token_type must round-trip as INT32 (5), not UINT32 (4) -- "
        "llama.cpp's vocab loader hard-requires INT32 and refuses the file "
        "otherwise (HF discussion lmcoleman/Qwen3.8-27B-MagicQuant-GGUF#1)"
    )


def test_untyped_list_keeps_historical_magnitude_inference():
    """A plain list with no recorded source type (never read from a GGUF
    file -- e.g. constructed directly by MagicQuant) must keep behaving
    exactly as before: narrowest tag that fits, UINT32 preferred for
    non-negative values. This is the pre-existing, still-correct behavior
    for values that never had a source type to preserve."""
    buf = io.BytesIO()
    _write_metadata_value(buf, [1, 2, 3])
    outer_tag, elem_tag, _ = _read_array_header(buf.getvalue())
    assert outer_tag == _GGUF_TYPE_ARRAY
    assert elem_tag == _GGUF_TYPE_UINT32


def test_typed_scalar_int_preserves_exact_type():
    buf = io.BytesIO()
    _write_metadata_value(buf, GGUFTypedInt(3, _GGUF_TYPE_INT8))
    data = buf.getvalue()
    tag = struct.unpack("<I", data[:4])[0]
    assert tag == _GGUF_TYPE_INT8
    assert struct.unpack("<b", data[4:5])[0] == 3


def test_empty_typed_array_preserves_element_type():
    buf = io.BytesIO()
    _write_metadata_value(buf, GGUFTypedArray([], _GGUF_TYPE_INT16))
    data = buf.getvalue()
    outer_tag, elem_tag, count = _read_array_header(data)
    assert outer_tag == _GGUF_TYPE_ARRAY
    assert elem_tag == _GGUF_TYPE_INT16
    assert count == 0


# ---------------------------------------------------------------------------
# End-to-end: real GGUF fixture -> GGUFReader -> create_hybrid_gguf ->
# independent verification with the UPSTREAM gguf package's own reader (not
# magicquant's), so the check doesn't share code with the thing under test.
# ---------------------------------------------------------------------------

# key -> (GGUFValueType, values). Covers every signed integer width plus
# token_type specifically, and an unsigned array (to confirm the fix doesn't
# just flip the bug the other way -- UINT64 must not get narrowed to
# UINT32 either, which the old magnitude-only logic also did).
_TYPED_ARRAYS = {
    "tokenizer.ggml.token_type": (gguf.GGUFValueType.INT32, [1, 2, 3, 4, 5, 6, 1]),
    "test.int8_arr": (gguf.GGUFValueType.INT8, [-5, 0, 127]),
    "test.int16_arr": (gguf.GGUFValueType.INT16, [-100, 0, 300]),
    "test.int32_arr": (gguf.GGUFValueType.INT32, [-1, 0, 70000]),
    "test.int64_arr": (gguf.GGUFValueType.INT64, [-1, 0, 5_000_000_000]),
    "test.uint64_arr": (gguf.GGUFValueType.UINT64, [0, 1, 5_000_000_000]),
}


@pytest.fixture
def typed_metadata_gguf(tmp_path):
    p = tmp_path / "typed_meta_source.gguf"
    w = gguf.GGUFWriter(str(p), arch="llama")
    for key, (vtype, values) in _TYPED_ARRAYS.items():
        w.add_key_value(key, values, gguf.GGUFValueType.ARRAY, vtype)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(p)


def test_reader_tags_every_array_with_its_source_element_type(typed_metadata_gguf):
    r = GGUFReader(typed_metadata_gguf)
    r.open()
    meta = r.get_metadata()
    for key, (vtype, values) in _TYPED_ARRAYS.items():
        v = meta[key]
        assert isinstance(v, GGUFTypedArray)
        assert v.gguf_type == int(vtype)
        assert list(v) == values


def test_create_hybrid_gguf_roundtrips_metadata_array_types(tmp_path, typed_metadata_gguf):
    out = str(tmp_path / "out.gguf")
    create_hybrid_gguf(
        output_path=out,
        base_model_path=typed_metadata_gguf,
        quant_config={"base": "F32", "groups": {}},
        verbose=False,
    )

    # Independent verification via the upstream `gguf` package's own
    # reader -- never touches magicquant.gguf.reader, so this can't pass
    # by coincidentally sharing the bug (or the fix) with the code under
    # test.
    ur = gguf.GGUFReader(out)
    for key, (vtype, values) in _TYPED_ARRAYS.items():
        field = ur.get_field(key)
        assert field is not None, f"{key} missing from rendered output"
        assert field.types == [gguf.GGUFValueType.ARRAY, vtype], (
            f"{key}: expected element type {vtype!r}, got {field.types!r} "
            f"-- source element type was not preserved"
        )
        # field.data holds indices into field.parts; resolve to concrete ints.
        decoded = [int(field.parts[idx][0]) for idx in field.data]
        assert decoded == values, f"{key}: values changed by the copy ({decoded} != {values})"
