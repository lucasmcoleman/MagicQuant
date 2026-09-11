"""Malformed headers cannot become cached models; custom alignment round-trips."""

import struct

import gguf
import numpy as np
import pytest

from magicquant.gguf.reader import GGUFReader, GGUFTypedInt
from magicquant.gguf.writer import ALIGNMENT, create_hybrid_gguf


def _model(path, alignment=32):
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_custom_alignment(alignment)
    values = np.arange(32, dtype=np.float32)
    writer.add_tensor("blk.0.attn_norm.weight", values)
    writer.add_tensor("blk.0.ffn_norm.weight", values + 1)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return values


@pytest.mark.parametrize("alignment", [64, 256, 4096])
def test_custom_alignment_reads_and_rewrites_correct_tensor_values(tmp_path, alignment):
    source, output = tmp_path / "source.gguf", tmp_path / "hybrid.gguf"
    values = _model(source, alignment)
    reader = GGUFReader(str(source))
    reader.open()
    reference = gguf.GGUFReader(str(source))
    assert reader.data_offset == reference.data_offset
    create_hybrid_gguf(str(output), str(source), {"base": "BF16", "groups": {}}, verbose=False)
    result = gguf.GGUFReader(str(output))
    assert result.alignment == ALIGNMENT
    np.testing.assert_array_equal(result.tensors[0].data.reshape(-1), values)
    np.testing.assert_array_equal(result.tensors[1].data.reshape(-1), values + 1)


def test_failed_open_never_caches_partial_metadata_and_can_retry(tmp_path):
    path = tmp_path / "broken.gguf"
    _model(path)
    valid = path.read_bytes()
    path.write_bytes(valid[:50])
    reader = GGUFReader(str(path))
    for _ in range(2):
        with pytest.raises(ValueError, match="Truncated"):
            reader.get_metadata()
        assert not reader._opened
        assert reader.metadata == {} and reader.tensors == []
    path.write_bytes(valid)
    assert len(reader.get_tensor_names()) == 2


def test_absurd_string_length_fails_before_allocation(tmp_path):
    path = tmp_path / "bad-string.gguf"
    path.write_bytes(struct.pack("<IIQQQ", GGUFReader.GGUF_MAGIC, 3, 0, 1, 2**63))
    with pytest.raises(ValueError, match="bounds"):
        GGUFReader(str(path)).open()


@pytest.mark.parametrize("alignment", [0, 3, -32, True, 32.5,
                                       GGUFTypedInt(32, 5), GGUFTypedInt(32, 10)])
def test_invalid_alignment_fails_before_tensor_reads(tmp_path, alignment):
    from magicquant.gguf.writer import _write_metadata_value, _write_string

    path = tmp_path / "bad-alignment.gguf"
    with path.open("wb") as handle:
        handle.write(struct.pack("<IIQQ", GGUFReader.GGUF_MAGIC, 3, 0, 1))
        _write_string(handle, "general.alignment")
        _write_metadata_value(handle, alignment)
    with pytest.raises(ValueError, match="general.alignment"):
        GGUFReader(str(path)).open()


@pytest.mark.parametrize("version", [0, 1, 4])
def test_unknown_header_version_fails_explicitly(tmp_path, version):
    path = tmp_path / "bad-version.gguf"
    path.write_bytes(struct.pack("<IIQQ", GGUFReader.GGUF_MAGIC, version, 0, 0))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        GGUFReader(str(path)).open()


def test_impossible_record_count_fails_explicitly(tmp_path):
    path = tmp_path / "bad-count.gguf"
    path.write_bytes(struct.pack("<IIQQ", GGUFReader.GGUF_MAGIC, 3, 2**63, 0))
    with pytest.raises(ValueError, match="record counts"):
        GGUFReader(str(path)).open()


def test_deeply_nested_metadata_is_rejected(tmp_path):
    path = tmp_path / "nested.gguf"
    path.write_bytes(
        struct.pack("<IIQQQI", GGUFReader.GGUF_MAGIC, 3, 0, 1, 0, 9)
        + struct.pack("<IQ", 9, 1) * 34 + struct.pack("<IQB", 0, 1, 0)
    )
    with pytest.raises(ValueError, match="deeply nested"):
        GGUFReader(str(path)).open()
