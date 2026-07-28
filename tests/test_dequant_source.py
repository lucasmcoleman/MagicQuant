"""Tests for the opt-in dequantize-already-quantized-source path.

Covers:
  - ``_allow_dequant_default()``'s env-var parsing.
  - ``GGUFSource``'s default (dequant off) behavior on a quantized tensor:
    ``can_decode`` False, ``read_tensor_f32`` None, ``read_tensor_raw``
    returns the tensor's exact on-disk bytes.
  - ``GGUFSource(allow_dequant=True)`` on a real Q8_0 tensor: real
    dequantization within tolerance, ``dequantized_types`` tracking.
  - The writer's pre-quantized guard (dequant off, different desired type)
    and the passthrough zero-fill regression (dequant off, SAME desired
    type -- output bytes must equal the source bytes, not zeros).
  - The writer's double-quantization path end-to-end with dequant enabled,
    including the DOUBLE-QUANTIZATION warning.

Fixture GGUFs are built with the upstream ``gguf`` package (same tool
``test_reader_autoopen.py`` uses), with the one quantized tensor's bytes
produced by magicquant's own ``encode_to_ggml_bytes`` so the "original"
float data is known and comparable after a real libggml decode.
"""
import os

import numpy as np
import pytest

gguf_pkg = pytest.importorskip("gguf")

from magicquant.gguf.source import (
    GGUFSource,
    _ALLOW_DEQUANT_ENV,
    _allow_dequant_default,
)
from magicquant.gguf.writer import create_hybrid_gguf
from magicquant.quant.converters import encode_to_ggml_bytes
from magicquant.quant.ggml_binding import LibggmlNotFound, get_handle


# ---------------------------------------------------------------------------
# Fixture: a tiny single-tensor GGUF with a real Q8_0 tensor
# ---------------------------------------------------------------------------

_TENSOR_NAME = "blk.0.attn_q.weight"


def _make_q8_gguf(tmp_path, seed=0, rows=4, cols=256):
    """Write a one-tensor GGUF with ``_TENSOR_NAME`` encoded as real Q8_0.

    2-D (rows, cols) rather than a flat 1-D array: the writer forces every
    1-D tensor (norms/biases) to F32 regardless of scheme (see writer.py's
    "1D tensors ... must stay at F32" rule), which would make the
    dequant-enabled re-quantize test assert on the wrong type for reasons
    unrelated to the feature under test. cols=256 keeps the row Q8_0-block-
    aligned (block size 32) AND K-quant-block-aligned (block size 256), so
    the Q4_K target used below doesn't trip the block-32 fallback either.

    Returns (path, original_f32_data, q8_blob) so callers can compare a
    decode against the known original and the raw blob against a
    passthrough copy.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((rows, cols)).astype(np.float32)
    blob = encode_to_ggml_bytes(data.flatten(), "Q8_0")
    raw = np.frombuffer(blob, dtype=np.uint8)

    # Q8_0: block_size=32, type_size=34 bytes/block -- one block per row here.
    bytes_per_row = (cols // 32) * 34
    assert bytes_per_row * rows == len(blob)

    path = str(tmp_path / "q8.gguf")
    w = gguf_pkg.GGUFWriter(path, arch="llama")
    w.add_tensor(
        _TENSOR_NAME, raw, raw_shape=(rows, bytes_per_row),
        raw_dtype=gguf_pkg.GGMLQuantizationType.Q8_0,
    )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, data.flatten(), blob


@pytest.fixture
def _require_libggml():
    """Skip the test if libggml can't be located (mirrors test_ggml_decode.py)."""
    try:
        get_handle()
    except LibggmlNotFound as e:
        pytest.skip(f"libggml not available: {e}")


# ---------------------------------------------------------------------------
# a. _allow_dequant_default(): env-var parsing
# ---------------------------------------------------------------------------

class TestAllowDequantDefault:
    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv(_ALLOW_DEQUANT_ENV, raising=False)
        assert _allow_dequant_default() is False

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "on", "ON", "yes", "Yes"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(_ALLOW_DEQUANT_ENV, value)
        assert _allow_dequant_default() is True

    @pytest.mark.parametrize("value", ["0", "", "false", "off", "no", "garbage"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv(_ALLOW_DEQUANT_ENV, value)
        assert _allow_dequant_default() is False


# ---------------------------------------------------------------------------
# b. GGUFSource default (dequant off) on a quantized tensor
# ---------------------------------------------------------------------------

class TestGGUFSourceDequantOff:
    def test_can_decode_false(self, tmp_path):
        path, _data, _blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path)
        try:
            assert src.can_decode(_TENSOR_NAME) is False
        finally:
            src.close()

    def test_read_tensor_f32_is_none(self, tmp_path):
        path, _data, _blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path)
        try:
            assert src.read_tensor_f32(_TENSOR_NAME) is None
        finally:
            src.close()

    def test_read_tensor_raw_returns_exact_bytes(self, tmp_path):
        path, _data, blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path)
        try:
            raw = src.read_tensor_raw(_TENSOR_NAME)
            assert raw == blob
        finally:
            src.close()


# ---------------------------------------------------------------------------
# c. GGUFSource(allow_dequant=True) on a real Q8_0 tensor
# ---------------------------------------------------------------------------

class TestGGUFSourceDequantOn:
    def test_can_decode_true(self, tmp_path, _require_libggml):
        path, _data, _blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path, allow_dequant=True)
        try:
            assert src.can_decode(_TENSOR_NAME) is True
        finally:
            src.close()

    def test_decode_matches_original_within_q8_0_tolerance(self, tmp_path, _require_libggml):
        path, data, _blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path, allow_dequant=True)
        try:
            out = src.read_tensor_f32(_TENSOR_NAME)
            assert out is not None
            assert out.shape == data.shape
            max_abs_err = np.max(np.abs(out - data))
            rel_err = max_abs_err / np.max(np.abs(data))
            assert rel_err < 0.01, f"Q8_0 decode relative error too large: {rel_err}"
        finally:
            src.close()

    def test_dequantized_types_tracked(self, tmp_path, _require_libggml):
        path, _data, _blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path, allow_dequant=True)
        try:
            assert src.dequantized_types == set()
            src.read_tensor_f32(_TENSOR_NAME)
            assert src.dequantized_types == {"Q8_0"}
        finally:
            src.close()

    def test_env_default_also_enables(self, tmp_path, monkeypatch, _require_libggml):
        """allow_dequant=None (the default) should take the env var's policy."""
        monkeypatch.setenv(_ALLOW_DEQUANT_ENV, "1")
        path, _data, _blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path)
        try:
            assert src.can_decode(_TENSOR_NAME) is True
            assert src.read_tensor_f32(_TENSOR_NAME) is not None
        finally:
            src.close()


# ---------------------------------------------------------------------------
# d/e/f. Writer end-to-end: guard, passthrough (zero-fill regression),
#         and dequant-enabled double-quantization with its warning.
# ---------------------------------------------------------------------------

class TestWriterDequantGuardAndPassthrough:
    def test_guard_raises_when_desired_type_differs(self, tmp_path, monkeypatch):
        """Dequant off + a DIFFERENT desired scheme -> ValueError naming the env var."""
        monkeypatch.delenv(_ALLOW_DEQUANT_ENV, raising=False)
        path, _data, _blob = _make_q8_gguf(tmp_path)
        out = str(tmp_path / "out.gguf")
        with pytest.raises(ValueError) as exc:
            create_hybrid_gguf(
                output_path=out, base_model_path=path,
                quant_config={"base": "Q4_K_M", "groups": {}},
                verbose=False,
            )
        assert _ALLOW_DEQUANT_ENV in str(exc.value)
        assert not os.path.exists(out)

    def test_passthrough_same_type_copies_bytes_not_zeros(self, tmp_path, monkeypatch):
        """Regression for the zero-fill landmine: dequant off + SAME desired
        scheme as the source ("Q8_0" base, tensor is already Q8_0) must copy
        the tensor's exact source bytes -- this fails against the pre-fix
        code, which wrote a zero-filled blob here."""
        monkeypatch.delenv(_ALLOW_DEQUANT_ENV, raising=False)
        path, _data, blob = _make_q8_gguf(tmp_path)
        out = str(tmp_path / "out.gguf")
        create_hybrid_gguf(
            output_path=out, base_model_path=path,
            quant_config={"base": "Q8_0", "groups": {}},
            verbose=False,
        )

        from magicquant.gguf.reader import GGUFReader
        reader = GGUFReader(out)
        reader.open()
        try:
            info = reader.get_tensor_info(_TENSOR_NAME)
            assert info["data_type"] == 8  # Q8_0
            with open(out, "rb") as f:
                f.seek(reader.data_offset + info["offset"])
                out_bytes = f.read(len(blob))
        finally:
            reader.close()

        assert out_bytes != b"\x00" * len(blob), (
            "output tensor is zero-filled -- the passthrough zero-fill "
            "landmine regressed"
        )
        assert out_bytes == blob

    def test_dequant_enabled_writes_valid_output_and_warns(
        self, tmp_path, monkeypatch, caplog, _require_libggml,
    ):
        import logging
        monkeypatch.setenv(_ALLOW_DEQUANT_ENV, "1")
        caplog.set_level(logging.WARNING, logger="magicquant.gguf.writer")

        path, data, _blob = _make_q8_gguf(tmp_path)
        out = str(tmp_path / "out.gguf")
        create_hybrid_gguf(
            output_path=out, base_model_path=path,
            quant_config={"base": "Q4_K_M", "groups": {}},
            verbose=False,
        )
        assert os.path.exists(out)

        from magicquant.gguf.reader import GGUFReader
        reader = GGUFReader(out)
        reader.open()
        try:
            info = reader.get_tensor_info(_TENSOR_NAME)
            # Re-quantized to the requested Q4_K target, not passed through.
            assert info["data_type"] == 12  # Q4_K
        finally:
            reader.close()

        assert any(
            "DOUBLE-QUANTIZATION" in rec.message and "Q8_0" in rec.message
            for rec in caplog.records
        ), "expected the double-quantization warning to fire"


# ---------------------------------------------------------------------------
# Same-type short-circuit and probe-failure memoization
# ---------------------------------------------------------------------------

class TestDequantOnSameTypePassthrough:
    def test_same_type_short_circuits_to_passthrough_no_warning(
        self, tmp_path, monkeypatch, caplog, _require_libggml,
    ):
        """Dequant ON + desired scheme == source type: the writer takes the
        verbatim byte-copy path instead of a dequant->re-encode round-trip.
        The distinguishing observable is the DOUBLE-QUANTIZATION warning:
        without the short-circuit the tensor counts as re-quantized and the
        warning fires; with it, nothing is re-quantized and it must not."""
        import logging
        monkeypatch.setenv(_ALLOW_DEQUANT_ENV, "1")
        caplog.set_level(logging.WARNING, logger="magicquant.gguf.writer")

        path, _data, blob = _make_q8_gguf(tmp_path)
        out = str(tmp_path / "out.gguf")
        create_hybrid_gguf(
            output_path=out, base_model_path=path,
            quant_config={"base": "Q8_0", "groups": {}},
            verbose=False,
        )

        from magicquant.gguf.reader import GGUFReader
        reader = GGUFReader(out)
        reader.open()
        try:
            info = reader.get_tensor_info(_TENSOR_NAME)
            assert info["data_type"] == 8  # Q8_0, passed through
            with open(out, "rb") as f:
                f.seek(reader.data_offset + info["offset"])
                out_bytes = f.read(len(blob))
        finally:
            reader.close()

        assert out_bytes == blob
        assert not any(
            "DOUBLE-QUANTIZATION" in rec.message for rec in caplog.records
        ), "same-type passthrough must not count as double-quantization"


class TestProbeFailureWarnsOnce:
    def test_can_decode_probe_failure_warns_once_per_type(
        self, tmp_path, monkeypatch, caplog,
    ):
        """A broken libggml must not spam one warning per tensor: the writer
        calls can_decode() once per tensor in Pass 1, so the probe failure
        is memoized per type name and warned exactly once."""
        import logging
        import magicquant.quant.ggml_binding as gb

        def _boom(_type_name):
            raise RuntimeError("libggml unavailable (test)")

        monkeypatch.setattr(gb, "supports_decode", _boom)
        caplog.set_level(logging.WARNING, logger="magicquant.gguf.source")

        path, _data, _blob = _make_q8_gguf(tmp_path)
        src = GGUFSource(path, allow_dequant=True)
        try:
            assert src.can_decode(_TENSOR_NAME) is False
            assert src.can_decode(_TENSOR_NAME) is False
            assert src.can_decode(_TENSOR_NAME) is False
        finally:
            src.close()

        probe_warnings = [
            r for r in caplog.records if "Dequant probe" in r.message
        ]
        assert len(probe_warnings) == 1
