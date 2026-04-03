"""Tests for quantization dtype guards and encoding correctness."""

import numpy as np
import pytest

from magicquant.quant.converters import encode_to_ggml_bytes, ggml_tensor_data_size


class TestDtypeGuards:
    """Verify that encode_to_ggml_bytes rejects non-floating-point input."""

    def test_rejects_int8_input(self):
        """Integer tensors must be rejected — they indicate pre-quantized data."""
        bad_weights = np.zeros(256, dtype=np.int8)
        with pytest.raises(ValueError, match="floating-point"):
            encode_to_ggml_bytes(bad_weights, "Q8_0")

    def test_rejects_uint8_input(self):
        bad_weights = np.zeros(256, dtype=np.uint8)
        with pytest.raises(ValueError, match="floating-point"):
            encode_to_ggml_bytes(bad_weights, "Q4_K")

    def test_rejects_int32_input(self):
        bad_weights = np.zeros(256, dtype=np.int32)
        with pytest.raises(ValueError, match="floating-point"):
            encode_to_ggml_bytes(bad_weights, "BF16")

    def test_accepts_float32(self):
        """Standard float32 input should work for all types."""
        weights = np.random.randn(256).astype(np.float32)
        blob = encode_to_ggml_bytes(weights, "Q8_0")
        expected_size = ggml_tensor_data_size("Q8_0", 256)
        assert len(blob) == expected_size

    def test_accepts_float16(self):
        """Float16 input should be auto-promoted to float32 and encoded."""
        weights = np.random.randn(256).astype(np.float16)
        blob = encode_to_ggml_bytes(weights, "Q8_0")
        expected_size = ggml_tensor_data_size("Q8_0", 256)
        assert len(blob) == expected_size

    def test_accepts_float64(self):
        """Float64 input should be narrowed to float32 and encoded."""
        weights = np.random.randn(256).astype(np.float64)
        blob = encode_to_ggml_bytes(weights, "Q8_0")
        expected_size = ggml_tensor_data_size("Q8_0", 256)
        assert len(blob) == expected_size

    def test_rejects_unknown_type(self):
        """Unknown ggml type name should raise ValueError."""
        weights = np.random.randn(256).astype(np.float32)
        with pytest.raises(ValueError, match="No ggml encoder"):
            encode_to_ggml_bytes(weights, "NONEXISTENT_TYPE")


class TestEncoderOutputSizes:
    """Verify that all encoders produce correct output sizes."""

    @pytest.mark.parametrize("ggml_type,n_elements", [
        ("F32", 256),
        ("F16", 256),
        ("BF16", 256),
        ("Q8_0", 256),
        ("Q4_0", 256),
        ("Q6_K", 256),
        ("Q5_K", 256),
        ("Q4_K", 256),
        ("IQ4_NL", 256),
        ("MXFP4", 256),
    ])
    def test_output_size_matches_expected(self, ggml_type, n_elements):
        weights = np.random.randn(n_elements).astype(np.float32)
        blob = encode_to_ggml_bytes(weights, ggml_type)
        expected = ggml_tensor_data_size(ggml_type, n_elements)
        assert len(blob) == expected, (
            f"{ggml_type}: got {len(blob)} bytes, expected {expected}"
        )

    def test_bf16_round_trip_preserves_values(self):
        """BF16 encode should preserve values within BF16 precision."""
        weights = np.array([1.0, -1.0, 0.0, 3.14, -2.71], dtype=np.float32)
        blob = encode_to_ggml_bytes(weights, "BF16")
        # Decode BF16: each value is 2 bytes, upper 16 bits of float32
        decoded = np.frombuffer(blob, dtype=np.uint16)
        f32_bytes = (decoded.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_allclose(f32_bytes, weights, rtol=1e-2)
