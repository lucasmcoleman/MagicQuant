"""Security/bounds hardening tests (L15)."""
import numpy as np
import pytest

from magicquant.quant.ggml_binding import get_handle
from magicquant.gguf.source import LoRAMergedSource


def test_ggml_encode_rejects_absurd_size(monkeypatch):
    """An absurd computed output size must raise (not allocate a huge buffer)."""
    import magicquant.quant.ggml_binding as gb

    handle = get_handle()
    small = np.zeros(256, dtype=np.float32)
    handle.encode(small, "Q8_0")  # sanity: small encode works

    # Force the size computation to report an absurd value for this small
    # input so we exercise the bound guard without allocating real memory.
    monkeypatch.setattr(gb, "_expected_size", lambda *a, **k: 64 * 1024 ** 3)
    with pytest.raises(ValueError, match="out of bounds"):
        handle.encode(small, "Q8_0")


def test_adapter_read_rejects_eof_overrun(tmp_path):
    """_read_adapter_tensor must raise when a read would pass EOF."""
    f = tmp_path / "adapter_model.safetensors"
    f.write_bytes(b"\x00" * 64)  # tiny file

    # Build a LoRAMergedSource without running __init__ (avoid needing a real
    # base model + adapter config).
    src = LoRAMergedSource.__new__(LoRAMergedSource)
    src._adapter_tensors = {
        "x.lora_A.weight": {
            "dtype": "F32",
            "shape": [4, 4],
            "filepath": str(f),
            "byte_offset": 0,
            "byte_length": 4 * 4 * 4,  # 64 bytes
            "data_start": 32,          # 32 + 64 = 96 > 64 file size
        }
    }
    with pytest.raises(ValueError, match="past EOF"):
        src._read_adapter_tensor("x.lora_A.weight")


def test_adapter_read_rejects_negative_offset(tmp_path):
    f = tmp_path / "adapter_model.safetensors"
    f.write_bytes(b"\x00" * 256)
    src = LoRAMergedSource.__new__(LoRAMergedSource)
    src._adapter_tensors = {
        "x.lora_A.weight": {
            "dtype": "F32", "shape": [2, 2], "filepath": str(f),
            "byte_offset": -8, "byte_length": 16, "data_start": 0,
        }
    }
    with pytest.raises(ValueError, match="negative"):
        src._read_adapter_tensor("x.lora_A.weight")
