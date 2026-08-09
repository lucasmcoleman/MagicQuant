"""Encoder parity tests — MagicQuant's ctypes binding vs llama-quantize subprocess.

Both code paths invoke the same ggml_quantize_chunk function from the same
libggml-cpu.so. Byte-for-byte equality is the natural expectation; any
mismatch indicates a binding bug (wrong type_id, wrong nrows/n_per_row,
non-contiguous memory, etc.).

These tests require:
  - libggml available via magicquant.quant.ggml_binding
  - llama-quantize on PATH (or override via LLAMA_QUANTIZE env var)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from magicquant.quant.ggml_binding import ggml_encode

# `gguf` is an undeclared-but-required dependency of this module. Use
# importorskip so the suite SKIPS (not fails collection) when it's absent.
gguf = pytest.importorskip("gguf")

REF_TENSOR_PATH = Path(__file__).parent.parent / "fixtures" / "reference_tensor.f32.npy"

# Schemes verified in this PR. Q4_0 is included because it's already encoded
# in the codebase and registered by the existing schemes; PR2 adds the rest
# of the legacy Q-quants. PR3 adds the IQ-quants.
SCHEMES_PR1 = [
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "IQ4_NL", "MXFP4", "Q4_0",
    "Q2_K", "Q3_K",
]


def _llama_quantize_path() -> str:
    """Locate the llama-quantize executable."""
    explicit = os.environ.get("LLAMA_QUANTIZE")
    if explicit:
        if not Path(explicit).is_file():
            pytest.fail(f"LLAMA_QUANTIZE={explicit} does not exist")
        return explicit
    # Prefer the local llama.cpp build (supports all schemes including MXFP4)
    for candidate in [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
        Path.home() / "llama.cpp-build" / "build" / "bin" / "llama-quantize",
    ]:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("llama-quantize")
    if found:
        return found
    pytest.skip("llama-quantize not found and LLAMA_QUANTIZE not set")


@pytest.fixture(scope="module")
def reference_tensor() -> np.ndarray:
    if not REF_TENSOR_PATH.exists():
        pytest.fail(f"Reference tensor missing: {REF_TENSOR_PATH}")
    return np.load(REF_TENSOR_PATH)


def _write_f32_gguf(tensor: np.ndarray, out_path: Path, tensor_name: str = "test.weight") -> None:
    """Write a minimal GGUF file containing one F32 tensor with enough llama
    metadata for llama-quantize to load it. Values are fake — llama-quantize
    only requires the keys to exist, it doesn't run inference."""
    writer = gguf.GGUFWriter(str(out_path), arch="llama")
    writer.add_context_length(2048)
    writer.add_embedding_length(tensor.shape[-1])
    writer.add_block_count(1)
    writer.add_feed_forward_length(tensor.shape[-1])
    writer.add_head_count(1)
    writer.add_head_count_kv(1)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_tensor(tensor_name, tensor, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _read_first_tensor_bytes(gguf_path: Path) -> Tuple[bytes, str]:
    """Read the raw bytes of the first tensor in a GGUF file.

    Returns (tensor_bytes, ggml_type_name).
    """
    reader = gguf.GGUFReader(str(gguf_path))
    if not reader.tensors:
        raise RuntimeError(f"no tensors in {gguf_path}")
    t = reader.tensors[0]
    # t.data is a memoryview of float32 if the tensor is F32; for quantized
    # tensors it's the raw bytes view.
    raw = bytes(t.data.tobytes() if hasattr(t.data, 'tobytes') else t.data)
    type_name = t.tensor_type.name  # e.g. "Q4_K" — see gguf.GGMLQuantizationType
    return raw, type_name


@pytest.mark.parametrize("scheme", SCHEMES_PR1)
def test_encoder_byte_for_byte_matches_llama_quantize(
    scheme: str, reference_tensor: np.ndarray, tmp_path: Path
) -> None:
    """Quantize the reference tensor via MagicQuant's ctypes path and via
    llama-quantize subprocess. Assert output bytes are identical."""
    quantize_bin = _llama_quantize_path()

    # Step 1: Build the F32 source GGUF
    src_path = tmp_path / "src.f32.gguf"
    _write_f32_gguf(reference_tensor, src_path)

    # Step 2: Run llama-quantize --pure <scheme>. The CLI uses the higher-
    # level scheme name (MXFP4_MOE), not the ggml type name (MXFP4); other
    # schemes match between layers. --pure bypasses llama-quantize's
    # per-tensor-name dispatch (e.g. MXFP4_MOE only applies to MoE expert
    # tensors by default; --pure forces it on every tensor).
    cli_scheme = {"MXFP4": "MXFP4_MOE"}.get(scheme, scheme)
    dst_path = tmp_path / f"dst.{scheme}.gguf"
    result = subprocess.run(
        [quantize_bin, "--pure", str(src_path), str(dst_path), cli_scheme],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"llama-quantize {scheme} failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    # Step 3: Extract llama-quantize's output bytes
    llama_bytes, llama_type_name = _read_first_tensor_bytes(dst_path)

    # Step 4: Quantize the same tensor via MagicQuant's binding
    magic_bytes = ggml_encode(reference_tensor, scheme)

    # Step 5: Assert byte equality
    assert len(magic_bytes) == len(llama_bytes), (
        f"size mismatch for {scheme}: magic={len(magic_bytes)}, "
        f"llama={len(llama_bytes)}"
    )
    if magic_bytes != llama_bytes:
        # Find first divergence
        diff_idx = next(
            (i for i in range(len(magic_bytes)) if magic_bytes[i] != llama_bytes[i]),
            -1,
        )
        pytest.fail(
            f"byte mismatch for {scheme} at offset {diff_idx}: "
            f"magic={magic_bytes[diff_idx:diff_idx+8].hex()}, "
            f"llama={llama_bytes[diff_idx:diff_idx+8].hex()}"
        )
