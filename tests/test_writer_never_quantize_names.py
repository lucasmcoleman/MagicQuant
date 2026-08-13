"""Tensor names llama.cpp's own quantizer refuses to quantize by name must
be written as F32 by MagicQuant's writer too, regardless of shape or the
scheme assigned to their group.

Ground truth: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (arch
nemotron_h_moe) crashed llama-perplexity on its very first sensitivity
probe with:

    ggml-cpu/binary-ops.cpp:135: binary_op: unsupported types:
    dst: f32, src0: f32, src1: q8_0

Root cause: the writer quantized blk.*.ssm_norm.weight to Q8_0. That
tensor is a 2-D {512, 8} per-group scale (not the usual 1-D norm vector),
so the 1-D norm/bias F32 rule never fires for it -- and its row width
(ne[0]=512) is both 32- and 256-divisible, so the block-size fallback
never fires either. llama.cpp consumes it as the RHS of a binary op
(build_norm -> ggml_mul in src/models/mamba-base.cpp), and ggml binary
ops require float operands on both sides, so inference aborts.

Also worth noting: "ssm_norm.weight" classifies into tensor group N
(TensorGroupClassifier's `_norm.weight$` pattern beats its `ssm_`
pattern), NOT group S -- so this bug was never reachable through
_is_f32_required_ssm_operand's group == "S" gate in the first place. The
fix here is deliberately name-based with no group restriction.

See magicquant.gguf.writer._is_never_quantized /
_NEVER_QUANTIZE_NAME_SUBSTRINGS, mirrored from llama.cpp's own
tensor_allows_quantization() in src/llama-quant.cpp (~lines 289-366).
"""
import numpy as np

import magicquant.gguf.source as source_mod
from magicquant.gguf.writer import (
    GGUFWriter,
    _is_never_quantized,
)

from tests.test_writer import StubSource


# ── unit: the name matcher ──────────────────────────────────────────────────

def test_norm_weight_names_match():
    assert _is_never_quantized("blk.0.ssm_norm.weight")
    assert _is_never_quantized("blk.0.attn_norm.weight")


def test_expert_gating_and_routing_names_match():
    assert _is_never_quantized("blk.0.ffn_gate_inp.weight")
    assert _is_never_quantized("blk.0.ffn_gate_tid2eid.weight")


def test_positional_and_token_type_names_match():
    assert _is_never_quantized("position_embd.weight")
    assert _is_never_quantized("token_types.weight")


def test_ordinary_projection_names_do_not_match():
    assert not _is_never_quantized("blk.0.ssm_in.weight")
    assert not _is_never_quantized("blk.0.ffn_down.weight")
    assert not _is_never_quantized("token_embd.weight")


# ── integration: the writer's Pass 1 plan honors it end to end ─────────────

def _tensor(name, shape):
    n = 1
    for d in shape:
        n *= d
    return (name, np.random.randn(n).astype(np.float32), shape)


def test_ssm_norm_2d_tensor_forced_to_f32_not_quantized(tmp_path, monkeypatch):
    # This is the exact bug: a 2-D ssm_norm.weight ({512, 8} in the real
    # run; ne[0]=512 -> the writer's last numpy dim here) with a
    # block-compatible row width must NOT be quantized, even though
    # neither the 1-D rule nor the block-size fallback catches it. The
    # row width (256, below) is deliberately both 32- and 256-divisible
    # -- an incompatible width would let the pre-existing block-size
    # fallback mask this bug by also (coincidentally) landing on F32.
    src = StubSource([_tensor("blk.0.ssm_norm.weight", (4, 256))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"N": "Q8_0"}}, verbose=False,
    )

    assert len(writer._fallbacks) == 1
    record = writer._fallbacks[0]
    assert record["tensor"] == "blk.0.ssm_norm.weight"
    assert record["requested"] == "Q8_0"
    assert record["actual"] == "F32"
    assert record["reason"] == "never-quantize-name"

    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(str(tmp_path / "h.gguf"))
    reader.open()
    try:
        info = reader.get_tensor_info("blk.0.ssm_norm.weight")
        assert info["data_type"] == 0  # F32 id
    finally:
        reader.close()


def test_ordinary_2d_ssm_projection_still_quantized(tmp_path, monkeypatch):
    # Proves the new rule doesn't over-match: an ordinary SSM projection
    # (not a norm, not a conv operand) with a block-compatible row width
    # must still be quantized as requested.
    src = StubSource([_tensor("blk.0.ssm_in.weight", (256, 256))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"S": "Q8_0"}}, verbose=False,
    )

    assert not any(r["tensor"] == "blk.0.ssm_in.weight" for r in writer._fallbacks)

    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(str(tmp_path / "h.gguf"))
    reader.open()
    try:
        info = reader.get_tensor_info("blk.0.ssm_in.weight")
        assert info["data_type"] == 8  # Q8_0 id -- actually quantized
    finally:
        reader.close()


def test_ffn_gate_inp_expert_gating_left_unquantized(tmp_path, monkeypatch):
    # A second upstream name-list entry, not just the norm rule: expert
    # gating must also come out F32 even at a 2-D shape that would
    # otherwise be fully block-compatible (row width 256 is both 32-
    # and 256-divisible, so the block-size fallback wouldn't catch it).
    src = StubSource([_tensor("blk.0.ffn_gate_inp.weight", (8, 256))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"R": "Q8_0"}}, verbose=False,
    )

    record = next(r for r in writer._fallbacks if r["tensor"] == "blk.0.ffn_gate_inp.weight")
    assert record["requested"] == "Q8_0"
    assert record["actual"] == "F32"
    assert record["reason"] == "never-quantize-name"


def test_ssm_conv1d_still_reports_f32_required_operand_not_never_quantize(tmp_path, monkeypatch):
    # ssm_conv1d is in BOTH the SSM-specific list and the new never-
    # quantize name list (by design -- see the comment on
    # _NEVER_QUANTIZE_NAME_SUBSTRINGS). The more specific, pre-existing
    # rule must win and keep reporting its own distinct reason, since the
    # never-quantize check is guarded on target_ggml_name != "F32" and
    # runs after it.
    src = StubSource([_tensor("blk.0.ssm_conv1d.weight", (4, 8))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"S": "Q8_0"}}, verbose=False,
    )

    record = next(r for r in writer._fallbacks if r["tensor"] == "blk.0.ssm_conv1d.weight")
    assert record["reason"] == "f32-required-operand"
    assert not any(
        r["tensor"] == "blk.0.ssm_conv1d.weight" and r["reason"] == "never-quantize-name"
        for r in writer._fallbacks
    )
