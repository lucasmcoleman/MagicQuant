"""SSM conv-weight operands must be F32 regardless of the group's configured
scheme -- including a FLOAT scheme (BF16/F16).

Ground truth: the completed Qwopus3.6-27B (qwen35 hybrid SSM+attention) run
had one candidate with group S at keep_scheme=BF16; its ssm_conv1d.weight
tensors were written BF16 (a float passthrough), and every probe/candidate
built that way crashed llama-perplexity on the fork's HIP kernel:
ggml-cuda/ssm-conv.cu:173 GGML_ASSERT(src1->nb[0] == sizeof(float)). The
writer's block-size fallback (_block32_fallback, which already special-cases
group "S" -> F32) never even runs for a float target, because F16/BF16 have
block_size==1 -- so the existing "S -> F32" rule was silently unreachable
whenever the group's scheme was itself a float type. See
magicquant.gguf.writer._is_f32_required_ssm_operand for the name-based,
scheme-agnostic fix.
"""
import numpy as np

import magicquant.gguf.source as source_mod
from magicquant.gguf.writer import (
    GGUFWriter,
    create_hybrid_gguf,
    _is_f32_required_ssm_operand,
)

from tests.test_writer import StubSource


# ── unit: the name matcher ──────────────────────────────────────────────────

def test_canonical_ssm_conv1d_name_matches():
    assert _is_f32_required_ssm_operand("blk.0.ssm_conv1d.weight")
    assert _is_f32_required_ssm_operand("blk.0.ssm_conv1d.bias")


def test_kimi_linear_conv1d_qkv_suffix_names_match():
    assert _is_f32_required_ssm_operand("blk.0.ssm_conv1d_q.weight")
    assert _is_f32_required_ssm_operand("blk.0.ssm_conv1d_k.weight")
    assert _is_f32_required_ssm_operand("blk.0.ssm_conv1d_v.weight")


def test_uncanonicalized_hf_style_kimi_names_match():
    # Defensive: if a source hasn't been mapped to the canonical GGUF name
    # yet, the raw HF-style Kimi Linear name should still be recognized.
    assert _is_f32_required_ssm_operand("model.layers.0.self_attn.q_conv1d.weight")


def test_ordinary_ssm_projection_names_do_not_match():
    # dt_proj / A_log / alpha / beta projections are ordinary matmul weights
    # (2D like any other linear layer) -- not conv operands, safe to quantize.
    assert not _is_f32_required_ssm_operand("blk.0.ssm_dt.weight")
    assert not _is_f32_required_ssm_operand("blk.0.ssm_a.weight")
    assert not _is_f32_required_ssm_operand("blk.0.ssm_alpha.weight")
    assert not _is_f32_required_ssm_operand("blk.0.ssm_beta.weight")


def test_non_ssm_names_do_not_match():
    assert not _is_f32_required_ssm_operand("blk.0.ffn_down.weight")
    assert not _is_f32_required_ssm_operand("token_embd.weight")


# ── integration: the writer's Pass 1 plan honors it end to end ─────────────

def _tensor(name, shape):
    n = 1
    for d in shape:
        n *= d
    return (name, np.random.randn(n).astype(np.float32), shape)


def test_ssm_conv1d_bf16_scheme_forced_to_f32_in_plan(tmp_path, monkeypatch):
    # Row width 6 is neither /256 nor /32 either, so if the F32-force didn't
    # run before the (skipped, float-target) block-size check, this would
    # otherwise have shipped as BF16-designated (written F16 on disk) --
    # exactly the crash-reproducing shape from the real run.
    src = StubSource([_tensor("blk.0.ssm_conv1d.weight", (4, 6))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"S": "BF16"}}, verbose=False,
    )

    assert len(writer._fallbacks) == 1
    record = writer._fallbacks[0]
    assert record["tensor"] == "blk.0.ssm_conv1d.weight"
    assert record["group"] == "S"
    assert record["requested"] == "BF16"
    assert record["actual"] == "F32"
    assert record["reason"] == "f32-required-operand"

    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(str(tmp_path / "h.gguf"))
    reader.open()
    try:
        info = reader.get_tensor_info("blk.0.ssm_conv1d.weight")
        assert info["data_type"] == 0  # F32 id
    finally:
        reader.close()


def test_ssm_conv1d_f16_scheme_also_forced_to_f32(tmp_path, monkeypatch):
    src = StubSource([_tensor("blk.0.ssm_conv1d.weight", (4, 8))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"S": "F16"}}, verbose=False,
    )

    record = next(r for r in writer._fallbacks if r["tensor"] == "blk.0.ssm_conv1d.weight")
    assert record["requested"] == "F16"
    assert record["actual"] == "F32"
    assert record["reason"] == "f32-required-operand"


def test_quantized_s_group_conv1d_row_incompatible_still_forced_f32(tmp_path, monkeypatch):
    # Scheme is quantized (not float) AND the row happens to be block-
    # incompatible: must still land on F32, same end result as before this
    # fix, just via the new name-based rule instead of (or in addition to)
    # the block-size path.
    src = StubSource([_tensor("blk.0.ssm_conv1d.weight", (4, 6))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"S": "Q4_K_M"}}, verbose=False,
    )

    record = next(r for r in writer._fallbacks if r["tensor"] == "blk.0.ssm_conv1d.weight")
    assert record["requested"] == "Q4_K"
    assert record["actual"] == "F32"
    assert record["reason"] == "f32-required-operand"


def test_non_conv1d_s_group_tensor_still_uses_block_size_fallback_rule(tmp_path, monkeypatch):
    # An ordinary SSM projection (not a conv operand) with an incompatible
    # row size for its quantized scheme must still be caught by the
    # pre-existing _block32_fallback "group S -> F32" rule, unaffected by
    # the new conv-specific check (which doesn't match this tensor's name).
    src = StubSource([_tensor("blk.0.ssm_dt.weight", (4, 6))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"S": "Q4_K_M"}}, verbose=False,
    )

    record = next(r for r in writer._fallbacks if r["tensor"] == "blk.0.ssm_dt.weight")
    assert record["requested"] == "Q4_K"
    assert record["actual"] == "F32"
    assert record["reason"] == "block-size"  # unchanged: the OLD rule, not ours


def test_non_ssm_bf16_tensor_still_passes_through_as_bf16_designated(tmp_path, monkeypatch):
    # A non-S-group tensor (ordinary FFN weight) assigned BF16 must NOT be
    # touched by the new SSM-scoped rule -- it still goes through the
    # pre-existing (unrelated) BF16->F16 on-disk downgrade, not F32.
    src = StubSource([_tensor("blk.0.ffn_down.weight", (256, 256))])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"D": "BF16"}}, verbose=False,
    )

    assert not any(r["reason"] == "f32-required-operand" for r in writer._fallbacks)

    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(str(tmp_path / "h.gguf"))
    reader.open()
    try:
        info = reader.get_tensor_info("blk.0.ffn_down.weight")
        assert info["data_type"] == 1  # F16 id (BF16 on-disk downgrade), not F32
    finally:
        reader.close()
