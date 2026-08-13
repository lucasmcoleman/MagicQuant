"""Writer-parity for v2 type resolution: the pure function must predict
exactly the per-tensor on-disk types a real writer Pass 1 produces."""

import numpy as np
import pytest

import magicquant.gguf.source as source_mod
from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.gguf.writer import create_hybrid_gguf
from magicquant.v2.resolve import resolve_tensor_type, tensor_bytes

from tests.test_writer import StubSource, _f32_tensor


def _build(tmp_path, tensors, quant_config, monkeypatch):
    src = StubSource(tensors)
    out = str(tmp_path / "out.gguf")
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
    create_hybrid_gguf(
        output_path=out,
        base_model_path="ignored",
        quant_config=quant_config,
        verbose=False,
    )
    from magicquant.gguf.reader import GGUFReader

    reader = GGUFReader(out)
    reader.open()
    return reader


def test_resolution_parity_with_writer(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    try:
        from magicquant.quant.ggml_binding import get_handle
        get_handle()
    except Exception:
        pytest.skip("libggml unavailable")

    tensors = [
        _f32_tensor("token_embd.weight", (256, 256)),
        _f32_tensor("blk.0.attn_q.weight", (256, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
        _f32_tensor("blk.0.ffn_up.weight", (256, 192)),   # 192 % 256 != 0 -> block fallback for K-quants
        _f32_tensor("blk.0.attn_norm.weight", (256,)),    # 1-D -> F32
        _f32_tensor("output.weight", (256, 256)),
        # 2-D, block-compatible (256 is 32- and 256-divisible), classifies
        # into group N (not S) -- only the never-quantize-name rule catches
        # this; neither the 1-D rule nor the block-size rule would.
        _f32_tensor("blk.0.ssm_norm.weight", (4, 256)),
    ]
    quant_config = {
        "base": "Q8_0",
        "groups": {"E": "BF16", "H": "Q6_K", "Q": "Q4_K_M", "D": "MXFP4_MOE",
                   "U": "Q4_K_M", "N": "Q8_0"},
    }
    reader = _build(tmp_path, tensors, quant_config, monkeypatch)
    try:
        classifier = TensorGroupClassifier()
        infos = {i["name"]: i for i in reader.get_all_tensors_info()}
        id_to_name = {v: k for k, v in __import__(
            "magicquant.gguf.writer", fromlist=["GGML_TYPE"]
        ).GGML_TYPE.items()}
        for (name, _arr, shape) in tensors:
            group = classifier.classify_tensor(name)
            scheme = quant_config["groups"].get(group, quant_config["base"])
            predicted, _reason = resolve_tensor_type(
                name, list(shape), len(shape), group, scheme
            )
            written_id = infos[name]["data_type"]
            written = id_to_name[written_id]
            assert written == predicted, (
                f"{name}: writer wrote {written}, resolve predicted {predicted}"
            )
            # Byte-size parity with what actually landed on disk isn't
            # directly readable per tensor from the reader info here, but the
            # size function must at least be exact ggml block math:
            n = int(np.prod(shape))
            assert tensor_bytes(predicted, list(shape)) > 0
            assert tensor_bytes("F32", list(shape)) == 4 * n
    finally:
        reader.close()


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        resolve_tensor_type("blk.0.ffn_up.weight", [256, 256], 2, "U", "NOT_A_SCHEME")


def test_bf16_downgrade_and_1d_rules():
    actual, reason = resolve_tensor_type(
        "token_embd.weight", [256, 256], 2, "E", "BF16"
    )
    assert actual == "F16" and reason == "bf16-downgrade"
    # A bias is 1-D AND does not end in "weight", and the weight-suffix gate
    # (llama.cpp's own first check) now runs before the 1-D rule -- so it is
    # reported as not-a-weight-tensor. Same F32 outcome, more accurate reason.
    actual, reason = resolve_tensor_type(
        "blk.0.attn_q.bias", [256], 1, "Q", "Q8_0"
    )
    assert actual == "F32" and reason == "not-a-weight-tensor"

    # The 1-D rule in isolation needs a 1-D tensor that DOES end in "weight"
    # and is not a norm (norms hit the never-quantize-name rule first).
    actual, reason = resolve_tensor_type(
        "blk.0.some_scale.weight", [256], 1, "Q", "Q8_0"
    )
    assert actual == "F32" and reason == "1d-f32"


def test_never_quantize_name_rule():
    # A 1-D norm's name alone is enough to route it to F32 via the
    # never-quantize-name rule, which runs BEFORE the 1D-F32 rule -- so a
    # "_norm.weight"-suffixed tensor reports this reason, not "1d-f32",
    # even though the end result (F32) is identical. Mirrors writer.py's
    # _is_never_quantized (single source of truth, imported not copied).
    actual, reason = resolve_tensor_type(
        "blk.0.attn_norm.weight", [256], 1, "N", "Q8_0"
    )
    assert actual == "F32" and reason == "never-quantize-name"

    # And this is what catches the exact-bug shape: 2-D, block-compatible
    # (256 is 32- and 256-divisible), group N not S -- see
    # tests/test_writer_never_quantize_names.py for the writer-side
    # equivalent of this exact case.
    actual, reason = resolve_tensor_type(
        "blk.0.ssm_norm.weight", [4, 256], 2, "N", "Q8_0"
    )
    assert actual == "F32" and reason == "never-quantize-name"


def test_block_fallback_rule():
    # 192-wide rows can't hold a 256-block K-quant.
    actual, reason = resolve_tensor_type(
        "blk.0.ffn_up.weight", [256, 192], 2, "U", "Q4_K_M"
    )
    assert reason == "block-size"
    assert actual != "Q4_K"


def test_not_a_weight_tensor_rule_covers_ssm_a_and_ssm_d():
    """The 2026-08-13 v2 abort. ssm_a is src3 of ggml_ssm_scan, which asserts
    nb[0] == sizeof(float) (ggml-cpu/ops.cpp:9655); ssm_d is src1 of a
    ggml_mul (mamba-base.cpp:136,267). Both are 2-D (64,1), so the 1-D rule
    misses them, and ne[0]=1 is not 32-divisible so a QUANTIZED target falls
    back to F32 and looks safe -- but a FLOAT target has block_size == 1,
    skips the block-size check, and is written F16, which aborts llama.cpp.

    v1 never hit this because it holds groups at Q8_0; v2 holds them at BF16.
    So both keep-schemes must now resolve to F32.
    """
    for name in ("blk.0.ssm_a", "blk.0.ssm_d"):
        for keep in ("BF16", "F16", "Q8_0", "Q4_K_M"):
            actual, reason = resolve_tensor_type(name, [64, 1], 2, "S", keep)
            assert actual == "F32", f"{name} @ {keep} -> {actual}"
            assert reason == "not-a-weight-tensor"


def test_ssm_matmul_operands_are_NOT_forced_to_f32():
    """The cleared half of the audit. ssm_in/ssm_out end in 'weight' and are
    build_lora_mm operands -- F16 is legal there. Forcing them to F32 would
    roughly double them (10304x2688 and 2688x4096) for no reason."""
    for name in ("blk.0.ssm_in.weight", "blk.0.ssm_out.weight"):
        actual, _ = resolve_tensor_type(name, [10304, 2688], 2, "S", "BF16")
        assert actual == "F16", f"{name} must stay F16 under a BF16 keep"
        actual, _ = resolve_tensor_type(name, [10304, 2688], 2, "S", "Q8_0")
        assert actual == "Q8_0", f"{name} must stay quantizable"
