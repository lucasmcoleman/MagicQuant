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
    ]
    quant_config = {
        "base": "Q8_0",
        "groups": {"E": "BF16", "H": "Q6_K", "Q": "Q4_K_M", "D": "MXFP4_MOE",
                   "U": "Q4_K_M", "N": "BF16"},
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
    actual, reason = resolve_tensor_type(
        "blk.0.attn_norm.weight", [256], 1, "N", "Q8_0"
    )
    assert actual == "F32" and reason == "1d-f32"


def test_block_fallback_rule():
    # 192-wide rows can't hold a 256-block K-quant.
    actual, reason = resolve_tensor_type(
        "blk.0.ffn_up.weight", [256, 192], 2, "U", "Q4_K_M"
    )
    assert reason == "block-size"
    assert actual != "Q4_K"
