"""Stream weights + streamed-bytes accounting (spec: /server/ai/docs/specs/magicquant-bandwidth.md §2.2)."""
import logging
import math

import numpy as np
import pytest

from magicquant.v2 import bandwidth as bw

QWEN_MD = {
    "general.architecture": "qwen35moe",
    "qwen35moe.expert_count": 256,
    "qwen35moe.expert_used_count": 8,
}


def _entry(group, shape, fixed=False, choices=None):
    return {"group": group, "shape": list(shape), "n_elems": int(np.prod(shape)),
            "fixed": fixed, "wnorm": None, "choices": choices or {}}


def test_by_group_from_hparams():
    w = bw.stream_weights_by_group(QWEN_MD)
    assert w["X"] == pytest.approx(8 / 256)
    assert w["E"] == 0.0 and w["V"] == 0.0 and w["UNKNOWN"] == 1.0
    for g in "QKOUDSRHN":
        assert w[g] == 1.0
    assert w["default"] == 1.0
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in w.values())


def test_override_wins():
    w = bw.stream_weights_by_group(QWEN_MD, {"H": 0.5})
    assert w["H"] == 0.5


def test_missing_hparams_prices_experts_hot_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group({"general.architecture": "qwen35moe"}, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "qwen35moe.expert_count" in msg and "qwen35moe.expert_used_count" in msg


def test_zero_expert_count_never_divides(caplog):
    md = dict(QWEN_MD, **{"qwen35moe.expert_count": 0})
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group(md, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    assert any("expert_count" in r.getMessage() for r in caplog.records)


def test_per_tensor_rules(caplog):
    tensors = {
        "token_embd.weight": _entry("E", (100, 64)),
        "per_layer_token_embd.weight": _entry("E", (100, 16)),
        "blk.0.nextn.embed_tokens.weight": _entry("H", (100, 64)),   # gathered by NAME inside H
        "blk.0.nextn.eh_proj.weight": _entry("H", (64, 128)),
        "output.weight": _entry("H", (100, 64)),
        "blk.0.ffn_down_exps.weight": _entry("X", (4, 64, 32)),
        "blk.0.attn_q.weight": _entry("Q", (64, 64)),
        "blk.0.per_layer_proj.weight": _entry("UNKNOWN", (64, 16)),   # PLE projection: read every token
        "blk.0.ffn_gate_inp.weight": _entry("R", (4, 64), fixed=True),
    }
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights(tensors, QWEN_MD, log=logging.getLogger("t"))
    assert w["token_embd.weight"] == 0.0
    assert w["per_layer_token_embd.weight"] == 0.0
    assert w["blk.0.nextn.embed_tokens.weight"] == 0.0
    assert w["blk.0.nextn.eh_proj.weight"] == 1.0
    assert w["output.weight"] == 1.0
    assert w["blk.0.ffn_down_exps.weight"] == pytest.approx(8 / 256)
    assert w["blk.0.attn_q.weight"] == 1.0
    assert w["blk.0.per_layer_proj.weight"] == 1.0
    assert w["blk.0.ffn_gate_inp.weight"] == 1.0
    assert any("UNKNOWN" in r.getMessage() for r in caplog.records)


def test_shape_sanity_warnings(caplog):
    tensors = {"blk.0.ffn_up_exps.weight": _entry("X", (64, 32)),   # 2-D X
               "blk.0.ffn_up.weight": _entry("U", (4, 64, 32))}      # 3-D non-X
    with caplog.at_level(logging.WARNING):
        bw.stream_weights(tensors, QWEN_MD, log=logging.getLogger("t"))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("blk.0.ffn_up_exps.weight" in m for m in msgs)
    assert any("blk.0.ffn_up.weight" in m for m in msgs)


def _table():
    return {
        "a": _entry("U", (8, 8), choices={"BF16": {"actual": "BF16", "bytes": 128, "werr": 0.0},
                                          "Q4_K_M": {"actual": "Q4_K_M", "bytes": 36, "werr": 1.0}}),
        "x": _entry("X", (2, 8, 8), choices={"BF16": {"actual": "BF16", "bytes": 256, "werr": 0.0},
                                             "Q4_K_M": {"actual": "Q4_K_M", "bytes": 72, "werr": 0.5}}),
        "n": _entry("N", (8,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 32, "werr": 0.0}}),
    }


def test_streamed_bytes_and_pure_loss_by_hand():
    t = _table()
    w = {"a": 1.0, "x": 0.25, "n": 1.0}
    assign = {"a": "Q4_K_M", "x": "BF16", "n": "F32"}
    assert bw.streamed_bytes(assign, t, w) == 36 + int(0.25 * 256) + 32
    assert bw.pure_loss(assign, t, {"U": 2.0, "X": 3.0}) == pytest.approx(2.0 * 1.0 + 0.0)
    assert bw.streamed_bytes({"a": "Q4_K_M"}, t, {}) == 36          # missing weight -> 1.0


def test_bpw_by_group_and_group_view(caplog):
    t = _table()
    assign = {"a": "Q4_K_M", "x": "Q4_K_M", "n": "F32"}
    b = bw.bpw_by_group(assign, t)
    assert b["U"] == pytest.approx(36 * 8 / 64)
    assert b["X"] == pytest.approx(72 * 8 / 128)
    by_group, exceptions = bw.group_view({"a": 1.0, "x": 0.25, "n": 1.0}, t)
    assert by_group == {"U": 1.0, "X": 0.25, "N": 1.0}
    assert exceptions == {}
    # a second N tensor at a different weight: modal value wins, the odd one is an exception, a WARNING fires
    t2 = dict(t)
    t2["n2"] = _entry("N", (8,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 32, "werr": 0.0}})
    t2["n3"] = _entry("N", (8,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 32, "werr": 0.0}})
    with caplog.at_level(logging.WARNING):
        by_group, exceptions = bw.group_view({"a": 1.0, "x": 0.25, "n": 1.0, "n2": 1.0, "n3": 0.0}, t2, log=logging.getLogger("t"))
    assert by_group["N"] == 1.0 and exceptions == {"n3": 0.0}
    assert any("distinct weights" in r.getMessage() for r in caplog.records)


def test_group_view_ignores_gathered_by_name_disagreement(caplog):
    # rule-2 tensors (MTP embedding inside H) are an intended exception: no WARNING
    t = {"output.weight": _entry("H", (100, 64)),
         "blk.0.nextn.embed_tokens.weight": _entry("H", (100, 64))}
    with caplog.at_level(logging.WARNING):
        by_group, exceptions = bw.group_view({"output.weight": 1.0, "blk.0.nextn.embed_tokens.weight": 0.0}, t, log=logging.getLogger("t"))
    assert by_group["H"] == 1.0 and exceptions == {"blk.0.nextn.embed_tokens.weight": 0.0}
    assert not any("distinct weights" in r.getMessage() for r in caplog.records)


def test_float_entry_stub_source_gets_expert_ratio(monkeypatch, tmp_path):
    from tests.test_v2_audit_regressions import _float_entry
    weights = np.full((2, 4, 32), 0.5, dtype=np.float32)
    entry = _float_entry(monkeypatch, tmp_path, weights, metadata=QWEN_MD)
    w = bw.stream_weights({"blk.0.ffn_down_exps.weight": entry}, QWEN_MD)
    assert w["blk.0.ffn_down_exps.weight"] == pytest.approx(8 / 256)


def test_float_entry_default_metadata_prices_hot(monkeypatch, tmp_path, caplog):
    from tests.test_v2_audit_regressions import _float_entry
    weights = np.full((2, 4, 32), 0.5, dtype=np.float32)
    entry = _float_entry(monkeypatch, tmp_path, weights)
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights({"blk.0.ffn_down_exps.weight": entry},
                              {"general.architecture": "llama"}, log=logging.getLogger("t"))
    assert w["blk.0.ffn_down_exps.weight"] == 1.0
    assert any("expert_count" in r.getMessage() for r in caplog.records)
