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


def test_infinite_expert_count_treated_as_missing_no_exception(caplog):
    md = dict(QWEN_MD, **{"qwen35moe.expert_count": float("inf")})
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group(md, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "expert_count" in msg and "inf" in msg   # raw hparam value visible


def test_nan_expert_count_treated_as_missing_no_exception(caplog):
    md = dict(QWEN_MD, **{"qwen35moe.expert_count": float("nan")})
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group(md, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "expert_count" in msg and "nan" in msg   # raw hparam value visible


def test_bogus_expert_used_count_shows_raw_value_and_spares_valid_expert_count(caplog):
    # expert_used_count is unparseable while expert_count parses fine: n and k
    # must convert independently (the valid one isn't discarded), and the
    # warning must show the RAW offending value, not a blanked-out None.
    md = dict(QWEN_MD, **{"qwen35moe.expert_used_count": "bogus"})
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group(md, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "bogus" in msg and "256" in msg


def test_nonsense_expert_hparams_fails_hot_not_clamped(caplog):
    md = dict(QWEN_MD, **{"qwen35moe.expert_used_count": 99, "qwen35moe.expert_count": 8})
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group(md, log=logging.getLogger("t"))
    assert w["X"] == 1.0         # out-of-range now fails hot -- never clamped to a
                                 # value derived from the bogus ratio
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "99" in msg and "8" in msg


def test_negative_expert_used_count_fails_hot_not_clamped_to_zero(caplog):
    # A negative expert_used_count previously clamped to w=0.0 (cold) -- the
    # one broken-hparam path that priced hot everywhere else. Must now fail
    # hot like every other broken-hparam case.
    md = dict(QWEN_MD, **{"qwen35moe.expert_used_count": -4})
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group(md, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "-4" in msg


def test_missing_architecture_gives_clear_warning_not_none_dot_expert_count(caplog):
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group({}, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "general.architecture missing" in msg
    assert "None.expert_count" not in msg and "None.expert_used_count" not in msg


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
    # exactly one UNKNOWN-group tensor in `tensors` above (per_layer_proj) --
    # the warning must fire exactly once and name that count.
    unknown_records = [r for r in caplog.records if "UNKNOWN" in r.getMessage()]
    assert len(unknown_records) == 1
    assert "1 UNKNOWN" in unknown_records[0].getMessage()


def test_per_tensor_override_beats_gathered_rule():
    # Rule order is override-by-group first, gathered-by-name second: an
    # explicit override for H must win even for the rule-2 MTP embedding.
    tensors = {
        "token_embd.weight": _entry("E", (100, 64)),
        "blk.0.nextn.embed_tokens.weight": _entry("H", (100, 64)),
    }
    w = bw.stream_weights(tensors, QWEN_MD, overrides={"H": 0.75})
    assert w["blk.0.nextn.embed_tokens.weight"] == 0.75
    assert w["token_embd.weight"] == 0.0        # group E, no override -> untouched

    w = bw.stream_weights(tensors, QWEN_MD, overrides={"E": 0.5})
    assert w["token_embd.weight"] == 0.5


def test_shape_sanity_warnings(caplog):
    tensors = {"blk.0.ffn_up_exps.weight": _entry("X", (64, 32)),   # 2-D X
               "blk.0.ffn_up.weight": _entry("U", (4, 64, 32))}      # 3-D non-X
    with caplog.at_level(logging.WARNING):
        bw.stream_weights(tensors, QWEN_MD, log=logging.getLogger("t"))
    msgs = [r.getMessage() for r in caplog.records]
    x_shape_msgs = [m for m in msgs if "blk.0.ffn_up_exps.weight" in m]
    other_shape_msgs = [m for m in msgs if "blk.0.ffn_up.weight" in m]
    assert x_shape_msgs and "not 3-D" in x_shape_msgs[0]
    assert other_shape_msgs and "not X" in other_shape_msgs[0]


def _table():
    return {
        "a": _entry("U", (8, 8), choices={"BF16": {"actual": "BF16", "bytes": 128, "werr": 0.0},
                                          "Q4_K_M": {"actual": "Q4_K_M", "bytes": 36, "werr": 1.0}}),
        "x": _entry("X", (2, 8, 8), choices={"BF16": {"actual": "BF16", "bytes": 256, "werr": 0.0},
                                             "Q4_K_M": {"actual": "Q4_K_M", "bytes": 72, "werr": 0.5}}),
        # werr=7.0 (not 0.0): if the fixed-tensor exclusion in pure_loss ever
        # broke, this tensor's contribution would be impossible to miss.
        "n": _entry("N", (8,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 32, "werr": 7.0}}),
    }


def test_streamed_bytes_and_pure_loss_by_hand():
    t = _table()
    w = {"a": 1.0, "x": 0.25, "n": 1.0}
    assign = {"a": "Q4_K_M", "x": "BF16", "n": "F32"}
    assert bw.streamed_bytes(assign, t, w) == 36 + int(0.25 * 256) + 32
    # "n" is fixed: its werr=7.0 must NOT contribute (2.0 not 2.0 + 7.0 == 9.0).
    assert bw.pure_loss(assign, t, {"U": 2.0, "X": 3.0}) == pytest.approx(2.0 * 1.0 + 0.0)
    assert bw.streamed_bytes({"a": "Q4_K_M"}, t, {}) == 36          # missing weight -> 1.0


def test_streamed_bytes_missing_tensor_raises_informative_keyerror():
    t = _table()
    with pytest.raises(KeyError, match=r"'missing' / 'Q4_K_M'"):
        bw.streamed_bytes({"missing": "Q4_K_M"}, t, {})


def test_streamed_bytes_missing_scheme_raises_informative_keyerror():
    t = _table()
    with pytest.raises(KeyError, match=r"'a' / 'NOPE'"):
        bw.streamed_bytes({"a": "NOPE"}, t, {})


def test_streamed_bytes_missing_bytes_key_raises_informative_keyerror():
    t = _table()
    del t["a"]["choices"]["Q4_K_M"]["bytes"]
    with pytest.raises(KeyError, match=r"'a' / 'Q4_K_M'.*no 'bytes'"):
        bw.streamed_bytes({"a": "Q4_K_M"}, t, {})


def test_pure_loss_missing_werr_key_raises_informative_keyerror():
    t = _table()
    del t["a"]["choices"]["Q4_K_M"]["werr"]
    with pytest.raises(KeyError, match=r"'a' / 'Q4_K_M'.*no 'werr'"):
        bw.pure_loss({"a": "Q4_K_M"}, t, {})


def test_pure_loss_werr_none_is_still_silently_excluded():
    # werr: null is the documented no-decode sentinel (sensitivity.py) -- it
    # must stay a silent skip, not be conflated with a missing "werr" key.
    t = _table()
    t["a"]["choices"]["Q4_K_M"]["werr"] = None
    assert bw.pure_loss({"a": "Q4_K_M"}, t, {"U": 2.0}) == 0.0


def test_bpw_by_group_missing_bytes_key_raises_informative_keyerror():
    t = _table()
    del t["x"]["choices"]["BF16"]["bytes"]
    with pytest.raises(KeyError, match=r"'x' / 'BF16'.*no 'bytes'"):
        bw.bpw_by_group({"x": "BF16"}, t)


def test_pure_loss_and_bpw_missing_group_key_defaults_to_unknown():
    t = {"z": {"shape": [2], "n_elems": 2, "fixed": False, "wnorm": None,
               "choices": {"Q": {"actual": "Q", "bytes": 4, "werr": 0.5}}}}
    assert "group" not in t["z"]
    assert bw.pure_loss({"z": "Q"}, t, {}) == pytest.approx(0.5)
    b = bw.bpw_by_group({"z": "Q"}, t)
    assert b["UNKNOWN"] == pytest.approx(4 * 8 / 2)


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
    # rule-2 tensors (MTP embedding inside H) are an intended exception: no
    # WARNING, and the modal weight must come from the non-gathered tensor
    # ("output.weight") regardless of dict insertion order -- with both
    # tensors tied 1-1 in the full population, the old implementation's
    # modal pick flipped depending on which tensor was inserted first.
    w_stream = {"output.weight": 1.0, "blk.0.nextn.embed_tokens.weight": 0.0}
    entries = {
        "output.weight": _entry("H", (100, 64)),
        "blk.0.nextn.embed_tokens.weight": _entry("H", (100, 64)),
    }
    for t in (dict(entries), dict(reversed(list(entries.items())))):
        with caplog.at_level(logging.WARNING):
            by_group, exceptions = bw.group_view(w_stream, t, log=logging.getLogger("t"))
        assert by_group["H"] == 1.0 and exceptions == {"blk.0.nextn.embed_tokens.weight": 0.0}
        assert not any("distinct weights" in r.getMessage() for r in caplog.records)


def test_group_view_tie_break_is_deterministic_not_insertion_order():
    # A genuine tie (one tensor each at two different weights, neither
    # gathered-by-name) must resolve to the higher ("hotter") weight
    # regardless of dict insertion order, and the loser must be an exception.
    entries = {"u1": _entry("U", (8,)), "u2": _entry("U", (8,))}
    w_stream = {"u1": 1.0, "u2": 0.5}
    for t in (dict(entries), dict(reversed(list(entries.items())))):
        by_group, exceptions = bw.group_view(w_stream, t)
        assert by_group["U"] == 1.0
        assert exceptions == {"u2": 0.5}


def test_group_view_all_gathered_group_still_warns_on_disagreement(caplog):
    # Every tensor in the group is gathered-by-name (rule 2), so the modal
    # falls back to the full population -- which must still be checked for
    # disagreement (not the now-empty non-gathered counter), and the pick
    # must still be order-independent.
    entries = {
        "token_embd.weight": _entry("E", (100, 64)),
        "blk.0.nextn.embed_tokens.weight": _entry("E", (100, 64)),
    }
    w_stream = {"token_embd.weight": 0.0, "blk.0.nextn.embed_tokens.weight": 1.0}
    for t in (dict(entries), dict(reversed(list(entries.items())))):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            by_group, exceptions = bw.group_view(w_stream, t, log=logging.getLogger("t"))
        assert by_group["E"] == 1.0
        assert exceptions == {"token_embd.weight": 0.0}
        assert any("distinct weights" in r.getMessage() for r in caplog.records)


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
