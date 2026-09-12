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

from magicquant.v2.allocate import Choice, Unit, allocate  # noqa: E402
from magicquant.v2 import BandwidthInfeasibleError  # noqa: E402


def _moe_table():
    """X carries ~95% of bytes at w=0.03; U and H are hot (w=1); N is fixed."""
    def ch(bf, q6, q6w, q4, q4w):
        return {"BF16": {"actual": "BF16", "bytes": bf, "werr": 0.0},
                "Q6_K": {"actual": "Q6_K", "bytes": q6, "werr": q6w},
                "Q4_K_M": {"actual": "Q4_K_M", "bytes": q4, "werr": q4w}}
    t = {}
    for i in range(4):
        t[f"blk.{i}.ffn_down_exps.weight"] = _entry("X", (8, 64, 64), choices=ch(65536, 26880, 0.02, 18432, 0.08))
        t[f"blk.{i}.attn_q.weight"] = _entry("U", (64, 64), choices=ch(8192, 3360, 0.05, 2304, 0.20))
    t["output.weight"] = _entry("H", (128, 64), choices=ch(16384, 6720, 0.05, 4608, 0.25))
    t["output_norm.weight"] = _entry("N", (64,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 256, "werr": 0.0}})
    return t


def _build_units(table, kappa, floors, w_stream, lam):
    units = []
    for name, entry in table["tensors"].items():
        if entry.get("fixed"):
            (s, c), = entry["choices"].items()
            units.append(Unit(name=name, group=entry["group"], choices=[Choice(s, c["actual"], int(c["bytes"]), 0.0)]))
            continue
        k = kappa.get(entry["group"], 1.0)
        chs = []
        for s, c in entry["choices"].items():
            loss = k * float(c["werr"])
            if lam:
                loss += lam * w_stream.get(name, 1.0) * int(c["bytes"]) / 2**30
            chs.append(Choice(s, c["actual"], int(c["bytes"]), loss))
        units.append(Unit(name=name, group=entry["group"], choices=chs))
    return units


def _solve(budget_bytes, budget_bw_bytes=None, lam_fixed=None):
    table = {"tensors": _moe_table()}
    w = bw.stream_weights(table["tensors"], {"general.architecture": "m", "m.expert_count": 100, "m.expert_used_count": 3})
    return bw.solve_bandwidth(_build_units, table, {"X": 1.0, "U": 1.0, "H": 1.0}, {}, w,
                              budget_bytes, budget_bw_bytes, lam_fixed=lam_fixed), table, w


def test_mode_off_is_lambda_zero():
    sol, table, w = _solve(budget_bytes=200_000)
    assert sol.mode == "off" and sol.lam == 0.0
    assert sol.streamed_bytes == sol.streamed_bytes_lambda0 == bw.streamed_bytes(sol.chosen.assignment, table["tensors"], w)
    assert sol.streamed_bytes_min is None and sol.budget_bw_bytes is None
    assert sol.predicted_loss_pure == pytest.approx(sol.chosen.total_loss)


def _s_min(budget_bytes=200_000):
    table = {"tensors": _moe_table()}
    w = bw.stream_weights(table["tensors"], {"general.architecture": "m", "m.expert_count": 100, "m.expert_used_count": 3})
    a = allocate(_build_units(table, {"X": 1.0, "U": 1.0, "H": 1.0}, {}, w, bw.LAM_MAX), budget_bytes)
    return bw.streamed_bytes(a.assignment, table["tensors"], w)


def test_budget_mode_moves_bits_from_experts_to_trunk():
    sol0, table, w = _solve(budget_bytes=200_000)
    target = (sol0.streamed_bytes_lambda0 + _s_min()) // 2          # strictly inside (s_min, s0)
    sol, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=target)
    assert sol.mode == "budget" and sol.lam > 0.0
    assert sol.streamed_bytes <= sol.budget_bw_bytes
    assert sol.chosen.total_bytes <= 200_000
    assert sol.streamed_bytes < sol.streamed_bytes_lambda0
    # direction: hot groups (U, H) LOSE bpw; cold X GAINS or holds
    assert sol.bpw_chosen["U"] + sol.bpw_chosen["H"] < sol.bpw_lambda0["U"] + sol.bpw_lambda0["H"]
    assert sol.bpw_chosen["X"] >= sol.bpw_lambda0["X"]
    assert sol.predicted_loss_pure < sol.total_loss_with_lambda
    assert sol.predicted_loss_pure == pytest.approx(bw.pure_loss(sol.chosen.assignment, table["tensors"], {"X": 1.0, "U": 1.0, "H": 1.0}))


def test_both_modes_is_an_error():
    with pytest.raises(ValueError):
        _solve(budget_bytes=200_000, budget_bw_bytes=60_000, lam_fixed=0.01)


def test_count_inversions():
    assert bw._count_inversions([(1e-3, 100), (1e-2, 90), (1e-1, 80)]) == 0
    assert bw._count_inversions([(1e-3, 100), (1e-2, 90), (1e-1, 95), (1.0, 80)]) == 1
    assert bw._count_inversions([]) == 0


def test_budget_already_met_returns_lambda_zero():
    sol0, _, _ = _solve(budget_bytes=200_000)
    sol, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=sol0.streamed_bytes_lambda0 * 2)
    assert sol.mode == "budget" and sol.lam == 0.0


def test_infeasible_budget_raises_and_never_returns_lam_max():
    with pytest.raises(BandwidthInfeasibleError) as ei:
        _solve(budget_bytes=200_000, budget_bw_bytes=1)
    assert ei.value.budget_bytes == 1 and ei.value.min_bytes > 1


def test_weight_mode_reports_given_lambda():
    sol, _, _ = _solve(budget_bytes=200_000, lam_fixed=0.05)
    assert sol.mode == "weight" and sol.lam == 0.05


def test_determinism():
    a, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=60_000)
    b, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=60_000)
    assert a.chosen.assignment == b.chosen.assignment and a.lam == b.lam


def test_to_json_shape():
    sol, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=60_000)
    j = sol.to_json()
    assert set(j) == {"mode", "lambda", "stream_weights", "weights_source", "streamed_bytes",
                      "streamed_bytes_lambda0", "streamed_bytes_min", "budget_bw_bytes",
                      "nonmonotone_probes", "predicted_loss_pure", "total_loss_with_lambda", "bpw_by_group"}
    assert set(j["stream_weights"]) == {"observed_by_group", "default", "exceptions"}
    assert j["stream_weights"]["default"] == 1.0
    assert set(j["bpw_by_group"]) == {"lambda0", "chosen"}


# ===========================================================================
# Integration: Task 5 wires solve_bandwidth into
# magicquant.v2.search.run_budget_search. Driven through the SAME
# characterization harness as tests/test_v2_search_characterization.py
# (its fixtures/helpers are reused directly, not re-derived) so the pipeline
# runs GPU-free and fully deterministic.
# ===========================================================================

import json  # noqa: E402

import magicquant.gguf.source as source_mod  # noqa: E402
import magicquant.v2.search as v2search  # noqa: E402
from magicquant.v2.search import V2Config, run_budget_search  # noqa: E402
from tests.test_v2_search_characterization import (  # noqa: E402
    SOURCE, _fake_table, _happy_ppl, _install_stubs, _make_cfg,
)
from tests.test_writer import StubSource  # noqa: E402


def test_run_budget_search_off_mode_is_byte_identical_shape(tmp_path, monkeypatch):
    """No bandwidth flags -> mode "off" and the pre-existing 0.07 loss pin
    survives untouched (results["allocation"]["predicted_loss"] is now
    routed through bandwidth.predicted_loss_pure, but at lambda=0 that is
    numerically the same quantity as the old chosen.total_loss)."""
    cfg = _make_cfg(tmp_path)
    _install_stubs(monkeypatch, _happy_ppl)

    results = run_budget_search(cfg)

    assert results["bandwidth"]["mode"] == "off"
    assert results["bandwidth"]["lambda"] == 0.0
    assert results["allocation"]["predicted_loss"] == pytest.approx(0.07)


def test_run_budget_search_wires_budget_mode_and_pure_loss(tmp_path, monkeypatch):
    # First, an unconstrained (bandwidth-off) run to learn this fixture's
    # lambda=0 streamed-bytes figure -- never hand-derived/hard-coded.
    cfg0 = _make_cfg(tmp_path / "off", anchors=1)
    _install_stubs(monkeypatch, _happy_ppl)
    results0 = run_budget_search(cfg0)
    assert results0["bandwidth"]["mode"] == "off"
    s0 = results0["bandwidth"]["streamed_bytes"]

    # Now a real streamed-bytes budget, strictly below s0, wired through a
    # genuine (non-fabricated) metadata source -- StubSource with real MoE
    # hparams -- so w_stream is derived from actual expert_count/
    # expert_used_count rather than the default _FakeSource's bare
    # "general.architecture": "llama".
    monkeypatch.setattr(
        source_mod, "open_model_source",
        lambda path, *a, **kw: StubSource([], metadata=QWEN_MD),
    )
    cfg = _make_cfg(tmp_path / "budget", anchors=1, budget_bw_gb=(s0 - 1) / 1024**3)
    results = run_budget_search(cfg)

    assert results["bandwidth"]["mode"] == "budget"
    assert results["bandwidth"]["weights_source"]["arch"] == "qwen35moe"
    assert results["allocation"]["predicted_loss"] == results["bandwidth"]["predicted_loss_pure"]
    for a in results["anchors"]:
        assert "streamed_bytes" in a

    frontier = json.loads((tmp_path / "budget" / "frontier.json").read_text())
    assert frontier["lambda"] == results["bandwidth"]["lambda"]

    table = _fake_table(cfg.schemes)
    recomputed = bw.pure_loss(
        results["allocation"]["assignment"], table["tensors"], results["kappa"]
    )
    assert results["allocation"]["predicted_loss"] == pytest.approx(recomputed)
    assert recomputed < results["bandwidth"]["total_loss_with_lambda"]


def test_tiny_bandwidth_weight_leaves_assignment_and_report_fit_unchanged(tmp_path, monkeypatch):
    """A lambda small enough to never flip a single allocation decision
    (invariant (h)): the assignment and the reporting-fit calibration must
    come out identical to the lambda=0 run."""
    _install_stubs(monkeypatch, _happy_ppl)

    cfg0 = _make_cfg(tmp_path / "lam0")
    results0 = run_budget_search(cfg0)

    cfg_tiny = _make_cfg(tmp_path / "tiny", bandwidth_weight=1e-9)
    results_tiny = run_budget_search(cfg_tiny)

    assert results_tiny["bandwidth"]["mode"] == "weight"
    assert results_tiny["allocation"]["assignment"] == results0["allocation"]["assignment"]
    assert results_tiny["report_fit_affine"] == results0["report_fit_affine"]


def test_unreadable_source_metadata_fails_soft_and_prices_hot(tmp_path, monkeypatch):
    # _moe_table (defined above) is X/U/H/N only -- no E or V groups, whose
    # stream weight is hard-pinned to 0.0 regardless of metadata (see
    # stream_weights_by_group). With this shape, EVERY group's
    # fallback-on-metadata-failure weight is genuinely 1.0, so "the run
    # prices everything hot when the source is unreadable" is actually
    # exercised end to end, not accidentally true for an unrelated reason
    # (the characterization suite's own fixture has a gathered-by-name
    # group-E tensor that is 0.0 independent of metadata, which would make
    # this assertion true for the wrong reason).
    ppl = lambda path: 10.0 if path == SOURCE else 10.1  # noqa: E731

    def _fake_compute_distortion_table(*a, **kw):
        return {"tensors": _moe_table(), "meta": {"version": 1, "schemes": ["BF16", "Q6_K", "Q4_K_M"]}}

    _install_stubs(monkeypatch, ppl)
    monkeypatch.setattr(v2search, "compute_distortion_table", _fake_compute_distortion_table)

    def _raise(path, *a, **kw):
        raise ValueError("unreadable source")

    monkeypatch.setattr(source_mod, "open_model_source", _raise)

    cfg = V2Config(
        source_model_path=SOURCE, output_dir=str(tmp_path), budget_gb=5_000_000 / 1024**3,
        schemes=["BF16", "Q6_K", "Q4_K_M"], use_imatrix=False, group_probes=False, anchors=1,
    )

    results = run_budget_search(cfg)

    assert "error" in results["bandwidth"]["weights_source"]
    observed = results["bandwidth"]["stream_weights"]["observed_by_group"]
    assert observed and all(v == 1.0 for v in observed.values())
