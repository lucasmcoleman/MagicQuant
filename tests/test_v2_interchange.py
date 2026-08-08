"""The interchange block is the on-disk contract letting v1 consumers (QAT,
ROCmFPX mq-hybrid, Foundry publish) read a v2 budget build. Reader and writer
live in different repos and never run in the same interpreter — pin the shape.
"""
import json

from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION
from magicquant.v2.interchange import budget_tier_key, write_interchange_block

RESULTS = {
    "algo": "v2-budget",
    "budget_gb": 12.5,
    "baseline_ppl": 6.01,
    "group_summary": {"U": "MXFP4_MOE", "Q": "Q5_K"},
    "allocation": {
        "assignment": {"blk.0.ffn_up.weight": "MXFP4_MOE",
                       "blk.0.attn_q.weight": "Q5_K"},
        "actual_types": {"blk.0.ffn_up.weight": "MXFP4",
                         "blk.0.attn_q.weight": "Q5_K"},
        "total_bytes": 13_000_000_000,
        "budget_bytes": 13_421_772_800,
    },
    "anchors": [{"tag": "budget", "actual_bytes": 13_100_000_000, "ppl": 6.11}],
}


def test_key_format_trims_trailing_zeros():
    assert budget_tier_key(100.0) == "BUDGET-100GiB"
    assert budget_tier_key(12.5) == "BUDGET-12.5GiB"


def test_creates_file_with_stamped_version(tmp_path):
    path = tmp_path / "search_results.json"
    key = write_interchange_block(path, RESULTS)
    data = json.loads(path.read_text())
    assert key == "BUDGET-12.5GiB"
    assert data["tier_scheme_version"] == CURRENT_TIER_SCHEME_VERSION
    block = data["tiered"][key]
    assert block["config"] == {"U": "MXFP4_MOE", "Q": "Q5_K"}
    assert block["tensor_config"]["blk.0.attn_q.weight"] == "Q5_K"
    assert block["predicted_bytes"] == 13_000_000_000
    assert block["actual_bytes"] == 13_100_000_000
    assert block["ppl"] == 6.11


def test_merge_preserves_existing_v1_tiers_byte_identical(tmp_path):
    path = tmp_path / "search_results.json"
    v1 = {"tier_scheme_version": 2,
          "tiered": {"Q4": {"config": {"U": "MXFP4_MOE"}, "size_gb": 14.2}}}
    path.write_text(json.dumps(v1))
    write_interchange_block(path, RESULTS)
    data = json.loads(path.read_text())
    assert data["tiered"]["Q4"] == v1["tiered"]["Q4"]     # untouched
    assert "BUDGET-12.5GiB" in data["tiered"]


def test_merge_into_legacy_file_does_not_relabel_it(tmp_path):
    """A pre-version-stamp v1 file must NOT gain a version=2 stamp — that
    would falsely relabel its old wide-band tiers as current-semantics."""
    path = tmp_path / "search_results.json"
    path.write_text(json.dumps({"tiered": {"Q5": {"config": {"U": "Q6_K"}}}}))
    write_interchange_block(path, RESULTS)
    data = json.loads(path.read_text())
    assert "tier_scheme_version" not in data
    assert data["tiered"]["Q5"]["config"] == {"U": "Q6_K"}


def test_rerun_replaces_own_key_only(tmp_path):
    path = tmp_path / "search_results.json"
    write_interchange_block(path, RESULTS)
    updated = dict(RESULTS, baseline_ppl=6.02)
    write_interchange_block(path, updated)
    data = json.loads(path.read_text())
    assert len(data["tiered"]) == 1
    assert data["tiered"]["BUDGET-12.5GiB"]["baseline_ppl"] == 6.02


def test_missing_anchor_bytes_reads_none(tmp_path):
    path = tmp_path / "search_results.json"
    write_interchange_block(path, dict(RESULTS, anchors=[]))
    data = json.loads(path.read_text())
    assert data["tiered"]["BUDGET-12.5GiB"]["actual_bytes"] is None


def test_corrupt_existing_file_warns_and_is_replaced(tmp_path, capsys):
    """A pre-existing search_results.json that isn't valid JSON is treated
    as absent (this module's documented treat-corrupt-as-absent policy):
    the corrupt content is replaced by a fresh file containing only this
    run's budget block, a warning naming the path is logged, and — because
    the corrupt file is indistinguishable from "no file" — it is treated as
    fresh and gets stamped with tier_scheme_version (same as the
    from-scratch case in test_creates_file_with_stamped_version)."""
    path = tmp_path / "search_results.json"
    path.write_text("{not json")
    key = write_interchange_block(path, RESULTS)

    out = capsys.readouterr().out
    assert "unreadable" in out
    assert str(path) in out

    data = json.loads(path.read_text())
    assert data["tier_scheme_version"] == CURRENT_TIER_SCHEME_VERSION
    assert set(data["tiered"]) == {key}
    assert data["tiered"][key]["config"] == RESULTS["group_summary"]
