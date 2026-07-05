"""Tests for magicquant.pareto -- the Pareto-frontier reporting tool.

MagicQuant's size-band tiers (Q4/Q5/Q6) each pick the best-quality winner
within their own band, which hides the real size/quality/(speed) tradeoff
across bands -- e.g. a Q6 tier can cost +11GB and -60% generation speed
over Q5 for only 0.6% better perplexity. These tests check that
pareto_frontier() correctly identifies the non-dominated set (excluding
dominated points, handling ties, 2- and 3-objective cases, and entries
missing optional fields), that format_pareto_report() renders a readable
table with marginal-cost lines, and that load_and_report() round-trips a
real search_results.json shape.
"""
import json

import pytest

from magicquant.pareto import (
    format_pareto_report,
    load_and_report,
    pareto_frontier,
)


def _entry(size_gb, ppl, tg_ts=None, config=None, **extra):
    entry = {
        "config": config or {"E": "BF16", "H": "BF16"},
        "ppl": ppl,
        "size_gb": size_gb,
    }
    if tg_ts is not None:
        entry["bench"] = {"tg_ts": tg_ts}
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# pareto_frontier correctness
# ---------------------------------------------------------------------------

def test_frontier_excludes_dominated_points():
    measurements = {
        "small": _entry(10.0, 7.0),   # smallest, worst ppl
        "mid": _entry(15.0, 6.5),     # better ppl, more size -- non-dominated
        "big_dominated": _entry(20.0, 6.6),  # bigger AND worse ppl than mid -- dominated
        "big_better": _entry(25.0, 6.0),  # biggest, best ppl -- non-dominated
    }
    frontier = pareto_frontier(measurements)
    keys = {item["key"] for item in frontier}
    assert keys == {"small", "mid", "big_better"}
    assert "big_dominated" not in keys


def test_frontier_sorted_by_size_ascending():
    measurements = {
        "c": _entry(30.0, 5.0),
        "a": _entry(10.0, 8.0),
        "b": _entry(20.0, 6.0),
    }
    frontier = pareto_frontier(measurements)
    sizes = [item["size_gb"] for item in frontier]
    assert sizes == sorted(sizes)
    assert [item["key"] for item in frontier] == ["a", "b", "c"]


def test_frontier_ties_both_kept():
    # Identical on every objective -- neither dominates the other (domination
    # requires strictly better on at least one axis), so both survive.
    measurements = {
        "x": _entry(10.0, 5.0, config={"E": "Q4_K_M"}),
        "y": _entry(10.0, 5.0, config={"E": "Q5_K"}),
    }
    frontier = pareto_frontier(measurements)
    assert {item["key"] for item in frontier} == {"x", "y"}


def test_frontier_single_point():
    measurements = {"only": _entry(10.0, 5.0)}
    frontier = pareto_frontier(measurements)
    assert len(frontier) == 1
    assert frontier[0]["key"] == "only"


def test_frontier_empty_measurements():
    assert pareto_frontier({}) == []


def test_frontier_three_objectives_higher_tg_is_better():
    measurements = {
        # Same size+ppl as "fast", but slower tg -- dominated on all 3 axes.
        "slow": _entry(10.0, 5.0, tg_ts=20.0),
        "fast": _entry(10.0, 5.0, tg_ts=50.0),
        # Bigger and worse ppl, but the fastest -- non-dominated (wins on tg_ts).
        "biggest_fastest": _entry(20.0, 6.0, tg_ts=80.0),
    }
    frontier = pareto_frontier(measurements, objectives=("size_gb", "ppl", "tg_ts"))
    keys = {item["key"] for item in frontier}
    assert "slow" not in keys, "slow is strictly worse than fast on all 3 axes"
    assert "fast" in keys
    assert "biggest_fastest" in keys


def test_frontier_three_objectives_skips_entries_missing_bench():
    measurements = {
        "has_bench": _entry(10.0, 5.0, tg_ts=50.0),
        "no_bench": _entry(15.0, 4.0),  # no "bench" key at all
    }
    frontier = pareto_frontier(measurements, objectives=("size_gb", "ppl", "tg_ts"))
    keys = {item["key"] for item in frontier}
    assert keys == {"has_bench"}, "entries missing a required objective must be excluded"


def test_frontier_default_two_objectives_ignores_bench():
    # With the default 2-objective frontier, bench presence/absence must not
    # affect membership at all.
    measurements = {
        "a": _entry(10.0, 5.0, tg_ts=50.0),
        "b": _entry(20.0, 4.0),  # no bench, but non-dominated on size/ppl
    }
    frontier = pareto_frontier(measurements)
    assert {item["key"] for item in frontier} == {"a", "b"}


def test_frontier_dominated_by_equal_size_better_ppl():
    measurements = {
        "worse": _entry(10.0, 7.0),
        "better_same_size": _entry(10.0, 6.0),
    }
    frontier = pareto_frontier(measurements)
    assert {item["key"] for item in frontier} == {"better_same_size"}


# ---------------------------------------------------------------------------
# format_pareto_report
# ---------------------------------------------------------------------------

def test_format_report_dominated_summary_line():
    measurements = {
        "small": _entry(10.0, 7.0),
        "mid": _entry(15.0, 6.5),
        "dominated": _entry(20.0, 6.6),
        "big": _entry(25.0, 6.0),
    }
    report = format_pareto_report(measurements)
    assert "dominated 1 of 4" in report


def test_format_report_shows_dash_for_missing_tg():
    measurements = {"a": _entry(10.0, 5.0)}
    report = format_pareto_report(measurements)
    assert "—" in report


def test_format_report_shows_tg_when_present():
    measurements = {"a": _entry(10.0, 5.0, tg_ts=42.5)}
    report = format_pareto_report(measurements)
    assert "42.5" in report


def test_format_report_includes_scheme_string():
    measurements = {"a": _entry(10.0, 5.0, config={"D": "Q4_K_M", "E": "Q6_K"})}
    report = format_pareto_report(measurements)
    assert "D:Q4_K_M|E:Q6_K" in report


def test_format_report_marginal_cost_between_adjacent_points():
    measurements = {
        "small": _entry(10.0, 8.0),
        "big": _entry(15.0, 4.0),  # +5GB for a 50% ppl drop
    }
    report = format_pareto_report(measurements)
    assert "marginal:" in report
    assert "+5.0GB" in report
    assert "-50.00% ppl" in report


def test_format_report_marginal_cost_includes_tg_delta_when_both_present():
    measurements = {
        "small": _entry(10.0, 8.0, tg_ts=50.0),
        "big": _entry(15.0, 4.0, tg_ts=20.0),
    }
    report = format_pareto_report(measurements)
    assert "-30.0tg" in report


def test_format_report_empty_measurements_does_not_crash():
    report = format_pareto_report({})
    assert "dominated 0 of 0" in report


def test_format_report_no_frontier_points_when_all_missing_objective():
    # 3-objective report where nothing has bench data -- frontier is empty,
    # not a crash.
    measurements = {"a": _entry(10.0, 5.0)}
    report = format_pareto_report(measurements, objectives=("size_gb", "ppl", "tg_ts"))
    assert "no measurements" in report
    assert "dominated 1 of 1" in report


# ---------------------------------------------------------------------------
# load_and_report
# ---------------------------------------------------------------------------

def test_load_and_report_reads_synthetic_search_results(tmp_path):
    data = {
        "baseline_ppl": 5.0,
        "measurements": {
            "small": _entry(10.0, 7.0),
            "big": _entry(20.0, 6.0),
        },
        "tiered": {},
    }
    path = tmp_path / "search_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    report = load_and_report(path)
    assert "dominated 0 of 2" in report
    assert "10.00" in report
    assert "20.00" in report


def test_load_and_report_tolerates_missing_measurements_key(tmp_path):
    path = tmp_path / "search_results.json"
    path.write_text(json.dumps({"baseline_ppl": 5.0}), encoding="utf-8")

    report = load_and_report(path)
    assert "dominated 0 of 0" in report


def test_load_and_report_tolerates_missing_bench_and_kl(tmp_path):
    data = {
        "measurements": {
            "a": {
                "config": {"E": "BF16"},
                "ppl": 5.0,
                "size_gb": 10.0,
                # no "bench", no "kl" keys at all
            },
        },
    }
    path = tmp_path / "search_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    report = load_and_report(str(path))
    assert "dominated 0 of 1" in report
