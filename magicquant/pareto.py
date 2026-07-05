"""Pareto-frontier reporting for MagicQuant search results.

MagicQuant's ``search_results.json`` groups candidates into size-band
"tiers" (Q4/Q5/Q6/...), and each tier's winner is picked purely on quality
within that band. That hides the real tradeoff: a real run can have its Q6
tier cost +11 GB and -60% generation speed over Q5 for only 0.6% better
perplexity -- a trade nobody would consciously choose, but the tiering
never surfaces it.

This module computes the actual Pareto frontier across the full
measurements dict (every candidate the search built and measured, not just
tier winners) so the size/quality/(optional) speed tradeoff is explicit:
every point on the frontier is a candidate no other candidate beats on
every axis at once.

Public API:
    pareto_frontier(measurements, objectives=("size_gb", "ppl")) -> list[dict]
    format_pareto_report(measurements, objectives=("size_gb", "ppl")) -> str
    load_and_report(search_results_path) -> str

Also runnable as a CLI: ``python -m magicquant.pareto <search_results.json>``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Direction of "better" per objective name. "min" = lower is better
# (size_gb, ppl, measured_loss); "max" = higher is better (throughput
# fields pulled out of a candidate's "bench" dict). Any objective name not
# listed here defaults to "min".
_MAXIMIZE = {"tg_ts", "pp_ts"}

# Objective names that live under entry["bench"][...] rather than directly
# on the measurement entry.
_BENCH_FIELDS = {"tg_ts", "pp_ts"}


def _direction(objective: str) -> str:
    return "max" if objective in _MAXIMIZE else "min"


def _extract(entry: Dict[str, Any], objective: str) -> Optional[float]:
    """Pull an objective's value out of a measurement entry.

    Tolerates missing fields (returns None) so callers can skip entries
    that don't carry a given objective (e.g. bench-derived tg_ts on a run
    that never speed-benched every candidate).
    """
    if objective in _BENCH_FIELDS:
        bench = entry.get("bench") or {}
        return bench.get(objective)
    return entry.get(objective)


def _dominates(
    a: Dict[str, float], b: Dict[str, float], objectives: Sequence[str]
) -> bool:
    """True if candidate ``a`` dominates candidate ``b``.

    Domination: ``a`` is better-or-equal to ``b`` on every objective, and
    strictly better on at least one. Direction-aware (lower-is-better vs.
    higher-is-better per objective, see ``_direction``).
    """
    strictly_better_any = False
    for objective in objectives:
        av, bv = a[objective], b[objective]
        if _direction(objective) == "min":
            if av > bv:
                return False
            if av < bv:
                strictly_better_any = True
        else:  # "max"
            if av < bv:
                return False
            if av > bv:
                strictly_better_any = True
    return strictly_better_any


def pareto_frontier(
    measurements: Dict[str, Dict[str, Any]],
    objectives: Sequence[str] = ("size_gb", "ppl"),
) -> List[Dict[str, Any]]:
    """Return the non-dominated subset of ``measurements``.

    A candidate is on the frontier if no other candidate is better-or-equal
    on ALL of ``objectives`` and strictly better on at least one. Lower is
    better for size_gb/ppl/measured_loss; higher is better for bench
    throughput fields (tg_ts, pp_ts). Unknown objective names default to
    "lower is better".

    Entries missing a value for any requested objective are excluded from
    consideration entirely (not just from domination checks) -- this is
    what lets a 3-objective (size_gb, ppl, tg_ts) frontier gracefully
    degrade on a run that didn't speed-bench every candidate: only the
    subset that has bench data participates.

    Each returned dict is a shallow copy of the original measurement entry
    with a "key" field added (the measurements dict key it came from), so
    the frontier list is self-describing once separated from the source
    dict. The list is sorted ascending by the first objective (size_gb by
    convention) for deterministic output.
    """
    candidates = []
    for key, entry in measurements.items():
        values: Dict[str, float] = {}
        complete = True
        for objective in objectives:
            v = _extract(entry, objective)
            if v is None:
                complete = False
                break
            values[objective] = v
        if not complete:
            continue
        candidates.append((key, entry, values))

    frontier = []
    for key, entry, values in candidates:
        dominated = False
        for _okey, _oentry, ovalues in candidates:
            if ovalues is values:
                continue
            if _dominates(ovalues, values, objectives):
                dominated = True
                break
        if not dominated:
            frontier.append((key, entry, values))

    sort_objective = objectives[0]
    frontier.sort(key=lambda item: item[2][sort_objective])

    result = []
    for key, entry, _values in frontier:
        item = dict(entry)
        item["key"] = key
        result.append(item)
    return result


def _scheme_str(entry: Dict[str, Any]) -> str:
    """Compact ``group:scheme`` string, groups sorted -- matches
    MagicQuantOrchestrator._config_key's format so a frontier entry's "key"
    (when present) and its recomputed scheme string always agree."""
    config = entry.get("config") or {}
    return "|".join(f"{g}:{config[g]}" for g in sorted(config))


def format_pareto_report(
    measurements: Dict[str, Dict[str, Any]],
    objectives: Sequence[str] = ("size_gb", "ppl"),
) -> str:
    """Render a readable text table of the Pareto frontier, sorted by size.

    Columns: size_gb, ppl, tg_ts (from bench, "--" if no candidate on the
    frontier has bench data), and a compact per-group scheme string.

    Below the table: a "dominated N of M" summary line, and, between each
    adjacent pair of frontier points, the marginal cost of stepping up to
    the larger one -- size and (when both sides have it) tg_ts deltas
    alongside the relative ppl change -- so the diminishing-returns shape
    of the tradeoff is explicit rather than hidden inside tier labels.
    """
    total = len(measurements)
    frontier = pareto_frontier(measurements, objectives=objectives)
    dominated = total - len(frontier)

    header = f"{'size_gb':>9}  {'ppl':>9}  {'tg_ts':>8}  scheme"
    rule = "-" * len(header)

    lines = [
        "MagicQuant Pareto Frontier (size / quality / speed tradeoff)",
        "=" * len(header),
        header,
        rule,
    ]

    if not frontier:
        lines.append("(no measurements with the required objective fields)")
        lines.append(rule)
        lines.append(f"dominated {dominated} of {total}")
        return "\n".join(lines)

    prev: Optional[Dict[str, Any]] = None
    for item in frontier:
        size_gb = item.get("size_gb")
        ppl = item.get("ppl")
        bench = item.get("bench") or {}
        tg_ts = bench.get("tg_ts")

        size_s = f"{size_gb:9.2f}" if isinstance(size_gb, (int, float)) else f"{'—':>9}"
        ppl_s = f"{ppl:9.4f}" if isinstance(ppl, (int, float)) else f"{'—':>9}"
        tg_s = f"{tg_ts:8.1f}" if isinstance(tg_ts, (int, float)) else f"{'—':>8}"
        scheme = _scheme_str(item)
        lines.append(f"{size_s}  {ppl_s}  {tg_s}  {scheme}")

        if prev is not None:
            prev_size = prev.get("size_gb")
            prev_ppl = prev.get("ppl")
            prev_bench = prev.get("bench") or {}
            prev_tg = prev_bench.get("tg_ts")

            parts = []
            if isinstance(size_gb, (int, float)) and isinstance(prev_size, (int, float)):
                parts.append(f"{size_gb - prev_size:+.1f}GB")
            if isinstance(tg_ts, (int, float)) and isinstance(prev_tg, (int, float)):
                parts.append(f"{tg_ts - prev_tg:+.1f}tg")
            cost = " / ".join(parts) if parts else "—"

            if (
                isinstance(ppl, (int, float))
                and isinstance(prev_ppl, (int, float))
                and prev_ppl
            ):
                pct = (ppl - prev_ppl) / prev_ppl * 100.0
                lines.append(f"    marginal: {cost} for {pct:+.2f}% ppl")
            else:
                lines.append(f"    marginal: {cost} for —% ppl")
        prev = item

    lines.append(rule)
    lines.append(f"dominated {dominated} of {total}")
    return "\n".join(lines)


def load_and_report(
    search_results_path: Union[str, Path],
    objectives: Sequence[str] = ("size_gb", "ppl"),
) -> str:
    """Read a ``search_results.json`` and return its Pareto frontier report.

    Tolerates a results file with no "measurements" key (prediction-only
    searches persist an empty measurements dict) -- reports zero frontier
    points rather than raising.
    """
    path = Path(search_results_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    measurements = data.get("measurements") or {}
    return format_pareto_report(measurements, objectives=objectives)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(
            "usage: python -m magicquant.pareto <search_results.json>",
            file=sys.stderr,
        )
        return 2
    print(load_and_report(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
