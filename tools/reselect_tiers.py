#!/usr/bin/env python3
"""Re-derive a finished run's tier ladder under the current tier scheme.

A ``search_results.json`` records every candidate the search actually
*measured* -- config, size and perplexity. The tier labels stored alongside
them are just a *selection* over that set, so a selection bug (v1's
overlapping size bands, loss-only within-band picking) can be corrected
after the fact from the same file: no GPU, no re-measurement.

That is the whole point of this tool. Runs finished before
``CURRENT_TIER_SCHEME_VERSION`` == 2 shipped artifacts whose tier *names*
disagree with their contents -- most visibly, v1's ``Q5`` band ``(0.33,
0.45]`` contained both uniform Q5_K (ratio 0.3441) and uniform Q6_K
(0.4102), and v1 picked within a band by loss alone, so Q6_K won every
time. Every v1 "Q5" is a Q6_K build, and the genuine Q5-sized candidate was
measured and then discarded.

Reported per run:

  * the corrected v2 ladder (per-band winner, size-aware),
  * ``MISLABELED``  -- the tier this candidate shipped under vs. what it is,
  * ``DOMINATED``   -- a strictly smaller candidate measured strictly lower
    loss, so shipping this one is never right,
  * ``IMPLAUSIBLE`` -- perplexity below baseline by more than measurement
    noise, which no quantization can achieve; runs predating the perplexity
    parser fix (a6f8dd0) could record a progress-line timing as a PPL, and
    such an entry would then *win* a band via ``min()``. Flagged entries are
    excluded from the corrected ladder entirely.
  * ``EMPTY``       -- a band with no usable candidate (nothing to ship
    without a new pack + measure).

The BF16 baseline size is not stored in ``search_results.json``, so it is
recovered by least squares: a candidate's size is linear in the per-group
parameter counts, one unknown per group plus a constant for the tensors no
scheme touches (norms/F32). With more measurements than groups the system
is overdetermined and the fit residual doubles as a self-check -- a bad fit
means the sizes are not internally consistent and the run should not be
trusted. ``--max-residual`` controls how much slop is tolerated.

Usage::

    tools/reselect_tiers.py path/to/search_results.json [more.json ...]
    tools/reselect_tiers.py --root /server/programming/Foundry/output
    tools/reselect_tiers.py --root output --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from magicquant.quant.schemes import get_scheme_by_name
from magicquant.quant.tiers import (
    CURRENT_TIER_SCHEME_VERSION,
    TIER_BOUNDARIES,
    classify_tier,
    tier_scheme_version,
)
from magicquant.utils.measurement import measurement_eps

# Bands from high to low precision, for stable report ordering.
TIER_ORDER = ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]

# A size fit worse than this (relative, max over candidates) means the
# recovered baseline is not trustworthy.
DEFAULT_MAX_RESIDUAL = 0.005


def _bpw(scheme_name: str) -> float:
    """Storage bits-per-weight for a scheme name, straight from the registry."""
    scheme = get_scheme_by_name(scheme_name)
    return float(scheme.bits_per_weight)


def _parse_key(key: str) -> Dict[str, str]:
    """``"D:Q4_K_M|E:BF16|..."`` -> ``{"D": "Q4_K_M", "E": "BF16", ...}``."""
    out: Dict[str, str] = {}
    for part in key.split("|"):
        group, _, scheme = part.partition(":")
        out[group] = scheme
    return out


class BaselineFitError(RuntimeError):
    """The per-group size model could not be recovered from the measurements."""


def recover_baseline_gb(
    candidates: List[Tuple[float, Dict[str, str]]],
    *,
    max_residual: float = DEFAULT_MAX_RESIDUAL,
) -> Tuple[float, float]:
    """Recover the BF16 size (GB) from measured (size, config) pairs.

    ``size_i = sum_g w_g * bpw(config_i[g]) + c``, where ``w_g`` is group
    ``g``'s parameter count expressed in GB-per-bit-per-weight and ``c``
    covers everything no scheme applies to. Solving for ``w`` and ``c``
    gives ``baseline = sum_g w_g * 16 + c``.

    Returns ``(baseline_gb, max_relative_residual)``.
    """
    groups = sorted({g for _, cfg in candidates for g in cfg})
    if len(candidates) < len(groups) + 1:
        raise BaselineFitError(
            f"{len(candidates)} measurements cannot determine "
            f"{len(groups)} group sizes + constant"
        )

    matrix = np.array(
        [[_bpw(cfg[g]) for g in groups] + [1.0] for _, cfg in candidates],
        dtype=float,
    )
    sizes = np.array([size for size, _ in candidates], dtype=float)

    solution, *_ = np.linalg.lstsq(matrix, sizes, rcond=None)
    predicted = matrix @ solution
    residual = float(np.max(np.abs(predicted - sizes) / np.maximum(sizes, 1e-9)))
    if residual > max_residual:
        raise BaselineFitError(
            f"size model fits poorly (max relative residual {residual:.4%} "
            f"> {max_residual:.4%}); measured sizes are not self-consistent"
        )

    baseline = float(solution[:-1].sum() * 16.0 + solution[-1])
    return baseline, residual


def pareto_front(rows: List[Dict[str, Any]]) -> None:
    """Mark rows that no smaller candidate beats on loss (in place)."""
    best = float("inf")
    for row in sorted(rows, key=lambda r: r["size_gb"]):
        if row["loss"] < best:
            best = row["loss"]
            row["pareto"] = True
        else:
            row["pareto"] = False


def analyze(path: Path, *, max_residual: float) -> Dict[str, Any]:
    """Re-derive one run's ladder. Never raises for data problems."""
    data = json.loads(path.read_text())
    measurements = data.get("measurements") or {}

    report: Dict[str, Any] = {
        "path": str(path),
        "stored_scheme_version": tier_scheme_version(data),
        "current_scheme_version": CURRENT_TIER_SCHEME_VERSION,
        "n_measurements": len(measurements),
        "error": None,
    }

    # A quantization cannot beat its own baseline by more than noise; an
    # entry that claims to is a broken measurement, not a good candidate.
    baseline_ppl = data.get("baseline_ppl")
    eps = measurement_eps(float(baseline_ppl)) if baseline_ppl else None
    report["baseline_ppl"] = baseline_ppl
    report["implausible_eps"] = eps

    candidates: List[Tuple[float, Dict[str, str]]] = []
    rows: List[Dict[str, Any]] = []
    for key, entry in measurements.items():
        size = entry.get("size_gb")
        loss = entry.get("measured_loss")
        if size is None or loss is None:
            continue
        config = entry.get("config") or _parse_key(key)
        # The size model is fit over every candidate: a bogus *perplexity*
        # says nothing about whether the file's byte count was measured
        # correctly, and dropping rows here would only weaken the fit.
        candidates.append((float(size), config))
        rows.append(
            {
                "key": key,
                "config": config,
                "size_gb": float(size),
                "loss": float(loss),
                "ppl": entry.get("ppl"),
                "implausible": eps is not None and float(loss) < -eps,
            }
        )

    if not rows:
        report["error"] = "no usable measurements"
        return report

    try:
        baseline_gb, residual = recover_baseline_gb(
            candidates, max_residual=max_residual
        )
    except (BaselineFitError, KeyError) as exc:
        report["error"] = f"baseline recovery failed: {exc}"
        return report

    report["baseline_gb"] = baseline_gb
    report["fit_residual"] = residual

    for row in rows:
        row["ratio"] = row["size_gb"] / baseline_gb
        row["tier"] = classify_tier(row["size_gb"], baseline_gb)

    usable = [r for r in rows if not r["implausible"]]
    pareto_front(usable)
    for row in rows:
        row.setdefault("pareto", False)
    rows.sort(key=lambda r: r["size_gb"])
    report["candidates"] = rows
    report["n_implausible"] = len(rows) - len(usable)

    # Corrected ladder: within a band prefer lower loss, then smaller.
    by_band: Dict[str, List[Dict[str, Any]]] = {}
    for row in usable:
        by_band.setdefault(row["tier"], []).append(row)
    corrected = {
        tier: sorted(band, key=lambda r: (r["loss"], r["size_gb"]))[0]
        for tier, band in by_band.items()
    }
    report["corrected"] = corrected
    report["empty_bands"] = [t for t in TIER_ORDER if t not in corrected]

    # What the run actually shipped, and how that reads under v2.
    shipped = data.get("tiered_survivors") or data.get("tiered") or {}
    findings: List[Dict[str, Any]] = []
    for tier, entry in shipped.items():
        size = entry.get("size_gb")
        if size is None:
            continue
        actual_tier = classify_tier(float(size), baseline_gb)
        shipped_loss = entry.get("measured_loss")
        better = [
            r
            for r in usable
            if r["size_gb"] < float(size) - 1e-9
            and r["loss"] < float(shipped_loss if shipped_loss is not None else "inf")
            - 1e-12
        ]
        flags = []
        if actual_tier != tier:
            flags.append("MISLABELED")
        if better:
            flags.append("DOMINATED")
        if eps is not None and shipped_loss is not None and float(shipped_loss) < -eps:
            flags.append("IMPLAUSIBLE")
        winner = min(better, key=lambda r: r["loss"]) if better else None
        findings.append(
            {
                "shipped_as": tier,
                "size_gb": float(size),
                "loss": entry.get("measured_loss"),
                "ppl": entry.get("ppl"),
                "actual_tier": actual_tier,
                "flags": flags,
                "dominated_by": winner["key"] if winner else None,
                "dominated_by_size_gb": winner["size_gb"] if winner else None,
                "dominated_by_loss": winner["loss"] if winner else None,
            }
        )
    report["shipped"] = findings
    return report


def _fmt(value: Optional[float], spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "  --"


def print_report(report: Dict[str, Any], *, verbose: bool) -> None:
    print("=" * 78)
    print(report["path"])
    stored = report["stored_scheme_version"]
    print(
        f"  tier scheme: stored v{stored} -> current "
        f"v{report['current_scheme_version']}"
        + ("   (STALE)" if stored < report["current_scheme_version"] else "")
    )
    if report["error"]:
        print(f"  !! {report['error']}")
        return
    print(
        f"  measurements: {report['n_measurements']}   "
        f"BF16 baseline: {report['baseline_gb']:.2f} GB   "
        f"(size-model residual {report['fit_residual']:.3%})"
    )
    if report.get("n_implausible"):
        print(
            f"  !! {report['n_implausible']} candidate(s) measured below "
            f"baseline by more than {report['implausible_eps']:.1%} "
            f"-- discarded as broken measurements"
        )

    if verbose:
        print()
        print(
            f"  {'size_gb':>9} {'ratio':>7} {'v2':>4} {'loss':>9} "
            f"{'ppl':>9}  front  config"
        )
        for row in report["candidates"]:
            mark = "  X  " if row["implausible"] else ("  *  " if row["pareto"] else "")
            print(
                f"  {row['size_gb']:9.2f} {row['ratio']:7.4f} {row['tier']:>4} "
                f"{row['loss']:9.4f} {_fmt(row['ppl'], '9.4f')}  "
                f"{mark:^5}  {row['key']}"
            )

    print()
    print("  corrected ladder (v2 bands, size-aware within band):")
    for tier in TIER_ORDER:
        row = report["corrected"].get(tier)
        if row is None:
            print(f"    {tier}: -- no measured candidate in band --")
            continue
        print(
            f"    {tier}: {row['size_gb']:8.2f} GB  loss {row['loss']:+.4f}  "
            f"ppl {_fmt(row['ppl'], '.4f')}  {row['key']}"
        )

    if report["shipped"]:
        print()
        print("  what this run shipped:")
        for item in report["shipped"]:
            flags = ("  " + " ".join(item["flags"])) if item["flags"] else "  ok"
            print(
                f"    shipped '{item['shipped_as']}': {item['size_gb']:8.2f} GB  "
                f"loss {_fmt(item['loss'], '+.4f')}  -> actually "
                f"{item['actual_tier']}{flags}"
            )
            if item["dominated_by"]:
                print(
                    f"        beaten by {item['dominated_by_size_gb']:.2f} GB @ "
                    f"loss {item['dominated_by_loss']:+.4f} "
                    f"({item['dominated_by']})"
                )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="search_results.json files")
    parser.add_argument("--root", type=Path, help="recursively find search_results.json")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument(
        "--max-residual",
        type=float,
        default=DEFAULT_MAX_RESIDUAL,
        help=f"reject a baseline fit worse than this (default {DEFAULT_MAX_RESIDUAL})",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="ladder only")
    args = parser.parse_args(argv)

    paths = list(args.paths)
    if args.root:
        paths.extend(sorted(args.root.rglob("search_results.json")))
    if not paths:
        parser.error("give at least one path or --root")

    reports = [analyze(p, max_residual=args.max_residual) for p in paths]
    for report in reports:
        print_report(report, verbose=not args.quiet)

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json}")

    stale = sum(
        1
        for r in reports
        if not r["error"] and r["stored_scheme_version"] < r["current_scheme_version"]
    )
    problems = sum(
        1 for r in reports if not r["error"] and any(s["flags"] for s in r["shipped"])
    )
    print(
        f"\n{len(reports)} run(s): {stale} on a stale tier scheme, "
        f"{problems} shipped a mislabeled or dominated tier."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
