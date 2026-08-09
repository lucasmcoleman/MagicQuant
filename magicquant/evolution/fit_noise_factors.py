"""
Fit per-scheme noise factors from real measured search_results.json data.

Lives in the package (not ``tools/``) so it can be imported at runtime by
``magicquant.orchestrator`` regardless of how ``magicquant`` was installed --
``tools/`` is a bare top-level package with no guaranteed presence outside a
git checkout (2026-08 packaging fix, F4). ``tools/fit_noise_factors.py`` is
now a thin shim that re-exports everything below so
``python tools/fit_noise_factors.py ...`` keeps working for hand-runs from a
checkout; this module is the single source of truth.

Unlike `tools/calibrate_noise_factors.py` (which builds and measures a
UNIFORM GGUF per scheme, one scheme at a time -- clean but ~2 hr of compute
and a dedicated calibration model+corpus), this tool fits noise factors
directly from whatever HYBRID (mixed-scheme) measured configs a real
MagicQuant measured search already produced. Useful whenever a full
per-scheme calibration run hasn't been done, but a `search_results.json`
from `run_measured_search` already exists.

Formula (mirrors ``PredictiveScorer.predict_loss`` in
``magicquant/evolution/predictor.py`` -- read that first if editing this):

    measured_loss ≈ sum_g( sensitivity_weight[g] * noise_factor[scheme[g]] )
                   + collapse_penalty_beta * n_compressed_sensitive_groups

The collapse penalty (compressing >=1 of E/H/O/R) uses the predictor's own
default ``collapse_penalty_beta`` -- it is NOT fitted here, since a handful
of hybrid configs can't reliably disentangle a fixed per-group-count penalty
from per-scheme noise contributions.

This reduces to a linear system ``A @ x = b`` across all measured configs,
where each column of ``A`` is a distinct scheme name (BF16 is fixed at 0.0,
matching the registry, and excluded from the unknowns) and ``b`` is
``measured_loss - collapse_term``. We solve it in the least-squares sense
with ``numpy.linalg.lstsq``. With few, heterogeneous real configs this is
typically UNDER-determined (more distinct schemes than measured configs);
`numpy.linalg.lstsq` returns the minimum-norm solution in that case, and
this tool reports each unknown's number of supporting observations so a
caller can judge how much to trust each fitted value.

A sibling ``sensitivity.json`` (same directory as each ``search_results
.json``) supplies the per-run group sensitivity weights
(``normalized_weights``) -- the exact values ``PredictiveScorer`` was
constructed with for that search (see ``orchestrator.py``'s
``run_measured_search``/``run_full_search``). A ``search_results.json``
without a sibling ``sensitivity.json`` is skipped with a warning: its
group-weighting is unknown, so its configs can't be fit consistently with
the predictor's formula.

Output shape matches what `magicquant/quant/calibration.py` reads (the
F2-fixed nested-envelope loader): ``{"schemes": {name: {"noise_factor":
...}}}``. This is opt-in, like the existing calibration mechanism -- running
this tool and writing its output to `tools/calibration_results.json` is a
separate, deliberate step; it never changes `schemes.py`'s registry
defaults or the seed42 regression fixture.

Usage:
    python tools/fit_noise_factors.py path/to/search_results.json [more.json ...]
    python tools/fit_noise_factors.py --output tools/calibration_results.json \\
        output/*/magicquant/search_results.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from magicquant.quant.schemes import get_scheme_by_name
from magicquant.evolution.predictor import (
    DEFAULT_COLLAPSE_PENALTY_BETA,
    PredictiveScorer,
)

HIGH_SENSITIVITY_GROUPS = PredictiveScorer.HIGH_SENSITIVITY_GROUPS
BASELINE_SCHEME = "BF16"  # fixed at noise_factor=0.0; never a fitted unknown


class FitInput:
    """One (config, measured_loss, sensitivity_weights) observation."""

    __slots__ = ("config", "measured_loss", "sensitivity_weights", "source")

    def __init__(self, config: Dict[str, str], measured_loss: float,
                 sensitivity_weights: Dict[str, float], source: str):
        self.config = config
        self.measured_loss = measured_loss
        self.sensitivity_weights = sensitivity_weights
        self.source = source


def _sensitivity_weights_for(search_results_path: Path) -> Optional[Dict[str, float]]:
    """Read the sibling sensitivity.json's normalized_weights, if present."""
    sibling = search_results_path.parent / "sensitivity.json"
    if not sibling.exists():
        return None
    try:
        data = json.loads(sibling.read_text())
    except (OSError, ValueError):
        return None
    weights = data.get("normalized_weights")
    if not isinstance(weights, dict):
        return None
    return {k: float(v) for k, v in weights.items()}


def load_fit_inputs(search_results_paths: List[Path]) -> Tuple[List[FitInput], List[str]]:
    """Load measured (config, measured_loss) pairs from one or more
    search_results.json files, paired with their sibling sensitivity
    weights. Returns (inputs, warnings)."""
    inputs: List[FitInput] = []
    warnings: List[str] = []

    for path in search_results_paths:
        path = Path(path)
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            warnings.append(f"{path}: could not read/parse ({exc}) -- skipped")
            continue

        weights = _sensitivity_weights_for(path)
        if weights is None:
            warnings.append(
                f"{path}: no sibling sensitivity.json with normalized_weights "
                "-- skipped (can't reproduce the predictor's group-weighting)"
            )
            continue

        measurements = data.get("measurements", {})
        if not isinstance(measurements, dict) or not measurements:
            warnings.append(f"{path}: no 'measurements' -- skipped")
            continue

        for entry in measurements.values():
            config = entry.get("config")
            measured_loss = entry.get("measured_loss")
            if not isinstance(config, dict) or not isinstance(measured_loss, (int, float)):
                continue
            inputs.append(FitInput(
                config=config, measured_loss=float(measured_loss),
                sensitivity_weights=weights, source=str(path),
            ))

    return inputs, warnings


def _collapse_term(config: Dict[str, str], collapse_penalty_beta: float) -> float:
    compressed_sensitive = sum(
        1 for g in config
        if g in HIGH_SENSITIVITY_GROUPS and config[g] != BASELINE_SCHEME
    )
    return collapse_penalty_beta * compressed_sensitive if compressed_sensitive > 0 else 0.0


def fit_noise_factors(
    inputs: List[FitInput],
    collapse_penalty_beta: float = DEFAULT_COLLAPSE_PENALTY_BETA,
) -> Dict[str, Dict[str, float]]:
    """Least-squares fit per-scheme noise factors from measured configs.

    Returns ``{scheme_name: {"noise_factor": float, "n_observations": int}}``
    for every non-BF16 scheme that appears in at least one input config.
    Empty dict if `inputs` is empty.
    """
    if not inputs:
        return {}

    schemes = sorted({
        scheme for fi in inputs for scheme in fi.config.values()
        if scheme != BASELINE_SCHEME
    })
    if not schemes:
        return {}
    col_index = {name: i for i, name in enumerate(schemes)}

    n_rows = len(inputs)
    n_cols = len(schemes)
    A = np.zeros((n_rows, n_cols), dtype=np.float64)
    b = np.zeros(n_rows, dtype=np.float64)
    n_obs = {name: 0 for name in schemes}

    for row, fi in enumerate(inputs):
        for group, scheme in fi.config.items():
            if scheme == BASELINE_SCHEME:
                continue
            sens_weight = fi.sensitivity_weights.get(
                group, 1.0 / max(len(fi.config), 1)
            )
            col = col_index[scheme]
            A[row, col] += sens_weight
            n_obs[scheme] += 1
        b[row] = fi.measured_loss - _collapse_term(fi.config, collapse_penalty_beta)

    solution, _residuals, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)

    return {
        name: {
            "noise_factor": round(max(0.0, float(solution[col_index[name]])), 4),
            "n_observations": n_obs[name],
        }
        for name in schemes
    }


def build_calibration_envelope(
    fitted: Dict[str, Dict[str, float]], sources: List[str],
) -> Dict:
    """Wrap fitted noise factors in the nested envelope
    ``magicquant/quant/calibration.py`` reads (the F2-fixed loader shape)."""
    schemes: Dict[str, Dict] = {
        name: {**info, "status": "fitted"} for name, info in fitted.items()
    }
    schemes[BASELINE_SCHEME] = {"noise_factor": 0.0, "status": "baseline"}
    return {
        "method": "least_squares_fit",
        "tool": "magicquant/evolution/fit_noise_factors.py",
        "sources": sources,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "schemes": schemes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("search_results", nargs="+", type=Path,
                     help="one or more search_results.json files")
    ap.add_argument("--output", type=Path, default=None,
                     help="write the calibration envelope here (default: print only)")
    args = ap.parse_args()

    inputs, warnings = load_fit_inputs(args.search_results)
    for w in warnings:
        print(f"[warn] {w}")

    if not inputs:
        print("No usable measured configs found -- nothing to fit.")
        return 1

    fitted = fit_noise_factors(inputs)
    envelope = build_calibration_envelope(fitted, [str(p) for p in args.search_results])

    print(f"\nFitted from {len(inputs)} measured config(s) across "
          f"{len({fi.source for fi in inputs})} file(s):")
    for name, info in sorted(fitted.items(), key=lambda kv: kv[1]["noise_factor"]):
        try:
            registry_value = get_scheme_by_name(name).noise_factor
        except ValueError:
            registry_value = None
        print(f"  {name:10s}  fitted={info['noise_factor']:7.3f}  "
              f"registry={registry_value}  n_obs={info['n_observations']}")

    if args.output:
        args.output.write_text(json.dumps(envelope, indent=2))
        print(f"\n[write] fitted noise factors -> {args.output}")
    else:
        print("\n(pass --output tools/calibration_results.json to write this out)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
