"""Group amplification calibration (κ) — the measured part of v2.

Raw per-tensor distortion ε is comparable within a group's algebra but
layers at different residual-stream positions amplify output perturbation
differently. κ_g corrects the group-relative scaling:

    κ_g = ΔPPL_rel(probe_g) / Σ_{t∈g} ε(t, s_probe)

Probes reuse v1's single-group build (group at s_probe, everything else
BF16) but are chunk-capped (κ is a ratio under identical conditions) and
STRICT: a probe that fails after a retry is recorded as failed and — by
default — aborts the run with a message naming the group. It is never
silently replaced with a fabricated number (docs/redesign.md §6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from magicquant.logging import get_logger
from magicquant.v2.outcome import MeasurementOutcome, ProbeMeasurementError

log = get_logger(__name__)

# Relative-ΔPPL floor: a probe measuring at/below baseline is measurement
# noise, not "this group is free" — clamp instead of emitting κ=0, which
# would tell the allocator the group can be crushed for free.
MIN_REL_DPPL = 1e-5
# Censoring: a probe whose rel-dPPL lands below CENSOR_FRAC x the median
# clearly-measured group's rel-dPPL is treated as below the probe's
# resolution and floored there (see fit_kappa) rather than taken literally.
CENSOR_FRAC = 0.25


def group_epsilon_sums(
    table: Dict[str, Any], probe_scheme: str
) -> Dict[str, float]:
    """Σ_{t∈g} ε(t, s_probe) per group, from the distortion table."""
    sums: Dict[str, float] = {}
    for name, entry in table["tensors"].items():
        if entry.get("fixed"):
            continue
        choice = entry["choices"].get(probe_scheme)
        if choice is None or choice.get("werr") is None:
            continue
        g = entry["group"]
        sums[g] = sums.get(g, 0.0) + float(choice["werr"])
    return sums


def run_group_probes(
    llama_tools,
    source_model_path: str,
    output_dir: Path,
    groups: List[str],
    baseline_ppl: float,
    probe_scheme: str = "Q4_K_M",
    probe_chunks: Optional[int] = 24,
    imatrix: Optional[Dict[str, Any]] = None,
    allow_partial: bool = False,
    retries: int = 1,
) -> Dict[str, MeasurementOutcome]:
    """Measure ΔPPL for each group probe. Strict failure semantics.

    Results (including failures) are cached to
    ``<output_dir>/v2_probes.json`` keyed by probe conditions, so a
    re-run/resume skips already-measured groups.
    """
    from magicquant.gguf.writer import create_hybrid_gguf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = output_dir / "_v2_probes"
    probe_dir.mkdir(exist_ok=True)
    cache_path = output_dir / "v2_probes.json"

    conditions = {
        "source": str(source_model_path),
        "probe_scheme": probe_scheme,
        "chunks": probe_chunks,
        "ctx_size": getattr(llama_tools, "ctx_size", None),
    }
    cached: Dict[str, Any] = {}
    if cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("conditions") == conditions:
                cached = data.get("probes", {})
                log.info(
                    "reusing %d cached v2 probe measurement(s)", len(cached),
                    stage="calibrate",
                )
        except (OSError, ValueError):
            cached = {}

    saved_chunks = getattr(llama_tools, "ppl_chunks", None)
    outcomes: Dict[str, MeasurementOutcome] = {}
    try:
        if probe_chunks is not None:
            llama_tools.ppl_chunks = probe_chunks

        # Slice-matched baseline: probe passes are chunk-capped, and the
        # first N wikitext chunks measure systematically differently than
        # the full corpus. Comparing a capped probe against a full-corpus
        # baseline would add a constant offset to every group's rel-dPPL,
        # distorting RELATIVE kappa (small-damage groups proportionally
        # more). Measure the source model once under the SAME cap and use
        # that as the probe baseline. Cached with the probes.
        slice_baseline: Optional[float] = None
        if "__slice_baseline__" in cached and cached["__slice_baseline__"].get(
            "status"
        ) == "ok":
            slice_baseline = cached["__slice_baseline__"]["value"]
        if slice_baseline is None:
            slice_baseline = llama_tools.calculate_perplexity(
                str(source_model_path), verbose=False
            )
            if slice_baseline is None:
                raise ProbeMeasurementError(
                    "Could not measure the slice-matched probe baseline "
                    f"(source model at chunks={probe_chunks}). Refusing to "
                    "calibrate kappa against a mismatched baseline."
                )
            outcomes["__slice_baseline__"] = MeasurementOutcome.success(
                slice_baseline, kind="slice-baseline", chunks=probe_chunks,
            )
            _write_probe_cache(cache_path, conditions, outcomes, cached)
        else:
            outcomes["__slice_baseline__"] = MeasurementOutcome.success(
                slice_baseline, kind="slice-baseline", chunks=probe_chunks,
                cached=True,
            )

        for group in groups:
            if group in cached and cached[group].get("status") == "ok":
                c = cached[group]
                outcomes[group] = MeasurementOutcome(
                    status="ok", value=c["value"], attempts=c.get("attempts", 1),
                    meta=c.get("meta", {}),
                )
                continue

            probe_path = probe_dir / f"probe_{group}.gguf"
            outcome: Optional[MeasurementOutcome] = None
            attempts = 0
            last_error = "unknown"
            while attempts <= retries and outcome is None:
                attempts += 1
                try:
                    quant_config = {
                        "base": "BF16",
                        "groups": {group: probe_scheme},
                    }
                    create_hybrid_gguf(
                        output_path=str(probe_path),
                        base_model_path=str(source_model_path),
                        quant_config=quant_config,
                        verbose=False,
                        imatrix=imatrix,
                    )
                    ppl = llama_tools.calculate_perplexity(
                        str(probe_path), verbose=False
                    )
                    if ppl is None:
                        last_error = (
                            "llama-perplexity produced no parseable PPL"
                        )
                        continue
                    outcome = MeasurementOutcome.success(
                        ppl, attempts=attempts, group=group,
                        probe_scheme=probe_scheme,
                    )
                except Exception as exc:  # noqa: BLE001 — recorded, not hidden
                    last_error = f"{type(exc).__name__}: {exc}"
                    log.warning(
                        "probe build/measure failed for group %s "
                        "(attempt %d/%d): %s",
                        group, attempts, retries + 1, last_error,
                        stage="calibrate",
                    )
                finally:
                    if probe_path.exists():
                        try:
                            probe_path.unlink()
                        except OSError:
                            pass

            if outcome is None:
                outcome = MeasurementOutcome.failure(
                    last_error, attempts=attempts, group=group,
                    probe_scheme=probe_scheme,
                )
            outcomes[group] = outcome

            # Persist after every probe (atomic) so a killed run resumes.
            _write_probe_cache(cache_path, conditions, outcomes, cached)
    finally:
        llama_tools.ppl_chunks = saved_chunks

    failed = [g for g, o in outcomes.items() if not o.ok]
    if failed and not allow_partial:
        raise ProbeMeasurementError(
            f"Group probe measurement failed for {failed} after "
            f"{retries + 1} attempt(s) each. Refusing to continue with "
            "fabricated sensitivities — fix the measurement (GPU busy? "
            "llama.cpp build? corpus?), re-run to resume from "
            f"{cache_path}, or pass --allow-partial-probes to continue "
            "with imputed-median kappa for the failed group(s)."
        )
    return outcomes


def _write_probe_cache(
    cache_path: Path,
    conditions: Dict[str, Any],
    outcomes: Dict[str, MeasurementOutcome],
    prior: Dict[str, Any],
) -> None:
    import os

    merged = dict(prior)
    merged.update({g: o.to_json() for g, o in outcomes.items()})
    payload = {"conditions": conditions, "probes": merged}
    tmp = str(cache_path) + ".tmp"
    Path(tmp).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, cache_path)


def fit_kappa(
    outcomes: Dict[str, MeasurementOutcome],
    eps_sums: Dict[str, float],
    baseline_ppl: float,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """κ_g per group from probe outcomes + distortion sums.

    Returns (kappa, provenance) where provenance[g] is ``"measured"`` or
    ``"imputed-median"`` (only reachable via allow_partial). Groups with no
    admissible probe-scheme distortion (e.g. all-fixed) get κ=0 with
    provenance ``"no-allocatable-mass"``.
    """
    kappa: Dict[str, float] = {}
    provenance: Dict[str, str] = {}

    # Compare probes against the slice-matched baseline measured under the
    # probes' own chunk cap (see run_group_probes) — the full-corpus
    # baseline_ppl is only the fallback when probes ran uncapped.
    sb = outcomes.get("__slice_baseline__")
    probe_baseline = sb.value if (sb is not None and sb.ok) else baseline_ppl

    # Pass 1: raw rel-dPPL per measured group.
    raw_rel: Dict[str, float] = {}
    for g, outcome in outcomes.items():
        if g == "__slice_baseline__":
            continue
        eps = eps_sums.get(g, 0.0)
        if eps <= 0.0:
            kappa[g] = 0.0
            provenance[g] = "no-allocatable-mass"
            continue
        if outcome.ok:
            raw_rel[g] = (outcome.value - probe_baseline) / probe_baseline

    # Censoring floor: a probe that measures at/below the measurement noise
    # must NOT assert "this group is free to crush" — that's the v1
    # sensitivity-0.0 hazard reborn (it put embeddings on Q4_K_M in a real
    # run and cost +3% PPL at the budget). A group whose damage is below
    # the probe's resolution gets floored at CENSOR_FRAC of the median
    # clearly-measured rel-dPPL, tagged "measured-censored".
    positives = sorted(r for r in raw_rel.values() if r > MIN_REL_DPPL)
    censor_floor = (
        CENSOR_FRAC * positives[len(positives) // 2] if positives else MIN_REL_DPPL
    )
    measured_vals: List[float] = []
    for g, rel in raw_rel.items():
        eps = eps_sums[g]
        if rel < censor_floor:
            kappa[g] = max(rel, censor_floor, MIN_REL_DPPL) / eps
            provenance[g] = "measured-censored"
            log.info(
                "kappa for group %s censored at floor (raw rel-dPPL %.3g "
                "below probe resolution; floored to %.3g)",
                g, rel, censor_floor, stage="calibrate",
            )
        else:
            kappa[g] = rel / eps
            provenance[g] = "measured"
        measured_vals.append(kappa[g])

    median = (
        sorted(measured_vals)[len(measured_vals) // 2] if measured_vals else 1.0
    )
    for g, outcome in outcomes.items():
        if g == "__slice_baseline__" or g in kappa:
            continue
        kappa[g] = median
        provenance[g] = "imputed-median"
        log.warning(
            "kappa for group %s imputed from median of measured groups "
            "(%.3g) — probe failed and --allow-partial-probes was set",
            g, median, stage="calibrate",
        )
    return kappa, provenance


def affine_report_fit(
    points: List[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """Least-squares (a, b) mapping predicted loss -> measured rel-ΔPPL,
    for REPORTING only (monotone maps don't change the allocation argmin).
    None with <2 points."""
    if len(points) < 2:
        return None
    import numpy as np

    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)
