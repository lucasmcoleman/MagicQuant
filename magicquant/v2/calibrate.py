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


def _drop_imatrix(conditions: Dict[str, Any]) -> Dict[str, Any]:
    """``conditions`` minus the ``imatrix`` key -- used to detect the
    narrow case where imatrix identity is the ONLY thing that changed, so
    the imatrix-independent ``__slice_baseline__`` entry can still be
    reused (see ``run_group_probes``)."""
    return {k: v for k, v in conditions.items() if k != "imatrix"}


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
    probe_mode: str = "single",
    keep_scheme: str = "BF16",
) -> Dict[str, MeasurementOutcome]:
    """Measure per-group probes for κ calibration. Strict failure semantics.

    ``probe_mode``:
      - ``"single"`` (default): probe G = ``{G: probe_scheme, rest: BF16}``,
        i.e. damage G alone against an otherwise-pristine model. κ is the
        PPL damage per unit of G's distortion. Underestimates layers whose
        error compounds downstream (see docs/redesign.md §10).
      - ``"cumulative"``: base = every allocatable group at ``probe_scheme``;
        probe G = base but ``{G: keep_scheme}`` (G held HIGH). κ is the PPL
        RECOVERED by keeping G high in a heavily-quantized context — G's
        marginal importance where the allocator actually operates. Adds one
        base-aggressive measurement (stored as ``__base_aggressive__``).

    Results (including failures) are cached to
    ``<output_dir>/v2_probes.json`` keyed by probe conditions (mode
    included), so a re-run/resume skips already-measured groups.
    """
    from magicquant.gguf.writer import create_hybrid_gguf
    # Function-local import: matches this file's existing convention (see
    # create_hybrid_gguf above, whose test monkeypatch seam depends on it).
    from magicquant.v2.sensitivity import _imatrix_identity

    if probe_mode not in ("single", "cumulative"):
        raise ValueError(
            f"probe_mode must be 'single' or 'cumulative', got {probe_mode!r}"
        )

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
        "probe_mode": probe_mode,
        "keep_scheme": keep_scheme,
        # Probe GGUFs are built with `imatrix=imatrix` (create_hybrid_gguf,
        # below), so a changed imatrix identity changes the measured PPL the
        # same way it changes compute_distortion_table's cache key
        # (sensitivity.py). Reused rather than duplicated so the two caches
        # can never define "same imatrix" differently.
        "imatrix": _imatrix_identity(imatrix),
    }

    def _probe_config(group: str) -> Dict[str, Any]:
        """Per-group quant_config for this probe mode."""
        if probe_mode == "single":
            # Damage only this group; everything else pristine.
            return {"base": keep_scheme, "groups": {group: probe_scheme}}
        # cumulative: everything aggressive, this group held HIGH.
        return {"base": probe_scheme, "groups": {group: keep_scheme}}
    cached: Dict[str, Any] = {}
    if cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            stored_conditions = data.get("conditions")
            if stored_conditions == conditions:
                cached = data.get("probes", {})
                log.info(
                    "reusing %d cached v2 probe measurement(s)", len(cached),
                    stage="calibrate",
                )
            elif isinstance(stored_conditions, dict) and _drop_imatrix(
                stored_conditions
            ) == _drop_imatrix(conditions):
                # Only the imatrix fingerprint changed (or a pre-imatrix-key
                # legacy cache is being upgraded). __slice_baseline__ is
                # measured directly on the unquantized source model (below)
                # and does not depend on imatrix, so it is still valid --
                # reuse just that one entry. Every imatrix-sensitive probe
                # (built via create_hybrid_gguf(..., imatrix=imatrix)) is
                # left uncached and will re-measure under the new imatrix.
                probes = data.get("probes", {})
                if "__slice_baseline__" in probes:
                    cached = {"__slice_baseline__": probes["__slice_baseline__"]}
                log.info(
                    "v2 probe cache: imatrix fingerprint %s -- "
                    "re-measuring group probes (reusing cached "
                    "slice-matched baseline, which does not depend on "
                    "imatrix)",
                    "changed" if "imatrix" in stored_conditions
                    else "absent (cache predates the imatrix key)",
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

        # Cumulative mode needs the all-aggressive base PPL (the reference
        # each leave-one-group-high probe is compared against). Measured once,
        # cached like the other probes.
        if probe_mode == "cumulative":
            base_ppl: Optional[float] = None
            if "__base_aggressive__" in cached and cached[
                "__base_aggressive__"
            ].get("status") == "ok":
                base_ppl = cached["__base_aggressive__"]["value"]
                outcomes["__base_aggressive__"] = MeasurementOutcome.success(
                    base_ppl, kind="base-aggressive", cached=True,
                )
            else:
                base_path = probe_dir / "probe__base_aggressive.gguf"
                b_outcome = _measure_probe(
                    llama_tools=llama_tools,
                    create_hybrid_gguf=create_hybrid_gguf,
                    source_model_path=source_model_path,
                    out_path=base_path,
                    quant_config={"base": probe_scheme, "groups": {}},
                    imatrix=imatrix,
                    retries=retries,
                    log_template="base-aggressive probe failed (attempt %d/%d): %s",
                    log_prefix_args=(),
                    meta={"kind": "base-aggressive"},
                )
                if not b_outcome.ok:
                    # Without the base, no cumulative κ can be computed at
                    # all — fail loudly (unless partial explicitly allowed,
                    # in which case fit_kappa falls back to imputed median).
                    if not allow_partial:
                        outcomes["__base_aggressive__"] = b_outcome
                        _write_probe_cache(cache_path, conditions, outcomes, cached)
                        raise ProbeMeasurementError(
                            "cumulative probe base-aggressive measurement "
                            f"failed: {b_outcome.error}. Every leave-one-group κ is "
                            "measured against this base; refusing to continue. "
                            "Fix the measurement and re-run to resume, or pass "
                            "--allow-partial-probes."
                        )
                outcomes["__base_aggressive__"] = b_outcome
                _write_probe_cache(cache_path, conditions, outcomes, cached)

        for group in groups:
            if group in cached and cached[group].get("status") == "ok":
                c = cached[group]
                outcomes[group] = MeasurementOutcome(
                    status="ok", value=c["value"], attempts=c.get("attempts", 1),
                    meta=c.get("meta", {}),
                )
                continue

            probe_path = probe_dir / f"probe_{group}.gguf"
            # _probe_config is a pure dict builder over an already-validated
            # probe_mode (checked at function entry) and cannot raise, so
            # evaluating it here rather than inside _measure_probe's retry
            # loop is behavior-preserving.
            quant_config = _probe_config(group)
            outcome = _measure_probe(
                llama_tools=llama_tools,
                create_hybrid_gguf=create_hybrid_gguf,
                source_model_path=source_model_path,
                out_path=probe_path,
                quant_config=quant_config,
                imatrix=imatrix,
                retries=retries,
                log_template="probe build/measure failed for group %s "
                "(attempt %d/%d): %s",
                log_prefix_args=(group,),
                meta={"group": group, "probe_scheme": probe_scheme},
            )
            outcomes[group] = outcome

            # Persist after every probe (atomic) so a killed run resumes.
            _write_probe_cache(cache_path, conditions, outcomes, cached)
    finally:
        llama_tools.ppl_chunks = saved_chunks

    failed = [
        g for g, o in outcomes.items()
        if not o.ok and g not in ("__slice_baseline__", "__base_aggressive__")
    ]
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


def _measure_probe(
    llama_tools,
    create_hybrid_gguf,
    source_model_path: str,
    out_path: Path,
    # Reused as the same dict object across every retry attempt below --
    # safe only while create_hybrid_gguf never mutates its quant_config/
    # group_schemes/tensor_overrides arguments.
    quant_config: Dict[str, Any],
    imatrix: Optional[Dict[str, Any]],
    retries: int,
    log_template: str,
    log_prefix_args: Tuple[Any, ...],
    meta: Dict[str, Any],
) -> MeasurementOutcome:
    """Build one probe GGUF and measure its PPL, retrying up to ``retries``
    times after the first attempt. Shared build/measure/retry/cleanup core
    of both the base-aggressive probe and each per-group probe in
    ``run_group_probes`` — the only things that differ between call sites
    are the quant_config, the log message, and the outcome ``meta``.

    ``create_hybrid_gguf`` is passed in rather than imported here so that
    ``run_group_probes``'s function-local import (relied on by a test that
    monkeypatches ``magicquant.gguf.writer.create_hybrid_gguf``) stays the
    single source of that binding.

    Returns the final ``MeasurementOutcome`` (success or failure) and never
    raises or decides what a failure means for the caller — the
    base-aggressive-vs-per-group asymmetry (abort immediately vs. record
    and defer to the aggregate check) stays entirely at the call sites.
    """
    outcome: Optional[MeasurementOutcome] = None
    attempts = 0
    last_error = "unknown"
    while attempts <= retries and outcome is None:
        attempts += 1
        try:
            create_hybrid_gguf(
                output_path=str(out_path),
                base_model_path=str(source_model_path),
                quant_config=quant_config,
                verbose=False,
                imatrix=imatrix,
            )
            ppl = llama_tools.calculate_perplexity(str(out_path), verbose=False)
            if ppl is None:
                last_error = "llama-perplexity produced no parseable PPL"
                continue
            outcome = MeasurementOutcome.success(ppl, attempts=attempts, **meta)
        except Exception as exc:  # noqa: BLE001 — recorded, not hidden
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                log_template, *log_prefix_args, attempts, retries + 1,
                last_error, stage="calibrate",
            )
        finally:
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
    if outcome is None:
        outcome = MeasurementOutcome.failure(last_error, attempts=attempts, **meta)
    return outcome


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

    Auto-detects the probe mode from ``outcomes``:
      - ``single``: raw rel-dPPL = ``(PPL_G − slice_baseline) / slice_baseline``
        — the damage of quantizing G alone.
      - ``cumulative`` (``__base_aggressive__`` present): raw rel-dPPL =
        ``(PPL_base − PPL_leave_G) / slice_baseline`` — the PPL RECOVERED by
        keeping G high in a fully-quantized context (its marginal importance).
    Both are (relative PPL ÷ distortion), so everything downstream
    (censoring, allocator) is identical.

    Returns (kappa, provenance) where provenance[g] is ``"measured"``,
    ``"measured-censored"``, ``"imputed-median"`` (only via allow_partial),
    or ``"no-allocatable-mass"`` (group with no admissible distortion).
    Pseudo-keys (``__slice_baseline__``, ``__base_aggressive__``) are never
    emitted as groups.
    """
    kappa: Dict[str, float] = {}
    provenance: Dict[str, str] = {}
    _pseudo = {"__slice_baseline__", "__base_aggressive__"}

    # Compare probes against the slice-matched baseline measured under the
    # probes' own chunk cap (see run_group_probes) — the full-corpus
    # baseline_ppl is only the fallback when probes ran uncapped.
    sb = outcomes.get("__slice_baseline__")
    probe_baseline = sb.value if (sb is not None and sb.ok) else baseline_ppl

    # Cumulative mode: rel-dPPL is recovery from the all-aggressive base.
    ba = outcomes.get("__base_aggressive__")
    cumulative = ba is not None and ba.ok
    base_ppl = ba.value if cumulative else None

    # Pass 1: raw rel-dPPL per measured group.
    raw_rel: Dict[str, float] = {}
    for g, outcome in outcomes.items():
        if g in _pseudo:
            continue
        eps = eps_sums.get(g, 0.0)
        if eps <= 0.0:
            kappa[g] = 0.0
            provenance[g] = "no-allocatable-mass"
            continue
        if outcome.ok:
            if cumulative:
                # Recovery from keeping G high in the all-quantized base.
                raw_rel[g] = (base_ppl - outcome.value) / probe_baseline
            else:
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
            # `rel` is provably < the other two args under this branch
            # guard (which also excludes NaN, since `nan < censor_floor`
            # is False), so it can never win the max and is safely omitted.
            kappa[g] = max(censor_floor, MIN_REL_DPPL) / eps
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
        if g in _pseudo or g in kappa:
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
