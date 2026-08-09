"""v2 budget search — compute → allocate → verify (docs/redesign.md §2).

GPU passes: 1 baseline + K anchor verifications (full corpus) + optional
chunk-capped group probes. Everything else is CPU.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from magicquant.logging import get_logger
from magicquant.v2.allocate import Allocation, Choice, Unit, allocate
from magicquant.v2.calibrate import (
    affine_report_fit,
    fit_kappa,
    group_epsilon_sums,
    run_group_probes,
)
from magicquant.v2.outcome import MeasurementOutcome
from magicquant.v2.sensitivity import compute_distortion_table

log = get_logger(__name__)

# Default allocation choice set: every scheme here must be losslessly
# supported by stock llama.cpp. Fork/IQ/legacy-Q4 families are appended by
# capability flags below.
DEFAULT_SCHEMES = [
    "BF16", "Q8_0", "Q6_K", "Q5_K", "Q4_K_M", "IQ4_NL", "MXFP4_MOE",
    "Q3_K", "Q2_K",
]
ROCMFPX_SCHEMES = ["ROCMFP8", "ROCMFP6", "ROCMFP4", "ROCMFP3"]
# Q4NX target profile: FLM_Q4NX_Converter packs GGUFs whose quantized
# tensors are Q4_0/Q4_1/Q8_0/MXFP4-family — restrict choices so the emitted
# hybrid converts losslessly for NPU serving (docs/redesign.md §7).
Q4NX_PROFILE_SCHEMES = ["BF16", "Q8_0", "Q4_0", "Q4_1", "MXFP4_MOE"]


@dataclass
class V2Config:
    source_model_path: str
    output_dir: str
    budget_gb: float
    llamacpp_path: Optional[str] = None
    data_file: Optional[str] = None            # perplexity corpus (None = auto-resolve)
    schemes: Optional[List[str]] = None       # None -> capability defaults
    enable_rocmfpx: bool = False
    target_profile: Optional[str] = None       # None | "q4nx"
    use_imatrix: bool = True
    imatrix_corpus: Optional[str] = None
    group_probes: bool = True
    probe_scheme: str = "Q4_K_M"
    probe_chunks: Optional[int] = 24
    # "single" (default, back-compat) or "cumulative" (leave-one-group-high;
    # measures marginal importance in a quantized context — docs/redesign.md
    # §10, the fix for single-group probes underweighting compounding layers
    # like embeddings).
    probe_mode: str = "single"
    allow_partial_probes: bool = False
    anchors: int = 2                            # frontier points to verify
    anchor_spread: float = 0.07                 # ±fraction of budget for neighbors
    measurement_chunks: Optional[int] = None    # None = full corpus for anchors
    sample_rows: Optional[int] = None
    floors: Dict[str, str] = field(default_factory=dict)  # group -> min scheme
    keep_anchors: bool = False
    model_name: Optional[str] = None


def _select_schemes(cfg: V2Config, imatrix_active: bool) -> List[str]:
    from magicquant.quant.schemes import get_scheme_by_name

    if cfg.schemes is not None:
        names = list(cfg.schemes)
    elif cfg.target_profile == "q4nx":
        names = list(Q4NX_PROFILE_SCHEMES)
    else:
        names = list(DEFAULT_SCHEMES)
        if cfg.enable_rocmfpx:
            names += ROCMFPX_SCHEMES

    kept: List[str] = []
    for n in names:
        try:
            s = get_scheme_by_name(n)
        except ValueError:
            log.warning("unknown scheme %s dropped from choice set", n)
            continue
        if getattr(s, "requires_imatrix", False) and not imatrix_active:
            log.info("scheme %s requires an imatrix — dropped (none active)", n)
            continue
        kept.append(n)
    return kept


def _build_units(
    table: Dict[str, Any],
    kappa: Dict[str, float],
    floors: Dict[str, str],
) -> List[Unit]:
    from magicquant.quant.schemes import get_scheme_by_name

    def _floor_bpw(group: str) -> Optional[float]:
        s = floors.get(group)
        if s is None:
            return None
        return get_scheme_by_name(s).bits_per_weight

    units: List[Unit] = []
    for name, entry in table["tensors"].items():
        group = entry["group"]
        choices: List[Choice] = []
        if entry.get("fixed"):
            (scheme, c), = entry["choices"].items()
            if c.get("bytes") is None:
                continue  # source-passthrough of unknown size: not allocatable
            choices = [Choice(scheme, c["actual"], int(c["bytes"]), 0.0)]
        else:
            k = kappa.get(group, 1.0)
            fb = _floor_bpw(group)
            for scheme, c in entry["choices"].items():
                if c.get("werr") is None:
                    continue  # no-decode or failed computation: excluded loudly
                if fb is not None:
                    try:
                        if get_scheme_by_name(scheme).bits_per_weight < fb:
                            continue
                    except ValueError:
                        pass
                choices.append(
                    Choice(scheme, c["actual"], int(c["bytes"]),
                           k * float(c["werr"]))
                )
        if choices:
            units.append(Unit(name=name, group=group, choices=choices))
    return units


def _dominant_group_schemes(
    table: Dict[str, Any], assignment: Dict[str, str]
) -> Dict[str, str]:
    """Per-group majority scheme (by parameter mass) — a v1-compatible
    summary of the per-tensor allocation for downstream consumers."""
    mass: Dict[str, Dict[str, int]] = {}
    for name, scheme in assignment.items():
        entry = table["tensors"].get(name)
        if entry is None or entry.get("fixed"):
            continue
        g = entry["group"]
        mass.setdefault(g, {})
        mass[g][scheme] = mass[g].get(scheme, 0) + int(entry["n_elems"])
    return {
        g: max(schemes.items(), key=lambda kv: kv[1])[0]
        for g, schemes in mass.items()
    }


def _measure_baseline(tools, cfg: V2Config) -> float:
    """── 1. Baseline (never fabricated) ──"""
    log.info("v2: measuring baseline perplexity", stage="baseline")
    baseline_ppl = tools.calculate_perplexity(
        cfg.source_model_path, verbose=False
    )
    if baseline_ppl is None:
        raise RuntimeError(
            "v2 search could not measure baseline perplexity for "
            f"{cfg.source_model_path}. A budget search against a fabricated "
            "baseline is meaningless — fix the llama.cpp build / corpus."
        )
    log.info("v2 baseline", stage="baseline", ppl=round(baseline_ppl, 4))
    return baseline_ppl


def _resolve_imatrix_and_schemes(
    tools, cfg: V2Config
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """── 2. imatrix (optional, loud when absent) ── + scheme selection."""
    imatrix = None
    if cfg.use_imatrix:
        from magicquant.imatrix import ensure_imatrix, resolve_imatrix_bin

        kwargs = {}
        resolved = resolve_imatrix_bin(tools)
        if resolved:
            kwargs["imatrix_bin"] = resolved
        imatrix = ensure_imatrix(
            cfg.source_model_path, corpus_path=cfg.imatrix_corpus, **kwargs
        )
        if imatrix is None:
            log.warning(
                "v2: imatrix unavailable — distortion table falls back to "
                "unweighted squared error and imatrix-requiring schemes are "
                "excluded (recorded in the manifest)", stage="imatrix",
            )

    schemes = _select_schemes(cfg, imatrix_active=imatrix is not None)
    if not schemes:
        raise RuntimeError("v2: empty scheme choice set after capability filtering")
    log.info("v2 choice set", stage="init", schemes=schemes)
    return imatrix, schemes


def _build_distortion_table(
    cfg: V2Config, schemes: List[str], imatrix: Optional[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    """── 3. Distortion table (CPU) ──"""
    return compute_distortion_table(
        cfg.source_model_path,
        schemes=schemes,
        imatrix=imatrix,
        cache_dir=str(out_dir / "_v2_cache"),
        sample_rows=cfg.sample_rows,
    )


def _calibrate_kappa(
    tools, cfg: V2Config, table: Dict[str, Any],
    imatrix: Optional[Dict[str, Any]], baseline_ppl: float, out_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, MeasurementOutcome], Dict[str, float], List[Dict[str, Any]]]:
    """── 4. κ calibration ──

    Returns (kappa, kappa_provenance, probe_outcomes, eps_sums, failures).
    probe_outcomes and eps_sums must both survive into the results/report-fit
    phase — probe_outcomes carries the "__slice_baseline__"/
    "__base_aggressive__" sentinel keys that phase 7's report fit and
    cumulative-mode detection read directly.
    """
    eps_sums = group_epsilon_sums(table, cfg.probe_scheme)
    groups = sorted(g for g, s in eps_sums.items() if s > 0)
    kappa: Dict[str, float] = {g: 1.0 for g in groups}
    kappa_provenance: Dict[str, str] = {g: "uncalibrated" for g in groups}
    probe_outcomes: Dict[str, MeasurementOutcome] = {}
    failures: List[Dict[str, Any]] = []
    if cfg.group_probes and groups:
        probe_outcomes = run_group_probes(
            tools,
            cfg.source_model_path,
            out_dir,
            groups,
            baseline_ppl,
            probe_scheme=cfg.probe_scheme,
            probe_chunks=cfg.probe_chunks,
            imatrix=imatrix,
            allow_partial=cfg.allow_partial_probes,
            probe_mode=cfg.probe_mode,
        )
        kappa, kappa_provenance = fit_kappa(
            probe_outcomes, eps_sums, baseline_ppl
        )
        for g, o in probe_outcomes.items():
            if not o.ok and g not in ("__slice_baseline__", "__base_aggressive__"):
                failures.append({"stage": "probe", "group": g, **o.to_json()})
    return kappa, kappa_provenance, probe_outcomes, eps_sums, failures


def _allocate_frontier_and_anchors(
    table: Dict[str, Any], kappa: Dict[str, float], cfg: V2Config,
    budget_bytes: int,
) -> Tuple[Allocation, List[Allocation], List[Dict[str, Any]]]:
    """── 5. Allocation + frontier ──

    ``chosen`` (the primary budget allocation) is allowed to raise
    BudgetInfeasibleError uncaught — only the NEIGHBOR anchors are wrapped
    in a try/except and recorded as failures.
    """
    units = _build_units(table, kappa, cfg.floors)
    chosen = allocate(units, budget_bytes)
    log.info(
        "v2 allocation solved",
        stage="allocate",
        size_gb=round(chosen.total_bytes / 1024**3, 3),
        predicted_loss=chosen.total_loss,
        frontier_points=len(chosen.frontier),
    )

    # Anchor allocations: the budget point plus neighbors on the frontier.
    anchor_allocs: List[Allocation] = [chosen]
    failures: List[Dict[str, Any]] = []
    for i in range(1, max(1, cfg.anchors)):
        sign = -1 if i % 2 else 1
        step = (i + 1) // 2
        factor = 1.0 + sign * cfg.anchor_spread * step
        try:
            anchor_allocs.append(allocate(units, int(budget_bytes * factor)))
        except Exception as exc:
            failures.append({
                "stage": "anchor-allocate", "factor": factor,
                "status": "failed", "error": str(exc),
            })
    return chosen, anchor_allocs, failures


def _build_and_verify_anchors(
    cfg: V2Config, tools, imatrix: Optional[Dict[str, Any]],
    baseline_ppl: float, anchor_allocs: List[Allocation], out_dir: Path,
) -> Tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]]]:
    """── 6. Build + verify anchors (full corpus; per-candidate failures
    recorded, run continues) ──

    Raises RuntimeError if every anchor fails (never report an unverified
    allocation).
    """
    from magicquant.gguf.writer import create_hybrid_gguf

    stem = cfg.model_name or Path(cfg.source_model_path).stem
    measured_anchors: List[Dict[str, Any]] = []
    final_model_path: Optional[str] = None
    failures: List[Dict[str, Any]] = []
    for idx, alloc in enumerate(anchor_allocs):
        gb = alloc.total_bytes / 1024**3
        tag = "budget" if idx == 0 else f"n{idx}"
        out_path = out_dir / f"{stem}-v2-{tag}-{gb:.2f}gb.gguf"
        outcome: MeasurementOutcome
        try:
            create_hybrid_gguf(
                output_path=str(out_path),
                base_model_path=cfg.source_model_path,
                quant_config={
                    "base": "BF16",
                    "groups": {},
                    "tensors": alloc.assignment,
                },
                verbose=False,
                imatrix=imatrix,
            )
            ppl = tools.calculate_perplexity(str(out_path), verbose=False)
            if ppl is None:
                outcome = MeasurementOutcome.failure(
                    "llama-perplexity produced no parseable PPL",
                )
            else:
                outcome = MeasurementOutcome.success(ppl)
        except Exception as exc:  # noqa: BLE001 — recorded, run continues
            outcome = MeasurementOutcome.failure(
                f"{type(exc).__name__}: {exc}"
            )

        actual_bytes = out_path.stat().st_size if out_path.is_file() else None
        entry = {
            "tag": tag,
            "path": str(out_path),
            "predicted_bytes": alloc.total_bytes,
            "actual_bytes": actual_bytes,
            "predicted_loss": alloc.total_loss,
            "measurement": outcome.to_json(),
        }
        if outcome.ok:
            entry["ppl"] = outcome.value
            entry["measured_rel_loss"] = (
                (outcome.value - baseline_ppl) / baseline_ppl
            )
        else:
            failures.append({"stage": "anchor", "tag": tag, **outcome.to_json()})
        measured_anchors.append(entry)

        if idx == 0 and outcome.ok:
            final_model_path = str(out_path)
        elif not cfg.keep_anchors and idx > 0 and out_path.is_file():
            out_path.unlink()
            entry["path"] = None

    if not any(a["measurement"]["status"] == "ok" for a in measured_anchors):
        raise RuntimeError(
            "v2 search verified ZERO anchors (every build or measurement "
            "failed). Refusing to report an unverified allocation — see "
            "'failures' in v2_results.json for per-anchor errors."
        )

    return measured_anchors, final_model_path, failures


def _compute_report_fit(
    measured_anchors: List[Dict[str, Any]],
    probe_outcomes: Dict[str, MeasurementOutcome],
    kappa: Dict[str, float],
    eps_sums: Dict[str, float],
    baseline_ppl: float,
) -> Optional[Tuple[float, float]]:
    """First half of ── 7. Report calibration + results ── — the reporting
    fit alone, independently of results-dict assembly."""
    fit_points = [
        (a["predicted_loss"], a["measured_rel_loss"])
        for a in measured_anchors
        if a["measurement"]["status"] == "ok"
    ]
    # Probe points anchor the low end of the reporting fit. In cumulative
    # mode κ·ε == recovery by construction (degenerate x==y), so only the
    # single-mode probe points carry independent signal; skip probes in
    # cumulative mode and let the anchors define the fit.
    _sb = probe_outcomes.get("__slice_baseline__")
    probe_baseline = _sb.value if (_sb is not None and _sb.ok) else baseline_ppl
    _cumulative = "__base_aggressive__" in probe_outcomes
    if not _cumulative:
        for g, o in probe_outcomes.items():
            if g in ("__slice_baseline__", "__base_aggressive__"):
                continue
            if o.ok and eps_sums.get(g, 0) > 0:
                fit_points.append(
                    (kappa[g] * eps_sums[g],
                     max((o.value - probe_baseline) / probe_baseline, 0.0))
                )
    return affine_report_fit(fit_points)


def _assemble_results(
    cfg: V2Config, tools, out_dir: Path, budget_bytes: int, t_start: float,
    baseline_ppl: float, imatrix: Optional[Dict[str, Any]], schemes: List[str],
    table: Dict[str, Any], kappa: Dict[str, float],
    kappa_provenance: Dict[str, str], eps_sums: Dict[str, float],
    chosen: Allocation, measured_anchors: List[Dict[str, Any]],
    final_model_path: Optional[str], failures: List[Dict[str, Any]],
    report_fit: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    """Second half of ── 7. Report calibration + results ── — results-dict
    assembly, the three file writes, and completion logging."""
    frontier_json = [p.to_json() for p in chosen.frontier]
    results = {
        "version": 2,
        "algo": "v2-budget",
        "source_model": cfg.source_model_path,
        "budget_gb": cfg.budget_gb,
        "baseline_ppl": baseline_ppl,
        "baseline_provenance": "measured",
        "measurement": {
            "corpus": tools._resolve_data_file(None),
            "ctx_size": getattr(tools, "ctx_size", None),
            "anchor_chunks": cfg.measurement_chunks,
            "probe_chunks": cfg.probe_chunks,
            "imatrix_active": imatrix is not None,
        },
        "schemes": schemes,
        "kappa": kappa,
        "kappa_provenance": kappa_provenance,
        "group_epsilon_sums": eps_sums,
        "allocation": chosen.to_json(),
        "group_summary": _dominant_group_schemes(table, chosen.assignment),
        "anchors": measured_anchors,
        "report_fit_affine": report_fit,
        "failures": failures,
        "final_model": final_model_path,
        "seconds": round(time.time() - t_start, 1),
    }

    _atomic_write_json(out_dir / "v2_results.json", results)

    from magicquant.v2.interchange import write_interchange_block
    write_interchange_block(out_dir / "search_results.json", results)

    _atomic_write_json(out_dir / "frontier.json", {
        "budget_bytes": budget_bytes,
        "kappa": kappa,
        "points": frontier_json,
        "measured": [
            {"gb": (a["actual_bytes"] or a["predicted_bytes"]) / 1024**3,
             "ppl": a.get("ppl"),
             "rel_loss": a.get("measured_rel_loss"),
             "tag": a["tag"]}
            for a in measured_anchors
        ],
    })

    if failures:
        log.warning(
            "v2 search completed WITH %d recorded failure(s) — see "
            "v2_results.json 'failures'", len(failures), stage="results",
        )
    log.info(
        "v2 search complete",
        stage="results",
        final_model=final_model_path,
        size_gb=round(chosen.total_bytes / 1024**3, 3),
        seconds=results["seconds"],
    )
    return results


def run_budget_search(cfg: V2Config) -> Dict[str, Any]:
    """Execute the v2 pipeline end to end. Returns the results dict (also
    written to ``<output_dir>/v2_results.json``)."""
    from magicquant.utils.llamacpp import LlamaCppTools

    t_start = time.time()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget_bytes = int(cfg.budget_gb * 1024**3)
    failures: List[Dict[str, Any]] = []

    # Single LlamaCppTools instance threaded through every phase below —
    # load-bearing for two mechanisms: _resolve_data_file's corpus pin
    # (checked in _assemble_results) and run_group_probes' ppl_chunks
    # save/restore (_calibrate_kappa), neither of which would work with a
    # freshly-constructed instance per phase.
    tools = LlamaCppTools(cfg.llamacpp_path, data_file=cfg.data_file)
    if cfg.measurement_chunks is not None:
        tools.ppl_chunks = cfg.measurement_chunks

    baseline_ppl = _measure_baseline(tools, cfg)
    imatrix, schemes = _resolve_imatrix_and_schemes(tools, cfg)
    table = _build_distortion_table(cfg, schemes, imatrix, out_dir)
    kappa, kappa_provenance, probe_outcomes, eps_sums, probe_failures = (
        _calibrate_kappa(tools, cfg, table, imatrix, baseline_ppl, out_dir)
    )
    failures.extend(probe_failures)
    chosen, anchor_allocs, anchor_allocate_failures = (
        _allocate_frontier_and_anchors(table, kappa, cfg, budget_bytes)
    )
    failures.extend(anchor_allocate_failures)
    measured_anchors, final_model_path, anchor_failures = (
        _build_and_verify_anchors(
            cfg, tools, imatrix, baseline_ppl, anchor_allocs, out_dir
        )
    )
    failures.extend(anchor_failures)

    report_fit = _compute_report_fit(
        measured_anchors, probe_outcomes, kappa, eps_sums, baseline_ppl
    )
    return _assemble_results(
        cfg, tools, out_dir, budget_bytes, t_start, baseline_ppl, imatrix,
        schemes, table, kappa, kappa_provenance, eps_sums, chosen,
        measured_anchors, final_model_path, failures, report_fit,
    )


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)
