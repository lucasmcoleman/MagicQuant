"""Bridge a v2 budget build into the v1 ``search_results.json`` interchange.

v1 consumers (QAT's ``load_hybrid_config``, Foundry's ROCmFPX mq-hybrid mode,
the publish stage) read ``tiered[<key>].config`` as ``{group: scheme}``. A v2
budget build is per-tensor; this module writes BOTH the per-group projection
(``config``, reusing v2's own ``group_summary``) and the exact per-tensor map
(``tensor_config``) under a ``BUDGET-<N>GiB`` pseudo-tier key.

MERGE-ONLY: an existing file's other tiers are never touched, and a legacy
(pre-version-stamp) file is never given a version stamp — that would falsely
relabel its old wide-band tier names as current-semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

from magicquant.logging import get_logger
from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION

log = get_logger(__name__)


def budget_tier_key(budget_gib: float) -> str:
    """Canonical pseudo-tier key for a budget build. One format, one place."""
    return f"BUDGET-{budget_gib:g}GiB"


def write_interchange_block(search_results_path: Path | str, results: dict) -> str:
    """Merge this budget run's block into ``search_results.json``.

    Returns the tier key written. Never raises for a missing file (creates
    it); a corrupt existing file is replaced (consistent with the rest of the
    pipeline's treat-corrupt-as-absent policy) after printing a warning.
    """
    path = Path(search_results_path)
    key = budget_tier_key(results["budget_gb"])
    alloc = results["allocation"]
    anchors = results.get("anchors") or []
    block = {
        "config": results["group_summary"],
        "tensor_config": alloc["assignment"],
        "tensor_actual_types": alloc["actual_types"],
        "algo": "v2-budget",
        "budget_bytes": alloc["budget_bytes"],
        "predicted_bytes": alloc["total_bytes"],
        "actual_bytes": anchors[0].get("actual_bytes") if anchors else None,
        "ppl": anchors[0].get("ppl") if anchors else None,
        "baseline_ppl": results["baseline_ppl"],
    }

    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            log.warning(
                f"Warning: {path} unreadable; rewriting with budget block only"
            )
            data = {}
    fresh = not data
    data.setdefault("tiered", {})[key] = block
    if fresh:
        data["tier_scheme_version"] = CURRENT_TIER_SCHEME_VERSION

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return key
