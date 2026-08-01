"""Measurement outcomes and failure semantics for the v2 search.

Doctrine (docs/redesign.md §6): a measurement either succeeded, or it
failed and the artifact records the failure. Nothing downstream ever
consumes a fabricated value; no single candidate failure kills a run that
can still make progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Canonical definition lives in the v1 prober (it is raised there too, and
# probing.py must stay import-light); re-exported here as the v2-facing name.
from magicquant.evolution.probing import ProbeMeasurementError  # noqa: F401,E402


class BudgetInfeasibleError(RuntimeError):
    """The requested byte budget is below the smallest achievable model
    size for the given choice set. Carries ``min_bytes`` so callers can
    report what IS achievable."""

    def __init__(self, budget_bytes: int, min_bytes: int):
        self.budget_bytes = budget_bytes
        self.min_bytes = min_bytes
        super().__init__(
            f"Budget {budget_bytes / 1024**3:.2f} GiB is infeasible: the "
            f"smallest allocation with the enabled schemes is "
            f"{min_bytes / 1024**3:.2f} GiB. Raise --budget-gb or enable "
            "more aggressive schemes (e.g. --enable-iq)."
        )


@dataclass
class MeasurementOutcome:
    """One build/measure step's result — success OR recorded failure."""

    status: str  # "ok" | "failed"
    value: Optional[float] = None  # the measured quantity (e.g. PPL)
    error: Optional[str] = None
    attempts: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @classmethod
    def success(cls, value: float, attempts: int = 1, **meta) -> "MeasurementOutcome":
        return cls(status="ok", value=value, attempts=attempts, meta=meta)

    @classmethod
    def failure(cls, error: str, attempts: int = 1, **meta) -> "MeasurementOutcome":
        return cls(status="failed", error=error, attempts=attempts, meta=meta)

    def to_json(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "error": self.error,
            "attempts": self.attempts,
            "meta": self.meta,
        }
