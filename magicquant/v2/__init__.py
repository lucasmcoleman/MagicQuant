"""MagicQuant v2 — budget-constrained mixed-precision allocation.

Design: docs/redesign.md. Enabled via ``magicquant search --algo v2
--budget-gb <B>`` (the v1 evolutionary path is the untouched default).

Public API:
    run_budget_search(V2Config) -> results dict   (search.py)
    compute_distortion_table(...)                 (sensitivity.py)
    allocate(...)                                 (allocate.py)
"""

from magicquant.v2.outcome import (  # noqa: F401
    MeasurementOutcome,
    ProbeMeasurementError,
    BudgetInfeasibleError,
)
from magicquant.v2.search import V2Config, run_budget_search  # noqa: F401
from magicquant.v2.interchange import budget_tier_key, write_interchange_block  # noqa: F401
