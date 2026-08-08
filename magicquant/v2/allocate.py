"""Budget-constrained mixed-precision allocation (MCKP solver).

Formulation (docs/redesign.md §4): for each unit (tensor) pick one scheme;
minimize total predicted distortion  Σ κ_g(t) · ε(t, s_t)  subject to
Σ bytes(t, s_t) ≤ budget.

Solver: per-unit lower convex hull + global slope-greedy (equivalent to a
Lagrange-multiplier sweep), then a bounded local-search polish over the raw
(non-hull) choices. The greedy trace IS the predicted quality-size
frontier: every prefix is the hull-optimal allocation for its size.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from magicquant.v2.outcome import BudgetInfeasibleError


@dataclass(frozen=True)
class Choice:
    """One admissible (scheme, actual on-disk type) option for a unit."""

    scheme: str          # registry scheme name (what the writer config carries)
    actual: str          # resolved on-disk ggml type (prices bytes + distortion)
    bytes: int           # exact on-disk size
    loss: float          # κ-scaled distortion ε (predicted quality cost)


@dataclass
class Unit:
    """An allocatable tensor with its admissible choices."""

    name: str
    group: str
    choices: List[Choice]                 # ≥1; fixed units have exactly 1
    hull: List[Choice] = field(default_factory=list)  # filled by _build_hull


@dataclass
class FrontierPoint:
    total_bytes: int
    total_loss: float
    changed_unit: Optional[str] = None
    new_scheme: Optional[str] = None

    def to_json(self) -> Dict:
        return {
            "bytes": self.total_bytes,
            "gb": self.total_bytes / 1024**3,
            "loss": self.total_loss,
            "changed_unit": self.changed_unit,
            "new_scheme": self.new_scheme,
        }


@dataclass
class Allocation:
    assignment: Dict[str, str]            # tensor name -> scheme name
    actual_types: Dict[str, str]          # tensor name -> resolved ggml type
    total_bytes: int
    total_loss: float
    frontier: List[FrontierPoint]
    budget_bytes: int
    polish_improvements: int = 0

    def to_json(self) -> Dict:
        return {
            "budget_bytes": self.budget_bytes,
            "budget_gb": self.budget_bytes / 1024**3,
            "total_bytes": self.total_bytes,
            "total_gb": self.total_bytes / 1024**3,
            "predicted_loss": self.total_loss,
            "polish_improvements": self.polish_improvements,
            "assignment": self.assignment,
            "actual_types": self.actual_types,
        }


def _build_hull(choices: List[Choice]) -> List[Choice]:
    """Lower convex hull of a unit's (bytes, loss) points.

    Sorted by bytes ascending; dominated points (another choice with
    ≤ bytes and ≤ loss, strict in one) are removed; then the lower convex
    envelope is kept, so consecutive-edge slopes |Δloss|/Δbytes are
    non-increasing along increasing bytes.
    """
    pts = sorted(choices, key=lambda c: (c.bytes, c.loss))
    # Remove dominated: keep strictly decreasing loss as bytes increase.
    nondom: List[Choice] = []
    for c in pts:
        if nondom and c.bytes == nondom[-1].bytes:
            continue  # same size, worse-or-equal loss (sort order) — dominated
        if nondom and nondom[-1].loss <= c.loss:
            continue  # bigger but not better than the best-so-far — dominated
        nondom.append(c)
    # Convex envelope (monotone chain on decreasing-loss curve).
    hull: List[Choice] = []
    for c in nondom:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # slope from a->b vs a->c: keep b only if it lies below chord a-c.
            # Using cross-product form to avoid division:
            lhs = (b.loss - a.loss) * (c.bytes - a.bytes)
            rhs = (c.loss - a.loss) * (b.bytes - a.bytes)
            if lhs >= rhs:  # b is above (or on) the chord — not on lower hull
                hull.pop()
            else:
                break
        hull.append(c)
    return hull


def allocate(
    units: List[Unit],
    budget_bytes: int,
    polish_sweeps: int = 2,
) -> Allocation:
    """Solve the MCKP: hull greedy + bounded polish. See module docstring.

    Raises BudgetInfeasibleError when even the smallest admissible
    configuration exceeds ``budget_bytes``.
    """
    if not units:
        raise ValueError("allocate() called with no units")

    for u in units:
        if not u.choices:
            raise ValueError(f"Unit {u.name!r} has no admissible choices")
        u.hull = _build_hull(u.choices)

    # Start: every unit at its smallest-bytes hull point.
    state: Dict[str, int] = {u.name: 0 for u in units}  # unit -> hull index
    by_name: Dict[str, Unit] = {u.name: u for u in units}
    total_bytes = sum(u.hull[0].bytes for u in units)
    total_loss = sum(u.hull[0].loss for u in units)

    if total_bytes > budget_bytes:
        raise BudgetInfeasibleError(budget_bytes, total_bytes)

    frontier: List[FrontierPoint] = [FrontierPoint(total_bytes, total_loss)]

    # Max-heap of next hull edges, keyed by loss reduction per byte.
    # Entry: (-slope, unit_name, hull_index_of_target_point).
    heap: List[Tuple[float, str, int]] = []

    def _push_next(u: Unit) -> None:
        idx = state[u.name]
        if idx + 1 < len(u.hull):
            cur, nxt = u.hull[idx], u.hull[idx + 1]
            dbytes = nxt.bytes - cur.bytes
            dloss = cur.loss - nxt.loss  # ≥ 0 on the hull
            if dbytes <= 0:
                return
            heapq.heappush(heap, (-(dloss / dbytes), u.name, idx + 1))

    for u in units:
        _push_next(u)

    remaining = budget_bytes - total_bytes
    while heap:
        neg_slope, name, target_idx = heapq.heappop(heap)
        u = by_name[name]
        if state[name] != target_idx - 1:
            continue  # stale entry
        cur, nxt = u.hull[target_idx - 1], u.hull[target_idx]
        dbytes = nxt.bytes - cur.bytes
        if dbytes > remaining:
            # This unit's later hull points are cumulatively even bigger —
            # no further upgrade of this unit can ever fit. Drop it.
            continue
        state[name] = target_idx
        remaining -= dbytes
        total_bytes += dbytes
        total_loss -= cur.loss - nxt.loss
        frontier.append(
            FrontierPoint(total_bytes, total_loss, name, nxt.scheme)
        )
        _push_next(u)

    # Bounded local-search polish over RAW choices (closes the hull /
    # integrality gap left by the greedy).
    assignment_choice: Dict[str, Choice] = {
        u.name: u.hull[state[u.name]] for u in units
    }
    improvements = 0
    for _ in range(max(0, polish_sweeps)):
        improved = False
        for u in units:
            cur = assignment_choice[u.name]
            best = cur
            for c in u.choices:
                if c is cur:
                    continue
                if c.loss < best.loss and (
                    total_bytes - cur.bytes + c.bytes
                ) <= budget_bytes:
                    best = c
            if best is not cur:
                total_bytes += best.bytes - cur.bytes
                total_loss += best.loss - cur.loss
                assignment_choice[u.name] = best
                improvements += 1
                improved = True
        if not improved:
            break
    if improvements:
        frontier.append(FrontierPoint(total_bytes, total_loss, None, None))

    return Allocation(
        assignment={n: c.scheme for n, c in assignment_choice.items()},
        actual_types={n: c.actual for n, c in assignment_choice.items()},
        total_bytes=total_bytes,
        total_loss=total_loss,
        frontier=frontier,
        budget_bytes=budget_bytes,
        polish_improvements=improvements,
    )
