"""MCKP allocator correctness: brute-force parity on small instances,
budget respect, frontier monotonicity, infeasibility."""

import itertools
import random

import pytest

from magicquant.v2.allocate import Choice, Unit, _build_hull, allocate
from magicquant.v2.outcome import BudgetInfeasibleError


def _mk_unit(name, group, pts):
    return Unit(
        name=name, group=group,
        choices=[Choice(f"S{i}", f"S{i}", b, l) for i, (b, l) in enumerate(pts)],
    )


def _brute_force(units, budget):
    best = None
    for combo in itertools.product(*[u.choices for u in units]):
        b = sum(c.bytes for c in combo)
        if b > budget:
            continue
        l = sum(c.loss for c in combo)
        if best is None or l < best[0]:
            best = (l, b, combo)
    return best


def test_hull_drops_dominated_and_nonconvex():
    pts = [(100, 10.0), (200, 6.0), (250, 7.0), (300, 5.9), (400, 1.0), (500, 0.5)]
    # (250, 7.0) dominated by (200, 6.0); (300, 5.9) is above chord 200->400.
    hull = _build_hull(
        [Choice(f"S{i}", f"S{i}", b, l) for i, (b, l) in enumerate(pts)]
    )
    hull_pts = [(c.bytes, c.loss) for c in hull]
    assert (250, 7.0) not in hull_pts
    assert (300, 5.9) not in hull_pts
    assert hull_pts[0] == (100, 10.0)
    assert hull_pts[-1] == (500, 0.5)
    # bytes strictly increasing, loss strictly decreasing
    assert all(a[0] < b[0] and a[1] > b[1] for a, b in zip(hull_pts, hull_pts[1:]))


def test_allocate_matches_bruteforce_on_random_instances():
    rng = random.Random(42)
    for trial in range(30):
        units = []
        for i in range(4):
            n_choices = rng.randint(2, 5)
            pts = sorted(
                {(rng.randint(10, 300), round(rng.uniform(0.1, 20.0), 3))
                 for _ in range(n_choices)}
            )
            units.append(_mk_unit(f"t{i}", "G", pts))
        min_b = sum(min(c.bytes for c in u.choices) for u in units)
        max_b = sum(max(c.bytes for c in u.choices) for u in units)
        budget = rng.randint(min_b, max_b)
        got = allocate(units, budget)
        exact = _brute_force(units, budget)
        assert exact is not None
        assert got.total_bytes <= budget
        # Hull-greedy + polish must land within 10% of exact optimum
        # (typically equal; the bound guards pathological non-convexity).
        assert got.total_loss <= exact[0] * 1.10 + 1e-9, (
            f"trial {trial}: got {got.total_loss} vs exact {exact[0]}"
        )


def test_budget_respected_and_frontier_monotone():
    units = [
        _mk_unit(f"t{i}", "G", [(50, 8.0), (100, 3.0), (200, 1.0), (400, 0.2)])
        for i in range(20)
    ]
    alloc = allocate(units, budget_bytes=3000)
    assert alloc.total_bytes <= 3000
    pts = [(p.total_bytes, p.total_loss) for p in alloc.frontier]
    assert all(b2 >= b1 for (b1, _), (b2, _) in zip(pts, pts[1:]))
    assert all(l2 <= l1 + 1e-12 for (_, l1), (_, l2) in zip(pts, pts[1:]))


def test_infeasible_budget_raises_with_min_bytes():
    units = [_mk_unit("a", "G", [(100, 1.0), (200, 0.5)]),
             _mk_unit("b", "G", [(150, 2.0)])]
    with pytest.raises(BudgetInfeasibleError) as ei:
        allocate(units, budget_bytes=200)
    assert ei.value.min_bytes == 250
    assert ei.value.budget_bytes == 200


def test_fixed_units_pass_through():
    units = [
        _mk_unit("fixed", "N", [(64, 0.0)]),
        _mk_unit("var", "U", [(100, 5.0), (200, 1.0)]),
    ]
    alloc = allocate(units, budget_bytes=264)
    assert alloc.assignment["fixed"] == "S0"
    assert alloc.assignment["var"] == "S1"
    assert alloc.total_bytes == 264


def test_bigger_budget_never_worse():
    rng = random.Random(7)
    units = [
        _mk_unit(f"t{i}", "G",
                 sorted({(rng.randint(20, 500), round(rng.uniform(0.1, 9.0), 2))
                         for _ in range(4)}))
        for i in range(10)
    ]
    min_b = sum(min(c.bytes for c in u.choices) for u in units)
    losses = []
    for budget in range(min_b, min_b + 3000, 300):
        losses.append(allocate(units, budget).total_loss)
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))
