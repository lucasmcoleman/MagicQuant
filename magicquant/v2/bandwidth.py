"""Stream weights and streamed-bytes accounting for the v2 allocator.

Spec: /server/ai/docs/specs/magicquant-bandwidth.md (sections 2.2-2.3);
design record: docs/redesign.md section 11.

A tensor's *stream weight* w_t in [0, 1] is the fraction of its stored bytes
a single decode token reads: 1.0 for the trunk (read in full every token),
expert_used_count / expert_count for routed experts, 0.0 for row-gathered
embedding tables. Weights are architectural -- derived from the group, the
name and the GGUF hparams -- never measured, and never written into the cached
distortion table (they are computed after the table is loaded).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from magicquant.v2.allocate import Allocation, Unit, allocate
from magicquant.v2.outcome import BandwidthInfeasibleError

KNOWN_GROUPS = frozenset({"E", "H", "X", "R", "Q", "K", "O", "S", "U", "D", "N", "V", "UNKNOWN"})

# Row-gathered embedding tables, matched on the NAME (any prefix), so the MTP
# module's own copy (blk.N.nextn.embed_tokens.weight, group H) is caught too.
_GATHERED_NAME = re.compile(r"(^|\.)(token_embd|embed_tokens|per_layer_token_embd)\.weight$")


def _warn(log, msg: str, *args) -> None:
    # Pre-format: callers may pass a stdlib logger (tests) or magicquant's
    # structlog logger (search.py), and structlog does not %-format positional args.
    if log is not None:
        log.warning(msg % args if args else msg)


def expert_ratio(metadata: Mapping[str, Any], *, log=None) -> Optional[float]:
    """expert_used_count / expert_count from the GGUF hparams, or None (with a
    warning) when either is missing, expert_count is zero, or the values are
    out of range (used-count negative or greater than the total -- fails hot,
    never clamped). ``n`` and ``k`` are converted independently, so a parse
    failure on one never discards a successfully-parsed value for the other;
    every warning names the RAW hparam value(s), not the (possibly-None)
    converted ones, so the offending value is visible. Never divides by zero;
    never returns a non-finite value or one outside [0,1]."""
    arch = metadata.get("general.architecture")
    if not arch:
        _warn(log, "stream weights: general.architecture missing; routed experts priced at w=1.0")
        return None
    key_n = f"{arch}.expert_count"
    key_k = f"{arch}.expert_used_count"
    n_raw = metadata.get(key_n)
    k_raw = metadata.get(key_k)
    try:
        n = int(n_raw) if n_raw is not None else None
    except (TypeError, ValueError, OverflowError):
        n = None
    try:
        k = int(k_raw) if k_raw is not None else None
    except (TypeError, ValueError, OverflowError):
        k = None
    if not n or k is None or n <= 0:
        _warn(log, "stream weights: %s / %s missing or zero (%r / %r); "
                   "routed experts priced at w=1.0 (today's pricing)", key_n, key_k, n_raw, k_raw)
        return None
    if k > n or k <= 0:
        _warn(log, "stream weights: %s=%r is inconsistent with %s=%r (used-count > "
                   "total, zero, or negative); routed experts priced at w=1.0 (fails hot, not clamped)", key_k, k_raw, key_n, n_raw)
        return None
    return min(1.0, max(0.0, k / n))


def stream_weights_by_group(metadata: Mapping[str, Any], overrides: Optional[Mapping[str, float]] = None,
                            *, log=None) -> Dict[str, float]:
    """One weight per group letter (plus "default"), the v1 entry point.
    Rule order: override > X ratio > E/V zero > everything else 1.0."""
    overrides = dict(overrides or {})
    for g in overrides:
        if g not in KNOWN_GROUPS:
            _warn(log, "stream weights: override for unknown group %r", g)
    ratio = expert_ratio(metadata, log=log)
    w: Dict[str, float] = {g: 1.0 for g in KNOWN_GROUPS}
    w["X"] = 1.0 if ratio is None else ratio
    w["E"] = 0.0
    w["V"] = 0.0
    for g, v in overrides.items():
        w[g] = float(v)
    w["default"] = 1.0
    for g, v in w.items():
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"stream weight for group {g!r} is not a finite value in [0,1]: {v!r}")
    return w


def stream_weights(table_tensors: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any],
                   overrides: Optional[Mapping[str, float]] = None, *, log=None) -> Dict[str, float]:
    """Per-tensor weights over a distortion table's ``tensors`` mapping.
    Rule order (first match wins): override by group > gathered by name >
    X ratio > E > V > UNKNOWN (1.0, warned) > everything else 1.0."""
    by_group = stream_weights_by_group(metadata, overrides, log=log)
    overrides = dict(overrides or {})
    out: Dict[str, float] = {}
    unknown = 0
    not_3d_in_x: List[str] = []
    is_3d_not_x: List[str] = []
    for name, entry in table_tensors.items():
        group = entry.get("group", "UNKNOWN")
        shape = list(entry.get("shape") or [])
        if group in overrides:
            w = float(overrides[group])
        elif _GATHERED_NAME.search(name):
            w = 0.0
        elif group == "X":
            w = by_group["X"]
        elif group in ("E", "V"):
            w = 0.0
        elif group == "UNKNOWN":
            w = 1.0
            unknown += 1
        else:
            w = 1.0
        if group == "X" and len(shape) != 3:
            not_3d_in_x.append(name)
        if group != "X" and len(shape) == 3:
            is_3d_not_x.append(name)
        if not (math.isfinite(w) and 0.0 <= w <= 1.0):
            raise ValueError(f"stream weight for {name!r} is not a finite value in [0,1]: {w!r}")
        out[name] = w
    if unknown:
        _warn(log, "stream weights: %d UNKNOWN-group tensor(s) priced at w=1.0", unknown)
    if not_3d_in_x:
        _warn(log, "stream weights: %d group-X tensor(s) not 3-D (first: %s)",
              len(not_3d_in_x), ", ".join(not_3d_in_x[:3]))
    if is_3d_not_x:
        _warn(log, "stream weights: %d tensor(s) are 3-D but not X (first: %s)",
              len(is_3d_not_x), ", ".join(is_3d_not_x[:3]))
    return out


def _choice(table_tensors: Mapping[str, Mapping[str, Any]], name: str, scheme: str) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """(entry, choice) = table_tensors[name], table_tensors[name]["choices"][scheme].
    Raises a KeyError naming both `name` and `scheme` if the tensor, its
    "choices", or that particular scheme is missing -- instead of a bare,
    uninformative KeyError pointing at one dict level."""
    try:
        entry = table_tensors[name]
        choice = entry["choices"][scheme]
    except KeyError:
        raise KeyError(f"{name!r} / {scheme!r}: tensor or scheme not found in distortion table") from None
    return entry, choice


def streamed_bytes(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]],
                   w_stream: Mapping[str, float]) -> int:
    """Sum over ALL assigned tensors (fixed included) of w_t * bytes(t, chosen).
    A tensor missing from w_stream is priced at 1.0."""
    total = 0.0
    for name, scheme in assignment.items():
        _, choice = _choice(table_tensors, name, scheme)
        try:
            b = int(choice["bytes"])
        except KeyError:
            raise KeyError(f"{name!r} / {scheme!r}: choice has no 'bytes' entry") from None
        total += w_stream.get(name, 1.0) * b
    return int(round(total))


def pure_loss(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]],
              kappa: Mapping[str, float]) -> float:
    """Sum of kappa_g * werr(t, chosen) over non-fixed tensors -- the quality
    objective with no lambda term (fixed tensors contribute 0.0, as today).
    A choice with ``werr: null`` (the documented no-decode sentinel, see
    sensitivity.py) is silently excluded, same as before; a choice with the
    "werr" key entirely absent is a malformed table entry and raises."""
    total = 0.0
    for name, scheme in assignment.items():
        entry, choice = _choice(table_tensors, name, scheme)
        if entry.get("fixed"):
            continue
        try:
            werr = choice["werr"]
        except KeyError:
            raise KeyError(f"{name!r} / {scheme!r}: choice has no 'werr' entry") from None
        if werr is None:
            continue
        total += float(kappa.get(entry.get("group", "UNKNOWN"), 1.0)) * float(werr)
    return total


def bpw_by_group(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Mean bits per weight per group for an assignment (bytes*8 / n_elems)."""
    bytes_g: Dict[str, int] = defaultdict(int)
    elems_g: Dict[str, int] = defaultdict(int)
    for name, scheme in assignment.items():
        entry, choice = _choice(table_tensors, name, scheme)
        try:
            b = int(choice["bytes"])
        except KeyError:
            raise KeyError(f"{name!r} / {scheme!r}: choice has no 'bytes' entry") from None
        group = entry.get("group", "UNKNOWN")
        bytes_g[group] += b
        elems_g[group] += int(entry.get("n_elems") or 0)
    return {g: (bytes_g[g] * 8.0 / elems_g[g]) for g in bytes_g if elems_g[g] > 0}


def group_view(w_stream: Mapping[str, float], table_tensors: Mapping[str, Mapping[str, Any]],
               *, log=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(observed_by_group, exceptions) for reporting: the modal weight over
    each group's tensors; exceptions lists every tensor whose weight differs
    from its group's modal value.

    The modal weight is computed from the NON-gathered population only (a
    group falls back to its full population when every tensor in it is
    gathered-by-name, i.e. rule-2). Within that population the modal weight
    is chosen deterministically: the weight with the HIGHEST COUNT, ties
    broken by the HIGHER weight (the conservative, "hotter" reading) -- never
    by dict insertion order, so the result is independent of it. A group
    with more than one distinct weight IN THE POPULATION ACTUALLY USED for
    its modal (non-gathered when non-empty, else the full population) logs a
    WARNING -- so an all-gathered group whose members disagree still warns,
    even though the non-gathered counter it would otherwise check is empty.
    Rule-2 tensors excluded from the modal population can never tip a tie
    and always land in `exceptions` when their weight disagrees with the
    rest of the group."""
    per_group: Dict[str, Counter] = defaultdict(Counter)      # full population (fallback)
    non_gathered: Dict[str, Counter] = defaultdict(Counter)   # rule-2-excluded population
    for name, entry in table_tensors.items():
        g = entry.get("group", "UNKNOWN")
        w = w_stream.get(name, 1.0)
        per_group[g][w] += 1
        if not _GATHERED_NAME.search(name):        # rule-2 tensors are an intended, silent exception
            non_gathered[g][w] += 1
    by_group: Dict[str, float] = {}
    for g, counts in per_group.items():
        pop = non_gathered[g] if non_gathered[g] else counts
        if len(pop) > 1:
            _warn(log, "stream weights: group %s has %d distinct weights %r", g, len(pop), sorted(pop))
        # Deterministic modal pick: highest count, ties broken by the higher
        # (hotter) weight -- never by Counter/dict insertion order.
        by_group[g] = max(pop.items(), key=lambda kv: (kv[1], kv[0]))[0]
    exceptions = {name: w_stream.get(name, 1.0) for name, entry in table_tensors.items()
                  if w_stream.get(name, 1.0) != by_group[entry.get("group", "UNKNOWN")]}
    return by_group, exceptions


LAM_MAX = 1e9          # feasibility probe only -- never returned as the chosen lambda
LAM_MIN = 1e-6
BISECTION_STEPS = 40


def _count_inversions(probes: Sequence[Tuple[float, int]]) -> int:
    """Number of probe pairs (i < j) where lambda rose but streamed bytes rose
    too -- evidence that S(lambda) is not monotone (allocate.py drops a unit
    permanently once its next edge exceeds the remaining budget)."""
    n = 0
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            if probes[i][0] < probes[j][0] and probes[i][1] < probes[j][1]:
                n += 1
    return n


BuildUnits = Callable[[Dict[str, Any], Mapping[str, float], Mapping[str, str], Mapping[str, float], float], List[Unit]]


@dataclass
class BandwidthSolution:
    """The chosen lambda, the resulting Allocation, and enough context to
    report why: bpw/streamed-bytes at both lambda=0 and the chosen lambda,
    the bisection's non-monotonicity count, and the stream-weight provenance
    used to compute it. Spec: /server/ai/docs/specs/magicquant-bandwidth.md §2.3."""

    mode: str                                   # "off" | "weight" | "budget"
    lam: float
    chosen: Allocation
    w_stream: Dict[str, float]
    by_group: Dict[str, float]
    exceptions: Dict[str, float]
    weights_source: Dict[str, Any]
    streamed_bytes: int
    streamed_bytes_lambda0: int
    streamed_bytes_min: Optional[int]
    budget_bw_bytes: Optional[int]
    nonmonotone_probes: int
    predicted_loss_pure: float
    total_loss_with_lambda: float
    storage_utilisation: float
    bpw_lambda0: Dict[str, float] = field(default_factory=dict)
    bpw_chosen: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "lambda": self.lam,
            "stream_weights": {"observed_by_group": self.by_group, "default": 1.0, "exceptions": self.exceptions},
            "weights_source": self.weights_source,
            "streamed_bytes": self.streamed_bytes,
            "streamed_bytes_lambda0": self.streamed_bytes_lambda0,
            "streamed_bytes_min": self.streamed_bytes_min,
            "budget_bw_bytes": self.budget_bw_bytes,
            "nonmonotone_probes": self.nonmonotone_probes,
            "predicted_loss_pure": self.predicted_loss_pure,
            "total_loss_with_lambda": self.total_loss_with_lambda,
            "storage_utilisation": self.storage_utilisation,
            "bpw_by_group": {"lambda0": self.bpw_lambda0, "chosen": self.bpw_chosen},
        }


def solve_bandwidth(build_units: BuildUnits, table: Dict[str, Any], kappa: Mapping[str, float],
                    floors: Mapping[str, str], w_stream: Mapping[str, float],
                    budget_bytes: int, budget_bw_bytes: Optional[int], *,
                    lam_fixed: Optional[float] = None,
                    weights_source: Optional[Dict[str, Any]] = None, log=None) -> BandwidthSolution:
    """Choose lambda and the allocation. The storage budget is always hard
    (enforced inside allocate()); lambda only steers which bytes are spent.
    Spec section 2.3.

    ``build_units`` is supplied by the caller (search.py's ``_build_units``
    in production, a local builder in tests) so this module never imports
    search.py -- that would cycle, since search.py calls solve_bandwidth."""
    tensors = table["tensors"]
    kappa = dict(kappa)
    if lam_fixed is not None and budget_bw_bytes is not None:
        raise ValueError("bandwidth_weight and budget_bw_gb are mutually exclusive")
    if lam_fixed is not None and not (math.isfinite(lam_fixed) and lam_fixed >= 0.0):
        raise ValueError(f"lam_fixed must be a finite value >= 0, got {lam_fixed!r}")
    if budget_bw_bytes is not None and not (math.isfinite(budget_bw_bytes) and budget_bw_bytes > 0):
        raise ValueError(f"budget_bw_bytes must be a finite value > 0, got {budget_bw_bytes!r}")
    probes: List[Tuple[float, int]] = []

    def solve(lam: float):
        alloc = allocate(build_units(table, kappa, floors, w_stream, lam), budget_bytes)
        s = streamed_bytes(alloc.assignment, tensors, w_stream)
        probes.append((lam, s))
        return alloc, s

    def finish(mode, lam, alloc, s, s0, s_min, nonmono, a0):
        by_group, exceptions = group_view(w_stream, tensors, log=log)
        utilisation = alloc.total_bytes / budget_bytes
        if utilisation < 0.98:
            _warn(log, "bandwidth: allocation uses only %.1f%% of the storage budget "
                       "-- at lambda>0 the greedy can leave budget unspent; compare "
                       "total_bytes across arms", utilisation * 100.0)
        return BandwidthSolution(
            mode=mode, lam=lam, chosen=alloc, w_stream=dict(w_stream),
            by_group=by_group,
            exceptions=exceptions,
            weights_source=dict(weights_source or {}),
            streamed_bytes=s, streamed_bytes_lambda0=s0, streamed_bytes_min=s_min,
            budget_bw_bytes=budget_bw_bytes if mode == "budget" else None,
            nonmonotone_probes=nonmono,
            predicted_loss_pure=pure_loss(alloc.assignment, tensors, kappa),
            total_loss_with_lambda=alloc.total_loss,
            storage_utilisation=utilisation,
            bpw_lambda0=bpw_by_group(a0.assignment, tensors),
            bpw_chosen=bpw_by_group(alloc.assignment, tensors),
        )

    a0, s0 = solve(0.0)
    if lam_fixed is not None:
        a, s = solve(float(lam_fixed))
        return finish("weight", float(lam_fixed), a, s, s0, None, _count_inversions(probes), a0)
    if budget_bw_bytes is None:
        return finish("off", 0.0, a0, s0, s0, None, _count_inversions(probes), a0)
    B = int(budget_bw_bytes)
    if s0 <= B:
        return finish("budget", 0.0, a0, s0, s0, None, _count_inversions(probes), a0)
    _, s_min = solve(LAM_MAX)
    if s_min > B:
        raise BandwidthInfeasibleError(B, s_min)
    lo, hi = LAM_MIN, LAM_MAX
    best_alloc: Optional[Allocation] = None
    best_lam = 0.0
    best_s = 0
    best_pure = math.inf
    for _ in range(BISECTION_STEPS):
        if hi / lo < 1.0 + 1e-6:
            # lo/hi have converged to the same measurement -- every further
            # probe would be bit-identical (results only change with the
            # allocation, which is already pinned between lo and hi). Cuts
            # ~40% of allocate() calls off a typical bisection.
            break
        mid = math.sqrt(lo * hi)
        a, s = solve(mid)
        if s <= B:
            hi = mid
            pl = pure_loss(a.assignment, tensors, kappa)
            if best_alloc is None or pl < best_pure:
                best_alloc, best_lam, best_s, best_pure = a, mid, s, pl
        else:
            lo = mid
    nonmono = _count_inversions(probes)
    if best_alloc is None:
        raise RuntimeError(
            f"bandwidth: bisection found no feasible lambda although the streamed-bytes "
            f"floor {s_min / 1024**3:.3f} GiB <= budget {B / 1024**3:.3f} GiB; "
            f"S(lambda) is non-monotone ({nonmono} inversions over {len(probes)} probes)"
        )
    if not (best_s <= B and best_alloc.total_bytes <= budget_bytes):
        raise RuntimeError(
            f"bandwidth: internal invariant violated -- the chosen allocation "
            f"exceeds a budget (streamed {best_s} vs {B}, stored "
            f"{best_alloc.total_bytes} vs {budget_bytes})"
        )
    if nonmono and log is not None:
        log.warning("bandwidth: S(lambda) non-monotone (%d inversions over %d probes)", nonmono, len(probes))
    return finish("budget", best_lam, best_alloc, best_s, s0, s_min, nonmono, a0)
