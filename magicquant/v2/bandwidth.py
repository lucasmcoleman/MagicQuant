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
from typing import Any, Dict, Mapping, Optional, Tuple

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
    if k > n or k < 0:
        _warn(log, "stream weights: %s=%r is inconsistent with %s=%r (used-count > "
                   "total, or negative); routed experts priced at w=1.0 (fails hot, not clamped)", key_k, k_raw, key_n, n_raw)
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
            _warn(log, "stream weights: group-X tensor %s is not 3-D (shape %r)", name, shape)
        if group != "X" and len(shape) == 3:
            _warn(log, "stream weights: 3-D tensor %s is in group %s, not X", name, group)
        if not (math.isfinite(w) and 0.0 <= w <= 1.0):
            raise ValueError(f"stream weight for {name!r} is not a finite value in [0,1]: {w!r}")
        out[name] = w
    if unknown:
        _warn(log, "stream weights: %d UNKNOWN-group tensor(s) priced at w=1.0", unknown)
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
