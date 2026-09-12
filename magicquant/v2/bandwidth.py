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
    warning) when either is missing or expert_count is zero. Never divides by
    zero; never returns a non-finite value."""
    arch = metadata.get("general.architecture")
    key_n = f"{arch}.expert_count"
    key_k = f"{arch}.expert_used_count"
    n = metadata.get(key_n) if arch else None
    k = metadata.get(key_k) if arch else None
    try:
        n = int(n) if n is not None else None
        k = int(k) if k is not None else None
    except (TypeError, ValueError):
        n, k = None, None
    if not n or k is None or n <= 0:
        _warn(log, "stream weights: %s / %s missing or zero (%r / %r); "
                   "routed experts priced at w=1.0 (today's pricing)", key_n, key_k, n, k)
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


def streamed_bytes(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]],
                   w_stream: Mapping[str, float]) -> int:
    """Sum over ALL assigned tensors (fixed included) of w_t * bytes(t, chosen).
    A tensor missing from w_stream is priced at 1.0."""
    total = 0.0
    for name, scheme in assignment.items():
        b = int(table_tensors[name]["choices"][scheme]["bytes"])
        total += w_stream.get(name, 1.0) * b
    return int(round(total))


def pure_loss(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]],
              kappa: Mapping[str, float]) -> float:
    """Sum of kappa_g * werr(t, chosen) over non-fixed tensors -- the quality
    objective with no lambda term (fixed tensors contribute 0.0, as today)."""
    total = 0.0
    for name, scheme in assignment.items():
        entry = table_tensors[name]
        if entry.get("fixed"):
            continue
        werr = entry["choices"][scheme].get("werr")
        if werr is None:
            continue
        total += float(kappa.get(entry["group"], 1.0)) * float(werr)
    return total


def bpw_by_group(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Mean bits per weight per group for an assignment (bytes*8 / n_elems)."""
    bytes_g: Dict[str, int] = defaultdict(int)
    elems_g: Dict[str, int] = defaultdict(int)
    for name, scheme in assignment.items():
        entry = table_tensors[name]
        bytes_g[entry["group"]] += int(entry["choices"][scheme]["bytes"])
        elems_g[entry["group"]] += int(entry.get("n_elems") or 0)
    return {g: (bytes_g[g] * 8.0 / elems_g[g]) for g in bytes_g if elems_g[g] > 0}


def group_view(w_stream: Mapping[str, float], table_tensors: Mapping[str, Mapping[str, Any]],
               *, log=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(observed_by_group, exceptions) for reporting: the modal weight over
    each group's tensors (a group with more than one distinct weight logs a
    WARNING, ignoring rule-2 gathered-by-name tensors, whose disagreement is
    intended); exceptions lists every tensor whose weight differs from its
    group's modal value."""
    per_group: Dict[str, Counter] = defaultdict(Counter)
    disagree_check: Dict[str, set] = defaultdict(set)
    for name, entry in table_tensors.items():
        g = entry.get("group", "UNKNOWN")
        w = w_stream.get(name, 1.0)
        per_group[g][w] += 1
        if not _GATHERED_NAME.search(name):        # rule-2 tensors are an intended, silent exception
            disagree_check[g].add(w)
    by_group: Dict[str, float] = {}
    for g, counts in per_group.items():
        if len(disagree_check[g]) > 1:
            _warn(log, "stream weights: group %s has %d distinct weights %r", g, len(disagree_check[g]), sorted(disagree_check[g]))
        by_group[g] = counts.most_common(1)[0][0]
    exceptions = {name: w_stream.get(name, 1.0) for name, entry in table_tensors.items()
                  if w_stream.get(name, 1.0) != by_group[entry.get("group", "UNKNOWN")]}
    return by_group, exceptions
