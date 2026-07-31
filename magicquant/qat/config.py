"""Load the per-group hybrid quant config for QAT from a search run.

MagicQuant's evolutionary search writes ``search_results.json`` with a ``tiered``
map; each tier's ``config`` is ``{group: scheme_name}`` where ``scheme_name`` is a
MagicQuant identifier (e.g. ``"MXFP4_MOE"``, ``"Q4_K_M"``). QAT's fake-quant
dispatches by the *ggml* block type name (``"MXFP4"``, ``"Q4_K"``), so
``load_hybrid_config`` resolves each scheme name to its ``ggml_type_name`` via the
canonical scheme registry (``magicquant.quant.schemes``).

Returns ``{group: ggml_type_name}`` ready to hand to ``wrap_model``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Union

from magicquant.logging import get_logger
from magicquant.quant.schemes import get_scheme_by_name
from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION, tier_scheme_version

PathLike = Union[str, "os.PathLike[str]"]

log = get_logger(__name__)


def load_hybrid_config(search_results_path: PathLike, tier: str) -> Dict[str, str]:
    """Load the per-group ggml_type_name map for ``tier`` from a search run.

    Args:
        search_results_path: Path to a ``search_results.json`` produced by
            ``magicquant search``.
        tier: Tier key to load (e.g. ``"Q4"``, ``"Q6"``).

    Returns:
        ``{group: ggml_type_name}`` — group is a tensor-group id (``"U"``, ``"Q"``,
        ...), ggml_type_name is the ggml block type the QAT fake-quant dispatches
        on (``"MXFP4"``, ``"Q4_K"``, ``"BF16"``, ...).

    Raises:
        KeyError: if ``tier`` (or the ``tiered`` block) isn't present; the message
            lists the tiers that *are* available.

    Compatibility: a ``search_results.json`` written before the 2026-07
    TIER_SCHEME_VERSION fix (``magicquant.quant.tiers``) has no
    ``tier_scheme_version`` field and its tier labels follow the OLD, wider
    size-ratio boundaries -- e.g. its "Q5" entry may actually be Q6_K-sized.
    This file STILL LOADS (the per-group config content stored under a tier
    key is unaffected by which boundaries produced the label -- only the
    label's human-facing meaning differs), but a non-fatal warning is logged
    so a QAT run doesn't silently assume "Q5" means what it means today.
    """
    with open(search_results_path, encoding="utf-8") as f:
        results = json.load(f)

    version = tier_scheme_version(results)
    if version < CURRENT_TIER_SCHEME_VERSION:
        log.warning(
            f"{search_results_path} was written under tier_scheme_version="
            f"{version} (current: {CURRENT_TIER_SCHEME_VERSION}) -- its tier "
            f"labels follow OLDER, wider size-ratio boundaries (see the "
            f"magicquant.quant.tiers module docstring). The requested tier "
            f"{tier!r} config still loads correctly, but its size may not "
            f"match what {tier!r} means under the current scheme (e.g. an "
            f"old 'Q5' can be Q6_K-sized). Re-run the search for labels "
            f"matching current semantics if that matters for this QAT run.",
            stage="qat_config", search_results_path=str(search_results_path),
            tier=tier, tier_scheme_version=version,
        )

    tiered = results.get("tiered")
    if not tiered:
        raise KeyError(
            f"search_results has no 'tiered' configs "
            f"(top-level keys: {sorted(results.keys())})"
        )

    if tier not in tiered:
        raise KeyError(
            f"tier {tier!r} not in search results; available tiers: "
            f"{sorted(tiered.keys())}"
        )

    config = tiered[tier].get("config", {})
    return {group: _to_ggml_type_name(scheme) for group, scheme in config.items()}


def _to_ggml_type_name(scheme_name: str) -> str:
    """Resolve a MagicQuant scheme name to its ggml block type name.

    Unknown scheme names are passed through unchanged so the fake-quant
    dispatcher can warn + fall back to BF16 (keeps a hybrid trainable rather than
    failing the whole load on one stray scheme).
    """
    try:
        return get_scheme_by_name(scheme_name).ggml_type_name
    except ValueError:
        return scheme_name
