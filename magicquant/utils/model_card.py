"""Model-card generation (M9).

Turns a MagicQuant ``search_results.json`` into a HuggingFace-style README
model card: a tier table with the measured PPL/loss/size and the per-group
scheme map for each tier. Purely local — no network. An optional ``--upload``
path (huggingface_hub) can be layered on top by the CLI.
"""

from typing import Dict, List, Optional


def _format_scheme_map(config: Dict[str, str]) -> str:
    """Render a per-group scheme map like ``E:BF16 H:BF16 Q:Q6_K ...``."""
    return " ".join(f"{g}:{config[g]}" for g in sorted(config))


def generate_model_card(
    results: Dict,
    base_model_name: str = "unknown",
    attribution_url: str = "https://github.com/magiccodingman/MagicQuant-Wiki",
) -> str:
    """Build a markdown model card from a parsed search_results.json dict.

    Args:
        results: the loaded ``search_results.json`` (must contain a ``tiered``
            mapping of tier -> {config, ppl, measured_loss, size_gb}).
        base_model_name: the source model name for the card header.
        attribution_url: link to the methodology wiki.

    Returns:
        A markdown string.
    """
    tiered: Dict[str, Dict] = results.get("tiered") or results.get(
        "tiered_survivors", {}
    )
    baseline_ppl = results.get("baseline_ppl")

    lines: List[str] = []
    lines.append(f"# {base_model_name} — MagicQuant Hybrid Quants")
    lines.append("")
    lines.append(
        "Hybrid per-tensor-group GGUF quants produced by "
        f"[MagicQuant]({attribution_url}). Different tensor groups "
        "(embeddings, attention, FFN, MoE experts) use different schemes "
        "based on measured sensitivity."
    )
    lines.append("")
    if baseline_ppl is not None:
        lines.append(f"**Baseline (BF16) perplexity:** {baseline_ppl}")
        lines.append("")

    lines.append("## Tiers")
    lines.append("")
    lines.append("| Tier | PPL | Loss | Size (GB) | Per-group schemes |")
    lines.append("|------|-----|------|-----------|-------------------|")

    # Stable tier ordering, best-quality first.
    order = ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]
    for tier in order:
        if tier not in tiered:
            continue
        info = tiered[tier]
        ppl = info.get("ppl")
        loss = info.get("measured_loss")
        size = info.get("size_gb")
        scheme_map = _format_scheme_map(info.get("config", {}))
        ppl_s = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else "—"
        loss_s = f"{loss * 100:.2f}%" if isinstance(loss, (int, float)) else "—"
        size_s = f"{size:.2f}" if isinstance(size, (int, float)) else "—"
        lines.append(f"| {tier} | {ppl_s} | {loss_s} | {size_s} | `{scheme_map}` |")

    lines.append("")
    lines.append("---")
    lines.append(
        f"Methodology: [magiccodingman MagicQuant Wiki]({attribution_url})."
    )
    lines.append("")
    return "\n".join(lines)
