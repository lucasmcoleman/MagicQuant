"""
Naming Scheme Generator - Generate and parse MagicQuant hybrid model names.

The naming scheme uses compact codes to represent which tensor groups use
different quantization schemes:
- E: Embeddings, H: LM Head, Q: Attention Query
- K: Attention Key/Value, O: Attention Output
- U: FFN Up/Gate, D: FFN Down
- X: MoE Experts, R: MoE Router

Example: Qwen3-4B-MXFP4-EH-B16-QKO-IQ4NL.gguf
- Base quantization: MXFP4
- Embeddings + Head: BF16 (higher precision)
- Attention Q/K/O: IQ4_NL
- Everything else: MXFP4 (base)
"""

from typing import Dict, Optional


def config_key(config: Dict[str, str]) -> str:
    """Canonical ``group:scheme`` key for a per-group hybrid config, groups
    sorted -- e.g. ``{"D": "Q4_K_M", "E": "Q6_K"}`` -> ``"D:Q4_K_M|E:Q6_K"``.

    CONTRACT: this format is a PERSISTED interchange format, not just an
    in-memory key. These strings become the keys of the orchestrator's
    ``self._measured``, which are serialized as the "measurements" dict
    keys in search_results.json AND into the measured-search checkpoint
    (magicquant.orchestrator's ``_write_measured_checkpoint`` /
    ``_config_key``), and are parsed back by ``tools/reselect_tiers.py``'s
    ``_parse_key`` (split on "|" then ":"). Do NOT change the separator,
    the sort, or add a prefix here -- that would silently break checkpoint
    resume (stale keys stop matching freshly-computed ones) and break
    reselect_tiers parsing. This is a pure move of the one-liner that used
    to be hand-duplicated in orchestrator.py, pareto.py, and
    evolution/predictor.py; it is unrelated to the SPACE-separated
    human-display variant used elsewhere (e.g. utils/model_card.py's
    ``_format_scheme_map``).
    """
    return "|".join(f"{g}:{config[g]}" for g in sorted(config))


# Group code definitions
GROUP_CODES = {
    'E': 'Embeddings',
    'H': 'LM Head', 
    'Q': 'Attention Query',
    'K': 'Attention Key/Value',
    'O': 'Attention Output',
    'U': 'FFN Up/Gate',
    'D': 'FFN Down',
    'X': 'MoE Experts',
    'R': 'MoE Router'
}


# Map MagicQuant tier labels to HuggingFace-recognized quant strings.
# HuggingFace parses filenames with a regex to generate the quant badge;
# only exact matches from GGMLFileQuantizationType enum names are recognized.
_TIER_TO_HF_LABEL = {
    "Q2": "Q2_K",
    "Q3": "Q3_K_M",
    "Q4": "Q4_K_M",
    "Q5": "Q5_K_M",
    "Q6": "Q6_K",
    "Q8": "Q8_0",
    "IQ4": "IQ4_NL",
}


def generate_name(
    model_name: str,
    base_quant: str,
    overrides: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate a clean MagicQuant hybrid model filename.

    The per-group quantization details are stored in GGUF metadata
    (magicquant.group_schemes), so the filename only carries the model
    name and compression tier.

    The tier portion of model_name (e.g. "Q5" at the end) is expanded
    to an HF-recognized quant string (e.g. "Q5_K_M") so HuggingFace
    shows the correct badge on the model page.

    Args:
        model_name: Base model name with tier suffix (e.g., "Qwen3-4B-Q5")
        base_quant: Base quantization scheme (kept for API compat)
        overrides: Group overrides (kept for API compat)

    Returns:
        Filename like "Qwen3-4B-Q5_K_M.gguf"
    """
    clean_name = model_name.replace(" ", "-")
    if clean_name.lower().endswith(".gguf"):
        clean_name = clean_name[:-5]

    # Expand tier suffix to HF-recognized quant label
    # e.g. "Model-Q5" -> "Model-Q5_K_M"
    for tier, hf_label in _TIER_TO_HF_LABEL.items():
        if clean_name.endswith(f"-{tier}"):
            clean_name = clean_name[: -len(tier)] + hf_label
            break

    return f"{clean_name}.gguf"


def get_group_names() -> Dict[str, str]:
    """Get the mapping of group codes to full names."""
    return GROUP_CODES.copy()