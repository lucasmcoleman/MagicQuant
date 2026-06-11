"""HF module name -> GGUF tensor name mapping for QAT group routing.

``wrap_model`` walks ``model.named_modules()`` (which yields module *paths*, e.g.
``model.layers.0.self_attn.q_proj`` with no trailing ``.weight``) and needs each
``nn.Linear``'s GGUF tensor name so it can classify the module into a tensor
group. This reuses MagicQuant's canonical ``source.py`` ``_HF_TO_GGUF_PATTERNS``
(via ``_hf_name_to_gguf``) so the QAT routing never drifts from the writer's name
mapping.

``hf_to_ggml_name(name)`` returns the GGUF tensor name (always ``.weight``-suffixed,
e.g. ``blk.0.attn_q.weight``) or ``None`` if the name doesn't map to a known
weight tensor.
"""

from __future__ import annotations

from typing import Optional

from magicquant.gguf.source import _HF_TO_GGUF_COMPILED, _hf_name_to_gguf


def hf_to_ggml_name(hf_module_name: str) -> Optional[str]:
    """Map a HuggingFace module/tensor name to its GGUF tensor name.

    Accepts either a module path (``model.layers.0.self_attn.q_proj``) or a full
    tensor name (``...q_proj.weight``). Returns the GGUF tensor name (e.g.
    ``blk.0.attn_q.weight``) or ``None`` if it doesn't match a known mapping.

    Unlike the underlying ``_hf_name_to_gguf`` (which falls back to returning the
    input unchanged), this returns ``None`` for unmatched names so callers can
    cleanly skip modules that aren't quantizable weights.
    """
    if not hf_module_name:
        return None

    # ``named_modules`` paths have no ``.weight`` suffix; the source patterns all
    # match the full tensor name. Normalize to the ``.weight`` form.
    name = hf_module_name
    if not name.endswith(".weight"):
        name = name + ".weight"

    mapped = _hf_name_to_gguf(name)

    # _hf_name_to_gguf returns the input unchanged when nothing matched. Detect
    # that "no match" case so we return None rather than a bogus passthrough.
    if mapped == name and not _matches_a_pattern(name):
        return None
    return mapped


def _matches_a_pattern(name: str) -> bool:
    """True if ``name`` (full tensor name) matches one of the HF->GGUF patterns.

    Mirrors ``_hf_name_to_gguf``'s prefix-stripping so the "did anything match?"
    check agrees with the actual mapping. ``output.weight``/``lm_head.weight`` are
    handled directly by ``_hf_name_to_gguf`` and count as matches.
    """
    if name in ("output.weight", "lm_head.weight"):
        return True
    stripped = name
    for prefix in ("model.language_model.", "language_model."):
        if stripped.startswith(prefix):
            stripped = "model." + stripped[len(prefix):]
            break
    return any(pat.match(stripped) for pat, _ in _HF_TO_GGUF_COMPILED)
