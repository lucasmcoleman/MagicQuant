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

``fused_expert_segments(name, shape)`` handles the OTHER shape a modern MoE takes:
a single fused 3-D ``nn.Parameter`` holding every routed expert
(``...mlp.experts.gate_up_proj`` with shape ``[n_experts, out, in]``), which is
not an ``nn.Linear`` at all and which GGUF splits across *several* tensor names.
See that function's docstring for the layout facts and how they were verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

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


# ── Fused 3-D MoE expert parameters ──────────────────────────────────────────
#
# Modern transformers MoE blocks (Qwen3.5/Qwen3.6, Llama4, GPT-OSS, Granite
# hybrid, ...) no longer build one ``nn.Linear`` per expert. They store every
# routed expert of a projection in ONE fused 3-D ``nn.Parameter``:
#
#   model.language_model.layers.N.mlp.experts.gate_up_proj  [E, 2*I, H]
#   model.language_model.layers.N.mlp.experts.down_proj     [E, H,   I]
#
# ``wrap_model``'s ``nn.Linear`` walk cannot see these (they are raw Parameters
# on a plain ``nn.Module``), which is why QAT covered ~7% of Qwen3.6-35B-A3B's
# weights before this existed.
#
# Layout facts this mapping depends on, and how each was verified against the
# actual artifacts rather than assumed:
#
#  * Torch layout is ``[E, out_features, in_features]`` -- the forward does
#    ``nn.functional.linear(x, self.gate_up_proj[e])``, so slot 1 is out and
#    slot 2 is in. (transformers ``Qwen3_5MoeExperts.forward``.)
#  * gate and up are stored CONCATENATED along the out axis (gate first), not
#    interleaved, and not transposed: transformers' ``use_experts_implementation``
#    decorator defaults are ``is_concatenated=True, is_transposed=False`` and
#    ``Qwen3_5MoeExperts`` takes those defaults; the eager forward then does
#    ``.chunk(2, dim=-1)`` on the projection output, i.e. contiguous halves.
#    ``expert_wrap`` re-checks the module's own ``is_concatenated``/
#    ``is_transposed`` attributes at wrap time and refuses to wrap when they
#    contradict this, so a future arch with the other layout is skipped loudly
#    instead of being fake-quantized against the wrong GGUF scheme.
#  * GGUF splits the fused HF parameter into SEPARATE per-projection tensors
#    ``blk.N.ffn_{gate,up,down}_exps.weight``, each ``[E, out, in]`` with the
#    same row width as the HF slice. Verified against the real artifact: the
#    Qwen3.6-35B-A3B budget run's ``search_results.json`` carries exactly
#    ``blk.{0..40}.ffn_{gate,up,down}_exps.weight`` (41 blocks = 40 decoder
#    layers + 1 MTP layer), never a fused ``ffn_gate_up_exps`` name.
#  * Granite-hybrid style ``block_sparse_moe.input_linear`` is the exception:
#    ``magicquant.gguf.source`` maps it to a single fused
#    ``ffn_gate_up_exps.weight``, so it gets ONE segment, not two.
#
# A tensor whose GGUF name can't be resolved (e.g. the MTP block, whose GGUF
# block index is ``num_hidden_layers + k`` and therefore not derivable from the
# HF name alone) still yields segments -- with ``gguf_name=None``, so the caller
# falls back to the *group* scheme (X for experts) rather than skipping the
# tensor entirely.


@dataclass(frozen=True)
class ExpertSegment:
    """One GGUF tensor's worth of a fused 3-D expert parameter.

    ``[start, stop)`` slices the parameter's **dim 1** (the out-features axis);
    ``gguf_name`` is the GGUF tensor that slice is written as, or ``None`` when
    the name can't be resolved (caller should fall back to the group scheme).
    """

    gguf_name: Optional[str]
    start: int
    stop: int


def _gate_up_halves(gate_name: Optional[str], up_name: Optional[str], d1: int):
    """Split ``[0, d1)`` into the gate half then the up half.

    Returns ``[]`` for an odd ``d1``: a fused gate+up parameter always has an
    even out dimension, so an odd one means the name matched something that
    isn't the layout this mapping assumes. Refusing beats silently emitting
    lopsided halves (``d1 // 2`` would happily produce 3 + 4).
    """
    if d1 % 2 != 0:
        return []
    half = d1 // 2
    return [
        ExpertSegment(gate_name, 0, half),
        ExpertSegment(up_name, half, d1),
    ]


# (regex over the prefix-normalized HF parameter name) -> builder taking the
# match and the parameter's dim-1 size, returning the segment list.
_FUSED_EXPERT_PATTERNS = [
    # Fused gate+up: two GGUF tensors, contiguous halves of the out axis.
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$"),
        lambda m, d1: _gate_up_halves(
            f"blk.{m.group(1)}.ffn_gate_exps.weight",
            f"blk.{m.group(1)}.ffn_up_exps.weight",
            d1,
        ),
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$"),
        lambda m, d1: [
            ExpertSegment(f"blk.{m.group(1)}.ffn_down_exps.weight", 0, d1)
        ],
    ),
    # Some architectures keep the three projections as separate 3-D parameters.
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_proj$"),
        lambda m, d1: [
            ExpertSegment(f"blk.{m.group(1)}.ffn_gate_exps.weight", 0, d1)
        ],
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.up_proj$"),
        lambda m, d1: [
            ExpertSegment(f"blk.{m.group(1)}.ffn_up_exps.weight", 0, d1)
        ],
    ),
    # Granite-hybrid: source.py maps input_linear to ONE fused GGUF tensor.
    (
        re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.input_linear\.weight$"),
        lambda m, d1: [
            ExpertSegment(f"blk.{m.group(1)}.ffn_gate_up_exps.weight", 0, d1)
        ],
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.output_linear\.weight$"),
        lambda m, d1: [
            ExpertSegment(f"blk.{m.group(1)}.ffn_down_exps.weight", 0, d1)
        ],
    ),
    # Multi-token-prediction blocks: same fused shape, but the GGUF block index
    # is num_hidden_layers + k, which the HF name alone doesn't carry. Emit
    # segments with no GGUF name so the caller uses the group scheme.
    (
        re.compile(r"^mtp\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$"),
        lambda m, d1: _gate_up_halves(None, None, d1),
    ),
    (
        re.compile(r"^mtp\.layers\.(\d+)\.mlp\.experts\.down_proj$"),
        lambda m, d1: [ExpertSegment(None, 0, d1)],
    ),
]


def _strip_decoder_prefix(name: str) -> str:
    """Normalize multimodal decoder prefixes to the plain ``model.layers.`` form.

    Mirrors ``_hf_name_to_gguf``'s own prefix stripping so both name paths agree
    on what ``model.language_model.layers.3...`` means.
    """
    for prefix in ("model.language_model.", "language_model."):
        if name.startswith(prefix):
            return "model." + name[len(prefix):]
    return name


def fused_expert_segments(
    hf_param_name: str, shape: Sequence[int]
) -> Optional[List[ExpertSegment]]:
    """Map a fused 3-D MoE expert parameter to the GGUF tensors it becomes.

    Args:
        hf_param_name: parameter path from ``model.named_parameters()``, e.g.
            ``model.language_model.layers.3.mlp.experts.gate_up_proj``. Unlike
            the 2-D path these are raw ``nn.Parameter``s, so there is no
            ``.weight`` suffix to add or strip.
        shape: the parameter's shape. Must be 3-D ``[E, out, in]``.

    Returns:
        A list of :class:`ExpertSegment` covering ``[0, shape[1])`` contiguously
        and in order, or ``None`` if the name isn't a known fused expert
        parameter (or the shape isn't 3-D / can't be split as the pattern
        requires -- e.g. a gate+up fusion with an odd out dimension).
    """
    if not hf_param_name or len(shape) != 3:
        return None
    d1 = int(shape[1])
    name = _strip_decoder_prefix(hf_param_name)
    for pattern, build in _FUSED_EXPERT_PATTERNS:
        m = pattern.match(name)
        if not m:
            continue
        segments = build(m, d1)
        # A gate+up split needs an even out dimension; anything that doesn't
        # tile [0, d1) exactly is a mapping bug, not something to guess through.
        if not segments or segments[0].start != 0 or segments[-1].stop != d1:
            return None
        if any(s.stop <= s.start for s in segments):
            return None
        if any(a.stop != b.start for a, b in zip(segments, segments[1:])):
            return None
        return segments
    return None


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
