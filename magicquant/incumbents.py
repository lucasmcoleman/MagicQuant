"""Incumbent configs — llama.cpp's own per-tensor mixtures, expressed as
MagicQuant per-group scheme assignments.

Real-world motivation (Qwopus3.6-27B-v2-MTP run,
``output/Qwopus3.6-27B-v2-MTP-GGUF/magicquant/search_results.json``): the
Q4-tier search winner (PPL 6.8353, 16.7GB) LOST to plain ``llama-quantize
--Q4_K_M`` (PPL 6.7022, 15.7GB) because the evolutionary search never
happened to generate anything shaped like stock Q4_K_M -- the predictor's
un-calibrated noise factors favored IQ layouts, so no stock-shaped candidate
ever got measured. Seeding the search with the actual incumbent mixtures
(this module) guarantees every measured run at least tries -- and records a
real measurement for -- "what llama.cpp would have done anyway", so the
search can only do better than stock, never silently lose to it.

Every mapping below is read directly out of llama.cpp's own quantizer
(``llama_tensor_get_type_impl`` in ``src/llama-quant.cpp``, the function
that decides each tensor's ggml type for a given ``--Q4_K_M``/``--Q5_K_M``/
``--Q6_K`` ftype) -- not guessed. Line numbers cited below are from the
version of that file read while writing this module; re-verify against
your checked-out llama.cpp if the upstream logic has since changed.

Category -> MagicQuant group mapping used throughout this module (see
``magicquant.gguf.tensor_groups.TensorGroupClassifier`` for the GGUF-name
side of the same grouping):

    llama.cpp ``tensor_category``           MagicQuant group
    ------------------------------           ----------------
    TOKEN_EMBD                               E (token_embd.weight)
    OUTPUT                                   H (output.weight / lm_head)
    ATTENTION_Q                              Q (attn_q.weight)
    ATTENTION_K, ATTENTION_V*                 K (attn_k.weight + attn_v.weight)
    ATTENTION_OUTPUT                          O (attn_output.weight)
    FFN_UP, FFN_GATE                          U (ffn_up / ffn_gate, dense
                                               AND MoE-expert-stacked: llama.cpp
                                               categorizes by a substring match
                                               on the tensor name -- see FFN_GATE
                                               note below -- so ``ffn_up_exps``/
                                               ``ffn_gate_exps`` land in the same
                                               category as the dense tensors)
    FFN_DOWN                                  D (ffn_down.weight; ffn_down_exps
                                               likewise falls in this category)
    (none -- see FFN_GATE note)               R (router / ffn_gate_inp)
    n/a (MoE experts share U/D/FFN_GATE)      X (mirrors U/D's dominant pick)
    n/a (SSM/Mamba has no stock analog)       S (mirrors D's choice, per the
                                               task spec -- llama.cpp has no
                                               quantization-mixture concept for
                                               linear-attention/SSM state)

    *category_is_attn_v() (line 153) treats ATTENTION_V, ATTENTION_QKV, and
    ATTENTION_KV_B as one bucket ("attn_v-like tensors"); MagicQuant's K group
    covers attn_k.weight + attn_v.weight, so it inherits attn_v's per-tensor
    outcome for the common (non-fused-qkv) case.

    FFN_GATE note: ``tensor_get_category`` (line 115) classifies by
    ``tensor_name.find(...)`` substring search, checked in this order:
    TOKEN_EMBD/OUTPUT exact match, then "attn_qkv.weight", "attn_kv_b.weight",
    "attn_v.weight", "attn_k.weight", "attn_q.weight", "attn_output.weight",
    then "ffn_up", "ffn_gate", "ffn_down" (each a bare substring test, no
    anchoring). The MoE router tensor name MagicQuant matches as R
    (``ffn_gate_inp`` -- see ``TensorGroupClassifier.GROUP_PATTERNS['R']``)
    contains the substring "ffn_gate", so llama.cpp's own quantizer places it
    in the SAME category as FFN_GATE/dense ffn_gate weights, with no
    router-specific ftype branch anywhere in ``llama_tensor_get_type_impl``.
    R therefore inherits U's dominant per-tier scheme below, sourced from
    that same (accidental, upstream) categorization rather than any deliberate
    "router quantization" logic in llama.cpp -- there isn't one.

For each tier, ``default_type`` is the base ggml type before any per-tensor
override (from the ``llama_ftype`` -> ``ggml_type`` switch around line 814):
Q4_K_M -> GGML_TYPE_Q4_K, Q5_K_M -> GGML_TYPE_Q5_K, Q6_K -> GGML_TYPE_Q6_K.
Below, "dominant" means the assignment the majority of tensors in that
category actually receive -- several branches only bump a minority of
layers (``use_more_bits``, roughly the first/last eighth plus every third
middle layer -- about 25%) to a higher scheme; the untouched majority stays
at ``default_type``, which is what this module records as the group's
incumbent scheme. A "slightly conservative" pick was never needed here: in
every branch below, the majority case coincides with the simple, uncontested
choice.

── Q4_K_M (default_type = Q4_K) ──────────────────────────────────────────
  * OUTPUT (H): no ``params->output_tensor_type`` override, arch isn't
    Falcon, row width is 256-divisible, ftype isn't an IQ2/IQ1 family
    member, and ``new_type != Q8_0`` -- so line 457-459's catch-all fires:
    ``new_type = GGML_TYPE_Q6_K``. H = Q6_K regardless of ftype (Q4/Q5/Q6
    all land here identically) whenever the tensor isn't tied to TOKEN_EMBD.
  * TOKEN_EMBD (E): the TOKEN_EMBD branch (line 469) only special-cases
    IQ2/IQ1/IQ3_XXS/TQ1_0/TQ2_0 ftypes -- none of those is Q4_K_M, so
    ``new_type`` falls through unchanged: E = Q4_K (default_type).
  * ATTENTION_Q (Q): the ATTENTION_Q branch (line 561) only special-cases
    IQ3_XS/IQ3_XXS -- Q4_K_M isn't one, so Q = Q4_K (default_type).
  * category_is_attn_v / ATTENTION_K (K): attn_v dominant is default_type
    (line 534's ``use_more_bits`` bump to Q6_K only hits ~25% of layers);
    attn_k has no Q4_K_M-specific branch (line 549 only special-cases
    n_expert==8 / IQ3_XS / IQ3_XXS). K = Q4_K (default_type), dominant.
  * ATTENTION_OUTPUT (O): the non-Falcon branch (line 622-628) only
    special-cases Q2_K/IQ3_XXS/Q3_K_M/Q3_K_L/IQ3_M -- none is Q4_K_M, so
    O = Q4_K (default_type). (The n_expert==8 special case at line 616-621
    bumps to Q5_K for 8-expert models specifically; not the general case.)
  * FFN_UP / FFN_GATE (U) and its MoE-expert/router aliases (X, R): neither
    branch (line 640, 648) special-cases Q4_K_M (only IQ3_XS) -- stays
    default_type. U = X = R = Q4_K.
  * FFN_DOWN (D): line 590-596, non-Falcon path -- ``if
    use_more_bits(i_layer, n_layer): new_type = Q6_K`` else unchanged.
    ~75% of layers stay at default_type. D = Q4_K, dominant.
  * S: no stock analog -- mirrors D's choice per the task spec. S = Q4_K.

── Q5_K_M (default_type = Q5_K) ──────────────────────────────────────────
  * OUTPUT (H): same catch-all as Q4_K_M above (ftype-independent) -> Q6_K.
  * TOKEN_EMBD (E): no Q5_K_M-specific branch -> Q5_K (default_type).
  * ATTENTION_Q (Q): no Q5_K_M-specific branch -> Q5_K (default_type).
  * category_is_attn_v / ATTENTION_K (K): line 534's bump to Q6_K applies to
    Q5_K_M too, same ~25%-of-layers minority; dominant stays Q5_K. attn_k has
    no Q5_K_M branch -> Q5_K. K = Q5_K, dominant.
  * ATTENTION_OUTPUT (O): non-Falcon branch has no Q5_K_M case -> Q5_K.
  * FFN_UP / FFN_GATE (U, X, R): no Q5_K_M case in either branch -> Q5_K.
  * FFN_DOWN (D): line 601, ``ftype == Q5_K_M and use_more_bits(...) ->
    Q6_K`` for the same ~25% minority; majority (~75%) stays Q5_K. D = Q5_K.
  * S: mirrors D. S = Q5_K.

── Q6_K (default_type = Q6_K) ────────────────────────────────────────────
  Scanning every ftype-gated branch in ``llama_tensor_get_type_impl``
  (attn_v/K/Q/O/QKV/FFN_GATE/FFN_UP/FFN_DOWN/TOKEN_EMBD), none of them
  special-cases ``LLAMA_FTYPE_MOSTLY_Q6_K`` at all -- every branch's
  condition list names other ftypes (Q2_K, Q3_K_*, Q4_K_*, Q5_K_M, IQ*,
  TQ*), never Q6_K. So every category (including OUTPUT, whose catch-all
  produces Q6_K anyway) falls straight through to ``default_type`` = Q6_K.
  Q6_K is effectively a uniform quant in llama.cpp's own mixture logic --
  no group deviates from it. All groups = Q6_K.

Caveats:
  * These derivations assume the common case: a dense (or MagicQuant-style
    fused-expert) architecture with separate attn_q/attn_k/attn_v (not a
    fused ``attn_qkv.weight``) and ``n_expert`` not exactly 8 (the 8-expert
    special cases in ATTENTION_OUTPUT/attn_v bump some assignments up,
    e.g. to Q8_0/Q5_K). A fused ATTENTION_QKV tensor gets a DIFFERENT,
    more aggressive bump for both Q4_K_M (-> Q5_K, line 637) and Q5_K_M
    (-> Q6_K, line 638) than the general Q/K-group case documented above;
    since separate q/k/v projections are the common architecture and
    MagicQuant's Q/K groups map to the per-projection tensors, that
    fused-QKV nuance is intentionally NOT reflected in the incumbent
    configs below.
  * "Incumbent" here means "the scheme most of that group's tensors would
    receive from stock llama-quantize" -- it is a coarse per-group
    approximation of a per-tensor-level mixture, exactly the same
    simplification MagicQuant's whole per-group model makes elsewhere.
"""

from typing import Dict, List

from magicquant.quant.schemes import get_scheme_by_name

# All ten MagicQuant tensor groups an incumbent config assigns a scheme to.
# (N and V -- normalization and vision tensors -- are out of scope: MagicQuant
# never varies them, and neither does llama.cpp's per-ftype mixture logic.)
INCUMBENT_GROUPS: List[str] = ["E", "H", "Q", "K", "O", "U", "D", "X", "R", "S"]

# Per-tier incumbent configs. Each dict is {group: scheme_name}, values are
# MagicQuant scheme names (registry keys from magicquant.quant.schemes), not
# raw ggml type names -- e.g. "Q4_K_M" is the scheme wrapping ggml's Q4_K
# block format (see the registry: bits_per_weight=4.5, ggml_type_name="Q4_K").
INCUMBENT_TIERS: Dict[str, Dict[str, str]] = {
    "Q4": {
        "E": "Q4_K_M", "H": "Q6_K", "Q": "Q4_K_M", "K": "Q4_K_M",
        "O": "Q4_K_M", "U": "Q4_K_M", "D": "Q4_K_M", "X": "Q4_K_M",
        "R": "Q4_K_M", "S": "Q4_K_M",
    },
    "Q5": {
        "E": "Q5_K", "H": "Q6_K", "Q": "Q5_K", "K": "Q5_K",
        "O": "Q5_K", "U": "Q5_K", "D": "Q5_K", "X": "Q5_K",
        "R": "Q5_K", "S": "Q5_K",
    },
    "Q6": {g: "Q6_K" for g in INCUMBENT_GROUPS},
}

# Sanity-check the registry at import time: every scheme name referenced
# above must actually exist. Fails loudly at import (not silently at search
# time) if the registry and this module ever drift apart.
for _tier, _config in INCUMBENT_TIERS.items():
    for _group, _scheme in _config.items():
        get_scheme_by_name(_scheme)
del _tier, _config, _group, _scheme


def get_incumbent_config(tier: str) -> Dict[str, str]:
    """Return the incumbent (stock-llama.cpp-approximating) group config for
    a tier ("Q4", "Q5", or "Q6").

    Returns a fresh copy (safe for callers to mutate/restrict-by-group
    without affecting the module-level registry).

    Raises:
        ValueError: ``tier`` isn't one of the known incumbent tiers.
    """
    if tier not in INCUMBENT_TIERS:
        raise ValueError(
            f"Unknown incumbent tier: {tier!r}. Available: {list(INCUMBENT_TIERS)}"
        )
    return dict(INCUMBENT_TIERS[tier])
