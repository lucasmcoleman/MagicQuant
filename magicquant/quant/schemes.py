"""
Quantization Schemes - Single source of truth for scheme metadata.

Each QuantizationScheme carries all attributes consumers (predictor, survival,
probing, orchestrator) need. This is the canonical registry — no other module
should hold parallel scheme dicts.

Bits-per-weight values are computed from the actual ggml block format:
  bpw = (block_bytes * 8) / block_elements

Noise factors are HEURISTIC values, ordered to match the published
perplexity ranking across Llama / Qwen / Mistral architectures. They are
NOT yet empirically calibrated: running tools/calibrate_noise_factors.py
(requires llama.cpp + a calibration model, ~2 hr compute) produces
tools/calibration_results.json, after which these values should be replaced
with the measured ppl_loss ratios and the refactor-regression fixture
regenerated (it is seed-pinned to the exact noise values here).
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


SchemeCategory = Literal["k_quant", "iq_quant", "legacy_q", "float", "mxfp4", "rocmfpx"]


@dataclass(frozen=True)
class QuantizationScheme:
    """Canonical metadata for one quantization scheme.

    Consumers read attributes off this class instead of maintaining parallel
    lookup dicts. New schemes are added by appending to the registry below;
    no consumer-side changes required for static metadata.
    """

    name: str                         # MagicQuant identifier ("Q4_K_M", "MXFP4_MOE", ...)
    ggml_type_name: str               # ggml block type ("Q4_K", "MXFP4", ...)
    ggml_type_id: int                 # numeric ggml type enum (used by the ctypes binding)
    bits_per_weight: float            # actual storage bpw from ggml block format
    noise_factor: float               # relative quantization noise (lower = better quality)
    speed_multiplier: float = 1.0     # relative inference speed vs BF16
    category: SchemeCategory = "k_quant"
    is_moe_optimized: bool = False
    requires_imatrix: bool = False    # IQ-quants benefit from importance matrices
    upgrade_neighbor: Optional[str] = None    # name of next-better scheme
    downgrade_neighbor: Optional[str] = None  # name of next-smaller scheme

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to BF16 (16 bits)."""
        return 16.0 / self.bits_per_weight

    def __repr__(self) -> str:
        return f"QuantScheme({self.name}, {self.bits_per_weight}bpw, noise={self.noise_factor})"


# ── Registry ─────────────────────────────────────────────────────────
# NOTE: ggml_type_id values are verified against ggml.h and cross-checked at
# startup by ggml_binding._verify_type_ids (the ctypes binding uses these IDs).

BF16 = QuantizationScheme(
    name="BF16",
    ggml_type_name="BF16",
    ggml_type_id=30,
    bits_per_weight=16.0,
    noise_factor=0.0,
    speed_multiplier=1.0,
    category="float",
    upgrade_neighbor=None,
    downgrade_neighbor="Q8_0",
)

Q8_0 = QuantizationScheme(
    name="Q8_0",
    ggml_type_name="Q8_0",
    ggml_type_id=8,
    bits_per_weight=8.5,
    noise_factor=1.0,
    speed_multiplier=1.75,
    category="legacy_q",
    upgrade_neighbor="BF16",
    downgrade_neighbor="Q6_K",
)

Q6_K = QuantizationScheme(
    name="Q6_K",
    ggml_type_name="Q6_K",
    ggml_type_id=14,
    bits_per_weight=6.5625,
    noise_factor=2.2,
    speed_multiplier=2.2,
    category="k_quant",
    upgrade_neighbor="Q8_0",
    downgrade_neighbor="Q5_K",
)

Q5_K = QuantizationScheme(
    name="Q5_K",
    ggml_type_name="Q5_K",
    ggml_type_id=13,
    bits_per_weight=5.5,
    noise_factor=3.0,
    speed_multiplier=2.7,
    category="k_quant",
    upgrade_neighbor="Q6_K",
    downgrade_neighbor="IQ4_NL",
)

# IQ4_NL: Non-linear lookup table optimized for weight distributions.
# Lower noise than Q4_K_M despite same bpw because the 16 levels are
# learned to minimize quantization error on real weight distributions.
IQ4_NL = QuantizationScheme(
    name="IQ4_NL",
    ggml_type_name="IQ4_NL",
    ggml_type_id=20,
    bits_per_weight=4.5,
    noise_factor=3.8,
    speed_multiplier=3.2,
    category="iq_quant",
    upgrade_neighbor="Q5_K",
    downgrade_neighbor="MXFP4_MOE",
)

# MXFP4: OCP MX Microscaling FP4 (E2M1 values + shared E8M0 exponent).
# Non-uniform FP4 levels (0, 0.5, 1, 1.5, 2, 3, 4, 6) are denser near
# zero, naturally matching the Gaussian-like weight distribution of
# transformers. Lower noise than integer Q4 at slightly better compression.
MXFP4_MOE = QuantizationScheme(
    name="MXFP4_MOE",
    ggml_type_name="MXFP4",
    ggml_type_id=39,
    bits_per_weight=4.25,
    noise_factor=4.0,
    speed_multiplier=3.8,
    category="mxfp4",
    is_moe_optimized=True,
    upgrade_neighbor="IQ4_NL",
    downgrade_neighbor="Q4_K_M",
)

Q4_K_M = QuantizationScheme(
    name="Q4_K_M",
    ggml_type_name="Q4_K",
    ggml_type_id=12,
    bits_per_weight=4.5,
    noise_factor=4.5,
    speed_multiplier=3.4,
    category="k_quant",
    upgrade_neighbor="MXFP4_MOE",
    downgrade_neighbor="Q3_K",
)


# ── Q3_K / Q2_K ──────────────────────────────────────────────────────
# Q3_K and Q2_K make the Q3 tier band reachable. Q2_K bpw=2.625 lands
# at ratio 0.164 — just outside the Q2 band (<=0.16); full Q2 band
# coverage requires sub-Q2 IQ-quants from PR3.
#
# noise_factor values are HEURISTIC (ordered above Q4_K_M), not yet
# calibrated — see the module docstring and tools/calibrate_noise_factors.py.

Q3_K = QuantizationScheme(
    name="Q3_K",
    ggml_type_name="Q3_K",
    ggml_type_id=11,
    bits_per_weight=3.4375,   # 110B * 8 / 256 = 3.4375
    noise_factor=8.0,         # heuristic; calibration pending
    speed_multiplier=4.0,     # ggml SIMD encoders are fast
    category="k_quant",
    upgrade_neighbor="Q4_K_M",
    downgrade_neighbor="Q2_K",
)

Q2_K = QuantizationScheme(
    name="Q2_K",
    ggml_type_name="Q2_K",
    ggml_type_id=10,
    bits_per_weight=2.625,    # 84B * 8 / 256 = 2.625
    noise_factor=15.0,        # heuristic; calibration pending
    speed_multiplier=4.5,     # smallest blocks → fastest dispatch
    category="k_quant",
    upgrade_neighbor="Q3_K",
    downgrade_neighbor=None,  # bottom of current registry; PR3 adds IQ-quants below
)


# ── ROCmFPX fork schemes ─────────────────────────────────────────────
# AMD-native (gfx1151-tuned) tensor formats from the ROCmFPX llama.cpp fork
# (https://github.com/ciru-ai/ROCmFPX). These are OPT-IN: the search only
# considers them when explicitly enabled AND the bound libggml is a ROCmFPX
# build that can encode them (gated in survival.py via the predictor's
# available-scheme set). GGUFs containing these types load only on the fork,
# not stock llama.cpp — documented as a hard caveat.
#
# bpw is the true storage cost from the fork's block structs (block=32):
#   fp3 = 14 B/blk → 3.5 bpw; fp4 = 18 → 4.5; fp6 = 26 → 6.5; fp8 = 33 → 8.25.
# noise_factor values are HEURISTIC, slotted next to the K-quant of matching
# bpw (calibration pending, like Q3_K/Q2_K).

ROCMFP8 = QuantizationScheme(
    name="ROCMFP8",
    ggml_type_name="Q8_0_ROCMFPX",
    ggml_type_id=103,
    bits_per_weight=8.25,
    noise_factor=1.05,          # ~Q8_0
    speed_multiplier=1.8,
    category="rocmfpx",
    upgrade_neighbor="BF16",
    downgrade_neighbor="ROCMFP6",
)

ROCMFP6 = QuantizationScheme(
    name="ROCMFP6",
    ggml_type_name="Q6_0_ROCMFPX",
    ggml_type_id=102,
    bits_per_weight=6.5,
    noise_factor=2.3,           # ~Q6_K
    speed_multiplier=2.3,
    category="rocmfpx",
    upgrade_neighbor="ROCMFP8",
    downgrade_neighbor="ROCMFP4",
)

ROCMFP4 = QuantizationScheme(
    name="ROCMFP4",
    ggml_type_name="Q4_0_ROCMFP4",
    ggml_type_id=100,
    bits_per_weight=4.5,
    noise_factor=4.2,           # ~Q4_K_M / MXFP4 band
    speed_multiplier=3.8,       # AMD-native FP4 kernel is the fork's fast path
    category="rocmfpx",
    is_moe_optimized=True,
    upgrade_neighbor="ROCMFP6",
    downgrade_neighbor="ROCMFP3",
)

ROCMFP3 = QuantizationScheme(
    name="ROCMFP3",
    ggml_type_name="Q3_0_ROCMFPX",
    ggml_type_id=104,
    bits_per_weight=3.5,
    noise_factor=8.5,           # ~Q3_K
    speed_multiplier=4.0,
    category="rocmfpx",
    upgrade_neighbor="ROCMFP4",
    downgrade_neighbor=None,
)


# ── Sub-4-bit IQ-quants ──────────────────────────────────────────────
# Stock-ggml importance-matrix-friendly non-linear quants below IQ4_NL.
# OPT-IN: excluded from the default search pool (see IQ_SCHEME_NAMES below
# and the enable_iq gate in evolution/survival.py) so the default search —
# and its seed-pinned regression fixture — is byte-identical to today.
#
# bpw is the true storage cost from the binding's block/type tables
# (magicquant/quant/ggml_binding.py _GGML_BLOCK_SIZE / _GGML_TYPE_SIZE; all
# block=256 except IQ4_NL's block=32, which is unaffected here):
#   IQ4_XS  = 136B/256  = 4.25   bpw
#   IQ3_S   = 110B/256  = 3.4375 bpw
#   IQ3_XXS =  98B/256  = 3.0625 bpw
#   IQ2_S   =  82B/256  = 2.5625 bpw
#   IQ2_XS  =  74B/256  = 2.3125 bpw
#   IQ2_XXS =  66B/256  = 2.0625 bpw
#   IQ1_M   =  56B/256  = 1.75   bpw
#   IQ1_S   =  50B/256  = 1.5625 bpw
# noise_factor values are HEURISTIC (ordered by published llama.cpp quality),
# consistent with the existing heuristic Q3_K=8.0/Q2_K=15.0 (calibration
# pending, see the module docstring).
#
# requires_imatrix is read from the REAL libggml
# (ggml_quantize_requires_imatrix), not guessed: verified against a stock
# (non-fork) libggml at ~/llama.cpp/build/bin — IQ2_XS, IQ2_XXS, and IQ1_S
# report True; IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, and IQ1_M report False.
# survival.py's enable_iq gate additionally drops every requires_imatrix
# scheme unconditionally (the search threads no imatrix).
#
# CRITICAL: only these NEW schemes reference existing registry entries
# (upward, via upgrade/downgrade_neighbor) — no existing scheme's neighbor
# fields are modified, so the default mutation chain is unchanged.

IQ4_XS = QuantizationScheme(
    name="IQ4_XS",
    ggml_type_name="IQ4_XS",
    ggml_type_id=23,
    bits_per_weight=4.25,
    noise_factor=4.1,
    speed_multiplier=3.3,
    category="iq_quant",
    requires_imatrix=False,
    upgrade_neighbor="MXFP4_MOE",
    downgrade_neighbor="IQ3_S",
)

IQ3_S = QuantizationScheme(
    name="IQ3_S",
    ggml_type_name="IQ3_S",
    ggml_type_id=21,
    bits_per_weight=3.4375,
    noise_factor=6.0,
    speed_multiplier=3.5,
    category="iq_quant",
    requires_imatrix=False,
    upgrade_neighbor="IQ4_XS",
    downgrade_neighbor="IQ3_XXS",
)

IQ3_XXS = QuantizationScheme(
    name="IQ3_XXS",
    ggml_type_name="IQ3_XXS",
    ggml_type_id=18,
    bits_per_weight=3.0625,
    noise_factor=7.5,
    speed_multiplier=3.6,
    category="iq_quant",
    requires_imatrix=False,
    upgrade_neighbor="IQ3_S",
    downgrade_neighbor="Q3_K",
)

IQ2_S = QuantizationScheme(
    name="IQ2_S",
    ggml_type_name="IQ2_S",
    ggml_type_id=22,
    bits_per_weight=2.5625,
    noise_factor=11.0,
    speed_multiplier=3.7,
    category="iq_quant",
    requires_imatrix=False,
    upgrade_neighbor="IQ3_XXS",
    downgrade_neighbor="IQ2_XS",
)

IQ2_XS = QuantizationScheme(
    name="IQ2_XS",
    ggml_type_name="IQ2_XS",
    ggml_type_id=17,
    bits_per_weight=2.3125,
    noise_factor=13.0,
    speed_multiplier=3.8,
    category="iq_quant",
    requires_imatrix=True,
    upgrade_neighbor="IQ2_S",
    downgrade_neighbor="IQ2_XXS",
)

IQ2_XXS = QuantizationScheme(
    name="IQ2_XXS",
    ggml_type_name="IQ2_XXS",
    ggml_type_id=16,
    bits_per_weight=2.0625,
    noise_factor=16.0,
    speed_multiplier=3.9,
    category="iq_quant",
    requires_imatrix=True,
    upgrade_neighbor="IQ2_XS",
    downgrade_neighbor="IQ1_M",
)

IQ1_M = QuantizationScheme(
    name="IQ1_M",
    ggml_type_name="IQ1_M",
    ggml_type_id=29,
    bits_per_weight=1.75,
    noise_factor=24.0,
    speed_multiplier=3.95,
    category="iq_quant",
    requires_imatrix=False,
    upgrade_neighbor="IQ2_XXS",
    downgrade_neighbor="IQ1_S",
)

IQ1_S = QuantizationScheme(
    name="IQ1_S",
    ggml_type_name="IQ1_S",
    ggml_type_id=19,
    bits_per_weight=1.5625,
    noise_factor=30.0,
    speed_multiplier=4.0,
    category="iq_quant",
    requires_imatrix=True,
    upgrade_neighbor="IQ1_M",
    downgrade_neighbor=None,
)


_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
    "Q3_K": Q3_K,
    "Q2_K": Q2_K,
    "ROCMFP8": ROCMFP8,
    "ROCMFP6": ROCMFP6,
    "ROCMFP4": ROCMFP4,
    "ROCMFP3": ROCMFP3,
    "IQ4_XS": IQ4_XS,
    "IQ3_S": IQ3_S,
    "IQ3_XXS": IQ3_XXS,
    "IQ2_S": IQ2_S,
    "IQ2_XS": IQ2_XS,
    "IQ2_XXS": IQ2_XXS,
    "IQ1_M": IQ1_M,
    "IQ1_S": IQ1_S,
}

# ROCmFPX fork scheme names (opt-in; excluded from the default search pool).
ROCMFPX_SCHEME_NAMES = frozenset({"ROCMFP8", "ROCMFP6", "ROCMFP4", "ROCMFP3"})

# Sub-4-bit IQ scheme names (opt-in; excluded from the default search pool).
# Deliberately does NOT include IQ4_NL, which is an existing default-pool
# scheme (see IQ4_NL above) and must stay sampled by default.
IQ_SCHEME_NAMES = frozenset({
    "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
})

# Group-class floors: minimum acceptable scheme per group class.
# "sensitive" (E, H, O, R) shouldn't go below Q8_0; "robust" (U, D, X)
# can go all the way to Q4_K_M (bottom of current registry).
# These were previously in survival.py as _MIN_SCHEME.
_GROUP_CLASS_FLOORS: Dict[str, str] = {
    "sensitive": "Q8_0",
    "robust": "Q4_K_M",
}


def get_scheme_by_name(name: str) -> QuantizationScheme:
    """Look up a scheme by its MagicQuant identifier."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown scheme: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_all_schemes() -> List[QuantizationScheme]:
    """Get all schemes ordered by noise (best quality first)."""
    return sorted(_REGISTRY.values(), key=lambda s: s.noise_factor)


def get_schemes_by_category(category: SchemeCategory) -> List[QuantizationScheme]:
    """Get all schemes in a given category, ordered by noise (best first)."""
    return [s for s in get_all_schemes() if s.category == category]


def get_floor_for_group_class(group_class: str) -> str:
    """Get the minimum acceptable scheme name for a group sensitivity class.

    group_class: "sensitive" or "robust".
    """
    return _GROUP_CLASS_FLOORS[group_class]
