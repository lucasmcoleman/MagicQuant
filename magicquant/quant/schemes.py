"""
Quantization Schemes - Single source of truth for scheme metadata.

Each QuantizationScheme carries all attributes consumers (predictor, survival,
probing, orchestrator) need. This is the canonical registry — no other module
should hold parallel scheme dicts.

Bits-per-weight values are computed from the actual ggml block format:
  bpw = (block_bytes * 8) / block_elements

Noise factors are calibrated against published perplexity benchmarks
across Llama / Qwen / Mistral architectures. (PR0 keeps the existing
heuristic values; PR1 will replace them with empirically-benched values
from tools/calibrate_noise_factors.py.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


SchemeCategory = Literal["k_quant", "iq_quant", "legacy_q", "float", "mxfp4"]


@dataclass(frozen=True)
class QuantizationScheme:
    """Canonical metadata for one quantization scheme.

    Consumers read attributes off this class instead of maintaining parallel
    lookup dicts. New schemes are added by appending to the registry below;
    no consumer-side changes required for static metadata.
    """

    name: str                         # MagicQuant identifier ("Q4_K_M", "MXFP4_MOE", ...)
    ggml_type_name: str               # ggml block type ("Q4_K", "MXFP4", ...)
    ggml_type_id: int                 # numeric ggml type enum (used by ctypes binding in PR1+)
    bits_per_weight: float            # actual storage bpw from ggml block format
    noise_factor: float               # relative quantization noise (lower = better quality)
    speed_multiplier: float = 1.0     # relative inference speed vs BF16
    category: SchemeCategory = "k_quant"
    is_moe_optimized: bool = False
    requires_imatrix: bool = False    # IQ-quants benefit from importance matrices
    min_for_group_class: Dict[str, str] = field(default_factory=dict)
    upgrade_neighbor: Optional[str] = None    # name of next-better scheme
    downgrade_neighbor: Optional[str] = None  # name of next-smaller scheme

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to BF16 (16 bits)."""
        return 16.0 / self.bits_per_weight

    def __repr__(self) -> str:
        return f"QuantScheme({self.name}, {self.bits_per_weight}bpw, noise={self.noise_factor})"


# ── Registry ─────────────────────────────────────────────────────────
# NOTE: ggml_type_id values verified against ggml.h. PR1 adds the ctypes
# binding that uses these IDs.

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
    downgrade_neighbor=None,  # bottom of current registry; PR1 adds Q3_K
)


_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
}

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
