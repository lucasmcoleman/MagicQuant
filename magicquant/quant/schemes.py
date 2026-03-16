"""
Quantization Schemes - Define quantization formats and their properties.

Bits-per-weight values are computed from the actual ggml block format:
  bpw = (block_bytes * 8) / block_elements

Noise factors are calibrated against published perplexity benchmarks
across Llama / Qwen / Mistral architectures.
"""

from typing import Dict, Tuple
import numpy as np


class QuantizationScheme:
    """
    Represents a quantization scheme with its properties.

    Attributes:
        name: Identifier string (e.g., "Q8_0", "MXFP4_MOE")
        bits_per_weight: Actual storage bpw from ggml block format
        noise_factor: Relative quantization noise (lower = better quality)
        speed_multiplier: Relative inference speed vs BF16
        is_moe_optimized: Whether scheme is optimized for MoE
    """

    def __init__(
        self,
        name: str,
        bits_per_weight: float,
        noise_factor: float,
        speed_multiplier: float = 1.0,
        is_moe_optimized: bool = False
    ):
        self.name = name
        self.bits_per_weight = bits_per_weight
        self.noise_factor = noise_factor
        self.speed_multiplier = speed_multiplier
        self.is_moe_optimized = is_moe_optimized

    def __repr__(self):
        return f"QuantScheme({self.name}, {self.bits_per_weight}bpw, noise={self.noise_factor})"

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to BF16 (16 bits)."""
        return 16.0 / self.bits_per_weight


# ── Scheme definitions ────────────────────────────────────────────────
# bpw = (type_size_bytes * 8) / block_size
# noise factors calibrated from published llama.cpp perplexity benchmarks

BF16 = QuantizationScheme(
    name="BF16",
    bits_per_weight=16.0,   # 2B * 8 / 1 = 16.0
    noise_factor=0.0,
    speed_multiplier=1.0
)

Q8_0 = QuantizationScheme(
    name="Q8_0",
    bits_per_weight=8.5,    # 34B * 8 / 32 = 8.5
    noise_factor=1.0,
    speed_multiplier=1.75
)

Q6_K = QuantizationScheme(
    name="Q6_K",
    bits_per_weight=6.5625, # 210B * 8 / 256 = 6.5625
    noise_factor=2.2,
    speed_multiplier=2.2
)

Q5_K = QuantizationScheme(
    name="Q5_K",
    bits_per_weight=5.5,    # 176B * 8 / 256 = 5.5
    noise_factor=3.0,
    speed_multiplier=2.7
)

# IQ4_NL: Non-linear lookup table optimized for weight distributions.
# Lower noise than Q4_K_M despite same bpw because the 16 levels are
# learned to minimize quantization error on real weight distributions.
IQ4_NL = QuantizationScheme(
    name="IQ4_NL",
    bits_per_weight=4.5,    # 18B * 8 / 32 = 4.5
    noise_factor=3.8,
    speed_multiplier=3.2
)

# MXFP4: OCP MX Microscaling FP4 (E2M1 values + shared E8M0 exponent).
# Non-uniform FP4 levels (0, 0.5, 1, 1.5, 2, 3, 4, 6) are denser near
# zero, naturally matching the Gaussian-like weight distribution of
# transformers.  Lower noise than integer Q4 at slightly better compression.
MXFP4_MOE = QuantizationScheme(
    name="MXFP4_MOE",
    bits_per_weight=4.25,   # 17B * 8 / 32 = 4.25
    noise_factor=4.0,
    speed_multiplier=3.8,
    is_moe_optimized=True
)

Q4_K_M = QuantizationScheme(
    name="Q4_K_M",
    bits_per_weight=4.5,    # 144B * 8 / 256 = 4.5
    noise_factor=4.5,
    speed_multiplier=3.4
)


def get_scheme_by_name(name: str) -> QuantizationScheme:
    """Get quantization scheme by its identifier string."""
    schemes = {
        "BF16": BF16,
        "Q8_0": Q8_0,
        "Q6_K": Q6_K,
        "Q5_K": Q5_K,
        "Q4_K_M": Q4_K_M,
        "IQ4_NL": IQ4_NL,
        "MXFP4_MOE": MXFP4_MOE
    }

    if name not in schemes:
        raise ValueError(f"Unknown scheme: {name}. Available: {list(schemes.keys())}")

    return schemes[name]


def get_all_schemes() -> list[QuantizationScheme]:
    """Get all available quantization schemes, ordered by noise (best first)."""
    return [BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M]
