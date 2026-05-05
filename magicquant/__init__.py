"""
MagicQuant - Evolutionary Tensor Search for Optimal LLM Compression

A hybrid quantization framework that dynamically groups tensors by architectural role
and employs evolutionary search to find optimal mixed-precision configurations.
"""

__version__ = "0.1.0"

from magicquant.quant.schemes import (
    BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M,
    QuantizationScheme,
    get_scheme_by_name,
    get_all_schemes,
    get_schemes_by_category,
)

from magicquant.gguf.tensor_groups import TensorGroupClassifier

__all__ = [
    "BF16", "Q8_0", "Q6_K", "Q5_K", "IQ4_NL", "MXFP4_MOE", "Q4_K_M",
    "QuantizationScheme",
    "get_scheme_by_name",
    "get_all_schemes",
    "get_schemes_by_category",
    "TensorGroupClassifier",
]
