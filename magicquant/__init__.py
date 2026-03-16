"""
MagicQuant - Evolutionary Tensor Search for Optimal LLM Compression

A hybrid quantization framework that dynamically groups tensors by architectural role
and employs evolutionary search to find optimal mixed-precision configurations.
"""

__version__ = "0.1.0"

from magicquant.quant.schemes import (
    Q8_0, Q6_K, Q5_K, Q4_K_M, IQ4_NL, MXFP4_MOE, BF16
)

from magicquant.gguf.tensor_groups import TensorGroupClassifier

__all__ = [
    "Q8_0", 
    "Q6_K", 
    "Q5_K", 
    "Q4_K_M", 
    "IQ4_NL", 
    "MXFP4_MOE", 
    "BF16",
    "TensorGroupClassifier"
]