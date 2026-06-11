"""MagicQuant QAT — Quantization-Aware Training (QAT-LoRA) package.

Differentiable per-group hybrid fake-quant (with a straight-through estimator)
plus the QAT-LoRA wrapping/training that compensates for MagicQuant's chosen
per-group hybrid quant config. Heavy training deps live in the optional ``[qat]``
extra; importing this package only requires ``torch``.

v1 public surface: ``fake_quant``, ``FakeQuantSTE``, ``SCHEME_FAKE_QUANT``,
``QATLinear``, ``wrap_model``, ``hf_to_ggml_name``, ``load_hybrid_config``.
(``run_qat`` is imported lazily — it pulls in transformers/trl from the ``[qat]``
extra, so it's not imported at package import time.)
"""

from magicquant.qat.fake_quant import (  # noqa: F401
    FakeQuantSTE,
    SCHEME_FAKE_QUANT,
    fake_quant,
)
from magicquant.qat.config import load_hybrid_config  # noqa: F401
from magicquant.qat.names import hf_to_ggml_name  # noqa: F401
from magicquant.qat.wrap import QATLinear, wrap_model  # noqa: F401

__all__ = [
    "fake_quant",
    "FakeQuantSTE",
    "SCHEME_FAKE_QUANT",
    "QATLinear",
    "wrap_model",
    "hf_to_ggml_name",
    "load_hybrid_config",
]
