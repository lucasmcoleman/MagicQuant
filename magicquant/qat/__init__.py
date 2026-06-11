"""MagicQuant QAT — Quantization-Aware Training (QAT-LoRA) package.

Differentiable per-group hybrid fake-quant (with a straight-through estimator)
plus the QAT-LoRA wrapping/training that compensates for MagicQuant's chosen
per-group hybrid quant config. Heavy training deps live in the optional ``[qat]``
extra; importing this package only requires ``torch``.

v1 public surface (kernels): ``fake_quant``, ``FakeQuantSTE``, ``SCHEME_FAKE_QUANT``.
(``QATLinear``, ``wrap_model``, ``run_qat`` land in later tasks.)
"""

from magicquant.qat.fake_quant import (  # noqa: F401
    FakeQuantSTE,
    SCHEME_FAKE_QUANT,
    fake_quant,
)

__all__ = ["fake_quant", "FakeQuantSTE", "SCHEME_FAKE_QUANT"]
