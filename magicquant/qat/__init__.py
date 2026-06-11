"""MagicQuant QAT — Quantization-Aware Training (QAT-LoRA) package.

Differentiable per-group hybrid fake-quant (with a straight-through estimator)
plus the QAT-LoRA wrapping/training that compensates for MagicQuant's chosen
per-group hybrid quant config.

Import policy: the **pure-Python** helpers (``hf_to_ggml_name``,
``load_hybrid_config``) import with no extra deps, so ``import magicquant.qat``
works in the core environment without the ``[qat]`` extra. The **torch-dependent**
surface (``fake_quant``, ``FakeQuantSTE``, ``SCHEME_FAKE_QUANT``, ``QATLinear``,
``wrap_model``, ``run_qat``) is loaded lazily on first access — so a torch-less
environment can still import the package and its pure submodules.
"""

from magicquant.qat.config import load_hybrid_config  # noqa: F401  (pure)
from magicquant.qat.names import hf_to_ggml_name  # noqa: F401  (pure)

__all__ = [
    "fake_quant",
    "FakeQuantSTE",
    "SCHEME_FAKE_QUANT",
    "QATLinear",
    "wrap_model",
    "bake_for_eval",
    "merge_qat_adapters",
    "run_qat",
    "hf_to_ggml_name",
    "load_hybrid_config",
]

# Torch-dependent names → (module, attribute), imported only on first access so
# the package (and its pure submodules) import without torch.
_LAZY = {
    "fake_quant": ("magicquant.qat.fake_quant", "fake_quant"),
    "FakeQuantSTE": ("magicquant.qat.fake_quant", "FakeQuantSTE"),
    "SCHEME_FAKE_QUANT": ("magicquant.qat.fake_quant", "SCHEME_FAKE_QUANT"),
    "QATLinear": ("magicquant.qat.wrap", "QATLinear"),
    "wrap_model": ("magicquant.qat.wrap", "wrap_model"),
    "bake_for_eval": ("magicquant.qat.wrap", "bake_for_eval"),
    "merge_qat_adapters": ("magicquant.qat.wrap", "merge_qat_adapters"),
    "run_qat": ("magicquant.qat.train", "run_qat"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
