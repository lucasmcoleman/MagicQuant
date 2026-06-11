"""Per-group QAT application: ``QATLinear`` + ``wrap_model``.

``QATLinear`` replaces an ``nn.Linear``: it freezes the original ("base") weight,
adds trainable LoRA ``A``/``B`` adapters, and on every forward fake-quantizes the
**merged** weight ``W + scaling·(B @ A)`` to the group's ggml scheme. Training
therefore sees exactly the quantized weight that will ship, and gradients (via the
straight-through estimator in ``fake_quant``) update only the small LoRA adapters
that compensate for the quantization error.

``wrap_model`` walks a model's ``nn.Linear`` modules, maps each to its GGUF tensor
name (``hf_to_ggml_name``), classifies it into a tensor group
(``TensorGroupClassifier``), looks up the group's scheme, and swaps in a
``QATLinear`` for that scheme. Groups that are BF16 (passthrough) or absent from
the scheme map are left untouched (no quant-awareness needed / wanted there).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from magicquant.qat.fake_quant import fake_quant
from magicquant.qat.names import hf_to_ggml_name


class QATLinear(nn.Module):
    """A ``nn.Linear`` whose merged base+LoRA weight is fake-quantized each forward.

    Trainable params: ``lora_A`` (r×in), ``lora_B`` (out×r). The base weight is a
    frozen (``requires_grad=False``) parameter, so it never accumulates a gradient.
    ``lora_B`` is zero-initialized so the initial output equals
    ``fake_quant(base) @ xᵀ`` (the adapter starts as a no-op).
    """

    def __init__(
        self,
        base_weight: torch.Tensor,
        ggml_type_name: str,
        lora_r: int,
        lora_alpha: float,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        out_features, in_features = base_weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.ggml_type_name = ggml_type_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r

        # Frozen base weight (kept as a non-trainable Parameter so it appears in
        # state_dict / .to(device) moves with the module but never trains).
        self.base_weight = nn.Parameter(
            base_weight.detach().clone(), requires_grad=False
        )
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA adapters: W_eff = base + scaling · (B @ A).
        # A: (r, in) kaiming-init; B: (out, r) zero-init -> adapter starts as no-op.
        self.lora_A = nn.Parameter(torch.empty(lora_r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        ggml_type_name: str,
        lora_r: int,
        lora_alpha: float,
    ) -> "QATLinear":
        """Build a ``QATLinear`` from an existing ``nn.Linear``."""
        return cls(
            base_weight=linear.weight.data,
            ggml_type_name=ggml_type_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            bias=linear.bias.data if linear.bias is not None else None,
        )

    def merged_weight(self) -> torch.Tensor:
        """The effective FP weight before fake-quant: base + scaled LoRA delta."""
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return self.base_weight + delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.merged_weight()
        w_fq = fake_quant(w_eff, self.ggml_type_name)
        return F.linear(x, w_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"ggml_type_name={self.ggml_type_name!r}, lora_r={self.lora_r}, "
            f"lora_alpha={self.lora_alpha}"
        )


# LoRA rank/alpha are uniform across wrapped layers in v1; run_qat overrides them
# via the keyword args below (the positional signature stays
# ``wrap_model(model, scheme_by_group, classifier)`` per the plan).
_DEFAULT_LORA_R = 32
_DEFAULT_LORA_ALPHA = 64


def wrap_model(
    model: nn.Module,
    scheme_by_group: Dict[str, str],
    classifier,
    *,
    lora_r: int = _DEFAULT_LORA_R,
    lora_alpha: float = _DEFAULT_LORA_ALPHA,
) -> nn.Module:
    """Replace routed ``nn.Linear`` modules with ``QATLinear`` in place.

    For each ``nn.Linear``: map its module path to a GGUF tensor name, classify it
    into a tensor group, look up the group's ggml scheme. If the scheme is present
    and not BF16 (passthrough), swap the module for a ``QATLinear``. Unmapped
    names, unrouted groups, and BF16 groups are left untouched.

    ``lora_r``/``lora_alpha`` are keyword-only (uniform across wrapped layers).

    Returns the same ``model`` (mutated in place).
    """
    # Collect first; mutating during named_modules() iteration is unsafe.
    to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        ggml_name = hf_to_ggml_name(name)
        if ggml_name is None:
            continue
        group = classifier.classify_tensor(ggml_name)
        scheme = scheme_by_group.get(group)
        if not scheme or scheme == "BF16":
            continue
        to_replace.append((name, module, scheme))

    for name, module, scheme in to_replace:
        qat = QATLinear.from_linear(
            module,
            ggml_type_name=scheme,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        _set_submodule(model, name, qat)

    return model


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Set ``root.<dotted_name>`` to ``new_module`` (handles ModuleList indices)."""
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)
