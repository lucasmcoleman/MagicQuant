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

``wrap_model`` also covers what the Linear walk structurally cannot see: FUSED
3-D MoE expert parameters (``mlp.experts.gate_up_proj`` and friends), which are
raw ``nn.Parameter``s on a plain module and are ~93% of a modern MoE's weights.
Those are handed to ``magicquant.qat.expert_wrap.wrap_fused_experts``, which
parametrizes the Parameter itself. Set ``wrap_experts=False`` to keep the old
Linear-only behaviour.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from magicquant.qat.expert_wrap import MODE_LIVE, wrap_fused_experts
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

        # Eval-time bake: ``fake_quant(merged_weight)`` precomputed once so a
        # no-grad perplexity pass skips the per-forward fake-quant. ``None`` =
        # live (training) path. Not a Parameter/buffer: it's a transient cache.
        self._baked_weight: Optional[torch.Tensor] = None

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
        # Eval fast-path: if a weight has been baked (see ``bake_for_eval``), the
        # fake-quant of the merged weight was precomputed once — just matmul it.
        if self._baked_weight is not None:
            return F.linear(x, self._baked_weight.to(x.dtype), self.bias)
        # Build the merged weight and fake-quant it in fp32 — the block-scale math
        # needs the range, and this lets the frozen base be bf16 (large models) while
        # the LoRA adapters + optimizer state stay fp32. Cast back to the input dtype
        # for the matmul so a bf16 activation path works end to end.
        w_eff = self.base_weight.float() + self.scaling * (self.lora_B.float() @ self.lora_A.float())
        w_fq = fake_quant(w_eff, self.ggml_type_name)
        return F.linear(x, w_fq.to(x.dtype), self.bias)

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


# Per-expert LoRA is a separate knob: 256 experts x 40 layers means the rank is
# multiplied by ~10k, not by the layer count, so the Linear default would be
# wildly out of scale. See expert_wrap.estimate_expert_qat_cost.
_DEFAULT_EXPERT_LORA_R = 4
_DEFAULT_EXPERT_LORA_ALPHA = 8.0


def wrap_model(
    model: nn.Module,
    scheme_by_group: Dict[str, str],
    classifier,
    *,
    lora_r: int = _DEFAULT_LORA_R,
    lora_alpha: float = _DEFAULT_LORA_ALPHA,
    scheme_by_tensor: Optional[Dict[str, str]] = None,
    wrap_experts: bool = True,
    expert_lora_r: int = _DEFAULT_EXPERT_LORA_R,
    expert_lora_alpha: float = _DEFAULT_EXPERT_LORA_ALPHA,
    expert_quant_mode: str = MODE_LIVE,
) -> nn.Module:
    """Wrap a model's quantized weights for QAT, in place.

    ``nn.Linear`` modules: map the module path to a GGUF tensor name, resolve its
    ggml scheme (per-tensor map first, then the module's tensor group), and swap
    in a ``QATLinear``. Unmapped names, unrouted groups, and BF16 schemes are
    left untouched. Linears whose weight isn't 2-D are skipped (a fused 3-D
    "Linear" is an expert stack, handled below, not a matrix ``QATLinear`` can
    take apart).

    Fused 3-D MoE expert parameters: delegated to
    ``expert_wrap.wrap_fused_experts`` (parametrization on the Parameter, with
    its own LoRA rank/alpha and quant mode). Disable with ``wrap_experts=False``.

    ``scheme_by_tensor`` (``{gguf_tensor_name: ggml_type_name}``, from a search
    run's ``tensor_config``) takes precedence over ``scheme_by_group`` wherever
    it names a tensor -- a budget build's per-tensor map really does disagree
    with its own group projection.

    Returns the same ``model`` (mutated in place).
    """
    # Collect first; mutating during named_modules() iteration is unsafe.
    to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if getattr(module, "weight", None) is None or module.weight.ndim != 2:
            continue
        ggml_name = hf_to_ggml_name(name)
        if ggml_name is None:
            continue
        scheme = None
        if scheme_by_tensor:
            scheme = scheme_by_tensor.get(ggml_name)
        if scheme is None:
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

    if wrap_experts:
        wrap_fused_experts(
            model,
            scheme_by_group,
            classifier,
            scheme_by_tensor,
            lora_r=expert_lora_r,
            lora_alpha=expert_lora_alpha,
            mode=expert_quant_mode,
        )

    return model


class _BakeHandle:
    """Toggle returned by :func:`bake_for_eval`.

    Holds the baked ``QATLinear`` modules so the fake-quant fast-path can be torn
    down (``restore``). Doubles as a context manager: ``with bake_for_eval(m):``
    bakes on enter and restores on exit.
    """

    def __init__(self, modules):
        self._modules = list(modules)

    def restore(self) -> None:
        """Drop the baked weights — every QATLinear goes back to the live path."""
        for m in self._modules:
            m._baked_weight = None

    def __enter__(self) -> "_BakeHandle":
        return self

    def __exit__(self, *exc) -> bool:
        self.restore()
        return False


@torch.no_grad()
def bake_for_eval(target: nn.Module) -> _BakeHandle:
    """Precompute ``fake_quant(merged_weight)`` for each QATLinear and swap its
    forward to a plain ``F.linear`` on that baked weight.

    For a no-grad perplexity eval this avoids re-running the (expensive)
    fake-quant kernel on every forward — the shipped weight is the same for the
    whole eval, so quantize it once. The baked output equals the live QATLinear
    output (the same ``fake_quant(merged_weight)`` is used), just without the
    per-call recompute.

    ``target`` may be a single ``QATLinear`` or any module containing them. Call
    the returned handle's ``restore()`` (or use it as a context manager) to drop
    the bake and return to the live (training) path.
    """
    if isinstance(target, QATLinear):
        modules = [target]
    else:
        modules = [m for m in target.modules() if isinstance(m, QATLinear)]
    for m in modules:
        w_eff = m.base_weight.float() + m.scaling * (m.lora_B.float() @ m.lora_A.float())
        m._baked_weight = fake_quant(w_eff, m.ggml_type_name).detach()
    return _BakeHandle(modules)


@torch.no_grad()
def merge_qat_adapters(model: nn.Module) -> nn.Module:
    """Replace each ``QATLinear`` with a plain ``nn.Linear`` holding the merged
    (base + scaled LoRA) weight — **not** fake-quantized.

    This is the handoff into MagicQuant's real pack: the exact ggml quantization
    of the merged weight happens later in ``magicquant generate`` (byte-identical
    to llama.cpp), so the merged Linear must carry the full-precision merged
    weight, not the fake-quant approximation used during training. Bias is
    preserved; the result is a vanilla module a standard HF save/convert handles.

    Mutates ``model`` in place and returns it.
    """
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            to_replace.append((name, module))

    for name, module in to_replace:
        merged = module.merged_weight().detach()
        linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        )
        linear = linear.to(device=merged.device, dtype=merged.dtype)
        linear.weight = nn.Parameter(merged.clone(), requires_grad=False)
        if module.bias is not None:
            linear.bias = nn.Parameter(
                module.bias.detach().clone().to(merged.dtype), requires_grad=False
            )
        _set_submodule(model, name, linear)

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
