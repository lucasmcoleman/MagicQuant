"""QATLinear (fake-quant the merged base+LoRA) + wrap_model group routing.

QATLinear freezes the base weight, adds trainable LoRA A/B, and fake-quantizes
the *merged* weight every forward (so training sees exactly what ships).
wrap_model walks a model's nn.Linear modules, maps each to its GGUF tensor name,
classifies it into a tensor group, and swaps in a QATLinear for the group's
scheme.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from magicquant.qat.wrap import QATLinear, wrap_model
from magicquant.qat.fake_quant import fake_quant
from magicquant.gguf.tensor_groups import TensorGroupClassifier


def test_qatlinear_forward_shape_and_trainables():
    base = nn.Linear(16, 8, bias=False)
    q = QATLinear.from_linear(base, ggml_type_name="Q8_0", lora_r=4, lora_alpha=8)
    x = torch.randn(2, 16)
    assert q(x).shape == (2, 8)
    trainable = [n for n, p in q.named_parameters() if p.requires_grad]
    assert set(trainable) == {"lora_A", "lora_B"}  # base frozen
    assert q.base_weight.requires_grad is False


def test_qatlinear_fakequants_merged_weight():
    base = nn.Linear(32, 32, bias=False)
    q = QATLinear.from_linear(base, "Q8_0", lora_r=4, lora_alpha=8)
    # with zero LoRA (lora_B is zero-init), output == fake_quant(base) @ x
    x = torch.randn(1, 32)
    expected = x @ fake_quant(base.weight, "Q8_0").T
    assert torch.allclose(q(x), expected, atol=1e-4)


def test_qatlinear_lora_b_zero_init_so_initial_is_base():
    base = nn.Linear(8, 8, bias=False)
    q = QATLinear.from_linear(base, "Q8_0", lora_r=4, lora_alpha=8)
    assert torch.count_nonzero(q.lora_B) == 0
    # lora_A is non-trivially initialized so a gradient can flow once B != 0
    assert torch.count_nonzero(q.lora_A) > 0


def test_qatlinear_preserves_bias():
    base = nn.Linear(8, 4, bias=True)
    with torch.no_grad():
        base.bias.copy_(torch.arange(4, dtype=torch.float32))
    q = QATLinear.from_linear(base, "Q8_0", lora_r=2, lora_alpha=4)
    x = torch.zeros(1, 8)
    # zero input -> output is just the bias (LoRA/base contribute nothing)
    assert torch.allclose(q(x).squeeze(0), torch.arange(4, dtype=torch.float32), atol=1e-5)


def test_qatlinear_gradients_reach_only_lora():
    base = nn.Linear(8, 8, bias=False)
    q = QATLinear.from_linear(base, "Q8_0", lora_r=4, lora_alpha=8)
    # perturb lora_B so the adapter actually contributes a gradient path
    with torch.no_grad():
        q.lora_B.add_(0.01)
    x = torch.randn(3, 8)
    q(x).sum().backward()
    assert q.lora_A.grad is not None and torch.isfinite(q.lora_A.grad).all()
    assert q.lora_B.grad is not None and torch.isfinite(q.lora_B.grad).all()
    assert q.base_weight.grad is None  # frozen base never accumulates grad


def test_wrap_model_routes_groups():
    class Toy(nn.Module):
        def __init__(s):
            super().__init__()
            s.model = nn.Module()
            s.model.layers = nn.ModuleList([nn.Module()])
            s.model.layers[0].self_attn = nn.Module()
            s.model.layers[0].self_attn.q_proj = nn.Linear(8, 8, bias=False)
            s.model.layers[0].mlp = nn.Module()
            s.model.layers[0].mlp.up_proj = nn.Linear(8, 8, bias=False)

    m = Toy()
    scheme_by_group = {"Q": "Q6_K", "U": "MXFP4"}
    wrap_model(m, scheme_by_group, TensorGroupClassifier())
    assert isinstance(m.model.layers[0].self_attn.q_proj, QATLinear)
    assert m.model.layers[0].self_attn.q_proj.ggml_type_name == "Q6_K"
    assert isinstance(m.model.layers[0].mlp.up_proj, QATLinear)
    assert m.model.layers[0].mlp.up_proj.ggml_type_name == "MXFP4"


def test_wrap_model_skips_unmapped_and_bf16_groups():
    class Toy(nn.Module):
        def __init__(s):
            super().__init__()
            s.model = nn.Module()
            s.model.layers = nn.ModuleList([nn.Module()])
            s.model.layers[0].self_attn = nn.Module()
            s.model.layers[0].self_attn.q_proj = nn.Linear(8, 8, bias=False)
            s.model.layers[0].self_attn.k_proj = nn.Linear(8, 8, bias=False)
            # an unmappable linear that classify won't route from a group present
            s.classifier_head = nn.Linear(8, 2, bias=False)

    m = Toy()
    # Q -> BF16 (passthrough, skip); K not in the map (skip)
    scheme_by_group = {"Q": "BF16"}
    wrap_model(m, scheme_by_group, TensorGroupClassifier())
    assert not isinstance(m.model.layers[0].self_attn.q_proj, QATLinear)
    assert not isinstance(m.model.layers[0].self_attn.k_proj, QATLinear)
    assert not isinstance(m.classifier_head, QATLinear)


def test_wrap_model_preserves_base_weights():
    base_w = None

    class Toy(nn.Module):
        def __init__(s):
            super().__init__()
            s.model = nn.Module()
            s.model.layers = nn.ModuleList([nn.Module()])
            s.model.layers[0].self_attn = nn.Module()
            s.model.layers[0].self_attn.q_proj = nn.Linear(8, 8, bias=False)

    m = Toy()
    base_w = m.model.layers[0].self_attn.q_proj.weight.detach().clone()
    wrap_model(m, {"Q": "Q8_0"}, TensorGroupClassifier())
    q = m.model.layers[0].self_attn.q_proj
    assert torch.allclose(q.base_weight, base_w, atol=0)
