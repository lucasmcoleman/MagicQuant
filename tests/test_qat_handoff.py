"""QATLinear eval/handoff helpers: bake_for_eval + merge_qat_adapters, plus the
bf16-base/fp32-LoRA forward path.

``bake_for_eval`` precomputes ``fake_quant(merged_weight)`` once and swaps each
QATLinear's forward to a plain ``F.linear`` on the baked weight, so a no-grad
perplexity eval doesn't pay the per-forward fake-quant. ``merge_qat_adapters``
replaces each QATLinear with a plain ``nn.Linear`` whose weight is the *un*-fake-
quantized merged weight (base + scaled LoRA) — the handoff path into MagicQuant's
real ggml pack.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from magicquant.qat.wrap import (
    QATLinear,
    wrap_model,
    bake_for_eval,
    merge_qat_adapters,
)
from magicquant.qat.fake_quant import fake_quant
from magicquant.gguf.tensor_groups import TensorGroupClassifier


def _toy_model():
    class Toy(nn.Module):
        def __init__(s):
            super().__init__()
            s.model = nn.Module()
            s.model.layers = nn.ModuleList([nn.Module()])
            s.model.layers[0].self_attn = nn.Module()
            s.model.layers[0].self_attn.q_proj = nn.Linear(16, 16, bias=False)
            s.model.layers[0].mlp = nn.Module()
            s.model.layers[0].mlp.up_proj = nn.Linear(16, 16, bias=True)

    return Toy()


def _wrapped_toy(lora_kick=0.05):
    m = _toy_model()
    wrap_model(m, {"Q": "Q6_K", "U": "MXFP4"}, TensorGroupClassifier(),
               lora_r=4, lora_alpha=8)
    # Kick LoRA off zero-init so merged != base (exercises the real adapter path).
    for mod in m.modules():
        if isinstance(mod, QATLinear):
            with torch.no_grad():
                mod.lora_B.add_(lora_kick)
                mod.lora_A.add_(lora_kick)
    return m


# ── bake_for_eval ──────────────────────────────────────────────────────────────

def test_bake_for_eval_matches_live_qatlinear():
    base = nn.Linear(32, 16, bias=False)
    q = QATLinear.from_linear(base, "Q6_K", lora_r=4, lora_alpha=8)
    with torch.no_grad():
        q.lora_B.add_(0.05)  # non-trivial adapter
    x = torch.randn(3, 32)
    live = q(x)
    bake_for_eval(q)
    baked = q(x)
    assert torch.allclose(baked, live, atol=1e-4)


def test_bake_for_eval_over_model_then_restore():
    m = _wrapped_toy()
    x = torch.randn(2, 16)
    live = m.model.layers[0].mlp.up_proj(x)

    handle = bake_for_eval(m)  # bake the whole model
    baked = m.model.layers[0].mlp.up_proj(x)
    assert torch.allclose(baked, live, atol=1e-4)

    # context-manager / restore toggle: after restore, forward is live again
    handle.restore()
    restored = m.model.layers[0].mlp.up_proj(x)
    assert torch.allclose(restored, live, atol=1e-4)


def test_bake_for_eval_context_manager():
    q = QATLinear.from_linear(nn.Linear(32, 32, bias=False), "Q6_K", 4, 8)
    with torch.no_grad():
        q.lora_B.add_(0.05)
    x = torch.randn(2, 32)
    live = q(x)
    with bake_for_eval(q):
        assert q._baked_weight is not None
        assert torch.allclose(q(x), live, atol=1e-4)
    # outside the with-block the bake is dropped (live fake-quant again)
    assert q._baked_weight is None
    assert torch.allclose(q(x), live, atol=1e-4)


def test_bake_for_eval_no_per_forward_fakequant(monkeypatch):
    """Baked forward must NOT call fake_quant (that's the whole point)."""
    q = QATLinear.from_linear(nn.Linear(32, 32, bias=False), "Q6_K", 4, 8)
    bake_for_eval(q)

    import magicquant.qat.wrap as wrap_mod
    calls = {"n": 0}
    real = wrap_mod.fake_quant

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(wrap_mod, "fake_quant", _counting)
    q(torch.randn(2, 32))
    assert calls["n"] == 0


# ── merge_qat_adapters ─────────────────────────────────────────────────────────

def test_merge_qat_adapters_replaces_with_plain_linear():
    m = _wrapped_toy()
    merge_qat_adapters(m)
    qp = m.model.layers[0].self_attn.q_proj
    up = m.model.layers[0].mlp.up_proj
    assert isinstance(qp, nn.Linear) and not isinstance(qp, QATLinear)
    assert isinstance(up, nn.Linear) and not isinstance(up, QATLinear)
    assert qp.weight.shape == (16, 16)
    assert up.bias is not None  # bias preserved


def test_merge_qat_adapters_weight_is_unquantized_merged():
    """The merged Linear weight must equal QATLinear.merged_weight() (base+scaled
    LoRA), NOT the fake-quantized weight — the real ggml pack happens later."""
    m = _wrapped_toy()
    q = m.model.layers[0].self_attn.q_proj
    expected = q.merged_weight().detach().clone()
    fq = fake_quant(q.merged_weight(), q.ggml_type_name).detach().clone()

    merge_qat_adapters(m)
    merged_linear = m.model.layers[0].self_attn.q_proj
    assert torch.allclose(merged_linear.weight, expected, atol=1e-6)
    # and it is NOT the fake-quantized weight (merged != fq for a real adapter)
    assert not torch.allclose(merged_linear.weight, fq, atol=1e-4)


def test_merge_qat_adapters_forward_matches_premerge():
    """Merged plain Linear reproduces the un-fake-quantized merged forward."""
    m = _wrapped_toy()
    q = m.model.layers[0].self_attn.q_proj
    x = torch.randn(2, 16)
    pre = x @ q.merged_weight().T  # bias=False on q_proj

    merge_qat_adapters(m)
    post = m.model.layers[0].self_attn.q_proj(x)
    assert torch.allclose(post, pre, atol=1e-5)


def test_merge_qat_adapters_preserves_bias_path():
    m = _wrapped_toy()
    up = m.model.layers[0].mlp.up_proj
    with torch.no_grad():
        up.bias.copy_(torch.arange(16, dtype=up.bias.dtype))
    x = torch.zeros(1, 16)
    merge_qat_adapters(m)
    out = m.model.layers[0].mlp.up_proj(x).squeeze(0)
    assert torch.allclose(out, torch.arange(16, dtype=out.dtype), atol=1e-5)


# ── bf16 base + fp32 LoRA forward ──────────────────────────────────────────────

def test_qatlinear_bf16_base_fp32_lora_finite():
    base = nn.Linear(32, 16, bias=False)
    q = QATLinear.from_linear(base, "Q6_K", lora_r=4, lora_alpha=8)
    # large-model layout: frozen base in bf16, LoRA adapters stay fp32
    q.base_weight.data = q.base_weight.data.to(torch.bfloat16)
    assert q.lora_A.dtype == torch.float32 and q.lora_B.dtype == torch.float32
    with torch.no_grad():
        q.lora_B.add_(0.05)
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    out = q(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out.float()).all()
    assert out.dtype == torch.bfloat16  # output follows the activation dtype


def test_qatlinear_bf16_base_bake_and_merge_finite():
    base = nn.Linear(32, 16, bias=False)
    q = QATLinear.from_linear(base, "Q6_K", lora_r=4, lora_alpha=8)
    q.base_weight.data = q.base_weight.data.to(torch.bfloat16)
    with torch.no_grad():
        q.lora_B.add_(0.05)
    # merged_weight must promote to fp32 and stay finite even with a bf16 base
    mw = q.merged_weight()
    assert torch.isfinite(mw.float()).all()
