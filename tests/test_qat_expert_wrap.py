"""QAT for FUSED 3-D MoE expert parameters.

A modern MoE stores every routed expert of a projection in one 3-D
``nn.Parameter`` (``mlp.experts.gate_up_proj``), not in per-expert ``nn.Linear``s
-- so ``wrap_model``'s Linear walk cannot see ~93% of a 35B MoE's weights.
``expert_wrap`` parametrizes the Parameter itself, which intercepts every
consumer without touching the MoE forward.

These tests use hand-built tiny modules (a [4, 8, 6] expert stack) and never
load a real model.
"""

import json
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.qat import expert_wrap as expert_wrap_mod
from magicquant.qat.expert_wrap import (
    MODE_FROZEN,
    MODE_LIVE,
    FusedExpertQAT,
    ResolvedSegment,
    estimate_expert_qat_cost,
    expert_cache_disabled,
    fused_expert_adapter_meta,
    fused_expert_adapter_state,
    iter_expert_parametrizations,
    merge_fused_expert_adapters,
    resolve_segment_schemes,
    wrap_fused_experts,
)
from magicquant.qat.fake_quant import fake_quant
from magicquant.qat.names import ExpertSegment, fused_expert_segments

# The tiny fixtures below use a [4, 8, 6] expert stack, whose 6-wide rows are not
# a multiple of any K-quant block. That legitimately trips the row-alignment
# guard (real expert rows are 512/2048 wide and never do), so it's silenced
# module-wide and asserted on directly in test_row_misalignment_warns.
pytestmark = pytest.mark.filterwarnings("ignore:.*is not a multiple of.*")


@pytest.fixture(autouse=True)
def _clean_expert_cache():
    """The expert-weight cache is module-global; don't let it leak across tests."""
    expert_wrap_mod._WEIGHT_CACHE.clear()
    yield
    expert_wrap_mod._WEIGHT_CACHE.clear()


# ── tiny stand-in for transformers' Qwen3_5MoeExperts ────────────────────────

class TinyExperts(nn.Module):
    """A fused 3-D expert stack with the real layout: [E, out, in].

    The forward mirrors transformers' eager MoE path -- index one expert out of
    the fused parameter and run ``F.linear`` -- which is exactly the access
    pattern the parametrization has to survive.
    """

    def __init__(self, n_experts=4, out_features=8, in_features=6):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(n_experts, out_features, in_features) * 0.05
        )
        self.down_proj = nn.Parameter(
            torch.randn(n_experts, in_features, out_features // 2) * 0.05
        )
        # transformers' use_experts_implementation decorator stamps these.
        self.is_concatenated = True
        self.is_transposed = False

    def forward(self, x, expert_idx):
        return torch.nn.functional.linear(x, self.gate_up_proj[expert_idx])


class TinyMoEModel(nn.Module):
    """Two decoder layers whose ``mlp.experts`` hold fused 3-D parameters."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = TinyExperts()
            self.model.layers.append(layer)


def _segments(*schemes):
    """Resolved segments splitting dim 1 (size 8) evenly across ``schemes``."""
    width = 8 // len(schemes)
    return [
        ResolvedSegment(f"blk.0.seg{i}.weight", i * width, (i + 1) * width, s)
        for i, s in enumerate(schemes)
    ]


# ── name mapping ──────────────────────────────────────────────────────────────

def test_gate_up_proj_splits_into_two_gguf_tensors():
    segs = fused_expert_segments(
        "model.language_model.layers.3.mlp.experts.gate_up_proj", (256, 1024, 2048)
    )
    assert [s.gguf_name for s in segs] == [
        "blk.3.ffn_gate_exps.weight",
        "blk.3.ffn_up_exps.weight",
    ]
    # Contiguous halves of the out axis, gate first.
    assert (segs[0].start, segs[0].stop) == (0, 512)
    assert (segs[1].start, segs[1].stop) == (512, 1024)


def test_down_proj_is_one_segment_and_prefixes_normalize():
    for name in (
        "model.language_model.layers.7.mlp.experts.down_proj",
        "language_model.layers.7.mlp.experts.down_proj",
        "model.layers.7.mlp.experts.down_proj",
    ):
        segs = fused_expert_segments(name, (256, 2048, 512))
        assert [s.gguf_name for s in segs] == ["blk.7.ffn_down_exps.weight"]
        assert (segs[0].start, segs[0].stop) == (0, 2048)


def test_mtp_experts_wrap_with_no_gguf_name():
    """MTP block index isn't derivable from the HF name -> group fallback."""
    segs = fused_expert_segments(
        "mtp.layers.0.mlp.experts.gate_up_proj", (256, 1024, 2048)
    )
    assert [s.gguf_name for s in segs] == [None, None]
    assert (segs[0].stop, segs[1].start) == (512, 512)


def test_non_expert_and_wrong_rank_names_do_not_map():
    assert fused_expert_segments("model.layers.0.self_attn.q_proj", (8, 6)) is None
    # 2-D parameter with an expert-shaped name is not a fused expert stack.
    assert fused_expert_segments(
        "model.layers.0.mlp.experts.down_proj", (2048, 512)
    ) is None
    # Odd out dimension can't be split into gate/up halves.
    assert fused_expert_segments(
        "model.layers.0.mlp.experts.gate_up_proj", (4, 7, 6)
    ) is None


# ── scheme resolution ─────────────────────────────────────────────────────────

def test_per_tensor_scheme_beats_group_scheme():
    segs = [
        ExpertSegment("blk.0.ffn_gate_exps.weight", 0, 4),
        ExpertSegment("blk.0.ffn_up_exps.weight", 4, 8),
    ]
    resolved = resolve_segment_schemes(
        segs,
        scheme_by_group={"X": "Q3_K"},
        classifier=TensorGroupClassifier(),
        scheme_by_tensor={"blk.0.ffn_gate_exps.weight": "Q2_K"},
    )
    # gate takes the per-tensor Q2_K; up falls back to the group's Q3_K.
    assert [s.ggml_type_name for s in resolved] == ["Q2_K", "Q3_K"]


def test_unnamed_segment_falls_back_to_expert_group():
    resolved = resolve_segment_schemes(
        [ExpertSegment(None, 0, 8)],
        scheme_by_group={"X": "Q4_K"},
        classifier=TensorGroupClassifier(),
        scheme_by_tensor={},
    )
    assert [s.ggml_type_name for s in resolved] == ["Q4_K"]


# ── the parametrization ───────────────────────────────────────────────────────

def test_forward_is_fake_quant_of_base_plus_lora():
    torch.manual_seed(0)
    experts = TinyExperts()
    base = experts.gate_up_proj.detach().clone()
    p = FusedExpertQAT(
        base.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4, mode=MODE_LIVE
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)

    # Non-zero adapter so the merged weight actually differs from the base.
    with torch.no_grad():
        p.lora_expert_A.normal_(0, 0.1)

    delta = (p.lora_alpha / p.lora_r) * torch.bmm(
        p.lora_expert_A.float(), p.lora_expert_B.float()
    )
    expected = fake_quant(base.float() + delta, "Q8_0")
    assert torch.allclose(experts.gate_up_proj, expected, atol=1e-6)


def test_adapter_starts_as_a_no_op():
    """Zero-init lora_expert_A means the delta is exactly zero at step 0."""
    torch.manual_seed(0)
    experts = TinyExperts()
    base = experts.gate_up_proj.detach().clone()
    p = FusedExpertQAT(
        base.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4, mode=MODE_LIVE
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    assert torch.allclose(experts.gate_up_proj, fake_quant(base.float(), "Q8_0"))


def test_segments_are_fake_quantized_independently():
    """A hybrid where gate and up land in different schemes must not be
    quantized as one tensor -- gate/up really do get separate GGUF entries."""
    torch.manual_seed(0)
    experts = TinyExperts()
    base = experts.gate_up_proj.detach().clone()
    p = FusedExpertQAT(
        base.shape, _segments("Q8_0", "Q2_K"), lora_r=0, lora_alpha=1,
        mode=MODE_LIVE,
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    out = experts.gate_up_proj
    assert torch.allclose(out[:, :4, :], fake_quant(base[:, :4, :].float(), "Q8_0"))
    assert torch.allclose(out[:, 4:, :], fake_quant(base[:, 4:, :].float(), "Q2_K"))


def test_chunking_over_experts_does_not_change_the_result():
    """Chunking bounds peak memory; it must not change a single value.

    Safe here because every fake-quant kernel blocks along the flattened row
    and one expert's slab is a whole number of rows.
    """
    torch.manual_seed(0)
    experts = TinyExperts(n_experts=4, out_features=8, in_features=64)
    base = experts.gate_up_proj.detach().clone()
    segs = [ResolvedSegment("blk.0.x.weight", 0, 8, "Q8_0")]
    whole = FusedExpertQAT(base.shape, segs, lora_r=0, lora_alpha=1, mode=MODE_LIVE)
    chunked = FusedExpertQAT(base.shape, segs, lora_r=0, lora_alpha=1, mode=MODE_LIVE)
    whole.chunk_experts = 4
    chunked.chunk_experts = 1
    assert torch.equal(whole(base), chunked(base))


def test_backward_updates_only_the_adapters():
    torch.manual_seed(0)
    experts = TinyExperts()
    p = FusedExpertQAT(
        experts.gate_up_proj.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    original = experts.parametrizations["gate_up_proj"].original
    original.requires_grad = False

    x = torch.randn(3, 6)
    experts(x, 1).sum().backward()

    assert original.grad is None, "frozen base must not accumulate a gradient"
    assert p.lora_expert_A.grad is not None
    assert p.lora_expert_B.grad is not None
    # Gradient reaches the indexed expert (the STE passes it straight through).
    assert p.lora_expert_A.grad[1].abs().sum() > 0


def test_frozen_mode_quantizes_the_base_once_and_adds_a_live_delta():
    torch.manual_seed(0)
    experts = TinyExperts()
    base = experts.gate_up_proj.detach().clone()
    p = FusedExpertQAT(
        base.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4, mode=MODE_FROZEN
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    p.quantize_base_(experts.parametrizations["gate_up_proj"].original.data)

    q_base = fake_quant(base.float(), "Q8_0")
    assert torch.allclose(
        experts.parametrizations["gate_up_proj"].original, q_base, atol=1e-6
    )
    with torch.no_grad():
        p.lora_expert_A.normal_(0, 0.1)
    delta = (p.lora_alpha / p.lora_r) * torch.bmm(
        p.lora_expert_A.float(), p.lora_expert_B.float()
    )
    # Frozen mode does NOT re-quantize the merged weight -- that's the whole
    # difference from live mode, and what makes it affordable.
    assert torch.allclose(experts.gate_up_proj, q_base + delta, atol=1e-6)


# ── model-level wrapping ──────────────────────────────────────────────────────

def test_wrap_fused_experts_covers_every_layer():
    torch.manual_seed(0)
    model = TinyMoEModel(n_layers=2)
    wrapped = wrap_fused_experts(
        model,
        scheme_by_group={"X": "Q8_0"},
        classifier=TensorGroupClassifier(),
        scheme_by_tensor=None,
        lora_r=2,
        lora_alpha=4,
    )
    # Both gate_up_proj and down_proj of both layers.
    assert len(wrapped) == 4
    assert len(list(iter_expert_parametrizations(model))) == 4
    names = sorted(p.param_name for p in wrapped)
    assert names == [
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.1.mlp.experts.gate_up_proj",
    ]


def test_wrap_refuses_a_transposed_layout():
    """A module that declares the other layout is skipped loudly, not guessed."""
    model = TinyMoEModel(n_layers=1)
    model.model.layers[0].mlp.experts.is_transposed = True
    with pytest.warns(UserWarning, match="is_transposed"):
        wrapped = wrap_fused_experts(
            model,
            scheme_by_group={"X": "Q8_0"},
            classifier=TensorGroupClassifier(),
            lora_r=2,
            lora_alpha=4,
        )
    assert wrapped == []


def test_wrap_skips_passthrough_schemes():
    model = TinyMoEModel(n_layers=1)
    wrapped = wrap_fused_experts(
        model,
        scheme_by_group={"X": "BF16"},
        classifier=TensorGroupClassifier(),
        lora_r=2,
        lora_alpha=4,
    )
    assert wrapped == []


def test_frozen_mode_wrap_quantizes_bases_in_place():
    torch.manual_seed(0)
    model = TinyMoEModel(n_layers=1)
    experts = model.model.layers[0].mlp.experts
    base = experts.gate_up_proj.detach().clone()
    wrap_fused_experts(
        model,
        scheme_by_group={"X": "Q8_0"},
        classifier=TensorGroupClassifier(),
        lora_r=2,
        lora_alpha=4,
        mode=MODE_FROZEN,
    )
    stored = experts.parametrizations["gate_up_proj"].original
    assert not torch.allclose(stored, base)
    # gate and up are SEPARATE GGUF tensors, so each is quantized on its own
    # block grid -- not as one flattened tensor. (With this tiny row width of 6
    # the two differ; at a real 2048-wide row they'd coincide, which is exactly
    # why the tiny case is the one worth asserting on.)
    half = base.shape[1] // 2
    expected = torch.cat(
        [
            fake_quant(base[:, :half, :].float(), "Q8_0"),
            fake_quant(base[:, half:, :].float(), "Q8_0"),
        ],
        dim=1,
    )
    assert torch.allclose(stored, expected, atol=1e-6)


# ── adapter save format (the merge lane's contract) ───────────────────────────

def test_adapter_state_keys_and_shapes_match_the_merge_contract():
    torch.manual_seed(0)
    model = TinyMoEModel(n_layers=1)
    wrap_fused_experts(
        model,
        scheme_by_group={"X": "Q8_0"},
        classifier=TensorGroupClassifier(),
        lora_r=2,
        lora_alpha=4,
    )
    state = fused_expert_adapter_state(model)
    assert set(state) == {
        "model.layers.0.mlp.experts.gate_up_proj.lora_expert_A",
        "model.layers.0.mlp.experts.gate_up_proj.lora_expert_B",
        "model.layers.0.mlp.experts.down_proj.lora_expert_A",
        "model.layers.0.mlp.experts.down_proj.lora_expert_B",
    }
    # Base [E=4, d1=8, d2=6] -> A (E, d1, r), B (E, r, d2); merge does
    # W[e] += scale * (A[e] @ B[e]).
    a = state["model.layers.0.mlp.experts.gate_up_proj.lora_expert_A"]
    b = state["model.layers.0.mlp.experts.gate_up_proj.lora_expert_B"]
    assert tuple(a.shape) == (4, 8, 2)
    assert tuple(b.shape) == (4, 2, 6)
    assert torch.bmm(a, b).shape == (4, 8, 6)
    assert a.dtype == torch.float32 and b.dtype == torch.float32


def test_adapter_state_is_consumable_by_the_merge_shape_check():
    """The saved shapes must satisfy magicquant.qat.merge._apply_3d exactly."""
    from magicquant.qat.merge import _apply_3d

    torch.manual_seed(0)
    model = TinyMoEModel(n_layers=1)
    wrap_fused_experts(
        model,
        scheme_by_group={"X": "Q8_0"},
        classifier=TensorGroupClassifier(),
        lora_r=2,
        lora_alpha=4,
    )
    p = next(iter_expert_parametrizations(model))
    with torch.no_grad():
        p.lora_expert_A.normal_(0, 0.1)
    state = fused_expert_adapter_state(model)
    key = "model.layers.0.mlp.experts.gate_up_proj"
    w = torch.randn(4, 8, 6)
    merged = _apply_3d(
        w,
        state[f"{key}.lora_expert_A"],
        state[f"{key}.lora_expert_B"],
        p.lora_alpha / p.lora_r,
        key,
    )
    expected = w + (p.lora_alpha / p.lora_r) * torch.bmm(
        p.lora_expert_A.detach(), p.lora_expert_B.detach()
    )
    assert torch.allclose(merged, expected, atol=1e-6)


def test_adapter_meta_records_shapes_and_segments():
    model = TinyMoEModel(n_layers=1)
    wrap_fused_experts(
        model,
        scheme_by_group={"X": "Q8_0"},
        classifier=TensorGroupClassifier(),
        lora_r=2,
        lora_alpha=4,
    )
    meta = fused_expert_adapter_meta(model)
    entry = next(m for m in meta if m["param"].endswith("gate_up_proj"))
    assert entry["base_shape"] == [4, 8, 6]
    assert entry["lora_expert_A_shape"] == [4, 8, 2]
    assert entry["lora_expert_B_shape"] == [4, 2, 6]
    assert [s["scheme"] for s in entry["segments"]] == ["Q8_0", "Q8_0"]
    # Must survive a JSON round-trip -- it goes into qat_meta.json.
    assert json.loads(json.dumps(meta)) == meta


# ── checkpoint / resume ───────────────────────────────────────────────────────

def test_expert_adapters_survive_a_lora_only_checkpoint_round_trip():
    """The overnight run's restart safety depends on this.

    ``_install_lora_only_checkpoint_save`` writes just the ``requires_grad``
    parameters and HF reloads with ``strict=False``. Parametrization moves the
    adapters under ``parametrizations.<attr>.0.*``; the names ``named_parameters``
    and ``state_dict`` use must still agree, or a resumed run would silently
    restore nothing for 93% of the model.
    """
    from magicquant.qat.train import _freeze_to_lora_only

    torch.manual_seed(0)
    model = TinyMoEModel(n_layers=1)
    wrap_fused_experts(
        model, {"X": "Q8_0"}, TensorGroupClassifier(), lora_r=2, lora_alpha=4
    )
    _freeze_to_lora_only(model)
    with torch.no_grad():
        for p in iter_expert_parametrizations(model):
            p.lora_expert_A.normal_(0, 0.1)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "expert adapters must be trainable"
    state = model.state_dict()
    assert trainable <= set(state), "trainable names missing from state_dict"

    restored = TinyMoEModel(n_layers=1)
    wrap_fused_experts(
        restored, {"X": "Q8_0"}, TensorGroupClassifier(), lora_r=2, lora_alpha=4
    )
    _, unexpected = restored.load_state_dict(
        {k: v for k, v in state.items() if k in trainable}, strict=False
    )
    assert unexpected == []
    for key in trainable:
        assert torch.equal(state[key], restored.state_dict()[key])


def test_freeze_counts_expert_adapter_elements():
    from magicquant.qat.train import _freeze_to_lora_only

    model = TinyMoEModel(n_layers=1)
    wrap_fused_experts(
        model, {"X": "Q8_0"}, TensorGroupClassifier(), lora_r=2, lora_alpha=4
    )
    # gate_up [4,8,6]: 4*8*2 + 4*2*6 = 112; down [4,6,4]: 4*6*2 + 4*2*4 = 80
    assert _freeze_to_lora_only(model) == 192
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_expert_" in name


# ── merge back into the base ──────────────────────────────────────────────────

def test_merge_writes_the_full_precision_merged_weight():
    """The pack quantizes later, so the merge must NOT bake the fake-quant."""
    torch.manual_seed(0)
    model = TinyMoEModel(n_layers=1)
    experts = model.model.layers[0].mlp.experts
    base = experts.gate_up_proj.detach().clone()
    wrap_fused_experts(
        model,
        scheme_by_group={"X": "Q8_0"},
        classifier=TensorGroupClassifier(),
        lora_r=2,
        lora_alpha=4,
    )
    p = next(iter_expert_parametrizations(model))
    with torch.no_grad():
        p.lora_expert_A.normal_(0, 0.1)
    delta = (p.lora_alpha / p.lora_r) * torch.bmm(
        p.lora_expert_A.detach(), p.lora_expert_B.detach()
    )

    merge_fused_expert_adapters(model)

    assert not parametrize.is_parametrized(experts, "gate_up_proj")
    assert torch.allclose(experts.gate_up_proj, base + delta, atol=1e-6)


# ── cost accounting ───────────────────────────────────────────────────────────

def test_cost_estimate_counts_params_and_bytes():
    model = TinyMoEModel(n_layers=2)
    wrapped = wrap_fused_experts(
        model,
        scheme_by_group={"X": "Q8_0"},
        classifier=TensorGroupClassifier(),
        lora_r=2,
        lora_alpha=4,
    )
    cost = estimate_expert_qat_cost(wrapped)
    # Per layer: gate_up [4,8,6] -> A 4*8*2 + B 4*2*6 = 64+48 = 112
    #            down    [4,6,4] -> A 4*6*2 + B 4*2*4 = 48+32 = 80
    assert cost["lora_params"] == 2 * (112 + 80)
    assert cost["n_expert_tensors"] == 4
    assert cost["base_elements"] == 2 * (4 * 8 * 6 + 4 * 6 * 4)
    # params + grads + 2 AdamW moments, fp32.
    assert cost["adapter_bytes"] == cost["lora_params"] * 4
    assert cost["train_bytes"] == cost["lora_params"] * 4 * 4
    assert cost["train_gib"] == pytest.approx(cost["train_bytes"] / 1024 ** 3)


def test_cost_estimate_scales_live_time_with_scheme_speed():
    """Q2_K is ~40x slower than Q3_K per element; the estimate must show it."""
    shape = (16, 8, 6)
    slow = FusedExpertQAT(
        shape, [ResolvedSegment("a", 0, 8, "Q2_K")], lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    fast = FusedExpertQAT(
        shape, [ResolvedSegment("a", 0, 8, "Q3_K")], lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    t_slow = estimate_expert_qat_cost([slow])["live_forward_seconds"]
    t_fast = estimate_expert_qat_cost([fast])["live_forward_seconds"]
    assert t_slow > 10 * t_fast > 0


def test_frozen_mode_reports_no_live_forward_cost():
    shape = (16, 8, 6)
    p = FusedExpertQAT(
        shape, [ResolvedSegment("a", 0, 8, "Q2_K")], lora_r=2, lora_alpha=4,
        mode=MODE_FROZEN,
    )
    cost = estimate_expert_qat_cost([p])
    assert cost["live_forward_seconds"] == 0.0
    assert cost["n_live_tensors"] == 0
    assert cost["lora_params"] > 0


def test_expert_lora_rank_zero_is_quant_only():
    """r=0 gives fake-quant with no adapter -- useful as an eval baseline."""
    torch.manual_seed(0)
    experts = TinyExperts()
    base = experts.gate_up_proj.detach().clone()
    p = FusedExpertQAT(
        base.shape, _segments("Q8_0"), lora_r=0, lora_alpha=1, mode=MODE_LIVE
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    assert p.lora_expert_A is None
    assert torch.allclose(experts.gate_up_proj, fake_quant(base.float(), "Q8_0"))
    assert fused_expert_adapter_state(experts) == {}


@pytest.mark.filterwarnings("default")
def test_row_misalignment_warns():
    """A row width the scheme can't tile means the fake-quant approximates a
    different partition than the writer will use -- say so, don't assume."""
    with pytest.warns(UserWarning, match="not a multiple of"):
        FusedExpertQAT(
            (4, 8, 6), [ResolvedSegment("a", 0, 8, "Q4_K")], lora_r=0,
            lora_alpha=1, mode=MODE_LIVE, param_name="tiny.experts.down_proj",
        )


@pytest.mark.filterwarnings("error")
def test_real_expert_row_widths_do_not_warn():
    """Qwen3.6-35B-A3B's actual widths (2048 and 512) tile every scheme used."""
    for row_width in (2048, 512):
        FusedExpertQAT(
            (2, 8, row_width),
            [ResolvedSegment("a", 0, 8, "Q2_K"), ResolvedSegment("b", 0, 8, "Q3_K")],
            lora_r=0, lora_alpha=1, mode=MODE_LIVE,
        )


# ── the per-forward cache ─────────────────────────────────────────────────────
#
# The eager MoE forward reads the fused parameter once per HIT EXPERT (up to 256
# times per layer). Without a cache that is 256 full rebuilds of a multi-GB
# weight, which is the difference between a run that finishes and one that
# doesn't -- so the cache's behaviour is asserted, not assumed.

def test_repeated_access_within_a_forward_is_computed_once():
    torch.manual_seed(0)
    experts = TinyExperts()
    p = FusedExpertQAT(
        experts.gate_up_proj.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    first = experts.gate_up_proj
    assert experts.gate_up_proj is first  # cache hit, same tensor object
    assert expert_wrap_mod._WEIGHT_CACHE.hits >= 1


def test_cache_is_invalidated_by_an_optimizer_style_update():
    torch.manual_seed(0)
    experts = TinyExperts()
    p = FusedExpertQAT(
        experts.gate_up_proj.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    before = experts.gate_up_proj.clone()
    with torch.no_grad():
        p.lora_expert_A.add_(0.5)  # what optimizer.step() does, in place
    after = experts.gate_up_proj
    assert not torch.allclose(before, after), "stale cached weight was served"


def test_cache_is_invalidated_by_grad_mode():
    """A value built under no_grad carries no graph and must not be reused."""
    torch.manual_seed(0)
    experts = TinyExperts()
    p = FusedExpertQAT(
        experts.gate_up_proj.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    with torch.no_grad():
        no_grad_w = experts.gate_up_proj
    assert no_grad_w.grad_fn is None
    with_grad_w = experts.gate_up_proj
    assert with_grad_w is not no_grad_w
    assert with_grad_w.grad_fn is not None


def test_cache_is_bounded_and_evicts():
    """Bounded at 2 entries: enough for one layer's gate_up + down, and no
    more, so a 41-layer model never holds 41 materialized expert weights."""
    torch.manual_seed(0)
    models = [TinyExperts() for _ in range(3)]
    ps = []
    for m in models:
        p = FusedExpertQAT(
            m.gate_up_proj.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4,
            mode=MODE_LIVE,
        )
        parametrize.register_parametrization(m, "gate_up_proj", p, unsafe=True)
        ps.append(p)

    first = models[0].gate_up_proj
    _second = models[1].gate_up_proj
    assert models[0].gate_up_proj is first  # still cached (2 entries)
    _third = models[2].gate_up_proj         # evicts the oldest
    assert models[0].gate_up_proj is not first
    assert len(expert_wrap_mod._WEIGHT_CACHE._entries) <= 2


def test_cached_and_uncached_results_agree():
    torch.manual_seed(0)
    experts = TinyExperts()
    p = FusedExpertQAT(
        experts.gate_up_proj.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    with torch.no_grad():
        p.lora_expert_A.normal_(0, 0.1)
    cached = experts.gate_up_proj.detach().clone()
    with expert_cache_disabled():
        uncached = experts.gate_up_proj.detach().clone()
    assert torch.equal(cached, uncached)


def test_gradients_still_flow_through_a_cached_weight():
    """Two consumers of one cached weight must both contribute gradient."""
    torch.manual_seed(0)
    experts = TinyExperts()
    p = FusedExpertQAT(
        experts.gate_up_proj.shape, _segments("Q8_0"), lora_r=2, lora_alpha=4,
        mode=MODE_LIVE,
    )
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    experts.parametrizations["gate_up_proj"].original.requires_grad = False
    x = torch.randn(3, 6)
    (experts(x, 0).sum() + experts(x, 2).sum()).backward()
    assert p.lora_expert_A.grad[0].abs().sum() > 0
    assert p.lora_expert_A.grad[2].abs().sum() > 0
    assert p.lora_expert_A.grad[1].abs().sum() == 0  # expert 1 was never used


# ── the real transformers MoE forward ─────────────────────────────────────────

def test_intercepts_the_real_qwen3_5_moe_experts_forward():
    """The whole design rests on parametrization intercepting the REAL consumer.

    transformers' ``Qwen3_5MoeExperts.forward`` reads ``self.gate_up_proj[e]``
    inside a per-expert loop -- exactly the fused 3-D shape the 35B model has.
    """
    modeling = pytest.importorskip(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    )
    configuration = pytest.importorskip(
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe"
    )
    cfg = configuration.Qwen3_5MoeTextConfig(
        hidden_size=32, moe_intermediate_size=16, num_experts=4,
        num_experts_per_tok=2, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=64,
    )
    cfg._experts_implementation = "eager"
    experts = modeling.Qwen3_5MoeExperts(cfg)

    # The layout facts names.py depends on, asserted against the real class.
    assert tuple(experts.gate_up_proj.shape) == (4, 32, 32)  # [E, 2*I, H]
    assert tuple(experts.down_proj.shape) == (4, 32, 16)     # [E, H, I]
    assert experts.is_concatenated is True
    assert experts.is_transposed is False

    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.05)
        experts.down_proj.normal_(0, 0.05)
    base = experts.gate_up_proj.detach().clone()

    x = torch.randn(6, 32)
    top_k_index = torch.randint(0, 4, (6, 2))
    top_k_weights = torch.rand(6, 2)
    reference = experts(x, top_k_index, top_k_weights)

    segs = [
        ResolvedSegment("blk.0.ffn_gate_exps.weight", 0, 16, "Q8_0"),
        ResolvedSegment("blk.0.ffn_up_exps.weight", 16, 32, "Q8_0"),
    ]
    p = FusedExpertQAT(base.shape, segs, lora_r=2, lora_alpha=4, mode=MODE_LIVE)
    parametrize.register_parametrization(experts, "gate_up_proj", p, unsafe=True)
    original = experts.parametrizations["gate_up_proj"].original
    original.requires_grad = False

    out = experts(x, top_k_index, top_k_weights)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, reference), "fake-quant never reached the forward"

    out.sum().backward()
    assert original.grad is None
    # A@B with A zero-init: the gradient lands on A first (B's is A^T @ dL, = 0
    # at step 0). Mirrors QATLinear's zero-init B; not a bug.
    assert p.lora_expert_A.grad.abs().sum() > 0

    expected = torch.cat(
        [
            fake_quant(base[:, :16, :].float(), "Q8_0"),
            fake_quant(base[:, 16:, :].float(), "Q8_0"),
        ],
        dim=1,
    )
    assert torch.allclose(experts.gate_up_proj[2], expected[2], atol=1e-6)


def test_kaiming_init_is_finite_and_nonzero():
    p = FusedExpertQAT(
        (4, 8, 6), _segments("Q8_0"), lora_r=2, lora_alpha=4, mode=MODE_LIVE
    )
    assert torch.all(p.lora_expert_A == 0)
    assert torch.isfinite(p.lora_expert_B).all()
    assert p.lora_expert_B.abs().sum() > 0
    assert p.scaling == pytest.approx(4 / 2)
    assert math.isclose(p.lora_alpha, 4.0)
