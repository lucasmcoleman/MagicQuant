"""Differential test: LoRAMergedSource's on-the-fly merge vs PEFT's own merge.

magicquant/gguf/source.py's LoRAMergedSource docstring states the merge
formula is ``W_merged = W_base + (lora_B @ lora_A) * scale`` -- a deliberate
mechanics divergence from PEFT (per-tensor merge-on-read vs PEFT's whole-
model ``merge_and_unload()``) that must stay MATH-equivalent to what PEFT
itself produces, including the scaling convention (plain ``alpha / rank`` vs
rsLoRA's ``alpha / sqrt(rank)``).

Comparison seam: ``LoRAMergedSource.read_tensor_f32()`` end to end (real
``adapter_config.json`` + ``adapter_model.safetensors`` on disk, real
``SafetensorsSource`` base) against PEFT's ``merge_and_unload()`` on an
equivalent toy model -- this exercises the exact code path a MagicQuant run
takes (``open_model_source`` -> ``LoRAMergedSource.__init__`` reading
``use_rslora`` -> ``read_tensor_f32``'s delta computation), not just the
scale arithmetic in isolation. The base uses HF arch ``qwen2`` (NEOX rope,
in ``_ARCH_NO_TRANSFORM_NEEDED``) specifically so ``apply_arch_value_transform``
is a no-op -- no Q/K rope permute, no qwen35 rules -- keeping the comparison
to PEFT honest (a permuted or nonlinearly-transformed base has no PEFT
equivalent to diff against).

INCIDENT (rsLoRA audit, 2026-07-28, now fixed): LoRAMergedSource.__init__
computed ``self._scale = self._alpha / self._rank`` unconditionally, never
reading adapter_config.json's ``use_rslora`` -- the exact same bug already
found and fixed in Foundry's ``fast_export.build_lora_map``
(github.com/lucasmcoleman/Foundry, commit d25f8bc), which trains with
``use_rslora=True`` by default. An unfixed LoRAMergedSource merges LoRA
deltas ``sqrt(rank)``x weaker than PEFT's own merge for any rsLoRA adapter --
e.g. r=4: alpha/r=2.0 vs the correct alpha/sqrt(r)=4.0. Fixed by reading
``use_rslora`` (default False) and branching the scale formula accordingly
(see the incident comment on ``LoRAMergedSource.__init__``); the rsLoRA case
below was confirmed to fail against the unfixed code before the fix landed.
"""

import copy
import json
import math
import struct

import numpy as np
import pytest

# torch comes from the heavy [qat] extra, which CI does not install. A bare
# `import torch` here raised at COLLECTION time, and a collection error aborts
# the entire pytest run rather than skipping one module -- so this single line
# failed the whole suite on every Python version the moment CI could actually
# parse its workflow file. Guard it the way test_qat_train.py already does.
torch = pytest.importorskip("torch")
nn = torch.nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

from magicquant.gguf.source import LoRAMergedSource  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

# Toy dims: hidden=16 (q_proj 16x16), ffn intermediate=32 (down_proj 16x32).
_HIDDEN = 16
_INTERMEDIATE = 32
_N_HEAD = 2


def _write_safetensors(path, tensors):
    """tensors: {name: float32 ndarray} -> minimal .safetensors file.

    Shared shape used both for the base model dir (model.safetensors) and
    the adapter dir (adapter_model.safetensors) -- LoRAMergedSource reads
    both formats identically (SafetensorsSource._parse_header).
    """
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        blob = arr.astype(np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    hdr = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for blob in blobs:
            f.write(blob)


class _ToyBlock(nn.Module):
    """Stand-in for one decoder layer's attention/MLP projections.

    Named so PEFT's key convention
    ("base_model.model.<path>.lora_A.default.weight") and LoRAMergedSource's
    key stripping (drop "base_model.model.", drop ".lora_A.weight") land on
    the same base-model key LoRAMergedSource expects from a real checkpoint,
    e.g. "model.layers.0.self_attn.q_proj.weight" -> "blk.0.attn_q.weight".
    """

    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(_INTERMEDIATE, _HIDDEN, bias=False)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyBlock()])

    def forward(self, x):
        # merge_and_unload() never calls forward(); unused but required for
        # get_peft_model() to treat this as a well-formed nn.Module.
        return x


def _build_toy(seed: int) -> _ToyModel:
    g = torch.Generator().manual_seed(seed)
    model = _ToyModel().float()
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.copy_(
            torch.randn(_HIDDEN, _HIDDEN, generator=g)
        )
        model.model.layers[0].mlp.down_proj.weight.copy_(
            torch.randn(_HIDDEN, _INTERMEDIATE, generator=g)
        )
    return model


def _randomize_lora_weights(peft_model, seed: int) -> None:
    """PEFT zero-inits lora_B by default, so an unperturbed merge would equal
    the base weight regardless of scaling -- give both A and B nonzero random
    values so a scaling error actually shows up in the merged weight."""
    g = torch.Generator().manual_seed(seed)
    for module in peft_model.modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or "default" not in lora_a:
            continue
        with torch.no_grad():
            lora_a["default"].weight.copy_(
                torch.randn(lora_a["default"].weight.shape, generator=g)
            )
            lora_b["default"].weight.copy_(
                torch.randn(lora_b["default"].weight.shape, generator=g)
            )


def _write_base_dir(base_dir, base_model) -> None:
    """Minimal qwen2 (NEOX rope, no value-transform) SafetensorsSource dir."""
    cfg = {
        "model_type": "qwen2",
        "hidden_size": _HIDDEN,
        "num_attention_heads": _N_HEAD,
        "num_key_value_heads": _N_HEAD,
        "num_hidden_layers": 1,
        "intermediate_size": _INTERMEDIATE,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }
    (base_dir / "config.json").write_text(json.dumps(cfg))
    tensors = {
        "model.layers.0.self_attn.q_proj.weight":
            base_model.model.layers[0].self_attn.q_proj.weight.detach().numpy(),
        "model.layers.0.mlp.down_proj.weight":
            base_model.model.layers[0].mlp.down_proj.weight.detach().numpy(),
    }
    _write_safetensors(base_dir / "model.safetensors", tensors)


def _write_adapter_dir(adapter_dir, peft_model, r, alpha, use_rslora) -> None:
    q_layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    d_layer = peft_model.base_model.model.model.layers[0].mlp.down_proj
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            q_layer.lora_A["default"].weight.detach().numpy(),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            q_layer.lora_B["default"].weight.detach().numpy(),
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight":
            d_layer.lora_A["default"].weight.detach().numpy(),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight":
            d_layer.lora_B["default"].weight.detach().numpy(),
    }
    _write_safetensors(adapter_dir / "adapter_model.safetensors", tensors)
    adapter_cfg = {
        "r": r,
        "lora_alpha": alpha,
        "target_modules": ["q_proj", "down_proj"],
        "use_rslora": use_rslora,
        "fan_in_fan_out": False,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg))


def _run_case(tmp_path, r: int, alpha: int, use_rslora: bool, seed: int = 0):
    """Merge a toy model's q_proj/down_proj both ways: LoRAMergedSource
    (real files on disk, real read_tensor_f32 call) vs PEFT's
    merge_and_unload(). Returns ((got_q, ref_q), (got_down, ref_down)) as
    numpy float32 arrays, both shaped like the base weight."""
    base = _build_toy(seed)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "down_proj"],
        use_rslora=use_rslora,
        bias="none",
    )
    peft_model = get_peft_model(copy.deepcopy(base), lora_config)
    _randomize_lora_weights(peft_model, seed + 1)

    # --- PEFT's own ground-truth merge ---
    merged_ref = copy.deepcopy(peft_model).merge_and_unload()
    ref_q = merged_ref.model.layers[0].self_attn.q_proj.weight.detach().numpy()
    ref_down = merged_ref.model.layers[0].mlp.down_proj.weight.detach().numpy()

    # --- LoRAMergedSource's path: real files, real __init__ + read_tensor_f32 ---
    base_dir = tmp_path / f"base_{seed}_{r}_{alpha}_{use_rslora}"
    base_dir.mkdir()
    _write_base_dir(base_dir, base)

    adapter_dir = tmp_path / f"adapter_{seed}_{r}_{alpha}_{use_rslora}"
    adapter_dir.mkdir()
    _write_adapter_dir(adapter_dir, peft_model, r, alpha, use_rslora)

    src = LoRAMergedSource(str(base_dir), str(adapter_dir))
    try:
        got_q = src.read_tensor_f32("blk.0.attn_q.weight").reshape(_HIDDEN, _HIDDEN)
        got_down = src.read_tensor_f32("blk.0.ffn_down.weight").reshape(
            _HIDDEN, _INTERMEDIATE
        )
    finally:
        src.close()

    return (got_q, ref_q), (got_down, ref_down)


@pytest.mark.parametrize(
    "r,alpha",
    [
        (4, 8),    # alpha == 2r
        (8, 8),    # alpha == r (scaling == 1)
        (4, 16),   # alpha == 4r
    ],
)
def test_plain_lora_matches_peft_merge(tmp_path, r, alpha):
    """LoRAMergedSource's scaling = alpha/r must match PEFT's plain-LoRA
    merge exactly (both compute the same alpha/r scaling; this locks the
    full read_tensor_f32 path, not just the scaling formula)."""
    (got_q, ref_q), (got_down, ref_down) = _run_case(tmp_path, r, alpha, use_rslora=False)
    np.testing.assert_allclose(got_q, ref_q, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(got_down, ref_down, atol=1e-5, rtol=1e-5)


def test_rslora_merge_matches_peft(tmp_path):
    """LoRAMergedSource.__init__ now reads adapter_config.json's
    'use_rslora' field and computes scale = alpha / sqrt(rank) for rsLoRA
    adapters, matching PEFT's own merge_and_unload() (see the fix + incident
    note on LoRAMergedSource.__init__ in magicquant/gguf/source.py). Confirmed
    to fail against the pre-fix code (alpha/rank unconditionally) before the
    fix landed."""
    r, alpha = 4, 8  # sqrt(4) = 2, so alpha/r=2.0 vs alpha/sqrt(r)=4.0 -- a
    # large, unmistakable gap, not a rounding-level difference.
    (got_q, ref_q), (got_down, ref_down) = _run_case(tmp_path, r, alpha, use_rslora=True)
    np.testing.assert_allclose(got_q, ref_q, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(got_down, ref_down, atol=1e-5, rtol=1e-5)


def test_rslora_scaling_formula_sanity():
    """Non-xfail control: confirms PEFT really does use alpha/sqrt(r) for
    rsLoRA (so the test above is measuring a real formula gap, not a test
    bug). If PEFT's convention ever changes, this fails independently of the
    merge test above and points straight at the assumption to revisit."""
    r, alpha = 4, 8
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=["q_proj"], use_rslora=True, bias="none",
    )
    base = _build_toy(seed=0)
    peft_model = get_peft_model(copy.deepcopy(base), lora_config)
    layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    expected = alpha / math.sqrt(r)
    assert layer.scaling["default"] == pytest.approx(expected)
    assert layer.scaling["default"] != pytest.approx(alpha / r)
