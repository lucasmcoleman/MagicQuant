"""magicquant.qat.merge: the streaming, on-disk base+LoRA merge.

Fixes the 2026-08-05 incident where Foundry's run_budget_qat.sh called
core.fast_export.streaming_merge against a MagicQuant adapter dir: that
function needs a PEFT adapter_config.json MagicQuant never writes, and
filters on the substring ".lora_A." (a trailing dot -- PEFT's
"...lora_A.weight" naming) while MagicQuant's own _save_adapters writes keys
ending in exactly ".lora_A" (no dot, no ".weight"). Even past the missing
config, that name mismatch would have matched zero keys and silently emitted
an unmodified copy of the base model.

These tests build a tiny hand-rolled safetensors model (2 shards) + a fake
adapter dir directly in MagicQuant's own on-disk format, so they exercise
exactly the shapes/keys _save_adapters actually writes -- no PEFT anywhere.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file, save_file

from magicquant.qat.merge import QATMergeError, merge_qat_adapters


# ── fixture builders ─────────────────────────────────────────────────────────

# Q_PROJ: a plain nn.Linear-shaped 2-D weight, gets a 2-D LoRA adapter.
_Q_KEY = "model.layers.0.self_attn.q_proj.weight"
_Q_OUT, _Q_IN = 8, 8

# DOWN_PROJ / EMBED: no adapter targets these -- must come out byte-identical.
_DOWN_KEY = "model.layers.0.mlp.down_proj.weight"
_EMBED_KEY = "model.embed_tokens.weight"

# EXPERTS: a fused 3-D MoE-expert parameter (no ".weight" suffix, matching
# Qwen3.6-35B-A3B's model.language_model.layers.N.mlp.experts.gate_up_proj),
# gets a 3-D "lora_expert_A/B" adapter.
_EXPERTS_KEY = "model.language_model.layers.0.mlp.experts.gate_up_proj"
_E, _EI, _EO = 4, 6, 10

_R, _ALPHA = 2, 4.0  # scale = alpha / r = 2.0


def _write_base_model(tmp_path, rng):
    """Two safetensors shards + model.safetensors.index.json + config.json."""
    model_dir = tmp_path / "base"
    model_dir.mkdir()

    shard1 = {
        _Q_KEY: torch.tensor(rng.standard_normal((_Q_OUT, _Q_IN)), dtype=torch.float32),
        _DOWN_KEY: torch.tensor(rng.standard_normal((8, 16)), dtype=torch.float32),
    }
    shard2 = {
        _EXPERTS_KEY: torch.tensor(rng.standard_normal((_E, _EI, _EO)), dtype=torch.float32),
        _EMBED_KEY: torch.tensor(rng.standard_normal((10, 8)), dtype=torch.float32),
    }
    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (model_dir / "config.json").write_text(json.dumps({"model_type": "toy"}))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"toy": True}))

    return model_dir, {**shard1, **shard2}


def _write_adapters(tmp_path, rng, *, include_2d=True, include_3d=False,
                     lora_r=_R, lora_alpha=_ALPHA, extra_state=None,
                     meta_overrides=None):
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()

    state = {}
    if include_2d:
        base = _Q_KEY[: -len(".weight")]
        state[f"{base}.lora_A"] = torch.tensor(
            rng.standard_normal((lora_r, _Q_IN)), dtype=torch.float32
        )
        state[f"{base}.lora_B"] = torch.tensor(
            rng.standard_normal((_Q_OUT, lora_r)), dtype=torch.float32
        )
    if include_3d:
        state[f"{_EXPERTS_KEY}.lora_expert_A"] = torch.tensor(
            rng.standard_normal((_E, _EI, lora_r)), dtype=torch.float32
        )
        state[f"{_EXPERTS_KEY}.lora_expert_B"] = torch.tensor(
            rng.standard_normal((_E, lora_r, _EO)), dtype=torch.float32
        )
    if extra_state:
        state.update(extra_state)

    save_file(state, str(adapter_dir / "adapter_model.safetensors"))

    meta = {
        "model": "toy", "lora_r": lora_r, "lora_alpha": lora_alpha,
        "adapter_file": "adapter_model.safetensors",
    }
    if meta_overrides:
        meta.update(meta_overrides)
    (adapter_dir / "qat_meta.json").write_text(json.dumps(meta))

    return adapter_dir, state


# ── 2-D merge round-trip: exact math ─────────────────────────────────────────

def test_2d_merge_matches_w_plus_scale_ba_exactly(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)

    model_dir, base_tensors = _write_base_model(tmp_path, rng)
    adapter_dir, adapter_state = _write_adapters(tmp_path, rng, include_2d=True)
    out_dir = tmp_path / "merged"

    result = merge_qat_adapters(str(model_dir), str(adapter_dir), str(out_dir))
    assert result == str(out_dir)

    merged = load_file(str(out_dir / "model-00001-of-00002.safetensors"))
    a = adapter_state["model.layers.0.self_attn.q_proj.lora_A"]
    b = adapter_state["model.layers.0.self_attn.q_proj.lora_B"]
    scale = _ALPHA / _R
    expected = base_tensors[_Q_KEY] + scale * (b @ a)

    assert torch.allclose(merged[_Q_KEY], expected, atol=1e-6)
    # never coincidentally equal to the un-merged base
    assert not torch.allclose(merged[_Q_KEY], base_tensors[_Q_KEY], atol=1e-6)


def test_tensors_without_adapters_are_byte_identical(tmp_path):
    import numpy as np
    rng = np.random.default_rng(1)

    model_dir, base_tensors = _write_base_model(tmp_path, rng)
    adapter_dir, _ = _write_adapters(tmp_path, rng, include_2d=True)
    out_dir = tmp_path / "merged"

    merge_qat_adapters(str(model_dir), str(adapter_dir), str(out_dir))

    merged1 = load_file(str(out_dir / "model-00001-of-00002.safetensors"))
    merged2 = load_file(str(out_dir / "model-00002-of-00002.safetensors"))

    assert torch.equal(merged1[_DOWN_KEY], base_tensors[_DOWN_KEY])
    assert torch.equal(merged2[_EMBED_KEY], base_tensors[_EMBED_KEY])
    # untouched-by-2D-only-run 3-D tensor is also untouched
    assert torch.equal(merged2[_EXPERTS_KEY], base_tensors[_EXPERTS_KEY])


def test_config_and_tokenizer_files_are_copied(tmp_path):
    import numpy as np
    rng = np.random.default_rng(2)

    model_dir, _ = _write_base_model(tmp_path, rng)
    adapter_dir, _ = _write_adapters(tmp_path, rng, include_2d=True)
    out_dir = tmp_path / "merged"

    merge_qat_adapters(str(model_dir), str(adapter_dir), str(out_dir))

    assert json.loads((out_dir / "config.json").read_text()) == {"model_type": "toy"}
    assert (out_dir / "tokenizer_config.json").is_file()
    assert (out_dir / "model.safetensors.index.json").is_file()


# ── 3-D expert-key path (declared format, no lane writes it yet) ────────────

def test_3d_expert_merge_matches_batched_bmm_exactly(tmp_path):
    import numpy as np
    rng = np.random.default_rng(3)

    model_dir, base_tensors = _write_base_model(tmp_path, rng)
    adapter_dir, adapter_state = _write_adapters(
        tmp_path, rng, include_2d=False, include_3d=True
    )
    out_dir = tmp_path / "merged"

    merge_qat_adapters(str(model_dir), str(adapter_dir), str(out_dir))

    merged = load_file(str(out_dir / "model-00002-of-00002.safetensors"))
    a = adapter_state[f"{_EXPERTS_KEY}.lora_expert_A"]
    b = adapter_state[f"{_EXPERTS_KEY}.lora_expert_B"]
    scale = _ALPHA / _R
    expected = base_tensors[_EXPERTS_KEY] + scale * torch.bmm(a, b)

    assert torch.allclose(merged[_EXPERTS_KEY], expected, atol=1e-6)
    # the 2-D tensor is untouched when only the 3-D lane is exercised
    merged1 = load_file(str(out_dir / "model-00001-of-00002.safetensors"))
    assert torch.equal(merged1[_Q_KEY], base_tensors[_Q_KEY])


def test_2d_and_3d_merge_together(tmp_path):
    import numpy as np
    rng = np.random.default_rng(4)

    model_dir, base_tensors = _write_base_model(tmp_path, rng)
    adapter_dir, adapter_state = _write_adapters(
        tmp_path, rng, include_2d=True, include_3d=True
    )
    out_dir = tmp_path / "merged"

    merge_qat_adapters(str(model_dir), str(adapter_dir), str(out_dir))

    merged1 = load_file(str(out_dir / "model-00001-of-00002.safetensors"))
    merged2 = load_file(str(out_dir / "model-00002-of-00002.safetensors"))
    scale = _ALPHA / _R

    a2 = adapter_state["model.layers.0.self_attn.q_proj.lora_A"]
    b2 = adapter_state["model.layers.0.self_attn.q_proj.lora_B"]
    assert torch.allclose(merged1[_Q_KEY], base_tensors[_Q_KEY] + scale * (b2 @ a2), atol=1e-6)

    a3 = adapter_state[f"{_EXPERTS_KEY}.lora_expert_A"]
    b3 = adapter_state[f"{_EXPERTS_KEY}.lora_expert_B"]
    expected3 = base_tensors[_EXPERTS_KEY] + scale * torch.bmm(a3, b3)
    assert torch.allclose(merged2[_EXPERTS_KEY], expected3, atol=1e-6)


# ── fails loudly ──────────────────────────────────────────────────────────────

def test_missing_adapter_dir_fails_loudly(tmp_path):
    import numpy as np
    rng = np.random.default_rng(5)
    model_dir, _ = _write_base_model(tmp_path, rng)

    with pytest.raises(FileNotFoundError):
        merge_qat_adapters(str(model_dir), str(tmp_path / "nope"), str(tmp_path / "out"))


def test_missing_qat_meta_fails_loudly(tmp_path):
    import numpy as np
    rng = np.random.default_rng(6)
    model_dir, _ = _write_base_model(tmp_path, rng)
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    save_file({"x": torch.zeros(1)}, str(adapter_dir / "adapter_model.safetensors"))

    with pytest.raises(FileNotFoundError, match="qat_meta.json"):
        merge_qat_adapters(str(model_dir), str(adapter_dir), str(tmp_path / "out"))


def test_adapter_file_with_no_lora_keys_fails_loudly(tmp_path):
    """An adapter dir with a meta file but zero usable lora key pairs must
    refuse rather than silently emit an unmodified copy of the base model --
    this is the exact bug class the trailing-dot PEFT-key mismatch caused."""
    import numpy as np
    rng = np.random.default_rng(7)
    model_dir, _ = _write_base_model(tmp_path, rng)
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    save_file({"unrelated.tensor": torch.zeros(1)},
              str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "qat_meta.json").write_text(
        json.dumps({"lora_r": 2, "lora_alpha": 4, "adapter_file": "adapter_model.safetensors"})
    )

    with pytest.raises(QATMergeError, match="no lora_A/lora_B"):
        merge_qat_adapters(str(model_dir), str(adapter_dir), str(tmp_path / "out"))


def test_lora_a_without_matching_lora_b_fails_loudly(tmp_path):
    import numpy as np
    rng = np.random.default_rng(8)
    model_dir, _ = _write_base_model(tmp_path, rng)
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    base = _Q_KEY[: -len(".weight")]
    save_file({f"{base}.lora_A": torch.zeros(_R, _Q_IN)},
              str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "qat_meta.json").write_text(
        json.dumps({"lora_r": _R, "lora_alpha": _ALPHA, "adapter_file": "adapter_model.safetensors"})
    )

    with pytest.raises(QATMergeError, match="no matching"):
        merge_qat_adapters(str(model_dir), str(adapter_dir), str(tmp_path / "out"))


def test_adapter_target_not_in_base_model_fails_loudly(tmp_path):
    import numpy as np
    rng = np.random.default_rng(9)
    model_dir, _ = _write_base_model(tmp_path, rng)
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    save_file(
        {
            "model.layers.99.self_attn.q_proj.lora_A": torch.zeros(_R, _Q_IN),
            "model.layers.99.self_attn.q_proj.lora_B": torch.zeros(_Q_OUT, _R),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "qat_meta.json").write_text(
        json.dumps({"lora_r": _R, "lora_alpha": _ALPHA, "adapter_file": "adapter_model.safetensors"})
    )

    with pytest.raises(QATMergeError, match="don't exist"):
        merge_qat_adapters(str(model_dir), str(adapter_dir), str(tmp_path / "out"))


def test_shape_mismatch_fails_loudly(tmp_path):
    import numpy as np
    rng = np.random.default_rng(10)
    model_dir, _ = _write_base_model(tmp_path, rng)
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    base = _Q_KEY[: -len(".weight")]
    # wrong rank between A and B (r=2 vs r=3)
    save_file(
        {
            f"{base}.lora_A": torch.zeros(2, _Q_IN),
            f"{base}.lora_B": torch.zeros(_Q_OUT, 3),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "qat_meta.json").write_text(
        json.dumps({"lora_r": 2, "lora_alpha": 4, "adapter_file": "adapter_model.safetensors"})
    )

    with pytest.raises(QATMergeError, match="shape mismatch"):
        merge_qat_adapters(str(model_dir), str(adapter_dir), str(tmp_path / "out"))


# ── real-world naming: MagicQuant's OWN format, not PEFT's ──────────────────

def test_rejects_peft_style_trailing_dot_keys_as_unusable(tmp_path):
    """Regression guard for the actual 2026-08-05 incident: keys saved in
    PEFT's '...lora_A.weight' convention (trailing dot + .weight) do NOT
    match MagicQuant's own '...lora_A' convention, so a dir containing only
    PEFT-style keys must be treated as having no usable pairs -- not silently
    merge zero deltas and call it done."""
    import numpy as np
    rng = np.random.default_rng(11)
    model_dir, _ = _write_base_model(tmp_path, rng)
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    base = _Q_KEY[: -len(".weight")]
    save_file(
        {
            f"{base}.lora_A.weight": torch.zeros(_R, _Q_IN),
            f"{base}.lora_B.weight": torch.zeros(_Q_OUT, _R),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "qat_meta.json").write_text(
        json.dumps({"lora_r": _R, "lora_alpha": _ALPHA, "adapter_file": "adapter_model.safetensors"})
    )

    with pytest.raises(QATMergeError, match="no lora_A/lora_B"):
        merge_qat_adapters(str(model_dir), str(adapter_dir), str(tmp_path / "out"))


# ── scale read from qat_meta.json, never hardcoded ───────────────────────────

def test_scale_uses_meta_lora_r_and_alpha_not_defaults(tmp_path):
    import numpy as np
    rng = np.random.default_rng(12)
    model_dir, base_tensors = _write_base_model(tmp_path, rng)
    # deliberately non-default r/alpha
    adapter_dir, adapter_state = _write_adapters(
        tmp_path, rng, include_2d=True, lora_r=5, lora_alpha=40.0
    )
    out_dir = tmp_path / "merged"

    merge_qat_adapters(str(model_dir), str(adapter_dir), str(out_dir))

    merged = load_file(str(out_dir / "model-00001-of-00002.safetensors"))
    a = adapter_state["model.layers.0.self_attn.q_proj.lora_A"]
    b = adapter_state["model.layers.0.self_attn.q_proj.lora_B"]
    expected = base_tensors[_Q_KEY] + (40.0 / 5) * (b @ a)
    assert torch.allclose(merged[_Q_KEY], expected, atol=1e-6)
