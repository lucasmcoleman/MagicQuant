"""Regression: qwen35/qwen35moe HF->GGUF value transforms.

MagicQuant's SafetensorsSource implemented the qwen3_5 (Qwen3.6-hybrid)
architecture as a NAME map + METADATA builder only -- it never implemented
the three VALUE transforms llama.cpp's convert_hf_to_gguf.py performs for
this arch (RMSNorm +1, A_log -> -exp(A_log), and a linear-attention V-head
grouped->tiled reorder). The name map is complete, so nothing failed loudly:
tensor names and shapes matched a reference GGUF exactly, but 64% of tensor
VALUES were silently wrong. llama-perplexity on the resulting GGUF reported
PPL == vocab_size (uniform logits) because the linear_attention decay gate
underflowed to zero in every layer.

This module proves the three transforms byte-exact against hand-derived
expected arrays on a tiny synthetic qwen3_5 checkpoint (hidden=16,
num_k_heads=2, num_v_heads=6 so r=3, head_k_dim=head_v_dim=4), covering both
a linear-attention layer (layer 0) and the norm/A_log rules in isolation.

VERIFIED TO FAIL PRE-FIX: every test in the "value transforms" section below
was run against the pre-fix source.py (stashed via `git stash` on this
file's dependency, `magicquant/gguf/source.py`) and failed -- the pre-fix
code returned the raw HF values unchanged for norm/A_log/every reordered
tensor, which do not equal the transformed expected arrays asserted here.
"""
import json
import struct

import numpy as np

from magicquant.gguf.source import SafetensorsSource, _reorder_v_heads


# ---------------------------------------------------------------------------
# helpers: hand-write a minimal single-file safetensors model dir
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors):
    """tensors: {name: float32 ndarray} -> minimal .safetensors file."""
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


HIDDEN = 16
NUM_K = 2
NUM_V = 6           # r = NUM_V // NUM_K = 3
HEAD_K_DIM = 4
HEAD_V_DIM = 4
QK_WIDTH = 2 * HEAD_K_DIM * NUM_K   # 16 (Q rows + K rows, verbatim)
V_WIDTH = NUM_V * HEAD_V_DIM        # 24
QKV_WIDTH = QK_WIDTH + V_WIDTH      # 40
R = NUM_V // NUM_K


def _qwen35_config(model_type="qwen3_5"):
    return {
        "model_type": model_type,
        "hidden_size": HIDDEN,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 1,
        "intermediate_size": HIDDEN * 2,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "linear_num_key_heads": NUM_K,
        "linear_num_value_heads": NUM_V,
        "linear_key_head_dim": HEAD_K_DIM,
        "linear_value_head_dim": HEAD_V_DIM,
    }


def _make_qwen35_model_dir(tmp_path, model_type="qwen3_5"):
    """One linear-attention layer + a couple of dense/norm tensors, covering
    every HF name kind the qwen3_5 value transforms key off."""
    (tmp_path / "config.json").write_text(json.dumps(_qwen35_config(model_type)))
    rng = np.random.default_rng(0)
    tensors = {
        "model.embed_tokens.weight": rng.standard_normal((32, HIDDEN)).astype(np.float32),
        "model.norm.weight": rng.standard_normal((HIDDEN,)).astype(np.float32),
        "model.layers.0.linear_attn.in_proj_qkv.weight":
            rng.standard_normal((QKV_WIDTH, HIDDEN)).astype(np.float32),
        "model.layers.0.linear_attn.in_proj_z.weight":
            rng.standard_normal((V_WIDTH, HIDDEN)).astype(np.float32),
        "model.layers.0.linear_attn.in_proj_a.weight":
            rng.standard_normal((NUM_V, HIDDEN)).astype(np.float32),
        "model.layers.0.linear_attn.in_proj_b.weight":
            rng.standard_normal((NUM_V, HIDDEN)).astype(np.float32),
        "model.layers.0.linear_attn.A_log":
            rng.standard_normal((NUM_V,)).astype(np.float32),
        "model.layers.0.linear_attn.conv1d.weight":
            rng.standard_normal((QKV_WIDTH, 1, 4)).astype(np.float32),
        "model.layers.0.linear_attn.dt_bias":
            rng.standard_normal((NUM_V,)).astype(np.float32),
        "model.layers.0.linear_attn.norm.weight":
            rng.standard_normal((HEAD_V_DIM,)).astype(np.float32),
        "model.layers.0.linear_attn.out_proj.weight":
            rng.standard_normal((HIDDEN, V_WIDTH)).astype(np.float32),
        "model.layers.0.input_layernorm.weight":
            rng.standard_normal((HIDDEN,)).astype(np.float32),
        "model.layers.0.post_attention_layernorm.weight":
            rng.standard_normal((HIDDEN,)).astype(np.float32),
        "model.layers.0.self_attn.q_norm.weight":
            rng.standard_normal((HEAD_K_DIM,)).astype(np.float32),
        "model.layers.0.self_attn.k_norm.weight":
            rng.standard_normal((HEAD_K_DIM,)).astype(np.float32),
    }
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    return tensors


# ---------------------------------------------------------------------------
# _reorder_v_heads: the pure permutation helper itself
# ---------------------------------------------------------------------------

def test_reorder_v_heads_matches_hand_written_permutation():
    # 6 = num_k_heads(2) * num_v_per_k(3) * head_dim(1); grouped [G0v0,G0v1,
    # G0v2,G1v0,G1v1,G1v2] -> tiled [G0v0,G1v0,G0v1,G1v1,G0v2,G1v2].
    a = np.array([0, 1, 2, 10, 11, 12], dtype=np.float32)
    expected = np.array([0, 10, 1, 11, 2, 12], dtype=np.float32)
    np.testing.assert_array_equal(
        _reorder_v_heads(a, axis=0, num_k_heads=2, num_v_per_k=3, head_dim=1),
        expected,
    )


def test_reorder_v_heads_is_pure_permutation():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(QKV_WIDTH - QK_WIDTH).astype(np.float32)
    out = _reorder_v_heads(a, axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=HEAD_V_DIM)
    assert sorted(a.tolist()) == sorted(out.tolist())
    assert not np.array_equal(a, out)


def test_reorder_v_heads_column_axis():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((HIDDEN, V_WIDTH)).astype(np.float32)
    out = _reorder_v_heads(a, axis=1, num_k_heads=NUM_K, num_v_per_k=R, head_dim=HEAD_V_DIM)
    assert out.shape == a.shape
    # every column of `a` reappears somewhere in `out` (pure column permute)
    a_cols = sorted(map(tuple, a.T))
    out_cols = sorted(map(tuple, out.T))
    assert a_cols == out_cols


# ---------------------------------------------------------------------------
# SafetensorsSource integration -- byte-exact against hand-derived expected
# ---------------------------------------------------------------------------

def test_rmsnorm_plus_one_output_norm(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    got = src.read_tensor_f32("output_norm.weight")
    np.testing.assert_allclose(got, raw["model.norm.weight"] + 1.0)


def test_rmsnorm_plus_one_attn_and_ffn_norm(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    np.testing.assert_allclose(
        src.read_tensor_f32("blk.0.attn_norm.weight"),
        raw["model.layers.0.input_layernorm.weight"] + 1.0,
    )
    np.testing.assert_allclose(
        src.read_tensor_f32("blk.0.post_attention_norm.weight"),
        raw["model.layers.0.post_attention_layernorm.weight"] + 1.0,
    )


def test_rmsnorm_plus_one_qk_norms(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    np.testing.assert_allclose(
        src.read_tensor_f32("blk.0.attn_q_norm.weight"),
        raw["model.layers.0.self_attn.q_norm.weight"] + 1.0,
    )
    np.testing.assert_allclose(
        src.read_tensor_f32("blk.0.attn_k_norm.weight"),
        raw["model.layers.0.self_attn.k_norm.weight"] + 1.0,
    )


def test_ssm_norm_is_excluded_from_plus_one(tmp_path):
    # llama.cpp deliberately excludes linear_attn.norm.weight from the +1
    # rule -- MagicQuant must reproduce the exclusion, not just the rule.
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    got = src.read_tensor_f32("blk.0.ssm_norm.weight")
    np.testing.assert_allclose(got, raw["model.layers.0.linear_attn.norm.weight"])
    assert not np.allclose(got, raw["model.layers.0.linear_attn.norm.weight"] + 1.0)


def test_a_log_negated_exp_and_reordered(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    expected = _reorder_v_heads(
        -np.exp(raw["model.layers.0.linear_attn.A_log"]),
        axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=1,
    )
    got = src.read_tensor_f32("blk.0.ssm_a")
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    # sanity: raw A_log must NOT appear unchanged, and -exp() alone
    # (unreordered) must not either -- both the sign/exp AND the reorder
    # have to have happened.
    assert not np.allclose(got, raw["model.layers.0.linear_attn.A_log"])
    assert not np.allclose(got, -np.exp(raw["model.layers.0.linear_attn.A_log"]))


def test_ssm_dt_bias_reordered_only(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    expected = _reorder_v_heads(
        raw["model.layers.0.linear_attn.dt_bias"],
        axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=1,
    )
    np.testing.assert_allclose(src.read_tensor_f32("blk.0.ssm_dt.bias"), expected)


def test_ssm_alpha_beta_reordered(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    for hf_key, gguf_name in (
        ("model.layers.0.linear_attn.in_proj_a.weight", "blk.0.ssm_alpha.weight"),
        ("model.layers.0.linear_attn.in_proj_b.weight", "blk.0.ssm_beta.weight"),
    ):
        expected = _reorder_v_heads(
            raw[hf_key], axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=1,
        )
        got = src.read_tensor_f32(gguf_name).reshape(NUM_V, HIDDEN)
        np.testing.assert_allclose(got, expected)


def test_attn_gate_reordered(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    expected = _reorder_v_heads(
        raw["model.layers.0.linear_attn.in_proj_z.weight"],
        axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=HEAD_V_DIM,
    )
    got = src.read_tensor_f32("blk.0.attn_gate.weight").reshape(V_WIDTH, HIDDEN)
    np.testing.assert_allclose(got, expected)


def test_attn_qkv_qk_rows_verbatim_v_rows_reordered(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    raw_qkv = raw["model.layers.0.linear_attn.in_proj_qkv.weight"]
    got = src.read_tensor_f32("blk.0.attn_qkv.weight").reshape(QKV_WIDTH, HIDDEN)

    # Q/K rows pass through byte-identical (no reorder for these).
    np.testing.assert_array_equal(got[:QK_WIDTH], raw_qkv[:QK_WIDTH])

    # V rows are reordered, and must actually have moved.
    expected_v = _reorder_v_heads(
        raw_qkv[QK_WIDTH:], axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=HEAD_V_DIM,
    )
    np.testing.assert_allclose(got[QK_WIDTH:], expected_v)
    assert not np.array_equal(got[QK_WIDTH:], raw_qkv[QK_WIDTH:])


def test_ssm_conv1d_qk_channels_verbatim_v_channels_reordered(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    # [C, 1, K] -> squeeze -> [C, K]
    raw_conv = raw["model.layers.0.linear_attn.conv1d.weight"].reshape(QKV_WIDTH, 4)
    got = src.read_tensor_f32("blk.0.ssm_conv1d.weight").reshape(QKV_WIDTH, 4)

    np.testing.assert_array_equal(got[:QK_WIDTH], raw_conv[:QK_WIDTH])

    expected_v = _reorder_v_heads(
        raw_conv[QK_WIDTH:], axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=HEAD_V_DIM,
    )
    np.testing.assert_allclose(got[QK_WIDTH:], expected_v)
    assert not np.array_equal(got[QK_WIDTH:], raw_conv[QK_WIDTH:])


def test_ssm_out_column_reorder(tmp_path):
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    raw_out = raw["model.layers.0.linear_attn.out_proj.weight"]
    expected = _reorder_v_heads(
        raw_out, axis=1, num_k_heads=NUM_K, num_v_per_k=R, head_dim=HEAD_V_DIM,
    )
    got = src.read_tensor_f32("blk.0.ssm_out.weight").reshape(HIDDEN, V_WIDTH)
    np.testing.assert_allclose(got, expected)
    assert not np.array_equal(got, raw_out)


def test_qwen35moe_gets_same_transforms(tmp_path):
    # qwen35moe (Qwen3.5's MoE variant) shares the same hybrid attention
    # block and must use the identical transform.
    raw = _make_qwen35_model_dir(tmp_path, model_type="qwen3_5_moe")
    src = SafetensorsSource(str(tmp_path))
    np.testing.assert_allclose(
        src.read_tensor_f32("output_norm.weight"), raw["model.norm.weight"] + 1.0,
    )
    expected_a = _reorder_v_heads(
        -np.exp(raw["model.layers.0.linear_attn.A_log"]),
        axis=0, num_k_heads=NUM_K, num_v_per_k=R, head_dim=1,
    )
    np.testing.assert_allclose(src.read_tensor_f32("blk.0.ssm_a"), expected_a, rtol=1e-6)


def test_control_dense_weights_untransformed(tmp_path):
    # Everything NOT covered by a qwen35 rule must round-trip byte-identical
    # -- proves the transform is scoped, not a blanket mangling of the arch.
    raw = _make_qwen35_model_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    np.testing.assert_array_equal(
        src.read_tensor_f32("token_embd.weight").reshape(32, HIDDEN),
        raw["model.embed_tokens.weight"],
    )


# ---------------------------------------------------------------------------
# num_k == num_v: the whole V-reorder block must be skipped (not just a
# no-op reorder), while norm/A_log rules still apply.
# ---------------------------------------------------------------------------

def _make_qwen35_uniform_heads_dir(tmp_path):
    """num_k_heads == num_v_heads: llama.cpp's own converter never reorders
    in this case (see _qwen35_value_transform's need_reorder guard)."""
    n = 4
    cfg = _qwen35_config()
    cfg["linear_num_key_heads"] = n
    cfg["linear_num_value_heads"] = n
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    rng = np.random.default_rng(3)
    tensors = {
        "model.layers.0.linear_attn.A_log": rng.standard_normal((n,)).astype(np.float32),
        "model.layers.0.linear_attn.dt_bias": rng.standard_normal((n,)).astype(np.float32),
    }
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    return tensors, n


def test_no_reorder_when_num_k_equals_num_v(tmp_path):
    raw, n = _make_qwen35_uniform_heads_dir(tmp_path)
    src = SafetensorsSource(str(tmp_path))
    # A_log still gets -exp(), just no reorder (identity reorder == raw here).
    got_a = src.read_tensor_f32("blk.0.ssm_a")
    np.testing.assert_allclose(got_a, -np.exp(raw["model.layers.0.linear_attn.A_log"]), rtol=1e-6)
    # dt_bias is untouched entirely (no norm/A_log rule applies, and the
    # reorder is skipped).
    np.testing.assert_array_equal(
        src.read_tensor_f32("blk.0.ssm_dt.bias"), raw["model.layers.0.linear_attn.dt_bias"],
    )
