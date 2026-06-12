"""Regression: llama-arch attn_q/attn_k must be rope-PERMUTED when packing.

llama.cpp stores Q/K projection weights for the ``llama`` architecture in an
interleaved ("NORM"-rope, rope type 0) layout; its converter permutes them at
conversion time. HF safetensors keep the half-split rotary layout. MagicQuant
copied Q/K verbatim, so every llama/mistral-family pack had scrambled attention:
Llama-3.2-1B f16 scored PPL ~1725 vs the reference convert's 18.9 on identical
weights — confirmed byte-exactly (permute(MQ_q, n_head) == REF_q, V identical).

Qwen2 (NEOX rope) takes HF layout directly and must NOT be permuted.
"""
import json
import struct

import numpy as np
import pytest

from magicquant.gguf.source import SafetensorsSource, _permute_qk_rows


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


def _make_model_dir(tmp_path, model_type, n_head=2, n_kv=1, hidden=8):
    head_dim = hidden // n_head
    cfg = {
        "model_type": model_type,
        "hidden_size": hidden,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "num_hidden_layers": 1,
        "intermediate_size": hidden * 2,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    rng = np.random.default_rng(0)
    tensors = {
        "model.embed_tokens.weight": rng.standard_normal((16, hidden)),
        "lm_head.weight": rng.standard_normal((16, hidden)),
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((hidden, hidden)),
        "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((n_kv * head_dim, hidden)),
        "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((n_kv * head_dim, hidden)),
    }
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    return tensors


# ---------------------------------------------------------------------------
# the permutation helper itself
# ---------------------------------------------------------------------------

def test_permute_matches_llamacpp_formula():
    # llama.cpp convert_hf_to_gguf.py LlamaModel.permute:
    #   reshape(n_head, 2, dim//n_head//2, *rest).swapaxes(1, 2).reshape(orig)
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    n_head = 2
    expected = (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
                 .swapaxes(1, 2).reshape(w.shape))
    np.testing.assert_array_equal(_permute_qk_rows(w, n_head), expected)


def test_permute_is_row_reorder():
    # Permutation must only reorder output rows, never mix values.
    w = np.arange(32, dtype=np.float32).reshape(8, 4)
    out = _permute_qk_rows(w, 2)
    assert sorted(map(tuple, out)) == sorted(map(tuple, w))
    assert not np.array_equal(out, w)


def test_permute_1d_bias():
    b = np.arange(8, dtype=np.float32)
    out = _permute_qk_rows(b, 2)
    expected = (b.reshape(2, 2, 2).swapaxes(1, 2).reshape(8))
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# SafetensorsSource integration
# ---------------------------------------------------------------------------

def test_llama_q_and_k_are_permuted(tmp_path):
    orig = _make_model_dir(tmp_path, "llama", n_head=2, n_kv=1, hidden=8)
    src = SafetensorsSource(str(tmp_path))

    q = src.read_tensor_f32("blk.0.attn_q.weight").reshape(8, 8)
    expected_q = _permute_qk_rows(
        orig["model.layers.0.self_attn.q_proj.weight"].astype(np.float32), 2)
    np.testing.assert_array_equal(q, expected_q)

    k = src.read_tensor_f32("blk.0.attn_k.weight").reshape(4, 8)
    expected_k = _permute_qk_rows(
        orig["model.layers.0.self_attn.k_proj.weight"].astype(np.float32), 1)
    np.testing.assert_array_equal(k, expected_k)


def test_llama_v_is_not_permuted(tmp_path):
    orig = _make_model_dir(tmp_path, "llama", n_head=2, n_kv=1, hidden=8)
    src = SafetensorsSource(str(tmp_path))
    v = src.read_tensor_f32("blk.0.attn_v.weight").reshape(4, 8)
    np.testing.assert_array_equal(
        v, orig["model.layers.0.self_attn.v_proj.weight"].astype(np.float32))


def test_qwen2_is_not_permuted(tmp_path):
    # NEOX-rope arch: llama.cpp consumes the HF layout directly.
    orig = _make_model_dir(tmp_path, "qwen2", n_head=2, n_kv=1, hidden=8)
    src = SafetensorsSource(str(tmp_path))
    q = src.read_tensor_f32("blk.0.attn_q.weight").reshape(8, 8)
    np.testing.assert_array_equal(
        q, orig["model.layers.0.self_attn.q_proj.weight"].astype(np.float32))


def test_mistral_maps_to_llama_arch_and_permutes(tmp_path):
    # model_type "mistral" -> GGUF arch "llama" -> must permute too.
    orig = _make_model_dir(tmp_path, "mistral", n_head=2, n_kv=1, hidden=8)
    src = SafetensorsSource(str(tmp_path))
    q = src.read_tensor_f32("blk.0.attn_q.weight").reshape(8, 8)
    expected_q = _permute_qk_rows(
        orig["model.layers.0.self_attn.q_proj.weight"].astype(np.float32), 2)
    np.testing.assert_array_equal(q, expected_q)
