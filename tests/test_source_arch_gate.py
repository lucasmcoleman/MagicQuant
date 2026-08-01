"""Regression: SafetensorsSource must refuse an architecture whose HF->GGUF
value transforms aren't implemented/verified, and must refuse a tensor name
it can't map -- both loudly, by default.

Ground truth: the qwen3_5 uniform-logits incident. qwen3_5's tensor NAME map
was complete (0 name diffs, 0 shape diffs vs a reference GGUF) but its VALUE
transforms were entirely unimplemented, so the pack loaded and ran while 64%
of tensor values were silently wrong (PPL == vocab_size). Nothing failed
loudly because name-mapping success was mistaken for "this arch is handled".
These tests prove the new gates catch that class of bug for any *future*
unhandled arch, rather than relying on someone noticing garbage output.

VERIFIED TO FAIL PRE-FIX: run against the pre-fix source.py (before
UnsupportedSourceArchitecture / the arch and unmapped-name gates existed),
every test below that expects ``pytest.raises(UnsupportedSourceArchitecture)``
failed with ``ImportError`` (the name didn't exist) / would otherwise have
passed silently through _ensure_loaded with no exception raised at all.
"""
import json
import logging
import struct

import numpy as np
import pytest

from magicquant.gguf.source import (
    SafetensorsSource,
    UnsupportedSourceArchitecture,
    _ALLOW_UNVALIDATED_ARCH_ENV,
    _build_gguf_metadata_from_config,
)


def _write_safetensors(path, tensors):
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


def _base_config(model_type):
    return {
        "model_type": model_type,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }


# ---------------------------------------------------------------------------
# Architecture gate
# ---------------------------------------------------------------------------

def test_unvalidated_arch_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(_ALLOW_UNVALIDATED_ARCH_ENV, raising=False)
    # "gemma3" is a real, recognized GGUF arch (arch_map maps model_type
    # "gemma3" -> "gemma3") but has no entry in _ARCH_VALUE_TRANSFORMS or
    # _ARCH_NO_TRANSFORM_NEEDED -- exactly the qwen3_5-before-the-fix shape.
    (tmp_path / "config.json").write_text(json.dumps(_base_config("gemma3")))
    rng = np.random.default_rng(0)
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": rng.standard_normal((16, 8)).astype(np.float32),
    })
    src = SafetensorsSource(str(tmp_path))
    with pytest.raises(UnsupportedSourceArchitecture, match="gemma3"):
        src.get_metadata()


def test_unvalidated_arch_env_var_downgrades_to_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(_ALLOW_UNVALIDATED_ARCH_ENV, "1")
    (tmp_path / "config.json").write_text(json.dumps(_base_config("gemma3")))
    rng = np.random.default_rng(0)
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": rng.standard_normal((16, 8)).astype(np.float32),
    })
    src = SafetensorsSource(str(tmp_path))
    with caplog.at_level(logging.ERROR):
        meta = src.get_metadata()  # must NOT raise
    assert meta["general.architecture"] == "gemma3"
    assert any("gemma3" in rec.message for rec in caplog.records)


def test_validated_arch_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.delenv(_ALLOW_UNVALIDATED_ARCH_ENV, raising=False)
    (tmp_path / "config.json").write_text(json.dumps(_base_config("qwen2")))
    rng = np.random.default_rng(0)
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": rng.standard_normal((16, 8)).astype(np.float32),
    })
    src = SafetensorsSource(str(tmp_path))
    meta = src.get_metadata()  # must not raise
    assert meta["general.architecture"] == "qwen2"


# ---------------------------------------------------------------------------
# Unmapped tensor-name gate
# ---------------------------------------------------------------------------

def test_unmapped_tensor_name_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(_ALLOW_UNVALIDATED_ARCH_ENV, raising=False)
    # qwen2 is a validated arch, but this tensor name matches no pattern in
    # _HF_TO_GGUF_PATTERNS -- the unmapped-name gate must catch it even
    # though the ARCHITECTURE gate passed.
    (tmp_path / "config.json").write_text(json.dumps(_base_config("qwen2")))
    rng = np.random.default_rng(0)
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "model.layers.0.totally_unrecognized_projection.weight":
            rng.standard_normal((8, 8)).astype(np.float32),
    })
    src = SafetensorsSource(str(tmp_path))
    with pytest.raises(UnsupportedSourceArchitecture, match="totally_unrecognized_projection"):
        src.get_metadata()


def test_unmapped_tensor_name_env_var_downgrades_to_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(_ALLOW_UNVALIDATED_ARCH_ENV, "1")
    (tmp_path / "config.json").write_text(json.dumps(_base_config("qwen2")))
    rng = np.random.default_rng(0)
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "model.layers.0.totally_unrecognized_projection.weight":
            rng.standard_normal((8, 8)).astype(np.float32),
    })
    src = SafetensorsSource(str(tmp_path))
    with caplog.at_level(logging.ERROR):
        names = src.get_tensor_names()  # must NOT raise
    # The tensor survives under its raw HF name (still present, just unmapped).
    assert "model.layers.0.totally_unrecognized_projection.weight" in names
    assert any("totally_unrecognized_projection" in rec.message for rec in caplog.records)


def test_mapped_names_do_not_trip_the_gate(tmp_path, monkeypatch):
    # Sanity/negative control: an ordinary, fully-mapped qwen2 checkpoint
    # must not raise at all.
    monkeypatch.delenv(_ALLOW_UNVALIDATED_ARCH_ENV, raising=False)
    (tmp_path / "config.json").write_text(json.dumps(_base_config("qwen2")))
    rng = np.random.default_rng(0)
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "lm_head.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "model.norm.weight": rng.standard_normal((8,)).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((8, 8)).astype(np.float32),
        "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((8, 8)).astype(np.float32),
        "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((8, 8)).astype(np.float32),
        "model.layers.0.self_attn.o_proj.weight": rng.standard_normal((8, 8)).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "model.layers.0.mlp.up_proj.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "model.layers.0.mlp.down_proj.weight": rng.standard_normal((8, 16)).astype(np.float32),
        "model.layers.0.input_layernorm.weight": rng.standard_normal((8,)).astype(np.float32),
        "model.layers.0.post_attention_layernorm.weight": rng.standard_normal((8,)).astype(np.float32),
    })
    src = SafetensorsSource(str(tmp_path))
    names = set(src.get_tensor_names())  # must not raise
    assert "blk.0.attn_q.weight" in names
    assert "blk.0.ffn_down.weight" in names


# ---------------------------------------------------------------------------
# arch_map / model_type gate (_build_gguf_metadata_from_config)
# ---------------------------------------------------------------------------
# One level up from the architecture-transform gate above: this one fires
# when the HF model_type has no entry in arch_map at all. Ground truth is
# the same qwen3_5 incident replayed one step earlier -- silently defaulting
# an unrecognized model_type to 'llama' builds every GGUF metadata key
# (context_length, attention.head_count, rope.freq_base, ...) under the
# WRONG architecture namespace, and the pack loads and often runs anyway.

def test_known_model_type_passes(monkeypatch):
    monkeypatch.delenv(_ALLOW_UNVALIDATED_ARCH_ENV, raising=False)
    meta = _build_gguf_metadata_from_config(_base_config("qwen2"))  # must not raise
    assert meta["general.architecture"] == "qwen2"


def test_unknown_model_type_raises(monkeypatch):
    monkeypatch.delenv(_ALLOW_UNVALIDATED_ARCH_ENV, raising=False)
    with pytest.raises(UnsupportedSourceArchitecture, match="totally_bogus_model_type"):
        _build_gguf_metadata_from_config(_base_config("totally_bogus_model_type"))


def test_unknown_model_type_env_var_downgrades_to_warning(monkeypatch, caplog):
    monkeypatch.setenv(_ALLOW_UNVALIDATED_ARCH_ENV, "1")
    with caplog.at_level(logging.ERROR):
        meta = _build_gguf_metadata_from_config(
            _base_config("totally_bogus_model_type")
        )  # must NOT raise
    assert meta["general.architecture"] == "llama"  # documented fallback
    assert any("totally_bogus_model_type" in rec.message for rec in caplog.records)
