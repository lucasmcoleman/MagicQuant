"""Regression test: HF→GGUF name mapping must handle .bias tensors.

Qwen2/Qwen2.5 (and other qkv-bias architectures) carry q/k/v projection biases.
Before the fix, `_hf_name_to_gguf` only mapped `.weight` tensors, so biases were
written to the GGUF under their HF names and llama.cpp failed to load the model
("missing tensor 'blk.0.attn_q.bias'").
"""
from magicquant.gguf.source import _hf_name_to_gguf


def test_attention_bias_maps_like_weight():
    assert _hf_name_to_gguf("model.layers.0.self_attn.q_proj.bias") == "blk.0.attn_q.bias"
    assert _hf_name_to_gguf("model.layers.0.self_attn.k_proj.bias") == "blk.0.attn_k.bias"
    assert _hf_name_to_gguf("model.layers.7.self_attn.v_proj.bias") == "blk.7.attn_v.bias"


def test_weight_mapping_unchanged():
    assert _hf_name_to_gguf("model.layers.0.self_attn.q_proj.weight") == "blk.0.attn_q.weight"
    assert _hf_name_to_gguf("model.layers.3.mlp.down_proj.weight") == "blk.3.ffn_down.weight"


def test_unmapped_bias_left_untouched():
    # A projection whose .weight doesn't map -> the bias is left as-is (no crash).
    name = "model.some.unknown_proj.bias"
    assert _hf_name_to_gguf(name) == name
