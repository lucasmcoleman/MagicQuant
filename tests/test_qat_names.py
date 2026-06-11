"""HF module name -> GGUF tensor name mapping for QAT group routing.

``wrap_model`` walks ``model.named_modules()`` and needs each ``nn.Linear``'s
GGUF tensor name to classify it into a tensor group. This reuses MagicQuant's
canonical ``source.py`` ``_HF_TO_GGUF_PATTERNS`` so the mapping never drifts from
the writer's.
"""

from magicquant.qat.names import hf_to_ggml_name


def test_maps_attention_and_ffn():
    assert hf_to_ggml_name("model.layers.0.self_attn.q_proj") == "blk.0.attn_q.weight"
    assert hf_to_ggml_name("model.layers.3.mlp.up_proj") == "blk.3.ffn_up.weight"
    assert hf_to_ggml_name("model.layers.3.mlp.down_proj") == "blk.3.ffn_down.weight"
    assert hf_to_ggml_name("lm_head") == "output.weight"
    assert hf_to_ggml_name("model.embed_tokens") == "token_embd.weight"


def test_maps_remaining_attention_and_ffn_projections():
    assert hf_to_ggml_name("model.layers.2.self_attn.k_proj") == "blk.2.attn_k.weight"
    assert hf_to_ggml_name("model.layers.2.self_attn.v_proj") == "blk.2.attn_v.weight"
    assert (
        hf_to_ggml_name("model.layers.2.self_attn.o_proj") == "blk.2.attn_output.weight"
    )
    assert hf_to_ggml_name("model.layers.5.mlp.gate_proj") == "blk.5.ffn_gate.weight"


def test_accepts_names_with_explicit_weight_suffix():
    # ``named_modules`` yields module paths (no .weight); accept either form.
    assert (
        hf_to_ggml_name("model.layers.0.self_attn.q_proj.weight")
        == "blk.0.attn_q.weight"
    )


def test_unknown_returns_none():
    assert hf_to_ggml_name("model.some.unknown") is None
    assert hf_to_ggml_name("") is None
