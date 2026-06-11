"""Regression test: rope_theta must be read from nested ``rope_parameters``.

transformers >=5 stopped emitting the flat ``rope_theta`` field in config.json and
now nests it inside ``rope_parameters`` (alongside ``rope_type`` etc.). MagicQuant's
generic field_map only reads the flat key, so a model re-saved by a recent
transformers (e.g. a merged QAT model) lost its RoPE base entirely. The GGUF then
fell back to the llama.cpp default (1e4) instead of Qwen2.5's 1e6, and the packed
model produced garbage perplexity (~8000 instead of ~22).

The fix adds a general fallback: if ``{arch}.rope.freq_base`` wasn't set by the flat
field_map, read it from ``rope_parameters.rope_theta`` for ANY architecture.
"""
from magicquant.gguf.source import _build_gguf_metadata_from_config


def test_flat_rope_theta_still_read():
    """Older configs with a flat rope_theta keep working (transformers <5)."""
    cfg = {"model_type": "qwen2", "rope_theta": 1000000.0}
    meta = _build_gguf_metadata_from_config(cfg)
    assert meta["qwen2.rope.freq_base"] == 1000000.0


def test_nested_rope_parameters_theta_read():
    """transformers >=5 nests rope_theta; the general fallback must pick it up."""
    cfg = {
        "model_type": "qwen2",
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }
    meta = _build_gguf_metadata_from_config(cfg)
    assert meta["qwen2.rope.freq_base"] == 1000000.0


def test_flat_takes_priority_over_nested():
    """If both are present, the flat field_map value wins (it runs first)."""
    cfg = {
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rope_parameters": {"rope_theta": 1000000.0},
    }
    meta = _build_gguf_metadata_from_config(cfg)
    assert meta["llama.rope.freq_base"] == 500000.0


def test_no_rope_theta_anywhere_leaves_key_unset():
    """No crash and no bogus default when neither location has rope_theta."""
    cfg = {"model_type": "llama"}
    meta = _build_gguf_metadata_from_config(cfg)
    assert "llama.rope.freq_base" not in meta
