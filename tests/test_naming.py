"""Naming tests — tier-suffix expansion to HF-recognized quant labels.

Locks the Q2-tier label gap fix: a model named "...-Q2" must expand to a
HuggingFace-recognized quant string ("Q2_K"), not stay as a bare "-Q2".
"""
from magicquant.utils.naming import config_key, generate_name, _TIER_TO_HF_LABEL


def test_q2_tier_label_expands():
    """Q2 tier suffix must map to an HF-recognized label (was missing)."""
    assert "Q2" in _TIER_TO_HF_LABEL
    name = generate_name("MyModel-Q2", base_quant="Q2_K", overrides={})
    assert name == "MyModel-Q2_K.gguf"


def test_all_default_tiers_have_labels():
    """Every tier the orchestrator emits by default must have an HF label."""
    for tier in ["Q2", "Q4", "Q5", "Q6", "Q8"]:
        assert tier in _TIER_TO_HF_LABEL, f"missing HF label for tier {tier}"


def test_q5_tier_label_expands():
    name = generate_name("Foo-Bar-Q5", base_quant="MXFP4_MOE", overrides={})
    assert name == "Foo-Bar-Q5_K_M.gguf"


def test_no_tier_suffix_unchanged():
    name = generate_name("PlainName", base_quant="MXFP4_MOE", overrides={})
    assert name == "PlainName.gguf"


def test_config_key_round_trips_through_reselect_tiers_parser():
    """config_key's output is a PERSISTED interchange format: it becomes the
    measurement keys of search_results.json / resume checkpoints, and
    tools/reselect_tiers.py's _parse_key parses it back. Guard the round trip
    directly so a format change can't ship on a green suite."""
    from tools.reselect_tiers import _parse_key

    config = {"E": "BF16", "U": "MXFP4_MOE", "D": "Q4_K_M"}
    assert _parse_key(config_key(config)) == config
