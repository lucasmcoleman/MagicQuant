"""Incumbent config tests (Part B item 1).

magicquant.incumbents approximates llama.cpp's own Q4_K_M/Q5_K_M/Q6_K
per-tensor mixtures as MagicQuant per-group configs. These tests check the
module's public contract: every tier has a config, every scheme name in it
resolves in the registry, groups are the documented set, and the shape is
evidence-based (not arbitrary) -- H is protected at Q6_K even in the Q4/Q5
tiers, and Q6 is uniform.
"""
import pytest

from magicquant.incumbents import (
    INCUMBENT_GROUPS,
    INCUMBENT_TIERS,
    get_incumbent_config,
)
from magicquant.quant.schemes import get_scheme_by_name


@pytest.mark.parametrize("tier", ["Q4", "Q5", "Q6"])
def test_get_incumbent_config_returns_all_groups(tier):
    config = get_incumbent_config(tier)
    assert set(config.keys()) == set(INCUMBENT_GROUPS)


@pytest.mark.parametrize("tier", ["Q4", "Q5", "Q6"])
def test_get_incumbent_config_schemes_are_valid(tier):
    config = get_incumbent_config(tier)
    for group, scheme in config.items():
        # Must not raise -- every scheme name must exist in the registry.
        resolved = get_scheme_by_name(scheme)
        assert resolved.name == scheme


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown incumbent tier"):
        get_incumbent_config("Q3")


def test_get_incumbent_config_returns_a_copy():
    a = get_incumbent_config("Q4")
    a["E"] = "MUTATED"
    b = get_incumbent_config("Q4")
    assert b["E"] != "MUTATED"


def test_q4_tier_protects_head_at_q6k():
    # H (output.weight) gets llama.cpp's OUTPUT catch-all (Q6_K) regardless
    # of ftype -- see llama_tensor_get_type_impl's OUTPUT branch.
    config = get_incumbent_config("Q4")
    assert config["H"] == "Q6_K"
    assert config["E"] == "Q4_K_M"


def test_q5_tier_protects_head_at_q6k():
    config = get_incumbent_config("Q5")
    assert config["H"] == "Q6_K"
    assert config["E"] == "Q5_K"


def test_q6_tier_is_uniform():
    config = get_incumbent_config("Q6")
    assert set(config.values()) == {"Q6_K"}


def test_s_group_mirrors_d_group_for_every_tier():
    # No stock analog for SSM/linear-attention -- documented as mirroring D.
    for tier in INCUMBENT_TIERS:
        config = get_incumbent_config(tier)
        assert config["S"] == config["D"]


def test_x_and_r_mirror_u_for_every_tier():
    # Experts (X) and router (R) share U's per-tensor category in
    # llama.cpp's own quantizer (ffn_up/ffn_gate substring match), see the
    # module docstring's FFN_GATE note.
    for tier in INCUMBENT_TIERS:
        config = get_incumbent_config(tier)
        assert config["X"] == config["U"]
        assert config["R"] == config["U"]
