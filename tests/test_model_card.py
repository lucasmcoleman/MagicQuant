"""Model-card generation tests (M9)."""
from magicquant.utils.model_card import generate_model_card


SAMPLE = {
    "baseline_ppl": 5.1234,
    "tiered": {
        "Q6": {
            "config": {"E": "BF16", "H": "BF16", "Q": "Q6_K", "D": "MXFP4_MOE"},
            "ppl": 5.20,
            "measured_loss": 0.015,
            "size_gb": 3.4,
        },
        "Q4": {
            "config": {"E": "BF16", "H": "BF16", "Q": "IQ4_NL", "D": "MXFP4_MOE"},
            "ppl": 5.55,
            "measured_loss": 0.083,
            "size_gb": 2.1,
        },
    },
}


def test_card_contains_each_tier_and_metrics():
    card = generate_model_card(SAMPLE, base_model_name="Qwen3-4B")
    assert "Qwen3-4B" in card
    # Both tiers present
    assert "| Q6 |" in card
    assert "| Q4 |" in card
    # Measured PPL shown
    assert "5.2000" in card
    assert "5.5500" in card
    # Size shown
    assert "3.40" in card and "2.10" in card
    # Per-group scheme maps shown
    assert "Q:Q6_K" in card
    assert "D:MXFP4_MOE" in card
    # Attribution
    assert "magiccodingman" in card


def test_card_handles_missing_metrics():
    minimal = {"tiered": {"Q5": {"config": {"E": "BF16"}}}}
    card = generate_model_card(minimal, base_model_name="X")
    assert "| Q5 |" in card
    assert "E:BF16" in card  # em-dash placeholders for missing ppl/size are fine


def test_card_uses_tiered_survivors_fallback():
    data = {
        "tiered_survivors": {
            "Q8": {"config": {"E": "BF16"}, "ppl": 5.0, "measured_loss": 0.0, "size_gb": 8.0}
        }
    }
    card = generate_model_card(data, base_model_name="Y")
    assert "| Q8 |" in card


def test_card_warns_on_legacy_tier_scheme_version():
    """A search_results.json with no tier_scheme_version (pre-2026-07 fix,
    like SAMPLE above) must surface a disclosure banner -- its tier labels
    follow the OLD, wider size-ratio boundaries."""
    card = generate_model_card(SAMPLE, base_model_name="Qwen3-4B")
    assert "tier_scheme_version=1" in card
    assert "OLDER, wider size-ratio boundaries" in card


def test_card_no_warning_on_current_tier_scheme_version():
    from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION

    current = dict(SAMPLE, tier_scheme_version=CURRENT_TIER_SCHEME_VERSION)
    card = generate_model_card(current, base_model_name="Qwen3-4B")
    assert "tier_scheme_version=" not in card
    assert "OLDER, wider size-ratio boundaries" not in card
