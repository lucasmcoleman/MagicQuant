"""stream_aware: bias streamed matmul groups (H/Q/K/O) BF16/F16 -> Q8_0.

Validated on a real 27B (2026-07-05): replacing BF16 with Q8_0 on the
streamed groups was PPL-identical (6.6107->6.6106) but -16% size / +18% tg
-- BF16 there is pure bandwidth waste. Off by default (fixture-safe).
"""
import random
from collections import Counter

import pytest

from magicquant.evolution.survival import EvolutionarySurvivor


class _FakePredictor:
    def predict_loss(self, config):
        return 0.0


def _survivor(**kw):
    return EvolutionarySurvivor(predictor=_FakePredictor(), baseline_config={}, **kw)


def _dist(stream_aware, group, n=4000, seed=1):
    random.seed(seed)
    sv = _survivor(stream_aware=stream_aware)
    return Counter(sv._generate_random_config([group])[group] for _ in range(n))


@pytest.mark.parametrize("group", ["H", "Q", "K", "O"])
def test_stream_aware_biases_streamed_group_away_from_bf16(group):
    off = _dist(False, group)
    on = _dist(True, group)
    off_float = off.get("BF16", 0) + off.get("F16", 0)
    on_float = on.get("BF16", 0) + on.get("F16", 0)
    assert on_float < off_float, f"{group}: stream_aware did not reduce float picks"
    assert on.get("Q8_0", 0) >= off.get("Q8_0", 0), f"{group}: Q8_0 mass did not grow"


@pytest.mark.parametrize("group", ["E", "U", "D", "N"])
def test_stream_aware_leaves_non_streamed_groups_untouched(group):
    # E is row-gathered (not streamed); U/D FFN + N norms are outside the set.
    assert _dist(False, group) == _dist(True, group), f"{group} distribution changed"


def test_stream_aware_default_off_is_identical():
    for g in ("H", "Q", "K", "O", "E", "U"):
        assert _dist(False, g) == _dist(False, g)  # deterministic
    # default construction leaves stream_aware False
    assert _survivor().stream_aware is False


def test_stream_aware_supersedes_head_aggressive_for_H():
    # Both on: H should follow stream_aware (Q8_0 target), not head_aggressive
    # (Q6_K/Q5_K target). Q8_0 picks under both-on should exceed the
    # head_aggressive-only distribution's Q8_0-and-below-K-quant balance.
    random.seed(1)
    both = _survivor(stream_aware=True, head_aggressive=True)
    both_d = Counter(both._generate_random_config(["H"])["H"] for _ in range(4000))
    random.seed(1)
    ha = _survivor(head_aggressive=True)
    ha_d = Counter(ha._generate_random_config(["H"])["H"] for _ in range(4000))
    # stream_aware pushes strongly to Q8_0; head_aggressive pushes to Q6_K/Q5_K
    assert both_d.get("Q8_0", 0) > ha_d.get("Q8_0", 0)


def test_stream_shift_moves_float_mass_to_legacy_q():
    shifted = EvolutionarySurvivor._stream_shift(
        {"float": 0.50, "legacy_q": 0.10, "k_quant": 0.40}
    )
    assert shifted["float"] == pytest.approx(0.02)
    assert shifted["legacy_q"] == pytest.approx(0.58)  # 0.10 + (0.50 - 0.02)
    assert shifted["k_quant"] == pytest.approx(0.40)  # untouched


def test_stream_shift_handles_no_float_key():
    shifted = EvolutionarySurvivor._stream_shift({"k_quant": 1.0})
    assert shifted["float"] == 0.0
    assert shifted["k_quant"] == pytest.approx(1.0)
