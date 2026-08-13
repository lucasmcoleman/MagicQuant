"""Q5_0/Q5_1 must be reachable exactly when they can help, and never otherwise.

Ground truth: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 shipped with
its Q5 AND Q6 tier bands structurally EMPTY. hidden_size=2688 and
moe_intermediate_size=1856 are 32- but not 256-divisible, so every block-256
K-quant is rewritten by the writer's block-size fallback and the achievable
block-32 prices jump straight from Q4_1 (5.0 bpw) to Q8_0 (8.5 bpw). Nothing
could land between 19.91 GiB and 32.54 GiB, which is precisely where the Q5
and Q6 bands sit. Q5_0 (5.5 bpw) and Q5_1 (6.0 bpw) fill that hole.

The design constraint is the other half of the story: on an ORDINARY
256-divisible model Q5_K is strictly better than Q5_0 at identical 5.5 bpw, so
making these globally reachable would make ordinary searches worse and would
churn tests/fixtures/refactor_regression_seed42.json. Hence shape-gating.

Two independent things have to hold, and both are tested here:
  1. the schemes are OFFERED (sampling pool + neighbour walk) for a qualifying
     group and not otherwise;
  2. the predictor PRICES a rewritten K-quant at what it really costs --
     without that, Q5_K looks smaller and cleaner than Q5_0 and wins every
     time, so the entries would exist and never once be selected.
"""

import pytest

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.gguf.writer import is_block32_only_tensor, _is_quantization_candidate
from magicquant.quant.schemes import BLOCK32_Q5_SCHEME_NAMES, get_scheme_by_name


# ── the shape predicate ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name,shape,expected", [
    # The real motivating widths.
    ("blk.0.ffn_down_exps.weight", (128, 1856), True),   # 1856 % 256 == 64
    ("blk.0.attn_q.weight", (2688, 2688), True),         # 2688 % 256 == 128
    # 256-divisible: a K-quant applies natively, so not block-32-only.
    ("blk.0.attn_q.weight", (4096, 4096), False),
    ("blk.0.ffn_down.weight", (512, 512), False),
    # Not 32-divisible at all: no block quant fits, the writer sends it to
    # F32, and offering Q5_0 would be meaningless.
    ("blk.0.attn_q.weight", (100, 100), False),
    # Not candidates at all -- these are F32 regardless of scheme, so they
    # must NOT drag their group into qualifying.
    ("blk.0.attn_norm.weight", (2688,), False),          # 1-D
    ("blk.0.ssm_norm.weight", (8, 512), False),          # never-quantize name
    ("blk.0.ffn_gate_inp.weight", (128, 2688), False),   # never-quantize name
])
def test_is_block32_only_tensor(name, shape, expected):
    assert is_block32_only_tensor(name, shape, len(shape)) is expected


def test_never_quantized_tensor_is_not_a_candidate_at_all():
    """The asymmetry that matters: a norm is not 'block-32-only', it is not a
    candidate. If it counted as block-32-only, a group made entirely of norms
    would qualify and get Q5_0/Q5_1 offered for a choice that can never take
    effect."""
    assert not _is_quantization_candidate("blk.0.ssm_norm.weight", (8, 512), 2)
    assert not is_block32_only_tensor("blk.0.ssm_norm.weight", (8, 512), 2)


# ── reachability: sampling pool ─────────────────────────────────────────────

class _StubPredictor:
    sensitivity_weights = {"X": 0.5, "O": 0.5}

    def predict_loss(self, cfg):
        return 1.0

    def predict_size(self, cfg):
        return 1.0

    def predict_tps(self, cfg):
        return 1.0


def _sampled_schemes(groups, block32_only_groups, n=250):
    import random
    random.seed(1234)
    surv = EvolutionarySurvivor(
        predictor=_StubPredictor(),
        baseline_config={"E": "BF16"},
        block32_only_groups=block32_only_groups,
    )
    seen = {g: set() for g in groups}
    for _ in range(n):
        cfg = surv._generate_random_config(groups)
        for g in groups:
            if g in cfg:
                seen[g].add(cfg[g])
    return seen


def test_q5_block32_schemes_are_offered_only_to_qualifying_groups():
    seen = _sampled_schemes(["X", "O"], block32_only_groups={"X"})
    assert seen["X"] & BLOCK32_Q5_SCHEME_NAMES, (
        "a block-32-only group must be able to draw Q5_0/Q5_1 -- without this "
        "the empty-tier-band problem they were added for is unfixed"
    )
    assert not (seen["O"] & BLOCK32_Q5_SCHEME_NAMES), (
        "a 256-divisible group must never draw them: Q5_K is strictly better "
        "at identical 5.5 bpw there"
    )


def test_no_group_draws_them_when_the_gate_is_empty():
    """The default. Empty gate must reproduce pre-feature behaviour exactly --
    this is the property tests/fixtures/refactor_regression_seed42.json
    depends on."""
    seen = _sampled_schemes(["X", "O", "U", "D"], block32_only_groups=set())
    for g, schemes in seen.items():
        assert not (schemes & BLOCK32_Q5_SCHEME_NAMES), g


# ── reachability: neighbour walk ────────────────────────────────────────────

def _survivor(block32_only_groups=()):
    return EvolutionarySurvivor(
        predictor=_StubPredictor(),
        baseline_config={"E": "BF16"},
        block32_only_groups=block32_only_groups,
    )


def test_chain_overlay_splices_q5_pair_in_for_a_qualifying_group():
    surv = _survivor({"X"})
    assert surv._downgrade("Q6_K", "X") == "Q5_1"
    assert surv._upgrade("Q5_K", "X") == "Q5_0"
    # ...and the new entries' own registry links complete the chain.
    assert surv._downgrade("Q5_1", "X") == "Q5_0"
    assert surv._upgrade("Q5_0", "X") == "Q5_1"


def test_chain_is_untouched_for_a_non_qualifying_group_and_with_no_group():
    surv = _survivor({"X"})
    assert surv._downgrade("Q6_K", "O") == "Q5_K"
    assert surv._downgrade("Q6_K") == "Q5_K"       # no group context at all
    plain = _survivor()
    assert plain._downgrade("Q6_K", "X") == "Q5_K"  # empty gate


# ── pricing: the half without which the feature never fires ─────────────────

def test_predictor_prices_a_rewritten_k_quant_at_its_real_cost():
    """On a block-32-only group a Q5_K assignment ships as Q8_0. If the
    predictor keeps believing 5.5 bpw, Q5_K looks both smaller and cleaner
    than Q5_0 and wins every time -- the registry entries would exist and
    never be selected once."""
    counts = {"X": 1_000_000}
    plain = PredictiveScorer({"X": 1.0}, parameter_counts=counts, baseline_size_gb=100.0)
    priced = PredictiveScorer(
        {"X": 1.0}, parameter_counts=counts, baseline_size_gb=100.0,
        effective_bpw={"X": {"Q5_K": 8.5}},   # rewritten by the block-size fallback
    )

    assert plain._bpw_for("X", "Q5_K") == pytest.approx(5.5)
    assert priced._bpw_for("X", "Q5_K") == pytest.approx(8.5)
    assert priced.predict_size({"X": "Q5_K"}) > plain.predict_size({"X": "Q5_K"})

    # Q5_0 is genuinely 5.5 bpw here, so it must now come out SMALLER than the
    # rewritten Q5_K -- which is the whole point.
    assert priced.predict_size({"X": "Q5_0"}) < priced.predict_size({"X": "Q5_K"})


def test_unpriced_pairs_fall_through_to_the_registry():
    """Absence of a table entry means 'no information, behave as before',
    never 'guess'."""
    p = PredictiveScorer({"X": 1.0}, effective_bpw={"X": {"Q5_K": 8.5}})
    assert p._bpw_for("X", "Q4_K_M") == pytest.approx(
        get_scheme_by_name("Q4_K_M").bits_per_weight
    )
    assert p._bpw_for("O", "Q5_K") == pytest.approx(5.5)   # different group
    empty = PredictiveScorer({"X": 1.0})
    assert empty._bpw_for("X", "Q5_K") == pytest.approx(5.5)


# ── registry facts ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,bpw,type_id", [("Q5_0", 5.5, 6), ("Q5_1", 6.0, 7)])
def test_registry_entries(name, bpw, type_id):
    s = get_scheme_by_name(name)
    assert s.bits_per_weight == bpw
    assert s.ggml_type_id == type_id
    assert s.category == "legacy_q"
    # Verified against ggml-quants.c, not copied from Q4_0's old wrong value;
    # tests/test_uses_imatrix_matches_ggml.py checks it behaviourally.
    assert s.uses_imatrix is True
    assert s.requires_imatrix is False


def test_q5_0_is_noisier_than_q5_k_at_identical_bpw():
    """The ordering that makes the whole thing coherent, and the reason these
    must not be reachable on 256-divisible models: same 5.5 bpw, legacy loses.
    Mirrors the Q4_0 (5.00) vs Q4_K_M (4.50) relationship already in the
    registry."""
    q5_0, q5_k = get_scheme_by_name("Q5_0"), get_scheme_by_name("Q5_K")
    assert q5_0.bits_per_weight == q5_k.bits_per_weight
    assert q5_0.noise_factor > q5_k.noise_factor


def test_q5_pair_is_not_dominated_by_mxfp4():
    """If MXFP4 were both smaller and cleaner, the search would never pick
    either one and the feature would be inert -- the failure mode the
    noise-factor measurement existed to rule out."""
    mxfp4 = get_scheme_by_name("MXFP4_MOE")
    for name in ("Q5_0", "Q5_1"):
        s = get_scheme_by_name(name)
        assert s.bits_per_weight > mxfp4.bits_per_weight
        assert s.noise_factor < mxfp4.noise_factor
