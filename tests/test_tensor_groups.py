"""Tensor-group classification tests.

Locks the MoE-expert classification fix (H1): ffn_{up,gate,down}_exps must all
map to group X (experts), not dense FFN U/D. Also pins dense + router + SSM names.
"""
import pytest

from magicquant.gguf.tensor_groups import TensorGroupClassifier


@pytest.fixture(scope="module")
def clf():
    return TensorGroupClassifier()


@pytest.mark.parametrize("name,expected", [
    # --- MoE experts: all three projections must be X (the H1 fix) ---
    ("blk.0.ffn_up_exps.weight", "X"),
    ("blk.0.ffn_gate_exps.weight", "X"),
    ("blk.0.ffn_down_exps.weight", "X"),
    ("blk.0.ffn_gate_up_exps.weight", "X"),      # fused gate+up variant
    ("blk.12.ffn_up_exps.weight", "X"),
    # --- dense FFN must stay U / D ---
    ("blk.0.ffn_up.weight", "U"),
    ("blk.0.ffn_gate.weight", "U"),
    ("blk.0.ffn_down.weight", "D"),
    # --- shared-expert MLP (dense-ish) stays U / D ---
    ("blk.0.ffn_up_shared.weight", "U"),
    ("blk.0.ffn_down_shared.weight", "D"),
    # --- router / gate-inp is R, not U ---
    ("blk.0.ffn_gate_inp.weight", "R"),
    # --- attention + embeddings + head unchanged ---
    ("blk.0.attn_q.weight", "Q"),
    ("blk.0.attn_k.weight", "K"),
    ("blk.0.attn_v.weight", "K"),
    ("blk.0.attn_output.weight", "O"),
    ("token_embd.weight", "E"),
    ("output.weight", "H"),
    # --- SSM / norms ---
    ("blk.0.ssm_in.weight", "S"),
    ("blk.0.attn_norm.weight", "N"),
])
def test_classify(clf, name, expected):
    assert clf.classify_tensor(name) == expected, f"{name} -> {clf.classify_tensor(name)}"


def test_moe_experts_not_in_dense_groups(clf):
    # regression guard: none of the expert matrices may land in U or D
    for n in ("blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight",
              "blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_up_exps.weight"):
        assert clf.classify_tensor(n) == "X"
