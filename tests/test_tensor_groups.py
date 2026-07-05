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


def test_qwen35_mtp_and_gated_attention_not_unknown(clf):
    """Regression for the Qwopus3.6-27B-v2-MTP failure mode: the two tensor
    families a real Qwen3.5 MTP GGUF carries that the classifier didn't know
    (verified against the actual model, 2026-07-04). UNKNOWN tensors hard-fail
    the writer's pre-scan, killing the whole pack.
    """
    # llama.cpp names MTP layers blk.N.nextn.* across GLM/DeepSeek/Qwen MTP
    # arches; head-adjacent (they predict tokens), same treatment as mtp.*
    assert clf.classify_tensor("blk.46.nextn.eh_proj.weight") == "H"
    assert clf.classify_tensor("blk.46.nextn.shared_head.head.weight") == "H"
    # Qwen3.5 gated attention: per-head gate multiplied into attention output
    assert clf.classify_tensor("blk.0.attn_gate.weight") == "O"


def test_nextn_norms_stay_out_of_matrix_groups(clf):
    # nextn's tiny 1-D norms classify as H via the explicit nextn. pattern;
    # harmless either way (the writer's 1-D compat rule forces norms to F32),
    # but pin the current behavior so a re-ordering doesn't silently change it.
    assert clf.classify_tensor("blk.46.nextn.enorm.weight") == "H"
