"""Router probability-correction biases must classify as R, not fall through.

Regression for a real Laguna-S-2.1 run where 47 `blk.N.exp_probs_b.bias`
tensors (llama.cpp's name for HF e_score_correction_bias) were reported as
having no group classification.
"""
import pytest

from magicquant.gguf.tensor_groups import TensorGroupClassifier


@pytest.mark.parametrize("name", [
    "blk.1.exp_probs_b.bias",
    "blk.47.exp_probs_b.bias",
    "blk.3.ffn_gate_inp.weight",
])
def test_router_tensors_classify_as_R(name):
    assert TensorGroupClassifier().classify_tensor(name) == "R"


def test_expert_tensors_still_classify_as_X():
    c = TensorGroupClassifier()
    assert c.classify_tensor("blk.1.ffn_down_exps.weight") == "X"
