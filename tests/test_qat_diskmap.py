"""magicquant.qat.diskmap: adapter target key <-> on-disk safetensors key
reconciliation.

Fixes the 2026-08-05 launch blocker where ``run_qat`` wrapped modules by their
LOADED model's own module-graph names (``model.layers...``) while the base
checkpoint's actual on-disk ``model.safetensors.index.json`` used a different
nesting (``model.language_model.layers...``) -- 390 of 391 adapter target
keys refused to merge, discovered only after an entire training run.

Pure-Python: no torch needed for these (the reconciliation itself is a string
transform over a key set).
"""

import pytest

from magicquant.qat.diskmap import (
    QATKeyReconciliationError,
    reconcile_key_to_disk,
    resolve_adapter_targets,
)


# ── reconcile_key_to_disk ────────────────────────────────────────────────────

def test_exact_match_short_circuits():
    keys = frozenset({"model.layers.0.self_attn.q_proj.weight"})
    assert (
        reconcile_key_to_disk("model.layers.0.self_attn.q_proj.weight", keys)
        == "model.layers.0.self_attn.q_proj.weight"
    )


def test_inserts_language_model_when_disk_key_is_nested():
    """The actual incident direction: module graph is bare, disk is nested."""
    keys = frozenset({"model.language_model.layers.0.self_attn.q_proj.weight"})
    resolved = reconcile_key_to_disk("model.layers.0.self_attn.q_proj.weight", keys)
    assert resolved == "model.language_model.layers.0.self_attn.q_proj.weight"


def test_removes_language_model_when_disk_key_is_bare():
    """The mirror-image case: module graph is nested, disk is bare."""
    keys = frozenset({"model.layers.0.self_attn.q_proj.weight"})
    resolved = reconcile_key_to_disk(
        "model.language_model.layers.0.self_attn.q_proj.weight", keys
    )
    assert resolved == "model.layers.0.self_attn.q_proj.weight"


def test_fused_expert_key_has_no_weight_suffix_and_still_reconciles():
    """3-D expert parameters are raw nn.Parameters -- no .weight suffix."""
    keys = frozenset({"model.language_model.layers.3.mlp.experts.gate_up_proj"})
    resolved = reconcile_key_to_disk(
        "model.layers.3.mlp.experts.gate_up_proj", keys
    )
    assert resolved == "model.language_model.layers.3.mlp.experts.gate_up_proj"


def test_no_match_anywhere_returns_none():
    keys = frozenset({"totally.unrelated.tensor.weight"})
    assert reconcile_key_to_disk("model.layers.0.self_attn.q_proj.weight", keys) is None


def test_a_name_that_is_not_model_prefixed_only_tries_exact_match():
    """lm_head.weight etc: neither transform applies, so only exact match works."""
    keys = frozenset({"lm_head.weight"})
    assert reconcile_key_to_disk("lm_head.weight", keys) == "lm_head.weight"
    assert reconcile_key_to_disk("head.weight", keys) is None


# ── resolve_adapter_targets ──────────────────────────────────────────────────

def test_resolve_adapter_targets_resolves_both_families():
    weight_map = {
        "model.language_model.layers.0.self_attn.q_proj.weight": "shard1",
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "shard2",
    }
    resolved_linear, resolved_expert = resolve_adapter_targets(
        ["model.layers.0.self_attn.q_proj"],
        ["model.layers.0.mlp.experts.gate_up_proj"],
        weight_map,
    )
    assert resolved_linear == {
        "model.layers.0.self_attn.q_proj": "model.language_model.layers.0.self_attn.q_proj"
    }
    assert resolved_expert == {
        "model.layers.0.mlp.experts.gate_up_proj":
            "model.language_model.layers.0.mlp.experts.gate_up_proj"
    }


def test_resolve_adapter_targets_raises_and_lists_every_failure_at_once():
    weight_map = {
        "model.language_model.layers.0.self_attn.q_proj.weight": "shard1",
        # no entry (in any reconcilable form) for down_proj or the expert.
    }
    with pytest.raises(QATKeyReconciliationError) as exc_info:
        resolve_adapter_targets(
            ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"],
            ["model.layers.0.mlp.experts.gate_up_proj"],
            weight_map,
        )
    msg = str(exc_info.value)
    assert "model.layers.0.mlp.down_proj.weight" in msg
    assert "model.layers.0.mlp.experts.gate_up_proj" in msg
    # the one that DOES resolve must not appear in the failure list
    assert "self_attn.q_proj" not in msg
    assert "2 of 3" in msg


def test_resolve_adapter_targets_empty_inputs_are_a_no_op():
    resolved_linear, resolved_expert = resolve_adapter_targets([], [], {})
    assert resolved_linear == {}
    assert resolved_expert == {}
