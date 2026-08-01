"""Tensor-group classification tests.

Locks the MoE-expert classification fix (H1): ffn_{up,gate,down}_exps must all
map to group X (experts), not dense FFN U/D. Also pins dense + router + SSM names.

Also covers the unknown-tensor gate (loud-on-unknown): a new architecture's
novel tensor names that hit no explicit pattern AND no keyword heuristic must
not vanish silently into a caller's default quant scheme -- they're tracked
per-instance and surfaced via a single summary warning (classify_tensors()'s
end-of-pass call to warn_unclassified_once()), or a hard raise under
MAGICQUANT_STRICT_CLASSIFY=1. Each gate test builds its OWN classifier
instance (never the shared module-scoped `clf` fixture) so accumulated
unclassified state / the warn-once flag / strict-mode raises from one test
can never leak into another.
"""
import logging

import pytest

from magicquant.gguf.tensor_groups import TensorGroupClassifier

_LOGGER_NAME = "magicquant.gguf.tensor_groups"


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


# ---------------------------------------------------------------------------
# Unknown-tensor gate (loud-on-unknown)
# ---------------------------------------------------------------------------

# Known names spanning every explicit-pattern group plus the mtp/ssm regression
# names above -- none of these may ever be flagged unclassified.
_KNOWN_ARCH_NAMES = [
    "token_embd.weight",
    "output.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.attn_gate.weight",
    "blk.0.attn_norm.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_down.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_down_exps.weight",
    "blk.0.ffn_gate_up_exps.weight",
    "blk.0.ffn_gate_inp.weight",
    "blk.0.ssm_in.weight",
    "blk.46.nextn.eh_proj.weight",
    "blk.46.nextn.shared_head.head.weight",
    "blk.46.nextn.enorm.weight",
    "model.visual.patch_embed.weight",
]


def test_known_arch_names_never_flagged_unclassified(caplog):
    clf = TensorGroupClassifier()
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        clf.classify_tensors(_KNOWN_ARCH_NAMES)

    assert clf.unclassified == {"count": 0, "examples": []}
    assert not any(r.name == _LOGGER_NAME for r in caplog.records)


def test_novel_tensor_name_warns_with_count_and_example(caplog):
    clf = TensorGroupClassifier()
    # 12 synthetic 2D-looking names from a hypothetical new architecture --
    # "frobnicator" matches no explicit pattern and no keyword heuristic.
    names = [f"blk.{i}.frobnicator.weight" for i in range(12)]

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        grouped = clf.classify_tensors(names)

    # Still classified as UNKNOWN for the caller (unchanged public contract) ...
    assert grouped["UNKNOWN"] == names

    # ... but tracked: full count kept, example list capped at 10.
    report = clf.unclassified
    assert report["count"] == 12
    assert len(report["examples"]) == 10
    assert report["examples"] == names[:10]

    # Exactly one warning fired (not one per tensor), and it names the count
    # plus at least one example.
    warnings = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "12" in message
    assert "blk.0.frobnicator.weight" in message


def test_warn_fires_once_per_instance_even_across_repeated_calls(caplog):
    clf = TensorGroupClassifier()
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        # Simulate a writer/orchestrator-style per-tensor loop (classify_tensor
        # called directly, repeatedly) followed by an explicit end-of-pass
        # check -- and then a second full pass reusing the same instance.
        for _ in range(3):
            for name in ("blk.0.frobnicator.weight", "blk.1.frobnicator.weight"):
                clf.classify_tensor(name)
            clf.warn_unclassified_once()

    warnings = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(warnings) == 1
    # Count keeps accumulating across every classify_tensor call this
    # instance ever sees, even though only one warning is ever logged.
    assert clf.unclassified["count"] == 6


def test_strict_mode_raises_on_first_unclassified_tensor(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_STRICT_CLASSIFY", "1")
    clf = TensorGroupClassifier()

    with pytest.raises(ValueError, match="MAGICQUANT_STRICT_CLASSIFY"):
        clf.classify_tensor("blk.0.frobnicator.weight")


def test_strict_mode_does_not_raise_for_known_names(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_STRICT_CLASSIFY", "1")
    clf = TensorGroupClassifier()

    for name in _KNOWN_ARCH_NAMES:
        clf.classify_tensor(name)  # must not raise

    assert clf.unclassified == {"count": 0, "examples": []}


@pytest.mark.parametrize("name", [
    "blk.0.frobnicator.bias",
    "blk.0.frobnicator_bias",
    "blk.0.frobnicator.scale",
    "blk.0.frobnicator_scale",
])
def test_1d_ish_novel_names_never_flagged_unclassified(name, caplog):
    clf = TensorGroupClassifier()
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = clf.classify_tensor(name)

    assert result == "UNKNOWN"  # still unrecognized as a group ...
    assert clf.unclassified == {"count": 0, "examples": []}  # ... but not tracked
    assert not any(r.name == _LOGGER_NAME for r in caplog.records)
