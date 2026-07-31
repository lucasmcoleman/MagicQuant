"""Loading the per-group hybrid config (group -> ggml_type_name) for QAT.

The orchestrator writes ``search_results.json`` with a ``tiered`` map; each tier's
``config`` is ``{group: scheme_name}`` where ``scheme_name`` is a MagicQuant
identifier ("MXFP4_MOE", "Q4_K_M", ...). QAT's fake-quant dispatches by the
*ggml* type name ("MXFP4", "Q4_K", ...), so ``load_hybrid_config`` maps each
scheme name to its ggml_type_name via the canonical scheme registry.
"""

import json
from pathlib import Path

import pytest

from magicquant.qat.config import load_hybrid_config

FIXTURE = Path(__file__).parent / "fixtures" / "search_results_sample.json"


def test_loads_tier_and_maps_to_ggml_type_names():
    cfg = load_hybrid_config(str(FIXTURE), tier="Q4")
    assert cfg == {
        "E": "BF16",
        "H": "Q8_0",
        "Q": "Q4_K",
        "K": "Q4_K",
        "O": "Q5_K",
        "U": "MXFP4",
        "D": "MXFP4",
        "X": "MXFP4",
        "R": "Q8_0",
    }


def test_loads_a_different_tier():
    cfg = load_hybrid_config(str(FIXTURE), tier="Q6")
    assert cfg == {
        "E": "BF16",
        "H": "BF16",
        "Q": "Q6_K",
        "K": "Q6_K",
        "O": "Q6_K",
        "U": "Q6_K",
        "D": "Q6_K",
    }


def test_accepts_a_path_object():
    cfg = load_hybrid_config(FIXTURE, tier="Q4")
    assert cfg["U"] == "MXFP4"


def test_missing_tier_raises_with_available_tiers(tmp_path):
    with pytest.raises(KeyError) as exc:
        load_hybrid_config(str(FIXTURE), tier="Q2")
    # error mentions the tiers actually present so the caller can recover
    assert "Q4" in str(exc.value) and "Q6" in str(exc.value)


def test_unknown_scheme_name_passes_through_unchanged(tmp_path):
    """A group whose scheme isn't in the registry keeps its name (so the
    fake-quant dispatcher can warn + fall back rather than the loader crashing)."""
    p = tmp_path / "sr.json"
    p.write_text(
        json.dumps({"tiered": {"Q4": {"config": {"U": "TOTALLY_MADE_UP"}}}})
    )
    cfg = load_hybrid_config(str(p), tier="Q4")
    assert cfg == {"U": "TOTALLY_MADE_UP"}


# ── tier_scheme_version compatibility (2026-07 fix) ──────────────────────

def test_legacy_fixture_has_no_tier_scheme_version():
    """FIXTURE predates the versioning stamp entirely -- confirms every test
    above already exercises the legacy/compat path, not just a version==1
    file constructed for the occasion."""
    raw = json.loads(FIXTURE.read_text())
    assert "tier_scheme_version" not in raw


def test_legacy_file_still_loads_but_warns(monkeypatch, tmp_path):
    """A pre-versioning search_results.json (no tier_scheme_version key)
    must still load successfully -- old artifacts keep working -- but logs a
    non-fatal warning so a QAT run doesn't silently assume 'Q5' means what
    it means today."""
    import magicquant.qat.config as config_mod

    warnings = []
    monkeypatch.setattr(
        config_mod.log, "warning", lambda msg, **kw: warnings.append((msg, kw))
    )
    cfg = load_hybrid_config(str(FIXTURE), tier="Q4")
    assert cfg["U"] == "MXFP4"  # loaded correctly despite being legacy
    assert len(warnings) == 1
    assert warnings[0][1]["tier_scheme_version"] == 1


def test_explicit_legacy_version_1_also_warns(monkeypatch, tmp_path):
    p = tmp_path / "sr.json"
    p.write_text(json.dumps({
        "tier_scheme_version": 1,
        "tiered": {"Q4": {"config": {"U": "MXFP4_MOE"}}},
    }))
    import magicquant.qat.config as config_mod

    warnings = []
    monkeypatch.setattr(
        config_mod.log, "warning", lambda msg, **kw: warnings.append((msg, kw))
    )
    load_hybrid_config(str(p), tier="Q4")
    assert len(warnings) == 1


def test_current_version_file_does_not_warn(monkeypatch, tmp_path):
    from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION
    import magicquant.qat.config as config_mod

    p = tmp_path / "sr.json"
    p.write_text(json.dumps({
        "tier_scheme_version": CURRENT_TIER_SCHEME_VERSION,
        "tiered": {"Q4": {"config": {"U": "MXFP4_MOE"}}},
    }))
    warnings = []
    monkeypatch.setattr(
        config_mod.log, "warning", lambda msg, **kw: warnings.append((msg, kw))
    )
    cfg = load_hybrid_config(str(p), tier="Q4")
    assert cfg["U"] == "MXFP4"
    assert warnings == []
