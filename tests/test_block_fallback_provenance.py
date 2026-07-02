"""Block-size fallback provenance: the writer must record WHICH tensors got
silently downgraded from their requested K-quant, not just print it under
verbose=True.

See ``magicquant.gguf.writer._block32_fallback`` (the policy: SSM/group "S"
or non-32-divisible rows -> F32; otherwise a block-32 quant, MXFP4 for
low-bit targets / Q8_0 for high-bit) and ``GGUFWriter._fallbacks`` (the
provenance log populated during Pass 1 of ``create_hybrid_gguf``).
"""
import numpy as np

import magicquant.gguf.source as source_mod
from magicquant.gguf.writer import GGUFWriter, _block32_fallback

from tests.test_writer import StubSource


# ── unit: the fallback chooser's type decisions (policy, unchanged here) ───

def test_ssm_group_stays_f32():
    assert _block32_fallback("Q4_K", row_size=128, group="S") == "F32"


def test_non_32_divisible_row_stays_f32():
    assert _block32_fallback("Q4_K", row_size=100, group="X") == "F32"


def test_low_bit_target_falls_back_to_mxfp4():
    assert _block32_fallback("Q4_K", row_size=128, group="D") == "MXFP4"


def test_high_bit_target_falls_back_to_q8_0():
    assert _block32_fallback("Q6_K", row_size=128, group="O") == "Q8_0"


# ── a fresh writer starts with an empty provenance log ──────────────────────

def test_fresh_writer_has_empty_fallback_log(tmp_path):
    writer = GGUFWriter(str(tmp_path / "out.gguf"))
    assert writer._fallbacks == []


# ── integration: the writer records provenance for a real fallback ─────────

def _expert(name, n_expert, out_f, in_f):
    # row-major shape (n_expert, out, in); writer's row width = shape[-1] = in_f
    n = n_expert * out_f * in_f
    return (name, np.random.randn(n).astype(np.float32), (n_expert, out_f, in_f))


def test_writer_records_block_size_fallback_provenance(tmp_path, monkeypatch):
    # in_f=128: not /256 (Q3_K K-quant fails) but /32 (MXFP4 fits) -> fallback.
    src = StubSource([
        _expert("blk.0.ffn_down_exps.weight", 4, 256, 128),
    ])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"X": "Q3_K"}}, verbose=False,
    )

    assert len(writer._fallbacks) == 1
    record = writer._fallbacks[0]
    assert record["tensor"] == "blk.0.ffn_down_exps.weight"
    assert record["group"] == "X"
    assert record["requested"] == "Q3_K"
    assert record["actual"] == "MXFP4"
    assert record["reason"] == "block-size"


def test_writer_no_fallback_when_rows_are_block_compatible(tmp_path, monkeypatch):
    # in_f=256: divides evenly into a K-quant's 256-block -> no fallback.
    src = StubSource([
        _expert("blk.0.ffn_down_exps.weight", 4, 256, 256),
    ])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    writer.create_hybrid_gguf(
        "ignored", {"base": "Q4_K_M", "groups": {"X": "Q3_K"}}, verbose=False,
    )

    assert writer._fallbacks == []


def test_writer_logs_one_summary_line_on_fallback(tmp_path, monkeypatch, caplog):
    # verbose=False must still surface a single summary log (data-integrity
    # notice), not the per-tensor [COMPAT] print (which is verbose-gated).
    src = StubSource([
        _expert("blk.0.ffn_down_exps.weight", 4, 256, 128),
    ])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "h.gguf"))
    with caplog.at_level("WARNING", logger="magicquant.gguf.writer"):
        writer.create_hybrid_gguf(
            "ignored", {"base": "Q4_K_M", "groups": {"X": "Q3_K"}}, verbose=False,
        )

    fallback_records = [
        r for r in caplog.records
        if "fell back from their requested quant" in r.getMessage()
    ]
    assert len(fallback_records) == 1, caplog.text
    assert "blk.0.ffn_down_exps.weight" in fallback_records[0].getMessage()
    assert "Q3_K->MXFP4" in fallback_records[0].getMessage()
