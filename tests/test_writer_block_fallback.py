"""K-quant block-size incompatibility must fall back to a block-32 quant, not F32.

K-quants use a 256-element block, so a tensor whose contiguous row width isn't a
multiple of 256 can't be K-quantized. The writer used to fall back to F32 — fine
for the tiny norms it assumed, but catastrophic for MoE experts: diffusiongemma's
ffn_down_exps (row width 128) bloated a Q3_K pack from ~14 GB to 39 GB (31 GB of
F32). A block-32 scheme (MXFP4 / Q8_0) encodes those rows and respects the user's
size intent; F32 is kept only where it's actually required (SSM conv operands) or
when the row isn't even 32-divisible.
"""
from pathlib import Path

import numpy as np

import magicquant.gguf.source as source_mod
from magicquant.gguf.writer import create_hybrid_gguf, _block32_fallback

from tests.test_writer import StubSource


# ── unit: the fallback chooser ──────────────────────────────────────────────

def test_low_bit_target_falls_back_to_mxfp4():
    assert _block32_fallback("Q3_K", row_size=128, group="X") == "MXFP4"
    assert _block32_fallback("Q4_K", row_size=128, group="D") == "MXFP4"


def test_high_bit_target_falls_back_to_q8_0():
    assert _block32_fallback("Q6_K", row_size=128, group="O") == "Q8_0"
    assert _block32_fallback("Q5_K", row_size=128, group="Q") == "Q8_0"


def test_sub_4bit_iq_targets_fall_back_to_mxfp4_not_q8_0():
    # M1 regression: IQ2/IQ3 (2.06-3.44 bpw) are sub-4-bit, robust-tier
    # schemes -- they must land on the MXFP4 (4.25 bpw) side of the
    # fallback, not silently balloon to Q8_0 (8.5 bpw, a ~2x size blow-up).
    assert _block32_fallback("IQ2_XXS", row_size=128, group="X") == "MXFP4"
    assert _block32_fallback("IQ3_S", row_size=128, group="D") == "MXFP4"


def test_iq4_family_falls_back_to_mxfp4():
    # IQ4_XS/IQ4_NL (4.25/4.5 bpw) sit right at the low-bit/high-bit
    # boundary the old hand-maintained tuple only partially covered
    # (IQ4_NL was listed, IQ4_XS was not).
    assert _block32_fallback("IQ4_XS", row_size=128, group="U") == "MXFP4"
    assert _block32_fallback("IQ4_NL", row_size=128, group="U") == "MXFP4"


def test_ssm_group_stays_f32():
    # SSM conv1d operands must be F32 (llama.cpp asserts).
    assert _block32_fallback("Q4_K", row_size=128, group="S") == "F32"


def test_non_32_divisible_row_stays_f32():
    # 100 % 32 != 0 -> no block-32 scheme fits either.
    assert _block32_fallback("Q4_K", row_size=100, group="X") == "F32"


# ── integration: the writer honors it end to end ────────────────────────────

def _expert(name, n_expert, out_f, in_f):
    # row-major shape (n_expert, out, in); writer's row width = shape[-1] = in_f
    n = n_expert * out_f * in_f
    return (name, np.random.randn(n).astype(np.float32), (n_expert, out_f, in_f))


def test_incompatible_expert_packs_as_mxfp4_not_f32(tmp_path, monkeypatch):
    # in_f = 128: not /256 (K-quant fails) but /32 (MXFP4 fits).
    src = StubSource([
        _expert("blk.0.ffn_down_exps.weight", 4, 256, 128),
    ])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
    out = str(tmp_path / "h.gguf")
    create_hybrid_gguf(out, "ignored",
                       {"base": "Q4_K_M", "groups": {"X": "Q3_K"}}, verbose=False)

    from gguf import GGUFReader
    types = {t.name: t.tensor_type.name for t in GGUFReader(out).tensors}
    assert types["blk.0.ffn_down_exps.weight"] == "MXFP4", types


def test_create_hybrid_gguf_with_per_expert_imatrix(tmp_path, monkeypatch):
    """End-to-end: a stacked `_exps` tensor quantized with a real per-expert
    imatrix dict (the shape magicquant.imatrix.load_imatrix now produces)
    must build successfully through the full worker-thread pipeline, not
    just through a direct encode_to_ggml_bytes call."""
    n_expert, out_f, in_f = 4, 8, 32  # in_f=32 -> MXFP4/Q8_0 eligible
    name = "blk.0.ffn_down_exps.weight"
    src = StubSource([_expert(name, n_expert, out_f, in_f)])
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    rng = np.random.default_rng(3)
    imatrix = {name: rng.random(n_expert * in_f).astype(np.float32)}

    out = str(tmp_path / "h.gguf")
    result = create_hybrid_gguf(
        out, "ignored", {"base": "Q8_0", "groups": {"X": "Q8_0"}},
        verbose=False, imatrix=imatrix,
    )
    assert Path(result).is_file()
    assert Path(result).stat().st_size > 0
