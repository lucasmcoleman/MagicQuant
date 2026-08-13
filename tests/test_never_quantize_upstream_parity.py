"""Drift tripwire: keep ``_NEVER_QUANTIZE_NAME_SUBSTRINGS`` level with the
never-quantize-by-name chain in llama.cpp's own quantizer.

The writer's list is a hand-transcribed mirror of upstream's
``quantize &= name.find("...") == std::string::npos;`` chain in
``src/llama-quant.cpp``. Hand transcription is exactly the failure mode this
file exists to catch: the first version of that list stopped 11 rules short,
and the omission of ``time_mix_lerp_fused.weight`` left an abort-class hole
(RWKV6/7 feed it to ``ggml_mul`` as ``src1``; a quantized copy makes
llama.cpp ``GGML_ABORT`` at load, the same way the motivating
``ssm_norm.weight`` bug did).

So rather than pin a second copy of the list, this re-derives it from a real
checkout and asserts coverage. Skips cleanly when no checkout is available,
the same way ``tests/integration/test_encoder_parity.py`` skips without
``llama-quantize``.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

import pytest

from magicquant.gguf.writer import (
    _NEVER_QUANTIZE_NAME_SUBSTRINGS,
    _is_never_quantized,
)

# `quantize &= name.find("<literal>") == std::string::npos;`
_UPSTREAM_RULE_RE = re.compile(
    r'quantize\s*&=\s*name\.find\(\s*"([^"]+)"\s*\)\s*==\s*std::string::npos'
)

# Upstream rules this mirror deliberately does NOT match literally, each for a
# stated reason. Anything that appears upstream and is not covered and not
# listed here is a genuine drift failure.
_KNOWN_DIVERGENCES = {
    # `quantize &= params->quantize_output_tensor || name != "output.weight"`
    # is conditional on a quantize param, not an unconditional refusal, and it
    # is an exact-name compare rather than a find(). MagicQuant models the
    # output tensor through its own head/H group instead.
    "output.weight",
}


def _find_llama_cpp_root() -> Optional[Path]:
    """The llama.cpp checkout to check against.

    Prefers the checkout whose build MagicQuant actually links, so this test
    tracks the library the encoder really calls rather than whichever source
    tree happens to be on disk.
    """
    env = os.environ.get("MAGICQUANT_LLAMA_CPP_DIR")
    if env and (Path(env) / "src" / "llama-quant.cpp").is_file():
        return Path(env)

    candidates: List[Path] = []
    try:
        from magicquant.quant.ggml_binding import _discover_libggml

        for lib in _discover_libggml():
            candidates.extend(Path(lib).resolve().parents)
    except Exception:
        pass

    candidates.extend(
        Path(p)
        for p in (
            "/home/lucas/llama.cpp",
            "/server/ai/llama.cpp",
            "/usr/local/src/llama.cpp",
        )
    )

    for cand in candidates:
        if (cand / "src" / "llama-quant.cpp").is_file():
            return cand
    return None


def _upstream_literals(root: Path) -> List[str]:
    text = (root / "src" / "llama-quant.cpp").read_text(errors="replace")
    return list(dict.fromkeys(_UPSTREAM_RULE_RE.findall(text)))


@pytest.fixture(scope="module")
def upstream_literals() -> List[str]:
    root = _find_llama_cpp_root()
    if root is None:
        pytest.skip(
            "no llama.cpp checkout found (set MAGICQUANT_LLAMA_CPP_DIR to enable "
            "the never-quantize drift tripwire)"
        )
    lits = _upstream_literals(root)
    if not lits:
        pytest.skip(
            f"{root}/src/llama-quant.cpp parsed to zero never-quantize rules -- "
            "upstream likely restructured the chain; update _UPSTREAM_RULE_RE"
        )
    return lits


def test_parser_finds_a_plausible_rule_count(upstream_literals):
    """Guard the guard: a regex that silently stops matching would make every
    coverage assertion below vacuously pass."""
    assert len(upstream_literals) >= 20, (
        f"only {len(upstream_literals)} never-quantize rules parsed out of "
        "llama-quant.cpp; the chain was probably restructured"
    )
    # Anchors present since the rule chain was introduced.
    assert "_norm.weight" in upstream_literals
    assert "ffn_gate_inp.weight" in upstream_literals


def test_every_upstream_rule_is_mirrored(upstream_literals):
    """Every name upstream refuses to quantize must also be refused here."""
    missing = [
        lit
        for lit in upstream_literals
        if lit not in _KNOWN_DIVERGENCES and not _is_never_quantized(lit)
    ]
    assert not missing, (
        "llama.cpp refuses to quantize these tensor names but MagicQuant's "
        "writer does not -- add them to _NEVER_QUANTIZE_NAME_SUBSTRINGS "
        f"(or to _KNOWN_DIVERGENCES with a reason): {missing}"
    )


def test_mirror_does_not_invent_rules(upstream_literals):
    """Every entry in our tuple should trace back to an upstream rule.

    Extras are not automatically wrong -- the two arch-templated names
    (POS_EMBD / TOKEN_TYPES) are matched by canonical GGUF name here because
    this writer has no per-arch tensor-name table -- but each one needs a
    stated reason, or the mirror quietly becomes a private policy that forces
    tensors to F32 for no upstream reason.
    """
    approximations = {
        # upstream: name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD, "weight")
        "position_embd.weight",
        # upstream: name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight")
        "token_types.weight",
    }
    unexplained = [
        substr
        for substr in _NEVER_QUANTIZE_NAME_SUBSTRINGS
        if substr not in approximations
        and not any(substr in lit for lit in upstream_literals)
    ]
    assert not unexplained, (
        "these entries do not correspond to any upstream never-quantize rule; "
        "either upstream dropped them or they were invented locally: "
        f"{unexplained}"
    )


@pytest.mark.parametrize(
    "name",
    [
        # The abort-class omission this tripwire was written for. ne is
        # (n_embd,1,1,6) -- 4-D, and n_embd is 256-divisible -- so neither the
        # 1-D rule nor the block-size fallback catches it.
        "blk.0.time_mix_lerp_fused.weight",
        # The rest of the time_mix set the first transcription stopped short of.
        "blk.3.time_mix_v0.weight",
        "blk.3.time_mix_a1.weight",
        "blk.3.time_mix_g2.weight",
        "blk.3.time_mix_decay_w2.weight",
        # get_rows operand.
        "blk.7.attn_rel_b.weight",
        # Audio projector: classifies into group E with a 32-divisible row, so
        # nothing else would have stopped it.
        "mm.a.code_embd.weight",
        # The original motivating bug, kept here so the two never regress apart.
        "blk.12.ssm_norm.weight",
    ],
)
def test_known_abort_class_names_are_refused(name):
    assert _is_never_quantized(name), f"{name} must never be quantized"
