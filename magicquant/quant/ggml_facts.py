"""
Canonical ggml type facts — single source of truth for name<->id, block
size, and type (encoded-byte) size, for every ggml quantization scheme
MagicQuant touches.

Why this module exists: before it, FOUR hand-maintained tables shadowed the
same upstream facts (converters.py's GGML_BLOCK_SIZE/GGML_TYPE_SIZE,
ggml_binding.py's _GGML_BLOCK_SIZE/_GGML_TYPE_SIZE/GGML_TYPE_IDS, writer.py's
GGML_TYPE, source.py's GGUFSource._TYPE_NAME) — copy-pasted from
ggml/src/ggml-quants.h by hand and free to drift from each other (they once
did: IQ4_XS was wrong in one copy, silently corrupting Pass-1 GGUF offsets
the moment PR3 registered it). Per the mission ("facts about formats/types/
archs should be IMPORTED from upstream packages ... never silently drift"),
stock facts now come from the `gguf` package — llama.cpp's own pure-python
package, a hard MagicQuant dependency (see pyproject.toml) — at import time.
Only the ROCmFPX fork's out-of-band types (ids 100-104, unknown to stock
ggml/gguf) are still hand-maintained, and ONLY here.

Exports:
    NAME_TO_ID : name -> ggml_type id                 (stock + fork)
    ID_TO_NAME : ggml_type id -> name                  (stock + fork)
    BLOCK_SIZE : name -> elements per block            (stock + fork)
    TYPE_SIZE  : name -> encoded bytes per block       (stock + fork)
    FORK_TYPES : the ONE fork-only registry (id/block/size/registered_name)
    ROCMFPX_TYPE_NAMES : frozenset of fork type names
    expected_size : name x n_elements -> encoded byte size (the ONE size formula)

Fail-safe: if `gguf` cannot be imported, this module raises ImportError at
import time (the normal Python behavior for a missing hard dependency) —
never falls back to a stale hand-table. Callers needing a softer failure
mode (e.g. purely optional tooling) should catch ImportError themselves;
MagicQuant's own quant path requires this module to succeed.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

import gguf.constants as _gguf_constants

_QuantType = _gguf_constants.GGMLQuantizationType
_QUANT_SIZES = _gguf_constants.GGML_QUANT_SIZES

# ---------------------------------------------------------------------------
# Stock types, derived from the installed `gguf` package.
#
# Name compatibility: MagicQuant's local type names (e.g. "Q4_K", "BF16",
# "MXFP4", "IQ4_XS") were verified byte-for-byte against
# gguf.constants.GGMLQuantizationType.__members__ (gguf package installed in
# the Foundry venv, 2026-07-28) — every name MagicQuant uses matches the
# upstream enum member's `.name` exactly, with the SAME numeric id and the
# SAME (block, size) from GGML_QUANT_SIZES. No normalization needed. If a
# future gguf upgrade renames or renumbers a member, the frozen-snapshot
# tripwire test (tests/test_ggml_facts_snapshot.py) fails loudly so a human
# reviews the diff instead of the tables silently drifting apart again.
# ---------------------------------------------------------------------------
_STOCK_NAME_TO_ID: Dict[str, int] = {
    member.name: int(member.value) for member in _QuantType
}

def _build_stock_size_tables():
    """Isolate loop variables in a function scope (nothing to scrub at module level)."""
    block_size: Dict[str, int] = {}
    type_size: Dict[str, int] = {}
    for member in _QuantType:
        sizes = _QUANT_SIZES.get(member)
        if sizes is None:
            continue  # reserved/extension ids with no published block layout
        block, size = sizes
        block_size[member.name] = block
        type_size[member.name] = size
    return block_size, type_size


_STOCK_BLOCK_SIZE, _STOCK_TYPE_SIZE = _build_stock_size_tables()

# ---------------------------------------------------------------------------
# Policy guard: names MagicQuant itself dispatches/writes MUST exist in
# whatever `gguf` version is installed, or a too-old `gguf` fails silently
# rather than loudly -- e.g. `gguf==0.10.0` predates the MXFP4/TQ1_0/TQ2_0
# GGMLQuantizationType members, so code referencing "MXFP4" either KeyErrors
# deep in a run or (worse) the GGUF writer's block-size compatibility check
# quietly falls back to Q4_0 instead of failing at import time.
#
# This is a POLICY list (which stock names MagicQuant's own scheme registry
# and writer actually use), NOT a shadow fact table -- the real facts
# (id/block/size) still come entirely from the installed `gguf` package
# above; REQUIRED_STOCK_NAMES only asserts those facts EXIST for the names
# MagicQuant needs.
#
# Kept LITERAL rather than derived from
# `{s.ggml_type_name for s in schemes.get_all_schemes()}`: this module
# cannot import magicquant.quant.schemes without circularity (schemes are
# meant to read verified facts FROM ggml_facts, not the other way around).
# magicquant/quant/schemes.py's registry is the reference this list must
# track by hand; tests/test_ggml_facts_snapshot.py cross-checks
# REQUIRED_STOCK_NAMES superset-of {s.ggml_type_name for s in
# get_all_schemes()} (that test CAN import both, so it catches drift this
# module structurally cannot check itself).
#
# F32/F16/BF16 are included even though schemes.py's registry only lists
# BF16 as a scheme target: writer.py dispatches to F32/F16 directly for
# forced-precision paths (SSM/group-S operand fallback to F32, BF16->F16
# offset math) even though they aren't scheme *encode* targets.
REQUIRED_STOCK_NAMES: FrozenSet[str] = frozenset({
    "F32", "F16", "BF16",
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K",
    "Q4_0", "Q4_1",
    "IQ4_NL", "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS",
    "IQ1_M", "IQ1_S",
    "MXFP4",
})


def _check_required_stock_names_present() -> None:
    """Raise ImportError (naming the missing entries) if the installed
    `gguf` package's GGMLQuantizationType enum lacks any name
    REQUIRED_STOCK_NAMES lists. The failure this prevents is NOT an
    exception at all otherwise -- it's a silently wrong quant (see the
    REQUIRED_STOCK_NAMES comment above), so this must run at import time,
    before any dispatch happens.
    """
    missing = sorted(REQUIRED_STOCK_NAMES - set(_STOCK_NAME_TO_ID))
    if missing:
        raise ImportError(
            f"magicquant.quant.ggml_facts: the installed `gguf` package is "
            f"missing GGMLQuantizationType member(s) {missing} that "
            f"MagicQuant requires (see REQUIRED_STOCK_NAMES in this module). "
            f"This usually means `gguf` is older than the pyproject.toml "
            f"floor -- upgrade gguf."
        )


_check_required_stock_names_present()

# ---------------------------------------------------------------------------
# Fork-only types: ROCmFPX (https://github.com/ciru-ai/ROCmFPX), a
# llama.cpp fork targeting this box's Strix Halo (gfx1151) hardware. These
# ids (100-104) sit past stock ggml's GGML_TYPE_COUNT and are NOT in the
# gguf package at all — they are genuinely fork-owned facts, not something
# any upstream import can supply. This dict is the ONE place they live;
# every other module (ggml_binding, writer, source, converters) imports
# from here rather than keeping its own copy.
#
# block/size verified against the fork's block structs (ggml/rocmfp4/
# rocmfp4.h, ggml/rocmfpx/rocmfpx.h): rocmfp4 = 16 qs + 2 scale bytes;
# fast = 16 qs + 1 scale byte; fp3 = 12 qs + 2 scale; fp6 = 24 qs + 2 scale;
# fp8 = 32 qs + 1 scale. All fork types use 32-element blocks.
#
# `registered_name` is the lowercase string the fork's ggml type_traits
# table registers the type under (used by ggml_binding's
# ggml_type_from_name-based support probe — see _probe_rocmfpx there).
#
# Validated at runtime against the loaded fork libggml, when one is loaded,
# by ggml_binding._LibggmlHandle._probe_rocmfpx / _verify_type_ids (never
# assumed present — a stock libggml legitimately doesn't have these).
# ---------------------------------------------------------------------------
FORK_TYPES: Dict[str, Dict[str, object]] = {
    "Q4_0_ROCMFP4": {
        "id": 100, "block": 32, "size": 18,
        "registered_name": "q4_0_rocmfp4",
    },
    "Q4_0_ROCMFP4_FAST": {
        "id": 101, "block": 32, "size": 17,
        "registered_name": "q4_0_rocmfp4_fast",
    },
    "Q6_0_ROCMFPX": {
        "id": 102, "block": 32, "size": 26,
        "registered_name": "q6_0_rocmfpx",
    },
    "Q8_0_ROCMFPX": {
        "id": 103, "block": 32, "size": 33,
        "registered_name": "q8_0_rocmfpx",
    },
    "Q3_0_ROCMFPX": {
        "id": 104, "block": 32, "size": 14,
        "registered_name": "q3_0_rocmfpx",
    },
}

ROCMFPX_TYPE_NAMES: FrozenSet[str] = frozenset(FORK_TYPES)

def _check_no_fork_stock_collision() -> None:
    """Refuse to import if a fork id lands on a stock id.

    That would mean the fork's ggml.h type enum now overlaps stock (e.g. a
    rebase picked up a new stock type at one of 100-104) — a silent
    corruption hazard (encode/decode would dispatch to the wrong scheme).
    """
    fork_ids = {info["id"] for info in FORK_TYPES.values()}
    collisions = fork_ids & set(_STOCK_NAME_TO_ID.values())
    if collisions:
        raise RuntimeError(
            f"magicquant.quant.ggml_facts: FORK_TYPES ids {sorted(collisions)} "
            f"collide with stock gguf ids. This means the ROCmFPX fork's "
            f"ggml.h type enum now overlaps stock ggml — check ROCmFPX's "
            f"ggml/include/ggml.h against the installed `gguf` package's "
            f"GGMLQuantizationType before proceeding; encode/decode dispatch "
            f"would otherwise silently hit the wrong scheme."
        )


_check_no_fork_stock_collision()

# ---------------------------------------------------------------------------
# Merged canonical tables (stock ∪ fork).
# ---------------------------------------------------------------------------
NAME_TO_ID: Dict[str, int] = dict(_STOCK_NAME_TO_ID)
NAME_TO_ID.update({name: info["id"] for name, info in FORK_TYPES.items()})

ID_TO_NAME: Dict[int, str] = {v: k for k, v in NAME_TO_ID.items()}

BLOCK_SIZE: Dict[str, int] = dict(_STOCK_BLOCK_SIZE)
BLOCK_SIZE.update({name: info["block"] for name, info in FORK_TYPES.items()})

TYPE_SIZE: Dict[str, int] = dict(_STOCK_TYPE_SIZE)
TYPE_SIZE.update({name: info["size"] for name, info in FORK_TYPES.items()})

# ---------------------------------------------------------------------------
# DOCUMENTED upstream-bug correction: Q8_1 (id=9).
#
# gguf==0.18.0's GGML_QUANT_SIZES publishes Q8_1 as (block=32, size=40) --
# the OLD "float d; float s; qs[32]" struct layout. Every real libggml
# cross-probed reports 36 bytes/block instead (the CURRENT
# "ggml_fp16_t d; ggml_fp16_t s; qs[32]" struct: 2+2+32 = 36), confirmed
# 2026-07-28 against FOUR independent builds: /server/ai/llama.cpp,
# /home/lucas/llama.cpp-build, /home/lucas/llama.cpp, and the ROCmFPX fork
# (/home/lucas/ROCmFPX/build-strix-rocmfp4). All four agree on 36; gguf-py
# is the outlier. Q8_1 is an ephemeral CPU dot-product intermediate type --
# never written into an on-disk GGUF and not dispatched to by any
# MagicQuant scheme -- so this correction has no on-disk-format
# implications; it only fixes TYPE_SIZE['Q8_1'] to match reality for
# anything that does look it up.
#
# tests/test_ggml_facts_snapshot.py's test_q8_1_known_upstream_staleness
# pins BOTH sides of this: our exported TYPE_SIZE['Q8_1'] == 36 (this
# override), AND gguf's raw GGML_QUANT_SIZES still says 40 (the bug is
# still present upstream). The moment a future gguf release fixes its
# Q8_1 constant to 36, that second assertion fails -- which is the signal
# to remove this override (it would then be redundant with upstream) and
# revisit ggml_binding.py's _verify_type_ids docstring, which still
# describes the pre-override state.
TYPE_SIZE["Q8_1"] = 36

# ---------------------------------------------------------------------------
# Encoded-byte-size arithmetic: single canonical body.
#
# magicquant.quant.converters.ggml_tensor_data_size (Pass-1 GGUF offset math)
# and magicquant.quant.ggml_binding._expected_size (ctypes output-buffer
# sizing) used to each hand-roll this exact ceil-div computation over their
# own module-local copies of BLOCK_SIZE/TYPE_SIZE. Both are now thin
# delegates to this function, so the writer's offset math and the encoder's
# buffer-size math are structurally the same computation rather than two
# independently-maintained copies that happen to agree today.
# ---------------------------------------------------------------------------

def expected_size(name: str, n_elements: int) -> int:
    """Return the encoded byte size for `n_elements` scalars of ggml type
    `name`.

    Fallback defaults are semantic, not sloppy: an unrecognized type name is
    treated as block=1 (no blocking) / type_size=2 bytes/element (F16-like).
    Do NOT tighten this to a KeyError -- this reproduces the exact fallback
    behavior of the two pre-fold copies (converters.ggml_tensor_data_size and
    ggml_binding._expected_size), and a behavior-preserving fold must keep it
    even though no current caller is known to hit it.
    """
    block_size = BLOCK_SIZE.get(name, 1)
    type_size = TYPE_SIZE.get(name, 2)
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * type_size
