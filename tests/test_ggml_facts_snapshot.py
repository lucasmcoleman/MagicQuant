"""Drift tripwire for magicquant.quant.ggml_facts.

ggml_facts derives its stock (name, id, block, size) tuples from the
installed `gguf` package instead of a hand-copied table (see that module's
docstring for the history of why: four separate hand tables used to shadow
these facts and could silently drift apart, which is exactly what happened
once with IQ4_XS).

This test pastes a FROZEN SNAPSHOT of the tuples the old hand-maintained
tables agreed on (converters.py / ggml_binding.py / writer.py / source.py,
as they stood before the ggml_facts rework) and asserts the derived tables
still produce them. If a future `gguf` upgrade renumbers or renames a type,
this test fails LOUDLY instead of the tables silently drifting to match
the new upstream values without anyone noticing -- a human has to look at
the diff and decide whether it's a legitimate upstream renumbering or
something to pin against.

Do NOT "fix" a failure here by just updating the snapshot to whatever
ggml_facts currently produces -- that defeats the entire point. Investigate
why the underlying `gguf` package changed first.
"""
from magicquant.quant import ggml_facts
from magicquant.quant.schemes import get_all_schemes


# (name, id, block_size, type_size) -- copied verbatim from the pre-rework
# hand tables. Every one of these was hand-verified against
# ggml/src/ggml-quants.h historically; this is the contract ggml_facts must
# keep reproducing.
# NOTE on Q8_1 (id=9): deliberately EXCLUDED from this snapshot. The
# pre-rework hand table said (32, 36); the installed `gguf` package
# (0.18.0) instead publishes (32, 40) -- a confirmed, narrow staleness in
# gguf-py itself (its GGML_QUANT_SIZES formula for Q8_1 is the OLD
# "float d; float s; qs[32]" struct; real ggml has used the current
# "ggml_fp16_t d; ggml_fp16_t s; qs[32]" struct = 36 bytes for a long time).
# Cross-probed 2026-07-28 against three independent stock libggml builds
# (/server/ai/llama.cpp, /home/lucas/llama.cpp-build, /home/lucas/llama.cpp)
# plus the ROCmFPX fork -- all four report 36, agreeing with the OLD hand
# table and disagreeing with gguf-py. Q8_1 is an ephemeral CPU
# dot-product intermediate type: never written into an on-disk GGUF, and
# not offered by any MagicQuant scheme (see get_all_schemes()), so
# ggml_binding._verify_type_ids' hard-crash gate is deliberately scoped to
# exclude it (see that method's docstring) rather than either (a) crashing
# construction of every libggml handle over a type nobody quantizes into,
# or (b) hand-overriding the upstream value here, which would recreate
# exactly the kind of shadow fact this whole rework exists to eliminate.
# _cross_check_block_type_sizes still logs this mismatch loudly every time
# a real handle is constructed (see test_ggml_binding_cross_check.py).
# See test_q8_1_known_upstream_staleness below, which pins gguf-py's
# CURRENT (wrong) value so a future gguf release that fixes it is
# immediately visible as a test failure here -- at which point this whole
# NOTE, the exclusion in _verify_type_ids, and this pinned test should be
# revisited together.
_FROZEN_STOCK_SNAPSHOT = [
    ("F32", 0, 1, 4),
    ("F16", 1, 1, 2),
    ("Q4_0", 2, 32, 18),
    ("Q4_1", 3, 32, 20),
    ("Q5_0", 6, 32, 22),
    ("Q5_1", 7, 32, 24),
    ("Q8_0", 8, 32, 34),
    ("Q2_K", 10, 256, 84),
    ("Q3_K", 11, 256, 110),
    ("Q4_K", 12, 256, 144),
    ("Q5_K", 13, 256, 176),
    ("Q6_K", 14, 256, 210),
    ("Q8_K", 15, 256, 292),
    ("IQ2_XXS", 16, 256, 66),
    ("IQ2_XS", 17, 256, 74),
    ("IQ3_XXS", 18, 256, 98),
    ("IQ1_S", 19, 256, 50),
    ("IQ4_NL", 20, 32, 18),
    ("IQ3_S", 21, 256, 110),
    ("IQ2_S", 22, 256, 82),
    ("IQ4_XS", 23, 256, 136),
    ("I8", 24, 1, 1),
    ("I16", 25, 1, 2),
    ("I32", 26, 1, 4),
    ("I64", 27, 1, 8),
    ("F64", 28, 1, 8),
    ("IQ1_M", 29, 256, 56),
    ("BF16", 30, 1, 2),
    ("MXFP4", 39, 32, 17),
]

# Fork-only types: (name, id, block_size, type_size, registered_name) --
# copied verbatim from the pre-rework ggml_binding.py ROCMFPX_TYPE_IDS /
# _GGML_BLOCK_SIZE / _GGML_TYPE_SIZE / _ROCMFPX_REGISTERED_NAME.
_FROZEN_FORK_SNAPSHOT = [
    ("Q4_0_ROCMFP4", 100, 32, 18, "q4_0_rocmfp4"),
    ("Q4_0_ROCMFP4_FAST", 101, 32, 17, "q4_0_rocmfp4_fast"),
    ("Q6_0_ROCMFPX", 102, 32, 26, "q6_0_rocmfpx"),
    ("Q8_0_ROCMFPX", 103, 32, 33, "q8_0_rocmfpx"),
    ("Q3_0_ROCMFPX", 104, 32, 14, "q3_0_rocmfpx"),
]


def test_stock_snapshot_unchanged():
    """Every stock (name, id, block, size) tuple the old hand tables agreed
    on must still come out of ggml_facts, derived from the `gguf` package."""
    for name, type_id, block, size in _FROZEN_STOCK_SNAPSHOT:
        assert name in ggml_facts.NAME_TO_ID, (
            f"{name}: no longer present in ggml_facts.NAME_TO_ID -- did a "
            f"gguf upgrade rename this type? Check "
            f"gguf.constants.GGMLQuantizationType before updating this "
            f"snapshot."
        )
        assert ggml_facts.NAME_TO_ID[name] == type_id, (
            f"{name}: id drifted from {type_id} to "
            f"{ggml_facts.NAME_TO_ID[name]} -- a gguf upgrade renumbered "
            f"this type. This is a breaking change for on-disk GGUFs; "
            f"investigate before updating the snapshot."
        )
        assert ggml_facts.BLOCK_SIZE[name] == block, (
            f"{name}: block_size drifted from {block} to "
            f"{ggml_facts.BLOCK_SIZE[name]}"
        )
        assert ggml_facts.TYPE_SIZE[name] == size, (
            f"{name}: type_size drifted from {size} to "
            f"{ggml_facts.TYPE_SIZE[name]}"
        )


def test_q8_1_known_upstream_staleness():
    """Pins BOTH sides of ggml_facts' documented Q8_1 correction.

    See the NOTE above _FROZEN_STOCK_SNAPSHOT and the override comment
    directly above ``TYPE_SIZE["Q8_1"] = 36`` in ggml_facts.py for the full
    story. Two independent facts are asserted:

      1. Our EXPORTED value is the corrected 36 (matching every real
         libggml cross-probed) -- this is the override actually in effect.
      2. gguf-py's RAW GGML_QUANT_SIZES constant is still the stale 40 --
         i.e. the upstream bug the override exists to correct is still
         present.

    If a future `gguf` release fixes its Q8_1 constant to 36, assertion (2)
    fails -- THIS is the signal to remove the override in ggml_facts.py
    (it would then be redundant with a now-correct upstream) and revisit
    ggml_binding._verify_type_ids' docstring, which still describes the
    pre-override state (uncorrected, matching gguf-py, Q8_1 scoped out of
    the load-bearing hard-crash gate for staleness reasons that would no
    longer apply). Do NOT "fix" a failure here by just updating (2)'s
    expected value -- that defeats the tripwire's purpose.
    """
    assert ggml_facts.NAME_TO_ID["Q8_1"] == 9
    assert ggml_facts.BLOCK_SIZE["Q8_1"] == 32
    assert ggml_facts.TYPE_SIZE["Q8_1"] == 36, (
        "ggml_facts.TYPE_SIZE['Q8_1'] is no longer 36 -- did the override "
        "in ggml_facts.py get removed or changed? It should keep exporting "
        "the real-libggml-verified 36, not gguf-py's raw (stale) value."
    )
    raw_size = _gguf_quant_sizes()[9][1]
    assert raw_size == 40, (
        f"gguf-py's raw GGML_QUANT_SIZES for Q8_1 (id=9) changed from the "
        f"known-stale 40 to {raw_size} -- if it's now 36 (matching real "
        f"libggml), the upstream bug ggml_facts.py's Q8_1 override exists "
        f"to correct has been fixed: remove that override, update its "
        f"comment and this test, and revisit "
        f"ggml_binding._verify_type_ids' docstring (see this test's "
        f"docstring above)."
    )


def _gguf_quant_sizes():
    """gguf-py's raw (uncorrected) GGML_QUANT_SIZES table, keyed by numeric
    id -- deliberately read straight from gguf.constants rather than through
    ggml_facts, so this test observes upstream's value independent of any
    correction ggml_facts.py applies on top of it."""
    import gguf.constants as _gguf_constants

    quant_type = _gguf_constants.GGMLQuantizationType
    sizes = _gguf_constants.GGML_QUANT_SIZES
    return {int(member.value): sizes[member] for member in quant_type if member in sizes}


def test_fork_snapshot_unchanged():
    """ROCmFPX fork facts are hand-maintained in ggml_facts.FORK_TYPES --
    this locks their values so an edit there is deliberate, not a typo."""
    for name, type_id, block, size, registered_name in _FROZEN_FORK_SNAPSHOT:
        assert name in ggml_facts.FORK_TYPES
        info = ggml_facts.FORK_TYPES[name]
        assert info["id"] == type_id
        assert info["block"] == block
        assert info["size"] == size
        assert info["registered_name"] == registered_name
        # Also reachable through the merged canonical tables.
        assert ggml_facts.NAME_TO_ID[name] == type_id
        assert ggml_facts.BLOCK_SIZE[name] == block
        assert ggml_facts.TYPE_SIZE[name] == size


def test_fork_ids_never_collide_with_stock():
    """Fork ids (100-104) must never coincide with a stock gguf id.

    ggml_facts already enforces this at import time (raises RuntimeError on
    collision); this test additionally locks the id RANGE assumption so a
    reviewer notices if a fork id is ever moved into the stock range.
    """
    fork_ids = {info["id"] for info in ggml_facts.FORK_TYPES.values()}
    stock_ids = {
        v for k, v in ggml_facts.NAME_TO_ID.items()
        if k not in ggml_facts.ROCMFPX_TYPE_NAMES
    }
    assert fork_ids.isdisjoint(stock_ids)
    assert all(fid >= 100 for fid in fork_ids), (
        "fork ids are expected to sit past stock ggml's GGML_TYPE_COUNT "
        "(historically >= 100); a fork id below that suggests an upstream "
        "type table change needs review"
    )


def test_required_stock_names_covers_schemes_registry():
    """ggml_facts.REQUIRED_STOCK_NAMES cannot import magicquant.quant.schemes
    (would be circular -- see the comment on REQUIRED_STOCK_NAMES), so it is
    kept as a hand-maintained literal that must track schemes.py's registry
    by hand. This test CAN import both, so it is the cross-check that
    catches drift between them: every stock (non-fork) ggml_type_name any
    scheme in the registry actually uses must be covered by
    REQUIRED_STOCK_NAMES. Fork-only names (ROCmFPX, e.g. "Q8_0_ROCMFPX")
    are excluded -- they're never in the installed `gguf` package at all,
    so REQUIRED_STOCK_NAMES deliberately doesn't (and can't) include them.
    """
    scheme_type_names = {s.ggml_type_name for s in get_all_schemes()}
    stock_scheme_type_names = scheme_type_names - ggml_facts.ROCMFPX_TYPE_NAMES
    missing = stock_scheme_type_names - ggml_facts.REQUIRED_STOCK_NAMES
    assert not missing, (
        f"schemes.py registry uses stock ggml_type_name(s) {sorted(missing)} "
        f"not covered by ggml_facts.REQUIRED_STOCK_NAMES -- a new scheme was "
        f"added without updating that literal list; add the missing name(s)."
    )


def test_id_to_name_is_consistent_inverse():
    """ID_TO_NAME must be exactly the inverse of NAME_TO_ID (no stale ids
    left over from a previous derivation, no accidental id collisions
    silently dropping a name)."""
    assert len(ggml_facts.NAME_TO_ID) == len(ggml_facts.ID_TO_NAME), (
        "NAME_TO_ID and ID_TO_NAME have different sizes -- likely two "
        "different names mapping to the same id, silently dropping one "
        "from the reverse mapping"
    )
    for name, type_id in ggml_facts.NAME_TO_ID.items():
        assert ggml_facts.ID_TO_NAME[type_id] == name
