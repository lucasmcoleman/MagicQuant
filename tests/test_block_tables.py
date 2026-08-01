"""Block-size / type-size table consistency tests.

The converters' GGML_BLOCK_SIZE / GGML_TYPE_SIZE tables must agree with the
authoritative tables in ggml_binding (the single source of truth). Locks the
IQ4_XS fix: converters previously had block=32/size=18 (IQ4_NL's values) which
disagreed with ggml_binding's correct block=256/size=136, a corruption trap.
"""
from magicquant.quant import converters
from magicquant.quant import ggml_binding
from magicquant.quant import ggml_facts


def test_iq4_xs_matches_binding():
    assert converters.GGML_BLOCK_SIZE["IQ4_XS"] == 256
    assert converters.GGML_TYPE_SIZE["IQ4_XS"] == 136
    assert ggml_binding._GGML_BLOCK_SIZE["IQ4_XS"] == 256
    assert ggml_binding._GGML_TYPE_SIZE["IQ4_XS"] == 136


def test_converters_tables_consistent_with_binding():
    """NOTE ON WHAT THIS TEST USED TO BE: it originally cross-checked
    converters.GGML_BLOCK_SIZE/GGML_TYPE_SIZE against
    ggml_binding._GGML_BLOCK_SIZE/_GGML_TYPE_SIZE directly, catching real
    drift back when each module hand-maintained its own copy of these tables
    (that's the IQ4_XS incident referenced in the module docstring: the two
    copies disagreed).

    Since the ggml_facts rework, BOTH tables are literally
    ``dict(ggml_facts.BLOCK_SIZE)`` / ``dict(ggml_facts.TYPE_SIZE)`` with no
    per-module overlay left on top (see converters.py's and
    ggml_binding.py's module-level assignments) -- comparing them to each
    other had become tautological: two copies of the same source dict are
    guaranteed equal by construction, regardless of whether either module
    still actually does the copy correctly. That guarantee is now
    STRUCTURAL (enforced by both modules' source reading `dict(ggml_facts.*)`
    rather than a hand table), not something a runtime assertion needs to
    re-verify between the two.

    What still needs an actual runtime check, because a future edit could
    silently break it without either side's simple `dict(ggml_facts.X)` line
    changing shape:
      (a) each consumer's table still equals ggml_facts' own canonical table
          (catches a consumer reintroducing a hand-overlay or hand-copy that
          silently diverges from ggml_facts again);
      (b) the ROCmFPX fork ids/names ggml_facts defines are actually present,
          with matching values, in BOTH consumers (catches a consumer that
          starts filtering/stripping fork entries out of its copy, e.g. an
          accidental `if name not in ROCMFPX_TYPE_NAMES` guard added to one
          side only).
    """
    # (a) Each consumer's table equals ggml_facts' own canonical table.
    assert converters.GGML_BLOCK_SIZE == ggml_facts.BLOCK_SIZE
    assert converters.GGML_TYPE_SIZE == ggml_facts.TYPE_SIZE
    assert ggml_binding._GGML_BLOCK_SIZE == ggml_facts.BLOCK_SIZE
    assert ggml_binding._GGML_TYPE_SIZE == ggml_facts.TYPE_SIZE

    # (b) Fork ids/names present, with matching values, in both consumers.
    for name, info in ggml_facts.FORK_TYPES.items():
        assert ggml_facts.NAME_TO_ID[name] == info["id"]
        for label, block_table, size_table in (
            ("converters", converters.GGML_BLOCK_SIZE, converters.GGML_TYPE_SIZE),
            ("ggml_binding", ggml_binding._GGML_BLOCK_SIZE, ggml_binding._GGML_TYPE_SIZE),
        ):
            assert name in block_table, f"{label}: missing fork type {name} from block-size table"
            assert block_table[name] == info["block"], (
                f"{label}: fork type {name} block_size {block_table[name]} "
                f"!= ggml_facts {info['block']}"
            )
            assert name in size_table, f"{label}: missing fork type {name} from type-size table"
            assert size_table[name] == info["size"], (
                f"{label}: fork type {name} type_size {size_table[name]} "
                f"!= ggml_facts {info['size']}"
            )


def test_iq4_xs_data_size_correct():
    """ggml_tensor_data_size for IQ4_XS uses block=256/size=136."""
    # 256 elements = 1 block = 136 bytes
    assert converters.ggml_tensor_data_size("IQ4_XS", 256) == 136
    # 512 elements = 2 blocks = 272 bytes
    assert converters.ggml_tensor_data_size("IQ4_XS", 512) == 272
