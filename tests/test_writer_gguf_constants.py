"""Drift tripwire for magicquant.gguf.writer's GGUF metadata-value-type tags
and default alignment.

Same "facts come from upstream, never drift" policy as
tests/test_ggml_facts_snapshot.py (see that file's docstring for the class
of bug this guards against). Before this rework, writer.py hand-typed the
GGUF metadata value-type tag ints (``_GGUF_TYPE_UINT8`` .. ``_GGUF_TYPE_
FLOAT64``) and the default tensor-data ``ALIGNMENT``, duplicating facts the
installed ``gguf`` package already publishes as
``gguf.constants.GGUFValueType`` and ``gguf.constants.GGUF_DEFAULT_
ALIGNMENT``. writer.py now derives them from that package; this test pastes
a FROZEN SNAPSHOT of the old hand-typed literals and asserts the derived
values still match, byte-for-byte, so on-disk GGUF metadata encoding is
unaffected by the refactor.

Do NOT "fix" a failure here by just updating the snapshot to whatever the
installed `gguf` package currently produces -- a mismatch means either the
package changed a stable, long-published enum (extremely unlikely and worth
investigating) or the derivation in writer.py is wrong.
"""
from magicquant.gguf import writer


# (writer attribute name, historical hand-typed literal) -- copied verbatim
# from writer.py before this refactor.
_FROZEN_TYPE_TAGS = [
    ("_GGUF_TYPE_UINT8", 0),
    ("_GGUF_TYPE_INT8", 1),
    ("_GGUF_TYPE_UINT16", 2),
    ("_GGUF_TYPE_INT16", 3),
    ("_GGUF_TYPE_UINT32", 4),
    ("_GGUF_TYPE_INT32", 5),
    ("_GGUF_TYPE_FLOAT32", 6),
    ("_GGUF_TYPE_BOOL", 7),
    ("_GGUF_TYPE_STRING", 8),
    ("_GGUF_TYPE_ARRAY", 9),
    ("_GGUF_TYPE_UINT64", 10),
    ("_GGUF_TYPE_INT64", 11),
    ("_GGUF_TYPE_FLOAT64", 12),
]

_FROZEN_ALIGNMENT = 32


def test_gguf_value_type_tags_match_frozen_snapshot():
    for attr_name, expected in _FROZEN_TYPE_TAGS:
        actual = getattr(writer, attr_name)
        assert actual == expected, (
            f"writer.{attr_name} drifted from the historical hand-typed "
            f"value {expected} to {actual} -- the installed `gguf` "
            f"package's GGUFValueType enum no longer matches the on-disk "
            f"GGUF format's stable value-type tags. Investigate before "
            f"updating this snapshot; every existing GGUF file (and every "
            f"reader) depends on these ints never changing."
        )
        assert isinstance(actual, int)


def test_alignment_matches_frozen_snapshot():
    assert writer.ALIGNMENT == _FROZEN_ALIGNMENT, (
        f"writer.ALIGNMENT drifted from the historical hand-typed value "
        f"{_FROZEN_ALIGNMENT} to {writer.ALIGNMENT} -- the installed "
        f"`gguf` package's GGUF_DEFAULT_ALIGNMENT no longer matches the "
        f"long-standing GGUF default. This changes tensor-data padding "
        f"in every GGUF written; investigate before updating this "
        f"snapshot."
    )


def test_type_tags_derived_from_installed_gguf_package():
    """Confirms the values genuinely come from gguf.constants.GGUFValueType
    (not just coincidentally-matching hand-typed literals) -- protects
    against a future edit reverting to hardcoded ints without anyone
    noticing, since that would also happen to pass the snapshot test above.
    """
    import gguf.constants as gguf_constants

    value_type = gguf_constants.GGUFValueType
    for attr_name, _ in _FROZEN_TYPE_TAGS:
        member_name = attr_name.removeprefix("_GGUF_TYPE_")
        assert getattr(writer, attr_name) == int(value_type[member_name])


def test_alignment_derived_from_installed_gguf_package():
    import gguf.constants as gguf_constants

    assert writer.ALIGNMENT == gguf_constants.GGUF_DEFAULT_ALIGNMENT


# (writer._ftype_map key, historical hand-typed literal) -- copied verbatim
# from writer.py's general.file_type table before it was derived from
# gguf.constants.LlamaFileType by member name (see gguf-io audit ITEM 0/1).
_FROZEN_FTYPE_MAP = [
    ("F32", 0),
    ("F16", 1),
    ("BF16", 32),
    ("Q8_0", 7),
    ("Q6_K", 18),
    ("Q5_K", 17),
    ("Q5_K_M", 17),
    ("Q5_K_S", 16),
    ("Q4_K", 15),
    ("Q4_K_M", 15),
    ("Q4_K_S", 14),
    ("Q3_K", 12),
    ("Q3_K_M", 12),
    ("Q2_K", 10),
    ("IQ4_NL", 25),
    ("IQ4_XS", 30),
]


def test_ftype_map_matches_frozen_snapshot():
    assert set(writer._ftype_map.keys()) == {k for k, _ in _FROZEN_FTYPE_MAP}, (
        "writer._ftype_map's key set changed -- this table is a deliberate "
        "union of ggml type names and legacy scheme-name aliases, used as "
        "BOTH a membership filter (dominant-scheme selection) and a "
        "second, scheme-keyed lookup (base_quant fallback). Do not add or "
        "remove keys without re-reading create_hybrid_gguf's "
        "_build_metadata; see that method's docstring."
    )
    for key, expected in _FROZEN_FTYPE_MAP:
        actual = writer._ftype_map[key]
        assert actual == expected, (
            f"writer._ftype_map[{key!r}] drifted from the historical "
            f"hand-typed value {expected} to {actual} -- the installed "
            f"`gguf` package's LlamaFileType enum no longer matches the "
            f"values MagicQuant has always written for general.file_type. "
            f"Investigate before updating this snapshot; a hand-typed "
            f"predecessor of this exact table already drifted once "
            f"(Q4_K->12, Q5_K->16, IQ4_NL->20 were wrong)."
        )
        assert isinstance(actual, int)


def test_ftype_map_derived_from_installed_gguf_package():
    """Confirms writer._ftype_map agrees with the installed package's
    LlamaFileType enum, member by member. Together with the frozen snapshot
    above this distinguishes "upstream renumbered the enum" (this test fails)
    from "our map drifted" (snapshot fails). Honest limitation: a revert to
    hand-typed literals that happen to match today's enum values would pass
    both tests -- what they guard is agreement with upstream, not the
    derivation mechanism itself.
    """
    import gguf.constants as gguf_constants

    _member_by_key = {
        "F32": "ALL_F32",
        "F16": "MOSTLY_F16",
        "BF16": "MOSTLY_BF16",
        "Q8_0": "MOSTLY_Q8_0",
        "Q6_K": "MOSTLY_Q6_K",
        "Q5_K": "MOSTLY_Q5_K_M",
        "Q5_K_M": "MOSTLY_Q5_K_M",
        "Q5_K_S": "MOSTLY_Q5_K_S",
        "Q4_K": "MOSTLY_Q4_K_M",
        "Q4_K_M": "MOSTLY_Q4_K_M",
        "Q4_K_S": "MOSTLY_Q4_K_S",
        "Q3_K": "MOSTLY_Q3_K_M",
        "Q3_K_M": "MOSTLY_Q3_K_M",
        "Q2_K": "MOSTLY_Q2_K",
        "IQ4_NL": "MOSTLY_IQ4_NL",
        "IQ4_XS": "MOSTLY_IQ4_XS",
    }
    file_type = gguf_constants.LlamaFileType
    for key, member_name in _member_by_key.items():
        assert writer._ftype_map[key] == int(file_type[member_name])
