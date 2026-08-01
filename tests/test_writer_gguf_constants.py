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
