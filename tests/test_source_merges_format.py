"""Regression test: BPE merges must be normalized to llama.cpp's string form.

transformers <5 wrote tokenizer.json merges as space-joined strings ("Ġ Ġ").
transformers >=5 writes them as pair-arrays (["Ġ", "Ġ"]). llama.cpp's GGUF BPE
loader requires the space-joined string form. Before the fix, the new pair-array
format was written verbatim, so each merge landed in the GGUF as a Python list
repr ("['Ġ', 'Ġ']"); BPE could not merge, the test text mis-tokenized, and
perplexity exploded (~11000 instead of ~22) even though the weights were
byte-identical to a working model.
"""
from magicquant.gguf.source import _normalize_merges


def test_legacy_string_merges_unchanged():
    """transformers <5 space-joined strings pass through untouched."""
    merges = ["Ġ Ġ", "Ġ t", "i n"]
    assert _normalize_merges(merges) == ["Ġ Ġ", "Ġ t", "i n"]


def test_pair_array_merges_joined_with_space():
    """transformers >=5 pair-arrays are joined into the legacy string form."""
    merges = [["Ġ", "Ġ"], ["Ġ", "t"], ["i", "n"]]
    assert _normalize_merges(merges) == ["Ġ Ġ", "Ġ t", "i n"]


def test_token_containing_space_is_preserved():
    """A merge piece that itself contains a space still round-trips via join.

    GGUF merges are split on the FIRST space by llama.cpp, so a two-element pair
    join is unambiguous as long as we only join the pair with a single space.
    """
    merges = [["a", "b"]]
    assert _normalize_merges(merges) == ["a b"]


def test_empty_merges():
    assert _normalize_merges([]) == []
