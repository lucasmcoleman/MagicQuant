"""The bundled calibration corpus must stay fit for purpose.

It replaced a 13 KB English-only file that produced ~5 chunks at ctx 512 --
too thin to estimate importance for a 27B's ~866 tensors. These pin the
properties that made it inadequate so a future shrink is caught, and pin the
one property that would silently invalidate a whole search: overlap with the
perplexity eval corpus.
"""
from pathlib import Path

import pytest

from magicquant.imatrix import DEFAULT_CORPUS_PATH, DEFAULT_CAPTURE_CHUNKS

EVAL_CORPUS = Path("/server/ai/wikitext/wikitext-2-raw/wiki.test.raw")


@pytest.fixture(scope="module")
def corpus() -> str:
    assert DEFAULT_CORPUS_PATH.exists(), f"missing {DEFAULT_CORPUS_PATH}"
    return DEFAULT_CORPUS_PATH.read_text(encoding="utf-8")


def test_corpus_large_enough_for_the_default_chunk_count(corpus):
    """Must hold at least DEFAULT_CAPTURE_CHUNKS worth of text, or the capture
    silently calibrates on less than it claims."""
    approx_tokens = len(corpus) / 3.0          # conservative multilingual ratio
    approx_chunks = approx_tokens / 512
    assert approx_chunks >= DEFAULT_CAPTURE_CHUNKS, (
        f"corpus yields ~{approx_chunks:.0f} chunks but the default capture "
        f"asks for {DEFAULT_CAPTURE_CHUNKS}"
    )


def test_corpus_is_multilingual(corpus):
    """A 248k-token vocab calibrated on Latin script alone leaves most
    embedding/head rows weighted at ~zero."""
    ranges = {
        "CJK": ("一", "鿿"), "Arabic": ("؀", "ۿ"),
        "Cyrillic": ("Ѐ", "ӿ"), "Devanagari": ("ऀ", "ॿ"),
    }
    present = {
        name for name, (lo, hi) in ranges.items()
        if any(lo <= ch <= hi for ch in corpus)
    }
    assert len(present) >= 3, f"only found scripts: {present or '{none}'}"


def test_corpus_covers_code_and_math(corpus):
    import re
    assert len(re.findall(r"\b(def |function |class |import |#include)", corpus)) > 50
    assert len(re.findall(r"(\\frac|\\sum|\$|\b\d+\s*[+\-*/=]\s*\d+)", corpus)) > 50


@pytest.mark.skipif(not EVAL_CORPUS.exists(), reason="eval corpus not present")
def test_corpus_is_disjoint_from_the_perplexity_eval_corpus(corpus):
    """THE important one. Calibrating on the text a run is scored against
    makes every measured_loss optimistic, and nothing in the output would
    show it."""
    ev = EVAL_CORPUS.read_text(encoding="utf-8", errors="ignore").split()
    ev_grams = {tuple(ev[i:i + 8]) for i in range(0, len(ev) - 8, 3)}
    cal = corpus.split()
    probes = [tuple(cal[i:i + 8]) for i in range(0, len(cal) - 8, 50)]
    hits = sum(1 for g in probes if g in ev_grams)
    assert hits / max(1, len(probes)) < 0.001, (
        f"{hits}/{len(probes)} sampled 8-grams also appear in the eval corpus"
    )
