"""Tests for magicquant.imatrix.ensure_imatrix — the capture/cache/load helper.

All tests monkeypatch capture_imatrix / load_imatrix so no real llama-imatrix
binary or corpus processing is needed; they exercise ensure_imatrix's own
control flow (GGUF-only gate, cache key/hit/miss, failure handling) plus the
presence of the bundled default corpus.
"""
from pathlib import Path

import numpy as np

from magicquant import imatrix


def test_ensure_imatrix_non_gguf_file_returns_none_without_capture(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("capture_imatrix must not be called for a non-GGUF source")

    monkeypatch.setattr(imatrix, "capture_imatrix", fail_if_called)

    source = tmp_path / "model.safetensors"
    source.write_bytes(b"not really safetensors")

    result = imatrix.ensure_imatrix(source)
    assert result is None


def test_ensure_imatrix_safetensors_dir_returns_none_without_capture(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("capture_imatrix must not be called for a directory source")

    monkeypatch.setattr(imatrix, "capture_imatrix", fail_if_called)

    source_dir = tmp_path / "model_dir"
    source_dir.mkdir()
    (source_dir / "model-00001-of-00001.safetensors").write_bytes(b"stub")

    result = imatrix.ensure_imatrix(source_dir)
    assert result is None


def test_ensure_imatrix_cache_hit_skips_capture(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"fake gguf bytes")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    corpus = imatrix.DEFAULT_CORPUS_PATH
    # Track the module default rather than a literal: `chunks` is part of the
    # cache key, so hardcoding -1 here made this test a cache MISS the moment
    # ensure_imatrix stopped defaulting to whole-corpus capture.
    key = imatrix._imatrix_cache_key(
        source, corpus, ctx_size=512, chunks=imatrix.DEFAULT_CAPTURE_CHUNKS
    )
    cache_path = cache_dir / f"{key}.imatrix.gguf"
    cache_path.write_bytes(b"pre-existing cached imatrix gguf")

    sentinel = {"blk.0.attn_q.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("capture_imatrix must not be called on a cache hit")

    def fake_load(path):
        assert Path(path) == cache_path
        return sentinel

    monkeypatch.setattr(imatrix, "capture_imatrix", fail_if_called)
    monkeypatch.setattr(imatrix, "load_imatrix", fake_load)

    result = imatrix.ensure_imatrix(source, cache_dir=cache_dir)
    assert result is sentinel


def test_ensure_imatrix_cache_miss_then_hit(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"fake gguf bytes")
    cache_dir = tmp_path / "cache2"

    capture_calls = []

    def fake_capture(model_path, corpus_path, output_path, **kwargs):
        capture_calls.append(Path(output_path))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"freshly captured imatrix gguf")
        return out

    sentinel = {"blk.0.ffn_down.weight": np.array([4.0], dtype=np.float32)}

    monkeypatch.setattr(imatrix, "capture_imatrix", fake_capture)
    monkeypatch.setattr(imatrix, "load_imatrix", lambda path: sentinel)

    result1 = imatrix.ensure_imatrix(source, cache_dir=cache_dir)
    assert result1 is sentinel
    assert len(capture_calls) == 1

    # Second call: cache file now exists, so capture must not run again.
    result2 = imatrix.ensure_imatrix(source, cache_dir=cache_dir)
    assert result2 is sentinel
    assert len(capture_calls) == 1


def test_ensure_imatrix_capture_failure_returns_none(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"fake gguf bytes")

    def failing_capture(*args, **kwargs):
        raise RuntimeError("llama-imatrix failed (rc=1):\nsome stderr")

    monkeypatch.setattr(imatrix, "capture_imatrix", failing_capture)

    result = imatrix.ensure_imatrix(source, cache_dir=tmp_path / "cache3")
    assert result is None


def test_ensure_imatrix_missing_source_returns_none(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("capture_imatrix must not be called for a missing source")

    monkeypatch.setattr(imatrix, "capture_imatrix", fail_if_called)

    missing = tmp_path / "does_not_exist.gguf"
    result = imatrix.ensure_imatrix(missing)
    assert result is None


def test_default_corpus_bundled_and_nontrivial():
    assert imatrix.DEFAULT_CORPUS_PATH.exists()
    assert imatrix.DEFAULT_CORPUS_PATH.is_file()
    size = imatrix.DEFAULT_CORPUS_PATH.stat().st_size
    assert size > 1024, f"expected bundled corpus > 1KB, got {size} bytes"
    text = imatrix.DEFAULT_CORPUS_PATH.read_text(encoding="utf-8")
    assert len(text.split()) > 100


def test_default_corpus_meets_llama_imatrix_minimum():
    """llama-imatrix refuses to run with fewer than 2*ctx_size tokens (1024 at
    the default ctx 512). The original 4.7KB bundled corpus tokenized to only
    926 Qwen tokens, so default-settings capture ALWAYS failed (found in a
    real validation run 2026-07-03). English prose tokenizes at roughly 1.3
    tokens/word across common BPE vocabularies, so require enough words for
    ~2x the minimum as a tokenizer-independent guard against the corpus
    shrinking below usability again.
    """
    text = imatrix.DEFAULT_CORPUS_PATH.read_text(encoding="utf-8")
    words = len(text.split())
    assert words >= 1600, (
        f"bundled corpus has {words} words (~{int(words * 1.3)} tokens); "
        f"llama-imatrix needs >=1024 tokens at default ctx 512, and we want "
        f"~2x headroom for tokenizer variation"
    )
