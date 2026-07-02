"""Unigram (SPM-style) tokenizer vocab must not crash the metadata reader.

Unigram tokenizer.json stores vocab as a LIST of [token, score] pairs (id =
index), not a {token: id} dict. The reader used to call .items() on it and
raise AttributeError, taking down any Unigram model.
"""
import json

from magicquant.gguf.source import _build_tokenizer_metadata


def test_unigram_list_vocab_parses(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "model": {
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["a", -1.5], ["b", -2.25], ["c", -3.0]],
        },
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
    meta = _build_tokenizer_metadata(str(tmp_path))
    assert meta["tokenizer.ggml.model"] == "llama"
    toks = meta["tokenizer.ggml.tokens"]
    assert toks[:4] == ["<unk>", "a", "b", "c"]
    # Unigram scores are captured, not left at 0.0.
    scores = meta["tokenizer.ggml.scores"]
    assert scores[1] == -1.5 and scores[2] == -2.25


def test_bpe_dict_vocab_still_works(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []},
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
    meta = _build_tokenizer_metadata(str(tmp_path))
    assert meta["tokenizer.ggml.model"] == "gpt2"
    assert meta["tokenizer.ggml.tokens"][:3] == ["a", "b", "c"]
