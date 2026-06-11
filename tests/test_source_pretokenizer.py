"""Regression test: GGUF must carry tokenizer.ggml.pre (pre-tokenizer type).

llama.cpp selects its pre-tokenizer regex from the ``tokenizer.ggml.pre`` GGUF
key. When the key is absent it falls back to 'default' and prints
"GENERATION QUALITY WILL BE DEGRADED!" — the text is then split with the wrong
regex, tokenizes differently from the reference, and perplexity inflates badly
(measured: Qwen2.5-0.5B base went 15.1 -> 21.9, a fine-tuned merge 12.6 -> 32.7).

MagicQuant reads tokenizer.json directly, so it identifies the pre-tokenizer by
the pre_tokenizer Split regex (copied verbatim across a model family) and maps it
to llama.cpp's canonical ``pre`` name.
"""
from magicquant.gguf.source import _detect_tokenizer_pre

QWEN2_RE = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
LLAMA3_RE = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
GPT2_RE = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def _seq(regex):
    """Build a tokenizer.json-shaped dict with a Sequence[Split, ByteLevel]."""
    return {
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated"},
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        }
    }


def test_qwen2_detected():
    assert _detect_tokenizer_pre(_seq(QWEN2_RE)) == "qwen2"


def test_llama3_detected():
    assert _detect_tokenizer_pre(_seq(LLAMA3_RE)) == "llama-bpe"


def test_gpt2_detected():
    # GPT-2 pre_tokenizer is often a bare Split (not wrapped in a Sequence).
    tok = {"pre_tokenizer": {"type": "Split", "pattern": {"Regex": GPT2_RE},
                             "behavior": "Isolated"}}
    assert _detect_tokenizer_pre(tok) == "gpt-2"


def test_unknown_regex_returns_none():
    assert _detect_tokenizer_pre(_seq(r"some unknown pattern")) is None


def test_missing_pretokenizer_returns_none():
    assert _detect_tokenizer_pre({}) is None
    assert _detect_tokenizer_pre({"pre_tokenizer": None}) is None
