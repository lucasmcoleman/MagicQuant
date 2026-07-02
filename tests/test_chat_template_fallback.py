"""Chat-template fallback to standalone chat_template.jinja/.json.

transformers >= 4.44 stores the chat template in a standalone file rather than
tokenizer_config.json; without a fallback the GGUF ships with no
tokenizer.chat_template (the known "GGUF needs chat-template patching" issue).
"""
import json

from magicquant.gguf.source import _build_tokenizer_metadata


def _min_tokenizer(dirpath):
    # A minimal tokenizer.json so the reader proceeds to the metadata step.
    (dirpath / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
    }))


def test_reads_standalone_jinja(tmp_path):
    _min_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))  # no chat_template
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}TEMPLATE_JINJA")
    meta = _build_tokenizer_metadata(str(tmp_path))
    assert meta.get("tokenizer.chat_template") == "{{ messages }}TEMPLATE_JINJA"


def test_reads_standalone_json(tmp_path):
    _min_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "TEMPLATE_JSON"}))
    meta = _build_tokenizer_metadata(str(tmp_path))
    assert meta.get("tokenizer.chat_template") == "TEMPLATE_JSON"


def test_tokenizer_config_takes_precedence(tmp_path):
    _min_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "FROM_CONFIG"}))
    (tmp_path / "chat_template.jinja").write_text("FROM_FILE")
    meta = _build_tokenizer_metadata(str(tmp_path))
    assert meta.get("tokenizer.chat_template") == "FROM_CONFIG"


def test_no_template_anywhere(tmp_path):
    _min_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
    meta = _build_tokenizer_metadata(str(tmp_path))
    assert "tokenizer.chat_template" not in meta
