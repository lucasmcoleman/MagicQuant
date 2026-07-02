"""llama-perplexity output parsing must read the STDERR stream.

Regression guard for the audit's rank-1 bug: llama-perplexity prints its
"Final estimate: PPL = ..." line to stderr, but both production parsers
(LlamaCppTools.calculate_perplexity and qat/validate.parse_perplexity) read
only stdout — silently collapsing the measured search + QAT validation. These
tests feed stderr-only output through both paths.
"""
import subprocess
from unittest import mock

import pytest

from magicquant.utils.llamacpp import LlamaCppTools, _parse_perplexity_output
from magicquant.qat.validate import parse_perplexity


_PPL_LINE = "Final estimate: PPL = 12.3456 +/- 0.06789"


def test_low_level_parser_finds_ppl_in_combined_output():
    combined = "some stdout noise\n" + _PPL_LINE
    assert _parse_perplexity_output(combined) == pytest.approx(12.3456)


def test_qat_parser_finds_ppl():
    assert parse_perplexity(f"loading...\n{_PPL_LINE}\n") == pytest.approx(12.3456)


def test_calculate_perplexity_reads_stderr(tmp_path, monkeypatch):
    """The PPL line on STDERR (empty stdout) must still be parsed."""
    model = tmp_path / "m.gguf"; model.write_bytes(b"gguf")
    corpus = tmp_path / "c.txt"; corpus.write_text("hello world\n")

    tools = LlamaCppTools.__new__(LlamaCppTools)  # bypass __init__/discovery
    tools.perplexity_tool = "/bin/true"
    tools.ctx_size = 512
    tools._resolve_data_file = lambda data_file: str(corpus)

    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=f"llama_print_timings...\n{_PPL_LINE}\n"
    )
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake):
        ppl = tools.calculate_perplexity(str(model), verbose=False)
    assert ppl == pytest.approx(12.3456)


def test_calculate_perplexity_none_when_absent(tmp_path):
    model = tmp_path / "m.gguf"; model.write_bytes(b"gguf")
    corpus = tmp_path / "c.txt"; corpus.write_text("hi\n")
    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.perplexity_tool = "/bin/true"
    tools.ctx_size = 512
    tools._resolve_data_file = lambda data_file: str(corpus)
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="no ppl here", stderr="nor here")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake):
        assert tools.calculate_perplexity(str(model), verbose=False) is None
