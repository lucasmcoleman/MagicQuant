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


# ---------------------------------------------------------------------------
# Regression: parser must never invent a number (the 2026-07 measured-search
# incident). A NaN model exits 0 and never prints "Final estimate: PPL =",
# but DOES print a per-chunk progress line ("perplexity: 2.74 seconds per
# pass - ETA ...") that the old second/third fallback patterns mistook for a
# real perplexity reading -- returning 2.74 instead of None. This drove 8/9
# group sensitivities to exactly 0.0 in the real incident run (baseline
# 34.8363, bogus probes 2.6-2.8).
# ---------------------------------------------------------------------------

_NAN_RUN_OUTPUT = """\
perplexity: tokenizing the input ..
perplexity: tokenization took 123.45 ms
perplexity: calculating perplexity over 10 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 2.74 seconds per pass - ETA 4.5 minutes
[1]nan,[2]nan,[3]nan,
Unexpected negative standard deviation of log(prob)

llama_perf_context_print:        load time =     500.00 ms
llama_perf_context_print: prompt eval time =   12345.00 ms
"""


def test_parser_returns_none_on_nan_run_not_the_progress_line_number():
    """Decisive regression: today's parser returns 2.74 (the progress
    line's seconds-per-pass) for this real-shaped NaN-run output; it must
    return None."""
    result = _parse_perplexity_output(_NAN_RUN_OUTPUT)
    assert result is None, (
        f"parser invented a PPL ({result!r}) from a NaN run's progress "
        "line instead of returning None"
    )


def test_parser_rejects_progress_line_alone_without_failure_marker():
    """Even without the explicit NaN-stddev marker, a bare progress line
    must never be mistaken for 'Final estimate: PPL ='."""
    output = "perplexity: 2.74 seconds per pass - ETA 4.5 minutes\n"
    assert _parse_perplexity_output(output) is None


def test_parser_rejects_bare_perplexity_colon_fallback():
    """The old '[Pp]erplexity[:=]' fallback is fully removed."""
    assert _parse_perplexity_output("perplexity: 9.99\n") is None
    assert _parse_perplexity_output("Perplexity = 9.99\n") is None


def test_parser_rejects_any_line_containing_ppl_fallback():
    """The old 'any line containing PPL' catch-all is fully removed."""
    assert _parse_perplexity_output("some noise PPL 9.99 more noise\n") is None


def test_parser_accepts_final_estimate():
    assert _parse_perplexity_output(_PPL_LINE) == pytest.approx(12.3456)


def test_parser_accepts_mean_ppl_q_kl_form():
    line = "Mean PPL(Q)                   :  13.821636 \xb1   3.046334"
    assert _parse_perplexity_output(line) == pytest.approx(13.821636)


def test_parser_none_for_final_estimate_nan():
    assert _parse_perplexity_output("Final estimate: PPL = nan +/- nan\n") is None


def test_parser_none_for_final_estimate_inf():
    assert _parse_perplexity_output("Final estimate: PPL = inf +/- inf\n") is None


def test_parser_none_on_failed_to_decode_marker_even_with_real_looking_line():
    output = "Final estimate: PPL = 5.0000 +/- 0.01\nfailed to decode batch\n"
    # The failure marker must win even though a well-formed PPL line is
    # ALSO present -- "failed to decode" means the run itself is suspect.
    assert _parse_perplexity_output(output) is None


def test_parser_logs_last_lines_at_warning_when_none(caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="magicquant.utils.llamacpp")
    noisy_output = "\n".join(f"noise line {i}" for i in range(30))
    result = _parse_perplexity_output(noisy_output)
    assert result is None
    assert any(
        "noise line 29" in rec.message or "noise line 29" in str(rec.args)
        for rec in caplog.records
    ), "expected the tail of the discarded output to be logged at WARNING"
