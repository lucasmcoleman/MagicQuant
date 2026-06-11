"""Perplexity comparison hook: QAT hybrid vs plain hybrid.

``compare_perplexity`` runs ``llama-perplexity`` on two GGUFs and returns their
PPLs + the delta (plain - qat; positive = QAT improved). The test stubs the
``perplexity_bin`` with a fake script that echoes a known ``Final estimate``
line, so it runs offline without llama.cpp or real models.
"""

import os
import stat

import pytest

from magicquant.qat.validate import compare_perplexity, parse_perplexity


def _make_fake_perplexity_bin(tmp_path, ppl_by_model):
    """Write a fake perplexity bin that prints a PPL keyed on the -m argument.

    ``ppl_by_model`` maps a substring of the model path to the PPL to emit.
    """
    script = tmp_path / "fake_perplexity.sh"
    lines = ["#!/usr/bin/env bash", 'model=""', "while [ $# -gt 0 ]; do",
             '  if [ "$1" = "-m" ]; then shift; model="$1"; fi', "  shift", "done"]
    for key, ppl in ppl_by_model.items():
        lines.append(f'if [[ "$model" == *{key}* ]]; then '
                     f'echo "Final estimate: PPL = {ppl} +/- 0.04200"; exit 0; fi')
    lines.append('echo "Final estimate: PPL = 99.0 +/- 0.1"; exit 0')
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def test_parse_perplexity_reads_final_estimate():
    out = "load stuff\n[1]5.1 [2]5.2\nFinal estimate: PPL = 12.3456 +/- 0.06789\n"
    assert parse_perplexity(out) == pytest.approx(12.3456)


def test_parse_perplexity_uses_last_match():
    out = "Final estimate: PPL = 9.99 +/- 0.1\nFinal estimate: PPL = 7.77 +/- 0.1\n"
    assert parse_perplexity(out) == pytest.approx(7.77)


def test_parse_perplexity_raises_when_absent():
    with pytest.raises(RuntimeError):
        parse_perplexity("no perplexity here\n")


def test_compare_perplexity_returns_plain_qat_delta(tmp_path):
    plain = tmp_path / "model-plain.gguf"
    qat = tmp_path / "model-qat.gguf"
    plain.write_text("x")
    qat.write_text("x")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")

    fake_bin = _make_fake_perplexity_bin(
        tmp_path, {"plain": "8.5000", "qat": "8.1000"}
    )
    result = compare_perplexity(str(plain), str(qat), str(corpus), fake_bin)
    assert result["plain"] == pytest.approx(8.5)
    assert result["qat"] == pytest.approx(8.1)
    # delta = plain - qat; positive means QAT lowered perplexity (improved)
    assert result["delta"] == pytest.approx(0.4)


def test_compare_perplexity_negative_delta_when_qat_worse(tmp_path):
    plain = tmp_path / "plain.gguf"
    qat = tmp_path / "qat.gguf"
    plain.write_text("x")
    qat.write_text("x")
    corpus = tmp_path / "c.txt"
    corpus.write_text("data\n")
    fake_bin = _make_fake_perplexity_bin(
        tmp_path, {"plain": "7.0", "qat": "7.5"}
    )
    result = compare_perplexity(str(plain), str(qat), str(corpus), fake_bin)
    assert result["delta"] == pytest.approx(-0.5)


def test_compare_perplexity_raises_on_bad_exit(tmp_path):
    plain = tmp_path / "p.gguf"
    qat = tmp_path / "q.gguf"
    plain.write_text("x")
    qat.write_text("x")
    corpus = tmp_path / "c.txt"
    corpus.write_text("data\n")
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/usr/bin/env bash\necho 'boom' >&2\nexit 3\n")
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(RuntimeError):
        compare_perplexity(str(plain), str(qat), str(corpus), str(bad))
