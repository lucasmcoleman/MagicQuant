"""Perplexity comparison hook: QAT hybrid vs plain hybrid.

``compare_perplexity`` runs ``llama-perplexity`` on two GGUFs and returns their
PPLs + the delta (plain - qat; positive = QAT improved). The test stubs the
``perplexity_bin`` with a fake script that echoes a known ``Final estimate``
line, so it runs offline without llama.cpp or real models.
"""

import os
import stat
import sys

import pytest

from magicquant.qat.validate import compare_perplexity, parse_perplexity


def _write_fake_bin(tmp_path, name, body):
    """An executable that runs *body* (Python source) with the real argv.

    The body is Python so the fake runs identically everywhere; the thin shim
    is what makes it directly executable by ``subprocess`` without a shell:
    ``#!/bin/sh`` + ``exec`` on POSIX, a ``.cmd`` file on Windows (where a
    shell script is not a valid executable -- WinError 193 -- and neither is
    a bare ``.py``).
    """
    script = tmp_path / f"{name}.py"
    script.write_text(body, encoding="utf-8")
    if os.name == "nt":
        shim = tmp_path / f"{name}.cmd"
        shim.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return str(shim)
    shim = tmp_path / name
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(shim)


def _make_fake_perplexity_bin(tmp_path, ppl_by_model):
    """Write a fake perplexity bin that prints a PPL keyed on the -m argument.

    ``ppl_by_model`` maps a substring of the model path to the PPL to emit.
    """
    body = (
        "import sys\n"
        "args = sys.argv[1:]\n"
        "model = ''\n"
        "for i, a in enumerate(args):\n"
        "    if a == '-m' and i + 1 < len(args):\n"
        "        model = args[i + 1]\n"
        f"for key, ppl in {dict(ppl_by_model)!r}.items():\n"
        "    if key in model:\n"
        "        print(f'Final estimate: PPL = {ppl} +/- 0.04200')\n"
        "        sys.exit(0)\n"
        "print('Final estimate: PPL = 99.0 +/- 0.1')\n"
    )
    return _write_fake_bin(tmp_path, "fake_perplexity", body)


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
    bad = _write_fake_bin(
        tmp_path, "bad", "import sys; print('boom', file=sys.stderr); sys.exit(3)\n"
    )
    with pytest.raises(RuntimeError):
        compare_perplexity(str(plain), str(qat), str(corpus), bad)
