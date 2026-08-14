"""``_run_captured(timeout=None)`` must mean "no deadline", as in subprocess.run.

The defect (2026-08-14). ``_run_captured`` was introduced to bound the
leaked-descriptor stall, and its docstring promises "contract is deliberately
identical to ``subprocess.run`` so callers and their tests do not change". It
was not identical: ``subprocess.run`` accepts ``timeout=None`` and waits
forever, while ``_run_captured`` computed ``time.monotonic() + timeout``
unconditionally and raised

    TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'

*before spawning anything*. ``capture_imatrix``'s ``timeout`` parameter is
declared ``Optional[float] = None``, so the documented default was the broken
path -- every imatrix capture that did not pass an explicit timeout.

Why it went unnoticed long enough to reach an artifact: ``ensure_imatrix``
wrapped the call in a bare ``except Exception`` and returned None, so a
TypeError in our own code was reported as an environmental "capture failed;
continuing without an imatrix". A Qwen3.8-27B campaign submitted with
``use_imatrix: true`` therefore quantized UNWEIGHTED, and the only evidence
was one warning line in a stage log.

Both halves are pinned here: the None default must work, and a programming
error must not be laundered into a graceful degrade.
"""

import subprocess
import sys

import pytest

from magicquant.utils import llamacpp


# ── the direct defect ───────────────────────────────────────────────────────

def test_timeout_none_runs_to_completion():
    """The exact call that raised TypeError before spawning."""
    proc = llamacpp._run_captured([sys.executable, "-c", "print('ok')"], timeout=None)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "ok"


def test_timeout_is_optional_at_the_call_site():
    """capture_imatrix relies on the DEFAULT, so the default must be usable
    without being named -- a signature requiring `timeout` positionally is how
    this broke."""
    proc = llamacpp._run_captured([sys.executable, "-c", "print('ok')"])
    assert proc.returncode == 0


def test_none_does_not_disable_the_error_contract():
    """No deadline must not mean no checking: the rest of the documented
    contract still holds."""
    with pytest.raises(subprocess.CalledProcessError):
        llamacpp._run_captured(
            [sys.executable, "-c", "import sys; sys.exit(3)"], timeout=None, check=True
        )
    proc = llamacpp._run_captured(
        [sys.executable, "-c", "import sys; sys.exit(3)"], timeout=None, check=False
    )
    assert proc.returncode == 3


def test_a_real_timeout_still_fires():
    """The None path must not have disarmed the deadline for everyone else."""
    with pytest.raises(subprocess.TimeoutExpired):
        llamacpp._run_captured(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1
        )


def test_stderr_is_captured_with_no_deadline():
    proc = llamacpp._run_captured(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom')"], timeout=None
    )
    assert "boom" in proc.stderr


# ── the fail-open that hid it ───────────────────────────────────────────────

def test_ensure_imatrix_does_not_swallow_programming_errors(tmp_path, monkeypatch):
    """A TypeError from our own code must propagate, not be logged as a
    capture failure and degraded to unweighted quantization."""
    from magicquant import imatrix as im

    model = tmp_path / "model-bf16.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    corpus = tmp_path / "calib.txt"
    corpus.write_text("hello world\n")

    def _boom(*a, **k):
        raise TypeError("unsupported operand type(s) for +: 'float' and 'NoneType'")

    monkeypatch.setattr(im, "capture_imatrix", _boom)

    with pytest.raises(TypeError):
        im.ensure_imatrix(model, corpus_path=corpus, cache_dir=tmp_path / "_im")


def test_ensure_imatrix_still_degrades_on_environmental_failure(tmp_path, monkeypatch):
    """The doctrine is unchanged: imatrix is optional, and a genuine capture
    failure (llama-imatrix missing, non-zero exit, timeout) still returns None
    rather than killing the run. Narrowing the catch must not have turned
    every environmental hiccup into a hard stop."""
    from magicquant import imatrix as im

    model = tmp_path / "model-bf16.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    corpus = tmp_path / "calib.txt"
    corpus.write_text("hello world\n")

    def _fail(*a, **k):
        raise RuntimeError("llama-imatrix failed (rc=1)")

    monkeypatch.setattr(im, "capture_imatrix", _fail)

    assert im.ensure_imatrix(
        model, corpus_path=corpus, cache_dir=tmp_path / "_im"
    ) is None
