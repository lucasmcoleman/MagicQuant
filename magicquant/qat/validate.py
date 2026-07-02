"""Validation hook: compare perplexity of the QAT hybrid vs the plain hybrid.

The QAT success metric (per the design spec) is a *lower* perplexity loss for the
quant-aware hybrid than the plain one. ``compare_perplexity`` builds nothing — it
takes two already-packed GGUFs (one from a plain hybrid pack, one from the
QAT-adapted pack), runs ``llama-perplexity`` on each over the same corpus, and
returns ``{"plain", "qat", "delta"}`` where ``delta = plain - qat`` (positive means
QAT lowered perplexity, i.e. improved quality).

``parse_perplexity`` factors the "Final estimate: PPL = ..." parsing shared with
``tools/calibrate_noise_factors.py``.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict


def parse_perplexity(output: str) -> float:
    """Parse the final perplexity from ``llama-perplexity`` output.

    Pass the combined stdout+stderr: llama-perplexity prints the
    ``Final estimate: PPL = <num> +/- <err>`` line to STDERR. Looks for the
    last such line (the last one wins, matching the running-estimate output
    where the final line is the most complete). Raises ``RuntimeError`` if no
    such line is present.
    """
    for line in reversed(output.splitlines()):
        if "Final estimate" in line and "PPL" in line:
            # "Final estimate: PPL = 12.3456 +/- 0.06789"
            parts = line.split("=")
            if len(parts) >= 2:
                return float(parts[1].strip().split()[0])
    raise RuntimeError(
        "could not parse perplexity from llama-perplexity output:\n"
        f"{output[-500:]}"
    )


def _run_perplexity(
    gguf_path: str,
    corpus: str,
    perplexity_bin: str,
    ctx_size: int = 512,
    timeout: int = 900,
) -> float:
    """Run ``perplexity_bin`` once on ``gguf_path`` over ``corpus`` and parse PPL."""
    cmd = [
        perplexity_bin,
        "-m", str(gguf_path),
        "-f", str(corpus),
        "--ctx-size", str(ctx_size),
        "--threads", str(os.cpu_count() or 4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity failed (rc={proc.returncode}) for {gguf_path}:\n"
            f"stderr: {proc.stderr[-500:]}"
        )
    # The "Final estimate: PPL =" line goes to STDERR, not stdout — scan both
    # or every QAT recovery measurement raises "could not parse".
    return parse_perplexity((proc.stdout or "") + "\n" + (proc.stderr or ""))


def compare_perplexity(
    plain_gguf: str,
    qat_gguf: str,
    corpus: str,
    perplexity_bin: str,
    ctx_size: int = 512,
) -> Dict[str, float]:
    """Compare perplexity of the plain hybrid vs the QAT hybrid.

    Args:
        plain_gguf: GGUF packed from the plain (non-QAT) hybrid.
        qat_gguf: GGUF packed from the QAT-adapted hybrid.
        corpus: Path to the text corpus for llama-perplexity.
        perplexity_bin: Path to the ``llama-perplexity`` binary.
        ctx_size: Context size for the perplexity run.

    Returns:
        ``{"plain": ppl_plain, "qat": ppl_qat, "delta": ppl_plain - ppl_qat}``.
        A positive ``delta`` means QAT lowered perplexity (the goal).
    """
    plain_ppl = _run_perplexity(plain_gguf, corpus, perplexity_bin, ctx_size)
    qat_ppl = _run_perplexity(qat_gguf, corpus, perplexity_bin, ctx_size)
    return {
        "plain": plain_ppl,
        "qat": qat_ppl,
        "delta": plain_ppl - qat_ppl,
    }
