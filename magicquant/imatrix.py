"""Importance-matrix (imatrix) capture and loading.

An importance matrix records the average squared activation seen by each input
column of every weight tensor while the model runs a calibration corpus. ggml's
quantizers use it to spend precision where activations are large; it improves
all K-quants and is REQUIRED for the very-low-bit IQ1/IQ2 types.

Capture wraps llama.cpp's ``llama-imatrix`` binary (the model must already be a
GGUF). Modern llama-imatrix writes a GGUF whose tensors come in pairs:

    <weight_name>.in_sum2   F32[n_per_row]   sum of squared activations/column
    <weight_name>.counts    F32[1]           number of activation rows summed

The per-column importance vector handed to ``ggml_quantize_chunk`` is
``in_sum2 / counts``. Loading reuses MagicQuant's own GGUF reader, so no extra
dependency is needed.

Usage:
    from magicquant.imatrix import capture_imatrix, load_imatrix
    capture_imatrix("model.gguf", "wiki.train.raw", "model.imatrix.gguf")
    imat = load_imatrix("model.imatrix.gguf")     # {tensor_name: float32[ncols]}
    create_hybrid_gguf(out, src, cfg, imatrix=imat)
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

_SUM_SUFFIX = ".in_sum2"
_COUNT_SUFFIX = ".counts"


def capture_imatrix(
    model_path: Union[str, Path],
    corpus_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    chunks: int = -1,
    ctx_size: int = 512,
    extra_args: Sequence[str] = (),
    imatrix_bin: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Path:
    """Run ``llama-imatrix`` over a calibration corpus and return the output path.

    Args:
        model_path: GGUF model to instrument (llama-imatrix cannot read
            safetensors — pack to GGUF first).
        corpus_path: plain-text calibration corpus.
        output_path: where to write the imatrix GGUF.
        chunks: max ctx_size-token chunks to process (-1 = whole corpus).
        ctx_size: chunk length in tokens.
        extra_args: extra raw CLI args appended to the command.
        imatrix_bin: explicit path to llama-imatrix (default: search PATH).
        timeout: optional subprocess timeout in seconds.
    """
    binary = imatrix_bin or shutil.which("llama-imatrix")
    if not binary:
        raise FileNotFoundError(
            "llama-imatrix not found on PATH. Install llama.cpp (e.g. "
            "`brew install llama.cpp`) or pass imatrix_bin=."
        )
    model_path, corpus_path = Path(model_path), Path(corpus_path)
    output_path = Path(output_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")

    cmd = [
        binary,
        "-m", str(model_path),
        "-f", str(corpus_path),
        "-o", str(output_path),
        "--ctx-size", str(ctx_size),
        "--chunks", str(chunks),
        "--no-ppl",
        *extra_args,
    ]
    logger.info("capturing imatrix: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"llama-imatrix failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-1000:]}"
        )
    return output_path


def load_imatrix(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load an imatrix GGUF into ``{weight_tensor_name: importance_vector}``.

    Each vector is float32 with one entry per input column of the weight it
    belongs to (``in_sum2 / counts``), ready to pass to
    ``create_hybrid_gguf(..., imatrix=...)``.
    """
    from magicquant.gguf.source import GGUFSource

    path = Path(path)
    source = GGUFSource(str(path))
    names = set(source.get_tensor_names())

    sum_names = {n for n in names if n.endswith(_SUM_SUFFIX)}
    if not sum_names:
        raise ValueError(
            f"{path} contains no '*{_SUM_SUFFIX}' tensors — not an imatrix "
            "GGUF (capture one with llama-imatrix / capture_imatrix)."
        )

    result: Dict[str, np.ndarray] = {}
    for sum_name in sorted(sum_names):
        weight_name = sum_name[: -len(_SUM_SUFFIX)]
        count_name = weight_name + _COUNT_SUFFIX
        if count_name not in names:
            raise ValueError(
                f"{path}: '{sum_name}' has no matching '{count_name}' — "
                "truncated or corrupt imatrix file."
            )
        sum2 = source.read_tensor_f32(sum_name)
        counts = source.read_tensor_f32(count_name)
        if sum2 is None or counts is None:
            raise ValueError(f"{path}: failed to read pair for '{weight_name}'")
        count = float(np.asarray(counts).ravel()[0])
        if count <= 0:
            logger.warning("imatrix: '%s' has count %s; skipping", weight_name, count)
            continue
        result[weight_name] = (
            np.asarray(sum2, dtype=np.float32).reshape(-1) / np.float32(count)
        )
    return result
