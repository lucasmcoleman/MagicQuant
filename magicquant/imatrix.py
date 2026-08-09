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

``ensure_imatrix`` wraps capture+load with an on-disk cache and a bundled
default calibration corpus, so callers don't have to hand-manage either:

    from magicquant.imatrix import ensure_imatrix
    imat = ensure_imatrix("model.gguf")           # None on any failure
    if imat is not None:
        create_hybrid_gguf(out, src, cfg, imatrix=imat)
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

_SUM_SUFFIX = ".in_sum2"
_COUNT_SUFFIX = ".counts"

# Bundled default calibration corpus — ~1 MB blended from
# eaddario/imatrix-calibration (MIT): 45% multilingual prose across 18
# languages, 20% code, 20% math, 15% agentic requests. Rebuild with
# tools/build_calib_corpus.py, which also asserts it stays disjoint from the
# wikitext eval corpus (currently 0.00000% shared 8-grams).
#
# It replaced a 13 KB English-only corpus that yielded ~5 chunks at ctx 512 --
# far too thin to estimate importance for a 27B's ~866 tensors, and with no
# non-Latin coverage at all, which matters because a 248k-token vocab
# calibrated on English alone leaves most embedding/head rows weighted at
# ~zero.
DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "calib_corpus.txt"

# Chunks to capture when the caller doesn't say. NOT -1 (whole corpus): the
# bundled corpus holds ~580 chunks, and capture costs roughly one perplexity
# pass per chunk-batch, so the whole thing would run ~1.5-2 h on a 27B for
# importance estimates that stop improving well before that. 200 chunks
# (~102k tokens) matches common llama.cpp practice and costs ~35-40 min once
# per model, cached thereafter. Pass chunks=-1 to use everything.
DEFAULT_CAPTURE_CHUNKS = 200


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


def _reduce_per_expert_imatrix(
    weight_name: str, sum2_2d: np.ndarray, counts_2d: np.ndarray
) -> Optional[np.ndarray]:
    """Divide per-expert sums by per-expert counts for a stacked MoE tensor.

    llama-imatrix tracks activation importance separately per expert for
    ``_exps`` tensors (confirmed empirically: capturing over gpt-oss-20b with
    a small corpus left several experts per layer at count=0, since MoE
    routing sends each token to only a handful of experts) — ``in_sum2``
    comes back shaped ``[n_experts, n_per_row]`` and ``counts`` shaped
    ``[n_experts, 1]``, not the flat ``[n_per_row]`` a dense tensor gets.

    An unvisited expert (count == 0) is filled with the mean of the visited
    experts' vectors rather than left at zero — a zero-everywhere importance
    row would make the encoder's weighted quantization degenerate for that
    expert's rows. Mirrors llama.cpp's own fallback for the same situation.
    Returns None (drop imatrix for this tensor entirely) only if NO expert
    was visited at all.
    """
    visited = counts_2d.ravel() > 0
    if not visited.any():
        logger.warning(
            "imatrix: '%s' has no visited experts (calibration corpus too "
            "small/narrow for this tensor's routing) -- skipping",
            weight_name,
        )
        return None

    per_expert = np.zeros_like(sum2_2d)
    per_expert[visited] = sum2_2d[visited] / counts_2d[visited]
    if not visited.all():
        fallback = per_expert[visited].mean(axis=0)
        per_expert[~visited] = fallback
        logger.info(
            "imatrix: '%s' had %d/%d unvisited expert(s); filled with the "
            "mean of visited experts",
            weight_name, int((~visited).sum()), len(visited),
        )
    return per_expert.reshape(-1)


def load_imatrix(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load an imatrix GGUF into ``{weight_tensor_name: importance_vector}``.

    Each vector is float32 with one entry per input column of the weight it
    belongs to (``in_sum2 / counts``), ready to pass to
    ``create_hybrid_gguf(..., imatrix=...)``. For a stacked MoE ``_exps``
    tensor, the vector is instead ``n_experts`` such slices concatenated
    expert-major (``[n_experts * n_per_row]``) — see
    ``_reduce_per_expert_imatrix`` — which ``ggml_binding.encode()``
    recognizes and quantizes expert-by-expert.
    """
    from magicquant.gguf.source import GGUFSource

    path = Path(path)
    source = GGUFSource(str(path))
    names = set(source.get_tensor_names())
    shapes = {info["name"]: info["shape"] for info in source.get_all_tensors_info()}

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

        sum_shape = shapes.get(sum_name)
        if sum_shape is not None and len(sum_shape) == 2:
            n_experts, n_per_row = sum_shape
            sum2_2d = np.asarray(sum2, dtype=np.float32).reshape(n_experts, n_per_row)
            counts_2d = np.asarray(counts, dtype=np.float32).reshape(n_experts, 1)
            vector = _reduce_per_expert_imatrix(weight_name, sum2_2d, counts_2d)
            if vector is not None:
                result[weight_name] = vector
            continue

        count = float(np.asarray(counts).ravel()[0])
        if count <= 0:
            logger.warning("imatrix: '%s' has count %s; skipping", weight_name, count)
            continue
        result[weight_name] = (
            np.asarray(sum2, dtype=np.float32).reshape(-1) / np.float32(count)
        )
    return result


def _imatrix_cache_key(
    source_path: Path, corpus_path: Path, ctx_size: int, chunks: int
) -> str:
    """Short, stable hash over the inputs that determine imatrix content.

    Keyed on the source model's identity (name + mtime + size — cheap stand-in
    for a content hash that still invalidates on re-export/re-quantize) plus
    the corpus and capture parameters, so a stale cache is never reused after
    the model, corpus, or capture settings change.
    """
    st = source_path.stat()
    payload = "|".join(
        [
            source_path.name,
            str(int(st.st_mtime)),
            str(st.st_size),
            corpus_path.name,
            str(ctx_size),
            str(chunks),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_imatrix_bin(tools: object) -> Optional[str]:
    """Resolve ``llama-imatrix`` as the SIBLING of ``tools``' already-resolved
    perplexity binary, instead of leaving capture to a bare PATH lookup.

    A bare ``shutil.which("llama-imatrix")`` (``capture_imatrix``'s own
    fallback) can silently resolve to a DIFFERENT llama.cpp build than the
    one a caller configured -- e.g. a stock PATH install that can't load an
    architecture only a configured fork build supports (this bit for real on
    a qwen35 MTP model, 2026-07-04; see orchestrator.enable_imatrix). Every
    caller that already resolved an ``LlamaCppTools`` instance should prefer
    the sibling of ITS perplexity tool so imatrix capture uses the same
    build as every other measurement pass.

    Args:
        tools: an ``LlamaCppTools`` instance (or ``None``/anything exposing
            ``perplexity_tool``). Untyped to avoid importing
            ``magicquant.utils.llamacpp`` here (avoids a needless import for
            callers that only want the resolution logic).

    Returns:
        The sibling path as a string if it exists on disk, else ``None`` --
        callers should leave ``imatrix_bin`` unset in that case so
        ``capture_imatrix``'s own PATH fallback applies (today's behavior).
    """
    perplexity = getattr(tools, "perplexity_tool", None)
    if not perplexity:
        return None
    sibling = Path(perplexity).parent / "llama-imatrix"
    return str(sibling) if sibling.exists() else None


def ensure_imatrix(
    source_model_path: Union[str, Path],
    *,
    corpus_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    chunks: int = DEFAULT_CAPTURE_CHUNKS,
    ctx_size: int = 512,
    imatrix_bin: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Capture (or reuse a cached) imatrix for ``source_model_path`` and load it.

    This is the orchestration layer over ``capture_imatrix`` / ``load_imatrix``:
    it derives a cache key from the source model's identity, the corpus, and
    the capture parameters, skips capture entirely on a cache hit, and always
    returns a usable ``{tensor_name: importance_vector}`` dict or ``None`` —
    it never raises, so callers can unconditionally do
    ``create_hybrid_gguf(..., imatrix=ensure_imatrix(...))``.

    Args:
        source_model_path: path to the model to instrument. llama-imatrix can
            only read a GGUF; if this doesn't end in ``.gguf`` (e.g. a
            safetensors file or checkpoint directory) this function does NOT
            attempt to pack one — it logs and returns ``None``. Callers with a
            safetensors model must pack a BF16 GGUF first (the pipeline
            orchestrator is expected to do this) and pass that path in.
        corpus_path: plain-text calibration corpus. Defaults to the bundled
            ``magicquant/data/calib_corpus.txt`` (a few KB of neutral, diverse
            English prose) when not given.
        cache_dir: directory to store captured imatrix GGUFs under. Defaults
            to ``<source_model_path's dir>/_imatrix``.
        chunks: max ctx_size-token chunks to process (-1 = whole corpus).
        ctx_size: chunk length in tokens.
        imatrix_bin: explicit path to llama-imatrix (default: search PATH).
        timeout: optional subprocess timeout in seconds, passed through to
            ``capture_imatrix``.

    Returns:
        ``{tensor_name: importance_vector}`` on success, or ``None`` if the
        source isn't a GGUF, doesn't exist, or capture failed for any reason
        (missing binary, non-zero exit, timeout, ...).
    """
    source_model_path = Path(source_model_path)

    if source_model_path.suffix.lower() != ".gguf":
        logger.warning(
            "ensure_imatrix: %s is not a GGUF — llama-imatrix can only "
            "instrument GGUFs. Pack a BF16 GGUF first; skipping imatrix "
            "capture.",
            source_model_path,
        )
        return None

    if not source_model_path.exists():
        logger.warning(
            "ensure_imatrix: source model not found: %s", source_model_path
        )
        return None

    corpus = Path(corpus_path) if corpus_path is not None else DEFAULT_CORPUS_PATH
    cache_root = (
        Path(cache_dir) if cache_dir is not None else source_model_path.parent / "_imatrix"
    )

    try:
        key = _imatrix_cache_key(source_model_path, corpus, ctx_size, chunks)
    except OSError:
        logger.warning(
            "ensure_imatrix: could not stat source model %s",
            source_model_path,
            exc_info=True,
        )
        return None
    cache_path = cache_root / f"{key}.imatrix.gguf"

    if cache_path.exists():
        logger.info("ensure_imatrix: cache hit at %s", cache_path)
    else:
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            capture_imatrix(
                source_model_path,
                corpus,
                cache_path,
                chunks=chunks,
                ctx_size=ctx_size,
                imatrix_bin=imatrix_bin,
                timeout=timeout,
            )
        except Exception:
            logger.warning(
                "ensure_imatrix: capture failed for %s; continuing without "
                "an imatrix",
                source_model_path,
                exc_info=True,
            )
            return None

    try:
        return load_imatrix(cache_path)
    except Exception:
        logger.warning(
            "ensure_imatrix: failed to load captured imatrix %s",
            cache_path,
            exc_info=True,
        )
        return None
