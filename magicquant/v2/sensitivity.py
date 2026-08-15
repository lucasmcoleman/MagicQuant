"""Per-tensor, per-scheme distortion estimation (docs/redesign.md §3).

For every (tensor, scheme) pair, encode the tensor with the SAME libggml
encoder the writer ships bytes through (imatrix-weighted when active),
decode it back, and record the imatrix-weighted squared error

    werr(t, s) = Σ_c m_t[c] · Σ_r (W[r,c] − Ŵ_s[r,c])²

— the expected squared output perturbation E‖(W−Ŵ)x‖² under the
independence approximation, with m = per-column mean squared activations
from llama-imatrix (m ≡ 1 when no imatrix is available; recorded in the
table's meta so consumers know which signal they got).

CPU-only; no perplexity passes. Cached to disk keyed on model identity,
scheme set, imatrix identity and sampling params.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.logging import get_logger
from magicquant.v2.resolve import resolve_tensor_type, tensor_bytes

log = get_logger(__name__)

TABLE_VERSION = 1

# Groups/type situations whose tensors are not allocatable (fixed by writer
# compatibility rules regardless of the requested scheme).
_FLOAT_TYPES = ("F32", "F16", "BF16")


# Bumped whenever the RESOLUTION logic changes -- i.e. anything that alters
# what `resolve_tensor_type` returns. The distortion table's contents are
# encode->decode error per tensor at its RESOLVED type, so a resolution change
# silently invalidates every cached entry while the key still matches. That
# happened: the 2026-08-13 weight-suffix gate changed ssm_a/ssm_d from F16 to
# F32, and nothing in the key could see it. TABLE_VERSION covers the table's
# SCHEMA; this covers its SEMANTICS. Bump it, don't reason about whether the
# delta was small enough to ignore.
RESOLUTION_VERSION = 1


def _model_identity(path: Path) -> Dict[str, Any]:
    """Identify the MODEL, not where it happens to sit.

    Deliberately excludes the absolute path. Keying on it meant any directory
    reorganisation invalidated the cache and cost a full recompute (~90 min on
    a 30B) for a file whose bytes never changed -- which is exactly what the
    2026-08-13 run-directory rename would have triggered.

    size+mtime alone would be too weak: two different models can collide, and
    a touch would falsely invalidate. So a cheap content hash anchors it --
    the first 1 MiB covers the GGUF magic, the full KV metadata block and most
    of the tensor-info table, which is the part that differs between models.
    Reading 1 MiB is negligible against building the table it guards.
    """
    try:
        st = path.stat()
        head = hashlib.sha256()
        with open(path, "rb") as fh:
            head.update(fh.read(1024 * 1024))
        return {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "head_sha256": head.hexdigest()[:16],
        }
    except OSError:
        # Unreadable -> no identity. Returning a path-derived stand-in would
        # let two unreadable models share a cache entry.
        return {"size": None, "mtime": None, "head_sha256": None}


def _imatrix_identity(imatrix: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Delegates to ``utils.measurement.imatrix_identity`` so this table's
    cache key and the v1 checkpoint-resume gate (``orchestrator.py``) hash
    "same imatrix" identically -- see that function's docstring."""
    from magicquant.utils.measurement import imatrix_identity

    return imatrix_identity(imatrix)


def _cache_key(
    model_id: Dict[str, Any],
    schemes: List[str],
    imatrix_id: Dict[str, Any],
    sample_rows: Optional[int],
) -> str:
    payload = json.dumps(
        {
            "v": TABLE_VERSION,
            "resolution_v": RESOLUTION_VERSION,
            "model": model_id,
            "schemes": sorted(schemes),
            "imatrix": imatrix_id,
            "sample_rows": sample_rows,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _weighted_sq_err(
    w: np.ndarray, w_hat: np.ndarray, m: Optional[np.ndarray]
) -> float:
    """Σ_c m[c] Σ_r (w−ŵ)²  for 2-D (rows, cols) arrays; m broadcast over rows."""
    d2 = np.square(w.astype(np.float32) - w_hat.astype(np.float32))
    if m is None:
        return float(d2.sum(dtype=np.float64))
    return float((d2.sum(axis=0, dtype=np.float64) * m.astype(np.float64)).sum())


def compute_distortion_table(
    source_model_path: str,
    schemes: List[str],
    imatrix: Optional[Dict[str, np.ndarray]] = None,
    cache_dir: Optional[str] = None,
    sample_rows: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build (or load from cache) the per-tensor × per-scheme distortion table.

    Returns::

        {
          "meta": {model, schemes, imatrix, sample_rows, seconds, version},
          "tensors": {
            name: {
              "group": str, "shape": [...], "n_elems": int,
              "fixed": bool,               # writer forces one type (norms, SSM conv)
              "wnorm": float,              # Σ m·W² (scale reference)
              "choices": {
                scheme: {"actual": str, "bytes": int, "werr": float,
                          "reason": str|None}
              }
            }
          }
        }

    Distortion is computed against the RESOLVED on-disk type (writer
    parity), so two schemes that resolve to the same actual type share one
    computation. A scheme that cannot be decoded by the loaded libggml is
    recorded with ``werr: null`` and reason ``"no-decode"`` — the allocator
    excludes it; it is never approximated (docs/redesign.md §3.1).
    """
    from magicquant.gguf.source import open_model_source
    from magicquant.quant.converters import encode_to_ggml_bytes
    from magicquant.quant.ggml_binding import ggml_decode, supports_decode

    src_path = Path(source_model_path)
    model_id = _model_identity(src_path)
    imatrix_id = _imatrix_identity(imatrix)
    key = _cache_key(model_id, schemes, imatrix_id, sample_rows)

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"distortion_{key}.json"
        if cache_path.is_file():
            try:
                table = json.loads(cache_path.read_text(encoding="utf-8"))
                if table.get("meta", {}).get("version") == TABLE_VERSION:
                    log.info(
                        "distortion table cache hit",
                        stage="sensitivity", path=str(cache_path),
                    )
                    return table
            except (OSError, ValueError) as exc:
                log.warning(
                    "distortion cache unreadable, recomputing",
                    stage="sensitivity", error=str(exc),
                )

    t0 = time.time()
    classifier = TensorGroupClassifier()
    tensors: Dict[str, Any] = {}

    source = open_model_source(str(src_path))
    try:
        infos = source.get_all_tensors_info()
        n_total = len(infos)
        for idx, info in enumerate(infos):
            name = info["name"]
            shape = [int(d) for d in info["shape"]]
            n_dims = int(info.get("n_dims", len(shape)))
            group = classifier.classify_tensor(name)
            n_elems = int(np.prod(shape)) if shape else 1

            entry: Dict[str, Any] = {
                "group": group,
                "shape": shape,
                "n_elems": n_elems,
                "fixed": False,
                "wnorm": None,
                "choices": {},
            }

            # Fixed units: writer forces F32 for 1-D tensors, SSM conv
            # operands, and never-quantize-by-name tensors (norms, expert
            # gating, etc. -- see writer._NEVER_QUANTIZE_NAME_SUBSTRINGS)
            # regardless of the requested scheme. One choice, zero distortion.
            probe_actual, probe_reason = resolve_tensor_type(
                name, shape, n_dims, group, schemes[0]
            )
            if probe_reason in (
                "f32-required-operand", "not-a-weight-tensor",
                "never-quantize-name", "1d-f32",
            ):
                entry["fixed"] = True
                entry["choices"]["F32"] = {
                    "actual": "F32",
                    "bytes": tensor_bytes("F32", shape),
                    "werr": 0.0,
                    "reason": probe_reason,
                }
                tensors[name] = entry
                continue

            flat = source.read_tensor_f32(name)
            if flat is None:
                # Pre-quantized/undecodable source tensor — v2 requires
                # BF16/F16/F32 sources exactly like the writer; record as
                # fixed passthrough with unknown distortion.
                src_type = source.get_source_type_name(name)
                entry["fixed"] = True
                entry["choices"][src_type] = {
                    "actual": src_type,
                    "bytes": tensor_bytes(src_type, shape)
                    if src_type in _FLOAT_TYPES else None,
                    "werr": 0.0,
                    "reason": "source-passthrough",
                }
                tensors[name] = entry
                continue

            w = np.asarray(flat, dtype=np.float32).reshape(shape)
            if w.ndim == 1:
                w = w.reshape(1, -1)
            cols = w.shape[-1]
            w2d = w.reshape(-1, cols)

            # Row sampling (unbiased strided estimator; docs §3.3). Encoding
            # is per-row independent in ggml, so sampled rows encode
            # identically to their full-tensor encoding.
            rows = w2d.shape[0]
            row_scale = 1.0
            if sample_rows is not None and rows > sample_rows:
                stride = max(1, rows // sample_rows)
                w2d_s = w2d[::stride][:sample_rows]
                row_scale = rows / w2d_s.shape[0]
            else:
                w2d_s = w2d

            # imatrix vector: per input column; MoE stacked _exps tensors
            # carry n_experts*cols expert-major — those need expert-shaped
            # weighting, and encode() handles the slicing internally; for
            # the error computation we broadcast per-expert below.
            m = None
            m_full = None
            if imatrix is not None:
                m_full = imatrix.get(name)
                if m_full is not None:
                    m_full = np.asarray(m_full, dtype=np.float32)
                    if m_full.size == cols:
                        m = m_full
                    elif w.ndim == 3 and m_full.size == w.shape[0] * cols:
                        m = None  # handled in the 3-D branch below
                    else:
                        log.warning(
                            "imatrix length mismatch for %s (%d vs cols=%d) "
                            "-- ignoring for this tensor",
                            name, m_full.size, cols,
                        )
                        m_full = None

            # Stacked MoE experts with a per-expert imatrix (encode/decode
            # below routes through the full-tensor 3-D branch for this).
            stacked_moe_imatrix = (
                w.ndim == 3 and m_full is not None and m_full.size == w.shape[0] * cols
            )

            sq = np.square(w2d_s.astype(np.float32)).sum(axis=0, dtype=np.float64)
            entry["wnorm"] = float(
                (sq * m.astype(np.float64)).sum() if m is not None else sq.sum()
            ) * row_scale

            # Compute distortion per RESOLVED type once, then map schemes.
            resolved: Dict[str, Any] = {}
            per_actual_err: Dict[str, Optional[float]] = {}
            for scheme in schemes:
                actual, reason = resolve_tensor_type(
                    name, shape, n_dims, group, scheme
                )
                resolved[scheme] = (actual, reason)
                if actual in per_actual_err:
                    continue
                if actual == "F32":
                    per_actual_err[actual] = 0.0
                    continue
                if actual == "F16":
                    w_hat = w2d_s.astype(np.float16).astype(np.float32)
                    per_actual_err[actual] = _weighted_sq_err(w2d_s, w_hat, m)
                    continue
                if not supports_decode(actual):
                    per_actual_err[actual] = None
                    continue
                try:
                    if stacked_moe_imatrix:
                        # Stacked MoE experts with per-expert imatrix: encode
                        # the full tensor (encoder slices the imatrix per
                        # expert), decode, and weight per expert.
                        blob = encode_to_ggml_bytes(
                            w, actual, imatrix=m_full, n_per_row=cols
                        )
                        w_hat = ggml_decode(blob, actual, w.size).reshape(w.shape)
                        m3 = m_full.reshape(w.shape[0], 1, cols)
                        d2 = np.square(w - w_hat)
                        per_actual_err[actual] = float(
                            (d2 * m3).sum(dtype=np.float64)
                        )
                    else:
                        blob = encode_to_ggml_bytes(
                            w2d_s, actual,
                            imatrix=m if m is not None else None,
                            n_per_row=cols,
                        )
                        w_hat = ggml_decode(blob, actual, w2d_s.size).reshape(
                            w2d_s.shape
                        )
                        per_actual_err[actual] = (
                            _weighted_sq_err(w2d_s, w_hat, m) * row_scale
                        )
                except Exception as exc:
                    # A failed encode/decode for one type must fail THAT
                    # choice loudly and leave the rest of the table valid.
                    log.warning(
                        "distortion computation failed for %s @ %s: %s",
                        name, actual, exc,
                    )
                    per_actual_err[actual] = None

            for scheme, (actual, reason) in resolved.items():
                err = per_actual_err.get(actual)
                entry["choices"][scheme] = {
                    "actual": actual,
                    "bytes": tensor_bytes(actual, shape),
                    "werr": err,
                    "reason": reason if err is not None else (reason or "no-decode"),
                }

            tensors[name] = entry
            if verbose and (idx + 1) % 25 == 0:
                log.info(
                    "distortion table progress",
                    stage="sensitivity",
                    done=idx + 1, total=n_total,
                )
    finally:
        source.close()

    table = {
        "meta": {
            "version": TABLE_VERSION,
            "model": model_id,
            "schemes": sorted(schemes),
            "imatrix": imatrix_id,
            "sample_rows": sample_rows,
            "seconds": round(time.time() - t0, 1),
        },
        "tensors": tensors,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(cache_path) + ".tmp"
        Path(tmp).write_text(json.dumps(table), encoding="utf-8")
        import os

        os.replace(tmp, cache_path)
        log.info(
            "distortion table cached",
            stage="sensitivity", path=str(cache_path),
            seconds=table["meta"]["seconds"],
        )

    return table
