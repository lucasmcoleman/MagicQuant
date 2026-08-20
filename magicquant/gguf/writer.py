"""
GGUF Writer - Create hybrid quantization GGUF files.

This module creates new GGUF files with mixed-precision quantization,
combining different quant schemes for different tensor groups.

Accepts both GGUF and safetensors as source formats via the ModelSource
abstraction in ``magicquant.gguf.source``.

Architecture:
  Pass 1 (header): Compute target types, data sizes, and offsets for every
      tensor without touching actual data. Write the complete GGUF header.
  Pass 2 (data): A pool of background threads reads + encodes tensors while
      the main thread writes blobs to disk in tensor order. This overlaps
      I/O with computation, and — since ggml_quantize_chunk is a ctypes call
      that releases the GIL for its duration — lets the encode workers run
      genuinely in parallel across CPU cores. Encoded blobs are still
      written out strictly in Pass-1 order regardless of which worker
      finished them, so output is byte-identical to the single-worker path
      no matter how many threads are used (see ``_parallel_encode_iter``).
      Thread count defaults to ``min(8, os.cpu_count() // 2)``; override with
      ``MAGICQUANT_ENCODE_THREADS`` (``1`` reproduces the exact historical
      single-worker code path).
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import collections
import concurrent.futures
import json
import re
import struct
import os
import time
import threading
import queue
import logging
import numpy as np

logger = logging.getLogger(__name__)

import gguf.constants as _gguf_constants

from magicquant.quant.converters import (
    encode_to_ggml_bytes,
    ggml_tensor_data_size,
    GGML_BLOCK_SIZE,
)
from magicquant.quant.schemes import get_all_schemes
from magicquant.quant import ggml_facts

# ggml_type_name -> bits_per_weight, derived from the scheme registry so the
# block-32 fallback's low/high-bit split (see _block32_fallback) can never
# drift from the registry the way the old hand-maintained tuple did.
_GGML_NAME_TO_BPW: Dict[str, float] = {
    s.ggml_type_name: s.bits_per_weight for s in get_all_schemes()
}


# ggml_type enum values used in GGUF tensor info. Derived from
# magicquant.quant.ggml_facts (stock names/ids from the installed `gguf`
# package, ROCmFPX fork types 100-104 overlaid from ggml_facts.FORK_TYPES —
# see that module's docstring). GGML_TYPE name kept for backward
# compatibility (external references, e.g. tests/test_writer_tensor_overrides.py).
GGML_TYPE = dict(ggml_facts.NAME_TO_ID)

# GGUF metadata value-type tags. Derived from the installed `gguf` package's
# own GGUFValueType enum (gguf.constants) rather than hand-typed ints — same
# "facts come from upstream, never drift" policy as ggml_facts.py's
# NAME_TO_ID (see that module's docstring for the incident this class of bug
# caused: a hand-copied table silently going stale). Local `_GGUF_TYPE_*`
# names are kept as aliases (many call sites below reference them, plus
# external references e.g. tests) — only the right-hand side now comes from
# upstream. Values are asserted identical to the historical literals in
# tests/test_writer_gguf_constants.py.
_GGUFValueType     = _gguf_constants.GGUFValueType
_GGUF_TYPE_UINT8   = int(_GGUFValueType.UINT8)
_GGUF_TYPE_INT8    = int(_GGUFValueType.INT8)
_GGUF_TYPE_UINT16  = int(_GGUFValueType.UINT16)
_GGUF_TYPE_INT16   = int(_GGUFValueType.INT16)
_GGUF_TYPE_UINT32  = int(_GGUFValueType.UINT32)
_GGUF_TYPE_INT32   = int(_GGUFValueType.INT32)
_GGUF_TYPE_FLOAT32 = int(_GGUFValueType.FLOAT32)
_GGUF_TYPE_BOOL    = int(_GGUFValueType.BOOL)
_GGUF_TYPE_STRING  = int(_GGUFValueType.STRING)
_GGUF_TYPE_ARRAY   = int(_GGUFValueType.ARRAY)
_GGUF_TYPE_UINT64  = int(_GGUFValueType.UINT64)
_GGUF_TYPE_INT64   = int(_GGUFValueType.INT64)
_GGUF_TYPE_FLOAT64 = int(_GGUFValueType.FLOAT64)

# Default tensor-data alignment (bytes), from the same package
# (gguf.constants.GGUF_DEFAULT_ALIGNMENT) rather than a hand-typed literal.
ALIGNMENT = _gguf_constants.GGUF_DEFAULT_ALIGNMENT

# struct.pack format code for each GGUF integer scalar/array element type.
# Used by _write_metadata_value to round-trip a KV value at its EXACT
# recorded source type (magicquant.gguf.reader.GGUFTypedInt/GGUFTypedArray's
# ``.gguf_type``) instead of re-deriving a type from the Python value's
# magnitude alone -- see that function's docstring for the incident this
# closes (tokenizer.ggml.token_type: source INT32, values 0-6, magnitude
# inference always prefers UINT32 for non-negative arrays -- HF discussion
# lmcoleman/Qwen3.8-27B-MagicQuant-GGUF#1).
_INT_STRUCT_FMT: Dict[int, str] = {
    _GGUF_TYPE_UINT8:  "B",
    _GGUF_TYPE_INT8:   "b",
    _GGUF_TYPE_UINT16: "H",
    _GGUF_TYPE_INT16:  "h",
    _GGUF_TYPE_UINT32: "I",
    _GGUF_TYPE_INT32:  "i",
    _GGUF_TYPE_UINT64: "Q",
    _GGUF_TYPE_INT64:  "q",
}

# Map MagicQuant scheme names to the ggml_type name we write into the file.
# Built from the canonical scheme registry; F16/F32 added as passthrough
# entries for source tensors that bypass quantization.
SCHEME_TO_GGML: Dict[str, str] = {s.name: s.ggml_type_name for s in get_all_schemes()}
SCHEME_TO_GGML["F16"] = "F16"
SCHEME_TO_GGML["F32"] = "F32"

# Map MagicQuant scheme/ggml names to the general.file_type value llama.cpp
# and HuggingFace use to report a GGUF's quantization type. Consulted by
# GGUFWriter._build_metadata, which determines the dominant scheme by
# counting actual parameter elements per scheme across all tensors (after
# Pass 1) and looks it up here. Aligned to llama.cpp's LLAMA_FTYPE enum.
# This is a cosmetic, human-readable badge only (each tensor carries its own
# ggml_type, so inference is unaffected). Generic "Q4_K"/"Q5_K" map to the
# _M variant. Values come from gguf.constants.LlamaFileType BY MEMBER NAME
# (not hand-typed ints) -- same "facts come from upstream, never drift"
# policy as the _GGUF_TYPE_*/GGML_TYPE tables above. This table's own
# hand-typed predecessor already drifted once (Q4_K->12, Q5_K->16,
# IQ4_NL->20 were wrong: 12=MOSTLY_Q4_1_SOME_F16-era, 16=Q5_K_S,
# 20=MOSTLY_IQ2_XS) -- deriving from the enum closes that class of bug.
# NOTE: the key set here is deliberate and load-bearing for
# _build_metadata's membership filter and fallback lookup -- do not add or
# remove keys based on the enum's full membership.
# Hoisted to module level (built once at import, not per create_hybrid_gguf
# call) so a renamed/removed upstream LlamaFileType member surfaces as an
# AttributeError at import time rather than mid-build; see
# tests/test_writer_gguf_constants.py for the drift-tripwire tests.
_LlamaFileType = _gguf_constants.LlamaFileType
_ftype_map: Dict[str, int] = {
    "F32": int(_LlamaFileType.ALL_F32),
    "F16": int(_LlamaFileType.MOSTLY_F16),
    "BF16": int(_LlamaFileType.MOSTLY_BF16),
    "Q8_0": int(_LlamaFileType.MOSTLY_Q8_0),
    "Q6_K": int(_LlamaFileType.MOSTLY_Q6_K),
    "Q5_K": int(_LlamaFileType.MOSTLY_Q5_K_M),
    "Q5_K_M": int(_LlamaFileType.MOSTLY_Q5_K_M),
    "Q5_K_S": int(_LlamaFileType.MOSTLY_Q5_K_S),
    "Q4_K": int(_LlamaFileType.MOSTLY_Q4_K_M),
    "Q4_K_M": int(_LlamaFileType.MOSTLY_Q4_K_M),
    "Q4_K_S": int(_LlamaFileType.MOSTLY_Q4_K_S),
    "Q3_K": int(_LlamaFileType.MOSTLY_Q3_K_M),
    "Q3_K_M": int(_LlamaFileType.MOSTLY_Q3_K_M),
    "Q2_K": int(_LlamaFileType.MOSTLY_Q2_K),
    "IQ4_NL": int(_LlamaFileType.MOSTLY_IQ4_NL),
    "IQ4_XS": int(_LlamaFileType.MOSTLY_IQ4_XS),
}

# llama.cpp's gguf-split KV keys. GGUFReader (magicquant.gguf.reader) parses
# these into plain Python ints with no memory of their on-disk GGUF type, and
# _write_metadata_value below re-derives the output type from the Python
# value's magnitude alone -- so a source split.no/split.count (u16) or
# split.tensors.count (i32) always round-trips as u32. llama.cpp's model
# loader type-checks these three keys strictly against u16/u16/i32 and
# refuses to load a file where they come back u32 ("key split.count has
# wrong type u32 but expected type u16"). They are excluded from
# _build_metadata's KV copy regardless of type fixing that, though, because
# this writer always emits exactly one file: per llama.cpp's gguf-split
# convention, absence of split.count means single-file, so these keys don't
# belong in a single-file artifact's metadata at all.
_SPLIT_KV_KEYS = frozenset({"split.no", "split.count", "split.tensors.count"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align(offset: int, alignment: int = ALIGNMENT) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def _tensor_n_elements(shape: List[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# GGUF binary serialisation helpers
# ---------------------------------------------------------------------------

def _write_string(f, s: str):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_metadata_value(f, value: Any):
    # Normalize numpy scalars (np.int64, np.float32, np.bool_) and 0-d arrays
    # to native Python types so the isinstance ladder below tags them
    # correctly. Without this, np.int64 fails `isinstance(value, int)` and
    # falls through to the STRING branch, writing e.g. head_count as text.
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    # A value read by magicquant.gguf.reader.GGUFReader (GGUFTypedInt /
    # GGUFTypedArray) carries the EXACT on-disk GGUF type it was parsed
    # as. When present, that recorded type is authoritative and used
    # verbatim below -- a copied KV must round-trip at its source element
    # type, never get re-derived from the Python value's magnitude (which
    # cannot tell a signed array of small non-negative values apart from an
    # unsigned one; see _INT_STRUCT_FMT's docstring). Values with no such
    # tag (freshly constructed by MagicQuant itself, e.g. plain lists in
    # tests) fall through to the historical magnitude/Python-type
    # inference unchanged.
    src_type = getattr(value, "gguf_type", None)

    if isinstance(value, bool):
        f.write(struct.pack("<I", _GGUF_TYPE_BOOL))
        f.write(struct.pack("<?", value))
    elif src_type in _INT_STRUCT_FMT and isinstance(value, int):
        f.write(struct.pack("<I", src_type))
        f.write(struct.pack("<" + _INT_STRUCT_FMT[src_type], int(value)))
    elif isinstance(value, int):
        if value < 0:
            f.write(struct.pack("<I", _GGUF_TYPE_INT64))
            f.write(struct.pack("<q", value))
        elif value <= 0xFFFFFFFF:
            f.write(struct.pack("<I", _GGUF_TYPE_UINT32))
            f.write(struct.pack("<I", value))
        else:
            f.write(struct.pack("<I", _GGUF_TYPE_UINT64))
            f.write(struct.pack("<Q", value))
    elif isinstance(value, float):
        f.write(struct.pack("<I", _GGUF_TYPE_FLOAT32))
        f.write(struct.pack("<f", value))
    elif isinstance(value, str):
        f.write(struct.pack("<I", _GGUF_TYPE_STRING))
        _write_string(f, value)
    elif isinstance(value, (list, tuple)):
        f.write(struct.pack("<I", _GGUF_TYPE_ARRAY))
        if not value:
            # Preserve the recorded element type for an empty array too
            # (e.g. an empty FLOAT32 array must not come back tagged
            # UINT32); only fall back to UINT32 when no type was recorded.
            f.write(struct.pack(
                "<I", src_type if src_type is not None else _GGUF_TYPE_UINT32,
            ))
            f.write(struct.pack("<Q", 0))
        elif src_type in _INT_STRUCT_FMT:
            fmt = _INT_STRUCT_FMT[src_type]
            ints = [
                int(item.item()) if isinstance(item, np.generic) else int(item)
                for item in value
            ]
            f.write(struct.pack("<I", src_type))
            f.write(struct.pack("<Q", len(ints)))
            for item in ints:
                f.write(struct.pack("<" + fmt, item))
        elif src_type == _GGUF_TYPE_FLOAT64:
            f.write(struct.pack("<I", _GGUF_TYPE_FLOAT64))
            f.write(struct.pack("<Q", len(value)))
            for item in value:
                f.write(struct.pack("<d", float(item)))
        elif src_type == _GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<I", _GGUF_TYPE_FLOAT32))
            f.write(struct.pack("<Q", len(value)))
            for item in value:
                f.write(struct.pack("<f", float(item)))
        elif src_type == _GGUF_TYPE_BOOL:
            f.write(struct.pack("<I", _GGUF_TYPE_BOOL))
            f.write(struct.pack("<Q", len(value)))
            for item in value:
                f.write(struct.pack("<?", bool(item)))
        elif src_type == _GGUF_TYPE_STRING:
            f.write(struct.pack("<I", _GGUF_TYPE_STRING))
            f.write(struct.pack("<Q", len(value)))
            for item in value:
                _write_string(f, str(item))
        else:
            # No recorded source type -- historical magnitude/Python-type
            # inference (untyped list/tuple, e.g. constructed directly by
            # MagicQuant or a test fixture).
            # Normalize numpy scalars in the list so type detection works.
            norm = [
                v.item() if isinstance(v, (np.generic,)) else v
                for v in value
            ]
            first = norm[0]
            if isinstance(first, str):
                f.write(struct.pack("<I", _GGUF_TYPE_STRING))
                f.write(struct.pack("<Q", len(norm)))
                for item in norm:
                    _write_string(f, str(item))
            elif isinstance(first, bool):
                # bool is a subclass of int — handle before the int branch.
                f.write(struct.pack("<I", _GGUF_TYPE_BOOL))
                f.write(struct.pack("<Q", len(norm)))
                for item in norm:
                    f.write(struct.pack("<?", bool(item)))
            elif isinstance(first, float):
                f.write(struct.pack("<I", _GGUF_TYPE_FLOAT32))
                f.write(struct.pack("<Q", len(norm)))
                for item in norm:
                    f.write(struct.pack("<f", float(item)))
            elif isinstance(first, int):
                # Pick the narrowest tag that fits ALL items so values >= 2^31
                # don't raise struct.error. Prefer UINT32 for non-negative
                # arrays, fall back to INT32, then INT64.
                ints = [int(item) for item in norm]
                lo, hi = min(ints), max(ints)
                if lo >= 0 and hi <= 0xFFFFFFFF:
                    f.write(struct.pack("<I", _GGUF_TYPE_UINT32))
                    f.write(struct.pack("<Q", len(ints)))
                    for item in ints:
                        f.write(struct.pack("<I", item))
                elif -(2 ** 31) <= lo and hi <= 2 ** 31 - 1:
                    f.write(struct.pack("<I", _GGUF_TYPE_INT32))
                    f.write(struct.pack("<Q", len(ints)))
                    for item in ints:
                        f.write(struct.pack("<i", item))
                else:
                    f.write(struct.pack("<I", _GGUF_TYPE_INT64))
                    f.write(struct.pack("<Q", len(ints)))
                    for item in ints:
                        f.write(struct.pack("<q", item))
            else:
                f.write(struct.pack("<I", _GGUF_TYPE_STRING))
                f.write(struct.pack("<Q", len(norm)))
                for item in norm:
                    _write_string(f, str(item))
    elif isinstance(value, dict):
        f.write(struct.pack("<I", _GGUF_TYPE_STRING))
        _write_string(f, json.dumps(value))
    else:
        f.write(struct.pack("<I", _GGUF_TYPE_STRING))
        _write_string(f, str(value))


# ---------------------------------------------------------------------------
# Pipeline worker (runs in background thread)
# ---------------------------------------------------------------------------

def _requires_imatrix(target_ggml_name: str) -> bool:
    """Does this ggml type REQUIRE an importance matrix to produce usable
    output (IQ1/IQ2 family)? Lazy libggml lookup; callers must only consult
    this for quantized targets (float passthroughs never need libggml)."""
    from magicquant.quant.ggml_binding import get_handle
    return get_handle().requires_imatrix(target_ggml_name)


def _block32_fallback(target_ggml_name: str, row_size: int, group: str) -> str:
    """Pick a fallback type when a K-quant (block 256) can't encode ``row_size``.

    K-quant block size is 256, so a row width that isn't a multiple of 256 can't
    be K-quantized. Rather than bloat to F32 (lossless but huge — this turned a
    ~14 GB MoE pack into 39 GB), prefer a block-32 quant that DOES fit: MXFP4 for
    low-bit targets, Q8_0 for high-bit. F32 is kept only where it's genuinely
    needed — SSM/linear-attention conv operands (group ``S``, which llama.cpp
    requires in F32) — or when the row isn't even 32-divisible (no block-32
    scheme fits either).

    "Low-bit" is derived from the registry's ``bits_per_weight`` rather than a
    hand-maintained name tuple, so a target ends up on the correct side by its
    actual size class instead of by whether someone remembered to list it here
    (the previous hard-coded tuple never grew to cover the IQ2/IQ3/IQ4 family,
    misrouting them to the much larger Q8_0 fallback). The 4.5 bpw threshold
    keeps the original 5 members (Q2_K, Q3_K, Q4_K, IQ4_NL, MXFP4, all <= 4.5
    bpw) on the low-bit side, above Q5_K (5.5 bpw) and everything higher.
    """
    if group == "S" or row_size % 32 != 0:
        return "F32"
    bpw = _GGML_NAME_TO_BPW.get(target_ggml_name)
    low_bit = bpw is not None and bpw <= 4.5
    return "MXFP4" if low_bit else "Q8_0"


def _is_quantization_candidate(name: str, shape: tuple, n_dims: int) -> bool:
    """Is this tensor one the search can meaningfully assign a scheme to?

    False for everything Pass 1 forces to F32 irrespective of scheme -- 1-D
    tensors, F32-required SSM conv operands, and the never-quantize-by-name
    list -- and for rows too narrow for any block quant at all. Lives beside
    _block32_fallback so the two cannot drift: this answers "does the scheme
    choice matter for this tensor", that answers "what does it become".
    """
    if n_dims <= 1:
        return False
    if _is_f32_required_ssm_operand(name) or _is_never_quantized(name):
        return False
    row_size = shape[-1] if shape else 0
    return row_size % 32 == 0


def is_block32_only_tensor(name: str, shape: tuple, n_dims: int) -> bool:
    """Is this tensor a real quantization candidate whose row width admits a
    block-32 quant but NOT a block-256 K-quant?

    Public (no underscore) because the search consumes it -- the orchestrator
    uses it to decide which groups get Q5_0/Q5_1 offered. Deliberately placed
    next to ``_block32_fallback``, whose rule it has to agree with: a tensor
    this returns True for is exactly one where any K-quant assignment gets
    rewritten by that fallback, so the registry's advertised bpw for a
    K-quant is a lie about this tensor's real cost.

    Note the asymmetry with ``_is_never_quantized``: a never-quantized tensor
    is not "block-32-only", it is not a candidate at all, so it returns False
    rather than True. Treating it as block-32-only would let a group made
    entirely of norms qualify for Q5_0/Q5_1, which can never take effect.
    """
    if not _is_quantization_candidate(name, shape, n_dims):
        return False
    row_size = shape[-1] if shape else 0
    return row_size % 256 != 0


def _encode_entry(source, entry, imatrix=None) -> bytes:
    """Read one tensor from ``source`` and encode it per its Pass-1 target.

    Pulled out of the historical single-worker loop so both the N=1 (exact
    historical) and N>1 (pooled) encode paths run the identical logic byte
    for byte — the only difference between them is how many threads call
    this function concurrently, never what it computes.
    """
    name = entry["name"]
    can_decode = entry["_can_decode"]
    target = entry["_target_ggml_name"]
    expected = entry["_expected_size"]

    # f32 is read ONLY on the can_decode branch below. Calling
    # read_tensor_f32 unconditionally used to be safe because an
    # undecodable tensor just returned None; now that dequant support lets
    # GGUFSource.read_tensor_f32 RAISE on a recognized-but-failed decode
    # (see source.py), calling it here for a passthrough tensor (can_decode
    # False, source type == target type, no decode ever intended) would
    # blow up a copy that was never supposed to touch the decoder at all.
    if can_decode:
        f32 = source.read_tensor_f32(name)
        if f32 is None:
            # Contract violation: can_decode() promised real data for this
            # tensor. Fail loudly rather than fall through to a zero-filled
            # blob (the old catch-all this replaces).
            raise RuntimeError(
                f"Tensor {name}: source.can_decode() returned True but "
                f"read_tensor_f32() returned None."
            )
        # Validate dtype before quantization dispatch.
        # Source should return float32, but guard against bugs
        # in source implementations that could return integer or
        # pre-quantized data, which would silently corrupt output.
        if not np.issubdtype(f32.dtype, np.floating):
            raise ValueError(
                f"Tensor {name}: expected floating-point data from "
                f"source but got dtype={f32.dtype}. Source model may "
                f"be pre-quantized. Use a BF16/F16/F32 source."
            )
        imat_vec = imatrix.get(name) if imatrix else None
        # Sources return flat buffers; the importance vector is per
        # input column, so supply the true row width from Pass-1
        # shape metadata (row-major convention: ne0 = shape[-1]).
        row_width = entry["shape"][-1] if imat_vec is not None else None
        blob = encode_to_ggml_bytes(
            f32, target, imatrix=imat_vec, n_per_row=row_width,
        )
    else:
        # Passthrough: the bad_tensors gate in Pass 1 already confirmed the
        # desired type equals the source type, so there's nothing to decode
        # or re-encode -- just copy the tensor's exact on-disk bytes.
        raw = source.read_tensor_raw(name)
        if raw is None:
            raise RuntimeError(
                f"Tensor {name}: passthrough (source type "
                f"{entry['_source_type_name']!r}) requires "
                f"source.read_tensor_raw() to return the tensor's raw "
                f"bytes, but it returned None. The source cannot produce a "
                f"verbatim byte copy for this tensor."
            )
        blob = raw

    # Validate blob size against expected
    if len(blob) != expected:
        if target in ("F32", "F16", "BF16"):
            # Safe to pad/trim uncompressed formats
            logger.warning(
                "Tensor %s: encoded blob size %d != expected %d "
                "(target type %s, %d elements); %s to fit",
                name, len(blob), expected, target,
                entry["_n_elems"],
                "padding" if len(blob) < expected else "trimming",
            )
            if len(blob) < expected:
                blob = blob + b"\x00" * (expected - len(blob))
            else:
                blob = blob[:expected]
        else:
            raise RuntimeError(
                f"Tensor {name}: encoder produced {len(blob)} bytes "
                f"but expected {expected} for type {target}"
            )

    return blob


# Tensor-name shapes (within group "S") that llama.cpp's own converter
# (convert_hf_to_gguf.py's tensor_force_quant ladder, MODEL_TENSOR.SSM_CONV1D
# / _Q / _K / _V) forces to F32 unconditionally -- these are the SSM/KDA conv
# weight (and its bias), never a quantizable projection matrix. The ggml-cuda
# ssm_conv kernel hard-asserts the conv weight's row stride is sizeof(float)
# (ggml-cuda/ssm-conv.cu), so a target of BF16/F16 is just as fatal as a
# quantized one -- this rule must win regardless of the group's configured
# scheme being a float type, unlike the block-32 fallback above (which only
# ever runs for quantized targets; a float target has block_size==1 and never
# reaches it). Matches both the canonical GGUF name (ssm_conv1d[_qkv]) and the
# Kimi-Linear HF name (q_conv1d/k_conv1d/v_conv1d) in case a source hasn't
# been canonicalized yet.
_SSM_F32_REQUIRED_NAME_RE = re.compile(
    r"(?:^|[._])conv1d(?:_[qkv])?(?:[._]|$)", re.IGNORECASE,
)


def _is_f32_required_ssm_operand(name: str) -> bool:
    """Is ``name`` an SSM conv-weight operand llama.cpp requires in F32?"""
    return bool(_SSM_F32_REQUIRED_NAME_RE.search(name))


def _is_quantizable_by_name(name: str) -> bool:
    """Mirror of llama.cpp's FIRST quantization gate, before any of its
    name-based refusals::

        bool quantize = name.rfind("weight") == name.size() - 6;

    A tensor whose name does not end in ``weight`` is not a quantization
    candidate at all upstream -- it is copied through at its source dtype,
    which for these is F32.

    This is the structural form of a rule that had been accumulating as
    special cases. Three members of one family were missed in turn by the
    hand-maintained name patterns: ``ssm_norm.weight`` (2026-08-13 morning,
    aborted llama.cpp in a binary op), then ``ssm_a`` and ``ssm_d``. The
    previous guard, ``_is_f32_required_ssm_operand``, matched only
    ``conv1d`` AND was gated on ``group == "S"`` -- and ``ssm_norm.weight``
    classifies into group ``N``, so the gate had already failed on its own
    terms before the pattern did.

    ``ssm_a`` and ``ssm_d`` are the reason this is not merely cosmetic. Both
    are 2-D ``(64, 1)``, so the 1-D rule does not fire; ``ne[0] == 1`` is not
    32-divisible, so a QUANTIZED target falls back to F32 and looks safe --
    but a FLOAT target has ``block_size == 1``, skips the block-size check
    entirely, and is written F16. llama.cpp then aborts: ``ssm_a`` is src3 of
    ``ggml_ssm_scan``, which asserts ``nb[0] == sizeof(float)``
    (ggml-cpu/ops.cpp:9655), and ``ssm_d`` is src1 of a ``ggml_mul``
    (mamba-base.cpp:136,267). v1 probing never hit either because it holds
    groups at Q8_0 -- a quantized keep-scheme. v2 holds them at BF16, which
    is how it found both.

    MEASURED BLAST RADIUS on nemotron_h_moe, every non-``weight`` tensor::

        exp_probs_b.bias   x24  (128,)     1-D -> already F32
        ssm_conv1d.bias    x23  (6144,)    1-D -> already F32
        ssm_dt.bias        x23  (64,)      1-D -> already F32
        ssm_a              x23  (64, 1)    2-D -> CHANGES: F16 -> F32
        ssm_d              x23  (64, 1)    2-D -> CHANGES: F16 -> F32

    So this rule widens behaviour by exactly the two tensors it exists to
    fix; every other non-weight tensor is already F32 via the 1-D rule.
    """
    return name.endswith("weight")


# Tensor names llama.cpp's own quantizer refuses to quantize outright,
# regardless of shape -- mirrored from the `quantize &= name.find(...)`
# chain in llama.cpp's src/llama-quant.cpp (llama_model_quantize_impl,
# the block starting at `bool quantize = name.rfind("weight") == ...`).
# tests/test_never_quantize_upstream_parity.py re-derives that chain
# from a real llama.cpp checkout and fails if this tuple falls behind
# it -- the first transcription of this list stopped 11 rules short.
# This is what covers e.g. nemotron_h_moe's blk.*.ssm_norm.weight: it's
# 2-D (so the 1D-F32 rule below doesn't fire) with ne[0]=512 (both
# 32- and 256-divisible, so the block-size fallback below doesn't fire
# either), so an un-mirrored writer quantizes it to Q8_0 and llama.cpp
# later aborts in ggml-cpu binary_op on the f32/q8_0 operand mismatch
# (ssm_norm feeds ggml_mul via build_norm in mamba-base.cpp).
#
# This is a DIFFERENT constraint from _is_f32_required_ssm_operand
# above: that one is about a CUDA kernel hard-asserting a float row
# stride, this one mirrors what upstream's quantizer will and won't
# touch by name.
#
# Note what the `target_ggml_name != "F32"` guard at the call site does
# and does NOT mean. It only skips re-logging a tensor the SSM rule
# already pinned; a BF16/F16 target still gets overridden to F32 here.
# That is a deliberate divergence from upstream, which copies a
# non-quantized tensor verbatim at its source dtype
# (llama-quant.cpp's `!quantize` branch). For "_norm.weight" the F32 is
# load-bearing -- the norm weight is src1 of ggml_mul and
# ggml-cpu/binary-ops.cpp accepts src1 only as F32 against an f32
# src0/dst, so "preserve the source dtype" would abort on f16 exactly
# as it did on q8_0. For the gemma3n-only names (altup*, laurel*,
# per_layer_model_proj) it is pure size: they are ggml_mul_mat operands
# where F16 would be legal, and forcing F32 costs ~106 MB on E4B.
# Keep both checks; this list intentionally includes "ssm_conv1d" and
# "shortconv.conv.weight" too so a group-"S" mismatch doesn't slip
# past both, even though _is_f32_required_ssm_operand already forces
# those to F32 first.
_NEVER_QUANTIZE_NAME_SUBSTRINGS = (
    "_norm.weight",             # do not quantize norm tensors
    "ffn_gate_inp.weight",      # expert gating
    "ffn_gate_tid2eid.weight",  # DeepSeek-V4 i32 routing table
    "altup",
    "laurel",
    "per_layer_model_proj",
    "ssm_conv1d",
    "shortconv.conv.weight",
    "indexer.k_proj.weight",
    "indexer.q_proj.weight",
    # RWKV time-mix operands. The full upstream set, not a prefix of it --
    # "time_mix_lerp_fused.weight" in particular is abort-class, not a
    # quality nicety: rwkv7-base.cpp:53 / rwkv6-base.cpp:70 feed it to
    # ggml_mul as src1, and it carries ne=(n_embd,1,1,6), so n_dims==4
    # dodges the 1-D rule while a 256-divisible ne[0] dodges the
    # block-size fallback -- exactly the hole ssm_norm.weight fell through.
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
    # Positional embeddings / token types: upstream matches these via
    # LLM_TN(arch)(LLM_TENSOR_POS_EMBD/TOKEN_TYPES, "weight"), which is
    # arch-templated per-architecture tensor naming. This writer has no
    # such per-arch table, so match on the GGUF canonical name instead.
    "position_embd.weight",
    "token_types.weight",
    # Vision / audio projector operands. Most of these are already forced
    # to F32 by the block-size fallback in practice (real .patch_embd rows
    # are 14/16 wide, .patch_merger 2), but "mm.a.code_embd.weight"
    # classifies into group E with a 32-divisible row and WOULD be
    # quantized here while upstream refuses it. Mirrored as a set so the
    # drift tripwire below keeps them current rather than rediscovering
    # them one incident at a time.
    ".position_embd",
    "sam.pos_embd",
    "sam.neck.",
    "sam.net_",
    ".rel_pos",
    ".patch_embd",
    ".patch_merger",
    "a.rvq.codebook",
    "mm.a.code_embd",
)


def _is_never_quantized(name: str) -> bool:
    """Does llama.cpp's own quantizer refuse to quantize ``name`` by name?"""
    return any(substr in name for substr in _NEVER_QUANTIZE_NAME_SUBSTRINGS)


# IEEE 754 half precision (F16): max finite magnitude. Values with |v| >
# this overflow to +/-Inf on conversion -- the hazard this check exists to
# catch (see the writer's own warning: "Out-of-F16-range values may become
# Inf/0").
_F16_MAX_FINITE = 65504.0


def _bf16_to_f16_would_corrupt(source, tensor_name: str) -> bool:
    """Would converting *tensor_name*'s real values from BF16 to F16
    silently produce Inf for any of them?

    Only called when a tensor is ACTUALLY about to be downgraded (so this
    never runs for tensors that stay at their requested scheme -- keeps
    Pass 1's "no data reading" fast path fast for everything else). Reads
    the tensor's real float values once via ``source.read_tensor_f32`` and
    does a single max-abs reduction over them -- cheap relative to the
    tensor read itself, which dominates.

    Conservative by construction: if the source can't decode this tensor at
    all (``read_tensor_f32`` returns ``None`` -- e.g. a quantized source
    with dequant not enabled), or it decodes to zero elements, this returns
    False (same as the historical unconditional-downgrade behavior) rather
    than blocking on data it doesn't have. Existing non-finite values in the
    SOURCE itself (already NaN/Inf before this conversion) also count as
    "would corrupt": true either way, this tensor writing through as
    plain F16 is not safe.

    Deliberately does NOT flag subnormal underflow (|v| < F16's smallest
    subnormal, ~5.96e-8): a prior version of this check did, and it fired
    on essentially every real weight tensor -- e.g. a 4096x4096 N(0, 0.02)
    tensor (max|v|~=0.11, nowhere near F16's 65504 ceiling) still has some
    element by chance landing below the subnormal floor, so EVERY BF16-
    designated group was silently substituted with Q8_0 across the board.
    Underflow of a ~1e-9 weight to zero is inconsequential; overflow to Inf
    (handled above) is the real hazard.
    """
    try:
        values = source.read_tensor_f32(tensor_name)
    except Exception:
        # A read failure here must not mask itself as "safe to downgrade" --
        # but it also must not crash Pass 1 (a real read failure surfaces
        # properly later, in the data pass that actually needs these bytes).
        return False
    if values is None or values.size == 0:
        return False

    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        # Non-finite in the SOURCE data already -- not something THIS
        # conversion introduces, but writing it through as F16 doesn't fix
        # it either.
        return True

    max_abs = float(np.max(np.abs(values)))
    if max_abs > _F16_MAX_FINITE:
        return True

    return False


def _read_encode_worker(source, entries, result_queue, imatrix=None):
    """
    Background thread: reads each tensor from source, encodes to ggml bytes,
    and pushes (entry, blob) onto the result queue.

    The bounded queue (maxsize=2) ensures at most 2 encoded tensors are
    buffered, preventing memory blowup on large models.

    imatrix: optional {tensor_name: importance_vector}; tensors with an entry
    are encoded imatrix-weighted, the rest unweighted.

    This is the historical single-worker path (MAGICQUANT_ENCODE_THREADS=1),
    kept byte-for-byte as it always was; see ``_parallel_encode_iter`` for
    the N>1 pooled path.
    """
    try:
        for entry in entries:
            blob = _encode_entry(source, entry, imatrix)
            result_queue.put((entry, blob))
    except Exception as exc:
        result_queue.put(exc)
    finally:
        result_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Parallel encode pool (N>1 worker threads)
# ---------------------------------------------------------------------------

# Fallback per-tensor byte estimate when there are no entries to average
# (degenerate empty-model case); arbitrary but harmless since capacity is
# re-derived from real entries whenever any exist.
_DEFAULT_TENSOR_BYTES = 4 * 1024 * 1024


def _resolve_encode_threads(env: Optional[Dict[str, str]] = None) -> int:
    """Number of encode worker threads: MAGICQUANT_ENCODE_THREADS overrides;
    default is min(8, os.cpu_count() // 2) (at least 1). ``1`` is the exact
    historical single-worker path, not just "a pool of size 1".
    """
    environ = os.environ if env is None else env
    raw = environ.get("MAGICQUANT_ENCODE_THREADS")
    if raw is not None:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n >= 1:
            return n
        # Non-positive / unparsable override: fall through to the default
        # rather than silently spawning zero workers.
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu // 2))


def _entry_footprint(entry: Dict[str, Any]) -> int:
    """Estimate the peak transient memory (bytes) one tensor occupies while
    in flight: the decoded float32 buffer (source.read_tensor_f32's return,
    the dominant allocation for anything but F32 passthrough) or the encoded
    blob, whichever is larger.
    """
    decode_bytes = entry["_n_elems"] * 4
    return max(decode_bytes, entry["_expected_size"])


def _resolve_encode_budget_bytes(
    tensor_entries: List[Dict[str, Any]], n_workers: int,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Total in-flight byte budget for the N-worker encode pool.

    ``MAGICQUANT_ENCODE_BUDGET_MB`` overrides directly (useful for tuning to
    the box's RAM headroom, or for deterministic small-budget tests). The
    default scales with both the model's actual tensor sizes (a model with
    170 MB dense tensors gets a bigger per-slot budget than one with 2 MB
    tensors) and the worker count (~2 tensors' worth of headroom per
    worker), and is never smaller than the single largest tensor's footprint
    so that tensor is never left unable to acquire its own budget alone.
    """
    environ = os.environ if env is None else env
    raw = environ.get("MAGICQUANT_ENCODE_BUDGET_MB")
    if raw:
        try:
            mb = float(raw)
            if mb > 0:
                return int(mb * 1024 * 1024)
        except ValueError:
            pass  # fall through to the size-derived default

    footprints = [_entry_footprint(e) for e in tensor_entries]
    if not footprints:
        return 2 * n_workers * _DEFAULT_TENSOR_BYTES
    avg_footprint = sum(footprints) / len(footprints)
    capacity = int(2 * n_workers * avg_footprint)
    return max(capacity, max(footprints))


class _ByteBudget:
    """Bounded byte budget gating how many decoded+encoded tensor buffers may
    be in flight at once.

    A plain bounded queue (the historical maxsize=2) caps tensor COUNT, not
    size — fine with one worker, but real models mix tiny 1D norms with
    100s-of-MB packed MoE expert tensors, so an N-worker, count-bounded queue
    could let N of the big ones decode simultaneously and blow memory on a
    box that shares RAM with the GPU. This caps total bytes instead.

    A single tensor larger than the whole budget is still admitted alone
    (acquire() only enforces the cap when something is already in flight) —
    the same guarantee the old one-tensor-at-a-time path gave for free, so a
    giant outlier tensor can never deadlock the pipeline.
    """

    def __init__(self, capacity_bytes: int):
        self._capacity = max(1, capacity_bytes)
        self._used = 0
        self._cv = threading.Condition()
        self.peak_used = 0  # instrumentation hook for tests

    def acquire(self, nbytes: int) -> None:
        with self._cv:
            while self._used > 0 and self._used + nbytes > self._capacity:
                self._cv.wait()
            self._used += nbytes
            self.peak_used = max(self.peak_used, self._used)

    def try_acquire(self, nbytes: int) -> bool:
        with self._cv:
            if self._used > 0 and self._used + nbytes > self._capacity:
                return False
            self._used += nbytes
            self.peak_used = max(self.peak_used, self._used)
            return True

    def release(self, nbytes: int) -> None:
        with self._cv:
            self._used -= nbytes
            self._cv.notify_all()


def _parallel_encode_iter(source, entries, imatrix, n_workers, budget):
    """Yield ``(entry, blob, footprint)`` for every entry in ``entries``, IN
    ORDER, encoding up to ``n_workers`` tensors concurrently via a thread
    pool, bounded by ``budget`` (a ``_ByteBudget``).

    Ordering is what makes this byte-identical to the single-worker path:
    tasks are submitted in ``entries`` order and results are consumed from a
    FIFO deque in that same order (``future.result()`` blocks on the oldest
    still-pending task, not on completion order), so the caller sees exactly
    the sequence it would from ``_read_encode_worker`` — just produced
    faster. The caller is responsible for calling ``budget.release(footprint)``
    once it is done with each yielded blob (i.e. after writing it), which is
    what lets ``_refill`` admit the next tensor.
    """
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=n_workers, thread_name_prefix="mq-encode",
    )
    # deque of (entry, footprint, future), oldest-submitted first.
    pending = collections.deque()
    n = len(entries)
    next_idx = 0

    def _submit(idx, blocking):
        entry = entries[idx]
        footprint = _entry_footprint(entry)
        if blocking:
            budget.acquire(footprint)
        elif not budget.try_acquire(footprint):
            return False
        fut = executor.submit(_encode_entry, source, entry, imatrix)
        pending.append((entry, footprint, fut))
        return True

    def _refill():
        nonlocal next_idx
        while next_idx < n and _submit(next_idx, blocking=False):
            next_idx += 1
        if not pending and next_idx < n:
            # Nothing in flight yet the next tensor's footprint alone won't
            # fit under try_acquire — must be a single oversized tensor.
            # Admit it unconditionally (acquire() never blocks when the
            # budget is empty) rather than deadlock.
            _submit(next_idx, blocking=True)
            next_idx += 1

    try:
        _refill()
        while pending:
            entry, footprint, fut = pending.popleft()
            # 600s per-tensor watchdog: the serial path aborts a hung encoder
            # rather than blocking forever; the pool must too. Timeout raises,
            # the finally-block cancels siblings, and the caller unlinks the
            # .partial -- no truncated file is ever published.
            try:
                blob = fut.result(timeout=600)  # propagates worker exceptions
            except concurrent.futures.TimeoutError:
                raise RuntimeError(
                    f"encode worker hung >600s on tensor {entry['name']}"
                )
            yield entry, blob, footprint
            # By now the caller has released this entry's footprint (it
            # writes the blob and releases immediately after each yield),
            # so there is room to keep the pipeline topped up.
            _refill()
    finally:
        # wait=True matters on the exception path: if fut.result() above
        # raised, sibling tasks already submitted (and possibly mid-encode,
        # reading from `source`) are still running. The caller's crash-
        # safety contract closes `source` as soon as this propagates, so we
        # must let every already-running task actually finish first --
        # otherwise a sibling worker thread could touch `source` after it's
        # closed. cancel_futures=True only drops tasks that never started.
        executor.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------------
# GGUFWriter
# ---------------------------------------------------------------------------

class GGUFWriter:
    """
    Write GGUF files with custom quantization configurations.

    Accepts any ModelSource (GGUF or safetensors) as input.
    Uses a pipelined architecture: a pool of background threads reads and
    encodes tensors (see MAGICQUANT_ENCODE_THREADS) while the main thread
    writes to disk in tensor order.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata: Dict[str, Any] = {}
        # One-time warning flag for the BF16 -> F16 on-disk downgrade.
        self._bf16_downgrade_warned = False
        # Provenance log for the block-32 fallback (see _block32_fallback):
        # one {"tensor", "group", "requested", "actual", "reason"} dict per
        # tensor whose requested K-quant was silently downgraded because its
        # row width wasn't block-divisible. Populated during Pass 1 of
        # create_hybrid_gguf; a summary is logged once the write completes.
        self._fallbacks: List[Dict[str, str]] = []

    def create_hybrid_gguf(
        self,
        base_model_path: str,
        quant_config: Dict,
        verbose: bool = True,
        adapter_path: Optional[str] = None,
        imatrix: Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        """
        Create a hybrid GGUF from any supported source format.

        Args:
            base_model_path: Path to source model — .gguf file, .safetensors
                file, or directory containing safetensors + config.json
            quant_config: {"base": "MXFP4_MOE", "groups": {"E": "BF16", ...},
                "tensors": {"blk.0.ffn_down.weight": "Q8_0", ...}}. "tensors"
                is optional: an exact-tensor-name -> scheme-name map that
                takes precedence over group/base resolution for exactly-
                matching tensor names. Resolution order is
                tensors[name] > groups[group] > base. Every scheme name in
                "tensors" must be known (ValueError otherwise, raised up
                front before any tensor is classified); a "tensors" entry
                whose name matches no tensor in the source is not an error
                (only a logged warning) since it doesn't affect the write,
                but a typo there would otherwise silently no-op.
            verbose: Print progress
            adapter_path: Optional path to a LoRA adapter directory.
            imatrix: Optional {gguf_tensor_name: importance_vector} from
                magicquant.imatrix.load_imatrix. Tensors with an entry are
                encoded imatrix-weighted (better quality, REQUIRED for the
                IQ1/IQ2 family); the rest encode unweighted.
        """
        from magicquant.gguf.source import open_model_source
        from magicquant.gguf.tensor_groups import TensorGroupClassifier

        scheme_map = SCHEME_TO_GGML

        if verbose:
            print(f"Loading source: {base_model_path}")
            if adapter_path:
                print(f"LoRA adapter: {adapter_path}")

        source = open_model_source(base_model_path, adapter_path=adapter_path)

        try:
            base_quant = quant_config.get("base", "Q4_K_M")
            group_schemes = quant_config.get("groups", {})
            tensor_overrides: Dict[str, str] = quant_config.get("tensors", {})

            self._validate_overrides(tensor_overrides, scheme_map)

            if verbose:
                print(f"Base quantization: {base_quant}")
                for grp, sch in group_schemes.items():
                    print(f"  Group {grp} -> {sch}")
                if tensor_overrides:
                    print(f"  Tensor overrides: {len(tensor_overrides)} tensor(s) configured")

            classifier = TensorGroupClassifier()
            source_metadata = source.get_metadata()
            all_tensors_info = source.get_all_tensors_info()

            self._warn_unmatched_overrides(tensor_overrides, all_tensors_info, verbose)
            self._prescan_unknown_tensors(all_tensors_info, classifier, verbose)

            tensor_entries = self._resolve_pass1_entries(
                all_tensors_info, tensor_overrides, group_schemes, base_quant,
                scheme_map, source, classifier, verbose,
            )

            self._validate_tensor_entries(tensor_entries, imatrix)

            filtered_meta = self._build_metadata(
                source_metadata, base_quant, group_schemes, tensor_entries,
            )

            return self._publish(
                filtered_meta, tensor_entries, source, verbose, imatrix,
            )

        finally:
            source.close()

    def _validate_overrides(self, tensor_overrides, scheme_map):
        """Raise if any quant_config['tensors'] scheme name is unknown.

        Extracted verbatim from create_hybrid_gguf's up-front override
        validation; see that method's docstring for the tensors/groups/base
        resolution order this guards.
        """
        # STRICT validation up front: every override scheme name must be
        # a key of the writer's scheme map. This is always a
        # configuration bug (unlike an override that names no tensor in
        # the source, handled below once tensor names are known) --
        # raise immediately, before any tensor is even classified.
        if tensor_overrides:
            unknown_scheme_entries = sorted(
                (name, sch) for name, sch in tensor_overrides.items()
                if sch not in scheme_map
            )
            if unknown_scheme_entries:
                listing = ", ".join(f"{n!r}: {s!r}" for n, s in unknown_scheme_entries)
                raise ValueError(
                    f"quant_config['tensors'] references unknown scheme "
                    f"name(s): {listing}. Available schemes: "
                    f"{sorted(scheme_map.keys())}"
                )

    def _warn_unmatched_overrides(self, tensor_overrides, all_tensors_info, verbose):
        """Warn (never raise) if a "tensors" override name matches no
        tensor in the source -- extracted verbatim from create_hybrid_gguf.
        """
        # A "tensors" override that matches NO tensor in the source is
        # never an error (the writer doesn't know the model's tensor
        # names until now) -- but a typo'd tensor name must not silently
        # no-op, so it's surfaced as a warning either way.
        if tensor_overrides:
            _source_tensor_names = {t["name"] for t in all_tensors_info}
            unmatched_overrides = [
                n for n in tensor_overrides if n not in _source_tensor_names
            ]
            if unmatched_overrides:
                shown = unmatched_overrides[:5]
                more = (f" (+{len(unmatched_overrides) - 5} more)"
                        if len(unmatched_overrides) > 5 else "")
                msg = (f"{len(unmatched_overrides)} tensor override name(s) in "
                       f"quant_config['tensors'] matched no tensor in the "
                       f"source (possible typo): {shown}{more}")
                logger.warning(msg)
                if verbose:
                    print(f"  WARNING: {msg}")

    def _prescan_unknown_tensors(self, all_tensors_info, classifier, verbose):
        """Cosmetic pre-scan: warn (verbose only) about tensors with no
        group classification. Extracted verbatim from create_hybrid_gguf;
        distinct from the hard-error undecodable-source-type gate in
        _validate_tensor_entries -- do not conflate the two.
        """
        # Pre-scan for UNKNOWN tensors so the user sees issues upfront
        unknown_tensors = [t["name"] for t in all_tensors_info
                           if classifier.classify_tensor(t["name"]) == "UNKNOWN"]
        if unknown_tensors and verbose:
            print(f"  WARNING: {len(unknown_tensors)} tensor(s) have no group classification "
                  f"(will use base quant): {unknown_tensors[:5]}"
                  + (f" ... and {len(unknown_tensors)-5} more" if len(unknown_tensors) > 5 else ""))

    def _resolve_pass1_entries(self, all_tensors_info, tensor_overrides, group_schemes, base_quant, scheme_map, source, classifier, verbose):
        """Pass 1: compute target ggml types and offsets for every tensor,
        no data reading. Extracted verbatim from create_hybrid_gguf's Pass-1
        loop -- see the module docstring and the inline [COMPAT] comments
        for the six-stage type-mutation pipeline (SSM-F32, never-quantize
        -name, 1D-F32, BF16->F16/Q8_0, block-32 fallback, can_decode
        passthrough) this loop runs in strict order for every tensor.
        """
        tensor_entries: List[Dict[str, Any]] = []
        data_offset = 0
        # Reset per-call so re-using a writer instance for a second
        # create_hybrid_gguf() call doesn't carry stale entries forward.
        self._fallbacks = []
        n_tensor_overrides_applied = 0

        for tinfo in all_tensors_info:
            name = tinfo["name"]
            shape = tinfo["shape"]
            n_dims = tinfo["n_dims"]

            group = classifier.classify_tensor(name)
            # Resolution order: tensors[name] > groups[group] > base.
            # Everything downstream (SSM F32 force, 1D F32, BF16->F16,
            # block-size fallback) treats this exactly like a
            # group-resolved scheme -- it only ever sees the resolved
            # name, never how it was resolved.
            if name in tensor_overrides:
                scheme = tensor_overrides[name]
                n_tensor_overrides_applied += 1
            else:
                scheme = group_schemes.get(group, base_quant)
            target_ggml_name = scheme_map.get(scheme, "Q4_0")
            target_ggml_id = GGML_TYPE.get(target_ggml_name, GGML_TYPE["Q4_0"])
            # Captured BEFORE the compat-mutation chain below (SSM-F32
            # force, never-quantize-name, 1D-F32, BF16->F16/Q8_0,
            # block-32 fallback, the can_decode passthrough overwrite) so
            # the bad_tensors gate further down can read the tensor's
            # PRE-compat desired type directly off the entry instead of
            # re-deriving scheme resolution from tensor_overrides/
            # group_schemes/base_quant a second time. Do not move this
            # past any of the mutations below -- see the bad_tensors
            # gate's comment for why the post-compat value would
            # silently disable that check.
            desired_ggml_name = target_ggml_name

            # SSM conv-weight operands (ssm_conv1d / ssm_conv1d_{q,k,v})
            # must be F32 no matter what scheme group S was configured
            # with -- see _is_f32_required_ssm_operand. This has to run
            # BEFORE the block-size fallback below: a float scheme (BF16/
            # F16) has block_size==1, so that check is skipped entirely
            # and would otherwise let a BF16-designated conv weight
            # through untouched (the real bug this guards against).
            if group == "S" and target_ggml_name != "F32" and _is_f32_required_ssm_operand(name):
                if verbose:
                    print(f"  [COMPAT] {name}: SSM conv operand requires F32 "
                          f"(llama.cpp kernel constraint), overriding {target_ggml_name}")
                self._fallbacks.append({
                    "tensor": name,
                    "group": group,
                    "requested": target_ggml_name,
                    "actual": "F32",
                    "reason": "f32-required-operand",
                })
                target_ggml_name = "F32"
                target_ggml_id = GGML_TYPE["F32"]

            # Never-quantize-by-name tensors (see
            # _NEVER_QUANTIZE_NAME_SUBSTRINGS): mirrors llama.cpp's own
            # quantizer so a tensor upstream refuses to quantize doesn't
            # slip through here just because its shape happens to pass
            # every shape-based check (1D-F32, block-size). Guarded on
            # target_ggml_name != "F32" so this never overwrites (or
            # re-logs under a different reason) a tensor the SSM check
            # above already forced to F32. Must run BEFORE the
            # block-size fallback below for the same reason as that
            # check: a float target has block_size==1 and skips it
            # entirely, which would let a quantizable-looking BF16/F16
            # scheme through untouched.
            # Not a "weight" tensor at all -> llama.cpp never quantizes it and
            # copies it at source dtype (F32). Must run BEFORE the block-size
            # fallback for the same reason as the SSM check above: a float
            # target has block_size == 1 and skips that check entirely, which
            # is exactly how ssm_a/ssm_d reached llama.cpp as F16 and aborted
            # it. See _is_quantizable_by_name for the measured blast radius.
            if target_ggml_name != "F32" and not _is_quantizable_by_name(name):
                if verbose:
                    print(f"  [COMPAT] {name}: not a 'weight' tensor; "
                          f"llama.cpp never quantizes these, keeping at F32")
                if n_dims >= 2:
                    self._fallbacks.append({
                        "tensor": name,
                        "group": group,
                        "requested": target_ggml_name,
                        "actual": "F32",
                        "reason": "not-a-weight-tensor",
                    })
                target_ggml_name = "F32"
                target_ggml_id = GGML_TYPE["F32"]

            if target_ggml_name != "F32" and _is_never_quantized(name):
                if verbose:
                    print(f"  [COMPAT] {name}: llama.cpp never quantizes this "
                          f"tensor by name, keeping at F32")
                # Record only 2-D+ matches. A 1-D match (every ordinary
                # norm/bias, and group N falls to base_quant on essentially
                # every generated config) would land at F32 one rule below
                # anyway, so logging it here says nothing an operator can
                # act on -- but it does crowd out the 2-D cases inside this
                # bucket. The summary line prints one example per reason, so
                # 17 trivial norms ahead of a real ssm_norm.weight means the
                # build advertises "output_norm.weight" and buries the tensor
                # this rule exists to catch. The F32 override itself stays
                # unconditional; only the telemetry is filtered.
                if n_dims >= 2:
                    self._fallbacks.append({
                        "tensor": name,
                        "group": group,
                        "requested": target_ggml_name,
                        "actual": "F32",
                        "reason": "never-quantize-name",
                    })
                target_ggml_name = "F32"
                target_ggml_id = GGML_TYPE["F32"]

            # 1D tensors (norms, biases) must stay at F32.  llama.cpp
            # uses f32 binary ops (e.g. element-wise mul in RMSNorm) and
            # does not support quantised or BF16 operands.  These tensors
            # are tiny so keeping them at F32 has negligible size impact.
            if n_dims <= 1 and target_ggml_name != "F32":
                if verbose:
                    print(f"  [COMPAT] {name}: 1D tensor (norm/bias), keeping at F32")
                target_ggml_name = "F32"
                target_ggml_id = GGML_TYPE["F32"]

            # BF16 → F16 conversion: llama.cpp has incomplete BF16
            # support in its compute graph (binary ops, some matmuls
            # assert sizeof(float) stride).  F16 is universally supported.
            # This is a deliberate compatibility tradeoff.
            #
            # CHOSEN BEHAVIOR (2026-07 incident fix -- see probing.py's
            # clamp guard and llamacpp.py's parser fix from the same
            # investigation): this used to warn ONCE and proceed
            # unconditionally, even though BF16 has ~8 more exponent
            # bits than F16 -- any source value with |v| > 65504 (F16's
            # max finite) or in F16's subnormal-underflow range becomes
            # Inf/0 on disk with nothing downstream reacting. Silently
            # Inf/0-poisoned tensors (typically embeddings/lm_head,
            # since those are the groups most often left at BF16) then
            # measure as NaN perplexity -- which is exactly the
            # degenerate input the parser/clamp fixes exist to catch,
            # but catching it downstream is strictly worse than not
            # producing it here.
            #
            # Detection is a single max-abs reduction over the
            # tensor's REAL values -- read
            # via source.read_tensor_f32 only when a downgrade is
            # actually happening (every other tensor in this header-only
            # pass never pays this cost; see the "Pass 1: ... (no data
            # reading)" comment above this loop).
            #
            # On detection: SUBSTITUTE Q8_0 for this one tensor rather
            # than raising. A raise would abort an otherwise-fine
            # multi-hour hybrid build over a single outlier tensor
            # (embeddings in particular routinely have some large
            # magnitude rows); Q8_0 keeps real dynamic range for exactly
            # this tensor at ~1/4 the size of F32/BF16, instead of
            # writing Inf/0 for it. If a future caller needs "abort
            # instead", it's a one-line change here.
            if target_ggml_name == "BF16":
                out_of_range = _bf16_to_f16_would_corrupt(source, name)

                if out_of_range:
                    logger.warning(
                        "%s: BF16->F16 downgrade would produce Inf/0 "
                        "(value(s) outside F16's finite/normal range) -- "
                        "substituting Q8_0 for this tensor instead of "
                        "writing corrupted data",
                        name,
                    )
                    self._fallbacks.append({
                        "tensor": name,
                        "group": group,
                        "requested": "BF16",
                        "actual": "Q8_0",
                        "reason": "bf16-f16-out-of-range",
                    })
                    target_ggml_name = "Q8_0"
                    target_ggml_id = GGML_TYPE["Q8_0"]
                else:
                    if not self._bf16_downgrade_warned:
                        logger.warning(
                            "BF16-designated group(s) written as F16 on disk "
                            "(llama.cpp BF16 compute-graph limitation). Out-of-F16-"
                            "range values may become Inf/0."
                        )
                        self._bf16_downgrade_warned = True
                    target_ggml_name = "F16"
                    target_ggml_id = GGML_TYPE["F16"]

            # Block-size compatibility check: quantized types require the
            # contiguous row dimension (ne[0] in GGUF) to be a multiple of
            # the block size.  The writer stores shapes in row-major order
            # and reverses when writing, so ne[0] = shape[-1].  K-quants use
            # a 256-block; rows that don't fit fall back to a block-32 quant
            # (MXFP4/Q8_0) rather than F32 — F32 is lossless but enormous for
            # big tensors like MoE experts (it once turned a ~14 GB pack into
            # 39 GB).  F32 is kept only where required (SSM conv operands) or
            # for rows that aren't 32-divisible either.
            row_size = shape[-1] if len(shape) >= 1 else 1
            block_size = GGML_BLOCK_SIZE.get(target_ggml_name, 1)
            if block_size > 1 and row_size % block_size != 0:
                requested_ggml_name = target_ggml_name
                fallback = _block32_fallback(target_ggml_name, row_size, group)
                if fallback != requested_ggml_name:
                    # Record the deviation so it's auditable even when
                    # verbose=False (data-integrity notice, not a
                    # progress message) — see the summary log below.
                    self._fallbacks.append({
                        "tensor": name,
                        "group": group,
                        "requested": requested_ggml_name,
                        "actual": fallback,
                        "reason": "block-size",
                    })
                if verbose:
                    print(f"  [COMPAT] {name}: row_size={row_size} not divisible by "
                          f"{target_ggml_name} block_size={block_size}, "
                          f"falling back to {fallback}")
                target_ggml_name = fallback
                target_ggml_id = GGML_TYPE[fallback]

            n_elems = _tensor_n_elements(shape)

            source_type_name = source.get_source_type_name(name)
            # Ask the source rather than testing the type name here: a
            # GGUFSource opened with dequant enabled can also decode
            # quantized types via libggml's dequantize_row_* kernels.
            can_decode = source.can_decode(name)
            if (
                can_decode
                and source_type_name not in ("F32", "F16", "BF16")
                and scheme_map.get(scheme, "Q4_0") == source_type_name
                and target_ggml_name == source_type_name
            ):
                # Dequant-enabled source, but the requested type IS the
                # source type: a verbatim byte copy is exact and free,
                # while a dequant->re-encode round-trip costs CPU and can
                # at best equal it. Route through the passthrough path
                # (this also keeps the DOUBLE-QUANTIZATION warning's
                # count honest -- such tensors aren't re-quantized).
                # Both equality checks matter: the scheme_map one mirrors
                # the bad_tensors gate's own derivation (so passthrough
                # never trips it), and the target one keeps the compat
                # mutations above (SSM/1D F32 forcing, block fallback)
                # authoritative -- a compat-mutated tensor still decodes
                # and re-encodes to its required type.
                can_decode = False
            if not can_decode:
                # The source tensor is not decodable to F32 (pre-quantized
                # or an unrecognized type). We pass it through verbatim, so
                # the target ggml type IS the source type. UNKNOWN source
                # types are a hard error (caught in the bad_tensors pass
                # below); never silently default to F32 (id 0), which would
                # produce a zero-filled blob masquerading as valid F32.
                target_ggml_name = source_type_name
                if source_type_name in GGML_TYPE:
                    target_ggml_id = GGML_TYPE[source_type_name]
                else:
                    # Unknown / undecodable source type — flag with a
                    # sentinel id; the bad_tensors pass raises before any
                    # data is written.
                    target_ggml_id = -1

            if target_ggml_id == -1:
                # No block/type-size fact exists for an unrecognized type, and
                # ggml_tensor_data_size now raises rather than guessing one
                # (guessing silently corrupts blocked types -- see
                # ggml_facts.expected_size). Size this entry at 0: the
                # bad_tensors pass a few lines below raises a ValueError
                # naming the tensor and its UNKNOWN type before any offset is
                # used or any byte is written, so the value is never read.
                # Computing it here would raise a KeyError first and replace
                # that specific, actionable error with an opaque one.
                expected_size = 0
            else:
                expected_size = ggml_tensor_data_size(target_ggml_name, n_elems)
            aligned_offset = _align(data_offset)

            tensor_entries.append({
                "name": name,
                "n_dims": n_dims,
                "shape": shape,
                "ggml_type": target_ggml_id,
                "offset": aligned_offset,
                "_target_ggml_name": target_ggml_name,
                "_n_elems": n_elems,
                "_expected_size": expected_size,
                "_can_decode": can_decode,
                "_group": group,
                "_source_type_name": source_type_name,
                "_desired_ggml_name": desired_ggml_name,
            })

            data_offset = aligned_offset + expected_size

        # Pass-1's tensor loop drives classify_tensor() name-by-name
        # (not classify_tensors(), which would fire this automatically)
        # -- so this instance's unclassified-tensor summary warning must
        # be triggered explicitly once the pass completes. See
        # TensorGroupClassifier.warn_unclassified_once()'s docstring.
        classifier.warn_unclassified_once()

        if tensor_overrides and verbose:
            print(f"  Tensor overrides applied: "
                  f"{n_tensor_overrides_applied}/{len(tensor_overrides)}")
        return tensor_entries

    def _validate_tensor_entries(self, tensor_entries, imatrix):
        """Post-hoc validation over Pass-1's tensor_entries: the
        undecodable/pre-quantized-source gate, the double-quantization
        warning, and the imatrix-required gate. Extracted verbatim from
        create_hybrid_gguf.
        """
        # ── Validate: detect pre-quantized / undecodable sources ──
        # Two distinct failure modes:
        #   1. UNKNOWN source type — the source could not even identify the
        #      tensor's format. This is ALWAYS a hard error (it would
        #      otherwise produce a zero-filled blob with a bogus type id).
        #   2. A recognized pre-quantized type (e.g. Q4_K) that the user
        #      asked to re-quantize to a different scheme — also an error,
        #      since MagicQuant requires high-precision source weights.
        bad_tensors = []
        undecodable_source_tensors = []
        for entry in tensor_entries:
            if not entry["_can_decode"]:
                source_type = entry["_source_type_name"]
                if source_type.startswith("UNKNOWN"):
                    undecodable_source_tensors.append((entry["name"], source_type))
                    continue
                # Recognized but pre-quantized: error only if the user
                # wanted a different type than the source already is.
                # Reads the PRE-compat desired type captured in Pass 1
                # (entry["_desired_ggml_name"], set before the
                # SSM-F32/never-quantize-name/1D-F32/BF16->F16/block-32/
                # can_decode compat mutations) rather than re-deriving
                # scheme resolution from tensor_overrides/group_schemes/
                # base_quant a second time -- so an override can't get
                # silently ignored by two independent copies of "what
                # scheme did the user actually ask for" drifting apart.
                # The entry's POST-compat "_target_ggml_name" is unusable
                # here: on the not-can_decode path it has already been overwritten
                # to the source type, which would make this comparison
                # always false and disable the gate silently.
                desired_ggml_name = entry["_desired_ggml_name"]
                if desired_ggml_name != source_type:
                    bad_tensors.append((entry["name"], source_type, desired_ggml_name))

        if undecodable_source_tensors:
            count = len(undecodable_source_tensors)
            first_name, first_type = undecodable_source_tensors[0]
            raise ValueError(
                f"Cannot encode {count} tensor(s) with an UNKNOWN/undecodable "
                f"source type. First: '{first_name}' (source type '{first_type}'). "
                f"The source model has tensors whose ggml type could not be "
                f"identified or decoded to F32. MagicQuant requires BF16, F16, "
                f"or F32 source weights."
            )

        if bad_tensors:
            count = len(bad_tensors)
            source_type = bad_tensors[0][1]
            raise ValueError(
                f"Cannot re-quantize {count} tensors: source is already quantized "
                f"({source_type}). MagicQuant requires BF16, F16, or F32 source weights. "
                f"Use a high-precision source model. If no high-precision "
                f"release exists for this model, you can opt into "
                f"dequantize-then-requantize by setting "
                f"MAGICQUANT_ALLOW_DEQUANT_SOURCE=1 (or passing "
                f"allow_dequant=True to open_model_source) -- the output is "
                f"then DOUBLE-quantized and strictly worse than one built "
                f"from source weights."
            )

        # ── Warn: re-quantizing from an already-quantized source ──
        # Reachable only with dequant explicitly enabled (otherwise the
        # bad_tensors guard above raised). Loud and unconditional: this
        # output is double-quantized, and nothing downstream -- model card,
        # search result, filename -- would otherwise say so.
        dequant_sources: Dict[str, int] = {}
        for entry in tensor_entries:
            st = entry["_source_type_name"]
            if entry["_can_decode"] and st not in ("F32", "F16", "BF16"):
                dequant_sources[st] = dequant_sources.get(st, 0) + 1
        if dequant_sources:
            breakdown = ", ".join(
                f"{n}x {t}" for t, n in sorted(dequant_sources.items())
            )
            logger.warning(
                "DOUBLE-QUANTIZATION: %d tensors are being dequantized from "
                "an already-quantized source and re-quantized (%s). Quality "
                "is bounded by the source quant's error floor -- a tier at or "
                "above the source precision cannot improve on it. Only use "
                "this when no BF16/F16/F32 release of the model exists, and "
                "say so on the model card.",
                sum(dequant_sources.values()), breakdown,
            )

        # ── Gate: imatrix-REQUIRING types must have an imatrix entry ──
        # IQ1/IQ2-family quantizers produce unusable output without an
        # importance matrix; fail fast in Pass 1 (before any bytes are
        # written) instead of silently shipping garbage. Only consulted
        # for quantized targets so float-only writes never load libggml.
        _FLOAT_TARGETS = ("F32", "F16", "BF16")
        missing_imatrix = [
            entry["name"] for entry in tensor_entries
            if entry["_can_decode"]
            and entry["_target_ggml_name"] not in _FLOAT_TARGETS
            and _requires_imatrix(entry["_target_ggml_name"])
            and (imatrix is None or entry["name"] not in imatrix)
        ]
        if missing_imatrix:
            first = ", ".join(missing_imatrix[:3])
            more = (f" (+{len(missing_imatrix) - 3} more)"
                    if len(missing_imatrix) > 3 else "")
            raise ValueError(
                f"{len(missing_imatrix)} tensor(s) target an imatrix-"
                f"REQUIRING quantization type but no imatrix entry was "
                f"provided: {first}{more}. Capture one with "
                f"magicquant.imatrix.capture_imatrix (or llama-imatrix) "
                f"and pass imatrix=load_imatrix(path), or choose a type "
                f"that does not require an importance matrix."
            )

    def _build_metadata(self, source_metadata, base_quant, group_schemes, tensor_entries):
        """Build self.metadata (WIPES any caller-set metadata -- existing
        behavior) and return the filtered dict ready for
        _write_gguf_body. Extracted verbatim from create_hybrid_gguf; KV
        insertion order here is output-byte load-bearing (_write_gguf_body
        writes filtered_meta.items() in dict order) -- do not reorder.
        """
        self.metadata = {}
        for k, v in source_metadata.items():
            # split.* excluded regardless of type: see _SPLIT_KV_KEYS --
            # per llama.cpp's gguf-split convention these keys don't belong
            # in a single-file artifact at all, independent of whether their
            # type round-trips. (GGUFReader now DOES retain each KV's
            # on-disk GGUF type -- GGUFTypedInt/GGUFTypedArray -- and
            # _write_metadata_value uses it, so every other copied key
            # round-trips at its source element type.)
            if k in _SPLIT_KV_KEYS:
                continue
            self.metadata[k] = v
        self.metadata["magicquant.hybrid"] = True
        self.metadata["magicquant.base_quant"] = base_quant
        self.metadata["magicquant.group_schemes"] = json.dumps(group_schemes)

        # Set general.file_type for llama.cpp compatibility. Determine the
        # dominant scheme by counting actual parameter elements per scheme
        # across all tensors (after Pass 1) and look it up in the
        # module-level _ftype_map (see that table's comment for the
        # enum-derivation rationale and the historical drift incident it
        # closes).
        from collections import Counter
        # Count elements per actual target ggml type from tensor_entries
        scheme_elements = Counter()
        for entry in tensor_entries:
            scheme_elements[entry["_target_ggml_name"]] += entry["_n_elems"]
        # Pick the scheme with the most parameters, preferring quantized
        # types over uncompressed (F16/F32/BF16) for display purposes
        quantized_types = {s for s in scheme_elements if s in _ftype_map and s not in ("F16", "F32", "BF16")}
        if quantized_types:
            dominant = max(quantized_types, key=lambda s: scheme_elements[s])
        elif scheme_elements:
            dominant = scheme_elements.most_common(1)[0][0]
        else:
            dominant = base_quant
        ftype = _ftype_map.get(dominant, _ftype_map.get(base_quant, 1))
        self.metadata["general.file_type"] = ftype

        # llama.cpp/convert_hf_to_gguf always emits this for a quantized
        # GGUF; the safetensors source path never did (its metadata
        # comes straight from config.json, which has no such key). Not
        # load-bearing for inference, but every reference-converted GGUF
        # carries it -- setdefault so a GGUF-source repack keeps its own
        # value if it already has one.
        self.metadata.setdefault("general.quantization_version", 2)

        filtered_meta = {k: v for k, v in self.metadata.items() if v is not None}
        return filtered_meta

    def _publish(self, filtered_meta, tensor_entries, source, verbose, imatrix=None):
        """Write the header+body via _write_gguf_body under crash-safe
        .partial/os.replace semantics, then log the grouped fallback
        summary. Extracted verbatim from create_hybrid_gguf.
        """
        # ==============================================================
        # Write header
        # ==============================================================
        if verbose:
            print(f"\nWriting output: {self.output_path}")
            print(f"Tensors: {len(tensor_entries)}")

        Path(self.output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

        t_start = time.monotonic()

        # Crash-safety: write to a sibling temp file and atomically rename
        # only after a fully-successful write. A worker exception (dtype
        # guard, size mismatch, OOM) or a hung encoder thread must leave NO
        # file at output_path and NO stray .partial behind.
        tmp_path = self.output_path + ".partial"
        try:
            self._write_gguf_body(
                tmp_path, filtered_meta, tensor_entries, source,
                t_start, verbose, imatrix=imatrix,
            )
        except BaseException:
            # Remove the partially-written temp file before propagating.
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise

        # Atomic publish (same directory -> os.replace is atomic).
        os.replace(tmp_path, self.output_path)

        elapsed = time.monotonic() - t_start
        output_size_mb = Path(self.output_path).stat().st_size / (1024 * 1024)
        if verbose:
            print(f"Done. {output_size_mb:.1f} MB in {elapsed:.1f}s "
                  f"({output_size_mb / max(elapsed, 0.001):.0f} MB/s)")

        # Data-integrity notice: surface fallbacks even when
        # verbose=False. self._fallbacks now carries more than one
        # reason (block-size, f32-required-operand, never-quantize-name,
        # and bf16-f16-out-of-range) -- a single line hardcoding
        # "due to block-size" misattributed every non-block-size
        # fallback observed live. Group by reason and summarize each
        # group with one example, instead of blaming them all on the
        # same cause. Per-tensor detail always lives in self._fallbacks.
        if self._fallbacks:
            by_reason: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
            for entry in self._fallbacks:
                by_reason[entry.get("reason", "unknown")].append(entry)
            for reason, entries in by_reason.items():
                first = entries[0]
                logger.warning(
                    "%d tensor(s) fell back from their requested quant "
                    "due to %s (e.g. %s: %s->%s)",
                    len(entries), reason, first["tensor"],
                    first["requested"], first["actual"],
                )

        return self.output_path

    def _write_gguf_body(
        self, tmp_path, filtered_meta, tensor_entries, source, t_start, verbose,
        imatrix=None,
    ) -> None:
        """Write the full GGUF (header + pipelined data) to ``tmp_path``.

        Raises on any worker error or a hung encoder thread; the caller is
        responsible for unlinking ``tmp_path`` on failure and renaming it into
        place on success.
        """
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<I", 0x46554747))  # magic
            f.write(struct.pack("<I", 3))            # version
            f.write(struct.pack("<Q", len(tensor_entries)))
            f.write(struct.pack("<Q", len(filtered_meta)))

            for key, value in filtered_meta.items():
                _write_string(f, key)
                _write_metadata_value(f, value)

            for entry in tensor_entries:
                _write_string(f, entry["name"])
                f.write(struct.pack("<I", entry["n_dims"]))
                for dim in reversed(entry["shape"]):
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", entry["ggml_type"]))
                f.write(struct.pack("<Q", entry["offset"]))

            header_end = f.tell()
            aligned_header = _align(header_end)
            if aligned_header > header_end:
                f.write(b"\x00" * (aligned_header - header_end))

            # ==========================================================
            # Pass 2: Pipelined read+encode -> write
            # ==========================================================
            data_section_start = f.tell()
            total = len(tensor_entries)
            bytes_written = 0

            def _write_entry_blob(idx: int, entry: Dict[str, Any], blob: bytes) -> None:
                nonlocal bytes_written
                aligned_offset = entry["offset"]

                # Write alignment padding
                current_pos = f.tell() - data_section_start
                padding = aligned_offset - current_pos
                if padding < 0:
                    raise RuntimeError(
                        f"Tensor {entry['name']}: file position {current_pos} "
                        f"exceeds expected offset {aligned_offset} by {-padding} bytes. "
                        f"GGUF is corrupt."
                    )
                if padding > 0:
                    f.write(b"\x00" * padding)

                f.write(blob)
                bytes_written += len(blob)

                if verbose:
                    elapsed = time.monotonic() - t_start
                    speed = bytes_written / (1024**2) / max(elapsed, 0.001)
                    eta = (elapsed / idx) * (total - idx) if idx > 0 else 0
                    print(
                        f"  [{idx}/{total}] {entry['name']}: "
                        f"{entry['_source_type_name']} -> {entry['_target_ggml_name']} "
                        f"({entry['_group']})  "
                        f"{speed:.0f} MB/s  ETA {eta:.0f}s",
                    )

            n_workers = _resolve_encode_threads()

            if n_workers <= 1:
                # ------------------------------------------------------
                # Exact historical single-worker path (also what
                # MAGICQUANT_ENCODE_THREADS=1 reproduces).
                # ------------------------------------------------------
                result_q: queue.Queue = queue.Queue(maxsize=2)
                worker = threading.Thread(
                    target=_read_encode_worker,
                    args=(source, tensor_entries, result_q, imatrix),
                    daemon=True,
                )
                worker.start()

                idx = 0
                while True:
                    item = result_q.get()

                    # Check for sentinel (done) or exception
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        # Drain queue so worker thread can finish
                        while True:
                            try:
                                result_q.get_nowait()
                            except queue.Empty:
                                break
                        worker.join(timeout=5)
                        raise item

                    entry, blob = item
                    idx += 1
                    _write_entry_blob(idx, entry, blob)

                # Wait for the worker to finish. If it didn't (hung encode),
                # raise so a truncated file is never renamed into place.
                worker.join(timeout=30)
                if worker.is_alive():
                    raise RuntimeError(
                        "Encoder worker thread did not finish within 30s; "
                        "aborting to avoid writing a truncated GGUF."
                    )
            else:
                # ------------------------------------------------------
                # Pooled N-worker path: entries still arrive (and get
                # written) in Pass-1 order, so output is byte-identical
                # to the path above -- see _parallel_encode_iter.
                # ------------------------------------------------------
                capacity_bytes = _resolve_encode_budget_bytes(tensor_entries, n_workers)
                budget = _ByteBudget(capacity_bytes)

                idx = 0
                for entry, blob, footprint in _parallel_encode_iter(
                    source, tensor_entries, imatrix, n_workers, budget,
                ):
                    idx += 1
                    _write_entry_blob(idx, entry, blob)
                    # Only now is this tensor's memory truly no longer
                    # needed -- release lets _parallel_encode_iter admit
                    # the next one under the byte cap.
                    budget.release(footprint)


def create_hybrid_gguf(
    output_path: str, base_model_path: str,
    quant_config: Dict, verbose: bool = True,
    adapter_path: Optional[str] = None,
    imatrix=None,
) -> str:
    """Convenience function to create a hybrid GGUF model.

    imatrix may be a ``{tensor_name: importance_vector}`` dict (from
    ``magicquant.imatrix.load_imatrix``) or a path to an imatrix GGUF
    captured by llama-imatrix, which is loaded here.
    """
    if isinstance(imatrix, (str, os.PathLike)):
        from magicquant.imatrix import load_imatrix
        imatrix = load_imatrix(imatrix)
    writer = GGUFWriter(output_path)
    return writer.create_hybrid_gguf(
        base_model_path, quant_config, verbose,
        adapter_path=adapter_path, imatrix=imatrix,
    )


if __name__ == "__main__":
    import sys
    import json as _json

    if len(sys.argv) < 4:
        print("Usage: python -m magicquant.gguf.writer <output.gguf> <source> <config.json>")
        print("  source: .gguf file, .safetensors file, or HF model directory")
        sys.exit(1)

    output_path = sys.argv[1]
    base_model_path = sys.argv[2]

    if os.path.exists(sys.argv[3]):
        with open(sys.argv[3]) as _f:
            quant_config = _json.load(_f)
    else:
        parts = sys.argv[3].split(",")
        groups = {}
        for part in parts:
            if ":" in part:
                group, scheme = part.split(":")
                groups[group] = scheme
        quant_config = {"base": "Q4_K_M", "groups": groups}

    result = create_hybrid_gguf(
        output_path=output_path,
        base_model_path=base_model_path,
        quant_config=quant_config,
        verbose=True,
    )
    print(f"\nCreated: {result}")
