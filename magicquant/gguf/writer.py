"""
GGUF Writer - Create hybrid quantization GGUF files.

This module creates new GGUF files with mixed-precision quantization,
combining different quant schemes for different tensor groups.

Accepts both GGUF and safetensors as source formats via the ModelSource
abstraction in ``magicquant.gguf.source``.

Architecture:
  Pass 1 (header): Compute target types, data sizes, and offsets for every
      tensor without touching actual data. Write the complete GGUF header.
  Pass 2 (data): A background thread reads + encodes tensors while the main
      thread writes blobs to disk. This overlaps I/O with computation for
      ~2x throughput on large models.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import struct
import os
import time
import threading
import queue
import logging
import numpy as np

logger = logging.getLogger(__name__)

from magicquant.quant.converters import (
    encode_to_ggml_bytes,
    ggml_tensor_data_size,
    GGML_BLOCK_SIZE,
)
from magicquant.quant.schemes import get_all_schemes

# ggml_type_name -> bits_per_weight, derived from the scheme registry so the
# block-32 fallback's low/high-bit split (see _block32_fallback) can never
# drift from the registry the way the old hand-maintained tuple did.
_GGML_NAME_TO_BPW: Dict[str, float] = {
    s.ggml_type_name: s.bits_per_weight for s in get_all_schemes()
}


# ggml_type enum values used in GGUF tensor info
GGML_TYPE = {
    "F32":      0,
    "F16":      1,
    "Q4_0":     2,
    "Q4_1":     3,
    "Q5_0":     6,
    "Q5_1":     7,
    "Q8_0":     8,
    "Q8_1":     9,
    "Q2_K":    10,
    "Q3_K":    11,
    "Q4_K":    12,
    "Q5_K":    13,
    "Q6_K":    14,
    "Q8_K":    15,
    "IQ2_XXS": 16,
    "IQ2_XS":  17,
    "IQ3_XXS": 18,
    "IQ1_S":   19,
    "IQ4_NL":  20,
    "IQ3_S":   21,
    "IQ2_S":   22,
    "IQ4_XS":  23,
    "I8":      24,
    "I16":     25,
    "I32":     26,
    "I64":     27,
    "F64":     28,
    "IQ1_M":   29,
    "BF16":    30,
    "MXFP4":   39,  # GGML_TYPE_MXFP4 (native llama.cpp support)
    # ROCmFPX fork types — only loadable by the ROCmFPX llama.cpp fork.
    "Q4_0_ROCMFP4":      100,
    "Q4_0_ROCMFP4_FAST": 101,
    "Q6_0_ROCMFPX":      102,
    "Q8_0_ROCMFPX":      103,
    "Q3_0_ROCMFPX":      104,
}

_GGML_TYPE_NAME = {v: k for k, v in GGML_TYPE.items()}

# GGUF metadata value-type tags
_GGUF_TYPE_UINT8   = 0
_GGUF_TYPE_INT8    = 1
_GGUF_TYPE_UINT16  = 2
_GGUF_TYPE_INT16   = 3
_GGUF_TYPE_UINT32  = 4
_GGUF_TYPE_INT32   = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL    = 7
_GGUF_TYPE_STRING  = 8
_GGUF_TYPE_ARRAY   = 9
_GGUF_TYPE_UINT64  = 10
_GGUF_TYPE_INT64   = 11
_GGUF_TYPE_FLOAT64 = 12

ALIGNMENT = 32

# Map MagicQuant scheme names to the ggml_type name we write into the file.
# Built from the canonical scheme registry; F16/F32 added as passthrough
# entries for source tensors that bypass quantization.
SCHEME_TO_GGML: Dict[str, str] = {s.name: s.ggml_type_name for s in get_all_schemes()}
SCHEME_TO_GGML["F16"] = "F16"
SCHEME_TO_GGML["F32"] = "F32"


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

    if isinstance(value, bool):
        f.write(struct.pack("<I", _GGUF_TYPE_BOOL))
        f.write(struct.pack("<?", value))
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
            f.write(struct.pack("<I", _GGUF_TYPE_UINT32))
            f.write(struct.pack("<Q", 0))
        else:
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
        import json
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


def _read_encode_worker(source, entries, result_queue, imatrix=None):
    """
    Background thread: reads each tensor from source, encodes to ggml bytes,
    and pushes (entry, blob) onto the result queue.

    The bounded queue (maxsize=2) ensures at most 2 encoded tensors are
    buffered, preventing memory blowup on large models.

    imatrix: optional {tensor_name: importance_vector}; tensors with an entry
    are encoded imatrix-weighted, the rest unweighted.
    """
    try:
        for entry in entries:
            name = entry["name"]
            can_decode = entry["_can_decode"]
            target = entry["_target_ggml_name"]
            expected = entry["_expected_size"]

            f32 = source.read_tensor_f32(name)

            if can_decode and f32 is not None:
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
            elif f32 is not None:
                blob = f32.view(np.uint8).tobytes()
            else:
                blob = b"\x00" * expected

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

            result_queue.put((entry, blob))
    except Exception as exc:
        result_queue.put(exc)
    finally:
        result_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# GGUFWriter
# ---------------------------------------------------------------------------

class GGUFWriter:
    """
    Write GGUF files with custom quantization configurations.

    Accepts any ModelSource (GGUF or safetensors) as input.
    Uses a pipelined architecture: a background thread reads and encodes
    tensors while the main thread writes to disk.
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
            quant_config: {"base": "MXFP4_MOE", "groups": {"E": "BF16", ...}}
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

            if verbose:
                print(f"Base quantization: {base_quant}")
                for grp, sch in group_schemes.items():
                    print(f"  Group {grp} -> {sch}")

            classifier = TensorGroupClassifier()
            source_metadata = source.get_metadata()
            all_tensors_info = source.get_all_tensors_info()

            # Pre-scan for UNKNOWN tensors so the user sees issues upfront
            unknown_tensors = [t["name"] for t in all_tensors_info
                               if classifier.classify_tensor(t["name"]) == "UNKNOWN"]
            if unknown_tensors and verbose:
                print(f"  WARNING: {len(unknown_tensors)} tensor(s) have no group classification "
                      f"(will use base quant): {unknown_tensors[:5]}"
                      + (f" ... and {len(unknown_tensors)-5} more" if len(unknown_tensors) > 5 else ""))

            # ==============================================================
            # Pass 1: Compute target types and offsets (no data reading)
            # ==============================================================
            tensor_entries: List[Dict[str, Any]] = []
            data_offset = 0
            # Reset per-call so re-using a writer instance for a second
            # create_hybrid_gguf() call doesn't carry stale entries forward.
            self._fallbacks = []

            for tinfo in all_tensors_info:
                name = tinfo["name"]
                shape = tinfo["shape"]
                n_dims = tinfo["n_dims"]

                group = classifier.classify_tensor(name)
                scheme = group_schemes.get(group, base_quant)
                target_ggml_name = scheme_map.get(scheme, "Q4_0")
                target_ggml_id = GGML_TYPE.get(target_ggml_name, GGML_TYPE["Q4_0"])

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
                # This is a deliberate compatibility tradeoff — make it
                # non-silent by warning once per writer instance.
                if target_ggml_name == "BF16":
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
                can_decode = source_type_name in ("F32", "F16", "BF16")
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
                })

                data_offset = aligned_offset + expected_size

            # ── Validate: detect pre-quantized / undecodable sources ──
            # Two distinct failure modes:
            #   1. UNKNOWN source type — the source could not even identify the
            #      tensor's format. This is ALWAYS a hard error (it would
            #      otherwise produce a zero-filled blob with a bogus type id).
            #   2. A recognized pre-quantized type (e.g. Q4_K) that the user
            #      asked to re-quantize to a different scheme — also an error,
            #      since MagicQuant requires high-precision source weights.
            bad_tensors = []
            unknown_tensors = []
            for entry in tensor_entries:
                if not entry["_can_decode"]:
                    source_type = entry["_source_type_name"]
                    if source_type.startswith("UNKNOWN"):
                        unknown_tensors.append((entry["name"], source_type))
                        continue
                    # Recognized but pre-quantized: error only if the user
                    # wanted a different type than the source already is.
                    group = entry["_group"]
                    scheme = group_schemes.get(group, base_quant)
                    desired_ggml_name = scheme_map.get(scheme, "Q4_0")
                    if desired_ggml_name != source_type:
                        bad_tensors.append((entry["name"], source_type, desired_ggml_name))

            if unknown_tensors:
                count = len(unknown_tensors)
                first_name, first_type = unknown_tensors[0]
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
                    f"Use a high-precision source model."
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

            # ── Prepare metadata ──
            self.metadata = {}
            for k, v in source_metadata.items():
                self.metadata[k] = v
            self.metadata["magicquant.hybrid"] = True
            self.metadata["magicquant.base_quant"] = base_quant
            import json
            self.metadata["magicquant.group_schemes"] = json.dumps(group_schemes)

            # Set general.file_type for llama.cpp compatibility.
            # llama.cpp and HuggingFace use this to report the quantization type.
            # For hybrid models, determine the dominant scheme by counting actual
            # parameter elements per scheme across all tensors (after Pass 1).
            # Aligned to llama.cpp's LLAMA_FTYPE enum. This is a cosmetic,
            # human-readable badge only (each tensor carries its own ggml_type,
            # so inference is unaffected). Generic "Q4_K"/"Q5_K" map to the _M
            # variant. Previous values (Q4_K->12, Q5_K->16, IQ4_NL->20) were
            # wrong (12=MOSTLY_Q4_1_SOME_F16-era, 16=Q5_K_S, 20=MOSTLY_IQ2_XS).
            _ftype_map = {
                "F32": 0, "F16": 1, "BF16": 32,
                "Q8_0": 7,
                "Q6_K": 18,
                "Q5_K": 17, "Q5_K_M": 17, "Q5_K_S": 16,
                "Q4_K": 15, "Q4_K_M": 15, "Q4_K_S": 14,
                "Q3_K": 12, "Q3_K_M": 12,
                "Q2_K": 10,
                "IQ4_NL": 25, "IQ4_XS": 30,
            }
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

            filtered_meta = {k: v for k, v in self.metadata.items() if v is not None}

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

            # Data-integrity notice: surface block-size fallbacks even when
            # verbose=False. One summary line regardless of how many tensors
            # were affected — per-tensor detail lives in self._fallbacks.
            if self._fallbacks:
                first = self._fallbacks[0]
                logger.warning(
                    "%d tensor(s) fell back from their requested quant due to "
                    "block-size (e.g. %s: %s->%s)",
                    len(self._fallbacks), first["tensor"],
                    first["requested"], first["actual"],
                )

            return self.output_path

        finally:
            source.close()

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

            # Start background read+encode thread
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
                    import queue as _queue_mod
                    while True:
                        try:
                            result_q.get_nowait()
                        except _queue_mod.Empty:
                            break
                    worker.join(timeout=5)
                    raise item

                entry, blob = item
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
                idx += 1

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

            # Wait for the worker to finish. If it didn't (hung encode),
            # raise so a truncated file is never renamed into place.
            worker.join(timeout=30)
            if worker.is_alive():
                raise RuntimeError(
                    "Encoder worker thread did not finish within 30s; "
                    "aborting to avoid writing a truncated GGUF."
                )

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata.copy()


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
