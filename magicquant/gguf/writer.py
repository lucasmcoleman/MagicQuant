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
            first = value[0]
            if isinstance(first, str):
                f.write(struct.pack("<I", _GGUF_TYPE_STRING))
                f.write(struct.pack("<Q", len(value)))
                for item in value:
                    _write_string(f, str(item))
            elif isinstance(first, float):
                f.write(struct.pack("<I", _GGUF_TYPE_FLOAT32))
                f.write(struct.pack("<Q", len(value)))
                for item in value:
                    f.write(struct.pack("<f", float(item)))
            elif isinstance(first, int):
                f.write(struct.pack("<I", _GGUF_TYPE_INT32))
                f.write(struct.pack("<Q", len(value)))
                for item in value:
                    f.write(struct.pack("<i", int(item)))
            else:
                f.write(struct.pack("<I", _GGUF_TYPE_STRING))
                f.write(struct.pack("<Q", len(value)))
                for item in value:
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

def _read_encode_worker(source, entries, result_queue):
    """
    Background thread: reads each tensor from source, encodes to ggml bytes,
    and pushes (entry, blob) onto the result queue.

    The bounded queue (maxsize=2) ensures at most 2 encoded tensors are
    buffered, preventing memory blowup on large models.
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
                blob = encode_to_ggml_bytes(f32, target)
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

    def create_hybrid_gguf(
        self,
        base_model_path: str,
        quant_config: Dict,
        verbose: bool = True,
        adapter_path: str = None,
    ) -> str:
        """
        Create a hybrid GGUF from any supported source format.

        Args:
            base_model_path: Path to source model — .gguf file, .safetensors
                file, or directory containing safetensors + config.json
            quant_config: {"base": "MXFP4_MOE", "groups": {"E": "BF16", ...}}
            verbose: Print progress
            adapter_path: Optional path to a LoRA adapter directory.
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

            for tinfo in all_tensors_info:
                name = tinfo["name"]
                shape = tinfo["shape"]
                n_dims = tinfo["n_dims"]

                group = classifier.classify_tensor(name)
                scheme = group_schemes.get(group, base_quant)
                target_ggml_name = scheme_map.get(scheme, "Q4_0")
                target_ggml_id = GGML_TYPE.get(target_ggml_name, GGML_TYPE["Q4_0"])

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
                if target_ggml_name == "BF16":
                    target_ggml_name = "F16"
                    target_ggml_id = GGML_TYPE["F16"]

                # Block-size compatibility check: quantized types require the
                # contiguous row dimension (ne[0] in GGUF) to be a multiple of
                # the block size.  The writer stores shapes in row-major order
                # and reverses when writing, so ne[0] = shape[-1].  Fall back
                # to F32 for tensors that don't meet the requirement.  F32 is
                # used (not F16) because some ops (SSM conv1d) assert F32
                # operands.  These tensors are small, so F32 is negligible.
                row_size = shape[-1] if len(shape) >= 1 else 1
                block_size = GGML_BLOCK_SIZE.get(target_ggml_name, 1)
                if block_size > 1 and row_size % block_size != 0:
                    if verbose:
                        print(f"  [COMPAT] {name}: row_size={row_size} not divisible by "
                              f"{target_ggml_name} block_size={block_size}, falling back to F32")
                    target_ggml_name = "F32"
                    target_ggml_id = GGML_TYPE["F32"]

                n_elems = _tensor_n_elements(shape)

                source_type_name = source.get_source_type_name(name)
                can_decode = source_type_name in ("F32", "F16", "BF16")
                if not can_decode:
                    target_ggml_name = source_type_name
                    target_ggml_id = GGML_TYPE.get(source_type_name, 0)

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

            # ── Validate: detect pre-quantized sources that can't be re-quantized ──
            bad_tensors = []
            for entry in tensor_entries:
                if not entry["_can_decode"]:
                    source_type = entry["_source_type_name"]
                    # Check if the user wanted a different type than the source
                    group = entry["_group"]
                    scheme = group_schemes.get(group, base_quant)
                    desired_ggml_name = scheme_map.get(scheme, "Q4_0")
                    if desired_ggml_name != source_type:
                        bad_tensors.append((entry["name"], source_type, desired_ggml_name))

            if bad_tensors:
                count = len(bad_tensors)
                source_type = bad_tensors[0][1]
                raise ValueError(
                    f"Cannot re-quantize {count} tensors: source is already quantized "
                    f"({source_type}). MagicQuant requires BF16, F16, or F32 source weights. "
                    f"Use a high-precision source model."
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
            _ftype_map = {
                "Q8_0": 7, "Q5_K": 16, "Q5_K_M": 17, "Q4_K": 12,
                "Q4_K_M": 15, "Q6_K": 18, "Q3_K": 11, "Q2_K": 10,
                "IQ4_NL": 20, "BF16": 32, "F16": 1, "F32": 0,
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

            with open(self.output_path, "wb") as f:
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
                    args=(source, tensor_entries, result_q),
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

                worker.join(timeout=5)

            elapsed = time.monotonic() - t_start
            output_size_mb = Path(self.output_path).stat().st_size / (1024 * 1024)
            if verbose:
                print(f"Done. {output_size_mb:.1f} MB in {elapsed:.1f}s "
                      f"({output_size_mb / max(elapsed, 0.001):.0f} MB/s)")

            return self.output_path

        finally:
            source.close()

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata.copy()


def create_hybrid_gguf(
    output_path: str, base_model_path: str,
    quant_config: Dict, verbose: bool = True,
    adapter_path: str = None,
) -> str:
    """Convenience function to create a hybrid GGUF model."""
    writer = GGUFWriter(output_path)
    return writer.create_hybrid_gguf(
        base_model_path, quant_config, verbose,
        adapter_path=adapter_path
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
