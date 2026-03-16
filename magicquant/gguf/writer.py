"""
GGUF Writer - Create hybrid quantization GGUF files.

This module creates new GGUF files with mixed-precision quantization,
combining different quant schemes for different tensor groups.

Quantization is delegated to ``magicquant.quant.converters`` which is
the single source of truth for all ggml block-format encoders.

GGUF binary format (version 3):
  [magic: 4 bytes "GGUF" LE] [version: uint32] [tensor_count: uint64]
  [metadata_kv_count: uint64] [metadata KV pairs...] [tensor info entries...]
  [padding to 32-byte alignment] [tensor data (each tensor 32-byte aligned)]

Tensor info entry:
  [name_len: uint64] [name: bytes] [n_dims: uint32] [dims: uint64 * n_dims]
  [type: uint32 (ggml_type enum)] [offset: uint64 (into data section)]

The writer uses a two-pass streaming architecture to avoid holding all
quantized tensor data in memory simultaneously:

  Pass 1 (header): Compute target types, data sizes, and offsets for every
      tensor without touching actual data. Write the complete GGUF header
      (magic, version, counts, metadata KV pairs, tensor info entries).

  Pass 2 (data): Stream tensor-by-tensor. For each tensor: read source
      bytes, decode to f32, quantize via encode_to_ggml_bytes(), write
      alignment padding + blob directly to the output file, then discard
      the buffer.
"""

from typing import Dict, List, Any
import struct
import os
import numpy as np

from magicquant.quant.converters import (
    encode_to_ggml_bytes,
    ggml_tensor_data_size,
    GGML_BLOCK_SIZE,
    GGML_TYPE_SIZE,
)


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
    "MXFP4":  100,  # MagicQuant custom: OCP MX FP4 (E2M1 + shared E8M0)
}

# Reverse lookup: integer id -> type name
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

# Map MagicQuant scheme names to the ggml_type name we write into the file
SCHEME_TO_GGML = {
    "BF16":      "BF16",
    "F16":       "F16",
    "F32":       "F32",
    "Q8_0":      "Q8_0",
    "Q6_K":      "Q6_K",
    "Q5_K":      "Q5_K",
    "Q4_K_M":    "Q4_K",   # Q4_K_M maps to ggml Q4_K
    "IQ4_NL":    "IQ4_NL",
    "MXFP4_MOE": "MXFP4",  # OCP MX FP4 (custom type, proper microscaling)
}


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


def _read_source_tensor_bytes(
    filepath: str, data_section_offset: int,
    tensor_offset: int, byte_length: int,
) -> bytes:
    with open(filepath, "rb") as f:
        f.seek(data_section_offset + tensor_offset)
        return f.read(byte_length)


# ---------------------------------------------------------------------------
# Source tensor decoding (float types only)
# ---------------------------------------------------------------------------

def _decode_tensor_to_f32(buf: bytes, source_ggml_type: int, n_elements: int):
    """
    Decode source tensor bytes into float32 numpy array.

    Returns None when the source is already quantised (cannot losslessly
    decode without full ggml dequant kernels).
    """
    type_name = _GGML_TYPE_NAME.get(source_ggml_type, "")
    if type_name == "F32":
        return np.frombuffer(buf, dtype=np.float32).copy()
    if type_name == "F16":
        return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    if type_name == "BF16":
        raw = np.frombuffer(buf, dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return None


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


def _skip_value(f, vtype: int):
    """Skip a GGUF metadata value of the given type."""
    if vtype in (_GGUF_TYPE_UINT8, _GGUF_TYPE_INT8, _GGUF_TYPE_BOOL):
        f.read(1)
    elif vtype in (_GGUF_TYPE_UINT16, _GGUF_TYPE_INT16):
        f.read(2)
    elif vtype in (_GGUF_TYPE_UINT32, _GGUF_TYPE_INT32, _GGUF_TYPE_FLOAT32):
        f.read(4)
    elif vtype in (_GGUF_TYPE_UINT64, _GGUF_TYPE_INT64, _GGUF_TYPE_FLOAT64):
        f.read(8)
    elif vtype == _GGUF_TYPE_STRING:
        f.read(struct.unpack("<Q", f.read(8))[0])
    elif vtype == _GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        length = struct.unpack("<Q", f.read(8))[0]
        for _ in range(length):
            _skip_value(f, elem_type)
    else:
        raise ValueError(f"Unknown GGUF type tag {vtype} while skipping value")


# ---------------------------------------------------------------------------
# GGUFWriter
# ---------------------------------------------------------------------------

class GGUFWriter:
    """
    Write GGUF files with custom quantization configurations.

    Quantization of individual tensors is delegated to
    ``magicquant.quant.converters.encode_to_ggml_bytes()``.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata: Dict[str, Any] = {}

    def create_hybrid_gguf(
        self,
        base_model_path: str,
        quant_config: Dict,
        verbose: bool = True,
    ) -> str:
        from magicquant.gguf.reader import GGUFReader
        from magicquant.gguf.tensor_groups import TensorGroupClassifier

        if verbose:
            print(f"Loading base model: {base_model_path}")

        reader = GGUFReader(base_model_path)
        reader.open()

        try:
            base_quant = quant_config.get("base", "Q4_K_M")
            group_schemes = quant_config.get("groups", {})

            if verbose:
                print(f"Base quantization: {base_quant}")
                for grp, sch in group_schemes.items():
                    print(f"  Group {grp} -> {sch}")

            classifier = TensorGroupClassifier()
            source_metadata = reader.get_metadata()
            all_tensors_info = reader.get_all_tensors_info()

            source_data_offset = self._find_source_data_offset(
                base_model_path, len(source_metadata), len(all_tensors_info)
            )

            # ==============================================================
            # Pass 1: Compute target types and offsets (no data reading)
            # ==============================================================
            tensor_entries: List[Dict[str, Any]] = []
            data_offset = 0

            for tinfo in all_tensors_info:
                name = tinfo["name"]
                shape = tinfo["shape"]
                n_dims = tinfo["n_dims"]
                source_type = tinfo["data_type"]

                group = classifier.classify_tensor(name)
                scheme = group_schemes.get(group, base_quant)
                target_ggml_name = SCHEME_TO_GGML.get(scheme, "Q4_0")
                target_ggml_id = GGML_TYPE.get(target_ggml_name, GGML_TYPE["Q4_0"])

                n_elems = _tensor_n_elements(shape)

                source_type_name = _GGML_TYPE_NAME.get(source_type, "F16")

                # Check if the source can be decoded to f32 for re-quantization.
                # If not, we will pass through raw bytes, so target = source.
                can_decode = source_type_name in ("F32", "F16", "BF16")
                if not can_decode:
                    target_ggml_name = source_type_name
                    target_ggml_id = source_type

                expected_size = ggml_tensor_data_size(target_ggml_name, n_elems)

                aligned_offset = _align(data_offset)

                tensor_entries.append({
                    "name": name,
                    "n_dims": n_dims,
                    "shape": shape,
                    "ggml_type": target_ggml_id,
                    "offset": aligned_offset,
                    # Fields used during Pass 2 (not written to header):
                    "_target_ggml_name": target_ggml_name,
                    "_source_type": source_type,
                    "_source_type_name": source_type_name,
                    "_source_offset": tinfo["offset"],
                    "_n_elems": n_elems,
                    "_expected_size": expected_size,
                    "_can_decode": can_decode,
                    "_group": group,
                })

                data_offset = aligned_offset + expected_size

            # ── Prepare metadata ──
            self.metadata = {}
            for k, v in source_metadata.items():
                self.metadata[k] = v
            self.metadata["magicquant.hybrid"] = True
            self.metadata["magicquant.base_quant"] = base_quant
            import json
            self.metadata["magicquant.group_schemes"] = json.dumps(group_schemes)

            filtered_meta = {k: v for k, v in self.metadata.items() if v is not None}

            # ==============================================================
            # Write header (magic, version, counts, metadata, tensor infos)
            # ==============================================================
            if verbose:
                print(f"\nWriting output: {self.output_path}")

            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

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
                # Pass 2: Stream tensor data directly to file
                # ==========================================================
                data_section_start = f.tell()

                for idx, entry in enumerate(tensor_entries):
                    name = entry["name"]
                    n_elems = entry["_n_elems"]
                    source_type = entry["_source_type"]
                    source_type_name = entry["_source_type_name"]
                    target_ggml_name = entry["_target_ggml_name"]
                    expected_size = entry["_expected_size"]
                    can_decode = entry["_can_decode"]
                    aligned_offset = entry["offset"]

                    # Write alignment padding to reach this tensor's offset
                    current_pos = f.tell() - data_section_start
                    padding = aligned_offset - current_pos
                    if padding > 0:
                        f.write(b"\x00" * padding)

                    # Read source bytes
                    source_byte_len = ggml_tensor_data_size(source_type_name, n_elems)
                    raw_bytes = _read_source_tensor_bytes(
                        base_model_path, source_data_offset,
                        entry["_source_offset"], source_byte_len,
                    )

                    # Decode and re-quantize, or pass through
                    if can_decode:
                        f32 = _decode_tensor_to_f32(raw_bytes, source_type, n_elems)
                        blob = encode_to_ggml_bytes(f32, target_ggml_name)
                    else:
                        blob = raw_bytes

                    # Pad / trim to exact expected size
                    if len(blob) < expected_size:
                        blob = blob + b"\x00" * (expected_size - len(blob))
                    elif len(blob) > expected_size:
                        blob = blob[:expected_size]

                    f.write(blob)

                    if verbose:
                        print(f"  [{idx+1}/{len(tensor_entries)}] {name}: "
                              f"{source_type_name} -> {target_ggml_name} "
                              f"(group={entry['_group']})")

            output_size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
            if verbose:
                print(f"Done. Output size: {output_size_mb:.1f} MB")

            return self.output_path

        finally:
            reader.close()

    @staticmethod
    def _find_source_data_offset(filepath: str, n_metadata: int, n_tensors: int) -> int:
        with open(filepath, "rb") as f:
            f.read(4 + 4 + 8 + 8)
            for _ in range(n_metadata):
                key_len = struct.unpack("<Q", f.read(8))[0]
                f.read(key_len)
                vtype = struct.unpack("<I", f.read(4))[0]
                _skip_value(f, vtype)
            for _ in range(n_tensors):
                name_len = struct.unpack("<Q", f.read(8))[0]
                f.read(name_len)
                n_dims = struct.unpack("<I", f.read(4))[0]
                f.read(n_dims * 8 + 4 + 8)
            return _align(f.tell())

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata.copy()


def create_hybrid_gguf(
    output_path: str, base_model_path: str,
    quant_config: Dict, verbose: bool = True,
) -> str:
    """Convenience function to create a hybrid GGUF model."""
    writer = GGUFWriter(output_path)
    return writer.create_hybrid_gguf(base_model_path, quant_config, verbose)


if __name__ == "__main__":
    import sys
    import json as _json

    if len(sys.argv) < 4:
        print("Usage: python gguf_writer.py <output.gguf> <base_model.gguf> <config.json>")
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
