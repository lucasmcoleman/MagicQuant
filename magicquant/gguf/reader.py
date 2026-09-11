"""
GGUF Reader - Parse GGUF model files and extract metadata.

The GGUF format is a binary format used by llama.cpp for storing quantized models.
This module provides functions to read and parse GGUF files without requiring
the full llama.cpp library, making it suitable for MagicQuant's preprocessing needs.
"""

from typing import Dict, List, Optional, Any
import struct
import os


class GGUFTypedArray(list):
    """A metadata array value tagged with its on-disk GGUF element type.

    Behaves exactly like a plain ``list`` (equality, iteration, indexing,
    JSON-serializes the same way) so every existing consumer of
    ``GGUFReader.get_metadata()`` keeps working unmodified. The one addition
    is ``gguf_type``: the GGUF wire-format element type id (see
    ``gguf.constants.GGUFValueType`` -- values 0-12, matching the
    ``data_type`` branches in ``_read_value`` below) this array was actually
    read as.

    Exists because a plain Python list has no memory of whether its ints
    came from an INT32 or UINT32 array on disk -- both decode to the same
    Python ``int``. Without this tag, a metadata-copying writer has no way
    to round-trip an array's element type and has to *guess* one from the
    values' magnitude alone, which cannot distinguish a signed array whose
    values happen to be small and non-negative (e.g.
    ``tokenizer.ggml.token_type``, INT32 on disk, values 0-6) from an
    unsigned one. See magicquant/gguf/writer.py's ``_write_metadata_value``.
    """

    def __init__(self, values, gguf_type: int):
        super().__init__(values)
        self.gguf_type = gguf_type


class GGUFTypedInt(int):
    """A metadata scalar int value tagged with its on-disk GGUF type.

    Same rationale as ``GGUFTypedArray`` above, for scalar (non-array)
    integer KV values -- an ``int`` subclass so every existing consumer
    (arithmetic, comparisons, ``isinstance(..., int)``) keeps working
    unmodified.

    (No ``__slots__`` here: CPython disallows a nonempty ``__slots__`` on a
    subtype of ``int`` -- its instances are already variable-length. A plain
    subclass gets a ``__dict__`` for free, which is fine for a value that's
    only ever constructed here and read via ``.gguf_type``.)
    """

    def __new__(cls, value: int, gguf_type: int):
        obj = super().__new__(cls, value)
        obj.gguf_type = gguf_type
        return obj


class GGUFReader:
    """
    Read and parse GGUF model files.
    
    This reader parses the GGUF binary format to extract:
        - Model metadata (parameters, architecture info)
        - Tensor information (names, shapes, data types)
        - Raw tensor data (optional)
    """
    
    # GGUF magic number: "GGUF" in little-endian
    GGUF_MAGIC = 0x46554747

    def __init__(self, filepath: str):
        """
        Initialize the GGUF reader.
        
        Args:
            filepath: Path to the GGUF model file
        """
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        self.data_offset: int = 0
        self._opened = False

    def _ensure_open(self):
        """Parse the file on first access if it hasn't been opened explicitly."""
        if not self._opened:
            self.open()

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self):
        """Parse once; publish reader state only after a complete valid header."""
        if self._opened:
            return
        metadata = {}
        tensors = []
        with open(self.filepath, "rb") as f:
            self.file_size = os.fstat(f.fileno()).st_size
            magic = struct.unpack("<I", self._read_exact(f, 4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {hex(magic)}. Expected {hex(self.GGUF_MAGIC)}")
            version = struct.unpack("<I", self._read_exact(f, 4))[0]
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}")
            tensor_count = struct.unpack("<Q", self._read_exact(f, 8))[0]
            metadata_key_count = struct.unpack("<Q", self._read_exact(f, 8))[0]
            # Every record consumes at least its length/type fields. Reject
            # impossible counts without looping over attacker-sized integers.
            if tensor_count > self.file_size // 24 or metadata_key_count > self.file_size // 12:
                raise ValueError("GGUF record counts exceed file size")
            for _ in range(metadata_key_count):
                key = self._read_string(f)
                data_type = struct.unpack("<I", self._read_exact(f, 4))[0]
                metadata[key] = self._read_value(f, data_type)

            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", self._read_exact(f, 4))[0]
                if not 1 <= n_dims <= 4:
                    raise ValueError(f"Tensor {name!r} has invalid dimension count: {n_dims}")
                shape = [struct.unpack("<Q", self._read_exact(f, 8))[0]
                         for _ in range(n_dims)][::-1]
                if any(dim == 0 for dim in shape):
                    raise ValueError(f"Tensor {name!r} has an empty dimension")
                tensor_type = struct.unpack("<I", self._read_exact(f, 4))[0]
                offset = struct.unpack("<Q", self._read_exact(f, 8))[0]
                tensors.append({"name": name, "n_dims": n_dims, "shape": shape,
                                "data_type": tensor_type, "offset": offset})

            alignment = metadata.get("general.alignment", 32)
            if (not isinstance(alignment, int) or isinstance(alignment, bool)
                    or alignment <= 0 or alignment > 2**32 - 1
                    or getattr(alignment, "gguf_type", 4) != 4
                    or alignment & (alignment - 1)):
                raise ValueError("GGUF general.alignment must be a positive power-of-two uint32")
            data_offset = ((f.tell() + alignment - 1) // alignment) * alignment
            for tensor in tensors:
                if tensor["offset"] % alignment:
                    raise ValueError(f"Unaligned GGUF tensor offset: {tensor['name']!r}")
                if data_offset + tensor["offset"] >= self.file_size:
                    raise ValueError(f"GGUF tensor starts beyond file data: {tensor['name']!r}")

        self.metadata = metadata
        self.tensors = tensors
        self.data_offset = data_offset
        self._opened = True

    def _read_exact(self, f, size: int) -> bytes:
        """Bound all reads by the file, including length-prefixed strings."""
        if size < 0 or size > self.file_size - f.tell():
            raise ValueError(f"Truncated GGUF: read of {size} bytes exceeds file bounds")
        data = f.read(size)
        if len(data) != size:
            raise ValueError("Truncated GGUF while reading header")
        return data

    def _read_string(self, f) -> str:
        """Read a GGUF string.

        Decode non-strictly: one non-UTF-8 byte in a vocab token or metadata
        value shouldn't abort parsing the whole file.
        """
        length = struct.unpack('<Q', self._read_exact(f, 8))[0]
        return self._read_exact(f, length).decode('utf-8', errors='replace')
    
    def _read_value(self, f, data_type: int, depth: int = 0) -> Any:
        """Read a value of the given GGUF type.

        Integer-family scalars (UINT8/INT8/UINT16/INT16/UINT32/INT32/
        UINT64/INT64) come back tagged with their exact on-disk type via
        ``GGUFTypedInt``, and ARRAY values via ``GGUFTypedArray`` -- see
        those classes' docstrings for why. Both subclass the plain Python
        type they'd otherwise be, so this is purely additive.
        """
        if depth > 32:
            raise ValueError("GGUF metadata arrays are too deeply nested")
        if data_type == 0:  # UINT8
            return GGUFTypedInt(struct.unpack('<B', self._read_exact(f, 1))[0], data_type)
        elif data_type == 1:  # INT8
            return GGUFTypedInt(struct.unpack('<b', self._read_exact(f, 1))[0], data_type)
        elif data_type == 2:  # UINT16
            return GGUFTypedInt(struct.unpack('<H', self._read_exact(f, 2))[0], data_type)
        elif data_type == 3:  # INT16
            return GGUFTypedInt(struct.unpack('<h', self._read_exact(f, 2))[0], data_type)
        elif data_type == 4:  # UINT32
            return GGUFTypedInt(struct.unpack('<I', self._read_exact(f, 4))[0], data_type)
        elif data_type == 5:  # INT32
            return GGUFTypedInt(struct.unpack('<i', self._read_exact(f, 4))[0], data_type)
        elif data_type == 6:  # FLOAT32
            return struct.unpack('<f', self._read_exact(f, 4))[0]
        elif data_type == 7:  # BOOL
            return struct.unpack('<?', self._read_exact(f, 1))[0]
        elif data_type == 8:  # STRING
            return self._read_string(f)
        elif data_type == 9:  # ARRAY
            elem_type = struct.unpack('<I', self._read_exact(f, 4))[0]
            length = struct.unpack('<Q', self._read_exact(f, 8))[0]
            if length > self.file_size - f.tell():
                raise ValueError("GGUF array length exceeds file bounds")
            items = [self._read_value(f, elem_type, depth + 1) for _ in range(length)]
            return GGUFTypedArray(items, elem_type)
        elif data_type == 10:  # UINT64
            return GGUFTypedInt(struct.unpack('<Q', self._read_exact(f, 8))[0], data_type)
        elif data_type == 11:  # INT64
            return GGUFTypedInt(struct.unpack('<q', self._read_exact(f, 8))[0], data_type)
        elif data_type == 12:  # FLOAT64
            return struct.unpack('<d', self._read_exact(f, 8))[0]
        else:
            raise ValueError(f"Unknown GGUF data type: {data_type}")
    
    def close(self):
        """Close the file (for context manager)."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get all model metadata."""
        self._ensure_open()
        return self.metadata.copy()
    
    def get_tensor_names(self) -> List[str]:
        """Get list of tensor names in the model."""
        self._ensure_open()
        return [t['name'] for t in self.tensors]
    
    def get_tensor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tensor."""
        self._ensure_open()
        for tensor in self.tensors:
            if tensor['name'] == name:
                return tensor
        return None
    
    def get_all_tensors_info(self) -> List[Dict[str, Any]]:
        """Get information about all tensors."""
        self._ensure_open()
        return [t.copy() for t in self.tensors]
    
    def get_model_architecture(self) -> str:
        """Get the model architecture name from metadata."""
        self._ensure_open()
        # Common metadata keys for architecture
        arch_keys = [
            'general.architecture',
            'architecture',
            'llama.architecture'
        ]
        
        for key in arch_keys:
            if key in self.metadata:
                return self.metadata[key]

        return 'unknown'
    
    def get_parameter_count(self) -> int:
        """Total element count across all tensors (including 1-D norms/biases)."""
        self._ensure_open()
        total = 0
        for tensor in self.tensors:
            shape = tensor['shape']
            params = 1
            for dim in shape:
                params *= dim
            total += params
        return total
    
    def get_file_size_gb(self) -> float:
        """Get file size in GB."""
        return self.file_size / (1024 ** 3)
    
    def get_bits_per_weight(self) -> float:
        """Estimate average bits per weight from model size."""
        self._ensure_open()
        params = self.get_parameter_count()
        if params == 0:
            return 8.0
        
        file_bytes = self.file_size
        return (file_bytes * 8) / params


def read_gguf_file(filepath: str) -> GGUFReader:
    """
    Create and open a GGUF reader (convenience function).
    
    Args:
        filepath: Path to the GGUF model file
        
    Returns:
        Initialized GGUFReader object
    """
    reader = GGUFReader(filepath)
    reader.open()
    return reader


if __name__ == "__main__":
    import sys
    from magicquant.gguf.tensor_groups import TensorGroupClassifier

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Reading GGUF file: {filepath}")

        with GGUFReader(filepath) as reader:
            print(f"Architecture: {reader.get_model_architecture()}")
            print(f"Parameters:   {reader.get_parameter_count():,}")
            print(f"File Size:    {reader.get_file_size_gb():.2f} GB")
            print(f"Bits/Weight:  {reader.get_bits_per_weight():.2f}")
            print()
            classifier = TensorGroupClassifier()
            grouped = classifier.classify_tensors(reader.get_tensor_names())
            for group, tensors in grouped.items():
                if tensors:
                    print(f"  {group}: {len(tensors)} tensors")
    else:
        print("Usage: python -m magicquant.gguf.reader <path_to_gguf_file>")
