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
        """Open and parse the GGUF file. Idempotent — safe to call repeatedly
        and via both the context manager and lazy accessor paths."""
        if self._opened:
            return
        self._opened = True
        with open(self.filepath, 'rb') as f:
            # Read magic number
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {hex(magic)}. Expected {hex(self.GGUF_MAGIC)}")
            
            # Read version (unused: advances the file cursor past the field;
            # GGUF version handling is not implemented by this reader).
            struct.unpack('<I', f.read(4))

            # Read tensor count
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata key count
            metadata_key_count = struct.unpack('<Q', f.read(8))[0]
            
            # Parse metadata keys
            for _ in range(metadata_key_count):
                key = self._read_string(f)
                data_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, data_type)
                self.metadata[key] = value
            
            # Parse tensor information
            for _ in range(tensor_count):
                tensor_name = self._read_string(f)

                # Read tensor shape (n dimensions, reverse order)
                n_dims = struct.unpack('<I', f.read(4))[0]
                shape = []
                for i in range(n_dims):
                    dim = struct.unpack('<Q', f.read(8))[0]
                    shape.insert(0, dim)  # Reverse order

                # Read tensor type
                tensor_type = struct.unpack('<I', f.read(4))[0]

                # Read offset
                offset = struct.unpack('<Q', f.read(8))[0]

                self.tensors.append({
                    'name': tensor_name,
                    'n_dims': n_dims,
                    'shape': shape,
                    'data_type': tensor_type,
                    'offset': offset
                })

            # Data section starts at next 32-byte alignment after ALL header data
            # (metadata KVs + tensor info entries)
            self.data_offset = ((f.tell() + 31) // 32) * 32
    
    def _read_string(self, f) -> str:
        """Read a GGUF string.

        Decode non-strictly: one non-UTF-8 byte in a vocab token or metadata
        value shouldn't abort parsing the whole file.
        """
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8', errors='replace')
    
    def _read_value(self, f, data_type: int) -> Any:
        """Read a value of the given GGUF type.

        Integer-family scalars (UINT8/INT8/UINT16/INT16/UINT32/INT32/
        UINT64/INT64) come back tagged with their exact on-disk type via
        ``GGUFTypedInt``, and ARRAY values via ``GGUFTypedArray`` -- see
        those classes' docstrings for why. Both subclass the plain Python
        type they'd otherwise be, so this is purely additive.
        """
        if data_type == 0:  # UINT8
            return GGUFTypedInt(struct.unpack('<B', f.read(1))[0], data_type)
        elif data_type == 1:  # INT8
            return GGUFTypedInt(struct.unpack('<b', f.read(1))[0], data_type)
        elif data_type == 2:  # UINT16
            return GGUFTypedInt(struct.unpack('<H', f.read(2))[0], data_type)
        elif data_type == 3:  # INT16
            return GGUFTypedInt(struct.unpack('<h', f.read(2))[0], data_type)
        elif data_type == 4:  # UINT32
            return GGUFTypedInt(struct.unpack('<I', f.read(4))[0], data_type)
        elif data_type == 5:  # INT32
            return GGUFTypedInt(struct.unpack('<i', f.read(4))[0], data_type)
        elif data_type == 6:  # FLOAT32
            return struct.unpack('<f', f.read(4))[0]
        elif data_type == 7:  # BOOL
            return struct.unpack('<?', f.read(1))[0]
        elif data_type == 8:  # STRING
            return self._read_string(f)
        elif data_type == 9:  # ARRAY
            elem_type = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<Q', f.read(8))[0]
            items = [self._read_value(f, elem_type) for _ in range(length)]
            return GGUFTypedArray(items, elem_type)
        elif data_type == 10:  # UINT64
            return GGUFTypedInt(struct.unpack('<Q', f.read(8))[0], data_type)
        elif data_type == 11:  # INT64
            return GGUFTypedInt(struct.unpack('<q', f.read(8))[0], data_type)
        elif data_type == 12:  # FLOAT64
            return struct.unpack('<d', f.read(8))[0]
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