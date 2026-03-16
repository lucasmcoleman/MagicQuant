"""
GGUF Reader - Parse GGUF model files and extract metadata.

The GGUF format is a binary format used by llama.cpp for storing quantized models.
This module provides functions to read and parse GGUF files without requiring
the full llama.cpp library, making it suitable for MagicQuant's preprocessing needs.
"""

from typing import Dict, List, Optional, Any
import struct
import os


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
    
    # Data types in GGUF
    GGUF_TYPES = {
        0: "UINT8",
        1: "INT8",
        2: "UINT16",
        3: "INT16",
        4: "UINT32",
        5: "INT32",
        6: "FLOAT32",
        7: "BOOL",
        8: "STRING",
        9: "ARRAY",
        10: "UINT64",
        11: "INT64",
        12: "FLOAT64"
    }
    
    # ggml_type enum — must match the canonical ggml enum exactly
    QUANT_TYPES = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K",
        12: "Q4_K",
        13: "Q5_K",
        14: "Q6_K",
        15: "Q8_K",
        16: "IQ2_XXS",
        17: "IQ2_XS",
        18: "IQ3_XXS",
        19: "IQ1_S",
        20: "IQ4_NL",
        21: "IQ3_S",
        22: "IQ2_S",
        23: "IQ4_XS",
        24: "I8",
        25: "I16",
        26: "I32",
        27: "I64",
        28: "F64",
        29: "IQ1_M",
        30: "BF16",
        100: "MXFP4",  # MagicQuant custom: OCP MX FP4
    }
    
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
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self):
        """Open and parse the GGUF file."""
        with open(self.filepath, 'rb') as f:
            # Read magic number
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: {hex(magic)}. Expected {hex(self.GGUF_MAGIC)}")
            
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            
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
        """Read a GGUF string."""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
    
    def _read_value(self, f, data_type: int) -> Any:
        """Read a value of the given GGUF type."""
        if data_type == 0:  # UINT8
            return struct.unpack('<B', f.read(1))[0]
        elif data_type == 1:  # INT8
            return struct.unpack('<b', f.read(1))[0]
        elif data_type == 2:  # UINT16
            return struct.unpack('<H', f.read(2))[0]
        elif data_type == 3:  # INT16
            return struct.unpack('<h', f.read(2))[0]
        elif data_type == 4:  # UINT32
            return struct.unpack('<I', f.read(4))[0]
        elif data_type == 5:  # INT32
            return struct.unpack('<i', f.read(4))[0]
        elif data_type == 6:  # FLOAT32
            return struct.unpack('<f', f.read(4))[0]
        elif data_type == 7:  # BOOL
            return struct.unpack('<?', f.read(1))[0]
        elif data_type == 8:  # STRING
            return self._read_string(f)
        elif data_type == 9:  # ARRAY
            elem_type = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, elem_type) for _ in range(length)]
        elif data_type == 10:  # UINT64
            return struct.unpack('<Q', f.read(8))[0]
        elif data_type == 11:  # INT64
            return struct.unpack('<q', f.read(8))[0]
        elif data_type == 12:  # FLOAT64
            return struct.unpack('<d', f.read(8))[0]
        else:
            raise ValueError(f"Unknown GGUF data type: {data_type}")
    
    def close(self):
        """Close the file (for context manager)."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get all model metadata."""
        return self.metadata.copy()
    
    def get_tensor_names(self) -> List[str]:
        """Get list of tensor names in the model."""
        return [t['name'] for t in self.tensors]
    
    def get_tensor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tensor."""
        for tensor in self.tensors:
            if tensor['name'] == name:
                return tensor
        return None
    
    def get_all_tensors_info(self) -> List[Dict[str, Any]]:
        """Get information about all tensors."""
        return [t.copy() for t in self.tensors]
    
    def get_model_architecture(self) -> str:
        """Get the model architecture name from metadata."""
        # Common metadata keys for architecture
        arch_keys = [
            'general.architecture',
            'architecture',
            'llama.architecture'
        ]
        
        for key in arch_keys:
            if key in self.metadata:
                return self.metadata[key]
        
        # Try to infer from tensor names
        for tensor_name in self.get_tensor_names():
            if 'transformer' in tensor_name.lower():
                return 'gpt-next'
            elif 'model' in tensor_name.lower():
                return 'llama'
        
        return 'unknown'
    
    def get_parameter_count(self) -> int:
        """Estimate total parameter count from tensor shapes."""
        total = 0
        for tensor in self.tensors:
            shape = tensor['shape']
            if len(shape) >= 2:  # Weight matrices have at least 2 dims
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
        params = self.get_parameter_count()
        if params == 0:
            return 8.0
        
        file_bytes = self.file_size
        return (file_bytes * 8) / params
    
    def find_tensors_by_group(self, group_name: str) -> List[Dict[str, Any]]:
        """
        Find tensors that belong to a specific functional group.
        
        Args:
            group_name: Group identifier ('E', 'H', 'Q', 'K', 'O', 'U', 'D')
            
        Returns:
            List of tensor info dicts matching the group
        """
        from magicquant.gguf.tensor_groups import TensorGroupClassifier
        
        classifier = TensorGroupClassifier()
        tensors = []
        
        for tensor in self.tensors:
            if classifier.classify_tensor(tensor['name']) == group_name:
                tensors.append(tensor)
        
        return tensors


def read_gguf_file(filepath: str) -> GGUFReader:
    """
    Create and open a GGUF reader (convenience function).
    
    Args:
        filepath: Path to the GGUF model file
        
    Returns:
        Initialized GGUFReader object
    """
    return GGUFReader(filepath)


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