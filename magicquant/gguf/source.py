"""
Model Source Abstraction - Unified interface for reading GGUF and safetensors.

Both GGUFSource and SafetensorsSource expose the same API so the writer
can consume either format transparently.
"""

from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import struct
import json
import os
import re
import numpy as np


class ModelSource(ABC):
    """Abstract interface for a model source (GGUF or safetensors)."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return GGUF-compatible metadata dict."""
        ...

    @abstractmethod
    def get_tensor_names(self) -> List[str]:
        """Return tensor names in GGUF convention."""
        ...

    @abstractmethod
    def get_all_tensors_info(self) -> List[Dict[str, Any]]:
        """Return list of tensor info dicts with keys:
        name, n_dims, shape (row-major reversed), data_type (ggml_type int)."""
        ...

    @abstractmethod
    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        """Read a tensor and return it as a flat float32 array.
        Returns None if the tensor is already quantised and can't be decoded."""
        ...

    @abstractmethod
    def get_source_type_name(self, tensor_name: str) -> str:
        """Return the ggml type name string for a tensor's source format."""
        ...

    def close(self):
        pass


# =====================================================================
# GGUF Source
# =====================================================================

class GGUFSource(ModelSource):
    """Read tensors from a GGUF file."""

    # ggml_type id -> name
    _TYPE_NAME = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
        8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
        13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
        18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
        23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
        28: "F64", 29: "IQ1_M", 30: "BF16", 100: "MXFP4",
    }

    def __init__(self, filepath: str):
        from magicquant.gguf.reader import GGUFReader
        self._path = filepath
        self._reader = GGUFReader(filepath)
        self._reader.open()
        self._data_offset = self._find_data_offset()

    def _find_data_offset(self) -> int:
        meta = self._reader.get_metadata()
        tensors = self._reader.get_all_tensors_info()
        # Walk the header to find the true data offset
        from magicquant.quant.converters import GGML_BLOCK_SIZE, GGML_TYPE_SIZE
        with open(self._path, "rb") as f:
            f.read(4 + 4 + 8 + 8)
            for _ in range(len(meta)):
                kl = struct.unpack("<Q", f.read(8))[0]; f.read(kl)
                vt = struct.unpack("<I", f.read(4))[0]
                self._skip_value(f, vt)
            for _ in range(len(tensors)):
                nl = struct.unpack("<Q", f.read(8))[0]; f.read(nl)
                nd = struct.unpack("<I", f.read(4))[0]
                f.read(nd * 8 + 4 + 8)
            return ((f.tell() + 31) // 32) * 32

    @staticmethod
    def _skip_value(f, vtype):
        if vtype in (0, 1, 7): f.read(1)
        elif vtype in (2, 3): f.read(2)
        elif vtype in (4, 5, 6): f.read(4)
        elif vtype in (10, 11, 12): f.read(8)
        elif vtype == 8:
            f.read(struct.unpack("<Q", f.read(8))[0])
        elif vtype == 9:
            et = struct.unpack("<I", f.read(4))[0]
            ln = struct.unpack("<Q", f.read(8))[0]
            for _ in range(ln):
                GGUFSource._skip_value(f, et)

    def get_metadata(self):
        return self._reader.get_metadata()

    def get_tensor_names(self):
        return self._reader.get_tensor_names()

    def get_all_tensors_info(self):
        return self._reader.get_all_tensors_info()

    def get_source_type_name(self, tensor_name: str) -> str:
        info = self._reader.get_tensor_info(tensor_name)
        if info is None:
            return "F16"
        return self._TYPE_NAME.get(info["data_type"], "F16")

    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        from magicquant.quant.converters import ggml_tensor_data_size
        info = self._reader.get_tensor_info(tensor_name)
        if info is None:
            return None
        type_name = self._TYPE_NAME.get(info["data_type"], "")
        n_elems = 1
        for d in info["shape"]:
            n_elems *= d
        byte_len = ggml_tensor_data_size(type_name, n_elems)
        with open(self._path, "rb") as f:
            f.seek(self._data_offset + info["offset"])
            buf = f.read(byte_len)
        if type_name == "F32":
            return np.frombuffer(buf, dtype=np.float32).copy()
        if type_name == "F16":
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        if type_name == "BF16":
            raw = np.frombuffer(buf, dtype=np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return None  # quantised — can't decode

    def close(self):
        self._reader.close()


# =====================================================================
# Safetensors Source
# =====================================================================

# HuggingFace tensor name -> GGUF tensor name mapping.
# Covers LLaMA / Qwen / Mistral / DeepSeek / Yi and similar architectures.
_HF_TO_GGUF_PATTERNS = [
    # Embeddings
    (r"^model\.embed_tokens\.weight$",              "token_embd.weight"),
    (r"^model\.embeddings\.word_embeddings\.weight$","token_embd.weight"),
    # Output head
    (r"^lm_head\.weight$",                          "output.weight"),
    # Final norm
    (r"^model\.norm\.weight$",                       "output_norm.weight"),
    (r"^model\.final_layernorm\.weight$",            "output_norm.weight"),
    # Per-layer attention
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_k.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_v.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_output.weight"),
    # QKV fused (some models)
    (r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_qkv.weight"),
    # Per-layer FFN
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_down.weight"),
    # Gate+Up fused
    (r"^model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_up.weight"),
    # Layer norms
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_norm.weight"),
    # MoE
    (r"^model\.layers\.(\d+)\.mlp\.gate\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_inp.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_up_exps.{m.group(2)}.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_exps.{m.group(2)}.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_down_exps.{m.group(2)}.weight"),
]

_HF_TO_GGUF_COMPILED = [(re.compile(p), r) for p, r in _HF_TO_GGUF_PATTERNS]


def _hf_name_to_gguf(hf_name: str) -> str:
    """Convert a HuggingFace tensor name to GGUF convention."""
    for pattern, replacement in _HF_TO_GGUF_COMPILED:
        m = pattern.match(hf_name)
        if m:
            if callable(replacement):
                return replacement(m)
            return replacement
    # Fallback: keep original name
    return hf_name


# safetensors dtype -> ggml type id
_ST_DTYPE_TO_GGML = {
    "F32": 0,
    "F16": 1,
    "BF16": 30,
    "I8": 24,
    "I16": 25,
    "I32": 26,
    "I64": 27,
    "F64": 28,
}

_ST_DTYPE_NUMPY = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # decoded manually
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F64": np.float64,
}


def _build_gguf_metadata_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build GGUF-compatible metadata from a HuggingFace config.json."""
    model_type = config.get("model_type", "llama")

    # Map HF model_type to GGUF architecture name
    arch_map = {
        "llama": "llama", "qwen2": "qwen2", "qwen2_moe": "qwen2moe",
        "mistral": "llama", "mixtral": "llama", "phi3": "phi3",
        "deepseek_v2": "deepseek2", "gemma": "gemma", "gemma2": "gemma2",
        "starcoder2": "starcoder2", "cohere": "command-r",
    }
    arch = arch_map.get(model_type, "llama")

    meta: Dict[str, Any] = {}
    meta["general.architecture"] = arch
    meta["general.name"] = config.get("_name_or_path", model_type)

    # Map config.json fields to GGUF metadata keys
    field_map = {
        "max_position_embeddings": f"{arch}.context_length",
        "hidden_size":             f"{arch}.embedding_length",
        "num_hidden_layers":       f"{arch}.block_count",
        "num_attention_heads":     f"{arch}.attention.head_count",
        "num_key_value_heads":     f"{arch}.attention.head_count_kv",
        "intermediate_size":       f"{arch}.feed_forward_length",
        "rope_theta":              f"{arch}.rope.freq_base",
        "rms_norm_eps":            f"{arch}.attention.layer_norm_rms_epsilon",
        "vocab_size":              f"{arch}.vocab_size",
    }

    for hf_key, gguf_key in field_map.items():
        if hf_key in config:
            val = config[hf_key]
            # GGUF expects integers for counts, floats for epsilon/theta
            if isinstance(val, float) and val == int(val) and "epsilon" not in hf_key and "theta" not in hf_key:
                val = int(val)
            meta[gguf_key] = val

    return meta


class SafetensorsSource(ModelSource):
    """
    Read tensors from a HuggingFace safetensors model directory.

    Accepts either:
    - A path to a single .safetensors file
    - A path to a directory containing .safetensors files + config.json
    """

    def __init__(self, path: str):
        if os.path.isfile(path) and path.endswith(".safetensors"):
            self._model_dir = os.path.dirname(path) or "."
            self._files = {path: None}  # header loaded lazily
        elif os.path.isdir(path):
            self._model_dir = path
            self._files = {}
        else:
            raise ValueError(f"Not a safetensors file or directory: {path}")

        self._tensor_map: Dict[str, Dict] = {}  # gguf_name -> info
        self._metadata: Dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        # Discover safetensors files
        if not self._files:
            index_path = os.path.join(self._model_dir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                for hf_name, filename in weight_map.items():
                    full = os.path.join(self._model_dir, filename)
                    self._files.setdefault(full, None)
            else:
                # Single file
                single = os.path.join(self._model_dir, "model.safetensors")
                if os.path.exists(single):
                    self._files[single] = None
                else:
                    raise FileNotFoundError(
                        f"No safetensors files found in {self._model_dir}"
                    )

        # Parse headers from all files
        for filepath in list(self._files.keys()):
            header, data_start = self._parse_header(filepath)
            self._files[filepath] = {"header": header, "data_start": data_start}

            for hf_name, info in header.items():
                if hf_name.startswith("__"):
                    continue
                gguf_name = _hf_name_to_gguf(hf_name)
                dtype = info.get("dtype", "F32")
                shape = info.get("shape", [])
                offsets = info.get("data_offsets", [0, 0])

                self._tensor_map[gguf_name] = {
                    "hf_name": hf_name,
                    "gguf_name": gguf_name,
                    "dtype": dtype,
                    "shape": list(reversed(shape)),  # reverse for GGUF convention
                    "shape_orig": shape,
                    "n_dims": len(shape),
                    "data_type": _ST_DTYPE_TO_GGML.get(dtype, 0),
                    "filepath": filepath,
                    "byte_offset": offsets[0],
                    "byte_length": offsets[1] - offsets[0],
                    "data_start": data_start,
                }

        # Load metadata from config.json
        config_path = os.path.join(self._model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self._metadata = _build_gguf_metadata_from_config(config)
        else:
            self._metadata = {"general.architecture": "llama"}

    @staticmethod
    def _parse_header(filepath: str) -> Tuple[Dict, int]:
        """Parse the safetensors header. Returns (header_dict, data_start_offset)."""
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            data_start = 8 + header_size
        header = json.loads(header_bytes)
        return header, data_start

    def get_metadata(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._metadata.copy()

    def get_tensor_names(self) -> List[str]:
        self._ensure_loaded()
        return list(self._tensor_map.keys())

    def get_all_tensors_info(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        result = []
        for gguf_name, info in self._tensor_map.items():
            result.append({
                "name": gguf_name,
                "n_dims": info["n_dims"],
                "shape": info["shape"],
                "data_type": info["data_type"],
                "offset": 0,  # not used — read_tensor_f32 handles offsets
            })
        return result

    def get_source_type_name(self, tensor_name: str) -> str:
        self._ensure_loaded()
        info = self._tensor_map.get(tensor_name)
        if info is None:
            return "F16"
        return info["dtype"]  # "F32", "F16", "BF16"

    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        self._ensure_loaded()
        info = self._tensor_map.get(tensor_name)
        if info is None:
            return None

        dtype = info["dtype"]
        np_dtype = _ST_DTYPE_NUMPY.get(dtype)
        if np_dtype is None:
            return None

        with open(info["filepath"], "rb") as f:
            f.seek(info["data_start"] + info["byte_offset"])
            buf = f.read(info["byte_length"])

        if dtype == "F32":
            return np.frombuffer(buf, dtype=np.float32).copy()
        elif dtype == "F16":
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif dtype == "BF16":
            raw = np.frombuffer(buf, dtype=np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            # Integer types — cast to float32
            return np.frombuffer(buf, dtype=np_dtype).astype(np.float32)

    def close(self):
        pass


# =====================================================================
# Factory
# =====================================================================

def open_model_source(path: str) -> ModelSource:
    """
    Open a model source, auto-detecting the format.

    Accepts:
    - A .gguf file path -> GGUFSource
    - A .safetensors file path -> SafetensorsSource
    - A directory containing .safetensors files -> SafetensorsSource
    """
    if os.path.isfile(path):
        if path.endswith(".gguf"):
            return GGUFSource(path)
        if path.endswith(".safetensors"):
            return SafetensorsSource(path)
        # Try to detect by magic
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"GGUF":
            return GGUFSource(path)
        # Assume safetensors
        return SafetensorsSource(path)

    if os.path.isdir(path):
        # Check for safetensors files
        has_st = any(
            f.endswith(".safetensors")
            for f in os.listdir(path)
        )
        if has_st:
            return SafetensorsSource(path)
        # Check for GGUF
        gguf_files = [f for f in os.listdir(path) if f.endswith(".gguf")]
        if gguf_files:
            return GGUFSource(os.path.join(path, gguf_files[0]))

    raise ValueError(f"Cannot detect model format for: {path}")
