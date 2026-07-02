"""
Model Source Abstraction - Unified interface for reading GGUF and safetensors.

Both GGUFSource and SafetensorsSource expose the same API so the writer
can consume either format transparently.
"""

from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import struct
import json
import logging
import os
import re
import numpy as np

_log = logging.getLogger(__name__)


def _flatten_to_max_dims(shape: List[int], max_dims: int = 4) -> List[int]:
    """Normalize tensor shape for GGUF compatibility.

    1. Squeeze singleton (size-1) inner dimensions.  For example,
       Conv1d weights [8192, 1, 4] become [8192, 4].
    2. Merge trailing dimensions so len(shape) <= max_dims.  GGUF only
       supports up to GGML_MAX_DIMS (4) dimensions.  For example,
       Conv3d weights [1152, 3, 2, 16, 16] become [1152, 3, 2, 256].

    The total element count is always preserved.
    """
    # Step 1: squeeze singleton dims (keep first and last dims intact
    # to avoid collapsing a scalar or vector)
    if len(shape) > 2:
        squeezed = [shape[0]]
        for d in shape[1:-1]:
            if d != 1:
                squeezed.append(d)
        squeezed.append(shape[-1])
        shape = squeezed

    # Step 2: merge trailing dims if still > max_dims
    if len(shape) <= max_dims:
        return shape
    keep = shape[:max_dims - 1]
    merge = shape[max_dims - 1:]
    merged = 1
    for d in merge:
        merged *= d
    return keep + [merged]


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
        28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4",
    }

    def __init__(self, filepath: str):
        from magicquant.gguf.reader import GGUFReader
        self._path = filepath
        self._reader = GGUFReader(filepath)
        self._reader.open()
        self._data_offset = self._reader.data_offset

    def get_metadata(self):
        return self._reader.get_metadata()

    def get_tensor_names(self):
        return self._reader.get_tensor_names()

    def get_all_tensors_info(self):
        return self._reader.get_all_tensors_info()

    def get_source_type_name(self, tensor_name: str) -> str:
        info = self._reader.get_tensor_info(tensor_name)
        if info is None:
            _log.warning("Tensor '%s' not found in GGUF source", tensor_name)
            return "UNKNOWN"
        type_name = self._TYPE_NAME.get(info["data_type"])
        if type_name is None:
            _log.warning(
                "Tensor '%s' has unknown ggml type id %d — cannot decode",
                tensor_name, info["data_type"],
            )
            return f"UNKNOWN({info['data_type']})"
        return type_name

    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        from magicquant.quant.converters import ggml_tensor_data_size
        info = self._reader.get_tensor_info(tensor_name)
        if info is None:
            return None
        type_name = self._TYPE_NAME.get(info["data_type"])
        if type_name is None:
            # Unknown ggml type — cannot decode
            return None
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
    # Attention Q/K norms (Qwen3.5 full-attention layers, also Cohere/Gemma)
    (r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q_norm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_k_norm.weight"),
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
    # Granite MoE Hybrid: fused expert tensors + shared MLP + Mamba
    (r"^model\.layers\.(\d+)\.block_sparse_moe\.input_linear\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_up_exps.weight"),
    (r"^model\.layers\.(\d+)\.block_sparse_moe\.output_linear\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_down_exps.weight"),
    (r"^model\.layers\.(\d+)\.block_sparse_moe\.router\.layer\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_inp.weight"),
    (r"^model\.layers\.(\d+)\.shared_mlp\.input_linear\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_up_shared.weight"),
    (r"^model\.layers\.(\d+)\.shared_mlp\.output_linear\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_down_shared.weight"),
    # Granite Mamba layers
    (r"^model\.layers\.(\d+)\.mamba\.in_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_in.weight"),
    (r"^model\.layers\.(\d+)\.mamba\.out_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_out.weight"),
    (r"^model\.layers\.(\d+)\.mamba\.conv1d\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_conv1d.weight"),
    (r"^model\.layers\.(\d+)\.mamba\.conv1d\.bias$",
     lambda m: f"blk.{m.group(1)}.ssm_conv1d.bias"),
    (r"^model\.layers\.(\d+)\.mamba\.dt_bias$",
     lambda m: f"blk.{m.group(1)}.ssm_dt.bias"),
    (r"^model\.layers\.(\d+)\.mamba\.A_log$",
     lambda m: f"blk.{m.group(1)}.ssm_a"),
    (r"^model\.layers\.(\d+)\.mamba\.D$",
     lambda m: f"blk.{m.group(1)}.ssm_d"),
    (r"^model\.layers\.(\d+)\.mamba\.norm\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_norm.weight"),
    # Qwen3.5 linear attention (SSM/Mamba-style) layers
    (r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_qkv.weight"),
    (r"^model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_gate.weight"),
    (r"^model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_alpha.weight"),
    (r"^model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_beta.weight"),
    (r"^model\.layers\.(\d+)\.linear_attn\.A_log$",
     lambda m: f"blk.{m.group(1)}.ssm_a"),
    (r"^model\.layers\.(\d+)\.linear_attn\.conv1d\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_conv1d.weight"),
    (r"^model\.layers\.(\d+)\.linear_attn\.dt_bias$",
     lambda m: f"blk.{m.group(1)}.ssm_dt.bias"),
    (r"^model\.layers\.(\d+)\.linear_attn\.norm\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_norm.weight"),
    (r"^model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ssm_out.weight"),
]

_HF_TO_GGUF_COMPILED = [(re.compile(p), r) for p, r in _HF_TO_GGUF_PATTERNS]


def _hf_name_to_gguf(hf_name: str, arch: str = "") -> str:
    """Convert a HuggingFace tensor name to GGUF convention.

    Args:
        hf_name: The original HuggingFace tensor name.
        arch: GGUF architecture string (e.g. "qwen35") for arch-specific
              name adjustments.
    """
    # Handle top-level output/lm_head directly
    if hf_name in ("output.weight", "lm_head.weight"):
        return "output.weight"

    # Bias tensors share their projection's mapping (the patterns below only cover
    # .weight). Map the corresponding .weight name and swap the suffix, so e.g.
    # Qwen2's q/k/v `*_proj.bias` becomes `blk.N.attn_{q,k,v}.bias` (llama.cpp
    # requires these for qkv-bias architectures; without it the GGUF won't load).
    if hf_name.endswith(".bias"):
        weight_name = hf_name[: -len(".bias")] + ".weight"
        mapped = _hf_name_to_gguf(weight_name, arch)
        if mapped != weight_name and mapped.endswith(".weight"):
            return mapped[: -len(".weight")] + ".bias"
        return hf_name  # projection's .weight didn't map -> leave bias untouched

    # Strip common multimodal prefixes so patterns match the LLM core
    stripped = hf_name
    for prefix in ("model.language_model.", "language_model."):
        if stripped.startswith(prefix):
            stripped = "model." + stripped[len(prefix):]
            break

    for pattern, replacement in _HF_TO_GGUF_COMPILED:
        m = pattern.match(stripped)
        if m:
            if callable(replacement):
                result = replacement(m)
            else:
                result = replacement
            # Architecture-specific name adjustments:
            # Qwen3.5 uses "post_attention_norm" instead of "ffn_norm"
            if arch in ("qwen35", "qwen35moe") and ".ffn_norm." in result:
                result = result.replace(".ffn_norm.", ".post_attention_norm.")
            return result
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
    # For multimodal/composite models, the LLM config is nested
    # under text_config, language_config, or llm_config.
    effective = config
    for sub_key in ("text_config", "language_config", "llm_config"):
        if sub_key in config and isinstance(config[sub_key], dict):
            effective = {**config, **config[sub_key]}
            break

    model_type = effective.get("model_type", "llama")

    # Synced from llama.cpp convert_hf_to_gguf.py (2026-03)
    arch_map = {
        "arctic": "arctic", "baichuan": "baichuan", "bloom": "bloom",
        "chatglm": "chatglm", "cohere": "command-r", "cohere2": "cohere2",
        "dbrx": "dbrx", "deepseek": "deepseek", "deepseek_v2": "deepseek2",
        "deepseek_v3": "deepseek2", "exaone": "exaone",
        "falcon": "falcon", "falcon_h1": "falcon-h1",
        "falcon_mamba": "mamba", "gemma": "gemma", "gemma2": "gemma2",
        "gemma3": "gemma3", "glm4": "glm4", "gpt2": "gpt2",
        "gpt_neox": "gptneox", "granite": "granite",
        "granitemoe": "granitemoe", "granitemoehybrid": "granitehybrid",
        "grok": "grok",
        "internlm2": "internlm2", "internlm3": "llama",
        "jamba": "jamba", "llama": "llama", "llama4": "llama4",
        "mamba": "mamba", "mamba2": "mamba2", "minicpm": "minicpm",
        "minicpm3": "minicpm3", "mistral": "llama", "mistral3": "mistral3",
        "mixtral": "llama", "nemotron": "nemotron",
        "olmo": "olmo", "olmo2": "olmo2", "olmoe": "olmoe",
        "phi": "phi2", "phi3": "phi3", "phimoe": "phimoe",
        "qwen": "qwen", "qwen2": "qwen2", "qwen2_moe": "qwen2moe",
        "qwen2_vl": "qwen2vl", "qwen3": "qwen3", "qwen3_5": "qwen35",
        "qwen3_5_text": "qwen35", "qwen3_5_moe": "qwen35moe",
        "qwen3_moe": "qwen3moe", "rwkv6": "rwkv6", "rwkv7": "rwkv7",
        "stablelm": "stablelm", "starcoder": "starcoder",
        "starcoder2": "starcoder2",
    }
    arch = arch_map.get(model_type)
    if arch is None:
        _log.warning(
            "Unknown model_type '%s' not in arch_map — defaulting to 'llama'. "
            "GGUF metadata keys may be wrong; consider adding a mapping or "
            "using a pre-converted GGUF source.",
            model_type,
        )
        arch = "llama"

    meta: Dict[str, Any] = {}
    meta["general.architecture"] = arch
    meta["general.name"] = effective.get("_name_or_path", config.get("_name_or_path", model_type))

    # Map config.json fields to GGUF metadata keys.
    # Note: vocab_size is intentionally omitted -- llama.cpp infers it
    # from the tokenizer token count.  Setting it explicitly causes
    # mismatches for multimodal models with padded vocabularies.
    field_map = {
        "max_position_embeddings": f"{arch}.context_length",
        "hidden_size":             f"{arch}.embedding_length",
        "num_hidden_layers":       f"{arch}.block_count",
        "num_attention_heads":     f"{arch}.attention.head_count",
        "num_key_value_heads":     f"{arch}.attention.head_count_kv",
        "intermediate_size":       f"{arch}.feed_forward_length",
        "rope_theta":              f"{arch}.rope.freq_base",
        "rms_norm_eps":            f"{arch}.attention.layer_norm_rms_epsilon",
    }

    for hf_key, gguf_key in field_map.items():
        if hf_key in effective:
            val = effective[hf_key]
            # GGUF expects integers for counts, floats for epsilon/theta
            if isinstance(val, float) and val == int(val) and "epsilon" not in hf_key and "theta" not in hf_key:
                val = int(val)
            meta[gguf_key] = val

    # transformers >=5 nests rope_theta inside ``rope_parameters`` and drops the
    # flat ``rope_theta`` field, so the field_map above misses it for any model
    # re-saved by a recent transformers (e.g. a merged/QAT model). Fall back to
    # rope_parameters.rope_theta for ANY arch — without it the GGUF gets the default
    # RoPE base and the model outputs garbage (Qwen2.5 needs 1e6, not the 1e4 default).
    if f"{arch}.rope.freq_base" not in meta:
        rope_params = effective.get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta")
        if rope_theta is not None:
            meta[f"{arch}.rope.freq_base"] = float(rope_theta)

    # ── Architecture-specific metadata ──
    # Qwen3.5 requires several additional keys that llama.cpp checks for:
    #   - rope.dimension_sections (MRoPE sections from rope_parameters)
    #   - rope.freq_base, rope.dimension_count
    #   - attention.key_length, attention.value_length
    #   - full_attention_interval (hybrid attention pattern)
    #   - ssm.* fields (for linear attention / Mamba-style layers)
    if arch in ("qwen35", "qwen35moe"):
        rope_params = effective.get("rope_parameters", {})

        # MRoPE dimension sections [time, height, width, extra] -- padded to 4
        mrope = rope_params.get("mrope_section", [])
        if mrope:
            sections = list(mrope)
            while len(sections) < 4:
                sections.append(0)
            meta[f"{arch}.rope.dimension_sections"] = sections[:4]

        # rope.freq_base from rope_parameters.rope_theta (takes priority
        # over the generic field_map which reads the top-level rope_theta)
        rope_theta = rope_params.get("rope_theta")
        if rope_theta is not None:
            meta[f"{arch}.rope.freq_base"] = float(rope_theta)

        # rope.dimension_count = partial_rotary_factor * head_dim
        head_dim = effective.get("head_dim",
                                 effective.get("hidden_size", 0) //
                                 max(effective.get("num_attention_heads", 1), 1))
        partial_rotary = effective.get("partial_rotary_factor",
                                       rope_params.get("partial_rotary_factor", 1.0))
        rope_dim_count = int(partial_rotary * head_dim)
        if rope_dim_count > 0:
            meta[f"{arch}.rope.dimension_count"] = rope_dim_count

        # attention key/value lengths
        if head_dim > 0:
            meta[f"{arch}.attention.key_length"] = head_dim
            meta[f"{arch}.attention.value_length"] = head_dim

        # full_attention_interval (hybrid attention pattern)
        fai = effective.get("full_attention_interval")
        if fai is not None:
            meta[f"{arch}.full_attention_interval"] = int(fai)

        # SSM / linear attention fields
        linear_key_head_dim = effective.get("linear_key_head_dim")
        linear_num_key_heads = effective.get("linear_num_key_heads")
        linear_num_value_heads = effective.get("linear_num_value_heads")
        linear_value_head_dim = effective.get("linear_value_head_dim")
        conv_kernel = effective.get("linear_conv_kernel_dim")

        if linear_key_head_dim is not None and linear_num_key_heads is not None:
            meta[f"{arch}.ssm.state_size"] = int(linear_key_head_dim)
            meta[f"{arch}.ssm.group_count"] = int(linear_num_key_heads)
        if linear_num_value_heads is not None and linear_value_head_dim is not None:
            meta[f"{arch}.ssm.inner_size"] = int(linear_num_value_heads * linear_value_head_dim)
            meta[f"{arch}.ssm.time_step_rank"] = int(linear_num_value_heads)
        if conv_kernel is not None:
            meta[f"{arch}.ssm.conv_kernel"] = int(conv_kernel)

    return meta


# GGUF architectures whose Q/K projections llama.cpp stores rope-PERMUTED
# ("NORM"-style / rope type 0, interleaved pairs). HF safetensors keep the
# half-split rotary layout, so these arches need the converter's permutation;
# NEOX-rope arches (qwen2, gemma, phi3, ...) consume the HF layout directly.
# Mirrors LlamaModel.permute in llama.cpp's convert_hf_to_gguf.py. Note
# model_type mistral/mixtral/internlm3 all map to GGUF arch "llama".
_QK_PERMUTED_ARCHS = {"llama", "baichuan"}


def _permute_qk_rows(weights: np.ndarray, n_head: int) -> np.ndarray:
    """Reorder Q/K output rows from HF half-split rotary layout to llama.cpp's
    interleaved layout (llama.cpp converter's ``LlamaModel.permute``).

    Works for 2-D weights (out, in) and 1-D biases (out,). Pure row reorder —
    values are never mixed. Without this, every llama-arch pack had scrambled
    attention (Llama-3.2-1B f16: PPL ~1725 vs 18.9 for the reference convert;
    proven byte-exact: permute(ours) == reference, V tensors identical).
    """
    out_dim = weights.shape[0]
    rest = weights.shape[1:]
    return np.ascontiguousarray(
        weights.reshape(n_head, 2, out_dim // n_head // 2, *rest)
               .swapaxes(1, 2)
    ).reshape(weights.shape)


def _normalize_merges(merges: list) -> list:
    """Normalize BPE merges to llama.cpp's space-joined string form.

    transformers <5 stored each merge in tokenizer.json as a space-joined string
    ("Ġ Ġ"); transformers >=5 stores it as a pair-array (["Ġ", "Ġ"]). llama.cpp's
    GGUF BPE loader only understands the string form — it splits each merge on the
    first space to recover the pair. If the pair-array form is written verbatim it
    lands in the GGUF as a Python list repr ("['Ġ', 'Ġ']"), BPE merging silently
    fails, and any model re-saved by a recent transformers (e.g. a merged QAT
    model) tokenizes to garbage even though its weights are byte-identical to a
    working model.
    """
    normalized = []
    for m in merges:
        if isinstance(m, (list, tuple)):
            normalized.append(" ".join(m))
        else:
            normalized.append(m)
    return normalized


# Map HuggingFace pre_tokenizer Split regexes -> llama.cpp's ``tokenizer.ggml.pre``
# names. These regexes are copied verbatim across a model family, so an exact match
# reliably identifies the pre-tokenizer. llama.cpp picks its splitting regex from
# this name; without it llama.cpp falls back to 'default' and prints "GENERATION
# QUALITY WILL BE DEGRADED!", tokenizing text wrongly (perplexity inflates badly).
_PRETOK_REGEX_TO_PRE = {
    # Qwen2 / Qwen2.5 (also deepseek-r1-qwen) — note the bare ``\p{N}``.
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "qwen2",
    # Llama-3 and the many llama-bpe descendants — ``\p{N}{1,3}`` groups digits.
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "llama-bpe",
    # GPT-2 / GPT-NeoX / Falcon / MPT / OLMo family (the classic GPT-2 regex).
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+": "gpt-2",
}


def _detect_tokenizer_pre(tok_json: Dict[str, Any]):
    """Identify llama.cpp's ``tokenizer.ggml.pre`` name from a tokenizer.json.

    Returns the canonical pre name (e.g. ``"qwen2"``) or ``None`` if the
    pre_tokenizer regex isn't recognized (caller should leave the key unset so
    llama.cpp surfaces its own degradation warning rather than us masking it).
    """
    pre = tok_json.get("pre_tokenizer")
    if not isinstance(pre, dict):
        return None
    # pre_tokenizer is either a single Split or a Sequence of pre-tokenizers.
    candidates = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
    for c in candidates:
        if isinstance(c, dict) and c.get("type") == "Split":
            pattern = c.get("pattern", {})
            regex = pattern.get("Regex") if isinstance(pattern, dict) else None
            if regex is not None:
                return _PRETOK_REGEX_TO_PRE.get(regex)
    return None


def _build_tokenizer_metadata(model_dir: str) -> Dict[str, Any]:
    """
    Read tokenizer data from a HuggingFace model directory and return
    GGUF-compatible tokenizer metadata.

    Handles the common case: BPE tokenizer from tokenizer.json
    (covers LLaMA, Qwen, Mistral, GPT-NeoX, Falcon, etc.).
    """
    meta: Dict[str, Any] = {}

    # ── tokenizer.json (BPE vocab + merges) ──
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        return meta

    with open(tokenizer_path, encoding="utf-8") as f:
        tok = json.load(f)

    model_info = tok.get("model", {})
    tok_type = model_info.get("type", "BPE")

    if tok_type == "BPE":
        meta["tokenizer.ggml.model"] = "gpt2"
    elif tok_type == "Unigram":
        meta["tokenizer.ggml.model"] = "llama"
    else:
        meta["tokenizer.ggml.model"] = "gpt2"

    # Pre-tokenizer type. llama.cpp REQUIRES this to pick the correct splitting
    # regex; without it, it warns "GENERATION QUALITY WILL BE DEGRADED" and
    # tokenizes with the wrong regex (perplexity inflates). Leave the key unset
    # when unrecognized so llama.cpp's own warning still surfaces.
    pre = _detect_tokenizer_pre(tok)
    if pre is not None:
        meta["tokenizer.ggml.pre"] = pre
    else:
        _log.warning(
            "Unrecognized BPE pre-tokenizer regex — leaving tokenizer.ggml.pre "
            "unset. llama.cpp will warn 'GENERATION QUALITY WILL BE DEGRADED'. "
            "Add the regex to _PRETOK_REGEX_TO_PRE in gguf/source.py to fix."
        )

    # Extract vocabulary. BPE stores it as a {token: id} dict; Unigram (SPM)
    # stores a LIST of [token, score] pairs where the id is the list index.
    # Calling .items() on the list form used to crash with AttributeError.
    vocab = model_info.get("vocab", {})
    unigram_scores: Dict[int, float] = {}
    if isinstance(vocab, list):
        sorted_tokens = []
        for idx, entry in enumerate(vocab):
            if isinstance(entry, (list, tuple)) and entry:
                sorted_tokens.append((entry[0], idx))
                if len(entry) > 1 and isinstance(entry[1], (int, float)):
                    unigram_scores[idx] = float(entry[1])
            else:
                sorted_tokens.append((entry, idx))
        vocab = sorted_tokens  # truthy guard below
    else:
        sorted_tokens = sorted(vocab.items(), key=lambda x: x[1]) if vocab else []
    if vocab:
        max_id = sorted_tokens[-1][1] if sorted_tokens else 0

        # Added tokens may have IDs beyond the base vocab (e.g. Qwen3.5
        # special tokens at 248044+).  Also, config.json vocab_size may be
        # larger still (padding for alignment).  Allocate enough room for
        # all of them.
        added = tok.get("added_tokens", [])
        if added:
            max_added_id = max(at.get("id", -1) for at in added)
            max_id = max(max_id, max_added_id)

        # If a config.json exists, use its vocab_size to pad the token
        # list so it matches the embedding tensor dimension.
        config_path_for_vocab = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path_for_vocab):
            with open(config_path_for_vocab) as _f:
                _cfg = json.load(_f)
            # Resolve nested text_config for multimodal models
            _eff = _cfg
            for _sub in ("text_config", "language_config", "llm_config"):
                if _sub in _cfg and isinstance(_cfg[_sub], dict):
                    _eff = {**_cfg, **_cfg[_sub]}
                    break
            cfg_vocab_size = _eff.get("vocab_size", 0)
            if cfg_vocab_size > max_id + 1:
                max_id = cfg_vocab_size - 1

        tokens = [""] * (max_id + 1)
        scores = [0.0] * (max_id + 1)
        token_types = [0] * (max_id + 1)  # 0 = normal

        for token_str, token_id in sorted_tokens:
            if token_id < len(tokens):
                tokens[token_id] = token_str
                if token_id in unigram_scores:
                    scores[token_id] = unigram_scores[token_id]

        # Fill in added_tokens (special tokens with IDs beyond base vocab)
        for at in added:
            tid = at.get("id", -1)
            content = at.get("content", "")
            special = at.get("special", False)
            if 0 <= tid < len(tokens):
                tokens[tid] = content
                if special:
                    token_types[tid] = 3  # 3 = control token

        meta["tokenizer.ggml.tokens"] = tokens
        meta["tokenizer.ggml.scores"] = scores
        meta["tokenizer.ggml.token_type"] = token_types

    # Extract BPE merges (normalizing transformers>=5's pair-array format back
    # to the space-joined string form llama.cpp's GGUF BPE loader requires).
    merges = model_info.get("merges", [])
    if merges:
        meta["tokenizer.ggml.merges"] = _normalize_merges(merges)

    # ── tokenizer_config.json (special token IDs) ──
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            tok_cfg = json.load(f)

        # Map special token config keys to GGUF metadata keys
        special_map = {
            "bos_token": "tokenizer.ggml.bos_token_id",
            "eos_token": "tokenizer.ggml.eos_token_id",
            "pad_token": "tokenizer.ggml.padding_token_id",
            "unk_token": "tokenizer.ggml.unknown_token_id",
        }

        # Build a complete token->id lookup including added tokens
        # (special tokens like <|im_end|> are often only in added_tokens,
        # not in the base BPE vocab)
        all_token_ids = dict(vocab)
        for at in added:
            content = at.get("content", "")
            tid = at.get("id", -1)
            if content and tid >= 0:
                all_token_ids[content] = tid

        for hf_key, gguf_key in special_map.items():
            val = tok_cfg.get(hf_key)
            if val is None:
                continue
            # Value can be a string or a dict with "content" key
            if isinstance(val, dict):
                val = val.get("content", "")
            if isinstance(val, str) and val in all_token_ids:
                meta[gguf_key] = all_token_ids[val]

        # Whether to prepend BOS at tokenization time. llama.cpp otherwise
        # applies its own per-arch default (often True), which silently corrupts
        # perplexity for models that don't use BOS (e.g. Qwen has it False).
        if "add_bos_token" in tok_cfg:
            meta["tokenizer.ggml.add_bos_token"] = bool(tok_cfg["add_bos_token"])
        if "add_eos_token" in tok_cfg and tok_cfg["add_eos_token"] is not None:
            meta["tokenizer.ggml.add_eos_token"] = bool(tok_cfg["add_eos_token"])

        # Chat template
        chat_template = tok_cfg.get("chat_template")
        if isinstance(chat_template, list):
            # Find the "default" template, or use the first one
            for entry in chat_template:
                if isinstance(entry, dict):
                    if entry.get("name") == "default":
                        chat_template = entry.get("template", "")
                        break
            else:
                if chat_template and isinstance(chat_template[0], dict):
                    chat_template = chat_template[0].get("template", "")
        if isinstance(chat_template, str) and chat_template:
            meta["tokenizer.chat_template"] = chat_template

    # Fallback: transformers >= 4.44 stores the chat template in a standalone
    # chat_template.jinja (or legacy chat_template.json) file, not in
    # tokenizer_config.json. Without this, GGUFs ship with no
    # tokenizer.chat_template and can't be chatted/tool-called without a manual
    # patch — the known Foundry "GGUF needs chat-template patching" issue.
    if "tokenizer.chat_template" not in meta:
        jinja_path = os.path.join(model_dir, "chat_template.jinja")
        json_path = os.path.join(model_dir, "chat_template.json")
        if os.path.exists(jinja_path):
            with open(jinja_path, encoding="utf-8") as f:
                tmpl = f.read().strip()
            if tmpl:
                meta["tokenizer.chat_template"] = tmpl
        elif os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                tmpl = data.get("chat_template") if isinstance(data, dict) else None
                if isinstance(tmpl, str) and tmpl.strip():
                    meta["tokenizer.chat_template"] = tmpl.strip()
            except (json.JSONDecodeError, OSError):
                pass

    # A template file that exists but yielded nothing is worth flagging — the
    # resulting GGUF would silently lack a usable chat template.
    if "tokenizer.chat_template" not in meta:
        for name in ("chat_template.jinja", "chat_template.json"):
            if os.path.exists(os.path.join(model_dir, name)):
                _log.warning(
                    "chat template file %s present in %s but no template emitted",
                    name, model_dir,
                )
                break

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

        # Load metadata from config.json first — we need the architecture
        # to apply arch-specific tensor name mappings.
        config_path = os.path.join(self._model_dir, "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self._metadata = _build_gguf_metadata_from_config(config)
        else:
            self._metadata = {"general.architecture": "llama"}

        arch = self._metadata.get("general.architecture", "llama")

        # Rope permutation setup for NORM-rope arches (see _QK_PERMUTED_ARCHS):
        # resolve head counts from the (text_config-aware) effective config.
        self._qk_heads = None
        if arch in _QK_PERMUTED_ARCHS:
            effective = config
            for sub_key in ("text_config", "language_config", "llm_config"):
                if sub_key in config and isinstance(config[sub_key], dict):
                    effective = {**config, **config[sub_key]}
                    break
            n_head = effective.get("num_attention_heads")
            n_kv = effective.get("num_key_value_heads", n_head)
            if n_head:
                self._qk_heads = {"q": int(n_head), "k": int(n_kv or n_head)}
            else:
                _log.warning(
                    "arch '%s' needs Q/K rope permutation but config has no "
                    "num_attention_heads — packing UNPERMUTED (model will be "
                    "broken in llama.cpp).", arch,
                )

        # Parse headers from all files
        for filepath in list(self._files.keys()):
            header, data_start = self._parse_header(filepath)
            self._files[filepath] = {"header": header, "data_start": data_start}

            for hf_name, info in header.items():
                if hf_name.startswith("__"):
                    continue

                # Strip multimodal prefixes before mapping
                stripped = hf_name
                for prefix in ("model.language_model.", "language_model."):
                    if stripped.startswith(prefix):
                        stripped = "model." + stripped[len(prefix):]
                        break

                # Skip vision encoder and MTP (multi-token prediction)
                # tensors — vision tensors belong in a separate mmproj GGUF,
                # and MTP tensors are not used by llama.cpp inference.
                # Including them causes assertion failures during load.
                if stripped.startswith("model.visual.") or hf_name.startswith("mtp."):
                    continue

                gguf_name = _hf_name_to_gguf(hf_name, arch=arch)
                dtype = info.get("dtype", "F32")
                shape = info.get("shape", [])
                offsets = info.get("data_offsets", [0, 0])

                # GGUF supports at most GGML_MAX_DIMS (4) dimensions.
                # Merge trailing dims for tensors that exceed this (e.g.
                # Conv3d patch_embed weights with shape [1152, 3, 2, 16, 16]).
                gguf_shape = _flatten_to_max_dims(list(shape), max_dims=4)

                self._tensor_map[gguf_name] = {
                    "hf_name": hf_name,
                    "gguf_name": gguf_name,
                    "dtype": dtype,
                    "shape": gguf_shape,  # row-major, at most 4-D
                    "shape_orig": list(shape),
                    "n_dims": len(gguf_shape),
                    "data_type": _ST_DTYPE_TO_GGML.get(dtype, 0),
                    "filepath": filepath,
                    "byte_offset": offsets[0],
                    "byte_length": offsets[1] - offsets[0],
                    "data_start": data_start,
                }

        # Handle tied weights: if output.weight is missing and embeddings are tied,
        # create a reference to token_embd.weight
        if "output.weight" not in self._tensor_map and "token_embd.weight" in self._tensor_map:
            if config.get("tie_word_embeddings", True):
                ref = dict(self._tensor_map["token_embd.weight"])
                ref["gguf_name"] = "output.weight"
                self._tensor_map["output.weight"] = ref

        # Load tokenizer data
        tokenizer_meta = _build_tokenizer_metadata(self._model_dir)
        self._metadata.update(tokenizer_meta)

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

    def get_qk_permute_heads(self, tensor_name: str) -> Optional[int]:
        """Head count to rope-permute ``tensor_name`` with, or None.

        Non-None only for attn_q/attn_k weights+biases of NORM-rope arches
        (see ``_QK_PERMUTED_ARCHS``). Exposed so wrappers that add deltas on
        top of the base weights (LoRAMergedSource) can permute their deltas
        identically.
        """
        self._ensure_loaded()
        heads = getattr(self, "_qk_heads", None)
        if not heads:
            return None
        base = tensor_name.rsplit(".", 1)[0]  # strip .weight/.bias
        if base.endswith(".attn_q"):
            return heads["q"]
        if base.endswith(".attn_k"):
            return heads["k"]
        return None

    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        self._ensure_loaded()
        info = self._tensor_map.get(tensor_name)
        if info is None:
            return None

        dtype = info["dtype"]
        np_dtype = _ST_DTYPE_NUMPY.get(dtype)
        if np_dtype is None:
            return None

        # Use memory-mapped I/O for zero-copy reads
        mmap = self._get_mmap(info["filepath"])
        start = info["data_start"] + info["byte_offset"]
        end = start + info["byte_length"]
        buf = mmap[start:end]

        if dtype == "F32":
            flat = np.frombuffer(buf, dtype=np.float32).copy()
        elif dtype == "F16":
            flat = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif dtype == "BF16":
            raw = np.frombuffer(buf, dtype=np.uint16)
            flat = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            flat = np.frombuffer(buf, dtype=np_dtype).astype(np.float32)

        # NORM-rope arches: llama.cpp expects Q/K rows interleaved.
        n_head = self.get_qk_permute_heads(tensor_name)
        if n_head:
            shaped = flat.reshape(info["shape"])
            flat = _permute_qk_rows(shaped, n_head).reshape(-1)
        return flat

    def _get_mmap(self, filepath: str):
        """Get or create a memory-mapped view of a safetensors file."""
        if not hasattr(self, "_mmaps"):
            self._mmaps: Dict[str, Any] = {}
            self._mmap_files: Dict[str, Any] = {}
        if filepath not in self._mmaps:
            import mmap
            f = open(filepath, "rb")
            self._mmap_files[filepath] = f
            self._mmaps[filepath] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[filepath]

    def close(self):
        for mm in getattr(self, "_mmaps", {}).values():
            mm.close()
        for f in getattr(self, "_mmap_files", {}).values():
            f.close()
        self._mmaps = {}
        self._mmap_files = {}


# =====================================================================
# LoRA Merged Source
# =====================================================================

class LoRAMergedSource(ModelSource):
    """
    Wraps a base model source and merges LoRA adapter weights on-the-fly.

    For each tensor that has a LoRA delta (lora_A + lora_B matrices), the
    merge formula is:
        W_merged = W_base + (lora_B @ lora_A) * (alpha / rank)

    Tensors without LoRA adapters pass through from the base model unchanged.
    No full merged copy is written to disk — merging happens per-tensor as
    the writer reads each one.
    """

    def __init__(self, base_path: str, adapter_path: str):
        """
        Args:
            base_path: Path to the base model (directory or .safetensors/.gguf)
            adapter_path: Path to the LoRA adapter directory (contains
                adapter_config.json + adapter_model.safetensors)
        """
        from magicquant.gguf.source import open_model_source

        self._base = open_model_source(base_path)

        # Capture the base model's architecture so LoRA tensor-name mapping
        # picks up arch-specific adjustments (e.g. Qwen3.5 ffn_norm renaming).
        try:
            self._base_arch = self._base.get_metadata().get(
                "general.architecture", ""
            )
        except Exception:
            self._base_arch = ""

        # Load adapter config
        if os.path.isdir(adapter_path):
            adapter_dir = adapter_path
        else:
            adapter_dir = os.path.dirname(adapter_path)

        config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

        with open(config_path) as f:
            adapter_cfg = json.load(f)

        self._rank = adapter_cfg.get("r", 8)
        self._alpha = adapter_cfg.get("lora_alpha", self._rank)
        self._scale = self._alpha / self._rank
        self._fan_in_fan_out = adapter_cfg.get("fan_in_fan_out", False)
        self._target_modules = set(adapter_cfg.get("target_modules", []))

        # Load adapter tensors
        adapter_st = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_st):
            raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")

        self._adapter_tensors: Dict[str, Dict] = {}
        header, data_start = SafetensorsSource._parse_header(adapter_st)
        for name, info in header.items():
            if name.startswith("__"):
                continue
            self._adapter_tensors[name] = {
                "dtype": info.get("dtype", "F32"),
                "shape": info.get("shape", []),
                "filepath": adapter_st,
                "byte_offset": info["data_offsets"][0],
                "byte_length": info["data_offsets"][1] - info["data_offsets"][0],
                "data_start": data_start,
            }

        # Build map: base HF tensor name -> (lora_A_key, lora_B_key)
        self._lora_map: Dict[str, Tuple[str, str]] = {}
        lora_a_keys = [k for k in self._adapter_tensors if ".lora_A." in k]
        for a_key in lora_a_keys:
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            if b_key in self._adapter_tensors:
                # Extract the base tensor name:
                # "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
                # -> "model.layers.0.self_attn.q_proj.weight"
                base_name = a_key.replace(".lora_A.", ".")
                if base_name.startswith("base_model.model."):
                    base_name = base_name[len("base_model.model."):]
                elif base_name.startswith("base_model."):
                    base_name = base_name[len("base_model."):]
                # Convert to GGUF name (arch-aware, matching the base source).
                gguf_name = _hf_name_to_gguf(base_name, arch=self._base_arch)
                self._lora_map[gguf_name] = (a_key, b_key)

    def _read_adapter_tensor(self, key: str) -> np.ndarray:
        info = self._adapter_tensors[key]
        byte_offset = info["byte_offset"]
        byte_length = info["byte_length"]
        data_start = info["data_start"]

        # Bounds-validate against the actual file size before reading so a
        # malformed/malicious safetensors header can't drive an out-of-range
        # read (or a short read that silently reshapes garbage).
        if byte_offset < 0 or byte_length < 0:
            raise ValueError(
                f"Adapter tensor '{key}' has negative byte_offset/byte_length "
                f"({byte_offset}/{byte_length})."
            )
        file_size = os.path.getsize(info["filepath"])
        end = data_start + byte_offset + byte_length
        if end > file_size:
            raise ValueError(
                f"Adapter tensor '{key}' would read past EOF: "
                f"data_start({data_start}) + byte_offset({byte_offset}) + "
                f"byte_length({byte_length}) = {end} > file size {file_size}."
            )

        with open(info["filepath"], "rb") as f:
            f.seek(data_start + byte_offset)
            buf = f.read(byte_length)
        dtype = info["dtype"]
        if dtype == "BF16":
            raw = np.frombuffer(buf, dtype=np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
        elif dtype == "F16":
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(info["shape"])
        else:
            return np.frombuffer(buf, dtype=np.float32).copy().reshape(info["shape"])

    def get_metadata(self):
        return self._base.get_metadata()

    def get_tensor_names(self):
        return self._base.get_tensor_names()

    def get_all_tensors_info(self):
        return self._base.get_all_tensors_info()

    def get_source_type_name(self, tensor_name: str) -> str:
        return self._base.get_source_type_name(tensor_name)

    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        base_f32 = self._base.read_tensor_f32(tensor_name)
        if base_f32 is None:
            return None

        if tensor_name not in self._lora_map:
            return base_f32

        a_key, b_key = self._lora_map[tensor_name]
        lora_a = self._read_adapter_tensor(a_key)  # (rank, in_features)
        lora_b = self._read_adapter_tensor(b_key)  # (out_features, rank)

        # Merge: W = W_base + (B @ A) * scale
        delta = (lora_b @ lora_a) * self._scale
        # fan_in_fan_out: transpose delta for Conv1D-based models (GPT-2 style)
        if self._fan_in_fan_out:
            delta = delta.T

        # The adapter delta is in HF layout; if the base source rope-permuted
        # this tensor (llama-arch Q/K), permute the delta identically or the
        # merge would mix the two layouts.
        permute_heads = getattr(self._base, "get_qk_permute_heads", None)
        if permute_heads is not None:
            n_head = permute_heads(tensor_name)
            if n_head:
                delta = _permute_qk_rows(delta, n_head)

        # Shape guard: a mismatched delta would silently corrupt the merge
        # (reshape could broadcast/raise obscurely). Fail loud, naming the
        # tensor and both shapes.
        if base_f32.size != delta.size:
            raise ValueError(
                f"LoRA merge shape mismatch for tensor '{tensor_name}': "
                f"base has {base_f32.size} elements but delta (B@A) has "
                f"{delta.size} (delta shape {delta.shape}). Check the adapter's "
                f"rank/target_modules or fan_in_fan_out setting."
            )
        base_f32 = base_f32.reshape(delta.shape) + delta

        return base_f32.flatten()

    def close(self):
        self._base.close()


# =====================================================================
# Factory
# =====================================================================

def open_model_source(
    path: str,
    adapter_path: Optional[str] = None,
) -> ModelSource:
    """
    Open a model source, auto-detecting the format.

    Accepts:
    - A .gguf file path -> GGUFSource
    - A .safetensors file path -> SafetensorsSource
    - A directory containing .safetensors files -> SafetensorsSource
    - A LoRA adapter directory (adapter_config.json) -> LoRAMergedSource
      (auto-downloads or locates the base model)

    If *adapter_path* is given, the result wraps the base model with
    LoRA merge-on-read.
    """
    # If the path itself is a LoRA adapter directory, resolve the base
    if os.path.isdir(path):
        adapter_cfg = os.path.join(path, "adapter_config.json")
        if os.path.exists(adapter_cfg) and adapter_path is None:
            with open(adapter_cfg) as f:
                cfg = json.load(f)
            # base_model_name_or_path comes from an untrusted file. We only
            # follow it if it resolves to an EXISTING LOCAL DIRECTORY (never a
            # HF repo id / URL — those would require an explicit override).
            base_model = cfg.get("base_model_name_or_path", "")
            if base_model and os.path.isabs(base_model) and os.path.isdir(base_model):
                return LoRAMergedSource(base_path=base_model, adapter_path=path)
            # Relative paths are resolved against the adapter directory, not the
            # CWD, to avoid surprising lookups.
            if base_model and not os.path.isabs(base_model):
                candidate = os.path.normpath(os.path.join(path, base_model))
                if os.path.isdir(candidate):
                    return LoRAMergedSource(base_path=candidate, adapter_path=path)
            raise ValueError(
                f"LoRA adapter detected at {path} but base model "
                f"'{base_model}' could not be resolved to a local directory. "
                f"Download it first or pass the base model path explicitly."
            )

    # Explicit adapter
    if adapter_path is not None:
        return LoRAMergedSource(base_path=path, adapter_path=adapter_path)

    # Standard format detection
    if os.path.isfile(path):
        if path.endswith(".gguf"):
            return GGUFSource(path)
        if path.endswith(".safetensors"):
            return SafetensorsSource(path)
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"GGUF":
            return GGUFSource(path)
        return SafetensorsSource(path)

    if os.path.isdir(path):
        has_st = any(f.endswith(".safetensors") for f in os.listdir(path))
        if has_st:
            return SafetensorsSource(path)
        gguf_files = [f for f in os.listdir(path) if f.endswith(".gguf")]
        if gguf_files:
            return GGUFSource(os.path.join(path, gguf_files[0]))

    raise ValueError(f"Cannot detect model format for: {path}")
