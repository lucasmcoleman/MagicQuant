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
import math
import os
import re
import numpy as np

from magicquant.quant import ggml_facts

_log = logging.getLogger(__name__)

# Opt-in: allow an ALREADY-QUANTIZED GGUF to be used as a source by
# dequantizing its tensors back to F32 via libggml's dequantize_row_* kernels.
#
# This is OFF by default and must stay that way. MagicQuant's whole premise is
# per-group sensitivity search against high-precision weights; a quantized
# source means every output is DOUBLE-quantized and the search optimizes
# against weights it cannot reconstruct. The default hard error (writer.py's
# "source is already quantized" guard) exists to stop that happening by
# accident.
#
# It is enabled deliberately for the one case where there is no alternative:
# a model published only as GGUF, with no BF16/F16 sibling and no safetensors
# release. Dequantizing a Q8_0 source (~8.5 bpw, per-32 block scales) and
# re-quantizing to a Q4 tier lands close to the same tier built from BF16; a
# Q5/Q6 tier off the same source cannot beat the source's own error floor and
# is largely pointless.
#
# The env var is the propagation mechanism (not a threaded kwarg) so that a
# single set in the calling stage reaches every internal open_model_source
# call site -- writer, orchestrator, probing, sensitivity -- and any
# subprocess they spawn. The explicit ``allow_dequant`` kwarg overrides it for
# direct API use and tests.
_ALLOW_DEQUANT_ENV = "MAGICQUANT_ALLOW_DEQUANT_SOURCE"

# Types read straight to float32 with no ggml kernel involved.
_NATIVE_FLOAT_TYPES = ("F32", "F16", "BF16")


def _allow_dequant_default() -> bool:
    """Resolve the default dequant-source policy from the environment."""
    return os.environ.get(_ALLOW_DEQUANT_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


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

    def can_decode(self, tensor_name: str) -> bool:
        """True if ``read_tensor_f32`` will return real data for this tensor.

        The writer uses this to decide between re-quantizing a tensor and
        passing its bytes through verbatim, so it must agree with
        ``read_tensor_f32`` exactly -- a False positive here writes a
        zero-filled blob. The default is the conservative answer (native
        float types only); ``GGUFSource`` widens it when dequantization of a
        quantized source is explicitly enabled.
        """
        return self.get_source_type_name(tensor_name) in _NATIVE_FLOAT_TYPES

    def read_tensor_raw(self, tensor_name: str) -> Optional[bytes]:
        """Return this tensor's exact on-disk bytes, undecoded.

        Used by the writer's passthrough path (source type == desired
        type), which needs a verbatim byte copy and must never go through
        ``read_tensor_f32`` -- with dequant enabled that can now RAISE for
        a recognized-but-undecodable type, which is the wrong failure mode
        for a straight copy. The default is None (most sources -- e.g.
        safetensors -- only ever expose decoded float data, so a copy path
        never applies to them); ``GGUFSource`` overrides this to return the
        tensor's raw block bytes.
        """
        return None

    def close(self):
        pass


# =====================================================================
# GGUF Source
# =====================================================================

class GGUFSource(ModelSource):
    """Read tensors from a GGUF file."""

    # ggml_type id -> name. Derived from magicquant.quant.ggml_facts (stock
    # ids/names come from the installed `gguf` package — see that module's
    # docstring). Deliberately excludes the ROCmFPX fork ids (100-104):
    # a GGUF read as a *source* here needs decoding to F32 for re-encoding,
    # and MagicQuant has no fork-type dequant path (ggml_binding's fork
    # dequant symbols exist, but wiring a fork-quantized source through this
    # reader hasn't been validated) -- recognizing but mishandling those ids
    # would be worse than the existing "UNKNOWN(id)" fallback.
    _TYPE_NAME = {
        k: v for k, v in ggml_facts.ID_TO_NAME.items()
        if k not in {info["id"] for info in ggml_facts.FORK_TYPES.values()}
    }

    def __init__(self, filepath: str, allow_dequant: Optional[bool] = None):
        from magicquant.gguf.reader import GGUFReader
        self._path = filepath
        self._reader = GGUFReader(filepath)
        self._reader.open()
        self._data_offset = self._reader.data_offset
        # Cached read handle: re-quantizing a GGUF source touches every tensor,
        # so a per-tensor open/seek/close (thousands of syscalls on a big MoE)
        # is wasteful. Opened lazily, closed in close().
        self._fh = None
        # None -> take the environment default. See _ALLOW_DEQUANT_ENV.
        self._allow_dequant = (
            _allow_dequant_default() if allow_dequant is None else bool(allow_dequant)
        )
        # Types this source actually dequantized, for the writer's warning.
        self._dequantized_types: set = set()
        # Types whose dequant probe already failed+warned: the writer calls
        # can_decode() once per tensor in Pass 1, so without memoization a
        # broken libggml would emit thousands of identical warnings on a
        # big model. One warning per type per source is enough.
        self._probe_warned: set = set()

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

    def can_decode(self, tensor_name: str) -> bool:
        """True for native float types, plus quantized types when dequant is on.

        For a quantized type the answer additionally depends on the loaded
        libggml actually exporting that ``dequantize_row_*`` symbol -- an
        IQ3_S/IQ2_XXS tensor has no such kernel in a stock build, so those
        still fall through to the writer's pre-quantized hard error rather
        than being silently zero-filled.
        """
        type_name = self.get_source_type_name(tensor_name)
        if type_name in _NATIVE_FLOAT_TYPES:
            return True
        if not self._allow_dequant or type_name.startswith("UNKNOWN"):
            return False
        if type_name in self._probe_warned:
            return False
        from magicquant.quant.ggml_binding import supports_decode
        try:
            return supports_decode(type_name)
        except Exception as exc:  # libggml missing/unloadable -> not decodable
            self._probe_warned.add(type_name)
            _log.warning(
                "Dequant probe for type '%s' failed (%s); treating as "
                "undecodable", type_name, exc,
            )
            return False

    @property
    def dequantized_types(self) -> set:
        """ggml types this source dequantized on read (empty in the normal case)."""
        return set(self._dequantized_types)

    def _read_raw_bytes(self, tensor_name: str, byte_len: int, pos: int) -> bytes:
        """Gather exactly ``byte_len`` bytes for a tensor at file offset ``pos``.

        Shared by ``read_tensor_f32`` (which decodes the result) and
        ``read_tensor_raw`` (which returns it verbatim) -- both need the
        identical pread loop, so it's factored here rather than duplicated.

        os.pread is position-independent: the writer's parallel encode pool
        calls into this from N threads on this ONE shared handle, and a
        seek()+read() pair races (the GIL drops between the two calls; a
        wrong tensor comes back with the RIGHT byte count -- silent output
        corruption). pread needs no lock and never touches the handle's own
        file position. A single pread syscall returns at most ~2 GiB on
        Linux, so gather in a loop -- a 27B model's token_embd (2.4 GiB
        BF16) exceeds the cap and came back short in one call.
        """
        if self._fh is None:
            self._fh = open(self._path, "rb")
        fd = self._fh.fileno()
        remaining = byte_len
        chunks = []
        while remaining > 0:
            chunk = os.pread(fd, remaining, pos)
            if not chunk:
                raise IOError(
                    f"unexpected EOF reading tensor {tensor_name!r}: "
                    f"{remaining} of {byte_len} bytes missing at offset {pos}"
                )
            chunks.append(chunk)
            pos += len(chunk)
            remaining -= len(chunk)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def read_tensor_raw(self, tensor_name: str) -> Optional[bytes]:
        """Return this tensor's exact on-disk bytes, undecoded.

        See ``ModelSource.read_tensor_raw`` for why the writer needs this
        instead of routing passthrough tensors through ``read_tensor_f32``.
        """
        from magicquant.quant.converters import ggml_tensor_data_size
        info = self._reader.get_tensor_info(tensor_name)
        if info is None:
            return None
        type_name = self._TYPE_NAME.get(info["data_type"])
        if type_name is None:
            return None
        n_elems = 1
        for d in info["shape"]:
            n_elems *= d
        byte_len = ggml_tensor_data_size(type_name, n_elems)
        return self._read_raw_bytes(
            tensor_name, byte_len, self._data_offset + info["offset"],
        )

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
        buf = self._read_raw_bytes(
            tensor_name, byte_len, self._data_offset + info["offset"],
        )
        if type_name in _NATIVE_FLOAT_TYPES:
            return _decode_st_bytes_to_f32(type_name, buf)
        # Quantized source tensor. Undecodable unless dequant was explicitly
        # enabled; the writer's pre-quantized guard reports the error, so just
        # signal "no data" here rather than raising.
        if not self._allow_dequant:
            return None
        from magicquant.quant.ggml_binding import ggml_decode
        try:
            out = ggml_decode(buf, type_name, n_elems)
        except Exception as exc:
            # can_decode() probes the same symbol, so reaching here means the
            # libggml handle changed under us or the block layout disagreed.
            # Returning None would zero-fill the tensor; fail loudly instead.
            raise RuntimeError(
                f"Dequantizing tensor {tensor_name!r} from {type_name} failed: "
                f"{exc}"
            ) from exc
        self._dequantized_types.add(type_name)
        return out

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
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
    # NOTE: per-expert projection tensors (model.layers.N.mlp.experts.E.*,
    # block_sparse_moe.experts.E.*) are intentionally NOT matched here.
    # SafetensorsSource._ensure_loaded() intercepts them via
    # _detect_moe_expert_tensor() BEFORE this table is consulted and stacks
    # them into one 3-D ffn_{gate,up,down}_exps tensor per layer (see the
    # "MoE expert stacking" section below) -- matching them 1:1 here would
    # collapse all experts of a projection onto the same GGUF name (last
    # expert silently wins), producing an unloadable GGUF.
    (r"^model\.layers\.(\d+)\.mlp\.gate\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_inp.weight"),
    # Mixtral router lives under block_sparse_moe instead of mlp.
    (r"^model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_inp.weight"),
    # Shared expert(s) (DeepSeek/Qwen MoE): always-on expert(s) that run
    # alongside the routed ones. Each is already a single HF tensor per
    # projection (no per-expert axis), so it maps 1:1 like any other tensor.
    # DeepSeek/DeepSeek2 use the plural "shared_experts"; Qwen2MoE/Llama4 use
    # the singular "shared_expert" (confirmed against llama.cpp's
    # gguf-py/gguf/tensor_mapping.py FFN_{GATE,UP,DOWN}_SHEXP entries).
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_shexp.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_up_shexp.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_down_shexp.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_shexp.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert\.up_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_up_shexp.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert\.down_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_down_shexp.weight"),
    # Qwen2MoE's shared-expert gate: a [1, hidden] linear producing the
    # per-token sigmoid weight that blends the shared expert's output in.
    # GGUF stores it as a 1-D {n_embd} tensor (llama-model.cpp QWEN2MOE
    # loader requires it unconditionally, no TENSOR_NOT_REQUIRED).
    (r"^model\.layers\.(\d+)\.mlp\.shared_expert_gate\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate_inp_shexp.weight"),
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
    # TODO(deepseek-v2/v3 MLA): DeepSeek-V2/V3's Multi-head Latent Attention
    # uses a different projection set than the plain q/k/v/o above:
    #   self_attn.q_a_proj            -> blk.N.attn_q_a.weight
    #   self_attn.q_b_proj            -> blk.N.attn_q_b.weight
    #   self_attn.kv_a_proj_with_mqa  -> blk.N.attn_kv_a_mqa.weight
    #   self_attn.kv_b_proj           -> blk.N.attn_kv_b.weight
    #   self_attn.q_a_layernorm       -> blk.N.attn_q_a_norm.weight
    #   self_attn.kv_a_layernorm      -> blk.N.attn_kv_a_norm.weight
    # (names confirmed against llama.cpp's gguf-py/gguf/tensor_mapping.py
    # ATTN_Q_A / ATTN_Q_B / ATTN_KV_A_MQA / ATTN_KV_B / ATTN_Q_A_NORM /
    # ATTN_KV_A_NORM entries). Not added here to keep this change focused on
    # MoE expert-stacking + Mixtral; deepseek2/deepseek_v3 checkpoints will
    # currently fall through to the generic q/k/v/o patterns (no match, since
    # the attribute names differ) and keep their raw HF names, which
    # llama.cpp's deepseek2 loader won't recognize. Add a dedicated pattern
    # block here (mirroring the q/k/v block above) when MLA support is needed.
    # --- draft_upstream_sync.py candidates (UNVERIFIED -- see PR body checklist before merging) ---
    (r"^layers\.(\d+)\.attention\.wkv_a_with_mqa\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_kv_a_mqa.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_KV_A_MQA -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_kv_a_mqa.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_KV_A_MQA -- draft_upstream_sync, UNVERIFIED
    (r"^layers\.(\d+)\.attention\.kv_a_norm\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_kv_a_norm.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_KV_A_NORM -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.kv_a_layernorm\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_kv_a_norm.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_KV_A_NORM -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_kv_b.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_KV_B -- draft_upstream_sync, UNVERIFIED
    (r"^layers\.(\d+)\.attention\.k_b_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_k_b.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_K_B -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.k_b_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_k_b.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_K_B -- draft_upstream_sync, UNVERIFIED
    (r"^layers\.(\d+)\.attention\.wq_a\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q_a.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_Q_A -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q_a.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_Q_A -- draft_upstream_sync, UNVERIFIED
    (r"^layers\.(\d+)\.attention\.q_a_norm\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q_a_norm.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_Q_A_NORM -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.q_a_layernorm\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q_a_norm.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_Q_A_NORM -- draft_upstream_sync, UNVERIFIED
    (r"^layers\.(\d+)\.attention\.wq_b\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q_b.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_Q_B -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q_b.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_Q_B -- draft_upstream_sync, UNVERIFIED
    (r"^layers\.(\d+)\.attention\.v_b_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_v_b.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_V_B -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.self_attn\.v_b_proj\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_v_b.weight"),  # DEEPSEEK2OCR, MISTRAL4 / ATTN_V_B -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.post_feedforward_layernorm_1\.weight$",
     lambda m: f"blk.{m.group(1)}.post_ffw_norm_1.weight"),  # GEMMA4 / FFN_POST_NORM_1 -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.post_feedforward_layernorm_2\.weight$",
     lambda m: f"blk.{m.group(1)}.post_ffw_norm_2.weight"),  # GEMMA4 / FFN_POST_NORM_2 -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.pre_feedforward_layernorm_2\.weight$",
     lambda m: f"blk.{m.group(1)}.pre_ffw_norm_2.weight"),  # GEMMA4 / FFN_PRE_NORM_2 -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.layer_scalar\.weight$",
     lambda m: f"blk.{m.group(1)}.layer_output_scale.weight"),  # GEMMA4 / LAYER_OUT_SCALE -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.per_layer_input_gate\.weight$",
     lambda m: f"blk.{m.group(1)}.inp_gate.weight"),  # GEMMA4 / PER_LAYER_INP_GATE -- draft_upstream_sync, UNVERIFIED
    (r"^model\.per_layer_model_projection\.weight$",
     'per_layer_model_proj.weight'),  # GEMMA4 / PER_LAYER_MODEL_PROJ -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.post_per_layer_input_norm\.weight$",
     lambda m: f"blk.{m.group(1)}.post_norm.weight"),  # GEMMA4 / PER_LAYER_POST_NORM -- draft_upstream_sync, UNVERIFIED
    (r"^model\.layers\.(\d+)\.per_layer_projection\.weight$",
     lambda m: f"blk.{m.group(1)}.proj.weight"),  # GEMMA4 / PER_LAYER_PROJ -- draft_upstream_sync, UNVERIFIED
    (r"^model\.per_layer_projection_norm\.weight$",
     'per_layer_proj_norm.weight'),  # GEMMA4 / PER_LAYER_PROJ_NORM -- draft_upstream_sync, UNVERIFIED
    (r"^model\.embed_tokens_per_layer\.weight$",
     'per_layer_token_embd.weight'),  # GEMMA4 / PER_LAYER_TOKEN_EMBD -- draft_upstream_sync, UNVERIFIED
]

_HF_TO_GGUF_COMPILED = [(re.compile(p), r) for p, r in _HF_TO_GGUF_PATTERNS]


def _hf_name_to_gguf(
    hf_name: str, arch: str = "", *, strict: bool = False
) -> Optional[str]:
    """Convert a HuggingFace tensor name to GGUF convention.

    Args:
        hf_name: The original HuggingFace tensor name.
        arch: GGUF architecture string (e.g. "qwen35") for arch-specific
              name adjustments.
        strict: When False (default), an unmapped name falls back to
                ``hf_name`` unchanged -- the original contract every existing
                caller depends on (e.g. the unmapped-name gates below key off
                ``gguf_name == hf_name``). When True, an unmapped name
                returns ``None`` instead, so a caller can tell "matched
                nothing" apart from "matched to an identical name" without a
                second pattern-table scan (see ``magicquant.qat.names
                .hf_to_ggml_name``).
    """
    # Handle top-level output/lm_head directly. Ahead of any strict fallback:
    # this is the one legitimate case where a name maps to itself via a
    # pattern-independent self-map, not a "nothing matched" no-op.
    if hf_name in ("output.weight", "lm_head.weight"):
        return "output.weight"

    # Bias tensors share their projection's mapping (the patterns below only cover
    # .weight). Map the corresponding .weight name and swap the suffix, so e.g.
    # Qwen2's q/k/v `*_proj.bias` becomes `blk.N.attn_{q,k,v}.bias` (llama.cpp
    # requires these for qkv-bias architectures; without it the GGUF won't load).
    if hf_name.endswith(".bias"):
        weight_name = hf_name[: -len(".bias")] + ".weight"
        # Deliberately non-strict regardless of the outer `strict`: this
        # recursion tests "did the .weight name map to something new", which
        # only the "returns hf_name unchanged" sentinel expresses directly.
        mapped = _hf_name_to_gguf(weight_name, arch)
        if mapped != weight_name and mapped.endswith(".weight"):
            return mapped[: -len(".weight")] + ".bias"
        # projection's .weight didn't map -> leave bias untouched
        return None if strict else hf_name

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
    # Fallback: no pattern matched.
    return None if strict else hf_name


# =====================================================================
# MoE expert stacking
# =====================================================================
#
# HF MoE checkpoints store each expert's projection as its own separate
# 2-D tensor, e.g.:
#   model.layers.3.mlp.experts.7.gate_proj.weight            (generic/Qwen/DeepSeek)
#   model.layers.3.block_sparse_moe.experts.7.w1.weight       (Mixtral)
#
# llama.cpp's GGUF format instead expects ONE 3-D tensor per (layer,
# projection), with every expert's 2-D weight stacked along a new leading
# axis in ascending expert-index order:
#   blk.3.ffn_gate_exps.weight   shape [n_expert, out_features, in_features]
#   blk.3.ffn_up_exps.weight
#   blk.3.ffn_down_exps.weight
#
# This mirrors llama.cpp convert_hf_to_gguf.py's per-arch expert handling
# (e.g. MixtralModel.modify_tensors / Qwen2MoeModel.modify_tensors), which
# accumulates each expert's tensor into a per-layer dict keyed by HF name
# and, once all n_experts are collected, does
# ``torch.stack([experts[i] for i in range(n_experts)], dim=0)`` before
# handing the merged tensor to the normal write path -- i.e. expert 0's
# weight occupies index 0 of the new leading axis, expert 1 index 1, etc.
#
# Mixtral's w1/w2/w3 map to gate/down/up respectively (confirmed against
# llama.cpp's gguf-py/gguf/tensor_mapping.py FFN_GATE_EXP / FFN_DOWN_EXP /
# FFN_UP_EXP entries, which list "...block_sparse_moe.experts.w1" under
# FFN_GATE_EXP, "...w2" under FFN_DOWN_EXP, and "...w3" under FFN_UP_EXP).
#
# Router (ffn_gate_inp) and shared-expert (ffn_*_shexp) tensors are NOT
# per-expert -- each is already a single HF tensor, so they map 1:1 via the
# ordinary _HF_TO_GGUF_PATTERNS table above and never reach this code.
#
# NOTE: expert *bias* tensors (model.layers.N.mlp.experts.E.gate_proj.bias
# etc.) are not handled here -- most MoE Linear layers ship with
# bias=False, and no architecture in this codebase's test fleet currently
# needs it. An unmatched expert bias falls through to _hf_name_to_gguf
# unmatched and keeps its raw HF name (harmless: llama.cpp's MoE loaders
# don't require a bias tensor to be present). Extend
# _MOE_EXPERT_PATTERNS with a ".bias" variant if a checkpoint needs it.
_MOE_EXPERT_PATTERNS = [
    # Generic / Qwen / DeepSeek
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$"), "gate"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$"), "up"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$"), "down"),
    # Mixtral (w1=gate, w2=down, w3=up)
    (re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$"), "gate"),
    (re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.weight$"), "down"),
    (re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.weight$"), "up"),
]


def _detect_moe_expert_tensor(stripped_name: str) -> Optional[Tuple[str, int, str]]:
    """Identify a per-expert MoE projection tensor.

    Args:
        stripped_name: HF tensor name with multimodal prefixes already
            stripped (same preprocessing SafetensorsSource applies before
            calling ``_hf_name_to_gguf``).

    Returns:
        ``(gguf_stacked_name, expert_idx, proj_key)`` if *stripped_name*
        names one expert's slice of a gate/up/down projection (e.g.
        ``("blk.3.ffn_gate_exps.weight", 7, "gate")``), else ``None``.
        ``gguf_stacked_name`` is the SAME name for every expert index of a
        given (layer, projection) -- callers group by it to stack.
    """
    for pattern, proj_key in _MOE_EXPERT_PATTERNS:
        m = pattern.match(stripped_name)
        if m:
            layer, expert_idx = m.group(1), int(m.group(2))
            return f"blk.{layer}.ffn_{proj_key}_exps.weight", expert_idx, proj_key
    return None


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


def _decode_st_bytes_to_f32(dtype: str, buf) -> Optional[np.ndarray]:
    """Decode a raw safetensors-style byte buffer to a flat float32 array.

    Shared by SafetensorsSource.read_tensor_f32 (single tensor) and its
    stacked-MoE-expert reader (one call per expert part, concatenated),
    GGUFSource.read_tensor_f32 (gated to the three native float types --
    see _NATIVE_FLOAT_TYPES), and LoRAMergedSource._read_adapter_tensor.
    Returns None for an unrecognized dtype.
    """
    np_dtype = _ST_DTYPE_NUMPY.get(dtype)
    if np_dtype is None:
        return None
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).copy()
    if dtype == "F16":
        return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    if dtype == "BF16":
        raw = np.frombuffer(buf, dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(buf, dtype=np_dtype).astype(np.float32)


def _resolve_effective_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective LLM config for multimodal/composite models.

    For multimodal/composite models, the LLM config is nested under
    text_config, language_config, or llm_config (checked in that order,
    first match wins). The sub-config's keys WIN over the top-level
    config's on conflict -- e.g. qwen3_5's "qwen3_5_text" model_type only
    ever appears inside text_config and must override the parent's.
    Returns ``config`` itself (same object) when no sub-config is present.
    """
    effective = config
    for sub_key in ("text_config", "language_config", "llm_config"):
        if sub_key in config and isinstance(config[sub_key], dict):
            effective = {**config, **config[sub_key]}
            break
    return effective


def _build_gguf_metadata_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build GGUF-compatible metadata from a HuggingFace config.json."""
    # For multimodal/composite models, the LLM config is nested
    # under text_config, language_config, or llm_config.
    effective = _resolve_effective_config(config)

    model_type = effective.get("model_type", "llama")

    # Synced from llama.cpp convert_hf_to_gguf.py (2026-03)
    arch_map = {
        "arctic": "arctic", "baichuan": "baichuan", "bloom": "bloom",
        "chatglm": "chatglm", "cohere": "command-r", "cohere2": "cohere2",
        "deepseek2ocr": "deepseek2-ocr",  # DEEPSEEK2OCR -- draft_upstream_sync, UNVERIFIED
        "deepseek2ocr_text": "deepseek2-ocr",  # DEEPSEEK2OCR -- draft_upstream_sync, UNVERIFIED
        "dbrx": "dbrx", "deepseek": "deepseek", "deepseek_v2": "deepseek2",
        "deepseek_v3": "deepseek2", "exaone": "exaone",
        "falcon": "falcon", "falcon_h1": "falcon-h1",
        "falcon_mamba": "mamba", "gemma": "gemma", "gemma2": "gemma2",
        "gemma4": "gemma4",  # GEMMA4 -- draft_upstream_sync, UNVERIFIED
        "gemma4_text": "gemma4",  # GEMMA4 -- draft_upstream_sync, UNVERIFIED
        "gemma3": "gemma3", "glm4": "glm4", "gpt2": "gpt2",
        "gpt_neox": "gptneox", "granite": "granite",
        "granitemoe": "granitemoe", "granitemoehybrid": "granitehybrid",
        "grok": "grok",
        "hunyuan_vl": "hunyuan_vl",  # HUNYUAN_VL -- draft_upstream_sync, UNVERIFIED
        "hunyuan_vl_text": "hunyuan_vl",  # HUNYUAN_VL -- draft_upstream_sync, UNVERIFIED
        "internlm2": "internlm2", "internlm3": "llama",
        "jamba": "jamba", "llama": "llama", "llama4": "llama4",
        "mamba": "mamba", "mamba2": "mamba2", "minicpm": "minicpm",
        "minicpm3": "minicpm3", "mistral": "llama", "mistral3": "mistral3",
        "mistral4": "mistral4",  # MISTRAL4 -- draft_upstream_sync, UNVERIFIED
        "mistral4_text": "mistral4",  # MISTRAL4 -- draft_upstream_sync, UNVERIFIED
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
        # Same disease as the qwen3_5 uniform-logits incident: silently
        # defaulting an unrecognized model_type to 'llama' produces a GGUF
        # whose metadata keys (context_length, block_count, attention.*,
        # rope.freq_base, ...) are built under the WRONG architecture
        # namespace. It loads and often even runs, with hparams silently
        # wrong. Fail loudly instead -- extend arch_map (cross-check against
        # llama.cpp's convert_hf_to_gguf.py MODEL_ARCH/architecture registry)
        # rather than guessing.
        msg = (
            f"Unknown HF model_type '{model_type}' has no entry in "
            f"arch_map (_build_gguf_metadata_from_config, magicquant/gguf/"
            f"source.py) -- refusing to guess a GGUF architecture. Add a "
            f"verified '{model_type}' -> GGUF-arch mapping (cross-check "
            f"llama.cpp's convert_hf_to_gguf.py architecture registry), or "
            f"set {_ALLOW_UNVALIDATED_ARCH_ENV}=1 to proceed anyway at your "
            f"own risk (falls back to 'llama'; GGUF metadata keys may be "
            f"wrong)."
        )
        if _allow_unvalidated_arch():
            _log.error(msg)
            arch = "llama"
        else:
            raise UnsupportedSourceArchitecture(msg)

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

    # ── MoE metadata (qwen2moe/qwen3moe/mixtral/etc.) ──
    # llama.cpp reads expert_count/expert_used_count generically for every
    # arch (llama-model.cpp load_hparams, optional=true) but ASSERTS on
    # n_expert_used>0 whenever n_expert>0, and sizes the ffn_*_exps tensor
    # creation off hparams.n_expert -- without these keys a from-safetensors
    # MoE pack has real 3-D expert tensors on disk but hparams.n_expert==0,
    # so llama.cpp either mis-sizes or refuses the tensors as "not found".
    # HF field names per llama.cpp convert_hf_to_gguf.py's base TextModel
    # (num_local_experts/num_experts, num_experts_per_tok/num_experts_per_token)
    # and Qwen2MoeModel (moe_intermediate_size, shared_expert_intermediate_size).
    n_experts = effective.get("num_local_experts", effective.get("num_experts"))
    if n_experts is not None:
        meta[f"{arch}.expert_count"] = int(n_experts)
    n_experts_used = effective.get("num_experts_per_tok", effective.get("num_experts_per_token"))
    if n_experts_used is not None:
        meta[f"{arch}.expert_used_count"] = int(n_experts_used)
    moe_ffn_len = effective.get("moe_intermediate_size")
    if moe_ffn_len is not None:
        meta[f"{arch}.expert_feed_forward_length"] = int(moe_ffn_len)
    shared_ffn_len = effective.get("shared_expert_intermediate_size")
    if shared_ffn_len is not None:
        meta[f"{arch}.expert_shared_feed_forward_length"] = int(shared_ffn_len)

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


def _reorder_v_heads(a: np.ndarray, axis: int, num_k_heads: int,
                      num_v_per_k: int, head_dim: int) -> np.ndarray:
    """Reorder V heads from HF grouped order [G0_v0..v{r-1}, G1_v0..] to
    ggml's tiled broadcast order [K0,K1,..,K0,K1,..]. Pure index permutation
    -- mirrors llama.cpp convert_hf_to_gguf.py's ``_reorder_v_heads``
    (convert_hf_to_gguf.py:5405-5416), used by qwen3_5's linear-attention
    layers whenever ``linear_num_value_heads != linear_num_key_heads``.
    """
    shape = list(a.shape)
    if axis < 0:
        axis += len(shape)
    new_shape = shape[:axis] + [num_k_heads, num_v_per_k, head_dim] + shape[axis + 1:]
    t = a.reshape(new_shape).swapaxes(axis, axis + 1)
    return np.ascontiguousarray(t).reshape(shape)


def _qwen35_value_transform(hf_name: str, gguf_name: str, arr: np.ndarray,
                             shape: List[int], cfg: Dict[str, Optional[int]]
                             ) -> np.ndarray:
    """Apply qwen35/qwen35moe's HF->GGUF value transforms.

    Mirrors llama.cpp convert_hf_to_gguf.py's ``Qwen3NextModel.modify_tensors``
    (RMSNorm +1 for every ``*norm.weight`` EXCEPT ``linear_attn.norm.weight``;
    ``A_log -> -exp(A_log)``) plus the linear-attention V-head grouped->tiled
    reorder from ``_LinearAttentionVReorderBase``, which fires whenever
    ``linear_num_value_heads != linear_num_key_heads``. Without these, the
    packed GGUF loads and runs but every linear_attention layer's decay gate
    underflows to zero and the model's logits collapse to uniform -- see the
    qwen3_5 uniform-logits incident (PPL == vocab size). All three rules were
    verified byte-exact against a real ``convert_hf_to_gguf`` output during
    that investigation; see ``tests/test_source_qwen35_transforms.py``.

    Args:
        hf_name: original HuggingFace tensor name -- drives the RMSNorm/A_log
            rules, which llama.cpp keys off the HF side.
        gguf_name: mapped GGUF tensor name -- drives the V-reorder rules.
        arr: flat float32 array, UNTRANSFORMED (no rope permute either).
        shape: this tensor's GGUF shape (row-major, already squeezed by
            ``_flatten_to_max_dims`` -- squeezing size-1 dims never reorders
            memory, so reshaping the flat buffer into it is safe).
        cfg: ``{"num_k", "num_v", "head_k_dim", "head_v_dim"}`` resolved by
            ``SafetensorsSource._ensure_loaded``; any may be ``None`` if the
            source config didn't have the field (in which case the reorder
            is skipped but norm/A_log rules still apply -- they don't need
            head counts).
    """
    out = arr.reshape(shape) if shape else arr

    # RMSNorm +1, except linear_attn.norm.weight (ssm_norm) which llama.cpp
    # deliberately excludes (it's already correctly-scaled upstream).
    if hf_name.endswith("norm.weight") and not hf_name.endswith("linear_attn.norm.weight"):
        return (out + 1.0).reshape(-1)

    num_k, num_v = cfg.get("num_k"), cfg.get("num_v")
    head_k_dim, head_v_dim = cfg.get("head_k_dim"), cfg.get("head_v_dim")
    need_reorder = bool(num_k and num_v and num_k != num_v)
    r = (num_v // num_k) if need_reorder else None

    if hf_name.endswith(".A_log"):
        # Nonlinear -- MUST run before the (linear) reorder.
        out = -np.exp(out)
        if need_reorder:
            out = _reorder_v_heads(out, axis=0, num_k_heads=num_k, num_v_per_k=r, head_dim=1)
        return out.reshape(-1)

    if not need_reorder:
        return out.reshape(-1)

    if gguf_name.endswith((".ssm_dt.bias", ".ssm_alpha.weight", ".ssm_beta.weight")):
        out = _reorder_v_heads(out, axis=0, num_k_heads=num_k, num_v_per_k=r, head_dim=1)
    elif gguf_name.endswith(".attn_gate.weight"):
        out = _reorder_v_heads(out, axis=0, num_k_heads=num_k, num_v_per_k=r, head_dim=head_v_dim)
    elif gguf_name.endswith(".attn_qkv.weight") or gguf_name.endswith(".ssm_conv1d.weight"):
        # Leading Q/K rows (or conv channels) pass through verbatim; only the
        # trailing V portion needs reordering.
        qk_width = 2 * head_k_dim * num_k
        head_part, v_part = out[:qk_width], out[qk_width:]
        v_part = _reorder_v_heads(v_part, axis=0, num_k_heads=num_k, num_v_per_k=r, head_dim=head_v_dim)
        out = np.concatenate([head_part, v_part], axis=0)
    elif gguf_name.endswith(".ssm_out.weight"):
        # out_proj: the V axis is COLUMNS (axis=1), not rows.
        out = _reorder_v_heads(out, axis=1, num_k_heads=num_k, num_v_per_k=r, head_dim=head_v_dim)

    return out.reshape(-1)


class UnsupportedSourceArchitecture(RuntimeError):
    """A safetensors source whose arch needs HF->GGUF value transforms
    MagicQuant has not implemented (or verified). Packing it anyway
    produces a GGUF that loads and runs but emits garbage -- see the
    qwen3_5 uniform-logits incident (PPL == vocab size, every tensor name
    and shape matched the reference, but 64% of tensor VALUES were wrong).
    """


# Sentinel for arches whose only needed value transform is the existing
# NORM-rope Q/K permute (_QK_PERMUTED_ARCHS / _permute_qk_rows, applied by
# apply_arch_value_transform independently of this table). Their presence
# here is only to satisfy the architecture gate below -- it is never called.
_QK_PERMUTE_ONLY = object()

# Arches whose HF->GGUF value transforms are implemented AND verified
# byte-exact against llama.cpp's own convert_hf_to_gguf output.
_ARCH_VALUE_TRANSFORMS: Dict[str, Any] = {
    "llama": _QK_PERMUTE_ONLY,
    "baichuan": _QK_PERMUTE_ONLY,
    "qwen35": _qwen35_value_transform,
    "qwen35moe": _qwen35_value_transform,
}

# Arches verified -- by the test fleet exercising SafetensorsSource end to
# end against real HF tensor names -- to need NO value transform beyond
# name mapping + metadata. Deliberately conservative: an arch not proven
# either way fails loudly (UnsupportedSourceArchitecture) rather than
# shipping silently-wrong weights. Extend only after adding/verifying
# coverage, not on a hunch that "it's probably fine".
_ARCH_NO_TRANSFORM_NEEDED = frozenset({"qwen2", "qwen2moe"})

# Opt-in escape hatch: downgrades UnsupportedSourceArchitecture (both the
# arch gate and the unmapped-tensor-name gate below) to a single loud
# _log.error so an experimenter can proceed at their own risk. Default is
# OFF -- the whole point of the gate is that it can never be silently
# bypassed by accident.
_ALLOW_UNVALIDATED_ARCH_ENV = "MAGICQUANT_ALLOW_UNVALIDATED_ARCH"


def _allow_unvalidated_arch() -> bool:
    return os.environ.get(_ALLOW_UNVALIDATED_ARCH_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


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


def _extract_tokenizer_vocab(
    tok: Dict[str, Any], arch: str, config: Optional[Dict[str, Any]], model_dir: str,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Extract tokenizer.ggml.model/pre, the vocab (tokens/scores/token_type,
    including config.json vocab_size padding and the config.json-derived
    bos_token_id), and merges. Extracted verbatim from
    _build_tokenizer_metadata's tokenizer.json vocab-extraction section.

    Returns (meta, vocab, added): ``vocab`` (post BPE/Unigram-rebind) and
    ``added`` (added_tokens) are threaded on to _resolve_special_tokens,
    which needs them to build its own token->id lookup.
    """
    meta: Dict[str, Any] = {}
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
    if arch in ("qwen35", "qwen35moe"):
        # qwen3.5's tokenizer.json pre_tokenizer regex is byte-identical to
        # qwen2's Split regex -- regex matching alone genuinely cannot tell
        # them apart (llama.cpp's own converter disambiguates via a checksum
        # of tokenizing a fixed test string, not regex matching, which this
        # simplified table doesn't replicate). We already know the arch from
        # config.json, so use it directly instead of the ambiguous table.
        pre = "qwen35"
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
        #
        # NOTE: `added` is bound ONLY inside this `if vocab:` block, same as
        # HEAD's pre-decomposition single function. This function's `return
        # meta, vocab, added` below unconditionally references it, so an
        # empty/falsy vocab makes THIS function raise UnboundLocalError at
        # its own return statement -- a pre-existing latent bug, not touched
        # or fixed here. Decomposition widened its reach: HEAD only hit this
        # when the vocab was empty AND tokenizer_config.json existed (the
        # only place `added` was read downstream); now it fires whenever the
        # vocab is empty, tokenizer_config.json or not, because `added` must
        # cross this function's return boundary to reach
        # _resolve_special_tokens. See the CHANGELOG "Decomposed two more
        # god-functions" entry for the accepted-edge-case disclosure -- no
        # real BPE/Unigram tokenizer.json ships an empty vocab.
        added = tok.get("added_tokens", [])
        if added:
            max_added_id = max(at.get("id", -1) for at in added)
            max_id = max(max_id, max_added_id)

        # If a config.json exists, use its vocab_size to pad the token
        # list so it matches the embedding tensor dimension. If the caller
        # already parsed config.json (see the ``config`` arg docstring
        # above), use that directly instead of re-reading it from disk.
        if config is not None:
            _cfg = config
        else:
            config_path_for_vocab = os.path.join(model_dir, "config.json")
            _cfg = None
            if os.path.exists(config_path_for_vocab):
                with open(config_path_for_vocab) as _f:
                    _cfg = json.load(_f)
        if _cfg is not None:
            _eff = _resolve_effective_config(_cfg)
            cfg_vocab_size = _eff.get("vocab_size", 0)
            if cfg_vocab_size > max_id + 1:
                max_id = cfg_vocab_size - 1

            # bos_token_id: for qwen3_5, config.json's top-level bos_token_id
            # is absent (it lives under text_config, already merged into
            # _eff above) and was previously dropped entirely, leaving
            # llama.cpp to default BOS to token 0. Emit generically for any
            # arch that has it. This is the FIRST of two bos_token_id
            # writes -- _resolve_special_tokens's tokenizer_config.json
            # bos_token write is the second and takes priority when it
            # resolves (see the comment there).
            bos_id = _eff.get("bos_token_id")
            if bos_id is not None:
                meta["tokenizer.ggml.bos_token_id"] = int(bos_id)

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

    return meta, vocab, added


_NO_TOKENIZER_CFG = object()
# Sentinel distinguishing "tokenizer_config.json does not exist" from "it
# exists and parses to JSON null" (tok_cfg is then a real ``None``). HEAD's
# original single-function code only ever entered its special-token/chat-
# template block behind ``if os.path.exists(config_path):`` -- a present-
# but-null file got in, then crashed with AttributeError the first time it
# called ``tok_cfg.get(...)``. Using bare ``None`` as the "absent" default
# here would conflate the two cases and turn that crash into a silent
# skip. _build_tokenizer_metadata passes this sentinel when the file is
# absent; _resolve_special_tokens/_resolve_chat_template check identity
# against it (not ``is None``) so a real ``None`` still falls through to
# ``.get()`` and raises exactly as before.


def _resolve_special_tokens(
    tok_cfg: Any, vocab: Any, added: Any,
) -> Dict[str, Any]:
    """Resolve bos/eos/pad/unk token ids and add_bos_token/add_eos_token
    from tokenizer_config.json. Extracted verbatim from
    _build_tokenizer_metadata's special-token-resolution section (the
    ``if os.path.exists(config_path):`` block, minus the chat_template
    read now in _resolve_chat_template). ``vocab``/``added`` are
    _extract_tokenizer_vocab's return values, needed to build the
    token->id lookup here. ``tok_cfg`` is the parsed tokenizer_config.json
    contents, ``_NO_TOKENIZER_CFG`` if the file doesn't exist, or ``None``
    if it exists and parses to JSON null (see ``_NO_TOKENIZER_CFG``'s
    docstring -- that case is NOT skipped, it falls through to ``.get()``
    and raises, matching HEAD).
    """
    meta: Dict[str, Any] = {}
    if tok_cfg is _NO_TOKENIZER_CFG:
        return meta

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

    # bos_token_id double-write, priority is INTENTIONAL: this is the
    # SECOND write. The first write is _extract_tokenizer_vocab's
    # config.json-derived tokenizer.ggml.bos_token_id, above/before this
    # function runs. Here, tokenizer_config.json's bos_token OVERWRITES it
    # -- takes priority over the config.json value, the same way the
    # qwen35 rope_theta override takes priority over the generic
    # field_map (see _build_gguf_metadata_from_config). The overwrite is
    # conditional, not unconditional: it only fires when bos_token
    # resolves to a known id below (`val in all_token_ids`); if it doesn't
    # (e.g. Qwen's tokenizer_config.json has bos_token: null), this loop
    # skips it and the config.json value from the first write survives.
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

    return meta


def _resolve_chat_template(
    model_dir: str, tok_cfg: Any,
) -> Dict[str, Any]:
    """Resolve tokenizer.chat_template from tokenizer_config.json, then the
    standalone chat_template.jinja/.json file fallbacks, then warn if a
    template file exists but yielded nothing. Extracted verbatim from
    _build_tokenizer_metadata's chat-template section (the chat_template
    read out of the same ``if os.path.exists(config_path):`` block as
    _resolve_special_tokens, plus the two file-based fallbacks and the
    warn-if-nothing-emitted check). ``tok_cfg`` -- see _NO_TOKENIZER_CFG's
    docstring for the file-absent-vs-null distinction; in practice a real
    ``None`` never reaches here because _resolve_special_tokens (called
    first by _build_tokenizer_metadata) already raises AttributeError on
    it, matching HEAD.
    """
    meta: Dict[str, Any] = {}

    # Chat template
    if tok_cfg is not _NO_TOKENIZER_CFG:
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
        if "tokenizer.chat_template" not in meta and os.path.exists(json_path):
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


def _build_tokenizer_metadata(
    model_dir: str, arch: str = "", config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read tokenizer data from a HuggingFace model directory and return
    GGUF-compatible tokenizer metadata.

    Handles the common case: BPE tokenizer from tokenizer.json
    (covers LLaMA, Qwen, Mistral, GPT-NeoX, Falcon, etc.).

    Args:
        arch: GGUF architecture string, used only to disambiguate
            ``tokenizer.ggml.pre`` for qwen35/qwen35moe (see below --
            regex matching alone can't tell it apart from qwen2).
        config: Already-parsed config.json contents, if the caller has
            one (e.g. SafetensorsSource._ensure_loaded, which reads
            config.json before this is called). When given, skips this
            function's own config.json read. Pass ``{}`` (not None) to
            represent "config.json does not exist" -- an empty dict
            resolves through the same effective-config/vocab_size/
            bos_token_id lookups as a missing file and yields identical
            metadata. Leave as ``None`` (the default) to preserve the
            original read-from-disk behavior.
    """
    meta: Dict[str, Any] = {}

    # ── tokenizer.json (BPE vocab + merges) ──
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        return meta

    with open(tokenizer_path, encoding="utf-8") as f:
        tok = json.load(f)

    vocab_meta, vocab, added = _extract_tokenizer_vocab(tok, arch, config, model_dir)
    meta.update(vocab_meta)

    # ── tokenizer_config.json (special token IDs) ──
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    tok_cfg = _NO_TOKENIZER_CFG
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            tok_cfg = json.load(f)

    meta.update(_resolve_special_tokens(tok_cfg, vocab, added))
    meta.update(_resolve_chat_template(model_dir, tok_cfg))

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
        # Set BEFORE any work: if the arch gate or unmapped-name gate below
        # raises, the object is deliberately left flagged loaded with a
        # partially-built _tensor_map, so a subsequent call returns silently
        # instead of re-raising. LoRAMergedSource.__init__ depends on this
        # indirectly (it calls self._base.get_metadata() inside a bare
        # `except Exception`, swallowing that first raise). Do NOT move this
        # to the end of the method.
        self._loaded = True

        self._discover_files()
        config, arch = self._load_config_and_gate_arch()
        self._resolve_qk_permute_cfg(arch, config)
        self._resolve_qwen35_cfg(arch, config)
        expert_groups = self._parse_headers_and_map_names(arch)
        self._stack_moe_experts(expert_groups)
        self._alias_tied_embeddings(config)

        # Load tokenizer data
        tokenizer_meta = _build_tokenizer_metadata(self._model_dir, arch=arch, config=config)
        self._metadata.update(tokenizer_meta)

    def _discover_files(self):
        """Discover safetensors shard files. Extracted verbatim from
        _ensure_loaded's shard-discovery stage.
        """
        if not self._files:
            index_path = os.path.join(self._model_dir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                for _hf_name, filename in weight_map.items():
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

    def _load_config_and_gate_arch(self) -> Tuple[Dict[str, Any], str]:
        """Load config.json, build metadata + resolve arch, and run the
        architecture gate. Extracted verbatim from _ensure_loaded's
        config-load / arch-gate stage. Returns (config, arch); also sets
        self._metadata as a side effect, matching the original inline code.
        """
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

        # Architecture gate: an arch with unimplemented/unverified HF->GGUF
        # value transforms must never silently ship garbage (see
        # UnsupportedSourceArchitecture / the qwen3_5 uniform-logits
        # incident). Checked BEFORE any tensor is read.
        if arch not in _ARCH_VALUE_TRANSFORMS and arch not in _ARCH_NO_TRANSFORM_NEEDED:
            msg = (
                f"Architecture '{arch}' has no verified HF->GGUF value-"
                f"transform coverage in SafetensorsSource (see "
                f"_ARCH_VALUE_TRANSFORMS / _ARCH_NO_TRANSFORM_NEEDED in "
                f"magicquant/gguf/source.py). Packing it anyway can produce "
                f"a GGUF that LOADS AND RUNS but emits garbage -- every "
                f"tensor name and shape can match the reference while the "
                f"VALUES are silently wrong (the qwen3_5 uniform-logits "
                f"incident). Add and verify a transform for '{arch}', or "
                f"set {_ALLOW_UNVALIDATED_ARCH_ENV}=1 to proceed anyway at "
                f"your own risk."
            )
            if _allow_unvalidated_arch():
                _log.error(msg)
            else:
                raise UnsupportedSourceArchitecture(msg)

        return config, arch

    def _resolve_qk_permute_cfg(self, arch: str, config: Dict[str, Any]):
        """Set self._qk_heads for NORM-rope arches. Extracted verbatim from
        _ensure_loaded's QK-permute head-count setup stage.
        """
        # Rope permutation setup for NORM-rope arches (see _QK_PERMUTED_ARCHS):
        # resolve head counts from the (text_config-aware) effective config.
        self._qk_heads = None
        if arch in _QK_PERMUTED_ARCHS:
            effective = _resolve_effective_config(config)
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

    def _resolve_qwen35_cfg(self, arch: str, config: Dict[str, Any]):
        """Set self._qwen35_cfg for the qwen3_5 hybrid-arch value transform.
        Extracted verbatim from _ensure_loaded's qwen35 linear-attention
        cfg-setup stage.
        """
        # qwen3_5 (qwen35/qwen35moe) hybrid-arch value-transform setup:
        # resolve the linear-attention head counts read_tensor_f32 needs to
        # apply _qwen35_value_transform's V-head reorder. None of these are
        # required for the RMSNorm +1 / A_log -> -exp() rules (they don't
        # need head counts), only for the reorder.
        self._qwen35_cfg = None
        if arch in ("qwen35", "qwen35moe"):
            effective = _resolve_effective_config(config)
            self._qwen35_cfg = {
                "num_k": effective.get("linear_num_key_heads"),
                "num_v": effective.get("linear_num_value_heads"),
                "head_k_dim": effective.get("linear_key_head_dim"),
                "head_v_dim": effective.get("linear_value_head_dim"),
            }

    def _parse_headers_and_map_names(self, arch: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Parse every shard's header, map HF tensor names to GGUF names
        (populating self._tensor_map directly), group MoE per-expert
        projections for later stacking, and run the unmapped-name gate.
        Extracted verbatim from _ensure_loaded's per-file header-parse /
        MoE-expert-grouping / name-mapping loop, plus the unmapped-name
        gate that must run immediately after (same gate, same escape
        hatch as the architecture gate). Returns expert_groups for
        _stack_moe_experts.
        """
        # Per-expert MoE projection tensors accumulate here instead of going
        # straight into self._tensor_map: gguf_stacked_name -> {expert_idx: raw_info}.
        # Stacked into single virtual 3-D tensors after all files are parsed
        # (see "MoE expert stacking" below _hf_name_to_gguf).
        expert_groups: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # HF tensor names that fell through _hf_name_to_gguf unmapped (kept
        # under their raw HF name). llama.cpp silently ignores unrecognized
        # tensor names, so an unmapped tensor is effectively MISSING from the
        # packed GGUF -- gated below alongside the architecture check.
        unmapped_names: List[str] = []

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

                dtype = info.get("dtype", "F32")
                shape = info.get("shape", [])
                offsets = info.get("data_offsets", [0, 0])

                # MoE per-expert projections: group by (layer, projection)
                # instead of adding to _tensor_map directly -- they're
                # stacked into one virtual 3-D tensor once every file has
                # been scanned (an expert's shard may not be the first/last
                # file for its layer).
                moe_hit = _detect_moe_expert_tensor(stripped)
                if moe_hit is not None:
                    stacked_name, expert_idx, _proj_key = moe_hit
                    expert_groups.setdefault(stacked_name, {})[expert_idx] = {
                        "dtype": dtype,
                        "shape": list(shape),
                        "filepath": filepath,
                        "byte_offset": offsets[0],
                        "byte_length": offsets[1] - offsets[0],
                        "data_start": data_start,
                    }
                    continue

                gguf_name = _hf_name_to_gguf(hf_name, arch=arch)
                # "output.weight" is excluded: it's the one legitimate case
                # where hf_name can already equal the GGUF name verbatim
                # (_hf_name_to_gguf's top-of-function passthrough), not a
                # fallback miss.
                if gguf_name == hf_name and hf_name != "output.weight":
                    unmapped_names.append(hf_name)

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

        # Unmapped-name gate: a name _hf_name_to_gguf() couldn't map is
        # dropped on the floor by llama.cpp (unrecognized names are silently
        # ignored), the mirror-image failure mode of the qwen3_5 incident
        # (there every name mapped fine but the VALUES were wrong). Same
        # gate, same escape hatch.
        if unmapped_names:
            shown = unmapped_names[:10]
            more = (f" (+{len(unmapped_names) - 10} more)"
                    if len(unmapped_names) > 10 else "")
            msg = (
                f"{len(unmapped_names)} tensor name(s) in this safetensors "
                f"source have no GGUF name mapping for arch '{arch}' and "
                f"were kept under their raw HF name: {shown}{more}. "
                f"llama.cpp silently ignores unrecognized tensor names, so "
                f"these will be MISSING from the packed GGUF. Add a pattern "
                f"to _HF_TO_GGUF_PATTERNS, or set "
                f"{_ALLOW_UNVALIDATED_ARCH_ENV}=1 to proceed anyway at your "
                f"own risk."
            )
            if _allow_unvalidated_arch():
                _log.error(msg)
            else:
                raise UnsupportedSourceArchitecture(msg)

        return expert_groups

    def _stack_moe_experts(self, expert_groups: Dict[str, Dict[int, Dict[str, Any]]]):
        """Stack grouped per-expert MoE projections into virtual 3-D
        tensors in self._tensor_map. Extracted verbatim from
        _ensure_loaded's MoE expert-stacking stage. Must run after
        _parse_headers_and_map_names (needs its expert_groups) and before
        _alias_tied_embeddings (tensor insertion order determines GGUF
        on-disk tensor order).
        """
        # Stack each MoE expert group into one virtual 3-D tensor: shape
        # [n_expert, out_features, in_features], experts in ascending index
        # order along the new leading axis (matches llama.cpp's
        # torch.stack(..., dim=0) -- see _detect_moe_expert_tensor's docstring).
        for gguf_name, experts_by_idx in expert_groups.items():
            idxs = sorted(experts_by_idx.keys())
            if idxs != list(range(len(idxs))):
                _log.warning(
                    "MoE expert group '%s' has non-contiguous/unexpected "
                    "expert indices %s (expected a dense 0..%d range) -- "
                    "stacking in ascending-index order anyway; a missing "
                    "expert will shift every later expert into the wrong "
                    "slot.", gguf_name, idxs, len(idxs) - 1,
                )
            ordered_parts = [experts_by_idx[i] for i in idxs]
            first = ordered_parts[0]
            for part in ordered_parts[1:]:
                if part["dtype"] != first["dtype"]:
                    _log.warning(
                        "MoE expert group '%s' has mixed dtypes (%s vs %s) "
                        "across experts -- decoding all parts to f32 "
                        "independently, but this is unexpected.",
                        gguf_name, first["dtype"], part["dtype"],
                    )

            per_expert_shape = list(first["shape"])
            stacked_shape_orig = [len(ordered_parts)] + per_expert_shape
            stacked_shape = _flatten_to_max_dims(stacked_shape_orig, max_dims=4)

            self._tensor_map[gguf_name] = {
                "hf_name": None,  # synthesized from N per-expert HF tensors
                "gguf_name": gguf_name,
                "dtype": first["dtype"],
                "shape": stacked_shape,  # row-major, at most 4-D
                "shape_orig": stacked_shape_orig,
                "n_dims": len(stacked_shape),
                "data_type": _ST_DTYPE_TO_GGML.get(first["dtype"], 0),
                "is_expert_stack": True,
                "expert_parts": ordered_parts,  # ascending expert-index order
            }

    def _alias_tied_embeddings(self, config: Dict[str, Any]):
        """Alias output.weight to token_embd.weight for tied embeddings.
        Extracted verbatim from _ensure_loaded's tied-embedding-aliasing
        stage. Must run after _stack_moe_experts (tensor insertion order
        determines GGUF on-disk tensor order).
        """
        # Handle tied weights: if output.weight is missing and embeddings are tied,
        # create a reference to token_embd.weight
        if "output.weight" not in self._tensor_map and "token_embd.weight" in self._tensor_map:
            if config.get("tie_word_embeddings", True):
                ref = dict(self._tensor_map["token_embd.weight"])
                ref["gguf_name"] = "output.weight"
                self._tensor_map["output.weight"] = ref

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

    def _read_untransformed(self, tensor_name: str) -> Optional[np.ndarray]:
        """Decode ``tensor_name`` to a flat float32 array with NO value
        transforms applied -- no Q/K rope permute, no arch-specific
        transform (e.g. qwen35's RMSNorm +1 / A_log -> -exp / V-reorder).

        Split out from ``read_tensor_f32`` so ``LoRAMergedSource`` can merge
        its delta onto the raw (untransformed) base and call
        ``apply_arch_value_transform`` exactly once, post-merge -- merging
        onto an ALREADY-transformed base and re-transforming would
        double-apply an additive rule (norm's ``+1``) and feed a nonlinear
        rule (``A_log``'s ``-exp()``) a value it never actually produces.
        """
        self._ensure_loaded()
        info = self._tensor_map.get(tensor_name)
        if info is None:
            return None

        if info.get("is_expert_stack"):
            return self._read_stacked_experts(info)

        dtype = info["dtype"]

        # Use memory-mapped I/O for zero-copy reads
        mmap = self._get_mmap(info["filepath"])
        start = info["data_start"] + info["byte_offset"]
        end = start + info["byte_length"]
        buf = mmap[start:end]

        return _decode_st_bytes_to_f32(dtype, buf)

    def apply_arch_value_transform(self, tensor_name: str,
                                    flat: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply this source's HF->GGUF value transform(s) to an already-
        decoded flat float32 array (as returned by ``_read_untransformed``).

        Idempotent-once: call exactly one time per tensor. Currently: the
        NORM-rope Q/K permute (``_QK_PERMUTED_ARCHS``) and, for
        qwen35/qwen35moe, ``_qwen35_value_transform``.
        """
        self._ensure_loaded()
        if flat is None:
            return None
        info = self._tensor_map.get(tensor_name)
        if info is None:
            return flat

        # NORM-rope arches: llama.cpp expects Q/K rows interleaved.
        n_head = self.get_qk_permute_heads(tensor_name)
        if n_head:
            shaped = flat.reshape(info["shape"])
            flat = _permute_qk_rows(shaped, n_head).reshape(-1)

        # qwen35/qwen35moe: RMSNorm +1 / A_log -> -exp / linear-attn V-reorder.
        # Never applies to expert-stack tensors (hf_name is None for those --
        # they're synthesized from N per-expert tensors, and none of the
        # qwen35 rules target expert weights anyway).
        if self._qwen35_cfg is not None and info.get("hf_name"):
            flat = _qwen35_value_transform(
                info["hf_name"], tensor_name, flat, info["shape"], self._qwen35_cfg,
            )

        return flat

    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        return self.apply_arch_value_transform(
            tensor_name, self._read_untransformed(tensor_name)
        )

    def _read_stacked_experts(self, info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Read a virtual stacked-MoE-expert tensor (see ``_detect_moe_expert_tensor``).

        Each expert's 2-D weight is decoded to a flat f32 array individually,
        then concatenated in ascending expert-index order. Because
        ``read_tensor_f32`` always returns a FLAT array and the stacked axis
        is a new LEADING axis, simple concatenation in index order already
        produces the correct row-major memory layout for
        ``[n_expert, out_features, in_features]`` -- no reshape/transpose
        needed.
        """
        parts = []
        for part in info["expert_parts"]:
            mmap = self._get_mmap(part["filepath"])
            start = part["data_start"] + part["byte_offset"]
            end = start + part["byte_length"]
            buf = mmap[start:end]
            flat = _decode_st_bytes_to_f32(part["dtype"], buf)
            if flat is None:
                return None
            parts.append(flat)
        return np.concatenate(parts)

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
        W_merged = W_base + (lora_B @ lora_A) * scale

    scale is (alpha / rank) for plain LoRA, but PEFT computes
    alpha / sqrt(rank) instead when the adapter was trained with rsLoRA
    (rank-stabilized LoRA, adapter_config.json's "use_rslora": true) --
    see PEFT's LoraLayer.scaling.

    Tensors without LoRA adapters pass through from the base model unchanged.
    No full merged copy is written to disk — merging happens per-tensor as
    the writer reads each one.
    """

    def __init__(self, base_path: str, adapter_path: str,
                 allow_dequant: Optional[bool] = None):
        """
        Args:
            base_path: Path to the base model (directory or .safetensors/.gguf)
            adapter_path: Path to the LoRA adapter directory (contains
                adapter_config.json + adapter_model.safetensors)
            allow_dequant: Forwarded to the base source; see open_model_source.
        """
        from magicquant.gguf.source import open_model_source

        self._base = open_model_source(base_path, allow_dequant=allow_dequant)

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
        # INCIDENT (rsLoRA audit, 2026-07-28): this used to compute
        # alpha / rank unconditionally, ignoring adapter_config.json's
        # "use_rslora" flag. PEFT uses alpha / sqrt(rank) instead for
        # rsLoRA adapters (see PEFT's LoraLayer.scaling) -- the same bug
        # already found and fixed in Foundry's fast_export.build_lora_map
        # (core/fast_export.py, commit d25f8bc), which trains with
        # use_rslora=True by default. An unfixed LoRAMergedSource merges
        # deltas sqrt(rank)x weaker than PEFT's own merge_and_unload() for
        # any rsLoRA adapter. See test_lora_merged_source_peft_differential.py.
        self._use_rslora = bool(adapter_cfg.get("use_rslora", False))
        self._scale = (
            self._alpha / math.sqrt(self._rank) if self._use_rslora
            else self._alpha / self._rank
        )
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
        flat = _decode_st_bytes_to_f32(dtype, buf)
        if flat is None:
            raise ValueError(f"Adapter tensor {key!r} has unsupported dtype {dtype!r}")
        return flat.reshape(info["shape"])

    def get_metadata(self):
        return self._base.get_metadata()

    def get_tensor_names(self):
        return self._base.get_tensor_names()

    def get_all_tensors_info(self):
        return self._base.get_all_tensors_info()

    def get_source_type_name(self, tensor_name: str) -> str:
        return self._base.get_source_type_name(tensor_name)

    def can_decode(self, tensor_name: str) -> bool:
        # Merging happens in float on top of whatever the base yields, so
        # decodability is exactly the base's answer (including its dequant
        # policy) -- not the conservative float-types-only default.
        return self._base.can_decode(tensor_name)

    def read_tensor_raw(self, tensor_name: str) -> Optional[bytes]:
        # A raw byte passthrough only makes sense when there's nothing to
        # merge: a tensor with a LoRA delta MUST go through read_tensor_f32
        # (decode -> merge -> re-encode), so returning raw bytes for it
        # here would silently ship the unmerged (and possibly quantized)
        # base tensor. Tensors with no adapter entry just forward to the
        # base source, including its own passthrough/undecodable policy.
        if tensor_name in self._lora_map:
            return None
        return self._base.read_tensor_raw(tensor_name)

    def read_tensor_f32(self, tensor_name: str) -> Optional[np.ndarray]:
        # Bases that separate raw decode from arch value-transforms
        # (currently only SafetensorsSource) get the "merge onto raw,
        # transform once" path: mandatory for qwen3_5-family bases, where
        # the transform includes a NONLINEAR step (A_log -> -exp(A_log)) --
        # merging a LoRA delta onto an ALREADY-transformed base and then
        # re-transforming the result would double-apply norm's additive
        # "+1" and feed -exp() a value it never produces. Bases without the
        # split (e.g. GGUFSource) keep the original behavior: merge onto the
        # already-transformed base, separately permuting just the delta.
        read_untransformed = getattr(self._base, "_read_untransformed", None)
        apply_transform = getattr(self._base, "apply_arch_value_transform", None)
        use_raw_merge = read_untransformed is not None and apply_transform is not None

        if use_raw_merge:
            base_f32 = read_untransformed(tensor_name)
        else:
            base_f32 = self._base.read_tensor_f32(tensor_name)
        if base_f32 is None:
            return None

        if tensor_name not in self._lora_map:
            if use_raw_merge:
                return apply_transform(tensor_name, base_f32)
            return base_f32

        a_key, b_key = self._lora_map[tensor_name]
        lora_a = self._read_adapter_tensor(a_key)  # (rank, in_features)
        lora_b = self._read_adapter_tensor(b_key)  # (out_features, rank)

        # Merge: W = W_base + (B @ A) * scale
        delta = (lora_b @ lora_a) * self._scale
        # fan_in_fan_out: transpose delta for Conv1D-based models (GPT-2 style)
        if self._fan_in_fan_out:
            delta = delta.T

        if not use_raw_merge:
            # base_f32 is already transformed (incl. rope permute for
            # NORM-rope arches); permute the delta identically before adding
            # so the two layouts match.
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
        merged = base_f32.reshape(delta.shape) + delta

        if use_raw_merge:
            # Raw (untransformed, unpermuted) merge -- apply this source's
            # value transform (rope permute + qwen35 rules, if any) exactly
            # once, to the merged result.
            return apply_transform(tensor_name, merged.flatten())

        return merged.flatten()

    def close(self):
        self._base.close()


# =====================================================================
# Factory
# =====================================================================

def open_model_source(
    path: str,
    adapter_path: Optional[str] = None,
    allow_dequant: Optional[bool] = None,
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

    *allow_dequant* opts a GGUF source into dequantizing already-quantized
    tensors back to F32 so they can be re-quantized (see _ALLOW_DEQUANT_ENV
    for why this is off by default and what it costs). ``None`` -- the
    default -- takes the policy from the environment, which is how the
    setting reaches nested open_model_source calls and subprocesses. It has
    no effect on safetensors sources, which are already high-precision.
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
                return LoRAMergedSource(base_path=base_model, adapter_path=path,
                                        allow_dequant=allow_dequant)
            # Relative paths are resolved against the adapter directory, not the
            # CWD, to avoid surprising lookups.
            if base_model and not os.path.isabs(base_model):
                candidate = os.path.normpath(os.path.join(path, base_model))
                if os.path.isdir(candidate):
                    return LoRAMergedSource(base_path=candidate, adapter_path=path,
                                            allow_dequant=allow_dequant)
            raise ValueError(
                f"LoRA adapter detected at {path} but base model "
                f"'{base_model}' could not be resolved to a local directory. "
                f"Download it first or pass the base model path explicitly."
            )

    # Explicit adapter
    if adapter_path is not None:
        return LoRAMergedSource(base_path=path, adapter_path=adapter_path,
                                allow_dequant=allow_dequant)

    # Standard format detection
    if os.path.isfile(path):
        if path.endswith(".gguf"):
            return GGUFSource(path, allow_dequant=allow_dequant)
        if path.endswith(".safetensors"):
            return SafetensorsSource(path)
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"GGUF":
            return GGUFSource(path, allow_dequant=allow_dequant)
        return SafetensorsSource(path)

    if os.path.isdir(path):
        has_st = any(f.endswith(".safetensors") for f in os.listdir(path))
        if has_st:
            return SafetensorsSource(path)
        gguf_files = sorted(f for f in os.listdir(path) if f.endswith(".gguf"))
        if len(gguf_files) > 1:
            # FAIL CLOSED. This used to take gguf_files[0] from an UNORDERED
            # os.listdir, so a directory holding several GGUFs silently
            # resolved to an arbitrary one -- not even deterministically
            # across machines. Observed 2026-08-13: a Foundry run directory
            # held model-bf16.gguf (417 tensors) beside model-bf16-nomtp.gguf
            # (401 tensors, MTP block removed) and the directory resolved to
            # the MTP-FREE variant. Quantizing the wrong model is not an error
            # anything downstream can detect: the file is valid, the tensor
            # count is plausible, and the artifact ships.
            #
            # Same doctrine as ggml_facts.expected_size: an ambiguous input is
            # refused, never guessed. Naming both candidates is the point --
            # the caller knows which they meant, and this module cannot.
            listing = "\n".join(f"    {f}" for f in gguf_files)
            raise ValueError(
                f"{path} contains {len(gguf_files)} .gguf files and there is "
                f"no basis for choosing between them:\n{listing}\n"
                f"Pass the specific file rather than the directory. (If these "
                f"are shards of one multi-part model, merge them first with "
                f"`llama-gguf-split --merge` -- multi-part sources are not "
                f"stitched automatically.)"
            )
        if gguf_files:
            return GGUFSource(os.path.join(path, gguf_files[0]),
                              allow_dequant=allow_dequant)

    raise ValueError(f"Cannot detect model format for: {path}")
