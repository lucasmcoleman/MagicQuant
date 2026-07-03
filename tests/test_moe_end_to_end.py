"""End-to-end MoE validation against a REAL llama.cpp binary.

``tests/test_moe_stacking.py`` proved expert-stacking at the unit level only:
synthetic tiny fixtures round-tripped through ``SafetensorsSource`` in-process,
with no GGUF writer and no llama.cpp loader involved. This module closes that
gap for two things that had never been exercised end-to-end before:

1. SafetensorsSource MoE expert-stacking + ``create_hybrid_gguf`` producing a
   GGUF that a real ``llama-completion``/``llama-cli`` binary actually loads
   and generates tokens from (routed experts, a router, AND a qwen2moe shared
   expert branch — llama.cpp's QWEN2MOE loader requires the shared-expert
   tensors unconditionally, no ``TENSOR_NOT_REQUIRED``).
2. Per-expert imatrix capture (``llama-imatrix`` over the packed GGUF) fed
   back through ``magicquant.imatrix.load_imatrix`` and ``create_hybrid_gguf``
   to re-pack the routed experts as Q4_K, then loaded again.

FINDINGS from building this (see ``magicquant/gguf/source.py``):

- ``_build_gguf_metadata_from_config`` never emitted ``expert_count`` /
  ``expert_used_count`` / ``expert_feed_forward_length`` /
  ``expert_shared_feed_forward_length``. llama.cpp reads
  ``%s.expert_count``/``%s.expert_used_count`` generically for every arch and
  sizes ``ffn_*_exps`` tensor creation off ``hparams.n_expert`` — without
  these keys a from-safetensors MoE pack has real 3-D expert tensors on disk
  but ``hparams.n_expert == 0``, so llama.cpp's QWEN2MOE loader throws
  (``n_expert must be > 0 for QWEN2MOE``) before even getting to the tensors.
  Fixed by reading ``num_experts``/``num_local_experts``,
  ``num_experts_per_tok``/``num_experts_per_token``, ``moe_intermediate_size``,
  and ``shared_expert_intermediate_size`` generically for any arch.
- The HF->GGUF shared-expert pattern table only matched the DeepSeek/DeepSeek2
  plural ``mlp.shared_experts.*``. Real Qwen2MoE (and Llama4) checkpoints use
  the SINGULAR ``mlp.shared_expert.{gate,up,down}_proj.weight`` (confirmed
  against llama.cpp's ``gguf-py/gguf/tensor_mapping.py``) — unmapped, those
  tensors fell through to their raw HF names and llama.cpp's QWEN2MOE loader
  refused to load (``missing tensor 'blk.0.ffn_gate_shexp.weight'`` etc., and
  ``missing tensor 'blk.0.ffn_gate_inp_shexp.weight'`` for the shared-expert
  gate scalar, which had no mapping at all). Both are now mapped.

With those two fixes, a from-scratch synthetic qwen2_moe checkpoint (no
pretrained weights, real Qwen2.5-0.5B tokenizer files for authentic vocab/
chat-template metadata) packs to a GGUF that loads and generates in real
llama.cpp, at both BF16 and imatrix-weighted Q4_K-experts precision.
"""
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from magicquant.gguf.source import SafetensorsSource
from magicquant.gguf.writer import create_hybrid_gguf
from magicquant.imatrix import capture_imatrix, load_imatrix

HIDDEN = 256
MOE_INTER = 256
SHARED_INTER = HIDDEN * 2
N_EXPERTS = 8
N_EXPERTS_PER_TOK = 2
N_LAYERS = 2
N_HEADS = 4
N_KV_HEADS = 4
VOCAB = 151936

_LLAMACPP_BUILD_BIN = Path("/home/lucas/llama.cpp/build/bin")
_TOKENIZER_SRC = Path(
    os.environ.get("MOE_E2E_TOKENIZER_DIR", "/server/ai/models/source/Qwen2.5-0.5B")
)
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")

_CALIB_CORPUS = """\
The quick brown fox jumps over the lazy dog near the riverbank at dawn.
Scientists study the behavior of neurons to understand how memories form in the brain.
In the mountains, the weather can change quickly from sunshine to storms within an hour.
The history of computing spans mechanical calculators, vacuum tubes, and modern silicon chips.
Cooking a good meal requires fresh ingredients, careful timing, and a bit of patience.
The stock market fluctuates based on investor sentiment, economic data, and global events.
Ancient civilizations built monuments that still stand as testaments to human ingenuity.
Machine learning models learn patterns from data by adjusting millions of parameters.
The ocean covers most of the planet and remains largely unexplored by humans.
Music theory explains how rhythm, harmony, and melody combine to create emotion.
Space exploration has revealed thousands of exoplanets orbiting distant stars.
Gardening in spring means preparing the soil, planting seeds, and watering regularly.
The novel's plot twisted unexpectedly when the detective discovered a hidden letter.
Renewable energy sources like solar and wind are becoming more cost effective every year.
Chess is a game of strategy where every move can shift the balance of the board.
"""


def _find_binary(name):
    """Prefer the explicit build this validation was run against; fall back to PATH."""
    explicit = _LLAMACPP_BUILD_BIN / name
    if explicit.exists():
        return str(explicit)
    return shutil.which(name)


_LLAMA_COMPLETION = _find_binary("llama-completion") or _find_binary("llama-cli")
_LLAMA_IMATRIX = _find_binary("llama-imatrix")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_LLAMA_COMPLETION and _LLAMA_IMATRIX and _TOKENIZER_SRC.is_dir()),
        reason=(
            "needs llama-completion/llama-cli + llama-imatrix binaries "
            f"(looked in {_LLAMACPP_BUILD_BIN} and PATH) and a real HF "
            f"tokenizer directory at {_TOKENIZER_SRC} (override with "
            "MOE_E2E_TOKENIZER_DIR)"
        ),
    ),
]


def _randmat(rng, rows, cols, scale=0.02):
    return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)


def _bf16_bytes(arr_f32):
    u32 = arr_f32.astype(np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def _write_safetensors(path, tensors, bf16_names=()):
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        if name in bf16_names:
            blob = _bf16_bytes(arr).tobytes()
            dtype = "BF16"
        else:
            blob = arr.astype(np.float32).tobytes()
            dtype = "F32"
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    header["__metadata__"] = {"format": "pt"}
    hdr_bytes = json.dumps(header).encode("utf-8")
    hdr_bytes += b" " * ((-len(hdr_bytes)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for blob in blobs:
            f.write(blob)


def _run_completion(model_path, prompt="Hello", n_predict=8, timeout=60):
    cmd = [
        _LLAMA_COMPLETION, "-m", str(model_path),
        "-p", prompt, "-n", str(n_predict), "--no-warmup",
    ]
    return subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
    )


def _assert_loaded_and_generated(result):
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "missing tensor" not in combined, combined
    assert "error loading model" not in combined, combined
    assert "eval time" in combined, combined


@pytest.fixture(scope="session")
def moe_checkpoint_dir(tmp_path_factory):
    """A tiny but realistic qwen2_moe safetensors checkpoint: coherent
    attention/expert dims, real Qwen2.5-0.5B tokenizer files for authentic
    vocab + chat-template metadata, random weights (load/shape correctness
    is what's under test, not output quality)."""
    out_dir = tmp_path_factory.mktemp("moe_checkpoint")
    for fname in _TOKENIZER_FILES:
        shutil.copy(_TOKENIZER_SRC / fname, out_dir / fname)

    config = {
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "moe_intermediate_size": MOE_INTER,
        "shared_expert_intermediate_size": SHARED_INTER,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": N_EXPERTS_PER_TOK,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "vocab_size": VOCAB,
        "max_position_embeddings": 2048,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
    }
    (out_dir / "config.json").write_text(json.dumps(config))

    rng = np.random.default_rng(1234)
    tensors = {
        "model.embed_tokens.weight": _randmat(rng, VOCAB, HIDDEN, scale=0.01),
        "lm_head.weight": _randmat(rng, VOCAB, HIDDEN, scale=0.01),
        "model.norm.weight": np.ones(HIDDEN, dtype=np.float32),
    }
    for layer in range(N_LAYERS):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = _randmat(rng, HIDDEN, HIDDEN)
        tensors[f"{p}.self_attn.k_proj.weight"] = _randmat(rng, HIDDEN, HIDDEN)
        tensors[f"{p}.self_attn.v_proj.weight"] = _randmat(rng, HIDDEN, HIDDEN)
        tensors[f"{p}.self_attn.o_proj.weight"] = _randmat(rng, HIDDEN, HIDDEN)

        tensors[f"{p}.mlp.gate.weight"] = _randmat(rng, N_EXPERTS, HIDDEN, scale=0.05)
        for e in range(N_EXPERTS):
            tensors[f"{p}.mlp.experts.{e}.gate_proj.weight"] = _randmat(rng, MOE_INTER, HIDDEN)
            tensors[f"{p}.mlp.experts.{e}.up_proj.weight"] = _randmat(rng, MOE_INTER, HIDDEN)
            tensors[f"{p}.mlp.experts.{e}.down_proj.weight"] = _randmat(rng, HIDDEN, MOE_INTER)

        tensors[f"{p}.mlp.shared_expert_gate.weight"] = (
            rng.standard_normal(HIDDEN) * 0.05
        ).astype(np.float32)
        tensors[f"{p}.mlp.shared_expert.gate_proj.weight"] = _randmat(rng, SHARED_INTER, HIDDEN)
        tensors[f"{p}.mlp.shared_expert.up_proj.weight"] = _randmat(rng, SHARED_INTER, HIDDEN)
        tensors[f"{p}.mlp.shared_expert.down_proj.weight"] = _randmat(rng, HIDDEN, SHARED_INTER)

    _write_safetensors(
        out_dir / "model.safetensors", tensors,
        bf16_names={"model.embed_tokens.weight", "lm_head.weight"},
    )
    return out_dir


@pytest.fixture(scope="session")
def moe_bf16_gguf(moe_checkpoint_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("moe_gguf") / "bf16.gguf"
    create_hybrid_gguf(str(out), str(moe_checkpoint_dir), {"base": "BF16", "groups": {}}, verbose=False)
    return out


@pytest.fixture(scope="session")
def moe_imatrix_path(moe_bf16_gguf, tmp_path_factory):
    corpus = tmp_path_factory.mktemp("moe_corpus") / "corpus.txt"
    corpus.write_text(_CALIB_CORPUS)
    out = tmp_path_factory.mktemp("moe_imatrix") / "imatrix.gguf"
    capture_imatrix(
        moe_bf16_gguf, corpus, out,
        chunks=8, ctx_size=64, imatrix_bin=_LLAMA_IMATRIX, timeout=90,
    )
    return out


# --- Step 2 checkpoint: does the metadata builder support qwen2_moe? -------

def test_metadata_has_moe_keys(moe_checkpoint_dir):
    src = SafetensorsSource(str(moe_checkpoint_dir))
    meta = src.get_metadata()
    assert meta["general.architecture"] == "qwen2moe"
    assert meta["qwen2moe.expert_count"] == N_EXPERTS
    assert meta["qwen2moe.expert_used_count"] == N_EXPERTS_PER_TOK
    assert meta["qwen2moe.expert_feed_forward_length"] == MOE_INTER
    assert meta["qwen2moe.expert_shared_feed_forward_length"] == SHARED_INTER


def test_expert_and_shared_expert_tensors_present(moe_checkpoint_dir):
    src = SafetensorsSource(str(moe_checkpoint_dir))
    infos = {i["name"]: i for i in src.get_all_tensors_info()}
    for layer in range(N_LAYERS):
        assert infos[f"blk.{layer}.ffn_gate_exps.weight"]["shape"] == [N_EXPERTS, MOE_INTER, HIDDEN]
        assert infos[f"blk.{layer}.ffn_up_exps.weight"]["shape"] == [N_EXPERTS, MOE_INTER, HIDDEN]
        assert infos[f"blk.{layer}.ffn_down_exps.weight"]["shape"] == [N_EXPERTS, HIDDEN, MOE_INTER]
        assert infos[f"blk.{layer}.ffn_gate_inp.weight"]["shape"] == [N_EXPERTS, HIDDEN]
        # qwen2moe's shared-expert branch: llama.cpp's loader requires all
        # four of these unconditionally (no TENSOR_NOT_REQUIRED).
        assert infos[f"blk.{layer}.ffn_gate_inp_shexp.weight"]["shape"] == [HIDDEN]
        assert infos[f"blk.{layer}.ffn_gate_shexp.weight"]["shape"] == [SHARED_INTER, HIDDEN]
        assert infos[f"blk.{layer}.ffn_up_shexp.weight"]["shape"] == [SHARED_INTER, HIDDEN]
        assert infos[f"blk.{layer}.ffn_down_shexp.weight"]["shape"] == [HIDDEN, SHARED_INTER]


# --- Step 3: real llama.cpp load test of the packed BF16 GGUF --------------

def test_bf16_gguf_loads_and_generates(moe_bf16_gguf):
    _assert_loaded_and_generated(_run_completion(moe_bf16_gguf))


# --- Step 4: per-expert imatrix, Q4_K re-pack, load test again -------------

def test_imatrix_per_expert_entries_are_expert_major_sized(moe_imatrix_path):
    imat = load_imatrix(moe_imatrix_path)
    for layer in range(N_LAYERS):
        for proj in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            assert name in imat, sorted(imat)
            assert imat[name].shape == (N_EXPERTS * MOE_INTER,)
            assert np.isfinite(imat[name]).all()


def test_q4k_hybrid_with_imatrix_loads_and_generates(moe_checkpoint_dir, moe_imatrix_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("moe_q4k") / "q4k_hybrid.gguf"
    create_hybrid_gguf(
        str(out), str(moe_checkpoint_dir),
        {"base": "BF16", "groups": {"X": "Q4_K_M"}},
        verbose=False, imatrix=str(moe_imatrix_path),
    )
    _assert_loaded_and_generated(_run_completion(out))
