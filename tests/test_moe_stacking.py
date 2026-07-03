"""MoE expert-stacking tests.

HF MoE checkpoints store each expert's gate/up/down projection as its own
separate 2-D tensor (e.g. ``model.layers.0.mlp.experts.3.gate_proj.weight``).
llama.cpp's GGUF format instead expects all experts of a (layer, projection)
stacked into ONE 3-D tensor (``blk.0.ffn_gate_exps.weight``, shape
``[n_expert, out_features, in_features]``), experts in ascending index order
along the new leading axis. Before this fix, SafetensorsSource mapped each
expert 1:1 (last expert silently overwriting the others in ``_tensor_map``),
producing an unloadable GGUF.

VALIDATION COVERAGE: everything here is UNIT-level. It exercises
``_detect_moe_expert_tensor`` directly (pure function, no I/O) and
round-trips synthetic, hand-written .safetensors fixtures (tiny dims,
n_expert=4) through SafetensorsSource -- confirming the stacked tensor's
name/shape/dtype and the byte-for-byte concatenation order. It does NOT
load a real MoE checkpoint or feed the result through the GGUF writer /
llama.cpp loader: no small real-world MoE model is available on this box.
Full end-to-end validation (writing an actual GGUF and loading it in
llama.cpp) is still outstanding and needs a real (or at least
realistically-shaped) MoE checkpoint.
"""
import json
import struct

import numpy as np
import pytest

from magicquant.gguf.source import SafetensorsSource, _detect_moe_expert_tensor


# ---------------------------------------------------------------------------
# helpers: hand-write a minimal multi-tensor safetensors model dir
# (same pattern as tests/test_source_qk_permute.py)
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors):
    """tensors: {name: float32 ndarray} -> minimal .safetensors file."""
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        blob = arr.astype(np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    hdr = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for blob in blobs:
            f.write(blob)


def _write_config(tmp_path, model_type, n_expert, hidden=4, inter=6):
    cfg = {
        "model_type": model_type,
        "hidden_size": hidden,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": inter,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "num_local_experts": n_expert,
        "num_experts_per_tok": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))


def _expert_tensor(expert_idx, shape):
    """A distinct, easily-checked per-expert tensor: every element == expert_idx."""
    return np.full(shape, float(expert_idx), dtype=np.float32)


# ---------------------------------------------------------------------------
# (a)+(b)+(c): generic (Qwen/DeepSeek) per-expert naming -> stacked tensor
# ---------------------------------------------------------------------------

def test_generic_experts_stack_name_shape_ndims(tmp_path):
    n_expert, hidden, inter = 4, 4, 6
    _write_config(tmp_path, "qwen2_moe", n_expert, hidden, inter)

    tensors = {}
    for e in range(n_expert):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = _expert_tensor(e, (inter, hidden))
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = _expert_tensor(e, (inter, hidden))
        tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = _expert_tensor(e, (hidden, inter))
    tensors["model.layers.0.mlp.gate.weight"] = np.arange(n_expert * hidden, dtype=np.float32).reshape(n_expert, hidden)
    _write_safetensors(tmp_path / "model.safetensors", tensors)

    src = SafetensorsSource(str(tmp_path))
    infos = {i["name"]: i for i in src.get_all_tensors_info()}

    for proj, shape in (("gate", [n_expert, inter, hidden]),
                         ("up", [n_expert, inter, hidden]),
                         ("down", [n_expert, hidden, inter])):
        name = f"blk.0.ffn_{proj}_exps.weight"
        assert name in infos, f"missing stacked tensor {name}"
        info = infos[name]
        assert info["n_dims"] == 3
        assert info["shape"] == shape

    # router is untouched, unstacked, correct name
    assert "blk.0.ffn_gate_inp.weight" in infos
    assert infos["blk.0.ffn_gate_inp.weight"]["n_dims"] == 2
    assert infos["blk.0.ffn_gate_inp.weight"]["shape"] == [n_expert, hidden]


def test_generic_experts_read_order_and_no_individual_names(tmp_path):
    n_expert, hidden, inter = 4, 4, 6
    _write_config(tmp_path, "qwen2_moe", n_expert, hidden, inter)

    tensors = {}
    # Insert in REVERSE expert order to prove the stacker sorts by index
    # rather than relying on header/file iteration order.
    for e in reversed(range(n_expert)):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = _expert_tensor(e, (inter, hidden))
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = _expert_tensor(e, (inter, hidden))
        tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = _expert_tensor(e, (hidden, inter))
    _write_safetensors(tmp_path / "model.safetensors", tensors)

    src = SafetensorsSource(str(tmp_path))

    for proj, shape in (("gate", (n_expert, inter, hidden)),
                         ("up", (n_expert, inter, hidden)),
                         ("down", (n_expert, hidden, inter))):
        flat = src.read_tensor_f32(f"blk.0.ffn_{proj}_exps.weight")
        assert flat is not None
        stacked = flat.reshape(shape)
        for e in range(n_expert):
            # every element of expert e's slab must equal e (expert 0 first, ...)
            assert np.all(stacked[e] == float(e)), f"expert slot {e} corrupted for {proj}"

    # (c) individual per-expert HF-derived names must NOT appear
    names = set(src.get_tensor_names())
    for e in range(n_expert):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.0.mlp.experts.{e}.{proj}.weight" not in names
        for proj_key in ("gate", "up", "down"):
            assert f"blk.0.ffn_{proj_key}_exps.{e}.weight" not in names


# ---------------------------------------------------------------------------
# (d): Mixtral w1/w2/w3 -> gate/down/up
# ---------------------------------------------------------------------------

def test_mixtral_w1_w2_w3_map_to_gate_down_up(tmp_path):
    n_expert, hidden, inter = 4, 4, 6
    _write_config(tmp_path, "mixtral", n_expert, hidden, inter)

    tensors = {}
    for e in range(n_expert):
        tensors[f"model.layers.0.block_sparse_moe.experts.{e}.w1.weight"] = _expert_tensor(e, (inter, hidden))  # gate
        tensors[f"model.layers.0.block_sparse_moe.experts.{e}.w3.weight"] = _expert_tensor(e, (inter, hidden))  # up
        tensors[f"model.layers.0.block_sparse_moe.experts.{e}.w2.weight"] = _expert_tensor(e, (hidden, inter))  # down
    tensors["model.layers.0.block_sparse_moe.gate.weight"] = np.arange(n_expert * hidden, dtype=np.float32).reshape(n_expert, hidden)
    _write_safetensors(tmp_path / "model.safetensors", tensors)

    src = SafetensorsSource(str(tmp_path))
    infos = {i["name"]: i for i in src.get_all_tensors_info()}

    assert infos["blk.0.ffn_gate_exps.weight"]["shape"] == [n_expert, inter, hidden]
    assert infos["blk.0.ffn_up_exps.weight"]["shape"] == [n_expert, inter, hidden]
    assert infos["blk.0.ffn_down_exps.weight"]["shape"] == [n_expert, hidden, inter]

    gate = src.read_tensor_f32("blk.0.ffn_gate_exps.weight").reshape(n_expert, inter, hidden)
    up = src.read_tensor_f32("blk.0.ffn_up_exps.weight").reshape(n_expert, inter, hidden)
    down = src.read_tensor_f32("blk.0.ffn_down_exps.weight").reshape(n_expert, hidden, inter)
    for e in range(n_expert):
        assert np.all(gate[e] == float(e))
        assert np.all(up[e] == float(e))
        assert np.all(down[e] == float(e))

    # Mixtral router maps to the same ffn_gate_inp name as the generic case,
    # and is NOT absorbed into the expert stack.
    assert "blk.0.ffn_gate_inp.weight" in infos
    assert infos["blk.0.ffn_gate_inp.weight"]["n_dims"] == 2
    names = set(src.get_tensor_names())
    for e in range(n_expert):
        for w in ("w1", "w2", "w3"):
            assert f"model.layers.0.block_sparse_moe.experts.{e}.{w}.weight" not in names


# ---------------------------------------------------------------------------
# (e): shared experts map 1:1, NOT stacked
# ---------------------------------------------------------------------------

def test_shared_experts_map_1to1_not_stacked(tmp_path):
    n_expert, hidden, inter = 4, 4, 6
    _write_config(tmp_path, "qwen2_moe", n_expert, hidden, inter)

    rng = np.random.default_rng(42)
    shared_gate = rng.standard_normal((inter, hidden)).astype(np.float32)
    shared_up = rng.standard_normal((inter, hidden)).astype(np.float32)
    shared_down = rng.standard_normal((hidden, inter)).astype(np.float32)

    tensors = {
        "model.layers.0.mlp.shared_experts.gate_proj.weight": shared_gate,
        "model.layers.0.mlp.shared_experts.up_proj.weight": shared_up,
        "model.layers.0.mlp.shared_experts.down_proj.weight": shared_down,
    }
    # Also include routed experts so the stacked group exists alongside shared.
    for e in range(n_expert):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = _expert_tensor(e, (inter, hidden))
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = _expert_tensor(e, (inter, hidden))
        tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = _expert_tensor(e, (hidden, inter))
    _write_safetensors(tmp_path / "model.safetensors", tensors)

    src = SafetensorsSource(str(tmp_path))
    infos = {i["name"]: i for i in src.get_all_tensors_info()}

    for proj, name, expected in (
        ("gate", "blk.0.ffn_gate_shexp.weight", shared_gate),
        ("up", "blk.0.ffn_up_shexp.weight", shared_up),
        ("down", "blk.0.ffn_down_shexp.weight", shared_down),
    ):
        assert name in infos, f"missing shared-expert tensor {name}"
        assert infos[name]["n_dims"] == 2
        assert infos[name]["shape"] == list(expected.shape)
        got = src.read_tensor_f32(name).reshape(expected.shape)
        np.testing.assert_array_equal(got, expected)

    # Shared-expert tensors must not have been absorbed into the routed stack.
    assert infos["blk.0.ffn_gate_exps.weight"]["shape"] == [n_expert, inter, hidden]


# ---------------------------------------------------------------------------
# _detect_moe_expert_tensor: pure-function unit tests (no I/O)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("model.layers.0.mlp.experts.0.gate_proj.weight", ("blk.0.ffn_gate_exps.weight", 0, "gate")),
    ("model.layers.0.mlp.experts.3.up_proj.weight", ("blk.0.ffn_up_exps.weight", 3, "up")),
    ("model.layers.12.mlp.experts.7.down_proj.weight", ("blk.12.ffn_down_exps.weight", 7, "down")),
    ("model.layers.0.block_sparse_moe.experts.0.w1.weight", ("blk.0.ffn_gate_exps.weight", 0, "gate")),
    ("model.layers.0.block_sparse_moe.experts.2.w3.weight", ("blk.0.ffn_up_exps.weight", 2, "up")),
    ("model.layers.0.block_sparse_moe.experts.1.w2.weight", ("blk.0.ffn_down_exps.weight", 1, "down")),
])
def test_detect_moe_expert_tensor_matches(name, expected):
    assert _detect_moe_expert_tensor(name) == expected


@pytest.mark.parametrize("name", [
    "model.layers.0.mlp.gate.weight",                       # router, not per-expert
    "model.layers.0.block_sparse_moe.gate.weight",           # Mixtral router
    "model.layers.0.mlp.shared_experts.gate_proj.weight",    # shared expert, 1:1
    "model.layers.0.mlp.shared_experts.up_proj.weight",
    "model.layers.0.mlp.shared_experts.down_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",                # unrelated
    "model.layers.0.mlp.up_proj.weight",                     # dense FFN, no expert index
])
def test_detect_moe_expert_tensor_none_for_non_expert_names(name):
    assert _detect_moe_expert_tensor(name) is None


def test_detect_moe_expert_tensor_stacked_name_shared_across_experts():
    # all experts of the same (layer, projection) must produce the SAME
    # stacked name -- that's what lets the header-parsing loop group them.
    names = [f"model.layers.5.mlp.experts.{e}.gate_proj.weight" for e in range(6)]
    stacked_names = {_detect_moe_expert_tensor(n)[0] for n in names}
    assert stacked_names == {"blk.5.ffn_gate_exps.weight"}
