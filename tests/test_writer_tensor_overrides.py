"""Per-tensor scheme overrides in the GGUF writer's create_hybrid_gguf.

quant_config gains an optional third key, "tensors": {exact_tensor_name:
scheme_name}, which takes precedence over group/base resolution for
exactly-matching tensor names. Resolution order: tensors[name] > groups[group]
> base. Mirrors the synthetic-model StubSource fixture pattern used in
tests/test_writer.py (kept self-contained here rather than imported, since
this file must not modify test_writer.py).
"""
import numpy as np
import pytest

from magicquant.gguf.source import ModelSource
from magicquant.gguf.writer import GGUFWriter, create_hybrid_gguf, GGML_TYPE


# ---------------------------------------------------------------------------
# Stub source (same shape as tests/test_writer.py's StubSource)
# ---------------------------------------------------------------------------

class StubSource(ModelSource):
    """In-memory F32 source with a few tensors."""

    def __init__(self, tensors, *, metadata=None):
        # tensors: list of (name, np.ndarray-f32, shape)
        self._tensors = tensors
        self._metadata = metadata or {"general.architecture": "llama"}

    def get_metadata(self):
        return dict(self._metadata)

    def get_tensor_names(self):
        return [n for (n, _a, _s) in self._tensors]

    def get_all_tensors_info(self):
        infos = []
        for (name, arr, shape) in self._tensors:
            infos.append({
                "name": name,
                "n_dims": len(shape),
                "shape": list(shape),
                "data_type": 0,  # F32
            })
        return infos

    def read_tensor_f32(self, tensor_name):
        for (name, arr, shape) in self._tensors:
            if name == tensor_name:
                return arr.astype(np.float32)
        return None

    def get_source_type_name(self, tensor_name):
        return "F32"


def _f32_tensor(name, shape):
    n = 1
    for d in shape:
        n *= d
    return (name, np.random.randn(n).astype(np.float32), shape)


def _ffn_down_source():
    return StubSource([
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
        _f32_tensor("blk.1.ffn_down.weight", (256, 256)),
        _f32_tensor("blk.2.ffn_down.weight", (256, 256)),
    ])


def _libggml_available():
    try:
        from magicquant.quant.ggml_binding import get_handle
        get_handle()
        return True
    except Exception:
        return False


requires_libggml = pytest.mark.skipif(
    not _libggml_available(), reason="libggml not available"
)


# ---------------------------------------------------------------------------
# (a) override on one FFN tensor is honored; group siblings keep group scheme
# ---------------------------------------------------------------------------

@requires_libggml
def test_tensor_override_honored_siblings_keep_group_scheme(tmp_path, monkeypatch):
    import magicquant.gguf.source as source_mod

    src = _ffn_down_source()
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    out = str(tmp_path / "out.gguf")
    quant_config = {
        "base": "Q8_0",
        "groups": {"D": "Q6_K"},
        "tensors": {"blk.1.ffn_down.weight": "Q4_K_M"},
    }
    create_hybrid_gguf(
        output_path=out, base_model_path="ignored",
        quant_config=quant_config, verbose=False,
    )

    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(out)
    reader.open()
    try:
        info0 = reader.get_tensor_info("blk.0.ffn_down.weight")
        info1 = reader.get_tensor_info("blk.1.ffn_down.weight")
        info2 = reader.get_tensor_info("blk.2.ffn_down.weight")
    finally:
        reader.close()

    assert info0["data_type"] == GGML_TYPE["Q6_K"], "sibling 0 must keep group scheme"
    assert info2["data_type"] == GGML_TYPE["Q6_K"], "sibling 2 must keep group scheme"
    assert info1["data_type"] == GGML_TYPE["Q4_K"], "overridden tensor must use its own scheme"


# ---------------------------------------------------------------------------
# (b) quant_config without "tensors" == groups-only call (byte-identical
#     Pass-1 type decisions)
# ---------------------------------------------------------------------------

@requires_libggml
def test_missing_tensors_key_matches_groups_only_call(tmp_path, monkeypatch):
    import magicquant.gguf.source as source_mod

    def _build(out_name, quant_config):
        src = _ffn_down_source()
        monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
        out = str(tmp_path / out_name)
        create_hybrid_gguf(
            output_path=out, base_model_path="ignored",
            quant_config=quant_config, verbose=False,
        )
        from magicquant.gguf.reader import GGUFReader
        reader = GGUFReader(out)
        reader.open()
        try:
            infos = reader.get_all_tensors_info()
            return {t["name"]: t["data_type"] for t in infos}
        finally:
            reader.close()

    groups_only = {"base": "Q8_0", "groups": {"D": "Q6_K"}}
    with_empty_tensors = {"base": "Q8_0", "groups": {"D": "Q6_K"}, "tensors": {}}

    types_a = _build("a.gguf", groups_only)
    types_b = _build("b.gguf", with_empty_tensors)

    assert types_a == types_b


# ---------------------------------------------------------------------------
# (c) unknown scheme name in overrides raises ValueError up front
# ---------------------------------------------------------------------------

def test_unknown_scheme_in_overrides_raises_value_error(tmp_path, monkeypatch):
    import magicquant.gguf.source as source_mod

    src = _ffn_down_source()
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    out = str(tmp_path / "out.gguf")
    quant_config = {
        "base": "Q8_0",
        "groups": {"D": "Q6_K"},
        "tensors": {"blk.1.ffn_down.weight": "NOT_A_REAL_SCHEME"},
    }
    with pytest.raises(ValueError, match="NOT_A_REAL_SCHEME"):
        create_hybrid_gguf(
            output_path=out, base_model_path="ignored",
            quant_config=quant_config, verbose=False,
        )

    assert not (tmp_path / "out.gguf").exists()
    assert not (tmp_path / "out.gguf.partial").exists()


# ---------------------------------------------------------------------------
# (d) override naming a nonexistent tensor: warns, does not raise, file builds
# ---------------------------------------------------------------------------

@requires_libggml
def test_override_naming_nonexistent_tensor_warns_but_builds(tmp_path, monkeypatch, caplog):
    import logging
    import magicquant.gguf.source as source_mod

    caplog.set_level(logging.WARNING, logger="magicquant.gguf.writer")

    src = _ffn_down_source()
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    out = str(tmp_path / "out.gguf")
    quant_config = {
        "base": "Q8_0",
        "groups": {"D": "Q6_K"},
        "tensors": {"blk.999.ffn_down.weight": "Q4_K_M"},
    }
    result = create_hybrid_gguf(
        output_path=out, base_model_path="ignored",
        quant_config=quant_config, verbose=False,
    )

    assert result == out
    from pathlib import Path
    assert Path(out).is_file()
    assert any(
        "blk.999.ffn_down.weight" in rec.message or "matched no tensor" in rec.message
        for rec in caplog.records
    ), "expected a warning naming the unmatched override"

    # Sibling tensors must still use the group scheme (override was a no-op
    # for the actual write, since it named nothing real).
    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(out)
    reader.open()
    try:
        info0 = reader.get_tensor_info("blk.0.ffn_down.weight")
    finally:
        reader.close()
    assert info0["data_type"] == GGML_TYPE["Q6_K"]


# ---------------------------------------------------------------------------
# Unknown-scheme validation also usable directly off GGUFWriter (not just the
# module-level convenience function).
# ---------------------------------------------------------------------------

def test_unknown_scheme_raises_via_writer_class_method(tmp_path, monkeypatch):
    import magicquant.gguf.source as source_mod

    src = _ffn_down_source()
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)

    writer = GGUFWriter(str(tmp_path / "out.gguf"))
    with pytest.raises(ValueError):
        writer.create_hybrid_gguf(
            "ignored",
            {"base": "Q8_0", "groups": {}, "tensors": {"x": "TOTALLY_BOGUS"}},
            verbose=False,
        )
