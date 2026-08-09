"""Regression tests for FIX 4: the BF16->F16 downgrade (gguf/writer.py) must
detect out-of-F16-range values BEFORE writing them, instead of silently
producing Inf with only an advisory warning.

Incident context: gguf/writer.py:778-791 downgrades any BF16-designated
group to F16 on disk (llama.cpp's BF16 compute-graph limitation), warning
once but proceeding unconditionally. F16's finite range is much narrower
than BF16's -- a value with |v| > 65504 becomes Inf. Those Inf tensors then
measure as NaN perplexity, which is exactly the degenerate input the parser
(llamacpp.py) and clamp (probing.py) fixes exist to catch downstream --
this fix stops it at the source instead.

MAJOR 1 (later fix, same file): an earlier version of this detector ALSO
flagged subnormal underflow (very small nonzero values going to 0), which
fired on essentially every real weight tensor and silently substituted
Q8_0 for BF16-designated groups across the board. That branch was deleted;
see test_detector_does_not_flag_subnormal_underflow_value and
test_detector_does_not_flag_a_realistic_weight_tensor below. Underflow to
0 is inconsequential; only overflow to Inf (and pre-existing non-finite
source values) are real hazards.
"""
import numpy as np

from magicquant.gguf.source import ModelSource
from magicquant.gguf.writer import create_hybrid_gguf, _bf16_to_f16_would_corrupt


class _StubSource(ModelSource):
    """In-memory source with one controllable tensor's real values, mirroring
    tests/test_writer.py's StubSource."""

    def __init__(self, tensors, metadata=None):
        # tensors: list of (name, np.ndarray-f32, shape)
        self._tensors = tensors
        self._metadata = metadata or {"general.architecture": "llama"}

    def get_metadata(self):
        return dict(self._metadata)

    def get_tensor_names(self):
        return [n for (n, _a, _s) in self._tensors]

    def get_all_tensors_info(self):
        return [
            {"name": name, "n_dims": len(shape), "shape": list(shape), "data_type": 0}
            for (name, _arr, shape) in self._tensors
        ]

    def read_tensor_f32(self, tensor_name):
        for (name, arr, _shape) in self._tensors:
            if name == tensor_name:
                return arr.astype(np.float32)
        return None

    def get_source_type_name(self, tensor_name):
        return "F32"


def _f32_tensor(name, shape, values=None, seed=0):
    n = 1
    for d in shape:
        n *= d
    if values is None:
        # Seeded: an unseeded randn() draw is ~1-in-320 flaky here (some
        # element occasionally lands close enough to 0.0, or far enough
        # into the tail, to change which branch a test exercises). Pin the
        # seed so the fixture is deterministic.
        rng = np.random.RandomState(seed)
        arr = rng.randn(n).astype(np.float32)
    else:
        arr = np.array(values, dtype=np.float32)
        assert arr.size == n
    return (name, arr, shape)


def _build(tmp_path, src, group_schemes, base="Q8_0"):
    out = str(tmp_path / "out.gguf")
    import magicquant.gguf.source as source_mod
    orig = source_mod.open_model_source
    source_mod.open_model_source = lambda *a, **k: src
    try:
        create_hybrid_gguf(
            output_path=out,
            base_model_path="ignored",
            quant_config={"base": base, "groups": group_schemes},
            verbose=False,
        )
    finally:
        source_mod.open_model_source = orig

    from magicquant.gguf.reader import GGUFReader
    reader = GGUFReader(out)
    reader.open()
    return reader


# ---------------------------------------------------------------------------
# Unit tests for the detector itself
# ---------------------------------------------------------------------------

def test_detector_flags_overflow_value():
    src = _StubSource([_f32_tensor("t", (4,), [1.0, 2.0, 70000.0, 3.0])])
    assert _bf16_to_f16_would_corrupt(src, "t") is True


def test_detector_does_not_flag_subnormal_underflow_value():
    """MAJOR 1 regression: a prior version of the detector flagged
    subnormal underflow (|v| < F16's smallest subnormal, ~5.96e-8) as
    "would corrupt". That branch fired on essentially every real weight
    tensor (any large real tensor has SOME element by chance landing that
    close to 0.0), so every BF16-designated group was silently substituted
    with Q8_0 across the board. Underflow to 0 is inconsequential; only
    overflow to Inf (tested above) and pre-existing non-finite values
    (tested below) are real hazards."""
    src = _StubSource([_f32_tensor("t", (4,), [1.0, 2.0, 1e-30, 3.0])])
    assert _bf16_to_f16_would_corrupt(src, "t") is False


def test_detector_does_not_flag_a_realistic_weight_tensor():
    """Direct regression for the incident that motivated this fix: a
    4096x4096 N(0, 0.02)-initialized tensor (a completely ordinary weight
    matrix, max|v| ~= 0.11 -- nowhere near F16's 65504 ceiling) must NOT be
    flagged, even though some of its ~16.7M values are inevitably very
    close to zero."""
    rng = np.random.RandomState(0)
    values = (rng.randn(4096 * 4096) * 0.02).astype(np.float32)
    assert float(np.max(np.abs(values))) < 65504.0
    src = _StubSource([("t", values, (4096, 4096))])
    assert _bf16_to_f16_would_corrupt(src, "t") is False


def test_detector_flags_existing_nonfinite_source_values():
    src = _StubSource([_f32_tensor("t", (4,), [1.0, float("nan"), 3.0, 4.0])])
    assert _bf16_to_f16_would_corrupt(src, "t") is True


def test_detector_passes_ordinary_values():
    src = _StubSource([_f32_tensor("t", (4,), [0.1, -0.2, 1.5, -3.0])])
    assert _bf16_to_f16_would_corrupt(src, "t") is False


def test_detector_conservative_when_source_cannot_decode():
    class _NoDecodeSource(_StubSource):
        def read_tensor_f32(self, tensor_name):
            return None

    src = _NoDecodeSource([_f32_tensor("t", (4,))])
    assert _bf16_to_f16_would_corrupt(src, "t") is False


# ---------------------------------------------------------------------------
# End-to-end: the writer must substitute Q8_0, not silently write Inf/0
# ---------------------------------------------------------------------------

def test_bf16_downgrade_of_out_of_range_tensor_substitutes_q8_0_not_f16(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="magicquant.gguf.writer")

    # A row of otherwise-ordinary values plus one value far outside F16's
    # finite range (BF16 can represent this; F16 cannot).
    values = [1.0] * 255 + [1.0e10]
    src = _StubSource([
        _f32_tensor("token_embd.weight", (1, 256), values),
        _f32_tensor("blk.0.attn_q.weight", (256, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
        _f32_tensor("output.weight", (256, 256)),
    ])

    reader = _build(tmp_path, src, {"E": "BF16"})
    try:
        info = reader.get_tensor_info("token_embd.weight")
        # Q8_0's ggml type id; must NOT be F16 (id 1) or BF16.
        from magicquant.gguf.writer import GGML_TYPE
        assert info["data_type"] == GGML_TYPE["Q8_0"], (
            "out-of-range BF16 tensor must be substituted with Q8_0, not "
            "silently written as F16 (which would be Inf for this value)"
        )
    finally:
        reader.close()

    assert any(
        "out-of-F16-range" in rec.message.lower()
        or "substituting q8_0" in rec.message.lower()
        for rec in caplog.records
    ), "expected a WARNING naming the out-of-range tensor and the Q8_0 substitution"


def test_bf16_downgrade_of_in_range_tensor_still_uses_f16(tmp_path):
    """Ordinary (in-range) values must still take the historical F16 path --
    this fix must not change behavior for the common case."""
    src = _StubSource([
        _f32_tensor("token_embd.weight", (256, 256)),  # random, in range
        _f32_tensor("blk.0.attn_q.weight", (256, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
        _f32_tensor("output.weight", (256, 256)),
    ])
    reader = _build(tmp_path, src, {"E": "BF16"})
    try:
        info = reader.get_tensor_info("token_embd.weight")
        assert info["data_type"] == 1  # F16
    finally:
        reader.close()
