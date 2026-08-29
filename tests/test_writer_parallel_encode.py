"""Parallel encode pool (MAGICQUANT_ENCODE_THREADS) tests.

The writer's Pass 2 historically ran a single background read+encode thread
feeding a bounded queue (maxsize=2). That thread is now one of N pool workers
(default min(8, os.cpu_count() // 2), MAGICQUANT_ENCODE_THREADS overrides;
``1`` reproduces the exact historical single-worker path). These tests check
the three invariants the pool must preserve:

  1. Byte-identical output regardless of thread count, for mixed quant
     schemes including dense-imatrix and per-expert-imatrix tensors.
  2. The crash-safety contract (.partial + os.replace, worker exception ->
     unlink, no leaked in-flight threads) still holds with N>1.
  3. The byte budget (_ByteBudget) actually caps in-flight decoded+encoded
     bytes rather than just tensor count.
"""
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import magicquant.gguf.source as source_mod
import magicquant.gguf.writer as writer_mod
from magicquant.gguf.writer import create_hybrid_gguf

from tests.test_writer import StubSource


# ---------------------------------------------------------------------------
# Shared synthetic multi-tensor model: a mix of groups/shapes/schemes so a
# single-vs-N-thread comparison exercises K-quants, block-32 quants, a BF16
# downgrade, 1D norm-forced-F32, and a block-size fallback all in one build.
# ---------------------------------------------------------------------------

def _f32(name, shape, seed):
    n = 1
    for d in shape:
        n *= d
    rng = np.random.default_rng(seed)
    return (name, rng.standard_normal(n).astype(np.float32), shape)


def _expert(name, n_expert, out_f, in_f, seed):
    n = n_expert * out_f * in_f
    rng = np.random.default_rng(seed)
    return (name, rng.standard_normal(n).astype(np.float32), (n_expert, out_f, in_f))


def _build_tensors():
    return [
        _f32("token_embd.weight", (512, 256), 1),           # E
        _f32("blk.0.attn_q.weight", (256, 256), 2),         # Q (imatrix-weighted)
        _f32("blk.0.attn_k.weight", (256, 256), 3),         # K
        _f32("blk.0.attn_output.weight", (256, 256), 4),    # O
        _f32("blk.0.ffn_up.weight", (256, 256), 5),         # U
        _f32("blk.0.ffn_down.weight", (256, 256), 6),       # D
        _f32("blk.0.attn_norm.weight", (256,), 7),          # N (1D -> forced F32)
        _f32("blk.1.attn_q.weight", (256, 256), 8),         # Q
        _f32("blk.1.ffn_down.weight", (256, 256), 9),       # D
        _expert("blk.0.ffn_down_exps.weight", 4, 256, 128, 10),  # X (block-size fallback + per-expert imatrix)
        _f32("output.weight", (256, 256), 11),              # H (BF16 -> F16 downgrade)
    ]


_QUANT_CONFIG = {
    "base": "Q4_K_M",
    "groups": {
        "E": "Q8_0",
        "Q": "Q4_K_M",
        "K": "Q6_K",
        "O": "Q5_K",
        "U": "IQ4_NL",
        "D": "Q4_K_M",
        "H": "BF16",
        "X": "Q3_K",  # row width 128 on the expert tensor -> block-size fallback to MXFP4
    },
}


def _imatrix_for(tensors):
    imat = {}
    for name, _arr, shape in tensors:
        if name == "blk.0.attn_q.weight":
            imat[name] = np.linspace(0.1, 2.0, shape[-1], dtype=np.float32)
        if name == "blk.0.ffn_down_exps.weight":
            rng = np.random.default_rng(42)
            n_expert = shape[0]
            in_f = shape[-1]
            imat[name] = rng.random(n_expert * in_f).astype(np.float32) + 0.01
    return imat


def _build_gguf(tmp_path, out_name, n_threads, monkeypatch, budget_mb=None):
    tensors = _build_tensors()
    src = StubSource(tensors)
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
    monkeypatch.setenv("MAGICQUANT_ENCODE_THREADS", str(n_threads))
    if budget_mb is not None:
        monkeypatch.setenv("MAGICQUANT_ENCODE_BUDGET_MB", str(budget_mb))
    else:
        monkeypatch.delenv("MAGICQUANT_ENCODE_BUDGET_MB", raising=False)
    out = str(tmp_path / out_name)
    create_hybrid_gguf(
        output_path=out,
        base_model_path="ignored",
        quant_config=_QUANT_CONFIG,
        verbose=False,
        imatrix=_imatrix_for(tensors),
    )
    return out


# ---------------------------------------------------------------------------
# 1. Byte-identical output across thread counts
# ---------------------------------------------------------------------------

def test_byte_identical_output_across_thread_counts(tmp_path, monkeypatch):
    out1 = _build_gguf(tmp_path, "n1.gguf", 1, monkeypatch)
    b1 = Path(out1).read_bytes()

    out3 = _build_gguf(tmp_path, "n3.gguf", 3, monkeypatch)
    b3 = Path(out3).read_bytes()
    assert b1 == b3, "3-thread output must be byte-identical to the single-worker path"

    out8 = _build_gguf(tmp_path, "n8.gguf", 8, monkeypatch)
    b8 = Path(out8).read_bytes()
    assert b1 == b8, "8-thread output must be byte-identical to the single-worker path"


def test_byte_identical_with_tiny_budget_forcing_serialization(tmp_path, monkeypatch):
    # A budget smaller than nearly every tensor's footprint forces the pool
    # down to ~1-in-flight-at-a-time in practice; output must still match.
    out1 = _build_gguf(tmp_path, "n1.gguf", 1, monkeypatch)
    b1 = Path(out1).read_bytes()

    out_tight = _build_gguf(tmp_path, "tight.gguf", 4, monkeypatch, budget_mb=0.1)
    b_tight = Path(out_tight).read_bytes()
    assert b1 == b_tight


# ---------------------------------------------------------------------------
# 2. Crash-safety contract holds with N>1
# ---------------------------------------------------------------------------

def test_crash_safety_with_n_workers_no_partial_left(tmp_path, monkeypatch):
    tensors = _build_tensors()
    fail_name = "blk.0.ffn_down.weight"
    src = StubSource(tensors, raise_on=fail_name)
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
    monkeypatch.setenv("MAGICQUANT_ENCODE_THREADS", "4")

    out = str(tmp_path / "out.gguf")
    with pytest.raises(Exception):
        create_hybrid_gguf(
            output_path=out,
            base_model_path="ignored",
            quant_config=_QUANT_CONFIG,
            verbose=False,
            imatrix=_imatrix_for(tensors),
        )

    assert not (tmp_path / "out.gguf").exists()
    assert not (tmp_path / "out.gguf.partial").exists()


def test_crash_safety_with_n_workers_bad_dtype(tmp_path, monkeypatch):
    tensors = _build_tensors()
    bad_name = "blk.1.ffn_down.weight"
    src = StubSource(tensors, bad_dtype_on=bad_name)
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: src)
    monkeypatch.setenv("MAGICQUANT_ENCODE_THREADS", "4")

    out = str(tmp_path / "out.gguf")
    with pytest.raises(ValueError, match="floating-point"):
        create_hybrid_gguf(
            output_path=out,
            base_model_path="ignored",
            quant_config=_QUANT_CONFIG,
            verbose=False,
            imatrix=_imatrix_for(tensors),
        )

    assert not (tmp_path / "out.gguf").exists()
    assert not (tmp_path / "out.gguf.partial").exists()


# ---------------------------------------------------------------------------
# 3. Byte budget caps in-flight bytes, not just tensor count
# ---------------------------------------------------------------------------

def test_writer_caps_in_flight_bytes_with_n_workers(tmp_path, monkeypatch):
    tensors = _build_tensors()
    footprints_total = sum(arr.nbytes for (_n, arr, _s) in tensors)
    one_tensor_bytes = 256 * 256 * 4  # matches the dense 256x256 f32 tensors above
    budget_mb = (one_tensor_bytes * 2) / (1024 * 1024)

    captured = {}
    real_cls = writer_mod._ByteBudget

    class RecordingBudget(real_cls):
        def __init__(self, capacity_bytes):
            super().__init__(capacity_bytes)
            captured["budget"] = self

    monkeypatch.setattr(writer_mod, "_ByteBudget", RecordingBudget)

    _build_gguf(tmp_path, "out.gguf", 4, monkeypatch, budget_mb=budget_mb)

    budget = captured["budget"]
    assert budget.peak_used <= budget._capacity, (
        "peak in-flight bytes must never exceed the configured budget "
        "when the budget already covers the largest single tensor"
    )
    assert budget.peak_used < footprints_total, (
        "expected the byte budget to actually throttle concurrency -- "
        "peak usage should stay well below 'every tensor decoded at once'"
    )


# ---------------------------------------------------------------------------
# Unit tests: _ByteBudget
# ---------------------------------------------------------------------------

def test_byte_budget_rejects_over_capacity_when_something_in_flight():
    budget = writer_mod._ByteBudget(1000)
    budget.acquire(400)
    assert budget.try_acquire(400) is True   # 800 <= 1000
    assert budget.try_acquire(300) is False  # 1100 > 1000
    budget.release(400)
    assert budget.try_acquire(300) is True   # 700 <= 1000
    assert budget.peak_used == 800


def test_byte_budget_never_deadlocks_on_oversized_single_item():
    budget = writer_mod._ByteBudget(100)
    # Nothing in flight yet -> admitted even though it alone exceeds capacity.
    budget.acquire(500)
    assert budget.peak_used == 500
    # A second item cannot join while the oversized one is still in flight.
    assert budget.try_acquire(1) is False
    budget.release(500)
    assert budget.try_acquire(1) is True


def test_byte_budget_acquire_blocks_until_release():
    budget = writer_mod._ByteBudget(100)
    budget.acquire(80)
    results = []

    def waiter():
        budget.acquire(80)  # must block until the first 80 is released
        results.append("acquired")

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.1)
    assert results == [], "acquire() must block while budget is full"
    budget.release(80)
    t.join(timeout=2)
    assert results == ["acquired"]


# ---------------------------------------------------------------------------
# Unit tests: _resolve_encode_threads / _resolve_encode_budget_bytes
# ---------------------------------------------------------------------------

def test_resolve_encode_threads_env_override(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_ENCODE_THREADS", "3")
    assert writer_mod._resolve_encode_threads() == 3


def test_resolve_encode_threads_one_is_literal(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_ENCODE_THREADS", "1")
    assert writer_mod._resolve_encode_threads() == 1


def test_resolve_encode_threads_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_ENCODE_THREADS", "not-a-number")
    monkeypatch.setattr(writer_mod.os, "cpu_count", lambda: 16)
    assert writer_mod._resolve_encode_threads() == 8  # min(8, 16 // 2)


def test_resolve_encode_threads_non_positive_falls_back(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_ENCODE_THREADS", "0")
    monkeypatch.setattr(writer_mod.os, "cpu_count", lambda: 10)
    assert writer_mod._resolve_encode_threads() == 5  # min(8, 10 // 2)


def test_resolve_encode_threads_default_scales_with_cpu_count(monkeypatch):
    monkeypatch.delenv("MAGICQUANT_ENCODE_THREADS", raising=False)
    monkeypatch.setattr(writer_mod.os, "cpu_count", lambda: 4)
    assert writer_mod._resolve_encode_threads() == 2  # min(8, 4 // 2)


def test_resolve_encode_budget_bytes_env_override_mb(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_ENCODE_BUDGET_MB", "10")
    assert writer_mod._resolve_encode_budget_bytes([], 4) == 10 * 1024 * 1024


def test_resolve_encode_budget_bytes_default_scales_with_workers_and_size(monkeypatch):
    monkeypatch.delenv("MAGICQUANT_ENCODE_BUDGET_MB", raising=False)
    entries = [
        {"_n_elems": 1000, "_expected_size": 500},
        {"_n_elems": 2000, "_expected_size": 1000},
    ]
    # footprints: max(4000,500)=4000, max(8000,1000)=8000; avg=6000
    n_workers = 3
    expected = max(int(2 * 3 * 6000), 8000)
    assert writer_mod._resolve_encode_budget_bytes(entries, n_workers) == expected


def test_resolve_encode_budget_bytes_floor_is_largest_single_tensor(monkeypatch):
    monkeypatch.delenv("MAGICQUANT_ENCODE_BUDGET_MB", raising=False)
    entries = [
        {"_n_elems": 1, "_expected_size": 1},
        {"_n_elems": 10_000_000, "_expected_size": 1},
    ]
    cap = writer_mod._resolve_encode_budget_bytes(entries, 1)
    assert cap >= 10_000_000 * 4


@pytest.mark.skipif(not hasattr(os, "pread"), reason="os.pread is POSIX-only")
def test_gguf_source_pread_short_reads_are_gathered(tmp_path, monkeypatch):
    """os.pread returns at most ~2GiB per syscall on Linux; a 27B model's
    token_embd (2.4GiB BF16) came back short in one call, truncating the
    tensor and aborting the build (caught live on the real Qwopus 27B,
    2026-07-05). Simulate the cap with a tiny per-call limit and assert the
    gather loop reassembles the exact bytes."""
    import numpy as np
    import magicquant.gguf.source as source_mod

    # Minimal single-tensor F32 GGUF via the project's own writer round-trip
    # is heavy; instead monkeypatch a GGUFSource around a raw file.
    data = np.arange(4096, dtype=np.float32)
    raw = tmp_path / "raw.bin"
    raw.write_bytes(data.tobytes())

    src = source_mod.GGUFSource.__new__(source_mod.GGUFSource)
    src._path = str(raw)
    src._fh = None
    src._data_offset = 0
    src._reader = type("R", (), {"get_tensor_info": staticmethod(
        lambda name: {"data_type": 0, "shape": [4096], "offset": 0}
    )})()

    real_pread = __import__("os").pread
    calls = []

    def capped_pread(fd, n, pos):
        n = min(n, 1000)  # simulate the syscall cap
        calls.append(n)
        return real_pread(fd, n, pos)

    monkeypatch.setattr(source_mod.os, "pread", capped_pread)
    out = src.read_tensor_f32("token_embd.weight")
    assert len(calls) > 1, "cap never engaged -- test is vacuous"
    np.testing.assert_array_equal(out, data)


def _raw_source_over(path, tensors):
    """GGUFSource over a raw file with a stub reader (no header parsing).

    ``tensors``: {name: (offset, n_f32_elems)}.
    """
    src = source_mod.GGUFSource.__new__(source_mod.GGUFSource)
    src._path = str(path)
    src._fh = None
    src._data_offset = 0
    src._reader = type("R", (), {"get_tensor_info": staticmethod(
        lambda name: {"data_type": 0, "shape": [tensors[name][1]],
                      "offset": tensors[name][0]}
    )})()
    return src


def test_gguf_source_pread_fallback_is_thread_safe(tmp_path, monkeypatch):
    """Where os.pread is missing (Windows), _read_raw_bytes falls back to
    lseek+read under a lock. The whole point of pread was that N pool threads
    share ONE handle -- an unlocked seek+read pair returns the wrong tensor
    with the right byte count. Force the fallback on every platform and hammer
    it from many threads; every read must come back byte-exact."""
    n_tensors, n_elems = 16, 2048
    blocks = [np.full(n_elems, i, dtype=np.float32).tobytes() for i in range(n_tensors)]
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"".join(blocks))
    tensors = {f"t{i}": (i * n_elems * 4, n_elems) for i in range(n_tensors)}
    src = _raw_source_over(raw, tensors)

    monkeypatch.setattr(source_mod, "_HAS_PREAD", False)
    # Poison os.pread so the fallback path is provably the one exercised.
    if hasattr(os, "pread"):
        monkeypatch.setattr(source_mod.os, "pread",
                            lambda *a: pytest.fail("fallback bypassed"))

    errors, start = [], threading.Barrier(8)

    def worker(wid):
        # Every failure mode lands in `errors`: a worker exception must fail
        # the test, not surface as a PytestUnhandledThreadExceptionWarning
        # while the assertion below passes vacuously. (The lazy-open EBADF
        # race was found exactly this way.)
        try:
            start.wait()
            for rep in range(50):
                i = (wid * 7 + rep) % n_tensors
                got = src.read_tensor_raw(f"t{i}")
                if got != blocks[i]:
                    errors.append((wid, i, "misread"))
        except Exception as exc:  # noqa: BLE001 - re-raised via assert below
            errors.append((wid, None, repr(exc)))

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, f"{len(errors)} cross-thread failures: {errors[:5]}"


def test_gguf_source_pread_fallback_gathers_short_reads(tmp_path, monkeypatch):
    """The fallback shares the gather loop: os.read on Windows clamps a single
    call to INT_MAX, so a capped read must be reassembled exactly."""
    data = np.arange(4096, dtype=np.float32)
    raw = tmp_path / "raw.bin"
    raw.write_bytes(data.tobytes())
    src = _raw_source_over(raw, {"token_embd.weight": (0, 4096)})

    monkeypatch.setattr(source_mod, "_HAS_PREAD", False)
    real_read, calls = os.read, []

    def capped_read(fd, n):
        n = min(n, 1000)
        calls.append(n)
        return real_read(fd, n)

    monkeypatch.setattr(source_mod.os, "read", capped_read)
    out = src.read_tensor_f32("token_embd.weight")
    assert len(calls) > 1, "cap never engaged -- test is vacuous"
    np.testing.assert_array_equal(out, data)
