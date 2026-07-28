"""Tests for _LibggmlHandle._cross_check_block_type_sizes.

Same skip pattern as test_ggml_decode.py / test_rocmfpx_schemes.py: skip
the whole module when libggml can't be located, since this exercises a
real ctypes handle rather than pure-python tables.
"""
from __future__ import annotations

import logging

import pytest

from magicquant.quant import ggml_binding
from magicquant.quant.ggml_binding import LibggmlNotFound, get_handle


@pytest.fixture(autouse=True)
def _require_libggml():
    try:
        get_handle()
    except LibggmlNotFound as e:
        pytest.skip(f"libggml not available: {e}")


def test_handle_construction_runs_cross_check_without_raising():
    """Constructing the (already-cached) handle must have run the cross-check
    as part of __init__ without raising -- a bad BLOCK_SIZE/TYPE_SIZE entry
    for a type this build doesn't fully support must log, not crash."""
    handle = get_handle()
    assert hasattr(handle, "_cross_check_block_type_sizes")
    # Calling it again directly must also be side-effect-free (idempotent,
    # log-only).
    handle._cross_check_block_type_sizes()


def test_cross_check_logs_on_injected_block_size_mismatch(monkeypatch, caplog):
    """Injecting a wrong BLOCK_SIZE for a real stock type must produce a
    warning log line naming the type -- not raise."""
    handle = get_handle()
    bad_table = dict(ggml_binding._GGML_BLOCK_SIZE)
    bad_table["Q4_K"] = bad_table["Q4_K"] + 1  # deliberately wrong
    monkeypatch.setattr(ggml_binding, "_GGML_BLOCK_SIZE", bad_table)

    with caplog.at_level(logging.WARNING, logger=ggml_binding.__name__):
        handle._cross_check_block_type_sizes()  # must not raise

    assert any(
        "Q4_K" in rec.message and "block_size" in rec.message
        for rec in caplog.records
    ), f"expected a Q4_K block_size mismatch warning, got: {[r.message for r in caplog.records]}"


def test_cross_check_skips_unsupported_fork_types(monkeypatch):
    """A fork type the loaded lib doesn't support must never be probed
    (calling ggml_type_size/ggml_blck_size on an out-of-range id would be
    undefined behavior on a stock lib's type_traits array)."""
    handle = get_handle()
    if handle.rocmfpx_supported:
        pytest.skip("this handle's libggml supports ROCmFPX -- nothing to skip here")
    # Should complete without touching any fork id.
    handle._cross_check_block_type_sizes()
