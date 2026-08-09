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


def test_requires_imatrix_cross_check_logs_on_static_live_mismatch(monkeypatch, caplog):
    """Injecting a scheme whose static requires_imatrix field disagrees with
    this handle's live requires_imatrix() answer must produce a warning log
    line naming the scheme -- not raise. Regression for the requires_imatrix
    drift guard: schemes.py's static field and the live
    ggml_quantize_requires_imatrix answer are two independent sources of
    truth with no runtime check tying them together otherwise. Handle
    construction itself already exercises the no-mismatch, no-raise path
    (get_handle() in the autouse fixture above would already fail loudly if
    _cross_check_requires_imatrix raised)."""
    import dataclasses

    import magicquant.quant.schemes as schemes_mod

    handle = get_handle()
    real_scheme = schemes_mod.get_scheme_by_name("Q8_0")
    assert real_scheme.requires_imatrix is False
    assert handle.requires_imatrix("Q8_0") is False  # sanity: no real drift today
    fake_scheme = dataclasses.replace(real_scheme, requires_imatrix=True)
    monkeypatch.setattr(schemes_mod, "get_all_schemes", lambda: [fake_scheme])

    with caplog.at_level(logging.WARNING, logger=ggml_binding.__name__):
        handle._cross_check_requires_imatrix()  # must not raise

    assert any(
        "Q8_0" in rec.message and "requires_imatrix" in rec.message
        for rec in caplog.records
    ), (
        "expected a requires_imatrix drift warning naming Q8_0, got: "
        f"{[r.message for r in caplog.records]}"
    )
