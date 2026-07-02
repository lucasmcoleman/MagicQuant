"""ROCmFPX fork-scheme support (registry + binding gate + search gating).

The AMD-native ROCmFPX fork types (ROCMFP3/4/6/8) are opt-in: registered in
the scheme registry but excluded from the default evolutionary search, and
only encodable when the bound libggml is a ROCmFPX build. These tests pin:
  * registry metadata (ids, bpw, category, neighbor links),
  * the default search never samples them (backward-compat / fixture safety),
  * enabling them injects seeds + sampling mass,
  * the binding's byte-size table matches the fork's block structs.

The live-encode round trip against a real ROCmFPX libggml is a separate GPU/
fork-gated check; here we verify the pure tables and gating without a lib.
"""
import random

import numpy as np
import pytest

from magicquant.quant.schemes import (
    get_scheme_by_name, get_all_schemes, ROCMFPX_SCHEME_NAMES,
)
from magicquant.quant.ggml_binding import (
    ROCMFPX_TYPE_IDS, _GGML_BLOCK_SIZE, _GGML_TYPE_SIZE,
)
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor


# ── registry ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,type_id,bpw,ggml_name", [
    ("ROCMFP8", 103, 8.25, "Q8_0_ROCMFPX"),
    ("ROCMFP6", 102, 6.5, "Q6_0_ROCMFPX"),
    ("ROCMFP4", 100, 4.5, "Q4_0_ROCMFP4"),
    ("ROCMFP3", 104, 3.5, "Q3_0_ROCMFPX"),
])
def test_rocmfpx_scheme_metadata(name, type_id, bpw, ggml_name):
    s = get_scheme_by_name(name)
    assert s.ggml_type_id == type_id
    assert s.bits_per_weight == bpw
    assert s.ggml_type_name == ggml_name
    assert s.category == "rocmfpx"


def test_rocmfpx_neighbor_chain_is_closed():
    # Family forms a monotone chain fp3 -> fp4 -> fp6 -> fp8 -> BF16.
    assert get_scheme_by_name("ROCMFP3").upgrade_neighbor == "ROCMFP4"
    assert get_scheme_by_name("ROCMFP4").upgrade_neighbor == "ROCMFP6"
    assert get_scheme_by_name("ROCMFP6").upgrade_neighbor == "ROCMFP8"
    assert get_scheme_by_name("ROCMFP8").downgrade_neighbor == "ROCMFP6"
    assert get_scheme_by_name("ROCMFP3").downgrade_neighbor is None


def test_rocmfpx_scheme_names_constant_matches_registry():
    reg_rocmfpx = {s.name for s in get_all_schemes() if s.category == "rocmfpx"}
    assert reg_rocmfpx == set(ROCMFPX_SCHEME_NAMES)


# ── binding byte-size tables (match fork block structs) ─────────────────────

@pytest.mark.parametrize("name,block,size", [
    # block=32 for all; size = qs bytes + scale bytes per the fork's structs.
    ("Q4_0_ROCMFP4", 32, 18),   # 16 nibbles + 2 UE4M3 scale bytes
    ("Q3_0_ROCMFPX", 32, 14),   # (32*3/8)=12 + 2
    ("Q6_0_ROCMFPX", 32, 26),   # (32*6/8)=24 + 2
    ("Q8_0_ROCMFPX", 32, 33),   # 32 int8 + 1 scale
])
def test_binding_block_tables(name, block, size):
    assert _GGML_BLOCK_SIZE[name] == block
    assert _GGML_TYPE_SIZE[name] == size


def test_rocmfpx_type_ids_registered():
    assert ROCMFPX_TYPE_IDS["Q4_0_ROCMFP4"] == 100
    assert ROCMFPX_TYPE_IDS["Q8_0_ROCMFPX"] == 103


# ── search gating ────────────────────────────────────────────────────────────

def _survivor(enable_rocmfpx):
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X"]
    predictor = PredictiveScorer(
        sensitivity_weights={g: 1.0 for g in groups},
        parameter_counts={g: 100_000_000 for g in groups},
        baseline_size_gb=5.0,
        baseline_tps=100.0,
    )
    return EvolutionarySurvivor(
        predictor=predictor,
        baseline_config={"E": "BF16", "H": "BF16"},
        max_generations=6,
        population_size=60,
        epsilon=0.2,
        enable_rocmfpx=enable_rocmfpx,
    )


def _all_schemes_used(configs):
    used = set()
    for c in configs:
        used.update(c["config"].values())
    return used


def test_default_search_never_samples_rocmfpx():
    random.seed(11)
    np.random.seed(11)
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X"]
    configs = _survivor(enable_rocmfpx=False).run_evolution(groups=groups, verbose=False)
    assert not (_all_schemes_used(configs) & ROCMFPX_SCHEME_NAMES), (
        "rocmfpx schemes leaked into a default (disabled) search"
    )


def test_enabled_search_samples_rocmfpx():
    random.seed(11)
    np.random.seed(11)
    groups = ["E", "H", "Q", "K", "O", "U", "D", "X"]
    configs = _survivor(enable_rocmfpx=True).run_evolution(groups=groups, verbose=False)
    # Seeds alone guarantee rocmfpx configs are evaluated even if sampling/
    # tiering reshapes the rest.
    assert _all_schemes_used(configs) & ROCMFPX_SCHEME_NAMES, (
        "rocmfpx schemes never appeared despite enable_rocmfpx=True"
    )


# ── live encode round trip (fork libggml required; skipped otherwise) ────────
#
# Run in a SUBPROCESS: loading a second (fork) libggml into an interpreter that
# already ctypes-loaded a different libggml build (RTLD_GLOBAL) clashes symbols
# and aborts the process. A clean child avoids that and mirrors how the entry
# points MAGICQUANT_LIBGGML_DIR at the fork before importing anything.

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_FORK_BIN = Path("~/ROCmFPX/build-strix-rocmfp4/bin").expanduser()

_ENCODE_CHILD = textwrap.dedent("""
    import sys, numpy as np
    from magicquant.quant.ggml_binding import get_handle
    h = get_handle()
    if not h.rocmfpx_supported:
        print("NO_FORK"); sys.exit(0)
    checks = {"Q4_0_ROCMFP4": 18, "Q3_0_ROCMFPX": 14,
              "Q6_0_ROCMFPX": 26, "Q8_0_ROCMFPX": 33}
    w = np.random.randn(32 * 20).astype("float32")
    for t, blk in checks.items():
        n = len(h.encode(w, t))
        assert n == 20 * blk, f"{t}: {n} != {20*blk}"
    print("OK")
""")


@pytest.mark.skipif(not _FORK_BIN.is_dir(), reason="ROCmFPX fork build not present")
def test_live_encode_matches_block_size_subprocess():
    env = dict(os.environ, MAGICQUANT_LIBGGML_DIR=str(_FORK_BIN))
    proc = subprocess.run(
        [sys.executable, "-c", _ENCODE_CHILD],
        env=env, capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout.strip()
    if out.endswith("NO_FORK"):
        pytest.skip("bound libggml does not support fork types")
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.endswith("OK"), f"unexpected child output: {out!r}\n{proc.stderr}"
