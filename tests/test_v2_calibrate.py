"""kappa fitting: slice-baseline use, censoring of below-resolution probes,
pseudo-key exclusion."""

from magicquant.v2.calibrate import CENSOR_FRAC, fit_kappa
from magicquant.v2.outcome import MeasurementOutcome as MO


def test_censoring_floors_below_resolution_groups():
    outcomes = {
        "__slice_baseline__": MO.success(10.0),
        "D": MO.success(10.6),   # rel 0.06
        "O": MO.success(10.2),   # rel 0.02
        "E": MO.success(9.99),   # rel -0.001 -> below resolution
    }
    eps = {"D": 30.0, "O": 2.0, "E": 600.0}
    kappa, prov = fit_kappa(outcomes, eps, baseline_ppl=999.0)
    assert "__slice_baseline__" not in kappa
    assert prov == {"D": "measured", "O": "measured", "E": "measured-censored"}
    # floor = CENSOR_FRAC * median(positive rels [0.02, 0.06]) = 0.25 * 0.06
    assert abs(kappa["E"] - (CENSOR_FRAC * 0.06) / 600.0) < 1e-12
    # cleanly measured groups are untouched
    assert abs(kappa["D"] - 0.06 / 30.0) < 1e-12
    assert abs(kappa["O"] - 0.02 / 2.0) < 1e-12


def test_slice_baseline_preferred_over_full_baseline():
    outcomes = {
        "__slice_baseline__": MO.success(20.0),  # capped-slice baseline
        "D": MO.success(21.0),                   # rel vs slice = 0.05
    }
    kappa, prov = fit_kappa(outcomes, {"D": 10.0}, baseline_ppl=10.0)
    assert abs(kappa["D"] - 0.05 / 10.0) < 1e-12


def test_failed_probe_imputes_median_when_allowed():
    outcomes = {
        "__slice_baseline__": MO.success(10.0),
        "D": MO.success(10.6),
        "K": MO.failure("boom"),
    }
    kappa, prov = fit_kappa(outcomes, {"D": 30.0, "K": 5.0}, baseline_ppl=10.0)
    assert prov["K"] == "imputed-median"
    assert kappa["K"] == kappa["D"]


def test_no_allocatable_mass_group():
    outcomes = {"__slice_baseline__": MO.success(10.0), "N": MO.success(10.1)}
    kappa, prov = fit_kappa(outcomes, {"N": 0.0}, baseline_ppl=10.0)
    assert kappa["N"] == 0.0
    assert prov["N"] == "no-allocatable-mass"


# --- cumulative "leave-one-group-high" probe mode (docs/redesign.md §10) ---

def test_cumulative_kappa_uses_recovery_from_base():
    # base-aggressive PPL 22.0; keeping E high recovers a lot (leave_E=20.0),
    # keeping D high recovers less but still above the censor floor
    # (leave_D=21.4). Slice baseline 18.0.
    outcomes = {
        "__slice_baseline__": MO.success(18.0),
        "__base_aggressive__": MO.success(22.0),
        "E": MO.success(20.0),   # recovery (22-20)/18 = 0.1111
        "D": MO.success(21.4),   # recovery (22-21.4)/18 = 0.0333 (> floor)
    }
    eps = {"E": 600.0, "D": 30.0}
    kappa, prov = fit_kappa(outcomes, eps, baseline_ppl=999.0)
    assert "__base_aggressive__" not in kappa and "__slice_baseline__" not in kappa
    assert prov["E"] == "measured" and prov["D"] == "measured"
    assert abs(kappa["E"] - (2.0/18.0) / 600.0) < 1e-9
    assert abs(kappa["D"] - (0.6/18.0) / 30.0) < 1e-9


def test_cumulative_rescues_embedding_vs_single():
    # The exact failure from validation: an embedding whose SINGLE-group probe
    # barely moves PPL (tiny kappa) but whose CUMULATIVE probe recovers a lot
    # (large kappa). Same eps in both; only the measurement context differs.
    eps = {"E": 600.0, "K": 5.0}
    # single: E quantized alone barely hurts (18.02 vs 18.0 baseline);
    #         K quantized alone hurts more (18.30).
    single = {
        "__slice_baseline__": MO.success(18.0),
        "E": MO.success(18.02),
        "K": MO.success(18.30),
    }
    ks, _ = fit_kappa(single, eps, baseline_ppl=999.0)
    # cumulative: from an all-quantized base (22.0), keeping E high recovers
    #             a lot (20.0), keeping K high recovers little (21.9).
    cumulative = {
        "__slice_baseline__": MO.success(18.0),
        "__base_aggressive__": MO.success(22.0),
        "E": MO.success(20.0),
        "K": MO.success(21.9),
    }
    kc, _ = fit_kappa(cumulative, eps, baseline_ppl=999.0)
    # Single mode ranks K >> E (embedding looks cheap to crush — the bug).
    assert ks["K"] > ks["E"] * 5
    # Cumulative mode raises E's kappa by orders of magnitude, so E is no
    # longer the cheapest thing to crush per byte.
    assert kc["E"] > ks["E"] * 20


def test_cumulative_censoring_and_pseudo_key_exclusion():
    outcomes = {
        "__slice_baseline__": MO.success(18.0),
        "__base_aggressive__": MO.success(22.0),
        "E": MO.success(20.0),    # recovery 0.111 -> measured
        "Q": MO.success(21.999),  # recovery ~5.5e-5 -> below floor -> censored
    }
    eps = {"E": 600.0, "Q": 4.0}
    kappa, prov = fit_kappa(outcomes, eps, baseline_ppl=999.0)
    assert set(kappa) == {"E", "Q"}
    assert prov["E"] == "measured"
    assert prov["Q"] == "measured-censored"


def test_probe_config_shapes_per_mode():
    # Directly exercise the per-mode probe quant_config builder via a fake
    # llama_tools that records the configs create_hybrid_gguf is called with,
    # without touching the GPU or writing real GGUFs.
    import magicquant.v2.calibrate as cal

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            return 18.0  # constant: baseline == every probe

    captured = []
    def _fake_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        captured.append(quant_config)
        from pathlib import Path as _P
        _P(output_path).write_bytes(b"x")
        return output_path

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _fake_create
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # single mode
            captured.clear()
            cal.run_group_probes(_Tools(), "m.gguf", d + "/s", ["E", "D"], 18.0,
                                 probe_mode="single", allow_partial=True)
            singles = [c for c in captured]
            assert {"base": "BF16", "groups": {"E": "Q4_K_M"}} in singles
            assert {"base": "BF16", "groups": {"D": "Q4_K_M"}} in singles
            # cumulative mode
            captured.clear()
            cal.run_group_probes(_Tools(), "m.gguf", d + "/c", ["E", "D"], 18.0,
                                 probe_mode="cumulative", allow_partial=True)
            cums = [c for c in captured]
            assert {"base": "Q4_K_M", "groups": {}} in cums          # base-aggressive
            assert {"base": "Q4_K_M", "groups": {"E": "BF16"}} in cums  # leave E high
            assert {"base": "Q4_K_M", "groups": {"D": "BF16"}} in cums
    finally:
        wmod.create_hybrid_gguf = orig


def test_invalid_probe_mode_rejected():
    import pytest
    import magicquant.v2.calibrate as cal
    class _Tools:
        ppl_chunks = None
        ctx_size = 512
    with pytest.raises(ValueError):
        cal.run_group_probes(_Tools(), "m.gguf", "/tmp/x", ["E"], 18.0,
                             probe_mode="bogus")


# --- failure-path coverage for the shared build/measure/retry/cleanup core
# extracted into calibrate._measure_probe (used by both the base-aggressive
# probe and each per-group probe in run_group_probes) ---

def test_group_probe_retries_then_succeeds():
    """A build that raises on the first attempt and succeeds on the second:
    attempts is recorded correctly, and the probe GGUF from the failed
    attempt (and the successful one, once measured) is cleaned up — the
    ``finally: unlink`` runs on every attempt."""
    import tempfile
    import magicquant.v2.calibrate as cal
    from pathlib import Path as _P

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            return 18.0

    calls = {"n": 0}

    def _flaky_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated build failure")
        _P(output_path).write_bytes(b"x")
        return output_path

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _flaky_create
    try:
        with tempfile.TemporaryDirectory() as d:
            outcomes = cal.run_group_probes(
                _Tools(), "m.gguf", d, ["E"], 18.0,
                probe_mode="single", retries=1,
            )
            assert outcomes["E"].ok
            assert outcomes["E"].attempts == 2
            # unlinked on both the failed and the successful attempt
            assert not list(_P(d).glob("_v2_probes/*.gguf"))
    finally:
        wmod.create_hybrid_gguf = orig


def test_group_probe_exhausts_retries_records_failure_with_allow_partial():
    """A PPL that never parses: attempts == retries + 1, the failure is
    recorded (not raised) under --allow-partial-probes, and the probe file
    is cleaned up."""
    import tempfile
    import magicquant.v2.calibrate as cal
    from pathlib import Path as _P

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            # slice-baseline (measured against source_model_path) succeeds;
            # the per-group probe GGUF never parses.
            if str(path) == "m.gguf":
                return 18.0
            return None  # unparsable PPL, every attempt

    def _fake_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        _P(output_path).write_bytes(b"x")
        return output_path

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _fake_create
    try:
        with tempfile.TemporaryDirectory() as d:
            outcomes = cal.run_group_probes(
                _Tools(), "m.gguf", d, ["E"], 18.0,
                probe_mode="single", retries=1, allow_partial=True,
            )
            assert outcomes["E"].status == "failed"
            assert outcomes["E"].attempts == 2
            assert outcomes["E"].error == "llama-perplexity produced no parseable PPL"
            assert not list(_P(d).glob("_v2_probes/*.gguf"))
    finally:
        wmod.create_hybrid_gguf = orig


def test_group_probe_failure_without_allow_partial_raises():
    """Per-group failure defers to the aggregate check at the end of
    run_group_probes, which raises ProbeMeasurementError when
    allow_partial is False."""
    import tempfile
    import pytest
    import magicquant.v2.calibrate as cal
    from magicquant.v2.outcome import ProbeMeasurementError
    from pathlib import Path as _P

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            if str(path) == "m.gguf":
                return 18.0
            return None

    def _fake_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        _P(output_path).write_bytes(b"x")
        return output_path

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _fake_create
    try:
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ProbeMeasurementError):
                cal.run_group_probes(
                    _Tools(), "m.gguf", d, ["E"], 18.0,
                    probe_mode="single", retries=0, allow_partial=False,
                )
    finally:
        wmod.create_hybrid_gguf = orig


def test_base_aggressive_failure_raises_and_persists_cache_before_raise():
    """Base-aggressive probe failure (cumulative mode) raises
    ProbeMeasurementError immediately when allow_partial is False — every
    leave-one-group kappa is measured against that base — and the failure
    is written to v2_probes.json BEFORE the raise, so a resumed run sees
    it."""
    import tempfile
    import json
    import pytest
    import magicquant.v2.calibrate as cal
    from magicquant.v2.outcome import ProbeMeasurementError
    from pathlib import Path as _P

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            # slice-baseline (measured against source_model_path) succeeds;
            # the base-aggressive probe GGUF never parses.
            if "base_aggressive" in str(path):
                return None
            return 18.0

    def _fake_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        _P(output_path).write_bytes(b"x")
        return output_path

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _fake_create
    try:
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ProbeMeasurementError, match="base-aggressive"):
                cal.run_group_probes(
                    _Tools(), "m.gguf", d, ["E"], 18.0,
                    probe_mode="cumulative", retries=0, allow_partial=False,
                )
            cache = json.loads((_P(d) / "v2_probes.json").read_text())
            assert cache["probes"]["__base_aggressive__"]["status"] == "failed"
            assert not list(_P(d).glob("_v2_probes/*.gguf"))
    finally:
        wmod.create_hybrid_gguf = orig


# --- imatrix identity in run_group_probes' cache key (v2-budget-search
# finding item 2): the probe cache must invalidate when imatrix changes,
# like the sibling compute_distortion_table cache already does, but must
# still reuse __slice_baseline__ across an imatrix-only change since that
# entry is measured on the unquantized source model. ---

def test_probe_cache_hit_when_imatrix_unchanged():
    """Two run_group_probes calls into the SAME output dir with an
    IDENTICAL imatrix: the second call is a full cache hit -- no group
    probe is rebuilt."""
    import tempfile
    import numpy as np
    import magicquant.v2.calibrate as cal
    from pathlib import Path as _P

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            return 18.0

    build_calls = []
    def _fake_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        build_calls.append(output_path)
        _P(output_path).write_bytes(b"x")
        return output_path

    imatrix = {"blk.0.weight": np.ones(4, dtype=np.float32)}

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _fake_create
    try:
        with tempfile.TemporaryDirectory() as d:
            cal.run_group_probes(_Tools(), "m.gguf", d, ["E", "D"], 18.0,
                                 probe_mode="single", allow_partial=True,
                                 imatrix=imatrix)
            n_after_first = len(build_calls)
            assert n_after_first == 2  # E and D each built once

            cal.run_group_probes(_Tools(), "m.gguf", d, ["E", "D"], 18.0,
                                 probe_mode="single", allow_partial=True,
                                 imatrix=imatrix)
            assert len(build_calls) == n_after_first  # full cache hit
    finally:
        wmod.create_hybrid_gguf = orig


def test_probe_cache_stale_when_imatrix_differs_but_slice_baseline_reused():
    """A second run_group_probes call with a DIFFERENT imatrix must
    re-measure every group probe (the cache is stale for imatrix-sensitive
    measurements), but should still reuse the cached __slice_baseline__
    entry -- it is measured directly against the unquantized source model
    and does not depend on imatrix."""
    import tempfile
    import numpy as np
    import magicquant.v2.calibrate as cal
    from pathlib import Path as _P

    ppl_calls = []

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            ppl_calls.append(path)
            return 18.0

    build_calls = []
    def _fake_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        build_calls.append(output_path)
        _P(output_path).write_bytes(b"x")
        return output_path

    imatrix_a = {"blk.0.weight": np.ones(4, dtype=np.float32)}
    imatrix_b = {"blk.0.weight": np.full(4, 2.0, dtype=np.float32)}

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _fake_create
    try:
        with tempfile.TemporaryDirectory() as d:
            cal.run_group_probes(_Tools(), "m.gguf", d, ["E", "D"], 18.0,
                                 probe_mode="single", allow_partial=True,
                                 imatrix=imatrix_a)
            n_builds_first = len(build_calls)
            assert n_builds_first == 2
            assert ppl_calls.count("m.gguf") == 1  # slice baseline measured once

            build_calls.clear()
            cal.run_group_probes(_Tools(), "m.gguf", d, ["E", "D"], 18.0,
                                 probe_mode="single", allow_partial=True,
                                 imatrix=imatrix_b)
            # stale cache under the new imatrix: both groups rebuilt
            assert len(build_calls) == n_builds_first
            # slice baseline is imatrix-independent: not remeasured
            assert ppl_calls.count("m.gguf") == 1
    finally:
        wmod.create_hybrid_gguf = orig


def test_probe_cache_legacy_without_imatrix_key_re_measures_probes():
    """A v2_probes.json written BEFORE the imatrix cache key existed must be
    treated as stale for every imatrix-sensitive probe (its PPLs were measured
    under an UNKNOWN imatrix) while still reusing __slice_baseline__. Locks the
    documented one-time re-measure migration so a future _drop_imatrix
    normalization can't silently flip legacy caches back to a full hit."""
    import json
    import tempfile
    import numpy as np
    import magicquant.v2.calibrate as cal
    from pathlib import Path as _P

    ppl_paths = []

    class _Tools:
        ppl_chunks = None
        ctx_size = 512
        def calculate_perplexity(self, path, verbose=True, **kw):
            ppl_paths.append(str(path))
            return 18.0

    build_calls = []
    def _fake_create(output_path, base_model_path, quant_config, verbose=False, **kw):
        build_calls.append(output_path)
        _P(output_path).write_bytes(b"x")
        return output_path

    imatrix = {"blk.0.weight": np.ones(4, dtype=np.float32)}

    import magicquant.gguf.writer as wmod
    orig = wmod.create_hybrid_gguf
    wmod.create_hybrid_gguf = _fake_create
    try:
        with tempfile.TemporaryDirectory() as d:
            cal.run_group_probes(_Tools(), "m.gguf", d, ["E", "D"], 18.0,
                                 probe_mode="single", allow_partial=True,
                                 imatrix=imatrix)
            n_first_builds = len(build_calls)
            assert n_first_builds == 2

            # Rewrite the cache as a LEGACY one: drop the imatrix key.
            cache_path = _P(d) / "v2_probes.json"
            data = json.loads(cache_path.read_text())
            assert "imatrix" in data["conditions"]
            del data["conditions"]["imatrix"]
            cache_path.write_text(json.dumps(data))

            ppl_paths.clear()
            cal.run_group_probes(_Tools(), "m.gguf", d, ["E", "D"], 18.0,
                                 probe_mode="single", allow_partial=True,
                                 imatrix=imatrix)
            # Group probes re-built (legacy cache stale for probes)...
            assert len(build_calls) == n_first_builds + 2
            # ...but the slice baseline was reused: no PPL call against the
            # unquantized source model itself.
            assert "m.gguf" not in ppl_paths
    finally:
        wmod.create_hybrid_gguf = orig
