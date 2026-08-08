"""Characterization test for ``magicquant.v2.search.run_budget_search``,
built BEFORE the extract-method decomposition proposed in findings
v2-budget-search ITEM 1 (docs/redesign.md's v2 pipeline, magicquant/v2/
search.py). It is run and made green against the UNMODIFIED function first;
every assertion here describes what the function currently DOES, not what
it "should" do. If the decomposition changes one of these, that is a real
behavior change and must be justified, not silenced.

Stub pattern follows tests/test_v2_calibrate.py::test_probe_config_shapes_per_mode
(a fake LlamaCppTools + monkeypatched magicquant.gguf.writer.create_hybrid_gguf),
extended with a fake magicquant.v2.search.compute_distortion_table and a fake
magicquant.imatrix.ensure_imatrix so the whole pipeline runs GPU-free and
fully deterministic. All the intercepted names are imported FUNCTION-LOCALLY
by search.py/calibrate.py (resolved at call time), so patching the module
attribute is enough -- see the docstrings on the fakes below for exactly
which binding each one defeats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

import magicquant.gguf.writer as wmod
import magicquant.imatrix as imatrix_mod
import magicquant.utils.llamacpp as llamacpp_mod
import magicquant.v2.search as v2search
from magicquant.v2.interchange import budget_tier_key
from magicquant.v2.outcome import BudgetInfeasibleError
from magicquant.v2.search import V2Config, run_budget_search

SOURCE = "src.gguf"

# Every unit at its cheapest choice (Q2_K/Q2_K/F32): 2100 + 1344 + 256.
MIN_TOTAL_BYTES = 3700


# ---------------------------------------------------------------------
# Fabricated distortion table: two allocatable groups (E, D) over three
# REAL registry schemes (BF16/Q4_K_M/Q2_K -- none require an imatrix, so
# _select_schemes's imatrix filter never touches them), plus one
# writer-fixed tensor (group N) to exercise the `fixed` branch of
# _build_units / _dominant_group_schemes. Numbers below (bytes, werr) were
# chosen by hand and cross-checked by directly driving
# magicquant.v2.allocate.allocate() with the equivalent Unit/Choice objects.
# ---------------------------------------------------------------------

def _fake_table(schemes: List[str]) -> Dict[str, Any]:
    def choices(bf16_b, q4_b, q4_w, q2_b, q2_w):
        return {
            "BF16": {"actual": "BF16", "bytes": bf16_b, "werr": 0.0, "reason": None},
            "Q4_K_M": {"actual": "Q4_K_M", "bytes": q4_b, "werr": q4_w, "reason": None},
            "Q2_K": {"actual": "Q2_K", "bytes": q2_b, "werr": q2_w, "reason": None},
        }

    return {
        "meta": {"version": 1, "schemes": sorted(schemes)},
        "tensors": {
            "token_embd.weight": {
                "group": "E", "shape": [100, 64], "n_elems": 6400,
                "fixed": False, "wnorm": 1.0,
                "choices": choices(12800, 3600, 1.0, 2100, 5.0),
            },
            "blk.0.ffn_down.weight": {
                "group": "D", "shape": [64, 64], "n_elems": 4096,
                "fixed": False, "wnorm": 1.0,
                "choices": choices(8192, 2304, 0.5, 1344, 2.5),
            },
            "output_norm.weight": {
                "group": "N", "shape": [64], "n_elems": 64,
                "fixed": True, "wnorm": None,
                "choices": {
                    "F32": {"actual": "F32", "bytes": 256, "werr": 0.0,
                             "reason": "1d-f32"},
                },
            },
        },
    }


def _make_cfg(tmp_path: Path, **overrides) -> V2Config:
    kwargs: Dict[str, Any] = dict(
        source_model_path=SOURCE,
        output_dir=str(tmp_path),
        budget_gb=8000 / 1024**3,
        schemes=["BF16", "Q4_K_M", "Q2_K"],
        use_imatrix=True,
        imatrix_corpus=None,
        group_probes=True,
        probe_scheme="Q4_K_M",
        probe_chunks=24,
        allow_partial_probes=False,
        anchors=2,
        anchor_spread=0.07,
        measurement_chunks=100,
        keep_anchors=False,
    )
    kwargs.update(overrides)
    return V2Config(**kwargs)


class _FakeTools:
    """Stand-in for magicquant.utils.llamacpp.LlamaCppTools.

    run_budget_search does ``from magicquant.utils.llamacpp import
    LlamaCppTools`` FUNCTION-LOCALLY, so monkeypatching the module attribute
    (see _install_stubs) is enough to intercept construction -- same
    mechanism relied on for create_hybrid_gguf below.
    """

    def __init__(self, llamacpp_path=None, data_file=None):
        self.llamacpp_path = llamacpp_path
        self.data_file = data_file
        self.ctx_size = 512
        self.ppl_chunks = None
        self.perplexity_tool = None  # skip the llama-imatrix sibling probe
        self._pinned_corpus = None
        self.calls: List[str] = []
        self._ppl_lookup: Callable[[str], Optional[float]] = lambda p: None

    def _resolve_data_file(self, data_file=None):
        return data_file or self.data_file or "corpus.raw"

    def calculate_perplexity(self, path, verbose=False, **kw):
        p = str(path)
        self.calls.append(p)
        return self._ppl_lookup(p)


def _install_stubs(
    monkeypatch,
    ppl_lookup: Callable[[str], Optional[float]],
    *,
    build_fail: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ensure_imatrix_return: Any = "__unset__",
    byte_size: int = 4096,
):
    """Wire every stub the characterization tests share.

    Returns (tools_registry, build_log, ensure_imatrix_calls) for
    inspection -- tools_registry lets tests assert LlamaCppTools was
    constructed exactly once (instance identity is load-bearing per the
    v2-budget-search/1 findings: _resolve_data_file's corpus pin and
    run_group_probes' ppl_chunks save/restore both depend on the SAME
    instance being threaded through every phase).
    """
    tools_registry: List[_FakeTools] = []
    build_log: List[Dict[str, Any]] = []
    ensure_imatrix_calls: List[Dict[str, Any]] = []

    def _tools_factory(llamacpp_path=None, data_file=None):
        t = _FakeTools(llamacpp_path, data_file)
        t._ppl_lookup = ppl_lookup
        tools_registry.append(t)
        return t

    def _fake_create_hybrid_gguf(output_path, base_model_path, quant_config,
                                  verbose=False, imatrix=None, **kw):
        build_log.append({
            "output_path": str(output_path),
            "base_model_path": str(base_model_path),
            "quant_config": quant_config,
            "imatrix": imatrix,
        })
        if build_fail is not None and build_fail(str(output_path), quant_config):
            raise RuntimeError("simulated build failure")
        Path(output_path).write_bytes(b"x" * byte_size)
        return output_path

    def _fake_compute_distortion_table(source_model_path, schemes, imatrix=None,
                                        cache_dir=None, sample_rows=None,
                                        verbose=True):
        return _fake_table(schemes)

    def _fake_ensure_imatrix(model_path, corpus_path=None, **kw):
        ensure_imatrix_calls.append(
            {"model_path": model_path, "corpus_path": corpus_path, "kwargs": kw}
        )
        if ensure_imatrix_return == "__unset__":
            return {"token_embd.weight": [1.0]}
        return ensure_imatrix_return

    monkeypatch.setattr(llamacpp_mod, "LlamaCppTools", _tools_factory)
    monkeypatch.setattr(wmod, "create_hybrid_gguf", _fake_create_hybrid_gguf)
    monkeypatch.setattr(v2search, "compute_distortion_table",
                         _fake_compute_distortion_table)
    monkeypatch.setattr(imatrix_mod, "ensure_imatrix", _fake_ensure_imatrix)

    return tools_registry, build_log, ensure_imatrix_calls


def _happy_ppl(path: str) -> Optional[float]:
    if path == SOURCE:
        return 10.0
    if "probe_D" in path:
        return 10.2
    if "probe_E" in path:
        return 10.5
    if "-v2-budget-" in path:
        return 10.1
    if "-v2-n1-" in path:
        return 10.3
    raise AssertionError(f"unexpected calculate_perplexity call: {path}")


# ===========================================================================
# Happy path: baseline -> imatrix+schemes -> distortion table -> kappa via
# group probes -> allocation+neighbors -> anchor build/verify -> results.
# ===========================================================================

@pytest.fixture
def happy_run(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    tools_registry, build_log, ensure_imatrix_calls = _install_stubs(
        monkeypatch, _happy_ppl
    )
    results = run_budget_search(cfg)
    return {
        "cfg": cfg,
        "results": results,
        "tools": tools_registry[0],
        "tools_registry": tools_registry,
        "build_log": build_log,
        "ensure_imatrix_calls": ensure_imatrix_calls,
        "tmp_path": tmp_path,
    }


def test_happy_path_results_json_shape_and_values(happy_run):
    results = happy_run["results"]
    cfg = happy_run["cfg"]

    assert set(results.keys()) == {
        "version", "algo", "source_model", "budget_gb", "baseline_ppl",
        "baseline_provenance", "measurement", "schemes", "kappa",
        "kappa_provenance", "group_epsilon_sums", "allocation",
        "group_summary", "anchors", "report_fit_affine", "failures",
        "final_model", "seconds",
    }
    assert results["version"] == 2
    assert results["algo"] == "v2-budget"
    assert results["source_model"] == SOURCE
    assert results["budget_gb"] == cfg.budget_gb
    assert results["baseline_ppl"] == pytest.approx(10.0)
    assert results["baseline_provenance"] == "measured"
    assert results["schemes"] == ["BF16", "Q4_K_M", "Q2_K"]

    m = results["measurement"]
    assert set(m.keys()) == {"corpus", "ctx_size", "anchor_chunks",
                              "probe_chunks", "imatrix_active"}
    assert m["corpus"] == "corpus.raw"
    assert m["ctx_size"] == 512
    assert m["anchor_chunks"] == cfg.measurement_chunks
    assert m["probe_chunks"] == cfg.probe_chunks
    assert m["imatrix_active"] is True

    assert results["group_epsilon_sums"] == {"D": 0.5, "E": 1.0}
    assert results["kappa_provenance"] == {"D": "measured", "E": "measured"}
    assert results["kappa"]["D"] == pytest.approx(0.02 / 0.5)
    assert results["kappa"]["E"] == pytest.approx(0.05 / 1.0)

    alloc = results["allocation"]
    assert set(alloc.keys()) == {
        "budget_bytes", "budget_gb", "total_bytes", "total_gb",
        "predicted_loss", "polish_improvements", "assignment", "actual_types",
    }
    assert alloc["assignment"] == {
        "token_embd.weight": "Q4_K_M",
        "blk.0.ffn_down.weight": "Q4_K_M",
        "output_norm.weight": "F32",
    }
    assert alloc["total_bytes"] == 6160
    assert alloc["predicted_loss"] == pytest.approx(0.07)

    assert results["group_summary"] == {"E": "Q4_K_M", "D": "Q4_K_M"}

    anchors = results["anchors"]
    assert len(anchors) == 2
    assert [a["tag"] for a in anchors] == ["budget", "n1"]
    for a in anchors:
        assert set(a.keys()) == {
            "tag", "path", "predicted_bytes", "actual_bytes",
            "predicted_loss", "measurement", "ppl", "measured_rel_loss",
        }
        assert a["measurement"]["status"] == "ok"
        assert a["predicted_bytes"] == 6160
        assert a["actual_bytes"] == 4096
    assert anchors[0]["ppl"] == pytest.approx(10.1)
    assert anchors[1]["ppl"] == pytest.approx(10.3)
    assert anchors[0]["measured_rel_loss"] == pytest.approx((10.1 - 10.0) / 10.0)
    assert anchors[1]["measured_rel_loss"] == pytest.approx((10.3 - 10.0) / 10.0)

    # Note #3 (load-bearing): actual_bytes is read via stat() BEFORE the
    # non-primary anchor's file is unlinked, so it survives even though the
    # file itself is gone and "path" is nulled out.
    assert anchors[0]["path"] is not None
    assert Path(anchors[0]["path"]).exists()
    assert anchors[1]["path"] is None

    # Pinned, not just shape-checked: _compute_report_fit's fit_points are
    # the 2 anchor points PLUS the 2 single-mode probe points (D, E), each
    # computed against probe_baseline (== slice_baseline here). Ground truth
    # independently derived by running the unmodified pipeline against this
    # exact fixture (see scratchpad/derive_fit.py) -- not copied from the
    # review that reported it, since that number must be re-earned, not
    # trusted. This pin is what catches a mutation that (a) drops the probe
    # points from the fit entirely, or (b) mis-detects single- vs
    # cumulative-mode (both silently change fit_points from 4 -> 2 members,
    # here happening to coincide but caught either way by the exact floats).
    assert results["report_fit_affine"] == pytest.approx(
        (-0.04477611940298202, 0.029850746268656542)
    )

    assert results["failures"] == []
    assert results["final_model"] == anchors[0]["path"]
    assert isinstance(results["seconds"], float)
    assert results["seconds"] >= 0.0


def test_happy_path_frontier_json_shape_and_values(happy_run):
    tmp_path = happy_run["tmp_path"]
    results = happy_run["results"]

    frontier = json.loads((tmp_path / "frontier.json").read_text())
    assert set(frontier.keys()) == {"budget_bytes", "kappa", "points", "measured"}
    assert frontier["budget_bytes"] == int(happy_run["cfg"].budget_gb * 1024**3)
    assert frontier["kappa"] == results["kappa"]

    assert len(frontier["points"]) >= 1
    for p in frontier["points"]:
        assert set(p.keys()) == {"bytes", "gb", "loss", "changed_unit", "new_scheme"}

    measured = frontier["measured"]
    assert len(measured) == 2
    assert measured[0] == {
        "gb": 4096 / 1024**3, "ppl": pytest.approx(10.1),
        "rel_loss": pytest.approx((10.1 - 10.0) / 10.0), "tag": "budget",
    }
    assert measured[1] == {
        "gb": 4096 / 1024**3, "ppl": pytest.approx(10.3),
        "rel_loss": pytest.approx((10.3 - 10.0) / 10.0), "tag": "n1",
    }


def test_happy_path_tool_call_order_and_single_instantiation(happy_run):
    tools_registry = happy_run["tools_registry"]
    tools = happy_run["tools"]
    cfg = happy_run["cfg"]

    # Note #1 (load-bearing): a single LlamaCppTools instance must be
    # threaded through every phase, never reconstructed.
    assert len(tools_registry) == 1

    calls = tools.calls
    assert len(calls) == 6
    assert calls[0] == SOURCE                       # phase 1: baseline
    assert calls[1] == SOURCE                        # phase 4: slice baseline
    assert "probe_D" in calls[2]                     # phase 4: group probes,
    assert "probe_E" in calls[3]                     # sorted(groups) = [D, E]
    assert "-v2-budget-" in calls[4]                 # phase 6: primary anchor
    assert "-v2-n1-" in calls[5]                     # phase 6: neighbor anchor

    # Note #2 (load-bearing): measurement_chunks is set before probes run
    # and restored (by run_group_probes' finally) before anchor measurement.
    assert tools.ppl_chunks == cfg.measurement_chunks


def test_happy_path_build_log_quant_configs_and_imatrix_identity(happy_run):
    build_log = happy_run["build_log"]
    results = happy_run["results"]
    assignment = results["allocation"]["assignment"]

    assert len(build_log) == 4
    assert build_log[0]["quant_config"] == {"base": "BF16", "groups": {"D": "Q4_K_M"}}
    assert build_log[1]["quant_config"] == {"base": "BF16", "groups": {"E": "Q4_K_M"}}
    assert build_log[2]["quant_config"] == {
        "base": "BF16", "groups": {}, "tensors": assignment,
    }
    assert build_log[3]["quant_config"] == {
        "base": "BF16", "groups": {}, "tensors": assignment,
    }

    # The imatrix resolved once in phase 2 is threaded, by identity, through
    # every subsequent create_hybrid_gguf call (probes AND anchors).
    imatrices = [entry["imatrix"] for entry in build_log]
    assert all(im is imatrices[0] for im in imatrices)
    assert imatrices[0] == {"token_embd.weight": [1.0]}


def test_returned_dict_matches_written_v2_results_json(happy_run):
    tmp_path = happy_run["tmp_path"]
    results = happy_run["results"]
    on_disk = json.loads((tmp_path / "v2_results.json").read_text())
    # Compare through a JSON round-trip so tuple-vs-list (report_fit_affine)
    # doesn't produce a false mismatch.
    assert on_disk == json.loads(json.dumps(results))


def test_search_results_interchange_written(happy_run):
    tmp_path = happy_run["tmp_path"]
    cfg = happy_run["cfg"]
    data = json.loads((tmp_path / "search_results.json").read_text())
    key = budget_tier_key(cfg.budget_gb)
    assert key in data.get("tiered", {})
    assert data["tiered"][key]["algo"] == "v2-budget"


# ===========================================================================
# The three hard RuntimeError paths.
# ===========================================================================

def test_baseline_none_raises_runtime_error(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, use_imatrix=False)
    _install_stubs(monkeypatch, lambda path: None)

    with pytest.raises(RuntimeError, match="baseline perplexity"):
        run_budget_search(cfg)

    assert not (tmp_path / "v2_results.json").exists()


def test_empty_scheme_set_raises_runtime_error(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, use_imatrix=False, schemes=[])
    _install_stubs(monkeypatch, lambda path: 10.0 if path == SOURCE else None)

    with pytest.raises(RuntimeError, match="empty scheme choice set"):
        run_budget_search(cfg)

    assert not (tmp_path / "v2_results.json").exists()


def test_zero_anchors_verified_raises_runtime_error(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path, use_imatrix=False, group_probes=False, anchors=1,
        budget_gb=8000 / 1024**3,
    )
    ppl = lambda path: 10.0 if path == SOURCE else 10.1  # noqa: E731
    _install_stubs(monkeypatch, ppl, build_fail=lambda path, cfg: True)

    with pytest.raises(RuntimeError, match="verified ZERO anchors"):
        run_budget_search(cfg)

    assert not (tmp_path / "v2_results.json").exists()


# ===========================================================================
# BudgetInfeasibleError -- not caught anywhere in run_budget_search when it
# is the PRIMARY (budget) allocation that fails.
# ===========================================================================

def test_budget_infeasible_primary_raises_and_writes_nothing(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path, use_imatrix=False, group_probes=False,
        budget_gb=1000 / 1024**3,  # below MIN_TOTAL_BYTES (3700)
    )
    _install_stubs(monkeypatch, lambda path: 10.0 if path == SOURCE else None)

    with pytest.raises(BudgetInfeasibleError) as excinfo:
        run_budget_search(cfg)

    assert excinfo.value.min_bytes == MIN_TOTAL_BYTES
    assert excinfo.value.budget_bytes == 1000
    assert not (tmp_path / "v2_results.json").exists()
    assert not (tmp_path / "frontier.json").exists()
    assert not (tmp_path / "search_results.json").exists()


# ===========================================================================
# `failures` accumulation from each of its three distinct call sites.
# ===========================================================================

def test_anchor_allocate_failure_recorded_in_failures(tmp_path, monkeypatch):
    """A neighbor anchor whose shrunk budget is infeasible is caught by the
    per-neighbor try/except (search.py's anchor-allocate loop) and recorded
    -- unlike the PRIMARY allocation, which raises uncaught."""
    cfg = _make_cfg(
        tmp_path, use_imatrix=False, group_probes=False, anchors=2,
        budget_gb=3750 / 1024**3,  # primary feasible (3700), 0.93x (3487) not
    )
    ppl = lambda path: 10.0 if path == SOURCE else 10.1  # noqa: E731
    _install_stubs(monkeypatch, ppl)

    results = run_budget_search(cfg)

    failures = results["failures"]
    assert len(failures) == 1
    f = failures[0]
    assert f["stage"] == "anchor-allocate"
    assert f["status"] == "failed"
    assert f["factor"] == pytest.approx(0.93)
    assert "error" in f

    # The primary anchor still built and verified fine -- no RuntimeError.
    anchors = results["anchors"]
    assert len(anchors) == 1
    assert anchors[0]["tag"] == "budget"
    assert anchors[0]["measurement"]["status"] == "ok"
    assert results["allocation"]["assignment"] == {
        "token_embd.weight": "Q2_K",
        "blk.0.ffn_down.weight": "Q2_K",
        "output_norm.weight": "F32",
    }


def test_probe_failure_recorded_and_kappa_imputed_median(tmp_path, monkeypatch):
    """Group D's probe never parses a PPL; allow_partial_probes lets the run
    continue, records the failure, and imputes D's kappa from the median of
    the successfully-measured groups (here, just E)."""
    cfg = _make_cfg(
        tmp_path, use_imatrix=False, anchors=1,
        allow_partial_probes=True,
    )

    def ppl(path):
        if path == SOURCE:
            return 10.0
        if "probe_D" in path:
            return None  # unparseable, every attempt
        if "probe_E" in path:
            return 10.5
        if "-v2-budget-" in path:
            return 10.1
        raise AssertionError(path)

    tools_registry, build_log, _ = _install_stubs(monkeypatch, ppl)

    results = run_budget_search(cfg)

    probe_failures = [f for f in results["failures"] if f["stage"] == "probe"]
    assert len(probe_failures) == 1
    pf = probe_failures[0]
    assert pf["group"] == "D"
    assert pf["status"] == "failed"
    assert pf["attempts"] == 2  # retries defaults to 1 -> 2 attempts
    assert pf["error"] == "llama-perplexity produced no parseable PPL"
    assert pf["meta"] == {"group": "D", "probe_scheme": "Q4_K_M"}

    assert results["kappa_provenance"]["E"] == "measured"
    assert results["kappa_provenance"]["D"] == "imputed-median"
    assert results["kappa"]["E"] == pytest.approx(0.05 / 1.0)
    assert results["kappa"]["D"] == pytest.approx(results["kappa"]["E"])

    # D's probe was attempted twice (build+measure both times), E once.
    d_probe_builds = [b for b in build_log if b["quant_config"].get("groups", {}).get("D")]
    e_probe_builds = [b for b in build_log if b["quant_config"].get("groups", {}).get("E")]
    assert len(d_probe_builds) == 2
    assert len(e_probe_builds) == 1

    # The run overall still succeeds (allow_partial_probes=True).
    assert results["anchors"][0]["measurement"]["status"] == "ok"


def test_anchor_stage_failure_recorded_without_raising_and_frontier_fallback(
    tmp_path, monkeypatch
):
    """One anchor (n1) fails to build; the other (budget) succeeds, so the
    run completes and records the failure rather than raising -- contrast
    with test_zero_anchors_verified_raises_runtime_error where EVERY anchor
    fails."""
    cfg = _make_cfg(
        tmp_path, use_imatrix=False, group_probes=False, anchors=2,
        budget_gb=8000 / 1024**3,
    )

    def ppl(path):
        if path == SOURCE:
            return 10.0
        if "-v2-budget-" in path:
            return 10.1
        raise AssertionError(f"unexpected calculate_perplexity call: {path}")

    _install_stubs(
        monkeypatch, ppl, build_fail=lambda path, cfg: "-v2-n1-" in path
    )

    results = run_budget_search(cfg)

    anchors = results["anchors"]
    assert len(anchors) == 2
    ok_anchor, failed_anchor = anchors[0], anchors[1]
    assert ok_anchor["tag"] == "budget" and ok_anchor["measurement"]["status"] == "ok"
    assert failed_anchor["tag"] == "n1" and failed_anchor["measurement"]["status"] == "failed"

    # A failed anchor's dict has no ppl/measured_rel_loss keys at all.
    assert set(failed_anchor.keys()) == {
        "tag", "path", "predicted_bytes", "actual_bytes",
        "predicted_loss", "measurement",
    }
    assert failed_anchor["actual_bytes"] is None  # build raised before writing
    # The failed anchor's intended path is preserved (never nulled: the
    # unlink branch only fires when the file actually exists).
    assert failed_anchor["path"] is not None
    assert not Path(failed_anchor["path"]).exists()

    anchor_failures = [f for f in results["failures"] if f["stage"] == "anchor"]
    assert len(anchor_failures) == 1
    assert anchor_failures[0]["tag"] == "n1"
    assert anchor_failures[0]["error"] == "RuntimeError: simulated build failure"

    frontier = json.loads((tmp_path / "frontier.json").read_text())
    n1_measured = next(m for m in frontier["measured"] if m["tag"] == "n1")
    # actual_bytes is None -> falls back to predicted_bytes/1024**3 (line
    # 404's `or` fallback), with ppl/rel_loss both None (.get on a dict that
    # never got the "ppl"/"measured_rel_loss" keys).
    assert n1_measured["gb"] == pytest.approx(failed_anchor["predicted_bytes"] / 1024**3)
    assert n1_measured["ppl"] is None
    assert n1_measured["rel_loss"] is None


# ===========================================================================
# imatrix resolution: the "unavailable" branch (ensure_imatrix -> None).
# ===========================================================================

def test_imatrix_unavailable_recorded_inactive(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path, use_imatrix=True, group_probes=False, anchors=1,
        budget_gb=8000 / 1024**3,
    )
    ppl = lambda path: 10.0 if path == SOURCE else 10.1  # noqa: E731
    _, build_log, ensure_imatrix_calls = _install_stubs(
        monkeypatch, ppl, ensure_imatrix_return=None
    )

    results = run_budget_search(cfg)

    assert len(ensure_imatrix_calls) == 1
    assert results["measurement"]["imatrix_active"] is False
    assert all(entry["imatrix"] is None for entry in build_log)


# ===========================================================================
# Cumulative probe mode (__base_aggressive__): the report-fit "skip probe
# points" branch of _compute_report_fit is only reachable when this
# pseudo-key is present in probe_outcomes. No test above ever sets
# probe_mode="cumulative", so that branch went completely unexercised.
# ===========================================================================

def test_cumulative_mode_report_fit_skips_probe_points(tmp_path, monkeypatch):
    """probe_mode="cumulative" makes run_group_probes measure a
    __base_aggressive__ baseline plus leave-one-group-high recoveries.
    _compute_report_fit must detect "__base_aggressive__" in probe_outcomes
    and skip adding probe points to the fit entirely (per its own comment:
    in cumulative mode kappa*eps == recovery by construction, a degenerate
    x==y point that would corrupt the fit). With a single anchor -- the
    only point _compute_report_fit would then see -- affine_report_fit's
    own <2-points rule means report_fit_affine MUST be None. If the
    cumulative branch is instead mis-detected (e.g. the "in" check
    inverted) and the probe points leak in anyway, fit_points grows to 3
    members and a real (non-None) fit comes out -- an easy, unambiguous
    kill. Ground truth (including the None) independently re-derived by
    running the unmodified pipeline against this exact fixture
    (scratchpad/derive_fit3.py), not assumed from the code reading alone.
    """
    cfg = _make_cfg(
        tmp_path, use_imatrix=False, anchors=1, probe_mode="cumulative",
    )

    def ppl(path):
        if path == SOURCE:
            return 10.0
        if "base_aggressive" in path:
            return 12.0
        if "probe_D" in path:
            return 11.7  # leave-D-high recovery: (12.0-11.7)/10.0 = 0.03
        if "probe_E" in path:
            return 11.2  # leave-E-high recovery: (12.0-11.2)/10.0 = 0.08
        if "-v2-budget-" in path:
            return 10.5
        raise AssertionError(path)

    _, build_log, _ = _install_stubs(monkeypatch, ppl)

    results = run_budget_search(cfg)

    # The cumulative machinery genuinely ran (not silently skipped): a
    # base-aggressive build happened, distinct from the two leave-high probes.
    base_aggressive_builds = [
        b for b in build_log if b["quant_config"] == {"base": "Q4_K_M", "groups": {}}
    ]
    assert len(base_aggressive_builds) == 1

    assert results["kappa_provenance"] == {"D": "measured", "E": "measured"}
    assert results["kappa"]["D"] == pytest.approx(0.06)
    assert results["kappa"]["E"] == pytest.approx(0.08)

    # The one and only fit_points member is the single anchor point --
    # correctly too few for affine_report_fit's own >=2-points rule.
    assert results["report_fit_affine"] is None


# ===========================================================================
# probe_baseline (the slice-matched baseline, measured inside
# run_group_probes) vs baseline_ppl (the full-corpus baseline measured in
# phase 1): _compute_report_fit's probe rel-dPPL formula must use
# probe_baseline. Every fixture above queries the SOURCE path for both, so
# the two values were always numerically identical there and a
# probe_baseline<->baseline_ppl substitution would go undetected. Here they
# differ.
# ===========================================================================

def test_probe_baseline_distinguished_from_full_baseline_in_report_fit(
    tmp_path, monkeypatch
):
    """The SOURCE path is queried twice: once for the phase-1 baseline
    (first call) and once more inside run_group_probes for the slice-matched
    probe baseline (second call). A stateful fake makes the two calls return
    DIFFERENT values, so a mutation that substitutes baseline_ppl for
    probe_baseline in the probe rel-dPPL formula changes the result -- e.g.
    D's probe PPL (9.8) sits BELOW baseline_ppl (10.0) but ABOVE
    probe_baseline (9.5): using baseline_ppl would clamp D's contribution to
    exactly 0.0 (the formula's own max(rel, 0.0)) where the correct
    probe_baseline-relative value is a real positive number. Ground truth
    independently re-derived by running the unmodified pipeline against
    this exact fixture (scratchpad/derive_fit2.py)."""
    source_calls = {"n": 0}

    def ppl(path):
        if path == SOURCE:
            source_calls["n"] += 1
            return 10.0 if source_calls["n"] == 1 else 9.5
        if "probe_D" in path:
            return 9.8
        if "probe_E" in path:
            return 10.5
        if "-v2-budget-" in path:
            return 10.6
        raise AssertionError(path)

    cfg = _make_cfg(tmp_path, use_imatrix=False, anchors=1)
    _install_stubs(monkeypatch, ppl)

    results = run_budget_search(cfg)

    assert source_calls["n"] == 2  # phase-1 baseline + slice-matched baseline
    assert results["baseline_ppl"] == pytest.approx(10.0)

    assert results["report_fit_affine"] == pytest.approx(
        (0.3993670886075936, 0.029180546302465138)
    )


# ===========================================================================
# keep_anchors=True: the no-unlink branch (`elif not cfg.keep_anchors and
# idx > 0 ...`) never fires, so every anchor file -- not just the primary --
# survives on disk with its "path" left non-None.
# ===========================================================================

def test_keep_anchors_true_preserves_every_anchor_file(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, keep_anchors=True)
    _install_stubs(monkeypatch, _happy_ppl)

    results = run_budget_search(cfg)

    anchors = results["anchors"]
    assert len(anchors) == 2
    for a in anchors:
        assert a["path"] is not None
        assert Path(a["path"]).exists()
        assert a["actual_bytes"] == 4096
