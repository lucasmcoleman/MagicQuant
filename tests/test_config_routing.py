"""Config-routing tests (M5).

cmd_search / cmd_generate / cmd_hybrid must build MagicQuantSettings (honoring
env / .env) and let explicit CLI values override. Verifies a single source of
truth for defaults (30/80).
"""
import argparse
import json
import sys

import pytest

import magicquant.__main__ as cli
from magicquant.__main__ import _settings_from_args


def _args(**kw):
    base = dict(
        model="model.gguf", output_dir=None, llamacpp_path=None, adapter=None,
        target_quant=None, generations=None, population=None, rounds=None,
        candidates=None, patience=None,
        use_imatrix=None, imatrix_corpus=None, enable_kl=None, kl_weight=None,
        enable_speed_bench=None, enable_rocmfpx=None, enable_iq=None,
        stream_aware=None, head_aggressive=None,
        seed=None, measurement_chunks=None,
        speed_weight=None, use_bytes_tps=None,
        write_calibration=None, calibration_source=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_env_honored(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_SEARCH_GENERATIONS", "7")
    settings = _settings_from_args(_args())
    assert settings.search_generations == 7


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_SEARCH_GENERATIONS", "7")
    settings = _settings_from_args(_args(generations=9))
    assert settings.search_generations == 9


def test_unified_default_is_30_80(monkeypatch):
    # No env, no CLI -> config.py defaults (not the old argparse 50/100).
    monkeypatch.delenv("MAGICQUANT_SEARCH_GENERATIONS", raising=False)
    monkeypatch.delenv("MAGICQUANT_POPULATION_SIZE", raising=False)
    settings = _settings_from_args(_args())
    assert settings.search_generations == 30
    assert settings.population_size == 80


def test_model_path_threaded(monkeypatch):
    settings = _settings_from_args(_args(model="/tmp/foo.gguf"))
    assert settings.source_model_path == "/tmp/foo.gguf"


# ── orchestrator knobs (use_imatrix, enable_kl, enable_rocmfpx, ...) ────────


def test_orchestrator_knobs_default_off(monkeypatch):
    for var in (
        "MAGICQUANT_USE_IMATRIX", "MAGICQUANT_IMATRIX_CORPUS",
        "MAGICQUANT_ENABLE_KL", "MAGICQUANT_KL_WEIGHT",
        "MAGICQUANT_ENABLE_SPEED_BENCH", "MAGICQUANT_ENABLE_ROCMFPX",
        "MAGICQUANT_ENABLE_IQ", "MAGICQUANT_STREAM_AWARE",
        "MAGICQUANT_HEAD_AGGRESSIVE", "MAGICQUANT_SEED",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = _settings_from_args(_args())
    assert settings.use_imatrix is False
    assert settings.imatrix_corpus is None
    assert settings.enable_kl is False
    assert settings.kl_weight == 0.1
    assert settings.enable_speed_bench is False
    assert settings.enable_rocmfpx is False
    assert settings.enable_iq is False
    assert settings.stream_aware is False
    assert settings.head_aggressive is False
    assert settings.seed is None


def test_orchestrator_knobs_env_honored(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_USE_IMATRIX", "true")
    monkeypatch.setenv("MAGICQUANT_IMATRIX_CORPUS", "/tmp/corpus.txt")
    monkeypatch.setenv("MAGICQUANT_ENABLE_KL", "true")
    monkeypatch.setenv("MAGICQUANT_KL_WEIGHT", "0.5")
    monkeypatch.setenv("MAGICQUANT_ENABLE_SPEED_BENCH", "true")
    monkeypatch.setenv("MAGICQUANT_ENABLE_ROCMFPX", "true")
    monkeypatch.setenv("MAGICQUANT_ENABLE_IQ", "true")
    monkeypatch.setenv("MAGICQUANT_STREAM_AWARE", "true")
    monkeypatch.setenv("MAGICQUANT_HEAD_AGGRESSIVE", "true")
    monkeypatch.setenv("MAGICQUANT_SEED", "42")

    settings = _settings_from_args(_args())

    assert settings.use_imatrix is True
    assert settings.imatrix_corpus == "/tmp/corpus.txt"
    assert settings.enable_kl is True
    assert settings.kl_weight == 0.5
    assert settings.enable_speed_bench is True
    assert settings.enable_rocmfpx is True
    assert settings.enable_iq is True
    assert settings.stream_aware is True
    assert settings.head_aggressive is True
    assert settings.seed == 42


def test_orchestrator_knobs_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_USE_IMATRIX", "true")
    monkeypatch.setenv("MAGICQUANT_KL_WEIGHT", "0.5")
    monkeypatch.setenv("MAGICQUANT_STREAM_AWARE", "true")
    monkeypatch.setenv("MAGICQUANT_HEAD_AGGRESSIVE", "true")
    monkeypatch.setenv("MAGICQUANT_SEED", "42")

    settings = _settings_from_args(
        _args(use_imatrix=False, kl_weight=0.9,
              stream_aware=False, head_aggressive=False, seed=7)
    )

    assert settings.use_imatrix is False
    assert settings.kl_weight == 0.9
    assert settings.stream_aware is False
    assert settings.head_aggressive is False
    assert settings.seed == 7


# ── measurement_chunks (measurement-corpus-size setting) ────────────────────


def test_measurement_chunks_default_none(monkeypatch):
    monkeypatch.delenv("MAGICQUANT_MEASUREMENT_CHUNKS", raising=False)
    settings = _settings_from_args(_args())
    assert settings.measurement_chunks is None


def test_measurement_chunks_env_honored(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_MEASUREMENT_CHUNKS", "12")
    settings = _settings_from_args(_args())
    assert settings.measurement_chunks == 12


def test_measurement_chunks_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_MEASUREMENT_CHUNKS", "12")
    settings = _settings_from_args(_args(measurement_chunks=5))
    assert settings.measurement_chunks == 5


# ── speed_weight / use_bytes_tps / write_calibration / calibration_source ───
# (LANE B: tunable tps-aware objective + cross-run noise calibration)


def test_tps_objective_knobs_default_off(monkeypatch):
    for var in (
        "MAGICQUANT_SPEED_WEIGHT", "MAGICQUANT_USE_BYTES_TPS",
        "MAGICQUANT_WRITE_CALIBRATION", "MAGICQUANT_CALIBRATION_SOURCE",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = _settings_from_args(_args())
    assert settings.speed_weight is None
    assert settings.use_bytes_tps is False
    assert settings.write_calibration is False
    assert settings.calibration_source == ""


def test_tps_objective_knobs_env_honored(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_SPEED_WEIGHT", "0.4")
    monkeypatch.setenv("MAGICQUANT_USE_BYTES_TPS", "true")
    monkeypatch.setenv("MAGICQUANT_WRITE_CALIBRATION", "true")
    monkeypatch.setenv("MAGICQUANT_CALIBRATION_SOURCE", "/tmp/calib.json")

    settings = _settings_from_args(_args())

    assert settings.speed_weight == 0.4
    assert settings.use_bytes_tps is True
    assert settings.write_calibration is True
    assert settings.calibration_source == "/tmp/calib.json"


def test_tps_objective_knobs_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_SPEED_WEIGHT", "0.4")
    monkeypatch.setenv("MAGICQUANT_USE_BYTES_TPS", "true")
    monkeypatch.setenv("MAGICQUANT_CALIBRATION_SOURCE", "/tmp/calib.json")

    settings = _settings_from_args(
        _args(speed_weight=0.7, use_bytes_tps=False,
              calibration_source="/tmp/other.json")
    )

    assert settings.speed_weight == 0.7
    assert settings.use_bytes_tps is False
    assert settings.calibration_source == "/tmp/other.json"


# ── cmd_search forwarding to the orchestrator ───────────────────────────────


class _FakeSearchOrchestrator:
    """Captures the kwargs cmd_search forwards to run_measured_search /
    run_full_search without running any real search."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.measured_calls = []
        self.full_calls = []
        type(self).last = self

    def run_measured_search(self, **kwargs):
        self.measured_calls.append(kwargs)
        return [], {}

    def run_full_search(self, **kwargs):
        self.full_calls.append(kwargs)
        return [], {}


def _search_args(**kw):
    base = dict(
        model="/tmp/base.gguf", output_dir=None, llamacpp_path=None,
        adapter=None, target_quant=None, generations=None, population=None,
        rounds=None, candidates=None, patience=None,
        use_imatrix=None, imatrix_corpus=None, enable_kl=None,
        kl_weight=None, enable_speed_bench=None, enable_rocmfpx=None,
        enable_iq=None, stream_aware=None, head_aggressive=None,
        seed=None, measurement_chunks=None,
        speed_weight=None, use_bytes_tps=None,
        write_calibration=None, calibration_source=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cmd_search_forwards_knobs_to_run_measured_search(monkeypatch):
    monkeypatch.setattr(
        "magicquant.orchestrator.MagicQuantOrchestrator", _FakeSearchOrchestrator
    )

    cli.cmd_search(_search_args(
        rounds=1,
        use_imatrix=True, imatrix_corpus="/tmp/corpus.txt",
        enable_kl=True, kl_weight=0.7, enable_speed_bench=True,
        enable_rocmfpx=True, enable_iq=True,
        stream_aware=True, head_aggressive=True, seed=42,
        measurement_chunks=8,
        speed_weight=0.4, use_bytes_tps=True,
        write_calibration=True, calibration_source="/tmp/calib.json",
    ))

    orch = _FakeSearchOrchestrator.last
    assert orch is not None
    assert len(orch.measured_calls) == 1
    call = orch.measured_calls[0]
    assert call["use_imatrix"] is True
    assert call["imatrix_corpus"] == "/tmp/corpus.txt"
    assert call["enable_kl"] is True
    assert call["kl_weight"] == 0.7
    assert call["enable_speed_bench"] is True
    assert call["enable_rocmfpx"] is True
    assert call["enable_iq"] is True
    assert call["stream_aware"] is True
    assert call["head_aggressive"] is True
    assert call["seed"] == 42
    assert call["measurement_chunks"] == 8
    assert call["speed_weight"] == 0.4
    assert call["use_bytes_tps"] is True
    assert call["write_calibration"] is True
    assert call["calibration_source"] == "/tmp/calib.json"


def test_cmd_search_forwards_knobs_to_run_full_search(monkeypatch):
    monkeypatch.setattr(
        "magicquant.orchestrator.MagicQuantOrchestrator", _FakeSearchOrchestrator
    )

    cli.cmd_search(_search_args(
        rounds=0,
        use_imatrix=True, imatrix_corpus="/tmp/corpus.txt",
        enable_kl=True, kl_weight=0.7, enable_speed_bench=True,
        enable_rocmfpx=True, enable_iq=True,
        stream_aware=True, head_aggressive=True, seed=42,
        measurement_chunks=8,
        speed_weight=0.4, use_bytes_tps=True,
        write_calibration=True, calibration_source="/tmp/calib.json",
    ))

    orch = _FakeSearchOrchestrator.last
    assert orch is not None
    assert len(orch.full_calls) == 1
    call = orch.full_calls[0]
    assert call["use_imatrix"] is True
    assert call["imatrix_corpus"] == "/tmp/corpus.txt"
    assert call["enable_rocmfpx"] is True
    assert call["enable_iq"] is True
    assert call["stream_aware"] is True
    assert call["head_aggressive"] is True
    assert call["seed"] == 42
    assert call["measurement_chunks"] == 8
    assert call["speed_weight"] == 0.4
    assert call["use_bytes_tps"] is True
    assert call["calibration_source"] == "/tmp/calib.json"
    # run_full_search has no KL / speed-bench / write_calibration params --
    # must not be forwarded.
    assert "enable_kl" not in call
    assert "kl_weight" not in call
    assert "write_calibration" not in call
    assert "enable_speed_bench" not in call


# ── search subparser real-argparse routing ──────────────────────────────────
# Regression test for a bug where search_parser's own --output-dir/--target-quant
# argparse defaults ("./output" / "MXFP4_MOE", non-None) always won inside
# _maybe(), so MAGICQUANT_OUTPUT_DIR / MAGICQUANT_TARGET_BASE_QUANT could never
# reach MagicQuantSettings for `search`. _args()/_search_args() above build a
# Namespace by hand with output_dir=None/target_quant=None and so cannot catch
# this class of bug -- this test builds the real parser via cli.main() instead.


def test_search_real_parser_env_and_flag_output_dir_target_quant(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "magicquant.orchestrator.MagicQuantOrchestrator", _FakeSearchOrchestrator
    )

    env_out = tmp_path / "env_out"
    monkeypatch.setenv("MAGICQUANT_OUTPUT_DIR", str(env_out))
    monkeypatch.setenv("MAGICQUANT_TARGET_BASE_QUANT", "Q6_K")

    # No --output-dir / --target-quant on the CLI -> env vars must resolve.
    monkeypatch.setattr(
        sys, "argv", ["magicquant", "search", "/tmp/base.gguf", "--rounds", "0"]
    )
    cli.main()

    orch = _FakeSearchOrchestrator.last
    assert orch is not None
    assert orch.kwargs["output_dir"] == str(env_out)
    assert orch.full_calls[0]["target_base_quant"] == "Q6_K"

    # Explicit CLI flags still win over the env vars.
    cli_out = tmp_path / "cli_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "magicquant", "search", "/tmp/base.gguf", "--rounds", "0",
            "--output-dir", str(cli_out), "--target-quant", "Q4_K_M",
        ],
    )
    cli.main()

    orch2 = _FakeSearchOrchestrator.last
    assert orch2.kwargs["output_dir"] == str(cli_out)
    assert orch2.full_calls[0]["target_base_quant"] == "Q4_K_M"


# ── cmd_search --algo v2 flag hygiene (E5) ───────────────────────────────────
# _run_v2_search's V2Config construction reads none of v1's evolutionary-
# search flags (--rounds, --enable-kl, ... -- see cli._V2_IGNORED_V1_FLAGS),
# so cmd_search must warn about any of them the user explicitly passed
# alongside --algo v2, and must hard-exit on --adapter (which isn't a silent
# no-op under v2 -- it would silently build from the wrong, un-merged model).


def test_cmd_search_v2_real_parser_warns_ignored_v1_flags_and_still_runs(
    monkeypatch, capsys,
):
    calls = []
    monkeypatch.setattr(
        "magicquant.v2.run_budget_search", lambda cfg: calls.append(cfg) or {}
    )

    monkeypatch.setattr(
        sys, "argv",
        [
            "magicquant", "search", "/tmp/base.gguf", "--algo", "v2",
            "--budget-gb", "5", "--rounds", "3", "--enable-kl",
            "--bytes-tps", "--target-quant", "Q6_K",
        ],
    )
    cli.main()

    assert len(calls) == 1, "the v2 search must still run"
    out = capsys.readouterr().out
    assert "WARNING" in out
    for flag in ("--rounds", "--enable-kl", "--bytes-tps", "--target-quant"):
        assert flag in out
    # Flags v2 DOES honor must not be named as ignored (the message itself
    # legitimately mentions --algo, so check for the ignored-list wording
    # rather than a bare substring match on that one).
    ignored_clause = out.split("ignores the following v1-only flag(s)")[-1]
    for flag in ("--output-dir", "--budget-gb"):
        assert flag not in ignored_clause


def test_cmd_search_v2_enable_iq_not_warned_and_reaches_v2config(monkeypatch, capsys):
    """F9: --enable-iq stopped being a v2 no-op, so it must (a) not appear
    in the ignored-v1-flags warning and (b) actually reach V2Config, making
    BudgetInfeasibleError's "--enable-iq" advice true."""
    calls = []
    monkeypatch.setattr(
        "magicquant.v2.run_budget_search", lambda cfg: calls.append(cfg) or {}
    )

    monkeypatch.setattr(
        sys, "argv",
        [
            "magicquant", "search", "/tmp/base.gguf", "--algo", "v2",
            "--budget-gb", "5", "--enable-iq",
        ],
    )
    cli.main()

    assert len(calls) == 1
    assert calls[0].enable_iq is True
    out = capsys.readouterr().out
    assert "--enable-iq" not in out


def test_cmd_search_v2_default_enable_iq_is_false(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "magicquant.v2.run_budget_search", lambda cfg: calls.append(cfg) or {}
    )

    cli.cmd_search(_search_args(algo="v2", budget_gb=5.0))

    assert len(calls) == 1
    assert calls[0].enable_iq is False


def test_cmd_search_v2_no_warning_without_v1_only_flags(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "magicquant.v2.run_budget_search", lambda cfg: calls.append(cfg) or {}
    )

    cli.cmd_search(_search_args(algo="v2", budget_gb=5.0))

    assert len(calls) == 1
    assert "WARNING" not in capsys.readouterr().out


def test_cmd_search_v2_with_adapter_is_hard_exit(monkeypatch):
    monkeypatch.setattr(
        "magicquant.v2.run_budget_search",
        lambda cfg: pytest.fail("v2 search must not run when --adapter is set"),
    )

    with pytest.raises(SystemExit, match="not supported with --algo v2"):
        cli.cmd_search(_search_args(algo="v2", budget_gb=5.0, adapter="/tmp/lora"))


def test_cmd_search_v2_env_adapter_also_hard_exits(monkeypatch):
    """MAGICQUANT_ADAPTER_PATH must not bypass the v2 adapter gate -- the
    check is at the settings level precisely because a set adapter is always
    wrong under v2, unlike the warn-only v1 flags (reviewer finding)."""
    monkeypatch.setenv("MAGICQUANT_ADAPTER_PATH", "/tmp/lora")
    monkeypatch.setattr(
        "magicquant.v2.run_budget_search",
        lambda cfg: pytest.fail("v2 search must not run with an env adapter"),
    )
    with pytest.raises(SystemExit, match="MAGICQUANT_ADAPTER_PATH"):
        cli.cmd_search(_search_args(algo="v2", budget_gb=5.0))


# ── cmd_generate routing (M5 finish) ────────────────────────────────────────


class _FakeOrchestrator:
    """Captures constructor kwargs and the generate call so the env-routing
    can be asserted without building any real GGUF."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.baseline_ppl = None
        self.tiered_calls = []
        self.top_calls = []
        type(self).last = self

    def generate_tiered_models(self, **kwargs):
        self.tiered_calls.append(kwargs)
        return []

    def generate_top_models(self, **kwargs):
        self.top_calls.append(kwargs)
        return []


def _write_tiered_results(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_results.json").write_text(
        json.dumps({"tiered": {"Q4": {"config": {"E": "BF16"}, "ppl": 5.0}}})
    )


def _gen_args(**kw):
    base = dict(
        model="/tmp/base.gguf", output_dir=None, llamacpp_path=None,
        target_quant=None, tiers=None, verify=None, adapter=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cmd_generate_honors_env(monkeypatch, tmp_path):
    """MAGICQUANT_* env is honored by cmd_generate: output_dir, target quant,
    and tiers flow from the environment into the orchestrator call."""
    env_out = tmp_path / "env_out"
    _write_tiered_results(env_out)

    monkeypatch.setenv("MAGICQUANT_OUTPUT_DIR", str(env_out))
    monkeypatch.setenv("MAGICQUANT_TARGET_BASE_QUANT", "Q6_K")
    monkeypatch.setenv("MAGICQUANT_TIERS", '["Q5","Q6"]')
    monkeypatch.setattr(
        "magicquant.orchestrator.MagicQuantOrchestrator", _FakeOrchestrator
    )

    cli.cmd_generate(_gen_args())

    orch = _FakeOrchestrator.last
    assert orch is not None
    # Output dir came from env, not the ./output argparse default.
    assert orch.kwargs["output_dir"] == str(env_out)
    # The tiered-generation path was used with the env-provided tier list.
    assert orch.tiered_calls, "expected generate_tiered_models to be called"
    assert orch.tiered_calls[0]["tiers"] == ["Q5", "Q6"]


def test_cmd_generate_cli_overrides_env(monkeypatch, tmp_path):
    """Explicit CLI flags override the MAGICQUANT_* env values."""
    env_out = tmp_path / "env_out"
    cli_out = tmp_path / "cli_out"
    _write_tiered_results(cli_out)

    monkeypatch.setenv("MAGICQUANT_OUTPUT_DIR", str(env_out))
    monkeypatch.setenv("MAGICQUANT_TIERS", '["Q5","Q6"]')
    monkeypatch.setattr(
        "magicquant.orchestrator.MagicQuantOrchestrator", _FakeOrchestrator
    )

    cli.cmd_generate(_gen_args(output_dir=str(cli_out), tiers="Q4,Q8"))

    orch = _FakeOrchestrator.last
    assert orch.kwargs["output_dir"] == str(cli_out)
    assert orch.tiered_calls[0]["tiers"] == ["Q4", "Q8"]


# ── cmd_hybrid routing (M5 finish) ──────────────────────────────────────────


def test_cmd_hybrid_honors_env_output_dir(monkeypatch, tmp_path):
    """cmd_hybrid routes its output directory through MagicQuantSettings so
    MAGICQUANT_OUTPUT_DIR is honored uniformly with the other commands."""
    pytest.importorskip("yaml")
    import yaml

    # A real (tiny) source file so the existence check passes.
    src = tmp_path / "src.gguf"
    src.write_bytes(b"GGUF")

    cfg_path = tmp_path / "hybrid.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"model": {"name": "M", "source": str(src)},
             "quantization": {"base": "Q4_K_M", "groups": {}}}
        )
    )

    env_out = tmp_path / "hybrid_env_out"
    monkeypatch.setenv("MAGICQUANT_OUTPUT_DIR", str(env_out))

    captured = {}

    def _fake_create(output_path, **kwargs):
        captured["output_path"] = output_path
        return output_path

    monkeypatch.setattr(
        "magicquant.gguf.writer.create_hybrid_gguf", _fake_create
    )

    cli.cmd_hybrid(argparse.Namespace(config=str(cfg_path), output_dir=None))

    assert captured["output_path"].startswith(str(env_out))


# ── cmd_imatrix --llamacpp-path routing (E6) ────────────────────────────────
# cmd_imatrix used to have no --llamacpp-path flag at all, so llama-imatrix
# resolution was a bare `shutil.which` PATH lookup with no override --
# unlike orchestrator.enable_imatrix / v2's group-probe imatrix resolution,
# which both aim llama-imatrix at the SIBLING of the already-resolved
# perplexity binary to guarantee the same llama.cpp build is used throughout
# a run. These tests drive cmd_imatrix (and the real parser) end to end with
# capture_imatrix/load_imatrix faked, so no real llama.cpp binary is needed.


def _imatrix_args(**kw):
    base = dict(
        model="/tmp/model.gguf", corpus="/tmp/corpus.txt", output=None,
        chunks=-1, ctx_size=512, llamacpp_path=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class _FakeImatrixTools:
    """Stands in for LlamaCppTools -- no real llama.cpp binary involved."""

    def __init__(self, llamacpp_path=None, **kwargs):
        self.llamacpp_path = llamacpp_path
        self.perplexity_tool = None


def test_cmd_imatrix_resolves_sibling_of_perplexity_tool(monkeypatch, tmp_path):
    bin_dir = tmp_path / "fork" / "build" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "llama-imatrix").write_text("")

    class _Tools(_FakeImatrixTools):
        def __init__(self, llamacpp_path=None, **kwargs):
            super().__init__(llamacpp_path, **kwargs)
            self.perplexity_tool = str(bin_dir / "llama-perplexity")

    monkeypatch.setattr("magicquant.utils.llamacpp.LlamaCppTools", _Tools)

    captured = {}

    def _fake_capture(model, corpus, out, **kwargs):
        captured.update(kwargs)
        return out

    monkeypatch.setattr("magicquant.imatrix.capture_imatrix", _fake_capture)
    monkeypatch.setattr("magicquant.imatrix.load_imatrix", lambda out: {})

    cli.cmd_imatrix(_imatrix_args(llamacpp_path=str(tmp_path / "fork")))

    assert captured["imatrix_bin"] == str(bin_dir / "llama-imatrix")


def test_cmd_imatrix_falls_back_to_none_without_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr("magicquant.utils.llamacpp.LlamaCppTools", _FakeImatrixTools)

    captured = {}

    def _fake_capture(model, corpus, out, **kwargs):
        captured.update(kwargs)
        return out

    monkeypatch.setattr("magicquant.imatrix.capture_imatrix", _fake_capture)
    monkeypatch.setattr("magicquant.imatrix.load_imatrix", lambda out: {})

    cli.cmd_imatrix(_imatrix_args())

    assert captured["imatrix_bin"] is None


def test_cmd_imatrix_survives_llamacpp_tools_construction_failure(monkeypatch):
    """A machine with no llama-quantize (only a PATH llama-imatrix) must
    keep working exactly as it did before --llamacpp-path existed --
    LlamaCppTools construction failing must not crash cmd_imatrix, mirroring
    orchestrator.llama_tools' try/except -> None."""

    def _raise(*a, **k):
        raise FileNotFoundError("no llama-quantize found")

    monkeypatch.setattr("magicquant.utils.llamacpp.LlamaCppTools", _raise)

    captured = {}

    def _fake_capture(model, corpus, out, **kwargs):
        captured.update(kwargs)
        return out

    monkeypatch.setattr("magicquant.imatrix.capture_imatrix", _fake_capture)
    monkeypatch.setattr("magicquant.imatrix.load_imatrix", lambda out: {})

    cli.cmd_imatrix(_imatrix_args())

    assert captured["imatrix_bin"] is None


def test_cmd_imatrix_real_parser_has_llamacpp_path_flag(monkeypatch, tmp_path):
    """End-to-end through the real argparse parser (not a hand-built
    Namespace), so a dest mismatch on --llamacpp-path would be caught."""
    bin_dir = tmp_path / "fork" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "llama-imatrix").write_text("")

    class _Tools(_FakeImatrixTools):
        def __init__(self, llamacpp_path=None, **kwargs):
            super().__init__(llamacpp_path, **kwargs)
            self.perplexity_tool = str(bin_dir / "llama-perplexity")

    monkeypatch.setattr("magicquant.utils.llamacpp.LlamaCppTools", _Tools)

    captured = {}

    def _fake_capture(model, corpus, out, **kwargs):
        captured.update(kwargs)
        return out

    monkeypatch.setattr("magicquant.imatrix.capture_imatrix", _fake_capture)
    monkeypatch.setattr("magicquant.imatrix.load_imatrix", lambda out: {})

    monkeypatch.setattr(
        sys, "argv",
        [
            "magicquant", "imatrix", "/tmp/model.gguf", "-f", "/tmp/corpus.txt",
            "--llamacpp-path", str(tmp_path / "fork"),
        ],
    )
    cli.main()

    assert captured["imatrix_bin"] == str(bin_dir / "llama-imatrix")
