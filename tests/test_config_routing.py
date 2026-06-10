"""Config-routing tests (M5).

cmd_search / cmd_generate / cmd_hybrid must build MagicQuantSettings (honoring
env / .env) and let explicit CLI values override. Verifies a single source of
truth for defaults (30/80).
"""
import argparse
import json

import pytest

import magicquant.__main__ as cli
from magicquant.__main__ import _settings_from_args


def _args(**kw):
    base = dict(
        model="model.gguf", output_dir=None, llamacpp_path=None, adapter=None,
        target_quant=None, generations=None, population=None, rounds=None,
        candidates=None,
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
