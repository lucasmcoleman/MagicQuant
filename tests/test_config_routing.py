"""Config-routing tests (M5).

cmd_search must build MagicQuantSettings (honoring env / .env) and let explicit
CLI values override. Verifies a single source of truth for defaults (30/80).
"""
import argparse

import pytest

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
