"""Orchestrator-level incumbent-seeding tests (Part B item 3).

Uses the same fake-tools pattern as test_orchestrator_measurement.py: real
EvolutionarySurvivor/PredictiveScorer/SensitivityProber run unmocked, only
the I/O boundary (model source, llama.cpp tools, candidate GGUF building) is
faked.
"""
import json

import magicquant.gguf.source as source_mod
from magicquant.incumbents import get_incumbent_config
from magicquant.orchestrator import MagicQuantOrchestrator


_TENSOR_NAMES = [
    "token_embd.weight",
    "output.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
]

_BASE_GROUPS = {"E", "H", "Q", "K", "O", "U", "D"}


class _FakeSource:
    def get_tensor_names(self):
        return list(_TENSOR_NAMES)

    def get_all_tensors_info(self):
        return [{"name": n, "shape": [4, 4]} for n in _TENSOR_NAMES]

    def close(self):
        pass


class _FakeLlamaTools:
    def __init__(self):
        self.ctx_size = 512

    def calculate_perplexity(self, path, verbose=False, **kw):
        return 5.0

    def _resolve_data_file(self, data_file=None):
        return "/fake/corpus.txt"


def _make_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setattr(source_mod, "open_model_source", lambda *a, **k: _FakeSource())
    orch = MagicQuantOrchestrator(
        source_model_path=str(tmp_path / "nonexistent.gguf"),
        output_dir=str(tmp_path / "out"),
    )
    fake_tools = _FakeLlamaTools()
    orch._llama_tools = fake_tools

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    counter = {"n": 0}

    def fake_build_candidate(config, name, base_quant):
        counter["n"] += 1
        p = candidates_dir / f"{name}_{counter['n']}.gguf"
        p.write_bytes(b"0" * 1024)
        return str(p)

    monkeypatch.setattr(orch, "_build_candidate", fake_build_candidate)
    return orch, fake_tools


# ── _build_incumbent_seeds ───────────────────────────────────────────────


def test_build_incumbent_seeds_restricts_to_search_groups(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch._search_groups = ["E", "H", "Q"]
    seed_configs, tier_by_key = orch._build_incumbent_seeds(True)

    assert len(seed_configs) == 3  # Q4, Q5, Q6
    for cfg in seed_configs:
        assert set(cfg.keys()) == {"E", "H", "Q"}
    assert set(tier_by_key.values()) == {"Q4", "Q5", "Q6"}


def test_build_incumbent_seeds_disabled_returns_empty(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)
    orch._search_groups = ["E", "H", "Q"]
    seed_configs, tier_by_key = orch._build_incumbent_seeds(False)

    assert seed_configs == []
    assert tier_by_key == {}


# ── run_measured_search: default seeding + forced round-1 measurement ────


def test_default_seed_incumbents_force_measures_all_three_tiers(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=1, verbose=False,
        probe_kl=False,  # _FakeLlamaTools has no save_base_logits
    )

    incumbent_tags = {
        info["incumbent"] for info in orch._measured.values() if info.get("incumbent")
    }
    assert incumbent_tags == {"Q4", "Q5", "Q6"}

    # Each tagged measurement's config must actually be the restricted
    # incumbent config for its tier.
    for info in orch._measured.values():
        tier = info.get("incumbent")
        if not tier:
            continue
        expected = {
            g: s for g, s in get_incumbent_config(tier).items() if g in _BASE_GROUPS
        }
        assert info["config"] == expected

    results = json.loads((orch.output_dir / "search_results.json").read_text())
    saved_tags = {
        v.get("incumbent") for v in results["measurements"].values() if v.get("incumbent")
    }
    assert saved_tags == {"Q4", "Q5", "Q6"}


def test_seed_incumbents_false_disables_seeding_and_measurement(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=1, candidates_per_round=2, verbose=False,
        seed_incumbents=False,
        probe_kl=False,  # _FakeLlamaTools has no save_base_logits
    )

    assert not any("incumbent" in info for info in orch._measured.values())


def test_incumbent_not_remeasured_if_already_measured_in_a_later_round(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    orch.run_measured_search(
        search_generations=2, population_size=8,
        measurement_rounds=2, candidates_per_round=1, verbose=False,
        probe_kl=False,  # _FakeLlamaTools has no save_base_logits
    )

    # Every incumbent config_key must be measured exactly once across all
    # rounds (the round-1 force-measure + the "skip if in self._measured"
    # guard in the build loop must not double-build/measure the same key).
    incumbent_keys = [k for k, v in orch._measured.items() if v.get("incumbent")]
    assert len(incumbent_keys) == len(set(incumbent_keys))


# ── run_full_search: seeding without measurement ─────────────────────────


def test_run_full_search_seeds_incumbents_into_discovered_configs(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    best_configs, _tiered = orch.run_full_search(
        max_generations=2, population_size=8, verbose=False, seed=123,
    )

    configs = [c["config"] for c in best_configs]
    q4_seed = {
        g: s for g, s in get_incumbent_config("Q4").items() if g in orch._search_groups
    }
    assert q4_seed in configs


def test_run_full_search_seed_incumbents_false_omits_incumbent_configs(tmp_path, monkeypatch):
    orch, _ = _make_orchestrator(tmp_path, monkeypatch)

    best_configs, _tiered = orch.run_full_search(
        max_generations=2, population_size=8, verbose=False, seed_incumbents=False,
        seed=123,
    )

    configs = [c["config"] for c in best_configs]
    q4_seed = {
        g: s for g, s in get_incumbent_config("Q4").items() if g in orch._search_groups
    }
    assert q4_seed not in configs
