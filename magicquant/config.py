"""
MagicQuant Configuration - Pydantic-settings based configuration.

Loads settings from environment variables (MAGICQUANT_ prefix) and
optionally from a .env file in the current working directory.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class MagicQuantSettings(BaseSettings):
    """Configuration for MagicQuant search and generation.

    Values are loaded from environment variables with the ``MAGICQUANT_``
    prefix (e.g. ``MAGICQUANT_SOURCE_MODEL_PATH``) and optionally from a
    ``.env`` file.
    """

    source_model_path: str
    output_dir: str = "./output"
    llamacpp_path: Optional[str] = None
    adapter_path: Optional[str] = None
    target_base_quant: str = "MXFP4_MOE"
    search_generations: int = 30
    population_size: int = 80
    measurement_rounds: int = 3
    candidates_per_round: int = 4
    # Early-stopping patience for the evolutionary search. None (default)
    # disables early-stop so the full generation budget runs.
    patience: Optional[int] = None
    tiers: list[str] = ["Q4", "Q5", "Q6"]
    verify: bool = False
    verbose: bool = True
    # Orchestrator knobs (run_measured_search / run_full_search) -- off by
    # default so the historical unweighted/no-KL/no-bench behavior is
    # unchanged unless explicitly opted into.
    use_imatrix: bool = False
    imatrix_corpus: Optional[str] = None
    enable_kl: bool = False
    kl_weight: float = 0.1
    enable_speed_bench: bool = False
    enable_rocmfpx: bool = False
    enable_iq: bool = False
    # Search-bias knobs (EvolutionarySurvivor sampling, not scoring): off by
    # default so the unbiased sampling behavior is unchanged unless opted in.
    stream_aware: bool = False
    head_aggressive: bool = False
    seed: Optional[int] = None
    # Cap on ctx_size-token chunks per perplexity/KL pass during a measured
    # search (forwarded to LlamaCppTools.ppl_chunks, overriding its own
    # MAGICQUANT_PPL_CHUNKS env fallback when set). None = whole corpus.
    measurement_chunks: Optional[int] = None
    # Tunable search-objective knobs (PredictiveScorer.score_hybrid /
    # EvolutionarySurvivor): off by default so the fixed 0.50/0.35/0.15
    # weights and speed_multiplier-based tps scoring are unchanged unless
    # explicitly opted into.
    speed_weight: Optional[float] = None
    use_bytes_tps: bool = False
    # Cross-run noise calibration (magicquant/quant/calibration.py): off by
    # default so predictor noise factors/speed multipliers keep reading the
    # fixed tools/calibration_results.json path (or the static registry)
    # unless explicitly opted into.
    write_calibration: bool = False
    calibration_source: str = ""
    # Search algorithm selector: "v1" (default — the evolutionary
    # Predict->Measure->Learn path, unchanged) or "v2" (budget-constrained
    # per-tensor allocation, docs/redesign.md). v2 requires budget_gb.
    algo: str = "v1"
    # Target model size in GiB for --algo v2 (weights only; leave headroom
    # for ctx/KV inside the GTT envelope).
    budget_gb: Optional[float] = None

    model_config = {
        "env_prefix": "MAGICQUANT_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def output_path(self) -> Path:
        """Return output_dir as a resolved Path."""
        return Path(self.output_dir).resolve()

    @property
    def source_path(self) -> Path:
        """Return source_model_path as a resolved Path."""
        return Path(self.source_model_path).resolve()

    def validate_paths(self) -> list[str]:
        """Check that required paths exist. Returns list of error messages."""
        errors: list[str] = []
        src = self.source_path
        if not src.exists():
            errors.append(f"Source model not found: {src}")
        if self.llamacpp_path and not Path(self.llamacpp_path).exists():
            errors.append(f"llama.cpp path not found: {self.llamacpp_path}")
        if self.adapter_path and not Path(self.adapter_path).exists():
            errors.append(f"Adapter path not found: {self.adapter_path}")
        return errors
