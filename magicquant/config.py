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
