"""
MagicQuant Orchestrator - Coordinates evolutionary search with llama.cpp.

This script:
1. Runs sensitivity probing (real or heuristic) to weight tensor groups
2. Runs evolutionary search to find optimal hybrid configurations
3. Uses the hybrid GGUF writer to generate per-group quantized models
4. Optionally validates results with llama.cpp perplexity measurements
"""

import os
import json
import time
from typing import Dict, List, Optional
from pathlib import Path

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.evolution.probing import SensitivityProber
from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.utils.naming import generate_name
from magicquant.utils.llamacpp import LlamaCppTools, get_llamacpp_quant_type


class MagicQuantOrchestrator:
    """Orchestrate MagicQuant search with llama.cpp execution."""

    def __init__(
        self,
        source_model_path: str,
        output_dir: str,
        llamacpp_path: Optional[str] = None,
    ):
        """
        Args:
            source_model_path: Path to source GGUF (BF16/F16)
            output_dir: Directory for output models
            llamacpp_path: Path to llama.cpp (auto-detect if None)
        """
        self.source_model_path = source_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize llama.cpp tools
        self.llama_tools = LlamaCppTools(llamacpp_path)

        # Will be populated during run
        self.baseline_ppl: Optional[float] = None
        self.sensitivity_weights: Optional[Dict[str, float]] = None

    def run_full_search(
        self,
        target_base_quant: str = "MXFP4_MOE",
        max_generations: int = 50,
        population_size: int = 100,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Run complete MagicQuant workflow.

        Returns:
            List of discovered configurations sorted by score
        """
        if verbose:
            print("=" * 60)
            print("MagicQuant Hybrid Quantization Search")
            print("=" * 60)
            print(f"Source Model: {self.source_model_path}")
            print(f"Output Directory: {self.output_dir}")
            print(f"Target Base Quant: {target_base_quant}")
            print()

        # Step 1: Calculate baseline perplexity
        if verbose:
            print("Step 1: Calculating baseline perplexity...")

        self.baseline_ppl = self.llama_tools.calculate_perplexity(
            self.source_model_path, verbose=verbose
        )

        if self.baseline_ppl is None:
            print("WARNING: Could not calculate baseline PPL. Using default.")
            self.baseline_ppl = 5.0

        # Step 2: Run real sensitivity probing
        if verbose:
            print("\nStep 2: Running sensitivity probing...")

        prober = SensitivityProber(
            base_model_path=self.source_model_path,
            baseline_perplexity=self.baseline_ppl,
            perplexity_calculator=self.llama_tools,
            output_dir=str(self.output_dir / "_probes"),
        )

        groups = ["E", "H", "Q", "K", "O", "U", "D"]
        prober.probe_all_groups(
            groups=groups,
            aggressive_scheme="Q4_K_M",
            verbose=verbose,
        )

        self.sensitivity_weights = prober.get_normalized_weights()

        # Save sensitivity results
        sensitivity_path = str(self.output_dir / "sensitivity.json")
        prober.save_results(sensitivity_path)

        if verbose:
            print("\nSensitivity weights:")
            for group, weight in self.sensitivity_weights.items():
                print(f"  {group}: {weight:.3f}")

        # Step 3: Get baseline size and speed estimates
        baseline_size_gb = self._estimate_model_size(self.source_model_path)
        baseline_tps = 360

        if verbose:
            print(f"\nBaseline Size: {baseline_size_gb:.2f} GB")
            print(f"Baseline TPS: {baseline_tps} tokens/sec")

        # Step 4: Run evolutionary search
        if verbose:
            print(f"\nStep 3: Running evolutionary search "
                  f"({max_generations} generations)...")

        predictor = PredictiveScorer(
            sensitivity_weights=self.sensitivity_weights,
            baseline_size_gb=baseline_size_gb,
            baseline_tps=baseline_tps,
        )

        survivor = EvolutionarySurvivor(
            predictor=predictor,
            baseline_config={"E": "BF16", "H": "BF16"},
            max_generations=max_generations,
            population_size=population_size,
            epsilon=0.2,
        )

        best_configs = survivor.run_evolution(verbose=verbose)

        # Step 5: Pick the best config per tier
        tiered = self._pick_best_per_tier(best_configs, baseline_size_gb)

        if verbose:
            print("\n" + "=" * 60)
            print("Best Configuration Per Tier:")
            print("=" * 60)
            for tier_name, cfg in tiered.items():
                c = cfg['config']
                ratio = cfg.get('predicted_size_gb', 0) / max(baseline_size_gb, 0.01)
                print(f"\n  {tier_name} ({ratio:.0%} of BF16):")
                print(f"    Loss={cfg.get('predicted_loss', 0):.3f}  "
                      f"Size={cfg.get('predicted_size_gb', 0):.1f}GB  "
                      f"Score={cfg.get('composite_score', 0):.3f}")
                for g in sorted(c):
                    print(f"      {g}: {c[g]}")

        # Return all configs but attach tier info
        for cfg in best_configs:
            if 'tier' not in cfg:
                size_gb = cfg.get('predicted_size_gb', 0)
                cfg['tier'] = self._classify_tier(size_gb, baseline_size_gb)

        return best_configs, tiered

    @staticmethod
    def _classify_tier(size_gb: float, baseline_gb: float) -> str:
        if baseline_gb <= 0:
            return "Q4"
        ratio = size_gb / baseline_gb
        if ratio > 0.55:
            return "Q6"
        elif ratio > 0.40:
            return "Q5"
        elif ratio > 0.28:
            return "Q4"
        else:
            return "Q3"

    @staticmethod
    def _pick_best_per_tier(
        configs: List[Dict], baseline_gb: float
    ) -> Dict[str, Dict]:
        """Select the highest-scoring config from each compression tier."""
        from collections import defaultdict
        by_tier: Dict[str, List[Dict]] = defaultdict(list)
        for cfg in configs:
            size_gb = cfg.get('predicted_size_gb', 0)
            tier = MagicQuantOrchestrator._classify_tier(size_gb, baseline_gb)
            by_tier[tier].append(cfg)
        result = {}
        for tier in ["Q6", "Q5", "Q4", "Q3"]:
            if tier in by_tier:
                best = max(by_tier[tier], key=lambda x: x.get('composite_score', 0))
                result[tier] = best
        return result

    def generate_hybrid_model(
        self,
        config: Dict[str, str],
        model_name: str,
        base_quant: str = "MXFP4_MOE",
        verify: bool = True,
    ) -> Optional[str]:
        """
        Generate a true per-group hybrid GGUF model.

        Uses the hybrid GGUF writer to read the source model, quantise
        each tensor according to its group assignment in *config*, and
        write a single output GGUF file.

        Args:
            config: Group -> quant scheme mapping
            model_name: Base model name for output file
            base_quant: Default scheme for groups not in *config*
            verify: Calculate PPL after generation

        Returns:
            Path to generated model or None on failure
        """
        from magicquant.gguf.writer import create_hybrid_gguf

        output_filename = generate_name(model_name, base_quant, config)
        output_path = self.output_dir / output_filename

        print(f"\nGenerating hybrid GGUF: {output_filename}")
        print(f"  Base scheme: {base_quant}")
        for grp, sch in config.items():
            print(f"  Group {grp} -> {sch}")

        try:
            quant_config = {
                "base": base_quant,
                "groups": config,
            }

            result = create_hybrid_gguf(
                output_path=str(output_path),
                base_model_path=self.source_model_path,
                quant_config=quant_config,
                verbose=True,
            )

            if not os.path.isfile(result):
                print("Failed to generate model")
                return None

        except Exception as exc:
            print(f"Failed to generate model: {exc}")
            return None

        # Verify if requested
        if verify:
            print(f"\nVerifying {output_filename}...")
            ppl = self.llama_tools.calculate_perplexity(str(output_path))

            if ppl and self.baseline_ppl:
                loss = (ppl - self.baseline_ppl) / self.baseline_ppl
                print(f"  Baseline PPL: {self.baseline_ppl:.4f}")
                print(f"  Quantized PPL: {ppl:.4f}")
                print(f"  Precision Loss: {loss*100:.2f}%")

        return str(output_path)

    def generate_tiered_models(
        self,
        tiered: Dict[str, Dict],
        model_name_prefix: str = "Model",
        tiers: Optional[List[str]] = None,
        verify: bool = False,
    ) -> List[str]:
        """
        Generate one hybrid GGUF per compression tier.

        Args:
            tiered: Dict of tier_name -> best config (from run_full_search)
            model_name_prefix: Prefix for output filenames
            tiers: Which tiers to generate (default: all available)
            verify: Run perplexity verification

        Returns:
            List of paths to successfully generated models
        """
        if tiers is None:
            tiers = ["Q6", "Q5", "Q4"]

        generated = []
        for tier in tiers:
            if tier not in tiered:
                print(f"\nNo config found for tier {tier}, skipping")
                continue

            entry = tiered[tier]
            config = entry["config"]
            name = f"{model_name_prefix}-{tier}"

            # Use the most aggressive scheme in the config as the "base" for naming
            base_quant = max(
                set(config.values()),
                key=lambda s: {
                    "BF16": 0, "Q8_0": 1, "Q6_K": 2, "Q5_K": 3,
                    "IQ4_NL": 4, "MXFP4_MOE": 5, "Q4_K_M": 6
                }.get(s, 3)
            )

            print(f"\n{'='*60}")
            print(f"Generating {tier} tier: {name}")
            print(f"  Predicted: loss={entry.get('predicted_loss', 0):.3f}  "
                  f"size={entry.get('predicted_size_gb', 0):.1f}GB")
            print(f"{'='*60}")

            path = self.generate_hybrid_model(
                config=config,
                model_name=name,
                base_quant=base_quant,
                verify=verify,
            )

            if path:
                generated.append(path)
                print(f"  -> {path}")
            else:
                print(f"  -> FAILED")

        return generated

    def generate_top_models(
        self,
        results: List[Dict],
        top_n: int = 3,
        model_name_prefix: str = "Model",
        base_quant: str = "MXFP4_MOE",
        verify: bool = False,
    ) -> List[str]:
        """Generate hybrid GGUFs for the top-N search results by score."""
        generated = []
        for i, entry in enumerate(results[:top_n], 1):
            config = entry["config"]
            name = f"{model_name_prefix}-Config{i}"

            print(f"\n{'='*60}")
            print(f"Generating Configuration {i}/{top_n}")
            print(f"{'='*60}")

            path = self.generate_hybrid_model(
                config=config,
                model_name=name,
                base_quant=base_quant,
                verify=verify,
            )

            if path:
                generated.append(path)
                print(f"  -> {path}")
            else:
                print(f"  -> FAILED")

        return generated

    def _estimate_model_size(self, model_path: str) -> float:
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 ** 3)


def main():
    """Main entry point for orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MagicQuant Orchestrator - Hybrid Quantization Search"
    )
    parser.add_argument(
        "source_model",
        help="Path to source GGUF model (BF16/F16)",
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Output directory for generated models",
    )
    parser.add_argument(
        "--target-quant", default="MXFP4_MOE",
        help="Target base quantization scheme",
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Number of evolutionary generations",
    )
    parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory",
    )

    args = parser.parse_args()

    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.source_model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path,
    )

    best_configs = orchestrator.run_full_search(
        target_base_quant=args.target_quant,
        max_generations=args.generations,
        verbose=True,
    )

    # Generate top 3 configurations
    print("\n" + "=" * 60)
    print("Generating Top 3 Configurations...")
    print("=" * 60)

    orchestrator.generate_top_models(
        results=best_configs,
        top_n=3,
        model_name_prefix="Qwen3-Coder-30B",
        base_quant=args.target_quant,
        verify=True,
    )


if __name__ == "__main__":
    main()
