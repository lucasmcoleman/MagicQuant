"""
MagicQuant Orchestrator - Coordinates the full Predict -> Measure -> Learn pipeline.

The core loop:
1. Sensitivity probing — measure per-group PPL impact
2. Evolutionary search — generate candidate hybrid configs, predict performance
3. Build & measure — create GGUFs for tier winners, run real perplexity
4. Active learning — feed residuals (measured - predicted) back into predictor
5. Repeat until convergence or budget exhausted
6. Output the best verified survivor per tier
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.evolution.probing import SensitivityProber
from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.utils.naming import generate_name
from magicquant.utils.llamacpp import LlamaCppTools, get_llamacpp_quant_type
from magicquant.logging import get_logger

log = get_logger(__name__)


class MagicQuantOrchestrator:
    """
    Orchestrate the full MagicQuant search with real measurement feedback.

    The key difference from a prediction-only search: after each evolutionary
    round, the top candidates are actually built as GGUF files and measured
    with llama-perplexity. The residuals (measured_loss - predicted_loss)
    are fed back into the predictor, making it increasingly accurate for
    this specific model architecture.
    """

    def __init__(
        self,
        source_model_path: str,
        output_dir: str,
        llamacpp_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ):
        self.source_model_path = source_model_path
        self.adapter_path = adapter_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._llamacpp_path = llamacpp_path
        self._llama_tools: Optional[LlamaCppTools] = None

        self.baseline_ppl: Optional[float] = None
        self.sensitivity_weights: Optional[Dict[str, float]] = None
        self.predictor: Optional[PredictiveScorer] = None

        # Track all measured configs across rounds
        self._measured: Dict[str, Dict] = {}  # config_key -> {config, ppl, loss, path}

    @property
    def llama_tools(self) -> Optional[LlamaCppTools]:
        """Lazily initialize LlamaCppTools on first access."""
        if self._llama_tools is None:
            try:
                self._llama_tools = LlamaCppTools(self._llamacpp_path)
            except Exception as exc:
                log.warning("llama.cpp not available", error=str(exc))
                return None
        return self._llama_tools

    # ------------------------------------------------------------------
    # Full measured search (the real MagicQuant pipeline)
    # ------------------------------------------------------------------

    def run_measured_search(
        self,
        target_base_quant: str = "MXFP4_MOE",
        search_generations: int = 30,
        population_size: int = 80,
        measurement_rounds: int = 3,
        candidates_per_round: int = 4,
        verbose: bool = True,
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Run the full Predict -> Measure -> Learn loop.

        Args:
            target_base_quant: Default base quantization scheme
            search_generations: Evolutionary generations per round
            population_size: Candidates per generation
            measurement_rounds: How many build-measure-learn cycles
            candidates_per_round: How many configs to actually build and
                measure per round (tier winners + epsilon-greedy picks)
            verbose: Print progress

        Returns:
            (all_configs, tiered_best) where tiered_best maps tier names
            to the best *measured* config for that tier.
        """
        if verbose:
            log.info(
                "MagicQuant Measured Hybrid Search",
                stage="init",
                source=self.source_model_path,
                adapter=self.adapter_path,
                output_dir=str(self.output_dir),
                rounds=measurement_rounds,
                candidates_per_round=candidates_per_round,
            )

        if self.llama_tools is None:
            raise RuntimeError(
                "run_measured_search requires llama.cpp. Install it or use "
                "prediction-only mode (--rounds 0)."
            )

        # ── Step 1: Baseline perplexity ──
        if verbose:
            log.info("Baseline perplexity", stage="baseline")

        self.baseline_ppl = self.llama_tools.calculate_perplexity(
            self.source_model_path, verbose=verbose
        )
        if self.baseline_ppl is None:
            log.warning("Could not measure baseline PPL, using default", stage="baseline", default_ppl=5.0)
            self.baseline_ppl = 5.0

        # ── Step 2: Sensitivity probing ──
        if verbose:
            log.info("Sensitivity probing", stage="probing")

        prober = SensitivityProber(
            base_model_path=self.source_model_path,
            baseline_perplexity=self.baseline_ppl,
            perplexity_calculator=self.llama_tools,
            output_dir=str(self.output_dir / "_probes"),
        )

        groups = ["E", "H", "Q", "K", "O", "U", "D"]
        # Add MoE/SSM groups if present in the model
        classifier = TensorGroupClassifier()
        from magicquant.gguf.source import open_model_source
        _src = open_model_source(self.source_model_path)
        try:
            tensor_names = _src.get_tensor_names()
        finally:
            _src.close()
        if any(classifier.classify_tensor(t) in ("X", "R") for t in tensor_names):
            groups.extend(["X", "R"])
        if any(classifier.classify_tensor(t) == "S" for t in tensor_names):
            groups.append("S")

        prober.probe_all_groups(groups=groups, aggressive_scheme="Q4_K_M", verbose=verbose)
        self.sensitivity_weights = prober.get_normalized_weights()
        prober.save_results(str(self.output_dir / "sensitivity.json"))

        if verbose:
            log.info(
                "Sensitivity weights computed",
                stage="probing",
                weights={g: round(w, 3) for g, w in self.sensitivity_weights.items()},
            )

        # ── Step 3: Initialize predictor ──
        baseline_size_gb = self._estimate_model_size(self.source_model_path)

        self.predictor = PredictiveScorer(
            sensitivity_weights=self.sensitivity_weights,
            baseline_size_gb=baseline_size_gb,
            baseline_tps=360,
        )

        # ── Step 4: Measured search rounds ──
        all_configs = []

        for round_idx in range(measurement_rounds):
            if verbose:
                log.info(
                    "Measurement round starting",
                    stage="measurement",
                    round=round_idx + 1,
                    total_rounds=measurement_rounds,
                )

            # 4a. Run evolutionary search with current predictor
            survivor = EvolutionarySurvivor(
                predictor=self.predictor,
                baseline_config={"E": "BF16", "H": "BF16"},
                max_generations=search_generations,
                population_size=population_size,
                epsilon=0.2,
            )

            round_configs = survivor.run_evolution(verbose=verbose)
            all_configs.extend(round_configs)

            # 4b. Pick candidates to measure: tier winners + epsilon picks
            to_measure = self._select_measurement_candidates(
                round_configs, baseline_size_gb, candidates_per_round
            )

            if verbose:
                log.info(
                    "Candidates selected for measurement",
                    stage="measurement",
                    count=len(to_measure),
                )

            # 4c. Build, measure, learn
            for i, candidate in enumerate(to_measure):
                config = candidate["config"]
                config_key = self._config_key(config)

                # Skip if already measured
                if config_key in self._measured:
                    if verbose:
                        log.debug(
                            "Candidate already measured, skipping",
                            stage="measurement",
                            progress=f"{i+1}/{len(to_measure)}",
                        )
                    continue

                if verbose:
                    schemes = " ".join(f"{g}:{s}" for g, s in sorted(config.items()))
                    log.info(
                        "Building candidate",
                        stage="measurement",
                        progress=f"{i+1}/{len(to_measure)}",
                        schemes=schemes,
                    )

                # Build GGUF
                model_name = f"round{round_idx+1}_candidate{i+1}"
                path = self._build_candidate(config, model_name, target_base_quant)

                if path is None:
                    continue

                # Measure perplexity
                ppl = self.llama_tools.calculate_perplexity(path, verbose=verbose)

                if ppl is not None:
                    measured_loss = (ppl - self.baseline_ppl) / self.baseline_ppl
                    predicted_loss = self.predictor.predict_loss(config)
                    residual = measured_loss - predicted_loss

                    # Record measurement
                    candidate_path = Path(path)
                    self._measured[config_key] = {
                        "config": config,
                        "ppl": ppl,
                        "measured_loss": measured_loss,
                        "predicted_loss": predicted_loss,
                        "residual": residual,
                        "path": path,
                        "size_gb": candidate_path.stat().st_size / (1024 ** 3),
                    }

                    # Active learning: feed residual back
                    self.predictor.record_residual(config, residual)

                    if verbose:
                        log.info(
                            "Candidate measured",
                            stage="measurement",
                            ppl=round(ppl, 4),
                            measured_loss=round(measured_loss, 4),
                            predicted_loss=round(predicted_loss, 4),
                            residual=round(residual, 4),
                        )
                else:
                    if verbose:
                        log.warning("Measurement failed", stage="measurement")

                # Clean up candidate GGUF to save disk (keep only final survivors)
                # We'll rebuild the final survivors at the end
                candidate_file = Path(path)
                if candidate_file.exists():
                    candidate_file.unlink()

            # 4d. Log round summary
            if verbose and self._measured:
                avg_residual = sum(
                    abs(m["residual"]) for m in self._measured.values()
                ) / len(self._measured)
                log.info(
                    "Round summary",
                    stage="measurement",
                    round=round_idx + 1,
                    total_measurements=len(self._measured),
                    mean_abs_residual=round(avg_residual, 4),
                )

        # ── Step 5: Select final survivors per tier ──
        tiered = self._select_final_survivors(baseline_size_gb)

        # Save all results
        self._save_results(all_configs, tiered)

        if verbose:
            for tier, info in tiered.items():
                c = info["config"]
                schemes = " ".join(f"{g}:{s}" for g, s in sorted(c.items()))
                log.info(
                    "Final verified survivor",
                    stage="results",
                    tier=tier,
                    ppl=round(info["ppl"], 4),
                    measured_loss=round(info["measured_loss"], 4),
                    size_gb=round(info["size_gb"], 2),
                    schemes=schemes,
                )

        return all_configs, tiered

    def _select_measurement_candidates(
        self,
        configs: List[Dict],
        baseline_gb: float,
        n: int,
    ) -> List[Dict]:
        """Pick the best candidates to actually build and measure."""
        # Get tier winners
        tiered = self._pick_best_per_tier(configs, baseline_gb)
        candidates = list(tiered.values())

        # Add epsilon-greedy: random picks that might be surprise winners
        import random
        remaining = [c for c in configs if c not in candidates]
        if remaining and len(candidates) < n:
            random.shuffle(remaining)
            candidates.extend(remaining[:n - len(candidates)])

        # Skip already-measured configs
        candidates = [
            c for c in candidates
            if self._config_key(c["config"]) not in self._measured
        ]

        return candidates[:n]

    def _build_candidate(
        self, config: Dict[str, str], name: str, base_quant: str
    ) -> Optional[str]:
        """Build a hybrid GGUF for measurement. Returns path or None."""
        from magicquant.gguf.writer import create_hybrid_gguf

        output_filename = generate_name(name, base_quant, config)
        candidates_dir = self.output_dir / "_candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(candidates_dir / output_filename)

        try:
            return create_hybrid_gguf(
                output_path=output_path,
                base_model_path=self.source_model_path,
                quant_config={"base": base_quant, "groups": config},
                verbose=False,
                adapter_path=self.adapter_path,
            )
        except Exception as exc:
            log.error("Build failed", stage="build", error=str(exc))
            return None

    def _select_final_survivors(self, baseline_gb: float) -> Dict[str, Dict]:
        """From all measured configs, pick the best per tier."""
        by_tier: Dict[str, List[Dict]] = defaultdict(list)
        for info in self._measured.values():
            tier = self._classify_tier(info["size_gb"], baseline_gb)
            by_tier[tier].append(info)

        result = {}
        for tier in ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]:
            if tier in by_tier:
                # Best = lowest measured loss within the tier
                best = min(by_tier[tier], key=lambda x: x["measured_loss"])
                result[tier] = best
        return result

    def _save_results(self, all_configs, tiered):
        """Persist search results and measurements to JSON."""
        results = {
            "baseline_ppl": self.baseline_ppl,
            "measurements": {
                k: {
                    "config": v["config"],
                    "ppl": v["ppl"],
                    "measured_loss": v["measured_loss"],
                    "predicted_loss": v["predicted_loss"],
                    "residual": v["residual"],
                    "size_gb": v["size_gb"],
                }
                for k, v in self._measured.items()
            },
            "tiered_survivors": {
                tier: {
                    "config": info["config"],
                    "ppl": info["ppl"],
                    "measured_loss": info["measured_loss"],
                    "size_gb": info["size_gb"],
                }
                for tier, info in tiered.items()
            },
            "tiered": {
                tier: {
                    "config": info["config"],
                    "ppl": info.get("ppl"),
                    "measured_loss": info.get("measured_loss"),
                    "predicted_loss": info.get("predicted_loss"),
                    "size_gb": info.get("size_gb"),
                }
                for tier, info in tiered.items()
            },
        }

        results_path = self.output_dir / "search_results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Prediction-only search (no llama.cpp needed)
    # ------------------------------------------------------------------

    def run_full_search(
        self,
        target_base_quant: str = "MXFP4_MOE",
        max_generations: int = 50,
        population_size: int = 100,
        verbose: bool = True,
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Run prediction-only evolutionary search (no real measurements).
        Use run_measured_search() for the full Predict->Measure->Learn loop.
        """
        if verbose:
            log.info(
                "MagicQuant Prediction-Only Search",
                stage="init",
                source=self.source_model_path,
            )

        # Baseline PPL
        _llama = self.llama_tools
        if _llama is not None:
            self.baseline_ppl = _llama.calculate_perplexity(
                self.source_model_path, verbose=verbose
            )
            if self.baseline_ppl is None:
                self.baseline_ppl = 5.0
        else:
            log.warning(
                "llama.cpp unavailable, using default baseline PPL",
                stage="baseline",
                default_ppl=5.0,
            )
            self.baseline_ppl = 5.0

        # Sensitivity probing
        if verbose:
            log.info("Sensitivity probing", stage="probing")
        prober = SensitivityProber(
            base_model_path=self.source_model_path,
            baseline_perplexity=self.baseline_ppl,
            perplexity_calculator=_llama,
            output_dir=str(self.output_dir / "_probes"),
        )
        groups = ["E", "H", "Q", "K", "O", "U", "D"]
        # Add MoE/SSM groups if present in the model
        classifier = TensorGroupClassifier()
        from magicquant.gguf.source import open_model_source
        _src = open_model_source(self.source_model_path)
        try:
            tensor_names = _src.get_tensor_names()
        finally:
            _src.close()
        if any(classifier.classify_tensor(t) in ("X", "R") for t in tensor_names):
            groups.extend(["X", "R"])
        if any(classifier.classify_tensor(t) == "S" for t in tensor_names):
            groups.append("S")

        prober.probe_all_groups(groups=groups, aggressive_scheme="Q4_K_M", verbose=verbose)
        self.sensitivity_weights = prober.get_normalized_weights()
        prober.save_results(str(self.output_dir / "sensitivity.json"))

        baseline_size_gb = self._estimate_model_size(self.source_model_path)

        self.predictor = PredictiveScorer(
            sensitivity_weights=self.sensitivity_weights,
            baseline_size_gb=baseline_size_gb,
            baseline_tps=360,
        )

        survivor = EvolutionarySurvivor(
            predictor=self.predictor,
            baseline_config={"E": "BF16", "H": "BF16"},
            max_generations=max_generations,
            population_size=population_size,
            epsilon=0.2,
        )

        best_configs = survivor.run_evolution(verbose=verbose)
        tiered = self._pick_best_per_tier(best_configs, baseline_size_gb)

        for cfg in best_configs:
            if 'tier' not in cfg:
                cfg['tier'] = self._classify_tier(
                    cfg.get('predicted_size_gb', 0), baseline_size_gb
                )

        return best_configs, tiered

    # ------------------------------------------------------------------
    # Model generation
    # ------------------------------------------------------------------

    def generate_hybrid_model(
        self, config: Dict[str, str], model_name: str,
        base_quant: str = "MXFP4_MOE", verify: bool = True,
    ) -> Optional[str]:
        """Generate a hybrid GGUF model."""
        from magicquant.gguf.writer import create_hybrid_gguf

        output_filename = generate_name(model_name, base_quant, config)
        output_path = self.output_dir / output_filename

        log.info(
            "Generating hybrid GGUF",
            stage="generate",
            filename=output_filename,
            group_schemes={g: s for g, s in sorted(config.items())},
        )

        try:
            result = create_hybrid_gguf(
                output_path=str(output_path),
                base_model_path=self.source_model_path,
                quant_config={"base": base_quant, "groups": config},
                verbose=True,
                adapter_path=self.adapter_path,
            )
            if not Path(result).is_file():
                return None
        except Exception as exc:
            log.error("Generation failed", stage="generate", error=str(exc))
            return None

        if verify and self.baseline_ppl:
            ppl = self.llama_tools.calculate_perplexity(str(output_path))
            if ppl:
                loss = (ppl - self.baseline_ppl) / self.baseline_ppl
                log.info(
                    "Verification complete",
                    stage="generate",
                    ppl=round(ppl, 4),
                    loss_pct=round(loss * 100, 2),
                )

        return str(output_path)

    def generate_tiered_models(
        self, tiered: Dict[str, Dict], model_name_prefix: str = "Model",
        tiers: Optional[List[str]] = None, verify: bool = False,
    ) -> List[str]:
        """Generate one hybrid GGUF per compression tier."""
        if tiers is None:
            tiers = ["Q8", "Q6", "Q5", "Q4", "Q2"]

        generated = []
        for tier in tiers:
            if tier not in tiered:
                log.info("No config for tier, skipping", stage="generate", tier=tier)
                continue

            entry = tiered[tier]
            config = entry["config"]
            name = f"{model_name_prefix}-{tier}"

            base_quant = max(
                set(config.values()),
                key=lambda s: {
                    "BF16": 0, "Q8_0": 1, "Q6_K": 2, "Q5_K": 3,
                    "IQ4_NL": 4, "MXFP4_MOE": 5, "Q4_K_M": 6
                }.get(s, 3)
            )

            log.info(
                "Generating tier model",
                stage="generate",
                tier=tier,
                name=name,
                ppl=round(entry["ppl"], 4) if "ppl" in entry else None,
                measured_loss=round(entry["measured_loss"], 4) if "measured_loss" in entry else None,
            )

            path = self.generate_hybrid_model(
                config=config, model_name=name,
                base_quant=base_quant, verify=verify,
            )
            if path:
                generated.append(path)
            else:
                log.error("Tier generation failed", stage="generate", tier=tier)

        return generated

    def generate_top_models(
        self, results: List[Dict], top_n: int = 3,
        model_name_prefix: str = "Model", base_quant: str = "MXFP4_MOE",
        verify: bool = False,
    ) -> List[str]:
        """Generate hybrid GGUFs for the top-N results by score."""
        generated = []
        for i, entry in enumerate(results[:top_n], 1):
            path = self.generate_hybrid_model(
                config=entry["config"],
                model_name=f"{model_name_prefix}-Config{i}",
                base_quant=base_quant, verify=verify,
            )
            if path:
                generated.append(path)
        return generated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _config_key(config: Dict[str, str]) -> str:
        return "|".join(f"{g}:{config[g]}" for g in sorted(config))

    @staticmethod
    def _classify_tier(size_gb: float, baseline_gb: float) -> str:
        if baseline_gb <= 0:
            return "Q4"
        ratio = size_gb / baseline_gb
        # Tighter boundaries: Q6 targets 45-65% of BF16 (not open-ended)
        # Configs above 65% are over-protected and wasteful
        if 0.45 < ratio <= 0.65:
            return "Q6"
        elif 0.33 < ratio <= 0.45:
            return "Q5"
        elif 0.22 < ratio <= 0.33:
            return "Q4"
        elif 0.16 < ratio <= 0.22:
            return "Q3"
        elif ratio <= 0.16:
            return "Q2"
        return "Q8"  # ratio > 0.65 — barely compressed, separate tier

    @staticmethod
    def _pick_best_per_tier(configs: List[Dict], baseline_gb: float) -> Dict[str, Dict]:
        by_tier: Dict[str, List[Dict]] = defaultdict(list)
        for cfg in configs:
            size_gb = cfg.get('predicted_size_gb', 0)
            tier = MagicQuantOrchestrator._classify_tier(size_gb, baseline_gb)
            by_tier[tier].append(cfg)
        result = {}
        for tier in ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]:
            if tier in by_tier:
                result[tier] = max(by_tier[tier], key=lambda x: x.get('composite_score', 0))
        return result

    def _estimate_model_size(self, model_path: str) -> float:
        """Compute BF16 baseline size in GB from the total parameter count.

        Using ``parameter_count * 2`` bytes gives the true BF16 baseline
        regardless of the source format (which could be a pre-quantized GGUF
        with a smaller on-disk size).
        """
        from magicquant.gguf.source import open_model_source
        try:
            src = open_model_source(model_path)
            try:
                total_elements = 0
                for info in src.get_all_tensors_info():
                    n = 1
                    for d in info["shape"]:
                        n *= d
                    total_elements += n
                if total_elements > 0:
                    return (total_elements * 2) / (1024 ** 3)
            finally:
                src.close()
        except Exception as exc:
            log.warning(
                "Could not count parameters for baseline size",
                model_path=model_path,
                error=str(exc),
            )

        # Last-resort fallback: file size (may be wrong for pre-quantized)
        p = Path(model_path)
        if p.is_file():
            return p.stat().st_size / (1024 ** 3)
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
            return total / (1024 ** 3)
        log.warning(
            "Could not estimate model size, using default",
            model_path=model_path,
            default_size_gb=1.0,
        )
        return 1.0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MagicQuant Orchestrator - Hybrid Quantization Search"
    )
    parser.add_argument("source_model", help="Path to source model (GGUF, safetensors, or directory)")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--target-quant", default="MXFP4_MOE")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3, help="Measurement rounds (0 = prediction only)")
    parser.add_argument("--candidates", type=int, default=4, help="Candidates to measure per round")
    parser.add_argument("--llamacpp-path", help="Path to llama.cpp directory")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")

    args = parser.parse_args()

    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.source_model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path,
        adapter_path=args.adapter,
    )

    if args.rounds > 0:
        all_configs, tiered = orchestrator.run_measured_search(
            target_base_quant=args.target_quant,
            search_generations=args.generations,
            measurement_rounds=args.rounds,
            candidates_per_round=args.candidates,
            verbose=True,
        )
    else:
        all_configs, tiered = orchestrator.run_full_search(
            target_base_quant=args.target_quant,
            max_generations=args.generations,
            verbose=True,
        )

    # Generate final survivors
    log.info("Generating final survivors", stage="generate")

    orchestrator.generate_tiered_models(
        tiered=tiered,
        model_name_prefix=Path(args.source_model).stem,
        tiers=["Q2", "Q4", "Q5", "Q6", "Q8"],
        verify=False,
    )


if __name__ == "__main__":
    main()
