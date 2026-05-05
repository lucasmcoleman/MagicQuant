"""
Sensitivity Probing - Measure tensor group response to quantization.

This module implements the "Probe Phase" described in MagicQuant, where
we measure how each tensor group responds to aggressive compression,
creating sensitivity weights that guide the evolutionary search.

Strategy:
  For each tensor group G, create a temporary GGUF where G is quantised
  with an aggressive scheme (e.g. Q4_K_M) while all other groups stay at
  BF16.  Measure perplexity of each probe model and compare to baseline.
"""

from typing import Dict, List, Tuple, Optional
import os
import json
import tempfile
import numpy as np

from magicquant.quant.schemes import get_scheme_by_name


class SensitivityProber:
    """
    Probe model sensitivity by testing individual groups with aggressive quantization.

    The probe strategy creates temporary hybrid models where only one group
    is compressed to a low precision while all others remain high-precision.
    This reveals which groups are most sensitive to quantization noise.
    """

    def __init__(
        self,
        base_model_path: str,
        baseline_perplexity: float,
        perplexity_calculator=None,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            base_model_path: Path to the source model (BF16 / F16 GGUF)
            baseline_perplexity: PPL of the uncompressed model
            perplexity_calculator: LlamaCppTools instance (or any object
                with a ``calculate_perplexity(model_path, verbose)`` method).
                When *None*, the prober falls back to heuristic estimates.
            output_dir: Directory for temporary probe GGUFs.  A temp dir is
                used when omitted.
        """
        self.base_model_path = base_model_path
        self.baseline_ppl = baseline_perplexity
        self.perplexity_calculator = perplexity_calculator
        self.output_dir = output_dir

        # Results from probes
        self.sensitivity_results: Dict[str, float] = {}
        self.probe_models: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe_all_groups(
        self,
        groups: List[str],
        aggressive_scheme: str = "Q4_K_M",
        keep_scheme: str = "BF16",
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Probe sensitivity for all tensor groups.

        Args:
            groups: Group identifiers to probe ('E', 'H', 'Q', …)
            aggressive_scheme: Scheme applied to the probed group
            keep_scheme: Scheme for all *other* groups (baseline precision)
            verbose: Print progress

        Returns:
            Dictionary mapping group -> sensitivity score
        """
        if verbose:
            print("Running Sensitivity Probes")
            print(f"Baseline PPL: {self.baseline_ppl}")
            print()

        for group in groups:
            ppl = self._probe_single_group(
                group, aggressive_scheme, keep_scheme, verbose
            )

            sensitivity = max(0.0, ppl - self.baseline_ppl) / self.baseline_ppl

            self.sensitivity_results[group] = sensitivity
            self.probe_models.append({
                "group": group,
                "aggressive_scheme": aggressive_scheme,
                "probe_ppl": ppl,
                "sensitivity": sensitivity,
            })

            if verbose:
                print(f"  Group '{group}': PPL={ppl:.4f}, "
                      f"Sensitivity={sensitivity:.4f}")

        return self.sensitivity_results

    def get_normalized_weights(self) -> Dict[str, float]:
        """
        Get normalized sensitivity weights that sum to 1.0.
        """
        total = sum(max(0, s) for s in self.sensitivity_results.values())

        if total == 0:
            return {g: 1.0 / len(self.sensitivity_results)
                    for g in self.sensitivity_results}

        return {g: max(0, s) / total
                for g, s in self.sensitivity_results.items()}

    def get_high_sensitivity_groups(self, threshold: float = 0.1) -> List[str]:
        return [g for g, s in self.sensitivity_results.items() if s > threshold]

    def get_probes(self) -> List[Dict]:
        return self.probe_models.copy()

    def save_results(self, path: str):
        """Persist sensitivity data to *path* as JSON."""
        data = {
            "baseline_ppl": self.baseline_ppl,
            "sensitivity": self.sensitivity_results,
            "normalized_weights": self.get_normalized_weights(),
            "probes": self.probe_models,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _probe_single_group(
        self,
        group: str,
        scheme: str,
        keep_scheme: str,
        verbose: bool = True,
    ) -> float:
        """
        Create a probe GGUF where only *group* is quantised with *scheme*
        and all others stay at *keep_scheme*, then measure perplexity.

        Falls back to heuristic estimation when no perplexity_calculator or
        no writable source model is available.
        """
        if verbose:
            print(f"  Probing group '{group}' with {scheme}...")

        # If we have both a calculator and a usable source GGUF, do a real probe
        if (
            self.perplexity_calculator is not None
            and os.path.isfile(self.base_model_path)
        ):
            return self._real_probe(group, scheme, keep_scheme, verbose)

        # Fallback: heuristic estimate
        return self._heuristic_probe(group, scheme)

    def _real_probe(
        self,
        group: str,
        scheme: str,
        keep_scheme: str,
        verbose: bool,
    ) -> float:
        """Build a real probe GGUF, run perplexity, clean up."""
        from magicquant.gguf.writer import create_hybrid_gguf

        # Determine a directory for probe files
        probe_dir = self.output_dir or tempfile.mkdtemp(prefix="mq_probe_")
        os.makedirs(probe_dir, exist_ok=True)
        probe_path = os.path.join(probe_dir, f"probe_{group}.gguf")

        try:
            # Build quant config: every group at keep_scheme except the target
            from magicquant.gguf.reader import GGUFReader
            from magicquant.gguf.tensor_groups import TensorGroupClassifier

            reader = GGUFReader(self.base_model_path)
            reader.open()
            classifier = TensorGroupClassifier()
            all_groups = set()
            for name in reader.get_tensor_names():
                g = classifier.classify_tensor(name)
                if g != "UNKNOWN":
                    all_groups.add(g)
            reader.close()

            group_overrides = {g: keep_scheme for g in all_groups}
            group_overrides[group] = scheme

            quant_config = {
                "base": keep_scheme,
                "groups": group_overrides,
            }

            if verbose:
                print(f"    Creating probe model: {probe_path}")

            create_hybrid_gguf(
                output_path=probe_path,
                base_model_path=self.base_model_path,
                quant_config=quant_config,
                verbose=False,
            )

            # Measure perplexity
            ppl = self.perplexity_calculator.calculate_perplexity(
                probe_path, verbose=verbose
            )

            if ppl is None:
                if verbose:
                    print(f"    WARNING: PPL measurement failed for group '{group}', "
                          "falling back to heuristic")
                return self._heuristic_probe(group, scheme)

            return ppl

        except Exception as exc:
            if verbose:
                print(f"    Probe failed ({exc}), using heuristic")
            return self._heuristic_probe(group, scheme)

        finally:
            # Clean up temporary probe file
            if os.path.exists(probe_path):
                try:
                    os.remove(probe_path)
                    if verbose:
                        print(f"    Cleaned up {probe_path}")
                except OSError:
                    pass

    def _heuristic_probe(self, group: str, scheme: str) -> float:
        """
        Estimate probe PPL without actually creating a model.

        Uses empirical sensitivity factors observed across a range of
        LLaMA / Qwen / Mistral architectures.
        """
        # Empirical sensitivity multipliers (relative PPL increase when
        # the group is quantised to ~4 bpw while everything else is BF16)
        _GROUP_SENSITIVITY = {
            "E": 2.0,   # Embeddings — very sensitive
            "H": 1.8,   # LM Head — very sensitive
            "O": 1.6,   # Attention output — sensitive
            "R": 1.5,   # MoE router — sensitive
            "Q": 1.2,   # Attention query — moderate
            "K": 1.1,   # Attention key/value — moderate
            "U": 0.6,   # FFN up/gate — robust
            "D": 0.7,   # FFN down — robust
            "X": 0.5,   # MoE experts — robust
        }

        # Scheme aggressiveness scaled to the heuristic's [0, 1] range.
        # Registry's noise_factor uses Q8_0=1.0 anchor; we rescale here so
        # Q4_K_M=1.0 maps to "max heuristic aggressiveness". This preserves
        # the original heuristic's behavior pre-refactor.
        try:
            registry_noise = get_scheme_by_name(scheme).noise_factor
            # Q4_K_M (registry noise=4.5) maps to 1.0; linearly scale others.
            noise = registry_noise / 4.5
        except ValueError:
            noise = 1.0

        sensitivity = _GROUP_SENSITIVITY.get(group, 1.0)

        ppl_increase_pct = sensitivity * noise * 0.05  # ~5% per unit at baseline
        return self.baseline_ppl * (1 + ppl_increase_pct)


class SensitivityAnalysis:
    """Analyze sensitivity data and generate recommendations."""

    @staticmethod
    def recommend_protected_groups(
        sensitivity_results: Dict[str, float],
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        sorted_groups = sorted(
            sensitivity_results.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_groups[:top_n]

    @staticmethod
    def identify_robust_groups(
        sensitivity_results: Dict[str, float],
        max_sensitivity: float = 0.05,
    ) -> List[str]:
        return [g for g, s in sensitivity_results.items() if s <= max_sensitivity]


if __name__ == "__main__":
    print("Sensitivity Probing Demo")
    print("=" * 50)

    prober = SensitivityProber(
        base_model_path="dummy.gguf",
        baseline_perplexity=5.23,
        perplexity_calculator=None,
    )

    groups_to_probe = ["E", "H", "Q", "K", "O", "U", "D"]

    sensitivity = prober.probe_all_groups(
        groups=groups_to_probe,
        aggressive_scheme="Q4_K_M",
        verbose=True,
    )

    print()
    print("Sensitivity Results:")
    for group, sens in sensitivity.items():
        print(f"  {group}: {sens:.4f}")

    print()
    weights = prober.get_normalized_weights()
    print("Normalized Weights (sum to 1):")
    for group, w in weights.items():
        print(f"  {group}: {w:.4f}")
