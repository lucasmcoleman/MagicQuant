"""
Predictive Scoring Model - Estimate performance of hybrid quantizations.

This module implements the predictive model that estimates:
- Loss (perplexity degradation) for a given hybrid configuration
- File size based on parameter counts and quantization schemes
- Inference speed (TPS) estimates

The model uses learned sensitivity weights from probing to make predictions,
with non-linear collapse penalties for compressing multiple sensitive layers.

Noise factors are calibrated against published perplexity benchmarks.
Compression ratios are derived from actual ggml block format byte sizes.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class PredictiveScorer:
    """
    Predict performance of hybrid quantization configurations.

    Uses a physics-based model with sensitivity weights to estimate:
        - Loss (PPL degradation)
        - Size (file size in GB)
        - Speed (TPS estimates)

    Also implements the non-linear "Collapse" penalty when multiple
    sensitive layers are compressed simultaneously.
    """

    # Noise factors calibrated from llama.cpp perplexity benchmarks.
    # Lower = less quantization noise = better quality.
    #
    # Key insight: non-linear schemes (IQ4_NL, MXFP4) produce lower noise
    # than integer schemes at comparable bpw because their quantization
    # levels better match the Gaussian-like weight distribution of
    # transformers.
    QUANT_NOISE_FACTORS = {
        "BF16":      0.0,
        "Q8_0":      1.0,
        "Q6_K":      2.2,
        "Q5_K":      3.0,
        "IQ4_NL":    3.8,   # non-linear lookup table, best ~4-bit quality
        "MXFP4_MOE": 4.0,   # FP4 levels, better than integer Q4
        "Q4_K_M":    4.5,   # integer 4-bit with sub-block scales
    }

    # Compression ratios from actual ggml block format:
    # ratio = 16.0 / (block_bytes * 8 / block_elements)
    QUANT_COMPRESSION = {
        "BF16":      1.0,    # 16.0 bpw
        "Q8_0":      1.88,   # 8.5 bpw
        "Q6_K":      2.44,   # 6.5625 bpw
        "Q5_K":      2.91,   # 5.5 bpw
        "IQ4_NL":    3.56,   # 4.5 bpw
        "MXFP4_MOE": 3.76,   # 4.25 bpw — best compression of the ~4-bit schemes
        "Q4_K_M":    3.56,   # 4.5 bpw
    }

    # Relative speed multipliers (vs BF16).
    # MXFP4 is fast due to simple block format (shared exponent + nibbles).
    QUANT_SPEED = {
        "BF16":      1.0,
        "Q8_0":      1.75,
        "Q6_K":      2.2,
        "Q5_K":      2.7,
        "IQ4_NL":    3.2,
        "Q4_K_M":    3.4,
        "MXFP4_MOE": 3.8,
    }

    def __init__(
        self,
        sensitivity_weights: Dict[str, float],
        parameter_counts: Optional[Dict[str, int]] = None,
        baseline_size_gb: float = 0,
        baseline_tps: float = 0
    ):
        self.sensitivity_weights = sensitivity_weights
        self.parameter_counts = parameter_counts or {}
        self.baseline_size_gb = baseline_size_gb
        self.baseline_tps = baseline_tps

        # Learnable residual cache for active learning
        self.residual_cache: Dict[str, float] = {}

        # Collapse penalty: when multiple "brain" layers (E, H, O, R) are
        # compressed, quality degrades super-linearly.
        self.collapse_penalty_alpha = 1.5
        self.collapse_penalty_beta = 0.02

    def predict_loss(self, group_schemes: Dict[str, str]) -> float:
        """
        Predict perplexity loss for a hybrid configuration.

        Formula:
            loss = sum(sensitivity_weight[g] * noise_factor[scheme[g]])
            + collapse penalty if >= 2 brain layers compressed
        """
        total_loss = 0.0
        high_sensitivity_groups = ['E', 'H', 'O', 'R']
        compressed_sensitive_count = 0

        for group, scheme in group_schemes.items():
            sens_weight = self.sensitivity_weights.get(
                group, 1.0 / max(len(group_schemes), 1)
            )
            noise_factor = self.QUANT_NOISE_FACTORS.get(scheme, 3.0)
            total_loss += sens_weight * noise_factor

            if group in high_sensitivity_groups and scheme != "BF16":
                compressed_sensitive_count += 1

        if compressed_sensitive_count >= 2:
            collapse_penalty = (
                total_loss * self.collapse_penalty_alpha +
                self.collapse_penalty_beta
            )
            total_loss *= (1 + collapse_penalty)

        config_key = self._make_config_key(group_schemes)
        if config_key in self.residual_cache:
            total_loss += self.residual_cache[config_key]

        return total_loss

    def predict_size(self, group_schemes: Dict[str, str]) -> float:
        """Predict file size in GB for a hybrid configuration."""
        if not self.baseline_size_gb or not self.parameter_counts:
            return self._estimate_simple_size(group_schemes)

        total_weighted_bits = 0.0
        total_params = sum(self.parameter_counts.values())

        for group, scheme in group_schemes.items():
            params_in_group = self.parameter_counts.get(group, 0)
            compression = self.QUANT_COMPRESSION.get(scheme, 2.0)
            bits = 16.0 / compression
            total_weighted_bits += params_in_group * bits

        if total_params > 0:
            avg_bpw = total_weighted_bits / total_params
            return self.baseline_size_gb * (avg_bpw / 16.0)

        return self._estimate_simple_size(group_schemes)

    def predict_tps(self, group_schemes: Dict[str, str]) -> float:
        """Predict inference speed (tokens/second) for a configuration."""
        if not self.baseline_tps or not self.parameter_counts:
            return self._estimate_simple_tps(group_schemes)

        total_weighted_speed = 0.0
        total_weight = 0

        for group, scheme in group_schemes.items():
            params_in_group = self.parameter_counts.get(group, 0)
            speed_mult = self.QUANT_SPEED.get(scheme, 1.5)

            # FFN layers dominate inference time
            if group in ['U', 'D', 'X']:
                weight = params_in_group * 2
            else:
                weight = params_in_group

            total_weighted_speed += weight * speed_mult
            total_weight += weight

        if total_weight == 0:
            return self.baseline_tps

        return self.baseline_tps * min(total_weighted_speed / total_weight, 4.0)

    def _estimate_simple_tps(self, group_schemes: Dict[str, str]) -> float:
        """TPS estimation without parameter counts."""
        # Approximate parameter distribution for a dense transformer
        param_dist = {
            'E': 0.04, 'H': 0.04, 'Q': 0.12, 'K': 0.12,
            'O': 0.06, 'U': 0.31, 'D': 0.31,
        }

        total_weighted_speed = 0.0
        total_weight = 0.0

        for group, scheme in group_schemes.items():
            dist = param_dist.get(group, 0.05)
            speed_mult = self.QUANT_SPEED.get(scheme, 1.5)
            weight = dist * (2 if group in ['U', 'D', 'X'] else 1)
            total_weighted_speed += weight * speed_mult
            total_weight += weight

        if total_weight == 0:
            return self.baseline_tps if self.baseline_tps else 1.0

        avg_speed_mult = total_weighted_speed / total_weight

        if self.baseline_tps:
            return self.baseline_tps * min(avg_speed_mult, 4.0)
        return avg_speed_mult

    def _estimate_simple_size(self, group_schemes: Dict[str, str]) -> float:
        """Size estimation without parameter counts."""
        if not self.baseline_size_gb:
            return 1.0

        # Approximate parameter distribution for a dense transformer (sums to 1.0)
        param_dist = {
            'E': 0.04, 'H': 0.04, 'Q': 0.12, 'K': 0.12,
            'O': 0.06, 'U': 0.31, 'D': 0.31,
        }

        # Compute weighted average bpw, then scale baseline size
        total_weighted_bpw = 0.0
        total_dist = 0.0

        for group, scheme in group_schemes.items():
            dist = param_dist.get(group, 0.05)
            compression = self.QUANT_COMPRESSION.get(scheme, 2.0)
            bpw = 16.0 / compression
            total_weighted_bpw += dist * bpw
            total_dist += dist

        if total_dist == 0:
            return self.baseline_size_gb

        avg_bpw = total_weighted_bpw / total_dist
        # baseline_size_gb is the BF16 (16 bpw) model size
        return self.baseline_size_gb * (avg_bpw / 16.0)

    def _make_config_key(self, group_schemes: Dict[str, str]) -> str:
        return "|".join(f"{g}:{group_schemes[g]}" for g in sorted(group_schemes))

    def score_hybrid(
        self,
        group_schemes: Dict[str, str],
        precision_weight: float = 0.50,
        size_weight: float = 0.35,
        speed_weight: float = 0.15
    ) -> Dict:
        """
        Score a hybrid configuration using weighted objectives.

        Default weights prioritize quality and compression (the tool's
        primary mission) over inference speed.
        """
        predicted_loss = self.predict_loss(group_schemes)
        predicted_size = self.predict_size(group_schemes)
        predicted_tps = self.predict_tps(group_schemes)

        # Normalize loss: 0 (all BF16) to ~5 (heavy quant).
        loss_score = max(0, 1 - predicted_loss / 5.0)

        if self.baseline_size_gb > 0:
            size_score = min(1, self.baseline_size_gb / max(predicted_size, 0.01))
        else:
            size_score = min(1, 1.0 / max(predicted_size, 0.01))

        if self.baseline_tps > 0:
            tps_score = min(1, predicted_tps / self.baseline_tps)
        elif predicted_tps > 0:
            tps_score = min(1, predicted_tps / 4.0)
        else:
            tps_score = 0.0

        composite_score = (
            precision_weight * loss_score +
            size_weight * size_score +
            speed_weight * tps_score
        )

        return {
            'predicted_loss': predicted_loss,
            'predicted_size_gb': predicted_size,
            'predicted_tps': predicted_tps,
            'loss_score': loss_score,
            'size_score': size_score,
            'tps_score': tps_score,
            'composite_score': composite_score
        }

    def record_residual(self, config: Dict[str, str], residual: float):
        key = self._make_config_key(config)
        self.residual_cache[key] = residual

    def get_sensitivity_weights(self) -> Dict[str, float]:
        return self.sensitivity_weights.copy()

    def update_sensitivity_weights(self, new_weights: Dict[str, float]):
        self.sensitivity_weights = new_weights


class TierClassifier:
    """Classify hybrids into standard tiers based on size ratio to baseline."""

    TIER_BOUNDARIES = {
        "Q8": (0.95, float('inf')),
        "Q7": (0.80, 0.95),
        "Q6": (0.65, 0.80),
        "Q5": (0.50, 0.65),
        "Q4": (0.35, 0.50),
        "Q3": (0.20, 0.35),
        "Q2": (0.10, 0.20),
    }

    @staticmethod
    def classify_by_size(size_gb: float, baseline_size_gb: float) -> str:
        ratio = size_gb / baseline_size_gb
        for tier, (low, high) in TierClassifier.TIER_BOUNDARIES.items():
            if low < ratio <= high:
                return tier
        if ratio > 1.0:
            return "Q8+"
        return "Q2-"
