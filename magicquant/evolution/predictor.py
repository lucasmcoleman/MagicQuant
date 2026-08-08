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

When constructed with `imatrix_active=True` (the search/build is quantizing
with an importance matrix), noise factors for imatrix-consuming schemes
(k_quant, iq_quant) get a discount that MXFP4/rocmfpx/float schemes don't --
see `magicquant.quant.schemes.effective_noise_factor`.
"""

from typing import Dict, List, Optional

from magicquant.quant import calibration
from magicquant.quant.schemes import effective_noise_factor, get_scheme_by_name


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

    # Scheme attributes (noise_factor, compression_ratio, speed_multiplier)
    # are read from the scheme registry — see magicquant.quant.schemes.

    def __init__(
        self,
        sensitivity_weights: Dict[str, float],
        parameter_counts: Optional[Dict[str, int]] = None,
        baseline_size_gb: float = 0,
        baseline_tps: float = 0,
        imatrix_active: bool = False,
        calibration_source: str = "",
    ):
        self.sensitivity_weights = sensitivity_weights
        self.parameter_counts = parameter_counts or {}
        self.baseline_size_gb = baseline_size_gb
        self.baseline_tps = baseline_tps
        # Whether this search/build is quantizing with an importance matrix.
        # Gates the noise discount in `_noise_factor_for` (via
        # magicquant.quant.schemes.effective_noise_factor) for schemes whose
        # ggml encoder actually consumes one -- k_quant/iq_quant, not
        # MXFP4/rocmfpx/float. Off by default: preserves every caller that
        # doesn't pass this (including the seed42 regression fixture).
        self.imatrix_active = imatrix_active
        # Optional override path for empirically calibrated noise factors /
        # speed multipliers (see magicquant.quant.calibration), letting one
        # run LOAD a specific calibration file instead of the fixed
        # tools/calibration_results.json path. "" (default) preserves that
        # historical lookup exactly -- required for the seed-pinned
        # regression fixture, which builds PredictiveScorer directly.
        self.calibration_source = calibration_source

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

        for group, scheme in group_schemes.items():
            sens_weight = self.sensitivity_weights.get(
                group, 1.0 / max(len(group_schemes), 1)
            )
            noise_factor = self._noise_factor_for(scheme)
            total_loss += sens_weight * noise_factor

        # Additive collapse penalty: penalise compressing sensitive groups
        # proportionally to how many are compressed, without quadratic
        # self-amplification (old formula was total_loss *= 1+1.5*total_loss).
        compressed_sensitive = sum(
            1 for g in group_schemes
            if g in high_sensitivity_groups and group_schemes[g] != "BF16"
        )
        if compressed_sensitive > 0:
            total_loss += self.collapse_penalty_beta * compressed_sensitive

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
            compression = self._compression_for(scheme)
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
            speed_mult = self._speed_for(scheme)

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

    def _compute_param_dist(self, groups: List[str]) -> Dict[str, float]:
        """Compute parameter distribution from actual tensor group sizes.

        Falls back to a dense-transformer heuristic when parameter_counts
        are unavailable or zero.
        """
        if self.parameter_counts:
            total = sum(self.parameter_counts.get(g, 0) for g in groups)
            if total > 0:
                return {g: self.parameter_counts.get(g, 0) / total for g in groups}
        # Fallback distribution when no real parameter_counts are available.
        # If MoE groups (X/R) are present, use a MoE-leaning approximation
        # where experts hold the bulk of the weights; otherwise use the dense
        # transformer split. This only matters when parameter_counts is empty
        # (the real fix is passing actual counts from _estimate_model_size).
        is_moe = 'X' in groups or 'R' in groups
        if is_moe:
            _DEFAULT = {
                'E': 0.02, 'H': 0.02, 'Q': 0.04, 'K': 0.04,
                'O': 0.02, 'U': 0.05, 'D': 0.05, 'X': 0.70,
                'R': 0.01, 'S': 0.05,
            }
        else:
            _DEFAULT = {
                'E': 0.04, 'H': 0.04, 'Q': 0.12, 'K': 0.12,
                'O': 0.06, 'U': 0.31, 'D': 0.31, 'S': 0.10,
            }
        return {g: _DEFAULT.get(g, 0.05) for g in groups}

    def _estimate_simple_tps(self, group_schemes: Dict[str, str]) -> float:
        """TPS estimation without parameter counts."""
        param_dist = self._compute_param_dist(list(group_schemes.keys()))

        total_weighted_speed = 0.0
        total_weight = 0.0

        for group, scheme in group_schemes.items():
            dist = param_dist.get(group, 0.05)
            speed_mult = self._speed_for(scheme)
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

        param_dist = self._compute_param_dist(list(group_schemes.keys()))

        # Compute weighted average bpw, then scale baseline size
        total_weighted_bpw = 0.0
        total_dist = 0.0

        for group, scheme in group_schemes.items():
            dist = param_dist.get(group, 0.05)
            compression = self._compression_for(scheme)
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

    def _noise_factor_for(self, scheme: str) -> float:
        """Prefer an empirically calibrated noise_factor (from
        `self.calibration_source` when set, else tools/calibration_results
        .json, when present) over the static registry value; fall back to
        3.0 if the scheme is unknown to both.

        When `self.imatrix_active`, the result is routed through
        `effective_noise_factor` so schemes whose ggml encoder actually
        consumes an imatrix (k_quant/iq_quant) get the imatrix noise
        discount -- schemes that ignore it (MXFP4, rocmfpx, float) don't.
        An unknown scheme (ValueError below) has no registry entry to read
        `uses_imatrix` off, so it falls back unscaled.
        """
        calibrated = calibration.calibrated_noise_factor(
            scheme, self.calibration_source or None
        )
        try:
            scheme_obj = get_scheme_by_name(scheme)
        except ValueError:
            return calibrated if calibrated is not None else 3.0
        return effective_noise_factor(
            scheme_obj, self.imatrix_active, base_noise_factor=calibrated
        )

    @staticmethod
    def _compression_for(scheme: str) -> float:
        """Look up compression_ratio from registry; fallback to 2.0 if unknown."""
        try:
            return get_scheme_by_name(scheme).compression_ratio
        except ValueError:
            return 2.0

    def _speed_for(self, scheme: str) -> float:
        """Prefer an empirically calibrated speed_multiplier (from
        `self.calibration_source` when set, else tools/calibration_results
        .json, when present) over the static registry value; fall back to
        1.5 if the scheme is unknown to both.

        The registry value is what feeds the seed-pinned evolution fixture
        (predict_tps -> score_hybrid -> survival.py's selection), so real
        bench corrections (see the speed_multiplier comments on
        Q4_K_M/IQ4_XS/IQ4_NL/MXFP4_MOE in schemes.py) route through this
        opt-in calibration file rather than changing the registry default.
        """
        calibrated = calibration.calibrated_speed_multiplier(
            scheme, self.calibration_source or None
        )
        if calibrated is not None:
            return calibrated
        try:
            return get_scheme_by_name(scheme).speed_multiplier
        except ValueError:
            return 1.5

    # Floor used by score_hybrid's use_bytes_tps path to avoid a divide-by-
    # zero when a (degenerate) predicted_size is 0.
    _BYTES_TPS_EPS = 1e-9
    # Compression ratio (vs the BF16 baseline) that earns a full tps_score of
    # 1.0. tg is memory-bandwidth-bound, so a model N-times smaller generates
    # ~N-times faster; 4.0 (~a 4-bit quant of a 16-bit baseline) is the
    # practical ceiling. This gradient is what makes speed_weight actually
    # discriminate: EVERY quantized config is smaller than the BF16 baseline,
    # so the old min(1, baseline/predicted) saturated to 1.0 for all of them
    # and the speed term was inert (caught by live A/B, 2026-07-05).
    _BYTES_TPS_MAX_SPEEDUP = 4.0

    def score_hybrid(
        self,
        group_schemes: Dict[str, str],
        precision_weight: float = 0.50,
        size_weight: float = 0.35,
        speed_weight: float = 0.15,
        use_bytes_tps: bool = False,
    ) -> Dict:
        """
        Score a hybrid configuration using weighted objectives.

        Default weights prioritize quality and compression (the tool's
        primary mission) over inference speed.

        use_bytes_tps: when True, replace the tps_score's normally
        predict_tps-derived value (which rides the noisy per-scheme
        speed_multiplier) with a deterministic bandwidth-bound proxy --
        min(1, baseline_size_gb / predicted_size). Generation is memory-
        bandwidth-bound, so bytes-per-token IS the tg cost (measured, see
        docs) -- smaller predicted size deterministically means faster tg,
        unlike speed_multiplier. Off by default: byte-identical to the
        historical predict_tps-based scoring, required for the seed-pinned
        refactor-regression fixture.
        """
        predicted_loss = self.predict_loss(group_schemes)
        predicted_size = self.predict_size(group_schemes)
        predicted_tps = self.predict_tps(group_schemes)

        # Normalize loss: 0 (all BF16) to ~5 (heavy quant).
        loss_score = max(0, 1 - predicted_loss / 5.0)

        # size_score: smaller predicted size relative to baseline = better.
        # 1.0 - (predicted / baseline) gives 0.0 for no compression and
        # approaches 1.0 for heavy compression, discriminating between tiers.
        if self.baseline_size_gb > 0:
            size_score = max(0.0, 1.0 - predicted_size / self.baseline_size_gb)
        else:
            size_score = max(0.0, 1.0 - predicted_size)

        if use_bytes_tps:
            # Map the compression ratio to [0, 1] so it DISCRIMINATES across
            # the quantized range: baseline-sized -> 0, >=MAX_SPEEDUP-smaller
            # -> 1. (A bare min(1, baseline/predicted) saturates at 1.0 for
            # every quantized config, making the speed term useless.)
            speedup = self.baseline_size_gb / max(predicted_size, self._BYTES_TPS_EPS)
            tps_score = (speedup - 1.0) / (self._BYTES_TPS_MAX_SPEEDUP - 1.0)
            tps_score = min(1.0, max(0.0, tps_score))
        elif self.baseline_tps > 0:
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
