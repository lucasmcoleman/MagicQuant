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
from magicquant.utils.naming import config_key as _naming_config_key

# Default coefficient for the non-linear "collapse" penalty applied when
# multiple high-sensitivity ("brain") groups are compressed simultaneously.
# Single home for this value -- tools/fit_noise_factors.py imports it rather
# than hand-copying the literal.
DEFAULT_COLLAPSE_PENALTY_BETA = 0.02


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

    # "Brain" groups whose compression triggers the collapse penalty in
    # predict_loss. Single home for this tuple -- tools/fit_noise_factors.py
    # imports it rather than hand-copying the literal. (survival.py:103's
    # _HIGH_SENSITIVITY set is a separate, deliberately-uncoupled copy used
    # for "brain" vs "attention" sampling categories; see the comment there.)
    HIGH_SENSITIVITY_GROUPS = ('E', 'H', 'O', 'R')

    def __init__(
        self,
        sensitivity_weights: Dict[str, float],
        parameter_counts: Optional[Dict[str, int]] = None,
        baseline_size_gb: float = 0,
        baseline_tps: float = 0,
        imatrix_active: bool = False,
        calibration_source: str = "",
        effective_bpw: Optional[Dict[str, Dict[str, float]]] = None,
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
        # {group: {scheme: real_bpw}} -- what each (group, scheme) pair will
        # actually cost once the writer's compat chain has had its say on THIS
        # model, supplied by the orchestrator. Only groups/schemes whose real
        # cost differs from the registry's advertised bpw need appear; see
        # _bpw_for, which falls through to the registry for everything else.
        #
        # None (default) is the pre-2026-08 behaviour exactly, which is what
        # the seed-pinned regression fixture and every direct-construction
        # caller rely on.
        #
        # Deliberately NOT paired with an effective-NOISE table. On a
        # block-32-only model a Q5_K assignment ships as Q8_0, so its true
        # noise is 1.0 while the predictor keeps a pessimistic 3.0. That is
        # directionally safe -- Q5_K then looks both large AND noisy, so it is
        # correctly abandoned in favour of Q8_0 or the Q5_0/Q5_1 pair that
        # really do hold that size band -- but it does mean predicted_loss is
        # wrong for those configs and will show up as inflated residuals in
        # the measured-search active-learning loop. Fixing that properly needs
        # a measured noise number per rewritten pair, not an inferred one.
        self.effective_bpw = effective_bpw or {}

        # Learnable residual cache for active learning. Values are ALWAYS in
        # noise units -- see the "Active learning" section below for why that
        # invariant exists and what broke when it didn't hold.
        self.residual_cache: Dict[str, float] = {}
        # config_key -> (uncorrected predicted_loss, measured_loss). The raw
        # material the measured->noise-unit scale is fitted from.
        self._measurement_pairs: Dict[str, tuple] = {}
        # Fitted slope mapping a relative-PPL measurement into noise units.
        # None = not enough signal to fit; no correction is applied.
        self._loss_scale: Optional[float] = None

        # Collapse penalty: when multiple "brain" layers (E, H, O, R) are
        # compressed, quality degrades super-linearly.
        self.collapse_penalty_alpha = 1.5
        self.collapse_penalty_beta = DEFAULT_COLLAPSE_PENALTY_BETA

    def predict_loss(self, group_schemes: Dict[str, str]) -> float:
        """
        Predict perplexity loss for a hybrid configuration.

        Formula:
            loss = sum(sensitivity_weight[g] * noise_factor[scheme[g]])
            + collapse penalty if >= 2 brain layers compressed
        """
        total_loss = 0.0
        high_sensitivity_groups = self.HIGH_SENSITIVITY_GROUPS

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

    def _predict_loss_uncorrected(self, group_schemes: Dict[str, str]) -> float:
        """predict_loss WITHOUT the active-learning correction.

        Calibration must be fitted against the model's own raw output, or the
        fit is contaminated by the correction derived from it.
        """
        key = self._make_config_key(group_schemes)
        saved = self.residual_cache.pop(key, None)
        try:
            return self.predict_loss(group_schemes)
        finally:
            if saved is not None:
                self.residual_cache[key] = saved

    def predict_size(self, group_schemes: Dict[str, str]) -> float:
        """Predict file size in GB for a hybrid configuration."""
        if not self.baseline_size_gb or not self.parameter_counts:
            return self._estimate_simple_size(group_schemes)

        total_weighted_bits = 0.0
        total_params = sum(self.parameter_counts.values())

        for group, scheme in group_schemes.items():
            params_in_group = self.parameter_counts.get(group, 0)
            bits = self._bpw_for(group, scheme)
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
            bpw = self._bpw_for(group, scheme)
            total_weighted_bpw += dist * bpw
            total_dist += dist

        if total_dist == 0:
            return self.baseline_size_gb

        avg_bpw = total_weighted_bpw / total_dist
        # baseline_size_gb is the BF16 (16 bpw) model size
        return self.baseline_size_gb * (avg_bpw / 16.0)

    def _make_config_key(self, group_schemes: Dict[str, str]) -> str:
        # Delegates to magicquant.utils.naming.config_key (search-v1/4).
        # Unlike MagicQuantOrchestrator._config_key, this instance's key
        # only feeds the in-memory self.residual_cache (written by
        # record_residual, read by predict_loss) -- there's no enforced
        # cross-module contract here, it just happens to share the same
        # format.
        return _naming_config_key(group_schemes)

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

    def _bpw_for(self, group: str, scheme: str) -> float:
        """Bits-per-weight this (group, scheme) pair will ACTUALLY cost on the
        model being searched, not what the registry advertises.

        The two diverge whenever the writer's compat chain rewrites a scheme.
        The case this exists for: on a model whose rows are 32- but not
        256-divisible, a Q5_K assignment is rewritten to Q8_0 by the
        block-size fallback, so it costs 8.5 bpw and not 5.5. Without this,
        the predictor prices Q5_K at 5.5, believes it is both smaller and
        cleaner than Q5_0 (5.5 bpw, noise 3.4), and picks it every time --
        while the actual write silently produces an 8.5 bpw tensor. The Q5_0
        entry would then exist in the registry and never once be selected, and
        the empty-tier-band problem it was added to solve would remain,
        looking like the schemes simply were not competitive.

        Falls through to the registry's own ratio -- the exact historical
        expression, `16.0 / self._compression_for(scheme)` -- whenever the
        orchestrator could not supply a table (no model open, read failed, or
        this group/scheme was not priced). Unknown means "behave exactly as
        before", never "guess".
        """
        if self.effective_bpw:
            per_group = self.effective_bpw.get(group)
            if per_group:
                bpw = per_group.get(scheme)
                if bpw is not None:
                    return bpw
        return 16.0 / self._compression_for(scheme)

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

    # ------------------------------------------------------------------
    # Active learning
    # ------------------------------------------------------------------
    #
    # THE INVARIANT: predict_loss ALWAYS returns noise units. It must not
    # silently change what it measures depending on whether a config has
    # been measured.
    #
    # It used to. `residual = measured_loss - predicted_loss` was computed in
    # the orchestrator and added straight back here, but the two sides are
    # different quantities:
    #
    #   measured_loss = (ppl - baseline_ppl) / baseline_ppl   RELATIVE FRACTION, ~0.005
    #   predict_loss  = Σ(sens_weight × noise_factor) + ...   NOISE UNITS,      ~2.0
    #
    # so `predicted + (measured - predicted)` returned the MEASUREMENT, in
    # the wrong units. Downstream, `score_hybrid`'s
    # `loss_score = 1 - predicted_loss / 5.0` mapped ~0.005 to 0.999 and
    # ~2.0 to 0.60, and `_tournament_selection` sorts every candidate in a
    # tier on the resulting composite and keeps the top 3. A measured config
    # therefore carried ~+0.20 composite (on a 0-1 scale, where the whole
    # precision term spans 0.50) purely from the unit mismatch -- it won its
    # tier almost regardless of merit, and the next generation was built from
    # its mutants. Exploration collapsed after round 1. Measured 2026-08-14.
    #
    # THE FIX: convert the measurement INTO noise units before differencing,
    # using a scale fitted from the (predicted, measured) pairs themselves --
    # a least-squares slope through the origin. Then the residual is a real
    # calibration correction in the units predict_loss speaks, and measured
    # and unmeasured configs are once again comparable.
    #
    # Deliberately conservative: with fewer than MIN_SCALE_PAIRS usable
    # pairs the scale is not fitted and NO correction is applied. An
    # unfitted scale is "no information", never a guess -- the same doctrine
    # as ggml_facts.expected_size and PredictiveScorer._bpw_for.

    MIN_SCALE_PAIRS = 2

    def record_measurement(
        self, config: Dict[str, str], measured_loss: float
    ) -> None:
        """Record a real measurement and refresh the calibration.

        Stores the (predicted, measured) pair, refits the measured->noise-unit
        scale over every pair seen, and recomputes every cached residual
        against the new scale (an updated scale changes older residuals too --
        leaving them stale would mix calibrations).
        """
        key = self._make_config_key(config)
        # The UNCORRECTED prediction: this pair calibrates the model, so it
        # must not be contaminated by the correction derived from it.
        raw_predicted = self._predict_loss_uncorrected(config)
        self._measurement_pairs[key] = (raw_predicted, measured_loss)
        self._refit_loss_scale()

    def _refit_loss_scale(self) -> None:
        """Least-squares slope through the origin over the measured pairs.

        Through the origin because both quantities are zero for a lossless
        config, so an intercept would be unphysical. Only strictly-positive
        measurements are fitted: a measurement at or below baseline is inside
        the noise floor (and may be flagged measurement_invalid upstream), so
        it carries no calibration signal even though it still RECEIVES a
        correction once a scale exists.
        """
        usable = [(p, m) for p, m in self._measurement_pairs.values() if m > 0]
        if len(usable) < self.MIN_SCALE_PAIRS:
            self._loss_scale = None
            self.residual_cache = {}
            return
        num = sum(p * m for p, m in usable)
        den = sum(m * m for _, m in usable)
        self._loss_scale = (num / den) if den > 0 else None

        self.residual_cache = {}
        if self._loss_scale is None:
            return
        for key, (raw_predicted, measured) in self._measurement_pairs.items():
            # measured * scale is the measurement expressed in noise units.
            self.residual_cache[key] = measured * self._loss_scale - raw_predicted

    def residual_for(self, config: Dict[str, str]) -> Optional[float]:
        """This config's calibrated residual in NOISE units, or None when no
        scale has been fitted yet. None is the honest answer -- a residual
        computed against an unfitted scale would be a guess."""
        return self.residual_cache.get(self._make_config_key(config))

    def record_residual(self, config: Dict[str, str], residual: float):
        """Legacy direct-residual API, retained for callers that compute the
        correction themselves and already have it in noise units. Prefer
        ``record_measurement`` -- it cannot get the units wrong."""
        key = self._make_config_key(config)
        self.residual_cache[key] = residual
