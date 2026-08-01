"""Guards that separate "no signal" from "zero sensitivity".

The 2026-07 MoE failure had one shape: probes that measured nothing were
recorded as measurements of zero, normalized into a confident weight vector,
and handed to a search that then optimized noise for hours. Every assertion
here pins one link in that chain.

Numbers in this file are from the real Laguna-S/Laguna-XS runs, not invented.
"""

import logging

import pytest

from magicquant.utils.measurement import (
    MIN_TAU_SAMPLES,
    PROBE_IMPLAUSIBLE,
    PROBE_RESOLVED,
    PROBE_UNRESOLVED,
    classify_probe_signal,
    predictor_is_tracking,
    predictor_rank_correlation,
    resolution_coverage,
)

# Laguna-S per-group parameter shares, recovered by least squares over the
# run's own measured candidate sizes (fit residual 0.000%).
LAGUNA_S_MASS = {
    "X": 105_750, "O": 2_340, "Q": 2_320, "U": 690, "E": 570,
    "H": 570, "K": 560, "D": 350, "R": 30,
}

# What that run's probes actually resolved: U/K/H cleared the noise floor,
# the other six -- including X, 93.4% of the model -- did not.
LAGUNA_S_RESOLUTIONS = {
    "U": PROBE_RESOLVED, "K": PROBE_RESOLVED, "H": PROBE_RESOLVED,
    "X": PROBE_UNRESOLVED, "Q": PROBE_UNRESOLVED, "O": PROBE_UNRESOLVED,
    "D": PROBE_UNRESOLVED, "E": PROBE_UNRESOLVED, "R": PROBE_UNRESOLVED,
}


class TestClassifyProbeSignal:
    def test_signal_well_above_its_error_is_resolved(self):
        # The real X probe measured by KL: 0.154163 +/- 0.001946 (79 sigma).
        assert classify_probe_signal(0.154163, 0.001946) == PROBE_RESOLVED

    def test_same_delta_against_a_coarse_error_is_unresolved(self):
        # The same probe judged by unpaired perplexity: 0.4308 +/- 0.79.
        # Identical underlying model, but the estimator cannot see it.
        assert classify_probe_signal(0.4308, 0.79) == PROBE_UNRESOLVED

    def test_far_below_zero_is_implausible(self):
        # Laguna-XS's NaN-cascade probes: PPL 2.7 against a 34.84 baseline.
        assert classify_probe_signal(-32.1, 0.78) == PROBE_IMPLAUSIBLE

    def test_slightly_below_zero_is_unresolved_not_implausible(self):
        # Six Laguna-S probes came in just under baseline. That is a real
        # corpus-specific effect, not a broken run -- it must not be
        # escalated to IMPLAUSIBLE, which would exclude the group entirely.
        assert classify_probe_signal(-0.098, 0.78) == PROBE_UNRESOLVED

    def test_no_error_and_no_floor_refuses_to_call_anything_resolved(self):
        # Conservative on purpose: an unresolved group is merely dropped from
        # the weights; a falsely resolved one steers the whole search.
        assert classify_probe_signal(1e9, None) == PROBE_UNRESOLVED

    def test_falls_back_to_absolute_floor_when_error_is_missing(self):
        assert classify_probe_signal(2.0, None, fallback_floor=1.0) == PROBE_RESOLVED
        assert classify_probe_signal(0.5, None, fallback_floor=1.0) == PROBE_UNRESOLVED

    def test_zero_and_negative_errors_do_not_divide_by_zero(self):
        assert classify_probe_signal(1.0, 0.0) == PROBE_UNRESOLVED
        assert classify_probe_signal(1.0, -1.0) == PROBE_UNRESOLVED


class TestResolutionCoverage:
    def test_mass_weighting_exposes_what_group_counting_hides(self):
        """The whole reason this function takes parameter counts.

        Three of nine groups resolved reads as 33% coverage. Those three held
        1.6% of Laguna-S's weights; X, unresolved, held 93.4%.
        """
        by_group = resolution_coverage(LAGUNA_S_RESOLUTIONS)
        by_mass = resolution_coverage(LAGUNA_S_RESOLUTIONS, LAGUNA_S_MASS)

        assert by_group == pytest.approx(1 / 3, abs=1e-9)
        assert by_mass < 0.02
        assert by_mass < by_group / 10

    def test_resolving_the_dominant_group_is_most_of_the_coverage(self):
        resolutions = dict(LAGUNA_S_RESOLUTIONS, X=PROBE_RESOLVED)
        assert resolution_coverage(resolutions, LAGUNA_S_MASS) > 0.9

    def test_implausible_does_not_count_as_resolved(self):
        resolutions = dict(LAGUNA_S_RESOLUTIONS, X=PROBE_IMPLAUSIBLE)
        assert resolution_coverage(resolutions, LAGUNA_S_MASS) < 0.02

    def test_missing_counts_fall_back_to_group_counting(self):
        assert resolution_coverage(LAGUNA_S_RESOLUTIONS, {}) == pytest.approx(1 / 3)

    def test_empty_inputs_are_zero_not_an_error(self):
        assert resolution_coverage({}) == 0.0
        assert resolution_coverage({}, LAGUNA_S_MASS) == 0.0

    def test_counts_that_do_not_cover_the_groups_are_zero(self):
        assert resolution_coverage(LAGUNA_S_RESOLUTIONS, {"ZZZ": 1}) == 0.0


class TestPredictorTracking:
    """The output-side guard: did the predictions order anything correctly?"""

    def test_uncorrelated_predictions_are_caught(self):
        """The exact pairs Laguna-S recorded, verbatim from its
        search_results.json. Kendall tau over them is -0.0426: the ordering
        the search spent hours optimizing was fractionally worse than a coin
        flip, and nothing at the time computed this."""
        predicted = [
            4.253018, 2.966267, 2.28, 1.439624, 0.491492, 0.491492, 0.0,
            0.913953, 1.385445, 1.385445, 1.385445, 3.064993, 0.446342,
            1.033281, 0.857834, 0.446342,
        ]
        measured = [
            0.025496, 0.013969, 0.004845, 0.047605, 0.045422, 0.020864,
            0.025925, 0.244713, 0.046464, 0.051313, 0.240506, 0.21743,
            0.047818, 0.043852, 0.241443, 0.025642,
        ]
        tracking, tau = predictor_is_tracking(predicted, measured)
        assert tracking is False
        assert tau == pytest.approx(-0.0426, abs=1e-3)

    def test_a_predictor_that_orders_correctly_passes(self):
        measured = [i / 100 for i in range(MIN_TAU_SAMPLES + 4)]
        predicted = list(measured)
        tracking, tau = predictor_is_tracking(predicted, measured)
        assert tracking is True
        assert tau == pytest.approx(1.0)

    def test_constant_offset_still_counts_as_tracking(self):
        # Only the ORDER is consumed downstream, so bias is irrelevant.
        measured = [i / 100 for i in range(MIN_TAU_SAMPLES + 4)]
        predicted = [m * 10 + 5 for m in measured]
        tracking, _ = predictor_is_tracking(predicted, measured)
        assert tracking is True

    def test_too_few_pairs_is_unknown_not_broken(self):
        n = MIN_TAU_SAMPLES - 1
        tracking, tau = predictor_is_tracking(list(range(n)), list(range(n)))
        assert tracking is None and tau is None

    def test_degenerate_predictions_are_unknown_not_broken(self):
        n = MIN_TAU_SAMPLES + 4
        tracking, _ = predictor_is_tracking([1.0] * n, list(range(n)))
        assert tracking is None

    def test_mismatched_lengths_are_unknown(self):
        assert predictor_rank_correlation([1, 2, 3], [1, 2]) == (None, None)


class TestProbeArtifactVerification:
    """A probe that never quantized its group reports 'insensitive'."""

    @staticmethod
    def _prober():
        from magicquant.evolution.probing import SensitivityProber
        return SensitivityProber(
            base_model_path="/nonexistent.gguf", baseline_perplexity=10.0
        )

    @staticmethod
    def _artifact(monkeypatch, types_by_name):
        """Stub the upstream reader to return the given tensor types."""
        import gguf

        class _T:
            def __init__(self, name, type_name):
                self.name = name
                self.tensor_type = type(
                    "TT", (), {"name": type_name}
                )()

        class _R:
            def __init__(self, path):
                self.tensors = [_T(n, t) for n, t in types_by_name.items()]

        monkeypatch.setattr(gguf, "GGUFReader", _R)

    class _Classifier:
        """Everything named blk.* is group X; nothing else matters here."""
        @staticmethod
        def classify_tensor(name):
            return "X" if name.startswith("blk") else "O"

    def test_untouched_group_raises(self, monkeypatch):
        from magicquant.evolution.probing import ProbeMeasurementError

        self._artifact(monkeypatch, {
            "blk.0.ffn_up_exps.weight": "F16",
            "blk.1.ffn_up_exps.weight": "F16",
        })
        with pytest.raises(ProbeMeasurementError, match="still at full precision"):
            self._prober()._verify_probe_artifact(
                "p.gguf", "X", "Q4_K_M", "BF16", self._Classifier
            )

    def test_properly_quantized_group_passes(self, monkeypatch):
        self._artifact(monkeypatch, {
            "blk.0.ffn_up_exps.weight": "Q4_K",
            "blk.1.ffn_up_exps.weight": "Q4_K",
        })
        self._prober()._verify_probe_artifact(
            "p.gguf", "X", "Q4_K_M", "BF16", self._Classifier
        )

    def test_group_absent_from_artifact_raises(self, monkeypatch):
        from magicquant.evolution.probing import ProbeMeasurementError

        self._artifact(monkeypatch, {"output.weight": "F16"})
        with pytest.raises(ProbeMeasurementError, match="no tensors"):
            self._prober()._verify_probe_artifact(
                "p.gguf", "X", "Q4_K_M", "BF16", self._Classifier
            )

    def test_partial_quantization_warns_but_proceeds(self, monkeypatch, caplog):
        # Block-size fallbacks are real and documented; they dilute the probe
        # rather than invalidating it.
        self._artifact(monkeypatch, {
            "blk.0.ffn_up_exps.weight": "Q4_K",
            "blk.1.ffn_up_exps.weight": "F32",
        })
        with caplog.at_level(logging.WARNING):
            self._prober()._verify_probe_artifact(
                "p.gguf", "X", "Q4_K_M", "BF16", self._Classifier
            )
        assert "stayed at full precision" in caplog.text
