"""
Sensitivity Probing - Measure tensor group response to quantization.

This module implements the "Probe Phase" described in MagicQuant, where
we measure how each tensor group responds to aggressive compression,
creating sensitivity weights that guide the evolutionary search.

Strategy:
  For each tensor group G, create a temporary GGUF where G is quantised
  with an aggressive scheme (e.g. Q4_K_M) while all other groups stay at a
  near-lossless keep scheme.  Measure each probe model and compare to
  baseline.

Why the keep scheme is Q8_0 and not BF16
----------------------------------------
It was BF16 until 2026-07-31, and that silently produced BROKEN probe
models. llama.cpp cannot run a BF16 compute graph here, so the writer
downgrades BF16-designated tensors to F16 on disk (it warns, and the warning
was treated as cosmetic). Mixing those F16 tensors with a *quantized* probed
group yields a model that does not merely measure badly -- it measures
nothing:

    probe H -> Q4_K_M, rest "BF16"   PPL = nan on every chunk
    probe H -> Q6_K,   rest "BF16"   PPL = 100352.0000  (= the vocab size,
                                     i.e. a uniform distribution)
    probe H -> Q4_K_M, rest Q8_0     PPL = 50.8959  (baseline 50.8509)

The same head quantized to Q6_K inside a fully uniform Q6_K build measures a
healthy 34.9771, so neither the head nor the scheme is at fault -- the F16
mixture is. A NaN probe exits 0 without printing "Final estimate", which is
exactly what the pre-a6f8dd0 perplexity parser turned into a fabricated
reading by matching the seconds-per-pass progress line (the 2.6-3.0 "PPLs"
against a 34.84 baseline in Laguna-XS's stored sensitivity.json).

Q8_0 is near-lossless, block-scaled, roughly half the bytes of F16, and
keeps every tensor on a path llama.cpp actually executes.
"""

from typing import Any, Dict, List, Tuple, Optional
import os
import json
import tempfile
import logging

from magicquant.quant import calibration
from magicquant.quant.schemes import get_scheme_by_name
from magicquant.gguf.writer import (
    _is_f32_required_ssm_operand,
    _NEVER_QUANTIZE_NAME_SUBSTRINGS,
)
from magicquant.utils.measurement import (
    DEFAULT_RELATIVE_EPS,
    PROBE_FIXED,
    PROBE_RESOLVED,
    PROBE_UNRESOLVED,
    classify_probe_signal,
    measurement_eps,
    resolution_coverage,
)

_log = logging.getLogger(__name__)

# Fraction of the model's parameters whose sensitivity must be resolved for
# the resulting weight vector to be worth searching on. Deliberately modest:
# the bar is "the probes said something about the bulk of the model", not
# "every group resolved". Laguna-S scored 0.016 against this and would have
# been refused; a healthy dense run resolves most of its mass.
MIN_RESOLVED_MASS = 0.50

# A probe measuring worse than this multiple of baseline is a broken build,
# not a sensitive group. Generous on purpose: the worst *legitimate* single-
# group probe observed is a few percent over baseline, while the real broken
# ones land at ~2900x (uniform logits: PPL == vocab size) or NaN.
BROKEN_PROBE_RATIO = 50.0

# Fallback resolution floor for KL-scored probes when llama.cpp's "Mean KLD"
# line carries no "+/- err" term. Real per-group readings on this hardware sit
# around 0.10-0.16 nats with errors near 0.002.
MIN_KL_SIGNAL = 0.001


def _never_quantize_match(name: str) -> Optional[str]:
    """Which llama.cpp never-quantize-by-name substring matched *name*, if
    any -- the same list ``_is_never_quantized`` checks, exposed here so a
    structurally-fixed-group log line can name the actual pattern (e.g.
    "ffn_gate_inp.weight") instead of a generic label."""
    for substr in _NEVER_QUANTIZE_NAME_SUBSTRINGS:
        if substr in name:
            return substr
    return None


# A tensor can be un-probeable without being un-optimizable, and the two
# must not be conflated. The writer's block-size fallback is gated on
# `block_size > 1` (writer.py Pass 1), and BF16 -- the registry's only
# category="float" scheme -- has block_size == 1, so it sails past that
# check and is written F16 at 2 B/elem instead of F32 at 4. Every OTHER
# reason below (SSM operand, never-quantize-by-name, 1-D) holds for every
# target type, which is what "structurally fixed" actually means.
#
# So a group that is fixed only for this reason still gets its probe
# skipped -- no quantized scheme can move it, so the probe would measure
# nothing and _verify_probe_artifact would refuse it -- but it must STAY
# in the mutable search set, because BF16 is a real, reachable, half-size
# choice for it. Dropping it would silently delete that option.
_SCHEME_DEPENDENT_FIXED_REASON = "non-32-divisible row"


def _tensor_fixed_reason(
    name: str, tensor_info: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Why is *name* unquantizable by KNOWN writer policy, if any?

    Mirrors the writer's Pass-1 compat chain closely enough to tell a
    STRUCTURALLY unquantizable tensor (never-quantize-by-name,
    F32-required SSM operand, 1-D, or a non-32-divisible row -- see
    gguf/writer.py's Pass-1 loop) from one that merely CAME BACK
    untouched from a probe. The latter is the original bug
    ``_verify_probe_artifact`` exists to catch and must keep catching --
    this function must never blur the two.

    Returns a short human-readable token, or ``None`` if *name* is a
    legitimate quantization candidate -- including when *tensor_info*
    (this tensor's shape) isn't available. Absence of proof is not proof
    of absence: an unconfirmed tensor is treated as a legitimate
    candidate, never as fixed.

    Not every reason here is scheme-invariant -- see
    ``_SCHEME_DEPENDENT_FIXED_REASON``.
    """
    if _is_f32_required_ssm_operand(name):
        return "f32-required-ssm-operand"
    substr = _never_quantize_match(name)
    if substr is not None:
        return substr
    if tensor_info is None:
        return None
    n_dims = tensor_info.get("n_dims", len(tensor_info.get("shape") or ()))
    if n_dims <= 1:
        return "1-D"
    shape = tensor_info.get("shape") or ()
    row_size = shape[-1] if shape else 1
    if row_size % 32 != 0:
        return "non-32-divisible row"
    return None


class ProbeMeasurementError(RuntimeError):
    """A sensitivity probe failed to produce a real measurement in a
    context that requires one (``SensitivityProber(strict=True)``).

    Exists to kill a silent-degradation bug class: historically a probe
    whose build/measurement failed for ANY reason fell back to
    ``_heuristic_probe`` — a FABRICATED perplexity from static per-group
    constants — and the surrounding *measured* search then ranked every
    candidate, in every round, on made-up sensitivities while still
    reporting success (provenance said "partial"; nothing failed, nothing
    was excluded). In strict mode the failure is raised loudly instead.
    Also re-exported as ``magicquant.v2.outcome.ProbeMeasurementError``.
    """


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
        strict: bool = False,
        baseline_ppl_err: Optional[float] = None,
        parameter_counts: Optional[Dict[str, int]] = None,
        kl_base_logits_path: Optional[str] = None,
        kl_corpus_path: Optional[str] = None,
        imatrix: Optional[Dict[str, Any]] = None,
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
            strict: when True, a *real* probe whose build or measurement
                fails raises :class:`ProbeMeasurementError` (after one
                measurement retry) instead of silently substituting the
                heuristic estimate. The measured search sets this — a
                multi-hour run must never silently rank candidates on
                fabricated sensitivities. Default False preserves the
                historical fallback for prediction-only use (no
                perplexity_calculator), where the heuristic is the
                documented design, not a degradation.
            baseline_ppl_err: this run's own reported error for
                ``baseline_perplexity`` (e.g. llama-perplexity's "+/- <err>"
                term), when the caller has it. Sizes the "physically
                impossible probe" clamp tolerance in ``probe_all_groups`` off
                real measurement noise instead of a flat guess. *None* (the
                common case -- ``calculate_perplexity`` only returns a bare
                float) falls back to a default in utils/measurement.py; see
                ``magicquant.utils.measurement.measurement_eps``.
            imatrix: the same per-tensor importance matrix the search's
                candidate builds and final tier generation are quantized
                with (``MagicQuantOrchestrator._imatrix``), threaded into
                every real probe's ``create_hybrid_gguf`` call so probes are
                measured under the SAME calibration state that steers them.
                *None* (default) reproduces the historical unweighted-probe
                behaviour exactly -- callers that never pass this see no
                change. See CLAUDE.md / issue #5 for why an imatrix mismatch
                between probes and candidates silently mis-ranks groups
                rather than merely mis-scaling them.
        """
        self.base_model_path = base_model_path
        self.baseline_ppl = baseline_perplexity
        self.baseline_ppl_err = baseline_ppl_err
        self.perplexity_calculator = perplexity_calculator
        self.output_dir = output_dir
        self.strict = strict
        self.imatrix = imatrix
        # Per-group parameter counts, when the caller has them. Used to judge
        # probe coverage by MODEL MASS rather than by group count -- three of
        # nine groups resolving reads as 33% coverage but was 1.6% of the
        # actual model on Laguna-S. See utils.measurement.resolution_coverage.
        self.parameter_counts: Dict[str, int] = parameter_counts or {}

        # Per-group probe resolution ("resolved"/"unresolved"/"implausible"),
        # and the mass-weighted fraction of the model actually resolved.
        # Populated by probe_all_groups.
        self.resolutions: Dict[str, str] = {}
        self.resolved_mass_fraction: float = 0.0

        # When set, probes are scored by KL divergence against these saved
        # reference logits instead of by their own perplexity. Perplexity's
        # reported error is the spread over chunk means (~2% of baseline at
        # 100 chunks), which is too coarse to separate a real sub-percent
        # probe delta from zero; KL's is over every evaluated token. Both
        # come from the same single llama-perplexity invocation.
        self.kl_base_logits_path = kl_base_logits_path
        self.kl_corpus_path = kl_corpus_path

        # Error term reported alongside the most recent probe reading, used
        # to decide whether that reading resolved anything.
        self._last_probe_err: Optional[float] = None

        # Results from probes
        self.sensitivity_results: Dict[str, float] = {}
        self.probe_models: List[Dict] = []

        # How the sensitivity weights above were obtained: "measured" (every
        # probe got a real llama-perplexity reading), "partial" (some probes
        # fell back to the heuristic estimate), "heuristic" (ALL of them did
        # -- the search then ran entirely on static empirical guesses, not
        # this model's actual behavior), or "suspect" (more than half of the
        # real measurements were physically impossible -- below baseline --
        # and got clamped to 0.0 sensitivity; see probe_all_groups). Set by
        # probe_all_groups; "unknown" until then. Threaded into
        # search_results.json by the orchestrator.
        self.probing_provenance: str = "unknown"

        # True once get_normalized_weights() has returned the uniform 1/N
        # fallback because every sensitivity was <=0 -- "no signal" is
        # otherwise indistinguishable from "genuinely flat", both in the
        # returned dict itself and (pre-fix) in this run's logs. See
        # get_normalized_weights().
        self.weights_degenerate: bool = False

        # Groups probe_all_groups determined are STRUCTURALLY unquantizable
        # on this model -- every tensor classified into the group is forced
        # to F32 by writer policy regardless of scheme (see
        # _detect_fixed_groups) -- mapped to {"reason": <str>,
        # "tensor_count": <int>}. Populated by probe_all_groups; empty until
        # then. The orchestrator reads this to drop these groups from the
        # evolutionary search's mutable-group set (no scheme choice for a
        # fixed group can ever take effect) and threads it into
        # search_results.json for downstream visibility.
        self.fixed_groups: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe_all_groups(
        self,
        groups: List[str],
        aggressive_scheme: str = "Q4_K_M",
        keep_scheme: str = "Q8_0",
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

        NOTE on the one-ahead build/measure overlap used by
        ``MagicQuantOrchestrator.run_measured_search`` (see orchestrator.py):
        it is NOT applied here. There, building the next candidate GGUF and
        measuring the current one are already two separate top-level calls
        in the loop body, so overlapping them is a small change. Here,
        ``_probe_single_group`` -> ``_real_probe`` builds, measures, AND
        cleans up its probe file all inside one method (with a ``finally``
        that deletes the probe GGUF right after measuring it) -- prefetching
        group i+1's build while group i measures would require splitting
        ``_real_probe`` into separate build/measure phases and threading a
        shared build-ahead queue through this loop, which is a real
        restructuring, not a small one. Left sequential; each probe still
        cleans up its own temporary GGUF immediately after use.
        """
        if verbose:
            print("Running Sensitivity Probes")
            print(f"Baseline PPL: {self.baseline_ppl}")
            print()

        # Some groups can be STRUCTURALLY unquantizable on this model --
        # every tensor classified into them is forced to F32 by writer
        # policy (never-quantize-by-name, 1-D, non-32-divisible row, or an
        # F32-required SSM operand) no matter what scheme this search would
        # otherwise assign. A probe of such a group is numerically identical
        # to the baseline on EVERY scheme, on every run -- the exact signal
        # _verify_probe_artifact exists to catch as a bug, except here it
        # isn't one. Detect and skip those up front rather than let each one
        # build a real probe GGUF only to have _verify_probe_artifact refuse
        # it (correctly, but for the wrong reason).
        self.fixed_groups = self._detect_fixed_groups(groups)

        # Tolerance below which a real (measured) probe PPL coming in under
        # baseline is still plausible measurement noise rather than a
        # physically impossible reading (a quantized probe cannot genuinely
        # beat the unquantized baseline). Sized off this run's own reported
        # baseline error when the caller supplied one, else a flat ~2%
        # default -- see magicquant.utils.measurement.measurement_eps.
        eps = measurement_eps(self.baseline_ppl, self.baseline_ppl_err)

        measured_count = 0
        clamped_count = 0
        for group in groups:
            fixed = self.fixed_groups.get(group)
            if fixed is not None:
                _log.info(
                    "group '%s': all %d tensors are never-quantizable "
                    "(%s) -- skipping probe, sensitivity fixed at 0.0",
                    group, fixed["tensor_count"], fixed["reason"],
                )
                if verbose:
                    print(
                        f"  Group '{group}': all {fixed['tensor_count']} "
                        f"tensors are never-quantizable ({fixed['reason']}) "
                        f"-- skipping probe, sensitivity fixed at 0.0"
                    )
                self.resolutions[group] = PROBE_FIXED
                self.sensitivity_results[group] = 0.0
                self.probe_models.append({
                    "group": group,
                    "aggressive_scheme": aggressive_scheme,
                    "probe_ppl": None,
                    "delta": None,
                    "sensitivity": 0.0,
                    "resolution": PROBE_FIXED,
                    "measured": False,
                    "clamped": False,
                    "fixed": True,
                    "fixed_reason": fixed["reason"],
                })
                continue

            ppl, measured = self._probe_single_group(
                group, aggressive_scheme, keep_scheme, verbose
            )
            if measured:
                measured_count += 1

            # A real probe reading below baseline*(1-eps) is physically
            # impossible (quantizing a group cannot IMPROVE perplexity) --
            # not just "low sensitivity". The unguarded
            # ``max(0.0, ppl - baseline)`` below already silently clamps this
            # to exactly 0.0 with no record of WHY; that 0.0 is indistin-
            # guishable from a genuinely flat/robust group, and the whole
            # entry still gets ``measured: true`` (see incident notes: a
            # NaN-driven measured search recorded 8/9 groups this way).
            # ...and it applies to PERPLEXITY readings only. In KL mode the
            # reading is a divergence, not a perplexity: it is non-negative by
            # construction, 0.0 means "identical to the reference", and there
            # is no baseline to come in under. Comparing a KLD of ~0.1 against
            # baseline_ppl*(1-eps) (~33 for this model) flags every single
            # probe as physically impossible, which then trips the
            # >half-clamped rule and downgrades a perfectly healthy run to
            # provenance "suspect". Observed exactly that: 9/9 groups resolved
            # with 100% mass coverage, and the run still failed its own gate.
            is_clamped = (
                measured
                and not self.kl_base_logits_path
                and ppl < self.baseline_ppl * (1 - eps)
            )
            if is_clamped:
                clamped_count += 1
                _log.warning(
                    "sensitivity probe for group '%s' measured PPL %.4f "
                    "below baseline %.4f (tolerance %.4f) -- a quantized "
                    "probe cannot genuinely beat its own baseline; clamping "
                    "sensitivity to 0.0 and flagging this probe as "
                    "'clamped' rather than treating it as a real zero",
                    group, ppl, self.baseline_ppl, eps,
                )
                if verbose:
                    print(
                        f"  WARNING: group '{group}' probe PPL {ppl:.4f} "
                        f"below baseline {self.baseline_ppl:.4f} -- clamped"
                    )

            # In KL mode ``ppl`` holds the probe's mean KL divergence, which
            # is already a non-negative measure of how far this group's
            # quantization moved the output distribution -- no baseline to
            # subtract and no sign to clamp.
            # A probe that came back catastrophically worse than baseline is
            # a broken model, not a sensitive group. Two real signatures:
            # NaN (which llama.cpp reports without a "Final estimate" line)
            # and PPL equal to the vocabulary size, i.e. a uniform output
            # distribution -- an H probe measured exactly 100352.0000 here
            # against a 34.84 baseline. Ranking such a reading as
            # "maximum sensitivity" would hand that group full weight and
            # steer the whole search off one corrupt build.
            if measured and not self.kl_base_logits_path:
                if ppl != ppl or ppl > self.baseline_ppl * BROKEN_PROBE_RATIO:
                    raise ProbeMeasurementError(
                        f"Sensitivity probe for group '{group}' measured PPL "
                        f"{ppl} against baseline {self.baseline_ppl} "
                        f"({BROKEN_PROBE_RATIO}x is the sanity bound). That "
                        f"is a broken probe model, not a sensitivity "
                        f"reading -- a uniform output distribution scores "
                        f"PPL == vocab size. Refusing to rank it."
                    )

            if self.kl_base_logits_path:
                delta = ppl
                sensitivity = max(0.0, ppl)
                # Fallback for builds whose "Mean KLD" line carries no
                # "+/- err" term. Without a floor here, a missing error would
                # send classify_probe_signal down its no-basis path and mark
                # EVERY group unresolved -- turning a cosmetic parsing gap
                # into a refused run. Real per-group readings on this box sit
                # around 0.10-0.16 nats with errors near 0.002, so a
                # thousandth of a nat is comfortably below any genuine signal
                # while still rejecting numerical dust.
                floor = MIN_KL_SIGNAL
            else:
                # Keep the SIGNED delta as well as the clamped sensitivity. A
                # probe reading below baseline is a real, reproducible,
                # corpus-specific effect (llama-perplexity is deterministic
                # on a fixed model+corpus), so max(0, ...) discards
                # information rather than filtering noise.
                delta = ppl - self.baseline_ppl
                sensitivity = max(0.0, delta) / self.baseline_ppl
                floor = self.baseline_ppl * DEFAULT_RELATIVE_EPS

            # Was this reading big enough to mean anything? The estimator's
            # precision decides -- NOT its reproducibility. Perplexity's
            # "+/- err" is the spread over chunk means (~2% of baseline at
            # 100 chunks), so sub-percent probe deltas are exact and still
            # uninformative; KL's is over every evaluated token.
            resolution = (
                classify_probe_signal(
                    delta, self._last_probe_err, fallback_floor=floor
                )
                if measured
                else PROBE_UNRESOLVED
            )
            self.resolutions[group] = resolution

            self.sensitivity_results[group] = sensitivity
            self.probe_models.append({
                "group": group,
                "aggressive_scheme": aggressive_scheme,
                "probe_ppl": ppl,
                "delta": delta,
                "sensitivity": sensitivity,
                "resolution": resolution,
                "measured": measured,
                "clamped": is_clamped,
            })

            if verbose:
                # Name the metric that was actually used. In KL mode this
                # number is a divergence, not a perplexity, and calling it
                # "PPL" in the log is the same class of mislabelling that
                # made this module's failure so hard to see.
                metric = "KLD" if self.kl_base_logits_path else "PPL"
                print(f"  Group '{group}': {metric}={ppl:.6f}, "
                      f"Sensitivity={sensitivity:.6f} [{resolution}]")

        # Fixed groups were never candidates for measurement -- excluded so
        # a model with one structurally-fixed group doesn't get downgraded
        # from "measured" to "partial" for correctly skipping it.
        total = len(groups) - len(self.fixed_groups)
        if measured_count == total:
            self.probing_provenance = "measured"
        elif measured_count == 0:
            self.probing_provenance = "heuristic"
            if total > 0:
                _log.warning(
                    "sensitivity probing produced ZERO real measurements -- "
                    "weights are heuristic (every probe fell back; the "
                    "evolutionary search will run on static empirical "
                    "guesses, not this model's actual quantization behavior)"
                )
                if verbose:
                    print(
                        "WARNING: sensitivity probing produced ZERO real "
                        "measurements -- weights are heuristic"
                    )
        else:
            self.probing_provenance = "partial"

        # More than half the groups' MEASURED probes came back physically
        # impossible (clamped) -- the run technically has "measured"/
        # "partial" provenance, but that provenance claims a real signal
        # this run doesn't actually have. Downgrade to "suspect" so
        # search_results.json / the orchestrator stop asserting a
        # measurement that's mostly noise-floor artifacts (the exact
        # incident: 8/9 groups clamped, provenance still said "measured").
        if total > 0 and clamped_count > total / 2:
            _log.warning(
                "sensitivity probing: %d/%d groups' measured probes were "
                "physically impossible (clamped below baseline) -- "
                "downgrading probing_provenance from %r to 'suspect'",
                clamped_count, total, self.probing_provenance,
            )
            self.probing_provenance = "suspect"

        # How much of the MODEL -- by parameter mass, not by group count --
        # did these probes actually resolve? This is the check that the
        # 2026-07 MoE runs needed and did not have. Laguna-S resolved three
        # of nine groups, which sounds like a third of the signal; those
        # three held 1.6% of the weights while X, at 93.4%, went unresolved
        # and was handed weight 0.000.
        self.resolved_mass_fraction = resolution_coverage(
            self.resolutions, self.parameter_counts
        )
        resolved_groups = [
            g for g, s in self.resolutions.items() if s == PROBE_RESOLVED
        ]
        # Only applies to runs that CLAIM a real measurement. "heuristic"
        # (every probe fell back) and "suspect" (most readings physically
        # impossible) already say something more specific about why the
        # signal is missing; overwriting them with "insufficient" would lose
        # the more useful diagnosis.
        if (
            total > 0
            and self.probing_provenance in ("measured", "partial")
            and self.resolved_mass_fraction < MIN_RESOLVED_MASS
        ):
            _log.warning(
                "sensitivity probing resolved only %.1f%% of the model by "
                "parameter mass (%d/%d groups: %s) -- below the %.0f%% "
                "needed to steer a search. The weight vector this produces "
                "is mostly zeros for the groups that hold the bytes; "
                "downgrading probing_provenance from %r to 'insufficient' "
                "so the caller can fall back to incumbents instead of "
                "optimizing an uninformative objective.%s",
                100 * self.resolved_mass_fraction, len(resolved_groups),
                total, ",".join(sorted(resolved_groups)) or "none",
                100 * MIN_RESOLVED_MASS, self.probing_provenance,
                ""
                if self.kl_base_logits_path
                else (
                    " These probes were scored by PERPLEXITY, whose reported "
                    "error is the spread over chunk means -- too coarse to "
                    "resolve a single-group probe at all. Re-run with "
                    "probe_kl=True (MagicQuantOrchestrator.run_measured_"
                    "search's default) so probes score by KL divergence "
                    "instead (~144x the resolution on a measured probe, for "
                    "one extra base-logits pass) -- if this run already had "
                    "probe_kl=True, KL base-logits capture itself failed; "
                    "check the calibration corpus and llama.cpp build."
                ),
            )
            if verbose:
                print(
                    f"WARNING: probes resolved only "
                    f"{100 * self.resolved_mass_fraction:.1f}% of the model "
                    f"by parameter mass -- provenance 'insufficient'"
                )
            self.probing_provenance = "insufficient"

        return self.sensitivity_results

    def get_normalized_weights(self) -> Dict[str, float]:
        """
        Get normalized sensitivity weights that sum to 1.0.

        When every sensitivity is <=0, the search has NO real signal to
        weight groups by -- the uniform 1/N fallback below is kept (callers
        depend on always getting a usable vector), but this is otherwise
        indistinguishable from a genuinely flat result. Logs a WARNING and
        sets ``self.weights_degenerate = True`` so a caller (e.g. the
        orchestrator, via sensitivity.json) can surface that this run's
        evolutionary search proceeded without sensitivity signal.
        """
        total = sum(max(0, s) for s in self.sensitivity_results.values())

        if total == 0:
            self.weights_degenerate = True
            if self.sensitivity_results:
                _log.warning(
                    "sensitivity probing produced a total weight of 0.0 "
                    "across all %d group(s) -- the evolutionary search will "
                    "run WITHOUT sensitivity signal (uniform weights)",
                    len(self.sensitivity_results),
                )
            return {g: 1.0 / len(self.sensitivity_results)
                    for g in self.sensitivity_results}

        self.weights_degenerate = False
        return {g: max(0, s) / total
                for g, s in self.sensitivity_results.items()}

    def save_results(self, path: str):
        """Persist sensitivity data to *path* as JSON."""
        normalized_weights = self.get_normalized_weights()
        data = {
            "baseline_ppl": self.baseline_ppl,
            "sensitivity": self.sensitivity_results,
            "normalized_weights": normalized_weights,
            "weights_degenerate": self.weights_degenerate,
            "probes": self.probe_models,
            "probing_provenance": self.probing_provenance,
            "fixed_groups": self.fixed_groups,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_fixed_groups(self, groups: List[str]) -> Dict[str, Dict[str, Any]]:
        """Which of *groups* are structurally unquantizable on this model:
        every tensor classified into the group is forced to F32 by writer
        policy (see ``_tensor_fixed_reason``) regardless of whatever scheme
        this search would otherwise assign it, so no probe of it can ever
        measure anything -- ``_verify_probe_artifact`` would refuse it on
        every scheme, on every run, forever.

        Requires a real, readable base model -- same gate
        ``_probe_single_group`` uses for whether to attempt a real probe at
        all. Returns ``{}`` (defer entirely to the normal per-group probe
        path) when there's no calculator, no real file, or the read fails
        for any reason: an inability to inspect the model is never grounds
        to call a group fixed.
        """
        if not (
            self.perplexity_calculator is not None
            and os.path.isfile(self.base_model_path)
        ):
            return {}

        from magicquant.gguf.reader import GGUFReader
        from magicquant.gguf.tensor_groups import TensorGroupClassifier

        try:
            reader = GGUFReader(self.base_model_path)
            reader.open()
            classifier = TensorGroupClassifier()
            names_by_group: Dict[str, List[str]] = {}
            for name in reader.get_tensor_names():
                names_by_group.setdefault(
                    classifier.classify_tensor(name), []
                ).append(name)
            reader.close()
        except Exception:
            # Can't inspect this model up front -- the normal probe path
            # (which opens the same reader again) will surface whatever the
            # real problem is; this pre-check just declines to guess.
            return {}

        get_tensor_info = getattr(reader, "get_tensor_info", None)
        if not callable(get_tensor_info):
            get_tensor_info = None

        fixed: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            names = names_by_group.get(group, [])
            reason = self._group_fixed_reason(names, get_tensor_info)
            if reason is not None:
                fixed[group] = {
                    "reason": reason,
                    "tensor_count": len(names),
                    # Whether this group may also be dropped from the mutable
                    # search set, or only skipped for probing -- see
                    # _reason_is_scheme_invariant.
                    "scheme_invariant": self._reason_is_scheme_invariant(reason),
                }
        return fixed

    @staticmethod
    def _group_fixed_reason(tensor_names: List[str], get_tensor_info) -> Optional[str]:
        """If every one of *tensor_names* is unquantizable for a KNOWN
        reason (see ``_tensor_fixed_reason``), return a short summary of the
        distinct reasons seen (e.g. ``"ffn_gate_inp.weight / 1-D"``). Returns
        ``None`` if the group is empty, or has ANY tensor that is -- or
        might be, if *get_tensor_info* can't confirm its shape -- a
        legitimate quantization candidate. That is the original bug
        ``_verify_probe_artifact`` exists to catch, so this must fall
        through to the normal probe path rather than guess.
        """
        if not tensor_names:
            return None
        reasons: List[str] = []
        for name in tensor_names:
            info = get_tensor_info(name) if get_tensor_info is not None else None
            reason = _tensor_fixed_reason(name, info)
            if reason is None:
                return None
            if reason not in reasons:
                reasons.append(reason)
        return " / ".join(reasons)

    @staticmethod
    def _reason_is_scheme_invariant(reason: str) -> bool:
        """Does *reason* (as returned by ``_group_fixed_reason``) hold for
        EVERY target type, or only for quantized ones?

        Only a scheme-invariant group may be dropped from the mutable
        search set -- see ``_SCHEME_DEPENDENT_FIXED_REASON``. A group fixed
        even partly for the scheme-dependent reason still skips its probe
        (nothing quantized can move it) but keeps its search slot, because
        BF16 remains a real half-size choice for it.
        """
        return _SCHEME_DEPENDENT_FIXED_REASON not in reason

    def _probe_single_group(
        self,
        group: str,
        scheme: str,
        keep_scheme: str,
        verbose: bool = True,
    ) -> Tuple[float, bool]:
        """
        Create a probe GGUF where only *group* is quantised with *scheme*
        and all others stay at *keep_scheme*, then measure perplexity.

        Falls back to heuristic estimation when no perplexity_calculator or
        no writable source model is available.

        Returns (ppl, measured) -- ``measured`` is True only for a real
        llama-perplexity reading, False for any heuristic estimate (used by
        probe_all_groups to stamp probing_provenance).
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
        return self._heuristic_probe(group, scheme), False

    def _real_probe(
        self,
        group: str,
        scheme: str,
        keep_scheme: str,
        verbose: bool,
    ) -> Tuple[float, bool]:
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
                imatrix=self.imatrix,
            )

            # A probe measures nothing unless the group it names actually
            # got requantized. Verify that against the artifact before
            # spending a perplexity pass on it -- a probe that silently kept
            # its target at BF16 reports "this group is insensitive", which
            # is the most damaging wrong answer this module can produce.
            self._verify_probe_artifact(
                probe_path, group, scheme, keep_scheme, classifier
            )

            # KL mode: compare this probe's per-token output distribution
            # against the reference logits instead of its corpus-mean
            # perplexity. Same single llama-perplexity pass, but the error
            # term is over every evaluated token rather than over chunk
            # means, so a probe delta that perplexity cannot separate from
            # zero resolves cleanly. Measured on one real probe: 79 sigma by
            # KL vs 0.55 sigma by unpaired perplexity.
            if self.kl_base_logits_path:
                kl = self.perplexity_calculator.calculate_kl_divergence(
                    probe_path, self.kl_base_logits_path, self.kl_corpus_path
                )
                if kl is None or kl.get("mean_kl") is None:
                    if self.strict:
                        raise ProbeMeasurementError(
                            f"KL probe for group '{group}' produced no "
                            "parseable KL divergence. Refusing to substitute "
                            "a fabricated value in a measured search."
                        )
                    return self._heuristic_probe(group, scheme), False
                self._last_probe_err = kl.get("mean_kl_err")
                return float(kl["mean_kl"]), True

            # Measure perplexity
            self._last_probe_err = self.baseline_ppl_err
            ppl = self.perplexity_calculator.calculate_perplexity(
                probe_path, verbose=verbose
            )

            if ppl is None and self.strict:
                # One retry: the common cause on a shared box is transient
                # GPU contention, which a second attempt often clears.
                _log.warning(
                    "PPL measurement returned no value for group '%s' — "
                    "retrying once (strict mode)", group,
                )
                ppl = self.perplexity_calculator.calculate_perplexity(
                    probe_path, verbose=verbose
                )
                if ppl is None:
                    raise ProbeMeasurementError(
                        f"Sensitivity probe for group '{group}' failed to "
                        "measure after a retry (llama-perplexity produced no "
                        "parseable PPL). Refusing to substitute a fabricated "
                        "heuristic value in a measured search — check the "
                        "llama.cpp build, corpus, and GPU availability."
                    )

            if ppl is None:
                if verbose:
                    print(f"    WARNING: PPL measurement failed for group '{group}', "
                          "falling back to heuristic")
                return self._heuristic_probe(group, scheme), False

            return ppl, True

        except ValueError as exc:
            # ValueErrors come from the writer's contract guards (pre-quantized
            # source, UNKNOWN tensor type, dtype mismatch, LoRA shape). These
            # are real build bugs / bad inputs, not transient measurement
            # failures — surface them instead of masking with a fabricated
            # heuristic PPL.
            _log.error(
                "Probe build failed for group '%s' (writer contract error) — "
                "re-raising rather than falling back to heuristic",
                group, exc_info=exc,
            )
            raise
        except ProbeMeasurementError:
            raise
        except Exception as exc:
            # Other failures (subprocess / measurement). In strict mode these
            # are raised — a measured search must never continue on fabricated
            # sensitivities (the silent-degradation bug this class kills).
            if self.strict:
                _log.error(
                    "Probe failed for group '%s' in strict mode — raising",
                    group, exc_info=exc,
                )
                raise ProbeMeasurementError(
                    f"Sensitivity probe for group '{group}' failed "
                    f"({type(exc).__name__}: {exc}). Refusing to substitute "
                    "a fabricated heuristic value in a measured search."
                ) from exc
            # Non-strict (prediction-only) keeps the historical fallback,
            # with the full traceback logged so the cause is visible.
            _log.warning(
                "Probe failed for group '%s' (%s) — using heuristic estimate",
                group, exc, exc_info=exc,
            )
            if verbose:
                print(f"    Probe failed ({exc}), using heuristic")
            return self._heuristic_probe(group, scheme), False

        finally:
            # Clean up temporary probe file
            if os.path.exists(probe_path):
                try:
                    os.remove(probe_path)
                    if verbose:
                        print(f"    Cleaned up {probe_path}")
                except OSError:
                    pass

    def _verify_probe_artifact(
        self,
        probe_path: str,
        group: str,
        scheme: str,
        keep_scheme: str,
        classifier,
    ) -> None:
        """Confirm the probe GGUF actually requantized the group it names.

        A probe's whole claim is "group G was compressed and everything else
        was not". Nothing downstream re-checks that, so any failure to apply
        the target scheme -- a tensor-name pattern that misses this
        architecture, a writer fallback, a config key that never reaches the
        encoder -- produces a probe numerically identical to the baseline.
        That reads as "group G is completely insensitive to quantization",
        which is the most damaging wrong answer this module can emit: the
        group gets weight 0.0 and the search is then free to crush it.

        The check is exact rather than size-based: read the tensor types back
        off the artifact and require that no tensor in the probed group is
        still sitting at the keep scheme's type. Block-size fallbacks to
        *other* quantized types are tolerated (the writer documents and
        reports those); staying at full precision is not.

        Raises:
            ProbeMeasurementError: if the probed group was not requantized.
                Always raised, in strict mode or not -- an unverified probe
                is worse than a missing one, because it looks like data.
        """
        from gguf import GGUFReader as _UpstreamReader

        keep_type = get_scheme_by_name(keep_scheme).ggml_type_name
        target_type = get_scheme_by_name(scheme).ggml_type_name

        try:
            reader = _UpstreamReader(probe_path)
            tensors = [
                t for t in reader.tensors
                if classifier.classify_tensor(t.name) == group
            ]
        except Exception as exc:  # unreadable artifact == unusable probe
            raise ProbeMeasurementError(
                f"Sensitivity probe for group '{group}' produced an "
                f"unreadable GGUF ({type(exc).__name__}: {exc})."
            ) from exc

        if not tensors:
            raise ProbeMeasurementError(
                f"Sensitivity probe for group '{group}' contains no tensors "
                f"classified into that group -- the probe measured nothing. "
                f"Check TensorGroupClassifier's patterns against this "
                f"architecture's tensor names."
            )

        # F32 is the writer's documented floor for 1-D and non-32-divisible
        # rows; those were never candidates for the target scheme.
        untouched = [
            t.name for t in tensors
            if t.tensor_type.name in (keep_type, "F32", "F16", "BF16")
            and t.tensor_type.name != target_type
        ]
        quantized = len(tensors) - len(untouched)

        if quantized == 0:
            raise ProbeMeasurementError(
                f"Sensitivity probe for group '{group}' requested "
                f"{scheme} ({target_type}) but all {len(tensors)} of the "
                f"group's tensors are still at full precision "
                f"({keep_type}/F32/F16). The probe is numerically identical "
                f"to the baseline, so its perplexity says nothing about "
                f"'{group}'. Refusing to report it as a measurement. "
                f"First few: {untouched[:3]}"
            )

        if untouched:
            _log.warning(
                "sensitivity probe for group '%s': %d/%d tensors stayed at "
                "full precision (expected %s). Probe signal is diluted; "
                "treat this group's sensitivity as a lower bound. First: %s",
                group, len(untouched), len(tensors), target_type,
                untouched[:3],
            )

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
        # the original heuristic's behavior pre-refactor. Prefer the
        # empirically calibrated noise_factor (tools/calibration_results.json)
        # over the static registry value when it's available.
        try:
            calibrated_noise = calibration.calibrated_noise_factor(scheme)
            registry_noise = (
                calibrated_noise if calibrated_noise is not None
                else get_scheme_by_name(scheme).noise_factor
            )
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
