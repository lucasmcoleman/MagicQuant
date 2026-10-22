"""
Shared "is this measurement physically possible" guard.

A quantized model cannot have LOWER perplexity than the unquantized (or
less-quantized) baseline it's being compared against -- any real llama-
perplexity reading that comes in below baseline is measurement noise (small
corpora, few chunks) at best and a genuinely broken run (NaN cascade,
mismatched corpus) at worst, never a real quality win. Both the sensitivity
prober (magicquant/evolution/probing.py) and the measured-search orchestrator
(magicquant/orchestrator.py) need the same "how far below baseline is still
plausible noise" tolerance, so it lives here once instead of drifting apart
in two copies.

Incident context (2026-07 measured-search investigation): probing.py used to
silently clamp any negative "sensitivity" to exactly 0.0 with
``max(0.0, ppl - baseline) / baseline`` and no log line -- indistinguishable
from a genuinely flat (zero-sensitivity) group. orchestrator.py made the
identical mistake for candidate measurements (``measured_loss = (ppl -
baseline) / baseline`` with no validity check), which let a NaN-driven
``measured_loss=-0.9225`` WIN a tier via ``min()``. See CLAUDE.md / the
project's incident notes for the full root-cause chain.
"""

import hashlib
from typing import Any, Dict, Optional

from magicquant.logging import get_logger

log = get_logger(__name__)

# Default tolerance when no run-specific measurement error is available:
# ~5% of baseline PPL.
#
# The previous 2% default was TIGHTER than real 1-sigma noise observed on
# this box: a real measured-search log reported 34.8363 +/- 0.78041 at 100
# ppl_chunks, i.e. ~2.24% -- already above the old 2% "impossible" cutoff on
# its own, before even accounting for the ~sqrt(2)x wider stderr expected at
# the 50-chunk setting some runs actually use (fewer chunks -> fewer
# perplexity-window samples -> proportionally noisier mean). A cutoff this
# tight risks flagging genuine measurement jitter as "physically impossible"
# on exactly the runs this guard exists to protect. 5% keeps real noise
# comfortably inside the tolerance while still catching the NaN-cascade-
# scale violations (measured_loss ~ -0.92) that motivated this guard in the
# first place.
DEFAULT_RELATIVE_EPS = 0.05

# When a run reports its own measurement error, use a MULTIPLE of it rather
# than 1 sigma. At 1 sigma a genuinely near-lossless candidate (true loss a
# few tenths of a percent, e.g. a Q8-tier config) sits close enough to the
# boundary that ordinary jitter flags it 'physically impossible' maybe 1 run
# in 7 -- and this guard EXCLUDES flagged candidates from tier competition,
# so a false positive silently costs a legitimate winner. 3 sigma puts that
# false-positive rate under a percent while still catching the NaN-cascade
# violations (-92%) this exists for.
REPORTED_ERR_SIGMAS = 3.0


def measurement_eps(
    baseline_ppl: float,
    reported_err: Optional[float] = None,
    *,
    default_relative_eps: float = DEFAULT_RELATIVE_EPS,
) -> float:
    """Return the relative tolerance below which a probe/candidate PPL
    reading below *baseline_ppl* is still plausible noise rather than a
    physically-impossible measurement.

    Args:
        baseline_ppl: The baseline perplexity being compared against.
        reported_err: This run's own reported measurement error (e.g.
            llama-perplexity's "+/- <err>" term, as surfaced in
            ``calculate_kl_divergence``'s ``ppl_err``), when reachable.
            *None* (the common case -- most call sites only have a bare PPL
            float, not its error term) falls back to
            ``default_relative_eps``.
        default_relative_eps: Fallback tolerance (relative to
            ``baseline_ppl``) used when *reported_err* isn't available.

    Returns:
        A relative epsilon in [0, 1] (or larger, if the reported error is
        genuinely huge) -- callers compare
        ``ppl < baseline_ppl * (1 - eps)``.
    """
    if baseline_ppl <= 0:
        # Degenerate baseline -- nothing sensible to scale off; fall back to
        # the flat default rather than dividing by zero/negative.
        return default_relative_eps
    if reported_err is not None and reported_err > 0:
        return max(REPORTED_ERR_SIGMAS * reported_err / baseline_ppl,
                   default_relative_eps)
    return default_relative_eps


# ---------------------------------------------------------------------------
# Probe resolution
# ---------------------------------------------------------------------------
#
# The clamp above answers "is this reading physically impossible?". It does
# NOT answer the question that actually sank the 2026-07 MoE runs: "did this
# probe measure anything at all?"
#
# On Laguna-S every one of nine probes landed within 0.70% of baseline and six
# came in *below* it. None tripped the 5% impossibility clamp, so
# ``max(0.0, ppl - baseline)`` silently turned six of them into sensitivity
# 0.0 -- indistinguishable from "this group is genuinely robust". The three
# survivors were normalized into a confident-looking weight vector, and the
# group holding 93.4% of the model's parameters got weight exactly 0.000.
# Predicted-vs-measured rank correlation for that run was -0.043.
#
# Note what the problem is NOT. llama-perplexity is deterministic on a fixed
# model+corpus+chunk count: re-running Laguna-XS's X probe in 2026-07
# reproduced its stored reading (PPL 35.2671) to four decimal places. So the
# probe deltas are EXACT, and a reading below baseline is a real corpus-
# specific effect, not jitter -- destroying its sign discards information.
#
# The problem is ESTIMATOR PRECISION, which is a different thing from
# reproducibility. llama-perplexity's ``+/- err`` is the spread over chunks:
# it says what you would get on a *different* sample of text. At 100 chunks
# that is ~2.2% of baseline, so a 0.4% probe delta is exact but says almost
# nothing about the model's true sensitivity. The metric has too little
# dynamic range to rank groups spanning 0.3% to 93% of a model's parameters.
#
# This is the argument for probing with KL divergence instead: KL is computed
# per token against saved reference logits, so its reported error is over
# ~50k paired samples rather than 100 chunk means, and it is non-negative by
# construction -- no sign to destroy. Perplexity-mode probing keeps working
# but is necessarily conservative about what it will call resolved.
#
# "No signal" and "zero sensitivity" are different facts and must not share a
# representation. These three states keep them apart.

PROBE_RESOLVED = "resolved"        # signal is distinguishable from zero
PROBE_UNRESOLVED = "unresolved"    # inside the noise floor -- no information
PROBE_IMPLAUSIBLE = "implausible"  # below zero by more than noise -- broken
# Structurally unquantizable (every tensor in the group is forced to F32 by
# writer policy -- never-quantize-by-name, 1-D, non-32-divisible row, or an
# F32-required SSM operand -- regardless of scheme). Sensitivity is KNOWN to
# be zero, not merely unmeasured: this is a DIFFERENT fact than
# PROBE_UNRESOLVED (noise floor, no information) and must not share its
# representation, for the same reason PROBE_RESOLVED/UNRESOLVED are kept
# apart from each other. See magicquant.evolution.probing._detect_fixed_groups.
PROBE_FIXED = "fixed"

# Multiple of the reported error a probe signal must clear to count as real.
# Nine groups are probed per run, so the largest of nine pure-noise readings
# is expected around 1.5 sigma; a 2-sigma bar would promote that to "signal"
# more often than not. 3 sigma keeps the per-run false-positive rate low
# enough that "resolved" means something.
PROBE_SIGNAL_SIGMAS = 3.0


def classify_probe_signal(
    value: float,
    err: Optional[float],
    *,
    sigmas: float = PROBE_SIGNAL_SIGMAS,
    fallback_floor: Optional[float] = None,
) -> str:
    """Classify one probe reading as resolved / unresolved / implausible.

    Args:
        value: The probe's signal, oriented so MORE damage is MORE positive
            and zero means "quantizing this group changed nothing". For a KL
            probe that is the mean KL divergence; for a perplexity probe it
            is ``probe_ppl - baseline_ppl``.
        err: The reading's own reported error -- llama-perplexity prints
            ``+/- <err>`` for both its final PPL and its mean-KLD lines.
            *None* means the caller could not obtain one, and
            *fallback_floor* is used instead.
        sigmas: How many multiples of *err* the signal must clear.
        fallback_floor: Absolute threshold used when *err* is unavailable.
            With neither an error term nor a floor there is no basis on which
            to call anything resolved, so the reading is reported UNRESOLVED.
            Deliberately conservative: an unresolved group is merely excluded
            from the weight vector, whereas a falsely "resolved" one steers
            the entire search.

    Returns:
        One of ``PROBE_RESOLVED``, ``PROBE_UNRESOLVED``, ``PROBE_IMPLAUSIBLE``.
    """
    if err is not None and err > 0:
        threshold = sigmas * err
    elif fallback_floor is not None and fallback_floor > 0:
        threshold = fallback_floor
    else:
        return PROBE_UNRESOLVED

    if value > threshold:
        return PROBE_RESOLVED
    if value < -threshold:
        return PROBE_IMPLAUSIBLE
    return PROBE_UNRESOLVED


def resolution_coverage(
    resolutions: Dict[str, str],
    parameter_counts: Optional[Dict[str, int]] = None,
) -> float:
    """Fraction of the model whose sensitivity the probes actually resolved.

    Counting *groups* rather than *parameters* is what made the Laguna-S
    failure look survivable: three of nine groups resolved reads as "a third
    of the signal", but those three held 2.3% of the weights while the
    unresolved X held 93.4%. Weighting by parameter mass reports 0.023
    instead of 0.333, and the run is correctly refused.

    Args:
        resolutions: ``{group: one of the PROBE_* states}``.
        parameter_counts: ``{group: parameter count}``. When omitted or
            empty, groups are counted equally -- better than nothing, but
            that is exactly the blind spot above, so callers that can supply
            real counts should.

    Returns:
        A fraction in [0, 1]; 0.0 when there is nothing to measure.
    """
    if not resolutions:
        return 0.0

    # PROBE_FIXED groups never needed resolving -- their sensitivity is
    # KNOWN (zero), not merely unmeasured. Counting their mass in the
    # denominator would make an otherwise fully-resolved run look
    # partially blind; counting it in the numerator would credit the
    # probes with signal they never measured. Excluded from both.
    countable = {g: s for g, s in resolutions.items() if s != PROBE_FIXED}
    if not countable:
        return 0.0

    if not parameter_counts:
        resolved = sum(1 for s in countable.values() if s == PROBE_RESOLVED)
        return resolved / len(countable)

    total = sum(parameter_counts.get(g, 0) for g in countable)
    if total <= 0:
        return 0.0
    resolved = sum(
        parameter_counts.get(g, 0)
        for g, state in countable.items()
        if state == PROBE_RESOLVED
    )
    return resolved / total


# ---------------------------------------------------------------------------
# Predictor tracking
# ---------------------------------------------------------------------------
#
# Everything above checks an INPUT to the search. This checks its OUTPUT, and
# it is the guard that would have caught the 2026-07 failure no matter which
# layer was at fault.
#
# A measured search predicts each candidate's quality loss, then measures some
# of them. If the predictions carry information, predicted and measured loss
# should rank candidates similarly. On Laguna-S, Kendall tau over 115
# predicted/measured pairs was -0.043: the ranking the search optimized was
# uncorrelated with -- fractionally worse than -- reality. Nothing computed
# that number at the time, so the run reported success and shipped.
#
# Rank correlation is used rather than a fit residual because the search only
# ever consumes the ORDERING of predictions. A predictor with a large constant
# bias but the right order is fine; one with small residuals in the wrong
# order is worthless.

# Below this, predictions carry no usable ordering information. Set at 0.0
# rather than something positive because the claim being tested is minimal --
# "better than a coin flip" -- and anything at or below it means the search is
# not being steered by its own model.
MIN_USEFUL_TAU = 0.0

# Rank correlation on a handful of pairs is mostly noise; below this many the
# verdict is "unknown", never "broken".
MIN_TAU_SAMPLES = 12


def predictor_rank_correlation(
    predicted: "list[float]",
    measured: "list[float]",
) -> "tuple[Optional[float], Optional[float]]":
    """Kendall tau-b between predicted and measured loss, with its p-value.

    Args:
        predicted: Predicted quality losses.
        measured: Measured quality losses, index-aligned with *predicted*.

    Returns:
        ``(tau, p_value)``, or ``(None, None)`` when there are too few pairs
        or the input is degenerate (e.g. every prediction identical, which
        makes tau undefined rather than zero).
    """
    if len(predicted) != len(measured) or len(predicted) < MIN_TAU_SAMPLES:
        return None, None
    if len(set(predicted)) < 2 or len(set(measured)) < 2:
        return None, None

    try:
        from scipy.stats import kendalltau
    except ImportError:
        # scipy backs this one diagnostic and lives in the [dev] extra, not the
        # core deps. A measured search runs for hours; forfeiting it to a
        # missing optional import at the reporting step would be absurd.
        # (None, None) is the documented "could not compute" return that every
        # caller already handles.
        log.warning(
            "scipy is not installed, so predictor rank correlation cannot be "
            "computed. The search itself is unaffected -- only this diagnostic "
            "is skipped. Install the [dev] extra to enable it.",
            stage="measurement",
        )
        return None, None

    result = kendalltau(predicted, measured)
    tau = float(result.statistic)
    if tau != tau:  # NaN
        return None, None
    return tau, float(result.pvalue)


def predictor_is_tracking(
    predicted: "list[float]",
    measured: "list[float]",
    *,
    min_tau: float = MIN_USEFUL_TAU,
) -> "tuple[Optional[bool], Optional[float]]":
    """Is the loss predictor ordering candidates better than chance?

    Returns:
        ``(verdict, tau)``. *verdict* is True when tau exceeds *min_tau*,
        False when it does not, and *None* when there is not enough data to
        judge -- callers must distinguish "not tracking" from "unknown", since
        only the first is evidence of a broken run.
    """
    tau, _ = predictor_rank_correlation(predicted, measured)
    if tau is None:
        return None, None
    return tau > min_tau, tau


def imatrix_identity(imatrix: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Cheap content fingerprint of an imatrix, or ``{"active": False}``.

    Shared by the v1 sensitivity prober (``evolution/probing.py``, via the
    measured-search orchestrator's checkpoint-resume gate) and the v2
    distortion-table cache (``v2/sensitivity.py``) so "same imatrix" means
    the same thing in both places -- v2's cache key and v1's checkpoint
    resume gate must invalidate on identical conditions, not merely similar
    ones. Full-content hashing would read hundreds of MB per tensor, so this
    hashes tensor names plus a coarse sample of each vector instead of every
    value.

    Never raises (PR6 review F2): this feeds ``_write_measured_checkpoint``,
    which must be able to persist a checkpoint mid-run regardless of what an
    imatrix happens to contain (mirrors ``_json_safe``'s same guarantee for
    measurement values). The WHOLE per-tensor body -- name coercion, array
    conversion, and the coarse-sample slice -- is guarded, not just the
    ``np.asarray`` call, since a 0-d array (e.g. from a ``None`` value) makes
    the slice itself raise ``IndexError``, a non-``str`` key makes naive
    ``name.encode()`` raise ``AttributeError``, and mutually unorderable
    keys make a plain ``sorted(imatrix)`` raise ``TypeError`` before any
    per-tensor code even runs.

    A single malformed ENTRY (a ``None`` value, a ragged list numpy can't
    coerce, an object whose ``__array__``/``__repr__`` raises) is skipped
    rather than allowed to abort the whole fingerprint or collapse it onto a
    shared "something failed" sentinel -- either would make two otherwise-
    different imatrices compare equal. The entry's raw value is hashed via
    ``repr()`` instead of its array bytes when array conversion fails --
    coarser, but still distinguishes "this imatrix" from "a different one"
    for every real (numpy-array-valued) imatrix this ever actually sees.
    Each entry's contribution (name bytes + value bytes) is only fed into
    the running hash once BOTH halves are known good -- never partially, so
    a value that fails after its name was already about to be hashed can't
    leave a half-written entry behind; when even ``repr()`` raises, the
    whole entry (name included) is dropped from the hash.
    """
    if imatrix is None:
        return {"active": False}
    import numpy as np

    try:
        names = sorted(imatrix, key=str)
    except Exception:
        # str(key) itself failing (a pathological __str__) is exotic enough
        # that a canonical sort isn't worth chasing -- fall back to
        # whatever order the dict already iterates in. Still deterministic
        # for a given imatrix construction (dicts preserve insertion
        # order); just not a canonical sort. For every real imatrix
        # (str-keyed, from ensure_imatrix) this is identical to plain
        # sorted(imatrix) and produces the same hash as before.
        names = list(imatrix)

    h = hashlib.sha256()
    for name in names:
        try:
            name_bytes = str(name).encode()
        except Exception:
            # Can't even name this entry (a pathological __str__ on a key
            # that also broke sorting above) -- nothing safe to hash, skip
            # it entirely rather than raise.
            continue
        try:
            v = np.asarray(imatrix[name], dtype=np.float32)
            value_bytes = v[:: max(1, v.size // 16)].tobytes()
        except Exception:
            try:
                value_bytes = repr(imatrix[name]).encode()
            except Exception:
                # repr() itself raising is rare enough (a pathological
                # __repr__) that there's nothing safe left to hash for this
                # one entry -- drop it and keep going. The remaining
                # entries still differentiate this imatrix from a
                # different one.
                continue
        h.update(name_bytes)
        h.update(value_bytes)
    return {"active": True, "n_tensors": len(imatrix), "hash": h.hexdigest()[:16]}
