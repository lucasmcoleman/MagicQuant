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

from typing import Optional

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
