"""
Tier classification — leaf module with the canonical size-ratio boundaries.

A model's compression tier (Q8/Q6/Q5/Q4/Q3/Q2) is derived purely from the
ratio of its predicted/measured size to the BF16 baseline size. This logic
used to live on ``MagicQuantOrchestrator._classify_tier``, which forced the
leaf evolution modules (survival.py, predictor.py) to do function-local
imports of the top-level orchestrator to dodge a circular import.

Putting the pure arithmetic here — with no upward dependencies — lets every
module import ``classify_tier`` directly. ``MagicQuantOrchestrator._classify_tier``
now delegates here so a single set of boundaries is used everywhere.

── TIER_SCHEME_VERSION 2 (2026-07) — labels now mean what they say ──────────

The v1 boundaries below (preserved as ``TIER_BOUNDARIES_V1``) were SIZE
BANDS, not scheme names, and they didn't line up with the real scheme
registry: v1's "Q5" band was ``(0.33, 0.45]`` of BF16 size, but uniform
Q6_K is 6.5625 bpw = ratio 0.4102 -- inside that band. Every "Q5" file v1
ever produced was actually a uniform-Q6_K-sized artifact (18-19% larger
than a stock Q5_K_M for ~0% quality gain), and no v1 output was ever a
genuine Q5-sized artifact. v1's "Q6" band, ``(0.45, 0.65]``, didn't even
contain real Q8_0 (ratio 0.5312) -- it fell into "Q6" too, so nothing but
BF16 itself could ever earn a "Q8" label.

v2's boundaries below were derived by computing every registry scheme's
``bits_per_weight / 16.0`` ratio (see ``magicquant.quant.schemes``) and
placing each boundary at the midpoint of the gap between adjacent canonical
schemes, so every uniform build of a named scheme classifies under its own
name. Verified against the live registry (2026-07-30):

    scheme                  ratio     v1 tier   v2 tier
    BF16                    1.0000    Q8        Q8
    Q8_0                    0.5312    Q6 (!)    Q8
    ROCMFP8                 0.5156    Q6 (!)    Q8
    Q6_K                    0.4102    Q5 (!)    Q6
    ROCMFP6                 0.4062    Q5 (!)    Q6
    Q5_K                    0.3438    Q5        Q5
    Q4_1                    0.3125    Q4        Q4
    Q4_K_M / IQ4_NL / Q4_0
      / ROCMFP4             0.2812    Q4        Q4
    MXFP4_MOE / IQ4_XS      0.2656    Q4        Q4
    ROCMFP3                 0.2188    Q3        Q3
    Q3_K / IQ3_S            0.2148    Q3        Q3
    IQ3_XXS                 0.1914    Q3        Q3
    Q2_K                    0.1641    Q3 (!)    Q2
    IQ2_S                   0.1602    Q3 (!)    Q2
    IQ2_XS / IQ2_XXS
      / IQ1_M / IQ1_S       <0.16     Q2        Q2

    (!) marks where v1 misclassified a canonical scheme.

Re-verify with ``tools/`` or a REPL any time the scheme registry changes:
every ``(name, bits_per_weight/16.0)`` pair should classify under a tier
matching its own name. If a new/changed scheme lands in the wrong band,
move the nearest boundary (never widen past the neighboring scheme's ratio)
and re-run ``tests/test_tiers.py::test_registry_schemes_classify_correctly``.

``search_results.json`` is stamped with ``tier_scheme_version`` (see
``CURRENT_TIER_SCHEME_VERSION`` below) so a reader can tell which boundary
set produced its tier labels. Old (pre-version) files predate this fix and
must be read as v1: their "Q5" entries are Q6_K-sized, not Q5_K-sized.
Boundaries are NOT retroactively applied to old files -- the actual
per-group config stored under a tier key is unaffected by the label's
meaning, so old artifacts still load correctly through
``load_hybrid_config``/``_load_mq_tier_config``; only the human-facing
meaning of the tier NAME differs pre/post v2. Consumers that display or
choose based on the tier label should call ``tier_scheme_version()`` on the
loaded JSON and disclose when it's < ``CURRENT_TIER_SCHEME_VERSION``.
"""

# Tier boundaries as (lower_exclusive, upper_inclusive) on size_gb / baseline_gb.
#
# v1 (legacy, TIER_SCHEME_VERSION == 1 or absent from search_results.json).
# Kept only so old artifacts' provenance is documented -- NOT used to
# reclassify anything; see the module docstring.
TIER_BOUNDARIES_V1 = [
    ("Q6", 0.45, 0.65),
    ("Q5", 0.33, 0.45),
    ("Q4", 0.22, 0.33),
    ("Q3", 0.16, 0.22),
]

# v2 (current, TIER_SCHEME_VERSION == 2) -- boundaries placed at the
# midpoint of the gap between each pair of adjacent canonical schemes'
# bits_per_weight/16.0 ratio (see the module docstring's classification
# table). "Q8" is anything above the Q6 band's upper bound (no explicit
# upper limit -- BF16 itself is ratio 1.0); "Q2" is anything at or below
# the Q3 band's lower bound.
TIER_BOUNDARIES_V2 = [
    ("Q6", 0.375, 0.46),
    ("Q5", 0.328, 0.375),
    ("Q4", 0.242, 0.328),
    ("Q3", 0.178, 0.242),
]

# Active boundaries -- what classify_tier() uses. Bumping this to a new
# TIER_BOUNDARIES_V<n> is a TIER_SCHEME_VERSION-worthy change: update
# CURRENT_TIER_SCHEME_VERSION alongside it and re-verify the registry
# classification table in the module docstring.
TIER_BOUNDARIES = TIER_BOUNDARIES_V2

# Ratio at/below which a config falls off the bottom of TIER_BOUNDARIES into
# "Q2" (rather than "Q8", the fallback for ratios ABOVE the top of
# TIER_BOUNDARIES). Derived from TIER_BOUNDARIES's own lowest entry so the
# two can never drift apart, unlike the old code's separately hardcoded 0.16.
_Q2_CEILING = min(low for _, low, _ in TIER_BOUNDARIES)

# Tier assigned when baseline_gb is unusable (<= 0).
DEFAULT_TIER = "Q4"

# ── Versioning (search_results.json compatibility) ──────────────────────────
#
# Stamped into search_results.json's top level by
# MagicQuantOrchestrator._save_results so a reader can tell which boundary
# set produced the file's tier labels. Absence of the field (any file
# written before this stamp existed) means LEGACY_TIER_SCHEME_VERSION.
CURRENT_TIER_SCHEME_VERSION = 2
LEGACY_TIER_SCHEME_VERSION = 1


def tier_scheme_version(search_results: dict) -> int:
    """Return the ``tier_scheme_version`` a ``search_results.json`` dict was
    written under, defaulting to ``LEGACY_TIER_SCHEME_VERSION`` when the
    field is absent (any file written before this versioning existed).

    Pure read-side helper -- never mutates the boundaries actually used by
    ``classify_tier`` (which always classifies fresh data under the CURRENT
    scheme). This only tells a consumer how to *interpret* tier labels that
    are already baked into an existing file, so old artifacts still load
    (see the module docstring for why no reclassification happens).
    """
    version = search_results.get("tier_scheme_version")
    # bool is a subclass of int in Python (isinstance(True, int) is True), so
    # a stray {"tier_scheme_version": True} would otherwise pass the
    # isinstance check and be returned as-is -- guard it out explicitly
    # rather than silently treating a boolean as a real version number.
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        return LEGACY_TIER_SCHEME_VERSION
    return version


def describe_tier_band(tier: str) -> str:
    """Human-readable size-ratio band for *tier* under the ACTIVE
    (``TIER_BOUNDARIES``) scheme -- e.g. ``"(0.328, 0.375]"`` for "Q5", or
    the open-ended top/bottom bands' text for "Q8"/"Q2". Used to name the
    band in diagnostics (e.g. an explicitly requested tier that came back
    empty) so a reader doesn't have to cross-reference TIER_BOUNDARIES by
    hand.
    """
    for name, low, high in TIER_BOUNDARIES:
        if name == tier:
            return f"({low}, {high}]"
    if tier == "Q2":
        return f"(0, {_Q2_CEILING}]"
    if tier == "Q8":
        top = max(high for _, _, high in TIER_BOUNDARIES)
        return f"({top}, 1.0]"
    return "unknown tier"


def classify_tier(size_gb: float, baseline_gb: float) -> str:
    """Classify a model size into a compression tier by ratio to baseline.

    Always classifies under the CURRENT boundaries (``TIER_BOUNDARIES``) --
    this is for producing NEW labels, not for reinterpreting old ones (see
    ``tier_scheme_version`` for reading old files' existing labels).

    Args:
        size_gb: The model size in GB (predicted or measured).
        baseline_gb: The BF16 baseline size in GB.

    Returns:
        One of "Q8", "Q6", "Q5", "Q4", "Q3", "Q2".
    """
    if baseline_gb <= 0:
        return DEFAULT_TIER
    ratio = size_gb / baseline_gb
    for tier, low, high in TIER_BOUNDARIES:
        if low < ratio <= high:
            return tier
    if ratio <= _Q2_CEILING:
        return "Q2"
    return "Q8"  # ratio above the top of TIER_BOUNDARIES — barely compressed
