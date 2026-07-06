"""Loader for empirically-measured noise factors (and, opt-in, speed
multipliers).

`tools/calibrate_noise_factors.py` runs an offline calibration bench
(llama.cpp + a calibration model) and writes measured (ppl, ppl_loss,
noise_factor) triples per scheme to `tools/calibration_results.json`. This
module loads that file when present so the predictor/probing code can prefer
measured noise factors over the static heuristic values in
`magicquant/quant/schemes.py`. The same per-scheme entries may also carry a
`speed_multiplier` key (read by `calibrated_speed_multiplier`) for whenever a
real llama-bench speed calibration is merged in -- see that function's
docstring; `schemes.py`'s static `speed_multiplier` values feed the
seed-pinned evolution fixture directly (via PredictiveScorer.predict_tps),
so real-bench corrections route through this opt-in file rather than editing
the registry.

The file is optional: if it doesn't exist (or is unreadable/malformed), every
lookup here returns None and callers are expected to fall back to the
registry. This keeps `schemes.py` as the source of truth until calibration
has actually been run.

Every lookup also accepts an optional ``source_path`` (a run's
``calibration_source`` override, see ``PredictiveScorer``) to load a
SPECIFIC calibration file instead of the fixed ``_CALIBRATION_PATH`` -- e.g.
a per-run ``noise_calibration.json`` written by
``MagicQuantOrchestrator._write_noise_calibration`` (opt-in,
``write_calibration=True``). Cached separately (keyed by path) from the
default-path cache below, so pointing one run at a custom file never
clobbers what every other caller reads from the default path.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Union

# Resolved relative to the repo root: magicquant/quant/calibration.py ->
# parents[0]=quant, [1]=magicquant, [2]=repo root.
_CALIBRATION_PATH: Path = Path(__file__).resolve().parents[2] / "tools" / "calibration_results.json"

# Module-level cache. `None` means "not loaded yet"; an (possibly empty)
# dict means "loaded" (including the case where the file was absent or
# unreadable, represented as {}).
_cache: Optional[Dict[str, dict]] = None

# Per-path cache for `source_path` overrides, keyed by the (stringified)
# path. Separate from `_cache` above so pointing a run at a custom
# calibration file never interferes with the default-path lookup other
# callers still use.
_source_cache: Dict[str, Dict[str, dict]] = {}


def _reset_cache() -> None:
    """Clear both module-level caches. For tests only."""
    global _cache
    _cache = None
    _source_cache.clear()


def _load() -> Dict[str, dict]:
    """Load and cache the calibration results dict from the default
    `_CALIBRATION_PATH`.

    Returns an empty dict (and caches it) if the file is missing, unreadable,
    or malformed, so callers never need to handle exceptions.
    """
    global _cache
    if _cache is not None:
        return _cache

    try:
        raw = _CALIBRATION_PATH.read_text()
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}

    _cache = data
    return _cache


def _load_source(source_path: Union[str, Path]) -> Dict[str, dict]:
    """Load and cache the calibration results dict from an explicit
    `source_path` override, cached per-path in `_source_cache`.

    Same missing/malformed tolerance as `_load`: never raises, an
    unreadable file just yields an empty dict.
    """
    key = str(source_path)
    if key in _source_cache:
        return _source_cache[key]

    try:
        raw = Path(source_path).read_text()
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}

    _source_cache[key] = data
    return data


def _calibrated_value(
    scheme_name: str, key: str, source_path: Optional[Union[str, Path]] = None
) -> Optional[float]:
    """Shared lookup: read `key` off `scheme_name`'s entry in the calibration
    file, if present and a finite number. Backs both
    `calibrated_noise_factor` and `calibrated_speed_multiplier`.

    `source_path`, when given (non-empty), loads from that file instead of
    the default `_CALIBRATION_PATH` -- see `_load_source`.
    """
    data = _load_source(source_path) if source_path else _load()
    # The calibration tool writes a nested `{"schemes": {name: {...}}}`
    # envelope alongside run metadata (model/corpus/date/baseline_ppl); a
    # bare `{name: {...}}` dict is also accepted so hand-written fixtures
    # keep working.
    schemes = data.get("schemes", data)
    if not isinstance(schemes, dict):
        return None
    entry = schemes.get(scheme_name)
    if not isinstance(entry, dict):
        return None

    value = entry.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(value):
        return None

    return float(value)


def calibrated_noise_factor(
    scheme_name: str, source_path: Optional[Union[str, Path]] = None
) -> Optional[float]:
    """Return the empirically measured noise_factor for `scheme_name`.

    Returns None if no calibration file exists, the scheme isn't present in
    it, or the recorded value isn't a finite number. `source_path`, when
    given, overrides the default `_CALIBRATION_PATH` (see
    `PredictiveScorer.calibration_source`).
    """
    return _calibrated_value(scheme_name, "noise_factor", source_path)


def calibrated_speed_multiplier(
    scheme_name: str, source_path: Optional[Union[str, Path]] = None
) -> Optional[float]:
    """Return an empirically measured speed_multiplier for `scheme_name`,
    if the calibration file's entry for it carries one.

    `tools/calibrate_noise_factors.py` doesn't write this key today (it's a
    perplexity-only calibration run) -- this is the read side of the same
    opt-in mechanism for whenever a real llama-bench-driven speed
    calibration is run and its results are merged into the same JSON file
    (schemes.py documents the real bench-derived ratios that would go here;
    see the speed_multiplier comments on Q4_K_M/IQ4_XS/IQ4_NL/MXFP4_MOE).
    Runtime behavior is opt-in and additive: absent this key, every caller
    falls back to the static registry value exactly as before. `source_path`,
    when given, overrides the default `_CALIBRATION_PATH`.
    """
    return _calibrated_value(scheme_name, "speed_multiplier", source_path)
