"""Loader for empirically-measured noise factors.

`tools/calibrate_noise_factors.py` runs an offline calibration bench
(llama.cpp + a calibration model) and writes measured (ppl, ppl_loss,
noise_factor) triples per scheme to `tools/calibration_results.json`. This
module loads that file when present so the predictor/probing code can prefer
measured noise factors over the static heuristic values in
`magicquant/quant/schemes.py`.

The file is optional: if it doesn't exist (or is unreadable/malformed), every
lookup here returns None and callers are expected to fall back to the
registry. This keeps `schemes.py` as the source of truth until calibration
has actually been run.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional

# Resolved relative to the repo root: magicquant/quant/calibration.py ->
# parents[0]=quant, [1]=magicquant, [2]=repo root.
_CALIBRATION_PATH: Path = Path(__file__).resolve().parents[2] / "tools" / "calibration_results.json"

# Module-level cache. `None` means "not loaded yet"; an (possibly empty)
# dict means "loaded" (including the case where the file was absent or
# unreadable, represented as {}).
_cache: Optional[Dict[str, dict]] = None


def _reset_cache() -> None:
    """Clear the module-level cache. For tests only."""
    global _cache
    _cache = None


def _load() -> Dict[str, dict]:
    """Load and cache the calibration results dict.

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


def calibrated_noise_factor(scheme_name: str) -> Optional[float]:
    """Return the empirically measured noise_factor for `scheme_name`.

    Returns None if no calibration file exists, the scheme isn't present in
    it, or the recorded value isn't a finite number.
    """
    data = _load()
    entry = data.get(scheme_name)
    if not isinstance(entry, dict):
        return None

    value = entry.get("noise_factor")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(value):
        return None

    return float(value)
