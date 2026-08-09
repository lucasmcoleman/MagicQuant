"""Thin CLI shim over ``magicquant.evolution.fit_noise_factors``.

The fitting logic used to live here directly, but ``tools/`` installs as a
bare top-level package with no guaranteed presence outside a git checkout
(``pyproject.toml`` excludes ``tools*`` from packaging) -- yet
``magicquant.orchestrator._write_noise_calibration`` needed to import this
logic at runtime regardless of how ``magicquant`` was installed. 2026-08
packaging fix (F4): the real implementation moved to
``magicquant/evolution/fit_noise_factors.py`` (part of the installed
package); this file just re-exports it so `python tools/fit_noise_factors.py
...` keeps working for hand-runs from a checkout. Edit the real module, not
this one.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `magicquant` importable when running this script directly from a
# checkout without an editable/site install on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from magicquant.evolution.fit_noise_factors import (  # noqa: E402,F401
    BASELINE_SCHEME,
    HIGH_SENSITIVITY_GROUPS,
    FitInput,
    build_calibration_envelope,
    fit_noise_factors,
    load_fit_inputs,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
