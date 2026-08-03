#!/usr/bin/env python3
"""Report where MagicQuant has fallen behind upstream llama.cpp.

Two things go stale whenever llama.cpp moves, and both currently need a human
to notice:

  1. **New ggml quant types.** ``ggml_facts`` derives block/type sizes from the
     installed ``gguf`` package, so new types are *readable* automatically --
     but a type MagicQuant has no QuantizationScheme for can never be chosen by
     the search. It is invisible rather than broken, which is worse.
  2. **New model architectures.** ``source.py``'s ``arch_map`` translates an HF
     ``model_type`` to a GGUF architecture, and an unmapped one hard-errors.
     That is the failure people actually hit: a model releases, and converting
     it fails until someone adds a line.

Both are checked against the ``gguf`` package -- llama.cpp's own pure-python
package and already a hard dependency -- rather than by scraping the llama.cpp
source tree, so this has no network dependency and no HTML parsing to rot.

Exit codes:
    0  in sync
    1  drift found (details on stdout, machine-readable with --json)
    2  could not check (missing dependency etc.)

Run it locally any time:  python tools/check_upstream_drift.py
CI runs it weekly; see .github/workflows/upstream-watch.yml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Architectures gguf knows about that MagicQuant deliberately does not map.
# Keep this SMALL and justified -- it is a suppression list, and anything here
# is a thing the tool will never warn about again.
ARCH_IGNORE = {
    "MMPROJ",       # vision projector side-car, not a text architecture
    "CLIP_VISION",  # ditto
}


def _known_arch_values() -> set[str]:
    """GGUF architecture strings MagicQuant's arch_map can already emit."""
    src = (REPO / "magicquant" / "gguf" / "source.py").read_text()
    m = re.search(r"arch_map\s*=\s*\{(.*?)\n    \}", src, re.S)
    if not m:
        raise RuntimeError(
            "could not locate arch_map in magicquant/gguf/source.py -- this "
            "checker greps it, so a refactor there needs a matching edit here"
        )
    return set(re.findall(r':\s*"([^"]+)"', m.group(1)))


def _known_scheme_ggml_types() -> set[str]:
    sys.path.insert(0, str(REPO))
    from magicquant.quant.schemes import get_all_schemes
    return {s.ggml_type_name for s in get_all_schemes()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--update-baseline", action="store_true",
                    help="accept the current upstream state as the new baseline")
    args = ap.parse_args()

    try:
        import gguf.constants as gc
    except ImportError:
        print("cannot check: the `gguf` package is not installed", file=sys.stderr)
        return 2

    report: dict[str, object] = {}

    # ---- 1. quant types upstream has that we cannot SELECT -----------------
    upstream_types = set(gc.GGMLQuantizationType.__members__)
    known_types = _known_scheme_ggml_types()
    # Non-quantized storage types are not search candidates; ignore them.
    passthrough = {"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"}
    unschemed = sorted(upstream_types - known_types - passthrough)
    report["new_quant_types"] = unschemed

    # ---- 2. architectures upstream has that we cannot MAP ------------------
    upstream_arch = {
        name for name in gc.MODEL_ARCH.__members__ if name not in ARCH_IGNORE
    }
    known_arch = {a.upper().replace("-", "_") for a in _known_arch_values()}
    unmapped = sorted(
        a for a in upstream_arch if a.replace("-", "_") not in known_arch
    )
    report["unmapped_architectures"] = unmapped

    try:
        from importlib.metadata import version as _v
        report["gguf_version"] = _v("gguf")
    except Exception:
        report["gguf_version"] = "unknown"

    # ---- 3. diff against the last accepted state --------------------------
    # Reporting every unmapped architecture would emit ~70 lines a week and be
    # ignored within a month. What is actionable is what appeared SINCE the
    # last look, so the baseline records the accepted state and this reports
    # only the delta.
    baseline_path = REPO / "tools" / "upstream_baseline.json"
    baseline = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
    new_types = sorted(set(unschemed) - set(baseline.get("new_quant_types", [])))
    new_arch = sorted(
        set(unmapped) - set(baseline.get("unmapped_architectures", []))
    )
    report["newly_appeared_quant_types"] = new_types
    report["newly_appeared_architectures"] = new_arch
    report["has_new"] = bool(new_types or new_arch)

    if args.update_baseline:
        baseline_path.write_text(json.dumps({
            "new_quant_types": unschemed,
            "unmapped_architectures": unmapped,
            "gguf_version": report["gguf_version"],
        }, indent=2) + "\n")
        print(f"baseline updated: {baseline_path}")
        return 0

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"gguf package version: {report['gguf_version']}")
        print(f"\nQuant types upstream defines that no scheme can select "
              f"({len(unschemed)}):")
        print("  " + (", ".join(unschemed) if unschemed else "(none -- in sync)"))
        print(f"\nArchitectures upstream defines that arch_map does not "
              f"translate ({len(unmapped)}):")
        if unmapped:
            for a in unmapped:
                print(f"  {a}")
        else:
            print("  (none -- in sync)")
        print("\nNeither list is automatically a bug: MagicQuant only needs a "
              "scheme for types worth searching, and only needs an arch entry "
              "for models you actually convert.")
        if report["has_new"]:
            print("\n=== NEW since the last accepted baseline ===")
            if new_types:
                print(f"  quant types:    {', '.join(new_types)}")
            if new_arch:
                print(f"  architectures:  {', '.join(new_arch)}")
            print("\nIf a new architecture is one you want to quantize, add it "
                  "to arch_map in magicquant/gguf/source.py. Once reviewed, run "
                  "with --update-baseline to stop reporting it.")
        else:
            print("\nNothing new since the last accepted baseline.")

    # Only NEW drift is worth failing on; the standing backlog is not news.
    return 1 if report["has_new"] else 0


if __name__ == "__main__":
    sys.exit(main())
