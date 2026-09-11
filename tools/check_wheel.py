"""Install a built wheel in isolation and check its CLI and runtime data."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


SMOKE = r'''
from importlib.metadata import distribution
from pathlib import Path
import sys

site = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(site))
import magicquant
from magicquant.imatrix import DEFAULT_CORPUS_PATH

assert Path(magicquant.__file__).resolve().is_relative_to(site), magicquant.__file__
assert DEFAULT_CORPUS_PATH.is_relative_to(site), DEFAULT_CORPUS_PATH
assert len(DEFAULT_CORPUS_PATH.read_text(encoding="utf-8")) > 10_000
entry = next(e for e in distribution("magicquant").entry_points if e.name == "magicquant")
sys.argv = ["magicquant", "--help"]
entry.load()()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="magicquant-wheel-") as directory:
        site = Path(directory) / "site"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             "--target", str(site), str(wheel)], check=True,
        )
        # Ignore PYTHONPATH and run outside the source checkout. Otherwise an
        # editable install can hide missing files in the release artifact.
        subprocess.run(
            [sys.executable, "-I", "-c", SMOKE, str(site)],
            cwd=directory, check=True,
        )
    print("Wheel CLI and bundled calibration corpus passed.")


if __name__ == "__main__":
    main()
