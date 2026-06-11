"""
Empirical noise-factor calibration for MagicQuant schemes.

For each registered scheme, this script:
  1. Builds a hybrid GGUF where every tensor group uses that scheme uniformly.
  2. Runs llama-perplexity against a calibration corpus.
  3. Records (scheme, ppl, ppl_loss = ppl - baseline_ppl).

Output: tools/calibration_results.json with per-scheme measurements.
Noise factors are normalized so Q8_0's ppl_loss = noise_factor 1.0.

Usage:
    python tools/calibrate_noise_factors.py \
        --model /path/to/Llama-3.2-1B-Instruct-bf16 \
        --corpus /path/to/wikitext-2-raw/wiki.test.raw \
        --output tools/calibration_results.json

Both --model and --corpus default to canonical reference paths.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Make magicquant importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from magicquant.quant.schemes import get_all_schemes  # noqa: E402


DEFAULT_MODEL = os.environ.get(
    "MAGICQUANT_CALIBRATION_MODEL",
    str(Path.home() / "models" / "Llama-3.2-1B-Instruct-bf16"),
)
DEFAULT_CORPUS = os.environ.get(
    "MAGICQUANT_CALIBRATION_CORPUS",
    "/home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw",
)


def _check_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"required tool '{name}' not on PATH")
    return found


def _baseline_ppl(model_path: Path, corpus: Path, perplexity_bin: str) -> float:
    """Run llama-perplexity on the unquantized BF16 model. Returns scalar ppl."""
    print(f"[baseline] computing BF16 perplexity for {model_path.name}...")
    return _run_perplexity(model_path, corpus, perplexity_bin)


def _run_perplexity(gguf_path: Path, corpus: Path, perplexity_bin: str) -> float:
    """Run llama-perplexity once and parse final perplexity from stdout."""
    cmd = [
        perplexity_bin,
        "-m", str(gguf_path),
        "-f", str(corpus),
        "--ctx-size", "512",
        "--threads", str(os.cpu_count() or 4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity failed (rc={proc.returncode}):\n"
            f"stderr: {proc.stderr[-500:]}"
        )
    # Single source of truth for the "Final estimate: PPL = ..." parsing.
    from magicquant.qat.validate import parse_perplexity
    return parse_perplexity(proc.stdout)


def _build_uniform_gguf(
    model_path: Path, scheme_name: str, output_dir: Path,
    create_hybrid_gguf,
) -> Optional[Path]:
    """Quantize the source model uniformly with `scheme_name`. Returns path or None."""
    out_path = output_dir / f"calib_{scheme_name}.gguf"
    print(f"[build] {scheme_name} → {out_path.name}")
    try:
        create_hybrid_gguf(
            output_path=str(out_path),
            base_model_path=str(model_path),
            quant_config={"base": scheme_name, "groups": {}},
            verbose=False,
        )
    except Exception as exc:
        print(f"  failed: {exc}")
        return None
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    ap.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    ap.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "calibration_results.json",
    )
    ap.add_argument(
        "--skip", nargs="*", default=["BF16"],
        help="scheme names to skip (BF16 is the baseline; default skips it)",
    )
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")

    perplexity_bin = _check_tool("llama-perplexity")

    from magicquant.gguf.writer import create_hybrid_gguf  # noqa: E402

    with tempfile.TemporaryDirectory(prefix="magicquant-calib-") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Baseline: source model is already BF16, so use it directly.
        baseline_ppl = _baseline_ppl(args.model, args.corpus, perplexity_bin)
        print(f"[baseline] PPL = {baseline_ppl:.4f}")

        results: Dict[str, Dict[str, float]] = {}

        for scheme in get_all_schemes():
            if scheme.name in args.skip:
                print(f"[skip] {scheme.name} (in --skip list)")
                continue
            gguf_path = _build_uniform_gguf(
                args.model, scheme.name, tmpdir_path, create_hybrid_gguf
            )
            if gguf_path is None:
                results[scheme.name] = {
                    "ppl": float("nan"),
                    "ppl_loss": float("nan"),
                    "noise_factor": 50.0,
                    "status": "build_failed",
                }
                continue
            try:
                ppl = _run_perplexity(gguf_path, args.corpus, perplexity_bin)
                results[scheme.name] = {
                    "ppl": ppl,
                    "ppl_loss": ppl - baseline_ppl,
                    "noise_factor": 0.0,  # filled below after Q8_0 anchor known
                    "status": "ok",
                }
                print(f"  {scheme.name}: ppl={ppl:.4f}, loss={ppl - baseline_ppl:+.4f}")
            except Exception as exc:
                print(f"  {scheme.name}: perplexity failed: {exc}")
                results[scheme.name] = {
                    "ppl": float("nan"),
                    "ppl_loss": float("nan"),
                    "noise_factor": 50.0,
                    "status": "perplexity_failed",
                }
            finally:
                if gguf_path and gguf_path.exists():
                    gguf_path.unlink()

        # Normalize: anchor Q8_0's ppl_loss → noise_factor 1.0
        anchor = results.get("Q8_0", {}).get("ppl_loss")
        if anchor is None or anchor != anchor:  # NaN check
            print("WARNING: Q8_0 anchor unavailable; falling back to absolute scale")
            anchor = 1.0
        else:
            print(f"[normalize] Q8_0 anchor: ppl_loss = {anchor:.4f}")

        for name, r in results.items():
            if r["status"] == "ok":
                r["noise_factor"] = round(max(0.0, r["ppl_loss"] / anchor), 3)
        # BF16 is the baseline by definition
        results["BF16"] = {
            "ppl": baseline_ppl, "ppl_loss": 0.0, "noise_factor": 0.0,
            "status": "baseline",
        }

        out = {
            "model": args.model.name,
            "corpus": args.corpus.name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "baseline_ppl": baseline_ppl,
            "schemes": results,
        }
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\n[write] calibration results → {args.output}")
        print("\nSummary:")
        for name, r in sorted(results.items(),
                              key=lambda kv: kv[1].get("noise_factor", 99)):
            print(f"  {name:10s}  noise={r['noise_factor']:6.3f}  "
                  f"ppl={r.get('ppl', 0):.4f}  status={r['status']}")


if __name__ == "__main__":
    main()
