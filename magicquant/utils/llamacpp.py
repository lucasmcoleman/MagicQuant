"""
llama.cpp integration - Wrapper for calling llama.cpp quantization tools.
"""

import json
import subprocess
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Default timeout for subprocess calls (seconds)
_SUBPROCESS_TIMEOUT = 7200  # 2 hours (35B baseline perplexity ~67 min)
_QUANTIZE_TIMEOUT = 1800   # 30 minutes for large model quantization
_BENCH_TIMEOUT = 300       # 5 minutes for llama-bench pp/tg speed measurement
# A real saved-logits (--kl-divergence-base) file is tens of MB even for a
# tiny model/corpus; a corpus too short for the requested ctx_size*chunks
# makes llama-perplexity exit 0 but write only a ~12-byte header stub. 4 KiB
# comfortably separates "real" from "stub" without depending on model size.
_MIN_LOGITS_FILE_BYTES = 4096


class LlamaCppTools:
    """Interface to llama.cpp quantization tools."""

    def __init__(
        self,
        llamacpp_path: Optional[str] = None,
        data_file: Optional[str] = None,
        ctx_size: int = 512,
    ):
        """
        Initialize llama.cpp tools wrapper.

        Args:
            llamacpp_path: Path to llama.cpp directory (auto-detect if None)
            data_file: Path to the dataset file used for perplexity evaluation
                (e.g. wikitext-2-raw/wiki.test.raw).  When *None* the tool
                will look in common locations relative to the llama.cpp dir.
            ctx_size: Context size for perplexity evaluation (default 512
                for fast evaluation; increase for more accurate results).
        """
        self.llamacpp_path = llamacpp_path or self._find_llamacpp()
        self.quantize_tool = self._find_quantize_tool()
        self.perplexity_tool = self._find_perplexity_tool()
        self.bench_tool = _find_bench_tool(self.perplexity_tool)
        self.data_file = data_file
        self.ctx_size = ctx_size

    def _find_llamacpp(self) -> str:
        """Auto-detect llama.cpp installation."""
        common_paths = [
            Path("C:/llama.cpp"),
            Path("C:/Program Files/llama.cpp"),
            Path.home() / "llama.cpp",
            Path("/usr/local/bin"),
        ]

        for p in common_paths:
            if p.exists():
                return str(p)

        # Try to find in PATH
        which_cmd = "where" if os.name == "nt" else "which"
        try:
            result = subprocess.run(
                [which_cmd, "llama-quantize"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return str(Path(result.stdout.strip()).parent)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise FileNotFoundError(
                "Could not find llama.cpp. Please install or provide path."
            )

    def _find_quantize_tool(self) -> str:
        """Find the quantize executable."""
        possible_names = ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]
        base = Path(self.llamacpp_path)
        search_dirs = [
            base,
            base / "build" / "bin",
            base / "build",
            base / "bin",
        ]

        for d in search_dirs:
            for name in possible_names:
                candidate = d / name
                if candidate.exists():
                    return str(candidate)

        raise FileNotFoundError(f"Could not find quantize tool in {self.llamacpp_path}")

    def _find_perplexity_tool(self) -> str:
        """Find the perplexity executable."""
        possible_names = ["llama-perplexity.exe", "llama-perplexity", "perplexity.exe", "perplexity"]
        base = Path(self.llamacpp_path)
        search_dirs = [
            base,
            base / "build" / "bin",
            base / "build",
            base / "bin",
        ]

        for d in search_dirs:
            for name in possible_names:
                candidate = d / name
                if candidate.exists():
                    return str(candidate)

        raise FileNotFoundError(f"Could not find perplexity tool in {self.llamacpp_path}")

    def _resolve_data_file(self, data_file: Optional[str] = None) -> Optional[str]:
        """Resolve the dataset file for perplexity evaluation.

        Priority:
        1. Explicit *data_file* argument
        2. Instance-level ``self.data_file``
        3. Common locations relative to the llama.cpp directory

        Returns:
            Absolute path to the data file, or *None* with a printed error.
        """
        candidate = data_file or self.data_file

        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                return str(candidate_path.resolve())

        # Search common locations relative to the llama.cpp directory
        base = Path(self.llamacpp_path)
        search_paths = [
            base / "wikitext-2-raw" / "wiki.test.raw",
            base / "wikitext-2" / "wiki.test.raw",
            base / "models" / "wikitext-2-raw" / "wiki.test.raw",
            base.parent / "wikitext-2-raw" / "wiki.test.raw",
        ]

        for p in search_paths:
            if p.is_file():
                return str(p.resolve())

        # Nothing found -- print a clear message
        print(
            "ERROR: No perplexity data file found.\n"
            "  llama-perplexity requires a dataset file (e.g. wikitext-2-raw/wiki.test.raw).\n"
            "  Download it with:\n"
            "    curl -LO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip\n"
            "    unzip wikitext-2-raw-v1.zip\n"
            f"  Then place 'wikitext-2-raw/wiki.test.raw' inside {self.llamacpp_path}\n"
            "  or pass data_file=<path> to LlamaCppTools / calculate_perplexity()."
        )
        return None

    def quantize_model(
        self,
        input_path: str,
        output_path: str,
        quant_type: str,
        verbose: bool = True,
    ) -> bool:
        """
        Quantize a model using llama.cpp.

        Args:
            input_path: Source model (BF16/F16)
            output_path: Output quantized model
            quant_type: Quantization type (Q4_K_M, Q6_K, IQ4_NL, etc.)
            verbose: Print output

        Returns:
            True if successful
        """
        cmd = [
            self.quantize_tool,
            input_path,
            output_path,
            quant_type,
        ]

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=_QUANTIZE_TIMEOUT,
            )

            if verbose:
                print(result.stdout)

            return True

        except subprocess.CalledProcessError as e:
            print(f"Quantization failed: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print(f"Quantization timed out after {_QUANTIZE_TIMEOUT}s")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(subprocess.CalledProcessError),
        reraise=True,
    )
    def _run_perplexity_subprocess(
        self,
        cmd: list[str],
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """Run the perplexity subprocess with retry logic.

        Retries up to 3 times with exponential backoff on
        CalledProcessError (e.g. transient GPU OOM or file lock).
        """
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )

    def calculate_perplexity(
        self,
        model_path: str,
        verbose: bool = True,
        data_file: Optional[str] = None,
        ctx_size: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate perplexity for a model.

        Args:
            model_path: Path to GGUF model
            verbose: Print output
            data_file: Path to dataset file (overrides instance default)
            ctx_size: Context size (overrides instance default)

        Returns:
            Perplexity value or None if failed
        """
        resolved_data_file = self._resolve_data_file(data_file)
        if resolved_data_file is None:
            return None

        effective_ctx = ctx_size if ctx_size is not None else self.ctx_size

        cmd = [
            self.perplexity_tool,
            "-m", model_path,
            "-f", resolved_data_file,
            "--ctx-size", str(effective_ctx),
            "--batch-size", "512",
            "--ubatch-size", "128",
        ]

        if verbose:
            print(f"Calculating perplexity for {Path(model_path).name}...")

        try:
            result = self._run_perplexity_subprocess(
                cmd, timeout=_SUBPROCESS_TIMEOUT,
            )

            # Parse perplexity from output. llama-perplexity prints the
            # "Final estimate: PPL = ..." line to STDERR, not stdout, so scan
            # both streams (matches tools/calibrate_noise_factors.py). Parsing
            # stdout only silently returned None here — collapsing the entire
            # measured search + QAT validation to prediction-only.
            ppl = _parse_perplexity_output(
                (result.stdout or "") + "\n" + (result.stderr or "")
            )

            if ppl is not None and verbose:
                print(f"  Perplexity: {ppl:.4f}")
            return ppl

        except subprocess.CalledProcessError as e:
            print(f"Perplexity calculation failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("Perplexity calculation timed out")
            return None

    def bench(
        self,
        model_path: str,
        *,
        n_prompt: int = 32,
        n_gen: int = 32,
        reps: int = 2,
        timeout: int = _BENCH_TIMEOUT,
    ) -> Optional[dict]:
        """Measure prompt-processing and token-generation throughput.

        Runs ``llama-bench -m <model> -p <n_prompt> -n <n_gen> -r <reps> -o
        json``, which reports two rows: a prompt-processing row (n_gen == 0,
        whose avg_ts is the pp t/s) and a generation row (n_prompt == 0,
        whose avg_ts is the tg t/s). Confirmed empirically against the
        ROCmFPX llama-bench build (see tests/test_llamacpp_measure.py).

        Args:
            model_path: Path to GGUF model to benchmark.
            n_prompt: Prompt length (tokens) for the pp test.
            n_gen: Generation length (tokens) for the tg test.
            reps: Repetitions per test (-r).
            timeout: Subprocess timeout in seconds.

        Returns:
            {"pp_ts": float, "tg_ts": float} (tokens/sec), or None if
            llama-bench is unavailable or the run/parse failed.
        """
        if not self.bench_tool:
            print("llama-bench not found; skipping speed measurement")
            return None

        cmd = [
            self.bench_tool,
            "-m", model_path,
            "-p", str(n_prompt),
            "-n", str(n_gen),
            "-r", str(reps),
            "-o", "json",
        ]

        try:
            result = self._run_perplexity_subprocess(cmd, timeout=timeout)
        except subprocess.CalledProcessError as e:
            print(f"llama-bench failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("llama-bench timed out")
            return None

        parsed = _parse_bench_json(result.stdout or "")
        if parsed is None:
            print("llama-bench: could not parse pp_ts/tg_ts from JSON output")
        return parsed

    def save_base_logits(
        self,
        base_model_path: str,
        corpus_path: str,
        out_logits_path: str,
        *,
        ctx_size: int = 512,
        chunks: int = -1,
        timeout: int = _SUBPROCESS_TIMEOUT,
    ) -> bool:
        """Run the base model once, saving per-token logits to disk.

        These saved logits are the reference distribution that later
        ``calculate_kl_divergence`` calls compare quantized models against.
        Wraps ``llama-perplexity -m <base> -f <corpus>
        --kl-divergence-base <out_logits_path>``.

        Args:
            base_model_path: Path to the (typically un-quantized/BF16 or
                highest-fidelity) reference GGUF model.
            corpus_path: Path to a plain-text corpus file.
            out_logits_path: Where to write the saved logits.
            ctx_size: Context size for the pass.
            chunks: Number of context-sized chunks to process (-1 = all).
            timeout: Subprocess timeout in seconds.

        Returns:
            True if the subprocess succeeded and out_logits_path exists.
        """
        cmd = [
            self.perplexity_tool,
            "-m", base_model_path,
            "-f", corpus_path,
            "--kl-divergence-base", out_logits_path,
            "--ctx-size", str(ctx_size),
            "--chunks", str(chunks),
        ]

        try:
            self._run_perplexity_subprocess(cmd, timeout=timeout)
        except subprocess.CalledProcessError as e:
            print(f"Saving base logits failed: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("Saving base logits timed out")
            return False

        # llama-perplexity exits 0 even when it can't actually run (e.g. the
        # corpus tokenizes to fewer tokens than ctx_size*chunks requires) --
        # it still creates the output file, but as a ~12-byte header stub
        # with no real logits (empirically: a valid file is tens of MB for a
        # small model/corpus). is_file() alone can't tell success from a
        # stub, so also require a minimum size.
        out_path = Path(out_logits_path)
        return out_path.is_file() and out_path.stat().st_size > _MIN_LOGITS_FILE_BYTES

    def calculate_kl_divergence(
        self,
        quant_model_path: str,
        base_logits_path: str,
        corpus_path: str,
        *,
        ctx_size: int = 512,
        chunks: int = -1,
        timeout: int = _SUBPROCESS_TIMEOUT,
    ) -> Optional[dict]:
        """Compute KL divergence of a quantized model against saved base logits.

        Wraps ``llama-perplexity -m <quant> -f <corpus> --kl-divergence
        --kl-divergence-base <base_logits_path>`` and parses the "KL
        divergence statistics" block it prints to stdout. Label/format
        confirmed empirically (see tests/test_llamacpp_measure.py):

            Mean    KLD:  -0.000019 +/-   0.000001
            Maximum KLD:   0.000001
            90.0%   KLD:  -0.000005

        Args:
            quant_model_path: Path to the quantized GGUF model to evaluate.
            base_logits_path: Path to logits previously written by
                save_base_logits().
            corpus_path: Path to the same plain-text corpus used to save
                the base logits (chunking must match).
            ctx_size: Context size for the pass (must match the base-logits
                run).
            chunks: Number of context-sized chunks to process (-1 = all;
                must match the base-logits run).
            timeout: Subprocess timeout in seconds.

        Returns:
            {"mean_kl": float, "max_kl": float, "p90_kl": float} (the
            latter two omitted if absent from the output), or None if the
            run failed or the "Mean KLD" line couldn't be found.
        """
        cmd = [
            self.perplexity_tool,
            "-m", quant_model_path,
            "-f", corpus_path,
            "--kl-divergence",
            "--kl-divergence-base", base_logits_path,
            "--ctx-size", str(ctx_size),
            "--chunks", str(chunks),
        ]

        try:
            result = self._run_perplexity_subprocess(cmd, timeout=timeout)
        except subprocess.CalledProcessError as e:
            print(f"KL divergence calculation failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("KL divergence calculation timed out")
            return None

        parsed = _parse_kl_output((result.stdout or "") + "\n" + (result.stderr or ""))
        if parsed is None:
            print("KL divergence: could not find 'Mean KLD' in output")
        return parsed


def _parse_perplexity_output(output: str) -> Optional[float]:
    """Extract perplexity value from llama-perplexity output.

    Args:
        output: Combined stdout+stderr from llama-perplexity (the
            "Final estimate: PPL =" line is emitted on stderr).

    Returns:
        The parsed perplexity float, or None if not found.
    """
    for line in reversed(output.split("\n")):
        # llama.cpp "Final estimate: PPL = 5.2345 +/- 0.0123"
        m = re.search(r"Final estimate.*?PPL\s*=\s*(\d+\.?\d*)", line)
        if m:
            return float(m.group(1))
        # Alternative: "perplexity = 5.2345"
        m = re.search(r"[Pp]erplexity\s*[:=]\s*(\d+\.?\d*)", line)
        if m:
            return float(m.group(1))
        # Last resort: any line containing "PPL" with a float
        if "PPL" in line:
            m = re.search(r"(\d+\.\d+)", line)
            if m:
                return float(m.group(1))
    return None


def _parse_bench_json(text: str) -> Optional[dict]:
    """Extract pp_ts/tg_ts from llama-bench's ``-o json`` output.

    llama-bench (with ``-o json``) prints a JSON array with one object per
    test row. Confirmed empirically: the prompt-processing row has
    ``n_gen == 0`` (its ``avg_ts`` is the pp t/s); the generation row has
    ``n_prompt == 0`` (its ``avg_ts`` is the tg t/s).

    Args:
        text: llama-bench stdout (the JSON array; some builds may print
            extra banner/log lines around it, so the outermost ``[...]``
            is isolated before parsing).

    Returns:
        {"pp_ts": float, "tg_ts": float}, or None if the JSON can't be
        parsed or the expected rows aren't both present.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    pp_ts = None
    tg_ts = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if pp_ts is None and row.get("n_gen") == 0:
            pp_ts = row.get("avg_ts")
        if tg_ts is None and row.get("n_prompt") == 0:
            tg_ts = row.get("avg_ts")

    if pp_ts is None or tg_ts is None:
        return None

    return {"pp_ts": float(pp_ts), "tg_ts": float(tg_ts)}


def _parse_kl_output(output: str) -> Optional[dict]:
    """Extract KL-divergence statistics from llama-perplexity output.

    ``llama-perplexity --kl-divergence`` prints a "KL divergence
    statistics" block to stdout with lines like (real format, confirmed by
    running a q8_0 model against its own saved logits -- see
    tests/test_llamacpp_measure.py)::

        ====== KL divergence statistics ======
        Mean    KLD:  -0.000019 ±   0.000001
        Maximum KLD:   0.000001
        90.0%   KLD:  -0.000005

    Args:
        output: Combined stdout+stderr from llama-perplexity.

    Returns:
        {"mean_kl": float, "max_kl": float, "p90_kl": float} (max_kl/p90_kl
        omitted if not present in the output), or None if no "Mean ... KLD:"
        line is found.
    """
    result: dict = {}

    m = re.search(r"Mean\s+KLD:\s*(-?\d+\.?\d*)", output)
    if not m:
        return None
    result["mean_kl"] = float(m.group(1))

    m = re.search(r"Maximum\s+KLD:\s*(-?\d+\.?\d*)", output)
    if m:
        result["max_kl"] = float(m.group(1))

    m = re.search(r"90\.0%\s+KLD:\s*(-?\d+\.?\d*)", output)
    if m:
        result["p90_kl"] = float(m.group(1))

    return result


def _find_bench_tool(perplexity_tool_path: str) -> Optional[str]:
    """Locate the llama-bench executable next to the resolved perplexity tool.

    Mirrors LlamaCppTools._find_perplexity_tool, but returns None instead of
    raising when the binary is absent -- bench() must degrade gracefully
    (return None) rather than prevent LlamaCppTools from being constructed.
    """
    possible_names = ["llama-bench.exe", "llama-bench"]
    base = Path(perplexity_tool_path).parent

    for name in possible_names:
        candidate = base / name
        if candidate.exists():
            return str(candidate)

    # Fall back to PATH
    which_cmd = "where" if os.name == "nt" else "which"
    try:
        result = subprocess.run(
            [which_cmd, "llama-bench"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        found = result.stdout.strip().splitlines()
        return found[0] if found else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


# Quantization type mapping from MagicQuant to llama.cpp
QUANT_TYPE_MAP: Dict[str, str] = {
    "BF16": "BF16",  # Keep as-is
    "Q8_0": "Q8_0",
    "Q6_K": "Q6_K",
    "Q5_K": "Q5_K",
    "Q4_K_M": "Q4_K_M",
    "IQ4_NL": "IQ4_NL",
    "MXFP4_MOE": "MXFP4",  # native ggml type 39 (GGML_TYPE_MXFP4)
}


def get_llamacpp_quant_type(magicquant_type: str) -> str:
    """Convert MagicQuant scheme name to llama.cpp type."""
    return QUANT_TYPE_MAP.get(magicquant_type, "Q4_K_M")
