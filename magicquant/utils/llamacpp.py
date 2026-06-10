"""
llama.cpp integration - Wrapper for calling llama.cpp quantization tools.
"""

import subprocess
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Default timeout for subprocess calls (seconds)
_SUBPROCESS_TIMEOUT = 600  # 10 minutes
_QUANTIZE_TIMEOUT = 1800   # 30 minutes for large model quantization


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
            "--perplexity",
        ]

        if verbose:
            print(f"Calculating perplexity for {Path(model_path).name}...")

        try:
            result = self._run_perplexity_subprocess(
                cmd, timeout=_SUBPROCESS_TIMEOUT,
            )

            # Parse perplexity from output -- try specific patterns first,
            # then fall back to generic extraction.
            ppl = _parse_perplexity_output(result.stdout)

            if ppl is not None and verbose:
                print(f"  Perplexity: {ppl:.4f}")
            return ppl

        except subprocess.CalledProcessError as e:
            print(f"Perplexity calculation failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("Perplexity calculation timed out")
            return None


def _parse_perplexity_output(stdout: str) -> Optional[float]:
    """Extract perplexity value from llama-perplexity output.

    Args:
        stdout: The stdout text from llama-perplexity.

    Returns:
        The parsed perplexity float, or None if not found.
    """
    for line in reversed(stdout.split("\n")):
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
