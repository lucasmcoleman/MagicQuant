"""
llama.cpp integration - Wrapper for calling llama.cpp quantization tools.
"""

import subprocess
import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path


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
        # Check common locations
        common_paths = [
            "C:/llama.cpp",
            "C:/Program Files/llama.cpp",
            os.path.expanduser("~/llama.cpp"),
            "/usr/local/bin",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find in PATH
        try:
            result = subprocess.run(
                ["where" if os.name == "nt" else "which", "llama-quantize"],
                capture_output=True,
                text=True,
                check=True
            )
            return os.path.dirname(result.stdout.strip())
        except subprocess.CalledProcessError:
            raise FileNotFoundError(
                "Could not find llama.cpp. Please install or provide path."
            )
    
    def _find_quantize_tool(self) -> str:
        """Find the quantize executable."""
        possible_names = ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]
        search_dirs = [
            self.llamacpp_path,
            os.path.join(self.llamacpp_path, "build", "bin"),
            os.path.join(self.llamacpp_path, "build"),
            os.path.join(self.llamacpp_path, "bin"),
        ]

        for d in search_dirs:
            for name in possible_names:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    return path

        raise FileNotFoundError(f"Could not find quantize tool in {self.llamacpp_path}")
    
    def _find_perplexity_tool(self) -> str:
        """Find the perplexity executable."""
        possible_names = ["llama-perplexity.exe", "llama-perplexity", "perplexity.exe", "perplexity"]
        search_dirs = [
            self.llamacpp_path,
            os.path.join(self.llamacpp_path, "build", "bin"),
            os.path.join(self.llamacpp_path, "build"),
            os.path.join(self.llamacpp_path, "bin"),
        ]

        for d in search_dirs:
            for name in possible_names:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    return path

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

        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

        # Search common locations relative to the llama.cpp directory
        search_paths = [
            os.path.join(self.llamacpp_path, "wikitext-2-raw", "wiki.test.raw"),
            os.path.join(self.llamacpp_path, "wikitext-2", "wiki.test.raw"),
            os.path.join(self.llamacpp_path, "models", "wikitext-2-raw", "wiki.test.raw"),
            # One level up (build dir inside llama.cpp checkout)
            os.path.join(self.llamacpp_path, "..", "wikitext-2-raw", "wiki.test.raw"),
        ]

        for p in search_paths:
            if os.path.isfile(p):
                return os.path.abspath(p)

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
        verbose: bool = True
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
            quant_type
        ]
        
        if verbose:
            print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if verbose:
                print(result.stdout)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Quantization failed: {e.stderr}")
            return False
    
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
            print(f"Calculating perplexity for {os.path.basename(model_path)}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minute timeout
            )
            
            # Parse perplexity from output — try specific patterns first,
            # then fall back to generic extraction.
            ppl = None
            for line in reversed(result.stdout.split('\n')):
                # llama.cpp "Final estimate: PPL = 5.2345 +/- 0.0123"
                m = re.search(r'Final estimate.*?PPL\s*=\s*(\d+\.?\d*)', line)
                if m:
                    ppl = float(m.group(1)); break
                # Alternative: "perplexity = 5.2345"
                m = re.search(r'[Pp]erplexity\s*[:=]\s*(\d+\.?\d*)', line)
                if m:
                    ppl = float(m.group(1)); break
                # Last resort: any line containing "PPL" with a float
                if 'PPL' in line:
                    m = re.search(r'(\d+\.\d+)', line)
                    if m:
                        ppl = float(m.group(1)); break

            if ppl is not None and verbose:
                print(f"  Perplexity: {ppl:.4f}")
            return ppl
            
        except subprocess.CalledProcessError as e:
            print(f"Perplexity calculation failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("Perplexity calculation timed out")
            return None


# Quantization type mapping from MagicQuant to llama.cpp
QUANT_TYPE_MAP = {
    "BF16": "BF16",  # Keep as-is
    "Q8_0": "Q8_0",
    "Q6_K": "Q6_K",
    "Q5_K": "Q5_K",
    "Q4_K_M": "Q4_K_M",
    "IQ4_NL": "IQ4_NL",
    "MXFP4_MOE": "MXFP4"  # MagicQuant custom type (not native llama.cpp)
}


def get_llamacpp_quant_type(magicquant_type: str) -> str:
    """Convert MagicQuant scheme name to llama.cpp type."""
    return QUANT_TYPE_MAP.get(magicquant_type, "Q4_K_M")
