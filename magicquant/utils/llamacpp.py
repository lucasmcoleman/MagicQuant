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
    
    def __init__(self, llamacpp_path: Optional[str] = None):
        """
        Initialize llama.cpp tools wrapper.
        
        Args:
            llamacpp_path: Path to llama.cpp directory (auto-detect if None)
        """
        self.llamacpp_path = llamacpp_path or self._find_llamacpp()
        self.quantize_tool = self._find_quantize_tool()
        self.perplexity_tool = self._find_perplexity_tool()
        
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
        
        for name in possible_names:
            path = os.path.join(self.llamacpp_path, name)
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(f"Could not find quantize tool in {self.llamacpp_path}")
    
    def _find_perplexity_tool(self) -> str:
        """Find the perplexity executable."""
        possible_names = ["llama-perplexity.exe", "llama-perplexity", "perplexity.exe", "perplexity"]
        
        for name in possible_names:
            path = os.path.join(self.llamacpp_path, name)
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(f"Could not find perplexity tool in {self.llamacpp_path}")
    
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
        verbose: bool = True
    ) -> Optional[float]:
        """
        Calculate perplexity for a model.
        
        Args:
            model_path: Path to GGUF model
            verbose: Print output
            
        Returns:
            Perplexity value or None if failed
        """
        cmd = [
            self.perplexity_tool,
            "-m", model_path,
            "--perplexity"
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
