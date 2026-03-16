# MagicQuant Hybrid Implementation Plan (Option 2)
## For Coding Agent Execution

---

## 🎯 **Objective**
Integrate MagicQuant's evolutionary search with llama.cpp's quantization tools to generate optimal MXFP4 hybrid quantizations for Qwen3-Coder-30B-A3B-Instruct.

---

## 📋 **Prerequisites to Verify**

Before starting implementation, verify these exist:

1. **llama.cpp installed and built** with quantize tool
   - Location should be discoverable or configured
   - Test: `llama-quantize --help` or `./quantize --help`

2. **Source model available**
   - Path: `F:\models\Qwen3-Coder-30B-A3B-Instruct-GGUF`
   - Should contain a BF16 or F16 GGUF file

3. **Python environment** with dependencies:
   - numpy, pyyaml (already in requirements.txt)

---

## 🔧 **Implementation Tasks**

### **TASK 1: Create llama.cpp Wrapper Module**
**File to create:** `magicquant/utils/llamacpp.py`

**Purpose:** Wrapper to call llama.cpp tools from Python

**Code to write:**

```python
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
            
            # Parse perplexity from output
            # Look for line like "Final estimate: PPL = 5.2345"
            for line in result.stdout.split('\n'):
                if 'PPL' in line or 'perplexity' in line.lower():
                    match = re.search(r'(\d+\.\d+)', line)
                    if match:
                        ppl = float(match.group(1))
                        if verbose:
                            print(f"  Perplexity: {ppl:.4f}")
                        return ppl
            
            return None
            
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
    "MXFP4_MOE": "Q4_0"  # Map to closest llama.cpp equivalent
}


def get_llamacpp_quant_type(magicquant_type: str) -> str:
    """Convert MagicQuant scheme name to llama.cpp type."""
    return QUANT_TYPE_MAP.get(magicquant_type, "Q4_K_M")
```

**Verification:**
- File exists at `magicquant/utils/llamacpp.py`
- Can be imported: `from magicquant.utils.llamacpp import LlamaCppTools`

---

### **TASK 2: Create Orchestrator Script**
**File to create:** `magicquant/orchestrator.py`

**Purpose:** Main script that runs evolutionary search and coordinates with llama.cpp

**Code to write:**

```python
"""
MagicQuant Orchestrator - Coordinates evolutionary search with llama.cpp.

This script:
1. Runs evolutionary search to find optimal hybrid configurations
2. Uses llama.cpp to generate the actual quantized models
3. Validates results with perplexity measurements
"""

import os
import time
from typing import Dict, List, Optional
from pathlib import Path

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.evolution.probing import SensitivityProber
from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.utils.naming import generate_name
from magicquant.utils.llamacpp import LlamaCppTools, get_llamacpp_quant_type


class MagicQuantOrchestrator:
    """Orchestrate MagicQuant search with llama.cpp execution."""
    
    def __init__(
        self,
        source_model_path: str,
        output_dir: str,
        llamacpp_path: Optional[str] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            source_model_path: Path to source GGUF (BF16/F16)
            output_dir: Directory for output models
            llamacpp_path: Path to llama.cpp (auto-detect if None)
        """
        self.source_model_path = source_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize llama.cpp tools
        self.llama_tools = LlamaCppTools(llamacpp_path)
        
        # Will be populated during run
        self.baseline_ppl: Optional[float] = None
        self.sensitivity_weights: Optional[Dict[str, float]] = None
    
    def run_full_search(
        self,
        target_base_quant: str = "MXFP4_MOE",
        max_generations: int = 50,
        population_size: int = 100,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Run complete MagicQuant workflow.
        
        Args:
            target_base_quant: Target base quantization scheme
            max_generations: Evolutionary generations
            population_size: Population size per generation
            verbose: Print progress
            
        Returns:
            List of discovered configurations sorted by score
        """
        if verbose:
            print("=" * 60)
            print("MagicQuant Hybrid Quantization Search")
            print("=" * 60)
            print(f"Source Model: {self.source_model_path}")
            print(f"Output Directory: {self.output_dir}")
            print(f"Target Base Quant: {target_base_quant}")
            print()
        
        # Step 1: Calculate baseline perplexity
        if verbose:
            print("Step 1: Calculating baseline perplexity...")
        
        self.baseline_ppl = self.llama_tools.calculate_perplexity(
            self.source_model_path,
            verbose=verbose
        )
        
        if self.baseline_ppl is None:
            print("WARNING: Could not calculate baseline PPL. Using default.")
            self.baseline_ppl = 5.0  # Default fallback
        
        # Step 2: Run sensitivity probing (simulated for now)
        if verbose:
            print("\nStep 2: Estimating sensitivity weights...")
        
        # Use heuristic weights based on model type
        self.sensitivity_weights = {
            'E': 0.15,  # Embeddings - high sensitivity
            'H': 0.08,  # Head - moderate
            'Q': 0.12,  # Attention Q - moderate-high
            'K': 0.12,  # Attention K/V - moderate-high
            'O': 0.15,  # Attention Output - high
            'U': 0.20,  # FFN Up - lower (robust)
            'D': 0.18   # FFN Down - lower (robust)
        }
        
        if verbose:
            print("Sensitivity weights:")
            for group, weight in self.sensitivity_weights.items():
                print(f"  {group}: {weight:.3f}")
        
        # Step 3: Get baseline size and speed estimates
        baseline_size_gb = self._estimate_model_size(self.source_model_path)
        baseline_tps = 360  # Estimated tokens/second for BF16 30B model
        
        if verbose:
            print(f"\nBaseline Size: {baseline_size_gb:.2f} GB")
            print(f"Baseline TPS: {baseline_tps} tokens/sec")
        
        # Step 4: Run evolutionary search
        if verbose:
            print(f"\nStep 3: Running evolutionary search ({max_generations} generations)...")
        
        predictor = PredictiveScorer(
            sensitivity_weights=self.sensitivity_weights,
            baseline_size_gb=baseline_size_gb,
            baseline_tps=baseline_tps
        )
        
        survivor = EvolutionarySurvivor(
            predictor=predictor,
            baseline_config={'E': 'BF16', 'H': 'BF16'},
            max_generations=max_generations,
            population_size=population_size,
            epsilon=0.2
        )
        
        best_configs = survivor.run_evolution(verbose=verbose)
        
        # Step 5: Display top configurations
        if verbose:
            print("\n" + "=" * 60)
            print("Top 10 Discovered Configurations:")
            print("=" * 60)
            
            for i, config in enumerate(best_configs[:10], 1):
                print(f"\n{i}. Configuration:")
                print(f"   Groups: {config['config']}")
                print(f"   Predicted Loss: {config.get('predicted_loss', 0):.4f}")
                print(f"   Predicted Size: {config.get('predicted_size_gb', 0):.2f} GB")
                print(f"   Predicted TPS: {config.get('predicted_tps', 0):.1f}")
                print(f"   Composite Score: {config.get('composite_score', 0):.4f}")
        
        return best_configs
    
    def generate_hybrid_model(
        self,
        config: Dict[str, str],
        model_name: str,
        base_quant: str = "MXFP4_MOE",
        verify: bool = True
    ) -> Optional[str]:
        """
        Generate a hybrid model using llama.cpp.
        
        NOTE: llama.cpp doesn't directly support per-group quantization.
        This is a SIMPLIFIED approach that creates a single quantization.
        For true hybrid, would need to:
        1. Extract tensors individually
        2. Quantize each with appropriate scheme
        3. Reassemble into new GGUF
        
        Args:
            config: Group -> quant scheme mapping
            model_name: Base model name for output file
            base_quant: Base quantization scheme
            verify: Calculate PPL after generation
            
        Returns:
            Path to generated model or None if failed
        """
        # Generate output filename
        output_filename = generate_name(model_name, base_quant, config)
        output_path = self.output_dir / output_filename
        
        # For now, just quantize with the base scheme
        # A full implementation would handle per-group quantization
        llamacpp_type = get_llamacpp_quant_type(base_quant)
        
        print(f"\nGenerating: {output_filename}")
        print(f"Using llama.cpp type: {llamacpp_type}")
        
        success = self.llama_tools.quantize_model(
            input_path=self.source_model_path,
            output_path=str(output_path),
            quant_type=llamacpp_type,
            verbose=True
        )
        
        if not success:
            print("Failed to generate model")
            return None
        
        # Verify if requested
        if verify:
            print(f"\nVerifying {output_filename}...")
            ppl = self.llama_tools.calculate_perplexity(str(output_path))
            
            if ppl:
                loss = (ppl - self.baseline_ppl) / self.baseline_ppl
                print(f"  Baseline PPL: {self.baseline_ppl:.4f}")
                print(f"  Quantized PPL: {ppl:.4f}")
                print(f"  Precision Loss: {loss*100:.2f}%")
        
        return str(output_path)
    
    def _estimate_model_size(self, model_path: str) -> float:
        """Estimate model size from file size."""
        size_bytes = os.path.getsize(model_path)
        size_gb = size_bytes / (1024 ** 3)
        return size_gb


def main():
    """Main entry point for orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MagicQuant Orchestrator - Hybrid Quantization Search"
    )
    parser.add_argument(
        "source_model",
        help="Path to source GGUF model (BF16/F16)"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for generated models"
    )
    parser.add_argument(
        "--target-quant",
        default="MXFP4_MOE",
        help="Target base quantization scheme"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of evolutionary generations"
    )
    parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory"
    )
    
    args = parser.parse_args()
    
    # Run orchestrator
    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.source_model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path
    )
    
    best_configs = orchestrator.run_full_search(
        target_base_quant=args.target_quant,
        max_generations=args.generations,
        verbose=True
    )
    
    # Generate top 3 configurations
    print("\n" + "=" * 60)
    print("Generating Top 3 Configurations...")
    print("=" * 60)
    
    for i, config in enumerate(best_configs[:3], 1):
        model_name = f"Qwen3-Coder-30B-Config{i}"
        path = orchestrator.generate_hybrid_model(
            config=config['config'],
            model_name=model_name,
            base_quant=args.target_quant,
            verify=True
        )
        
        if path:
            print(f"\n✓ Generated: {path}")


if __name__ == "__main__":
    main()
```

**Verification:**
- File exists at `magicquant/orchestrator.py`
- Can run: `python -m magicquant.orchestrator --help`

---

### **TASK 3: Update CLI Entry Point**
**File to modify:** `magicquant/__main__.py`

**Changes needed:**

Replace the entire file content with:

```python
"""
MagicQuant CLI - Main Entry Point

Usage:
    magicquant search <model.gguf>         Run evolutionary search
    magicquant generate <model.gguf>       Generate hybrid GGUF
"""

import argparse
import sys
import os

from magicquant.orchestrator import MagicQuantOrchestrator


def cmd_search(args):
    """Run evolutionary search to find optimal configurations."""
    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path
    )
    
    best_configs = orchestrator.run_full_search(
        target_base_quant=args.target_quant,
        max_generations=args.generations,
        population_size=args.population,
        verbose=True
    )
    
    # Save results to file
    import json
    results_file = os.path.join(args.output_dir, "search_results.json")
    
    with open(results_file, 'w') as f:
        json.dump([
            {
                'config': c['config'],
                'predicted_loss': c.get('predicted_loss', 0),
                'predicted_size_gb': c.get('predicted_size_gb', 0),
                'predicted_tps': c.get('predicted_tps', 0),
                'composite_score': c.get('composite_score', 0)
            }
            for c in best_configs[:20]
        ], f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


def cmd_generate(args):
    """Generate hybrid model from search results."""
    import json
    
    # Load search results
    results_file = os.path.join(args.output_dir, "search_results.json")
    
    if not os.path.exists(results_file):
        print(f"Error: Search results not found at {results_file}")
        print("Please run 'magicquant search' first")
        sys.exit(1)
    
    with open(results_file) as f:
        results = json.load(f)
    
    if not results:
        print("No configurations found in results")
        sys.exit(1)
    
    # Take top N configurations
    top_n = args.top_n
    configs_to_generate = results[:top_n]
    
    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path
    )
    
    # Calculate baseline once
    orchestrator.baseline_ppl = orchestrator.llama_tools.calculate_perplexity(
        args.model,
        verbose=True
    )
    
    print(f"\nGenerating top {top_n} configurations...")
    
    for i, config_data in enumerate(configs_to_generate, 1):
        config = config_data['config']
        model_name = f"Qwen3-Coder-30B-Config{i}"
        
        print(f"\n{'='*60}")
        print(f"Generating Configuration {i}/{top_n}")
        print(f"{'='*60}")
        
        path = orchestrator.generate_hybrid_model(
            config=config,
            model_name=model_name,
            base_quant=args.target_quant,
            verify=args.verify
        )
        
        if path:
            print(f"✓ Generated: {path}")
        else:
            print(f"✗ Failed to generate configuration {i}")


def main():
    parser = argparse.ArgumentParser(
        prog="magicquant",
        description="Evolutionary Tensor Search for Optimal LLM Compression"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Run evolutionary search for optimal configurations"
    )
    search_parser.add_argument("model", help="Path to source GGUF model (BF16/F16)")
    search_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)"
    )
    search_parser.add_argument(
        "--target-quant",
        default="MXFP4_MOE",
        help="Target base quantization (default: MXFP4_MOE)"
    )
    search_parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations (default: 50)"
    )
    search_parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size (default: 100)"
    )
    search_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)"
    )
    search_parser.set_defaults(func=cmd_search)
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate hybrid models from search results"
    )
    generate_parser.add_argument("model", help="Path to source GGUF model")
    generate_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)"
    )
    generate_parser.add_argument(
        "--target-quant",
        default="MXFP4_MOE",
        help="Target base quantization (default: MXFP4_MOE)"
    )
    generate_parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Generate top N configurations (default: 3)"
    )
    generate_parser.add_argument(
        "--verify",
        action="store_true",
        help="Calculate perplexity after generation"
    )
    generate_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)"
    )
    generate_parser.set_defaults(func=cmd_generate)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
```

**Verification:**
- File modified successfully
- Can run: `python -m magicquant --help`
- Shows "search" and "generate" commands

---

### **TASK 4: Test the Implementation**
**Steps to execute:**

1. **Test CLI help:**
```bash
python -m magicquant --help
python -m magicquant search --help
python -m magicquant generate --help
```

2. **Run search on Qwen model:**
```bash
python -m magicquant search "F:\models\Qwen3-Coder-30B-A3B-Instruct-GGUF\<filename>.gguf" --output-dir "./output" --target-quant MXFP4_MOE --generations 50
```

3. **Generate top configurations:**
```bash
python -m magicquant generate "F:\models\Qwen3-Coder-30B-A3B-Instruct-GGUF\<filename>.gguf" --output-dir "./output" --target-quant MXFP4_MOE --top-n 3 --verify
```

**Expected Results:**
- Search completes and saves `output/search_results.json`
- Generate creates 3 GGUF files in `output/` directory
- Each with name like `Qwen3-Coder-30B-Config1-MXFP4_MOE-*.gguf`

---

## ⚠️ **Important Notes for Coding Agent**

### **Limitation: True Per-Group Quantization**

The current implementation uses llama.cpp's standard quantization, which applies a single scheme to the entire model. **This is NOT true hybrid/per-group quantization**.

True hybrid quantization would require:
1. Extracting individual tensors from source GGUF
2. Quantizing each tensor with its assigned scheme
3. Reassembling into a new GGUF file

This would need implementation of:
- Tensor extraction from GGUF (reading raw bytes)
- Per-tensor quantization
- GGUF writing with mixed precision tensors

**For MVP:** The current approach finds optimal configurations via evolutionary search, but generates standard single-scheme quantizations.

**For Full Implementation:** Would need to complete `magicquant/gguf/writer.py` as outlined in "Approach A" of the original plan.

### **Dependencies to Add**

No new dependencies needed - existing requirements.txt is sufficient.

### **Windows Compatibility**

Code uses `os.name` checks for Windows vs. Unix commands:
- `where` vs `which` for finding executables
- Path handling via `pathlib.Path` for cross-platform compatibility

---

## ✅ **Success Criteria**

After implementation:
1. ✅ Can run `magicquant search` command
2. ✅ Evolutionary search completes and finds configurations
3. ✅ Can run `magicquant generate` command
4. ✅ Generates quantized GGUF files via llama.cpp
5. ✅ Outputs show predicted loss, size, TPS metrics
6. ✅ Generated models can be loaded in llama.cpp

---

## 📝 **Implementation Checklist**

- [ ] Create `magicquant/utils/llamacpp.py`
- [ ] Create `magicquant/orchestrator.py`
- [ ] Update `magicquant/__main__.py`
- [ ] Test CLI help commands
- [ ] Run search on test/small model first
- [ ] Run search on Qwen3-Coder-30B-A3B-Instruct
- [ ] Generate top 3 configurations
- [ ] Verify generated models load in llama.cpp
- [ ] Document any issues or limitations found

---

## 🚀 **Ready to Implement**

This plan is complete and ready for execution by a coding agent. All file paths, code snippets, and verification steps are included.

**Note:** Replace `<filename>.gguf` in the test commands with the actual filename of your Qwen3-Coder-30B-A3B-Instruct GGUF file.
