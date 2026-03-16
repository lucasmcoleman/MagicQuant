"""
Naming Scheme Generator - Generate and parse MagicQuant hybrid model names.

The naming scheme uses compact codes to represent which tensor groups use
different quantization schemes:
- E: Embeddings, H: LM Head, Q: Attention Query
- K: Attention Key/Value, O: Attention Output
- U: FFN Up/Gate, D: FFN Down
- X: MoE Experts, R: MoE Router

Example: Qwen3-4B-MXFP4-EH-B16-QKO-IQ4NL.gguf
- Base quantization: MXFP4
- Embeddings + Head: BF16 (higher precision)
- Attention Q/K/O: IQ4_NL
- Everything else: MXFP4 (base)
"""

from typing import Dict, List, Optional, Tuple
import re


# Group code definitions
GROUP_CODES = {
    'E': 'Embeddings',
    'H': 'LM Head', 
    'Q': 'Attention Query',
    'K': 'Attention Key/Value',
    'O': 'Attention Output',
    'U': 'FFN Up/Gate',
    'D': 'FFN Down',
    'X': 'MoE Experts',
    'R': 'MoE Router'
}


def generate_name(
    model_name: str,
    base_quant: str,
    overrides: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate a MagicQuant hybrid model name.
    
    Args:
        model_name: Base model name (e.g., "Qwen3-4B-Instruct")
        base_quant: Base quantization scheme (e.g., "MXFP4_MOE", "IQ4_NL")
        overrides: Dict of group codes -> quant schemes that differ from base
        
    Returns:
        Complete filename with extension (e.g., "Qwen3-4B-MXFP4-EH-B16-QKO-IQ4NL.gguf")
        
    Example:
        generate_name(
            model_name="Qwen3-4B-Instruct",
            base_quant="MXFP4_MOE",
            overrides={"E": "BF16", "H": "BF16", "Q": "IQ4_NL"}
        )
        Returns: "Qwen3-4B-MXFP4-EH-B16-Q-IQ4NL.gguf"
    """
    # Clean model name for filename
    clean_name = model_name.replace(" ", "-").replace("_", "-")
    
    # Build override blocks
    if not overrides:
        return f"{clean_name}-{base_quant}.gguf"
    
    # Group overrides by quant scheme for compact representation
    scheme_to_groups: Dict[str, List[str]] = {}
    for group, scheme in overrides.items():
        if scheme not in scheme_to_groups:
            scheme_to_groups[scheme] = []
        scheme_to_groups[scheme].append(group)
    
    # Sort groups within each scheme alphabetically for consistency
    for scheme in scheme_to_groups:
        scheme_to_groups[scheme].sort()
    
    # Build override string
    override_parts = []
    for scheme, groups in scheme_to_groups.items():
        group_str = ''.join(groups)  # e.g., "EH" or "QKO"
        # Normalize scheme name for filename (remove underscores for readability)
        scheme_clean = scheme.replace("_", "")
        override_parts.append(f"{group_str}-{scheme_clean}")
    
    override_str = "-".join(override_parts)
    
    return f"{clean_name}-{base_quant}-{override_str}.gguf"


def parse_name(name: str) -> Dict:
    """
    Parse a MagicQuant hybrid model name into components.
    
    Args:
        name: Complete filename (e.g., "Qwen3-4B-MXFP4-EH-B16-QKO-IQ4NL.gguf")
        
    Returns:
        Dictionary with parsed components:
            - model_name: Base model name
            - base_quant: Base quantization scheme
            - overrides: Dict of group codes -> quant schemes
    """
    # Remove .gguf extension if present
    if name.endswith('.gguf'):
        name = name[:-5]
    
    parts = name.split('-')
    
    if len(parts) < 3:
        return {
            'error': f'Invalid format: {name}',
            'original_name': name
        }
    
    # First part(s) is model name (may contain hyphens)
    # Last part is base quantization
    base_quant = parts[-1]
    
    # Middle parts are override blocks
    overrides: Dict[str, str] = {}
    
    i = 1  # Skip first part which is model name start
    while i < len(parts) - 1:
        block = parts[i]
        
        # Check if this looks like an override block (has a number after dash)
        if '-' in block or re.match(r'^[EHQKOUXRD]+-[A-Z]', block):
            # This is an override block: "EH-B16" or "Q-IQ4NL"
            group_part, scheme_part = block.rsplit('-', 1)
            
            # Normalize scheme
            scheme_normalized = normalize_scheme(scheme_part)
            
            for group in group_part:
                overrides[group] = scheme_normalized
        
        i += 1
    
    model_name = '-'.join(parts[:-2]) if len(parts) > 3 else parts[0]
    
    return {
        'model_name': model_name,
        'base_quant': base_quant,
        'overrides': overrides
    }


def normalize_scheme(scheme: str) -> str:
    """Normalize a quantization scheme name for consistent comparison."""
    # Add underscore back where appropriate
    if scheme == "IQ4NL":
        return "IQ4_NL"
    elif scheme == "MXFP4MOE" or scheme == "mxfp4moe":
        return "MXFP4_MOE"
    elif scheme == "Q4K" or scheme == "q4k":
        return "Q4_K_M"
    elif scheme == "Q6K" or scheme == "q6k":
        return "Q6_K"
    elif scheme == "Q5K" or scheme == "q5k":
        return "Q5_K"
    else:
        # Return as-is if no known normalization
        return scheme


def get_group_names() -> Dict[str, str]:
    """Get the mapping of group codes to full names."""
    return GROUP_CODES.copy()


def generate_config_for_quant(
    model_name: str,
    base_quant: str,
    overrides: Dict[str, str]
) -> Dict:
    """
    Generate a configuration dictionary for hybrid quant creation.
    
    This can be used to create the config.yaml needed by the hybrid generator.
    
    Args:
        model_name: Base model name
        base_quant: Base quantization scheme
        overrides: Which groups get different quantization
        
    Returns:
        Configuration dictionary in format:
        {
            "model": {...},
            "quantization": {
                "base": "...",
                "groups": {
                    "...": "..."
                }
            }
        }
    """
    return {
        "model": {"name": model_name, "source": None},  # source will be set by user
        "quantization": {
            "base": base_quant,
            "groups": overrides
        }
    }


def calculate_expected_size(
    base_model_size: float,
    base_quant_bits: float,
    overrides: Dict[str, float]
) -> float:
    """
    Estimate the size of a hybrid quant model.
    
    Args:
        base_model_size: Original model size in GB (BF16)
        base_quant_bits: Bits per weight for base quantization
        overrides: Dict mapping group names to their quant bits
        
    Returns:
        Estimated size in GB
    """
    # Total parameters is proportional to base size * 16 / base_quant_bits
    total_params = base_model_size * (16.0 / base_quant_bits)
    
    # This is a simplified estimate - actual size depends on parameter distribution
    # A real implementation would need to know the parameter counts per group
    
    return total_params * (base_quant_bits / 16.0)  # Simplified


def get_scheme_bits(scheme_name: str) -> float:
    """Get the bits per weight for a quantization scheme."""
    # Actual bpw from ggml block format: (block_bytes * 8) / block_elements
    SCHEME_BITS = {
        "BF16": 16.0,
        "Q8_0": 8.5,
        "Q6_K": 6.5625,
        "Q5_K": 5.5,
        "IQ4_NL": 4.5,
        "MXFP4_MOE": 4.25,
        "Q4_K_M": 4.5,
    }
    
    return SCHEME_BITS.get(scheme_name, 8.0)


if __name__ == "__main__":
    # Test the naming scheme
    print("Testing MagicQuant Naming Scheme")
    print("=" * 50)
    
    # Example 1: MXFP4 base with protected embeddings
    name1 = generate_name(
        model_name="Qwen3-4B-Instruct",
        base_quant="MXFP4_MOE",
        overrides={"E": "BF16", "H": "BF16"}
    )
    print(f"Example 1: {name1}")
    
    parsed1 = parse_name(name1)
    print(f"Parsed: {parsed1}")
    print()
    
    # Example 2: IQ4_NL base with high-precision attention
    name2 = generate_name(
        model_name="Qwen3-30B-A3B",
        base_quant="IQ4_NL",
        overrides={"Q": "Q6_K", "K": "Q6_K", "O": "Q8_0"}
    )
    print(f"Example 2: {name2}")
    
    parsed2 = parse_name(name2)
    print(f"Parsed: {parsed2}")