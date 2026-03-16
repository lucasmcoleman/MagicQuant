"""
Tensor Group Classification - Identify tensor groups by architectural role.

Groups:
- E: Embeddings (token_embd.weight)
- H: LM Head (output.weight / lm_head.weight)
- Q: Attention Query (attn_q.weight)
- K: Attention Key/Value (attn_k.weight + attn_v.weight)
- O: Attention Output (attn_output.weight)
- U: FFN Up/Gate (ffn_up + ffn_gate)
- D: FFN Down (ffn_down.weight)
- X: MoE Experts (ffn_*_expert.*)
- R: MoE Router (router.*, gate.*)
- S: SSM / Linear Attention (linear_attn.*, mamba.*)
- N: Normalization layers (attn_norm, ffn_norm, etc.) — tiny, keep at source precision
- V: Vision encoder (model.visual.*) — separate modality
"""

from typing import Dict, List, Tuple
import re


class TensorGroupClassifier:
    """
    Classifies GGUF tensors into functional groups based on their names.

    This is crucial for MagicQuant because different tensor groups have
    different sensitivity levels to quantization noise. By grouping them,
    we can apply hybrid quantization strategies effectively.
    """

    # Group definitions with regex patterns.
    # Order matters: first match wins. More specific patterns come first.
    GROUP_PATTERNS = {
        'V': [r'model\.visual\.', r'vision_model\.', r'visual\.'],
        'N': [r'_norm\.weight$', r'layernorm', r'_norm\.bias$',
              r'q_norm\.weight$', r'k_norm\.weight$'],
        'E': [r'token_embd\.weight'],
        'H': [r'output\.weight', r'lm_head\.weight', r'mtp\.'],
        'X': [r'ffn.*expert'],
        'R': [r'ffn_gate_inp', r'router'],
        'Q': [r'attn_q\.weight', r'attn_qkv\.weight'],
        'K': [r'attn_k\.weight', r'attn_v\.weight'],
        'O': [r'attn_output\.weight'],
        'S': [r'linear_attn\.', r'mamba\.', r'ssm\.'],
        'U': [r'ffn_up', r'ffn_gate'],
        'D': [r'ffn_down'],
    }
    
    def __init__(self):
        self.patterns = {
            group: [re.compile(p, re.IGNORECASE) for p in patterns]
            for group, patterns in self.GROUP_PATTERNS.items()
        }
    
    def classify_tensor(self, tensor_name: str) -> str:
        """
        Classify a single tensor into its functional group.
        
        Args:
            tensor_name: The full name of the tensor from GGUF
            
        Returns:
            Single character group identifier (E, H, Q, K, O, U, D, X, R)
            or 'UNKNOWN' if no match found
        """
        tensor_lower = tensor_name.lower()
        
        for group, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(tensor_lower):
                    return group
        
        return 'UNKNOWN'
    
    def classify_tensors(self, tensors: List[str]) -> Dict[str, List[str]]:
        """Classify multiple tensors into groups."""
        grouped = {group: [] for group in self.GROUP_PATTERNS}
        grouped['UNKNOWN'] = []
        
        for tensor_name in tensors:
            group = self.classify_tensor(tensor_name)
            grouped[group].append(tensor_name)
        
        return grouped
    
    def get_group_info(self, model_metadata: Dict) -> Dict:
        """
        Get comprehensive information about tensor groups in a model.
        
        Args:
            model_metadata: GGUF metadata containing tensor names and counts
            
        Returns:
            Dictionary with group analysis including:
                - tensor_counts: Number of tensors per group
                - param_counts: Estimated parameters per group
                - sensitivity_flags: High-sensitivity groups (E, H, O, R)
                - is_moe: Whether model uses Mixture-of-Experts
        """
        tensors = model_metadata.get('tensors', [])
        
        grouped = self.classify_tensors(tensors)
        
        # Calculate tensor counts per group
        tensor_counts = {group: len(t) for group, t in grouped.items()}
        
        # Identify MoE models (have R or X groups with tensors)
        is_moe = len(grouped.get('X', [])) > 0 or len(grouped.get('R', [])) > 0
        
        # High sensitivity groups - these are critical to preserve
        high_sensitivity_groups = ['E', 'H', 'O']  # Embeddings, Head, Output
        if is_moe:
            high_sensitivity_groups.append('R')  # MoE Router is also critical
        
        return {
            'tensor_counts': tensor_counts,
            'grouped_tensors': grouped,
            'is_moe': is_moe,
            'high_sensitivity_groups': high_sensitivity_groups
        }