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
import logging
import os
import re

logger = logging.getLogger(__name__)


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
        # mtp./nextn.: multi-token-prediction layers (llama.cpp names them
        # blk.N.nextn.{eh_proj,embed_tokens,shared_head...} across GLM/DeepSeek/
        # Qwen3.5 MTP arches). Head-adjacent: they predict tokens, so treat them
        # with H's sensitivity rather than inventing a new group.
        'H': [r'^output\.weight$', r'lm_head\.weight', r'mtp\.', r'nextn\.'],
        # MoE experts. GGUF names them ffn_{up,gate,down}_exps — the prior
        # `ffn.*expert` (literal "expert") missed them, so ffn_up_exps/ffn_gate_exps
        # fell through to dense group U. ffn_gate_up_exps (fused) needs its own pattern.
        'X': [r'ffn_(up|gate|down)_exps', r'ffn_gate_up_exps', r'ffn.*expert',
              r'block_sparse_moe\.(input|output)_linear'],
        'R': [r'ffn_gate_inp', r'router', r'block_sparse_moe\.router'],
        'Q': [r'attn_q\.weight', r'attn_qkv\.weight'],
        'K': [r'attn_k\.weight', r'attn_v\.weight'],
        # attn_gate: Qwen3.5 gated attention -- a per-head gate multiplied into
        # the attention output, so it shares O's sensitivity band.
        'O': [r'attn_output\.weight', r'attn_gate\.weight'],
        'S': [r'linear_attn\.', r'mamba\.', r'ssm\.', r'ssm_'],
        'U': [r'ffn_up', r'ffn_gate(?!_inp)', r'ffn_up_shared',
              r'shared_mlp\.input_linear'],
        'D': [r'ffn_down(?!_exps)', r'ffn_down_shared',
              r'shared_mlp\.output_linear'],
    }
    
    # Name suffixes that mark a tensor as (almost certainly) 1-D: norms,
    # biases, and per-channel scales. classify_tensor() only ever sees a
    # name -- no shape -- so this is a name-based *proxy* for "is 1-D",
    # used solely to keep the unknown-tensor gate (below) from flagging
    # noise: the GGUF writer forces every 1-D tensor to F32 regardless of
    # assigned group, so an unrecognized norm/bias name doesn't degrade
    # search quality the way an unrecognized 2D+ matrix does. This is a
    # heuristic, not a guarantee -- documented honestly: a 2D+ tensor whose
    # name happens to end in one of these suffixes would be (harmlessly)
    # excluded from the report, and a 1-D tensor with an unexpected suffix
    # would be (harmlessly, just noisily) included in it.
    _LIKELY_1D_SUFFIXES = (
        '.bias', '_bias',
        'norm.weight', 'norm.bias',
        '.scale', '_scale',
        'layernorm.weight', 'layernorm.bias',
    )

    # Cap on stored example names per instance (report/warning payload size).
    _MAX_UNCLASSIFIED_EXAMPLES = 10

    def __init__(self):
        self.patterns = {
            group: [re.compile(p, re.IGNORECASE) for p in patterns]
            for group, patterns in self.GROUP_PATTERNS.items()
        }

        # Unknown-tensor gate state (loud-on-unknown). Populated as
        # classify_tensor() runs across this instance's lifetime; a new
        # architecture's novel tensor names hitting no explicit pattern AND
        # no keyword heuristic land here instead of silently vanishing into
        # a caller's default quant scheme. See warn_unclassified_once() /
        # the `unclassified` property for how this gets surfaced.
        self._unclassified_examples: List[str] = []
        self._unclassified_count: int = 0
        self._warned: bool = False

    # Heuristic keywords for fallback classification when explicit patterns miss.
    # Maps substrings found in tensor names to groups. Checked only if no
    # explicit pattern matched — prevents wack-a-mole with new architectures.
    _HEURISTIC_KEYWORDS = {
        'N': ['norm', 'layernorm', 'rmsnorm'],
        'E': ['embed', 'embd', 'wte'],
        'H': ['lm_head', 'nextn', 'mtp'],
        'R': ['router', 'gate_inp', 'gating'],
        'X': ['expert', 'moe'],
        'S': ['ssm', 'mamba', 'conv1d', 'dt_bias', 'a_log', 'recurrence'],
        'Q': ['q_proj', 'query'],
        'K': ['k_proj', 'v_proj', 'key', 'value'],
        'O': ['o_proj', 'out_proj', 'attn_output', 'attn_out', 'attn_gate'],
        'U': ['up_proj', 'gate_proj', 'input_linear', 'in_proj',
              'ffn_up', 'ffn_gate', 'w1', 'w3'],
        'D': ['down_proj', 'output_linear', 'ffn_down', 'w2'],
        'V': ['visual', 'vision', 'image'],
    }

    def classify_tensor(self, tensor_name: str) -> str:
        """
        Classify a single tensor into its functional group.

        Uses explicit regex patterns first, then falls back to keyword
        heuristics so new architectures get reasonable defaults without
        needing pattern updates for every model.

        Returns:
            Single character group identifier (E, H, Q, K, O, U, D, X, R, S, N, V)
            or 'UNKNOWN' if no match found
        """
        tensor_lower = tensor_name.lower()

        # Pass 1: explicit patterns (high confidence)
        for group, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(tensor_lower):
                    return group

        # Pass 2: keyword heuristics (reasonable defaults)
        for group, keywords in self._HEURISTIC_KEYWORDS.items():
            for kw in keywords:
                if kw in tensor_lower:
                    return group

        # Neither explicit pattern nor keyword heuristic matched: this name
        # hit no classification pattern at all. A caller resolving group ->
        # quant scheme (e.g. group_schemes.get(group, base_quant)) will
        # silently fall back to its own default for 'UNKNOWN' -- fine for
        # noise (1-D norms/biases, forced F32 regardless of group anyway)
        # but a quiet, quality-degrading surprise for a real 2D+ matrix from
        # an architecture these patterns don't know yet. Track it so that
        # can be surfaced loudly instead of drifting silently.
        if not self._looks_1d(tensor_name):
            self._record_unclassified(tensor_name)

        return 'UNKNOWN'

    def _looks_1d(self, tensor_name: str) -> bool:
        """Name-based proxy for '1-D tensor' -- see _LIKELY_1D_SUFFIXES."""
        name_lower = tensor_name.lower()
        return any(name_lower.endswith(suffix) for suffix in self._LIKELY_1D_SUFFIXES)

    def _record_unclassified(self, tensor_name: str) -> None:
        """Record a substantial (non-1D-looking) unclassified tensor name.

        Strict mode (MAGICQUANT_STRICT_CLASSIFY=1) raises immediately, right
        here, on whichever tensor first trips it -- fail-fast doesn't need
        to wait for a full pass. Loud (non-strict) mode just accumulates;
        the summary warning fires once via warn_unclassified_once(), which
        classify_tensors() calls automatically at the end of a batch pass.
        """
        self._unclassified_count += 1
        if len(self._unclassified_examples) < self._MAX_UNCLASSIFIED_EXAMPLES:
            self._unclassified_examples.append(tensor_name)

        if os.environ.get("MAGICQUANT_STRICT_CLASSIFY") == "1":
            raise ValueError(
                f"MAGICQUANT_STRICT_CLASSIFY=1: tensor '{tensor_name}' matched "
                "no classification pattern (explicit or keyword heuristic). "
                "New architecture? magicquant/gguf/tensor_groups.py patterns "
                "need extending -- refusing to let it silently default to a "
                "base quant scheme."
            )

    @property
    def unclassified(self) -> Dict[str, object]:
        """Report of substantial tensors seen (via classify_tensor) that hit
        no explicit pattern and no keyword heuristic this instance's
        lifetime. {'count': total N (uncapped), 'examples': up to the first
        10 names}. 1-D-looking names (norms/biases/scales) are excluded --
        see _LIKELY_1D_SUFFIXES."""
        return {
            'count': self._unclassified_count,
            'examples': list(self._unclassified_examples),
        }

    def warn_unclassified_once(self) -> None:
        """Emit ONE loud warning summarizing this instance's unclassified
        tensors, if any were seen and this instance hasn't already warned.

        classify_tensors() calls this automatically at the end of its batch
        pass (that's "a classification pass over a model" in the literal
        sense -- it's handed the model's full tensor-name list at once).
        Callers that instead drive classify_tensor() one name at a time
        across a full model (e.g. the writer's/orchestrator's per-tensor
        loops) must call this explicitly once their own pass completes to
        get the same loud summary -- classify_tensor() alone only accumulates,
        it never assumes it has seen "the whole model" that way.
        """
        if self._warned or self._unclassified_count == 0:
            return
        self._warned = True
        examples = self._unclassified_examples
        more = self._unclassified_count - len(examples)
        more_suffix = f" (+{more} more)" if more > 0 else ""
        logger.warning(
            "MagicQuant tensor classifier: %d tensor(s) matched no "
            "classification pattern (explicit or keyword heuristic) and "
            "will silently default to a base/fallback quant scheme wherever "
            "a caller resolves group -> scheme. New architecture? "
            "magicquant/gguf/tensor_groups.py patterns need extending -- "
            "search quality degrades silently otherwise. Examples: %s%s",
            self._unclassified_count, examples, more_suffix,
        )

    def classify_tensors(self, tensors: List[str]) -> Dict[str, List[str]]:
        """Classify multiple tensors into groups.

        Represents one full classification pass over a model's tensor list:
        fires the unclassified-tensor summary warning (once) at the end, see
        warn_unclassified_once().
        """
        grouped = {group: [] for group in self.GROUP_PATTERNS}
        grouped['UNKNOWN'] = []

        for tensor_name in tensors:
            group = self.classify_tensor(tensor_name)
            grouped[group].append(tensor_name)

        self.warn_unclassified_once()
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