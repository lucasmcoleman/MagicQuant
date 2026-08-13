"""Writer-parity type resolution for v2 planning.

The GGUF writer's Pass 1 applies compatibility rules that can change a
tensor's on-disk type away from the requested scheme (SSM conv operands →
F32, 1-D tensors → F32, BF16 → F16, K-quant 256-block rows that aren't
divisible → block-32 fallback). The v2 allocator must price each (tensor,
scheme) choice by the bytes and distortion of the type that would ACTUALLY
ship, so this module exposes the same resolution as a pure function.

Parity strategy: import the writer's own helpers (single source of truth)
rather than mirroring their logic; ``tests/test_v2_resolve.py`` asserts
parity against a real writer Pass 1 on a synthetic model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from magicquant.gguf.writer import (
    SCHEME_TO_GGML,
    _block32_fallback,
    _is_f32_required_ssm_operand,
    _is_never_quantized,
    _is_quantizable_by_name,
    _tensor_n_elements,
)
from magicquant.quant.converters import GGML_BLOCK_SIZE, ggml_tensor_data_size


def resolve_tensor_type(
    name: str,
    shape: List[int],
    n_dims: int,
    group: str,
    scheme_name: str,
) -> Tuple[str, Optional[str]]:
    """Resolve the actual on-disk ggml type for quantizing ``name`` with
    ``scheme_name``, applying the writer's Pass-1 compatibility rules in
    the writer's order.

    Returns ``(actual_ggml_type_name, reason)`` where ``reason`` is None
    when the requested scheme maps through unchanged, else one of
    ``"f32-required-operand" | "not-a-weight-tensor" |
    "never-quantize-name" | "1d-f32" |
    "bf16-downgrade" | "block-size"``.

    Raises ValueError for a scheme name unknown to the writer's map (the
    writer would silently default it to Q4_0; planning must never do that).
    """
    if scheme_name not in SCHEME_TO_GGML:
        raise ValueError(
            f"Unknown scheme name {scheme_name!r} (known: "
            f"{sorted(SCHEME_TO_GGML)})"
        )
    target = SCHEME_TO_GGML[scheme_name]
    reason: Optional[str] = None

    # 1. SSM conv operands must be F32 (kernel constraint) — checked before
    #    the block-size rule because float schemes (block_size 1) would skip it.
    if group == "S" and target != "F32" and _is_f32_required_ssm_operand(name):
        return "F32", "f32-required-operand"

    # 2. Not a "weight" tensor -- llama.cpp's first quantization gate. Must
    #    precede the 1-D and block-size rules for the same reason as #1:
    #    ssm_a/ssm_d are 2-D with ne[0]=1, so only a quantized target hits
    #    the block-size fallback; a float target slips through as F16 and
    #    aborts llama.cpp. See writer._is_quantizable_by_name.
    if target != "F32" and not _is_quantizable_by_name(name):
        return "F32", "not-a-weight-tensor"

    # 3. Never-quantize-by-name tensors (llama.cpp's own quantizer refuses
    #    these regardless of shape/group -- see writer.py's
    #    _NEVER_QUANTIZE_NAME_SUBSTRINGS). Checked before the 1-D and
    #    block-size rules for the same reason as #1: this can be the only
    #    thing catching a 2-D, block-compatible tensor like ssm_norm.weight.
    if target != "F32" and _is_never_quantized(name):
        return "F32", "never-quantize-name"

    # 4. 1-D tensors (norms/biases) stay F32.
    if n_dims <= 1 and target != "F32":
        return "F32", "1d-f32"

    # 5. BF16 is written as F16 (llama.cpp compute-graph limitation).
    if target == "BF16":
        target, reason = "F16", "bf16-downgrade"

    # 6. Block-size fallback for non-divisible rows.
    row_size = shape[-1] if len(shape) >= 1 else 1
    block_size = GGML_BLOCK_SIZE.get(target, 1)
    if block_size > 1 and row_size % block_size != 0:
        fallback = _block32_fallback(target, row_size, group)
        if fallback != target:
            return fallback, "block-size"

    return target, reason


def tensor_bytes(ggml_type_name: str, shape: List[int]) -> int:
    """Exact on-disk bytes for a tensor of ``shape`` stored as
    ``ggml_type_name`` (ggml block math — identical to the writer's Pass-1
    ``expected_size``)."""
    return ggml_tensor_data_size(ggml_type_name, _tensor_n_elements(list(shape)))
