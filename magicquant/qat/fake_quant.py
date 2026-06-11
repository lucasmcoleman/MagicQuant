"""Differentiable per-scheme fake-quant kernels with a straight-through estimator.

Each kernel quantize->dequantizes a weight tensor in its ggml scheme's block
structure, entirely in torch (so it runs on GPU and is autograd-friendly). The
forward pass is a faithful *approximation* of what libggml would store and read
back; the ``FakeQuantSTE`` autograd Function makes the backward pass a
straight-through identity so gradients flow to the underlying (continuous)
weight as if the quantization weren't there.

Public API:
    fake_quant(w, ggml_type_name) -> Tensor   (the dispatcher; STE-wrapped)
    FakeQuantSTE                              (the autograd Function)
    SCHEME_FAKE_QUANT                         (name -> kernel registry)

Fidelity contract: ``tests/test_fake_quant.py`` asserts each kernel's dequant
output is within a per-scheme tolerance of the real libggml round-trip
(``ggml_encode`` -> ``dequantize_row_*``). These are GPU-fast approximations,
not byte-exact reimplementations — STE makes exact gradients moot.

Scheme set so far (Task 1): BF16, F16, F32, Q8_0. Any scheme without a kernel
falls back to BF16 passthrough with a logged warning, so a hybrid is always
trainable (just not quant-aware for the unmapped groups). MXFP4/Q4_K/Q6_K/Q5_K
are added in Tasks 2-3.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict

import torch


# ── Block-structure helpers ──────────────────────────────────────────────────

def _to_blocks(w: torch.Tensor, block: int):
    """Flatten ``w`` and view it as (n_blocks, block).

    Returns (blocks, original_shape, flat_numel). Real weight rows are multiples
    of 256/32, but if the flat length isn't a multiple of ``block`` the tail is
    zero-padded to a full block (and trimmed back on reconstruction).
    """
    orig_shape = w.shape
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block)
    return blocks, orig_shape, n


def _from_blocks(blocks: torch.Tensor, orig_shape, n: int) -> torch.Tensor:
    flat = blocks.reshape(-1)[:n]
    return flat.reshape(orig_shape)


def _round_to_fp16(x: torch.Tensor) -> torch.Tensor:
    """Round a tensor to fp16 precision (ggml stores block scales as fp16)."""
    return x.to(torch.float16).to(x.dtype)


# ── Float passthroughs ───────────────────────────────────────────────────────

def _fq_bf16(w: torch.Tensor) -> torch.Tensor:
    return w.bfloat16().float().to(w.dtype)


def _fq_f16(w: torch.Tensor) -> torch.Tensor:
    return w.to(torch.float16).to(w.dtype)


def _fq_f32(w: torch.Tensor) -> torch.Tensor:
    return w.to(torch.float32).to(w.dtype)


# ── Q8_0: per-32 block, symmetric int8 × fp16 scale ──────────────────────────

def _fq_q8_0(w: torch.Tensor) -> torch.Tensor:
    block = 32
    blocks, shape, n = _to_blocks(w, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # ggml: d = amax / 127; scale stored as fp16. id = 1/d applied to quantize.
    d = _round_to_fp16(amax / 127.0)
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.round(blocks * inv), -127, 127)
    deq = q * d
    return _from_blocks(deq, shape, n)


# ── Registry + dispatcher ────────────────────────────────────────────────────

SCHEME_FAKE_QUANT: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "BF16": _fq_bf16,
    "F16": _fq_f16,
    "F32": _fq_f32,
    "Q8_0": _fq_q8_0,
}


class FakeQuantSTE(torch.autograd.Function):
    """Straight-through fake-quant.

    forward:  apply the scheme's quant->dequant kernel ``fn(w)``.
    backward: identity — pass the upstream gradient straight through to ``w``
              (and ``None`` for the non-tensor ``fn`` argument).
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]):
        return fn(w)

    @staticmethod
    def backward(ctx, grad_output):
        # identity for the weight, None for the function argument
        return grad_output, None


def fake_quant(w: torch.Tensor, ggml_type_name: str) -> torch.Tensor:
    """Differentiably fake-quantize ``w`` to ``ggml_type_name``.

    Dispatches to the scheme's kernel through ``FakeQuantSTE`` so the backward
    pass is a straight-through identity. Schemes with no kernel fall back to BF16
    passthrough with a ``UserWarning`` (a hybrid stays trainable for unmapped
    groups, just not quant-aware).
    """
    fn = SCHEME_FAKE_QUANT.get(ggml_type_name)
    if fn is None:
        warnings.warn(
            f"No fake-quant kernel for ggml type {ggml_type_name!r}; "
            f"falling back to BF16 passthrough (not quant-aware for this group).",
            UserWarning,
            stacklevel=2,
        )
        fn = _fq_bf16
    return FakeQuantSTE.apply(w, fn)
