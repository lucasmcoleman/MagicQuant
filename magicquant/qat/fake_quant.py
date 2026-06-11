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

Scheme set so far (Task 2): BF16, F16, F32, Q8_0, MXFP4, Q4_K. Any scheme without
a kernel falls back to BF16 passthrough with a logged warning, so a hybrid is
always trainable (just not quant-aware for the unmapped groups). Q6_K/Q5_K are
added in Task 3.
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


# ── MXFP4: 32-elem blocks, E8M0 shared exponent + E2M1 grid ──────────────────

# Signed E2M1 magnitudes (the 8 representable magnitudes, mirrored for sign).
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _fq_mxfp4(w: torch.Tensor) -> torch.Tensor:
    """MXFP4: 32-elem blocks with a shared power-of-two (E8M0) block scale and
    each element rounded to the nearest E2M1 grid value. ggml picks the block
    scale so the block's max element lands near the top of the grid (max
    magnitude 6 ~= 2^2.585), i.e. ``scale = 2^(floor(log2(amax)) - 2)``."""
    block = 32
    blocks, shape, n = _to_blocks(w, block)
    grid = _E2M1_GRID.to(device=w.device, dtype=blocks.dtype)

    amax = blocks.abs().amax(dim=-1, keepdim=True)
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    exp = torch.floor(torch.log2(safe_amax)) - 2.0
    scale = torch.where(amax > 0, torch.exp2(exp), torch.ones_like(amax))

    scaled = blocks / scale  # bring into the grid's domain
    sign = torch.sign(scaled)
    mag = scaled.abs()
    # nearest E2M1 magnitude: (n_blocks, block, n_grid) -> argmin over grid
    diff = (mag.unsqueeze(-1) - grid.view(1, 1, -1)).abs()
    idx = diff.argmin(dim=-1)
    q_mag = grid[idx]
    deq = sign * q_mag * scale
    return _from_blocks(deq, shape, n)


# ── K-quant common: ggml make_qkx2 scale/min search ──────────────────────────

def _make_qkx2(
    sb: torch.Tensor,
    maxq: int,
    nstep: int = 20,
    rmin: float = -1.0,
    rdelta: float = 0.1,
) -> tuple:
    """Vectorized port of ggml's ``make_qkx2_quants`` (unit weights).

    For each group (last dim), find the (scale, min) that minimizes the squared
    reconstruction error of an asymmetric integer quantizer ``round((x-min)/scale)``
    in ``[0, maxq]``. ggml scans candidate ``iscale`` values, assigns integer
    levels, then least-squares-fits (scale, min) to those levels, keeping the
    best. ``min`` is constrained ``<= 0`` (ggml stores ``-min >= 0``).

    Returns ``(scale, min)`` each with shape ``(..., 1)``.
    """
    gmin = sb.amin(dim=-1, keepdim=True).clamp(max=0.0)
    gmax = sb.amax(dim=-1, keepdim=True)
    span = (gmax - gmin).clamp(min=1e-12)

    scale0 = (gmax - gmin) / maxq
    best_scale = scale0.clone()
    best_min = gmin.clone()
    best_err = torch.full_like(gmax, float("inf"))

    n = sb.shape[-1]
    for istep in range(nstep + 1):
        iscale = (rmin + rdelta * istep + maxq) / span
        levels = torch.clamp(torch.round(iscale * (sb - gmin)), 0, maxq)
        # least-squares fit of (scale, min) to the integer levels
        suml = levels.sum(dim=-1, keepdim=True)
        sumll = (levels * levels).sum(dim=-1, keepdim=True)
        sumx = sb.sum(dim=-1, keepdim=True)
        sumlx = (levels * sb).sum(dim=-1, keepdim=True)
        det = n * sumll - suml * suml
        ok = det.abs() > 1e-12
        scale = torch.where(ok, (n * sumlx - suml * sumx) / det, scale0)
        mn = torch.where(ok, (sumll * sumx - suml * sumlx) / det, gmin)
        mn = torch.minimum(mn, torch.zeros_like(mn))  # ggml forces min <= 0
        recon = scale * levels + mn
        err = ((recon - sb) ** 2).sum(dim=-1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
        best_min = torch.where(better, mn, best_min)

    return best_scale, best_min


# ── Q4_K: asymmetric K-quant super-block (scale + min) ────────────────────────

def _fq_kquant_min(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Asymmetric K-quant (Q4_K=4 bits; Q5_K=5 bits in Task 3): 256-elem
    super-block of 8 sub-blocks of 32, each with its own (scale, min) found via
    ggml's ``make_qkx2`` error-minimizing search. The per-sub (scale, min) are
    kept at fp16 precision (ggml stores them 6-bit-quantized against a super
    factor; the fake-quant approximates that with fp16 scales — well within
    tolerance, and a clean fixed point so ``fq(fq(w)) == fq(w)``)."""
    super_block = 256
    sub = 32
    n_sub = super_block // sub
    maxq = (1 << n_bits) - 1

    blocks, shape, n = _to_blocks(w, super_block)
    sb = blocks.reshape(blocks.shape[0], n_sub, sub)  # (n_super, n_sub, sub)

    scale, mn = _make_qkx2(sb, maxq)  # (n_super, n_sub, 1)
    scale = _round_to_fp16(scale)
    mn = _round_to_fp16(mn)

    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.clamp(torch.round((sb - mn) * inv_scale), 0, maxq)
    deq = q * scale + mn

    deq = deq.reshape(blocks.shape[0], super_block)
    return _from_blocks(deq, shape, n)


def _fq_q4_k(w: torch.Tensor) -> torch.Tensor:
    return _fq_kquant_min(w, n_bits=4)


# ── Registry + dispatcher ────────────────────────────────────────────────────

SCHEME_FAKE_QUANT: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "BF16": _fq_bf16,
    "F16": _fq_f16,
    "F32": _fq_f32,
    "Q8_0": _fq_q8_0,
    "MXFP4": _fq_mxfp4,
    "Q4_K": _fq_q4_k,
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
