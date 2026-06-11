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

Scheme set: BF16, F16, F32, Q8_0, MXFP4, Q4_K, Q6_K, Q5_K (v1) plus the
aggressive low-bit tiers Q3_K, Q2_K, IQ4_NL. Any scheme without a kernel falls
back to BF16 passthrough with a logged warning, so a hybrid is always trainable
(just not quant-aware for the unmapped groups).
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

# IQ4_NL's 16-entry non-linear lookup table (the signed levels each 4-bit code
# maps to), mirrored verbatim from libggml's ``kvalues_iq4nl`` in
# ggml-common.h. Monotonically increasing, so "nearest entry" is unambiguous.
_IQ4_NL_KVALUES = torch.tensor(
    [-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
     1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0]
)


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


def _fq_q5_k(w: torch.Tensor) -> torch.Tensor:
    return _fq_kquant_min(w, n_bits=5)


# ── Q6_K: symmetric 6-bit, 16 sub-blocks of 16 ───────────────────────────────

def _fq_q6_k(w: torch.Tensor) -> torch.Tensor:
    """Symmetric 6-bit K-quant: 256-elem super-block of 16 sub-blocks of 16,
    each with its own scale. ggml stores per-sub scale as int8 × a single fp16
    super factor; the fake-quant approximates with a per-sub fp16 scale.

    The signed level range is clamped symmetrically to ``[-31, 31]`` (ggml uses
    ``[-32, 31]``; dropping the lone ``-32`` level makes the quantizer a clean
    fixed point — ``max|q| == 31`` always, so re-deriving ``scale = amax/31``
    from the dequant output reproduces the same scale and ``fq(fq(w)) == fq(w)``).
    The fidelity cost is negligible (well within the Q6_K tolerance)."""
    super_block = 256
    sub = 16
    n_sub = super_block // sub  # 16
    lvl = 31

    blocks, shape, n = _to_blocks(w, super_block)
    sb = blocks.reshape(blocks.shape[0], n_sub, sub)

    amax = sb.abs().amax(dim=-1, keepdim=True)  # (n_super, n_sub, 1)
    scale = _round_to_fp16(amax / lvl)
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.clamp(torch.round(sb * inv_scale), -lvl, lvl)
    deq = q * scale

    deq = deq.reshape(blocks.shape[0], super_block)
    return _from_blocks(deq, shape, n)


# ── Q3_K: symmetric 3-bit, 16 sub-blocks of 16, 6-bit scale-of-scales ─────────

def _make_q3_scale(sb: torch.Tensor, nmax: int = 4) -> torch.Tensor:
    """Per-sub-block scale for Q3_K, ported from ggml's ``make_q3_quants``.

    ggml derives the (signed) sub-block scale from the level set
    ``L = clamp(round(-nmax/max * x), -nmax, nmax-1)`` and a weighted (``w=x²``)
    least-squares fit ``scale = Σ w·x·L / Σ w·L²``. The reference then runs five
    sequential per-element RMSE refinement sweeps; those are dropped here — the
    refinement only tightens the libggml match from ~0.02 to ~0.007 (both far
    inside the Q3_K tolerance) and the sequential 16×5 update would dominate the
    per-step cost of a kernel that runs every training step.

    Returns the signed scale with shape ``(..., 1)``.
    """
    amax_idx = sb.abs().argmax(dim=-1, keepdim=True)
    mx = torch.gather(sb, -1, amax_idx)  # signed value at the max-magnitude slot
    iscale = torch.where(mx != 0, -float(nmax) / mx, torch.zeros_like(mx))
    levels = torch.clamp(torch.round(iscale * sb), -nmax, nmax - 1)
    w = sb * sb
    sumlx = (w * sb * levels).sum(dim=-1, keepdim=True)
    suml2 = (w * levels * levels).sum(dim=-1, keepdim=True)
    return torch.where(suml2 > 0, sumlx / suml2, torch.zeros_like(suml2))


def _fq_q3_k_once(w: torch.Tensor) -> torch.Tensor:
    """One faithful Q3_K quant->dequant pass (see ``_fq_q3_k`` for the wrapper).

    256-elem super-block of 16 sub-blocks of 16. Per ggml's ``quantize_row_q3_K_ref``:
    each sub-block gets a signed scale (``_make_q3_scale``); those 16 scales are
    themselves 6-bit-quantized against a single fp16 super factor
    (``d_all = -max_scale/32``, level ``l = clamp(round(scale/d_all), -32, 31)``);
    each weight is then a signed 3-bit level in ``[-4, 3]`` against ``d_all·l``.
    """
    super_block = 256
    sub = 16
    n_sub = super_block // sub  # 16

    blocks, shape, n = _to_blocks(w, super_block)
    sb = blocks.reshape(blocks.shape[0], n_sub, sub)

    scales = _make_q3_scale(sb)  # (n_super, n_sub, 1) signed

    # 6-bit quantize the sub scales against a single fp16 super factor.
    amax_idx = scales.abs().squeeze(-1).argmax(dim=-1, keepdim=True).unsqueeze(-1)
    max_scale = torch.gather(scales, 1, amax_idx)  # (n_super, 1, 1) signed
    nz = max_scale != 0
    iscale = torch.where(nz, -32.0 / max_scale, torch.zeros_like(max_scale))
    d_all = _round_to_fp16(torch.where(nz, 1.0 / iscale, torch.zeros_like(iscale)))
    sc_lvl = torch.clamp(torch.round(iscale * scales), -32, 31)  # (n_super, n_sub, 1)

    d = d_all * sc_lvl  # per-sub dequant scale
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.round(sb * inv), -4, 3)
    deq = (q * d).reshape(blocks.shape[0], super_block)
    return _from_blocks(deq, shape, n)


def _fq_q3_k(w: torch.Tensor) -> torch.Tensor:
    """Q3_K fake-quant, applied twice so the result is a fixed point.

    A single ``_fq_q3_k_once`` faithfully reproduces the libggml encode->dequant,
    but it is not a one-step fixed point: re-quantizing its output picks a
    slightly different scale-of-scales (the per-sub argmax can move), so
    ``fq(w)`` and ``fq(fq(w))`` differ. The map converges after the second
    application, so the kernel projects onto that fixed point here; the fixed
    point sits ~0.02 from the real round-trip — well inside the Q3_K tolerance —
    and ``fq(fq(w)) == fq(w)`` holds exactly."""
    return _fq_q3_k_once(_fq_q3_k_once(w))


# ── Q2_K: asymmetric 2-bit, 16 sub-blocks of 16, 4-bit scale/min-of-scales ────

def _make_qkx2_weighted(
    sb: torch.Tensor,
    weights: torch.Tensor,
    maxq: int = 3,
    nstep: int = 15,
    rmin: float = -0.5,
    rdelta: float = 0.1,
) -> tuple:
    """Weighted (MAD) port of ggml's ``make_qkx2_quants`` for Q2_K.

    Unlike the unit-weight :func:`_make_qkx2` used by Q4_K/Q5_K, ggml's Q2_K path
    weights the (scale, min) error by ``|x|`` and minimises mean-absolute (not
    squared) deviation, scanning ``nstep+1`` candidate ``iscale`` values and
    least-squares-fitting (scale, min) to the resulting levels. ``min`` is
    constrained ``<= 0`` (ggml stores ``-min >= 0``).

    Returns ``(scale, the_min)`` each with shape ``(..., 1)``; ``scale >= 0`` and
    ``the_min = -min >= 0`` so dequant is ``q*scale - the_min``.
    """
    gmin = sb.amin(dim=-1, keepdim=True).clamp(max=0.0)
    gmax = sb.amax(dim=-1, keepdim=True)
    sum_w = weights.sum(dim=-1, keepdim=True)
    sum_x = (weights * sb).sum(dim=-1, keepdim=True)
    rng = gmax - gmin
    degenerate = rng == 0
    rng_safe = torch.where(degenerate, torch.ones_like(rng), rng)

    iscale = maxq / rng_safe
    scale = 1.0 / iscale
    levels = torch.clamp(torch.round(iscale * (sb - gmin)), 0, maxq)
    err = (weights * (scale * levels + gmin - sb).abs()).sum(dim=-1, keepdim=True)

    best_err = err
    best_scale = scale.clone()
    best_min = gmin.clone()

    for istep in range(nstep + 1):
        iscale = (rmin + rdelta * istep + maxq) / rng_safe
        laux = torch.clamp(torch.round(iscale * (sb - gmin)), 0, maxq)
        sum_l = (weights * laux).sum(dim=-1, keepdim=True)
        sum_l2 = (weights * laux * laux).sum(dim=-1, keepdim=True)
        sum_xl = (weights * laux * sb).sum(dim=-1, keepdim=True)
        det = sum_w * sum_l2 - sum_l * sum_l
        ok = det > 0
        det_safe = torch.where(ok, det, torch.ones_like(det))
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / det_safe
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det_safe
        # ggml forces min <= 0; when the fit wants min > 0, refit scale alone.
        min_pos = this_min > 0
        sum_l2_safe = torch.where(sum_l2 != 0, sum_l2, torch.ones_like(sum_l2))
        this_scale = torch.where(min_pos, sum_xl / sum_l2_safe, this_scale)
        this_min = torch.where(min_pos, torch.zeros_like(this_min), this_min)
        recon = this_scale * laux + this_min
        cur_err = (weights * (recon - sb).abs()).sum(dim=-1, keepdim=True)
        better = ok & (cur_err < best_err)
        best_err = torch.where(better, cur_err, best_err)
        best_scale = torch.where(better, this_scale, best_scale)
        best_min = torch.where(better, this_min, best_min)

    the_min = -best_min
    best_scale = torch.where(degenerate, torch.zeros_like(best_scale), best_scale)
    the_min = torch.where(degenerate, -gmin, the_min)
    return best_scale, the_min


def _fq_q2_k_once(w: torch.Tensor) -> torch.Tensor:
    """One faithful Q2_K quant->dequant pass (see ``_fq_q2_k`` for the wrapper).

    256-elem super-block of 16 sub-blocks of 16. Per ggml's
    ``quantize_row_q2_K_ref``: each sub-block gets an asymmetric (scale, min)
    from ``_make_qkx2_weighted``; those 16 scales and 16 mins are each 4-bit
    quantized against their own fp16 super factor (``d = max_scale/15``,
    ``dmin = max_min/15``); each weight is a 2-bit level in ``[0, 3]``, dequant
    ``q·(d·sl) - (dmin·ml)``.
    """
    super_block = 256
    sub = 16
    n_sub = super_block // sub  # 16
    q4scale = 15.0

    blocks, shape, n = _to_blocks(w, super_block)
    sb = blocks.reshape(blocks.shape[0], n_sub, sub)

    weights = sb.abs()
    scale, mn = _make_qkx2_weighted(sb, weights)  # scale>=0, mn>=0

    max_scale = scale.amax(dim=1, keepdim=True)  # (n_super, 1, 1)
    max_min = mn.amax(dim=1, keepdim=True)
    sc_pos = max_scale > 0
    mn_pos = max_min > 0
    isc = torch.where(sc_pos, q4scale / torch.where(sc_pos, max_scale, torch.ones_like(max_scale)),
                      torch.zeros_like(max_scale))
    imn = torch.where(mn_pos, q4scale / torch.where(mn_pos, max_min, torch.ones_like(max_min)),
                      torch.zeros_like(max_min))
    sl = torch.clamp(torch.round(isc * scale), 0, 15)  # 4-bit sub scale level
    ml = torch.clamp(torch.round(imn * mn), 0, 15)      # 4-bit sub min level
    d = _round_to_fp16(torch.where(sc_pos, max_scale / q4scale, torch.zeros_like(max_scale)))
    dmin = _round_to_fp16(torch.where(mn_pos, max_min / q4scale, torch.zeros_like(max_min)))

    dl = d * sl     # per-sub dequant scale
    dm = dmin * ml  # per-sub dequant min (subtracted)
    inv = torch.where(dl != 0, 1.0 / dl, torch.zeros_like(dl))
    q = torch.clamp(torch.round((sb + dm) * inv), 0, 3)
    deq = (q * dl - dm).reshape(blocks.shape[0], super_block)
    return _from_blocks(deq, shape, n)


def _fq_q2_k(w: torch.Tensor) -> torch.Tensor:
    """Q2_K fake-quant, applied twice so the result is a fixed point.

    Like Q3_K, a single faithful pass matches libggml closely (~0.04 mean rel
    error) but is not a one-step fixed point — the ``|x|``-weighted (scale, min)
    search re-derives slightly different values when the (quantized) input
    changes the weights. The map converges after the second application, so the
    kernel projects onto that fixed point here; it stays well inside the Q2_K
    tolerance and ``fq(fq(w)) == fq(w)`` holds exactly."""
    return _fq_q2_k_once(_fq_q2_k_once(w))


# ── IQ4_NL: 32-elem blocks, non-linear 16-entry lookup table ──────────────────

def _fq_iq4_nl_once(w: torch.Tensor) -> torch.Tensor:
    """One faithful IQ4_NL quant->dequant pass (see ``_fq_iq4_nl`` for the wrapper).

    32-elem blocks; each element maps to the nearest entry in the 16-value
    non-linear lookup table (``_IQ4_NL_KVALUES``), scaled by a per-block fp16
    factor. Mirrors ggml's ``quantize_row_iq4_nl_impl`` (block_size == the full
    super-block here, so the simple single-scale branch): the block scale is
    seeded from ``-max/values[0]`` then refined over ``2·ntry+1`` candidate
    inverse-scales, each scored by the weighted (``w=x²``) least-squares
    objective ``sumqx²/sumq2``, keeping the best."""
    block = 32
    ntry = 7
    blocks, shape, n = _to_blocks(w, block)
    kv = _IQ4_NL_KVALUES.to(device=w.device, dtype=blocks.dtype)
    v0 = kv[0]  # -127

    amax = blocks.abs().amax(dim=-1, keepdim=True)
    nonzero = amax >= 1e-8
    amax_idx = blocks.abs().argmax(dim=-1, keepdim=True)
    mx = torch.gather(blocks, -1, amax_idx)  # signed value at the max-magnitude slot

    w2 = blocks * blocks

    def _best(idx_scaled):
        # nearest lookup-table entry for each scaled element, then the
        # least-squares-optimal scale and its score for that assignment.
        diff = (idx_scaled.unsqueeze(-1) - kv.view(1, 1, -1)).abs()
        q = kv[diff.argmin(dim=-1)]
        sumqx = (w2 * q * blocks).sum(dim=-1, keepdim=True)
        sumq2 = (w2 * q * q).sum(dim=-1, keepdim=True)
        return sumqx, sumq2

    # seed: ntry > 0 -> d = -max/values[0]
    d0 = -mx / v0
    id0 = torch.where(d0 != 0, 1.0 / d0, torch.zeros_like(d0))
    sumqx, sumq2 = _best(id0 * blocks)
    d = torch.where(sumq2 > 0, sumqx / sumq2, d0)
    best = d * sumqx

    for itry in range(-ntry, ntry + 1):
        idt = torch.where(mx != 0, (itry + v0) / mx, torch.zeros_like(mx))
        sumqx, sumq2 = _best(idt * blocks)
        cond = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        d_new = torch.where(sumq2 > 0, sumqx / sumq2, d)
        d = torch.where(cond, d_new, d)
        best = torch.where(cond, d_new * sumqx, best)

    d = _round_to_fp16(d)
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    diff = (inv * blocks).unsqueeze(-1) - kv.view(1, 1, -1)
    q = kv[diff.abs().argmin(dim=-1)]
    deq = torch.where(nonzero, d * q, torch.zeros_like(blocks))
    return _from_blocks(deq, shape, n)


def _fq_iq4_nl(w: torch.Tensor) -> torch.Tensor:
    """IQ4_NL fake-quant, applied twice so the result is a fixed point.

    A single ``_fq_iq4_nl_once`` matches libggml almost exactly, but the
    ``ntry``-step scale refinement makes it not a one-step fixed point — the
    refined scale can move slightly when the (quantized) input changes the
    objective. The map converges after the second application, so the kernel
    projects onto that fixed point; the result still matches the libggml
    round-trip to ~5e-4 and ``fq(fq(w)) == fq(w)`` holds exactly."""
    return _fq_iq4_nl_once(_fq_iq4_nl_once(w))


# ── Registry + dispatcher ────────────────────────────────────────────────────

SCHEME_FAKE_QUANT: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "BF16": _fq_bf16,
    "F16": _fq_f16,
    "F32": _fq_f32,
    "Q8_0": _fq_q8_0,
    "MXFP4": _fq_mxfp4,
    "Q4_K": _fq_q4_k,
    "Q6_K": _fq_q6_k,
    "Q5_K": _fq_q5_k,
    "Q3_K": _fq_q3_k,
    "Q2_K": _fq_q2_k,
    "IQ4_NL": _fq_iq4_nl,
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
