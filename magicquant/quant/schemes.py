"""
Quantization Schemes - Single source of truth for scheme metadata.

Each QuantizationScheme carries all attributes consumers (predictor, survival,
probing, orchestrator) need. This is the canonical registry — no other module
should hold parallel scheme dicts.

Bits-per-weight values are computed from the actual ggml block format:
  bpw = (block_bytes * 8) / block_elements

Noise factors are HEURISTIC values, ordered to match the published
perplexity ranking across Llama / Qwen / Mistral architectures. They are
NOT yet empirically calibrated: running tools/calibrate_noise_factors.py
(requires llama.cpp + a calibration model, ~2 hr compute) produces
tools/calibration_results.json, after which these values should be replaced
with the measured ppl_loss ratios and the refactor-regression fixture
regenerated (it is seed-pinned to the exact noise values here).
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


SchemeCategory = Literal["k_quant", "iq_quant", "legacy_q", "float", "mxfp4", "rocmfpx"]


@dataclass(frozen=True)
class QuantizationScheme:
    """Canonical metadata for one quantization scheme.

    Consumers read attributes off this class instead of maintaining parallel
    lookup dicts. New schemes are added by appending to the registry below;
    no consumer-side changes required for static metadata.
    """

    name: str                         # MagicQuant identifier ("Q4_K_M", "MXFP4_MOE", ...)
    ggml_type_name: str               # ggml block type ("Q4_K", "MXFP4", ...)
    ggml_type_id: int                 # numeric ggml type enum (used by the ctypes binding)
    bits_per_weight: float            # actual storage bpw from ggml block format
    noise_factor: float               # relative quantization noise (lower = better quality)
    speed_multiplier: float = 1.0     # relative inference speed vs BF16
    category: SchemeCategory = "k_quant"
    requires_imatrix: bool = False    # IQ-quants benefit from importance matrices
    # Whether ggml's `quantize_<type>` function actually reads the imatrix
    # pointer when one is passed to `ggml_quantize_chunk` -- distinct from
    # `requires_imatrix` above, which is about whether the type can run
    # WITHOUT one at all. A scheme can consume an imatrix without requiring
    # it (e.g. Q4_K, IQ4_XS: better with one, fine without) or ignore an
    # imatrix it's handed entirely (MXFP4, Q8_0, float passthroughs -- their
    # quantize_* functions take a `quant_weights` argument and immediately
    # discard it, `(void)quant_weights;`/`GGML_UNUSED(quant_weights);` --
    # verified against ggml/src/ggml-quants.c). True for every k_quant and
    # iq_quant scheme; False for float/legacy_q/mxfp4/rocmfpx.
    uses_imatrix: bool = False
    upgrade_neighbor: Optional[str] = None    # name of next-better scheme
    downgrade_neighbor: Optional[str] = None  # name of next-smaller scheme

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to BF16 (16 bits)."""
        return 16.0 / self.bits_per_weight

    def __repr__(self) -> str:
        return f"QuantScheme({self.name}, {self.bits_per_weight}bpw, noise={self.noise_factor})"


# ── Registry ─────────────────────────────────────────────────────────
# NOTE: ggml_type_id values are verified against ggml.h and cross-checked at
# startup by ggml_binding._verify_type_ids (the ctypes binding uses these IDs).

BF16 = QuantizationScheme(
    name="BF16",
    ggml_type_name="BF16",
    ggml_type_id=30,
    bits_per_weight=16.0,
    noise_factor=0.0,
    speed_multiplier=1.0,
    category="float",
    upgrade_neighbor=None,
    downgrade_neighbor="Q8_0",
)

Q8_0 = QuantizationScheme(
    name="Q8_0",
    ggml_type_name="Q8_0",
    ggml_type_id=8,
    bits_per_weight=8.5,
    noise_factor=1.0,
    speed_multiplier=1.75,
    category="legacy_q",
    upgrade_neighbor="BF16",
    downgrade_neighbor="Q6_K",
)

Q6_K = QuantizationScheme(
    name="Q6_K",
    ggml_type_name="Q6_K",
    ggml_type_id=14,
    bits_per_weight=6.5625,
    noise_factor=2.2,
    speed_multiplier=2.2,
    category="k_quant",
    uses_imatrix=True,
    upgrade_neighbor="Q8_0",
    downgrade_neighbor="Q5_K",
)

Q5_K = QuantizationScheme(
    name="Q5_K",
    ggml_type_name="Q5_K",
    ggml_type_id=13,
    bits_per_weight=5.5,
    noise_factor=3.0,
    speed_multiplier=2.7,
    category="k_quant",
    uses_imatrix=True,
    upgrade_neighbor="Q6_K",
    downgrade_neighbor="IQ4_NL",
)

# IQ4_NL: Non-linear lookup table optimized for weight distributions.
# WITH an imatrix active, this genuinely has lower noise than Q4_K_M despite
# the same bpw, because the 16 levels are learned to minimize quantization
# error on real weight distributions -- backing noise_factor=3.8 below.
# WITHOUT calibration this claim does not hold: see the
# IMATRIX_DEPENDENT_SCHEME_NAMES section later in this file, where the same
# lookup table -- optimal for unweighted reconstruction error -- loses every
# measured unweighted comparison against Q5_K/Q6_K/MXFP4_MOE.
#
# speed_multiplier=3.2 (near-parity with Q4_K_M's 3.4) does NOT match real
# hardware: llama-bench on this box (fork, -ngl 99, pp512/tg64, from the
# Qwopus3.6-27B run) measured an IQ4_NL-dominant hybrid at 122.8 pp512 vs
# 231.6 for stock Q4_K_M -- roughly HALF the prompt-processing throughput
# (ratio ~0.53), not the ~94% the current values imply. This value feeds
# the seed-pinned evolution fixture (PredictiveScorer.predict_tps ->
# score_hybrid -> survival.py selection) directly, so it is NOT changed
# here -- see magicquant/quant/calibration.py's calibrated_speed_multiplier
# for the opt-in route a real calibration file would use instead.
IQ4_NL = QuantizationScheme(
    name="IQ4_NL",
    ggml_type_name="IQ4_NL",
    ggml_type_id=20,
    bits_per_weight=4.5,
    noise_factor=3.8,
    speed_multiplier=3.2,
    category="iq_quant",
    uses_imatrix=True,
    upgrade_neighbor="Q5_K",
    downgrade_neighbor="MXFP4_MOE",
)

# MXFP4: OCP MX Microscaling FP4 (E2M1 values + shared E8M0 exponent).
# Non-uniform FP4 levels (0, 0.5, 1, 1.5, 2, 3, 4, 6) are denser near
# zero, naturally matching the Gaussian-like weight distribution of
# transformers. Lower noise than integer Q4 at slightly better compression.
#
# speed_multiplier=3.8 is UNCHANGED (same fixture constraint as IQ4_NL
# above), but note for context: the real bench data above didn't include a
# pure MXFP4-dominant hybrid, so this value hasn't been directly checked
# against hardware the way Q4_K_M/IQ4_XS/IQ4_NL have been.
MXFP4_MOE = QuantizationScheme(
    name="MXFP4_MOE",
    ggml_type_name="MXFP4",
    ggml_type_id=39,
    bits_per_weight=4.25,
    noise_factor=4.0,
    speed_multiplier=3.8,
    category="mxfp4",
    upgrade_neighbor="IQ4_NL",
    downgrade_neighbor="Q4_K_M",
)

# speed_multiplier=3.4 is our reference anchor for the real-hardware ratios
# documented on IQ4_NL/IQ4_XS/MXFP4_MOE above/below: real llama-bench (fork,
# -ngl 99, pp512, Qwopus3.6-27B run) measured stock Q4_K_M at 231.6 pp512 --
# the 100% baseline those other schemes' real/registry ratios are quoted
# against. Also fixture-pinned; not changed here.
Q4_K_M = QuantizationScheme(
    name="Q4_K_M",
    ggml_type_name="Q4_K",
    ggml_type_id=12,
    bits_per_weight=4.5,
    noise_factor=4.5,
    speed_multiplier=3.4,
    category="k_quant",
    uses_imatrix=True,
    upgrade_neighbor="MXFP4_MOE",
    downgrade_neighbor="Q3_K",
)


# ── Q3_K / Q2_K ──────────────────────────────────────────────────────
# Q3_K and Q2_K make the Q3 tier band reachable. Q2_K bpw=2.625 lands
# at ratio 0.164 — just outside the Q2 band (<=0.16); full Q2 band
# coverage requires sub-Q2 IQ-quants from PR3.
#
# noise_factor values are HEURISTIC (ordered above Q4_K_M), not yet
# calibrated — see the module docstring and tools/calibrate_noise_factors.py.

Q3_K = QuantizationScheme(
    name="Q3_K",
    ggml_type_name="Q3_K",
    ggml_type_id=11,
    bits_per_weight=3.4375,   # 110B * 8 / 256 = 3.4375
    noise_factor=8.0,         # heuristic; calibration pending
    speed_multiplier=4.0,     # ggml SIMD encoders are fast
    category="k_quant",
    uses_imatrix=True,
    upgrade_neighbor="Q4_K_M",
    downgrade_neighbor="Q2_K",
)

Q2_K = QuantizationScheme(
    name="Q2_K",
    ggml_type_name="Q2_K",
    ggml_type_id=10,
    bits_per_weight=2.625,    # 84B * 8 / 256 = 2.625
    noise_factor=15.0,        # heuristic; calibration pending
    speed_multiplier=4.5,     # smallest blocks → fastest dispatch
    category="k_quant",
    uses_imatrix=True,
    upgrade_neighbor="Q3_K",
    downgrade_neighbor=None,  # bottom of current registry; PR3 adds IQ-quants below
)


# ── Legacy (non-K) Q4 quants ─────────────────────────────────────────
# Stock-ggml pre-K-quant block=32 legacy types (llama.cpp's original Q4_0/
# Q4_1, predating the block=256 K-quant family). v2-ONLY: excluded from v1's
# random-config sampling pool (see LEGACY_Q4_SCHEME_NAMES below and
# survival.py's _generate_random_config) so the default evolutionary search
# -- and its seed-pinned regression fixture -- stays byte-identical. These
# exist in the registry for v2's explicit per-tensor scheme-override
# allocation (writer.py create_hybrid_gguf's "tensors" override key), not
# for the evolutionary sampler.
#
# bpw from the real ggml block struct (block=32, matches
# ggml_binding._GGML_BLOCK_SIZE/_GGML_TYPE_SIZE):
#   Q4_0 = 18B/32 = 4.5 bpw (4-bit codes + 1 f16 scale, no min term)
#   Q4_1 = 20B/32 = 5.0 bpw (4-bit codes + f16 scale + f16 min term)
# noise_factor values are HEURISTIC, slotted just below Q4_K_M (4.5) and
# above Q3_K (8.0) -- legacy Q4 lacks the K-quant super-block min/scale
# refinement, so it's noisier than Q4_K_M at the same or slightly higher
# bpw; Q4_1's extra min term makes it slightly cleaner than Q4_0.
# Calibration pending, like the rest of the registry (see module docstring).
#
# upgrade_neighbor=None for both, and no EXISTING scheme's upgrade_neighbor
# is changed to point at them -- the default mutation (Protector/Crusher)
# neighbor-walk can never reach these from any v1-reachable scheme.
#
# uses_imatrix=True, verified against ggml rather than assumed from the
# "legacy" label. ggml_quantize_chunk routes GGML_TYPE_Q4_0/Q4_1 to
# quantize_q4_0/quantize_q4_1, which fall through to the unweighted _ref
# encoders ONLY under `if (!quant_weights)`; given a non-NULL imatrix they
# call quantize_row_q4_0_impl / _q4_1_impl (ggml-quants.c:1893/1936), which
# build weight[j] = qw[j] * sqrtf(sigma2 + x[j]^2) and feed
# make_qx_quants / make_qkx3_quants -- the same weighted machinery the
# K-quants use. Measured through the live libggml: 159/227 differing bytes
# and weighted MSE -22.7% / -39.9% with an imatrix versus without.
#
# This was registered False until 2026-08. Nothing shipped wrong -- the
# encoder passes the imatrix through unconditionally, so the BYTES were
# always weighted; `uses_imatrix` has exactly one functional consumer,
# effective_noise_factor, reached only from PredictiveScorer, which cannot
# see these schemes while they stay out of v1 sampling. It was wrong
# metadata one line away from mattering. tests/test_legacy_q4_schemes.py
# pinned the wrong value; the registry-wide behavioural cross-check in
# tests/test_uses_imatrix_matches_ggml.py is what keeps it honest now.

Q4_0 = QuantizationScheme(
    name="Q4_0",
    ggml_type_name="Q4_0",
    ggml_type_id=2,
    bits_per_weight=4.5,
    noise_factor=5.0,
    speed_multiplier=3.4,   # ~parity with Q4_K_M; both are 4-bit block quants
    category="legacy_q",
    uses_imatrix=True,      # see the imatrix note above the Q4_0 block
    requires_imatrix=False,
    upgrade_neighbor=None,
    downgrade_neighbor=None,
)

Q4_1 = QuantizationScheme(
    name="Q4_1",
    ggml_type_name="Q4_1",
    ggml_type_id=3,
    bits_per_weight=5.0,
    noise_factor=4.7,
    speed_multiplier=3.4,   # ~parity with Q4_K_M; both are 4-bit block quants
    category="legacy_q",
    uses_imatrix=True,      # see the imatrix note above the Q4_0 block
    requires_imatrix=False,
    upgrade_neighbor=None,
    downgrade_neighbor=None,
)


# ── Block-32 Q5 quants ───────────────────────────────────────────────
# The reason these exist: a model whose rows are not 256-divisible sends
# every block-256 scheme through the writer's block-size fallback, so the
# achievable sizes collapse to whatever the block-32 family offers. That
# family ran Q4_1 (5.0 bpw) -> Q8_0 (8.5 bpw) with nothing between, which
# is a hole exactly wide enough to swallow the Q5 and Q6 tier bands. On
# nemotron_h_moe (hidden_size 2688, moe_intermediate_size 1856 -- neither
# 256-divisible) both bands came out structurally EMPTY: nothing could land
# between 19.91 GiB and 32.54 GiB. Q5_0 at 5.5 bpw lands inside Q5.
#
# bpw from the real ggml block structs (ggml-common.h):
#   Q5_0 = 22B/32 = 5.5 bpw  (block_q5_0: d + qh[4] + qs[16])
#   Q5_1 = 24B/32 = 6.0 bpw  (block_q5_1: d + m + qh[4] + qs[16])
#
# uses_imatrix=True, verified in ggml-quants.c the same way Q4_0/Q4_1 were
# (see the note above Q4_0): quantize_q5_0/quantize_q5_1 fall through to the
# unweighted _ref encoders only under `if (!quant_weights)`; with an imatrix
# they call quantize_row_q5_0_impl (make_qx_quants) / _q5_1_impl
# (make_qkx3_quants). Confirmed behaviourally by
# tests/test_uses_imatrix_matches_ggml.py, which encodes every registry
# scheme with and without an imatrix and asserts the bytes differ iff this
# flag says they should. Do NOT copy Q4_0's pre-2026-08 value here.
#
# noise_factor: measured, not interpolated by eye. Encode->decode through
# the real libggml and take the imatrix-weighted squared error relative to
# Q8_0 (v2.sensitivity._weighted_sq_err -- the same metric v2 prices tensors
# with), then interpolate on ln(ratio) between the two bracketing anchors
# already in this registry, Q5_K (3.00) and Q4_K_M (4.50):
#     Q5_1  ratio  48.6 -> 3.10        Q5_0  ratio  63.4 -> 3.39
# Measured on two models -- nemotron_h_moe 30B-A3B and a 235GB bf16 -- whose
# ratios agreed within 4% for every scalar block quant. The CONTROL that
# makes those numbers usable: recovering Q6_K by the same interpolation from
# only Q8_0/Q5_K gives 2.26 against its registered 2.20. A global log-linear
# fit across the whole 8.5->2.6 bpw range is NOT good (RMS 2.17), so this is
# an interpolation between tight anchors and must not be extrapolated.
#
# Two schemes were deliberately excluded from that fit, and the exclusion is
# the point rather than a convenience: MXFP4's ratio moved 441 -> 1072
# between the two models while every scalar quant moved <5%, and IQ4_NL's
# reconstruction error famously does not track its perplexity (see the
# imatrix invariants in CLAUDE.md). Reconstruction-error pricing is only
# defensible WITHIN the scalar block-quant family, which is what Q5_0/Q5_1
# are. Independent unweighted rel-RMS derivations landed on 3.3/3.1, i.e.
# within 0.1 of the above.
#
# Sanity of the resulting order, which is the real test: Q5_0 (5.5 bpw,
# legacy) at 3.40 is noisier than Q5_K (5.5 bpw, K-quant) at 3.00 -- the
# same "legacy loses to the K-quant at equal bpw" relationship the registry
# already encodes for Q4_0 (5.00) vs Q4_K_M (4.50). And neither is dominated
# by MXFP4 (4.25 bpw / 4.00), which is what makes the search able to pick
# them at all.
#
# speed_multiplier is NOT derived -- 2.7 is slotted between Q6_K and Q4_K_M
# by bpw. Nothing in this registry has a benchmarked speed multiplier; do not
# single these two out for correction.
#
# REACHABILITY: like Q4_0/Q4_1 these are orphans in the STATIC chain, and no
# existing scheme's neighbour is re-pointed at them here. They become
# reachable only through EvolutionarySurvivor's runtime overlay, and only
# for groups whose tensors are block-32-only on the model actually being
# searched. A static re-point would change the neighbour walk for every
# model, including the 256-divisible ones where Q5_K is strictly better than
# Q5_0 at identical bpw.

Q5_0 = QuantizationScheme(
    name="Q5_0",
    ggml_type_name="Q5_0",
    ggml_type_id=6,
    bits_per_weight=5.5,    # 22 B/block * 8 / 32
    noise_factor=3.4,
    speed_multiplier=2.7,   # not benchmarked -- see note above
    category="legacy_q",
    uses_imatrix=True,
    requires_imatrix=False,
    upgrade_neighbor="Q5_1",
    downgrade_neighbor="Q5_K",
)

Q5_1 = QuantizationScheme(
    name="Q5_1",
    ggml_type_name="Q5_1",
    ggml_type_id=7,
    bits_per_weight=6.0,    # 24 B/block * 8 / 32
    noise_factor=3.1,
    speed_multiplier=2.7,   # not benchmarked -- see note above
    category="legacy_q",
    uses_imatrix=True,
    requires_imatrix=False,
    upgrade_neighbor="Q6_K",
    downgrade_neighbor="Q5_0",
)


# ── NVFP4 ────────────────────────────────────────────────────────────
# NVIDIA FP4: block 64 with a UE4M3 scale per 16-element sub-block
# (QK_NVFP4=64, QK_NVFP4_SUB=16, block_nvfp4 = 4 + 32 = 36 B -> 4.5 bpw).
# Upstream ggml type 40, not a fork type.
#
# OPT-IN, default off (ENABLE via enable_nvfp4). Three separate reasons, all
# verified on this box rather than assumed:
#
#  1. uses_imatrix=False, and it means it. quantize_nvfp4 opens with
#     GGML_UNUSED(quant_weights) and calls quantize_row_nvfp4_ref, whose
#     signature (x, y, k) has no weights parameter at all. Unlike Q2_0/Q1_0 --
#     which merely LOOK imatrix-aware because they branch on quant_weights
#     before calling the unweighted kernel in both arms -- NVFP4 does not even
#     pretend. An uncalibrated scheme competing in an imatrix-on search is the
#     exact situation that cost IQ4_NL all 11 of its measured comparisons.
#
#  2. The bound libggml can ENCODE it but not DECODE it (dequantize_row_nvfp4
#     is absent from this build; verified live). So v2's distortion table
#     cannot price it, and the encode->decode method used to derive the
#     Q5_0/Q5_1 noise factors cannot measure it here either. noise_factor 3.7
#     is therefore NOT locally verified -- it comes from an external
#     derivation and should be treated as provisional until this build gains
#     a dequant path.
#
#  3. The llama-perplexity on PATH (linuxbrew b7090) has no nvfp4 symbols, so
#     it cannot LOAD an NVFP4 GGUF. A measured search that selects NVFP4 will
#     fail at measurement time on the default runtime. gfx1151 compute kernels
#     do exist in the newer checkout, so this is a tooling gap, not a hardware
#     one -- but it is why this must not be reachable by default.
#
# Orphaned in the chain (no neighbours, nothing points at it) for the same
# reason as Q4_0/Q4_1: reachability is opt-in, never incidental.
NVFP4 = QuantizationScheme(
    name="NVFP4",
    ggml_type_name="NVFP4",
    ggml_type_id=40,
    bits_per_weight=4.5,    # 36 B/block * 8 / 64
    noise_factor=3.7,       # provisional -- see (2) above
    speed_multiplier=3.5,   # not benchmarked
    category="legacy_q",
    uses_imatrix=False,
    requires_imatrix=False,
    upgrade_neighbor=None,
    downgrade_neighbor=None,
)


# ── ROCmFPX fork schemes ─────────────────────────────────────────────
# AMD-native (gfx1151-tuned) tensor formats from the ROCmFPX llama.cpp fork
# (https://github.com/ciru-ai/ROCmFPX). These are OPT-IN: the search only
# considers them when explicitly enabled AND the bound libggml is a ROCmFPX
# build that can encode them (gated in survival.py via the predictor's
# available-scheme set). GGUFs containing these types load only on the fork,
# not stock llama.cpp — documented as a hard caveat.
#
# bpw is the true storage cost from the fork's block structs (block=32):
#   fp3 = 14 B/blk → 3.5 bpw; fp4 = 18 → 4.5; fp6 = 26 → 6.5; fp8 = 33 → 8.25.
# noise_factor values are HEURISTIC, slotted next to the K-quant of matching
# bpw (calibration pending, like Q3_K/Q2_K).
#
# uses_imatrix left at the False default: the fork's quantize_* internals
# for these types haven't been independently verified to consume an
# imatrix the way stock ggml's K/IQ-quants do (see the `uses_imatrix` field
# docstring above), so we don't credit them the imatrix noise discount
# without evidence either way.

ROCMFP8 = QuantizationScheme(
    name="ROCMFP8",
    ggml_type_name="Q8_0_ROCMFPX",
    ggml_type_id=103,
    bits_per_weight=8.25,
    noise_factor=1.05,          # ~Q8_0
    speed_multiplier=1.8,
    category="rocmfpx",
    upgrade_neighbor="BF16",
    downgrade_neighbor="ROCMFP6",
)

ROCMFP6 = QuantizationScheme(
    name="ROCMFP6",
    ggml_type_name="Q6_0_ROCMFPX",
    ggml_type_id=102,
    bits_per_weight=6.5,
    noise_factor=2.3,           # ~Q6_K
    speed_multiplier=2.3,
    category="rocmfpx",
    upgrade_neighbor="ROCMFP8",
    downgrade_neighbor="ROCMFP4",
)

ROCMFP4 = QuantizationScheme(
    name="ROCMFP4",
    ggml_type_name="Q4_0_ROCMFP4",
    ggml_type_id=100,
    bits_per_weight=4.5,
    noise_factor=4.2,           # ~Q4_K_M / MXFP4 band
    speed_multiplier=3.8,       # AMD-native FP4 kernel is the fork's fast path
    category="rocmfpx",
    upgrade_neighbor="ROCMFP6",
    downgrade_neighbor="ROCMFP3",
)

ROCMFP3 = QuantizationScheme(
    name="ROCMFP3",
    ggml_type_name="Q3_0_ROCMFPX",
    ggml_type_id=104,
    bits_per_weight=3.5,
    noise_factor=8.5,           # ~Q3_K
    speed_multiplier=4.0,
    category="rocmfpx",
    upgrade_neighbor="ROCMFP4",
    downgrade_neighbor=None,
)


# ── Sub-4-bit IQ-quants ──────────────────────────────────────────────
# Stock-ggml importance-matrix-friendly non-linear quants below IQ4_NL.
# OPT-IN: excluded from the default search pool (see IQ_SCHEME_NAMES below
# and the enable_iq gate in evolution/survival.py) so the default search —
# and its seed-pinned regression fixture — is byte-identical to today.
#
# bpw is the true storage cost from the binding's block/type tables
# (magicquant/quant/ggml_binding.py _GGML_BLOCK_SIZE / _GGML_TYPE_SIZE; all
# block=256 except IQ4_NL's block=32, which is unaffected here):
#   IQ4_XS  = 136B/256  = 4.25   bpw
#   IQ3_S   = 110B/256  = 3.4375 bpw
#   IQ3_XXS =  98B/256  = 3.0625 bpw
#   IQ2_S   =  82B/256  = 2.5625 bpw
#   IQ2_XS  =  74B/256  = 2.3125 bpw
#   IQ2_XXS =  66B/256  = 2.0625 bpw
#   IQ1_M   =  56B/256  = 1.75   bpw
#   IQ1_S   =  50B/256  = 1.5625 bpw
# noise_factor values are HEURISTIC (ordered by published llama.cpp quality),
# consistent with the existing heuristic Q3_K=8.0/Q2_K=15.0 (calibration
# pending, see the module docstring).
#
# requires_imatrix is read from the REAL libggml
# (ggml_quantize_requires_imatrix), not guessed: verified against a stock
# (non-fork) libggml at ~/llama.cpp/build/bin — IQ2_XS, IQ2_XXS, and IQ1_S
# report True; IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, and IQ1_M report False.
# survival.py's enable_iq gate additionally drops every requires_imatrix
# scheme unconditionally (the search threads no imatrix).
#
# CRITICAL: only these NEW schemes reference existing registry entries
# (upward, via upgrade/downgrade_neighbor) — no existing scheme's neighbor
# fields are modified, so the default mutation chain is unchanged.
#
# speed_multiplier: IQ4_XS=3.3 (near-parity with Q4_K_M's 3.4) does NOT match
# real hardware either -- same real llama-bench run referenced on IQ4_NL
# above measured an IQ4_XS-dominant hybrid at 119.7 pp512 vs Q4_K_M's 231.6
# (ratio ~0.52, i.e. roughly HALF, not ~97%). IQ kernels are consistently
# ~half Q4_K's prompt-processing speed on this ROCm hardware; a real
# calibration would proportionally scale the rest of the IQ family
# (IQ3_S/IQ3_XXS/IQ2_S/IQ2_XS/IQ2_XXS/IQ1_M/IQ1_S below) down from their
# current near-Q4_K_M speed_multipliers by roughly the same ~0.5 ratio. None
# of these values are changed here (fixture constraint, see Q4_K_M/IQ4_NL
# above); route real corrections through calibration.calibrated_speed_multiplier.

IQ4_XS = QuantizationScheme(
    name="IQ4_XS",
    ggml_type_name="IQ4_XS",
    ggml_type_id=23,
    bits_per_weight=4.25,
    noise_factor=4.1,
    speed_multiplier=3.3,
    category="iq_quant",
    requires_imatrix=False,
    uses_imatrix=True,
    upgrade_neighbor="MXFP4_MOE",
    downgrade_neighbor="IQ3_S",
)

IQ3_S = QuantizationScheme(
    name="IQ3_S",
    ggml_type_name="IQ3_S",
    ggml_type_id=21,
    bits_per_weight=3.4375,
    noise_factor=6.0,
    speed_multiplier=3.5,
    category="iq_quant",
    requires_imatrix=False,
    uses_imatrix=True,
    upgrade_neighbor="IQ4_XS",
    downgrade_neighbor="IQ3_XXS",
)

IQ3_XXS = QuantizationScheme(
    name="IQ3_XXS",
    ggml_type_name="IQ3_XXS",
    ggml_type_id=18,
    bits_per_weight=3.0625,
    noise_factor=7.5,
    speed_multiplier=3.6,
    category="iq_quant",
    requires_imatrix=False,
    uses_imatrix=True,
    upgrade_neighbor="IQ3_S",
    downgrade_neighbor="Q3_K",
)

IQ2_S = QuantizationScheme(
    name="IQ2_S",
    ggml_type_name="IQ2_S",
    ggml_type_id=22,
    bits_per_weight=2.5625,
    noise_factor=11.0,
    speed_multiplier=3.7,
    category="iq_quant",
    requires_imatrix=False,
    uses_imatrix=True,
    upgrade_neighbor="IQ3_XXS",
    downgrade_neighbor="IQ2_XS",
)

IQ2_XS = QuantizationScheme(
    name="IQ2_XS",
    ggml_type_name="IQ2_XS",
    ggml_type_id=17,
    bits_per_weight=2.3125,
    noise_factor=13.0,
    speed_multiplier=3.8,
    category="iq_quant",
    requires_imatrix=True,
    uses_imatrix=True,
    upgrade_neighbor="IQ2_S",
    downgrade_neighbor="IQ2_XXS",
)

IQ2_XXS = QuantizationScheme(
    name="IQ2_XXS",
    ggml_type_name="IQ2_XXS",
    ggml_type_id=16,
    bits_per_weight=2.0625,
    noise_factor=16.0,
    speed_multiplier=3.9,
    category="iq_quant",
    requires_imatrix=True,
    uses_imatrix=True,
    upgrade_neighbor="IQ2_XS",
    downgrade_neighbor="IQ1_M",
)

IQ1_M = QuantizationScheme(
    name="IQ1_M",
    ggml_type_name="IQ1_M",
    ggml_type_id=29,
    bits_per_weight=1.75,
    noise_factor=24.0,
    speed_multiplier=3.95,
    category="iq_quant",
    requires_imatrix=False,
    uses_imatrix=True,
    upgrade_neighbor="IQ2_XXS",
    downgrade_neighbor="IQ1_S",
)

IQ1_S = QuantizationScheme(
    name="IQ1_S",
    ggml_type_name="IQ1_S",
    ggml_type_id=19,
    bits_per_weight=1.5625,
    noise_factor=30.0,
    speed_multiplier=4.0,
    category="iq_quant",
    requires_imatrix=True,
    uses_imatrix=True,
    upgrade_neighbor="IQ1_M",
    downgrade_neighbor=None,
)


_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
    "Q3_K": Q3_K,
    "Q2_K": Q2_K,
    "Q4_0": Q4_0,
    "Q4_1": Q4_1,
    "Q5_0": Q5_0,
    "Q5_1": Q5_1,
    "NVFP4": NVFP4,
    "ROCMFP8": ROCMFP8,
    "ROCMFP6": ROCMFP6,
    "ROCMFP4": ROCMFP4,
    "ROCMFP3": ROCMFP3,
    "IQ4_XS": IQ4_XS,
    "IQ3_S": IQ3_S,
    "IQ3_XXS": IQ3_XXS,
    "IQ2_S": IQ2_S,
    "IQ2_XS": IQ2_XS,
    "IQ2_XXS": IQ2_XXS,
    "IQ1_M": IQ1_M,
    "IQ1_S": IQ1_S,
}

# ROCmFPX fork scheme names (opt-in; excluded from the default search pool).
ROCMFPX_SCHEME_NAMES = frozenset({"ROCMFP8", "ROCMFP6", "ROCMFP4", "ROCMFP3"})

# Legacy Q4_0/Q4_1 scheme names (v2-only; excluded from v1's random-config
# sampling pool -- see the "Legacy (non-K) Q4 quants" section above and
# survival.py's _generate_random_config). Reserved for explicit per-tensor
# scheme overrides (writer.py's "tensors" key), not the evolutionary sampler.
LEGACY_Q4_SCHEME_NAMES = frozenset({"Q4_0", "Q4_1"})

# Block-32 Q5 scheme names. Like LEGACY_Q4_SCHEME_NAMES these are kept out of
# v1's default sampling pool, but for a narrower reason: they are added back
# for a specific GROUP when that group's tensors on THIS model are
# block-32-only (32-divisible rows that aren't 256-divisible), which is the
# only situation where they beat the K-quant of the same bpw. On an ordinary
# 256-divisible model Q5_K is strictly better than Q5_0 at identical 5.5 bpw,
# so making them globally reachable would make ordinary searches worse -- and
# would churn the seed-pinned fixture. See EvolutionarySurvivor's
# block32_only_groups and _BLOCK32_Q5_CHAIN_OVERLAY.
BLOCK32_Q5_SCHEME_NAMES = frozenset({"Q5_0", "Q5_1"})

# NVFP4: opt-in only (enable_nvfp4). Excluded from the default sampling pool
# for the reasons in the NVFP4 block above -- imatrix-blind, unpriceable on
# this build (no dequant), and unloadable by the llama-perplexity on PATH.
NVFP4_SCHEME_NAMES = frozenset({"NVFP4"})

# Sub-4-bit IQ scheme names (opt-in; excluded from the default search pool).
# Deliberately does NOT include IQ4_NL, which is gated separately by
# IMATRIX_DEPENDENT_SCHEME_NAMES below -- on imatrix availability rather than
# the enable_iq opt-in, since its problem is calibration, not bit width.
IQ_SCHEME_NAMES = frozenset({
    "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
})

# Schemes that are only COMPETITIVE with an importance matrix. Distinct from
# `requires_imatrix` (cannot encode at all without one) and from `uses_imatrix`
# (will consume one if offered -- true of every K-quant, all of which are fine
# without). These encode fine unweighted, they just lose, so they are dropped
# from the search pool when no imatrix is in play: every candidate spent on one
# is a wasted measurement.
#
# IQ4_NL, measured 2026-08-03 over 11 candidates on two 27B models with
# use_imatrix=false -- ThinkingCap on GPU and FableFusion on CPU, so neither a
# kernel nor a hardware artifact:
#
#     U=Q5_K       0.0036             U=MXFP4_MOE  0.0009, 0.0184
#     U=Q6_K       0.0021, 0.0075     U=IQ4_NL     0.0149, 0.0223, 0.0240,
#                                                  0.0335, 0.0336
#
# It never won a comparison. The cleanest case had IQ4_NL as the ONLY low-bit
# group (loss 0.0149) losing 16x to a config with TWO low-bit groups (0.0009).
# Ruled out: corrupt block/type metadata (all match the upstream gguf package),
# a bad HIP kernel (CPU and GPU agree), and config confounding (case above).
#
# Cause: its non-linear lookup places levels to minimise UNWEIGHTED error when
# no imatrix is supplied. In isolation that wins the metric it optimises --
# IQ4_NL round-trips real ffn_up weights at 0.051 relative RMS vs MXFP4's
# 0.101 -- and loses badly on the one that matters. The "Lower noise than
# Q4_K_M" comment above the IQ4_NL definition and its noise_factor=3.8 now
# carry an explicit imatrix qualifier for exactly this reason -- both were
# silently assuming calibration before that qualifier was added.
IMATRIX_DEPENDENT_SCHEME_NAMES = frozenset({"IQ4_NL"})

# Group-class floors: minimum acceptable scheme per group class.
# "sensitive" (E, H, O, R) shouldn't go below Q8_0; "robust" (U, D, X)
# can go all the way to Q4_K_M (bottom of current registry).
# These were previously in survival.py as _MIN_SCHEME.
_GROUP_CLASS_FLOORS: Dict[str, str] = {
    "sensitive": "Q8_0",
    "robust": "Q4_K_M",
}


def get_scheme_by_name(name: str) -> QuantizationScheme:
    """Look up a scheme by its MagicQuant identifier."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown scheme: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_all_schemes() -> List[QuantizationScheme]:
    """Get all schemes ordered by noise (best quality first)."""
    return sorted(_REGISTRY.values(), key=lambda s: s.noise_factor)


def get_schemes_by_category(category: SchemeCategory) -> List[QuantizationScheme]:
    """Get all schemes in a given category, ordered by noise (best first)."""
    return [s for s in get_all_schemes() if s.category == category]


def get_floor_for_group_class(group_class: str) -> str:
    """Get the minimum acceptable scheme name for a group sensitivity class.

    group_class: "sensitive" or "robust".
    """
    return _GROUP_CLASS_FLOORS[group_class]


# ── Imatrix-aware noise scaling ─────────────────────────────────────
#
# Ground truth from the completed Qwopus3.6-27B (qwen35 hybrid SSM+attention,
# MTP) measured search, run with imatrix ON
# (output/Qwopus3.6-27B-v2-MTP-GGUF/magicquant/search_results.json):
# predicted_loss overestimated measured_loss by 20-100x across all 7 measured
# configs (mean_abs_residual ~0.97), but NOT uniformly -- the config leaning
# on IQ4_XS/IQ4_NL in FFN-up (a `uses_imatrix=True` group) measured 26x below
# its prediction, the best (lowest) ratio of any config in the run, while
# configs leaning on MXFP4_MOE in the same slot measured 45-52x below theirs.
# MXFP4's ggml encoder was independently verified to IGNORE the imatrix
# pointer entirely (byte-identical output with/without one, 2026-07-04),
# while Q4_K/Q5_K/IQ4_XS/IQ4_NL's encoders read it and change their output.
# So under imatrix, IQ/K-quant groups get a real quality boost MXFP4 doesn't,
# and the static noise_factor table -- which has no imatrix concept -- can't
# reflect that; left alone, it misranks an IQ/K-quant group as no better than
# an MXFP4 group of similar nominal noise once an imatrix is active.
#
# IMATRIX_NOISE_SCALE=0.85 is a deliberately conservative (not curve-fit)
# correction: an imatrix-consuming scheme's *effective* noise, once an
# imatrix is active, is treated as 15% lower than its static registry value.
# It only nudges the predictor's relative ranking toward what the run showed
# (imatrix-aware K/IQ beats otherwise-equal MXFP4) rather than asserting a
# precise magnitude -- the 20-100x scale of the actual overestimate is a
# separate, much bigger problem than this ranking correction addresses (see
# tools/fit_noise_factors.py for per-scheme calibration from measured data).
IMATRIX_NOISE_SCALE = 0.85


def effective_noise_factor(
    scheme: QuantizationScheme,
    imatrix_active: bool,
    base_noise_factor: Optional[float] = None,
) -> float:
    """Return the noise factor to use for `scheme` given whether an imatrix
    is active for this search/build.

    `base_noise_factor` lets callers substitute an empirically calibrated
    value (magicquant.quant.calibration.calibrated_noise_factor) for the
    static `scheme.noise_factor` while still applying the same
    imatrix-awareness gating documented at ``IMATRIX_NOISE_SCALE`` above.
    Defaults to `scheme.noise_factor` when omitted.

    Schemes with `uses_imatrix=False` (MXFP4, rocmfpx, float, legacy Q8_0)
    are returned unscaled regardless of `imatrix_active`: their ggml encoders
    ignore the imatrix pointer, so an active imatrix changes nothing about
    their actual quantization noise.
    """
    base = scheme.noise_factor if base_noise_factor is None else base_noise_factor
    if imatrix_active and scheme.uses_imatrix:
        return base * IMATRIX_NOISE_SCALE
    return base
