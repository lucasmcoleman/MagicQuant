"""QAT for FUSED 3-D MoE expert parameters (``mlp.experts.gate_up_proj`` & co).

``magicquant.qat.wrap`` covers ``nn.Linear`` modules. Modern MoE architectures
don't store routed experts as Linears at all -- Qwen3.5/Qwen3.6, Llama4, GPT-OSS
and friends fuse every expert of a projection into ONE 3-D ``nn.Parameter``:

    model.language_model.layers.N.mlp.experts.gate_up_proj   [E, 2*I, H]
    model.language_model.layers.N.mlp.experts.down_proj      [E, H,   I]

On Qwen3.6-35B-A3B those two parameters per layer are ~93% of the model, so a
QAT run that only wraps Linears compensates the 7% that quantization barely
hurts and skips the experts that low-bit quantization crushes hardest.

Mechanism
---------
``torch.nn.utils.parametrize.register_parametrization`` on the fused Parameter.
The parametrization's ``forward(W)`` returns the QAT view of the weight, so
**every** consumer of ``module.gate_up_proj`` sees it -- the MoE forward is never
touched and never needs to know QAT is on. Trainable state is a per-expert LoRA
pair, batched over the leading expert axis:

    lora_expert_A  (E, W.shape[1], r)   zero-init
    lora_expert_B  (E, r, W.shape[2])   kaiming-init
    delta          = scaling * bmm(A, B)          # (E, W.shape[1], W.shape[2])
    scaling        = lora_alpha / lora_r

The A/B roles (which factor is zeroed) are inverted relative to ``QATLinear``
because the merge lane's on-disk contract fixes the operand order as
``W[e] += scale * (A[e] @ B[e])`` (``magicquant.qat.merge._apply_3d``). With
``lora_expert_A`` zeroed the delta still starts at exactly zero, which is the
property that matters. See ``_save_adapters`` in ``train.py`` for the key names.

Quantization modes
------------------
A fused expert parameter is enormous, and the fake-quant kernels are not cheap:
measured on this box's gfx1151 GPU, ``Q2_K`` runs at ~2.7 Melem/s and ``Q3_K``
at ~110 Melem/s. Qwen3.6-35B-A3B has ~33e9 expert elements, so re-quantizing
them all on every forward is ~90 minutes **per training step** -- not a tuning
problem, an infeasibility. Hence two modes:

``"live"`` (default)
    ``fake_quant(W + delta)`` every forward, exactly ``QATLinear``'s semantics:
    what you train is what ships, because the pack quantizes the merged weight
    too. Correct, validated, and the right choice whenever the expert tensors
    are small enough to afford it. ``estimate_expert_qat_cost`` reports what
    "afford" means for a given model before a run burns a night on it.

``"frozen"``
    Fake-quant the base ONCE at wrap time (in place, chunked, no_grad), then
    every forward is just ``W_q + delta``. Per-step cost collapses to one bmm.
    The honest caveat: the adapter's own delta is never re-quantized during
    training, while the shipped weight is ``quant(W_q + delta)``. At Q2_K/Q3_K a
    small delta can be largely rounded away at pack time, so frozen-mode
    recovery is *not* equivalent to live-mode recovery and has not been
    validated end to end. It exists because for a 35B MoE the alternative isn't
    "slower QAT", it's "no QAT at all".

Both modes are implemented by the same parametrization; ``mode`` only decides
whether the fake-quant happens per forward or once at wrap.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from magicquant.qat.fake_quant import fake_quant
from magicquant.qat.names import ExpertSegment, fused_expert_segments

# Schemes that mean "leave it alone" -- no quantization awareness wanted.
PASSTHROUGH_SCHEMES = frozenset({"BF16", "F16", "F32"})

MODE_LIVE = "live"
MODE_FROZEN = "frozen"
EXPERT_QUANT_MODES = (MODE_LIVE, MODE_FROZEN)

# Target element count for one fake-quant chunk. The kernels build several
# intermediates per element (MXFP4 materializes an 8-wide grid-distance tensor),
# so chunking the expert axis bounds peak memory independently of how many
# experts a layer has. 16M elements ~= 64 MB fp32 before the kernel's own
# temporaries.
_CHUNK_TARGET_ELEMS = 16 * 1024 * 1024


@dataclass(frozen=True)
class ResolvedSegment:
    """An :class:`ExpertSegment` with its ggml scheme resolved."""

    gguf_name: Optional[str]
    start: int
    stop: int
    ggml_type_name: str


# ── per-forward result cache ──────────────────────────────────────────────────
#
# The eager MoE forward reads the fused parameter ONCE PER HIT EXPERT:
#
#     for expert_idx in expert_hit:
#         ... nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
#
# With 256 experts and a 512-token batch essentially every expert is hit, so a
# parametrization with no cache is evaluated ~256 times per layer per forward --
# each evaluation rebuilding the whole [E, d1, d2] weight. That turns a 1-second
# bmm into a 4-minute one, and a live fake-quant into something that never
# finishes.
#
# torch's own ``parametrize.cached()`` is the documented remedy but is unusable
# here: it caches every parametrized tensor for the whole context, which for
# this model is ~66 GB of materialized expert weights held at once. So the cache
# is local and BOUNDED instead. Size 2 is enough because the loop alternates
# gate_up_proj / down_proj within one layer and never revisits a previous
# layer's experts; moving to the next layer evicts naturally.
#
# Validity is keyed on the tensors' ``_version`` counters (bumped by any in-place
# write, which is what ``optimizer.step()`` does) plus the grad mode, so a cached
# value can never outlive the state it was computed from. This also means a
# gradient-checkpoint recompute in the backward pass gets a correct value --
# either a fresh recomputation (normal, since the entry was evicted long before)
# or an entry whose versions still match.
#
# Not thread-safe; training here is single-process, single-threaded.

_CACHE_MAXSIZE = 2


class _ExpertWeightCache:
    """Tiny bounded cache of computed expert weights, keyed by parametrization."""

    def __init__(self, maxsize: int = _CACHE_MAXSIZE):
        self.maxsize = maxsize
        self.enabled = True
        # owner id -> (owner, signature, tensor); insertion-ordered = LRU order.
        self._entries: "Dict[int, Tuple[Any, Tuple, torch.Tensor]]" = {}
        self.hits = 0
        self.misses = 0

    def get(self, owner, signature):
        if not self.enabled:
            return None
        entry = self._entries.get(id(owner))
        # `owner is` guards against an id() reused by a new object after a GC.
        if entry is None or entry[0] is not owner or entry[1] != signature:
            self.misses += 1
            return None
        self.hits += 1
        return entry[2]

    def put(self, owner, signature, value) -> None:
        if not self.enabled:
            return
        self._entries.pop(id(owner), None)
        self._entries[id(owner)] = (owner, signature, value)
        while len(self._entries) > self.maxsize:
            self._entries.pop(next(iter(self._entries)))

    def clear(self) -> None:
        self._entries.clear()


_WEIGHT_CACHE = _ExpertWeightCache()


class expert_cache_disabled:
    """Context manager turning the expert-weight cache off (tests/debugging)."""

    def __enter__(self):
        self._prev = _WEIGHT_CACHE.enabled
        _WEIGHT_CACHE.enabled = False
        _WEIGHT_CACHE.clear()
        return self

    def __exit__(self, *exc):
        _WEIGHT_CACHE.enabled = self._prev
        _WEIGHT_CACHE.clear()
        return False


def _chunk_size(d1: int, d2: int) -> int:
    """How many experts to fake-quant at once, given one expert's slab size."""
    per_expert = max(1, d1 * d2)
    return max(1, _CHUNK_TARGET_ELEMS // per_expert)


def _warn_on_row_misalignment(param_name: str, row_width: int, segments) -> None:
    """Warn when a scheme's blocks would straddle rows in the fake-quant.

    ``fake_quant`` blocks the FLATTENED tensor, which equals real per-row ggml
    blocking only when the row width is a multiple of the scheme's block size.
    Every fused expert tensor in practice has a row width of 512/2048 (both
    multiples of 256), so this never fires on a real model -- but if it ever
    does, the fake-quant is approximating a different partition than the writer
    will actually use, and that must be visible rather than assumed away.
    """
    try:
        from magicquant.quant.converters import GGML_BLOCK_SIZE
    except Exception:  # pragma: no cover - only if the core deps are absent
        return
    for seg in segments:
        if seg.ggml_type_name in PASSTHROUGH_SCHEMES:
            continue
        block = GGML_BLOCK_SIZE.get(seg.ggml_type_name, 1)
        if block > 1 and row_width % block != 0:
            warnings.warn(
                f"{param_name or '<fused expert>'}: row width {row_width} is not "
                f"a multiple of {seg.ggml_type_name}'s block size {block}, so the "
                f"QAT fake-quant blocks across row boundaries while the GGUF "
                f"writer blocks per row (and may fall back to another type "
                f"entirely). Training sees an approximation of a different "
                f"partition than the one that ships.",
                UserWarning,
                stacklevel=3,
            )


class FusedExpertQAT(nn.Module):
    """Parametrization giving a fused 3-D expert Parameter QAT semantics.

    Registered via ``parametrize.register_parametrization(experts_module,
    "gate_up_proj", FusedExpertQAT(...))``. ``forward`` receives the frozen base
    ``W`` (which parametrize keeps as ``.original``) and returns the QAT weight.

    Args:
        shape: the fused parameter's shape ``[E, d1, d2]``.
        segments: contiguous, in-order slices of dim 1 with their resolved ggml
            scheme -- one per GGUF tensor the parameter is written as. Segments
            whose scheme is a passthrough (BF16/F16/F32) are copied through
            un-fake-quantized, so a hybrid where only gate is low-bit works.
        lora_r: per-expert LoRA rank. ``0`` disables the adapter entirely (the
            parametrization then only applies fake-quant).
        lora_alpha: LoRA alpha; ``scaling = lora_alpha / lora_r``.
        mode: ``"live"`` or ``"frozen"`` (see the module docstring).
        param_name: the parameter's ORIGINAL ``named_parameters()`` path, kept
            because registration moves the parameter under
            ``parametrizations.<attr>.original`` and the adapter file's keys must
            still be the base model's real safetensors keys.
    """

    def __init__(
        self,
        shape: Sequence[int],
        segments: Sequence[ResolvedSegment],
        lora_r: int,
        lora_alpha: float,
        *,
        mode: str = MODE_LIVE,
        param_name: str = "",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if len(shape) != 3:
            raise ValueError(f"fused expert parameter must be 3-D, got {tuple(shape)}")
        if mode not in EXPERT_QUANT_MODES:
            raise ValueError(
                f"mode must be one of {EXPERT_QUANT_MODES}, got {mode!r}"
            )
        n_experts, d1, d2 = (int(s) for s in shape)
        self.n_experts = n_experts
        self.d1 = d1
        self.d2 = d2
        self.mode = mode
        self.param_name = param_name
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 0.0
        self.segments: Tuple[ResolvedSegment, ...] = tuple(segments)
        self.chunk_experts = _chunk_size(d1, d2)
        # Set once the base has been fake-quantized in place (frozen mode).
        self.base_quantized = False
        _warn_on_row_misalignment(param_name, d2, self.segments)

        if self.lora_r > 0:
            # Orientation is fixed by the merge contract: W[e] += scale*(A[e]@B[e]).
            self.lora_expert_A = nn.Parameter(
                torch.zeros(
                    n_experts, d1, self.lora_r, device=device, dtype=torch.float32
                )
            )
            self.lora_expert_B = nn.Parameter(
                torch.empty(
                    n_experts, self.lora_r, d2, device=device, dtype=torch.float32
                )
            )
            nn.init.kaiming_uniform_(self.lora_expert_B, a=math.sqrt(5))
        else:
            self.register_parameter("lora_expert_A", None)
            self.register_parameter("lora_expert_B", None)

    # ── the QAT weight ────────────────────────────────────────────────────────

    def _delta(self, lo: int, hi: int) -> Optional[torch.Tensor]:
        """Scaled LoRA delta for experts ``[lo, hi)``, or ``None`` if rank 0."""
        if self.lora_r <= 0:
            return None
        a = self.lora_expert_A[lo:hi].float()
        b = self.lora_expert_B[lo:hi].float()
        return self.scaling * torch.bmm(a, b)

    def _quantize_slab(self, part: torch.Tensor) -> torch.Tensor:
        """Fake-quantize one expert chunk segment-by-segment along dim 1."""
        if len(self.segments) == 1:
            seg = self.segments[0]
            if seg.ggml_type_name in PASSTHROUGH_SCHEMES:
                return part
            return fake_quant(part, seg.ggml_type_name)
        pieces = []
        for seg in self.segments:
            sl = part[:, seg.start:seg.stop, :]
            if seg.ggml_type_name in PASSTHROUGH_SCHEMES:
                pieces.append(sl)
            else:
                pieces.append(fake_quant(sl, seg.ggml_type_name))
        return torch.cat(pieces, dim=1)

    def _cache_signature(self, W: torch.Tensor) -> Tuple:
        """State the computed weight depends on, cheap enough to check per access.

        ``_version`` bumps on any in-place write -- an ``optimizer.step()`` on the
        adapters, or ``quantize_base_`` on the base -- so a stale entry can never
        be served. Grad mode is in the key because a value built under
        ``no_grad`` carries no graph and must not be reused once grad is back on.
        """
        if self.lora_r > 0:
            versions = (
                W._version, self.lora_expert_A._version, self.lora_expert_B._version,
            )
        else:
            versions = (W._version,)
        return versions + (torch.is_grad_enabled(),)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        signature = self._cache_signature(W)
        cached = _WEIGHT_CACHE.get(self, signature)
        if cached is not None:
            return cached
        out = self._compute(W)
        _WEIGHT_CACHE.put(self, signature, out)
        return out

    def _compute(self, W: torch.Tensor) -> torch.Tensor:
        out_dtype = W.dtype
        if self.mode == MODE_FROZEN:
            # Base was fake-quantized in place at wrap time; nothing to do per
            # forward but add the (full-precision) adapter delta. Deliberately
            # NOT chunked: a bmm+add has no intermediate blow-up (peak is one
            # fp32 copy of the output, ~2 GB for the largest real tensor), and
            # chunking would trade that for 32 small bmms plus a cat. Measured
            # 0.089 s for a [256, 1024, 2048] tensor on gfx1151.
            if self.lora_r <= 0:
                return W
            delta = self._delta(0, self.n_experts)
            return (W.float() + delta).to(out_dtype)

        # Live mode IS chunked: the fake-quant kernels build several
        # intermediates per element (MXFP4 materializes an 8-wide grid-distance
        # tensor, the K-quant scale searches keep a handful of super-block
        # temporaries), so an unchunked pass over a 537M-element tensor would
        # allocate tens of GB.
        chunks: List[torch.Tensor] = []
        for lo in range(0, self.n_experts, self.chunk_experts):
            hi = min(lo + self.chunk_experts, self.n_experts)
            part = W[lo:hi].float()
            delta = self._delta(lo, hi)
            if delta is not None:
                part = part + delta
            chunks.append(self._quantize_slab(part))
        if len(chunks) == 1:
            return chunks[0].to(out_dtype)
        return torch.cat(chunks, dim=0).to(out_dtype)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Assigning to the parametrized attribute writes straight to the base.

        Required so ``module.gate_up_proj = tensor`` (and parametrize's own
        bookkeeping) has a defined meaning; the adapter is left untouched.
        """
        return X

    # ── frozen-mode base preparation ──────────────────────────────────────────

    @torch.no_grad()
    def quantize_base_(self, W: torch.Tensor) -> torch.Tensor:
        """Fake-quantize ``W`` in place, chunk by chunk (frozen mode setup).

        Done on the base parameter's own storage so no second copy of a
        multi-GB expert tensor is ever live. Safe to call once; idempotent in
        exact arithmetic anyway, since every fake-quant kernel is a fixed point.
        """
        for lo in range(0, self.n_experts, self.chunk_experts):
            hi = min(lo + self.chunk_experts, self.n_experts)
            q = self._quantize_slab(W[lo:hi].float())
            W[lo:hi].copy_(q.to(W.dtype))
        self.base_quantized = True
        # The in-place writes above bump W._version, so any cached weight is
        # already invalid by signature; clearing is belt-and-braces for the case
        # where a caller hands in a different tensor than the one cached against.
        _WEIGHT_CACHE.clear()
        return W

    def extra_repr(self) -> str:
        segs = ", ".join(
            f"{s.gguf_name or '<group>'}[{s.start}:{s.stop}]={s.ggml_type_name}"
            for s in self.segments
        )
        return (
            f"n_experts={self.n_experts}, shape=({self.n_experts}, {self.d1}, "
            f"{self.d2}), lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, "
            f"mode={self.mode!r}, segments=[{segs}]"
        )


# ── scheme resolution ─────────────────────────────────────────────────────────

def resolve_segment_schemes(
    segments: Sequence[ExpertSegment],
    scheme_by_group: Dict[str, str],
    classifier,
    scheme_by_tensor: Optional[Dict[str, str]] = None,
    *,
    default_group: str = "X",
) -> List[ResolvedSegment]:
    """Attach a ggml scheme to each segment: per-tensor first, then per-group.

    The per-tensor map (a search run's ``tensor_config``) is authoritative when
    it names the segment's GGUF tensor -- on the Qwen3.6 budget build the group
    map says X=Q3_K while 54 of the 123 expert tensors are actually Q2_K, so
    routing experts by group alone would fake-quant nearly half of them a full
    bit above what ships. Falls back to the group scheme (``X`` for experts, or
    whatever the classifier says for a resolvable GGUF name).
    """
    resolved: List[ResolvedSegment] = []
    for seg in segments:
        scheme = None
        if seg.gguf_name and scheme_by_tensor:
            scheme = scheme_by_tensor.get(seg.gguf_name)
        if scheme is None:
            group = default_group
            if seg.gguf_name is not None and classifier is not None:
                group = classifier.classify_tensor(seg.gguf_name)
            scheme = scheme_by_group.get(group)
        if scheme is None:
            return []
        resolved.append(
            ResolvedSegment(seg.gguf_name, seg.start, seg.stop, scheme)
        )
    return resolved


def _layout_is_supported(owner: nn.Module, param_attr: str) -> Tuple[bool, str]:
    """Whether ``owner``'s fused expert layout matches what the mapping assumes.

    transformers' ``use_experts_implementation`` decorator stamps the experts
    module with ``is_concatenated`` / ``is_transposed``. The GGUF segment
    mapping in ``names.py`` assumes concatenated (gate first) and untransposed
    ``[E, out, in]``; if a module says otherwise, wrapping it would fake-quant
    the wrong slices against the wrong schemes. Absent attributes mean an older
    transformers (or a hand-built module) -- the documented default is assumed.
    """
    if getattr(owner, "is_transposed", False):
        return False, "module reports is_transposed=True ([E, in, out] layout)"
    if not getattr(owner, "is_concatenated", True):
        return False, "module reports is_concatenated=False (interleaved gate/up)"
    return True, ""


def _split_param_path(name: str) -> Tuple[str, str]:
    """``a.b.c`` -> ``("a.b", "c")``; a bare name -> ``("", name)``."""
    if "." not in name:
        return "", name
    parent, _, attr = name.rpartition(".")
    return parent, attr


def _get_submodule(root: nn.Module, dotted: str) -> nn.Module:
    module = root
    if not dotted:
        return module
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


# ── wrapping ──────────────────────────────────────────────────────────────────

def wrap_fused_experts(
    model: nn.Module,
    scheme_by_group: Dict[str, str],
    classifier,
    scheme_by_tensor: Optional[Dict[str, str]] = None,
    *,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    mode: str = MODE_LIVE,
) -> List["FusedExpertQAT"]:
    """Register a :class:`FusedExpertQAT` on every fused 3-D expert parameter.

    Walks ``model.named_parameters()`` for 3-D parameters whose names map to
    GGUF expert tensors (``magicquant.qat.names.fused_expert_segments``),
    resolves each segment's scheme, and registers the parametrization. In
    ``"frozen"`` mode the base is fake-quantized in place immediately after
    registration.

    Skipped (with a warning, never silently): parameters whose owning module
    reports an unsupported layout, and parameters whose every segment resolves
    to a passthrough scheme (nothing to be quantization-aware about).

    Returns the list of registered parametrizations, in wrap order.
    """
    if mode not in EXPERT_QUANT_MODES:
        raise ValueError(f"mode must be one of {EXPERT_QUANT_MODES}, got {mode!r}")

    # Collect first: registering rebinds attributes and would disturb the walk.
    targets: List[Tuple[str, nn.Module, str, List[ResolvedSegment]]] = []
    for name, param in model.named_parameters():
        if param.ndim != 3:
            continue
        segments = fused_expert_segments(name, tuple(param.shape))
        if not segments:
            continue
        parent_path, attr = _split_param_path(name)
        owner = _get_submodule(model, parent_path)
        ok, why = _layout_is_supported(owner, attr)
        if not ok:
            warnings.warn(
                f"Skipping QAT for fused expert parameter {name!r}: {why}. "
                f"The GGUF segment mapping assumes concatenated, untransposed "
                f"[E, out, in] experts; wrapping it anyway would fake-quant the "
                f"wrong slices.",
                UserWarning,
                stacklevel=2,
            )
            continue
        resolved = resolve_segment_schemes(
            segments, scheme_by_group, classifier, scheme_by_tensor
        )
        if not resolved:
            continue
        if all(s.ggml_type_name in PASSTHROUGH_SCHEMES for s in resolved):
            continue
        targets.append((name, owner, attr, resolved))

    registered: List[FusedExpertQAT] = []
    for name, owner, attr, resolved in targets:
        param = getattr(owner, attr)
        param.requires_grad = False
        p = FusedExpertQAT(
            tuple(param.shape),
            resolved,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            mode=mode,
            param_name=name,
            device=param.device,
        )
        # unsafe=True skips parametrize's eager "call it once and compare
        # shape/dtype" check. That check would run a FULL fake-quant of a
        # multi-GB expert tensor per registration in live mode (the exact cost
        # the mode exists to schedule deliberately); shape/dtype invariance is
        # guaranteed by construction here and covered by tests instead.
        parametrize.register_parametrization(owner, attr, p, unsafe=True)
        if mode == MODE_FROZEN:
            p.quantize_base_(owner.parametrizations[attr].original.data)
        registered.append(p)
    return registered


def iter_expert_parametrizations(model: nn.Module) -> Iterable[FusedExpertQAT]:
    """Every :class:`FusedExpertQAT` registered on ``model``."""
    for module in model.modules():
        if isinstance(module, FusedExpertQAT):
            yield module


def fused_expert_adapter_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Adapter tensors for every wrapped fused expert, keyed for the merge lane.

    Keys are ``"<base safetensors key>.lora_expert_A"`` / ``".lora_expert_B"``
    where the base key is the fused parameter's ORIGINAL
    ``named_parameters()`` path (e.g.
    ``model.language_model.layers.3.mlp.experts.gate_up_proj``) -- raw
    Parameters, so no ``.weight`` suffix. Shapes are ``(E, W.shape[1], r)`` and
    ``(E, r, W.shape[2])``; the merge is ``W[e] += scale * (A[e] @ B[e])``.
    Matches ``magicquant.qat.merge._apply_3d`` exactly.
    """
    state: Dict[str, torch.Tensor] = {}
    for p in iter_expert_parametrizations(model):
        if p.lora_r <= 0 or not p.param_name:
            continue
        state[f"{p.param_name}.lora_expert_A"] = (
            p.lora_expert_A.detach().to(torch.float32).cpu()
        )
        state[f"{p.param_name}.lora_expert_B"] = (
            p.lora_expert_B.detach().to(torch.float32).cpu()
        )
    return state


def fused_expert_adapter_meta(model: nn.Module) -> List[Dict[str, Any]]:
    """Per-tensor record of what was wrapped, for ``qat_meta.json``."""
    meta: List[Dict[str, Any]] = []
    for p in iter_expert_parametrizations(model):
        meta.append({
            "param": p.param_name,
            "base_shape": [p.n_experts, p.d1, p.d2],
            "lora_expert_A_shape": [p.n_experts, p.d1, p.lora_r],
            "lora_expert_B_shape": [p.n_experts, p.lora_r, p.d2],
            "mode": p.mode,
            "segments": [
                {
                    "gguf_name": s.gguf_name,
                    "start": s.start,
                    "stop": s.stop,
                    "scheme": s.ggml_type_name,
                }
                for s in p.segments
            ],
        })
    return meta


@torch.no_grad()
def merge_fused_expert_adapters(model: nn.Module) -> nn.Module:
    """Bake ``W + scale*(A@B)`` into the base and remove the parametrizations.

    The full-precision merged weight, NOT the fake-quantized one -- the real
    ggml pack happens later and must see the continuous merged weight (same
    contract as ``wrap.merge_qat_adapters`` for Linears). In ``"frozen"`` mode
    the base is already the fake-quantized one, which is exactly what training
    saw, so merging is still consistent with the trained forward.

    Mutates ``model`` in place and returns it.
    """
    targets = []
    for module in model.modules():
        plist = getattr(module, "parametrizations", None)
        if plist is None:
            continue
        for attr in list(plist.keys()):
            for p in plist[attr]:
                if isinstance(p, FusedExpertQAT):
                    targets.append((module, attr, p))
                    break

    for module, attr, p in targets:
        original = module.parametrizations[attr].original
        if p.lora_r > 0:
            for lo in range(0, p.n_experts, p.chunk_experts):
                hi = min(lo + p.chunk_experts, p.n_experts)
                delta = p._delta(lo, hi)
                original.data[lo:hi] = (
                    original.data[lo:hi].float() + delta
                ).to(original.dtype)
        parametrize.remove_parametrizations(module, attr, leave_parametrized=False)
    _WEIGHT_CACHE.clear()
    return model


# ── cost / memory accounting ──────────────────────────────────────────────────

# Measured on this box (gfx1151, torch 2.11 ROCm, fp32, 4M-element tensors) --
# elements per second for one fake_quant call. Used only to turn "how big are
# the experts" into an honest wall-clock estimate in the wrap-time log; being
# off by 2x doesn't change any decision these numbers inform.
MEASURED_FAKE_QUANT_ELEMS_PER_S = {
    "Q2_K": 2.7e6,
    "Q3_K": 1.1e8,
    "Q4_K": 2.4e7,
    "Q5_K": 2.4e7,
    "Q6_K": 5.0e7,
    "MXFP4": 5.8e7,
    "IQ4_NL": 1.0e7,
    "Q8_0": 1.0e8,
}
_DEFAULT_ELEMS_PER_S = 2.5e7


def estimate_expert_qat_cost(
    parametrizations: Sequence[FusedExpertQAT],
    *,
    optimizer_states: int = 2,
    param_bytes: int = 4,
) -> Dict[str, Any]:
    """Train-time cost of the wrapped fused-expert adapters.

    Args:
        parametrizations: the wrapped experts (``wrap_fused_experts``' return).
        optimizer_states: moments the optimizer keeps per trainable element
            (AdamW = 2).
        param_bytes: bytes per adapter element (fp32 = 4).

    Returns a dict with the adapter parameter count, the bytes for
    params+grads+optimizer state, the total base elements covered, and -- for
    ``"live"`` parametrizations -- a fake-quant seconds-per-forward estimate
    from :data:`MEASURED_FAKE_QUANT_ELEMS_PER_S`. ``live_forward_seconds`` is
    0.0 when nothing runs live.
    """
    n_tensors = len(parametrizations)
    lora_params = 0
    base_elems = 0
    live_seconds = 0.0
    n_live = 0
    for p in parametrizations:
        if p.lora_r > 0:
            lora_params += p.n_experts * p.d1 * p.lora_r
            lora_params += p.n_experts * p.lora_r * p.d2
        base_elems += p.n_experts * p.d1 * p.d2
        if p.mode != MODE_LIVE:
            continue
        n_live += 1
        for seg in p.segments:
            if seg.ggml_type_name in PASSTHROUGH_SCHEMES:
                continue
            elems = p.n_experts * (seg.stop - seg.start) * p.d2
            rate = MEASURED_FAKE_QUANT_ELEMS_PER_S.get(
                seg.ggml_type_name, _DEFAULT_ELEMS_PER_S
            )
            live_seconds += elems / rate

    # params + grads + optimizer moments, all at param_bytes each.
    bytes_per_elem = param_bytes * (2 + optimizer_states)
    return {
        "n_expert_tensors": n_tensors,
        "n_live_tensors": n_live,
        "lora_params": lora_params,
        "base_elements": base_elems,
        "adapter_bytes": lora_params * param_bytes,
        "train_bytes": lora_params * bytes_per_elem,
        "train_gib": lora_params * bytes_per_elem / (1024 ** 3),
        "live_forward_seconds": live_seconds,
    }
