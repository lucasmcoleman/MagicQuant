"""Streaming base+LoRA merge for QAT adapters -> a real merged safetensors model.

``magicquant.qat.train.run_qat`` freezes a base model, wraps its routed
``nn.Linear`` modules with fake-quant ``QATLinear`` (``magicquant/qat/wrap.py``),
trains only the LoRA adapters, and saves ONLY those adapters to
``<out>/adapter_model.safetensors`` plus a companion ``<out>/qat_meta.json``
recording the run's hyperparameters (``lora_r``, ``lora_alpha``,
``scheme_by_group``, ...). Nothing in MagicQuant writes a *merged* model to
disk on its own.

This module is that missing piece: the on-disk, streaming counterpart to
``magicquant.qat.wrap.merge_qat_adapters`` (which merges an *in-memory*
``nn.Module`` and is used inside training/eval). It streams the base model's
safetensors shards from disk one at a time, applies
``W += scale * (adapter delta)`` to every tensor an adapter targets, and
writes merged shards to ``out_dir`` -- the base model is never fully
materialized in memory (pattern: Foundry's ``core/fast_export.py``
``streaming_merge``, ~6 GB peak on a 35B model).

INCIDENT (2026-08-05): Foundry's own merge step (``run_budget_qat.sh`` step 4)
called ``core.fast_export.streaming_merge`` against a MagicQuant adapter dir.
That function (a) requires a PEFT ``adapter_config.json`` MagicQuant never
writes (immediate ``FileNotFoundError``), and (b) filters LoRA keys on the
substring ``".lora_A."`` -- a TRAILING DOT, i.e. PEFT's ``...lora_A.weight``
naming -- while MagicQuant's own ``_save_adapters`` writes keys ending in
exactly ``".lora_A"`` (no trailing dot, no ``.weight``). Even with a config
stub supplied, that mismatch would match ZERO keys and silently emit an
unmodified copy of the base model with no error at all. This module reads
MagicQuant's actual on-disk format directly -- no PEFT config, no borrowed
naming assumptions -- and refuses loudly (see ``QATMergeError`` below) rather
than degrading to a silent no-op merge.

Adapter key shapes handled (2-D always; 3-D only if present in the file):

  2-D (``nn.Linear`` LoRA, written by ``QATLinear``/``_save_adapters``):
    keys ``"<module_path>.lora_A"`` / ``"<module_path>.lora_B"`` --
    ``module_path`` is exactly what ``model.named_modules()`` yields (no
    ``.weight`` suffix). Shapes: ``lora_A`` is ``(r, in_features)``,
    ``lora_B`` is ``(out_features, r)``. The base tensor is
    ``"<module_path>.weight"``. Merge: ``W += scale * (B @ A)`` -- matches
    ``QATLinear.merged_weight``/``wrap.merge_qat_adapters`` exactly.

  3-D (fused MoE-expert LoRA -- opt-in, only applied if such keys exist):
    keys ``"<tensor_name>.lora_expert_A"`` / ``"<tensor_name>.lora_expert_B"``,
    where ``tensor_name`` is the exact base safetensors key of a fused expert
    parameter (e.g. Qwen3.6-35B-A3B's
    ``model.language_model.layers.N.mlp.experts.{gate_up,down}_proj`` -- these
    are raw ``nn.Parameter``s, not an ``nn.Linear.weight``, so unlike the 2-D
    case there is no ``.weight`` to append). Shapes: ``lora_expert_A`` is
    ``(E, in, r)``, ``lora_expert_B`` is ``(E, r, out)``, batched over the
    leading expert axis ``E``. Merge: ``W[e] += scale * (A[e] @ B[e])`` for
    every expert ``e`` (``torch.bmm``). No lane in this repo writes this
    format yet (see CLAUDE.md's QAT section, "routed experts are FUSED 3-D
    PARAMETERS") -- if the adapter file has no ``.lora_expert_A``/
    ``.lora_expert_B`` keys, this path is a pure no-op and the merge is
    2-D-only, which must work today regardless of whether that lane lands.

``scale`` = ``lora_alpha / lora_r``, read from ``qat_meta.json`` -- NEVER
hardcoded, since a run launched with non-default ``--lora-r``/``--lora-alpha``
must merge at the values it actually trained with. There is no rsLoRA option
in MagicQuant's own QAT (unlike PEFT), so this is the one formula, always. A
future 3-D expert lane trained at its own rank/alpha can record
``expert_lora_r``/``expert_lora_alpha`` in ``qat_meta.json``; absent that,
the 3-D path reuses the run's single ``lora_r``/``lora_alpha``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from magicquant.logging import get_logger

log = get_logger(__name__)

_META_FILENAME = "qat_meta.json"
_DEFAULT_ADAPTER_FILENAME = "adapter_model.safetensors"

_LORA_EXPERT_A_SUFFIX = ".lora_expert_A"
_LORA_EXPERT_B_SUFFIX = ".lora_expert_B"
_LORA_A_SUFFIX = ".lora_A"
_LORA_B_SUFFIX = ".lora_B"

# Config/tokenizer/preprocessor files copied verbatim from the base model dir
# into the merged output. None of these carry weights (the "byte-identical
# elsewhere" contract is about tensors), they just need to be present for the
# merged directory to be a normal, loadable HF model directory on its own.
_COPY_IF_PRESENT = (
    "config.json", "generation_config.json", "special_tokens_map.json",
    "preprocessor_config.json", "processor_config.json",
    "video_preprocessor_config.json", "configuration.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "vocab.json", "merges.txt",
)


class QATMergeError(RuntimeError):
    """A merge can't proceed safely and must not silently no-op instead."""


def _resolve_base_model_dir(base_model: str) -> str:
    """Return a local directory holding the base model's safetensors.

    ``base_model`` is used as-is if it's already a local directory (tonight's
    actual use: a Foundry ``output/.../source`` dir). Otherwise it's treated
    as an HF Hub repo id and downloaded (``*.gguf`` siblings some repos also
    publish are skipped -- a merge only ever touches safetensors). The
    ``huggingface_hub`` import is local to this branch so a purely local
    merge never needs network or that dependency importable.
    """
    if os.path.isdir(base_model):
        return base_model
    from huggingface_hub import snapshot_download

    log.info(
        "Base model is not a local directory; downloading from the Hub",
        base_model=base_model,
    )
    return snapshot_download(base_model, ignore_patterns=["*.gguf"])


def _read_qat_meta(adapter_dir: str) -> Dict:
    meta_path = os.path.join(adapter_dir, _META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path!r} not found. qat-merge reads the run's own "
            f"hyperparameters (lora_r, lora_alpha) from qat_meta.json rather "
            f"than hardcoding them -- a run launched with non-default "
            f"--lora-r/--lora-alpha must merge at ITS values. Point "
            f"--adapters at a directory `magicquant qat` actually wrote to."
        )
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _load_adapter_state(adapter_dir: str, meta: Dict) -> Dict[str, torch.Tensor]:
    adapter_file = meta.get("adapter_file", _DEFAULT_ADAPTER_FILENAME)
    adapter_path = os.path.join(adapter_dir, adapter_file)
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(
            f"Adapter weights not found: {adapter_path!r} "
            f"(qat_meta.json names {adapter_file!r})."
        )
    return load_file(adapter_path, device="cpu")


def _build_lora_maps(
    state: Dict[str, torch.Tensor], scale: float, expert_scale: float
) -> Tuple[
    Dict[str, Tuple[torch.Tensor, torch.Tensor, float]],
    Dict[str, Tuple[torch.Tensor, torch.Tensor, float]],
]:
    """Pair up ``lora_A``/``lora_B`` (and ``lora_expert_A``/``lora_expert_B``) keys.

    Returns ``(two_d_map, three_d_map)``, each
    ``{base_tensor_key: (A, B, scale)}``. The expert suffix is checked FIRST
    since ``.lora_expert_A`` also ends in the substring ``A`` but must never
    be treated as a bare ``.lora_A`` match (and vice versa -- the two suffix
    sets are disjoint by construction, checked with ``endswith`` on the full
    suffix, not a loose substring).
    """
    two_d: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}
    three_d: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}

    for key, a in state.items():
        if key.endswith(_LORA_EXPERT_A_SUFFIX):
            base = key[: -len(_LORA_EXPERT_A_SUFFIX)]
            b_key = base + _LORA_EXPERT_B_SUFFIX
            b = state.get(b_key)
            if b is None:
                raise QATMergeError(
                    f"{key!r} has no matching {b_key!r} in the adapter file."
                )
            three_d[base] = (a, b, expert_scale)
        elif key.endswith(_LORA_A_SUFFIX):
            base = key[: -len(_LORA_A_SUFFIX)]
            b_key = base + _LORA_B_SUFFIX
            b = state.get(b_key)
            if b is None:
                raise QATMergeError(
                    f"{key!r} has no matching {b_key!r} in the adapter file."
                )
            two_d[base + ".weight"] = (a, b, scale)

    return two_d, three_d


def _apply_2d(
    w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, name: str
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or w.ndim != 2:
        raise QATMergeError(
            f"{name!r}: expected 2-D base/lora_A/lora_B, got "
            f"{tuple(w.shape)}/{tuple(a.shape)}/{tuple(b.shape)}"
        )
    out_features, r = b.shape
    r2, in_features = a.shape
    if r != r2 or tuple(w.shape) != (out_features, in_features):
        raise QATMergeError(
            f"{name!r}: shape mismatch -- base {tuple(w.shape)}, "
            f"lora_A {tuple(a.shape)}, lora_B {tuple(b.shape)}"
        )
    orig_dtype = w.dtype
    delta = scale * (b.float() @ a.float())
    return (w.float() + delta).to(orig_dtype)


def _apply_3d(
    w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, name: str
) -> torch.Tensor:
    if a.ndim != 3 or b.ndim != 3 or w.ndim != 3:
        raise QATMergeError(
            f"{name!r}: expected 3-D base/lora_expert_A/lora_expert_B, got "
            f"{tuple(w.shape)}/{tuple(a.shape)}/{tuple(b.shape)}"
        )
    e, i, r = a.shape
    e2, r2, o = b.shape
    if e != e2 or r != r2 or tuple(w.shape) != (e, i, o):
        raise QATMergeError(
            f"{name!r}: shape mismatch -- base {tuple(w.shape)}, "
            f"lora_expert_A {tuple(a.shape)}, lora_expert_B {tuple(b.shape)}"
        )
    orig_dtype = w.dtype
    delta = scale * torch.bmm(a.float(), b.float())
    return (w.float() + delta).to(orig_dtype)


def _load_weight_map(model_dir: str) -> Tuple[Dict[str, str], Dict]:
    """Return ``(weight_map, index_metadata)`` for a local safetensors model dir.

    Handles both the sharded (``model.safetensors.index.json``) and
    single-file (``model.safetensors``) layouts.
    """
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        return idx["weight_map"], idx.get("metadata", {})
    if os.path.isfile(single_path):
        with safe_open(single_path, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}, {}
    raise FileNotFoundError(f"No safetensors files found in {model_dir!r}")


def merge_qat_adapters(base_model_dir: str, adapter_dir: str, out_dir: str) -> str:
    """Merge QAT-LoRA adapters into ``base_model_dir``'s safetensors, streamed
    shard-by-shard, and write the result to ``out_dir``.

    - Every tensor an adapter targets gets ``W += scale * delta`` (2-D:
      ``scale * (B @ A)``; 3-D expert, if present: per-expert
      ``scale * (A[e] @ B[e])``). ``scale = lora_alpha / lora_r`` from
      ``qat_meta.json``.
    - Every other tensor is copied through unchanged (byte-identical).
    - Config/tokenizer files are copied from ``base_model_dir``.

    Raises:
        FileNotFoundError: ``adapter_dir`` doesn't exist, or has no
            ``qat_meta.json`` / no adapter weights file.
        QATMergeError: the adapter file has no usable lora key pairs at all,
            an adapter's A/B shapes disagree with each other or with the base
            tensor, or an adapter's target tensor doesn't exist anywhere in
            the base model's weight map. Each of these would otherwise
            silently produce a partially- or fully-unmodified copy of the
            base model -- the exact failure mode this module exists to close.

    Returns ``out_dir``.
    """
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir!r}")

    meta = _read_qat_meta(adapter_dir)
    lora_r = float(meta["lora_r"])
    lora_alpha = float(meta["lora_alpha"])
    scale = lora_alpha / lora_r
    # Forward-compatible with a future 3-D expert lane trained at its own
    # rank/alpha; falls back to the run-wide scale when absent (today).
    expert_r = float(meta.get("expert_lora_r", lora_r))
    expert_alpha = float(meta.get("expert_lora_alpha", lora_alpha))
    expert_scale = expert_alpha / expert_r

    state = _load_adapter_state(adapter_dir, meta)
    two_d_map, three_d_map = _build_lora_maps(state, scale, expert_scale)
    if not two_d_map and not three_d_map:
        keys = sorted(state.keys())
        raise QATMergeError(
            f"{adapter_dir!r}'s adapter file has no lora_A/lora_B (or "
            f"lora_expert_A/lora_expert_B) key pairs -- refusing to write "
            f"out what would silently be an unmodified copy of the base "
            f"model. Keys found: {keys[:10]}" + (" ..." if len(keys) > 10 else "")
        )

    model_dir = _resolve_base_model_dir(base_model_dir)
    weight_map, idx_metadata = _load_weight_map(model_dir)

    # Fail BEFORE touching disk if any adapter target doesn't exist anywhere
    # in the base model -- catches a name-mapping bug up front rather than
    # after burning I/O on a partial merge.
    all_targets = set(two_d_map) | set(three_d_map)
    unmapped = all_targets - set(weight_map.keys())
    if unmapped:
        missing = sorted(unmapped)
        raise QATMergeError(
            f"{len(unmapped)} adapter target tensor(s) don't exist in "
            f"{model_dir!r}'s weight map -- refusing to merge a subset while "
            f"silently dropping the rest: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    log.info(
        "Loaded QAT adapters", adapter_dir=adapter_dir, lora_r=lora_r,
        lora_alpha=lora_alpha, scale=scale, two_d_targets=len(two_d_map),
        three_d_targets=len(three_d_map),
    )

    shards: Dict[str, List[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for fname in _COPY_IF_PRESENT:
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, out_path / fname)

    applied_2d = set()
    applied_3d = set()

    for shard_idx, (shard_name, tensor_names) in enumerate(sorted(shards.items())):
        shard_path = os.path.join(model_dir, shard_name)
        shard_data = load_file(shard_path, device="cpu")
        try:
            for name in tensor_names:
                if name in two_d_map:
                    a, b, s = two_d_map[name]
                    shard_data[name] = _apply_2d(shard_data[name], a, b, s, name)
                    applied_2d.add(name)
                elif name in three_d_map:
                    a, b, s = three_d_map[name]
                    shard_data[name] = _apply_3d(shard_data[name], a, b, s, name)
                    applied_3d.add(name)
                # else: no adapter targets this tensor -- left byte-identical.

            # Write to a temp file then atomically rename, so a crash
            # mid-write can't leave a corrupt shard in out_dir.
            tmp_path = (out_path / shard_name).with_suffix(".tmp")
            save_file(shard_data, str(tmp_path))
            tmp_path.rename(out_path / shard_name)
        finally:
            del shard_data

        log.info(
            "Merged shard", shard=shard_name, index=shard_idx + 1, total=len(shards),
        )

    if os.path.isfile(os.path.join(model_dir, "model.safetensors.index.json")):
        with open(out_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump({"metadata": idx_metadata, "weight_map": weight_map}, f, indent=2)

    log.info(
        "QAT adapter merge complete", out_dir=out_dir, merged_2d=len(applied_2d),
        merged_3d=len(applied_3d), total_tensors=len(weight_map),
    )
    return out_dir
