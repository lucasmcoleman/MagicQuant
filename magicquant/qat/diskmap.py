"""Reconcile QAT adapter target keys against the base checkpoint's on-disk
safetensors key set.

INCIDENT (2026-08-05, launch blocker #2): ``run_qat`` discovers what to wrap
by walking the LOADED model's own module graph (``named_modules()`` /
``named_parameters()``), whose paths come from however the HF model CLASS
happens to name its submodules. On Qwen3.6-35B-A3B those paths were bare
``model.layers.N...``, while the SAME model's checkpoint on disk
(``model.safetensors.index.json``'s ``weight_map``) used
``model.language_model.layers.N...`` -- a nesting difference the loaded
module graph never surfaces, because nothing before this module ever
compared the two. ``_save_adapters``/``fused_expert_adapter_state`` wrote the
module-graph names verbatim, so 390 of 391 adapter target keys didn't exist
in the base model's weight map -- discovered by ``magicquant.qat.merge``
*after* an entire overnight training run, not before it.

This module is the fix, in both places the review specified:

  * at adapter SAVE time (``train._save_adapters`` /
    ``expert_wrap.fused_expert_adapter_state``), every key actually written
    to ``adapter_model.safetensors`` is reconciled to its disk form first, so
    the file ``magicquant.qat.merge`` reads later always has real keys;
  * at PRE-FLIGHT (``train.run_qat``, before a single training step runs),
    every would-be adapter target key is checked against the same weight map
    so an unresolvable name fails in the run's first minutes, not after
    hours of training -- see :func:`resolve_adapter_targets`.

The reconciliation itself is bounded and exact-match-first, never fuzzy:
try the literal key, then try inserting or removing ``'language_model.'``
immediately after a leading ``'model.'`` -- the ONE nesting difference this
was built to catch -- and only accept the result if it lands on a UNIQUE key
in the weight map. Anything else is a loud failure, never a silent guess.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

_LANGUAGE_MODEL = "language_model."
_MODEL_PREFIX = "model."
_MODEL_LM_PREFIX = _MODEL_PREFIX + _LANGUAGE_MODEL  # "model.language_model."


class QATKeyReconciliationError(RuntimeError):
    """An adapter target key doesn't exist in the base checkpoint's on-disk
    weight map, even after bounded prefix reconciliation.

    Raised instead of silently saving/using a key that ``magicquant.qat.merge``
    would later refuse -- see this module's docstring for the incident this
    closes.
    """


def reconcile_key_to_disk(key: str, weight_map_keys: FrozenSet[str]) -> Optional[str]:
    """Resolve ``key`` (a name from the loaded model's own module graph)
    against ``weight_map_keys`` (the base checkpoint's real on-disk tensor
    names).

    Exact match first. Failing that, try inserting or removing
    ``'language_model.'`` immediately after a leading ``'model.'`` -- the two
    directions are mutually exclusive by construction (a key either already
    has the prefix or it doesn't), so this can only ever produce ONE
    candidate; "requiring a unique match" is enforced explicitly anyway so a
    future edit that loosens the transform can't silently start resolving
    ambiguously.

    Returns the disk key, or ``None`` if nothing resolved.
    """
    if key in weight_map_keys:
        return key

    candidates = set()
    if key.startswith(_MODEL_PREFIX) and not key.startswith(_MODEL_LM_PREFIX):
        candidate = _MODEL_LM_PREFIX + key[len(_MODEL_PREFIX):]
        if candidate in weight_map_keys:
            candidates.add(candidate)
    if key.startswith(_MODEL_LM_PREFIX):
        candidate = _MODEL_PREFIX + key[len(_MODEL_LM_PREFIX):]
        if candidate in weight_map_keys:
            candidates.add(candidate)

    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def resolve_adapter_targets(
    linear_module_names: Iterable[str],
    expert_param_names: Iterable[str],
    weight_map: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Resolve every would-be adapter target key to its on-disk form, batched.

    Used by ``run_qat``'s PRE-FLIGHT check: called once, right after wrapping
    and before any training happens, so an unresolvable name is a run-start
    failure rather than an after-hours one.

    Args:
        linear_module_names: ``QATLinear`` module paths from
            ``model.named_modules()`` (base tensor key = ``f"{name}.weight"``).
        expert_param_names: fused 3-D expert parameters' ORIGINAL
            ``named_parameters()`` paths (the base tensor key IS the name --
            no ``.weight`` suffix; see ``expert_wrap``'s module docstring).
        weight_map: ``{tensor_name: shard_filename}`` from the base
            checkpoint's ``model.safetensors.index.json`` (or the
            single-file equivalent).

    Returns:
        ``(linear_name -> disk_module_path, expert_name -> disk_param_name)``,
        covering every input name.

    Raises:
        QATKeyReconciliationError: any key -- 2-D or 3-D -- failed to
            resolve. Lists every failure from BOTH families at once (never
            one at a time), so a partial-resolution run is unambiguous about
            scope on the first try.
    """
    weight_map_keys: FrozenSet[str] = frozenset(weight_map.keys())
    linear_names = list(linear_module_names)
    expert_names = list(expert_param_names)

    resolved_linear: Dict[str, str] = {}
    resolved_expert: Dict[str, str] = {}
    failures: List[str] = []

    for name in linear_names:
        target = f"{name}.weight"
        disk = reconcile_key_to_disk(target, weight_map_keys)
        if disk is None:
            failures.append(target)
        else:
            resolved_linear[name] = disk[: -len(".weight")]

    for name in expert_names:
        disk = reconcile_key_to_disk(name, weight_map_keys)
        if disk is None:
            failures.append(name)
        else:
            resolved_expert[name] = disk

    if failures:
        total = len(linear_names) + len(expert_names)
        failures.sort()
        raise QATKeyReconciliationError(
            f"{len(failures)} of {total} adapter target key(s) could not be "
            f"resolved against the base checkpoint's on-disk weight map -- "
            f"neither an exact match nor a unique 'language_model.' "
            f"insert/remove match (the 2026-08-05 incident class: the "
            f"loaded model's module-graph names disagree with its own "
            f"checkpoint's safetensors names). Caught here, at run start, "
            f"instead of after training. Unresolved: {failures[:20]}"
            + (" ..." if len(failures) > 20 else "")
        )

    return resolved_linear, resolved_expert
