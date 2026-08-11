"""QAT-LoRA training loop (``run_qat``) + completion-only collator.

``run_qat(cfg)`` loads an HF causal-LM, wraps its routed ``nn.Linear`` modules with
per-group fake-quant ``QATLinear``s (so every forward sees the quantized weight
that will ship), trains only the LoRA adapters on a chat dataset with
**completion-only loss** (system/user turns masked), and saves the adapters plus a
``qat_meta.json`` describing the run. Returns the output directory.

Heavy deps (transformers/peft/accelerate, +transitively tokenizers/safetensors/
huggingface_hub) are imported lazily inside ``run_qat`` so ``import magicquant.qat``
stays light (only torch).

Offline fallback: if the named HF model can't be downloaded, ``run_qat`` builds a
tiny ``LlamaForCausalLM`` from a small config (with a minimal byte-level
tokenizer) so the CPU smoke test still runs without network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from magicquant.qat.config import load_hybrid_config, load_tensor_config
from magicquant.qat.diskmap import (
    QATKeyReconciliationError,
    reconcile_key_to_disk,
    resolve_adapter_targets,
)
from magicquant.qat.expert_wrap import (
    EXPERT_QUANT_MODES,
    MODE_LIVE,
    estimate_expert_qat_cost,
    fused_expert_adapter_meta,
    fused_expert_adapter_state,
    iter_expert_parametrizations,
)
from magicquant.qat.wrap import QATLinear, wrap_model
from magicquant.gguf.tensor_groups import TensorGroupClassifier

_log = logging.getLogger("magicquant.qat.train")

# Label id that the loss ignores (HF convention).
IGNORE_INDEX = -100

# ``loaded_from`` value when the requested model couldn't be loaded and
# ``_build_offline_tiny_model`` filled in instead (see
# ``_load_model_and_tokenizer``). No real on-disk checkpoint backs this model,
# so adapter-key disk reconciliation (``_try_load_base_weight_map``) is a
# guaranteed no-op for it -- checked by identity against this constant rather
# than re-deriving "is this a real model" some other way.
_OFFLINE_TINY_MODEL_SENTINEL = "offline-tiny-llama"


# ── completion-only collator ──────────────────────────────────────────────────

@dataclass
class CompletionOnlyCollator:
    """Pad a batch of pre-masked {input_ids, labels} examples to a common length.

    Each example already has ``labels`` with non-completion tokens set to
    ``IGNORE_INDEX`` (done at tokenization time, where the prompt/completion split
    is known). This collator only pads to the batch max: ``input_ids`` with
    ``pad_token_id`` (attention-masked out), ``labels`` with ``IGNORE_INDEX``.
    """

    pad_token_id: int

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── dataset construction ──────────────────────────────────────────────────────

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ids_from_chat_template(tokenizer, messages, add_generation_prompt: bool):
    """Apply the chat template and return a flat list of token ids.

    ``apply_chat_template`` returns either a list of ids or a dict with
    ``input_ids`` depending on the transformers version; normalize to a list.
    """
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    # transformers >=5 returns a BatchEncoding (dict-like, not a dict subclass);
    # older versions return a flat list of ids. Normalize to a flat list.
    if hasattr(out, "keys") or (isinstance(out, dict)):
        out = out["input_ids"]
    return list(out)


def _encode_example(
    tokenizer, messages: List[Dict[str, str]], max_seq_len: int
) -> Optional[Dict[str, List[int]]]:
    """Tokenize one chat example with completion-only labels.

    The full conversation is the input. The prompt (everything up to and
    including the final user turn's generation prompt) is masked with
    ``IGNORE_INDEX`` so loss is computed only on the assistant completion.
    """
    full = _ids_from_chat_template(tokenizer, messages, add_generation_prompt=False)

    # Prompt = all messages before the final assistant turn, with a generation
    # prompt appended so the boundary matches where the completion begins.
    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant = i
            break
    if last_assistant is None:
        return None  # nothing to learn from

    prompt_msgs = messages[:last_assistant]
    prompt_ids = _ids_from_chat_template(
        tokenizer, prompt_msgs, add_generation_prompt=True
    )
    prompt_len = min(len(prompt_ids), len(full))

    full = full[:max_seq_len]
    labels = list(full)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX

    # If masking left nothing to learn (completion got truncated away), drop it.
    if all(t == IGNORE_INDEX for t in labels):
        return None
    return {"input_ids": full, "labels": labels}


def _build_dataset(tokenizer, rows: List[Dict[str, Any]], max_seq_len: int):
    examples = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        enc = _encode_example(tokenizer, messages, max_seq_len)
        if enc is not None:
            examples.append(enc)
    return examples


# ── model loading (with offline fallback) ─────────────────────────────────────

def _resolve_dtype(dtype) -> torch.dtype:
    """Map a cfg dtype (str | torch.dtype | None) to a torch dtype.

    None → bf16 on GPU (fits large models), fp32 on CPU (bf16 matmul is spotty there).
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}[str(dtype).lower()]


def _from_pretrained(cls, model_id, dtype):
    """Load with the transformers-5 ``dtype=`` kwarg, falling back to ``torch_dtype=``."""
    try:
        return cls.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype)


def _load_model_and_tokenizer(model_id: str, dtype=None):
    """Load an HF model (causal-LM or multimodal conditional-gen) + tokenizer,
    falling back to a tiny offline model if the download fails.

    Tries causal-LM first, then multimodal auto-classes (so QAT can target the text
    decoder of models like Gemma-3/4, which are ``*ForConditionalGeneration`` /
    ``*ForMultimodalLM``, not ``*ForCausalLM``). The vision/audio Linears simply
    don't route in ``wrap_model`` (their names don't map to GGUF tensor names), so
    only the text-decoder weights get fake-quant QAT.

    Returns ``(model, tokenizer, loaded_from)``.
    """
    import transformers
    from transformers import AutoTokenizer

    dtype = _resolve_dtype(dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # network/cache miss -> offline tiny model
        _log.warning("Could not load tokenizer for %r (%s); offline tiny fallback.", model_id, exc)
        return _build_offline_tiny_model()

    # In order of specificity: plain causal LM, then the multimodal/conditional-gen
    # auto-classes a modern Gemma exposes.
    candidates = [
        "AutoModelForCausalLM",
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModel",
    ]
    last_exc = None
    for cls_name in candidates:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = _from_pretrained(cls, model_id, dtype)
        except Exception as exc:
            last_exc = exc
            continue
        _log.info("Loaded %r with %s (dtype=%s).", model_id, cls_name, dtype)
        return model, _ensure_pad_and_template(tokenizer), model_id

    msg = (
        f"Could not load {model_id!r} with any auto-class (last error: {last_exc}); "
        "falling back to an offline tiny LlamaForCausalLM (smoke/CI path)."
    )
    # Both _log.warning and print: this is a must-not-be-missed message --
    # the run silently trains on a 2-layer toy model instead of the requested
    # one and continues to completion -- and this file echoes those messages
    # to stdout regardless of log level, since an unattended run's stdout is
    # what actually gets captured and read.
    _log.warning(msg)
    print(f"WARNING: {msg}", flush=True)
    return _build_offline_tiny_model()


def _ensure_pad_and_template(tokenizer):
    """Make sure the tokenizer has a pad token and a chat template."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = _SIMPLE_CHAT_TEMPLATE
    return tokenizer


# A minimal ChatML-ish template for tokenizers that ship without one.
_SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


def _build_offline_tiny_model():
    """Construct a tiny LlamaForCausalLM + byte-level tokenizer entirely offline."""
    from transformers import LlamaConfig, LlamaForCausalLM

    tokenizer = _build_byte_tokenizer()
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config).to(torch.float32)
    return model, tokenizer, _OFFLINE_TINY_MODEL_SENTINEL


def _build_byte_tokenizer():
    """A tiny offline byte-level tokenizer (256 bytes + a few specials).

    No network, no files: builds a ``PreTrainedTokenizerFast`` around a byte-level
    BPE with an empty merge table, which tokenizes any UTF-8 text by bytes.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    # The ByteLevel pre-tokenizer maps each raw byte to one of 256 printable
    # unicode chars; the BPE vocab must use those exact tokens (one per byte) so
    # text actually tokenizes to distinct ids (not all <unk>).
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    byte_vocab = {ch: i for i, ch in enumerate(alphabet)}
    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    for s in specials:
        byte_vocab[s] = len(byte_vocab)

    tok = Tokenizer(models.BPE(vocab=byte_vocab, merges=[], unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast.chat_template = _SIMPLE_CHAT_TEMPLATE
    return fast


# ── the training entry point ──────────────────────────────────────────────────

def _resolve_scheme_by_group(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Get the {group: ggml_type_name} map from cfg (direct or via search config)."""
    if cfg.get("scheme_by_group"):
        return dict(cfg["scheme_by_group"])
    config_path = cfg.get("config")
    tier = cfg.get("tier")
    if config_path and tier:
        return load_hybrid_config(config_path, tier)
    raise ValueError(
        "run_qat cfg needs either 'scheme_by_group' or both 'config' and 'tier'"
    )


def _resolve_scheme_by_tensor(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Get the optional {gguf_tensor_name: ggml_type_name} map from cfg.

    Empty dict when the run has no per-tensor map (the ladder search path) --
    routing then falls back entirely to ``scheme_by_group``, which is what
    happened before budget builds existed.
    """
    if cfg.get("scheme_by_tensor"):
        return dict(cfg["scheme_by_tensor"])
    config_path = cfg.get("config")
    tier = cfg.get("tier")
    if config_path and tier:
        try:
            return load_tensor_config(config_path, tier)
        except (KeyError, OSError, ValueError) as exc:
            _log.warning(
                "Could not load a per-tensor config from %r/%r (%s); "
                "falling back to the per-group map only.", config_path, tier, exc,
            )
    return {}


def _try_load_base_weight_map(
    model_id: str, loaded_from: str
) -> Optional[Dict[str, str]]:
    """Best-effort: the base checkpoint's on-disk ``{tensor_name: shard}`` map.

    Feeds both halves of the 2026-08-05 adapter-key fix (see
    ``magicquant.qat.diskmap``): ``run_qat``'s pre-flight key check and
    ``_save_adapters``'s save-time reconciliation both need the SAME map, so
    it's loaded exactly once here and threaded through.

    Reuses ``magicquant.qat.merge``'s own resolver/loader (not a second
    implementation) specifically so the map used to reconcile adapter keys at
    train time is IDENTICAL to the one ``magicquant qat-merge`` will look
    tensors up in later -- any drift between the two would defeat the whole
    point.

    Returns ``None`` (never raises) when there's nothing real to check
    against:
      * ``loaded_from`` is the offline-tiny-model sentinel (see
        ``_load_model_and_tokenizer`` / ``_build_offline_tiny_model``) -- no
        on-disk checkpoint backs that model at all, so there's nothing to
        resolve keys against and no point spending a network round-trip
        finding that out.
      * any other resolution failure (bad path, no safetensors found, a Hub
        lookup error) -- a model that loaded some other way but has no
        derivable weight map means reconciliation simply can't run, not that
        the run itself is broken. Logged as a WARNING either way, so a
        silently skipped preflight is still visible in the run's log.
    """
    if loaded_from == _OFFLINE_TINY_MODEL_SENTINEL:
        return None
    from magicquant.qat.merge import _load_weight_map, _resolve_base_model_dir

    try:
        model_dir = _resolve_base_model_dir(model_id)
        weight_map, _meta = _load_weight_map(model_dir)
        return weight_map
    except Exception as exc:  # best-effort -- see docstring
        _log.warning(
            "Could not load a base-checkpoint weight map for %r (%s); "
            "adapter target keys will NOT be reconciled against on-disk "
            "names or preflight-checked this run. If the loaded model's "
            "module names differ from its own checkpoint's safetensors "
            "names (the 2026-08-05 'language_model' nesting incident), that "
            "will only surface later, at merge time.", model_id, exc,
        )
        return None


def _config_hash(
    model_id: str,
    scheme_by_group: Dict[str, str],
    lora_r: int,
    lora_alpha: float,
    scheme_by_tensor: Optional[Dict[str, str]] = None,
    expert_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Identity of everything a resumed checkpoint's adapters were trained against.

    ``lora_r``/``lora_alpha`` are folded in UNCONDITIONALLY, alongside
    ``scheme_by_group`` -- unlike ``scheme_by_tensor``/``expert_config``, every
    run has a base LoRA rank/alpha, so there is no "old checkpoint predates
    this key" case to preserve. Pass the RESOLVED values (what ``wrap_model``
    was actually called with), not a raw ``cfg.get(...)`` -- a run that omits
    the key and one that passes the default explicitly must hash identically.
    A changed ``lora_r`` fails loudly later inside the trainer's checkpoint
    load (a torch shape mismatch on ``lora_A``/``lora_B``); a changed
    ``lora_alpha`` alone changes no tensor shape and would otherwise resume
    silently under a different effective adapter scale -- exactly the failure
    class ``_check_config_identity`` exists to stop.

    BEHAVIOR DELTA: because this makes the hash unconditional, any checkpoint
    already written under a version of this function without ``lora_r``/
    ``lora_alpha`` in the payload will refuse to resume (its saved hash no
    longer matches), even if lora_r/lora_alpha are unchanged. Restart it with
    ``--no-resume``.

    ``scheme_by_tensor`` and ``expert_config`` are folded in only when non-empty,
    so a run with neither hashes exactly as it did before they existed (an old
    Linear-only checkpoint stays resumable). They must be *in* the hash when
    present: adapters trained against a per-tensor Q2_K expert layout, or at a
    different expert rank/quant mode, are as wrong to resume into a changed
    config as adapters trained against a different scheme_by_group -- which is
    the exact failure ``_check_config_identity`` exists to stop.
    """
    payload: Dict[str, Any] = {
        "model": model_id,
        "scheme_by_group": scheme_by_group,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }
    if scheme_by_tensor:
        payload["scheme_by_tensor"] = scheme_by_tensor
    if expert_config:
        payload["expert_config"] = expert_config
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _freeze_to_lora_only(model) -> int:
    """Freeze everything except LoRA params. Returns trainable element count.

    Covers both adapter families: ``QATLinear``'s 2-D ``lora_A``/``lora_B`` and
    the fused 3-D expert parametrizations' ``lora_expert_A``/``lora_expert_B``.
    """
    for p in model.parameters():
        p.requires_grad = False
    n_trainable = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            n_trainable += module.lora_A.numel() + module.lora_B.numel()
    for p in iter_expert_parametrizations(model):
        if p.lora_r <= 0:
            continue
        p.lora_expert_A.requires_grad = True
        p.lora_expert_B.requires_grad = True
        n_trainable += p.lora_expert_A.numel() + p.lora_expert_B.numel()
    return n_trainable


def _enable_gradient_checkpointing(model) -> bool:
    """Turn on gradient checkpointing if the model supports it.

    Cuts activation memory (re-computes activations in the backward pass instead
    of storing them) for big models. Guarded: models without HF's
    ``gradient_checkpointing_enable`` (or where it raises) are left untouched with
    a warning, so an exotic architecture never fails the run. Returns whether it
    was actually enabled.
    """
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        _log.warning(
            "gradient_checkpointing requested but %s has no callable "
            "gradient_checkpointing_enable; skipping.", type(model).__name__,
        )
        return False
    try:
        # use_reentrant=False is the non-deprecated path; pass it when accepted.
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
    except Exception as exc:  # never let an opt-in memory knob fail the run
        _log.warning("gradient_checkpointing_enable failed (%s); skipping.", exc)
        return False
    # Checkpointing needs inputs that require grad; HF exposes this helper.
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:  # best-effort
            pass
    _log.info("Gradient checkpointing enabled.")
    return True


def _save_adapters(
    model, out_dir: str, weight_map: Optional[Dict[str, str]] = None
) -> str:
    """Save every LoRA adapter to ``adapter_model.safetensors``.

    Two key families, both consumed by ``magicquant.qat.merge``:

    2-D (``QATLinear``):
        ``"<module_path>.lora_A"`` ``(r, in_features)`` and
        ``"<module_path>.lora_B"`` ``(out_features, r)``. ``module_path`` is the
        ``named_modules()`` path; the merge appends ``.weight`` to find the base
        tensor. Merge: ``W += scale * (B @ A)``.

    3-D (fused MoE experts, ``expert_wrap.FusedExpertQAT``):
        ``"<base_tensor_key>.lora_expert_A"`` ``(E, W.shape[1], r)`` and
        ``"<base_tensor_key>.lora_expert_B"`` ``(E, r, W.shape[2])``.
        ``base_tensor_key`` is the fused parameter's exact safetensors key (a raw
        ``nn.Parameter``, so NO ``.weight`` suffix), e.g.
        ``model.language_model.layers.3.mlp.experts.gate_up_proj``. Merge:
        ``W[e] += expert_scale * (A[e] @ B[e])`` with
        ``expert_scale = expert_lora_alpha / expert_lora_r`` from
        ``qat_meta.json``.

    All tensors are written fp32 on CPU.

    SAVE-TIME key reconciliation (2026-08-05 fix -- see
    ``magicquant.qat.diskmap``): ``module_path``/``base_tensor_key`` above are
    the loaded model's OWN module-graph names, which can differ from what the
    base checkpoint's safetensors actually call the same tensor (Qwen3.6:
    ``model.layers...`` in the module graph vs. ``model.language_model.layers...``
    on disk). If ``weight_map`` is given (the base checkpoint's
    ``model.safetensors.index.json`` weight map, normally loaded once by
    ``run_qat`` and threaded through here), every 2-D key is reconciled
    against it before being written, and the SAME map is passed to
    ``fused_expert_adapter_state`` for the 3-D keys -- so what actually lands
    in ``adapter_model.safetensors`` is always the DISK key, never a name the
    merge step would refuse. ``weight_map=None`` (the default) preserves the
    pre-fix behavior for callers with no real checkpoint to reconcile against.

    Raises:
        QATKeyReconciliationError: ``weight_map`` was given and one or more
            ``QATLinear`` target keys didn't resolve against it (listed all
            at once), or (propagated from ``fused_expert_adapter_state``) one
            or more fused-expert target keys didn't. ``run_qat``'s preflight
            should already have caught this before training started; this is
            the save-time backstop.
    """
    from safetensors.torch import save_file

    weight_map_keys = frozenset(weight_map.keys()) if weight_map is not None else None
    state: Dict[str, torch.Tensor] = {}
    unresolved: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, QATLinear):
            continue
        disk_module_path = name
        if weight_map_keys is not None:
            resolved = reconcile_key_to_disk(f"{name}.weight", weight_map_keys)
            if resolved is None:
                unresolved.append(f"{name}.weight")
                continue
            disk_module_path = resolved[: -len(".weight")]
        state[f"{disk_module_path}.lora_A"] = module.lora_A.detach().to(torch.float32).cpu()
        state[f"{disk_module_path}.lora_B"] = module.lora_B.detach().to(torch.float32).cpu()
    if unresolved:
        raise QATKeyReconciliationError(
            f"{len(unresolved)} QATLinear adapter target key(s) could not be "
            f"resolved against the base checkpoint's on-disk weight map at "
            f"save time: {sorted(unresolved)}"
        )
    state.update(fused_expert_adapter_state(model, weight_map))
    path = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(state, path)
    return path


# ── checkpoint / resume ─────────────────────────────────────────────────────
#
# HF Trainer's default periodic checkpoint saves the model's FULL state_dict --
# every frozen base-model weight, not just the trainable LoRA adapters. For a
# 35B base that's tens of GB written to disk on every save, which would itself
# risk filling the disk on an unattended overnight run (the exact failure mode
# this feature exists to prevent). The base is fully and deterministically
# reconstructed on every run anyway (same source model + same scheme_by_group
# -> identical fake-quant wrapping, see run_qat below), so only the trainable
# LoRA params need to survive a restart. `_install_lora_only_checkpoint_save`
# patches a Trainer instance's `_save` so its checkpoints hold only those --
# optimizer/scheduler/RNG state is untouched (already small: AdamW only
# tracks trainable params).
#
# Loading a partial state dict back in is a documented HF path, not a hack:
# `Trainer._load_from_checkpoint` calls `model.load_state_dict(state_dict,
# False)` -- `strict=False` -- so missing base-weight keys are silently
# tolerated (logged as "missing keys", never raised).

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")

# Files that mark a checkpoint dir as fully written (vs. killed mid-save).
_CHECKPOINT_STATE_FILE = "trainer_state.json"
_CHECKPOINT_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

# Records which (model, scheme_by_group) a checkpoint was written under -- see
# _check_config_identity below.
_CONFIG_HASH_FILENAME = "qat_config_hash.txt"


def _write_config_hash(checkpoint_dir: str, config_hash: str) -> None:
    """Best-effort: record which cfg/scheme produced ``checkpoint_dir``.

    Called from the periodic-save path (see ``_install_lora_only_checkpoint_save``),
    so it inherits the same warn-and-continue contract: a write failure here
    (e.g. disk full, same failure mode the save itself just hit) must not take
    down an otherwise-successful checkpoint write, let alone the whole run.
    """
    try:
        with open(
            os.path.join(checkpoint_dir, _CONFIG_HASH_FILENAME), "w", encoding="utf-8"
        ) as f:
            f.write(config_hash)
    except Exception as exc:  # best-effort -- see docstring
        _log.warning("Could not write config-identity hash to %s: %s", checkpoint_dir, exc)


def _check_config_identity(checkpoint_dir: str, config_hash: str) -> None:
    """Refuse to resume from ``checkpoint_dir`` if it was written under a
    DIFFERENT (model, scheme_by_group) than the current run.

    A checkpoint with no hash file predates this guard (or its best-effort
    write previously failed) -- resuming from it is unchanged prior behavior,
    not a new risk, so it's allowed through silently. A checkpoint WITH a
    hash file that disagrees means the model id, per-group/per-tensor quant
    scheme, base or expert LoRA rank/alpha, or expert config changed since it
    was written: resuming those LoRA adapters would train them against a
    frozen base / fake-quant config (or adapter shape/scale) that no longer
    matches what this run is compensating for, silently corrupting the
    result. That must stop the run outright rather than degrade to a fresh
    start -- the safe fallback is explicit, named in the error: ``--no-resume``.
    """
    hash_path = os.path.join(checkpoint_dir, _CONFIG_HASH_FILENAME)
    if not os.path.isfile(hash_path):
        return
    with open(hash_path, encoding="utf-8") as f:
        saved_hash = f.read().strip()
    if saved_hash and saved_hash != config_hash:
        raise RuntimeError(
            f"Refusing to resume from {checkpoint_dir!r}: its config_hash "
            f"({saved_hash}) does not match the current run's config_hash "
            f"({config_hash}) -- the model, scheme_by_group/scheme_by_tensor, "
            f"lora_r/lora_alpha, or expert_config changed since this "
            f"checkpoint was written. Resuming LoRA adapters trained against "
            f"a different frozen base/quant config or adapter rank/scale "
            f"would silently corrupt the run. Start fresh with --no-resume "
            f"(or resume=False) if this is intentional."
        )


def _install_lora_only_checkpoint_save(trainer, config_hash: Optional[str] = None) -> None:
    """Patch ``trainer`` so its periodic checkpoints save only trainable
    (LoRA) params, in place, via an instance-level override of ``_save``.

    An instance-level shadow (same pattern as this module's
    ``gradient_checkpointing_enable`` test spies) rather than a Trainer
    subclass: it works uniformly whether ``trainer`` is the real
    ``transformers.Trainer`` or a lightweight test double, and never disturbs
    what class ``trainer`` actually is.

    No-ops (logs at debug and returns) if ``trainer`` has no ``_save`` to
    override -- an unrecognized/minimal Trainer implementation should keep
    whatever default checkpoint behavior it has, not crash.

    The periodic save itself is best-effort: an exception here (disk full,
    permissions, anything) is logged as a WARNING and swallowed rather than
    propagated, so it can never abort a multi-hour unattended run over one
    checkpoint write. This does NOT apply to the run's FINAL adapter save
    (``_save_adapters``, called once after ``trainer.train()`` returns) --
    that one still raises on failure, since losing the actual trained result
    at the end of the run is not something to shrug off.

    If ``config_hash`` is given, it's written into the checkpoint dir after
    a successful save (see ``_write_config_hash`` / ``_check_config_identity``).
    """
    original_save = getattr(trainer, "_save", None)
    if not callable(original_save):
        _log.debug(
            "%s has no _save method to patch; periodic checkpoints (if any) "
            "will use its own default full-state-dict save.",
            type(trainer).__name__,
        )
        return

    def _lora_only_save(output_dir=None, state_dict=None):
        try:
            if state_dict is None:
                model = trainer.model
                trainable_names = {
                    n for n, p in model.named_parameters() if p.requires_grad
                }
                state_dict = {
                    k: v for k, v in model.state_dict().items() if k in trainable_names
                }
            original_save(output_dir, state_dict=state_dict)
            if config_hash and output_dir:
                _write_config_hash(output_dir, config_hash)
        except Exception as exc:  # best-effort -- see docstring
            msg = (
                f"Periodic checkpoint save to {output_dir!r} failed ({exc}); "
                "continuing training without this checkpoint. The final "
                "adapter save at the end of the run still raises on failure."
            )
            _log.warning(msg)
            print(f"WARNING: {msg}", flush=True)

    trainer._save = _lora_only_save


def _list_checkpoint_dirs(trainer_output_dir: str) -> List[str]:
    """``checkpoint-<N>`` subdirectories of ``trainer_output_dir``, newest first.

    Returns ``[]`` (never raises) if the directory doesn't exist yet -- the
    normal state before a run's first checkpoint.
    """
    if not os.path.isdir(trainer_output_dir):
        return []
    found = []
    for name in os.listdir(trainer_output_dir):
        m = _CHECKPOINT_DIR_RE.match(name)
        if not m:
            continue
        path = os.path.join(trainer_output_dir, name)
        if os.path.isdir(path):
            found.append((int(m.group(1)), path))
    found.sort(key=lambda t: t[0], reverse=True)
    return [path for _, path in found]


def _is_checkpoint_complete(path: str) -> bool:
    """Whether ``path`` has the files a resume needs (trainer state + weights).

    A process killed mid-checkpoint-write (this feature's whole reason for
    existing) can leave a ``checkpoint-N`` directory with some files but not
    others. Resuming from that half-written state would raise inside HF's
    loader instead of degrading gracefully, so it's treated as absent.
    """
    if not os.path.isfile(os.path.join(path, _CHECKPOINT_STATE_FILE)):
        return False
    return any(
        os.path.isfile(os.path.join(path, w)) for w in _CHECKPOINT_WEIGHT_FILES
    )


def _resolve_resume_checkpoint(trainer_output_dir: str, resume: bool) -> Optional[str]:
    """Pick the newest COMPLETE checkpoint under ``trainer_output_dir``, or
    ``None`` to start fresh.

    Never raises. ``resume=False``, no checkpoints, or every checkpoint found
    being incomplete/corrupt all resolve to ``None`` -- an absent or bad
    checkpoint must never fail an unattended run, it must just start over.
    """
    if not resume:
        return None
    for path in _list_checkpoint_dirs(trainer_output_dir):
        if _is_checkpoint_complete(path):
            return path
        _log.warning(
            "Skipping incomplete checkpoint %s (missing %s or a weights file); "
            "trying an older one.", path, _CHECKPOINT_STATE_FILE,
        )
    return None


@dataclass
class _RunConfig:
    """Resolved run_qat hyperparameters -- see run_qat's docstring for the
    documented default for each field. Plain data holder for the ~20 locals
    ``_parse_run_cfg`` resolves out of ``cfg``, threaded through the rest of
    ``run_qat`` and into ``qat_meta.json``.
    """

    lora_r: int
    lora_alpha: float
    epochs: float
    max_steps: int
    lr: float
    max_seq_len: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler: str
    gradient_checkpointing: bool
    save_steps: int
    save_total_limit: int
    resume: bool
    expert_lora_r: int
    expert_lora_alpha: float
    expert_quant_mode: str
    wrap_experts: bool


def _parse_run_cfg(cfg: Dict[str, Any]) -> _RunConfig:
    """Resolve run_qat's hyperparameters out of ``cfg`` (defaults documented
    on run_qat itself). Raises ValueError on an invalid ``expert_quant_mode``
    -- BEFORE the model loads, so a bad config fails in ~1s, not after a
    multi-minute load on a large model (see run_qat).
    """
    lora_r = int(cfg.get("lora_r", 8))
    lora_alpha = float(cfg.get("lora_alpha", 16))
    epochs = float(cfg.get("epochs", 1))
    max_steps = int(cfg.get("max_steps", -1))
    lr = float(cfg.get("lr", 2e-4))
    max_seq_len = int(cfg.get("max_seq_len", 512))

    # Training schedule defaults (production-grade, all overridable via cfg).
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    lr_scheduler = str(cfg.get("lr_scheduler", "cosine"))
    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))

    # Checkpoint/resume defaults -- see the module-level "checkpoint / resume"
    # section for why only LoRA params get written to disk.
    save_steps = int(cfg.get("save_steps", 100))
    save_total_limit = int(cfg.get("save_total_limit", 3))
    resume = bool(cfg.get("resume", True))

    expert_lora_r = int(cfg.get("expert_lora_r", 4))
    expert_lora_alpha = float(cfg.get("expert_lora_alpha", 2 * expert_lora_r))
    expert_quant_mode = str(cfg.get("expert_quant_mode", MODE_LIVE))
    wrap_experts = bool(cfg.get("wrap_experts", True))
    if expert_quant_mode not in EXPERT_QUANT_MODES:
        raise ValueError(
            f"expert_quant_mode must be one of {EXPERT_QUANT_MODES}, "
            f"got {expert_quant_mode!r}"
        )
    return _RunConfig(
        lora_r=lora_r, lora_alpha=lora_alpha, epochs=epochs, max_steps=max_steps,
        lr=lr, max_seq_len=max_seq_len, warmup_ratio=warmup_ratio,
        weight_decay=weight_decay, max_grad_norm=max_grad_norm,
        lr_scheduler=lr_scheduler, gradient_checkpointing=gradient_checkpointing,
        save_steps=save_steps, save_total_limit=save_total_limit, resume=resume,
        expert_lora_r=expert_lora_r, expert_lora_alpha=expert_lora_alpha,
        expert_quant_mode=expert_quant_mode, wrap_experts=wrap_experts,
    )


def _log_expert_cost(
    expert_cost: Dict[str, Any],
    expert_lora_r: int,
    expert_lora_alpha: float,
    expert_quant_mode: str,
) -> None:
    """Log (and print) the fused 3-D expert QAT cost estimate.

    Both _log.info and print, for the same reason the resume message in
    run_qat does: this module's logger has no handler in a real invocation,
    and this is the number that decides whether the run can finish at all.
    """
    _expert_msg = (
        f"Fused 3-D expert QAT: {expert_cost['n_expert_tensors']} tensors, "
        f"{expert_cost['base_elements'] / 1e9:.1f}e9 base elements covered, "
        f"r={expert_lora_r} alpha={expert_lora_alpha} "
        f"mode={expert_quant_mode!r}; adapters "
        f"{expert_cost['lora_params'] / 1e6:.1f}M params "
        f"(~{expert_cost['train_gib']:.2f} GiB incl. grads + AdamW moments)"
    )
    if expert_cost["live_forward_seconds"] > 0:
        _expert_msg += (
            f"; estimated live fake-quant cost "
            f"~{expert_cost['live_forward_seconds']:.0f} s per forward pass"
        )
    _log.info(_expert_msg)
    print(_expert_msg, flush=True)
    if expert_cost["live_forward_seconds"] > 60:
        _slow = (
            f"WARNING: live expert fake-quant is estimated at "
            f"~{expert_cost['live_forward_seconds'] / 60:.0f} min per forward "
            f"pass on this model -- a training step costs at least that much. "
            f"Set expert_quant_mode='frozen' (fake-quantize the expert base "
            f"once at wrap time) to make the run finish, accepting that the "
            f"adapter delta is then not re-quantized during training."
        )
        _log.warning(_slow)
        print(_slow, flush=True)


def _build_training_args(trainer_output_dir: str, rc: _RunConfig, use_bf16: bool, use_fp16: bool):
    """Build run_qat's TrainingArguments (schedule/checkpoint defaults are
    documented on run_qat itself).

    transformers-5 compat: ``warmup_ratio`` was removed upstream and folded
    into ``warmup_steps``, which now accepts a float in [0, 1) meaning "ratio
    of total training steps" -- semantically identical to the old
    ``warmup_ratio``. Same try/TypeError fallback pattern as
    ``_from_pretrained``'s dtype= handling: pass the legacy kwarg first (older
    transformers), retry with the mapped one only when the TypeError names
    warmup_ratio (any other TypeError propagates untouched).
    """
    from transformers import TrainingArguments

    try:
        return _construct_training_args(
            TrainingArguments, trainer_output_dir, rc, use_bf16, use_fp16,
            warmup_kwargs={"warmup_ratio": rc.warmup_ratio},
        )
    except TypeError as exc:
        if "warmup_ratio" not in str(exc):
            raise
        return _construct_training_args(
            TrainingArguments, trainer_output_dir, rc, use_bf16, use_fp16,
            warmup_kwargs={"warmup_steps": float(rc.warmup_ratio)},
        )


def _construct_training_args(TrainingArguments, trainer_output_dir, rc,
                             use_bf16, use_fp16, *, warmup_kwargs):
    return TrainingArguments(
        output_dir=trainer_output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=rc.epochs,
        max_steps=rc.max_steps,
        learning_rate=rc.lr,
        **warmup_kwargs,
        weight_decay=rc.weight_decay,
        max_grad_norm=rc.max_grad_norm,
        lr_scheduler_type=rc.lr_scheduler,
        logging_steps=1,
        save_strategy="steps",
        save_steps=rc.save_steps,
        save_total_limit=rc.save_total_limit,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=0,
    )


def _build_run_meta(
    cfg: Dict[str, Any],
    rc: _RunConfig,
    loaded_from: str,
    base_weight_map: Optional[Dict[str, str]],
    scheme_by_group: Dict[str, str],
    scheme_by_tensor: Dict[str, str],
    config_hash: str,
    n_trainable: int,
    adapter_path: str,
    resume_from_checkpoint: Optional[str],
    expert_params: list,
    expert_cost: Dict[str, Any],
    model,
) -> Dict[str, Any]:
    """Build the ``qat_meta.json`` dict."""
    meta: Dict[str, Any] = {
        "model": cfg["model"],
        "loaded_from": loaded_from,
        "adapter_keys_reconciled_to_disk": base_weight_map is not None,
        "scheme_by_group": scheme_by_group,
        "config_hash": config_hash,
        "lora_r": rc.lora_r,
        "lora_alpha": rc.lora_alpha,
        "epochs": rc.epochs,
        "max_steps": rc.max_steps,
        "lr": rc.lr,
        "max_seq_len": rc.max_seq_len,
        "warmup_ratio": rc.warmup_ratio,
        "weight_decay": rc.weight_decay,
        "max_grad_norm": rc.max_grad_norm,
        "lr_scheduler": rc.lr_scheduler,
        "gradient_checkpointing": rc.gradient_checkpointing,
        "save_steps": rc.save_steps,
        "save_total_limit": rc.save_total_limit,
        "resume": rc.resume,
        "resumed_from_checkpoint": resume_from_checkpoint,
        "trainable_params": n_trainable,
        "adapter_file": os.path.basename(adapter_path),
        # Fused 3-D MoE experts. expert_lora_r/expert_lora_alpha are READ BY
        # THE MERGE (magicquant.qat.merge computes its 3-D scale from them and
        # falls back to lora_r/lora_alpha when absent), so they are written
        # unconditionally -- even when no expert was wrapped -- rather than only
        # when they differ from the Linear values.
        "expert_lora_r": rc.expert_lora_r,
        "expert_lora_alpha": rc.expert_lora_alpha,
        "expert_quant_mode": rc.expert_quant_mode,
        "wrap_experts": rc.wrap_experts,
        "n_expert_tensors": len(expert_params),
        "expert_adapter_params": expert_cost["lora_params"],
        "expert_adapters": fused_expert_adapter_meta(model),
    }
    if scheme_by_tensor:
        # The per-tensor map is what actually routed the run; record its size
        # and a hash rather than 750 entries inline.
        meta["scheme_by_tensor_count"] = len(scheme_by_tensor)
        meta["scheme_by_tensor_hash"] = hashlib.sha256(
            json.dumps(scheme_by_tensor, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    return meta


def run_qat(cfg: Dict[str, Any]) -> str:
    """Run QAT-LoRA per ``cfg`` and return the output adapter directory.

    Required cfg keys: ``model``, ``dataset``, ``out``, and either
    ``scheme_by_group`` or (``config`` + ``tier``).
    Optional: ``lora_r`` (8), ``lora_alpha`` (16), ``epochs`` (1), ``max_steps``
    (-1 = full), ``lr`` (2e-4), ``max_seq_len`` (512).

    Fused 3-D MoE experts (``mlp.experts.gate_up_proj`` & co) get their own
    adapters and their own knobs, because 256 experts x 40 layers multiplies a
    rank by ~10k: ``expert_lora_r`` (4), ``expert_lora_alpha`` (8),
    ``expert_quant_mode`` (``"live"``), ``wrap_experts`` (True). See
    ``magicquant.qat.expert_wrap`` for what ``live`` vs ``frozen`` costs and
    means; ``run_qat`` logs the measured per-forward fake-quant estimate for the
    wrapped experts before training starts, so an infeasible configuration is
    visible in the first seconds of a run rather than at dawn.

    When the config source carries a per-tensor map (``tensor_config``, written
    by budget builds), it takes precedence over the per-group map for both
    Linears and experts.

    Training schedule (sensible defaults, all overridable): ``warmup_ratio``
    (0.03), ``weight_decay`` (0.0), ``max_grad_norm`` (1.0), ``lr_scheduler``
    ('cosine'). Set ``gradient_checkpointing`` (False) to trade compute for
    activation memory on big models.

    Checkpoint/resume (all overridable): ``save_steps`` (100), ``save_total_limit``
    (3), ``resume`` (True). Checkpoints land under ``<out>/_trainer/checkpoint-*``
    and hold only the trainable LoRA params (see ``_install_lora_only_checkpoint_save``)
    -- the frozen base is never checkpointed, since it's rebuilt identically from
    ``model`` + the resolved ``scheme_by_group`` on every run. With ``resume=True``
    (the default), a prior run's newest *complete* checkpoint in ``<out>`` is
    resumed from automatically; a missing or corrupt checkpoint silently falls
    back to a fresh start rather than failing the run.
    """
    from transformers import Trainer

    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)

    rc = _parse_run_cfg(cfg)

    scheme_by_group = _resolve_scheme_by_group(cfg)
    scheme_by_tensor = _resolve_scheme_by_tensor(cfg)
    expert_config = {
        "wrap_experts": rc.wrap_experts,
        "expert_lora_r": rc.expert_lora_r,
        "expert_lora_alpha": rc.expert_lora_alpha,
        "expert_quant_mode": rc.expert_quant_mode,
    } if rc.wrap_experts else None
    # Computed once, reused for both the checkpoint config-identity guard
    # (below) and qat_meta.json -- see _check_config_identity's docstring.
    # lora_r/lora_alpha are the RESOLVED locals from _RunConfig (not a raw
    # cfg.get), matching what wrap_model below is actually called with.
    config_hash = _config_hash(
        cfg["model"], scheme_by_group, rc.lora_r, rc.lora_alpha,
        scheme_by_tensor, expert_config,
    )

    model, tokenizer, loaded_from = _load_model_and_tokenizer(
        cfg["model"], dtype=cfg.get("dtype")
    )

    # Wrap routed Linears as fake-quant QATLinears and fused 3-D MoE expert
    # parameters as FusedExpertQAT parametrizations, then freeze.
    wrap_model(
        model,
        scheme_by_group,
        TensorGroupClassifier(),
        lora_r=rc.lora_r,
        lora_alpha=rc.lora_alpha,
        scheme_by_tensor=scheme_by_tensor,
        wrap_experts=rc.wrap_experts,
        expert_lora_r=rc.expert_lora_r,
        expert_lora_alpha=rc.expert_lora_alpha,
        expert_quant_mode=rc.expert_quant_mode,
    )
    n_qat = sum(1 for m in model.modules() if isinstance(m, QATLinear))
    expert_params = list(iter_expert_parametrizations(model))
    n_trainable = _freeze_to_lora_only(model)
    expert_cost = estimate_expert_qat_cost(expert_params)
    _log.info(
        "Wrapped %d Linear modules as QATLinear and %d fused 3-D expert "
        "parameters (%d trainable LoRA params).",
        n_qat, len(expert_params), n_trainable,
    )
    if n_qat == 0 and not expert_params:
        _log.warning(
            "No QAT layers were created (the scheme config matched no routable "
            "Linear or fused-expert names — check the model's text-decoder naming "
            "vs hf_to_ggml_name); training has no trainable parameters."
        )

    # ── PRE-FLIGHT: adapter target keys vs. the base checkpoint's real disk
    # keys (2026-08-05 fix, blocker #2 -- see magicquant.qat.diskmap). Every
    # key an adapter will eventually target is checked against the base
    # checkpoint's own model.safetensors.index.json weight map RIGHT NOW,
    # before a single training step runs. Without this, a module-graph vs.
    # on-disk naming mismatch (Qwen3.6: `model.layers...` in the loaded
    # model vs. `model.language_model.layers...` on disk) surfaces only when
    # `magicquant qat-merge` refuses the finished adapters -- 390 of 391 keys,
    # after an entire overnight run, on the actual incident this closes.
    base_weight_map = _try_load_base_weight_map(cfg["model"], loaded_from)
    if base_weight_map is not None:
        linear_names = [
            name for name, m in model.named_modules() if isinstance(m, QATLinear)
        ]
        expert_names = [
            p.param_name for p in expert_params if p.lora_r > 0 and p.param_name
        ]
        # Raises QATKeyReconciliationError (with the full unmapped list) if
        # anything fails to resolve -- deliberately NOT caught here. An
        # unattended overnight run should fail loudly in its first minutes,
        # not silently proceed toward a merge that will refuse its adapters.
        resolve_adapter_targets(linear_names, expert_names, base_weight_map)
        _log.info(
            "Adapter target key preflight OK: %d Linear + %d expert key(s) "
            "resolve against the base checkpoint's on-disk weight map.",
            len(linear_names), len(expert_names),
        )

    if expert_params:
        _log_expert_cost(
            expert_cost, rc.expert_lora_r, rc.expert_lora_alpha, rc.expert_quant_mode
        )

    if rc.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    rows = _read_jsonl(cfg["dataset"])
    train_examples = _build_dataset(tokenizer, rows, rc.max_seq_len)
    if not train_examples:
        raise ValueError(
            f"No trainable examples built from {cfg['dataset']!r} "
            "(each row needs a 'messages' list with an assistant turn)."
        )

    collator = CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id or 0)

    # Match the trainer's mixed-precision flag to the model's loaded dtype.
    model_dtype = next(model.parameters()).dtype
    use_bf16 = model_dtype == torch.bfloat16
    use_fp16 = model_dtype == torch.float16

    trainer_output_dir = os.path.join(out_dir, "_trainer")

    # Resolve resume BEFORE constructing TrainingArguments purely for logging
    # order (the decision doesn't depend on args); never raises -- see
    # _resolve_resume_checkpoint's docstring.
    #
    # Both _log.info AND print: this module's `_log` is a bare stdlib logger
    # that magicquant.logging.configure_logging() never attaches a handler to
    # (pre-existing across several modules, not new here), so on its own it is
    # silent in every real invocation. The resume/fresh-start decision is the
    # one thing an unattended overnight run most needs visible in its captured
    # stdout log, so it also goes through print(..., flush=True) here.
    resume_from_checkpoint = _resolve_resume_checkpoint(trainer_output_dir, rc.resume)
    if resume_from_checkpoint:
        # Config-identity guard: an incomplete/corrupt checkpoint degrades
        # silently to a fresh start (see _resolve_resume_checkpoint above),
        # but a checkpoint that's fully valid yet belongs to a DIFFERENT
        # (model, scheme_by_group) is a distinct, intentional refusal -- see
        # _check_config_identity's docstring. This raises, not warns.
        _check_config_identity(resume_from_checkpoint, config_hash)
        _resume_msg = f"Resuming QAT from checkpoint: {resume_from_checkpoint}"
    elif rc.resume:
        _resume_msg = f"No usable checkpoint found in {trainer_output_dir}; starting fresh."
    else:
        _resume_msg = (
            f"resume=False; starting fresh even if a checkpoint exists in "
            f"{trainer_output_dir}."
        )
    _log.info(_resume_msg)
    print(_resume_msg, flush=True)

    args = _build_training_args(trainer_output_dir, rc, use_bf16, use_fp16)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_examples,
        data_collator=collator,
    )
    _install_lora_only_checkpoint_save(trainer, config_hash=config_hash)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── save adapters + metadata ──
    # weight_map is the SAME map the preflight above already validated every
    # key against (loaded once; see _try_load_base_weight_map) -- reused here
    # so every saved key is the base checkpoint's real DISK key, not the
    # loaded model's module-graph name. See _save_adapters's docstring.
    adapter_path = _save_adapters(model, out_dir, weight_map=base_weight_map)
    meta = _build_run_meta(
        cfg=cfg,
        rc=rc,
        loaded_from=loaded_from,
        base_weight_map=base_weight_map,
        scheme_by_group=scheme_by_group,
        scheme_by_tensor=scheme_by_tensor,
        config_hash=config_hash,
        n_trainable=n_trainable,
        adapter_path=adapter_path,
        resume_from_checkpoint=resume_from_checkpoint,
        expert_params=expert_params,
        expert_cost=expert_cost,
        model=model,
    )
    with open(os.path.join(out_dir, "qat_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir
