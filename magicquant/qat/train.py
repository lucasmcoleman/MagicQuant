"""QAT-LoRA training loop (``run_qat``) + completion-only collator.

``run_qat(cfg)`` loads an HF causal-LM, wraps its routed ``nn.Linear`` modules with
per-group fake-quant ``QATLinear``s (so every forward sees the quantized weight
that will ship), trains only the LoRA adapters on a chat dataset with
**completion-only loss** (system/user turns masked), and saves the adapters plus a
``qat_meta.json`` describing the run. Returns the output directory.

Heavy deps (transformers/trl/datasets) are imported lazily inside ``run_qat`` so
``import magicquant.qat`` stays light (only torch).

Offline fallback: if the named HF model can't be downloaded, ``run_qat`` builds a
tiny ``LlamaForCausalLM`` from a small config (with a minimal byte-level
tokenizer) so the CPU smoke test still runs without network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from magicquant.qat.config import load_hybrid_config
from magicquant.qat.wrap import QATLinear, wrap_model
from magicquant.gguf.tensor_groups import TensorGroupClassifier

_log = logging.getLogger("magicquant.qat.train")

# Label id that the loss ignores (HF convention).
IGNORE_INDEX = -100


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

def _load_model_and_tokenizer(model_id: str):
    """Load an HF causal-LM + tokenizer, falling back to a tiny offline model.

    Returns ``(model, tokenizer, loaded_from)`` where ``loaded_from`` is the model
    id or ``"offline-tiny-llama"`` if the download failed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # QAT runs in float32: the fake-quant kernels upcast to fp32 for their
    # block-scale math anyway, and CPU bf16 matmul support is spotty. A bf16
    # base would also mismatch the fp32-init LoRA params in QATLinear.forward.
    dtype = torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        return model, _ensure_pad_and_template(tokenizer), model_id
    except Exception as exc:  # network/cache miss -> offline tiny model
        _log.warning(
            "Could not load %r (%s); falling back to an offline tiny LlamaForCausalLM "
            "so the run can proceed (smoke/CI path).",
            model_id, exc,
        )
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
    return model, tokenizer, "offline-tiny-llama"


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


def _config_hash(model_id: str, scheme_by_group: Dict[str, str]) -> str:
    payload = json.dumps(
        {"model": model_id, "scheme_by_group": scheme_by_group},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _freeze_to_lora_only(model) -> int:
    """Freeze everything except QATLinear LoRA params. Returns trainable count."""
    for p in model.parameters():
        p.requires_grad = False
    n_trainable = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            n_trainable += module.lora_A.numel() + module.lora_B.numel()
    return n_trainable


def _save_adapters(model, out_dir: str) -> str:
    """Save all QATLinear LoRA params to ``adapter_model.safetensors``."""
    from safetensors.torch import save_file

    state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().to(torch.float32).cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().to(torch.float32).cpu()
    path = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(state, path)
    return path


def run_qat(cfg: Dict[str, Any]) -> str:
    """Run QAT-LoRA per ``cfg`` and return the output adapter directory.

    Required cfg keys: ``model``, ``dataset``, ``out``, and either
    ``scheme_by_group`` or (``config`` + ``tier``).
    Optional: ``lora_r`` (8), ``lora_alpha`` (16), ``epochs`` (1), ``max_steps``
    (-1 = full), ``lr`` (2e-4), ``max_seq_len`` (512).
    """
    from transformers import Trainer, TrainingArguments

    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)

    lora_r = int(cfg.get("lora_r", 8))
    lora_alpha = float(cfg.get("lora_alpha", 16))
    epochs = float(cfg.get("epochs", 1))
    max_steps = int(cfg.get("max_steps", -1))
    lr = float(cfg.get("lr", 2e-4))
    max_seq_len = int(cfg.get("max_seq_len", 512))

    scheme_by_group = _resolve_scheme_by_group(cfg)

    model, tokenizer, loaded_from = _load_model_and_tokenizer(cfg["model"])

    # Wrap routed Linears with per-group fake-quant QATLinears, then freeze.
    wrap_model(
        model,
        scheme_by_group,
        TensorGroupClassifier(),
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    n_trainable = _freeze_to_lora_only(model)
    if n_trainable == 0:
        _log.warning(
            "No QATLinear layers were created (scheme_by_group matched nothing); "
            "training has no trainable parameters."
        )

    rows = _read_jsonl(cfg["dataset"])
    train_examples = _build_dataset(tokenizer, rows, max_seq_len)
    if not train_examples:
        raise ValueError(
            f"No trainable examples built from {cfg['dataset']!r} "
            "(each row needs a 'messages' list with an assistant turn)."
        )

    collator = CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id or 0)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "_trainer"),
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_examples,
        data_collator=collator,
    )
    trainer.train()

    # ── save adapters + metadata ──
    adapter_path = _save_adapters(model, out_dir)
    meta = {
        "model": cfg["model"],
        "loaded_from": loaded_from,
        "scheme_by_group": scheme_by_group,
        "config_hash": _config_hash(cfg["model"], scheme_by_group),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "max_steps": max_steps,
        "lr": lr,
        "max_seq_len": max_seq_len,
        "trainable_params": n_trainable,
        "adapter_file": os.path.basename(adapter_path),
    }
    with open(os.path.join(out_dir, "qat_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir
