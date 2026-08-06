"""End-to-end coverage for the 2026-08-05 adapter-key reconciliation fix,
through the real ``run_qat`` entry point.

The incident: ``run_qat`` wraps modules by walking the LOADED model's own
module graph (``model.layers...``), while Qwen3.6-35B-A3B's actual checkpoint
on disk (``model.safetensors.index.json``) names the same tensors
``model.language_model.layers...``. ``_save_adapters`` wrote the module-graph
names verbatim -- 390 of 391 adapter target keys didn't exist in the base
model's weight map, discovered only when ``magicquant qat-merge`` refused
them, after an entire training run.

These tests build a REAL tiny ``LlamaForCausalLM`` (so ``wrap_model`` routes
real ``nn.Linear``s under real ``model.layers...`` names) and a hand-written
``model.safetensors.index.json`` in a temp directory standing in for the base
checkpoint, then monkeypatch ``_load_model_and_tokenizer`` to return that
model with ``loaded_from`` set to the temp directory -- so
``_try_load_base_weight_map`` resolves it exactly the way it would a real
local Foundry ``output/.../source`` directory (no network, no download).
"""

import json

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file

from magicquant.qat import train as train_mod
from magicquant.qat.diskmap import QATKeyReconciliationError
from magicquant.qat.wrap import QATLinear, wrap_model
from magicquant.gguf.tensor_groups import TensorGroupClassifier


# ── shared trainer double (mirrors test_qat_train.py's pattern) ─────────────

class _RecordingTrainer:
    last_args = None
    last_train_called = False

    def __init__(self, model=None, args=None, train_dataset=None, data_collator=None):
        type(self).last_args = args
        self.model = model
        self.args = args

    def train(self, resume_from_checkpoint=None):
        type(self).last_train_called = True
        return None


@pytest.fixture
def patched_trainer(monkeypatch):
    import transformers
    import transformers.trainer as trainer_mod

    _RecordingTrainer.last_args = None
    _RecordingTrainer.last_train_called = False
    monkeypatch.setattr(trainer_mod, "Trainer", _RecordingTrainer, raising=False)
    monkeypatch.setattr(transformers, "Trainer", _RecordingTrainer, raising=False)
    return _RecordingTrainer


# ── tiny real Llama, wired the same way run_qat would load one ─────────────

_SCHEME_BY_GROUP = {"Q": "Q6_K", "U": "MXFP4"}


def _tiny_llama():
    """A real LlamaForCausalLM whose module graph is bare `model.layers...`
    (never `model.language_model.layers...`) -- exactly the incident's
    module-graph side."""
    from transformers import LlamaConfig, LlamaForCausalLM

    tokenizer = train_mod._build_byte_tokenizer()
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config).to(torch.float32)
    return model, tokenizer


def _wrapped_linear_names(model):
    """What run_qat would target, computed the same way wrap_model does --
    used only to build fixture weight maps, never as production logic."""
    probe = wrap_model(
        model, _SCHEME_BY_GROUP, TensorGroupClassifier(), lora_r=2, lora_alpha=4,
        wrap_experts=False,
    )
    names = [n for n, m in probe.named_modules() if isinstance(m, QATLinear)]
    return names


def _write_ds(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        json.dumps({"messages": [{"role": "user", "content": "a"},
                                 {"role": "assistant", "content": "b"}]}) + "\n"
    )
    return str(ds)


def _write_index(model_dir, weight_map):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )


# ── preflight catches an unresolvable key before training starts ──────────

def test_preflight_raises_before_a_single_training_step_on_unresolvable_keys(
    tmp_path, patched_trainer, monkeypatch
):
    """The exact failure mode this fix closes: NONE of the model's own
    module-graph names appear in the base checkpoint's weight map, in any
    reconcilable form. Must fail loudly, and BEFORE trainer.train() runs --
    not after an entire overnight run, when magicquant qat-merge would have
    refused the adapters instead.
    """
    model, tokenizer = _tiny_llama()
    real_names = _wrapped_linear_names(_tiny_llama()[0])
    assert real_names, "fixture sanity: at least one Linear must route"

    base_dir = tmp_path / "base"
    # Deliberately unrelated names -- neither exact nor language_model-
    # reconcilable against anything wrap_model will ever produce.
    _write_index(base_dir, {
        f"totally.unrelated.{i}.weight": "model.safetensors" for i in range(len(real_names))
    })

    def _fake_loader(model_id, dtype=None):
        return model, tokenizer, model_id

    monkeypatch.setattr(train_mod, "_load_model_and_tokenizer", _fake_loader)

    cfg = {
        "model": str(base_dir),
        "scheme_by_group": _SCHEME_BY_GROUP,
        "dataset": _write_ds(tmp_path),
        "out": str(tmp_path / "adapters"),
        "lora_r": 2, "lora_alpha": 4, "epochs": 1, "max_steps": 1,
        "lr": 2e-4, "max_seq_len": 32,
    }

    with pytest.raises(QATKeyReconciliationError):
        train_mod.run_qat(cfg)

    assert patched_trainer.last_train_called is False, (
        "trainer.train() must never be reached when adapter keys can't "
        "resolve -- this is a run-start failure, not an after-hours one"
    )
    assert not (tmp_path / "adapters" / "adapter_model.safetensors").exists()


# ── saved keys are the DISK form, not the module-graph form ────────────────

def test_saved_adapter_keys_use_the_disk_form_with_language_model_nesting(
    tmp_path, patched_trainer, monkeypatch
):
    """The actual incident's shape: the checkpoint on disk nests every tensor
    under 'language_model.' one level deeper than the loaded model's own
    module graph does. Preflight must pass, and the adapters actually
    written to disk must carry the RECONCILED (nested) keys -- the form
    magicquant.qat.merge will look up against the same checkpoint.
    """
    model, tokenizer = _tiny_llama()
    probe_model, _ = _tiny_llama()
    real_names = _wrapped_linear_names(probe_model)
    assert real_names, "fixture sanity: at least one Linear must route"

    base_dir = tmp_path / "base"
    weight_map = {
        f"model.language_model.{name[len('model.'):]}.weight": "model.safetensors"
        for name in real_names
    }
    _write_index(base_dir, weight_map)

    def _fake_loader(model_id, dtype=None):
        return model, tokenizer, model_id

    monkeypatch.setattr(train_mod, "_load_model_and_tokenizer", _fake_loader)

    out_dir = tmp_path / "adapters"
    cfg = {
        "model": str(base_dir),
        "scheme_by_group": _SCHEME_BY_GROUP,
        "dataset": _write_ds(tmp_path),
        "out": str(out_dir),
        "lora_r": 2, "lora_alpha": 4, "epochs": 1, "max_steps": 1,
        "lr": 2e-4, "max_seq_len": 32,
    }

    result = train_mod.run_qat(cfg)
    assert result == str(out_dir)

    meta = json.loads((out_dir / "qat_meta.json").read_text())
    assert meta["adapter_keys_reconciled_to_disk"] is True

    saved = load_file(str(out_dir / "adapter_model.safetensors"))
    saved_bases = {
        k[: -len(".lora_A")] for k in saved if k.endswith(".lora_A")
    }
    expected_bases = {
        f"model.language_model.{name[len('model.'):]}" for name in real_names
    }
    assert saved_bases == expected_bases, (
        f"adapter keys must use the on-disk 'language_model.'-nested form, "
        f"not the loaded model's bare module-graph names. "
        f"got={saved_bases} expected={expected_bases}"
    )
    # And the bare (unreconciled) module-graph form must NOT appear at all.
    for name in real_names:
        assert f"{name}.lora_A" not in saved
        assert f"{name}.lora_B" not in saved
