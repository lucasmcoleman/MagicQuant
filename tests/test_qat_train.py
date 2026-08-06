"""run_qat production wiring: training defaults, gradient checkpointing, the
multimodal-loader fallback order, and the bf16 dtype path.

These tests avoid a full training run by capturing the ``TrainingArguments`` that
``run_qat`` builds (a fake Trainer records the args and no-ops ``train``) and by
exercising ``_resolve_dtype`` / ``_load_model_and_tokenizer`` with stubs. The
end-to-end smoke (one real training step) lives in ``test_qat_smoke.py``.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from magicquant.qat import train as train_mod
from magicquant.qat.train import _resolve_dtype, _load_model_and_tokenizer


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_ds(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        json.dumps({"messages": [{"role": "user", "content": "a"},
                                 {"role": "assistant", "content": "b"}]}) + "\n"
    )
    return str(ds)


class _RecordingTrainer:
    """Stand-in for transformers.Trainer that records args and no-ops train().

    Deliberately has no ``_save`` method: ``_install_lora_only_checkpoint_save``
    must no-op (not crash) against a Trainer double that doesn't expose one, so
    these production-wiring tests also double as coverage for that guard.
    """

    last_args = None
    last_model = None
    last_resume_from_checkpoint = "NOT_CALLED"

    def __init__(self, model=None, args=None, train_dataset=None, data_collator=None):
        type(self).last_args = args
        type(self).last_model = model
        self.model = model
        self.args = args

    def train(self, resume_from_checkpoint=None):
        type(self).last_resume_from_checkpoint = resume_from_checkpoint
        return None


@pytest.fixture
def patched_trainer(monkeypatch):
    """Patch the lazily-imported Trainer inside run_qat.

    run_qat does ``from transformers import Trainer`` at call time. ``transformers``
    is a lazy module, so ``from … import Trainer`` resolves through both
    ``transformers.Trainer`` (the package ``__dict__``) and the backing
    ``transformers.trainer.Trainer`` depending on whether the symbol was already
    materialized; patch both so the fake is picked up regardless of order.
    """
    import transformers
    import transformers.trainer as trainer_mod

    _RecordingTrainer.last_args = None
    _RecordingTrainer.last_model = None
    _RecordingTrainer.last_resume_from_checkpoint = "NOT_CALLED"
    monkeypatch.setattr(trainer_mod, "Trainer", _RecordingTrainer, raising=False)
    monkeypatch.setattr(transformers, "Trainer", _RecordingTrainer, raising=False)
    return _RecordingTrainer


def _base_cfg(tmp_path):
    return {
        "model": "offline-please-fall-back",  # forces the offline tiny model
        "scheme_by_group": {"U": "MXFP4", "Q": "Q6_K"},
        "dataset": _write_ds(tmp_path),
        "out": str(tmp_path / "adapters"),
        "lora_r": 4, "lora_alpha": 8, "epochs": 1, "max_steps": 1,
        "lr": 2e-4, "max_seq_len": 64,
    }


# ── training defaults wired into TrainingArguments ─────────────────────────────

def test_run_qat_defaults_wired_into_training_args(tmp_path, patched_trainer):
    train_mod.run_qat(_base_cfg(tmp_path))
    args = patched_trainer.last_args
    assert args.warmup_ratio == pytest.approx(0.03)
    assert args.weight_decay == pytest.approx(0.0)
    assert args.max_grad_norm == pytest.approx(1.0)
    assert str(args.lr_scheduler_type) == "cosine" or args.lr_scheduler_type == "cosine"


def test_run_qat_overrides_training_defaults(tmp_path, patched_trainer):
    cfg = _base_cfg(tmp_path)
    cfg.update(
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=0.5,
        lr_scheduler="linear",
    )
    train_mod.run_qat(cfg)
    args = patched_trainer.last_args
    assert args.warmup_ratio == pytest.approx(0.1)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.max_grad_norm == pytest.approx(0.5)
    assert str(args.lr_scheduler_type) == "linear" or args.lr_scheduler_type == "linear"


def test_run_qat_backward_compat_existing_keys_only(tmp_path, patched_trainer):
    """A cfg with only the original keys still trains (no new key required)."""
    cfg = _base_cfg(tmp_path)  # only original keys
    out = train_mod.run_qat(cfg)
    meta = json.loads((tmp_path / "adapters" / "qat_meta.json").read_text())
    # new defaults are recorded in meta for reproducibility
    assert meta["warmup_ratio"] == pytest.approx(0.03)
    assert meta["lr_scheduler"] == "cosine"
    assert out == cfg["out"]


# ── fused 3-D MoE expert knobs ────────────────────────────────────────────────

def test_meta_always_records_the_expert_scale_the_merge_reads(tmp_path, patched_trainer):
    """``magicquant.qat.merge`` computes its 3-D expert scale from
    ``expert_lora_r``/``expert_lora_alpha`` in qat_meta.json. They are written
    on EVERY run -- including this one, whose tiny offline llama has no fused
    experts at all -- so the merge never has to infer which fallback applied.
    """
    cfg = _base_cfg(tmp_path)
    train_mod.run_qat(cfg)
    meta = json.loads((tmp_path / "adapters" / "qat_meta.json").read_text())
    assert meta["expert_lora_r"] == 4
    assert meta["expert_lora_alpha"] == pytest.approx(8.0)
    assert meta["expert_quant_mode"] == "live"
    assert meta["wrap_experts"] is True
    assert meta["n_expert_tensors"] == 0        # a dense llama has none
    assert meta["expert_adapters"] == []
    assert meta["expert_adapter_params"] == 0


def test_expert_knobs_flow_from_cfg_into_meta(tmp_path, patched_trainer):
    cfg = dict(_base_cfg(tmp_path), expert_lora_r=8, expert_lora_alpha=24.0,
               expert_quant_mode="frozen")
    train_mod.run_qat(cfg)
    meta = json.loads((tmp_path / "adapters" / "qat_meta.json").read_text())
    assert meta["expert_lora_r"] == 8
    assert meta["expert_lora_alpha"] == pytest.approx(24.0)
    assert meta["expert_quant_mode"] == "frozen"


def test_unknown_expert_quant_mode_is_refused(tmp_path, patched_trainer):
    cfg = dict(_base_cfg(tmp_path), expert_quant_mode="occasionally")
    with pytest.raises(ValueError, match="expert_quant_mode"):
        train_mod.run_qat(cfg)


# ── checkpoint / resume wiring ──────────────────────────────────────────────────

def test_checkpoint_defaults_wired_into_training_args(tmp_path, patched_trainer):
    """Unchanged cfg -> save_strategy='steps' with the documented defaults, not
    the old save_strategy='no' (this is the behavior-visible half of the
    default-safe contract: checkpoints now get written even with no new keys)."""
    train_mod.run_qat(_base_cfg(tmp_path))
    args = patched_trainer.last_args
    assert str(args.save_strategy) == "steps" or args.save_strategy == "steps"
    assert args.save_steps == 100
    assert args.save_total_limit == 3


def test_checkpoint_save_steps_and_limit_overridable(tmp_path, patched_trainer):
    cfg = _base_cfg(tmp_path)
    cfg.update(save_steps=5, save_total_limit=2)
    train_mod.run_qat(cfg)
    args = patched_trainer.last_args
    assert args.save_steps == 5
    assert args.save_total_limit == 2


def test_resume_defaults_to_true_with_no_checkpoint_present(tmp_path, patched_trainer):
    """Default-safe: with nothing under --out, a fresh tmp_path has no checkpoint
    to find, so resume_from_checkpoint must be None (fresh start) even though
    resume=True by default -- absence of a checkpoint is not an error."""
    train_mod.run_qat(_base_cfg(tmp_path))
    assert patched_trainer.last_resume_from_checkpoint is None


def test_resume_picks_up_newest_complete_checkpoint(tmp_path, patched_trainer):
    """A prior (killed) run's checkpoint under <out>/_trainer is detected and
    handed to trainer.train(resume_from_checkpoint=...) without a real Trainer
    ever running -- this isolates the *detection* wiring from the actual HF
    resume machinery (covered separately by the real end-to-end smoke)."""
    cfg = _base_cfg(tmp_path)
    trainer_dir = tmp_path / "adapters" / "_trainer"
    older = trainer_dir / "checkpoint-2"
    newer = trainer_dir / "checkpoint-8"
    for d in (older, newer):
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00")
    train_mod.run_qat(cfg)
    assert patched_trainer.last_resume_from_checkpoint == str(newer)


def test_resume_false_ignores_existing_checkpoint(tmp_path, patched_trainer):
    cfg = _base_cfg(tmp_path)
    cfg["resume"] = False
    trainer_dir = tmp_path / "adapters" / "_trainer"
    ck = trainer_dir / "checkpoint-8"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text("{}")
    (ck / "model.safetensors").write_bytes(b"\x00")
    train_mod.run_qat(cfg)
    assert patched_trainer.last_resume_from_checkpoint is None


def test_resume_skips_incomplete_newest_checkpoint(tmp_path, patched_trainer):
    """A checkpoint dir missing its weights file (killed mid-write) is treated
    as absent, falling back to the next older COMPLETE one instead of handing
    HF's loader a half-written directory."""
    cfg = _base_cfg(tmp_path)
    trainer_dir = tmp_path / "adapters" / "_trainer"
    complete = trainer_dir / "checkpoint-4"
    complete.mkdir(parents=True)
    (complete / "trainer_state.json").write_text("{}")
    (complete / "model.safetensors").write_bytes(b"\x00")
    incomplete = trainer_dir / "checkpoint-9"
    incomplete.mkdir(parents=True)
    (incomplete / "trainer_state.json").write_text("{}")
    # no weights file -- simulates a kill mid-checkpoint-write
    train_mod.run_qat(cfg)
    assert patched_trainer.last_resume_from_checkpoint == str(complete)


def test_checkpoint_meta_records_resume_fields(tmp_path, patched_trainer):
    cfg = _base_cfg(tmp_path)
    train_mod.run_qat(cfg)
    meta = json.loads((tmp_path / "adapters" / "qat_meta.json").read_text())
    assert meta["save_steps"] == 100
    assert meta["save_total_limit"] == 3
    assert meta["resume"] is True
    assert meta["resumed_from_checkpoint"] is None


# ── gradient checkpointing ─────────────────────────────────────────────────────

def test_gradient_checkpointing_off_by_default(tmp_path, patched_trainer, monkeypatch):
    called = {"n": 0}
    real_loader = train_mod._load_model_and_tokenizer

    def _spy_loader(model_id, dtype=None):
        model, tok, src = real_loader(model_id, dtype)
        orig = model.gradient_checkpointing_enable

        def _counting(*a, **k):
            called["n"] += 1
            return orig(*a, **k)

        model.gradient_checkpointing_enable = _counting
        return model, tok, src

    monkeypatch.setattr(train_mod, "_load_model_and_tokenizer", _spy_loader)
    train_mod.run_qat(_base_cfg(tmp_path))
    assert called["n"] == 0


def test_gradient_checkpointing_enabled_when_flag_set(tmp_path, patched_trainer, monkeypatch):
    called = {"n": 0}
    real_loader = train_mod._load_model_and_tokenizer

    def _spy_loader(model_id, dtype=None):
        model, tok, src = real_loader(model_id, dtype)
        orig = model.gradient_checkpointing_enable

        def _counting(*a, **k):
            called["n"] += 1
            return orig(*a, **k)

        model.gradient_checkpointing_enable = _counting
        return model, tok, src

    monkeypatch.setattr(train_mod, "_load_model_and_tokenizer", _spy_loader)
    cfg = _base_cfg(tmp_path)
    cfg["gradient_checkpointing"] = True
    train_mod.run_qat(cfg)
    assert called["n"] == 1


def test_gradient_checkpointing_guarded_for_unsupported_model(tmp_path, patched_trainer, monkeypatch):
    """A model whose gradient_checkpointing_enable isn't callable doesn't crash."""
    real_loader = train_mod._load_model_and_tokenizer

    def _spy_loader(model_id, dtype=None):
        model, tok, src = real_loader(model_id, dtype)
        # simulate an exotic model lacking the HF mixin's enable hook (shadow on
        # the instance so other tests' models are unaffected)
        model.gradient_checkpointing_enable = None  # not callable
        return model, tok, src

    monkeypatch.setattr(train_mod, "_load_model_and_tokenizer", _spy_loader)
    cfg = _base_cfg(tmp_path)
    cfg["gradient_checkpointing"] = True
    # must not raise
    out = train_mod.run_qat(cfg)
    assert out == cfg["out"]


# ── _resolve_dtype (bf16 path) ─────────────────────────────────────────────────

def test_resolve_dtype_bf16_aliases():
    assert _resolve_dtype("bf16") is torch.bfloat16
    assert _resolve_dtype("bfloat16") is torch.bfloat16
    assert _resolve_dtype(torch.bfloat16) is torch.bfloat16


def test_resolve_dtype_other_paths():
    assert _resolve_dtype("fp16") is torch.float16
    assert _resolve_dtype("float32") is torch.float32
    # None -> bf16 on GPU, fp32 on CPU
    expected = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    assert _resolve_dtype(None) is expected
    assert _resolve_dtype("auto") is expected


# ── multimodal loader fallback order ───────────────────────────────────────────

def test_load_model_fallback_order_honored(monkeypatch):
    """_load_model_and_tokenizer tries causal-LM, then the multimodal/conditional
    auto-classes in order, returning the first that loads."""
    import transformers

    tried = []

    class _FakeModel:
        pass

    def _make_cls(name, succeed):
        class _Cls:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                tried.append(name)
                if succeed:
                    return _FakeModel()
                raise RuntimeError(f"{name} cannot load this model")
        return _Cls

    # AutoTokenizer must succeed (else we fall to the offline tiny model).
    class _FakeTok:
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        chat_template = "x"

        @staticmethod
        def from_pretrained(model_id):
            return _FakeTok()

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTok)
    # Causal-LM fails; the first multimodal class (MultimodalLM) succeeds.
    monkeypatch.setattr(transformers, "AutoModelForCausalLM",
                        _make_cls("AutoModelForCausalLM", succeed=False), raising=False)
    monkeypatch.setattr(transformers, "AutoModelForMultimodalLM",
                        _make_cls("AutoModelForMultimodalLM", succeed=True), raising=False)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText",
                        _make_cls("AutoModelForImageTextToText", succeed=True), raising=False)

    model, tok, src = _load_model_and_tokenizer("some/gemma-multimodal")
    assert isinstance(model, _FakeModel)
    # causal-LM tried first, then the conditional-gen class — and we stopped there
    assert tried == ["AutoModelForCausalLM", "AutoModelForMultimodalLM"]
    assert src == "some/gemma-multimodal"


def test_load_model_falls_back_to_offline_when_all_classes_fail(monkeypatch):
    """If every auto-class fails to load, we get the offline tiny Llama."""
    import transformers

    class _FailCls:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            raise RuntimeError("nope")

    class _FakeTok:
        pad_token_id = 0
        eos_token = "</s>"
        chat_template = "x"

        @staticmethod
        def from_pretrained(model_id):
            return _FakeTok()

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTok)
    for name in ["AutoModelForCausalLM", "AutoModelForMultimodalLM",
                 "AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"]:
        monkeypatch.setattr(transformers, name, _FailCls, raising=False)

    model, tok, src = _load_model_and_tokenizer("some/unloadable")
    assert src == "offline-tiny-llama"
