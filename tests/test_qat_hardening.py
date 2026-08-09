"""Hardening added to magicquant.qat.train ahead of the 2026-08-05 overnight
QAT run (35B MoE, multi-hour, unattended):

  - periodic checkpoint saves are best-effort: an exception during a periodic
    save is logged as a WARNING and training continues, rather than aborting
    a multi-hour run over one checkpoint write. The FINAL adapter save
    (_save_adapters, called once training finishes) is untouched and still
    raises on failure.
  - a config-identity guard: each checkpoint records a hash of
    (model, scheme_by_group, base lora_r/lora_alpha, and -- since fused 3-D
    MoE expert QAT landed -- the per-tensor scheme map and the expert
    rank/alpha/quant-mode). Resuming from a checkpoint whose hash disagrees
    with the CURRENT run's cfg is refused with a clear error naming
    --no-resume, rather than silently training LoRA adapters that were
    initialized against a checkpoint from a different frozen base/quant
    config or a different adapter rank/scale.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from magicquant.qat import train as train_mod


# ── periodic save is best-effort (does not abort the run) ──────────────────

class _NoParamModel:
    def named_parameters(self):
        return []

    def state_dict(self):
        return {}


class _RaisingSaveTrainer:
    """original _save always raises -- e.g. a disk-full write mid-checkpoint."""

    def __init__(self):
        self.model = _NoParamModel()

    def _save(self, output_dir=None, state_dict=None):
        raise OSError("disk full")


def test_periodic_save_exception_does_not_propagate():
    trainer = _RaisingSaveTrainer()
    train_mod._install_lora_only_checkpoint_save(trainer)
    # must not raise: an unattended multi-hour run can't die over one
    # best-effort checkpoint write.
    trainer._save(output_dir="/does/not/matter")


def test_periodic_save_exception_with_explicit_state_dict_does_not_propagate():
    trainer = _RaisingSaveTrainer()
    train_mod._install_lora_only_checkpoint_save(trainer)
    trainer._save(output_dir="/x", state_dict={"a": "b"})


def test_periodic_save_success_path_is_unaffected(tmp_path):
    """The happy path (no exception) still saves exactly as before -- the
    try/except wrapper must not change what a successful save does."""
    calls = []

    class _OkTrainer:
        def __init__(self):
            self.model = _NoParamModel()

        def _save(self, output_dir=None, state_dict=None):
            calls.append((output_dir, state_dict))

    trainer = _OkTrainer()
    train_mod._install_lora_only_checkpoint_save(trainer)
    trainer._save(output_dir=str(tmp_path), state_dict={"lora_A": "A"})
    assert calls == [(str(tmp_path), {"lora_A": "A"})]


def test_config_hash_written_after_successful_periodic_save(tmp_path):
    class _OkTrainer:
        def __init__(self):
            self.model = _NoParamModel()

        def _save(self, output_dir=None, state_dict=None):
            pass  # real _save would write weight files here

    trainer = _OkTrainer()
    train_mod._install_lora_only_checkpoint_save(trainer, config_hash="myhash123")
    trainer._save(output_dir=str(tmp_path))
    hash_file = tmp_path / train_mod._CONFIG_HASH_FILENAME
    assert hash_file.read_text() == "myhash123"


def test_config_hash_not_written_when_periodic_save_raises(tmp_path):
    """If the underlying save fails, no hash file should appear either --
    a checkpoint dir that doesn't actually hold weights shouldn't claim an
    identity."""
    trainer = _RaisingSaveTrainer()
    train_mod._install_lora_only_checkpoint_save(trainer, config_hash="myhash123")
    trainer._save(output_dir=str(tmp_path))
    assert not (tmp_path / train_mod._CONFIG_HASH_FILENAME).exists()


# ── _write_config_hash / _check_config_identity (unit level) ───────────────

def test_write_config_hash_creates_file(tmp_path):
    ck = tmp_path / "checkpoint-1"
    ck.mkdir()
    train_mod._write_config_hash(str(ck), "myhash")
    assert (ck / train_mod._CONFIG_HASH_FILENAME).read_text() == "myhash"


def test_write_config_hash_best_effort_on_unwritable_dir():
    # parent doesn't exist -- open() raises inside; must be swallowed, not raised.
    train_mod._write_config_hash("/nonexistent/deeply/nested/dir", "myhash")


def test_check_config_identity_noop_when_no_hash_file(tmp_path):
    ck = tmp_path / "checkpoint-1"
    ck.mkdir()
    # must not raise -- a checkpoint predating this guard resumes as before.
    train_mod._check_config_identity(str(ck), "somehash")


def test_check_config_identity_noop_when_hash_matches(tmp_path):
    ck = tmp_path / "checkpoint-1"
    ck.mkdir()
    (ck / train_mod._CONFIG_HASH_FILENAME).write_text("abc123")
    train_mod._check_config_identity(str(ck), "abc123")  # must not raise


def test_check_config_identity_raises_on_mismatch(tmp_path):
    ck = tmp_path / "checkpoint-1"
    ck.mkdir()
    (ck / train_mod._CONFIG_HASH_FILENAME).write_text("old-hash")
    with pytest.raises(RuntimeError, match="--no-resume"):
        train_mod._check_config_identity(str(ck), "new-hash")


# ── end-to-end wiring through run_qat ───────────────────────────────────────

def _write_ds(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        json.dumps({"messages": [{"role": "user", "content": "a"},
                                 {"role": "assistant", "content": "b"}]}) + "\n"
    )
    return str(ds)


def _base_cfg(tmp_path):
    return {
        "model": "offline-please-fall-back",  # forces the offline tiny model
        "scheme_by_group": {"U": "MXFP4", "Q": "Q6_K"},
        "dataset": _write_ds(tmp_path),
        "out": str(tmp_path / "adapters"),
        "lora_r": 4, "lora_alpha": 8, "epochs": 1, "max_steps": 1,
        "lr": 2e-4, "max_seq_len": 64,
    }


class _RecordingTrainer:
    last_resume_from_checkpoint = "NOT_CALLED"

    def __init__(self, model=None, args=None, train_dataset=None, data_collator=None):
        self.model = model
        self.args = args

    def train(self, resume_from_checkpoint=None):
        type(self).last_resume_from_checkpoint = resume_from_checkpoint
        return None


@pytest.fixture
def patched_trainer(monkeypatch):
    import transformers
    import transformers.trainer as trainer_mod

    _RecordingTrainer.last_resume_from_checkpoint = "NOT_CALLED"
    monkeypatch.setattr(trainer_mod, "Trainer", _RecordingTrainer, raising=False)
    monkeypatch.setattr(transformers, "Trainer", _RecordingTrainer, raising=False)
    return _RecordingTrainer


def test_run_qat_refuses_resume_when_config_hash_mismatches(tmp_path, patched_trainer):
    cfg = _base_cfg(tmp_path)
    trainer_dir = tmp_path / "adapters" / "_trainer"
    ck = trainer_dir / "checkpoint-8"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text("{}")
    (ck / "model.safetensors").write_bytes(b"\x00")
    (ck / train_mod._CONFIG_HASH_FILENAME).write_text("stale-hash-from-a-different-run")

    with pytest.raises(RuntimeError, match="--no-resume"):
        train_mod.run_qat(cfg)
    # never got as far as calling trainer.train()
    assert patched_trainer.last_resume_from_checkpoint == "NOT_CALLED"


# The expert config run_qat folds into the hash for a cfg that names no expert
# knobs -- i.e. run_qat's own defaults. Kept next to the tests that need it so a
# default change fails loudly here rather than silently un-guarding a resume.
_DEFAULT_EXPERT_CONFIG = {
    "wrap_experts": True,
    "expert_lora_r": 4,
    "expert_lora_alpha": 8.0,
    "expert_quant_mode": "live",
}


def _expected_hash(cfg, expert_config=_DEFAULT_EXPERT_CONFIG):
    # float(...) matches run_qat's own coercion of lora_alpha (train.py's
    # _parse_run_cfg) -- _base_cfg sets an int (8), but the hash must be
    # computed against the resolved float (8.0) or this helper silently
    # diverges from what run_qat actually hashes.
    return train_mod._config_hash(
        cfg["model"], cfg["scheme_by_group"],
        cfg["lora_r"], float(cfg["lora_alpha"]),
        {}, expert_config,
    )


def test_run_qat_resumes_when_config_hash_matches(tmp_path, patched_trainer):
    cfg = _base_cfg(tmp_path)
    matching_hash = _expected_hash(cfg)
    trainer_dir = tmp_path / "adapters" / "_trainer"
    ck = trainer_dir / "checkpoint-8"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text("{}")
    (ck / "model.safetensors").write_bytes(b"\x00")
    (ck / train_mod._CONFIG_HASH_FILENAME).write_text(matching_hash)

    train_mod.run_qat(cfg)  # must not raise
    assert patched_trainer.last_resume_from_checkpoint == str(ck)


def test_run_qat_resumes_when_checkpoint_has_no_hash_file(tmp_path, patched_trainer):
    """A checkpoint predating this guard (no hash file at all) resumes exactly
    as before -- the guard must not retroactively break old checkpoints."""
    cfg = _base_cfg(tmp_path)
    trainer_dir = tmp_path / "adapters" / "_trainer"
    ck = trainer_dir / "checkpoint-8"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text("{}")
    (ck / "model.safetensors").write_bytes(b"\x00")

    train_mod.run_qat(cfg)  # must not raise
    assert patched_trainer.last_resume_from_checkpoint == str(ck)


def test_run_qat_meta_records_config_hash(tmp_path, patched_trainer):
    cfg = _base_cfg(tmp_path)
    train_mod.run_qat(cfg)
    meta = json.loads((tmp_path / "adapters" / "qat_meta.json").read_text())
    assert meta["config_hash"] == _expected_hash(cfg)


# ── the hash covers the fused-expert config too ───────────────────────────────

def test_config_hash_is_unchanged_without_expert_or_tensor_config():
    """An old Linear-only checkpoint must stay resumable: with neither a
    per-tensor map nor an expert config, the hash is byte-for-byte what the
    four-argument form (model, scheme_by_group, lora_r, lora_alpha) produces
    when scheme_by_tensor/expert_config are omitted vs. explicit {}/None."""
    schemes = {"U": "MXFP4", "Q": "Q6_K"}
    assert train_mod._config_hash("m", schemes, 4, 8.0) == train_mod._config_hash(
        "m", schemes, 4, 8.0, {}, None
    )


@pytest.mark.parametrize("changed", [
    {"expert_lora_r": 8},
    {"expert_quant_mode": "frozen"},
    {"wrap_experts": False},
    {"expert_lora_alpha": 16.0},
])
def test_config_hash_changes_when_the_expert_config_changes(changed):
    """Adapters trained at a different expert rank/mode have different shapes
    and different semantics; resuming across that change must be refused, which
    only works if the hash moves."""
    schemes = {"U": "MXFP4"}
    baseline = train_mod._config_hash("m", schemes, 4, 8.0, {}, _DEFAULT_EXPERT_CONFIG)
    other = dict(_DEFAULT_EXPERT_CONFIG, **changed)
    assert train_mod._config_hash("m", schemes, 4, 8.0, {}, other) != baseline


def test_config_hash_changes_when_the_per_tensor_map_changes():
    schemes = {"X": "Q3_K"}
    a = train_mod._config_hash(
        "m", schemes, 4, 8.0, {"blk.0.ffn_gate_exps.weight": "Q2_K"}
    )
    b = train_mod._config_hash(
        "m", schemes, 4, 8.0, {"blk.0.ffn_gate_exps.weight": "Q3_K"}
    )
    assert a != b


# ── the hash covers base lora_r/lora_alpha too (E3) ─────────────────────────

@pytest.mark.parametrize("changed_kwargs", [
    {"lora_r": 8},
    {"lora_alpha": 16.0},
])
def test_config_hash_changes_when_base_lora_r_or_alpha_changes(changed_kwargs):
    """Resuming with a different base --lora-r/--lora-alpha than the
    checkpoint was trained with must be refused. lora_r changes the adapter's
    tensor shape (caught, eventually, by a torch error inside the trainer's
    checkpoint load) but lora_alpha alone changes only the LoRA scale
    (lora_alpha / lora_r) with no shape change at all -- nothing but this
    hash would ever catch that divergence."""
    schemes = {"U": "MXFP4"}
    baseline_kwargs = {"lora_r": 4, "lora_alpha": 8.0}
    baseline = train_mod._config_hash("m", schemes, **baseline_kwargs)
    other = dict(baseline_kwargs, **changed_kwargs)
    assert train_mod._config_hash("m", schemes, **other) != baseline
