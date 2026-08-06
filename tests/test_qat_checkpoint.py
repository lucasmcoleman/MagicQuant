"""Checkpoint/resume helpers in magicquant.qat.train, isolated from a real
HF Trainer/model: checkpoint-dir discovery + completeness, resume resolution,
and the LoRA-only periodic-save patch.

magicquant.qat.train imports torch unconditionally (see its module docstring:
only config.py/names.py are the torch-free pure submodules), so this file is
gated the same way test_qat_train.py is.
"""

import pytest

torch = pytest.importorskip("torch")

from magicquant.qat import train as train_mod


# ── _list_checkpoint_dirs ────────────────────────────────────────────────────

def test_list_checkpoint_dirs_empty_when_missing(tmp_path):
    assert train_mod._list_checkpoint_dirs(str(tmp_path / "nope")) == []


def test_list_checkpoint_dirs_sorted_newest_first_numerically(tmp_path):
    """Numeric sort, not lexical -- checkpoint-10 must sort ahead of
    checkpoint-2 (a naive string sort would put "2" after "10")."""
    d = tmp_path / "_trainer"
    for n in (2, 10, 1):
        (d / f"checkpoint-{n}").mkdir(parents=True)
    got = train_mod._list_checkpoint_dirs(str(d))
    assert got == [str(d / "checkpoint-10"), str(d / "checkpoint-2"), str(d / "checkpoint-1")]


def test_list_checkpoint_dirs_ignores_non_checkpoint_entries(tmp_path):
    d = tmp_path / "_trainer"
    (d / "checkpoint-3").mkdir(parents=True)
    (d / "runs").mkdir(parents=True)          # unrelated dir (e.g. tb logs)
    (d / "checkpoint-not-a-number").mkdir(parents=True)
    (d / "checkpoint-3-extra").mkdir(parents=True)  # doesn't match ^checkpoint-\d+$
    (d / "checkpoint-5").write_text("not a dir")     # a file, not a dir
    got = train_mod._list_checkpoint_dirs(str(d))
    assert got == [str(d / "checkpoint-3")]


# ── _is_checkpoint_complete ──────────────────────────────────────────────────

def test_checkpoint_incomplete_missing_trainer_state(tmp_path):
    ck = tmp_path / "checkpoint-1"
    ck.mkdir()
    (ck / "model.safetensors").write_bytes(b"\x00")
    assert train_mod._is_checkpoint_complete(str(ck)) is False


def test_checkpoint_incomplete_missing_weights(tmp_path):
    ck = tmp_path / "checkpoint-1"
    ck.mkdir()
    (ck / "trainer_state.json").write_text("{}")
    assert train_mod._is_checkpoint_complete(str(ck)) is False


@pytest.mark.parametrize("weight_file", [
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
])
def test_checkpoint_complete_with_any_recognized_weight_file(tmp_path, weight_file):
    ck = tmp_path / "checkpoint-1"
    ck.mkdir()
    (ck / "trainer_state.json").write_text("{}")
    (ck / weight_file).write_bytes(b"\x00")
    assert train_mod._is_checkpoint_complete(str(ck)) is True


# ── _resolve_resume_checkpoint ───────────────────────────────────────────────

def _make_checkpoint(trainer_dir, step, complete=True):
    ck = trainer_dir / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    if complete:
        (ck / "trainer_state.json").write_text("{}")
        (ck / "model.safetensors").write_bytes(b"\x00")
    else:
        (ck / "trainer_state.json").write_text("{}")  # weights missing
    return ck


def test_resolve_resume_none_when_dir_absent(tmp_path):
    assert train_mod._resolve_resume_checkpoint(str(tmp_path / "nope"), True) is None


def test_resolve_resume_none_when_resume_false_even_with_checkpoint(tmp_path):
    trainer_dir = tmp_path / "_trainer"
    _make_checkpoint(trainer_dir, 5)
    assert train_mod._resolve_resume_checkpoint(str(trainer_dir), False) is None


def test_resolve_resume_returns_newest_complete(tmp_path):
    trainer_dir = tmp_path / "_trainer"
    _make_checkpoint(trainer_dir, 2)
    newest = _make_checkpoint(trainer_dir, 8)
    got = train_mod._resolve_resume_checkpoint(str(trainer_dir), True)
    assert got == str(newest)


def test_resolve_resume_falls_back_past_incomplete_newest(tmp_path):
    trainer_dir = tmp_path / "_trainer"
    complete = _make_checkpoint(trainer_dir, 4, complete=True)
    _make_checkpoint(trainer_dir, 9, complete=False)  # killed mid-write
    got = train_mod._resolve_resume_checkpoint(str(trainer_dir), True)
    assert got == str(complete)


def test_resolve_resume_none_when_every_checkpoint_incomplete(tmp_path):
    trainer_dir = tmp_path / "_trainer"
    _make_checkpoint(trainer_dir, 3, complete=False)
    _make_checkpoint(trainer_dir, 7, complete=False)
    assert train_mod._resolve_resume_checkpoint(str(trainer_dir), True) is None


# ── _install_lora_only_checkpoint_save ──────────────────────────────────────

class _FakeParam:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad


class _FakeModel:
    def __init__(self, params, state):
        self._params = params  # {name: _FakeParam}
        self._state = state    # {name: value}

    def named_parameters(self):
        return list(self._params.items())

    def state_dict(self):
        return dict(self._state)


class _FakeTrainerWithSave:
    def __init__(self, model):
        self.model = model
        self.calls = []  # list of (output_dir, state_dict)

    def _save(self, output_dir=None, state_dict=None):
        self.calls.append((output_dir, state_dict))


def test_install_filters_checkpoint_to_trainable_params_only():
    model = _FakeModel(
        params={
            "base.weight": _FakeParam(False),
            "lora_A": _FakeParam(True),
            "lora_B": _FakeParam(True),
        },
        state={"base.weight": "BASE", "lora_A": "A", "lora_B": "B"},
    )
    trainer = _FakeTrainerWithSave(model)
    train_mod._install_lora_only_checkpoint_save(trainer)

    trainer._save(output_dir="/out")

    assert len(trainer.calls) == 1
    out_dir, state_dict = trainer.calls[0]
    assert out_dir == "/out"
    assert state_dict == {"lora_A": "A", "lora_B": "B"}


def test_install_passes_through_an_explicit_state_dict_unfiltered():
    """If a caller (e.g. an unusual HF code path) supplies its own state_dict,
    the patch must not override it with the filtered one."""
    model = _FakeModel(params={"p": _FakeParam(True)}, state={"p": "X"})
    trainer = _FakeTrainerWithSave(model)
    train_mod._install_lora_only_checkpoint_save(trainer)

    explicit = {"whatever": "value"}
    trainer._save(output_dir="/out", state_dict=explicit)

    out_dir, state_dict = trainer.calls[0]
    assert state_dict is explicit


def test_install_noops_on_trainer_without_save():
    class _NoSaveTrainer:
        pass

    t = _NoSaveTrainer()
    train_mod._install_lora_only_checkpoint_save(t)  # must not raise
    assert not hasattr(t, "_save")
