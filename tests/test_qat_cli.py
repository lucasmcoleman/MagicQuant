"""CLI wiring for `magicquant qat`'s checkpoint/resume flags.

Two layers, matching how the flags actually flow:
  1. argparse defaults/parsing (`main()` dispatch) -- do `--save-steps`,
     `--resume`, `--no-resume` parse to the right `args` values.
  2. `cmd_qat` -> cfg dict -- does `args.save_steps`/`args.resume` actually
     land in the cfg passed to `run_qat`.

Neither layer imports torch/transformers (cmd_qat's heavy imports are local,
`from magicquant.qat.train import run_qat`, patched away below), so this file
runs in the core (torch-less) test environment.
"""

import argparse

import pytest

import magicquant.__main__ as main_mod


# ── layer 1: argparse defaults/parsing ──────────────────────────────────────

def _parse_qat_args(monkeypatch, extra_argv):
    """Run main() with a captured cmd_qat, return the argparse.Namespace it saw."""
    captured = {}

    def _fake_cmd_qat(args: argparse.Namespace) -> None:
        captured["args"] = args

    monkeypatch.setattr(main_mod, "cmd_qat", _fake_cmd_qat)
    argv = [
        "magicquant", "qat", "some/model",
        "--config", "sr.json", "--tier", "Q4", "--dataset", "data.jsonl",
    ] + extra_argv
    monkeypatch.setattr("sys.argv", argv)
    main_mod.main()
    return captured["args"]


def test_save_steps_defaults_to_100(monkeypatch):
    args = _parse_qat_args(monkeypatch, [])
    assert args.save_steps == 100


def test_save_steps_overridable(monkeypatch):
    args = _parse_qat_args(monkeypatch, ["--save-steps", "5"])
    assert args.save_steps == 5


def test_resume_defaults_to_true(monkeypatch):
    args = _parse_qat_args(monkeypatch, [])
    assert args.resume is True


def test_no_resume_flag_sets_resume_false(monkeypatch):
    args = _parse_qat_args(monkeypatch, ["--no-resume"])
    assert args.resume is False


def test_explicit_resume_flag_is_still_true(monkeypatch):
    """--resume is accepted explicitly (e.g. by a script that always passes it)
    and is a no-op against the True default."""
    args = _parse_qat_args(monkeypatch, ["--resume"])
    assert args.resume is True


# ── layer 2: cmd_qat -> cfg dict ────────────────────────────────────────────

def _run_cmd_qat_capturing_cfg(monkeypatch, tmp_path, **arg_overrides):
    # cmd_qat's `from magicquant.qat.train import run_qat` is a local import,
    # and magicquant.qat.train unconditionally imports torch (see its module
    # docstring) -- so this layer needs torch even though layer 1 (argparse)
    # doesn't. Skipped, not failed, in the torch-less core test environment.
    pytest.importorskip("torch")
    captured = {}

    def _fake_run_qat(cfg):
        captured["cfg"] = cfg
        return cfg["out"]

    monkeypatch.setattr("magicquant.qat.train.run_qat", _fake_run_qat)

    defaults = dict(
        source_model="some/model",
        config="sr.json",
        tier="Q4",
        dataset="data.jsonl",
        out=str(tmp_path / "adapters"),
        lora_r=32,
        lora_alpha=64.0,
        epochs=1.0,
        max_steps=-1,
        lr=2e-4,
        max_seq_len=512,
        save_steps=100,
        resume=True,
        expert_lora_r=4,
        expert_lora_alpha=None,
        expert_quant_mode="live",
        wrap_experts=True,
        gradient_checkpointing=False,
    )
    defaults.update(arg_overrides)
    args = argparse.Namespace(**defaults)
    main_mod.cmd_qat(args)
    return captured["cfg"]


def test_cmd_qat_passes_save_steps_into_cfg(monkeypatch, tmp_path):
    cfg = _run_cmd_qat_capturing_cfg(monkeypatch, tmp_path, save_steps=7)
    assert cfg["save_steps"] == 7


def test_cmd_qat_passes_resume_true_into_cfg(monkeypatch, tmp_path):
    cfg = _run_cmd_qat_capturing_cfg(monkeypatch, tmp_path, resume=True)
    assert cfg["resume"] is True


def test_cmd_qat_passes_resume_false_into_cfg(monkeypatch, tmp_path):
    cfg = _run_cmd_qat_capturing_cfg(monkeypatch, tmp_path, resume=False)
    assert cfg["resume"] is False


# ── fused 3-D MoE expert knobs ────────────────────────────────────────────────

def test_expert_qat_defaults(monkeypatch):
    args = _parse_qat_args(monkeypatch, [])
    assert args.expert_lora_r == 4
    assert args.expert_lora_alpha is None  # -> 2 x rank, resolved in cmd_qat
    assert args.expert_quant_mode == "live"
    assert args.wrap_experts is True


def test_expert_qat_flags_parse(monkeypatch):
    args = _parse_qat_args(monkeypatch, [
        "--expert-lora-r", "8", "--expert-lora-alpha", "32",
        "--expert-quant-mode", "frozen", "--no-expert-qat",
    ])
    assert args.expert_lora_r == 8
    assert args.expert_lora_alpha == 32.0
    assert args.expert_quant_mode == "frozen"
    assert args.wrap_experts is False


def test_expert_quant_mode_rejects_unknown_values(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_qat_args(monkeypatch, ["--expert-quant-mode", "sometimes"])


def test_cmd_qat_defaults_expert_alpha_to_twice_the_rank(monkeypatch, tmp_path):
    cfg = _run_cmd_qat_capturing_cfg(
        monkeypatch, tmp_path, expert_lora_r=8, expert_lora_alpha=None
    )
    assert cfg["expert_lora_r"] == 8
    assert cfg["expert_lora_alpha"] == 16.0


def test_gradient_checkpointing_defaults_off_and_is_settable(monkeypatch):
    assert _parse_qat_args(monkeypatch, []).gradient_checkpointing is False
    args = _parse_qat_args(monkeypatch, ["--gradient-checkpointing"])
    assert args.gradient_checkpointing is True


def test_cmd_qat_passes_gradient_checkpointing_into_cfg(monkeypatch, tmp_path):
    """run_qat has always honoured this key; before fused-expert QAT the CLI
    had no way to set it, which on a wrapped MoE is the difference between a
    run and an OOM."""
    cfg = _run_cmd_qat_capturing_cfg(
        monkeypatch, tmp_path, gradient_checkpointing=True
    )
    assert cfg["gradient_checkpointing"] is True


def test_cmd_qat_passes_explicit_expert_knobs_into_cfg(monkeypatch, tmp_path):
    cfg = _run_cmd_qat_capturing_cfg(
        monkeypatch, tmp_path, expert_lora_r=2, expert_lora_alpha=9.0,
        expert_quant_mode="frozen", wrap_experts=False,
    )
    assert cfg["expert_lora_r"] == 2
    assert cfg["expert_lora_alpha"] == 9.0
    assert cfg["expert_quant_mode"] == "frozen"
    assert cfg["wrap_experts"] is False
