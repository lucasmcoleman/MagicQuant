"""End-to-end QAT-LoRA smoke: one training step on a tiny model (CPU, slow).

Marked ``slow``: it builds a tiny LlamaForCausalLM, wraps it with the per-group
fake-quant QATLinears, runs a single training step with completion-only loss, and
saves LoRA adapters + ``qat_meta.json``. Uses the tiny HF model id when it can be
downloaded; ``run_qat`` falls back to constructing a tiny model offline so the
smoke still runs without network.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.slow
def test_qat_one_step_runs(tmp_path):
    from magicquant.qat.train import run_qat

    ds = tmp_path / "d.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "yo"}]},
        {"messages": [{"role": "user", "content": "ping"},
                      {"role": "assistant", "content": "pong"}]},
    ]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    cfg = {
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "scheme_by_group": {"U": "MXFP4", "D": "MXFP4", "Q": "Q6_K",
                            "K": "Q6_K", "O": "Q8_0"},
        "dataset": str(ds),
        "out": str(tmp_path / "adapters"),
        "lora_r": 4,
        "lora_alpha": 8,
        "epochs": 1,
        "max_steps": 1,
        "lr": 2e-4,
        "max_seq_len": 32,
    }
    out = run_qat(cfg)

    assert os.path.exists(os.path.join(out, "qat_meta.json"))
    meta = json.loads(open(os.path.join(out, "qat_meta.json")).read())
    assert meta["scheme_by_group"] == cfg["scheme_by_group"]
    assert "config_hash" in meta
    files = os.listdir(out)
    assert any(
        f.endswith(".safetensors") or f == "adapter_model.bin" for f in files
    ), f"no adapter weights saved: {files}"


@pytest.mark.slow
def test_qat_saved_adapters_reload(tmp_path):
    """The saved adapter state dict reloads and holds the trained LoRA tensors."""
    from magicquant.qat.train import run_qat
    from safetensors.torch import load_file

    ds = tmp_path / "d.jsonl"
    ds.write_text(
        json.dumps({"messages": [{"role": "user", "content": "a"},
                                  {"role": "assistant", "content": "b"}]}) + "\n"
    )
    cfg = {
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "scheme_by_group": {"U": "MXFP4", "Q": "Q6_K"},
        "dataset": str(ds),
        "out": str(tmp_path / "adapters"),
        "lora_r": 4, "lora_alpha": 8, "epochs": 1, "max_steps": 1,
        "lr": 2e-4, "max_seq_len": 16,
    }
    out = run_qat(cfg)
    sd = load_file(os.path.join(out, "adapter_model.safetensors"))
    assert sd, "adapter state dict is empty"
    # every saved tensor is a LoRA adapter param
    assert all("lora_A" in k or "lora_B" in k for k in sd), list(sd)[:4]
    assert any(torch.isfinite(v).all() for v in sd.values())
