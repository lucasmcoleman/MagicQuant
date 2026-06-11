# MagicQuant Per-Group Hybrid QAT-LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` checkboxes. The numerical fake-quant kernels are **test-driven**: each test compares the torch fake-quant against the real `libggml` quantizer (`magicquant.quant.ggml_binding.ggml_encode` + dequant), so the test is the correctness contract — implement the kernel until the test passes.

**Goal:** Add Quantization-Aware Training (QAT-LoRA) that fine-tunes a model to be robust to MagicQuant's per-group hybrid quant config, and expose it as a stage in Foundry's pipeline UI.

**Architecture:** MagicQuant gains a `magicquant/qat/` package (differentiable per-scheme fake-quant with STE → `QATLinear` that fake-quants the merged base+adapter → QAT-LoRA loop + CLI), behind an optional `[qat]` extra. Foundry gains a `qat` pipeline stage (`_qat_entry.py` + `QATService` + UI card) following its existing per-stage pattern.

**Tech Stack:** Python 3.14, torch (ROCm), peft, trl, transformers, datasets; libggml (via the existing ctypes binding) as the fake-quant correctness reference. MagicQuant tests: `.venv/bin/python -m pytest tests/ -q`. Foundry tests: `.venv/bin/python -m pytest tests/ -m "not slow and not gpu" -q`.

---

## File structure

**MagicQuant (`magicquant/qat/`):**
- `__init__.py` — package exports (`fake_quant`, `QATLinear`, `wrap_model`, `run_qat`).
- `fake_quant.py` — per-scheme quant→dequant kernels + `FakeQuantSTE` autograd Function + `fake_quant(w, ggml_type_name)` dispatcher + `SCHEME_FAKE_QUANT` registry.
- `names.py` — `hf_to_ggml_name(hf_module_name)` reusing `magicquant/gguf/source.py` `_HF_TO_GGUF_PATTERNS`.
- `wrap.py` — `QATLinear(nn.Module)` + `wrap_model(model, scheme_by_group, classifier)`.
- `config.py` — `load_hybrid_config(search_results_path, tier) -> dict[group,str]` (per-group ggml_type_name map).
- `train.py` — `run_qat(cfg: dict) -> str` (returns adapter dir) + completion-only collator.
- `pyproject.toml` — add `[project.optional-dependencies] qat`.
- `magicquant/__main__.py` — add `qat` subcommand.

**MagicQuant tests:** `tests/test_fake_quant.py`, `tests/test_qat_names.py`, `tests/test_qat_wrap.py`, `tests/test_qat_config.py`, `tests/test_qat_smoke.py` (slow).

**Foundry:** `core/_qat_entry.py`, `core/services.py` (+`QATService`), `core/pipeline.py` (wire stage), `ui/` (stage card + Pydantic config field), `tests/test_qat_service.py`.

---

## Task 0: Environment — install the `[qat]` extra

**Files:** Modify `pyproject.toml`

- [ ] **Step 1:** Add the extra to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
qat = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "trl>=0.8.0",
    "datasets>=2.18.0",
]
```

- [ ] **Step 2:** Install into the venv (torch must be the ROCm build that already works on this box — match Foundry's: `.venv/bin/python -c "import torch;print(torch.__version__)"` in Foundry to get the version/index, then install the same into MagicQuant's `.venv`). Verify: `.venv/bin/python -c "import torch, peft, trl, transformers, datasets; print('qat deps ok', torch.version.hip or torch.version.cuda)"`.
- [ ] **Step 3:** Confirm the existing suite is still green: `.venv/bin/python -m pytest tests/ -q` → 108 passed.
- [ ] **Step 4:** Commit: `git add pyproject.toml && git commit -m "deps(qat): add optional [qat] extra (torch/peft/trl/transformers/datasets)"`

---

## Task 1: Fake-quant — STE harness + BF16/Q8_0 (the simple anchors)

**Files:** Create `magicquant/qat/__init__.py`, `magicquant/qat/fake_quant.py`; Test `tests/test_fake_quant.py`

- [ ] **Step 1: failing test** — uses libggml as the reference. The contract: `fake_quant`'s dequantized output is close to what shipping through ggml actually produces.

```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from magicquant.qat.fake_quant import fake_quant, FakeQuantSTE
from magicquant.quant.ggml_binding import ggml_encode
from magicquant.quant import converters  # for a dequant reference if available

def _ggml_roundtrip(w_np, ggml_type):
    """Quantize with libggml then dequantize back to f32 — the ship reference."""
    # ggml_encode -> bytes; dequant via numpy reference in converters or gguf.
    # (Implementer: use the same dequant the writer/source uses.)
    from magicquant.qat._ggml_ref import ggml_quant_dequant  # tiny helper added in this task
    return ggml_quant_dequant(w_np.astype(np.float32), ggml_type)

@pytest.mark.parametrize("ggml_type", ["BF16", "Q8_0"])
def test_fake_quant_matches_libggml(ggml_type):
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    fq = fake_quant(w, ggml_type).detach().cpu().numpy()
    ref = _ggml_roundtrip(w.cpu().numpy(), ggml_type)
    # close to the real ggml round-trip (faithful, not byte-exact)
    rel = np.abs(fq - ref).mean() / (np.abs(ref).mean() + 1e-8)
    assert rel < 0.05, f"{ggml_type} fake-quant deviates {rel:.3f} from libggml"

def test_fake_quant_idempotent():
    w = torch.randn(128, 128)
    once = fake_quant(w, "Q8_0")
    twice = fake_quant(once, "Q8_0")
    assert torch.allclose(once, twice, atol=1e-4)

def test_ste_gradient_passes_through():
    w = torch.randn(64, 64, requires_grad=True)
    out = fake_quant(w, "Q8_0")
    out.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert torch.allclose(w.grad, torch.ones_like(w.grad), atol=1e-5)  # STE = identity

def test_bf16_passthrough_is_near_identity():
    w = torch.randn(32, 32)
    assert torch.allclose(fake_quant(w, "BF16"), w.bfloat16().float(), atol=1e-2)
```

- [ ] **Step 2: run, expect fail** (`ModuleNotFoundError`).

- [ ] **Step 3: implement** `magicquant/qat/fake_quant.py`:
  - `class FakeQuantSTE(torch.autograd.Function)`: `forward(ctx, w, fn)` returns `fn(w)`; `backward(ctx, g)` returns `(g, None)` (straight-through identity).
  - Kernels (block-structured, vectorized torch): `_fq_bf16(w)` = `w.bfloat16().float()`; `_fq_q8_0(w)` = per-32-element-block: `scale = block.abs().max()/127; q = round(block/scale).clamp(-127,127); deq = q*scale` (matches Q8_0: int8 × fp16 scale, no min). Reshape to (..., n_blocks, 32) on the last dim, restore shape.
  - `SCHEME_FAKE_QUANT = {"BF16": _fq_bf16, "F16": _fq_f16, "F32": lambda w: w, "Q8_0": _fq_q8_0}` (more added in Task 2-3).
  - `def fake_quant(w, ggml_type_name): fn = SCHEME_FAKE_QUANT.get(ggml_type_name); if fn is None: warn+return _fq_bf16(w); return FakeQuantSTE.apply(w, fn)`.
  - Add `magicquant/qat/_ggml_ref.py` with `ggml_quant_dequant(w_np, ggml_type)` = `ggml_encode` then dequant (reuse the dequant the GGUFReader/source uses; for BF16/Q8_0 it's straightforward) — this is the test reference only.
  - `__init__.py` exports `fake_quant`, `FakeQuantSTE`.

- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** — `git commit -m "feat(qat): STE fake-quant harness + BF16/Q8_0 kernels validated vs libggml"`

---

## Task 2: Fake-quant — MXFP4 + Q4_K

**Files:** Modify `magicquant/qat/fake_quant.py`; extend `tests/test_fake_quant.py`

- [ ] **Step 1: failing test** — add `"MXFP4"` and `"Q4_K"` to the `test_fake_quant_matches_libggml` parametrize list (same `rel < 0.08` tolerance — wider for 4-bit), plus an MXFP4-specific test that values land on the E2M1 grid scaled by the block's E8M0 exponent.
- [ ] **Step 2: run, expect fail** (`MXFP4`/`Q4_K` not in registry → falls back to BF16 → deviates → assertion fails).
- [ ] **Step 3: implement** the kernels:
  - `_fq_mxfp4(w)`: reshape to 32-elem blocks; `block_scale_exp = floor(log2(block.abs().max())) ` (E8M0); `scaled = block / 2**block_scale_exp`; round `scaled` to the nearest E2M1 value (the 16-entry signed grid `{0,±0.5,±1,±1.5,±2,±3,±4,±6}`); `deq = grid_val * 2**block_scale_exp`. Mirror `converters` MXFP4 math (doubled kvalues table) for the grid.
  - `_fq_q4_k(w)`: K-quant super-block (256 elems = 8 sub-blocks of 32): per-sub-block `scale = (max-min)/15`, `q = round((x-min)/scale).clamp(0,15)`, `deq = q*scale + min`. (6-bit scale/min quantization of the K-quant format may be approximated by fp scales for the fake-quant — the test tolerance allows it.)
  - Register both in `SCHEME_FAKE_QUANT`.
- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** — `feat(qat): MXFP4 + Q4_K fake-quant kernels`

---

## Task 3: Fake-quant — Q6_K + Q5_K

**Files:** Modify `magicquant/qat/fake_quant.py`; extend `tests/test_fake_quant.py`

- [ ] **Step 1: failing test** — add `"Q6_K"`, `"Q5_K"` to the parametrized libggml-match test (`rel < 0.06`).
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `_fq_q6_k` (6-bit, super-block scale) and `_fq_q5_k` (5-bit + min), same super-block pattern as Q4_K with the bit width changed; register.
- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** — `feat(qat): Q6_K + Q5_K fake-quant kernels (v1 scheme set complete)`

---

## Task 4: HF→GGUF name mapping

**Files:** Create `magicquant/qat/names.py`; Test `tests/test_qat_names.py`

- [ ] **Step 1: failing test:**

```python
from magicquant.qat.names import hf_to_ggml_name
def test_maps_attention_and_ffn():
    assert hf_to_ggml_name("model.layers.0.self_attn.q_proj") == "blk.0.attn_q.weight"
    assert hf_to_ggml_name("model.layers.3.mlp.up_proj") == "blk.3.ffn_up.weight"
    assert hf_to_ggml_name("model.layers.3.mlp.down_proj") == "blk.3.ffn_down.weight"
    assert hf_to_ggml_name("lm_head") == "output.weight"
    assert hf_to_ggml_name("model.embed_tokens") == "token_embd.weight"
def test_unknown_returns_none():
    assert hf_to_ggml_name("model.some.unknown") is None
```

- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `hf_to_ggml_name` reusing `magicquant/gguf/source.py` `_HF_TO_GGUF_PATTERNS` (import the regex list; apply, append `.weight`). If patterns aren't importable cleanly, copy the minimal mapping for attn_{q,k,v,output}, ffn_{up,gate,down}, token_embd, output, norms.
- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** — `feat(qat): HF module -> GGUF tensor name mapping`

---

## Task 5: QATLinear + wrap_model

**Files:** Create `magicquant/qat/wrap.py`; Test `tests/test_qat_wrap.py`

- [ ] **Step 1: failing test:**

```python
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from magicquant.qat.wrap import QATLinear, wrap_model
from magicquant.gguf.tensor_groups import TensorGroupClassifier

def test_qatlinear_forward_shape_and_trainables():
    base = nn.Linear(16, 8, bias=False)
    q = QATLinear.from_linear(base, ggml_type_name="Q8_0", lora_r=4, lora_alpha=8)
    x = torch.randn(2, 16)
    assert q(x).shape == (2, 8)
    trainable = [n for n, p in q.named_parameters() if p.requires_grad]
    assert set(trainable) == {"lora_A", "lora_B"}        # base frozen
    assert q.base_weight.requires_grad is False

def test_qatlinear_fakequants_merged_weight():
    base = nn.Linear(32, 32, bias=False)
    q = QATLinear.from_linear(base, "Q8_0", lora_r=4, lora_alpha=8)
    # with zero LoRA, output == fake_quant(base) @ x
    x = torch.randn(1, 32)
    from magicquant.qat.fake_quant import fake_quant
    expected = x @ fake_quant(base.weight, "Q8_0").T
    assert torch.allclose(q(x), expected, atol=1e-4)

def test_wrap_model_routes_groups():
    # toy model with attn_q-like and ffn_up-like linears named per HF convention
    class Toy(nn.Module):
        def __init__(s):
            super().__init__()
            s.model = nn.Module()
            s.model.layers = nn.ModuleList([nn.Module()])
            s.model.layers[0].self_attn = nn.Module()
            s.model.layers[0].self_attn.q_proj = nn.Linear(8, 8, bias=False)
            s.model.layers[0].mlp = nn.Module()
            s.model.layers[0].mlp.up_proj = nn.Linear(8, 8, bias=False)
    m = Toy()
    scheme_by_group = {"Q": "Q6_K", "U": "MXFP4"}
    wrap_model(m, scheme_by_group, TensorGroupClassifier())
    assert isinstance(m.model.layers[0].self_attn.q_proj, QATLinear)
    assert m.model.layers[0].self_attn.q_proj.ggml_type_name == "Q6_K"
    assert m.model.layers[0].mlp.up_proj.ggml_type_name == "MXFP4"
```

- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `QATLinear`:
  - `from_linear(linear, ggml_type_name, lora_r, lora_alpha)` classmethod: store frozen `base_weight` (register_buffer or `nn.Parameter(requires_grad=False)`), optional bias, `lora_A` (r×in), `lora_B` (out×r) as trainable params (`lora_B` zero-init so initial output == fake_quant(base)), `scaling = lora_alpha/lora_r`.
  - `forward(x)`: `W = base_weight + scaling * (lora_B @ lora_A); Wfq = fake_quant(W, ggml_type_name); return F.linear(x, Wfq, bias)`.
  - `wrap_model(model, scheme_by_group, classifier)`: walk `model.named_modules()`; for each `nn.Linear`, `ggml = hf_to_ggml_name(name)`; if None skip; `group = classifier.classify_tensor(ggml)`; `scheme = scheme_by_group.get(group)`; if scheme and scheme != "BF16": replace the module (via parent setattr) with `QATLinear.from_linear(...)`. Return model.
- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** — `feat(qat): QATLinear (fake-quant merged base+LoRA) + wrap_model group routing`

---

## Task 6: hybrid-config loader

**Files:** Create `magicquant/qat/config.py`; Test `tests/test_qat_config.py`

- [ ] **Step 1: failing test** — given a sample `search_results.json` fixture with per-tier per-group scheme maps, `load_hybrid_config(path, tier="Q4")` returns `{group: ggml_type_name}` for that tier.
- [ ] **Step 2-4:** implement `load_hybrid_config` reading the orchestrator's `search_results.json` shape (per-tier `config` = group→scheme name; map MagicQuant scheme name → `ggml_type_name` via the scheme registry). Add a `tests/fixtures/search_results_sample.json`.
- [ ] **Step 5: commit** — `feat(qat): load per-group hybrid config from search_results.json`

---

## Task 7: run_qat training loop + CLI

**Files:** Create `magicquant/qat/train.py`; Modify `magicquant/__main__.py`; Test `tests/test_qat_smoke.py`

- [ ] **Step 1: failing test** (`@pytest.mark.slow`, CPU, tiny):

```python
import json, pytest
torch = pytest.importorskip("torch")
@pytest.mark.slow
def test_qat_one_step_runs(tmp_path):
    from magicquant.qat.train import run_qat
    # uses a tiny HF model id cached locally (e.g. hf-internal-testing/tiny-random-LlamaForCausalLM)
    ds = tmp_path/"d.jsonl"
    ds.write_text(json.dumps({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]})+"\n")
    cfg = {"model":"hf-internal-testing/tiny-random-LlamaForCausalLM",
           "scheme_by_group":{"U":"MXFP4","D":"MXFP4","Q":"Q6_K","K":"Q6_K","O":"Q8_0"},
           "dataset":str(ds), "out":str(tmp_path/"adapters"),
           "lora_r":4,"lora_alpha":8,"epochs":1,"max_steps":1,"lr":2e-4,"max_seq_len":32}
    out = run_qat(cfg)
    import os
    assert os.path.exists(os.path.join(out, "qat_meta.json"))
    assert any(f.endswith(".safetensors") or f=="adapter_model.bin" for f in os.listdir(out))
```

- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `run_qat(cfg)`:
  - Load tokenizer + `AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=bf16)`.
  - `scheme_by_group = cfg["scheme_by_group"]` (or `load_hybrid_config(cfg["config"], cfg["tier"])`).
  - `wrap_model(model, scheme_by_group, TensorGroupClassifier())`; ensure only LoRA params require grad.
  - Build a dataset from the JSONL (chat template) with **completion-only loss** (mask everything up to the last assistant turn — reuse the masking approach; a simple collator that sets label=-100 on non-assistant tokens).
  - Train with HF `Trainer` (or `trl.SFTTrainer`) honoring `epochs`/`max_steps`/`lr`/`max_seq_len`.
  - Save: `lora_A/lora_B` per `QATLinear` to `out/` as a state dict (`adapter_model.safetensors`) + `qat_meta.json` (model, scheme_by_group, hash, hyperparams). Return `out`.
  - CLI in `__main__.py`: `qat` subparser (`source_model`, `--config`, `--tier`, `--dataset`, `--out`, `--lora-r`, `--lora-alpha`, `--epochs`, `--max-steps`, `--lr`, `--max-seq-len`) → builds cfg → `run_qat`. Route through `MagicQuantSettings` for path/env consistency like the other commands.
- [ ] **Step 4: run, expect pass** (`pytest tests/test_qat_smoke.py -m slow -q`). Also run the full non-slow suite green.
- [ ] **Step 5: commit** — `feat(qat): run_qat QAT-LoRA loop (completion-only) + magicquant qat CLI`

---

## Task 8: validation hook (PPL with/without QAT)

**Files:** Create `magicquant/qat/validate.py`; Test `tests/test_qat_validate.py`

- [ ] **Step 1: failing test** — `compare_perplexity(plain_gguf, qat_gguf, corpus, perplexity_bin)` parses two `llama-perplexity` outputs and returns `{"plain": x, "qat": y, "delta": x-y}`; test with a stub `perplexity_bin` (a fake script echoing a known PPL) so it's offline.
- [ ] **Step 2-4:** implement the parser/compare (reuse `tools/calibrate_noise_factors.py`'s perplexity-parsing helper — factor it into a shared util if clean).
- [ ] **Step 5: commit** — `feat(qat): perplexity comparison hook for QAT vs plain hybrid`

---

## Task 9: Foundry — QAT stage (service + entry + pipeline)

**Files:** Create `Foundry/core/_qat_entry.py`; Modify `Foundry/core/services.py`, `Foundry/core/pipeline.py`; Test `Foundry/tests/test_qat_service.py`

- [ ] **Step 1: failing test** (Foundry repo) — mirror `test_script_equivalence.py`: `QATService(ROOT, "python").build_script(model=..., scheme_by_group=..., dataset=..., lora_r=..., ...)` returns a script that **compiles** and contains the expected `magicquant qat`/`_qat_entry` invocation + repr-escaped strings.
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** following the existing stage pattern:
  - `core/_qat_entry.py`: `run(cfg)` imports `magicquant.qat.run_qat` and calls it (subprocess context like `_magicquant_entry.py`).
  - `core/services.py`: `class QATService` with `build_config()` (JSON for the entry) + `build_script()` (`_entry_shim("_qat_entry", cfg, self.pipeline_root)`).
  - `core/pipeline.py`: add a `qat` stage between search and export; gate with `_stage_complete.json` marker; `--qat`/`--no-qat` flag (default off for back-compat).
- [ ] **Step 4: run, expect pass** (Foundry suite green: `pytest tests/ -m "not slow and not gpu" -q`).
- [ ] **Step 5: commit** (Foundry) — `feat(qat): Foundry QAT pipeline stage (service + entry + wiring)`

---

## Task 10: Foundry — UI stage card

**Files:** Modify `Foundry/ui/index.html`, the FastAPI app + `UIConfig` model (in `ui/`); Test `Foundry/tests/test_qat_service.py` (extend for the config model)

- [ ] **Step 1: failing test** — the UI `UIConfig` Pydantic model accepts the QAT fields (`qat_enabled`, `qat_dataset`, `qat_tier`, `qat_lora_r`, `qat_lora_alpha`, `qat_epochs`, `qat_lr`) and rejects unknown ones (`extra='forbid'`).
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement:** add the QAT fields to the UIConfig model; add a **QAT stage card** to `ui/index.html` (toggle + the config inputs) matching the existing stage cards; wire its run into the pipeline call + the existing WebSocket log stream + completion badge.
- [ ] **Step 4: run, expect pass;** load `index.html` in the FastAPI test client to confirm it serves (200) and contains the QAT card.
- [ ] **Step 5: commit** (Foundry) — `feat(qat): Foundry UI QAT stage card + config`

---

## Task 11: docs

**Files:** Modify `MagicQuant/CLAUDE.md`, `MagicQuant/README.md`, `Foundry/CLAUDE.md`, `Foundry/README.md`

- [ ] Document the `qat` command + `[qat]` extra (MagicQuant) and the QAT pipeline stage + UI (Foundry). Commit in each repo.

---

## Self-review notes

- **Spec coverage:** fake-quant ops T1-T3 (incl. validation-vs-libggml); STE T1; QATLinear merged-weight + wrap T5; name mapping T4; hybrid-config T6; run_qat + completion-only + CLI T7; validation/PPL T8; Foundry stage T9; UI T10; deps T0; docs T11. All spec sections mapped.
- **v1 scheme set** = BF16/Q8_0/Q6_K/Q5_K/Q4_K/MXFP4 (T1-T3); unmapped schemes fall back to BF16 passthrough with a warning (in `fake_quant`).
- **Type consistency:** `fake_quant(w, ggml_type_name)`, `QATLinear.from_linear(linear, ggml_type_name, lora_r, lora_alpha)`, `wrap_model(model, scheme_by_group, classifier)`, `run_qat(cfg)->out_dir`, `hf_to_ggml_name(name)->str|None` — names consistent across tasks.
- **Cross-repo:** MagicQuant tasks T0-T8,T11 on branch `feature/qat`; Foundry tasks T9-T10,T11 on a Foundry branch `feature/qat` — two repos, coordinate but each suite stays green independently.
- **Numerical kernels are test-driven against libggml** — the `rel < tolerance` comparisons are the contract; exact byte-match is NOT required (STE makes gradient exactness moot).
