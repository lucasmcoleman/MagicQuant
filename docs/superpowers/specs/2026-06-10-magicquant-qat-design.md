# MagicQuant — Per-Group Hybrid QAT (QAT-LoRA)

**Date:** 2026-06-10
**Status:** Approved (design)
**Spans:** MagicQuant (core feature) + Foundry (pipeline stage + Web UI)

## Problem

MagicQuant is purely post-training: it reads finished BF16 weights and packs a
per-group hybrid GGUF via libggml. At aggressive tiers (Q2/Q3/MXFP4) the model
loses accuracy it could recover if it had been *trained aware* of the exact
hybrid quantization it will ship as. We add **Quantization-Aware Training** that
fine-tunes a model to be robust to MagicQuant's chosen per-group hybrid config,
and expose it as a stage in Foundry's pipeline UI.

## Goals

- A **QAT-LoRA** path: freeze the base, fake-quantize it to the **per-group hybrid
  config** in the forward, train LoRA adapters that compensate, merge, then pack
  the exact hybrid GGUF.
- **Differentiable per-group fake-quant** mirroring MagicQuant's schemes with a
  straight-through estimator (STE), validated against the real libggml quantizer.
- Drive it from the existing search output (`search_results.json` per-group config).
- Run it from **Foundry's Web UI** as a first-class stage (toggle + config + live
  WebSocket logs + completion marker), following the existing stage pattern.
- Measure success: perplexity of the QAT'd hybrid vs the non-QAT hybrid.

## Non-Goals (follow-ons)

- Full-QAT (update all weights) — small-model option, later.
- Foundry fast-shard-loader integration for 40B-scale QAT — v1 uses standard HF
  loading (fits 9B comfortably on the 122 GB APU; big-model loading noted below).
- Multi-GPU / distributed training.
- A standalone MagicQuant UI (MagicQuant stays headless; the UI is Foundry's).

## Architecture

### Part A — `magicquant/qat/` (the core, in MagicQuant)

Heavy training deps go in an optional `[qat]` extra (`torch`, `transformers`,
`peft`, `trl`, `datasets`); the core install is unchanged.

**`fake_quant.py` — differentiable per-scheme fake-quant.**
- One function per ggml scheme that quantize→dequantizes a weight tensor in the
  scheme's block structure, all in torch (GPU). **v1 scheme set** (covers MagicQuant's
  default Q4/Q5/Q6 hybrid tiers): `BF16/F16/F32` (round to the float type), `Q8_0`
  (per-32 block fp16 scale × int8), `Q6_K/Q5_K/Q4_K` (K-quant super-block: per-sub-block
  scale + round to 6/5/4 bits), and `MXFP4` (32-elem blocks, E8M0 block scale + E2M1
  round). **Stretch** (the aggressive Q2/Q3 tiers, added once v1 is proven):
  `IQ4_NL` (nearest entry in the non-linear LUT), `Q3_K`, `Q2_K`. Any group whose
  scheme has no fake-quant op yet falls back to BF16 passthrough with a logged warning
  (so a hybrid is always trainable, just not quant-aware for those groups).
- `FakeQuantSTE(autograd.Function)`: forward = the scheme's quant→dequant;
  backward = straight-through (gradient identity, clamped to the representable
  range so out-of-range weights get a zero gradient).
- `fake_quant(w, ggml_type_name)` dispatches by the scheme's `ggml_type_name`.
- **Fidelity:** these are torch-native *faithful approximations* (GPU-fast), not
  byte-exact libggml. A test asserts each op's dequantized output is within a
  tolerance of `libggml`'s actual `ggml_encode`→dequant for random weights, so the
  training target tracks the ship target. The final pack still uses exact libggml;
  the residual train-vs-ship gap is small and measured (see Validation).

**`wrap.py` — per-group application.**
- `QATLinear(nn.Module)`: holds the frozen base weight `W` (FP), trainable LoRA
  `A,B` (+ scaling), and the group's `ggml_type_name`. Forward:
  `W_eff = W + scaling·(B @ A); W_fq = fake_quant(W_eff, type); return x @ W_fqᵀ (+bias)`.
  Critically the fake-quant is applied to the **merged** base+adapter each step, so
  training sees exactly the weight that will ship.
- `wrap_model(model, hybrid_config, classifier)`: walk `nn.Linear` modules, map each
  HF module name → ggml tensor name (reuse MagicQuant's `source.py`
  `_HF_TO_GGUF_PATTERNS`), classify → tensor group → scheme from `hybrid_config`,
  replace the Linear with a `QATLinear`. BF16 groups (passthrough) may be skipped to
  save compute. Only LoRA `A,B` are trainable; base weights stay frozen.

**`train.py` — the QAT-LoRA loop.**
- `run_qat(cfg)`: load the HF model (FP base), `wrap_model(...)`, train on a chat
  dataset with completion-only loss (system/user turns masked, mirroring Foundry),
  using a standard HF `Trainer`/loop. Saves LoRA adapters + a `qat_meta.json`
  (base model, hybrid config hash, hyperparams). Big-model note: FP base for a 9B is
  ~18 GB (fine); 40B FP base (~80 GB) fits the 122 GB APU only when services are
  idle — the documented path for 40B is the Foundry-fast-loader follow-on.

**CLI:** `magicquant qat <model> --config search_results.json --dataset data.jsonl
--out adapters/ [--lora-r 32 --lora-alpha 64 --epochs 1 --lr 2e-4]`.

### Part B — Foundry pipeline stage + Web UI

Follows Foundry's established per-stage pattern (`services.py` builder + thin
`_*_entry.py` shim run as a subprocess + completion marker + UI toggle/config +
WebSocket logs). Foundry already has `torch/peft/trl`.

- **`core/_qat_entry.py`** — importable stage body `run(cfg_json)` that imports
  `magicquant.qat.run_qat` and runs it (subprocess context, like `_magicquant_entry.py`).
- **`core/services.py`** — `QATService` with `build_config()` (JSON consumed by the
  entry) and `build_script()` (the shim). One source of truth for CLI + UI.
- **`core/pipeline.py`** — wire a `qat` stage; flow:
  `search (hybrid config) → QAT (adapters) → export (merge) → generate (pack) → validate`.
  Skip/resume via the existing `_stage_complete.json` markers.
- **`ui/`** — add **QAT** to the stage list: a toggle + a config panel (base model,
  hybrid-config source = a prior `search_results.json`, dataset path, LoRA r/alpha,
  epochs, lr), live log streaming over the existing WebSocket, and a completion badge.
  Mirrors the existing stage cards; `ui` UIConfig model gains the QAT fields
  (Pydantic, `extra='forbid'`).

## Data flow

```
MagicQuant search ──► search_results.json (per-group hybrid)
                          │
Foundry UI "QAT" stage ───┤ base model + dataset + LoRA cfg
                          ▼
       magicquant.qat.run_qat  ──► LoRA adapters (quant-aware to the hybrid)
                          ▼
       Foundry export (streaming merge) ──► merged BF16 model
                          ▼
       magicquant generate (exact libggml) ──► hybrid GGUF
                          ▼
       validate: perplexity(QAT hybrid) vs perplexity(plain hybrid)
```

## Validation / success metric

Reuse the perplexity/calibration infra: build the hybrid GGUF both with and without
QAT and compare `llama-perplexity`. Success = lower PPL loss for the QAT'd hybrid,
expected to be largest at Q2/Q3/MXFP4. Report the delta; a tiny end-to-end run on a
small model is the acceptance smoke.

## Testing (offline, mostly tiny)

- **`tests/test_fake_quant.py`** — each scheme's fake-quant: (a) dequant output within
  tolerance of `libggml` `ggml_encode`→dequant on random weights; (b) idempotent
  (`fq(fq(w)) == fq(w)`); (c) STE gradient passes through (`grad` finite, ≈ upstream);
  (d) BF16 passthrough is ~identity.
- **`tests/test_qat_wrap.py`** — `wrap_model` routes each Linear to the right group's
  scheme; `QATLinear` forward shapes; only `A,B` are trainable; HF→ggml name mapping.
- **`tests/test_qat_smoke.py`** (CPU, marked slow) — one training step on a toy model
  runs and loss is finite; adapters save/reload.
- **Foundry:** `tests/test_qat_service.py` — `QATService.build_script` compiles and
  contains the expected config; UI config model accepts/rejects fields. Existing
  MagicQuant 108 + Foundry 145 suites stay green.

## Dependencies

- MagicQuant: new optional extra `[qat] = torch, transformers, peft, trl, datasets`.
  Core (`pip install magicquant`) unchanged — no torch unless `[qat]`.
- Foundry: already has the training stack; no new deps.

## Phasing (becomes the implementation plan)

1. **Fake-quant ops** (`fake_quant.py`) + tests vs libggml — the core, highest risk.
2. **`QATLinear` + `wrap_model`** + tests (name mapping, group routing).
3. **`train.py` + CLI** + CPU smoke; adapter save/load.
4. **Validation** hook (PPL with/without QAT).
5. **Foundry stage** (`_qat_entry.py`, `QATService`, pipeline wiring) + tests.
6. **Foundry UI** (stage card, config panel, WebSocket logs, marker) + docs.

## Open questions

None — QAT-LoRA against the per-group hybrid config; MagicQuant-native core with an
optional `[qat]` extra; torch-native faithful fake-quant validated against libggml
with STE; fake-quant applied to the merged base+adapter each step; surfaced in
Foundry's UI as a pipeline stage. 40B-scale fast-loader QAT and full-QAT are
explicit follow-ons.
