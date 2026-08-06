# Quantization-Aware Training (QAT-LoRA)

MagicQuant ships hybrid GGUFs where each tensor group is quantized to a different
scheme (BF16 brain, MXFP4 FFN, K-quant attention). Quantization is lossy, and the
loss is fixed once the model ships — there is no chance to recover it. **QAT-LoRA**
adds that chance: it fine-tunes a small set of LoRA adapters *while the model is
fake-quantized to its target hybrid config*, so the adapters learn to compensate
for the exact quantization error the shipped model will carry.

This is an optional feature behind the `[qat]` extra and is surfaced as Foundry's
**QAT** pipeline stage. It lives in `magicquant/qat/`.

---

## What QAT-LoRA does

The core idea is a **fake-quant with a straight-through estimator (STE)** wrapped
around the weights, plus LoRA adapters that are the only trainable parameters.

### Per-group fake-quant + STE (`qat/fake_quant.py`)

`fake_quant(w, ggml_type_name)` runs a differentiable quantize→dequantize of a
weight tensor in its ggml scheme's block structure, entirely in torch (so it runs
on GPU and is autograd-friendly):

- **Forward** is a faithful *approximation* of what libggml would store and read
  back: the same block sizes (32 / 256), the same scale derivation, the same level
  grids. Schemes covered: `BF16`, `F16`, `F32`, `Q8_0`, `MXFP4`, `Q4_K`, `Q5_K`,
  `Q6_K`, plus the aggressive low-bit tiers `Q3_K`, `Q2_K`, `IQ4_NL`. (`Q3_K`,
  `Q2_K`, and `IQ4_NL` are applied twice so the kernel projects onto a clean fixed
  point, `fq(fq(w)) == fq(w)`.) Any scheme without a kernel falls back to BF16
  passthrough with a logged warning, so a hybrid is always trainable — just not
  quant-aware for the unmapped groups.
- **Backward** is the straight-through estimator: `FakeQuantSTE` passes the
  upstream gradient through the (non-differentiable) rounding unchanged, as if the
  quantization weren't there. The continuous weight therefore receives a real
  gradient even though the forward pass rounded to a grid.

**Fidelity contract:** `tests/test_fake_quant.py` asserts each kernel's dequant
output is within a per-scheme tolerance of the real libggml round-trip
(`ggml_encode` → `dequantize_row_*`). These are GPU-fast approximations, **not**
byte-exact reimplementations — and STE makes exact gradients moot. The byte-exact
quantization happens later, in the real pack (see *Handoff* below).

### `QATLinear` fake-quants the merged base+adapter (`qat/wrap.py`)

`QATLinear` replaces an `nn.Linear`. It:

1. Freezes the original ("base") weight (`requires_grad=False`).
2. Adds trainable LoRA adapters `A` (r×in, Kaiming-init) and `B` (out×r,
   **zero**-init, so the adapter starts as a no-op).
3. On **every forward**, builds the merged weight `W_eff = base + scaling·(B @ A)`
   in fp32 and fake-quantizes *that* to the group's ggml scheme:

   ```
   w_fq = fake_quant(base + scaling·(B @ A), ggml_type_name)
   y    = x @ w_fqᵀ
   ```

The crucial detail: it is the **merged** weight that gets fake-quantized, not the
base alone. Training therefore sees exactly the quantized weight that will ship,
and the STE routes the gradient to the small LoRA adapters — the adapters move the
continuous weight so that, *after* rounding to the quant grid, the output is closer
to the bf16 reference. Keeping the merged weight in fp32 for the block-scale math
lets the frozen base be bf16 (large models) while the LoRA adapters and optimizer
state stay fp32.

`wrap_model(model, scheme_by_group, classifier)` walks the model's `nn.Linear`
modules, maps each module path to its GGUF tensor name
(`qat.names.hf_to_ggml_name`, reusing the writer's `_HF_TO_GGUF_PATTERNS` so the
routing never drifts), classifies it into a tensor group
(`TensorGroupClassifier`), looks up the group's scheme, and swaps in a `QATLinear`
for that scheme. Groups that are BF16 (passthrough) or absent from the scheme map
are left untouched — no quant-awareness is needed or wanted there.

### Fused 3-D MoE experts (`qat/expert_wrap.py`)

`wrap_model`'s `nn.Linear` walk is structurally blind to modern MoE
architectures. Qwen3.5/Qwen3.6, Llama4, GPT-OSS and Granite-hybrid don't build one
Linear per expert — they fuse every routed expert of a projection into **one 3-D
`nn.Parameter`**:

```
model.language_model.layers.N.mlp.experts.gate_up_proj   [E, 2*I, H]
model.language_model.layers.N.mlp.experts.down_proj      [E, H,   I]
```

On Qwen3.6-35B-A3B those parameters are **33.0e9 of the model's 35.9e9 elements**.
Before this landed, QAT wrapped 2.44e9 elements — **6.8%** — and skipped the
experts, which is exactly where low-bit quantization does its damage (that build
puts 14.5e9 expert elements at Q2_K and 18.5e9 at Q3_K). With expert wrapping the
covered fraction is **98.6%**.

**Mechanism.** `torch.nn.utils.parametrize.register_parametrization` on the fused
Parameter. The parametrization's `forward(W)` returns the QAT weight, so *every*
consumer sees it and the MoE forward is never touched. Trainable state is a
per-expert LoRA pair batched over the leading expert axis:

```
lora_expert_A  (E, W.shape[1], r)   zero-init
lora_expert_B  (E, r, W.shape[2])   Kaiming-init
delta = (expert_lora_alpha / expert_lora_r) * bmm(A, B)
```

Which factor is zeroed is inverted relative to `QATLinear` because the merge
contract fixes the operand order as `W[e] += scale * (A[e] @ B[e])`
(`qat/merge.py::_apply_3d`). The delta still starts at exactly zero.

**GGUF splits what HF fuses.** `gate_up_proj` becomes *two* GGUF tensors,
`blk.N.ffn_gate_exps.weight` and `blk.N.ffn_up_exps.weight` — contiguous halves of
the out axis, gate first — and they can carry different schemes. `qat/names.py`
maps a fused parameter to a list of `ExpertSegment`s, and each segment is
fake-quantized against *its own* scheme. The concatenated-not-interleaved,
untransposed `[E, out, in]` layout is re-checked at wrap time against the module's
own `is_concatenated`/`is_transposed` flags (which transformers'
`use_experts_implementation` decorator stamps); a module declaring the other
layout is skipped with a warning rather than fake-quantized against the wrong
slices.

**Per-tensor schemes.** A budget build's `tensor_config` disagrees with its own
group projection — the 13.5 GiB Qwen3.6 build says `X: Q3_K` while 54 of its 123
expert tensors are Q2_K. `load_tensor_config` reads the per-tensor map and
`wrap_model` prefers it over the group scheme, for Linears and experts alike.

**Two quant modes**, because the fake-quant kernels are not free (measured on
gfx1151: `Q2_K` ~2.7 Melem/s, `Q3_K` ~110 Melem/s):

| mode | forward | cost on Qwen3.6-35B-A3B |
|---|---|---|
| `live` (default) | `fake_quant(W + delta)` every step — `QATLinear`'s exact semantics | **~92 min per forward pass** |
| `frozen` | base fake-quantized once at wrap; forward is `W_q + delta` | ~5.5 s per forward pass |

`live` is what you want whenever you can afford it: what trains is what ships,
because the pack quantizes the merged weight too. It is simply not affordable at
33e9 expert elements — `run_qat` prints the estimate at wrap time and warns when
it exceeds a minute per forward, so an infeasible run is visible in its first
seconds rather than at dawn. `frozen` makes a 35B run possible at the cost of a
real caveat: the adapter's delta is never re-quantized during training, while the
shipped weight is `quant(W_q + delta)`, so at Q2_K/Q3_K some of the compensation
can be rounded away at pack time. Frozen-mode recovery has **not** been validated
end to end.

**Why there is a cache.** The eager MoE forward reads the fused parameter once per
*hit expert* (`self.gate_up_proj[expert_idx]` inside the loop) — ~256 times per
layer at a 512-token batch. Torch's own `parametrize.cached()` is unusable here
because it caches every parametrized tensor for the whole context, ~66 GB for this
model. `expert_wrap` keeps a **bounded** cache of 2 entries instead, which is
exactly enough for the loop's `gate_up_proj`/`down_proj` alternation within one
layer, keyed on the tensors' `_version` counters and the grad mode so a stale
value can never be served.

**Adapter file.** Expert adapters are written into the same
`adapter_model.safetensors` as the Linear ones, under
`"<base safetensors key>.lora_expert_A"` / `".lora_expert_B"` — no `.weight`
suffix, because a fused expert stack is a raw Parameter, not an `nn.Linear`
weight. `qat_meta.json` records `expert_lora_r` / `expert_lora_alpha` (which the
merge reads for its 3-D scale), `expert_quant_mode`, and a per-tensor
`expert_adapters` block with every shape and segment scheme.

### Eval and handoff

- `bake_for_eval(model)` precomputes `fake_quant(merged_weight)` once and swaps each
  `QATLinear`'s forward to a plain `F.linear` on that baked weight, so a no-grad
  perplexity eval doesn't pay the per-forward fake-quant. It is also a context
  manager (`with bake_for_eval(m): ...`).
- `merge_qat_adapters(model)` replaces each `QATLinear` with a plain `nn.Linear`
  holding the **un-fake-quantized** merged weight (base + scaled LoRA). This is the
  handoff into MagicQuant's real pack: the exact ggml quantization happens later in
  `magicquant generate` (byte-identical to llama.cpp), so the merged Linear must
  carry the full-precision merged weight, not the training-time fake-quant
  approximation.

---

## The `[qat]` extra

The heavy training stack (torch / transformers / peft / trl / datasets) lives
behind an optional extra so the core MagicQuant install stays torch-free:

```bash
pip install -e ".[qat]"
```

> `torch` must be the build that matches your hardware (the ROCm wheel on AMD,
> CUDA on NVIDIA, or CPU). Install the matching wheel first if the default index
> doesn't provide it.

**Import policy.** `import magicquant.qat` works without the extra: the pure-Python
helpers (`hf_to_ggml_name`, `load_hybrid_config`) import with no extra deps, and the
torch-dependent surface (`fake_quant`, `QATLinear`, `wrap_model`, `run_qat`, …) is
loaded lazily on first attribute access. So a torch-less environment can still
import the package and its pure submodules (config parsing stays unit-testable
without the extra installed).

---

## The CLI

```bash
magicquant qat ./my-model \
    --config ./output/search_results.json \
    --tier Q4 \
    --dataset data/chat.jsonl \
    --out ./output/qat_adapters \
    --lora-r 32 --lora-alpha 64 --epochs 1 --lr 2e-4
```

| Flag | Default | Meaning |
|------|---------|---------|
| `source_model` (positional) | — | HF model id or local path to fine-tune. |
| `--config` | *required* | A prior `magicquant search`'s `search_results.json` — the per-group hybrid config source. |
| `--tier` | `Q4` | Tier within the search results to make the adapters robust to. |
| `--dataset` | *required* | Chat JSONL (`{"messages": [...]}` per line), trained with completion-only loss. |
| `--out` | `MAGICQUANT_OUTPUT_DIR/qat_adapters` | Output adapter directory. |
| `--lora-r` / `--lora-alpha` | 32 / 64 | LoRA rank / scaling. |
| `--epochs` / `--max-steps` / `--lr` / `--max-seq-len` | 1 / -1 / 2e-4 / 512 | Training schedule. |
| `--expert-lora-r` / `--expert-lora-alpha` | 4 / 2×r | Rank / scaling for FUSED 3-D MoE expert tensors. Separate from `--lora-r` because the rank is paid **per expert per layer** (256 × 41 on Qwen3.6-35B-A3B): r=4 is 236M adapter params (~3.5 GiB with grads + AdamW moments), r=8 is double that. |
| `--expert-quant-mode` | `live` | `live` re-quantizes base+LoRA every forward; `frozen` quantizes the expert base once at wrap time. See the mode table above — `live` is ~92 min/forward on a 35B MoE. |
| `--no-expert-qat` | off | Skip fused 3-D experts entirely (Linear-only QAT, the pre-2026-08 behaviour). |
| `--gradient-checkpointing` | off | Recompute activations in the backward pass. **Effectively required with fused-expert QAT on a large MoE**: without it the autograd graph retains every layer's materialized expert weight (~66 GiB on Qwen3.6-35B-A3B, on top of the base). |

`load_hybrid_config(search_results.json, tier)` resolves each MagicQuant scheme
name (`"MXFP4_MOE"`, `"Q4_K_M"`, …) to its ggml block type name (`"MXFP4"`,
`"Q4_K"`, …) — the name the fake-quant dispatcher uses — and returns
`{group: ggml_type_name}` for `wrap_model`. `load_tensor_config` does the same for
a run's per-tensor `tensor_config` (budget builds), which takes precedence.

The dataset is tokenized with **completion-only loss**: each chat example's prompt
(everything up to and including the final user turn's generation prompt) is masked
with `IGNORE_INDEX = -100`, so loss is computed only on the assistant completion.

`run_qat(cfg)` freezes everything except the `QATLinear` LoRA params, trains with an
HF `Trainer` (cosine schedule, warmup, grad clipping, optional gradient
checkpointing), then writes the adapters to `adapter_model.safetensors` plus a
`qat_meta.json` describing the run (base model, `scheme_by_group`, a config hash,
and all hyperparameters). Merge the adapters (`merge_qat_adapters`) and pack the
exact hybrid with `magicquant generate`.

---

## Validated recovery result

The QAT feature was validated in a **confound-controlled** experiment on
**Qwen2.5-0.5B (base)** with an **aggressive Q4_K-attention / MXFP4-FFN hybrid** —
deliberately one of the most damaging configs, to give QAT something real to
recover. Perplexity (lower is better, same corpus throughout):

| Model | PPL | Gap vs bf16 |
|-------|-----|-------------|
| bf16 (reference) | **16.35** | 0.00 |
| quant (plain hybrid) | **19.54** | **+3.19** (quantization damage) |
| quant + QAT-LoRA | **15.13** | **+1.67** vs bf16 after the LoRA's own domain adaptation* |
| bf16 + identical-LoRA (control) | **13.46** | — |

\* The QAT model (15.13) is even better than the bf16 reference, but that is the
LoRA's domain adaptation talking, not quantization recovery — which is precisely
why the control matters.

### Why the control matters (methodology)

A LoRA fine-tune lowers perplexity on the training distribution all by itself,
*independent* of any quantization awareness. If you only compared `quant` (19.54)
to `quant+QAT` (15.13) you would credit QAT with the entire 4.41-point drop — but
most of that is just domain adaptation that a plain LoRA on the bf16 model would
also buy you.

To isolate the *quantization-recovery* component, we ran the **same LoRA, same
data, same steps** on the **bf16** model as a control (13.46). The comparison is
then made on the **bf16-vs-quant gap**, holding the LoRA's domain adaptation fixed
on both sides:

- **Before QAT:** quant 19.54 − bf16 16.35 = **+3.19**
- **After QAT:** quant+QAT 15.13 − bf16+LoRA-control 13.46 = **+1.67**

The gap shrank from **+3.19 → +1.67**. The recovered fraction is

```
(3.19 − 1.67) / 3.19 = 1.52 / 3.19 ≈ 47.5%
```

> **QAT recovered 47.5% of the quantization loss — beyond what plain LoRA
> domain-adaptation already buys you.** Roughly half the quantization damage of an
> aggressive hybrid is genuinely recoverable by making the LoRA quant-aware.

The validation harness `qat/validate.py` (`compare_perplexity`) packs both a plain
hybrid and the QAT-adapted hybrid to real GGUFs and runs `llama-perplexity` on each
over the same corpus, returning `{plain, qat, delta}`. The confound-controlled
result above additionally builds the bf16+identical-LoRA control arm so the
recovery fraction can be computed honestly.

### Production-path proof (real libggml GGUF)

The result above was measured in torch fake-quant space. The same confound-controlled
design was then repeated **end-to-end through the production path**: every arm was
`merge_qat_adapters`-merged, saved with transformers, packed to a **real GGUF with
MagicQuant's libggml encoders** (byte-identical to llama.cpp), and scored with
`llama-perplexity` (wikitext-2, `-c 512 --chunks 40`). Same aggressive
Q4_K-attention / MXFP4-FFN hybrid on Qwen2.5-0.5B; QT/BT arms seed-matched, the
only difference being whether the training forward fake-quantizes:

| Arm | Pack | PPL |
|-----|------|-----|
| B — base | F32 | 12.56 |
| Q — base | hybrid | 14.21 (**+1.65** raw quant damage) |
| QT — QAT-merged | hybrid | **11.94** |
| BT — LoRA control | F32 | 10.92 (QT−BT = **+1.02** residual) |

```
recovery = (1.65 − 1.02) / 1.65 ≈ 38.1%
```

> **QAT recovery survives the real pack: 38.1% of the quantization damage is
> recovered in the actual shipped GGUF**, confound-controlled. The QAT'd hybrid
> (11.94) even lands below the un-quantized base (12.56) — the LoRA's domain
> adaptation pays for the rest. The real-pack figure is somewhat below the torch
> fake-quant figures (45–66% across tiers) because training sees a faithful but
> approximate quant; the gap is the price of the approximation, not a failure of
> the method.

Getting this number required fixing three GGUF-pack bugs that the proof itself
exposed (all with regression tests, all general MagicQuant fixes rather than
QAT-specific): missing `.bias` tensor name-mapping (Qwen qkv-bias models wouldn't
load), transformers>=5 format drift (BPE merges became pair-arrays, `rope_theta`
moved into `rope_parameters` — re-saved models packed to garbage), and a missing
`tokenizer.ggml.pre` key (llama.cpp fell back to the wrong pre-tokenizer regex,
inflating perplexity on *every* MagicQuant pack — base 21.9→12.6 after the fix).

---

## Multimodal and bf16 support

- **bf16 base, fp32 LoRA.** `QATLinear` builds the merged weight in fp32 for the
  block-scale math but keeps the frozen base in its loaded dtype, so a bf16 base
  trains end-to-end (bf16 base + fp32 LoRA/optimizer state, fake-quant in fp32, cast
  back to the activation dtype for the matmul). The trainer's mixed-precision flag
  is matched to the loaded model dtype.
- **Multimodal models.** `run_qat` loads via the multimodal/conditional-generation
  auto-classes (`AutoModelForImageTextToText`, `AutoModelForMultimodalLM`, …) in
  addition to `AutoModelForCausalLM`, so **Gemma-3/4 and similar `*ForConditional
  Generation` models target their text decoder.** The vision/audio Linears simply
  don't route in `wrap_model` (their names don't map to GGUF tensor names), so only
  the text-decoder weights get fake-quant QAT — which is exactly the part the GGUF
  pack quantizes.

---

## Honest caveats

- **Recovery scales with quant aggressiveness.** The 47.5% figure is for a
  deliberately aggressive Q4_K-attention / MXFP4-FFN hybrid. For a gentle tier
  (mostly BF16 brain + Q6/Q8 attention) there is much less quantization damage to
  recover, so the absolute and relative recovery will be smaller. QAT pays off most
  at the aggressive tiers (Q2 / Q3 / MXFP4-heavy); on a near-lossless hybrid it
  mostly just does ordinary LoRA domain adaptation.
- **Training uses a faithful torch fake-quant, not exact ggml.** The forward-pass
  fake-quant in `qat/fake_quant.py` is a GPU-fast *approximation* of libggml,
  validated to within a per-scheme tolerance — it is **not** byte-exact (some
  schemes drop a level, approximate scale-of-scales with fp16, or skip the final
  RMSE refinement sweeps; the discrepancy is well inside tolerance). The **GGUF pack
  of the final model is exact-ggml**: `merge_qat_adapters` hands the *un*-quantized
  merged weight to `magicquant generate`, whose encoder calls `libggml` directly and
  is byte-identical to `llama-quantize`. So there is a small, bounded train/ship
  mismatch: the adapter is optimized against the approximation, the shipped weights
  are quantized exactly. The 47.5% figure was measured in torch fake-quant space;
  the **production-path proof above re-measures on the real packed GGUFs and gets
  38.1%** — the difference is the measured cost of that mismatch.
- **The QAT model can beat the bf16 reference outright** (15.13 < 16.35 here), but
  that is LoRA domain adaptation, not quantization magic. Always cite the
  confound-controlled +3.19 → +1.67 gap, not the raw quant-vs-QAT drop, when
  attributing a number to QAT.
- **The validated result is a dense model, Linear-only.** Every number above comes
  from Qwen2.5-0.5B with `QATLinear` in `live` semantics. Fused 3-D MoE expert QAT
  reuses the same fake-quant kernels and the same merged-weight discipline, but no
  recovery figure has been measured for it — and `frozen` mode is a *weaker*
  approximation than anything that produced these numbers (see the mode table
  above). Do not quote 47.5% / 38.1% for an MoE expert run.

---

## Where it lives

```
magicquant/qat/
  fake_quant.py   — differentiable per-scheme fake-quant + STE (validated vs libggml)
  wrap.py         — QATLinear (fake-quants merged base+LoRA) + wrap_model + bake/merge handoff
  expert_wrap.py  — FusedExpertQAT: parametrization + per-expert LoRA for fused 3-D MoE experts
  names.py        — HF module path → GGUF tensor name(s); fused_expert_segments for 3-D experts
  config.py       — load_hybrid_config ({group: scheme}) + load_tensor_config ({tensor: scheme})
  merge.py        — streaming on-disk base+adapter merge (2-D and 3-D expert keys)
  train.py        — run_qat: completion-only QAT-LoRA loop + adapter/meta save (offline tiny fallback)
  validate.py     — compare_perplexity: QAT hybrid vs plain hybrid via llama-perplexity
```

Tests: `tests/test_fake_quant.py` (fidelity vs libggml), `tests/test_qat_wrap.py`,
`tests/test_qat_expert_wrap.py` (fused 3-D experts, incl. interception of the real
`Qwen3_5MoeExperts.forward`), `tests/test_qat_handoff.py`,
`tests/test_qat_config.py`, `tests/test_qat_names.py`, `tests/test_qat_merge.py`,
`tests/test_qat_validate.py`, `tests/test_qat_smoke.py` (end-to-end one-step run).
