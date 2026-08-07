> Archived verbatim from `Foundry/output/qat-control-exp/RESULTS.md` (2026-08-07, run
> `model-qat-control-exp.service`, 55 min). output/ is gitignored and periodically
> cleaned; this tracked copy is the durable record. Regeneration instructions at the
> bottom refer to the original run directory.

# QAT control experiment — Qwen2.5-0.5B, 4 arms

**Question.** Frozen-mode QAT on the Qwen3.6-35B-A3B BUDGET-13.5GiB build measured NEGATIVE (−6.1% recovery; PPL 7.2809 → 7.3150, confirmed on two backends). Is *frozen mode's semantics* the problem — the delta is trained against a frozen fake-quantized base but ships as `quant(W + delta)` — or was the negative result about scale, architecture, or training/eval domain?

This runs the validated 47.5%-recovery setup (Qwen2.5-0.5B, aggressive Q4_K-attention / MXFP4-FFN hybrid) with **live**, **frozen**, and **bf16-control** arms trained identically, so the only variable is the training forward.

## Setup

- Base model: `Qwen/Qwen2.5-0.5B` (loaded fp32; every arm re-saved through the identical transformers path)
- Dataset: `/server/programming/Foundry/data/qat_corpus.jsonl`
- Hybrid: attention `Q/K/O → Q4_K_M`, FFN `U/D → MXFP4_MOE`, everything else `F32`. Reference packs are `F32` everywhere, so the *only* difference between a reference pack and a hybrid pack is the attention/FFN groups — no embedding-precision confound.
- LoRA: r=32, alpha=64.0, lr=0.0002, seq=512, steps=500, cosine schedule, warmup 0.03, seed=0 (identical in all three training arms; hyperparameters mirror the 35B budget QAT run)
- Perplexity: `llama-perplexity -c 512 --chunks 100 -t 8 -ngl 0` on `/server/ai/wikitext/wikitext-2-raw/wiki.test.raw`, ROCmFPX build
- Device: cpu
- Wrapped Linears: 168 (requested → effective: `{'MXFP4 -> MXFP4': 72, 'Q4_K -> MXFP4': 96}`)

> **Q4_K is not representable on this model.** Qwen2.5-0.5B's hidden size is 896, and a K-quant needs a row width divisible by 256 (896 = 3.5 × 256). The GGUF writer therefore falls back to the block-32 type MXFP4 for 96 of the 168 routed tensors — so the *packed* hybrid is MXFP4 for attention as well as FFN, whatever the config asks for. The arms fake-quantize the **effective** type, and every hybrid pack is read back and asserted tensor-by-tensor against it, so training and the shipped file cannot disagree. (The same fallback applies to the original 'Q4_K-attention' validation packs on this model — they were MXFP4-attention in the GGUF too.)

## Measured perplexity

| Arm | What it is | PPL | Δ vs baseline | GGUF |
|---|---|---|---|---|
| `A0-baseline-f32` | baseline — base weights, F32 pack, no training | **13.9844** | — | 2410 MiB |
| `A1-quant-only` | quant-only — base weights, hybrid pack, no training | **17.3871** | +3.4027 | 1226 MiB |
| `A2-live-qat` | live QAT — forward fake-quants W + s·BA (mainline QATLinear) | **17.2078** | +3.2234 | 1226 MiB |
| `A3a-frozen-qat-ship-orig` | frozen QAT — trained on fq(W) + s·BA, merged onto W | **17.5081** | +3.5237 | 1226 MiB |
| `A3b-frozen-qat-ship-fq` | frozen QAT — trained on fq(W) + s·BA, merged onto fq(W) | **17.5554** | +3.5710 | 1226 MiB |
| `A4a-control-f32` | bf16 control — identical LoRA on the unquantized base, F32 pack | **14.5474** | +0.5630 | 2410 MiB |
| `A4b-control-hybrid` | bf16 control — same run, hybrid pack | **18.3021** | +4.3177 | 1226 MiB |

Quantization damage (the thing QAT is supposed to recover): **17.3871 − 13.9844 = +3.4027 PPL**.

## Confound-controlled recovery

A LoRA fine-tune lowers perplexity by itself, independent of any quantization awareness. The honest recovery figure therefore compares *gaps*, holding the LoRA's own domain adaptation fixed on both sides (the methodology behind the validated 47.5% figure, MagicQuant `docs/qat.md`):

```
damage    = PPL(A1 quant-only)  − PPL(A0 baseline)
residual  = PPL(arm)            − PPL(A4a bf16+identical-LoRA control)
recovery% = (damage − residual) / damage
```

| Arm | PPL | residual vs control | recovery |
|---|---|---|---|
| `A2-live-qat` | 17.2078 | +2.6604 | **+21.8%** |
| `A3a-frozen-qat-ship-orig` | 17.5081 | +2.9607 | **+13.0%** |
| `A3b-frozen-qat-ship-fq` | 17.5554 | +3.0080 | **+11.6%** |
| `A4b-control-hybrid` | 18.3021 | +3.7547 | **-10.3%** |

## Training arms

| Run | Mode | Steps | Wrapped | Trainable | Final loss | ‖Δ‖_F | max|Δ| | Runtime |
|---|---|---|---|---|---|---|---|---|
| `A2_live` | live | 500 | 168 | 17.60M | 1.3090 | 10.214 | 0.00734 | 31.5 min |
| `A3_frozen` | frozen | 500 | 168 | 17.60M | 1.3032 | 10.106 | 0.00871 | 10.3 min |
| `A4_control` | none | 500 | 168 | 17.60M | 1.2154 | 10.055 | 0.00727 | 9.7 min |

Identical wrap set, rank, seed and step count across arms — the only difference is the forward. A non-zero `max|Δ|` in every row is the guard that no arm silently trained a no-op adapter.

## Interpretation

1. The hybrid cost +3.4027 PPL (+24.33%), so there is real damage to recover.

2. **Live QAT reproduces meaningful recovery (+21.8%)** on this harness. The method and the harness are intact; the 35B negative result is therefore *not* explained by the harness having rotted.

3. **Frozen mode is materially worse than live** (+13.0% vs +21.8%, a -8.8 pt gap) at the same scale, architecture, dataset, hyperparameters and step count. Scale/architecture/domain are held constant here, so this isolates **the mode itself** as the cause of the 35B negative result.

4. Making the frozen merge self-consistent (delta onto `fq(W)`) changed recovery by only -1.4 pt, so the merge target is **not** the dominant problem — frozen mode is weak on its own terms.

5. Quant-awareness earns its keep: live QAT beats plain-LoRA-then-quantize by 32.2 pt (+21.8% vs -10.3%).

### What this experiment cannot say

- It is a **dense 0.5B**. Frozen mode does not exist for `nn.Linear` in mainline MagicQuant (only for fused 3-D MoE experts), so the frozen arm here is an experiment-local reimplementation of those exact semantics (`exp_qat.ExpLoRALinear`, mode `frozen`), asserted against mainline for the live path. It reproduces the *semantics*, not the MoE routing.
- The 35B run also differed in **eval domain** (chat-blend training, wikitext eval). That confound is present here too — which is exactly what the A4a control absorbs, and why recovery is computed against it rather than against the raw quant-vs-QAT drop.
- Absolute recovery percentages depend on how aggressive the hybrid is and how long training runs; the *comparison between arms* is what this experiment is powered to answer, because every other variable is pinned.

---

Regenerate this file from the artifacts with `python -c "import results,pathlib; results.render(pathlib.Path('.'))"`. Raw measurements: `ppl/*.json`, `ppl/*.log`, `runs/*/train_meta.json`, `config.json`.
