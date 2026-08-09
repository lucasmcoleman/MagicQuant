# MagicQuant v2 — Budget-Constrained Mixed-Precision Allocation

**Status:** design approved, implemented behind `--algo v2` (old path untouched).
**Author:** algorithmic-redesign session, 2026-07-11/12 (fable-sprint prompt 6).
**Entry points:** `magicquant search <model> --algo v2 --budget-gb <B>`,
`magicquant.v2.run_budget_search`.

---

## 1. What v1 actually does, and why it underperforms

The current pipeline (`magicquant/orchestrator.py` → `evolution/`) is:

1. **Baseline PPL** — one llama-perplexity pass.
2. **Sensitivity probing** (`evolution/probing.py`) — for each of 7–10 tensor
   *groups* (E/H/Q/K/O/U/D + X/R/S), build a probe GGUF with that group at
   **one** scheme (Q4_K_M) and everything else BF16, measure PPL. Sensitivity
   is the relative PPL increase, normalized so weights sum to 1.
3. **Evolutionary search** (`evolution/survival.py`) — random per-group
   scheme assignments, tournament selection per size tier, Protector/Crusher
   mutations, epsilon exploration — all scored by a **predictor**
   (`evolution/predictor.py`): `loss = Σ_g w_g · noise_factor(scheme_g)` plus
   a collapse penalty.
4. **Measured rounds** — per round, build + measure ~4 tier winners /
   epsilon picks; feed `residual = measured − predicted` back into the
   predictor.
5. **Tier survivors** — best measured config per post-hoc size band.

### The five structural problems

**P1 — Sensitivity is one number per group.** One probe, one scheme,
whole-group. There is no per-layer resolution (layer 0's `ffn_down` and layer
15's `ffn_down` get identical treatment; in practice first/last layers are
far more sensitive — this is exactly why llama.cpp's own Q4_K_M mixture
bumps specific layers). There is no per-scheme response curve: the measured
Q4_K_M response is extrapolated to every other scheme through *static*
`noise_factor` constants. The imatrix — an activation-statistics signal we
already capture (`magicquant/imatrix.py`) — is used to improve *encoding*
but never as a *sensitivity* signal.

**P2 — The "learning" doesn't generalize.** `PredictiveScorer.residual_cache`
keys the **exact** config string. A residual measured for config A adjusts
predictions for config A only — which is already measured. Every unmeasured
config is scored by the same static linear model regardless of how many
measurements exist. The Predict→Measure→Learn loop learns nothing usable.

**P3 — The evolutionary search is solving a problem that has a closed-form
answer.** Under an additive predictor, "best per-group assignment per size
tier" is a small multiple-choice knapsack — exactly solvable in
microseconds. Randomized evolution over 30 generations × 80 candidates adds
sampling noise and code complexity, not optimization power. Its outputs
mostly rediscover the obvious configs (admitted in CLAUDE.md's Known
Limitations).

**P4 — No budget constraint.** The user's real question on unified-memory
hardware is "best model that fits in N GB of GTT alongside ctx/KV". v1
answers "best config per ±size band that the random search happened to
visit". Nothing lets you *ask for* 11.5 GB.

**P5 — Robustness: fabricated measurements poison whole runs (the bug the
mission told us to find).** `SensitivityProber._real_probe`
(probing.py:297–306) catches any non-ValueError failure — subprocess crash,
transient GPU OOM while another process holds the GPU, timeout — and
**returns `_heuristic_probe()`: a fabricated PPL from static per-group
constants**, marked only by `measured=False` in a JSON nobody gates on. In a
*measured* search this means: one transient failure during probing → the
entire multi-hour run (every generation, every mutation decision, every
tier ranking) silently prices that group off made-up numbers. The run
"succeeds", `probing_provenance: "partial"` is stamped, nothing fails, no
candidate is excluded. Sibling instances of the same class:
`run_full_search` fabricates `baseline_ppl = 5.0` on measurement failure
(flagged by provenance, still fabricated); a probe PPL that lands *below*
baseline (pure noise) clamps to sensitivity 0.0, telling the search that
group is free to crush.

**Cost accounting (v1, defaults):** 1 baseline + 7–10 probes + 3 rounds × 4
candidates (+3 incumbents forced in round 1) ≈ **20–23 full llama-perplexity
passes** plus as many full GGUF builds. On a 27B on this box that is a
multi-day run; the audit trail (AUDIT_2026-07-01.md) shows exactly this
pain motivating chunk-capping.

---

## 2. v2 in one paragraph

Replace probe-and-evolve with **compute-allocate-verify**: (a) compute a
**per-tensor × per-scheme distortion table** — imatrix-weighted quantization
error, obtained by encoding and decoding every tensor with the *same
libggml the shipped file will use* — a CPU-only pass, no perplexity runs;
(b) calibrate per-group **amplification factors** with a handful of
chunk-capped probe measurements (optional but default); (c) solve the
allocation **exactly** as a multiple-choice knapsack under the user's byte
budget, which also yields the entire predicted quality–size **frontier** in
one solve; (d) spend the GPU only on **verifying 2–3 frontier anchors**
with full-corpus perplexity, recording failures loudly per-candidate. Net:
per-tensor (not per-group) bit allocation, a real budget constraint, and
**~4–6 full measurement passes instead of 20+ (≥5× reduction)** with
strictly more expressive allocations.

```
BF16 source ──► imatrix (cached, 1 short GPU pass)
    │
    ├─► [CPU] distortion table  ε(tensor, scheme)   — encode+decode via libggml,
    │         cached per (model, scheme-set, imatrix)
    │
    ├─► [GPU, capped, optional] group probes → κ_g amplification fit  (strict:
    │         a failed probe FAILS, it is never faked)
    │
    ├─► [CPU] MCKP allocator:  min Σ κ·ε  s.t.  Σ bytes ≤ budget
    │         → chosen per-tensor assignment + full predicted frontier
    │
    └─► [GPU] verify K frontier anchors (full corpus)  → v2_results.json,
              frontier.json, final hybrid GGUF
```

---

## 3. Sensitivity estimation (mission item 1)

### 3.1 The distortion model

For tensor `t` with weights `W ∈ R^{rows×cols}` (GGUF row-major convention;
`cols` = input dimension), quantized by scheme `s` into `Ŵ_s`, define

```
ε(t, s) = Σ_c m_t[c] · Σ_r ( W[r,c] − Ŵ_s[r,c] )²
```

where `m_t` is the imatrix vector for `t` — the mean squared activation per
input column captured by `llama-imatrix` (`in_sum2 / counts`, see
`magicquant/imatrix.py`). Under the standard independence approximation this
is `E‖(W − Ŵ_s) x‖²` up to cross-terms — the expected squared perturbation
of the layer's output on the calibration distribution. It is the same
objective ggml's own imatrix-weighted quantizers minimize locally, and the
same family of second-order proxies as HAWQ/GPTQ-style sensitivity, using
activation second moments as the (diagonal) curvature surrogate. Without an
imatrix, `m ≡ 1` (plain squared error) — still per-tensor, still per-scheme,
degraded but honest and *labeled as such* in the manifest.

`Ŵ_s` is produced by **encoding with `ggml_quantize_chunk` via the existing
byte-exact `magicquant/quant/ggml_binding.py` (imatrix-weighted when
active, i.e. the identical bytes the writer would ship) and decoding with
the corresponding exported `dequantize_row_*` symbol** (verified present in
the target libggml: `dequantize_row_{q8_0,q6_K,q5_K,q4_K,q3_K,q2_K,iq4_nl,
mxfp4,q4_0,q4_1}`; ROCmFPX fork exports `rocmfp4_dequantize_row_q4_0` and
`rocmfpx_dequantize_row_fp{3,6,8}`). No re-implementation of any format, no
MSE-vs-real-encoder gap. A scheme whose decode symbol is missing from the
loaded libggml is **excluded from the choice set with a logged reason**,
never approximated.

Per (tensor, scheme) we resolve the **actual** on-disk type first — the same
rules as the writer's Pass 1 (1-D → F32, SSM conv operands → F32, BF16 → F16
downgrade, 256-block K-quant on non-divisible rows → block-32 fallback,
`requires_imatrix` gate) — via a shared pure function
(`magicquant/v2/resolve.py`) with a writer-parity test. So ε and byte-size
are computed for the type that would actually ship, not the requested label.

### 3.2 Allocation units

The unit is the **individual tensor** (≈ 130 units for a 1B dense model,
~300–700 for larger/MoE). This is the finest granularity the GGUF container
supports, it is what llama.cpp's own mixtures use, and it subsumes v1's
groups (a group is just a set of units). Groups survive for two purposes:
κ calibration (below) and human-readable reporting. F32-forced tensors
(norms, SSM conv) are fixed units with a single choice.

### 3.3 Cost & caching

One streaming pass over the model per scheme, CPU-only, numpy-vectorized,
parallelizable per tensor. For the 1B validation model × 8 schemes this is
minutes; for a 27B it is bounded by disk reads (~8 model-reads) and still
requires **zero GPU seconds**. Results are cached at
`<output>/_v2_cache/distortion_<key>.json` keyed on (model identity
path+size+mtime, scheme set, imatrix key, sampling params). Optional row
subsampling (`--sensitivity-sample-rows N`, unbiased row-stride estimator)
for very large tensors; off by default, recorded in the manifest when used.

### 3.4 Group amplification κ (the *measured* part)

Raw ε is comparable *within* a layer's algebra but layers at different
depths amplify output perturbation differently (residual-stream position,
norm placement). We correct with per-group multipliers:

```
predicted_rel_ppl_loss(alloc) = Σ_t κ_{g(t)} · ε(t, alloc_t)          (†)
κ_g = ΔPPL_rel(probe_g) / Σ_{t∈g} ε(t, s_probe)
```

Probes reuse v1's build (group at `s_probe` = Q4_K_M, rest BF16) but are
**chunk-capped** (default `--probe-chunks 24`) — κ is a ratio of the same
measurement conditions, so a capped pass suffices; and they are **strict**
(§6). `--no-group-probes` skips them (κ≡1) for a zero-GPU-until-verify run;
`--kappa-from <file>` reuses a previous fit (κ is a *model* property, stable
across budgets — compute once, sweep budgets forever).

Because (†) is monotone in Σκε, any *global* miscalibration of scale does
not change the argmin under a byte constraint — κ only needs to get
*relative group scaling* right for the allocation to be right. This is why
a handful of capped probes is enough where v1 needed nothing less than full
fidelity everywhere.

### 3.5 What replaces the residual cache

Anchor measurements (§5) are fed back as a **generalizing** fit: a
least-squares refit of κ (and a global affine map ε→PPL for *reporting*)
over all measured (probe + anchor) points, persisted to
`<output>/v2_calibration.json`. Every future run of the same model — or a
re-run at a different budget — starts from the refined fit. Unlike v1's
exact-config residual cache, a measurement taken anywhere in config space
improves predictions everywhere, because the model is (†), not a lookup
table.

---

## 4. Allocation as constrained optimization (mission item 2)

### 4.1 Formulation

Multiple-choice knapsack (MCKP): for each unit `t` choose one scheme
`s ∈ C(t)`; minimize `Σ_t κ_{g(t)} ε(t,s_t)` subject to
`Σ_t bytes(t, s_t) ≤ B`. `C(t)` is the capability-filtered choice set:
registry schemes minus (`requires_imatrix` without imatrix; fork types
without fork libggml; types whose decode symbol is missing; user floors;
target-profile restrictions §7). `bytes(t,s)` is exact ggml block math on
the *resolved* type (§3.1), so predicted size ≡ file size.

### 4.2 Why Lagrangian/convex-hull greedy (the chosen formulation)

Considered:

- **Exhaustive / evolutionary over whole-model types** (v1): no budget
  handle, no per-tensor resolution; rejected for the reasons in §1.
- **Dynamic programming over discrete byte capacities:** exact, but requires
  discretizing the budget (MB cells × ~10⁵ unit-choices — fine for one
  budget) and yields **one** budget per solve; re-solving per frontier point
  is wasteful; exactness beyond the hull is worth ~nothing here (see gap
  bound below).
- **Plain greedy marginal-utility on raw points:** can be arbitrarily bad on
  non-convex per-unit curves (a scheme that is both bigger *and* worse than
  a mix of its neighbors traps the greedy).
- **Chosen: per-unit lower convex hull + global slope-greedy** (equivalent
  to sweeping the Lagrange multiplier λ through all hull-edge slopes):
  1. Per unit, compute the lower convex hull of its `(bytes, loss)` choice
     points — dominated and non-hull choices can never be part of a
     λ-optimal solution.
  2. Start every unit at its smallest-bytes hull point (if that total
     already exceeds B → **infeasible, fail loudly with the minimum
     achievable size** — no silent "closest effort").
  3. Repeatedly apply the hull edge with the best `Δloss/Δbytes` that still
     fits (max-heap); each application is one unit stepping one hull point
     up.
  4. Finish with a bounded local-search polish: single-unit upgrades
     (including non-hull points) that fit the residual budget and improve
     loss; ≤2 sweeps.

  Properties: `O(Σ|C| log)` — microseconds at our sizes; the greedy prefix
  trace **is the entire predicted quality–size frontier** (every prefix is
  the hull-optimal allocation for its size — one solve, all budgets), which
  §5 measures and `docs/validation.md` plots; and the integer-optimality gap
  is at most one hull edge of one unit — with hundreds of units each
  contributing ≤ ~1% of total bytes, that is far below measurement noise.
  The polish step usually closes even that.

### 4.3 Guardrails

Hard floors are **off by default** — measured ε already encodes what v1's
hand-coded "brain" floors were guessing at — but remain available
(`--floor E=Q6_K,H=Q6_K`) for users who want llama.cpp-incumbent-style
insurance, and the writer's compatibility forcings (§3.1) are always
respected. The output embedding/head choice interacts with streaming
bandwidth (v1's `stream_aware` finding: Q8_0 head ≈ PPL-neutral, −16%
bytes); v2 doesn't need a special dial — the byte cost is priced directly
by the budget constraint.

---

## 5. Search efficiency: measure the frontier, not the population (mission item 3)

v2's only full-corpus GPU passes are:

1. **Baseline** (1 pass; reused from cache/checkpoint when identity
   matches — same rules as v1's resume checkpoint).
2. **Anchor verification** (default K=2, `--anchors K`): the budget
   allocation itself, plus its nearest frontier neighbor(s) at ~±7% bytes.
   Anchors are built with the existing writer (per-tensor overrides) and
   measured on the full corpus. Purpose: (a) verify the chosen quant with a
   real number *before* it ships (v1 never measured the final tier build at
   all unless `verify=True`); (b) refine κ (§3.5); (c) place *measured*
   points on the frontier plot.
3. **Successive halving, only when it earns its keep** (`--race N`): when
   the predicted loss of the top-N frontier-adjacent allocations (e.g.
   MXFP4-heavy vs IQ4_NL-heavy vs Q4_K-heavy mixes near the same budget)
   differs by less than the surrogate's residual error, race them:
   all N at `--probe-chunks` cost → keep ceil(N/2) → full corpus for the
   final 1–2. Off by default (the surrogate separates typical candidates
   fine).

**Accounting vs v1 (defaults, same model):**

| | full-corpus PPL passes | GGUF builds | per-layer? | budget? |
|---|---|---|---|---|
| v1 measured search | 1 + 7–10 probes + ~12–15 candidates ≈ **20–23** | ~20 | no | no |
| v2 | 1 baseline + 2 anchors = **3** (+7–10 *capped* probes ≈ 1–2 full-pass-equivalents) | 7–12 (probes + anchors) | yes | yes |

≥5× fewer full measurements is met by construction; validation (§8)
demonstrates equal-or-better quality at matched size.

---

## 6. Robustness: no fabricated numbers, no aborted runs (mission item 4)

Doctrine: **a measurement either succeeded, or it failed and the artifact
records the failure.** Nothing downstream ever consumes a fabricated value;
no single candidate failure kills a run that can still make progress.

Mechanics (`magicquant/v2/outcome.py`):

- Every build/measure step returns `MeasurementOutcome{status: ok|failed,
  value, error, attempts}`. All outcomes — including failures — land in
  `v2_results.json` under `"failures"` with the underlying error text.
- **Probes:** one retry after a failure (transient GPU contention is the
  common case on this box). Still failing → `ProbeFailure` recorded; by
  default the run **aborts with a message naming the group and the fix**
  (strict), because κ built on a missing group is exactly the silent
  degradation we're killing. `--allow-partial-probes` continues with the
  failed group's κ set to the *median of measured κ*, tagged
  `"kappa_provenance": "imputed-median"` in the manifest and warned at
  the end-of-run summary — visible degradation, opt-in only.
- **Anchors:** a failed anchor build/measure is recorded, the next frontier
  candidate is promoted, and the run continues; ALL anchors failing →
  RuntimeError (a run that verified nothing must not claim success —
  mirrors the v1 fix for zero-measurement runs).
- **Baseline:** never fabricated in v2. No measurement → hard error (v1's
  measured path already does this; v1's `run_full_search` 5.0-fabrication
  stays quarantined in the legacy path).

**The v1 kill (shared code, gated):** `SensitivityProber` gains
`strict: bool`. `_real_probe` no longer silently substitutes heuristics on
generic failure when strict; `MagicQuantOrchestrator.run_measured_search`
constructs its prober with `strict=True` — a *measured* search now fails
loudly on a failed probe instead of silently ranking a multi-hour run on
static guesses. Prediction-only search (`run_full_search`, no llama.cpp)
keeps heuristics — there the fallback is the documented design, not a
degradation. This is a deliberate, narrow behavior change to the v1 measured
path, with tests (`tests/test_probe_errors.py` extended); everything else
about v1 is untouched.

---

## 7. Custom formats as first-class choices (mission item 5)

- **ROCmFPX (`/home/lucas/ROCmFPX`)** — real ggml types (ids 100–104) in a
  llama.cpp fork; already registered in `quant/schemes.py` (category
  `rocmfpx`). In v2 they are ordinary MCKP choices whenever the loaded
  libggml supports them (existing name-probe via `handle.supports`), with ε
  computed through the fork's exported dequant symbols
  (`rocmfp4_dequantize_row_q4_0`, `rocmfpx_dequantize_row_fp{3,6,8}`), and
  gated by `--enable-rocmfpx` exactly like v1 (files with these types load
  only on the fork). The IQ family (minus the sub-2-bit types §9 excludes) is
  likewise gated by `--enable-iq`, with the imatrix-requiring members
  (IQ2_XS/IQ2_XXS) admitted only when an imatrix is active — the
  BudgetInfeasibleError advice to pass `--enable-iq` is real as of the 2026-08
  cleanup pass. No sampling-mass hand-tuning needed anymore — measured
  ε prices them against MXFP4/K-quants directly, which is the head-to-head
  the fork always needed.
- **Q4NX (`/server/programming/FLM_Q4NX_Converter`)** — investigated:
  Q4NX is a **whole-file NPU container format** (FastLLM safetensors-side
  pack of Q4_0/Q4_1/Q8_0/MXFP4-family blocks), *not* a ggml tensor type;
  there is no type id a GGUF tensor could carry, so "per-tensor Q4NX inside
  a GGUF" does not exist as an operation. First-class treatment that is
  actually true to the format: a **target profile** (`--target-profile
  q4nx`) that restricts `C(t)` to the Q4NX-packable base types (adds
  Q4_0/Q4_1 to the registry as ordinary schemes; keeps Q8_0/MXFP4), so the
  budget-optimal hybrid GGUF v2 emits converts losslessly through the Q4NX
  packer for NPU serving, and its (size, quality) point sits on the same
  frontier plot as everything else. The design keeps a `TargetProfile`
  table so future container formats are one entry, not a redesign.

---

## 8. Validation protocol (ran on this machine; results → docs/validation.md)

- **Model:** `/server/ai/models/source/Llama-3.2-1B-Instruct-bf16.gguf`
  (2.8 GB BF16; existing measured scheme calibration at
  `/server/ai/calibration_results_2026-06-11.json`, baseline wikitext PPL
  14.3627 under ctx 512). Chosen for full-corpus statistical cleanliness
  under GPU-sharing constraints; the algorithm is size-agnostic.
- **Corpus:** `/server/ai/wikitext/wikitext-2-raw/wiki.test.raw`, ctx 512 —
  identical conditions both pipelines, recorded in both results files.
- **Old pipeline:** `run_measured_search` (defaults + seed pinned) →
  tiered survivors; take the Q4-band winner's measured PPL and exact bytes.
- **New pipeline:** `--algo v2 --budget-gb <exactly the v1 winner's size>`
  → measure the anchor. Same-budget comparison is the headline number;
  additionally 3–4 frontier points measured for the plot.
- **Downstream task:** HellaSwag accuracy via `llama-perplexity
  --hellaswag` (same binary, deterministic), both winners + BF16 reference.
- **Report:** `docs/validation.md` — table (config, GB, PPL, ΔPPL%, task
  score), frontier plot (predicted curve + measured points, v1 winners
  overlaid), measurement-count comparison, and an honest verdict: the
  redesign wins only if the v2 frontier dominates at matched size.
- All GPU commands wrapped in `flock /tmp/claude-gpu.lock` (shared box).

## 9. Non-goals

- Replacing the writer/encoder stack (byte-exactness is a solved, tested
  property — v2 builds on it).
- QAT integration changes (search_results.json consumers keep working; v2
  emits a compatible `tiered`-style block for its chosen allocation).
- Sub-2-bit IQ types, speed-aware objectives beyond byte-pricing, and
  KV-cache/ctx budgeting (the budget knob is weights-only; document the
  GTT arithmetic in the README instead).
- Removing v1. It remains the default until v2 has more mileage.

---

## 10. Post-validation addendum: cumulative "leave-one-group-high" κ probes

The first matched-budget validation (docs/validation.md) exposed a real
flaw in the §3.4 κ calibration: **single-group probes underestimate the
importance of any layer whose quantization error compounds downstream.**

### The failure

A single-group probe damages group G alone against an otherwise-pristine
model: `{G: Q4_K_M, rest: BF16}`. For `token_embd.weight` (21% of a 1B
model's params) this barely moved PPL — κ_E measured 1.4e-5 — because the
full-precision layers downstream absorbed the embedding error. The
allocator, told embeddings are nearly free to damage, crushed E to Q4_K_M
to buy precision elsewhere, and the model measured **+9.2% PPL vs v1's
+6.0%** at matched size. But in the *shipped* allocation nothing downstream
is full-precision — every layer is quantized — so embedding error is NOT
absorbed. The probe measured sensitivity in a context that doesn't match
deployment.

### The fix: marginal importance in a quantized context

Replace the single-group probe with a **cumulative "leave-one-group-high"**
probe (`--probe-mode cumulative`, the old behavior stays as `single`):

```
base_aggressive   = every allocatable group at s_probe (Q4_K_M)   → PPL_base
leave_G_high      = base_aggressive but G at keep_scheme (BF16)    → PPL_leave_G
recovery_G        = (PPL_base − PPL_leave_G) / slice_baseline_ppl
κ_G               = max(recovery_G, censor_floor) / Σ_{t∈G} ε(t, s_probe)
```

`recovery_G` is how much PPL is **recovered by keeping G high while
everything else stays quantized** — exactly G's marginal value in the
heavily-quantized regime the allocator actually operates in. It carries the
same units as the single-group rel-ΔPPL (relative PPL ÷ distortion), so
`fit_kappa`, the censoring floor, and the allocator are unchanged
downstream — only the *measurement* differs. An embedding layer that
compounds error downstream now shows a large `recovery_G` and gets a κ that
protects it; a genuinely-robust FFN layer recovers little and stays
crushable.

### Cost

`single` mode: 1 slice-baseline + N group probes = N+1 capped passes.
`cumulative` mode: 1 slice-baseline + 1 base-aggressive + N leave-one
probes = N+2 capped passes — one extra measurement, same order, still
dwarfed by v1's per-candidate full builds. The distortion table (§3) is
mode-independent and still computed once and cached.

### Why not make it the default immediately

`single` is retained and remains the default so existing v2 results
(validation-v2b) stay reproducible and the change is opt-in and A/B-able.
The validation in docs/validation.md reports whether `cumulative` closes
the v1–v2 gap **without** the manual `--floor E=Q6_K` guardrail — if it
does, `cumulative` becomes the recommended default in a follow-up once it
has mileage across more models.
