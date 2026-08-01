# WORKLOG — MagicQuant algorithmic redesign (v2 "budget search")

Mission: fable-sprint prompt 6 (/server/ai/docs/fable-sprint-prompts.md §6).
Session start: 2026-07-11.

## Status: IN PROGRESS

## Plan (priority order per mission)
1. [x] Audit — code (orchestrator/predictor/survival/probing/llamacpp/imatrix/writer/schemes)
       + audit docs digested (Sonnet lane) + environment inventory (Sonnet lane).
2. [ ] Design doc — docs/redesign.md
3. [ ] Implementation — magicquant/v2/ (sensitivity, allocation, surrogate search) behind
       `--algo v2`; old path untouched by default.
4. [ ] Robustness fix — kill the silent-degradation bug class (fabricated probe
       sensitivities in a measured search).
5. [ ] Validation — old vs new at same size budget on Llama-3.2-1B (wikitext PPL +
       downstream task), frontier plot, docs/validation.md. GPU work under
       `flock /tmp/claude-gpu.lock` and sequenced last.

## Audit findings (condensed; full rationale goes in docs/redesign.md)
- Sensitivity = 7–10 whole-group probes at ONE scheme (Q4_K_M), normalized to sum 1.
  No per-layer resolution, no per-scheme response curve, no use of imatrix as a signal.
- PredictiveScorer.predict_loss = sum(w_g * static noise_factor) + collapse penalty.
  The "active learning" residual_cache keys EXACT configs → zero generalization.
- EvolutionarySurvivor searches a per-group scheme space whose optimum, under the
  additive predictor, is directly solvable — the evolution machinery adds noise, not power.
- No size-budget constraint anywhere; tiers are post-hoc size bands.
- Robustness bug class (mission item 4): `SensitivityProber._real_probe` catches generic
  measurement failures and substitutes `_heuristic_probe` FABRICATED values into a
  measured search (probing.py:297-306). One transient GPU failure → whole run silently
  ranks configs on made-up sensitivities. Provenance is stamped "partial" but nothing
  fails and nothing gets excluded. Also: `run_full_search` fabricates baseline_ppl=5.0.
- Measurement cost today: ~1 baseline + 7-10 probes + ~12 candidates ≈ 20+ full
  llama-perplexity passes per run.

## Environment decisions
- Measurement tree: /home/lucas/ROCmFPX/build-strix-rocmfp4/bin (llama-perplexity/imatrix/
  quantize/bench + libggml with dequantize_row_* symbols for stock AND fork types).
  /server/ai/llama.cpp/build is being churned by the concurrent kernel-sprint agent — do not use.
- Validation model: /server/ai/models/source/Llama-3.2-1B-Instruct-bf16.gguf (2.8 GB;
  measured per-scheme calibration exists at /server/ai/calibration_results_2026-06-11.json,
  baseline wikitext PPL 14.3627).
- Corpus: /server/ai/wikitext/wikitext-2-raw/wiki.test.raw.
- Q4NX (FLM_Q4NX_Converter) is a whole-file NPU safetensors container, NOT a ggml type —
  first-class treatment = a target-profile that constrains the choice set to Q4NX-packable
  base types (Q4_0/Q4_1/Q8_0/MXFP4). ROCmFPX types (ggml ids 100-104) are per-tensor
  choices when the fork libggml is active; fork exports rocmfp4_dequantize_row_q4_0 /
  rocmfpx_dequantize_row_fp{3,6,8}.

## Log
- 2026-07-11 ~22:00 Audit complete (code read directly; audit docs + inventory via Sonnet lanes).
- 2026-07-11 ~22:20 WORKLOG created. Next: docs/redesign.md.
- 2026-07-12 (resumed after harness restart) docs/redesign.md written.
- 2026-07-12 Implementation complete:
  - magicquant/v2/{__init__,outcome,resolve,sensitivity,allocate,calibrate,search}.py (mine)
  - ggml_binding decode support + tests (Sonnet lane A)
  - writer per-tensor "tensors" overrides + Q4_0/Q4_1 v2-only schemes + tests (Sonnet lane B)
  - Robustness kill: SensitivityProber(strict=) + ProbeMeasurementError; measured search now
    strict=True (probing.py, orchestrator.py) — the fabricated-heuristic fallback is dead in
    measured runs, preserved in prediction-only mode where it's the documented design.
  - CLI: `magicquant search --algo v2 --budget-gb ...` (+ --anchors/--probe-chunks/
    --no-group-probes/--allow-partial-probes/--target-profile q4nx/--floor/
    --sensitivity-sample-rows/--keep-anchors); config.py gains algo/budget_gb
    (MAGICQUANT_ALGO / MAGICQUANT_BUDGET_GB).
  - Tests: 611 passed, 6 skipped (pre-existing GPU-gated) — seed-pinned v1 fixture unchanged;
    new: test_v2_allocate, test_v2_resolve, test_v2_sensitivity, test_strict_probing,
    test_ggml_decode, test_writer_tensor_overrides, test_legacy_q4_schemes.
- Ownership ledger (for verification): PRE-EXISTING uncommitted state, NOT this session's:
  AUDIT_2026-06-09.md / AUDIT_FIXPLAN_2026-06-09.md / README.md / Makefile modifications,
  Dockerfile deletion (relocated to docker/Dockerfile), .github/workflows/docker-build.yml.
  THIS session's: WORKLOG.md, docs/redesign.md, docs/validation.md, magicquant/v2/*,
  edits to probing.py/orchestrator.py/__main__.py/config.py, tests/test_v2_*.py,
  tests/test_strict_probing.py; via subagent lanes: ggml_binding.py, writer.py, schemes.py,
  survival.py, tests/test_ggml_decode.py, tests/test_writer_tensor_overrides.py,
  tests/test_legacy_q4_schemes.py.
- 2026-07-12 Validation complete (GPU, flock-gated per-subprocess wrappers):
  - Reference: had to re-convert a clean F16 1B from safetensors — the two pre-existing
    models/source/*.gguf are BROKEN June-permutation-bug exports (PPL 22794/1740), verified
    across 2 builds. Fresh ref baseline PPL 18.3675, HellaSwag@400 59.25%.
  - v1 measured search (seed 42, 32-chunk): 23 PPL passes; tier winners rebuilt+re-measured
    full-corpus. Q4 winner 0.7446 GiB, PPL 19.4738 (+6.02%); +imatrix control 18.8069 (+2.39%).
  - v2 @ matched 0.7373 GiB budget (128-chunk probes): 4 full + 7 capped passes (5.75x fewer
    full measurements — the >=5x goal MET). Anchor PPL 20.0529 (+9.20%).
  - VERDICT: v2 does NOT dominate v1 at matched size — honest negative result. Root cause
    diagnosed + confirmed CPU-side: allocator crushed token_embd (21.3% of params) to Q4_K_M
    because its single-group probe kappa_E=1.4e-5 underestimates embedding importance (error
    compounds downstream; additive surrogate misses it). Guardrail `--floor E=Q6_K` (design
    had it from the start) raises E back to Q6_K at identical budget — measured PPL of the
    floored config is the ONE deferred cell (GPU contended, window closing).
  - Two robustness fixes shipped: (1) the mission's fabricated-heuristic kill (strict probing);
    (2) a SECOND silent-degradation instance found DURING validation — noisy probe -> kappa~0
    -> crush — fixed via censoring (fit_kappa measured-censored, tests/test_v2_calibrate.py).
  - Artifacts: docs/validation.md, docs/validation-frontier.png, tools/plot_frontier.py,
    output/validation-v1/, output/validation-v2b/. Tests: 615 passed, 6 skipped.
- STATUS (first pass): COMPLETE. Only open item = measured PPL of the --floor E=Q6_K v2
  config (deferred post-cutoff GPU); does not change the reported verdict.

## Round 2 (2026-07-12, post-commit at 9f0906f) — cumulative probes + close the floor cell
- Implemented the diagnosed root-cause fix: CUMULATIVE "leave-one-group-high" kappa probes
  (--probe-mode cumulative), the old single-group mode kept as default. docs/redesign.md §10
  appended with the design. Files: magicquant/v2/calibrate.py (run_group_probes gains
  probe_mode + base-aggressive measurement; fit_kappa auto-detects mode via
  __base_aggressive__ and computes kappa from RECOVERY = PPL_base_aggressive - PPL_leave_G_high),
  magicquant/v2/search.py (thread probe_mode, cumulative report-fit handling),
  config.py + __main__.py (--probe-mode flag / MAGICQUANT_PROBE_MODE).
- CPU unit tests (no GPU): tests/test_v2_calibrate.py +5 — recovery math, embedding-rescue
  (single ranks K>>E i.e. the bug; cumulative raises kappa_E 20x+), censoring under cumulative,
  per-mode probe-config shapes, invalid-mode reject. Full suite: 620 passed, 6 skipped.
- Built --floor E=Q6_K config CPU-side from cached table (0.7446 GiB, E back to Q6_K).
- GPU (flock-gated, contended): measuring floored-E PPL+HellaSwag, and running full cumulative
  v2 run (reuses cached distortion table — mode-independent). Numbers -> docs/validation.md.
- MEASURED (folded into validation.md): --floor E=Q6_K @ matched 0.7446 GiB budget:
  wikitext PPL 19.5733 = +6.56% (down from unfloored +9.20%, competitive with v1 +6.02%) —
  CONFIRMS the embedding-crush diagnosis. HellaSwag@400 58.25% (edges v1's 57.00%, within
  eval noise; F16 ref 59.25%). So the floored per-tensor allocation is >= v1 on the downstream
  task despite marginally-worse PPL.
- PENDING post-cutoff GPU (contended, honestly marked in validation.md, no raw placeholders):
  cumulative-probe v2 matched-budget run (mechanism CPU-proven + floored-E-validated;
  measured end-to-end PPL deferred) and unfloored-v2 HellaSwag.
- NO git commit (coordinator commits after verifying).
