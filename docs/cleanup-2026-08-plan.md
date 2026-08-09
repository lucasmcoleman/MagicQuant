# MagicQuant Cleanup Plan — 2026-08-08

> **APPROVED 2026-08-08** (review-first flow): Bundles A, B, C, D in full; all 10 Bundle E fixes.
> Bundle F decisions: F1 archive the 4 superseded audit docs + WORKLOG to docs/history/ (AUDIT_2026-07-01 stays at root); F2 option (a) — delete quantize_model/tests/type-map, construction contract unchanged; F3 keep `supports()`, mark intentional; **F4 fix the tools/ packaging defect this pass** (move fit_noise_factors into magicquant/, exclude tools*); F5 leave qat/validate.py + comment; F6 remove the dead Makefile build target; F7 brief 0.3.0 CHANGELOG backfill; F8 delete all three heuristic duplicates; **F9 wire enable_iq into v2 this pass** (per verifier: must NOT reuse v1's unconditional LEGACY_Q4 drop — q4nx profile schemes survive).

**Baseline:** HEAD `06d4099`, clean tree. Tests: 903 passed / 20 skipped (`.venv`); QAT surface verifiable via `.venv-qat` (torch 2.12, baseline run in progress).
**Method:** 175-agent audit (7 subsystem lanes × dead-code/duplication/simplification + repo hygiene), every candidate adversarially verified by an independent skeptic with repo-wide reference searches, git-blame recency checks, and seed-pinned-fixture blast-radius checks. 101 confirmed safe, 29 need a decision, 11 refuted.
**Execution:** dedicated branch `cleanup/2026-08`, one atomic commit per unit below, CHANGELOG.md entry per commit, full test suite before/after every commit. Full verifier notes (load-bearing traps per item) are preserved and will be handed verbatim to each implementation agent.

**Hard constraint honored throughout:** no capability, behavior, or performance regression. Items that *do* change observable behavior (all of them bug-fixes toward documented intent) are quarantined in Bundle E as opt-in.

---

## Bundle A — Mechanical sweep (dead code, unused imports, provably-unreachable branches)

*Risk: low. All items verified zero-caller / zero-effect by independent skeptics (AST + repo-wide reference search incl. dynamic access, argparse dests, entry points). Verified against full suite.*

One commit per subsystem (~6 commits):

**A1. gguf/reader.py + tensor_groups.py** — delete `GGUF_TYPES` table, `find_tensors_by_group` (+ its local import), `get_group_info`, unused `Tuple` import; simplify `get_model_architecture`'s never-fired tensor-name-inference fallback to fall through to `"unknown"`.
**A2. gguf/writer.py** — delete `_GGML_TYPE_NAME` reverse table, dead `set_metadata`/`get_metadata` accessors; replace the 12-day-old-but-already-dead getattr duck-typing guards for `can_decode`/`read_tensor_raw` with direct calls (keeping the `raw is None` RuntimeError block); drop `import queue as _queue_mod` re-import; hoist the two local `import json` to module level; rename the second, unrelated `unknown_tensors` binding (all 5 references atomically).
**A3. orchestrator.py + evolution/** — delete the superseded standalone `main()`/`__main__` CLI (52 lines); drop unused `get_llamacpp_quant_type` import; delete `QUANT_TYPE_MAP` + `get_llamacpp_quant_type` from llamacpp.py (dead AND diverged from the registry); unused `Tuple`/`numpy` imports in predictor.py; dead `get_sensitivity_weights`/`update_sensitivity_weights`; survival.py's redundant local re-import; `self.history` + always-empty `get_discovered_configs` (keeping `tier_winners`/`get_best_config_per_tier`); probing.py's three zero-caller methods + `__main__` demo (keeping `recommend_protected_groups`); `_speed_metric` moved to `__init__` defaults (keeping the getattr fallbacks — they're load-bearing for bare-`__new__` tests); build `tiered_survivors` by deriving from `tiered` instead of building the same dict twice.
**A4. v2/** — delete `Unit.fixed` property, the provably-unreachable `rel` term in fit_kappa's censoring `max()`, the unreachable BF16 distortion branch; compute the 3-D stacked-MoE imatrix condition once.
**A5. quant/ + qat/_ggml_ref.py** — unused `Dict` import in converters.py; `is_moe_optimized` field + its two assignments (trap noted: no TODO text — a docstring test bans certain phrases); the 5 dead dequant rows in `_ggml_ref.py`; converters.py:98 `.flatten()` → `.ravel()` (verifier-corrected: halves the copy overhead per tensor; byte-identity empirically proven across 7 types × 6 layouts).
**A6. config.py + tools/ + naming.py** — dead `output_path` property; reselect_tiers.py's unreachable None-guard and triple-`min()` recompute; naming.py's genuinely dead `generate_config_for_quant`, `get_scheme_bits`, `__main__` demo. **Conflict resolved:** `GROUP_CODES`/`get_group_names` stay (they become the single home for cmd_analyze's duplicate table in C9) instead of being deleted per the competing dead-code finding.

Deliberately **kept**: tensor_groups.py's redundant `exp_probs_b` heuristic entry (9 days old, possibly deliberate belt-and-braces → Bundle F).

## Bundle B — Docs, comments, repo hygiene

*Risk: low; user-visible file moves/deletions are in Bundle F instead.*

**B1.** IQ4_NL header comment: qualify the claim its own file later disowns (comment-only; `noise_factor=3.8` untouched — fixture-pinned).
**B2.** Encoder docstrings: document `n_per_row` (conditionally-required, per verifier's corrected semantics), fix the imatrix-scope claim to "K-quants and the IQ family" (not "K-quants + IQ4_NL" — that drops 8 registered schemes), fix converters.py:9 module docstring too; CLAUDE.md's matching stale sentence.
**B3.** Q8_1 story told twice: keep the full narrative in ggml_facts.py, shrink ggml_binding's docstring to a pointer.
**B4.** README: fix converters.py architecture line (keep "single source of truth" phrasing — it's a CLAUDE.md invariant), add the 3 undocumented CLI commands (`imatrix`, `qat-merge`, `card` — wary of the two distinct `merge_qat_adapters` functions), refresh scheme table + architecture tree (curated, NOT a raw registry dump — ROCmFPX types need their fork-only caveat), fix the "Syntax check" mislabel.
**B5.** docs/qat.md: document the `qat-merge` CLI (flags copied verbatim from `__main__.py:1109-1129`).
**B6.** pyproject `[qat]` extra: drop never-imported `trl`/`datasets`, **add `accelerate>=1.1.0`** (actually required by `TrainingArguments`, currently only present transitively); fix train.py docstring + CLAUDE.md mentions.
**B7.** .gitignore: add `/o/` (anchored — unanchored `o/` would swallow any future `o` dir at any depth), `.ruff_cache/`, `.venv-qat/`.
**B8.** Makefile: add `.ruff_cache/` to `clean` (narrowed per verifier: NOT `output/` — it's 13 GB of real run artifacts, data-loss footgun; NOT `o/`).
**B9.** WORKLOG.md: one-line "session closed, see git log" header note (no blanket COMPLETE stamp — two cells are honestly marked never-measured).

## Bundle C — Single-home duplication folds

*Risk: medium — each has a verifier-documented trap list that the implementing agent receives verbatim. Each is one commit, test-gated.*

**C1. quant:** `_expected_size` ⇄ `ggml_tensor_data_size` → one function in ggml_facts.py (trap: call-time module-global table resolution must be preserved — tests monkeypatch the tables).
**C2. quant:** calibration.py `_load`/`_load_source` → extract shared `_read_json_dict_tolerant` (the variant that keeps the two deliberately-separate caches; ~10 tests monkeypatch `_CALIBRATION_PATH` at call time).
**C3. gguf/writer:** `_ftype_map` rebuilt from the already-imported `gguf.constants.LlamaFileType` (trap: key SET is load-bearing twice as a membership filter — do not "complete" it from the enum).
**C4. gguf/writer:** bad_tensors gate stops re-deriving Pass-1 scheme resolution — capture `_desired_ggml_name` at Pass 1 before the 5-stage mutation chain (chosen over the competing extract-a-function proposal: no NameError trap at line 951).
**C5. gguf/source:** `_resolve_effective_config` extracted, all 4 sites (sub-config-wins merge order proven load-bearing by `qwen3_5_text`); thread config into `_build_tokenizer_metadata` as an *optional* param (7 test callers pass only `tmp_path`).
**C6. gguf/source:** BF16/F16/F32 decode → the existing `_decode_st_bytes_to_f32`, with the verifier's mandatory amendments (gate on float names in `read_tensor_f32` so the dequant fall-through survives; keep the reshape at the LoRA site).
**C7. orchestrator:** one `_serialize_measurement` for the 4 hand-maintained field lists (traps: `path` inserted mid-order, two keys swapped between sites — key order preserved exactly; `tiered_survivors` vs `tiered` NEVER collapsed — external consumers read them with opposite precedence).
**C8. orchestrator:** canonical `config_key` → naming.py, all sites (persisted-interchange contract: pure code motion, zero format change).
**C9. cli:** cmd_analyze's verbatim copy of GROUP_CODES → `get_group_names()` + the one extra UNKNOWN entry.
**C10. llamacpp.py:** three tool-discovery loops → `_find_tool_in_dirs` (trap: dirs-outer/names-inner nesting order is load-bearing — legacy root binary wins over modern build/bin); shared `_effective_chunks` + the four copy-pasted subprocess-triage blocks (catch EXACTLY `CalledProcessError`/`TimeoutExpired` — an OSError test exists specifically to see one escape).
**C11. tools:** fit_noise_factors' hand-copied collapse-penalty constants → promoted constants on PredictiveScorer (verifier found a third copy in survival.py — cross-referenced with a comment; full unification is future work).
**C12. v2/calibrate:** the two ~60-line probe build/measure/retry/cleanup blocks → one helper (trap: the raise-immediately vs collect-and-defer asymmetry between base-aggressive and per-group probes is semantic — parameterized, not unified).
**C13. v2/interchange:** raw `print()` → structured logger like every sibling module, plus a small new test for the corrupt-file branch it lives in (currently zero coverage).

## Bundle D — God-function decompositions (pure code motion)

*Risk: highest of the behavior-preserving work — mitigated by: pure-move discipline (no tidying while moving), the verifier's must-preserve lists, and characterization tests where coverage is thin. One commit each; each independently revertable.*

**D1.** `ggml_binding.encode()` (162 lines, 4 jobs) — trap: the n_per_row-override-when-no-imatrix contract is documented API.
**D2.** `writer.create_hybrid_gguf` (~570 lines, 7 jobs) — trap: FIVE type-mutation stages in strict order (verifier found the fifth the proposal missed).
**D3.** `source._ensure_loaded` (~260 lines, 10 jobs) — trap: `self._loaded = True` is set at the TOP deliberately; do not move it.
**D4.** `source._build_tokenizer_metadata` (~230 lines) — trap: BOS double-write priority (config.json first, tokenizer_config overrides) verified by execution; preserve exactly.
**D5.** orchestrator: extract `_detect_search_groups` / `_build_predictor` shared by `run_measured_search`/`run_full_search` (byte-identical blocks, diff-verified); extract `_record_candidate_measurement` as a **pure move** (53 of its lines are a 9-day-old bugfix); function-local `open_model_source` import stays function-local (tests patch it there).
**D6.** v2 `run_budget_search` (~260 lines, 7 phases) — characterization test lands FIRST as its own commit, then the split.
**D7.** qat `run_qat` (309 lines, ~15 jobs) — verified via `.venv-qat`; ordering deps preserved (config_hash before checkpoint-identity check).
**D8.** `reselect_tiers.analyze()` split + `cmd_card` precedence rewrite (readability-only; semantics AST-verified identical) + `cmd_probe` hasattr simplification (keeping the truthiness guard — it prevents a real ZeroDivisionError on `--baseline-ppl 0`).

QAT-side folds (torch-verified, same commit series): `expert_cache_enabled/disabled` → one contextmanager factory; wrap.py's `_set_submodule` → expert_wrap's traversal primitives; `_bf16_roundtrip` → reuse `_encode_f32_to_bf16`; `hf_to_ggml_name` → additive `strict=` param on source.py's `_hf_name_to_gguf` (the "output"→"output.weight" self-map is the one way to silently break it — pinned first).

## Bundle E — Behavior-affecting fixes (opt-in; each changes observable behavior toward documented intent)

**E1.** `search` subcommand's argparse defaults silently defeat `MAGICQUANT_OUTPUT_DIR`/`MAGICQUANT_TARGET_BASE_QUANT` env vars → defaults to None like sibling parsers. *Changes: env vars start working for `search`.*
**E2.** `_write_pareto_report` is the ONE consumer of `self._measured` that doesn't filter `measurement_invalid` entries (both the frontier call and the logged table). *Changes: invalid measurements stop appearing in the Pareto report.*
**E3.** qat `_config_hash` ignores base-LoRA rank/alpha → resume could silently continue with wrong hyperparams. *Changes: existing checkpoints from differently-configured runs stop resuming.*
**E4.** train.py's most consequential fallback (auto-class-exhausted → toy model) is the one message skipping the file's print()-for-visibility convention. *Changes: one extra stdout line.*
**E5.** `cmd_search --algo v2` silently discards ~20 v1-only flags → one-line warning naming them. Special case: `--adapter` silently ignored means the v2 GGUF is built from the WRONG (un-merged) model — worth flagging loudly. *Changes: new warning output.*
**E6.** `cmd_imatrix` never plumbs `--llamacpp-path` (orchestrator fixed this same failure mode already) → add the flag. *Changes: new CLI flag.*
**E7.** v2 probe cache omits imatrix identity from its key (sibling distortion cache includes it) → add fingerprint. *Changes: existing v2_probes.json caches invalidate once.*
**E8.** `predictor_is_tracking` — the documented "would have caught the 2026-07 failure" guard is built, tested, and never called. Wire it end-of-run (per verifier: NOT per-round — sample count can't reach MIN_TAU_SAMPLES per round). *Changes: new diagnostic in results.*
**E9.** CI: install the `[qat]` extra in a matrix leg (today ~40 QAT tests NEVER run in CI); make ruff blocking (today `|| true` — and `tools/` is never linted at all). Known-open items M12/M17 from the live audit doc.
**E10.** `configure_logging`'s dead `verbose` param → delete (the safe branch of the verifier's two options).

## Bundle F — Needs your call (can't or shouldn't decide unilaterally)

**F1. Archive the four root audit docs + WORKLOG to `docs/history/`?** AUDIT_REPORT/LOGIC_AUDIT/AUDIT_2026-06-09/FIXPLAN are self-marked SUPERSEDED; AUDIT_2026-07-01 is live. Move the superseded four + WORKLOG, keep 07-01 at root? (8 relative links need fixing if docs/superpowers moves too.)
**F2. `LlamaCppTools.quantize_model` + `_find_quantize_tool`:** production-dead, but the completeness critic found `_find_llamacpp()` uses `which llama-quantize` as its discovery anchor — naive deletion breaks perplexity discovery. Options: (a) delete method, keep binary as anchor with comment; (b) full lazy-discovery refactor; (c) leave.
**F3. `handle.supports()`:** zero callers but name-checked by CLAUDE.md and docs/redesign.md as API. Keep-and-mark-intentional, or delete + update both docs?
**F4. `tools/` packaging defect:** installs as top-level `tools` package, but orchestrator does `from tools.fit_noise_factors import …` at runtime — the clean fix (move fit_noise_factors into the package) is a real refactor, not an exclude-line. Do it this pass or log as future work?
**F5. qat/validate.py `compare_perplexity`:** zero production callers but likely used out-of-repo for the docs/qat.md validation numbers. Wire as `qat-validate` CLI, move to tools/, or leave?
**F6. Makefile `build` target:** invokes undeclared `build` package (and a stale `./build/` dir shadows it anyway). Remove target, or wire it properly?
**F7. CHANGELOG.md backfill:** stuck at 0.2.0; package is 0.3.0. Backfill a 0.3.0 section from git history, or just start clean with the cleanup entries?
**F8.** `exp_probs_b` heuristic duplicate (9 days old): delete like its two siblings, or belt-and-braces?
**F9.** `--enable-iq` in v2's BudgetInfeasibleError message is a lie (v2 never reads it). Verifier: don't delete the message — thread `enable_iq` into v2 for real parity. Do the wiring this pass?

## Not this pass — logged as future work in CHANGELOG

Cross-cutting unifications the completeness critic surfaced (each is a design task, not a safe mechanical fold): atomic-publish implemented 7× across 4 subsystems (3-way diverged); two diverged dequant-symbol tables; safetensors shard discovery 2×; "load search_results.json tier" 3×; config-signature 6 sites. Plus: the tests/ tree itself (19.3k LOC) was never audited; `run_full_search` constructs its prober WITHOUT `parameter_counts` (real v1 behavioral divergence between measured/prediction paths — possible bug, needs its own investigation); latent v2 bug found during verification (`--target-profile q4nx` + `probe_scheme` interplay); v2's scheme filter missing v1's IMATRIX_DEPENDENT gate (compounds with E7's imatrix-default issue); SensitivityProber's structurally-unreachable `baseline_ppl_err`; remaining tail-discuss items.

## Refuted for the record (11)

Claims killed by verification, so nobody re-litigates them later — highlights: writer/reader `__main__` CLIs are NOT redundant (writer's exercises a path the real CLI can't); `_write_metadata_value`'s "duplication" is deliberately-diverged semantics; train.py's log+print pairs are 3-day-old deliberate hardening, not copy-paste; probing.py's 3-site strict/fallback repetition doesn't fold safely; cmd_card's settings-bypass claim overstated exclusivity; `search-parser` claim about `--enable-iq` deletion rested on a false premise.

## Verification protocol (every commit)

1. Full suite in `.venv` (903 passed / 20 skipped baseline) — plus `.venv-qat` suite for any commit touching `magicquant/qat/`. Characterized `.venv-qat` baseline: **1069 passed / 3 skipped / 3 pre-existing failures** (test_probe_resolution.py TestPredictorTracking ×3 — `.venv-qat` lacks scipy, so `predictor_rank_correlation` degrades to None; environment gap, not a code bug). Commits compare against this exact baseline; optionally `pip install scipy` into `.venv-qat` to green it.
2. Seed-pinned `test_refactor_regression.py` explicitly watched; any change requiring fixture regen is out of scope for this pass by definition.
3. For writer/encoder-adjacent commits: byte-identity spot-check (encode a reference tensor set before/after).
4. CHANGELOG.md entry lands in the same commit; discovered-not-fixed issues go to its Future Work section, not into scope.
