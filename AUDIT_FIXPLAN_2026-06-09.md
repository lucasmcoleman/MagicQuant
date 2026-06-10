# Trusted Fix-Plan — MagicQuant

Verified against current code by an independent pass (every audit finding re-confirmed or refuted with file:line evidence). This is the authoritative implementation plan.

## Findings

### [H1-moe-classify] HIGH — CONFIRMED
**Claim:** MoE up/gate expert tensors (ffn_up_exps, ffn_gate_exps) misclassify into dense-FFN group U instead of experts group X; ffn_down_exps -> X but the other two -> U.

**Evidence:** magicquant/gguf/tensor_groups.py:40-41 X list = [r'ffn.*expert', r'ffn_gate_up_exps', r'ffn_down_exps', r'block_sparse_moe\.(input|output)_linear']. U list line 47: [r'ffn_up', r'ffn_gate(?!_inp)', ...]. Empirically reproduced: `blk.0.ffn_up_exps.weight -> U`, `blk.0.ffn_gate_exps.weight -> U`, `blk.0.ffn_down_exps.weight -> X`, `blk.0.ffn_gate_up_exps.weight -> X`. r'ffn.*expert' requires literal 'expert' (GGUF uses '_exps'); ffn_up_exps/ffn_gate_exps are not listed, and U is iterated after X in the dict but matches r'ffn_up'/r'ffn_gate'. NOTE: ffn_down_exps reaches X only because the D pattern r'ffn_down(?!_exps)' explicitly excludes it AND X lists ffn_down_exps — so the X list IS partially reached, but up/gate are not.

**Fix:** In magicquant/gguf/tensor_groups.py GROUP_PATTERNS['X'] (line 40), replace the three FFN-expert patterns with a single robust one: add r'ffn_(up|gate|down)_exps' (keep r'ffn_gate_up_exps' and r'block_sparse_moe\.(input|output)_linear' for fused/grok variants). Final X list: [r'ffn_(up|gate|down)_exps', r'ffn_gate_up_exps', r'ffn.*expert', r'block_sparse_moe\.(input|output)_linear']. Because GROUP_PATTERNS is an ordered dict and X is iterated before U/D, the _exps tensors will now match X first. Do NOT remove the D-pattern negative-lookahead r'ffn_down(?!_exps)' — it must stay so dense ffn_down still maps to D.

**Test:** tests/test_tensor_groups.py: table-driven test asserting classify_tensor('blk.0.ffn_up_exps.weight')=='X', ffn_gate_exps=='X', ffn_down_exps=='X', ffn_gate_up_exps=='X', plus dense ffn_up.weight=='U', ffn_gate.weight=='U', ffn_down.weight=='D', ffn_gate_inp.weight=='R'.

**Risk:** Low. Only changes classification of *_exps tensors (which by definition only exist in MoE models) from U/X-mixed to uniformly X. Dense models have no *_exps tensors so behavior is unchanged. Verify r'ffn_gate_up_exps' still resolves (it matches r'ffn_(up|gate|down)_exps'? No — 'gate_up' is not 'gate'/'up'/'down' alone; keep the explicit fused pattern).


### [H2-search-ignores-xrs] HIGH — CONFIRMED
**Claim:** Evolutionary search ignores X/R/S groups: run_evolution is called without groups so it falls back to DEFAULT_GROUPS=['E','H','Q','K','O','U','D']; X/R/S tensors only ever get the base quant.

**Evidence:** magicquant/orchestrator.py:198 `round_configs = survivor.run_evolution(verbose=verbose)` and :505 `best_configs = survivor.run_evolution(verbose=verbose)` — neither passes groups=. magicquant/evolution/survival.py:46 DEFAULT_GROUPS = ['E','H','Q','K','O','U','D']; survival.py:100-101 `if groups is None: groups = self.DEFAULT_GROUPS`. The orchestrator already computes the full detected group list for probing (orchestrator.py:143-155, adds X/R/S) but discards it before calling run_evolution. So population/mutation/tournament never vary X/R/S.

**Fix:** Hoist the detected-group computation in run_measured_search and run_full_search into an instance attribute. In both methods, after building `groups` (orchestrator.py ~line 143-155 and ~471-483), store `self._search_groups = groups`. Then pass it: orchestrator.py:198 -> `survivor.run_evolution(groups=self._search_groups, verbose=verbose)` and :505 -> same. The EvolutionarySurvivor already threads `groups` through _initialize_population, _mutate_winners, _generate_random_config, _find_protector_target/_find_crusher_target, so no survival.py change is required for plumbing. Also extend survival.py _HIGH_SENSITIVITY/_LOW_SENSITIVITY: R is already in _HIGH_SENSITIVITY (line 65), X already in _LOW_SENSITIVITY (line 68); add S to a class — recommend leaving S out of both (treated as attention/moderate via the else branch in _generate_random_config, which is correct for SSM).

**Test:** Test that for an expert-bearing group list (e.g. groups=['E','H','U','D','X','R']), run_evolution returns configs where config['X'] takes more than one distinct scheme across the discovered set (i.e. X is actually varied), and that 'X' key is present in every returned config.

**Risk:** Medium. Expanding the searched group set widens the search space; population_size may need to remain adequate. Seeds in _initialize_population already use `{g: ... for g in groups}` so they handle X/R/S. The refactor-regression fixture (tests/fixtures/refactor_regression_seed42.json) pins a 7-group dense run via run_evolution(verbose=False) with default groups — it is unaffected because that test calls run_evolution directly without groups. Confirm the fixture test still passes.


### [M1-predictor-paramcounts] MEDIUM — CONFIRMED
**Claim:** Predictor never receives per-group parameter_counts; MoE size/speed predictions use the hardcoded dense _DEFAULT distribution with no X/R/S entries.

**Evidence:** magicquant/orchestrator.py:171-175 and :491-495 construct PredictiveScorer(sensitivity_weights=..., baseline_size_gb=..., baseline_tps=360) with NO parameter_counts arg. predictor.py:46 `self.parameter_counts = parameter_counts or {}`. predictor.py:144-153 _compute_param_dist: when parameter_counts empty, returns _DEFAULT={'E':.04,'H':.04,'Q':.12,'K':.12,'O':.06,'U':.31,'D':.31} and `{g:_DEFAULT.get(g,0.05) for g in groups}` so X/R/S get 0.05. predict_size (line 94) and predict_tps (line 114) early-return to _estimate_simple_* when parameter_counts is falsy. For MoE where experts are ~85% of params this ignores the bulk of the model; prediction-only mode (--rounds 0) relies entirely on this wrong distribution and is NOT residual-corrected (size has no residual loop).

**Fix:** In orchestrator._estimate_model_size (line 668-708) it already iterates src.get_all_tensors_info(); add per-group accumulation: instantiate a TensorGroupClassifier, and for each info compute n=prod(shape) and add to a dict param_counts[classifier.classify_tensor(info['name'])] += n. Return both the GB size and the param_counts dict (or store param_counts on self, e.g. self._param_counts). Then at orchestrator.py:171 and :491 pass parameter_counts=self._param_counts to PredictiveScorer. As a backstop, add X/R/S entries to predictor.py _DEFAULT (line 149-152), e.g. for a typical MoE shift U/D down and add 'X':0.55 — but the real fix is passing actual counts so the fallback is rarely hit.

**Test:** Unit test: PredictiveScorer with parameter_counts={'X':850_000_000,'U':50_000_000,...} predicts a smaller predict_size when X is set to MXFP4_MOE vs Q6_K, and that the difference scales with the X param share (not the 0.05 default).

**Risk:** Low-medium. Changing _estimate_model_size return type would touch its two callers; safer to store on self and keep the float return. Real param counts change predicted_size/tier classification for MoE — desired, but may shift which tier a config lands in; the measured loop self-corrects loss but not size, so verify tiers still populate.


### [M2-noise-uncalibrated] MEDIUM — CONFIRMED
**Claim:** noise_factor values for Q3_K/Q2_K are uncalibrated placeholders; the calibration bench was never run (calibration_results.json missing).

**Evidence:** magicquant/quant/schemes.py:164 Q3_K noise_factor=8.0 with comment `# placeholder; calibrated below`; :176 Q2_K noise_factor=15.0 `# placeholder; calibrated below`. Module docstring lines 11-14 still say 'PR1 will replace them with empirically-benched values from tools/calibrate_noise_factors.py.' `ls tools/calibration_results.json` -> No such file. tools/calibrate_noise_factors.py exists but was never run.

**Fix:** Run `python tools/calibrate_noise_factors.py` (requires llama.cpp + a calibration model; ~1.5-2 hr compute), commit tools/calibration_results.json, paste measured noise_factor values into schemes.py for ALL schemes (not just Q3_K/Q2_K — re-bench the full registry for internal consistency), and drop the `# placeholder` comments and the PR1-pending sentence in the docstring (lines 11-14, 156-157). If compute is unavailable, at minimum update the docstring to stop claiming PR1 will replace them and label all noise_factors as heuristic-until-calibrated.

**Test:** Test asserting every scheme in the registry has a noise_factor sourced from calibration_results.json (load the JSON, assert schemes.py value == json value within tolerance), and a guard test that fails if any scheme docstring/comment still contains 'placeholder'.

**Risk:** Low (data-only). Changing noise_factors shifts initial population bias and prediction-only loss ranking; the refactor-regression fixture pins behavior with the CURRENT values, so re-running calibration will require regenerating tests/fixtures/refactor_regression_seed42.json (the fixture is seed-pinned to the exact noise values). Document this coupling.


### [M3-writer-not-crashsafe] MEDIUM — CONFIRMED
**Claim:** Worker-thread exception leaves a partial/loadable GGUF on disk; no .partial + os.replace, finally only closes source.

**Evidence:** magicquant/gguf/writer.py:555-556 `finally: source.close()` — no unlink/os.replace. The file is opened directly at the final path: line 458 `with open(self.output_path, 'wb') as f:`. If _read_encode_worker raises (dtype guard line 207, shape/size error line 234, OOM), the main loop re-raises (line 513 `raise item`) but the partially-written .gguf with a valid header (magic+version+tensor-info already flushed by line 459-479) remains. This is the still-open April finding H-5.

**Fix:** In GGUFWriter.create_hybrid_gguf: write to a temp path. At line 454, compute `tmp_path = self.output_path + '.partial'` and open(tmp_path,'wb') instead of self.output_path (update lines 454, 458, and the stat/size reads at 548 to use tmp_path during the write). After `worker.join(timeout=5)` at line 545 succeeds AND no exception propagated, call `os.replace(tmp_path, self.output_path)`. Wrap the body in try/except that does `Path(tmp_path).unlink(missing_ok=True)` on any exception before re-raising. Also: worker.join(timeout=5) at line 545 can silently leave a still-running thread; check `worker.is_alive()` after join and raise if it didn't finish, so a hung encode doesn't produce a truncated-then-renamed file.

**Test:** Test that injects a source whose read_tensor_f32 raises on the 2nd tensor (or returns int dtype) and asserts create_hybrid_gguf raises AND no file exists at output_path afterward (and no .partial left behind).

**Risk:** Low. os.replace is atomic on same filesystem; ensure tmp is in the same dir (it is — same parent). The verbose ETA/size logging that reads Path(self.output_path).stat() at line 548 must read tmp_path. Probe path (probing.py) and orchestrator candidate builds rely on the returned path being the final path — unchanged since os.replace lands it there.


### [M4-imatrix-not-threaded] MEDIUM — CONFIRMED
**Claim:** imatrix is plumbed into the encoder leaf but never threaded through the writer; all K-quant/IQ encoding runs unweighted.

**Evidence:** converters.py:81-126 encode_to_ggml_bytes(weights, ggml_type_name, imatrix=None) accepts and forwards imatrix to ggml_encode; ggml_binding.encode (line 258) accepts it. But writer.py:212 calls `encode_to_ggml_bytes(f32, target)` with no imatrix. No magicquant/imatrix.py exists (not in file tree). PR4 unstarted. For imatrix-requiring IQ1/IQ2 this would produce unusable output (binding has requires_imatrix() check at line 313 but writer never consults it).

**Fix:** Implement PR4 per docs/superpowers/plans/2026-05-04-magicquant-pr4-imatrix-support.md: (1) add magicquant/imatrix.py that captures activation magnitudes (initially via subprocess `llama-imatrix`), returning Dict[tensor_name -> np.ndarray]; (2) thread Optional[Dict[str,np.ndarray]] through create_hybrid_gguf -> _read_encode_worker -> encode_to_ggml_bytes(f32, target, imatrix=imat.get(name)); (3) before encoding, if ggml_binding.get_handle().requires_imatrix(target) and no imatrix available, raise a clear error rather than silently producing garbage. Gate IQ1/IQ2 schemes behind imatrix availability. This is a prerequisite for PR3's IQ1/IQ2 parity tests to flip from xfail.

**Test:** Test that encode_to_ggml_bytes with a requires-imatrix type and imatrix=None raises a clear error; and a smoke test that imatrix capture returns arrays whose length matches each tensor's element count.

**Risk:** Medium. New subprocess dependency (llama-imatrix); large surface. Should land AFTER the classification/search fixes. Until implemented, the registry must not expose IQ1_S/IQ2_* as selectable schemes (it currently does not — registry only has 9 schemes, none imatrix-required), so the immediate risk is latent.


### [M5-dual-config] MEDIUM — CONFIRMED
**Claim:** Two parallel configuration systems; pydantic MagicQuantSettings only reaches --dry-run; defaults diverge; python-dotenv declared but never imported.

**Evidence:** magicquant/config.py MagicQuantSettings (env_prefix MAGICQUANT_, env_file .env) is instantiated ONLY in __main__.cmd_dry_run (line 293/300). cmd_search (line 130-166), cmd_generate (226), cmd_hybrid (168) read argparse directly. Defaults diverge: config.py:27-30 search_generations=30/population_size=80 vs __main__.py:428/434 --generations default 50, --population default 100. README.md:158-169 advertises MAGICQUANT_* env vars 'All settings can be provided via environment variables.' python-dotenv is a hard dep (pyproject.toml:17) but `grep dotenv` finds no import (pydantic-settings loads .env internally, so dotenv is genuinely unused).

**Fix:** Option A (preferred, honors README): route cmd_search/cmd_generate/cmd_hybrid through MagicQuantSettings — build settings = MagicQuantSettings() (picks up env/.env), then override with any explicitly-passed argparse values (use argparse defaults of None to detect 'not provided'), call settings.validate_paths(), and feed settings into the orchestrator. Reconcile defaults to a single source (pick 30/80 OR 50/100 — recommend matching config.py 30/80 and updating argparse defaults). Option B (minimal): scope README env-var section to the dry-run command only and drop python-dotenv from pyproject.toml:17. Either way, drop the unused python-dotenv dependency since pydantic-settings handles .env.

**Test:** Test that setting MAGICQUANT_SEARCH_GENERATIONS=7 in env is honored by cmd_search (assert the orchestrator receives 7), and that an explicit --generations 9 overrides the env.

**Risk:** Low. Routing real commands through settings changes default generations/population (50->30, 100->80) — document the behavior change. Detecting 'argparse not provided' requires defaults=None then fallback, a small refactor of the subparsers.


### [M6-dockerfiles] MEDIUM — CONFIRMED
**Claim:** Both Dockerfiles likely broken by the mandatory llama-cpp-python dependency (no compiler toolchain in the install layer).

**Evidence:** Root Dockerfile:4 ARG BASE_IMAGE=python:3.12-slim; builder stage runs `python -m build --wheel` (line 14-15) but the RUNTIME stage line 28-29 `pip install /tmp/*.whl` installs llama-cpp-python (hard dep pyproject.toml:21) into python:3.12-slim with no gcc/cmake — if no manylinux wheel matches the platform, pip compiles llama-cpp-python from sdist and fails. docker/Dockerfile:20 runtime stage is ubuntu:22.04 with only python3/python3-pip/libopenblas-dev (line 24-26), then line 34 `pip3 install -e .` pulls llama-cpp-python with no build-essential/cmake in that stage (build-essential exists only in the builder stage line 9-11).

**Fix:** Consolidate to one Dockerfile. For the install layer that runs pip install with llama-cpp-python: either (a) add `apt-get install -y build-essential cmake` (Debian/Ubuntu) before the pip install and accept compile time, or (b) pin a prebuilt CPU wheel via the official index, e.g. `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` so no compiler is needed. In docker/Dockerfile, since llama.cpp is already built in the builder stage and libggml is copied to /usr/local/lib (line 28-30, ldconfig line 30), set MAGICQUANT_LIBGGML_DIR=/usr/local/lib so ggml_binding discovers system libggml — then llama-cpp-python can be installed CPU-wheel-only or even made optional. Add a HEALTHCHECK-equivalent smoke `python -c 'from magicquant.quant.ggml_binding import get_handle; get_handle()'`.

**Test:** CI/smoke: `docker build` the consolidated image and run `docker run --rm IMAGE python -c 'import magicquant; from magicquant.quant.ggml_binding import get_handle; get_handle(); print(magicquant.__version__)'` — must exit 0.

**Risk:** Medium. Building llama-cpp-python from sdist in-image is slow and may still fail without the right headers; the prebuilt-wheel approach is more reliable but pins versions. Consolidating Dockerfiles touches Makefile docker-build target (root) and docker/docker-compose.yml (points at docker/Dockerfile line: dockerfile: docker/Dockerfile) — update both.


### [M7-requirements-stale] MEDIUM — CONFIRMED
**Claim:** requirements.txt is stale and install-breaking: lists only numpy and pyyaml, missing pydantic-settings/structlog/tenacity/llama-cpp-python.

**Evidence:** requirements.txt (full file) = `numpy>=1.21.0` and `pyyaml>=6.0`. pyproject.toml:12-22 real deps = numpy, pydantic-settings>=2.0.0, structlog>=24.0.0, tenacity>=8.2.0, python-dotenv>=1.0.0, llama-cpp-python>=0.3.0 (pyyaml is an OPTIONAL extra, pyproject.toml:25). Verified in this environment: `import structlog`/`import pydantic_settings` raise ModuleNotFoundError, so `pip install -r requirements.txt` yields an env where `magicquant.logging` can't import (test_refactor_regression already fails here on `import structlog`) and libggml can't bind. docker/Dockerfile:32 COPYs requirements.txt, compounding M6.

**Fix:** Delete requirements.txt (pyproject is the single source of truth; the standard install is `pip install -e .`), OR regenerate it from pyproject to exactly mirror dependencies. If docker/Dockerfile is kept and still COPYs requirements.txt, switch it to `pip install -e .` (it already does at line 34) and stop COPYing requirements.txt (line 32), or regenerate the file. Recommend deletion + ensure all docs say `pip install -e .`.

**Test:** Not a unit test; add a CI check that `pip install -e .` in a clean venv lets `import magicquant.logging`, `import magicquant.config`, and `from magicquant.quant.ggml_binding import GGML_TYPE_IDS` all succeed.

**Risk:** Very low. Deleting requirements.txt only affects anyone running `pip install -r requirements.txt`; the README/Makefile use `pip install -e .`.


### [M8-unknown-source-leaky] MEDIUM — CONFIRMED
**Claim:** Pre-quantized/UNKNOWN source handling is a leaky special case; UNKNOWN maps to ggml type 0 (F32) and read_tensor_f32 returns None -> zero-filled blob, can slip past bad_tensors.

**Evidence:** source.py:119-131 GGUFSource.get_source_type_name returns literal 'UNKNOWN' (line 123) or 'UNKNOWN(<id>)' (line 130) for missing/undecodable tensors; read_tensor_f32 returns None for quantized/unknown (line 156). writer.py:364-368: source_type_name = get_source_type_name; can_decode = type in (F32,F16,BF16); if not can_decode -> target_ggml_name = source_type_name; target_ggml_id = GGML_TYPE.get(source_type_name, 0) — 'UNKNOWN' is not in GGML_TYPE so maps to 0 (F32). worker (writer.py:215-216): f32 is None -> `blob = b'\x00'*expected` (zero-filled). bad_tensors check (writer.py:390-399) only flags a tensor when desired_ggml_name != source_type — an UNKNOWN tensor whose group's desired type happens to equal the (UNKNOWN) source_type slips past, producing a zero-filled F32 blob with no error.

**Fix:** Make the ModelSource contract explicit. In writer.py Pass 1 (after line 364): treat any source_type_name starting with 'UNKNOWN' as a hard error — append to bad_tensors unconditionally (don't gate on desired!=source). Tighten the GGML_TYPE.get fallback at writer.py:368 to raise instead of defaulting to 0 when source_type_name not in GGML_TYPE. Optionally add a `can_quantize()`/dtype property to ModelSource (source.py base class line ~72-78) and have read_tensor_f32 raise on undecodable rather than return None, so the zero-fill branch (worker line 215-216) becomes unreachable for real tensors. Consolidate the two validation passes (Pass-1 bad_tensors + worker dtype guard) into one contract check.

**Test:** Test with a stub source whose get_source_type_name returns 'UNKNOWN(99)' for a tensor: create_hybrid_gguf must raise ValueError naming the tensor, and must NOT write a zero-filled blob.

**Risk:** Medium. read_tensor_f32 returning None is currently also the signal used by the intentional zero-fill path; ensure no legitimate caller relies on None for a real tensor (probing/LoRA do not). Raising on UNKNOWN makes previously-silent corrupt runs fail loudly — desired, but could surface on edge models with exotic tensor types; provide a clear error message naming the tensor and its raw type id.


### [M9-no-hf-publish] MEDIUM — CONFIRMED
**Claim:** No HuggingFace publishing path / model-card generation despite the project being framed around publishing hybrid quants.

**Evidence:** README.md:9-14 frames the tool around producing models for HF collections ('Carbon Fiber Body, Ferrari Engine'). No publish/card command exists: __main__.py subparsers are only analyze/probe/search/hybrid/generate (lines 378-502). No model-card or huggingface_hub usage anywhere (grep finds none in magicquant/). search_results.json (orchestrator._save_results line 385-422) holds per-tier measured PPL/size/loss but is never turned into a card; provenance is lost on manual upload.

**Fix:** Add a `magicquant publish` (or `card`) subcommand in __main__.py that reads output/search_results.json + the generated GGUF metadata and emits a README.md model card: tier table (scheme map per group from magicquant.group_schemes, measured PPL/loss, size_gb), attribution to magiccodingman wiki, base model name. Optionally call huggingface_hub.upload_file/upload_folder behind an explicit --upload flag (add huggingface_hub as an optional extra in pyproject). The data already exists in search_results.json, so this is mostly templating.

**Test:** Test that given a sample search_results.json fixture, the card generator produces a markdown string containing each tier, its measured PPL, size, and the per-group scheme map.

**Risk:** Low (additive feature). Network/auth surface only when --upload is used; keep card generation purely local by default.


### [L1-stale-audit-docs] LOW — CONFIRMED
**Claim:** In-repo audit docs (AUDIT_REPORT.md 2026-04-03, LOGIC_AUDIT.md 2026-04-05) are substantially stale and reference deleted numpy K-quant encoders.

**Evidence:** AUDIT_REPORT.md header: Date 2026-04-03; LOGIC_AUDIT.md: Date 2026-04-05 — both predate the May refactor. converters.py is now 126 lines dispatching to libggml with NO numpy K-quant encoders (grep for _encode_q6_k/RMSE/_quantize_q4_k in converters.py returns nothing). The docs' CRITICAL/HIGH items (collapse penalty, three tier systems, K-quant clamping, dtype guard/timeout, file-size baseline) are all fixed at HEAD.

**Fix:** Add a header to both AUDIT_REPORT.md and LOGIC_AUDIT.md: 'SUPERSEDED — pre-libggml binding (April 2026). Audited commit predates the May 2026 ctypes refactor. See AUDIT_2026-06-09.md for current state.' Migrate the still-valid items (S/X/R search gap, GGUF-version reader gap, tied_word_embeddings default) into AUDIT_2026-06-09.md or an issue tracker, then optionally delete the April docs.

**Test:** N/A (documentation). Optionally a docs-lint check that flagged docs contain a SUPERSEDED banner.

**Risk:** None (docs-only). A maintainer trusting them today would chase phantom bugs.


### [L2-version-drift] LOW — CONFIRMED
**Claim:** Version string stuck at 0.1.0 while CHANGELOG documents 0.2.0.

**Evidence:** magicquant/__init__.py:8 `__version__ = "0.1.0"`; pyproject.toml:7 `version = "0.1.0"`; CHANGELOG.md:3 `## [0.2.0] - 2026-04-03`. Root Dockerfile HEALTHCHECK line 34 reports __version__, so 'which build' returns 0.1.0.

**Fix:** Bump magicquant/__init__.py:8 and pyproject.toml:7 to 0.2.0. Better: single-source the version — set it in pyproject.toml only and derive __version__ via importlib.metadata.version('magicquant') in __init__.py (with a try/except fallback for editable/uninstalled runs). Note CLAUDE.md / CHANGELOG describe 0.2.0 as 'Production Hardening' which predates the May libggml refactor, so consider 0.3.0 to reflect the encoder rewrite.

**Test:** Test asserting magicquant.__version__ == importlib.metadata.version('magicquant') (when installed), and that it is not '0.1.0'.

**Risk:** None.


### [L3-numpy-metadata-string] LOW — CONFIRMED
**Claim:** numpy integer/float metadata values silently serialized as strings (np.int64/np.float32 fail isinstance int/float -> str()).

**Evidence:** writer.py:124-177 _write_metadata_value uses isinstance(value, int)/isinstance(value, float). np.int64/np.float32 are not Python int/float, so they fall through to the final else (line 175-177) and write a STRING tag. config.json values typically arrive as Python ints (low hit rate) but SafetensorsSource/_build_gguf_metadata could pass numpy scalars for e.g. *.attention.head_count, which llama.cpp expects as integer.

**Fix:** At the top of _write_metadata_value (writer.py:124), normalize numpy scalars to natives: `if isinstance(value, np.generic): value = value.item()`. numpy is already imported (writer.py:26). This converts np.int64->int, np.float32->float, np.bool_->bool before the isinstance ladder.

**Test:** Test that _write_metadata_value with np.int64(32) writes a UINT32 tag (not STRING), and np.float32(1.5) writes FLOAT32.

**Risk:** Very low. Pure widening; np.ndarray (not np.generic) still handled by the list/tuple branch only if converted — confirm metadata never passes a 0-d ndarray (add `if isinstance(value, np.ndarray) and value.ndim==0: value=value.item()` for safety).


### [L4-int-array-int32] LOW — CONFIRMED
**Claim:** Integer metadata arrays hard-coded to INT32; values >= 2^31 raise struct.error.

**Evidence:** writer.py:161-165: for an int-first array it writes _GGUF_TYPE_INT32 and packs each item with `struct.pack('<i', int(item))`. Any value >= 2^31 (or < -2^31) raises struct.error. Latent today (vocab sizes < 2^31) but a hard failure rather than silent corruption.

**Fix:** In writer.py:161-165, before choosing the tag, scan the array: `if any(v < -2**31 or v > 2**31-1 for v in value): use INT64 (tag 11, '<q')` else if all >= 0 and max <= 2^32-1 consider UINT32, else INT32. Simplest robust choice: if all non-negative and max <= 2^32-1 -> UINT32 ('<I'); elif fits int32 -> INT32; else INT64 ('<q').

**Test:** Test that an int array containing 2**31 writes an INT64/UINT32 array without raising struct.error.

**Risk:** Very low. Changing the array element type tag must stay consistent across all items (it does — one tag per array).


### [L5-bf16-downgrade-silent] LOW — CONFIRMED
**Claim:** BF16 group request silently downgraded to F16 for tensor data.

**Evidence:** writer.py:342-344: `if target_ggml_name == 'BF16': target_ggml_name='F16'; target_ggml_id=GGML_TYPE['F16']` — unconditional, no log even when verbose. The BF16 encoder (converters._encode_f32_to_bf16, round-to-nearest-even) is therefore unreachable for tensor data. README.md:40-41,54 and CLAUDE.md advertise BF16 brain layers; the bytes written are F16 (5-bit exponent vs 8-bit; out-of-F16 values become Inf/0).

**Fix:** writer.py:342-344: emit a one-time warning (logger.warning, dedup with a flag on self) 'BF16-designated group written as F16 (llama.cpp BF16 compute-graph limitation)'. Update README.md (lines ~40-54) and CLAUDE.md Critical Invariants to state that BF16-designated groups are stored as F16 on disk. This is a deliberate compatibility tradeoff per the inline comment (writer.py:339-341); make it non-silent.

**Test:** Test that requesting group scheme 'BF16' produces tensors with ggml_type==F16 (id 1) in the output GGUF, and that a warning is logged once.

**Risk:** None (logging + docs only). Do not change the downgrade itself — it is intentional for llama.cpp compatibility.


### [L6-floor-not-enforced] LOW — CONFIRMED
**Claim:** Sensitive-group quantization floor (Q8_0) defined but never enforced during random config generation; sensitive groups can be assigned below floor.

**Evidence:** schemes.py:200-203 _GROUP_CLASS_FLOORS = {'sensitive':'Q8_0','robust':'Q4_K_M'}; get_floor_for_group_class exists. survival.py:73-76 _min_scheme_for_class reads it, but its only consumer is _find_crusher_target (survival.py:381, robust floor). _generate_random_config (survival.py:224-264) assigns brain groups via _BRAIN_CLASS_WEIGHTS which includes mxfp4:0.05 and k_quant:0.30 (Q2_K/Q3_K) — so E/H/O/R can be sampled down to MXFP4_MOE/Q2_K. The Protector only upgrades; nothing clamps a sub-floor sensitive pick out of the final population.

**Fix:** In survival.py _generate_random_config (after picking config[g], ~line 263), clamp sensitive-group picks to the sensitive floor: if g in self._HIGH_SENSITIVITY and get_scheme_by_name(config[g]).bits_per_weight < get_scheme_by_name(get_floor_for_group_class('sensitive')).bits_per_weight: config[g] = floor scheme. Also filter sub-floor winners before returning best_configs in run_evolution (line 140), or in tournament selection, so a sub-floor sensitive config never becomes a survivor. Keep the existing crusher robust-floor logic.

**Test:** Property test: over N _generate_random_config calls, assert no E/H/O/R group is ever assigned a scheme with bpw below Q8_0 (8.5).

**Risk:** Low-medium. Clamping narrows the brain search space (intended). Removing mxfp4/low-k from _BRAIN_CLASS_WEIGHTS would be cleaner but the clamp is safer and explicit. Ensure seeds (e.g. mxfp4_max sets O via base but E/H are BF16) are not accidentally clamped in a way that breaks the refactor-regression fixture — that fixture uses default groups and seeded configs; verify it still matches.


### [L7-no-early-stop] LOW — CONFIRMED
**Claim:** Search has no convergence/early-stopping; always runs full generation budget.

**Evidence:** survival.py:110-141 run_evolution loops `for generation in range(self.max_generations)` with no break on plateau. max_generations default 50 (CLI __main__.py:428) / 30 (orchestrator). In measured mode each round re-runs the full search.

**Fix:** Add an early-stop: track best composite_score across generations; if it doesn't improve by > epsilon_improve for `patience` consecutive generations, break. Add a --patience CLI flag (__main__.py search subparser) and a patience param to run_evolution. Default patience high enough to preserve current behavior unless opted in, or set a sensible default (e.g. 8) and regenerate the refactor fixture.

**Test:** Test that with patience=2 and a predictor returning constant scores, run_evolution stops before max_generations (assert fewer prediction calls than max_generations*population).

**Risk:** Low. Changing iteration count alters the discovered-config sequence -> breaks tests/fixtures/refactor_regression_seed42.json if patience triggers within 3 gens; keep the fixture run at a patience that never triggers (the fixture uses generations=3, population=20).


### [L8-tier-circular-import] LOW — CONFIRMED
**Claim:** Tier classification unified via an inverted dependency: evolution modules do function-local imports of MagicQuantOrchestrator to dodge a circular import.

**Evidence:** survival.py:280 `from magicquant.orchestrator import MagicQuantOrchestrator` inside _classify_into_tiers; predictor.py:301 same inside TierClassifier.classify_by_size. The pure-arithmetic _classify_tier (orchestrator.py:636-653) and its boundary constants live on the highest-level coordinator, forcing leaf modules to import upward.

**Fix:** Create magicquant/quant/tiers.py (a leaf module) with `classify_tier(size_gb, baseline_gb) -> str` and the boundary constants. Make orchestrator._classify_tier delegate to it (or re-export). Replace the function-local imports in survival.py:280 and predictor.py:301 with `from magicquant.quant.tiers import classify_tier`. Removes the circular dependency.

**Test:** Test that magicquant.quant.tiers.classify_tier and MagicQuantOrchestrator._classify_tier return identical labels across a grid of ratios spanning all boundaries.

**Risk:** Low. Behavior must stay byte-identical (same boundaries) so the refactor fixture and _pick_best_per_tier results are unchanged — copy the exact boundary logic.


### [L9-dead-scaffolding] LOW — CONFIRMED
**Claim:** Dead/scaffolding abstractions and a no-op size estimator (TierClassifier, HybridValidator, parse_name, calculate_expected_size).

**Evidence:** predictor.py:292-303 TierClassifier and survival.py:407-421 HybridValidator are imported nowhere (grep). naming.py:89-144 parse_name parses an old hyphen-block format that generate_name (line 51-86) no longer produces (generate_name now emits e.g. 'Model-Q5_K_M.gguf', not the 'EH-B16' blocks parse_name expects) — so parse_name is no longer the inverse of generate_name. naming.py:206-228 calculate_expected_size returns `total_params * (base_quant_bits/16.0)` where total_params = base_model_size*(16/base_quant_bits) — algebraically returns base_model_size unchanged and ignores `overrides` entirely.

**Fix:** Delete TierClassifier (predictor.py:292-303), HybridValidator (survival.py:407-421), and parse_name/normalize_scheme (naming.py:89-162) if truly unused (confirm no external callers). For calculate_expected_size (naming.py:206-228): either delete it or reimplement via predictor.predict_size (build a PredictiveScorer with the parameter_counts and call predict_size). If parse_name must stay for round-tripping, rewrite it to parse the current 'Name-Q5_K_M' format.

**Test:** If calculate_expected_size is reimplemented: test that for a uniform-MXFP4 config it returns ~baseline*4.25/16, not the input unchanged.

**Risk:** Low. Confirm nothing imports these (the audit and grep say nothing does). Deleting reduces misleading surface a maintainer could wire in.


### [L10-filetype-enum] LOW — CONFIRMED
**Claim:** general.file_type enum values partly wrong (cosmetic): Q4_K->12, IQ4_NL->20, Q5_K->16 don't match LLAMA_FTYPE.

**Evidence:** writer.py:423-426 _ftype_map: 'Q5_K':16, 'Q4_K':12, 'IQ4_NL':20. LLAMA_FTYPE: Q5_K_S=16 (not generic Q5_K), Q4_K_S=14/Q4_K_M=15 (12 is MOSTLY_Q4_1_SOME_F16-era / not Q4_K), IQ4_NL=25 (20 is MOSTLY_IQ2_XS). These set only the human-readable quant badge; each tensor carries its own ggml_type so inference is unaffected.

**Fix:** Align _ftype_map (writer.py:423-426) to current LLAMA_FTYPE values: Q4_K_M->15, Q5_K_M->17, Q6_K->18, Q8_0->7, IQ4_NL->25, Q3_K_M->12, Q2_K->10, F16->1, F32->0, BF16->32; for ambiguous generic 'Q4_K'/'Q5_K' pick the _M variant. Or add a comment documenting it as best-effort cosmetic and leave inference-irrelevant.

**Test:** Test that for a config dominated by Q6_K, the written general.file_type == 18 (LLAMA_FTYPE Q6_K).

**Risk:** None (cosmetic). Only affects displayed quant label.


### [L11-mxfp4-comment] LOW — CONFIRMED
**Claim:** Stale comment claims MXFP4 is non-native llama.cpp.

**Evidence:** magicquant/utils/llamacpp.py:319 `"MXFP4_MOE": "MXFP4",  # MagicQuant custom type (not native llama.cpp)` — contradicts the project invariant (MXFP4 = native ggml type 39, asserted across writer.py:69, ggml_binding.py:54, schemes.py:127-132).

**Fix:** Change the comment at llamacpp.py:319 to `# native ggml type 39 (GGML_TYPE_MXFP4)`.

**Test:** N/A (comment). Optionally a grep-based docs test that no source comment says 'not native llama.cpp' near MXFP4.

**Risk:** None (comment-only).


### [L12-adapter-optional-type] LOW — CONFIRMED
**Claim:** adapter_path: str = None instead of Optional[str].

**Evidence:** writer.py:268 `adapter_path: str = None` in GGUFWriter.create_hybrid_gguf signature; writer.py:568 `adapter_path: str = None` in module-level create_hybrid_gguf. Optional is imported (writer.py:18 `from typing import ... Optional`). Type-checker noise; no runtime effect.

**Fix:** Change both signatures (writer.py:268 and :568) to `adapter_path: Optional[str] = None`.

**Test:** N/A (type annotation). A mypy/pyright run in CI would catch it.

**Risk:** None.


### [L13-broad-except-no-exc-info] LOW — CONFIRMED
**Claim:** Broad exception handlers drop tracebacks; the probe path falls back to a fabricated heuristic PPL on any exception, masking real writer bugs.

**Evidence:** orchestrator.py:70 `except Exception as exc: log.warning(..., error=str(exc))` (no exc_info); :367 `log.error('Build failed', ..., error=str(exc))`; :548 `log.error('Generation failed', ..., error=str(exc))`. probing.py:230-233 `except Exception as exc: ... return self._heuristic_probe(...)` — ANY exception in the real probe (including a genuine create_hybrid_gguf bug) silently falls back to a fabricated heuristic PPL, biasing sensitivity weights without surfacing the error.

**Fix:** Add exc_info=exc to the structlog error/warning calls (orchestrator.py:70,367,548). In probing._real_probe (line 230-233), narrow the except to the expected failure classes (e.g. subprocess/measurement errors) OR at minimum log the full traceback (logger with exc_info) before falling back, and consider re-raising on writer-level exceptions (ValueError from the dtype/UNKNOWN guards) so real build bugs aren't hidden behind heuristic PPLs.

**Test:** Test that when create_hybrid_gguf raises a ValueError inside _real_probe, the error is logged with a traceback (and, after narrowing, propagates rather than silently returning a heuristic value).

**Risk:** Low. Narrowing the probe except could let a previously-swallowed exception propagate and abort a probe run — that is the desired behavior (fail loud on real bugs), but verify it doesn't break legitimate fallback-to-heuristic when llama.cpp is merely unavailable (that path is gated earlier at probing.py:160-167, not in the try).


### [L14-dup-config-onm] LOW — CONFIRMED
**Claim:** O(n*m) duplicate-config check in the evolution inner loop re-serializes every existing best_config on each membership test.

**Evidence:** survival.py:120-121: `config_key = str(sorted(winner['config'].items())); if config_key not in [str(sorted(c['config'].items())) for c in best_configs]:` — rebuilds the full list of serialized keys for every winner, every generation. Negligible at default scale but quadratic.

**Fix:** Maintain a `seen_keys: set` alongside best_configs in run_evolution (survival.py ~line 108). On each winner compute config_key once; `if config_key not in seen_keys: seen_keys.add(config_key); best_configs.append(winner)`. O(1) membership.

**Test:** Covered by existing tests/test_refactor_regression.py (output sequence must be unchanged); add a micro-benchmark assertion if desired.

**Risk:** None — must produce identical best_configs ordering to keep the refactor-regression fixture passing (it does: same insertion order, just faster membership).


### [L15-security-hardening] LOW — CONFIRMED
**Claim:** Light security hardening items (all local-access, low): unbounded ctypes out_size; unvalidated safetensors byte_offset/byte_length; untrusted base_model_name_or_path from adapter_config.json.

**Evidence:** ggml_binding.py:283-288 computes out_size = _expected_size(...) and allocates (c_uint8*out_size)() with no upper bound before the ctypes call. source.py:899-900 _read_adapter_tensor seeks data_start+byte_offset and reads byte_length with no validation against file size before the slice/reshape (line 904-908). source.py:974 `base_model = cfg.get('base_model_name_or_path', '')` from adapter_config.json is used to locate the base model with no trust check. None remotely exploitable (presupposes local control of env/lib dirs or a malicious local model/adapter).

**Fix:** ggml_binding: bound n_per_row/out_size (e.g. assert out_size < a few GB) before allocation; document that MAGICQUANT_LIBGGML_DIR must point to a trusted directory. source.py:899-908: validate byte_offset>=0 and data_start+byte_offset+byte_length <= file size before reading; raise on mismatch. source.py:974: treat base_model_name_or_path as untrusted — require an explicit --base-model override or restrict resolution to a configured models root rather than following the path verbatim.

**Test:** Test that _read_adapter_tensor raises when byte_length would read past EOF, and that ggml_encode raises (not segfaults) for an absurd element count.

**Risk:** Low. Adding bounds checks could reject legitimate-but-large tensors if thresholds are too tight; size the bound generously (e.g. 16 GB).


### [RIGHT-encoder-parity] LOW — CONFIRMED
**Claim:** Quantized encoding now delegates to libggml via ctypes; output byte-identical to llama-quantize (audit's 'What is Right').

**Evidence:** converters.py:106-126 encode_to_ggml_bytes dispatches all quantized types to ggml_encode (ggml_binding.py:331-340 -> ggml_quantize_chunk). No numpy K-quant encoders remain (grep). ggml_binding._verify_type_ids (line 237-256) cross-checks type sizes at startup; encode asserts written==expected (line 305-310). tests/integration/test_encoder_parity.py byte-compares vs llama-quantize for 9 schemes. This corroborates that the April audit's encoder CRITICALs are already_fixed.

**Fix:** No fix needed. Preserve this boundary; ensure CLAUDE.md is updated to describe it (see additional_issues — CLAUDE.md still claims numpy encoders).

**Test:** Keep test_encoder_parity.py running in CI with a skip-when-llama-quantize-absent guard (see additional_issues A1).

**Risk:** N/A.


## Additional issues the audit missed

- CLAUDE.md is stale and misleading (HIGH-equivalent, audit listed it but I confirm in detail). CLAUDE.md line 'No test suite exists yet' is false (tests/ has test_quantization_guards.py with 18 passing tests, test_refactor_regression.py, and tests/integration/test_encoder_parity.py). The 'Quantization (converters.py)' section says 'Encoders are vectorized with numpy. K-quant encoders (Q6_K, Q5_K, Q4_K) use RMSE-optimized scale selection (7 candidates per sub-block). The MXFP4 encoder matches llama.cpp...' — all describe DELETED numpy code; converters.py now dispatches to ggml_binding (libggml ctypes). Known Limitations says 'K-quant encoders use simple min/max with RMSE optimization... Quality gap is ~10-27% MSE vs llama.cpp native' — the MSE gap is now 0 (byte-identical). Fix: rewrite the Commands section ('No test suite exists yet' -> 'Run: pip install -e ".[dev]" && pytest'), the Quantization section (describe ggml_binding.py, _verify_type_ids drift check, byte-parity), and delete the MSE-gap Known-Limitation. This is the agent/maintainer contract; it currently describes an architecture deleted ~a month ago.
- IQ4_XS block-size/type-size table in converters.py is WRONG and inconsistent with ggml_binding.py (correctness trap for PR3). magicquant/quant/converters.py:34 sets GGML_BLOCK_SIZE['IQ4_XS']=32 and :47 GGML_TYPE_SIZE['IQ4_XS']=18 (those are IQ4_NL's values). The correct ggml values (and what ggml_binding.py:164,176 already have) are block=256, type_size=136. ggml_tensor_data_size('IQ4_XS', n) in converters.py therefore returns a size 256/32 * 18/136 wrong, which the writer uses for Pass-1 offset math (writer.py:370). IQ4_XS is not yet a registered scheme (registry has 9 schemes, none IQ4_XS), so this is latent — but PR3 plans to register IQ-quants and the writer's offsets would then be corrupt while ggml_binding's encode size would be correct, causing a 'wrote X bytes, expected Y' RuntimeError or misaligned GGUF. Fix BEFORE PR3: set converters.py IQ4_XS to block=256, size=136 to match ggml_binding.py. Recommend deriving converters.py's tables from ggml_binding._GGML_BLOCK_SIZE/_GGML_TYPE_SIZE (single source of truth) to prevent future drift.
- Encoder-parity test has an undeclared hard dependency on the `gguf` package and fails collection (not skip) when absent. tests/integration/test_encoder_parity.py:68,91 `import gguf` inside _write_f32_gguf/_read_first_tensor_bytes. `gguf` is not in pyproject.toml dev deps. In this environment `pytest tests/` raises ModuleNotFoundError: No module named 'gguf' and FAILS the test (the audit's 'skip-when-llama-quantize-absent' guard exists for the binary but not for the gguf import). Fix: add `gguf` to pyproject.toml [project.optional-dependencies] dev, OR wrap the import in a module-level `gguf = pytest.importorskip('gguf')` so the suite skips gracefully when gguf is missing. Currently a clean `pip install -e ".[dev]"` + pytest fails.
- Test infrastructure cannot run in a clean checkout because core runtime deps (structlog, pydantic_settings, tenacity, llama_cpp) are not installed in this environment and requirements.txt omits them. `pytest tests/test_refactor_regression.py` fails at import time with `ModuleNotFoundError: No module named 'structlog'` (via magicquant.orchestrator -> magicquant.logging). This blocks the refactor-regression fixture test and any orchestrator-touching test. Root cause overlaps M7 (stale requirements.txt) — but note specifically that the test suite is NOT runnable as-is here; the implementing engineer must `pip install -e ".[dev]"` (which pulls pyproject deps) first. Recommend a tox/CI config and a Makefile `test` target that installs `.[dev]` before pytest.
- Makefile lint target is a no-op stub (py_compile only) and format prints 'No formatter configured'; there is no CI, no linter, no lockfile. Makefile lines 8-19. The encoder-parity test (core invariant guard) is never run automatically. Fix: add ruff or black to dev extras, make `lint` actually run it, and add a minimal CI (GitHub Actions) that runs `pip install -e ".[dev]"` then the dtype-guard + refactor-regression tests on every push, and the encoder-parity tests with a skip-when-llama-quantize/gguf-absent guard. (The audit mentioned 'no CI' as a theme; I confirm the lint/format targets are stubs.)
- naming.py generate_name only expands a hardcoded tier-suffix table (_TIER_TO_HF_LABEL, naming.py:41-48) and silently no-ops for unknown suffixes. Tiers Q2 and IQ4 map (Q2->no entry! _TIER_TO_HF_LABEL has Q3/Q4/Q5/Q6/Q8/IQ4 but NO 'Q2'), so a model named 'Model-Q2' is written as 'Model-Q2.gguf' with no HF-recognized quant badge, and orchestrator.generate_tiered_models defaults to tiers including 'Q2' (orchestrator.py:570) and main() requests tiers=['Q2','Q4','Q5','Q6','Q8'] (orchestrator.py:756). Confirmed: _TIER_TO_HF_LABEL has no 'Q2' key. Fix: add 'Q2':'Q2_K' (and 'Q3':'Q3_K_M' already present) to _TIER_TO_HF_LABEL so the Q2 tier filename gets a recognized badge once PR3 makes the Q2 tier reachable.
- orchestrator.generate_tiered_models default tiers (line 570 ['Q8','Q6','Q5','Q4','Q2']) and main() (line 756 ['Q2','Q4','Q5','Q6','Q8']) request a Q2 tier that NO registered scheme can satisfy (lowest scheme Q2_K bpw=2.625 -> ratio 0.164, just OUTSIDE the Q2 band <=0.16 per orchestrator._classify_tier line 651 and the spec docs:337). So generate_tiered_models logs 'No config for tier, skipping' for Q2 every run. This is the concrete manifestation of the audit's 'Roadmap stalled at PR1 / Q2 tier unreachable' finding — confirmed at the call sites. Until PR3 lands sub-Q2 IQ-quants, either drop Q2 from the default tier lists or document that Q2 requires PR3.

## Ordered implementation plan

1. 0. PREP: In a clean venv run `pip install -e ".[dev]"` (plus `pip install gguf`) so the test suite is runnable; confirm `pytest tests/test_quantization_guards.py` passes (18 tests) and capture the current refactor-regression behavior. This unblocks all subsequent TDD.
2. 1. Extract testable units to break circular import FIRST (L8): create magicquant/quant/tiers.py with classify_tier + boundary constants; have orchestrator._classify_tier delegate; replace function-local orchestrator imports in survival.py:280 and predictor.py:301. Run tests — output must be unchanged.
3. 2. Add table-driven tensor-group classification test (currently MISSING) capturing CURRENT behavior, then fix the MoE classifier (H1): update tensor_groups.py:40 X list to r'ffn_(up|gate|down)_exps' (+ keep fused/grok patterns). Re-run the new test asserting ffn_{up,gate,down}_exps -> X. This MUST precede wiring classification into search.
4. 3. Fix the IQ4_XS table discrepancy (additional issue) in converters.py:34,47 to block=256/size=136 (match ggml_binding.py); ideally re-derive converters tables from ggml_binding's single source of truth. Latent but must be correct before PR3 registers IQ-quants.
5. 4. Thread per-group parameter_counts into the predictor (M1): accumulate counts in orchestrator._estimate_model_size (store on self), pass to PredictiveScorer at orchestrator.py:171 and :491; add X/R/S backstop entries to predictor _DEFAULT. Add the predict_size param-share test.
6. 5. Wire X/R/S into the search (H2): store detected groups on self in both search methods, pass groups=self._search_groups to run_evolution at orchestrator.py:198 and :505. DEPENDS on step 2 (classification correct) and benefits from step 4 (param counts). Add the 'X is varied' test. Verify refactor-regression fixture still passes (it calls run_evolution directly without groups).
7. 6. Enforce sensitive-group floor in random config generation (L6) and add early-stop scaffolding decision (L7) — note both may require regenerating tests/fixtures/refactor_regression_seed42.json; do floor-clamp now, gate early-stop behind a default-off/never-trigger patience to avoid fixture churn until step 12.
8. 7. Make the writer crash-safe (M3): write to output_path+'.partial', os.replace after worker.join succeeds, unlink .partial on exception, check worker.is_alive(). Add the crash-safety test (inject a raising source).
9. 8. Harden source/UNKNOWN handling (M8): treat UNKNOWN as a hard error in writer Pass 1; make GGML_TYPE.get not default to F32(0); optionally raise in read_tensor_f32. Add the UNKNOWN-tensor test. (Independent of writer crash-safety but touches the same Pass-1/worker code, so sequence after step 7.)
10. 9. Fix metadata serialization bugs (L3 np.generic normalize, L4 int-array width) and the file_type enum (L10) in writer.py; fix BF16->F16 silent downgrade to log once (L5) and document in README/CLAUDE. Add the metadata-type tests.
11. 10. Fix LoRA name-mapping arch arg + shape guard (L9-adjacent / from audit): source.py:893 pass arch=, source.py:939 add base_f32.size==delta.size check with a ValueError naming the tensor.
12. 11. Security hardening (L15): bound out_size in ggml_binding; validate safetensors byte_offset/byte_length vs file size; treat base_model_name_or_path as untrusted. Add the EOF/oversize tests.
13. 12. Calibrate noise factors (M2): run tools/calibrate_noise_factors.py, commit calibration_results.json, paste values into schemes.py, drop placeholders/PR1-pending docstring. THEN regenerate the refactor-regression fixture once (it is seed-pinned to noise values) and re-pin; also finalize early-stop default from step 6.
14. 13. Land PR3 (IQ-quants + lower robust floor) per docs/.../pr3-iq-quants.md so Q2/Q3 tiers fill — DEPENDS on step 3 (IQ4_XS tables correct) and step 12. Then PR4 (imatrix capture/threading, M4) so IQ1/IQ2 are usable and PR3 IQ parity tests flip from xfail.
15. 14. Reconcile config systems (M5): route cmd_search/generate/hybrid through MagicQuantSettings (argparse overrides), unify defaults (30/80 vs 50/100), drop python-dotenv. Add env-honored test.
16. 15. Fix install/ops drift: delete or regenerate requirements.txt (M7); repair/consolidate Dockerfiles for llama-cpp-python build (M6); bump __version__ + pyproject to 0.2.0 (or 0.3.0) and single-source via importlib.metadata (L2).
17. 16. Sync docs to HEAD: rewrite CLAUDE.md converters/Known-Limitations/'no test suite' sections (additional issue); mark AUDIT_REPORT.md/LOGIC_AUDIT.md SUPERSEDED (L1); fix README BF16/env-var claims; fix llamacpp.py:319 MXFP4 comment (L11); adapter_path Optional[str] (L12); add exc_info + narrow probe except (L13); seen-set dedup (L14); add Q2/Q3 tier labels to _TIER_TO_HF_LABEL (additional issue) or drop Q2 default tier; tidy dead scaffolding (L9-dead) and the HuggingFace publish command (M9) as a feature add.
18. 17. Add minimal CI + lint (additional issues): ruff/black in dev extras, Makefile lint runs it, GitHub Actions runs `pip install -e ".[dev]"` + dtype-guard + refactor-regression on every push, encoder-parity with skip-when-llama-quantize/gguf-absent guards, and an end-to-end create_hybrid_gguf smoke test (read->classify->write->reopen with GGUFReader->assert offsets/types).

## Test strategy

"EXISTING INFRA: pytest with tests/ split into unit (tests/test_quantization_guards.py — 18 passing, no external deps) and tests/integration/test_encoder_parity.py (needs llama-quantize binary AND the `gguf` pip package + a fixtures/reference_tensor.f32.npy). tests/test_refactor_regression.py is a seed-pinned golden fixture (tests/fixtures/refactor_regression_seed42.json) that pins evolutionary-search output and MUST be regenerated whenever noise_factors, group sets, floor-clamping, or early-stop change behavior. Run all: `pip install -e \".[dev]\"` then `python -m pytest tests/ -v`. CAVEAT (verified here): a bare checkout lacks structlog/pydantic_settings/tenacity/llama_cpp and `gguf`, so the refactor-regression and integration tests error on import until deps are installed; the dtype-guard tests pass standalone because they only import converters (which imports ggml_binding, which lazily binds libggml only on first encode — so size/guard tests that don't call a quantized encode still pass, but those that DO call ggml_encode, e.g. Q8_0/Q6_K output-size tests, require libggml from llama-cpp-python).\n\nHOW TO EXTEND: (1) Create tests/test_tensor_groups.py — pure-function, no deps — as the FIRST new test (catches the MoE bug and guards future arch additions); table of canonical dense/MoE/SSM names -> expected group. (2) Create tests/test_tiers.py asserting magicquant.quant.tiers.classify_tier == orchestrator._classify_tier across boundary grid. (3) Add a writer crash-safety test using a stub ModelSource (implement the abstract methods, raise/return-int on the 2nd tensor) — assert no file and no .partial remain. (4) Add an end-to-end smoke test: stub source with a few F32 tensors -> create_hybrid_gguf -> reopen with magicquant.gguf.reader.GGUFReader -> assert tensor count, ggml types per group, and that GGUFReader offsets are 32-aligned and monotonic. (5) For encoder-parity, change `import gguf` to `gguf = pytest.importorskip('gguf')` and add `gguf` to dev extras so the suite SKIPS (not fails) when absent; keep the existing pytest.skip for the llama-quantize binary. (6) For predictor param-counts, unit-test predict_size/predict_tps with explicit parameter_counts dicts (no model needed). (7) For config reconciliation, monkeypatch env vars and assert the orchestrator receives them.\n\nRUNNING SUBSETS: deps-free fast loop -> `pytest tests/test_tensor_groups.py tests/test_tiers.py -q`; libggml-required -> `pytest tests/test_quantization_guards.py -q` (needs llama-cpp-python); full parity -> `LLAMA_QUANTIZE=~/llama.cpp-build/build/bin/llama-quantize pytest tests/integration -q`. REGRESSION DISCIPLINE: after any change to survival.py/schemes.py noise_factors/group sets, run the refactor-regression test FIRST; if it legitimately changes, regenerate the fixture in a single dedicated commit (run _capture_run and write JSON) so the behavior delta is reviewable. Add CI (GitHub Actions) that installs `.[dev]`+gguf, runs the deps-free + libggml tests on every push, and the parity tests when a llama-quantize binary is cached/available (skip otherwise with a warning)."

## Notes

"Verification summary vs the prior audit (AUDIT_2026-06-09.md): I independently CONFIRMED every finding it listed — no false positives found in this audit. The headline MoE bug (H1) is empirically reproduced (ffn_up_exps/ffn_gate_exps -> U, ffn_down_exps -> X) — note the audit slightly understates it: ffn_down_exps actually DOES reach X (because the D pattern has a negative lookahead and X lists ffn_down_exps), so it's up+gate (2/3 of expert weight matrices) that misclassify, consistent with the audit's 'two-thirds' framing. H2 (search ignores X/R/S) confirmed at orchestrator.py:198,505. M3 (writer not crash-safe) confirmed at writer.py:555. M7 (requirements.txt) confirmed and I verified it actually breaks imports here. Version drift, stale CLAUDE.md/AUDIT_REPORT.md/LOGIC_AUDIT.md, Dockerfiles, MXFP4 comment, calibration_results.json absence — all confirmed.\n\nThree NET-NEW real issues the audit MISSED, in priority order: (A) IQ4_XS block/type-size table in converters.py is numerically WRONG (32/18 vs correct 256/136) and inconsistent with ggml_binding.py — latent now but a corruption trap the moment PR3 registers IQ4_XS; must be fixed as a dependency of PR3. (B) tests/integration/test_encoder_parity.py hard-imports `gguf` (undeclared dep) and FAILS collection rather than skipping when absent — the core invariant guard is silently un-runnable in a standard install. (C) generate_tiered_models requests a Q2 tier (orchestrator.py:570,756) that no scheme can satisfy AND naming._TIER_TO_HF_LABEL has no 'Q2' key, so Q2 outputs would get no HF badge even once reachable.\n\nThe May 2026 libggml refactor is real and verified: converters.py has zero numpy K-quant encoders, all quantized encoding routes through ggml_binding.ggml_quantize_chunk with a startup type-ID drift check — so the April audit docs' encoder CRITICALs are genuinely already_fixed. Highest-leverage work remains the MoE/SSM trio (H1+H2+M1), which should be fixed together and covered by one MoE end-to-end test, exactly as the prior audit recommends."