"""
MagicQuant Orchestrator - Coordinates the full Predict -> Measure -> Learn pipeline.

The core loop:
1. Sensitivity probing — measure per-group PPL impact
2. Evolutionary search — generate candidate hybrid configs, predict performance
3. Build & measure — create GGUFs for tier winners, run real perplexity
4. Active learning — feed residuals (measured - predicted) back into predictor
5. Repeat until convergence or budget exhausted
6. Output the best verified survivor per tier
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.evolution.probing import SensitivityProber
from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.utils.naming import generate_name
from magicquant.utils.llamacpp import LlamaCppTools, get_llamacpp_quant_type
from magicquant.logging import get_logger
from magicquant.quant.schemes import get_scheme_by_name

log = get_logger(__name__)


class MagicQuantOrchestrator:
    """
    Orchestrate the full MagicQuant search with real measurement feedback.

    The key difference from a prediction-only search: after each evolutionary
    round, the top candidates are actually built as GGUF files and measured
    with llama-perplexity. The residuals (measured_loss - predicted_loss)
    are fed back into the predictor, making it increasingly accurate for
    this specific model architecture.
    """

    def __init__(
        self,
        source_model_path: str,
        output_dir: str,
        llamacpp_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ):
        self.source_model_path = source_model_path
        self.adapter_path = adapter_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._llamacpp_path = llamacpp_path
        self._llama_tools: Optional[LlamaCppTools] = None

        self.baseline_ppl: Optional[float] = None
        # How baseline_ppl was obtained: "measured" (real llama-perplexity),
        # "fabricated" (measurement failed → default 5.0), or "prediction-only"
        # (no llama.cpp; heuristic search). Stamped into search_results.json so
        # QAT auto-detect and Foundry's rocmfpx MQ-hybrid can tell a verified
        # config from a guessed one.
        self.baseline_provenance: str = "unknown"
        self.sensitivity_weights: Optional[Dict[str, float]] = None
        # How self.sensitivity_weights was obtained: "measured" (every probe
        # got a real llama-perplexity reading), "partial" (some fell back),
        # or "heuristic" (ALL fell back -- the whole search then ran on
        # static empirical guesses, not this model's actual behavior). Copied
        # from SensitivityProber.probing_provenance right after probing;
        # stamped into search_results.json alongside baseline_provenance.
        self.probing_provenance: str = "unknown"
        self.predictor: Optional[PredictiveScorer] = None

        # Track all measured configs across rounds
        self._measured: Dict[str, Dict] = {}  # config_key -> {config, ppl, loss, path}

        # Per-group parameter counts, populated by _estimate_model_size and
        # fed to PredictiveScorer so MoE size/speed predictions use the real
        # (mostly-experts) distribution instead of the dense fallback.
        self._param_counts: Dict[str, int] = {}

        # Detected search groups (includes X/R/S when present), populated by
        # the search methods and passed to run_evolution.
        self._search_groups: List[str] = list(EvolutionarySurvivor.DEFAULT_GROUPS)

        # RNG seed for the last search (None = nondeterministic). Recorded in
        # search_results.json so a run can be reproduced / A-B compared.
        self._search_seed: Optional[int] = None

        # Importance matrix cache: {gguf_tensor_name: importance_vector} once
        # resolved via enable_imatrix(), or None (unweighted quantization,
        # the historical default). Applied to EVERY create_hybrid_gguf call
        # this orchestrator makes from here on -- candidate builds during a
        # measured search AND final tier generation, regardless of which
        # search path produced the config. Never blocks the pipeline: a
        # capture failure just leaves this None.
        self._imatrix: Optional[Dict[str, Any]] = None
        # Path to base-model logits saved via llama-perplexity
        # --kl-divergence-base, when enable_kl=True in run_measured_search.
        # None means KL-divergence scoring is inactive.
        self._kl_base_logits_path: Optional[str] = None
        # Corpus the base logits above were captured over -- every candidate's
        # KL calculation during the measured-search loop must reuse this
        # exact corpus to be comparable.
        self._kl_corpus_path: Optional[str] = None
        # Weight applied to |mean_kl| when blending KL into final-survivor
        # selection (see _select_final_survivors). Only has any effect when
        # a candidate actually has a "kl" measurement recorded.
        self._kl_weight: float = 0.0

    def _apply_seed(self, seed: Optional[int]) -> None:
        """Seed the RNGs once for a reproducible search.

        Seeds the global ``random`` module (used by survival.py's mutation /
        sampling and the orchestrator's candidate shuffle) plus numpy. Called
        ONCE at the start of a search — not per generation/round — so the
        sequence still evolves across rounds. ``None`` leaves RNG state
        untouched (nondeterministic; preserves the historical default and the
        seed-pinned regression fixture, which seeds globally in the test).
        """
        self._search_seed = seed
        if seed is None:
            return
        import random as _random
        _random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed & 0xFFFFFFFF)
        except Exception:
            pass

    def enable_imatrix(self, corpus_path: Optional[str] = None, **kwargs) -> bool:
        """Capture (or load a cached) importance matrix for the source model
        and cache it on ``self._imatrix`` for every subsequent
        ``create_hybrid_gguf`` call this orchestrator makes -- candidate
        builds during ``run_measured_search`` AND final tier generation via
        ``generate_hybrid_model``/``generate_tiered_models`` -- regardless of
        which search path (measured or prediction-only) produced the config.

        Requires ``self.source_model_path`` to be a GGUF (imatrix capture
        only reads GGUF); a safetensors source returns False and leaves
        quantization unweighted, same as never calling this at all.

        Returns True if an imatrix is now active, False otherwise (source
        isn't GGUF, or capture/load failed -- logged as a warning, never
        raised: this must never block the pipeline).
        """
        from magicquant.imatrix import ensure_imatrix

        # Default llama-imatrix to the sibling of the discovered perplexity
        # binary: ensure_imatrix's own fallback is a PATH lookup, which can
        # resolve to a DIFFERENT llama.cpp build than llamacpp_path -- e.g. a
        # stock brew install that can't load an arch only the configured fork
        # supports (bit for real on a qwen35 MTP model, 2026-07-04).
        if "imatrix_bin" not in kwargs:
            perplexity = getattr(self.llama_tools, "perplexity_tool", None)
            if perplexity:
                sibling = Path(perplexity).parent / "llama-imatrix"
                if sibling.exists():
                    kwargs["imatrix_bin"] = str(sibling)

        self._imatrix = ensure_imatrix(
            self.source_model_path, corpus_path=corpus_path, **kwargs
        )
        if self._imatrix is None:
            log.warning(
                "imatrix not active (source isn't GGUF, or capture failed) "
                "-- quantizing unweighted",
                stage="imatrix", source=self.source_model_path,
            )
        else:
            log.info(
                "imatrix active", stage="imatrix",
                n_tensors=len(self._imatrix),
            )
        return self._imatrix is not None

    @property
    def llama_tools(self) -> Optional[LlamaCppTools]:
        """Lazily initialize LlamaCppTools on first access."""
        if self._llama_tools is None:
            try:
                self._llama_tools = LlamaCppTools(self._llamacpp_path)
            except Exception as exc:
                log.warning("llama.cpp not available", error=str(exc), exc_info=exc)
                return None
        return self._llama_tools

    # ------------------------------------------------------------------
    # Full measured search (the real MagicQuant pipeline)
    # ------------------------------------------------------------------

    def run_measured_search(
        self,
        target_base_quant: str = "MXFP4_MOE",
        search_generations: int = 30,
        population_size: int = 80,
        measurement_rounds: int = 3,
        candidates_per_round: int = 4,
        verbose: bool = True,
        patience: Optional[int] = None,
        enable_rocmfpx: bool = False,
        enable_iq: bool = False,
        seed: Optional[int] = None,
        use_imatrix: bool = False,
        imatrix_corpus: Optional[str] = None,
        enable_kl: bool = False,
        kl_weight: float = 0.1,
        enable_speed_bench: bool = False,
        measurement_chunks: Optional[int] = None,
        seed_incumbents: bool = True,
        resume: bool = True,
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Run the full Predict -> Measure -> Learn loop.

        Args:
            target_base_quant: Default base quantization scheme
            search_generations: Evolutionary generations per round
            population_size: Candidates per generation
            measurement_rounds: How many build-measure-learn cycles
            candidates_per_round: How many configs to actually build and
                measure per round (tier winners + epsilon-greedy picks)
            verbose: Print progress
            use_imatrix: capture/reuse an importance matrix and weight every
                candidate build + final tier generation with it (see
                ``enable_imatrix``). Off by default (unweighted, historical
                behavior).
            imatrix_corpus: calibration corpus for imatrix capture; None uses
                the bundled default (magicquant/data/calib_corpus.txt).
            enable_kl: also measure real KL-divergence-to-base for each
                candidate (via llama-perplexity's built-in --kl-divergence)
                and blend it into final-survivor selection. Off by default.
            kl_weight: weight applied to |mean_kl| when blending into
                selection (see _select_final_survivors); only meaningful
                when enable_kl=True.
            enable_speed_bench: also measure real tokens/sec per candidate
                via llama-bench (informational; recorded in search_results
                .json, not fed into per-generation prediction scoring --
                bench-ing the full population every generation isn't
                tractable, only the small measured set is). Off by default.
            measurement_chunks: cap every perplexity/KL pass in this run to
                this many ctx_size-token chunks instead of the whole corpus
                (overrides LlamaCppTools' own MAGICQUANT_PPL_CHUNKS env
                fallback when set). None (default) measures the whole
                corpus every pass.
            seed_incumbents: seed the evolutionary search (every round) with
                llama.cpp's own Q4_K_M/Q5_K_M/Q6_K incumbent mixtures (see
                ``magicquant.incumbents``), restricted to the groups this
                search actually varies. On by default so a measured run can
                never silently lose to "what stock llama-quantize would have
                done anyway" (see magicquant.incumbents' module docstring for
                the real run that motivated this). Round 1 additionally
                force-measures every incumbent ahead of the normal
                tier-winner/epsilon picks (deduped against them), so every
                run records a real measurement for each one -- those entries
                carry an "incumbent": tier tag in search_results.json.
            resume: on start, look for ``<output_dir>/_measured_checkpoint
                .json`` from a prior (possibly killed) run of this exact
                search -- same seed, same source-model identity (path +
                size + mtime), same measurement conditions (chunks/ctx_size
                /corpus). If it matches, restore the baseline PPL,
                sensitivity weights, and every already-recorded measurement
                (skipping their rebuild via the existing config_key check)
                instead of re-running baseline measurement and probing from
                scratch. A missing/mismatched/corrupt checkpoint is logged
                and ignored -- the run proceeds fresh and overwrites it. A
                checkpoint is written after baseline+probing complete and
                after every successful candidate measurement (atomic
                tmp-then-``os.replace``, mirroring the GGUF writer's crash
                safety), and deleted once the run completes successfully.
                On by default; pass False to always start fresh.

        Returns:
            (all_configs, tiered_best) where tiered_best maps tier names
            to the best *measured* config for that tier.
        """
        self._apply_seed(seed)
        self._kl_weight = kl_weight if enable_kl else 0.0
        if verbose:
            log.info(
                "MagicQuant Measured Hybrid Search",
                stage="init",
                source=self.source_model_path,
                adapter=self.adapter_path,
                output_dir=str(self.output_dir),
                rounds=measurement_rounds,
                candidates_per_round=candidates_per_round,
                seed=seed,
            )

        if self.llama_tools is None:
            raise RuntimeError(
                "run_measured_search requires llama.cpp. Install it or use "
                "prediction-only mode (--rounds 0)."
            )
        if measurement_chunks is not None:
            self.llama_tools.ppl_chunks = measurement_chunks

        # ── Resume: look for a checkpoint from a prior (possibly killed) run
        # of this exact search before doing any real measurement work ──
        checkpoint_path = self._measured_checkpoint_path()
        checkpoint = (
            self._load_matching_checkpoint(checkpoint_path, verbose) if resume else None
        )

        # ── Step 1: Baseline perplexity ──
        # ``baseline_needs_standalone_measurement`` tracks whether we still
        # owe a real calculate_perplexity(source_model) pass: False when the
        # checkpoint already restored it, OR when Step 1b below fuses it in
        # from the KL base-logits save (that pass, even without
        # --kl-divergence, prints this same model's own "Final estimate:
        # PPL" -- see LlamaCppTools.save_base_logits). This turns "baseline
        # pass + KL-base-logits pass" into ONE llama-perplexity invocation
        # whenever enable_kl succeeds, instead of two.
        baseline_needs_standalone_measurement = True
        if checkpoint is not None:
            self.baseline_ppl = checkpoint["baseline_ppl"]
            self.baseline_provenance = checkpoint["baseline_provenance"]
            for key, entry in checkpoint.get("measured", {}).items():
                self._measured[key] = dict(entry)
            baseline_needs_standalone_measurement = False
            if verbose:
                log.info(
                    "Resumed baseline + measurements from checkpoint",
                    stage="resume", path=str(checkpoint_path),
                    measured=len(self._measured),
                )

        # ── Step 1b: optional imatrix + KL base logits (fuses in the ──
        # ── baseline measurement on a fresh run, see above) ──
        # Both are best-effort: a failure here degrades to the historical
        # behavior (unweighted quant / no KL score) rather than aborting a
        # real measured search over a secondary quality signal.
        if use_imatrix:
            # enable_imatrix -> ensure_imatrix already caches capture to disk
            # and reuses it on a hit, so re-calling this on resume is cheap
            # when the cache survived and correctly recomputes when it didn't
            # -- no separate resume bookkeeping needed for imatrix itself.
            self.enable_imatrix(imatrix_corpus)

        if enable_kl:
            # On resume, reuse the checkpoint's KL base-logits file if it's
            # still on disk -- regenerating it is one llama-perplexity pass
            # over the whole corpus, exactly the kind of work resume exists
            # to avoid. Falls through to a fresh capture if the file is gone.
            reused_kl = False
            if checkpoint is not None:
                ck_kl = checkpoint.get("kl") or {}
                base_path = ck_kl.get("base_logits_path")
                if ck_kl.get("enabled") and base_path and Path(base_path).is_file():
                    self._kl_base_logits_path = base_path
                    self._kl_corpus_path = ck_kl.get("corpus_path")
                    reused_kl = True
                    if verbose:
                        log.info(
                            "Reusing KL base logits from checkpoint",
                            stage="kl", path=base_path,
                        )
            if not reused_kl:
                # Reuse the SAME corpus already configured for baseline-PPL
                # measurement (not imatrix_corpus, a separate calibration-corpus
                # concept) -- KL only means something when base and candidate are
                # compared over identical text.
                corpus = self.llama_tools._resolve_data_file(None)
                if corpus is None:
                    log.warning(
                        "enable_kl requested but no calibration corpus resolved "
                        "-- skipping KL-divergence scoring", stage="kl",
                    )
                else:
                    base_logits_path = str(self.output_dir / "_kl_base_logits.kld")
                    saved_ppl = self.llama_tools.save_base_logits(
                        self.source_model_path, corpus, base_logits_path,
                        ctx_size=self.llama_tools.ctx_size,
                    )
                    if saved_ppl is not None:
                        self._kl_base_logits_path = base_logits_path
                        self._kl_corpus_path = corpus
                        log.info("KL base logits saved", stage="kl", path=base_logits_path)
                        if baseline_needs_standalone_measurement:
                            # Fuse: this pass's own PPL becomes the baseline,
                            # so the standalone baseline pass below is
                            # skipped entirely.
                            self.baseline_ppl = saved_ppl
                            self.baseline_provenance = "measured"
                            baseline_needs_standalone_measurement = False
                            if verbose:
                                log.info(
                                    "Baseline perplexity (fused with KL "
                                    "base-logits save)",
                                    stage="baseline", ppl=round(saved_ppl, 4),
                                )
                    else:
                        log.warning(
                            "Could not save base logits -- disabling KL-divergence "
                            "scoring for this run", stage="kl",
                        )

        # ── Step 1c: standalone baseline measurement ──
        # Skipped when the checkpoint already restored it, or Step 1b fused
        # it in above. This is the historical baseline pass, unchanged --
        # taken whenever enable_kl is off, or its fused attempt didn't pan
        # out (no corpus / save failure), matching the pre-fusion behavior.
        if baseline_needs_standalone_measurement:
            if verbose:
                log.info("Baseline perplexity", stage="baseline")

            self.baseline_ppl = self.llama_tools.calculate_perplexity(
                self.source_model_path, verbose=verbose
            )
            if self.baseline_ppl is None:
                # Measured search is worthless against a fabricated baseline: every
                # measured_loss=(ppl-baseline)/baseline and every survivor ranking
                # would be computed against a guess. Fail loudly rather than
                # silently emit "verified" tiers that were never verified.
                raise RuntimeError(
                    "Measured search could not measure baseline perplexity "
                    f"(llama-perplexity on {self.source_model_path}). Check the "
                    "llama.cpp build and calibration corpus. Refusing to proceed "
                    "with a fabricated baseline; use prediction-only search "
                    "(run_full_search) if no llama.cpp is available."
                )
            self.baseline_provenance = "measured"

        # ── Step 2: Sensitivity probing ──
        # Group detection is cheap tensor-name classification (no
        # measurement calls), so it always runs regardless of resume --
        # only the expensive probe_all_groups() below is skippable.
        groups = ["E", "H", "Q", "K", "O", "U", "D"]
        # Add MoE/SSM groups if present in the model
        classifier = TensorGroupClassifier()
        from magicquant.gguf.source import open_model_source
        _src = open_model_source(self.source_model_path)
        try:
            tensor_names = _src.get_tensor_names()
        finally:
            _src.close()
        if any(classifier.classify_tensor(t) in ("X", "R") for t in tensor_names):
            groups.extend(["X", "R"])
        if any(classifier.classify_tensor(t) == "S" for t in tensor_names):
            groups.append("S")
        # Remember the full detected group set so run_evolution actually
        # varies X/R/S (otherwise it falls back to DEFAULT_GROUPS).
        self._search_groups = groups

        if checkpoint is not None:
            self.sensitivity_weights = checkpoint["sensitivity_weights"]
            self.probing_provenance = checkpoint["probing_provenance"]
            if verbose:
                log.info(
                    "Resumed sensitivity weights from checkpoint", stage="resume",
                )
        else:
            if verbose:
                log.info("Sensitivity probing", stage="probing")

            prober = SensitivityProber(
                base_model_path=self.source_model_path,
                baseline_perplexity=self.baseline_ppl,
                perplexity_calculator=self.llama_tools,
                output_dir=str(self.output_dir / "_probes"),
            )
            prober.probe_all_groups(groups=groups, aggressive_scheme="Q4_K_M", verbose=verbose)
            self.sensitivity_weights = prober.get_normalized_weights()
            self.probing_provenance = prober.probing_provenance
            prober.save_results(str(self.output_dir / "sensitivity.json"))

            if verbose:
                log.info(
                    "Sensitivity weights computed",
                    stage="probing",
                    weights={g: round(w, 3) for g, w in self.sensitivity_weights.items()},
                )

        # Baseline + probing are complete (whether resumed or freshly
        # measured) -- checkpoint now so a kill during Step 4 can resume
        # past both without re-running either.
        self._write_measured_checkpoint(checkpoint_path)

        # ── Step 3: Initialize predictor ──
        # (_estimate_model_size also populates self._param_counts per group.)
        baseline_size_gb = self._estimate_model_size(self.source_model_path)

        self.predictor = PredictiveScorer(
            sensitivity_weights=self.sensitivity_weights,
            parameter_counts=self._param_counts,
            baseline_size_gb=baseline_size_gb,
            baseline_tps=360,
            imatrix_active=self._imatrix is not None,
        )

        # ── Step 3b: incumbent seeding ──
        # Build llama.cpp's own Q4_K_M/Q5_K_M/Q6_K mixtures (restricted to the
        # groups this search actually varies) so the evolutionary search is
        # anchored to "what stock llama-quantize would have done anyway" --
        # see magicquant.incumbents' module docstring for why this matters.
        seed_configs, incumbent_tier_by_key = self._build_incumbent_seeds(
            seed_incumbents
        )

        # ── Step 4: Measured search rounds ──
        all_configs = []

        for round_idx in range(measurement_rounds):
            if verbose:
                log.info(
                    "Measurement round starting",
                    stage="measurement",
                    round=round_idx + 1,
                    total_rounds=measurement_rounds,
                )

            # 4a. Run evolutionary search with current predictor
            survivor = EvolutionarySurvivor(
                predictor=self.predictor,
                baseline_config={"E": "BF16", "H": "BF16"},
                max_generations=search_generations,
                population_size=population_size,
                epsilon=0.2,
                enable_rocmfpx=enable_rocmfpx,
                enable_iq=enable_iq,
            )

            round_configs = survivor.run_evolution(
                groups=self._search_groups, verbose=verbose, patience=patience,
                seed_configs=seed_configs if seed_configs else None,
            )
            all_configs.extend(round_configs)

            # 4b. Pick candidates to measure: tier winners + epsilon picks
            to_measure = self._select_measurement_candidates(
                round_configs, baseline_size_gb, candidates_per_round
            )

            # Round 1 force-measures every incumbent ahead of the normal
            # picks (deduped against them and against anything already
            # measured), so every run records a real measurement for "what
            # stock llama-quantize would have done" regardless of whether
            # the evolutionary search happened to rediscover it on its own.
            if round_idx == 0 and seed_configs:
                already_key = {
                    self._config_key(c["config"]) for c in to_measure
                }
                forced = []
                for cfg in seed_configs:
                    key = self._config_key(cfg)
                    if key in already_key or key in self._measured:
                        continue
                    already_key.add(key)
                    forced.append({"config": cfg})
                to_measure = forced + to_measure

            if verbose:
                log.info(
                    "Candidates selected for measurement",
                    stage="measurement",
                    count=len(to_measure),
                )

            # 4c. Build, measure, learn
            for i, candidate in enumerate(to_measure):
                config = candidate["config"]
                config_key = self._config_key(config)

                # Skip if already measured
                if config_key in self._measured:
                    if verbose:
                        log.debug(
                            "Candidate already measured, skipping",
                            stage="measurement",
                            progress=f"{i+1}/{len(to_measure)}",
                        )
                    continue

                if verbose:
                    schemes = " ".join(f"{g}:{s}" for g, s in sorted(config.items()))
                    log.info(
                        "Building candidate",
                        stage="measurement",
                        progress=f"{i+1}/{len(to_measure)}",
                        schemes=schemes,
                    )

                # Build GGUF
                model_name = f"round{round_idx+1}_candidate{i+1}"
                path = self._build_candidate(config, model_name, target_base_quant)

                if path is None:
                    continue

                # Measure perplexity, fusing in the KL pass when active: a
                # --kl-divergence run against saved base logits ALSO prints
                # this candidate's own perplexity (Mean PPL(Q)), so when KL
                # scoring is active we get both signals from ONE
                # llama-perplexity invocation instead of two. Falls back to
                # the historical standalone calculate_perplexity call when
                # KL is off, its base logits aren't active, the KL call
                # raised, or its result doesn't carry "ppl" -- the "KL
                # failure must not abort/win" guarantee stays intact either
                # way (measured entry gets ppl either way, "kl" only
                # recorded when the KL call itself succeeded).
                kl_result = None
                if enable_kl and self._kl_base_logits_path:
                    try:
                        kl_result = self.llama_tools.calculate_kl_divergence(
                            path, self._kl_base_logits_path, self._kl_corpus_path,
                            ctx_size=self.llama_tools.ctx_size,
                        )
                    except Exception as exc:
                        log.warning(
                            "KL-divergence measurement failed for candidate; "
                            "continuing without it", stage="kl", error=str(exc),
                        )
                        kl_result = None

                if kl_result is not None and kl_result.get("ppl") is not None:
                    ppl = kl_result["ppl"]
                else:
                    ppl = self.llama_tools.calculate_perplexity(path, verbose=verbose)

                if ppl is not None:
                    measured_loss = (ppl - self.baseline_ppl) / self.baseline_ppl
                    predicted_loss = self.predictor.predict_loss(config)
                    residual = measured_loss - predicted_loss

                    # Record measurement
                    candidate_path = Path(path)
                    self._measured[config_key] = {
                        "config": config,
                        "ppl": ppl,
                        "measured_loss": measured_loss,
                        "predicted_loss": predicted_loss,
                        "residual": residual,
                        "path": path,
                        "size_gb": candidate_path.stat().st_size / (1024 ** 3),
                    }
                    if config_key in incumbent_tier_by_key:
                        self._measured[config_key]["incumbent"] = (
                            incumbent_tier_by_key[config_key]
                        )

                    if kl_result is not None:
                        self._measured[config_key]["kl"] = kl_result

                    # Optional secondary signal -- best-effort (None on
                    # failure), scored in _select_final_survivors alongside
                    # measured_loss rather than gating the candidate at all.
                    # bench() only catches CalledProcessError/TimeoutExpired
                    # internally; a missing or wrong-arch binary raises
                    # OSError/FileNotFoundError, which must not abort the
                    # rest of the search.
                    if enable_speed_bench:
                        try:
                            self._measured[config_key]["bench"] = self.llama_tools.bench(path)
                        except Exception as exc:
                            log.warning(
                                "Speed bench failed for candidate; continuing "
                                "without it", stage="bench", error=str(exc),
                            )

                    # Active learning: feed residual back
                    self.predictor.record_residual(config, residual)

                    if verbose:
                        log.info(
                            "Candidate measured",
                            stage="measurement",
                            ppl=round(ppl, 4),
                            measured_loss=round(measured_loss, 4),
                            predicted_loss=round(predicted_loss, 4),
                            residual=round(residual, 4),
                        )

                    # Persist after EVERY successful measurement -- a kill
                    # right after this point must resume with this candidate
                    # already recorded, not lost.
                    self._write_measured_checkpoint(checkpoint_path)
                else:
                    if verbose:
                        log.warning("Measurement failed", stage="measurement")

                # Clean up candidate GGUF to save disk (keep only final survivors)
                # We'll rebuild the final survivors at the end
                candidate_file = Path(path)
                if candidate_file.exists():
                    candidate_file.unlink()

            # 4d. Log round summary
            if verbose and self._measured:
                avg_residual = sum(
                    abs(m["residual"]) for m in self._measured.values()
                ) / len(self._measured)
                log.info(
                    "Round summary",
                    stage="measurement",
                    round=round_idx + 1,
                    total_measurements=len(self._measured),
                    mean_abs_residual=round(avg_residual, 4),
                )

        # A measured search whose every candidate build/measure failed must
        # not report success: self._measured stays empty, _select_final_
        # survivors would return {}, and _save_results would still write a
        # valid-looking search_results.json with zero measurements -- an
        # overnight run that silently accomplished nothing. Fail loudly
        # instead of falling through to Step 5.
        if measurement_rounds > 0 and not self._measured:
            raise RuntimeError(
                "Measured search completed all "
                f"{measurement_rounds} round(s) but produced zero successful "
                "measurements (every candidate build or perplexity "
                "measurement failed). Refusing to write search_results.json "
                "as if this were a normal partial run -- check the "
                "llama.cpp build, disk space, and per-candidate build errors "
                "logged above."
            )

        # ── Step 5: Select final survivors per tier ──
        tiered = self._select_final_survivors(baseline_size_gb)

        # Save all results
        self._save_results(all_configs, tiered)

        # Run completed successfully -- the checkpoint's job is done.
        checkpoint_path.unlink(missing_ok=True)

        if verbose:
            for tier, info in tiered.items():
                c = info["config"]
                schemes = " ".join(f"{g}:{s}" for g, s in sorted(c.items()))
                log.info(
                    "Final verified survivor",
                    stage="results",
                    tier=tier,
                    ppl=round(info["ppl"], 4),
                    measured_loss=round(info["measured_loss"], 4),
                    size_gb=round(info["size_gb"], 2),
                    schemes=schemes,
                )

        return all_configs, tiered

    def _select_measurement_candidates(
        self,
        configs: List[Dict],
        baseline_gb: float,
        n: int,
    ) -> List[Dict]:
        """Pick the best candidates to actually build and measure.

        Every discovered tier band contributes its winner unconditionally --
        tier winners are never truncated away by a small ``n`` or crowded
        out by epsilon-greedy random picks. ``n`` caps the *epsilon*
        exploration budget on top of the guaranteed tier winners, not the
        total: if more tiers were discovered than ``n``, every tier winner
        still ships (this round just measures more than ``n`` candidates).
        """
        tiered = self._pick_best_per_tier(configs, baseline_gb)
        tier_winners = list(tiered.values())
        winner_keys = {self._config_key(c["config"]) for c in tier_winners}

        # Tier winners already measured in a prior round don't need a
        # rebuild, but they still "count" as covering their band.
        to_build = [
            c for c in tier_winners
            if self._config_key(c["config"]) not in self._measured
        ]

        # Epsilon-greedy: random picks from the rest of the discovered pool,
        # filling up to n total on top of the guaranteed tier winners.
        import random
        remaining = [
            c for c in configs
            if self._config_key(c["config"]) not in winner_keys
            and self._config_key(c["config"]) not in self._measured
        ]
        budget = max(0, n - len(to_build))
        if remaining and budget:
            random.shuffle(remaining)
            seen = {self._config_key(c["config"]) for c in to_build}
            for c in remaining:
                key = self._config_key(c["config"])
                if key in seen:
                    continue
                seen.add(key)
                to_build.append(c)
                budget -= 1
                if budget <= 0:
                    break

        return to_build

    def _build_candidate(
        self, config: Dict[str, str], name: str, base_quant: str
    ) -> Optional[str]:
        """Build a hybrid GGUF for measurement. Returns path or None."""
        from magicquant.gguf.writer import create_hybrid_gguf

        output_filename = generate_name(name, base_quant, config)
        candidates_dir = self.output_dir / "_candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(candidates_dir / output_filename)

        try:
            return create_hybrid_gguf(
                output_path=output_path,
                base_model_path=self.source_model_path,
                quant_config={"base": base_quant, "groups": config},
                verbose=False,
                adapter_path=self.adapter_path,
                imatrix=self._imatrix,
            )
        except Exception as exc:
            log.error("Build failed", stage="build", error=str(exc), exc_info=exc)
            return None

    def _selection_score(self, info: Dict) -> float:
        """Ranking key for tier-winner selection: measured_loss, optionally
        blended with |mean_kl| when a "kl" measurement is present (only true
        when ``enable_kl=True`` was passed to ``run_measured_search`` AND
        base-logits capture succeeded for this run). ``self._kl_weight`` is
        0.0 whenever KL scoring is inactive, so this is a no-op in that case.
        """
        score = info["measured_loss"]
        kl = info.get("kl")
        if kl and kl.get("mean_kl") is not None:
            score += self._kl_weight * abs(kl["mean_kl"])
        return score

    def _select_final_survivors(self, baseline_gb: float) -> Dict[str, Dict]:
        """From all measured configs, pick the best per tier."""
        by_tier: Dict[str, List[Dict]] = defaultdict(list)
        for info in self._measured.values():
            tier = self._classify_tier(info["size_gb"], baseline_gb)
            by_tier[tier].append(info)

        result = {}
        for tier in ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]:
            if tier in by_tier:
                candidates = by_tier[tier]
                # A candidate whose KL measurement failed (calculate_kl_
                # divergence raised or returned None) must never look BETTER
                # than the worst candidate that actually measured KL in this
                # tier -- otherwise a measurement failure gets rewarded over
                # real (if poor) data. Only kicks in when at least one
                # sibling in the tier has KL data; falls back to plain
                # _selection_score (no-op when kl_weight is 0) otherwise.
                kl_vals = [
                    abs(c["kl"]["mean_kl"]) for c in candidates
                    if c.get("kl") and c["kl"].get("mean_kl") is not None
                ]
                worst_kl = max(kl_vals) if kl_vals else None

                def _score(info, worst_kl=worst_kl):
                    score = self._selection_score(info)
                    has_kl = info.get("kl") and info["kl"].get("mean_kl") is not None
                    if worst_kl is not None and not has_kl:
                        score += self._kl_weight * worst_kl
                    return score

                best = min(candidates, key=_score)
                result[tier] = best
        return result

    def _measurement_metadata(self) -> Dict[str, Any]:
        """Describe the conditions under which this run's numbers were
        measured, so results from different runs are never silently compared.

        Every field is read via ``getattr``/``.get`` with a fallback -- this
        must not raise for a prediction-only run (no ``_llama_tools``, no
        ``_imatrix``) or for older orchestrator state built via ``__new__``
        that predates these attributes entirely.
        """
        llama = getattr(self, "_llama_tools", None)
        corpus = None
        if llama is not None:
            try:
                corpus = llama._resolve_data_file(None)
            except Exception:
                corpus = None

        imatrix = getattr(self, "_imatrix", None)

        probing_provenance = None
        output_dir = getattr(self, "output_dir", None)
        if output_dir is not None:
            try:
                sensitivity_path = Path(output_dir) / "sensitivity.json"
                if sensitivity_path.is_file():
                    sensitivity_data = json.loads(sensitivity_path.read_text())
                    probing_provenance = sensitivity_data.get("probing_provenance")
            except Exception:
                probing_provenance = None

        return {
            "chunks": getattr(llama, "ppl_chunks", None),
            "ctx_size": getattr(llama, "ctx_size", None),
            "corpus": corpus,
            "imatrix_active": imatrix is not None,
            "imatrix_n_tensors": len(imatrix) if imatrix else None,
            "kl_enabled": bool(getattr(self, "_kl_base_logits_path", None)),
            "kl_weight": getattr(self, "_kl_weight", 0.0),
            "probing_provenance": probing_provenance,
        }

    def _save_results(self, all_configs, tiered):
        """Persist search results and measurements to JSON.

        Called from BOTH search paths. Prediction-only tiers (run_full_search)
        carry ``predicted_size_gb``/``predicted_loss`` and no measured fields,
        so every access is ``.get()`` with the predicted fallback — the
        measured path simply fills in more of the fields. Consumers (QAT's
        ``load_hybrid_config``, Foundry's rocmfpx MQ-hybrid mode) only require
        ``tiered[tier]["config"]``, which both paths provide.
        """
        results = {
            "baseline_ppl": self.baseline_ppl,
            "baseline_provenance": self.baseline_provenance,
            "probing_provenance": getattr(self, "probing_provenance", "unknown"),
            "seed": self._search_seed,
            "measurement": self._measurement_metadata(),
            "measurements": {
                k: {
                    "config": v["config"],
                    "ppl": v.get("ppl"),
                    "measured_loss": v.get("measured_loss"),
                    "predicted_loss": v.get("predicted_loss"),
                    "residual": v.get("residual"),
                    "size_gb": v.get("size_gb"),
                    "kl": v.get("kl"),
                    "bench": v.get("bench"),
                    "incumbent": v.get("incumbent"),
                }
                for k, v in self._measured.items()
            },
            "tiered_survivors": {
                tier: {
                    "config": info["config"],
                    "ppl": info.get("ppl"),
                    "measured_loss": info.get("measured_loss"),
                    "size_gb": info.get("size_gb", info.get("predicted_size_gb")),
                }
                for tier, info in tiered.items()
            },
            "tiered": {
                tier: {
                    "config": info["config"],
                    "ppl": info.get("ppl"),
                    "measured_loss": info.get("measured_loss"),
                    "predicted_loss": info.get("predicted_loss"),
                    "size_gb": info.get("size_gb", info.get("predicted_size_gb")),
                }
                for tier, info in tiered.items()
            },
        }

        results_path = self.output_dir / "search_results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Prediction-only search (no llama.cpp needed)
    # ------------------------------------------------------------------

    def run_full_search(
        self,
        target_base_quant: str = "MXFP4_MOE",
        max_generations: int = 50,
        population_size: int = 100,
        verbose: bool = True,
        patience: Optional[int] = None,
        enable_rocmfpx: bool = False,
        enable_iq: bool = False,
        seed: Optional[int] = None,
        use_imatrix: bool = False,
        imatrix_corpus: Optional[str] = None,
        measurement_chunks: Optional[int] = None,
        seed_incumbents: bool = True,
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Run prediction-only evolutionary search (no real measurements).
        Use run_measured_search() for the full Predict->Measure->Learn loop.

        use_imatrix/imatrix_corpus: prediction-only search never builds
        candidate GGUFs, so this has no effect on the search itself -- it
        only makes generate_hybrid_model/generate_tiered_models (called
        afterward with this same orchestrator) quantize with an importance
        matrix instead of unweighted. Off by default; safe for the fixture.

        measurement_chunks: cap the (single) baseline perplexity pass to
        this many ctx_size-token chunks instead of the whole corpus.
        Symmetric with run_measured_search's knob of the same name; a no-op
        when llama.cpp is unavailable (no baseline pass runs at all).

        seed_incumbents: same seeding as run_measured_search (see its
        docstring and magicquant.incumbents), minus the force-measurement --
        this path never measures anything real, it only seeds the
        evolutionary search's population so the incumbent mixtures are
        always among the discovered/scored configs. On by default.
        """
        self._apply_seed(seed)
        if use_imatrix:
            self.enable_imatrix(imatrix_corpus)
        if verbose:
            log.info(
                "MagicQuant Prediction-Only Search",
                stage="init",
                source=self.source_model_path,
            )

        # Baseline PPL. Prediction-only search doesn't strictly need it (the
        # predictor scores by relative noise), so a default is tolerable here —
        # but stamp provenance so consumers know the tiers are predicted, not
        # verified.
        _llama = self.llama_tools
        if _llama is not None:
            if measurement_chunks is not None:
                _llama.ppl_chunks = measurement_chunks
            self.baseline_ppl = _llama.calculate_perplexity(
                self.source_model_path, verbose=verbose
            )
            if self.baseline_ppl is None:
                log.warning(
                    "Baseline PPL measurement failed; using default (search "
                    "remains prediction-only)",
                    stage="baseline", default_ppl=5.0,
                )
                self.baseline_ppl = 5.0
                self.baseline_provenance = "fabricated"
            else:
                self.baseline_provenance = "measured"
        else:
            log.warning(
                "llama.cpp unavailable, using default baseline PPL",
                stage="baseline",
                default_ppl=5.0,
            )
            self.baseline_ppl = 5.0
            self.baseline_provenance = "prediction-only"

        # Sensitivity probing
        if verbose:
            log.info("Sensitivity probing", stage="probing")
        prober = SensitivityProber(
            base_model_path=self.source_model_path,
            baseline_perplexity=self.baseline_ppl,
            perplexity_calculator=_llama,
            output_dir=str(self.output_dir / "_probes"),
        )
        groups = ["E", "H", "Q", "K", "O", "U", "D"]
        # Add MoE/SSM groups if present in the model
        classifier = TensorGroupClassifier()
        from magicquant.gguf.source import open_model_source
        _src = open_model_source(self.source_model_path)
        try:
            tensor_names = _src.get_tensor_names()
        finally:
            _src.close()
        if any(classifier.classify_tensor(t) in ("X", "R") for t in tensor_names):
            groups.extend(["X", "R"])
        if any(classifier.classify_tensor(t) == "S" for t in tensor_names):
            groups.append("S")
        # Remember the full detected group set so run_evolution actually
        # varies X/R/S (otherwise it falls back to DEFAULT_GROUPS).
        self._search_groups = groups

        prober.probe_all_groups(groups=groups, aggressive_scheme="Q4_K_M", verbose=verbose)
        self.sensitivity_weights = prober.get_normalized_weights()
        self.probing_provenance = prober.probing_provenance
        prober.save_results(str(self.output_dir / "sensitivity.json"))

        # (_estimate_model_size also populates self._param_counts per group.)
        baseline_size_gb = self._estimate_model_size(self.source_model_path)

        self.predictor = PredictiveScorer(
            sensitivity_weights=self.sensitivity_weights,
            parameter_counts=self._param_counts,
            baseline_size_gb=baseline_size_gb,
            baseline_tps=360,
            imatrix_active=self._imatrix is not None,
        )

        seed_configs, _incumbent_tier_by_key = self._build_incumbent_seeds(
            seed_incumbents
        )

        survivor = EvolutionarySurvivor(
            predictor=self.predictor,
            baseline_config={"E": "BF16", "H": "BF16"},
            max_generations=max_generations,
            population_size=population_size,
            epsilon=0.2,
            enable_rocmfpx=enable_rocmfpx,
            enable_iq=enable_iq,
        )

        best_configs = survivor.run_evolution(
            groups=self._search_groups, verbose=verbose, patience=patience,
            seed_configs=seed_configs if seed_configs else None,
        )
        tiered = self._pick_best_per_tier(best_configs, baseline_size_gb)

        for cfg in best_configs:
            if 'tier' not in cfg:
                cfg['tier'] = self._classify_tier(
                    cfg.get('predicted_size_gb', 0), baseline_size_gb
                )

        # Persist search_results.json for downstream consumers (QAT's
        # auto-detect, Foundry's rocmfpx MQ-hybrid mode). Previously only the
        # measured path saved — the prediction-only path silently produced
        # nothing to hand off.
        self._save_results(best_configs, tiered)

        return best_configs, tiered

    # ------------------------------------------------------------------
    # Model generation
    # ------------------------------------------------------------------

    def generate_hybrid_model(
        self, config: Dict[str, str], model_name: str,
        base_quant: str = "MXFP4_MOE", verify: bool = True,
    ) -> Optional[str]:
        """Generate a hybrid GGUF model."""
        from magicquant.gguf.writer import create_hybrid_gguf

        output_filename = generate_name(model_name, base_quant, config)
        output_path = self.output_dir / output_filename

        log.info(
            "Generating hybrid GGUF",
            stage="generate",
            filename=output_filename,
            group_schemes={g: s for g, s in sorted(config.items())},
        )

        try:
            result = create_hybrid_gguf(
                output_path=str(output_path),
                base_model_path=self.source_model_path,
                quant_config={"base": base_quant, "groups": config},
                verbose=True,
                adapter_path=self.adapter_path,
                imatrix=self._imatrix,
            )
            if not Path(result).is_file():
                return None
        except Exception as exc:
            log.error("Generation failed", stage="generate", error=str(exc), exc_info=exc)
            return None

        if verify and self.baseline_ppl:
            ppl = self.llama_tools.calculate_perplexity(str(output_path))
            if ppl:
                loss = (ppl - self.baseline_ppl) / self.baseline_ppl
                log.info(
                    "Verification complete",
                    stage="generate",
                    ppl=round(ppl, 4),
                    loss_pct=round(loss * 100, 2),
                )

        return str(output_path)

    def generate_tiered_models(
        self, tiered: Dict[str, Dict], model_name_prefix: str = "Model",
        tiers: Optional[List[str]] = None, verify: bool = False,
    ) -> List[str]:
        """Generate one hybrid GGUF per compression tier.

        NOTE: the Q2 tier (size ratio <= 0.16) is currently UNREACHABLE — the
        smallest registered scheme is Q2_K (bpw=2.625 -> ratio ~0.164, just
        outside the band). It will log "No config for tier, skipping" until
        PR3 adds sub-Q2 IQ-quants. Q2 is kept in the default list (and now has
        an HF filename label) so it fills automatically once PR3 lands.
        """
        if tiers is None:
            tiers = ["Q8", "Q6", "Q5", "Q4", "Q2"]

        generated = []
        for tier in tiers:
            if tier not in tiered:
                log.info("No config for tier, skipping", stage="generate", tier=tier)
                continue

            entry = tiered[tier]
            config = entry["config"]
            name = f"{model_name_prefix}-{tier}"

            # base_quant: pick the scheme with highest bpw (least compressed) as
            # the "label" for this hybrid. Reads bpw from the canonical registry.
            def _bpw_or_default(s: str) -> float:
                try:
                    return get_scheme_by_name(s).bits_per_weight
                except ValueError:
                    return 4.5  # mid-range default for unknown schemes
            base_quant = max(set(config.values()), key=_bpw_or_default)

            log.info(
                "Generating tier model",
                stage="generate",
                tier=tier,
                name=name,
                ppl=round(entry["ppl"], 4) if "ppl" in entry else None,
                measured_loss=round(entry["measured_loss"], 4) if "measured_loss" in entry else None,
            )

            path = self.generate_hybrid_model(
                config=config, model_name=name,
                base_quant=base_quant, verify=verify,
            )
            if path:
                generated.append(path)
            else:
                log.error("Tier generation failed", stage="generate", tier=tier)

        return generated

    def generate_top_models(
        self, results: List[Dict], top_n: int = 3,
        model_name_prefix: str = "Model", base_quant: str = "MXFP4_MOE",
        verify: bool = False,
    ) -> List[str]:
        """Generate hybrid GGUFs for the top-N results by score."""
        generated = []
        for i, entry in enumerate(results[:top_n], 1):
            path = self.generate_hybrid_model(
                config=entry["config"],
                model_name=f"{model_name_prefix}-Config{i}",
                base_quant=base_quant, verify=verify,
            )
            if path:
                generated.append(path)
        return generated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _config_key(config: Dict[str, str]) -> str:
        return "|".join(f"{g}:{config[g]}" for g in sorted(config))

    def _build_incumbent_seeds(
        self, seed_incumbents: bool
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Build llama.cpp's own per-tier incumbent mixtures (see
        ``magicquant.incumbents``), restricted to the groups this search
        actually varies (``self._search_groups``, which must already be set
        by the time this is called).

        Returns ``(seed_configs, incumbent_tier_by_key)``: the restricted
        config dicts to feed straight into
        ``EvolutionarySurvivor.run_evolution(seed_configs=...)``, and a
        ``config_key -> tier`` map used to tag forced measurements as
        ``"incumbent"`` in search_results.json. Both are empty when
        ``seed_incumbents`` is False.
        """
        seed_configs: List[Dict[str, str]] = []
        incumbent_tier_by_key: Dict[str, str] = {}
        if not seed_incumbents:
            return seed_configs, incumbent_tier_by_key

        from magicquant.incumbents import get_incumbent_config, INCUMBENT_TIERS

        for tier in ["Q4", "Q5", "Q6"]:
            if tier not in INCUMBENT_TIERS:
                continue
            incumbent = get_incumbent_config(tier)
            restricted = {
                g: s for g, s in incumbent.items() if g in self._search_groups
            }
            if not restricted:
                continue
            seed_configs.append(restricted)
            incumbent_tier_by_key[self._config_key(restricted)] = tier
        return seed_configs, incumbent_tier_by_key

    # ------------------------------------------------------------------
    # Measured-search checkpoint / resume
    # ------------------------------------------------------------------

    def _measured_checkpoint_path(self) -> Path:
        return self.output_dir / "_measured_checkpoint.json"

    def _source_identity(self) -> Dict[str, Any]:
        """Identity fingerprint for the source model: path + total size +
        latest mtime. Comparing only the path would miss an in-place model
        swap at the same path between a killed run and its resume attempt.

        A directory (safetensors checkpoint) aggregates over its
        ``*.safetensors`` files, matching ``_estimate_model_size``'s own
        fallback glob. Any stat failure (missing file/dir) degrades to a
        ``None``-filled identity rather than raising -- a resume check must
        never crash the search, it should just conclude "doesn't match".
        """
        p = Path(self.source_model_path)
        try:
            if p.is_dir():
                total_size = 0
                latest_mtime = 0.0
                for f in sorted(p.glob("*.safetensors")):
                    st = f.stat()
                    total_size += st.st_size
                    latest_mtime = max(latest_mtime, st.st_mtime)
                return {"path": str(p), "size": total_size, "mtime": latest_mtime}
            st = p.stat()
            return {"path": str(p), "size": st.st_size, "mtime": st.st_mtime}
        except OSError:
            return {"path": str(p), "size": None, "mtime": None}

    def _safe_resolve_corpus(self) -> Optional[str]:
        try:
            return self.llama_tools._resolve_data_file(None)
        except Exception:
            return None

    def _current_measurement_conditions(self) -> Dict[str, Any]:
        """The subset of measurement conditions that must match between a
        checkpoint and the run attempting to resume it: the chunk cap, ctx
        size, and calibration corpus. (Fuller run metadata -- imatrix/KL
        state, probing provenance -- is recorded in the checkpoint too, but
        those are RESULTS of a run, not inputs to compare for eligibility.)
        """
        llama = getattr(self, "_llama_tools", None)
        return {
            "chunks": getattr(llama, "ppl_chunks", None),
            "ctx_size": getattr(llama, "ctx_size", None),
            "corpus": self._safe_resolve_corpus(),
        }

    def _load_matching_checkpoint(
        self, path: Path, verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Load ``_measured_checkpoint.json`` and return it only if it's
        valid JSON AND its seed + source-model identity + measurement
        conditions match this run. Any mismatch or parse failure logs why
        and returns None -- the caller then runs fresh (and eventually
        overwrites the stale/corrupt checkpoint).
        """
        if not path.is_file():
            return None
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning(
                "Checkpoint unreadable/corrupted -- running fresh",
                stage="resume", path=str(path), error=str(exc),
            )
            return None

        reasons = []
        if checkpoint.get("seed") != self._search_seed:
            reasons.append(
                f"seed {checkpoint.get('seed')!r} != {self._search_seed!r}"
            )
        current_source = self._source_identity()
        if checkpoint.get("source_model") != current_source:
            reasons.append("source model identity changed")
        current_conditions = self._current_measurement_conditions()
        if checkpoint.get("measurement_conditions") != current_conditions:
            reasons.append("measurement conditions changed")

        if reasons:
            if verbose:
                log.info(
                    "Checkpoint present but not resumable -- running fresh",
                    stage="resume", path=str(path), reasons=reasons,
                )
            return None
        return checkpoint

    @staticmethod
    def _json_safe(obj):
        """Coerce numpy scalars/arrays a measurement might carry (kl/bench
        values) so a checkpoint write can never crash the search mid-run."""
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return str(obj)

    def _write_measured_checkpoint(self, path: Path) -> None:
        """Atomically persist enough state to resume a killed measured
        search: baseline, sensitivity weights, every measurement recorded so
        far, and the identity/condition fields a later resume must match.
        Mirrors gguf/writer.py's tmp-then-``os.replace`` pattern -- a kill
        mid-write must never leave a half-written checkpoint a later resume
        attempts to parse.
        """
        checkpoint = {
            "version": 1,
            "seed": self._search_seed,
            "source_model": self._source_identity(),
            "measurement_conditions": self._current_measurement_conditions(),
            "baseline_ppl": self.baseline_ppl,
            "baseline_provenance": self.baseline_provenance,
            "sensitivity_weights": self.sensitivity_weights,
            "probing_provenance": self.probing_provenance,
            "kl": {
                "enabled": bool(self._kl_base_logits_path),
                "base_logits_path": self._kl_base_logits_path,
                "corpus_path": self._kl_corpus_path,
            },
            "imatrix": {
                "active": self._imatrix is not None,
                "n_tensors": len(self._imatrix) if self._imatrix else None,
            },
            "measured": {
                k: {
                    "config": v["config"],
                    "ppl": v.get("ppl"),
                    "measured_loss": v.get("measured_loss"),
                    "predicted_loss": v.get("predicted_loss"),
                    "residual": v.get("residual"),
                    "path": v.get("path"),
                    "size_gb": v.get("size_gb"),
                    "kl": v.get("kl"),
                    "bench": v.get("bench"),
                    "incumbent": v.get("incumbent"),
                }
                for k, v in self._measured.items()
            },
        }
        tmp_path = str(path) + ".tmp"
        Path(tmp_path).write_text(
            json.dumps(checkpoint, indent=2, default=self._json_safe), encoding="utf-8"
        )
        os.replace(tmp_path, path)

    @staticmethod
    def _classify_tier(size_gb: float, baseline_gb: float) -> str:
        # Delegates to the leaf module magicquant.quant.tiers so a single set
        # of boundaries is used everywhere (and leaf modules need not import
        # this orchestrator).
        from magicquant.quant.tiers import classify_tier
        return classify_tier(size_gb, baseline_gb)

    @staticmethod
    def _pick_best_per_tier(configs: List[Dict], baseline_gb: float) -> Dict[str, Dict]:
        by_tier: Dict[str, List[Dict]] = defaultdict(list)
        for cfg in configs:
            size_gb = cfg.get('predicted_size_gb', 0)
            tier = MagicQuantOrchestrator._classify_tier(size_gb, baseline_gb)
            by_tier[tier].append(cfg)
        result = {}
        for tier in ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]:
            if tier in by_tier:
                result[tier] = max(by_tier[tier], key=lambda x: x.get('composite_score', 0))
        return result

    def _estimate_model_size(self, model_path: str) -> float:
        """Compute BF16 baseline size in GB from the total parameter count.

        Using ``parameter_count * 2`` bytes gives the true BF16 baseline
        regardless of the source format (which could be a pre-quantized GGUF
        with a smaller on-disk size).

        Side effect: populates ``self._param_counts`` with per-group element
        counts (classified via TensorGroupClassifier) so the predictor can use
        the real parameter distribution — critical for MoE models where the
        experts group X holds the bulk of the weights.
        """
        from magicquant.gguf.source import open_model_source
        try:
            src = open_model_source(model_path)
            try:
                classifier = TensorGroupClassifier()
                param_counts: Dict[str, int] = defaultdict(int)
                total_elements = 0
                for info in src.get_all_tensors_info():
                    n = 1
                    for d in info["shape"]:
                        n *= d
                    total_elements += n
                    group = classifier.classify_tensor(info["name"])
                    param_counts[group] += n
                # Store for the predictor (drop UNKNOWN so it doesn't skew
                # group-relative shares; its weights still count toward size).
                self._param_counts = {
                    g: c for g, c in param_counts.items() if g != "UNKNOWN"
                }
                if total_elements > 0:
                    return (total_elements * 2) / (1024 ** 3)
            finally:
                src.close()
        except Exception as exc:
            log.warning(
                "Could not count parameters for baseline size",
                model_path=model_path,
                error=str(exc),
            )

        # Last-resort fallback: file size (may be wrong for pre-quantized)
        p = Path(model_path)
        if p.is_file():
            return p.stat().st_size / (1024 ** 3)
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
            return total / (1024 ** 3)
        log.warning(
            "Could not estimate model size, using default",
            model_path=model_path,
            default_size_gb=1.0,
        )
        return 1.0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MagicQuant Orchestrator - Hybrid Quantization Search"
    )
    parser.add_argument("source_model", help="Path to source model (GGUF, safetensors, or directory)")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--target-quant", default="MXFP4_MOE")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3, help="Measurement rounds (0 = prediction only)")
    parser.add_argument("--candidates", type=int, default=4, help="Candidates to measure per round")
    parser.add_argument("--llamacpp-path", help="Path to llama.cpp directory")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")

    args = parser.parse_args()

    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.source_model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path,
        adapter_path=args.adapter,
    )

    if args.rounds > 0:
        all_configs, tiered = orchestrator.run_measured_search(
            target_base_quant=args.target_quant,
            search_generations=args.generations,
            measurement_rounds=args.rounds,
            candidates_per_round=args.candidates,
            verbose=True,
        )
    else:
        all_configs, tiered = orchestrator.run_full_search(
            target_base_quant=args.target_quant,
            max_generations=args.generations,
            verbose=True,
        )

    # Generate final survivors
    log.info("Generating final survivors", stage="generate")

    orchestrator.generate_tiered_models(
        tiered=tiered,
        model_name_prefix=Path(args.source_model).stem,
        tiers=["Q2", "Q4", "Q5", "Q6", "Q8"],
        verify=False,
    )


if __name__ == "__main__":
    main()
