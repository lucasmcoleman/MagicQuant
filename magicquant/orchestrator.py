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

import concurrent.futures
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.evolution.probing import SensitivityProber
from magicquant.gguf.tensor_groups import TensorGroupClassifier
from magicquant.gguf.writer import _is_quantization_candidate, is_block32_only_tensor
from magicquant.utils.naming import generate_name, config_key as _naming_config_key
from magicquant.utils.llamacpp import (
    LlamaCppTools,
    LlamaBinaryArchError,
    binary_supports_arch,
    resolve_source_gguf_arch,
)
from magicquant.utils.measurement import (
    imatrix_identity,
    measurement_eps,
    predictor_is_tracking,
)
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
        # Last successfully-resolved corpus path (see _safe_resolve_corpus)
        # -- kept so a pin-violation RuntimeError there can report this
        # instead of silently going to None.
        self._last_resolved_corpus: Optional[str] = None
        # 'supported' | 'unknown' | 'skipped', set by run_measured_search's
        # fail-fast arch check (see _run_arch_support_check below) right
        # after tools construction, before any real measurement. None
        # before any measured search has run (prediction-only run_full_search
        # never sets this). Persisted into search_results.json /
        # the measured checkpoint (_save_results / _write_measured_checkpoint).
        self._llamacpp_arch_check: Optional[str] = None

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
        # True once SensitivityProber.get_normalized_weights() had to fall
        # back to uniform weights for this run's probing (total sensitivity
        # <= 0 across every group) -- copied from
        # SensitivityProber.weights_degenerate right after probing. See
        # _enforce_probing_signal_gate.
        self.weights_degenerate: bool = False
        # Set by probing: fraction of the model (by parameter mass) whose
        # sensitivity was resolved above the measurement's own noise, and the
        # per-group resolution states behind it.
        self.resolved_mass_fraction: float = 0.0
        self.probe_resolutions: Dict[str, str] = {}
        # Groups probing found STRUCTURALLY unquantizable on this model
        # (every tensor forced to F32 by writer policy regardless of
        # scheme -- see SensitivityProber._detect_fixed_groups), copied
        # from SensitivityProber.fixed_groups / a resumed checkpoint right
        # after probing. Subtracted from self._search_groups before the
        # evolutionary search runs, so no candidate wastes a slot mutating
        # a scheme that can never take effect.
        self.fixed_groups: Dict[str, Dict] = {}
        self.predictor: Optional[PredictiveScorer] = None

        # Track all measured configs across rounds
        self._measured: Dict[str, Dict] = {}  # config_key -> {config, ppl, loss, path}

        # Per-group parameter counts, populated by _estimate_model_size and
        # fed to PredictiveScorer so MoE size/speed predictions use the real
        # (mostly-experts) distribution instead of the dense fallback.
        self._param_counts: Dict[str, int] = {}

        # Groups whose every quantization candidate on THIS model has a row
        # width that is 32- but not 256-divisible, so any K-quant assigned to
        # them is rewritten by the writer's block-size fallback. Populated by
        # _estimate_model_size (same pass, no extra I/O) and handed to
        # EvolutionarySurvivor, which uses it to offer Q5_0/Q5_1 for exactly
        # those groups. Empty for every ordinary 256-divisible model, and
        # empty means byte-identical behaviour to before this existed.
        self.block32_only_groups: set = set()
        # {group: {scheme: real_bpw}} for pairs whose actual cost differs from
        # the registry's advertised bpw once the writer's compat chain has run
        # -- handed to PredictiveScorer. Without this the predictor prices a
        # rewritten Q5_K at 5.5 bpw instead of the 8.5 it really costs, always
        # prefers it to Q5_0, and the block-32 schemes never get chosen.
        self._effective_bpw: Dict[str, Dict[str, float]] = {}

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
        # --kl-divergence-base, captured whenever run_measured_search's
        # probe_kl (default True) or enable_kl (default False) requested it
        # and capture succeeded. None means neither the sensitivity probes
        # (Step 2) nor the candidate objective have KL data available --
        # probes fall back to raw PPL and enable_kl's blend is a no-op.
        # HUGE on disk -- roughly chunks * ctx_size * vocab_size * 2 bytes
        # (fp16 per-token logits over the full vocabulary), verified 69 GB
        # for a 27B model at 100 chunks. run_measured_search deletes this
        # file on successful completion unless keep_kl_base_logits=True or
        # MAGICQUANT_KEEP_KL_LOGITS=1 is set; a failed/killed run always
        # leaves it in place so resume can reuse it. See
        # run_measured_search's keep_kl_base_logits docstring.
        self._kl_base_logits_path: Optional[str] = None
        # Corpus the base logits above were captured over -- every candidate's
        # KL calculation during the measured-search loop must reuse this
        # exact corpus to be comparable.
        self._kl_corpus_path: Optional[str] = None
        # Whether run_measured_search's enable_kl (candidate objective
        # blend) was requested for the LAST search this orchestrator ran.
        # False before any measured search has run. Recorded (alongside
        # _kl_weight below) so _current_measurement_conditions can tell a
        # PPL-only checkpoint from a KL-blended one -- see F2 /
        # _measurement_conditions_match.
        self._enable_kl: bool = False
        # Weight applied to |mean_kl| when blending KL into final-survivor
        # selection (see _select_final_survivors). Only has any effect when
        # a candidate actually has a "kl" measurement recorded. Set from
        # run_measured_search's enable_kl/kl_weight ONLY -- probe_kl never
        # touches this: probe-KL scoring and the candidate objective blend
        # are independent knobs (see run_measured_search's docstring).
        self._kl_weight: float = 0.0
        # Set at the top of run_measured_search's KL step: whether base-logits
        # capture was requested at all (probe_kl or enable_kl) and, if so,
        # whether the attempt failed (no corpus / capture pass failed / build
        # lacks --kl-divergence-base). Both False before any measured search
        # has run, or when neither knob requested capture. Read by
        # _enforce_probing_signal_gate to tell an operator whether KL probe
        # capture was attempted-and-failed vs never attempted.
        self._kl_capture_requested: bool = False
        self._kl_capture_failed: bool = False
        # When True, _select_final_survivors re-ranks WITHIN a tier's
        # near-tied candidates (raw measured_loss within self._speed_epsilon
        # relative of the tier's best) by size_gb (default "bytes" metric) or
        # measured tg throughput ("bench"), instead of a bare argmin over the
        # KL-blended score. ON by default since the 2026-07 fix (see
        # run_measured_search's speed_aware/speed_epsilon params): a real
        # ThinkingCap run shipped a "Q5" winner that was BOTH larger and had
        # HIGHER raw PPL than a candidate it beat, purely because the KL
        # tiebreak tipped a near-noise-level PPL difference -- selection was
        # size-blind. Set speed_aware=False to restore the pre-fix bare
        # argmin-over-KL-blended-score behavior. See _speed_aware_pick.
        self._speed_aware: bool = True
        # None resolves lazily in _select_final_survivors to
        # magicquant.utils.measurement.measurement_eps(self.baseline_ppl, ...)
        # -- the same measurement-noise tolerance used elsewhere (
        # DEFAULT_RELATIVE_EPS=0.05, REPORTED_ERR_SIGMAS=3), rather than an
        # invented flat 0.5% that turned out to be too tight to absorb real
        # measurement noise (see run_measured_search's speed_epsilon
        # docstring for the ThinkingCap numbers that motivated this).
        self._speed_epsilon: Optional[float] = None
        # Metric _speed_aware_pick re-ranks near-tied candidates by: "bytes"
        # (size_gb, default) or "bench" (measured tg throughput). Overridden
        # by run_measured_search's speed_metric param. See the getattr
        # fallback at _select_final_survivors, which intentionally keeps a
        # DIFFERENT default ("bytes") for bare-__new__ test doubles that never
        # ran __init__ -- both defaults agree, so this is purely additive.
        self._speed_metric: str = "bytes"

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

    _ALLOW_DEGENERATE_PROBING_ENV = "MAGICQUANT_ALLOW_DEGENERATE_PROBING"
    # BLOCKER fix: the KL base-logits file (self._kl_base_logits_path) is
    # enormous -- roughly chunks * ctx_size * vocab_size * 2 bytes (fp16
    # per-token logits over the full vocabulary), verified at 69 GB for a
    # 27B model at 100 chunks -- and, with probe_kl defaulting True, every
    # measured run now creates one. Setting this env var to exactly "1"
    # keeps the file after a successful run instead of deleting it (same
    # override style as _ALLOW_DEGENERATE_PROBING_ENV above). See
    # run_measured_search's keep_kl_base_logits parameter.
    _KEEP_KL_LOGITS_ENV = "MAGICQUANT_KEEP_KL_LOGITS"

    def _kl_probe_capture_note(self) -> str:
        """Build the "how does KL probe scoring stand for this run" clause
        used by ``_enforce_probing_signal_gate``'s message -- distinguishes
        "KL was active and still couldn't resolve a signal" from "KL was
        attempted but capture failed" from "KL was never attempted", so an
        operator reading the failure knows which knob (if any) to look at
        instead of always being pointed at ``enable_kl`` regardless of what
        actually happened. Uses ``getattr`` throughout: called on orchestrator
        instances built via ``__new__`` in tests, which never ran
        ``__init__`` or ``run_measured_search`` and so carry none of these
        attributes.
        """
        if getattr(self, "_kl_base_logits_path", None):
            return (
                "KL probe scoring WAS active for this run's sensitivity "
                "probes (base logits were captured) and still could not "
                "resolve a signal -- check the llama.cpp build, corpus, and "
                "baseline measurement rather than probe_kl/enable_kl."
            )
        if getattr(self, "_kl_capture_requested", False) and getattr(
            self, "_kl_capture_failed", False
        ):
            return (
                "KL probe capture was ATTEMPTED for this run (probe_kl "
                "and/or enable_kl) but FAILED -- see the 'kl' warning logged "
                "earlier in this run. Fix the calibration corpus or "
                "llama.cpp build (needs --kl-divergence-base support) so "
                "probes can use KL scoring, whose per-token error is fine "
                "enough to resolve a dense group's sub-percent probe delta "
                "that raw PPL's ~2% chunk-mean error cannot."
            )
        return (
            "KL probe scoring was NEVER ATTEMPTED for this run (probe_kl "
            "and enable_kl were both off) -- on DENSE models this is the "
            "expected outcome of raw-PPL probe scoring: PPL's reported "
            "error is the spread over chunk means (~2% of baseline), while "
            "a dense group's whole-group probe delta is sub-percent, so no "
            "probe can resolve. Pass probe_kl=True (the default) to "
            "``run_measured_search`` so probes use KL-divergence scoring "
            "instead."
        )

    def _enforce_probing_signal_gate(self) -> None:
        """MAJOR 4: refuse to proceed on a search whose sensitivity probing
        came back with no reliable signal.

        Before this fix, ``probing_provenance == "suspect"`` (more than
        half the probed groups' MEASURED probes were physically-impossible
        clamped readings -- see ``probing.py``'s ``probe_all_groups``) and
        ``weights_degenerate`` (every sensitivity was <=0, so the search
        fell back to uniform 1/N weights -- see ``get_normalized_weights``)
        were WRITE-ONLY: stamped into ``sensitivity.json``/
        ``search_results.json`` for a human to notice later, but nothing
        actually gated on them. A search positively identified as
        signal-less still completed and shipped tiers exactly like a
        healthy measured run.

        Raises loudly by default. Set
        ``MAGICQUANT_ALLOW_DEGENERATE_PROBING=1`` to proceed anyway (e.g. a
        deliberate re-run against a known-flat/tiny model where a uniform
        weighting is expected, not a symptom of a broken measurement).
        """
        degenerate = (
            self.probing_provenance in ("suspect", "insufficient")
            or self.weights_degenerate
        )
        if not degenerate:
            return

        if os.environ.get(self._ALLOW_DEGENERATE_PROBING_ENV) == "1":
            log.warning(
                "Sensitivity probing produced no reliable signal but "
                f"{self._ALLOW_DEGENERATE_PROBING_ENV}=1 is set -- "
                "proceeding anyway. This search's tiers rank candidates on "
                "uniform/noise-floor weights, not real per-group "
                "sensitivity -- disclose this before shipping.",
                stage="probing",
                probing_provenance=self.probing_provenance,
                weights_degenerate=self.weights_degenerate,
            )
            return

        raise RuntimeError(
            "Sensitivity probing produced no reliable signal for this "
            f"search (probing_provenance={self.probing_provenance!r}, "
            f"weights_degenerate={self.weights_degenerate}): more than half "
            "the probed groups' measured probes were physically-impossible "
            "clamped readings, or every group's sensitivity was <=0. "
            "Proceeding would rank the evolutionary search's candidates on "
            "uniform/noise-floor weights while still shipping tiers as if "
            "this were a healthy measured search. "
            f"{self._kl_probe_capture_note()} "
            "Otherwise check the llama.cpp build, corpus, and baseline "
            "measurement, or set "
            f"{self._ALLOW_DEGENERATE_PROBING_ENV}=1 to proceed anyway."
        )

    # Default precision:size ratio score_hybrid's own weights carry
    # (0.50:0.35) -- _build_objective_weights preserves this ratio while
    # renormalizing the remainder after reserving a caller-chosen speed_weight.
    _DEFAULT_PRECISION_WEIGHT = 0.50
    _DEFAULT_SIZE_WEIGHT = 0.35

    @classmethod
    def _build_objective_weights(
        cls, speed_weight: Optional[float]
    ) -> Optional[Tuple[float, float, float]]:
        """Build a (precision, size, speed) objective_weights tuple for
        ``EvolutionarySurvivor`` from a single ``speed_weight`` knob.

        Reserves ``speed_weight`` for the speed term and renormalizes
        precision/size to fill the remainder while preserving their default
        0.50:0.35 ratio -- e.g. ``speed_weight=0.40`` gives about
        ``(0.35, 0.25, 0.40)``. Returns ``None`` when ``speed_weight`` is
        ``None`` (the default), which leaves ``EvolutionarySurvivor``'s own
        default -- ``score_hybrid``'s fixed 0.50/0.35/0.15 weights -- in
        effect, unchanged from historical behavior.
        """
        if speed_weight is None:
            return None
        # Clamp to [0, 1]: a speed_weight > 1 would make remainder negative and
        # yield negative precision/size weights (nonsensical scoring, not a
        # crash). Guards a typo'd --speed-weight (Opus review, 2026-07-05).
        speed_weight = min(1.0, max(0.0, speed_weight))
        remainder = 1.0 - speed_weight
        ratio_total = cls._DEFAULT_PRECISION_WEIGHT + cls._DEFAULT_SIZE_WEIGHT
        precision_weight = remainder * (cls._DEFAULT_PRECISION_WEIGHT / ratio_total)
        size_weight = remainder * (cls._DEFAULT_SIZE_WEIGHT / ratio_total)
        return (precision_weight, size_weight, speed_weight)

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
        from magicquant.imatrix import ensure_imatrix, resolve_imatrix_bin

        # Default llama-imatrix to the sibling of the discovered perplexity
        # binary: ensure_imatrix's own fallback is a PATH lookup, which can
        # resolve to a DIFFERENT llama.cpp build than llamacpp_path -- e.g. a
        # stock brew install that can't load an arch only the configured fork
        # supports (bit for real on a qwen35 MTP model, 2026-07-04).
        if "imatrix_bin" not in kwargs:
            resolved = resolve_imatrix_bin(self.llama_tools)
            if resolved:
                kwargs["imatrix_bin"] = resolved

        # Never calibrate on the text the run is SCORED against. Doing so
        # tunes quantization to the eval set and every measured_loss comes
        # back optimistic with nothing in the output hinting why. The two
        # default to different corpora, but nothing enforced it until now --
        # one imatrix_corpus pointed at wikitext would silently invalidate a
        # whole search's numbers.
        if corpus_path is not None:
            try:
                cal = Path(corpus_path).resolve()
                tools = self.llama_tools
                pinned = getattr(tools, "_pinned_corpus", None) if tools else None
                if pinned and Path(pinned).resolve() == cal:
                    log.error(
                        "refusing imatrix: the calibration corpus is the same "
                        "file as the perplexity eval corpus, which would make "
                        "every measured loss optimistic. Point imatrix_corpus "
                        "at different text, or leave it unset for the bundled "
                        "default.",
                        stage="imatrix", corpus=str(cal),
                    )
                    self._imatrix = None
                    return False
            except (OSError, ValueError):
                pass    # unresolvable path: let ensure_imatrix report it

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

    def _current_llamacpp_binary(self) -> Optional[str]:
        """The resolved perplexity-tool path for THIS run's LlamaCppTools
        instance, or None (prediction-only run / bare test double with no
        ``_llama_tools`` / a fake lacking the attribute). Read via
        ``getattr`` off ``self._llama_tools`` directly -- never via the
        ``llama_tools`` property -- so persisting results/checkpoint state
        can never trigger a lazy ``LlamaCppTools()`` auto-construction as a
        side effect.
        """
        return getattr(getattr(self, "_llama_tools", None), "perplexity_tool", None)

    def _run_arch_support_check(self) -> str:
        """Fail-fast pre-measurement check (t+0, before any real
        measurement subprocess runs): does the resolved perplexity tool's
        libllama actually contain the source model's GGUF architecture
        literal? See ``utils.llamacpp.binary_supports_arch`` for the
        ground-truth probe and the field incident this fixes -- multiple
        llama.cpp builds coexisting on one box, where a measured search
        auto-resolved to a build lacking the arch and died at baseline
        40+ minutes in with "unknown model architecture".

        Only applies when the source is a readable GGUF with an
        architecture key (see ``resolve_source_gguf_arch`` -- a
        safetensors source, or any source this reader can't parse, skips
        the check with a debug note; safetensors sources go through
        ``create_hybrid_gguf``'s own conversion path and this check has no
        equivalent signal to probe there).

        Escape hatch: ``MAGICQUANT_SKIP_ARCH_CHECK=1`` skips with a logged
        warning.

        Returns 'supported' | 'unknown' | 'skipped' -- the value
        ``run_measured_search`` persists as ``llamacpp_arch_check``.
        Raises ``LlamaBinaryArchError`` when the literal is PROVABLY
        absent (verdict False): the resolved binary cannot load this
        model.
        """
        if os.environ.get("MAGICQUANT_SKIP_ARCH_CHECK") == "1":
            log.warning(
                "MAGICQUANT_SKIP_ARCH_CHECK=1 -- skipping the "
                "pre-measurement llama.cpp architecture check",
                stage="init",
            )
            return "skipped"

        arch = resolve_source_gguf_arch(self.source_model_path)
        if arch is None:
            return "skipped"

        perplexity_tool = self._current_llamacpp_binary()
        # binary_supports_arch itself now guards a missing/empty
        # perplexity_tool (returns None) -- no need to short-circuit here.
        verdict = binary_supports_arch(perplexity_tool, arch)
        if verdict is False:
            raise LlamaBinaryArchError(
                f"The resolved llama.cpp binary ({perplexity_tool!r}) does "
                f"not contain the GGUF architecture literal {arch!r} -- it "
                "cannot load this model. This is a PRE-MEASUREMENT check "
                "(runs before any llama-perplexity subprocess); catching "
                "it now, at t+0, just saved an hours-long measured search "
                "from dying at baseline with llama.cpp's own 'unknown "
                "model architecture' error after burning real compute for "
                "zero measurements. Point llamacpp_path (or the "
                "MAGICQUANT_LLAMACPP_PATH env var) at a llama.cpp build "
                f"that supports {arch!r}, or set "
                "MAGICQUANT_SKIP_ARCH_CHECK=1 to bypass this check."
            )
        if verdict is None:
            log.debug(
                "arch pre-check: could not verify -- proceeding "
                "unverified",
                stage="init", perplexity_tool=perplexity_tool, arch=arch,
            )
            return "unknown"
        return "supported"

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
        head_aggressive: bool = False,
        stream_aware: bool = False,
        seed: Optional[int] = None,
        use_imatrix: bool = False,
        imatrix_corpus: Optional[str] = None,
        probe_kl: bool = True,
        enable_kl: bool = False,
        kl_weight: float = 0.1,
        keep_kl_base_logits: bool = False,
        enable_speed_bench: bool = False,
        speed_aware: bool = True,
        speed_epsilon: Optional[float] = None,
        speed_metric: str = "bytes",
        measurement_chunks: Optional[int] = None,
        seed_incumbents: bool = True,
        resume: bool = True,
        speed_weight: Optional[float] = None,
        use_bytes_tps: bool = False,
        write_calibration: bool = False,
        calibration_source: str = "",
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
            probe_kl: capture base-model reference logits (via
                llama-perplexity's --kl-divergence-base) and use them to
                score Step 2's per-group SENSITIVITY PROBES by KL-divergence
                instead of raw PPL. On by default. This is the fix for dense
                models: PPL's reported error is the spread over chunk means
                (~2% of baseline), while a dense group's whole-group probe
                delta is sub-percent, so raw-PPL probes can't resolve it and
                ``_enforce_probing_signal_gate`` kills the run. KL's per-token
                error is fine enough to resolve them (measured on one real
                probe: 79 sigma vs 0.55 sigma). Independent of ``enable_kl``:
                probe_kl affects ONLY how probes are scored in Step 2, never
                the evolutionary search's candidate objective -- turning
                probe_kl off does not disable ``enable_kl``'s blend, and
                turning probe_kl on (the default) does not, by itself, blend
                KL into candidate selection. Base-logits capture is attempted
                whenever ``probe_kl or enable_kl`` -- the two knobs share one
                capture pass so a run wanting both never pays for it twice.
                A capture failure while ONLY probe_kl requested it (no
                calibration corpus, a llama.cpp build without
                --kl-divergence-base, or the capture pass itself failing)
                logs a warning and falls back to raw-PPL probe scoring --
                ``_enforce_probing_signal_gate`` remains the backstop against
                a run that then can't resolve anything. A capture failure
                while ``enable_kl=True`` was explicitly requested keeps the
                historical warn-and-skip behavior (KL scoring disabled for
                the whole run, no exception) -- see ``enable_kl`` below.
            enable_kl: also measure real KL-divergence-to-base for each
                candidate (via llama-perplexity's built-in --kl-divergence)
                and blend it into final-survivor selection (candidate
                OBJECTIVE, not probe scoring -- see ``probe_kl`` above). Off
                by default. Setting this True also implies ``probe_kl``'s
                base-logits capture (they share the one capture pass) even
                if ``probe_kl=False`` was passed explicitly.
            kl_weight: weight applied to |mean_kl| when blending into
                selection (see _select_final_survivors); only meaningful
                when enable_kl=True. Never affects probe scoring.
            keep_kl_base_logits: keep the captured KL base-logits file
                (``<output_dir>/_kl_base_logits.kld``) on disk after a
                SUCCESSFUL run instead of deleting it. This file is
                enormous -- roughly ``chunks * ctx_size * vocab_size * 2``
                bytes (fp16 per-token logits over the full vocabulary),
                verified at 69 GB for a 27B model at 100 chunks -- and with
                ``probe_kl`` defaulting True, every measured run now
                creates one. False (the default) deletes it once the run
                completes successfully (results saved, checkpoint deleted --
                nothing left that could resume from it). A run that fails
                or is killed always leaves the file in place regardless of
                this flag, since a resume needs it to skip the ~18-min
                recapture (see ``resume``'s KL base-logits reuse). Also
                honored via the ``MAGICQUANT_KEEP_KL_LOGITS=1`` env var
                (exact string ``"1"``, same convention as
                ``MAGICQUANT_ALLOW_DEGENERATE_PROBING``) for callers that
                can't easily thread a new kwarg through (e.g. a CLI/UI
                wrapper) -- either one keeps the file.
            enable_speed_bench: also measure real tokens/sec per candidate
                via llama-bench (informational; recorded in search_results
                .json, not fed into per-generation prediction scoring --
                bench-ing the full population every generation isn't
                tractable, only the small measured set is). Off by default.
            head_aggressive: bias the evolutionary search's random-config
                sampling for the 'H' (LM head / output.weight) group only
                toward the smaller K-quants (Q6_K/Q5_K/Q8_0) and away from
                BF16 -- see ``EvolutionarySurvivor.__init__`` and
                ``_HEAD_AGGRESSIVE_CLASS_WEIGHTS`` for the rationale
                (output.weight streams in full every tg token, so its
                precision is a bandwidth tax the PPL objective never sees).
                A bias, not a hard exclusion. Off by default (unchanged
                sampling for every group, including H).
            speed_aware: within each tier, stop taking a bare argmin over the
                KL-blended selection score. Instead, find the tier's best
                RAW ``measured_loss`` (a Pareto/knee rule over
                (measured_loss, size_gb), not the KL-blended score -- KL can
                still legitimately separate two candidates whose PPL is
                genuinely different, but it must not be the thing that
                decides a near-noise-level PPL tie), band every candidate
                within ``speed_epsilon`` relative of that best, and among
                the band prefer the smallest ``size_gb`` (default
                ``speed_metric="bytes"``) or, when candidates carry bench
                data (``enable_speed_bench``), the best measured tg
                throughput (``speed_metric="bench"``) -- see
                ``_select_final_survivors`` / ``_speed_aware_pick``. Pure
                post-hoc re-ranking of already-recorded measurements -- no
                extra GPU calls.
                ON by default since the 2026-07 fix: a real ThinkingCap run
                shipped a "Q5" tier winner that was BOTH larger (20.89 GB vs
                17.68 GB) AND had HIGHER raw PPL (6.827036 vs 6.826419) than
                a candidate in the same tier -- it won purely because the KL
                tiebreak (kl_weight * |mean_kl|) tipped a PPL difference
                (~0.009%) that was two orders of magnitude below the run's
                own reported measurement error (+/- ~1.6%). Selection was
                size-blind and let noise-level KL swings override a real,
                smaller, better-PPL candidate. Pass ``speed_aware=False`` to
                restore the pre-fix bare argmin-over-KL-blended-score
                behavior.
            speed_epsilon: relative tolerance (applied as
                ``epsilon * abs(best_raw_measured_loss)``) defining
                "near-tied" for ``speed_aware``. ``None`` (the default)
                resolves lazily, per tier, once each tier's quality-winner
                is known, to
                ``magicquant.utils.measurement.measurement_eps(baseline_ppl,
                reported_err)`` -- the SAME measurement-noise tolerance used
                by the "is this PPL reading physically plausible" guard
                elsewhere in this module (``DEFAULT_RELATIVE_EPS=0.05``,
                ``REPORTED_ERR_SIGMAS=3``), rather than a second, invented
                tolerance. ``reported_err`` is that tier's quality-winner's
                own fused-KL-pass ``ppl_err`` when one was measured (same
                threading as the per-candidate ``measurement_invalid``
                check's ``kl_result.get("ppl_err")``), so the 3-sigma path
                actually gets exercised on real runs instead of always
                falling back to the flat default; it falls back to
                ``DEFAULT_RELATIVE_EPS`` when no such measurement is
                reachable (e.g. ``enable_kl=False``). The historical flat
                0.005 (0.5%) was too tight to absorb real measurement noise
                -- see the ThinkingCap numbers above, where the "near tied"
                PPL gap was itself only ~0.009% but still fell entirely
                within normal run-to-run jitter. Only meaningful when
                ``speed_aware=True``; pass an explicit float to pin a fixed
                tolerance instead of the noise-derived default.
                IMPORTANT: widening this epsilon is NOT what fixes the
                ThinkingCap bug above -- that candidate pair's raw PPL gap
                (~0.0091% of baseline in measured_loss units) sits
                well inside even the old, tighter 0.005
                (0.5%) band, so any reasonable epsilon puts them "near
                tied". What actually fixes it is banding on RAW
                ``measured_loss`` (this function's ``score_of`` param, see
                ``_speed_aware_pick``) instead of the KL-blended
                ``_selection_score`` -- the old bug was that KL, not PPL,
                decided the tier winner. Do not "tune" this epsilon
                expecting it to change tier-winner selection by itself.
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
                /corpus/KL objective/imatrix identity -- see
                ``_current_measurement_conditions``). If it matches, restore
                the baseline PPL,
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
            speed_weight: reserve this weight for the search's speed
                objective, renormalizing precision:size to fill the
                remainder while keeping their default 0.50:0.35 ratio (e.g.
                speed_weight=0.40 gives about 0.35/0.25/0.40). Built into an
                ``objective_weights`` tuple passed to ``EvolutionarySurvivor``
                (see ``_build_objective_weights``). ``None`` (default) means
                today's fixed 0.50/0.35/0.15 weights, unchanged -- required
                for the seed-pinned regression fixture.
            use_bytes_tps: score the search's speed term deterministically
                from predicted size (``PredictiveScorer.score_hybrid``'s
                bandwidth-bound proxy) instead of the noisy per-scheme
                speed_multiplier path. Off by default (unchanged scoring).
            write_calibration: after a successful measured search, fit
                per-scheme noise factors from THIS run's measurements +
                sensitivity weights (mirrors ``tools/fit_noise_factors.py``)
                and write ``<output_dir>/noise_calibration.json`` in the
                nested envelope ``magicquant.quant.calibration`` reads. Off
                by default; best-effort (a fitting failure is logged and
                never blocks a successful search from completing).
            calibration_source: load empirically calibrated noise
                factors/speed multipliers from this file instead of the
                fixed ``tools/calibration_results.json`` path (see
                ``magicquant.quant.calibration``). Passed straight through
                to the ``PredictiveScorer`` this search constructs. ``""``
                (default) means today's fixed-path lookup, unchanged.

        Returns:
            (all_configs, tiered_best) where tiered_best maps tier names
            to the best *measured* config for that tier.
        """
        self._apply_seed(seed)
        # Recorded (in addition to self._kl_weight) so
        # _current_measurement_conditions can distinguish "blending off" from
        # "blending on with weight 0" -- see F2's _measurement_conditions_match.
        self._enable_kl = enable_kl
        self._kl_weight = kl_weight if enable_kl else 0.0
        self._speed_aware = speed_aware
        self._speed_epsilon = speed_epsilon
        self._speed_metric = speed_metric
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

        # ── Fail-fast arch check + instrument visibility (t+0, before any
        # real measurement -- see _run_arch_support_check / the
        # multi-build-coexistence field incident it fixes). Auto-detect
        # warning: an unpinned llamacpp_path can silently resolve to a
        # DIFFERENT build across submission paths/runs -- flag it for any
        # measured run. ──
        self._llamacpp_arch_check = self._run_arch_support_check()
        log.info(
            "Resolved llama.cpp tools",
            stage="init",
            perplexity_tool=self._current_llamacpp_binary(),
            bench_tool=getattr(self._llama_tools, "bench_tool", None),
            arch_check=self._llamacpp_arch_check,
        )
        if self._llamacpp_path is None:
            log.warning(
                "llamacpp_path was not pinned -- LlamaCppTools auto-"
                "resolved a build on its own. For a measured run, pin an "
                "explicit llamacpp_path (or MAGICQUANT_LLAMACPP_PATH) so "
                "the instrument can't drift to a different coexisting "
                "build across runs.",
                stage="init",
                resolved_llamacpp_path=getattr(
                    self._llama_tools, "llamacpp_path", None
                ),
            )

        # ── Optional imatrix capture -- deliberately BEFORE the checkpoint
        # resume gate below, not after: ``_current_measurement_conditions``
        # (issue #5) fingerprints ``self._imatrix`` to decide whether a
        # checkpoint's sensitivity weights + measured candidates were built
        # under the SAME calibration state this run wants. Capturing it
        # first means that fingerprint reflects this run's real imatrix
        # state at gate-check time instead of always reading "inactive"
        # (self._imatrix's __init__ default) regardless of ``use_imatrix``.
        # Best-effort: a capture failure degrades to the historical
        # unweighted-quant behavior rather than aborting a real measured
        # search over a secondary quality signal.
        if use_imatrix:
            # enable_imatrix -> ensure_imatrix already caches capture to disk
            # and reuses it on a hit, so calling this ahead of resume is
            # cheap when the cache survived and correctly recomputes when it
            # didn't -- no separate resume bookkeeping needed for imatrix
            # capture itself.
            self.enable_imatrix(imatrix_corpus)

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
        # whenever ``probe_kl`` (the default) or ``enable_kl`` succeeds,
        # instead of two.
        baseline_needs_standalone_measurement = True
        if checkpoint is not None:
            self.baseline_ppl = checkpoint["baseline_ppl"]
            self.baseline_provenance = checkpoint["baseline_provenance"]
            # Pre-v2 checkpoints predate the measurement-validity flag AND
            # the strict perplexity parser, so their per-candidate readings may
            # contain fabricated values (a parsed progress-line timing) with no
            # way to tell them apart -- info.get("measurement_invalid") is
            # simply absent, reads falsy, and such an entry can win a tier. The
            # BASELINE is kept (it is a single expensive measurement and is
            # re-validated by the source-identity + conditions match above);
            # only the candidate measurements are discarded.
            ck_version = checkpoint.get("version", 1)
            if ck_version >= 2:
                for key, entry in checkpoint.get("measured", {}).items():
                    self._measured[key] = dict(entry)
            elif checkpoint.get("measured"):
                log.warning(
                    "Discarding pre-v2 checkpoint measurements (no validity "
                    "flag; may predate the strict PPL parser) -- baseline kept",
                    stage="resume", version=ck_version,
                    discarded=len(checkpoint.get("measured", {})),
                )
            baseline_needs_standalone_measurement = False
            if verbose:
                log.info(
                    "Resumed baseline + measurements from checkpoint",
                    stage="resume", path=str(checkpoint_path),
                    measured=len(self._measured),
                )

        # ── Step 1b: optional KL base logits (fuses in the baseline ──
        # ── measurement on a fresh run, see above). Imatrix capture already
        # ── happened earlier, ahead of the checkpoint resume gate. ──
        # Best-effort: a failure here degrades to the historical raw-PPL
        # probe scoring / no KL objective blend rather than aborting a real
        # measured search over a secondary quality signal.
        # Base-logits capture is attempted whenever EITHER knob wants it --
        # probe_kl (Step 2's per-group sensitivity probes, on by default) or
        # enable_kl (the candidate objective blend, off by default). They
        # share this one capture pass; which knob(s) requested it only
        # changes how a capture FAILURE is handled below, not whether the
        # attempt happens. Recorded on self so _enforce_probing_signal_gate
        # can later report whether KL probe capture was attempted-and-failed
        # vs never attempted at all.
        self._kl_capture_requested = bool(probe_kl or enable_kl)
        self._kl_capture_failed = False
        # Reset per run: without this, a second run on the same orchestrator
        # instance whose own capture failed would inherit the previous run's
        # pointer (probes scored against a stale baseline), and a
        # probe_kl=False + enable_kl=False run would skip the capture block
        # entirely and leave the attribute at whatever it was.
        self._kl_base_logits_path = None
        self._kl_corpus_path = None
        if self._kl_capture_requested:
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
                    self._kl_capture_failed = True
                    if enable_kl:
                        # Historical behavior, unchanged: enable_kl's own
                        # capture failure just disables the objective blend
                        # for this run rather than aborting a real measured
                        # search over a secondary quality signal.
                        log.warning(
                            "enable_kl requested but no calibration corpus resolved "
                            "-- skipping KL-divergence scoring", stage="kl",
                        )
                    else:
                        log.warning(
                            "probe_kl requested KL-scored sensitivity probing "
                            "but no calibration corpus resolved -- probes will "
                            "fall back to raw-PPL scoring for this run "
                            "(dense-model sensitivity may not resolve; see "
                            "_enforce_probing_signal_gate)", stage="kl",
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
                        self._kl_capture_failed = True
                        if enable_kl:
                            # Historical behavior, unchanged (see above).
                            log.warning(
                                "Could not save base logits -- disabling KL-divergence "
                                "scoring for this run", stage="kl",
                            )
                        else:
                            log.warning(
                                "probe_kl requested KL-scored sensitivity probing "
                                "but the base-logits capture pass failed (check "
                                "the llama.cpp build supports --kl-divergence-base) "
                                "-- probes will fall back to raw-PPL scoring for "
                                "this run", stage="kl",
                            )

        # ── Step 1c: standalone baseline measurement ──
        # Skipped when the checkpoint already restored it, or Step 1b fused
        # it in above. This is the historical baseline pass, unchanged --
        # taken whenever probe_kl and enable_kl are both off, or their fused
        # attempt didn't pan out (no corpus / save failure), matching the
        # pre-fusion behavior.
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
        groups = self._detect_search_groups()

        # A checkpoint can legitimately carry a baseline but NO sensitivities:
        # a run killed between the baseline measurement and probing, or a
        # deliberately injected baseline (measuring an oversized source once,
        # out-of-band, so the search itself never has to load it). Restoring a
        # null unconditionally used to skip probing and then crash in the
        # predictor with `'NoneType' object has no attribute 'get'`, which
        # reads as a corrupt checkpoint rather than a missing-probe state.
        # Treat absent/empty weights as "not restored" and probe normally.
        if checkpoint is not None and checkpoint.get("sensitivity_weights"):
            self.sensitivity_weights = checkpoint["sensitivity_weights"]
            self.probing_provenance = checkpoint["probing_provenance"]
            self.weights_degenerate = checkpoint.get("weights_degenerate", False)
            self.fixed_groups = checkpoint.get("fixed_groups", {})
            if verbose:
                log.info(
                    "Resumed sensitivity weights from checkpoint", stage="resume",
                )
        else:
            if verbose:
                log.info("Sensitivity probing", stage="probing")

            # Populate self._param_counts before probing so the prober can
            # judge its coverage by parameter MASS rather than group count --
            # three of nine groups resolving meant 1.6% of the model on
            # Laguna-S. (Header-only read; Step 3 calls this again for size.)
            self._estimate_model_size(self.source_model_path)

            prober = SensitivityProber(
                base_model_path=self.source_model_path,
                baseline_perplexity=self.baseline_ppl,
                perplexity_calculator=self.llama_tools,
                output_dir=str(self.output_dir / "_probes"),
                parameter_counts=self._param_counts,
                # When this run saved reference logits (enable_kl), score
                # probes by KL divergence instead of perplexity. Perplexity's
                # error is the spread over chunk means and cannot separate a
                # single-group probe's effect from zero; KL's is per token.
                # Measured on one real probe: 79 sigma vs 0.55 sigma.
                kl_base_logits_path=self._kl_base_logits_path,
                kl_corpus_path=self._kl_corpus_path,
                # A MEASURED search must never silently rank candidates on
                # fabricated (heuristic) sensitivities: a failed probe now
                # raises ProbeMeasurementError after a retry instead of
                # poisoning the whole run. run_full_search (prediction-only)
                # keeps the heuristic fallback — there it's the documented
                # design, not a degradation. See probing.ProbeMeasurementError.
                strict=True,
                # Probes must be measured under the SAME calibration state
                # as the candidates and final tiers they steer (issue #5) --
                # self._imatrix is already active by this point (captured
                # ahead of the checkpoint-resume gate, above).
                imatrix=self._imatrix,
            )
            prober.probe_all_groups(groups=groups, aggressive_scheme="Q4_K_M", verbose=verbose)
            self.sensitivity_weights = prober.get_normalized_weights()
            self.probing_provenance = prober.probing_provenance
            self.weights_degenerate = prober.weights_degenerate
            # Fraction of the model, by parameter mass, whose sensitivity the
            # probes actually resolved. Surfaced so a reader of
            # search_results.json can tell a search steered by real signal
            # from one steered by rounding.
            self.resolved_mass_fraction = prober.resolved_mass_fraction
            self.probe_resolutions = dict(prober.resolutions)
            self.fixed_groups = dict(prober.fixed_groups)
            prober.save_results(str(self.output_dir / "sensitivity.json"))

            if verbose:
                log.info(
                    "Sensitivity weights computed",
                    stage="probing",
                    weights={g: round(w, 3) for g, w in self.sensitivity_weights.items()},
                )

        # Structurally-fixed groups (see SensitivityProber._detect_fixed_groups)
        # can never take a scheme other than what the writer already forces
        # (F32) -- drop them from the mutable set BEFORE incumbent seeding /
        # run_evolution so no candidate wastes a slot on a choice that can
        # never take effect. Applies whether the weights above came from a
        # fresh probe or a resumed checkpoint.
        self._exclude_fixed_groups(verbose)

        # MAJOR 4: a search whose probing came back with no reliable signal
        # (suspect provenance / degenerate uniform weights) must not
        # silently proceed and ship tiers as if it were healthy.
        self._enforce_probing_signal_gate()

        # Baseline + probing are complete (whether resumed or freshly
        # measured) -- checkpoint now so a kill during Step 4 can resume
        # past both without re-running either.
        self._write_measured_checkpoint(checkpoint_path)

        # ── Step 3: Initialize predictor ──
        # (_estimate_model_size also populates self._param_counts per group.)
        baseline_size_gb = self._estimate_model_size(self.source_model_path)

        self.predictor = self._build_predictor(calibration_source, baseline_size_gb)

        # ── Step 3b: incumbent seeding ──
        # Build llama.cpp's own Q4_K_M/Q5_K_M/Q6_K mixtures (restricted to the
        # groups this search actually varies) so the evolutionary search is
        # anchored to "what stock llama-quantize would have done anyway" --
        # see magicquant.incumbents' module docstring for why this matters.
        seed_configs, incumbent_tier_by_key = self._build_incumbent_seeds(
            seed_incumbents
        )

        # Tunable speed objective (opt-in, see speed_weight's docstring):
        # built once, reused every round below. None when speed_weight is
        # unset, leaving EvolutionarySurvivor's own default (score_hybrid's
        # fixed weights) in effect.
        objective_weights = self._build_objective_weights(speed_weight)

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
                # Derived from the LOADED imatrix, not the use_imatrix request:
                # capture degrades gracefully (see enable_imatrix above), so a
                # requested-but-missing imatrix must not unlock IQ4_NL.
                has_imatrix=self._imatrix is not None,
                head_aggressive=head_aggressive,
                stream_aware=stream_aware,
                objective_weights=objective_weights,
                use_bytes_tps=use_bytes_tps,
                block32_only_groups=self.block32_only_groups,
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

            # 4c. Build, measure, learn.
            #
            # One-ahead pipeline: candidate i+1's CPU-bound GGUF build runs
            # on a single background thread while candidate i's GPU-bound
            # measurement subprocess runs on the main thread. Everything
            # that touches shared state -- self._measured, predictor
            # feedback, checkpoint writes -- stays on the main thread, in
            # the same order as the historical serial loop; only the BUILD
            # step moves off it, one candidate ahead.
            #
            # Thread-safety: a background build only reads self._imatrix
            # (a plain dict of numpy arrays populated once before this loop
            # and never mutated afterward -- safe to read concurrently) and
            # writes to an output path unique to its own candidate index. No
            # other shared mutable state is touched by _build_candidate.
            # Prefetching candidate i+1's build while candidate i is being
            # MEASURED stacks a concurrent CPU build's working set on top of
            # the measurement subprocess's full model load. On a big model /
            # small box that sum OOMs (killed a real 27B run at candidate 6/7
            # on this 124GB unified-memory box, 2026-07-05). Only overlap when
            # the source model is small enough that a concurrent build fits
            # alongside the measurement -- otherwise build serially (peak =
            # max(build, measure), never their sum).
            overlap_builds = self._should_overlap_builds()
            if verbose:
                log.info(
                    "Build/measure overlap",
                    stage="measurement",
                    enabled=overlap_builds,
                )

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

            def _submit_build(idx: int):
                """Submit candidate idx's build unless it's out of range or
                already measured (dedupe BEFORE submitting, same as the
                historical serial loop). Returns (future, config, config_key)
                or None."""
                if idx >= len(to_measure):
                    return None
                cfg = to_measure[idx]["config"]
                key = self._config_key(cfg)
                if key in self._measured:
                    return None
                name = f"round{round_idx+1}_candidate{idx+1}"
                future = executor.submit(self._build_candidate, cfg, name, target_base_quant)
                return (future, cfg, key)

            pending = _submit_build(0) if overlap_builds else None
            try:
                for i, candidate in enumerate(to_measure):
                    config = candidate["config"]
                    config_key = self._config_key(config)

                    if overlap_builds:
                        job = pending
                        # Prefetch the NEXT candidate's build now, so it
                        # overlaps with this candidate's measurement below --
                        # at most one extra candidate GGUF on disk at a time
                        # (prefetch depth 1).
                        pending = _submit_build(i + 1)
                    else:
                        # Serial: build THIS candidate now and block on it
                        # before measuring -- never a build and a measurement
                        # resident at once.
                        job = _submit_build(i)

                    # Skip if already measured (mirrors the historical guard;
                    # _submit_build applies the identical check before ever
                    # submitting a build for this index, so no orphaned build
                    # exists for a skipped candidate).
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

                    # job is None only if this index was already measured at
                    # submit time -- but that's exactly the branch just above,
                    # so reaching here always has a real in-flight/finished build.
                    _future, _cfg, _key = job
                    try:
                        path = _future.result()
                    except Exception as exc:
                        # _build_candidate already catches its own exceptions
                        # and returns None on failure -- this only fires for
                        # something escaping that (e.g. a cancelled/aborted
                        # future), and must degrade the same way: log + skip
                        # without killing the rest of the search.
                        log.error("Build failed", stage="build", error=str(exc), exc_info=exc)
                        path = None

                    if path is None:
                        continue

                    self._record_candidate_measurement(
                        path, config, config_key, incumbent_tier_by_key,
                        enable_kl, enable_speed_bench, checkpoint_path, verbose,
                    )
            finally:
                # Any prefetched build still outstanding (e.g. we're bailing
                # out via an exception raised somewhere above) must be
                # joined and its candidate GGUF cleaned up -- the candidates
                # dir must never leak a finished-but-unconsumed prefetch
                # build.
                if pending is not None:
                    _future, _cfg, _key = pending
                    _future.cancel()
                    try:
                        _leaked_path = _future.result()
                    except Exception:
                        _leaked_path = None
                    if _leaked_path:
                        _leaked = Path(_leaked_path)
                        if _leaked.exists():
                            _leaked.unlink()
                executor.shutdown(wait=True)

            # 4d. Log round summary
            if verbose and self._measured:
                # A measurement-timeout disclosure entry (see
                # _record_candidate_measurement's failure branch) has no
                # "residual" at all -- it never reached a real ppl
                # reading -- so filter rather than indexing unconditionally
                # (that used to be safe when every self._measured entry
                # came from the ppl-succeeded branch and therefore always
                # had one). Log UNCONDITIONALLY once verbose+non-empty,
                # even when every entry this round was a disclosure (no
                # residuals at all) -- reviewer Q1: suppressing the line
                # in exactly that case would drop round-level visibility
                # in precisely the failure scenario this whole fix exists
                # to surface, so mean_abs_residual reads None instead of
                # the log line vanishing.
                residuals = [
                    abs(m["residual"]) for m in self._measured.values()
                    if m.get("residual") is not None
                ]
                mean_abs_residual = (
                    round(sum(residuals) / len(residuals), 4) if residuals else None
                )
                log.info(
                    "Round summary",
                    stage="measurement",
                    round=round_idx + 1,
                    total_measurements=len(self._measured),
                    mean_abs_residual=mean_abs_residual,
                )

        # A measured search whose every candidate build/measure failed --
        # OR whose every candidate measurement came back physically
        # impossible (measurement_invalid, see the measurement loop above)
        # -- must not report success: _select_final_survivors excludes
        # invalid entries from tier competition, so counting emptiness of
        # self._measured itself misses the case where self._measured is
        # non-empty but every single entry is invalid (retained there
        # deliberately, for diagnostics). That let a run "successfully"
        # complete with zero tiers instead of failing loudly. Count VALID
        # measurements, matching the existing guard's style.
        n_valid = sum(
            1 for info in self._measured.values()
            if not info.get("measurement_invalid")
        )
        if measurement_rounds > 0 and n_valid == 0:
            raise RuntimeError(
                "Measured search completed all "
                f"{measurement_rounds} round(s) but produced zero VALID "
                "measurements (every candidate build/perplexity measurement "
                "either failed outright, or came back physically impossible "
                "-- measured_loss below -eps -- and was flagged "
                "measurement_invalid). Refusing to write search_results.json "
                "as if this were a normal partial run -- check the "
                "llama.cpp build, disk space, corpus, and per-candidate "
                "build errors logged above."
            )

        self._log_predictor_tracking()

        # ── Step 5: Select final survivors per tier ──
        tiered = self._select_final_survivors(baseline_size_gb)

        # Save all results
        self._save_results(all_configs, tiered)
        self._write_pareto_report()
        if write_calibration:
            self._write_noise_calibration()

        # Run completed successfully -- the checkpoint's job is done.
        checkpoint_path.unlink(missing_ok=True)

        # BLOCKER fix: the KL base-logits file is huge (see
        # keep_kl_base_logits's docstring above -- ~chunks * ctx_size *
        # vocab_size * 2 bytes, verified 69 GB for a 27B at 100 chunks) and
        # was never deleted. Only reached on a SUCCESSFUL completion (every
        # earlier failure path above -- baseline measurement failure, zero
        # valid measurements -- raises before this point), so a killed or
        # failed run always leaves the file in place for a later resume to
        # reuse without paying the capture cost again.
        # Two targets: the file this run adopted, plus any orphan at the
        # canonical path this run never adopted (a prior killed KL run's
        # capture followed by this run completing with KL off entirely --
        # without the orphan sweep that ~69 GB would outlive every
        # successful run).
        _kl_cleanup_targets = {
            p for p in (
                self._kl_base_logits_path,
                str(self.output_dir / "_kl_base_logits.kld"),
            ) if p
        }
        if not keep_kl_base_logits and not (
            os.environ.get(self._KEEP_KL_LOGITS_ENV) == "1"
        ):
            for _kl_path in _kl_cleanup_targets:
                try:
                    _kl_file = Path(_kl_path)
                    if not _kl_file.exists():
                        continue
                    _kl_file.unlink()
                    if verbose:
                        log.info(
                            "Deleted KL base logits file",
                            stage="kl", path=_kl_path,
                        )
                except OSError as exc:
                    log.warning(
                        "Failed to delete KL base logits file -- it will be "
                        "left on disk",
                        stage="kl", path=_kl_path,
                        error=str(exc),
                    )

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

    def _detect_search_groups(self) -> List[str]:
        """Detect which tensor groups this model actually has: dense
        E/H/Q/K/O/U/D always, plus X/R when MoE expert/router tensors are
        present and S when SSM/linear-attention tensors are present. Sets
        self._search_groups (read by _build_incumbent_seeds and
        run_evolution) and returns the same list, since callers also use
        it locally (e.g. as probe_all_groups's groups= argument).

        Byte-identical block previously duplicated in run_measured_search
        and run_full_search (diffed identical before this extraction).

        TRAP: the `open_model_source` import below is deliberately
        function-local, not hoisted to module scope. Several tests
        (e.g. tests/test_strict_probing.py's "Group detection opens the
        source -- stub it.") monkeypatch magicquant.gguf.source
        .open_model_source by name; a late import resolves that
        attribute at call time, which is what makes the patch take. A
        module-level import would bind the original function at import
        time and silently break every one of those tests.
        """
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
        # This loop drove classify_tensor() one name at a time (not
        # classify_tensors(), which fires the summary automatically) across
        # the full model's tensor_names -- surface the unclassified-tensor
        # summary now that the pass is complete. See
        # TensorGroupClassifier.warn_unclassified_once()'s docstring.
        classifier.warn_unclassified_once()
        # Remember the full detected group set so run_evolution actually
        # varies X/R/S (otherwise it falls back to DEFAULT_GROUPS).
        self._search_groups = groups
        return groups

    def _exclude_fixed_groups(self, verbose: bool) -> None:
        """Drop any group in ``self.fixed_groups`` (set right after probing,
        live or resumed -- see SensitivityProber._detect_fixed_groups) from
        ``self._search_groups``. Byte-identical block previously duplicated
        in run_measured_search and run_full_search.

        A structurally-fixed group's scheme can never take effect (the
        writer forces F32 regardless), so leaving it in the mutable set
        would waste candidate-generation slots on choices with zero real
        effect and mislead size prediction (a candidate "shrinking" R to
        Q4_K_M would predict bytes saved that the actual write never
        realizes). Must run AFTER self.fixed_groups is populated and
        BEFORE _build_incumbent_seeds / run_evolution read self._search_groups.
        """
        if not self.fixed_groups:
            return
        # Only groups fixed for a SCHEME-INVARIANT reason may be dropped. A
        # group fixed solely because its rows aren't 32-divisible still has
        # BF16 available (block_size == 1 skips the writer's block-size
        # fallback, so it is written F16 at half the F32 bytes) -- dropping
        # it would silently delete a real size choice. Such a group still
        # gets its probe skipped; it just keeps its search slot. Defaults to
        # True so a checkpoint written before this key existed keeps its
        # previous exclusion behaviour.
        excludable = {
            g for g, info in self.fixed_groups.items()
            if info.get("scheme_invariant", True)
        }
        if not excludable:
            return
        self._search_groups = [
            g for g in self._search_groups if g not in excludable
        ]
        if verbose:
            log.info(
                "Excluding structurally-fixed groups from the search",
                stage="probing",
                fixed_groups=sorted(excludable),
                probe_skipped_only=sorted(set(self.fixed_groups) - excludable),
            )

    def _build_predictor(self, calibration_source, baseline_size_gb: float) -> PredictiveScorer:
        """Construct this search's PredictiveScorer. Byte-identical
        construction call previously duplicated in run_measured_search and
        run_full_search (diffed identical before this extraction); each
        caller still assigns the result to self.predictor itself.
        """
        return PredictiveScorer(
            sensitivity_weights=self.sensitivity_weights,
            parameter_counts=self._param_counts,
            baseline_size_gb=baseline_size_gb,
            baseline_tps=360,
            imatrix_active=self._imatrix is not None,
            calibration_source=calibration_source,
            # Real per-(group, scheme) bpw on THIS model, so a K-quant that
            # the writer will rewrite to Q8_0 is priced at what it actually
            # costs. Empty for ordinary models, which is exactly the
            # historical behaviour.
            effective_bpw=self._effective_bpw,
        )

    def _record_candidate_measurement(
        self, path, config, config_key, incumbent_tier_by_key,
        enable_kl, enable_speed_bench, checkpoint_path, verbose,
    ):
        """Measure one built candidate GGUF (fusing KL+PPL when active),
        apply the physical-plausibility eps guard, populate
        self._measured, feed the predictor's active-learning residual
        cache, persist the checkpoint, and unlink the candidate GGUF.
        Pure code motion out of run_measured_search's per-candidate loop
        body -- no logic change, no reordering.

        PRECONDITION: the caller must have already handled `path is
        None` -- that guard stays at the call site, above this method's
        call, in run_measured_search. `Path(None)` below would raise.

        53 of this body's ~148 lines are a 9-day-old correctness fix
        (a6f8dd0, "never let a non-measurement become a number"): the
        `eps`/`measurement_invalid` guard and the residual-recording
        suppression it drives exist because a NaN-driven
        measured_loss=-0.9225 once won a tier. Do not simplify the eps
        expression, and do not hoist predictor.record_residual() out
        from under the `if not measurement_invalid:` check.

        Exception semantics are unchanged by this move: an exception
        escaping calculate_perplexity or the KL-fallback branch is not
        caught here and propagates to the caller exactly as it did out
        of the inline block -- including that the candidate GGUF is
        NOT unlinked on that path, since the cleanup at the bottom of
        this method never runs. That is pre-existing behavior; it is
        not fixed here.
        """
        # Measure perplexity, fusing in the KL pass when active: a
        # --kl-divergence run against saved base logits ALSO
        # prints this candidate's own perplexity (Mean PPL(Q)), so
        # when KL scoring is active we get both signals from ONE
        # llama-perplexity invocation instead of two. Falls back
        # to the historical standalone calculate_perplexity call
        # when KL is off, its base logits aren't active, the KL
        # call raised, or its result doesn't carry "ppl" -- the
        # "KL failure must not abort/win" guarantee stays intact
        # either way (measured entry gets ppl either way, "kl"
        # only recorded when the KL call itself succeeded).
        kl_result = None
        kl_timed_out = False
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
            if kl_result is None:
                # Distinguish "the KL subprocess itself timed out" from
                # any other KL failure (unparseable output, an OSError
                # caught above) -- see LlamaCppTools._run_subprocess_or_
                # none's _last_subprocess_failure. getattr-safe: a
                # hand-rolled fake tools object (several tests) never
                # sets this attribute at all.
                kl_failure = getattr(self.llama_tools, "_last_subprocess_failure", None)
                kl_timed_out = bool(kl_failure) and kl_failure.get("kind") == "timeout"

        # Track which corpus THIS measurement actually used --
        # recorded per-entry below (fix for CORPUS PROVENANCE:
        # search_results.json used to stamp one corpus value at
        # save time, which can't catch a corpus that changed
        # mid-run). The KL path already threads self._kl_corpus_
        # path through explicitly; the plain path re-resolves via
        # LlamaCppTools, which now pins its own auto-resolution
        # and raises if a later call would disagree -- so this is
        # cheap (no new subprocess) and guaranteed consistent
        # with what calculate_perplexity itself just measured
        # over.
        if kl_result is not None and kl_result.get("ppl") is not None:
            ppl = kl_result["ppl"]
            measurement_corpus = self._kl_corpus_path
            ppl_timed_out = False
        else:
            ppl = self.llama_tools.calculate_perplexity(path, verbose=verbose)
            measurement_corpus = self.llama_tools._resolve_data_file(None)
            # Same distinction as the KL leg above, for this (always the
            # LAST-attempted) leg -- whichever of the two determines
            # "ppl is None" below.
            ppl_failure = getattr(self.llama_tools, "_last_subprocess_failure", None)
            ppl_timed_out = bool(ppl_failure) and ppl_failure.get("kind") == "timeout"

        if ppl is not None:
            measured_loss = (ppl - self.baseline_ppl) / self.baseline_ppl

            # A quantized candidate cannot genuinely beat the
            # baseline it's a lossy compression OF -- a
            # measured_loss below -eps is a failed/noise-floor
            # measurement, not a quality win. Unguarded, this fed
            # straight into _select_final_survivors' min()
            # selection; the incident that motivated this fix saw
            # a NaN-driven "measured_loss=-0.9225" WIN a tier.
            # eps is sized off this candidate's own reported KL
            # ppl_err when available, else the same shared default
            # probing.py's clamp uses (magicquant.utils
            # .measurement.measurement_eps).
            eps = measurement_eps(
                self.baseline_ppl,
                kl_result.get("ppl_err") if kl_result else None,
            )
            measurement_invalid = measured_loss < -eps
            if measurement_invalid:
                log.warning(
                    "Candidate measurement is physically "
                    "impossible (measured_loss below -eps) -- "
                    "flagging invalid instead of letting it win "
                    "a tier",
                    stage="measurement",
                    measured_loss=round(measured_loss, 4),
                    eps=round(eps, 4),
                    ppl=round(ppl, 4),
                    baseline_ppl=round(self.baseline_ppl, 4),
                )

            predicted_loss = self.predictor.predict_loss(config)
            # NOTE the units. predicted_loss is in NOISE units (a ranking
            # score, ~1-3); measured_loss is a RELATIVE FRACTION
            # ((ppl-baseline)/baseline, ~0.005). Their raw difference is
            # dimensionally meaningless and must never be fed back as a
            # correction -- doing so used to make predict_loss return the
            # measurement itself, handing measured configs a ~+0.20 composite
            # advantage that collapsed exploration after round 1 (see
            # PredictiveScorer's "Active learning" section). The predictor now
            # owns the conversion; we read the real, noise-unit residual back
            # from it for the record.
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
                "corpus_path": measurement_corpus,
                "measurement_invalid": measurement_invalid,
            }
            if config_key in incumbent_tier_by_key:
                self._measured[config_key]["incumbent"] = (
                    incumbent_tier_by_key[config_key]
                )

            if kl_result is not None:
                self._measured[config_key]["kl"] = kl_result
            elif kl_timed_out:
                # KL leg timed out but the PPL fallback above still
                # succeeded -- this candidate gets scored WITHOUT the KL
                # term its siblings carry. Record that instead of
                # silently comparing it apples-to-oranges against
                # candidates that DID get a KL measurement (field
                # report, 2026-08).
                self._measured[config_key]["kl_timeout"] = True

            # Optional secondary signal -- best-effort (None on
            # failure), scored in _select_final_survivors
            # alongside measured_loss rather than gating the
            # candidate at all. bench() only catches
            # CalledProcessError/TimeoutExpired internally; a
            # missing or wrong-arch binary raises
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

            # Active learning: feed residual back -- but never
            # from an invalid measurement. A physically
            # impossible ppl reading would poison the predictor
            # with a bogus residual that then biases every LATER
            # candidate's predicted_loss in this search, not just
            # this one candidate's own record.
            if not measurement_invalid:
                self.predictor.record_measurement(config, measured_loss)
                # The calibrated, noise-unit residual -- the only one worth
                # persisting. None until enough pairs exist to fit a scale,
                # which is honest: "not enough signal yet", never a guess.
                residual = self.predictor.residual_for(config)

            if verbose:
                log.info(
                    "Candidate measured",
                    stage="measurement",
                    ppl=round(ppl, 4),
                    measured_loss=round(measured_loss, 4),
                    predicted_loss=round(predicted_loss, 4),
                    # None until the predictor has enough (predicted, measured)
                    # pairs to fit its measured->noise-unit scale. Logging
                    # "None" is the honest reading -- there is no calibrated
                    # residual yet, and mean_abs_residual already filters it.
                    residual=(round(residual, 4) if residual is not None else None),
                )

            # Persist after EVERY successful measurement -- a kill
            # right after this point must resume with this candidate
            # already recorded, not lost.
            self._write_measured_checkpoint(checkpoint_path)
        else:
            if kl_timed_out or ppl_timed_out:
                # Neither leg produced a usable measurement, and at
                # least one failure was a genuine subprocess TIMEOUT
                # (not a parse failure/OSError) -- without this, this
                # candidate leaves NO trace in self._measured,
                # indistinguishable from "never attempted" (field
                # report, 2026-08: a 37.8GB candidate burned ~4h of
                # healthy CPU across both legs and search_results.json
                # recorded nothing). This entry is diagnostics-only and
                # deliberately INERT to selection, using the same
                # "drop from selection, don't erase the record"
                # contract as every other measurement_invalid entry:
                # measurement_invalid=True makes _select_final_
                # survivors skip it before it ever touches size_gb
                # (which this entry doesn't even have), and the
                # existing n_valid all-invalid guard, noise-calibration
                # fit, and predictor-tracking diagnostic already filter
                # on measurement_invalid / measured_loss is not None,
                # so none of them need to change for this entry to stay
                # harmless. "config" is the one field _serialize_
                # measurement requires unconditionally (v["config"], no
                # fallback), so it must be present.
                self._measured[config_key] = {
                    "config": config,
                    "measurement_invalid": True,
                    "measurement_timeout": True,
                    "timeout_leg": "ppl" if ppl_timed_out else "kl",
                }
                if kl_timed_out:
                    self._measured[config_key]["kl_timeout"] = True
                # Persist so a kill shortly after this point resumes
                # without re-attempting a candidate already known to
                # time out at this size/timeout budget -- the whole
                # point of recording this is to stop burning hours on
                # it every round.
                self._write_measured_checkpoint(checkpoint_path)
            if verbose:
                log.warning("Measurement failed", stage="measurement")

        # Clean up candidate GGUF to save disk (keep only final survivors)
        # We'll rebuild the final survivors at the end
        candidate_file = Path(path)
        if candidate_file.exists():
            candidate_file.unlink()

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

    @staticmethod
    def _available_ram_bytes() -> Optional[int]:
        """Best-effort MemAvailable (Linux); None if unreadable."""
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    def _should_overlap_builds(self) -> bool:
        """Whether to prefetch candidate i+1's build during i's measurement.

        The overlap hides CPU build time behind the GPU measurement, but it
        also holds a build's working set AND the measurement subprocess's full
        model load in RAM at once -- which OOMs on a large model / small box
        (a real 27B run died this way, 2026-07-05). ``MAGICQUANT_OVERLAP_BUILDS``
        forces the choice ("0"/"false"/"no" off, anything else truthy on);
        unset auto-disables when the source model is large relative to
        available RAM (a concurrent build then has no headroom), so a tiny
        model still overlaps and a 27B on a 124GB box does not.
        """
        raw = os.environ.get("MAGICQUANT_OVERLAP_BUILDS")
        if raw is not None:
            return raw.strip().lower() not in ("0", "false", "no", "")

        try:
            src = Path(self.source_model_path)
            src_bytes = src.stat().st_size if src.is_file() else None
        except OSError:
            src_bytes = None
        avail = self._available_ram_bytes()
        if src_bytes is None or avail is None:
            return True  # can't reason about it -- keep the historical overlap
        # A concurrent build needs room beyond the measurement's ~model-sized
        # load; require the source to fit in ~35% of available RAM to overlap.
        return src_bytes < avail * 0.35

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

    @staticmethod
    def _speed_aware_pick(
        candidates: List[Dict], quality_best: Dict, epsilon: float, score_of,
        speed_metric: str = "bytes",
    ) -> Dict:
        """Within a tier, among candidates whose *raw measured_loss* is within
        ``epsilon`` relative of the tier's best, pick the smallest -- otherwise
        leave ``quality_best`` (the flat score winner) untouched.

        NOTE the ranking key is raw ``measured_loss``, NOT the KL-blended
        ``_selection_score``. That distinction IS the 2026-07 fix: banding on
        the blended score let a noise-level KL swing hand a tier to a larger,
        worse-PPL candidate. See ``speed_epsilon`` in ``run_measured_search``.

        ``speed_metric`` chooses how "fastest" is measured:
          - ``"bytes"`` (default): smallest ``size_gb``. tg is memory-
            bandwidth-bound, so fewer bytes IS faster generation -- and
            ``size_gb`` is recorded for every candidate, deterministic, and
            noise-free. A measured llama-bench tg swung ~1.85x for the SAME
            config across invocations (thermal + coexisting GPU users), so
            ranking on it can pick a genuinely worse config as "fastest";
            bytes sidesteps that entirely. This is the safe default.
          - ``"bench"``: highest measured ``bench["tg_ts"]``. Captures per-byte
            kernel-efficiency differences (e.g. IQ vs K-quant) that bytes
            can't see, but only trustworthy with reliable bench data (more
            reps, quiesced GPU). Requires ``enable_speed_bench=True``; when no
            candidate carries bench data this is a no-op and ``quality_best``
            is returned unchanged.

        ``score_of`` is the metric the epsilon band is computed over.
        ``_select_final_survivors`` passes raw ``measured_loss`` (NOT the
        KL-blended ``_selection_score``) -- the Pareto/knee rule is over
        (measured_loss, size_gb) per the 2026-07 fix: KL must not be able to
        smuggle a genuinely-worse-PPL, larger candidate past a real PPL
        advantage just because the KL term swung the blended score (this is
        exactly what happened on the real ThinkingCap run that motivated
        this fix -- see ``run_measured_search``'s ``speed_aware`` docstring).
        KL still matters -- it decided which candidate is ``quality_best``
        (used as the fallback return and to gate KL-failed candidates below)
        -- but no longer overrides a raw quality/size Pareto tradeoff within
        the noise band.
        The in-band filter separately guards on KL *presence*, independent
        of ``score_of``: a candidate whose KL measurement failed (no "kl"
        key) can never beat one that measured real KL when the quality
        winner itself is KL-confirmed -- otherwise a measurement failure
        gets rewarded over real (if poor) data.

        DELIBERATE DEMOTION -- KL's ``mean_kl`` VALUE plays no role in
        picking the in-band winner below (only its *presence*, via the
        gate above): the final ``min(..., key=size_gb)`` /
        ``max(..., key=tg_ts)`` picks purely on size/speed among in-band
        candidates, even when two in-band candidates have very different
        ``mean_kl``. This is intentional, not an oversight -- reinstating
        KL as a value-based tiebreak here would reopen exactly the
        ThinkingCap failure mode this function exists to close (a KL swing
        deciding the tier winner instead of raw quality/size), just scoped
        to a narrower slice of candidates. If a future run shows KL-value
        blindness within the band causing a real regression (analogous to
        the ThinkingCap PPL/size one), prefer adding a documented, narrowly-
        scoped exception here over quietly blending KL back into the
        ranking key.
        """
        best_has_kl = bool(
            quality_best.get("kl") and quality_best["kl"].get("mean_kl") is not None
        )
        best_score = min(score_of(c) for c in candidates)
        threshold = best_score + epsilon * abs(best_score)
        in_band = [
            c for c in candidates
            if score_of(c) <= threshold
            and (not best_has_kl
                 or (c.get("kl") and c["kl"].get("mean_kl") is not None))
        ]
        if speed_metric == "bench":
            contenders = [c for c in in_band
                          if c.get("bench") and c["bench"].get("tg_ts") is not None]
            if not contenders:
                return quality_best
            return max(contenders, key=lambda c: c["bench"]["tg_ts"])
        # "bytes": smaller = faster tg (bandwidth-bound), size_gb always present
        contenders = [c for c in in_band if c.get("size_gb") is not None]
        if not contenders:
            return quality_best
        return min(contenders, key=lambda c: c["size_gb"])

    def _select_final_survivors(self, baseline_gb: float) -> Dict[str, Dict]:
        """From all measured configs, pick the best per tier.

        Candidates flagged ``measurement_invalid`` (measured_loss below
        -eps -- a physically impossible reading, see the measurement loop in
        ``run_measured_search``) are excluded from tier competition here but
        left in ``self._measured`` / search_results.json for diagnostics --
        "drop from selection, don't erase the record".
        """
        by_tier: Dict[str, List[Dict]] = defaultdict(list)
        for info in self._measured.values():
            if info.get("measurement_invalid"):
                continue
            tier = self._classify_tier(info["size_gb"], baseline_gb)
            by_tier[tier].append(info)

        # Conservative getattr fallback (False/None/"bytes") for instances
        # built via bare __new__ that predate this feature entirely -- a
        # normally constructed orchestrator always has these attributes set
        # (True / None / "bytes" by __init__, then possibly overridden by
        # run_measured_search's own speed_aware/speed_epsilon/speed_metric
        # params), so real callers get the new default; only pre-feature/
        # stripped-down test doubles fall back here.
        speed_aware = getattr(self, "_speed_aware", False)
        speed_epsilon_override = getattr(self, "_speed_epsilon", None)
        speed_metric = getattr(self, "_speed_metric", "bytes")

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
                if speed_aware:
                    if speed_epsilon_override is not None:
                        speed_epsilon = speed_epsilon_override
                    else:
                        # Reuse the same measurement-noise tolerance the "is
                        # this PPL physically plausible" guard uses (see
                        # run_measured_search's speed_epsilon docstring),
                        # AND actually exercise its 3-sigma path: thread the
                        # tier-quality-winner's own reported KL ppl_err
                        # through when a fused KL pass measured one, exactly
                        # like the per-candidate measurement_invalid check
                        # above does with kl_result.get("ppl_err"). Without
                        # this, measurement_eps always fell back to its flat
                        # DEFAULT_RELATIVE_EPS regardless of how precise this
                        # run's own measurements actually were. NOTE: this
                        # epsilon width is NOT what fixes the ThinkingCap
                        # bug -- that fix is banding on raw measured_loss
                        # (score_of below / _speed_aware_pick's docstring)
                        # instead of the KL-blended score. A future reader
                        # should not "tune" this epsilon expecting it to
                        # change tier-winner selection by itself.
                        reported_err = (
                            best["kl"].get("ppl_err")
                            if best.get("kl") else None
                        )
                        baseline_ppl = getattr(self, "baseline_ppl", None) or 0.0
                        speed_epsilon = measurement_eps(baseline_ppl, reported_err)
                    best = self._speed_aware_pick(
                        candidates, best, speed_epsilon,
                        lambda info: info["measured_loss"], speed_metric,
                    )
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
        weights_degenerate = None
        output_dir = getattr(self, "output_dir", None)
        if output_dir is not None:
            try:
                sensitivity_path = Path(output_dir) / "sensitivity.json"
                if sensitivity_path.is_file():
                    sensitivity_data = json.loads(sensitivity_path.read_text())
                    probing_provenance = sensitivity_data.get("probing_provenance")
                    # Absent on sensitivity.json written before this field
                    # existed -- None rather than a misleading False.
                    weights_degenerate = sensitivity_data.get("weights_degenerate")
            except Exception:
                probing_provenance = None
                weights_degenerate = None

        return {
            "chunks": getattr(llama, "ppl_chunks", None),
            "ctx_size": getattr(llama, "ctx_size", None),
            "corpus": corpus,
            "imatrix_active": imatrix is not None,
            "imatrix_n_tensors": len(imatrix) if imatrix else None,
            # True whenever base-logits capture succeeded, whether it was
            # requested by probe_kl (Step 2 sensitivity probes, default on)
            # or enable_kl (candidate objective blend, default off) or both
            # -- they share one capture pass. Does NOT by itself mean the
            # candidate objective blends KL; see kl_objective_blend_active.
            "kl_enabled": bool(getattr(self, "_kl_base_logits_path", None)),
            # True only when enable_kl actually blended KL into candidate
            # selection this run (self._kl_weight is 0.0 whenever enable_kl
            # was off, regardless of probe_kl / kl_enabled above).
            "kl_objective_blend_active": bool(getattr(self, "_kl_weight", 0.0)),
            "kl_weight": getattr(self, "_kl_weight", 0.0),
            "probing_provenance": probing_provenance,
            # True when SensitivityProber.get_normalized_weights() had to
            # fall back to uniform weights (total sensitivity == 0) -- see
            # probing.py's weights_degenerate. None when unknown (no
            # sensitivity.json / prediction-only run).
            "weights_degenerate": weights_degenerate,
        }

    @staticmethod
    def _serialize_measurement(v: Dict[str, Any], *, include_path: bool) -> Dict[str, Any]:
        """Single field list for a per-measurement export entry, shared by
        ``_save_results``' "measurements" dict (include_path=False) and
        ``_write_measured_checkpoint``'s "measured" dict (include_path=True).
        Both used to independently re-list the same fields off
        ``self._measured.items()`` and had to be hand-kept in sync.

        CONTRACT: this whitelist feeds a PERSISTED interchange format.
        search_results.json's "measurements" is consumed by
        qat.config.load_hybrid_config, Foundry's rocmfpx MQ-hybrid mode,
        tools/reselect_tiers.py, and tools/fit_noise_factors.py. The
        checkpoint's "measured" dict is read back verbatim on resume
        (``self._measured[key] = dict(entry)``, no second filter) -- this
        whitelist is therefore the ONLY gate on what a resumed run keeps.
        A field added here lands in BOTH artifacts; a field dropped here
        is silently lost by every resumed run. See the BLOCKER note below
        for why that is not hypothetical.

        Per-site key ORDER is preserved exactly as it was before this
        helper existed, and is CONTRACTUAL: these artifacts are persisted
        interchange formats read by external tools, so treat raw JSON key
        order as part of the format -- do not collapse the two branches
        into one uniform order. ``_save_results`` ends
        ...incumbent, corpus_path, measurement_invalid; the checkpoint
        inserts "path" right after "residual" and ends
        ...incumbent, measurement_invalid, corpus_path. Both then get
        kl_timeout, measurement_timeout, timeout_leg appended at the TAIL
        (additive; see _record_candidate_measurement's timeout-disclosure
        fix -- these are new fields, not a reordering of the existing
        ones, so each site's pre-existing order above is unchanged).
        """
        entry: Dict[str, Any] = {
            "config": v["config"],
            "ppl": v.get("ppl"),
            "measured_loss": v.get("measured_loss"),
            "predicted_loss": v.get("predicted_loss"),
            "residual": v.get("residual"),
        }
        if include_path:
            # Checkpoint-only, write-only field: the candidate GGUF it
            # names is unlink()ed moments after the checkpoint write, and
            # nothing ever reads it back on resume. Deliberately NOT
            # emitted into search_results.json -- that would leak a dead
            # temp-build path into a published, externally-consumed
            # artifact.
            entry["path"] = v.get("path")
        entry["size_gb"] = v.get("size_gb")
        entry["kl"] = v.get("kl")
        entry["bench"] = v.get("bench")
        entry["incumbent"] = v.get("incumbent")
        if include_path:
            # BLOCKER fix: this whitelist used to omit
            # measurement_invalid/corpus_path, so a resumed run's entries
            # came back WITHOUT them -- info.get("measurement_invalid") is
            # None (falsy) post-resume, and a physically-impossible
            # candidate could win a tier again across a resume boundary.
            # See tests/test_measured_search_checkpoint_resume.py::
            # test_measurement_invalid_and_corpus_path_survive_checkpoint_round_trip.
            entry["measurement_invalid"] = v.get("measurement_invalid", False)
            entry["corpus_path"] = v.get("corpus_path")
        else:
            # Per-measurement corpus (fix for CORPUS PROVENANCE: the
            # top-level "measurement"/"corpus" field below is still
            # stamped once, kept for backward compat with older readers,
            # but each measurement now carries the corpus it was ACTUALLY
            # taken over, so a mid-run change would be visible here even
            # if the single summary field wasn't updated).
            entry["corpus_path"] = v.get("corpus_path")
            # True when this reading was physically impossible
            # (measured_loss below -eps) and excluded from
            # _select_final_survivors -- kept in the record for
            # diagnostics rather than silently dropped.
            entry["measurement_invalid"] = v.get("measurement_invalid", False)

        # Additive (2026-08 measurement-timeout fix), appended at the TAIL
        # of BOTH orders above -- see the docstring note. All three
        # default to falsy/None when absent, so an OLDER checkpoint or
        # search_results.json (written before this fix) round-trips
        # exactly as before: no KeyError, no behavior change, these just
        # read back as "no timeout info recorded" for every pre-existing
        # entry.
        entry["kl_timeout"] = v.get("kl_timeout", False)
        entry["measurement_timeout"] = v.get("measurement_timeout", False)
        entry["timeout_leg"] = v.get("timeout_leg")
        return entry

    def _log_predictor_tracking(self) -> None:
        """Kendall-tau diagnostic: does the loss predictor's ranking of
        candidates actually track measured reality? See
        ``magicquant.utils.measurement``'s "Predictor tracking" section --
        ``predictor_rank_correlation``/``predictor_is_tracking`` were built
        and unit-tested there (the module's own docstring calls this "the
        guard that would have caught the 2026-07 [Laguna-S] failure no
        matter which layer was at fault", tau -0.043 over that run's
        pairs) but were never wired into a live search until now.

        Computed ONCE, cumulatively, over every accumulated
        (predicted_loss, measured_loss) pair in ``self._measured`` --
        deliberately NOT per-round: ``candidates_per_round`` defaults to 4,
        far below ``MIN_TAU_SAMPLES`` (12), so a per-round call would
        report "unknown" forever and never actually check anything.
        ``measurement_invalid`` entries are excluded (their measured_loss
        is a physically-impossible reading that would corrupt the
        correlation the same way it would corrupt tier selection), and
        the filter clauses use ``.get(...)`` so incumbent-seeded entries
        (``predicted_loss`` present, ``measured_loss=None``) and
        foreign/hand-edited checkpoints missing either field are skipped
        before the value expression indexes them.

        Report, never gate: ``predictor_is_tracking`` returns a
        THREE-state verdict (True / False / None-for-"not enough data"),
        and only False is evidence of a broken run. A None verdict is
        logged the same as True (info) -- logging "not tracking" for
        "unknown" would be the exact "measured nothing reported as
        measured zero" defect this repo's audits keep flagging,
        reproduced in the reporting layer instead of the measurement
        layer. Never raises BY CONSTRUCTION (the whole body is wrapped,
        like the sibling reporting helpers): this runs before
        ``_save_results``, so an unexpected exception here -- e.g. an
        older self-installed scipy without ``SignificanceResult
        .statistic`` -- must not be able to destroy a multi-hour run's
        results. scipy being absent (``predictor_rank_correlation``
        degrades to ``(None, None)`` with its own one-time warning)
        surfaces as the ordinary "unknown" verdict, not a crash.

        Stores the verdict on ``self._predictor_tracking`` for
        ``_save_results`` to persist under the additive
        ``"predictor_tracking"`` search_results.json key.
        """
        try:
            self._log_predictor_tracking_inner()
        except Exception as exc:  # pragma: no cover - defensive, like siblings
            log.warning(
                "Predictor-tracking diagnostic failed (non-fatal)",
                stage="measurement", error=str(exc),
            )

    def _log_predictor_tracking_inner(self) -> None:
        pairs = [
            (info["predicted_loss"], info["measured_loss"])
            for info in self._measured.values()
            if not info.get("measurement_invalid")
            and info.get("predicted_loss") is not None
            and info.get("measured_loss") is not None
        ]
        predicted = [p for p, _m in pairs]
        measured = [m for _p, m in pairs]
        is_tracking, tau = predictor_is_tracking(predicted, measured)
        self._predictor_tracking = {
            "is_tracking": is_tracking,
            "tau": tau,
            "n_pairs": len(pairs),
        }
        if is_tracking is False:
            log.warning(
                "Predictor is NOT tracking measured reality over this "
                "run -- its ranking of candidates was no better than "
                "chance. The search optimized against a signal "
                "uncorrelated with quality; treat its tier winners with "
                "suspicion.",
                stage="measurement", tau=round(tau, 4), n_pairs=len(pairs),
            )
        else:
            log.info(
                "Predictor tracking check",
                stage="measurement",
                is_tracking=is_tracking,
                tau=round(tau, 4) if tau is not None else None,
                n_pairs=len(pairs),
            )

    def _band_histogram(self, all_configs) -> Dict[str, int]:
        """{tier_band: count} over every candidate the search PROPOSED, keyed
        by predicted size. Counts proposals, not measurements -- see the
        comment at its call site in _save_results.

        Defensive by design: a candidate without a usable predicted size is
        counted under "unknown" rather than dropped, so the histogram's total
        always reconciles against len(all_configs) and a silent shortfall
        can't hide a bookkeeping bug.
        """
        from magicquant.quant.tiers import classify_tier

        hist: Dict[str, int] = {}
        for cand in all_configs or []:
            size = None
            if isinstance(cand, dict):
                size = cand.get("predicted_size_gb", cand.get("size_gb"))
            if size is None or not self._baseline_size_gb_for_bands():
                band = "unknown"
            else:
                band = classify_tier(size, self._baseline_size_gb_for_bands())
            hist[band] = hist.get(band, 0) + 1
        return hist

    def _baseline_size_gb_for_bands(self) -> float:
        """Baseline the band histogram classifies against. Separate accessor so
        a missing/zero baseline degrades to "unknown" bands instead of raising
        inside a results-writing path."""
        val = getattr(self, "baseline_size_gb", None)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
        predictor = getattr(self, "predictor", None)
        val = getattr(predictor, "baseline_size_gb", 0) if predictor else 0
        return float(val) if isinstance(val, (int, float)) and val > 0 else 0.0

    def _save_results(self, all_configs, tiered):
        """Persist search results and measurements to JSON.

        Called from BOTH search paths. Prediction-only tiers (run_full_search)
        carry ``predicted_size_gb``/``predicted_loss`` and no measured fields,
        so every access is ``.get()`` with the predicted fallback — the
        measured path simply fills in more of the fields. Consumers (QAT's
        ``load_hybrid_config``, Foundry's rocmfpx MQ-hybrid mode) only require
        ``tiered[tier]["config"]``, which both paths provide.
        """
        from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION

        # Built once and reused for both "tiered" (full) and "tiered_survivors"
        # (the same per-tier entry minus predicted_loss) below -- the two used
        # to be independent dict comprehensions over the same `tiered.items()`
        # and had to be hand-kept in sync.
        _tiered_full = {
            tier: {
                "config": info["config"],
                "ppl": info.get("ppl"),
                "measured_loss": info.get("measured_loss"),
                "predicted_loss": info.get("predicted_loss"),
                "size_gb": info.get("size_gb", info.get("predicted_size_gb")),
                # "incumbent" when this tier's winner IS one of
                # magicquant.incumbents' seeded llama.cpp-mixture
                # configs (info["incumbent"] holds the seed tier tag,
                # e.g. "Q4"), "evolved" when the search itself produced
                # the winner. Previously invisible -- across four real
                # models the Q4/Q5 tiers were repeatedly won by the
                # incumbent seed with the search contributing nothing,
                # and there was no field recording that fact. Only
                # populated by the measured-search path (run_full_search's
                # prediction-only tiers don't carry seed provenance, so
                # they always read "evolved" here).
                "source": "incumbent" if info.get("incumbent") else "evolved",
            }
            for tier, info in tiered.items()
        }

        results = {
            # Which TIER_BOUNDARIES set classified the tier labels below.
            # Old files without this key predate the fix and must be read as
            # LEGACY_TIER_SCHEME_VERSION -- see magicquant.quant.tiers'
            # module docstring and tier_scheme_version() for the
            # compatibility read path. This never causes old artifacts to
            # fail to load: the per-tier config content is unaffected by the
            # label's meaning, only its human-facing interpretation is.
            "tier_scheme_version": CURRENT_TIER_SCHEME_VERSION,
            "baseline_ppl": self.baseline_ppl,
            "baseline_provenance": self.baseline_provenance,
            "probing_provenance": getattr(self, "probing_provenance", "unknown"),
            # Groups probing found structurally unquantizable on this model
            # (never-quantize-by-name / 1-D / non-32-divisible / SSM-F32) and
            # excluded from the mutable search -- see
            # SensitivityProber._detect_fixed_groups.
            "fixed_groups": getattr(self, "fixed_groups", {}),
            # PROPOSAL-side band histogram: how many candidates the search
            # PRODUCED per tier band, by predicted size, regardless of whether
            # any were ever measured. Without it, "zero configs measured in
            # the Q5 band" is ambiguous -- it cannot distinguish "the search
            # never proposed one" from "it proposed several and selection
            # dropped them all before measurement", and those are different
            # defects with different fixes. The 2026-08-13 empty-band
            # investigation stalled on exactly that ambiguity.
            #
            # Cheap: all_configs is already in hand and already carries
            # predicted sizes; this is a count, not a recomputation.
            "proposed_band_histogram": self._band_histogram(all_configs),
            "seed": self._search_seed,
            "measurement": self._measurement_metadata(),
            # Additive key: the Kendall-tau predicted-vs-measured ranking
            # check from _log_predictor_tracking(). None on a prediction-
            # only run (run_full_search never sets self._predictor_tracking
            # -- it has no measurements to check) or on a bare-__new__ test
            # double that calls _save_results directly. See
            # _log_predictor_tracking's docstring for the report-never-gate
            # semantics of the True/False/None verdict this carries.
            "predictor_tracking": getattr(self, "_predictor_tracking", None),
            "measurements": {
                k: self._serialize_measurement(v, include_path=False)
                for k, v in self._measured.items()
            },
            # Same per-tier entry as "tiered" below, minus predicted_loss --
            # derived from _tiered_full so the two can't drift apart again.
            "tiered_survivors": {
                tier: {k: v for k, v in entry.items() if k != "predicted_loss"}
                for tier, entry in _tiered_full.items()
            },
            "tiered": _tiered_full,
            # Additive (2026-08 fail-fast arch-check fix): which llama.cpp
            # binary took these measurements, and whether its libllama was
            # verified (before any measurement ran) to support the source
            # model's GGUF architecture -- see run_measured_search's
            # _run_arch_support_check. Top-level, not a per-measurement
            # field (_serialize_measurement's docstring): every measurement
            # in one run shares one instrument. None on a prediction-only
            # run_full_search (never sets _llamacpp_arch_check) or a
            # bare-__new__ test double with no _llama_tools.
            "llamacpp_binary": self._current_llamacpp_binary(),
            "llamacpp_arch_check": getattr(self, "_llamacpp_arch_check", None),
        }

        results_path = self.output_dir / "search_results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def _write_pareto_report(self) -> None:
        """Additive, read-only Pareto-frontier report over this run's
        measurements.

        The size-band tiers in search_results.json hide the real
        size/quality/(speed) tradeoff -- e.g. a Q6 tier can cost +11 GB and
        -60% generation speed for 0.6% better perplexity, a trade nobody
        would consciously choose. This surfaces it: writes pareto.json (the
        non-dominated candidate list, see magicquant.pareto.pareto_frontier)
        next to search_results.json, and logs the human-readable table at
        INFO.

        Pure reporting -- never touches search/selection state and must
        never fail the run: any error here is logged and swallowed. Guarded
        with getattr so bare-``__new__`` test orchestrators (missing
        output_dir or _measured entirely, e.g. ones that call
        ``_save_results`` directly without going through a real ``run_*``
        search) are skipped rather than crashing.

        Excludes ``measurement_invalid`` entries before building the
        frontier/table, same as ``_select_final_survivors`` and
        ``_write_noise_calibration`` -- an invalid entry's ppl is below
        baseline*(1-eps) by construction (a physically-impossible reading,
        see the eps guard in the measurement loop), so it has a lower ppl
        than every real candidate and would dominate the frontier on a
        mixed valid/invalid run. This is a caller-side fix only:
        ``magicquant.pareto.load_and_report()`` reading a persisted
        search_results.json off disk still shows invalid entries, since
        ``_save_results`` deliberately keeps them there for diagnostics.
        """
        output_dir = getattr(self, "output_dir", None)
        measured = getattr(self, "_measured", None)
        if output_dir is None or measured is None:
            return
        try:
            from magicquant.pareto import pareto_frontier, format_pareto_report

            usable = {
                k: v for k, v in measured.items()
                if not v.get("measurement_invalid")
            }
            excluded = len(measured) - len(usable)
            if excluded:
                log.info(
                    "Pareto report excluding measurement_invalid entries",
                    stage="pareto", excluded=excluded, total=len(measured),
                )
            frontier = pareto_frontier(usable)
            pareto_path = Path(output_dir) / "pareto.json"
            pareto_path.write_text(json.dumps(frontier, indent=2), encoding="utf-8")
            log.info(format_pareto_report(usable))
        except Exception as exc:
            log.warning(
                "Pareto report generation failed (non-fatal)",
                stage="pareto", error=str(exc), exc_info=exc,
            )

    def _write_noise_calibration(self) -> None:
        """Fit per-scheme noise factors from THIS run's measurements +
        sensitivity weights and write ``<output_dir>/noise_calibration.json``
        (opt-in, ``run_measured_search(write_calibration=True)``).

        Reuses ``magicquant.evolution.fit_noise_factors``'s least-squares
        fit directly (not re-implemented; ``tools/fit_noise_factors.py`` is
        now a thin CLI shim over the same module) but skips the round-trip
        through disk: it builds ``FitInput`` rows straight from ``self._measured``
        and ``self.sensitivity_weights`` instead of re-reading
        ``search_results.json``/``sensitivity.json`` back off disk. The
        output envelope matches the nested ``{"schemes": {...}}`` shape
        ``magicquant.quant.calibration`` reads, so a later run can point
        ``calibration_source`` at this exact file.

        Best-effort and additive: never raises. A fitting failure (or zero
        usable measurements) is logged and swallowed rather than failing an
        otherwise-successful measured search.
        """
        try:
            # F4 (2026-08 packaging fix): this used to be `from
            # tools.fit_noise_factors import ...` with a sys.path fallback,
            # because `tools/` is a bare top-level package with no guaranteed
            # presence outside a git checkout -- broken for any caller that
            # only has `magicquant` on its path (e.g. Foundry via PYTHONPATH,
            # run 3, 2026-07-06). The fitting logic now lives in-package, so
            # this import needs no fallback: it works wherever `magicquant`
            # itself is importable.
            from magicquant.evolution.fit_noise_factors import (
                FitInput, build_calibration_envelope, fit_noise_factors,
            )

            sensitivity_weights = self.sensitivity_weights or {}
            results_path = str(self.output_dir / "search_results.json")
            inputs = [
                FitInput(
                    config=info["config"],
                    measured_loss=info["measured_loss"],
                    sensitivity_weights=sensitivity_weights,
                    source=results_path,
                )
                for info in self._measured.values()
                # MAJOR 3: filtering only on "measured_loss is not None" let
                # measurement_invalid (physically-impossible, measured_loss
                # below -eps) readings into the noise-factor fit -- the
                # exact poisoning the measurement loop's active-learning
                # feed (record_residual) already avoids for the predictor,
                # made persistent here in noise_calibration.json instead.
                if info.get("measured_loss") is not None
                and not info.get("measurement_invalid")
            ]
            if not inputs:
                log.warning(
                    "write_calibration requested but no usable measurements "
                    "to fit -- skipping noise_calibration.json",
                    stage="calibration",
                )
                return

            fitted = fit_noise_factors(inputs)
            envelope = build_calibration_envelope(fitted, [results_path])
            calib_path = self.output_dir / "noise_calibration.json"
            calib_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
            log.info(
                "Noise calibration written", stage="calibration",
                path=str(calib_path), n_schemes=len(fitted),
            )
        except Exception as exc:
            log.warning(
                "Noise-factor calibration fit failed (non-fatal)",
                stage="calibration", error=str(exc), exc_info=exc,
            )

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
        head_aggressive: bool = False,
        stream_aware: bool = False,
        seed: Optional[int] = None,
        use_imatrix: bool = False,
        imatrix_corpus: Optional[str] = None,
        measurement_chunks: Optional[int] = None,
        seed_incumbents: bool = True,
        speed_weight: Optional[float] = None,
        use_bytes_tps: bool = False,
        calibration_source: str = "",
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Run prediction-only evolutionary search (no real measurements).
        Use run_measured_search() for the full Predict->Measure->Learn loop.

        head_aggressive: same H-only sampling bias as run_measured_search
        (see its docstring and EvolutionarySurvivor.__init__). Off by
        default; unchanged sampling for every group when False.

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

        speed_weight/use_bytes_tps: same tunable-objective knobs as
        run_measured_search (see its docstring and
        ``_build_objective_weights``) -- ``None``/``False`` (default) leaves
        the search's scoring unchanged.

        calibration_source: same as run_measured_search -- passed straight
        through to the ``PredictiveScorer`` this search constructs. ``""``
        (default) means today's fixed-path calibration lookup, unchanged.
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
            # Same calibration state as generate_hybrid_model's final tiers
            # (issue #5) -- enable_imatrix, above, already set self._imatrix.
            imatrix=self._imatrix,
        )
        groups = self._detect_search_groups()

        prober.probe_all_groups(groups=groups, aggressive_scheme="Q4_K_M", verbose=verbose)
        self.sensitivity_weights = prober.get_normalized_weights()
        self.probing_provenance = prober.probing_provenance
        self.weights_degenerate = prober.weights_degenerate
        self.fixed_groups = dict(prober.fixed_groups)
        prober.save_results(str(self.output_dir / "sensitivity.json"))

        # Structurally-fixed groups can never take a scheme other than what
        # the writer already forces (F32) -- see the same exclusion in
        # run_measured_search.
        self._exclude_fixed_groups(verbose)

        # MAJOR 4: same gate as run_measured_search -- "suspect"/degenerate
        # provenance can occur here too whenever a real llama_tools was
        # available and used for probing (this run's probes were not all
        # heuristic). A purely heuristic run (no llama.cpp) never reaches
        # "suspect" (see _enforce_probing_signal_gate's docstring / MAJOR 4
        # discussion), so this does not regress the documented
        # prediction-only heuristic-fallback design.
        self._enforce_probing_signal_gate()

        # (_estimate_model_size also populates self._param_counts per group.)
        baseline_size_gb = self._estimate_model_size(self.source_model_path)

        self.predictor = self._build_predictor(calibration_source, baseline_size_gb)

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
            has_imatrix=self._imatrix is not None,
            head_aggressive=head_aggressive,
            stream_aware=stream_aware,
            objective_weights=self._build_objective_weights(speed_weight),
            use_bytes_tps=use_bytes_tps,
            block32_only_groups=self.block32_only_groups,
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
        self._write_pareto_report()

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

        Under TIER_SCHEME_VERSION 2 (see magicquant.quant.tiers' module
        docstring), the Q2 tier IS reachable: its band is
        ``(0, _Q2_CEILING]`` = ``(0, 0.178]``, and both Q2_K (ratio ~0.1641)
        and IQ2_S (ratio ~0.1602) classify into it. (Under the old v1
        boundaries Q2 sat just outside the band and was genuinely
        unreachable without sub-Q2 IQ-quants -- that limitation no longer
        applies.) A tier can still legitimately come back empty if the
        search/incumbents never produced a config whose measured/predicted
        size lands in that tier's band; see the requested-but-empty warning
        below for how that's surfaced.
        """
        from magicquant.quant.tiers import describe_tier_band

        if tiers is None:
            tiers = ["Q8", "Q6", "Q5", "Q4", "Q2"]

        # Every tier in `tiers` (whether the caller passed it explicitly or
        # it came from the default list above) was REQUESTED -- a caller
        # asking for e.g. tiers=["Q4", "Q5", "Q6"] expects to get (up to)
        # three files back. Under the narrower v2 bands, a requested tier
        # with no measured/predicted config in it is a real, actionable gap
        # (fewer files shipped than asked for), not a routine no-op -- log
        # it loudly (not INFO) and record it so a downstream reader of
        # search_results.json (e.g. the publish path) can see it too,
        # instead of it only ever being visible in a log line. This does
        # NOT hard-fail: a genuinely empty band (nothing measured/predicted
        # landed there) is legitimate and the run's other tiers still ship.
        missing_requested = []
        generated = []
        for tier in tiers:
            if tier not in tiered:
                band = describe_tier_band(tier)
                log.warning(
                    "Requested tier has no config -- skipping "
                    "(narrower v2 tier bands can leave a tier empty; see "
                    "magicquant.quant.tiers module docstring)",
                    stage="generate", tier=tier, band=band,
                )
                missing_requested.append({"tier": tier, "band": band})
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
                # .get() is not None, not key presence: prediction-only
                # search results serialize ppl/measured_loss as null, so the
                # keys are PRESENT with value None (round(None) crashed here
                # for every `generate` after a --rounds 0 search).
                ppl=round(entry["ppl"], 4) if entry.get("ppl") is not None else None,
                measured_loss=round(entry["measured_loss"], 4) if entry.get("measured_loss") is not None else None,
            )

            path = self.generate_hybrid_model(
                config=config, model_name=name,
                base_quant=base_quant, verify=verify,
            )
            if path:
                generated.append(path)
            else:
                log.error("Tier generation failed", stage="generate", tier=tier)

        if missing_requested:
            self._record_missing_requested_tiers(missing_requested)

        return generated

    def _record_missing_requested_tiers(self, missing_requested: List[Dict[str, str]]) -> None:
        """Stamp ``requested_tiers_missing`` into the already-written
        search_results.json so a downstream reader (e.g. the publish path)
        can see that fewer files were generated than were asked for, without
        having to scrape logs. Best-effort: search_results.json is written
        by ``_save_results`` earlier in the run (both search paths call it
        before ``generate_tiered_models`` runs), but ``generate_tiered_models``
        can also be invoked standalone (see ``magicquant/__main__.py``'s
        ``generate`` subcommand, loading a prior run's file) against a
        directory where the file may be missing/unreadable -- in which case
        this logs and gives up rather than raising, since the GGUF files
        themselves were already generated successfully and that must not be
        undone by a diagnostics-only write failing.
        """
        results_path = self.output_dir / "search_results.json"
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "Could not read search_results.json to record missing "
                "requested tiers -- the tiers listed above were still "
                "skipped, but this run's search_results.json won't "
                "reflect it",
                stage="generate", error=str(exc),
            )
            return
        existing["requested_tiers_missing"] = missing_requested
        results_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

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
        # Thin delegate -- canonical implementation now lives in
        # magicquant.utils.naming.config_key (search-v1/4: was hand-
        # reimplemented identically here, in pareto._scheme_str, and in
        # evolution.predictor._make_config_key). Kept as a staticmethod
        # rather than inlined at call sites: it's called on the class
        # directly (tests/test_measurement_candidate_coverage.py) and has
        # ~10 internal call sites in the hot measured-search loop. Its
        # output is a PERSISTED interchange format -- see
        # magicquant.utils.naming.config_key's docstring.
        return _naming_config_key(config)

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
        """Resolve the corpus llama_tools would use, for measurement-
        condition comparisons -- never raises.

        MINOR fix: this used to swallow ALL exceptions, including the pin-
        violation ``RuntimeError`` ``LlamaCppTools._resolve_data_file`` now
        raises when the auto-resolved corpus disagrees with what it pinned
        earlier in this run (see its docstring). Swallowing THAT turned the
        corpus into ``None`` silently, voiding this run's own resume
        comparisons: a checkpoint's ``measurement_conditions["corpus"]``
        would then always "match" a future ``None`` instead of ever
        catching the change. A pin violation is a real invariant break, not
        a "no corpus configured" no-op -- log it loudly and keep reporting
        the last known-good corpus instead of erasing it.
        """
        llama = self.llama_tools
        if llama is None:
            return None
        try:
            resolved = llama._resolve_data_file(None)
        except RuntimeError as exc:
            log.error(
                "Corpus resolution failed (pin violation) -- keeping last "
                "known-good corpus for measurement-condition comparisons "
                "instead of silently voiding them",
                stage="resume", error=str(exc),
            )
            return getattr(self, "_last_resolved_corpus", None)
        except Exception as exc:
            log.warning(
                "Corpus resolution failed unexpectedly", stage="resume",
                error=str(exc), exc_info=exc,
            )
            return getattr(self, "_last_resolved_corpus", None)
        self._last_resolved_corpus = resolved
        return resolved

    def _current_measurement_conditions(self) -> Dict[str, Any]:
        """The subset of measurement conditions that must match between a
        checkpoint and the run attempting to resume it: the chunk cap, ctx
        size, calibration corpus, whether/how KL blends into the candidate
        OBJECTIVE (``enable_kl``/``kl_weight`` -- NOT ``probe_kl``, which only
        affects Step 2 sensitivity-probe scoring, a RESULT of a run rather
        than an input the resumed run's candidate ranking depends on), and
        the active imatrix's identity (``imatrix_id``).

        BLOCKER fix (F2): without ``enable_kl``/``kl_weight`` here, a
        checkpoint recorded under a PPL-only objective (``enable_kl=False``)
        could silently half-resume into a KL-blended run (``enable_kl=True``)
        -- every already-measured candidate's ``measured_loss`` would be
        reused unchanged, but nothing would ever backfill the ``"kl"``
        measurement those candidates never took, so they'd permanently
        compete on PPL-only scores inside a run everything else believes is
        KL-blended. See ``_measurement_conditions_match`` for the
        backward-compatible comparison this enables (a checkpoint written
        before this fix simply lacks these two keys).

        ``imatrix_id`` (issue #5): both ``self.sensitivity_weights`` (probes)
        and every entry in ``self._measured`` (candidates, via
        ``_build_candidate``'s ``imatrix=self._imatrix``) are built against
        whatever imatrix was active when the checkpointed run produced them.
        A resume with a DIFFERENT imatrix identity -- capture off vs on, or a
        different corpus/model producing a different capture -- must not
        reuse either: it would silently rank this run's candidates on
        weights and measurements taken under a calibration state that no
        longer matches what ``generate_hybrid_model`` will use for the final
        tiers. This mirrors v2's distortion-table cache, which already keys
        on imatrix identity (``v2/sensitivity.py``'s ``_imatrix_identity`` /
        ``_cache_key``) -- v1's checkpoint gate now invalidates the same way,
        via the same hash (``utils.measurement.imatrix_identity``).
        """
        llama = getattr(self, "_llama_tools", None)
        return {
            "chunks": getattr(llama, "ppl_chunks", None),
            "ctx_size": getattr(llama, "ctx_size", None),
            "corpus": self._safe_resolve_corpus(),
            "enable_kl": bool(getattr(self, "_enable_kl", False)),
            "kl_weight": getattr(self, "_kl_weight", 0.0),
            "imatrix_id": imatrix_identity(self._imatrix),
        }

    @staticmethod
    def _measurement_conditions_match(
        stored: Dict[str, Any], current: Dict[str, Any]
    ) -> bool:
        """Compare a checkpoint's stored measurement conditions against the
        current run's, with backward-compatible defaults for the KL fields
        (F2).

        ``chunks``/``ctx_size``/``corpus`` must match exactly, as before.

        ``enable_kl`` is compared with a MISSING stored key treated as
        ``False`` -- every checkpoint written before this fix predates the
        key entirely and was, without exception, produced by a PPL-only
        objective (``enable_kl`` didn't exist as a checkpoint condition
        until now), so "key absent" and "key present and False" mean the
        same thing here. Concretely: an old checkpoint + a fresh
        ``enable_kl=False`` config still resumes (``False == False``); an
        old checkpoint + a fresh ``enable_kl=True`` config is rejected
        (``False != True``), which is exactly the "changed objective" case
        this fix exists to catch; a new checkpoint resumes normally against
        a same-objective new run either way.

        ``kl_weight`` only participates in the comparison when the CURRENT
        run has ``enable_kl`` on -- a weight difference while blending is
        OFF changes nothing real about how candidates were ranked (
        ``run_measured_search`` forces ``self._kl_weight = 0.0`` whenever
        ``enable_kl`` is False), so it must not force an unnecessary
        re-measurement.

        ``imatrix_id`` (issue #5) is compared with a MISSING stored key
        treated as ``{"active": False}`` -- every checkpoint written before
        this key existed was produced by probes that never received an
        imatrix at all (the bug this fix closes), so "key absent" reads as
        "probes were unweighted", exactly like the ``enable_kl`` backward-
        compat default above. A stored inactive identity against a current
        run with an imatrix active is a mismatch and forces a fresh run --
        deliberately, even for the (possible but unrecorded) case where an
        old checkpoint's CANDIDATE measurements happened to already use an
        imatrix: that checkpoint predates this fix and cannot prove which
        state its probes vs. its candidates were each built under, so it is
        not trusted rather than partially trusted.
        """
        for key in ("chunks", "ctx_size", "corpus"):
            if stored.get(key) != current.get(key):
                return False

        stored_enable_kl = bool(stored.get("enable_kl", False))
        current_enable_kl = bool(current.get("enable_kl", False))
        if stored_enable_kl != current_enable_kl:
            return False

        if current_enable_kl and stored.get("kl_weight", 0.0) != current.get(
            "kl_weight", 0.0
        ):
            return False

        stored_imatrix_id = stored.get("imatrix_id", {"active": False})
        current_imatrix_id = current.get("imatrix_id", {"active": False})
        if stored_imatrix_id != current_imatrix_id:
            return False

        return True

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
        stored_conditions = checkpoint.get("measurement_conditions") or {}
        if not self._measurement_conditions_match(stored_conditions, current_conditions):
            reasons.append("measurement conditions changed")

        # Different llama.cpp BINARY = a different measurement instrument:
        # resurrecting a checkpoint's PPL/KL numbers next to a run using a
        # different llama-perplexity build would merge two runs whose
        # numbers are not comparable (this fix's whole point -- see
        # _run_arch_support_check). A checkpoint written before this fix
        # simply lacks "llamacpp_binary" -- treated as compatible (legacy),
        # same missing-key-is-compatible spirit as the enable_kl backward-
        # compat default above. Compared via os.path.realpath() on BOTH
        # sides (Opus review) so a relative-vs-absolute or symlinked
        # spelling of the SAME binary can't spuriously reject a valid
        # checkpoint -- that would throw away exactly the hours the
        # checkpoint protects; realpath() is safe to call on a path that
        # doesn't exist on disk (it just normalizes, never raises).
        # current_binary=None (no resolvable binary path at all this run,
        # e.g. tools missing a perplexity_tool) is DELIBERATELY still a
        # mismatch against a stored path: this run cannot confirm it is
        # even using a binary, let alone the same one.
        stored_binary = checkpoint.get("llamacpp_binary")
        if stored_binary is not None:
            current_binary = self._current_llamacpp_binary()
            resolved_stored = os.path.realpath(stored_binary)
            resolved_current = (
                os.path.realpath(current_binary) if current_binary is not None else None
            )
            if resolved_current != resolved_stored:
                reasons.append(
                    f"llamacpp_binary changed ({stored_binary!r} != {current_binary!r})"
                )

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
            "version": 2,
            "seed": self._search_seed,
            "source_model": self._source_identity(),
            "measurement_conditions": self._current_measurement_conditions(),
            "baseline_ppl": self.baseline_ppl,
            "baseline_provenance": self.baseline_provenance,
            "sensitivity_weights": self.sensitivity_weights,
            "probing_provenance": self.probing_provenance,
            # So a resumed run's _enforce_probing_signal_gate sees the same
            # signal a fresh run would have -- see MAJOR 4.
            "weights_degenerate": getattr(self, "weights_degenerate", False),
            # So a resumed run drops the same groups from self._search_groups
            # a fresh run would have -- see SensitivityProber.fixed_groups.
            "fixed_groups": getattr(self, "fixed_groups", {}),
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
                k: self._serialize_measurement(v, include_path=True)
                for k, v in self._measured.items()
            },
            # Additive (2026-08 fail-fast arch-check fix), same fields/
            # rationale as _save_results' tail -- see that site's comment.
            # Also read back by _load_matching_checkpoint's resume gate:
            # a stored llamacpp_binary that DIFFERS from this run's is a
            # different measurement instrument, not just a config change.
            "llamacpp_binary": self._current_llamacpp_binary(),
            "llamacpp_arch_check": getattr(self, "_llamacpp_arch_check", None),
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

    def _build_effective_bpw(
        self, group_tensors: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, float]]:
        """Price every (group, scheme) pair at what it will REALLY cost on this
        model, by asking the writer's own resolution rule rather than the
        registry's advertised bpw.

        Routes through ``v2.resolve.resolve_tensor_type`` -- the parity-tested
        mirror of the writer's Pass-1 chain -- instead of re-deriving those
        rules a third time. Only pairs whose real bpw differs from the
        advertised one are recorded, so the table stays small and
        ``PredictiveScorer._bpw_for`` falls through to the registry for
        everything else.

        Any failure here leaves the table empty, which the predictor treats as
        "no information, behave exactly as before". A wrong size prediction is
        worse than none.
        """
        from magicquant.quant.schemes import get_all_schemes
        from magicquant.quant.ggml_facts import expected_size
        from magicquant.v2.resolve import resolve_tensor_type

        table: Dict[str, Dict[str, float]] = {}
        schemes = [s.name for s in get_all_schemes()]
        for group, infos in group_tensors.items():
            per_scheme: Dict[str, float] = {}
            for scheme in schemes:
                total_bits = 0
                total_elems = 0
                for info in infos:
                    shape = tuple(info["shape"])
                    n = 1
                    for d in shape:
                        n *= d
                    try:
                        actual, _reason = resolve_tensor_type(
                            info["name"], shape, len(shape), group, scheme
                        )
                        total_bits += expected_size(actual, n) * 8
                    except Exception:
                        total_bits = 0
                        break
                    total_elems += n
                if total_bits and total_elems:
                    real = total_bits / total_elems
                    try:
                        from magicquant.quant.schemes import get_scheme_by_name
                        advertised = get_scheme_by_name(scheme).bits_per_weight
                    except ValueError:
                        continue
                    # Only record a genuine divergence; equality means the
                    # registry already tells the truth for this pair.
                    if abs(real - advertised) > 1e-6:
                        per_scheme[scheme] = real
            if per_scheme:
                table[group] = per_scheme
        return table

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
                # Block-32-only bookkeeping rides along on this same pass --
                # it already opens the source and visits every tensor's shape,
                # so this costs no extra I/O. A group qualifies only if EVERY
                # one of its tensors does (strict, matching probing.py's
                # _group_fixed_reason doctrine): a group mixing 32-only and
                # 256-divisible tensors would otherwise get Q5_0 offered for
                # the 256-divisible half too, where Q5_K is genuinely better
                # at identical bpw.
                block32_candidates: Dict[str, bool] = {}
                group_tensors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for info in src.get_all_tensors_info():
                    n = 1
                    for d in info["shape"]:
                        n *= d
                    total_elements += n
                    group = classifier.classify_tensor(info["name"])
                    param_counts[group] += n
                    group_tensors[group].append(info)
                    if is_block32_only_tensor(
                        info["name"], tuple(info["shape"]), len(info["shape"])
                    ):
                        block32_candidates.setdefault(group, True)
                    elif _is_quantization_candidate(
                        info["name"], tuple(info["shape"]), len(info["shape"])
                    ):
                        # A real candidate that is NOT block-32-only
                        # disqualifies the whole group.
                        block32_candidates[group] = False
                self.block32_only_groups = {
                    g for g, ok in block32_candidates.items() if ok
                }
                self._effective_bpw = self._build_effective_bpw(group_tensors)
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
