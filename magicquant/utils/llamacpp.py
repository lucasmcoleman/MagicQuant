"""
llama.cpp integration - Wrapper for calling llama.cpp quantization tools.
"""

import json
import logging
import math
import subprocess
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_log = logging.getLogger(__name__)


# Default timeout for subprocess calls (seconds)
_SUBPROCESS_TIMEOUT = 7200  # 2 hours (35B baseline perplexity ~67 min)
_QUANTIZE_TIMEOUT = 1800   # 30 minutes for large model quantization
_BENCH_TIMEOUT = 300       # 5 minutes for llama-bench pp/tg speed measurement
# A real saved-logits (--kl-divergence-base) file is tens of MB even for a
# tiny model/corpus; a corpus too short for the requested ctx_size*chunks
# makes llama-perplexity exit 0 but write only a ~12-byte header stub. 4 KiB
# comfortably separates "real" from "stub" without depending on model size.
_MIN_LOGITS_FILE_BYTES = 4096


def _env_int(name: str) -> Optional[int]:
    """Parse an optional int env var; unset/empty/invalid -> None (flag omitted)."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _find_tool_in_dirs(possible_names: List[str], search_dirs: List[Path]) -> Optional[str]:
    """Search *search_dirs* (outer) x *possible_names* (inner) for the first
    existing path, returning it as a string, or None if none exist.

    Dirs-outer, names-inner is LOAD-BEARING: a legacy root binary (e.g.
    ``<llamacpp_path>/quantize``) must keep winning over a modern
    build/bin one (e.g. ``<llamacpp_path>/build/bin/llama-quantize``) when
    both exist -- flipping the nesting order would silently change which
    binary gets selected. Matches on ``.exists()`` (not ``.is_file()``), so
    a same-named directory also counts, same as before this was extracted.
    """
    for d in search_dirs:
        for name in possible_names:
            candidate = d / name
            if candidate.exists():
                return str(candidate)
    return None


class LlamaCppTools:
    """Interface to llama.cpp quantization tools."""

    def __init__(
        self,
        llamacpp_path: Optional[str] = None,
        data_file: Optional[str] = None,
        ctx_size: int = 512,
        ngl: Optional[int] = None,
        threads: Optional[int] = None,
    ):
        """
        Initialize llama.cpp tools wrapper.

        Args:
            llamacpp_path: Path to llama.cpp directory (auto-detect if None)
            data_file: Path to the dataset file used for perplexity evaluation
                (e.g. wikitext-2-raw/wiki.test.raw).  When *None* the tool
                will look in common locations relative to the llama.cpp dir.
            ctx_size: Context size for perplexity evaluation (default 512
                for fast evaluation; increase for more accurate results).
            ngl: Number of layers to offload to GPU (``-ngl``) for the
                perplexity/bench subprocess calls. *None* (default) omits
                the flag entirely, matching historical CPU-only behavior.
                Falls back to the ``MAGICQUANT_NGL`` env var when not given.
            threads: CPU thread count (``-t`` for perplexity/bench,
                trailing positional ``nthreads`` for quantize). *None*
                (default) omits it, matching historical behavior. Falls
                back to the ``MAGICQUANT_THREADS`` env var when not given.
        """
        self.llamacpp_path = llamacpp_path or self._find_llamacpp()
        self.quantize_tool = self._find_quantize_tool()
        self.perplexity_tool = self._find_perplexity_tool()
        self.bench_tool = _find_bench_tool(self.perplexity_tool)
        self.data_file = data_file
        self.ctx_size = ctx_size
        self.ngl = ngl if ngl is not None else _env_int("MAGICQUANT_NGL")
        self.threads = threads if threads is not None else _env_int("MAGICQUANT_THREADS")
        # Cap on ctx_size-token chunks per perplexity/KL pass (--chunks).
        # None = whole corpus (historical). A full wikitext pass on a 27B
        # takes ~55 min on this box and a measured search needs ~20 of them;
        # capping trades some statistical resolution for tractable wall-clock
        # while keeping every measurement in the run on the same corpus slice.
        self.ppl_chunks = _env_int("MAGICQUANT_PPL_CHUNKS")
        # Set on first auto-resolution and enforced thereafter -- see
        # _resolve_data_file's pinning wrapper.
        self._pinned_corpus: Optional[str] = None

    def _gpu_flags(self) -> List[str]:
        """``-ngl``/``-t`` flags for perplexity/bench, omitted when unset.

        Reads via getattr (not self.ngl/self.threads directly) so callers
        that construct a bare instance with ``LlamaCppTools.__new__`` and
        set only the attributes they care about (a pattern several existing
        tests use) keep the pre-this-feature omitted-flag behavior instead
        of hitting an AttributeError.
        """
        flags: List[str] = []
        ngl = getattr(self, "ngl", None)
        threads = getattr(self, "threads", None)
        if ngl is not None:
            flags += ["-ngl", str(ngl)]
        if threads is not None:
            flags += ["-t", str(threads)]
        return flags

    @staticmethod
    def _perplexity_batch_flags() -> List[str]:
        """``--batch-size``/``--ubatch-size`` flags shared by every
        llama-perplexity invocation whose reading must be comparable to
        ``calculate_perplexity``'s.

        MINOR fix (F4): ``save_base_logits`` used to omit these while
        ``calculate_perplexity`` passed them, so a measured search's fused
        baseline (``run_measured_search``'s Step 1b, which takes the
        baseline PPL from THIS pass instead of a separate
        ``calculate_perplexity`` call -- see ``save_base_logits``'s
        docstring) was measured under different batching than every
        candidate. Batch size can shift llama.cpp's internal numerics
        slightly (different accumulation order), so "same corpus, same
        ctx_size, different batching" is still a real apples-to-oranges
        comparison, not just a performance knob. Extracted to one place so
        ``calculate_perplexity`` and ``save_base_logits`` cannot drift back
        out of parity with each other.
        """
        return ["--batch-size", "512", "--ubatch-size", "128"]

    def _find_llamacpp(self) -> str:
        """Auto-detect llama.cpp installation."""
        common_paths = [
            Path("C:/llama.cpp"),
            Path("C:/Program Files/llama.cpp"),
            Path.home() / "llama.cpp",
            Path("/usr/local/bin"),
        ]

        for p in common_paths:
            if p.exists():
                return str(p)

        # Try to find in PATH
        which_cmd = "where" if os.name == "nt" else "which"
        try:
            result = subprocess.run(
                [which_cmd, "llama-quantize"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return str(Path(result.stdout.strip()).parent)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise FileNotFoundError(
                "Could not find llama.cpp. Please install or provide path."
            )

    def _find_quantize_tool(self) -> str:
        """Find the quantize executable.

        MagicQuant does not quantize through this binary -- encoding goes
        through magicquant.quant.ggml_binding.ggml_encode (byte-identical to
        llama.cpp, see tests/integration/test_encoder_parity.py). It is kept
        as a llama.cpp-location anchor (construction fails fast if a build
        dir is missing llama-quantize) and for cmd_dry_run's diagnostic log
        line (self.quantize_tool), not as MagicQuant's own quantization path.
        """
        possible_names = ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]
        base = Path(self.llamacpp_path)
        search_dirs = [
            base,
            base / "build" / "bin",
            base / "build",
            base / "bin",
        ]

        found = _find_tool_in_dirs(possible_names, search_dirs)
        if found is None:
            raise FileNotFoundError(f"Could not find quantize tool in {self.llamacpp_path}")
        return found

    def _find_perplexity_tool(self) -> str:
        """Find the perplexity executable."""
        possible_names = ["llama-perplexity.exe", "llama-perplexity", "perplexity.exe", "perplexity"]
        base = Path(self.llamacpp_path)
        search_dirs = [
            base,
            base / "build" / "bin",
            base / "build",
            base / "bin",
        ]

        found = _find_tool_in_dirs(possible_names, search_dirs)
        if found is None:
            raise FileNotFoundError(f"Could not find perplexity tool in {self.llamacpp_path}")
        return found

    def _resolve_data_file(self, data_file: Optional[str] = None) -> Optional[str]:
        """Resolve the dataset file for perplexity evaluation, PINNED after
        first use.

        Priority:
        1. Explicit *data_file* argument
        2. Instance-level ``self.data_file``
        3. Common locations relative to the llama.cpp directory

        Every call made with ``data_file=None`` -- i.e. every implicit,
        instance-driven resolution, which is what every
        ``calculate_perplexity(path, ...)`` call during a search takes --
        resolves to the SAME corpus for this instance's whole lifetime: the
        first such resolution is cached on ``self._pinned_corpus``, and any
        later resolution that would disagree raises loudly instead of
        silently switching corpora. A measured search compares baseline and
        every candidate's PPL against each other under the assumption they
        all ran over the same text; a corpus that silently changed mid-run
        (e.g. ``self.data_file`` mutated, or the wikitext file disappearing
        out from under a fallback search) would make every number in that
        run's search_results.json quietly incomparable (see incident notes,
        point 5: CORPUS PROVENANCE). An explicit *data_file* argument is the
        CALLER choosing a corpus for that one call on purpose (e.g. an
        already-resolved KL corpus threaded through explicitly) and is never
        pinned or checked against the pin.

        Returns:
            Absolute path to the data file, or *None* with a printed error.
        """
        resolved = self._resolve_data_file_uncached(data_file)

        if data_file:
            # Explicit override: this call's choice, not the instance's
            # ambient corpus -- bypasses pinning entirely.
            return resolved

        pinned = getattr(self, "_pinned_corpus", None)
        if pinned is None:
            self._pinned_corpus = resolved
            return resolved
        if resolved != pinned:
            raise RuntimeError(
                f"PPL corpus resolution changed mid-run: this LlamaCppTools "
                f"instance pinned {pinned!r} at first use, but a later "
                f"auto-resolution now produces {resolved!r}. Every "
                "measurement in a run must share one corpus or PPL values "
                "are not comparable (see incident notes, point 5: CORPUS "
                "PROVENANCE). If a genuine corpus change is intended, "
                "construct a new LlamaCppTools instance for it."
            )
        return pinned

    def _resolve_data_file_uncached(self, data_file: Optional[str] = None) -> Optional[str]:
        """Do the actual resolution work (see ``_resolve_data_file``'s
        pinning wrapper, which is what every other caller should use)."""
        candidate = data_file or self.data_file

        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                return str(candidate_path.resolve())

        # Search common locations relative to the llama.cpp directory
        base = Path(self.llamacpp_path)
        search_paths = [
            base / "wikitext-2-raw" / "wiki.test.raw",
            base / "wikitext-2" / "wiki.test.raw",
            base / "models" / "wikitext-2-raw" / "wiki.test.raw",
            base.parent / "wikitext-2-raw" / "wiki.test.raw",
        ]

        for p in search_paths:
            if p.is_file():
                return str(p.resolve())

        # Last resort: MagicQuant's bundled calibration corpus. Much smaller
        # than wikitext (noisier per-candidate PPL, though baseline and
        # candidates stay internally comparable since they share it), but far
        # better than aborting a configured run because this particular
        # llama.cpp build dir doesn't happen to have wikitext next to it
        # (bit for real when llamacpp_path pointed at a ROCmFPX build dir).
        try:
            from magicquant.imatrix import DEFAULT_CORPUS_PATH

            if DEFAULT_CORPUS_PATH.is_file():
                print(
                    f"WARNING: no wikitext corpus found near {self.llamacpp_path} "
                    f"-- falling back to the bundled calibration corpus "
                    f"({DEFAULT_CORPUS_PATH.name}). For stabler perplexity "
                    "comparisons, place wikitext-2-raw/wiki.test.raw in the "
                    "llama.cpp dir or pass data_file=<path>."
                )
                return str(DEFAULT_CORPUS_PATH.resolve())
        except ImportError:
            pass

        # Nothing found -- print a clear message
        print(
            "ERROR: No perplexity data file found.\n"
            "  llama-perplexity requires a dataset file (e.g. wikitext-2-raw/wiki.test.raw).\n"
            "  Download it with:\n"
            "    curl -LO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip\n"
            "    unzip wikitext-2-raw-v1.zip\n"
            f"  Then place 'wikitext-2-raw/wiki.test.raw' inside {self.llamacpp_path}\n"
            "  or pass data_file=<path> to LlamaCppTools / calculate_perplexity()."
        )
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(subprocess.CalledProcessError),
        reraise=True,
    )
    def _run_perplexity_subprocess(
        self,
        cmd: list[str],
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """Run the perplexity subprocess with retry logic.

        Retries up to 3 times with exponential backoff on
        CalledProcessError (e.g. transient GPU OOM or file lock).
        """
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )

    def _effective_chunks(self, chunks: int) -> str:
        """``--chunks`` value shared by ``save_base_logits`` and
        ``calculate_kl_divergence``: the caller's explicit *chunks* if given
        (!= -1), else this instance's ``MAGICQUANT_PPL_CHUNKS`` cap, else -1
        (whole corpus).

        Uses ``getattr(self, "ppl_chunks", None)`` rather than
        ``self.ppl_chunks`` -- several tests construct a bare instance via
        ``LlamaCppTools.__new__`` without setting ``ppl_chunks``, and this
        must keep degrading to "no cap" instead of raising AttributeError.
        The ``or -1`` falsy-collapse is preserved verbatim (``ppl_chunks ==
        0`` still yields -1).
        """
        return str(chunks if chunks != -1 else (getattr(self, "ppl_chunks", None) or -1))

    def _run_subprocess_or_none(
        self, cmd: List[str], timeout: int, label: str
    ) -> Optional[subprocess.CompletedProcess]:
        """Run *cmd* via ``_run_perplexity_subprocess``, printing
        ``"<label> failed: <stderr>"`` / ``"<label> timed out"`` and
        returning None instead of propagating on the two subprocess-failure
        exceptions every measurement call site handles the same way.

        Catches EXACTLY ``subprocess.CalledProcessError`` and
        ``subprocess.TimeoutExpired`` -- never a broader ``OSError`` or
        ``Exception``. A missing/wrong-arch binary raises OSError/
        FileNotFoundError from ``subprocess.run`` itself, and that must keep
        propagating OUT of this helper: the orchestrator's measured-search
        loop depends on it to fail a candidate rather than the call site
        silently recording "no measurement" (see
        tests/test_orchestrator_measurement.py::
        test_measured_search_survives_kl_and_bench_raising_oserror).
        """
        try:
            return self._run_perplexity_subprocess(cmd, timeout=timeout)
        except subprocess.CalledProcessError as e:
            print(f"{label} failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print(f"{label} timed out")
            return None

    def calculate_perplexity(
        self,
        model_path: str,
        verbose: bool = True,
        data_file: Optional[str] = None,
        ctx_size: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate perplexity for a model.

        Args:
            model_path: Path to GGUF model
            verbose: Print output
            data_file: Path to dataset file (overrides instance default)
            ctx_size: Context size (overrides instance default)

        Returns:
            Perplexity value or None if failed
        """
        resolved_data_file = self._resolve_data_file(data_file)
        if resolved_data_file is None:
            return None

        effective_ctx = ctx_size if ctx_size is not None else self.ctx_size

        cmd = [
            self.perplexity_tool,
            "-m", model_path,
            "-f", resolved_data_file,
            "--ctx-size", str(effective_ctx),
        ] + self._perplexity_batch_flags() + self._gpu_flags()
        # Deliberately NOT _effective_chunks(): this site omits --chunks
        # entirely when ppl_chunks is None, whereas the KL/logits sites always
        # pass a value (-1 sentinel). Do not "finish the fold".
        ppl_chunks = getattr(self, "ppl_chunks", None)
        if ppl_chunks is not None:
            cmd += ["--chunks", str(ppl_chunks)]

        if verbose:
            print(f"Calculating perplexity for {Path(model_path).name}...")

        result = self._run_subprocess_or_none(cmd, _SUBPROCESS_TIMEOUT, "Perplexity calculation")
        if result is None:
            return None

        # Parse perplexity from output. llama-perplexity prints the
        # "Final estimate: PPL = ..." line to STDERR, not stdout, so scan
        # both streams (matches tools/calibrate_noise_factors.py). Parsing
        # stdout only silently returned None here — collapsing the entire
        # measured search + QAT validation to prediction-only.
        ppl = _parse_perplexity_output(
            (result.stdout or "") + "\n" + (result.stderr or "")
        )

        if ppl is not None and verbose:
            print(f"  Perplexity: {ppl:.4f}")
        return ppl

    def bench(
        self,
        model_path: str,
        *,
        n_prompt: Optional[int] = None,
        n_gen: Optional[int] = None,
        reps: Optional[int] = None,
        timeout: int = _BENCH_TIMEOUT,
    ) -> Optional[dict]:
        """Measure prompt-processing and token-generation throughput.

        Runs ``llama-bench -m <model> -p <n_prompt> -n <n_gen> -r <reps> -o
        json``, which reports two rows: a prompt-processing row (n_gen == 0,
        whose avg_ts is the pp t/s) and a generation row (n_prompt == 0,
        whose avg_ts is the tg t/s). Confirmed empirically against the
        ROCmFPX llama-bench build (see tests/test_llamacpp_measure.py).

        Defaults are 3 reps of 128 generated tokens (was 2x32): a short
        low-rep tg run swings widely -- the same 27B config was measured at
        4.44 and 8.19 t/s across invocations (2026-07-05), largely from
        thermal state and a *coexisting* GPU process (e.g. an unrelated
        llama-server) competing for the same unified memory bandwidth. More
        reps + a longer generation average the per-invocation noise; a
        candidate's own reported ``tg_ts_std`` lets callers judge confidence.
        For a trustworthy A/B, bench candidates back-to-back in one window and
        quiesce other GPU users. Env overrides: MAGICQUANT_BENCH_REPS /
        MAGICQUANT_BENCH_NGEN / MAGICQUANT_BENCH_NPROMPT.

        Args:
            model_path: Path to GGUF model to benchmark.
            n_prompt: Prompt length (tokens) for the pp test (default 32).
            n_gen: Generation length (tokens) for the tg test (default 128).
            reps: Repetitions per test (-r) (default 3).
            timeout: Subprocess timeout in seconds.

        Returns:
            ``{"pp_ts", "tg_ts", "pp_ts_std", "tg_ts_std"}`` (tokens/sec;
            the ``*_std`` are the per-row stddev, or None if the build omits
            it), or None if llama-bench is unavailable or the run/parse
            failed.
        """
        if not self.bench_tool:
            print("llama-bench not found; skipping speed measurement")
            return None

        n_prompt = n_prompt if n_prompt is not None else (_env_int("MAGICQUANT_BENCH_NPROMPT") or 32)
        n_gen = n_gen if n_gen is not None else (_env_int("MAGICQUANT_BENCH_NGEN") or 128)
        reps = reps if reps is not None else (_env_int("MAGICQUANT_BENCH_REPS") or 3)

        cmd = [
            self.bench_tool,
            "-m", model_path,
            "-p", str(n_prompt),
            "-n", str(n_gen),
            "-r", str(reps),
            "-o", "json",
        ] + self._gpu_flags()

        result = self._run_subprocess_or_none(cmd, timeout, "llama-bench")
        if result is None:
            return None

        parsed = _parse_bench_json(result.stdout or "")
        if parsed is None:
            print("llama-bench: could not parse pp_ts/tg_ts from JSON output")
        return parsed

    def save_base_logits(
        self,
        base_model_path: str,
        corpus_path: str,
        out_logits_path: str,
        *,
        ctx_size: int = 512,
        chunks: int = -1,
        timeout: int = _SUBPROCESS_TIMEOUT,
    ) -> Optional[float]:
        """Run the base model once, saving per-token logits to disk.

        These saved logits are the reference distribution that later
        ``calculate_kl_divergence`` calls compare quantized models against.
        Wraps ``llama-perplexity -m <base> -f <corpus>
        --kl-divergence-base <out_logits_path>``.

        This pass, even without ``--kl-divergence``, still prints the normal
        "Final estimate: PPL = ..." line for the base model itself -- so a
        caller that also needs the base model's own perplexity (e.g. as the
        measured-search baseline) can get it from THIS single invocation
        instead of running a separate ``calculate_perplexity`` pass over the
        same model/corpus (see ``run_measured_search``'s baseline+KL fusion).
        Passes the same ``--batch-size``/``--ubatch-size`` flags as
        ``calculate_perplexity`` (via ``_perplexity_batch_flags``) so that
        fused baseline is measured under identical batching to every
        candidate it's compared against (MINOR fix, F4: this used to omit
        them).

        Args:
            base_model_path: Path to the (typically un-quantized/BF16 or
                highest-fidelity) reference GGUF model.
            corpus_path: Path to a plain-text corpus file.
            out_logits_path: Where to write the saved logits.
            ctx_size: Context size for the pass.
            chunks: Number of context-sized chunks to process (-1 = all).
            timeout: Subprocess timeout in seconds.

        Returns:
            The parsed "Final estimate: PPL" value from this pass on success,
            or *None* if the subprocess failed, or the output logits file is
            missing/stub-sized, or no PPL line could be parsed. The stub-file
            guard always wins: a stub-sized output file means failure/None
            even if a PPL line happened to parse from stdout/stderr.
        """
        cmd = [
            self.perplexity_tool,
            "-m", base_model_path,
            "-f", corpus_path,
            "--kl-divergence-base", out_logits_path,
            "--ctx-size", str(ctx_size),
            "--chunks", self._effective_chunks(chunks),
        ] + self._perplexity_batch_flags() + self._gpu_flags()

        result = self._run_subprocess_or_none(cmd, timeout, "Saving base logits")
        if result is None:
            return None

        # llama-perplexity exits 0 even when it can't actually run (e.g. the
        # corpus tokenizes to fewer tokens than ctx_size*chunks requires) --
        # it still creates the output file, but as a ~12-byte header stub
        # with no real logits (empirically: a valid file is tens of MB for a
        # small model/corpus). is_file() alone can't tell success from a
        # stub, so also require a minimum size.
        out_path = Path(out_logits_path)
        if not (out_path.is_file() and out_path.stat().st_size > _MIN_LOGITS_FILE_BYTES):
            return None

        return _parse_perplexity_output((result.stdout or "") + "\n" + (result.stderr or ""))

    def calculate_kl_divergence(
        self,
        quant_model_path: str,
        base_logits_path: str,
        corpus_path: str,
        *,
        ctx_size: int = 512,
        chunks: int = -1,
        timeout: int = _SUBPROCESS_TIMEOUT,
    ) -> Optional[dict]:
        """Compute KL divergence of a quantized model against saved base logits.

        Wraps ``llama-perplexity -m <quant> -f <corpus> --kl-divergence
        --kl-divergence-base <base_logits_path>`` and parses the "KL
        divergence statistics" block it prints to stdout. Label/format
        confirmed empirically (see tests/test_llamacpp_measure.py):

            Mean    KLD:  -0.000019 +/-   0.000001
            Maximum KLD:   0.000001
            90.0%   KLD:  -0.000005

        Args:
            quant_model_path: Path to the quantized GGUF model to evaluate.
            base_logits_path: Path to logits previously written by
                save_base_logits().
            corpus_path: Path to the same plain-text corpus used to save
                the base logits (chunking must match).
            ctx_size: Context size for the pass (must match the base-logits
                run).
            chunks: Number of context-sized chunks to process (-1 = all;
                must match the base-logits run).
            timeout: Subprocess timeout in seconds.

        Returns:
            {"mean_kl": float, "max_kl": float, "p90_kl": float, "ppl": float,
            "ppl_err": float} (all but "mean_kl" omitted if absent from the
            output), or None if the run failed or the "Mean KLD" line
            couldn't be found. "ppl" is this pass's own "Mean PPL(Q)" --
            the evaluated (quantized) model's perplexity over the same
            chunks, printed in the "Perplexity statistics" block that always
            precedes "KL divergence statistics" -- so a caller needing both
            perplexity and KL divergence for a candidate can get both from
            this ONE invocation instead of a separate calculate_perplexity
            call (see run_measured_search's candidate-measurement fusion).
        """
        cmd = [
            self.perplexity_tool,
            "-m", quant_model_path,
            "-f", corpus_path,
            "--kl-divergence",
            "--kl-divergence-base", base_logits_path,
            "--ctx-size", str(ctx_size),
            "--chunks", self._effective_chunks(chunks),
        ] + self._perplexity_batch_flags() + self._gpu_flags()

        result = self._run_subprocess_or_none(cmd, timeout, "KL divergence calculation")
        if result is None:
            return None

        parsed = _parse_kl_output((result.stdout or "") + "\n" + (result.stderr or ""))
        if parsed is None:
            print("KL divergence: could not find 'Mean KLD' in output")
        return parsed


# Measurement-failure markers: llama-perplexity prints these and still
# exits 0 (perplexity.cpp), so a caller checking only the return code sees
# "success". A NaN model in particular hits the first one and then never
# reaches the "Final estimate: PPL =" print at all (perplexity.cpp:646-657
# gates it on nll2 > 0) -- so scanning for a PPL number in this output is
# guaranteed to either find nothing real or (pre-fix) find something bogus.
_NEGATIVE_STDDEV_MARKER = "Unexpected negative standard deviation of log(prob)"
_FAILED_DECODE_MARKER = "failed to decode"

# "Final estimate: PPL = 5.2345 +/- 0.0123" -- the only real per-run summary
# line llama-perplexity prints. Accepts a literal "nan"/"inf" token too (not
# just digits): a hypothetical future llama.cpp build that prints the final
# line even for a degenerate run must still be caught by the
# not-finite check below rather than silently failing to match at all.
_FINAL_ESTIMATE_RE = re.compile(
    r"Final estimate.*?PPL\s*=\s*(-?\d+\.?\d*|-?nan|-?inf)", re.IGNORECASE
)
# The KL block's "Mean PPL(Q) : 13.821636 +/- 3.046334" -- the evaluated
# model's own perplexity from a --kl-divergence run (see _parse_kl_output,
# which extracts the same field for KL-specific callers). Recognized here
# too so any caller that feeds KL output through this generic parser (rather
# than the KL-specific one) still gets a real PPL instead of nothing.
_MEAN_PPL_Q_RE = re.compile(
    r"Mean PPL\(Q\)\s*:\s*(-?\d+\.?\d*|-?nan|-?inf)\s*(?:\xb1|\+/-)", re.IGNORECASE
)


def _to_finite_float(raw: str) -> Optional[float]:
    """Parse *raw* as a float, returning None for NaN/Inf (never a sentinel
    that later arithmetic would silently propagate as "real" data)."""
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _tail(output: str, n: int = 20) -> str:
    """Last *n* lines of *output*, for diagnostic logging."""
    return "\n".join(output.splitlines()[-n:])


def _parse_perplexity_output(output: str) -> Optional[float]:
    """Extract perplexity value from llama-perplexity output.

    Accepts ONLY two forms (see module-level regexes above), scanned in
    REVERSE so the last (most complete) occurrence wins:
      1. "Final estimate: PPL = <float>"
      2. The KL block's "Mean PPL(Q) : <float> +/- <err>"

    Everything else that used to match here is GONE -- this is the fix for
    a real incident (2026-07): a measured search recorded sensitivity
    probes of 2.6-2.8 against a baseline of 34.8363, driving 8/9 group
    sensitivities to exactly 0.0. Root cause: llama.cpp prints a PROGRESS
    line "perplexity: 2.74 seconds per pass - ETA 4.5 minutes"
    (tools/perplexity/perplexity.cpp:605) for every chunk, and only prints
    "Final estimate: PPL = <x>" when nll2 > 0 (perplexity.cpp:646-657) -- a
    NaN model skips that and instead prints
    "Unexpected negative standard deviation of log(prob)", STILL EXITING 0.
    The old second pattern (``[Pp]erplexity\\s*[:=]\\s*(\\d+\\.?\\d*)``) and
    third ("any line containing PPL" + a float) both matched the progress
    line, so the parser returned 2.74 (the seconds-per-pass number) instead
    of None. Tell-tale in hindsight: bogus values had <=2 decimals (the
    progress line's %.2f), real ones had 4 (%.4lf) -- but the real fix is to
    never accept anything but the two named forms.

    Also returns None -- a genuine measurement failure, not "line not
    found" -- when the output contains either measurement-failure marker
    (NaN-model stddev message, or a decode failure), even if some other
    line happened to look parseable.

    Foundry discards llama-perplexity's stdout/stderr on a "successful"
    (exit 0) subprocess call, so a None here is otherwise undiagnosable
    after the fact -- the last ~20 lines of output are logged at WARNING
    whenever this returns None.

    Args:
        output: Combined stdout+stderr from llama-perplexity (the
            "Final estimate: PPL =" line is emitted on stderr).

    Returns:
        The parsed perplexity float, or None if no real measurement is
        present (not found, or found but NaN/Inf, or an explicit failure
        marker is present).
    """
    if _NEGATIVE_STDDEV_MARKER in output or _FAILED_DECODE_MARKER in output:
        _log.warning(
            "llama-perplexity output contains a measurement-failure marker "
            "(NaN model / decode failure) -- refusing to parse a PPL from "
            "it. Last %d lines of output:\n%s",
            20, _tail(output),
        )
        return None

    for line in reversed(output.split("\n")):
        m = _FINAL_ESTIMATE_RE.search(line)
        if m:
            value = _to_finite_float(m.group(1))
            if value is None:
                _log.warning(
                    "llama-perplexity printed a non-finite 'Final estimate' "
                    "PPL (%r) -- treating as a measurement failure. Last %d "
                    "lines of output:\n%s",
                    m.group(1), 20, _tail(output),
                )
            return value
        m = _MEAN_PPL_Q_RE.search(line)
        if m:
            value = _to_finite_float(m.group(1))
            if value is None:
                _log.warning(
                    "llama-perplexity printed a non-finite 'Mean PPL(Q)' "
                    "(%r) -- treating as a measurement failure. Last %d "
                    "lines of output:\n%s",
                    m.group(1), 20, _tail(output),
                )
            return value

    _log.warning(
        "no 'Final estimate: PPL =' or 'Mean PPL(Q)' line found in "
        "llama-perplexity output -- returning None instead of guessing. "
        "Last %d lines of output:\n%s",
        20, _tail(output),
    )
    return None


def _parse_bench_json(text: str) -> Optional[dict]:
    """Extract pp_ts/tg_ts from llama-bench's ``-o json`` output.

    llama-bench (with ``-o json``) prints a JSON array with one object per
    test row. Confirmed empirically: the prompt-processing row has
    ``n_gen == 0`` (its ``avg_ts`` is the pp t/s); the generation row has
    ``n_prompt == 0`` (its ``avg_ts`` is the tg t/s).

    Args:
        text: llama-bench stdout (the JSON array; some builds may print
            extra banner/log lines around it, so the outermost ``[...]``
            is isolated before parsing).

    Returns:
        {"pp_ts": float, "tg_ts": float}, or None if the JSON can't be
        parsed or the expected rows aren't both present.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    pp_ts = None
    tg_ts = None
    pp_std = None
    tg_std = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        # llama-bench names the per-row spread "stddev" on some builds and
        # "stddev_ts" on others; accept either, None if absent.
        std = row.get("stddev_ts", row.get("stddev"))
        if pp_ts is None and row.get("n_gen") == 0:
            pp_ts = row.get("avg_ts")
            pp_std = std
        if tg_ts is None and row.get("n_prompt") == 0:
            tg_ts = row.get("avg_ts")
            tg_std = std

    if pp_ts is None or tg_ts is None:
        return None

    return {
        "pp_ts": float(pp_ts),
        "tg_ts": float(tg_ts),
        "pp_ts_std": float(pp_std) if pp_std is not None else None,
        "tg_ts_std": float(tg_std) if tg_std is not None else None,
    }


def _parse_kl_output(output: str) -> Optional[dict]:
    """Extract KL-divergence statistics from llama-perplexity output.

    ``llama-perplexity --kl-divergence`` prints a "KL divergence
    statistics" block to stdout with lines like (real format, confirmed by
    running a q8_0 model against its own saved logits -- see
    tests/test_llamacpp_measure.py)::

        ====== KL divergence statistics ======
        Mean    KLD:  -0.000019 ±   0.000001
        Maximum KLD:   0.000001
        90.0%   KLD:  -0.000005

    It also prints a "Perplexity statistics" block just above the KL block,
    including the evaluated model's own perplexity::

        Mean PPL(Q)                   :  13.821636 ±   3.046334

    Args:
        output: Combined stdout+stderr from llama-perplexity.

    Returns:
        {"mean_kl": float, "max_kl": float, "p90_kl": float, "ppl": float,
        "ppl_err": float} (all but "mean_kl" omitted if not present in the
        output), or None if no "Mean ... KLD:" line is found.
    """
    result: dict = {}

    # llama.cpp prints "Mean    KLD:   0.154163 ±   0.001946". Capturing the
    # error term is what makes KL usable as a PROBE signal rather than just a
    # report: it is computed over every evaluated token (~50k at 100 chunks)
    # instead of over 100 chunk means, which is why one real probe resolved
    # at 79 sigma by KL against 0.55 sigma for the same probe judged by
    # perplexity. Optional in the pattern -- older builds print a bare mean.
    m = re.search(
        r"Mean\s+KLD:\s*(-?\d+\.?\d*)(?:\s*(?:\xb1|\+/-)\s*(\d+\.?\d*))?",
        output,
    )
    if not m:
        return None
    result["mean_kl"] = float(m.group(1))
    if m.group(2) is not None:
        result["mean_kl_err"] = float(m.group(2))

    m = re.search(r"Maximum\s+KLD:\s*(-?\d+\.?\d*)", output)
    if m:
        result["max_kl"] = float(m.group(1))

    m = re.search(r"90\.0%\s+KLD:\s*(-?\d+\.?\d*)", output)
    if m:
        result["p90_kl"] = float(m.group(1))

    # The evaluated model's own perplexity, from the "Perplexity statistics"
    # block that precedes "KL divergence statistics" -- lets a caller fuse a
    # candidate's PPL + KL measurement into this one invocation instead of
    # two (see llamacpp.py's calculate_kl_divergence docstring / orchestrator
    # .py's run_measured_search).
    m = re.search(
        r"Mean PPL\(Q\)\s*:\s*(\d+\.?\d*)\s*(?:\xb1|\+/-)\s*(\d+\.?\d*)", output
    )
    if m:
        result["ppl"] = float(m.group(1))
        result["ppl_err"] = float(m.group(2))

    return result


def _find_bench_tool(perplexity_tool_path: str) -> Optional[str]:
    """Locate the llama-bench executable next to the resolved perplexity tool.

    Mirrors LlamaCppTools._find_perplexity_tool, but returns None instead of
    raising when the binary is absent -- bench() must degrade gracefully
    (return None) rather than prevent LlamaCppTools from being constructed.
    """
    possible_names = ["llama-bench.exe", "llama-bench"]
    base = Path(perplexity_tool_path).parent

    found = _find_tool_in_dirs(possible_names, [base])
    if found is not None:
        return found

    # Fall back to PATH
    which_cmd = "where" if os.name == "nt" else "which"
    try:
        result = subprocess.run(
            [which_cmd, "llama-bench"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        found = result.stdout.strip().splitlines()
        return found[0] if found else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
