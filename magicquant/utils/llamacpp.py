"""
llama.cpp integration - Wrapper for calling llama.cpp quantization tools.
"""

import json
import logging
import math
import signal
import subprocess
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_log = logging.getLogger(__name__)


def _env_int(name: str) -> Optional[int]:
    """Parse an optional int env var; unset/empty/invalid -> None (flag omitted)."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# Default timeout for subprocess calls (seconds). MAGICQUANT_SUBPROCESS_TIMEOUT
# overrides the flat floor; large artifacts additionally scale this up per
# call site -- see LlamaCppTools._measure_timeout below. (2026-08 field
# report: a 37.8GB candidate's KL pass hit this flat 2h cap while healthy at
# ~96% CPU -- not a hang -- then the PPL fallback hit the same cap, burning
# 4h of compute for zero measurements.)
# Same `_env_int(...) or <default>` falsy-collapse as every other env knob
# in this file (see _effective_chunks' note on MAGICQUANT_PPL_CHUNKS): unset,
# empty, non-numeric, AND a literal "0" all collapse to the 7200 default --
# there is no way to request an actual zero timeout. A negative value (e.g.
# "-1") parses fine and is used as-is, unvalidated -- callers are trusted not
# to pass one.
_SUBPROCESS_TIMEOUT = _env_int("MAGICQUANT_SUBPROCESS_TIMEOUT") or 7200  # 2 hours (35B baseline perplexity ~67 min)
_QUANTIZE_TIMEOUT = 1800   # 30 minutes for large model quantization
_BENCH_TIMEOUT = 300       # 5 minutes for llama-bench pp/tg speed measurement
# Conservative floor bandwidth (bytes/sec) for scaling a measurement
# timeout to a specific file's size (LlamaCppTools._measure_timeout). Its
# ONLY job is to keep the cap FINITE for a genuine hang while never
# undercutting a healthy bandwidth-bound pass: 4 MB/s is far below any real
# llama-perplexity throughput on this box, so a run still making progress
# never hits this cap, while a real 0 B/s hang still times out eventually.
_MIN_MEASURE_BANDWIDTH = 4_000_000  # 4 MB/s
# How often, while waiting on a measurement subprocess, to check whether the
# child is actually still alive. Purely a liveness poll: the real wait is
# still bounded by the caller's timeout, this just lets us notice a dead
# child long before that timeout fires.
_LIVENESS_POLL_INTERVAL = 30  # seconds
# Once the child has exited, how long to keep draining its pipes before
# declaring them abandoned. A healthy child's pipes reach EOF the instant the
# last writer closes them, so this only has to cover the gap between "child
# reaped" and "buffered output read". 60s is enormous for that and still
# nothing next to a multi-hour measurement timeout.
_ABANDONED_PIPE_GRACE = 60  # seconds
# A real saved-logits (--kl-divergence-base) file is tens of MB even for a
# tiny model/corpus; a corpus too short for the requested ctx_size*chunks
# makes llama-perplexity exit 0 but write only a ~12-byte header stub. 4 KiB
# comfortably separates "real" from "stub" without depending on model size.
_MIN_LOGITS_FILE_BYTES = 4096
# Ceiling for the bounded load check (LlamaCppTools.verify_model_loads). This
# is deliberately NOT _measure_timeout: loading a model and running a single
# 512-token chunk is I/O-bound setup, not a measurement, so it gets a tight
# ceiling and a much more optimistic bandwidth floor. A load check that can
# run for hours defeats its own purpose.
_LOAD_CHECK_TIMEOUT = 600           # 10 minutes
_LOAD_CHECK_BANDWIDTH = 50_000_000  # 50 MB/s -- read throughput, not compute


# ---------------------------------------------------------------------------
# Process-group isolation for measurement subprocesses
#
# The stall cleanup below must be able to kill everything the child spawned,
# including a grandchild that outlived it and inherited its pipe write ends.
#
# POSIX: ``start_new_session=True`` makes the child lead its own session, so
# ``os.killpg`` on that group can never reach MagicQuant's own.
#
# Windows has no process groups in that sense, and the obvious substitute --
# ``taskkill /T`` -- walks the parent->child tree, which fails for the exact
# case this exists for (child already exited, orphaned grandchild holding the
# pipe; the tree walk finds nothing). The real equivalent is a Job Object:
# descendants inherit job membership automatically and TerminateJobObject
# kills every member regardless of tree state. Done via ctypes on kernel32.
#
# Capability-detected (hasattr), not platform-sniffed. A platform with neither
# degrades to ``proc.kill()`` on the child alone, and says so once.
# ---------------------------------------------------------------------------
_HAS_POSIX_PROCESS_GROUPS = hasattr(os, "getpgid") and hasattr(os, "killpg")

if not _HAS_POSIX_PROCESS_GROUPS and os.name == "nt":
    import ctypes
    from ctypes import wintypes as _wt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.restype = _wt.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [_wt.LPVOID, _wt.LPCWSTR]
    _kernel32.OpenProcess.restype = _wt.HANDLE
    _kernel32.OpenProcess.argtypes = [_wt.DWORD, _wt.BOOL, _wt.DWORD]
    _kernel32.AssignProcessToJobObject.restype = _wt.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [_wt.HANDLE, _wt.HANDLE]
    _kernel32.TerminateJobObject.restype = _wt.BOOL
    _kernel32.TerminateJobObject.argtypes = [_wt.HANDLE, _wt.UINT]
    _kernel32.CloseHandle.restype = _wt.BOOL
    _kernel32.CloseHandle.argtypes = [_wt.HANDLE]
    # AssignProcessToJobObject needs PROCESS_SET_QUOTA | PROCESS_TERMINATE.
    _PROCESS_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001
else:
    _kernel32 = None

_warned_no_group_isolation = False


def _popen_isolation_kwargs() -> dict:
    """Extra ``Popen`` kwargs that put the child in its own POSIX session.

    Nothing on Windows: isolation there comes from the Job Object attached
    right after the spawn (``_capture_process_group``), not from a spawn flag.
    """
    return {"start_new_session": True} if _HAS_POSIX_PROCESS_GROUPS else {}


def _capture_process_group(proc: "subprocess.Popen"):
    """Return the opaque token ``_kill_process_group`` needs, captured NOW.

    POSIX: the pgid. It MUST be read right after the spawn, while the leader
    is certainly alive -- by the time a stall is detected the child is
    typically already reaped and ``os.getpgid(proc.pid)`` raises
    ``ProcessLookupError``, silently skipping the kill and leaving exactly the
    grandchild we came to remove. (Caught by
    tests/test_measurement_pipe_stall.py, which failed only on the NEXT run
    because the surviving grandchild wrote its marker 30s later.)

    Windows: a Job Object handle with the child assigned. Children the child
    spawns from here on inherit membership. A grandchild spawned in the
    microseconds between CreateProcess returning and the assignment would not
    -- acceptable: llama.cpp's tools do not fork helpers at startup, and the
    pathology this guards against is a long-lived leaked descriptor, not a
    startup race.

    ``None`` when the platform offers neither (or the OS call failed):
    cleanup then falls back to killing the child alone.
    """
    global _warned_no_group_isolation
    if _HAS_POSIX_PROCESS_GROUPS:
        try:
            return os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return None
    if _kernel32 is not None:
        job = _kernel32.CreateJobObjectW(None, None)
        if job:
            hproc = _kernel32.OpenProcess(
                _PROCESS_SET_QUOTA_AND_TERMINATE, False, proc.pid
            )
            if hproc:
                try:
                    if _kernel32.AssignProcessToJobObject(job, hproc):
                        return job
                    err = ctypes.get_last_error()
                finally:
                    _kernel32.CloseHandle(hproc)
            else:
                err = ctypes.get_last_error()
            _kernel32.CloseHandle(job)
        else:
            err = ctypes.get_last_error()
        _log.warning(
            "could not attach measurement subprocess %s to a Job Object "
            "(WinError %s); stall cleanup will kill only the child, not "
            "anything it spawned", proc.pid, err,
        )
        return None
    if not _warned_no_group_isolation:
        _warned_no_group_isolation = True
        _log.warning(
            "this platform has neither POSIX process groups nor Windows Job "
            "Objects; measurement stall cleanup will kill only the child"
        )
    return None


def _kill_process_group(proc: "subprocess.Popen", group) -> None:
    """Kill the child's whole process group / job, then *proc* itself.

    The group matters: the failure this exists for is a grandchild that
    outlived the child and inherited its stdout/stderr write end. Killing
    only the child leaves that grandchild holding the pipe forever.

    *group* is whatever ``_capture_process_group`` returned for this spawn --
    a pgid on POSIX, a Job Object handle on Windows, or ``None`` (child-only
    fallback). Never looked up here; see that function for why.
    """
    if group is not None:
        if _HAS_POSIX_PROCESS_GROUPS:
            try:
                os.killpg(group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        elif _kernel32 is not None:
            _kernel32.TerminateJobObject(group, 1)
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _release_process_group(group) -> None:
    """Drop the token on every exit path. POSIX: nothing to release.

    Windows: closes the Job Object handle WITHOUT terminating its members --
    the job is created without KILL_ON_JOB_CLOSE on purpose, for parity with
    POSIX, where a clean child exit never signals its grandchildren. Killing
    is ``_kill_process_group``'s job, on the stall/timeout paths only.
    """
    if group is not None and not _HAS_POSIX_PROCESS_GROUPS and _kernel32 is not None:
        _kernel32.CloseHandle(group)


def _run_captured(
    cmd: List[str],
    timeout: Optional[float] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """``subprocess.run(cmd, capture_output=True, text=True, timeout=...)``
    that cannot hang past the death of its own child.

    Why this is not just ``subprocess.run``: ``run`` waits for the pipes to
    reach EOF, not for the child to exit. EOF arrives only when the LAST
    writer closes the write end -- and if anything else inherited that
    descriptor, the child can exit and be left unreaped while the parent
    sits in ``poll()`` until the full timeout expires.

    Observed 2026-08-13 on a ``--magicquant-budget-gib`` run: child
    ``llama-perplexity`` in state ``Z`` (exited, never reaped), parent in
    ``poll_schedule_timeout`` holding two read ends, six other processes
    still holding the write ends, 70 minutes of no output at 0% CPU on the
    measurement. Because ``_measure_timeout`` now scales with file size, a
    62 GB probe would have sat there ~4h (~8h for a KL leg) instead of the
    old flat 2h -- the size scaling is correct, but it turns this stall from
    expensive into most of a day.

    This does NOT try to prevent the descriptor leak -- ``close_fds=True``
    has been Python's default since 3.2 and is already in force, so the
    inheriting processes are not ones MagicQuant spawns through
    ``subprocess``. It bounds the damage instead: once the child is gone,
    its output is complete, so waiting longer than ``_ABANDONED_PIPE_GRACE``
    can only ever return the same bytes.

    Contract is deliberately identical to ``subprocess.run`` so callers and
    their tests do not change: raises ``subprocess.TimeoutExpired`` on the
    real timeout AND on an abandoned-pipe stall (the callers that degrade
    gracefully -- ``_run_subprocess_or_none`` -- already catch exactly that),
    ``subprocess.CalledProcessError`` when *check* and the exit status is
    non-zero, and lets ``OSError``/``FileNotFoundError`` from the spawn
    itself propagate untouched (``tests/test_orchestrator_measurement.py``
    pins that escape path -- never widen it).
    """
    # ``timeout=None`` means "no deadline", exactly as in ``subprocess.run``.
    # Getting this wrong is not theoretical: the first version computed
    # ``time.monotonic() + timeout`` unconditionally, so every caller that took
    # the documented default raised ``TypeError: unsupported operand type(s)
    # for +: 'float' and 'NoneType'`` before spawning anything. That broke
    # ``capture_imatrix`` (whose ``timeout`` defaults to None) on the
    # Qwen3.8-27B campaign, and because ``ensure_imatrix`` catches capture
    # failures and continues, the only trace was a warning -- the run went on
    # to quantize UNWEIGHTED against an explicit ``use_imatrix: true``.
    # The abandoned-pipe bound below still applies when there is no deadline;
    # that guard is about the child being gone, not about elapsed time.
    deadline = None if timeout is None else time.monotonic() + timeout
    # Own session (POSIX) / own Job Object (Windows) so a stall can be cleaned
    # up wholesale -- see the process-group seam above _kill_process_group.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **_popen_isolation_kwargs(),
    )
    # Captured NOW, while the leader is certainly alive -- see
    # _capture_process_group for why a later lookup silently kills nothing.
    group = _capture_process_group(proc)
    try:
        return _wait_captured(cmd, proc, group, timeout, deadline, check)
    finally:
        _release_process_group(group)


def _wait_captured(cmd, proc, group, timeout, deadline, check):
    """The poll loop of ``_run_captured``; split out so the group token is
    released on every exit path by the caller's ``finally``."""
    child_exited_at: Optional[float] = None
    while True:
        if deadline is None:
            poll_for = _LIVENESS_POLL_INTERVAL
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(proc, group)
                out, err = proc.communicate()
                raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)
            poll_for = min(_LIVENESS_POLL_INTERVAL, remaining)

        try:
            out, err = proc.communicate(timeout=poll_for)
        except subprocess.TimeoutExpired:
            pass
        else:
            # Normal path: pipes hit EOF and the child is reaped.
            if check and proc.returncode:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, output=out, stderr=err
                )
            return subprocess.CompletedProcess(cmd, proc.returncode, out, err)

        # Pipes are still open. Is the child even alive?
        if proc.poll() is None:
            child_exited_at = None
            continue

        now = time.monotonic()
        if child_exited_at is None:
            child_exited_at = now
            continue
        if now - child_exited_at < _ABANDONED_PIPE_GRACE:
            continue

        # Child is gone and the pipes STILL have not closed, so a descriptor
        # we do not own is holding the write end. Nothing further can ever
        # arrive; waiting out the remaining (possibly hours of) timeout would
        # buy exactly nothing.
        _log.error(
            "measurement subprocess %s exited (rc=%s) but its stdout/stderr "
            "stayed open for %ds afterwards -- a leaked descriptor in another "
            "process is holding the write end. Abandoning the read instead of "
            "waiting out the remaining %s of timeout. Command: %s",
            proc.pid, proc.returncode, _ABANDONED_PIPE_GRACE,
            "unbounded time" if deadline is None else f"{int(deadline - now)}s",
            " ".join(cmd),
        )
        _kill_process_group(proc, group)
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output=(
                f"measurement subprocess exited rc={proc.returncode} but its "
                f"pipes were held open by another process for "
                f"{_ABANDONED_PIPE_GRACE}s; output is unrecoverable"
            ),
            stderr=None,
        )


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


# ---------------------------------------------------------------------------
# Fail-fast arch support check (2026-08 multi-build-coexistence field fix)
#
# Multiple llama.cpp builds can coexist on one box (a stock-current build
# that knows a new arch; a pinned older build and a fork that don't), and
# LlamaCppTools' resolution inputs (hint / MAGICQUANT env / PATH) can vary
# across submission paths -- a measured search once auto-resolved to a
# build lacking the source model's arch and died at baseline 40+ minutes
# in with llama.cpp's own "unknown model architecture", burning the whole
# run's compute for zero measurements. The functions below let a caller
# catch that at t+0, before any subprocess runs, instead.
# ---------------------------------------------------------------------------


class LlamaBinaryArchError(RuntimeError):
    """The resolved llama.cpp binary's libllama does not contain the GGUF
    architecture literal the source model requires -- it provably cannot
    load this model. Raised by the fail-fast arch check
    (``binary_supports_arch`` + the wiring in
    ``MagicQuantOrchestrator.run_measured_search`` /
    ``magicquant.v2.search.run_budget_search``) BEFORE any measurement
    subprocess runs.
    """


# The probed binaries/libraries are 5-80MB; read in fixed-size chunks so a
# scan never has to hold a whole one in memory at once. *carry* (the tail of
# the previous chunk, sized len(literal)-1) is prepended to each new chunk
# so a literal that straddles a chunk boundary is still found.
_ARCH_SCAN_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB

# Directories (relative to the resolved binary's own directory; "" means
# that directory itself) and filename globs searched for a sibling
# libllama. Extended (Opus review) beyond "same dir as the binary" to also
# cover a standard install-prefix layout (<prefix>/bin/llama-perplexity +
# <prefix>/lib/libllama.so or .../lib64/...) and Windows shared-library
# names -- the original same-dir-only search returned a guaranteed-wrong
# False (not even None) for a real install-prefix layout, since it fell
# through to scanning the (dynamically-linked, arch-table-free) binary
# itself and found nothing.
_LIBLLAMA_SEARCH_DIRS = ("", "../lib", "../lib64")
_LIBLLAMA_NAME_GLOBS = ("libllama.so*", "libllama.dll", "llama.dll")

# The DT_NEEDED entry (e.g. "libllama.so", "libllama.so.1") that every
# dynamically-linked llama-perplexity binary observed in the field carries
# as plain bytes in its dynamic section -- present in all three real
# binaries checked during review, absent from a true static build (which
# instead carries the arch literals directly). Used ONLY to decide whether
# a negative binary-only scan is a real negative (static build) or
# uninformative (dynamic build whose real arch table lives in an
# unreachable libllama.so) -- see binary_supports_arch.
_DYNAMIC_LINK_MARKER = b"libllama"


def _scan_file_for_literal(path: Path, literal: bytes, chunk_size: int) -> bool:
    """True iff *literal* occurs anywhere in the file at *path*."""
    overlap = max(len(literal) - 1, 0)
    with open(path, "rb") as f:
        carry = b""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return False
            haystack = carry + chunk
            if literal in haystack:
                return True
            carry = haystack[-overlap:] if overlap else b""


def _library_candidates(tool_path: Path) -> List[Path]:
    """Every libllama-shaped file that might sit alongside *tool_path*'s
    llama-perplexity binary, across the layouts actually seen in the field:
    the binary's own directory (a common CMake build/bin/ layout), or a
    sibling ``../lib``/``../lib64`` (a standard install-prefix layout).
    Windows names are matched the same way. Existing files only, in a
    fixed (dir, name-glob) order for reproducibility -- every match found
    gets scanned (see binary_supports_arch's OR semantics), so this order
    affects only which match is scanned first, never the final verdict.
    """
    tool_dir = tool_path.parent
    found: List[Path] = []
    for rel_dir in _LIBLLAMA_SEARCH_DIRS:
        d = (tool_dir / rel_dir) if rel_dir else tool_dir
        if not d.is_dir():
            continue
        for name_glob in _LIBLLAMA_NAME_GLOBS:
            found.extend(sorted(d.glob(name_glob)))
    return found


def binary_supports_arch(
    perplexity_tool: Optional[str],
    arch: Optional[str],
    *,
    chunk_size: int = _ARCH_SCAN_CHUNK_BYTES,
) -> Optional[bool]:
    """Does *perplexity_tool*'s build support the GGUF architecture *arch*?

    Ground-truth probe: GGUF architecture names are string literals baked
    into libllama.so (or a statically-linked binary) -- llama.cpp
    dispatches model loading by looking up ``general.architecture`` against
    a compiled-in table, so a build that never registered an arch cannot
    contain its literal. This scans file BYTES for that literal WITHOUT
    loading any model and WITHOUT shelling out to ``strings`` (``strings
    <lib> | grep -c '<arch>'`` is the established equivalent; this
    reimplements it in pure Python for determinism/portability).

    Necessary, NOT sufficient, in TWO directions: a build that never
    registered an arch cannot contain its literal (this is what makes a
    real ``False`` trustworthy), but presence of the literal does not by
    itself guarantee full support. The likelier of the two ways presence
    can mislead is a substring false-positive: a build supporting only
    ``qwen3moe`` answers True for a scan of ``qwen3``, since the shorter
    name is a literal substring of the longer one -- a plain byte scan
    cannot tell "the arch table has this exact entry" from "the arch table
    has an entry that happens to contain these bytes". A stub/partial
    registration (the literal present but the dispatch incomplete) is the
    other, less likely, direction.

    Scans candidate sibling libraries (see ``_library_candidates`` -- same
    directory as the binary, or a sibling ``../lib``/``../lib64``, covering
    both a CMake build/bin/ layout and a standard install-prefix layout)
    AND the binary itself, OR-ing every scan together: a True from ANY
    candidate wins immediately. This is not just belt-and-suspenders --
    verified to also catch, for free, a stale/mismatched library sitting
    beside a STATIC binary that itself carries the current literal (the
    library-only scan would say False; the binary scan says True; the OR
    is correct). If nothing comes back True, a real ``False`` requires that
    at least one candidate was actually scanned successfully AND, when the
    only informative scan was of the binary itself, that the binary does
    NOT look dynamically linked against libllama (see
    ``_DYNAMIC_LINK_MARKER``) -- a dynamically-linked binary's own bytes
    never carry the arch table at all (it lives in the unreachable
    libllama.so), so a negative scan of ONLY the binary is uninformative,
    not a negative, and must return ``None`` rather than a guaranteed-wrong
    ``False``.

    Args:
        perplexity_tool: Path to the resolved llama-perplexity binary, or
            None/empty (returns None -- nothing resolvable to scan).
        arch: The GGUF ``general.architecture`` value to look for, or
            None/empty (returns None -- an empty needle would otherwise
            vacuously match, which must never look like "supported").
        chunk_size: Read chunk size in bytes (default 4 MiB). Exposed so
            tests can force a small chunk size to exercise the
            straddling-a-chunk-boundary case deterministically.

    Returns:
        True: some candidate (library or binary) contains the literal --
            the build supports this arch.
        False: at least one candidate was scanned and definitively
            answered the question (a library, or a binary that does not
            look dynamically linked) and none of them had the literal --
            the build provably does not support this arch.
        None: undeterminable -- nothing could be located/read at all, or
            the only informative-looking candidate was a dynamically-linked
            binary with no reachable library. Callers must treat this as
            "proceed, unverified", never as a negative result.
    """
    if not perplexity_tool:
        return None
    if not arch:
        _log.debug(
            "binary_supports_arch: empty/missing arch literal -- "
            "undeterminable (never a vacuous True)"
        )
        return None

    tool_path = Path(perplexity_tool)
    needle = arch.encode("utf-8")

    # 1. Every candidate sibling library. A True from ANY of them settles
    #    the question immediately.
    library_scanned = False
    for lib_path in _library_candidates(tool_path):
        try:
            found = _scan_file_for_literal(lib_path, needle, chunk_size)
        except OSError:
            continue
        library_scanned = True
        if found:
            return True

    # 2. Also scan the binary itself (OR, not "only if no library was
    #    found") -- catches a static build (the binary IS the arch table)
    #    and the stale-library-beside-a-static-binary case described above.
    binary_negative = False
    if tool_path.is_file():
        try:
            if _scan_file_for_literal(tool_path, needle, chunk_size):
                return True
            binary_negative = True
        except OSError:
            pass

    # 3. Nothing came back True -- decide a real False vs undeterminable.
    if library_scanned:
        # A definitive negative straight from the build's real arch table
        # (a library) stands regardless of what the separate binary scan
        # said (which may be uninformative if dynamically linked).
        # Assumes a discovered candidate library is the one the binary
        # actually loads; a stale decoy in ../lib (binary resolving its
        # real lib via RPATH/LD_LIBRARY_PATH) can still produce a wrong
        # False -- the MAGICQUANT_SKIP_ARCH_CHECK escape hatch covers
        # that residual.
        return False
    if binary_negative:
        # No library was found/scanned at all -- the binary's own "not
        # found" is only a real negative if the binary IS the arch table,
        # i.e. it does NOT look dynamically linked against libllama.
        try:
            dynamically_linked = _scan_file_for_literal(
                tool_path, _DYNAMIC_LINK_MARKER, chunk_size
            )
        except OSError:
            dynamically_linked = False
        if dynamically_linked:
            _log.debug(
                "binary_supports_arch: %s looks dynamically linked against "
                "libllama and no sibling library was found/readable -- "
                "undeterminable, not a negative", tool_path,
            )
            return None
        return False
    return None


def resolve_source_gguf_arch(source_model_path: str) -> Optional[str]:
    """Best-effort, header-only read of *source_model_path*'s GGUF
    ``general.architecture`` metadata, for the fail-fast arch check.

    Returns None -- meaning "the caller should skip the check entirely" --
    whenever the path isn't a readable GGUF with an architecture key: a
    safetensors source (file or directory), a directory, a corrupt/stub
    file (as several unit-test fixtures deliberately write), or any other
    parse failure. Uses ``magicquant.gguf.reader.GGUFReader``, which stops
    after the metadata + tensor-info header and never touches tensor data
    -- milliseconds even for a multi-GB file.
    """
    from magicquant.gguf.reader import GGUFReader

    try:
        reader = GGUFReader(source_model_path)
        reader.open()
        arch = reader.get_model_architecture()
    except Exception as exc:
        _log.debug(
            "arch pre-check: %s is not a readable GGUF (%s) -- skipping",
            source_model_path, exc,
        )
        return None
    if arch == "unknown":
        _log.debug(
            "arch pre-check: %s has no general.architecture metadata -- "
            "skipping", source_model_path,
        )
        return None
    return arch


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

    def _measure_timeout(self, model_path: str, kl: bool = False) -> int:
        """Size-aware subprocess timeout for one measurement pass over
        *model_path*.

        ``max(_SUBPROCESS_TIMEOUT, file_size_bytes // _MIN_MEASURE_BANDWIDTH)``
        -- never lower than the flat floor (small/typical models keep
        exactly the historical cap), but scaled up for a large artifact
        whose legitimate bandwidth-bound pass would otherwise exceed it
        (37.8GB -> ~2.6h for the plain-PPL leg at the 4 MB/s floor). *kl*
        doubles the result: the KL/logits legs (``save_base_logits``,
        ``calculate_kl_divergence``) are empirically slower than plain PPL
        over the same file -- same forward compute plus a saved-logits
        read plus full-vocab KL accumulation.

        ``os.path.getsize`` is read defensively: a missing/inaccessible
        file (already-deleted candidate, permissions, race -- ``OSError``)
        OR a bad argument (e.g. ``model_path=None`` -- ``TypeError``, which
        ``os.path.getsize`` raises rather than an ``OSError`` subclass)
        both degrade to the base timeout rather than raising -- this
        helper's job is to WIDEN a timeout for a known-large file, not to
        gate on the file's existence or the caller's argument being
        well-formed (the subprocess call itself fails loudly if the model
        path is actually bad). Always returns a finite int: this box has
        an OOM-livelock history, so hang protection must never be
        disabled, only sized correctly.
        """
        try:
            size_bytes = os.path.getsize(model_path)
        except (OSError, TypeError):
            size_bytes = 0
        ppl_timeout = max(_SUBPROCESS_TIMEOUT, size_bytes // _MIN_MEASURE_BANDWIDTH)
        return ppl_timeout * 2 if kl else ppl_timeout

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
                _log.debug(
                    "_find_llamacpp: resolved to %s (first common install "
                    "path that exists, of %s)", p, common_paths,
                )
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
            resolved = str(Path(result.stdout.strip()).parent)
            _log.debug(
                "_find_llamacpp: resolved to %s (none of the common install "
                "paths existed; fell back to `%s llama-quantize` on PATH)",
                resolved, which_cmd,
            )
            return resolved
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

        Goes through ``_run_captured`` rather than ``subprocess.run`` so a
        child that exits while something else still holds its pipe open
        cannot burn the whole (now size-scaled, multi-hour) timeout -- see
        that function.
        """
        return _run_captured(cmd, timeout=timeout, check=True)

    def verify_model_loads(
        self,
        model_path: str,
        *,
        data_file: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Cheaply answer "does llama.cpp accept this GGUF at all?".

        Returns ``(ok, detail)``. ``detail`` is the tail of the tool's output
        when it fails, and "" when it loads.

        This is a LOAD check, not a quality gate -- it deliberately does not
        look at the perplexity value. The point is that the errors worth
        catching early are all raised while the file is being opened::

            done_getting_tensors: wrong number of tensors; expected 417, got 408
            unknown model architecture: 'muse-glimmer'
            key split.count has wrong type u32 but expected type u16

        A full perplexity pass catches those too, but only after minutes to
        tens of minutes, and typically an hour downstream of the write that
        caused them -- by which point the diagnosis is no longer obvious.
        Running this immediately after a GGUF is written fails at the point
        where the cause is unambiguous. Keep the existing PPL smoke test as
        the separate quality gate it already is; this does not replace it.

        Implementation notes that matter:

        * ``llama-perplexity ... -c 512 --chunks 1``, NOT ``llama-cli``.
          ``llama-cli`` enters its interactive loop even with stdin at
          /dev/null -- it once spun and wrote a 16 GB log before anyone
          noticed. ``-no-cnv`` is not a fix; it still waits on stdin. The
          perplexity tool loads, runs exactly one chunk, and exits with a
          non-zero status on any load failure.
        * ``--chunks 1`` keeps the compute at one 512-token window, so the
          runtime is dominated by reading the file.
        * goes through ``_run_captured`` so a child that dies while something
          else holds its pipe cannot hang this either.
        """
        resolved = self._resolve_data_file(data_file)
        if resolved is None:
            return False, "no calibration/eval corpus available for a load check"

        if timeout is None:
            try:
                size_bytes = os.path.getsize(model_path)
            except (OSError, TypeError):
                size_bytes = 0
            timeout = max(_LOAD_CHECK_TIMEOUT, size_bytes // _LOAD_CHECK_BANDWIDTH)

        cmd = [
            self.perplexity_tool,
            "-m", model_path,
            "-f", resolved,
            "--ctx-size", "512",
            "--chunks", "1",
        ] + self._gpu_flags()

        try:
            proc = _run_captured(cmd, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return False, f"load check timed out after {timeout}s"
        # OSError/FileNotFoundError deliberately NOT caught: a missing or
        # wrong-arch binary is a caller problem, not a bad model, and every
        # other site in this file lets it propagate for that reason.

        if proc.returncode == 0:
            return True, ""
        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        return False, _tail(combined, 15)

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

        Also records the failure kind on the instance
        (``self._last_subprocess_failure = {"kind": "timeout"|"error",
        "label": label}``), cleared (``None``) on every call before the
        subprocess even runs -- so a caller can distinguish "this specific
        call timed out" from "this specific call failed for another
        reason" (e.g. unparseable output) without either kind being
        confused with a stale reading left over from a PRIOR call. An
        OSError that propagates out of this method (see above) leaves the
        attribute cleared too, since it was reset before the propagating
        call. Callers that need this must read it via
        ``getattr(tools, "_last_subprocess_failure", None)`` -- several
        tests construct a bare instance via ``LlamaCppTools.__new__`` that
        never runs ``__init__`` and so never gets this attribute set at
        all until the first call.
        """
        self._last_subprocess_failure = None
        try:
            result = self._run_perplexity_subprocess(cmd, timeout=timeout)
        except subprocess.CalledProcessError as e:
            print(f"{label} failed: {e.stderr}")
            self._last_subprocess_failure = {"kind": "error", "label": label}
            return None
        except subprocess.TimeoutExpired:
            print(f"{label} timed out")
            self._last_subprocess_failure = {"kind": "timeout", "label": label}
            return None
        return result

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
        # Cleared HERE, before the early return below, not just inside
        # _run_subprocess_or_none: the corpus-resolution early return can
        # fire before that helper (and therefore this flag) is ever
        # touched, which used to leave a PREVIOUS call's stale
        # "timeout"/"error" reading in place -- the orchestrator would
        # then misread an unrelated later failure (e.g. a vanished
        # corpus) as a timeout too (Q2 field review, 2026-08).
        self._last_subprocess_failure = None
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

        result = self._run_subprocess_or_none(
            cmd, self._measure_timeout(model_path), "Perplexity calculation"
        )
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
        timeout: Optional[int] = None,
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
            timeout: Subprocess timeout in seconds. *None* (default) uses
                the size-aware KL-leg timeout for *base_model_path* (see
                ``_measure_timeout``, ``kl=True`` -- this pass saves logits,
                grouped with the KL/logits legs, not the plain-PPL leg).

        Returns:
            The parsed "Final estimate: PPL" value from this pass on success,
            or *None* if the subprocess failed, or the output logits file is
            missing/stub-sized, or no PPL line could be parsed. The stub-file
            guard always wins: a stub-sized output file means failure/None
            even if a PPL line happened to parse from stdout/stderr.
        """
        # Cleared at the top defensively (Q2 field review) -- this method
        # currently always reaches _run_subprocess_or_none (which clears
        # it itself too), so there is no known early-return leak here
        # today, but the same class of bug as calculate_perplexity's
        # would be trivial to reintroduce with a future early return
        # added above the subprocess call.
        self._last_subprocess_failure = None
        cmd = [
            self.perplexity_tool,
            "-m", base_model_path,
            "-f", corpus_path,
            "--kl-divergence-base", out_logits_path,
            "--ctx-size", str(ctx_size),
            "--chunks", self._effective_chunks(chunks),
        ] + self._perplexity_batch_flags() + self._gpu_flags()

        effective_timeout = (
            timeout if timeout is not None
            else self._measure_timeout(base_model_path, kl=True)
        )
        result = self._run_subprocess_or_none(cmd, effective_timeout, "Saving base logits")
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
        timeout: Optional[int] = None,
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
            timeout: Subprocess timeout in seconds. *None* (default) uses
                the size-aware KL-leg timeout for *quant_model_path* (see
                ``_measure_timeout``, ``kl=True``).

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
        # Cleared at the top defensively -- see save_base_logits' identical
        # note (Q2 field review): no known early-return leak here today,
        # but cheap insurance against one being added later above the
        # subprocess call.
        self._last_subprocess_failure = None
        cmd = [
            self.perplexity_tool,
            "-m", quant_model_path,
            "-f", corpus_path,
            "--kl-divergence",
            "--kl-divergence-base", base_logits_path,
            "--ctx-size", str(ctx_size),
            "--chunks", self._effective_chunks(chunks),
        ] + self._perplexity_batch_flags() + self._gpu_flags()

        effective_timeout = (
            timeout if timeout is not None
            else self._measure_timeout(quant_model_path, kl=True)
        )
        result = self._run_subprocess_or_none(cmd, effective_timeout, "KL divergence calculation")
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
