"""A measurement subprocess that exits must not be able to hang the parent.

Field report, 2026-08-13, a ``--magicquant-budget-gib`` run: the child
``llama-perplexity`` was in state ``Z`` -- exited, never reaped -- while the
parent sat in ``poll_schedule_timeout`` holding the two read ends and six
other processes still held the write ends. 70 minutes of zero output at 0%
CPU on the measurement.

``subprocess.run`` waits for the pipes to reach EOF, not for the child to
exit. EOF only arrives when the LAST writer closes the descriptor, so a
grandchild that outlived the child pins the parent for the entire timeout.
That timeout is now size-scaled (``_measure_timeout``), so on a 62 GB probe
it is ~4 hours (~8 for a KL leg) rather than the old flat 2.

The scenario is reproduced here exactly, and cheaply: a Python child spawns
a Python grandchild that inherits its stdout/stderr and outlives it -- the
same shape as the production failure. It used to be ``sh -c '(sleep N) &
...'``, which was POSIX-only; the Python form runs everywhere, so the
Windows Job Object cleanup path is exercised by the same tests as the POSIX
killpg path.

Note this is NOT a v2-only bug. v1 (``evolution/probing.py``,
``orchestrator.py``) and v2 (``v2/search.py``, ``v2/calibrate.py``) all reach
the one ``LlamaCppTools.calculate_perplexity`` and the same spawn path, so
v1 was lucky rather than safe.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from magicquant.utils import llamacpp
from magicquant.utils.llamacpp import _run_captured


# ── portable child processes ────────────────────────────────────────────────
#
# Every child is ``sys.executable -c <script>`` so the tests run identically
# on POSIX and Windows. The grandchild is handed the child's stdout/stderr
# EXPLICITLY (stdout=sys.stdout): that is what pins the pipe write ends in the
# grandchild on both platforms -- relying on implicit handle inheritance is
# not portable (Windows only inherits std handles in some configurations).

def _py(script: str) -> list:
    return [sys.executable, "-c", script]


def _grandchild_holds_pipe(pidfile=None) -> list:
    """Child exits at once; its grandchild sleeps 30s holding the pipes.
    Optionally records the grandchild's PID (not its own -- see the killed
    test) to *pidfile*."""
    record = (
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n" if pidfile else ""
    )
    return _py(
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],"
        " stdout=sys.stdout, stderr=sys.stderr)\n"
        + record +
        "print('working'); sys.stdout.flush(); sys.exit(0)\n"
    )


def _pid_alive(pid: int) -> bool:
    """Liveness probe that is safe on Windows. NOT ``os.kill(pid, 0)`` there:
    Windows' os.kill maps any non-CTRL signal to TerminateProcess, so the
    POSIX "signal 0 = existence check" idiom would kill the process."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    SYNCHRONIZE, PROCESS_QUERY_LIMITED_INFORMATION = 0x00100000, 0x1000
    WAIT_TIMEOUT = 0x102
    h = k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        return k32.WaitForSingleObject(h, 0) == WAIT_TIMEOUT
    finally:
        k32.CloseHandle(h)


def _force_kill(pid: int) -> None:
    sig = signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, OSError):
        pass


@pytest.fixture
def fast_stall_detection(monkeypatch):
    """Shrink the liveness poll and the post-exit grace so the stall path
    resolves in about a second instead of a minute."""
    monkeypatch.setattr(llamacpp, "_LIVENESS_POLL_INTERVAL", 0.1)
    monkeypatch.setattr(llamacpp, "_ABANDONED_PIPE_GRACE", 0.5)


# ── the regression ──────────────────────────────────────────────────────────

def test_child_exit_with_pipe_held_open_does_not_wait_out_the_timeout(
    fast_stall_detection,
):
    """The exact production shape: child exits immediately, a grandchild
    keeps the write end open. Must give up in about the grace period, not in
    `timeout` seconds."""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_captured(
            _grandchild_holds_pipe(),
            timeout=30,          # stands in for the multi-hour real timeout
        )
    elapsed = time.monotonic() - started
    assert elapsed < 5, (
        f"took {elapsed:.1f}s -- it waited on the abandoned pipe instead of "
        "noticing the child had already exited"
    )


def test_the_lingering_grandchild_is_killed(fast_stall_detection, tmp_path):
    """Bounding the wait is not enough on its own -- the process still
    holding the descriptor has to go too, or it leaks for its own lifetime.

    This asserts on the grandchild's PID, not on a side effect it would
    produce later. An earlier version of this test waited 1.5s for a marker
    file the grandchild only writes at +30s, so a surviving grandchild passed
    the assertion and then failed the NEXT run by leaving the marker behind.
    It did catch a real bug that way -- ``_kill_process_group`` was looking up
    the pgid after the leader had been reaped, so it never killed anything --
    but only by accident, a run late, and in the wrong test.
    """
    pidfile = tmp_path / "grandchild.pid"

    # The GRANDCHILD's PID (p.pid in the child script), NOT the child's own.
    # The sh-era version of this test once recorded `$$` -- the child's PID,
    # already dead and reaped by the time we look -- and passed no matter
    # what the cleanup did.
    with pytest.raises(subprocess.TimeoutExpired):
        _run_captured(_grandchild_holds_pipe(pidfile), timeout=30)

    assert pidfile.exists(), "grandchild never started -- test is not exercising anything"
    grandchild = int(pidfile.read_text().strip())
    assert grandchild > 0

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not _pid_alive(grandchild):
            return  # gone, as required
        time.sleep(0.05)

    # Still alive: clean up so the leak does not outlive the test run.
    _force_kill(grandchild)
    pytest.fail(
        f"grandchild {grandchild} survived the stall cleanup -- it still holds "
        "the pipe write end, which is the whole failure being fixed"
    )


def test_stall_error_explains_itself():
    """An operator reading only the exception must be able to tell an
    abandoned-pipe stall from an ordinary slow measurement."""
    import magicquant.utils.llamacpp as m

    old_poll, old_grace = m._LIVENESS_POLL_INTERVAL, m._ABANDONED_PIPE_GRACE
    m._LIVENESS_POLL_INTERVAL, m._ABANDONED_PIPE_GRACE = 0.1, 0.3
    try:
        with pytest.raises(subprocess.TimeoutExpired) as exc:
            _run_captured(_grandchild_holds_pipe(), timeout=30)
    finally:
        m._LIVENESS_POLL_INTERVAL, m._ABANDONED_PIPE_GRACE = old_poll, old_grace

    assert "held open by another process" in str(exc.value.output)


# ── the contract callers depend on is unchanged ─────────────────────────────

def test_normal_success_returns_completed_process():
    result = _run_captured(
        _py("import sys; print('out'); print('err', file=sys.stderr)"), timeout=30
    )
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
    assert "out" in result.stdout
    assert "err" in result.stderr


def test_check_raises_called_process_error_with_output_attached():
    """`_run_subprocess_or_none` prints `<label> failed: <stderr>`, so stderr
    has to survive onto the exception."""
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_captured(
            _py("import sys; print('boom', file=sys.stderr); sys.exit(3)"),
            timeout=30, check=True,
        )
    assert exc.value.returncode == 3
    assert "boom" in exc.value.stderr


def test_check_false_returns_nonzero_instead_of_raising():
    result = _run_captured(_py("import sys; sys.exit(7)"), timeout=30, check=False)
    assert result.returncode == 7


def test_genuine_timeout_still_raises_and_kills_the_child():
    """A child that is genuinely still running must hit the ordinary
    timeout -- the liveness check must not shorten a healthy slow run."""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_captured(_py("import time; time.sleep(30)"), timeout=1)
    assert time.monotonic() - started < 5


def test_oserror_from_the_spawn_still_propagates():
    """tests/test_orchestrator_measurement.py depends on OSError escaping the
    subprocess triage rather than being converted -- never widen that."""
    with pytest.raises((OSError, FileNotFoundError)):
        _run_captured(["/nonexistent/mq/binary"], timeout=30)


@pytest.mark.skipif(not llamacpp._HAS_POSIX_PROCESS_GROUPS,
                    reason="POSIX sessions/process groups only")
def test_child_runs_in_its_own_session():
    """killpg is only safe because the child leads its own group. If this
    regresses, the stall cleanup would signal MagicQuant's own group."""
    proc = subprocess.Popen(
        _py("import time; time.sleep(5)"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **llamacpp._popen_isolation_kwargs(),
    )
    try:
        assert os.getpgid(proc.pid) == proc.pid
        assert os.getpgid(proc.pid) != os.getpgid(0)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def test_process_group_token_is_captured_and_kills_the_child():
    """Portable counterpart: on every supported platform the spawn must yield
    a real group token (pgid on POSIX, Job Object on Windows) -- ``None``
    means the cleanup silently degraded to killing the child alone, which is
    exactly the grandchild leak this module exists to prevent."""
    proc = subprocess.Popen(
        _py("import time; time.sleep(30)"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **llamacpp._popen_isolation_kwargs(),
    )
    token = llamacpp._capture_process_group(proc)
    try:
        assert token is not None, "no process-group isolation on this platform"
        llamacpp._kill_process_group(proc, token)
        proc.wait(timeout=5)
        assert proc.returncode is not None
    finally:
        llamacpp._release_process_group(token)
        proc.communicate()


# ── the bounded load check ──────────────────────────────────────────────────

class _FakeTools:
    """Just enough LlamaCppTools to exercise verify_model_loads."""

    def __init__(self, perplexity_tool, corpus="/fake/corpus.txt"):
        self.perplexity_tool = perplexity_tool
        self._corpus = corpus

    def _resolve_data_file(self, data_file=None):
        return self._corpus

    def _gpu_flags(self):
        return []


def _verify(tool_cmd, **kw):
    from magicquant.utils.llamacpp import LlamaCppTools
    tools = _FakeTools(tool_cmd)
    return LlamaCppTools.verify_model_loads(tools, "/fake/model.gguf", **kw)


def test_load_check_passes_on_a_zero_exit(monkeypatch):
    import magicquant.utils.llamacpp as m
    monkeypatch.setattr(m, "_run_captured",
                        lambda cmd, timeout, check=False:
                        subprocess.CompletedProcess(cmd, 0, "ok", ""))
    ok, detail = _verify("llama-perplexity")
    assert ok is True and detail == ""


def test_load_check_reports_the_real_llama_error(monkeypatch):
    """The whole value is that the operator sees the load error itself, not a
    downstream symptom an hour later."""
    import magicquant.utils.llamacpp as m
    err = ("llama_model_load: error loading model: done_getting_tensors: "
           "wrong number of tensors; expected 417, got 408")
    monkeypatch.setattr(m, "_run_captured",
                        lambda cmd, timeout, check=False:
                        subprocess.CompletedProcess(cmd, 1, "", err))
    ok, detail = _verify("llama-perplexity")
    assert ok is False
    assert "wrong number of tensors" in detail


def test_load_check_uses_one_chunk_and_not_llama_cli(monkeypatch):
    """llama-cli enters its interactive loop even with stdin at /dev/null --
    it once spun and wrote a 16 GB log. --chunks 1 is what keeps this bounded.
    """
    import magicquant.utils.llamacpp as m
    seen = {}

    def spy(cmd, timeout, check=False):
        seen["cmd"] = cmd
        seen["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(m, "_run_captured", spy)
    _verify("/opt/llama-perplexity")

    assert "llama-cli" not in " ".join(seen["cmd"])
    assert seen["cmd"][0] == "/opt/llama-perplexity"
    assert "--chunks" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--chunks") + 1] == "1"
    assert seen["cmd"][seen["cmd"].index("--ctx-size") + 1] == "512"
    # Bounded, and far below a measurement timeout.
    assert 0 < seen["timeout"] <= 600


def test_load_check_turns_a_timeout_into_a_failure_not_an_exception(monkeypatch):
    import magicquant.utils.llamacpp as m

    def boom(cmd, timeout, check=False):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(m, "_run_captured", boom)
    ok, detail = _verify("llama-perplexity")
    assert ok is False and "timed out" in detail


def test_load_check_lets_a_missing_binary_propagate(monkeypatch):
    """A missing/wrong-arch binary is a caller problem, not a bad model --
    same contract every other spawn site in this module keeps."""
    import magicquant.utils.llamacpp as m

    def boom(cmd, timeout, check=False):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(m, "_run_captured", boom)
    with pytest.raises((OSError, FileNotFoundError)):
        _verify("/nonexistent/llama-perplexity")


def test_load_check_fails_cleanly_with_no_corpus():
    from magicquant.utils.llamacpp import LlamaCppTools

    class _NoCorpus(_FakeTools):
        def _resolve_data_file(self, data_file=None):
            return None

    ok, detail = LlamaCppTools.verify_model_loads(
        _NoCorpus("llama-perplexity"), "/fake/model.gguf"
    )
    assert ok is False and "corpus" in detail
