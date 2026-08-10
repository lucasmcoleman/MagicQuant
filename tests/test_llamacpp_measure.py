"""Tests for LlamaCppTools.bench / save_base_logits / calculate_kl_divergence.

The unit tests below feed the REAL llama-bench JSON and llama-perplexity
KL-divergence stdout/stderr formats (captured by actually running the
ROCmFPX build against the tiny Qwen2.5-0.5B-Instruct q8_0 GGUF) through the
pure parser helpers, so the regexes/JSON handling are pinned to reality
rather than a guessed format.

Empirically confirmed invocation semantics (see docstrings on the
parsers/methods in magicquant/utils/llamacpp.py for the full detail):

- ``llama-bench -m <model> -p 8 -n 8 -r 1 -o json`` emits a JSON array with
  one row per test: the prompt-processing row has ``n_gen == 0`` (its
  ``avg_ts`` is pp t/s); the generation row has ``n_prompt == 0`` (its
  ``avg_ts`` is tg t/s).
- ``llama-perplexity ... --kl-divergence-base FNAME`` (no ``--kl-divergence``
  flag) SAVES per-token logits to FNAME from a model run.
  ``llama-perplexity -m <quant> -f <corpus> --kl-divergence
  --kl-divergence-base FNAME`` then COMPUTES KL divergence of <quant> against
  those saved logits, printing a "====== KL divergence statistics ======"
  block to stdout with a "Mean    KLD:  <value> ±   <stderr>" line
  (plus "Maximum KLD:", percentile lines like "90.0%   KLD:", "Median  KLD:",
  "Minimum KLD:" -- no "±" on those).
- Self-consistency check: saving base logits FROM the q8_0 model and then
  computing KL of that SAME q8_0 model against its own saved logits gives a
  deterministic ``Mean KLD`` of -0.000019 (reproduced twice, bit-identical)
  -- i.e. very close to the mathematically-expected 0, but not exactly 0 and
  even slightly negative. This is a precision artifact of llama.cpp's saved
  logits format (not run-to-run noise), so callers/tests should assert
  ``abs(mean_kl)`` is small rather than ``mean_kl >= 0``.
"""
import json
import os
from pathlib import Path
from unittest import mock

import pytest

import magicquant.utils.llamacpp as llamacpp_mod
from magicquant.utils.llamacpp import (
    LlamaCppTools,
    _env_int,
    _parse_bench_json,
    _parse_kl_output,
)


# --- Real captured formats -------------------------------------------------

# Trimmed from an actual `llama-bench -m qwen2.5-0.5b-instruct-q8_0.gguf -p 8
# -n 8 -r 1 -o json` run on the ROCmFPX build (build-strix-rocmfp4).
_BENCH_JSON = """[
  {
    "build_commit": "221402a",
    "model_type": "qwen2 1B Q8_0",
    "n_batch": 2048,
    "n_prompt": 8,
    "n_gen": 0,
    "avg_ns": 7187110,
    "avg_ts": 1113.103876,
    "samples_ts": [ 1113.1 ]
  },
  {
    "build_commit": "221402a",
    "model_type": "qwen2 1B Q8_0",
    "n_batch": 2048,
    "n_prompt": 0,
    "n_gen": 8,
    "avg_ns": 46294656,
    "avg_ts": 172.806123,
    "samples_ts": [ 172.806 ]
  }
]"""

# Real ROCm banner lines that land on stdout/stderr around the JSON on some
# builds -- the parser must isolate the JSON array from this noise.
_BENCH_JSON_WITH_BANNER_NOISE = (
    "ggml_rocm_init: found 1 ROCm devices (Total VRAM: 114688 MiB):\n"
    + _BENCH_JSON
    + "\n"
)

# Captured verbatim from `llama-perplexity -m qwen2.5-0.5b-instruct-q8_0.gguf
# -f corpus.txt --kl-divergence --kl-divergence-base base_logits.kld
# --ctx-size 128 --chunks 2` (self-consistency: q8_0 vs its own saved
# logits).
_KL_STDOUT = """
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1      14.7258 ±    4.9090      -0.00002 ±       -nan      -0.00002 ±    0.00000     0.000 ±  0.000 %    100.000 ±  0.000 %
   2      13.8216 ±    3.0463      -0.00002 ±       -nan      -0.00002 ±    0.00000     0.000 ±  0.000 %    100.000 ±  0.000 %

====== Perplexity statistics ======
Mean PPL(Q)                   :  13.821636 ±   3.046334
Mean PPL(base)                :  13.821891 ±   3.046407
Cor(ln(PPL(Q)), ln(PPL(base))): 100.00%
Mean ln(PPL(Q)/PPL(base))     :  -0.000018 ±       -nan
Mean PPL(Q)/PPL(base)         :   0.999982 ±       -nan
Mean PPL(Q)-PPL(base)         :  -0.000255 ±       -nan

====== KL divergence statistics ======
Mean    KLD:  -0.000019 ±   0.000001
Maximum KLD:   0.000001
99.9%   KLD:   0.000001
99.0%   KLD:   0.000000
95.0%   KLD:  -0.000001
90.0%   KLD:  -0.000005
Median  KLD:  -0.000018
10.0%   KLD:  -0.000033
 5.0%   KLD:  -0.000036
 1.0%   KLD:  -0.000041
 0.1%   KLD:  -0.000043
Minimum KLD:  -0.000043
"""

_KL_STDERR_NOISE = (
    "0.00.782.862 I kl_divergence: computing over 2 chunks, n_ctx=128, batch_size=2048, n_seq=16\n"
    "0.00.958.406 I kl_divergence: 0.18 seconds per pass - ETA\n"
)


# --- Pure-parser unit tests (no binary needed) ------------------------------


def test_parse_bench_json_extracts_pp_and_tg_ts():
    parsed = _parse_bench_json(_BENCH_JSON)
    assert parsed["pp_ts"] == pytest.approx(1113.103876) and parsed["tg_ts"] == pytest.approx(172.806123)


def test_parse_bench_json_ignores_surrounding_banner_noise():
    parsed = _parse_bench_json(_BENCH_JSON_WITH_BANNER_NOISE)
    assert parsed["pp_ts"] == pytest.approx(1113.103876) and parsed["tg_ts"] == pytest.approx(172.806123)


def test_parse_bench_json_returns_none_on_garbage():
    assert _parse_bench_json("no json here") is None


def test_parse_bench_json_returns_none_when_rows_missing():
    # Only a pp row, no tg row -> incomplete, should be None not a partial dict.
    only_pp = json.dumps([{"n_prompt": 8, "n_gen": 0, "avg_ts": 1113.1}])
    assert _parse_bench_json(only_pp) is None


def test_parse_kl_output_extracts_mean_max_p90():
    parsed = _parse_kl_output(_KL_STDOUT + _KL_STDERR_NOISE)
    assert parsed is not None
    assert parsed["mean_kl"] == pytest.approx(-0.000019)
    assert parsed["max_kl"] == pytest.approx(0.000001)
    assert parsed["p90_kl"] == pytest.approx(-0.000005)


def test_parse_kl_output_extracts_ppl_from_perplexity_statistics_block():
    """A --kl-divergence run also prints a 'Perplexity statistics' block with
    the evaluated model's own 'Mean PPL(Q)' -- extracting it here lets the
    orchestrator fuse a candidate's PPL + KL measurement into ONE
    llama-perplexity invocation (see run_measured_search)."""
    parsed = _parse_kl_output(_KL_STDOUT + _KL_STDERR_NOISE)
    assert parsed is not None
    assert parsed["ppl"] == pytest.approx(13.821636)
    assert parsed["ppl_err"] == pytest.approx(3.046334)


def test_parse_kl_output_returns_none_when_absent():
    assert _parse_kl_output("nothing relevant here") is None


def test_parse_kl_output_omits_ppl_when_perplexity_block_absent():
    """A KL block without the preceding Perplexity-statistics block (e.g. an
    unexpected/older output format) must still parse the KL fields -- just
    without "ppl"/"ppl_err" -- rather than returning None outright."""
    kl_only = (
        "====== KL divergence statistics ======\n"
        "Mean    KLD:  -0.000019 \xb1   0.000001\n"
    )
    parsed = _parse_kl_output(kl_only)
    assert parsed is not None
    assert parsed["mean_kl"] == pytest.approx(-0.000019)
    assert "ppl" not in parsed


# --- LlamaCppTools method tests (mocked subprocess) -------------------------


def _bare_tools(**attrs) -> LlamaCppTools:
    """Construct a LlamaCppTools bypassing __init__ (no binary discovery)."""
    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.perplexity_tool = "/bin/true"
    tools.bench_tool = "/bin/true"
    tools.ctx_size = 512
    tools.ngl = None
    tools.threads = None
    for k, v in attrs.items():
        setattr(tools, k, v)
    return tools


def test_bench_returns_none_when_bench_tool_missing():
    tools = _bare_tools(bench_tool=None)
    assert tools.bench(model_path="/does/not/matter.gguf") is None


def test_bench_parses_result_from_mocked_subprocess():
    import subprocess

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_BENCH_JSON, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        result = tools.bench("/some/model.gguf", n_prompt=8, n_gen=8, reps=1)
    assert result["pp_ts"] == pytest.approx(1113.103876) and result["tg_ts"] == pytest.approx(172.806123)
    # Sanity: bench invoked llama-bench with -o json and the given -p/-n/-r.
    cmd = run.call_args[0][0]
    assert cmd[0] == "/bin/true"
    assert "-o" in cmd and "json" in cmd
    assert "8" in cmd  # -p 8 / -n 8


def test_save_base_logits_returns_ppl_when_file_written(tmp_path):
    """save_base_logits now returns the pass's own parsed PPL (not a bare
    bool) on success -- this pass, even without --kl-divergence, still
    prints the normal 'Final estimate: PPL' line for the base model itself,
    which run_measured_search fuses in as the baseline measurement."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="Final estimate: PPL = 14.9619 +/- 0.85130"
    )

    def _fake_run(cmd, timeout):
        # Simulate llama-perplexity actually writing a real (not stub-sized)
        # logits file -- save_base_logits rejects anything <= a few KB as a
        # failed-but-exit-0 stub (see _MIN_LOGITS_FILE_BYTES).
        out_logits.write_bytes(b"fake-logits" * 1000)
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        ppl = tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert ppl == pytest.approx(14.9619)
    assert out_logits.is_file()


def test_save_base_logits_none_when_file_is_stub_sized(tmp_path):
    """A too-short corpus makes llama-perplexity exit 0 but write only a
    ~12-byte header stub -- save_base_logits must treat that as failure
    (None), not just check the file exists (regression: this previously
    returned True, then calculate_kl_divergence silently failed to find
    'Mean KLD'). The stub-file guard wins even if a PPL line parsed."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="Final estimate: PPL = 14.9619 +/- 0.85130"
    )

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"_logits_\x80\x00\x00\x00")  # the real 12-byte stub
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        ppl = tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert ppl is None


def test_save_base_logits_none_when_file_missing(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"  # never created

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake):
        ppl = tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert ppl is None


def test_save_base_logits_none_when_ppl_unparseable(tmp_path):
    """A valid (non-stub) logits file but no parseable 'Final estimate: PPL'
    line -- unrealistic for a real llama-perplexity run, but the return
    contract is a plain Optional[float], so this degrades to None rather
    than raising."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="no ppl line here", stderr="")

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"fake-logits" * 1000)
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        ppl = tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert ppl is None


def test_calculate_kl_divergence_parses_mocked_subprocess(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    base_logits = tmp_path / "base.kld"
    base_logits.write_bytes(b"fake-logits")

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_KL_STDOUT, stderr=_KL_STDERR_NOISE)
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        result = tools.calculate_kl_divergence("/quant/model.gguf", str(base_logits), str(corpus))
    assert result["mean_kl"] == pytest.approx(-0.000019)
    assert result["max_kl"] == pytest.approx(0.000001)
    cmd = run.call_args[0][0]
    assert "--kl-divergence" in cmd
    assert "--kl-divergence-base" in cmd


def test_calculate_kl_divergence_none_on_unparseable_output(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    base_logits = tmp_path / "base.kld"
    base_logits.write_bytes(b"fake-logits")

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="nothing useful", stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake):
        result = tools.calculate_kl_divergence("/quant/model.gguf", str(base_logits), str(corpus))
    assert result is None


# --- GPU offload (-ngl/-t) flag threading -----------------------------------
#
# Real GPU behavior was verified manually (not by these unit tests, which
# mock the subprocess): the plain /home/lucas/llama.cpp/build/bin build has
# no ROCm/Vulkan backend compiled in at all (only libggml-cpu.so present) --
# passing -ngl there is a harmless no-op, memory breakdown stays 100% Host.
# The ~/ROCmFPX/build-strix-rocmfp4/bin build DOES offload for real: with
# -ngl 99 explicit, `llama-perplexity -v` shows layers assigned to ROCm0/
# Vulkan0 devices and a memory breakdown with non-zero ROCm0/Vulkan0 buffers
# (pipeline parallelism across both). That build also defaults `-fit on`,
# which auto-offloads via its own layer-fitting heuristic even when -ngl is
# never passed -- orthogonal to this feature (we still only add the flag
# when the caller/env explicitly asks for it, to keep the omitted-flag cmd
# byte-identical to historical behavior).


def test_env_int_parses_valid_int():
    assert _env_int("MAGICQUANT_NGL_TEST_DOES_NOT_EXIST_XYZ") is None


def test_env_int_unset_returns_none(monkeypatch):
    monkeypatch.delenv("MQ_TEST_ENV_INT", raising=False)
    assert _env_int("MQ_TEST_ENV_INT") is None


def test_env_int_valid_value(monkeypatch):
    monkeypatch.setenv("MQ_TEST_ENV_INT", "42")
    assert _env_int("MQ_TEST_ENV_INT") == 42


def test_env_int_invalid_value_returns_none(monkeypatch):
    monkeypatch.setenv("MQ_TEST_ENV_INT", "not-an-int")
    assert _env_int("MQ_TEST_ENV_INT") is None


def test_env_int_empty_string_returns_none(monkeypatch):
    monkeypatch.setenv("MQ_TEST_ENV_INT", "")
    assert _env_int("MQ_TEST_ENV_INT") is None


def _patched_finders(monkeypatch):
    monkeypatch.setattr(LlamaCppTools, "_find_llamacpp", lambda self: "/fake/llamacpp")
    monkeypatch.setattr(LlamaCppTools, "_find_quantize_tool", lambda self: "/fake/llama-quantize")
    monkeypatch.setattr(LlamaCppTools, "_find_perplexity_tool", lambda self: "/fake/llama-perplexity")
    monkeypatch.setattr(
        "magicquant.utils.llamacpp._find_bench_tool", lambda perplexity_tool_path: None
    )


def test_init_defaults_ngl_threads_to_none_without_env(monkeypatch):
    monkeypatch.delenv("MAGICQUANT_NGL", raising=False)
    monkeypatch.delenv("MAGICQUANT_THREADS", raising=False)
    _patched_finders(monkeypatch)
    tools = LlamaCppTools()
    assert tools.ngl is None
    assert tools.threads is None


def test_init_reads_ngl_and_threads_from_env(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_NGL", "20")
    monkeypatch.setenv("MAGICQUANT_THREADS", "8")
    _patched_finders(monkeypatch)
    tools = LlamaCppTools()
    assert tools.ngl == 20
    assert tools.threads == 8


def test_init_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("MAGICQUANT_NGL", "20")
    monkeypatch.setenv("MAGICQUANT_THREADS", "8")
    _patched_finders(monkeypatch)
    tools = LlamaCppTools(ngl=99, threads=4)
    assert tools.ngl == 99
    assert tools.threads == 4


def test_calculate_perplexity_cmd_unchanged_when_unset(tmp_path):
    import subprocess

    data_file = tmp_path / "corpus.txt"
    data_file.write_text("hello world\n")

    tools = _bare_tools(data_file=str(data_file))
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="Final estimate: PPL = 5.0 +/- 0.1"
    )
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.calculate_perplexity("/some/model.gguf")
    cmd = run.call_args[0][0]
    assert "-ngl" not in cmd
    assert "-t" not in cmd


def test_calculate_perplexity_cmd_includes_flags_when_set(tmp_path):
    import subprocess

    data_file = tmp_path / "corpus.txt"
    data_file.write_text("hello world\n")

    tools = _bare_tools(data_file=str(data_file), ngl=99, threads=16)
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="Final estimate: PPL = 5.0 +/- 0.1"
    )
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.calculate_perplexity("/some/model.gguf")
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("-ngl") + 1] == "99"
    assert cmd[cmd.index("-t") + 1] == "16"


def test_bench_cmd_unchanged_when_unset():
    import subprocess

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_BENCH_JSON, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.bench("/some/model.gguf", n_prompt=8, n_gen=8, reps=1)
    cmd = run.call_args[0][0]
    assert "-ngl" not in cmd
    assert "-t" not in cmd


def test_bench_cmd_includes_flags_when_set():
    import subprocess

    tools = _bare_tools(ngl=30, threads=12)
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_BENCH_JSON, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.bench("/some/model.gguf", n_prompt=8, n_gen=8, reps=1)
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("-ngl") + 1] == "30"
    assert cmd[cmd.index("-t") + 1] == "12"


def test_save_base_logits_cmd_unchanged_when_unset(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"fake-logits" * 1000)
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run) as run:
        tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    cmd = run.call_args[0][0]
    assert "-ngl" not in cmd
    assert "-t" not in cmd


def test_save_base_logits_cmd_includes_flags_when_set(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools(ngl=99, threads=16)
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"fake-logits" * 1000)
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run) as run:
        tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("-ngl") + 1] == "99"
    assert cmd[cmd.index("-t") + 1] == "16"


# ── F4: save_base_logits must be measured under the SAME batching as
# ── calculate_perplexity, or the fused baseline and every candidate it's
# ── compared against are apples-to-oranges. ─────────────────────────────────


def test_save_base_logits_cmd_includes_batch_flags(tmp_path):
    """save_base_logits used to omit --batch-size/--ubatch-size entirely
    while calculate_perplexity passed them -- the fused baseline (see
    run_measured_search's Step 1b) was then measured under different
    batching than every candidate's calculate_perplexity/calculate_kl_
    divergence pass."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"fake-logits" * 1000)
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run) as run:
        tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--batch-size") + 1] == "512"
    assert cmd[cmd.index("--ubatch-size") + 1] == "128"


def test_all_three_measurement_paths_batch_flags_match(tmp_path):
    """Three-way invocation-parity check: calculate_perplexity,
    save_base_logits AND calculate_kl_divergence must pass identical
    batch/ubatch values. All three feed one measured_loss computation --
    baseline PPL comes from the fused save_base_logits pass while every
    candidate's PPL comes from calculate_kl_divergence's own 'Mean PPL(Q)'
    (or, KL off, from calculate_perplexity) -- so a batching mismatch on ANY
    pair silently measures baseline and candidates under different
    conditions. Pinned via the shared _perplexity_batch_flags() helper
    rather than independently hand-maintained literal lists."""
    import subprocess

    data_file = tmp_path / "corpus.txt"
    data_file.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"
    out_logits.write_bytes(b"fake-logits")

    tools = _bare_tools(data_file=str(data_file))
    ppl_fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="Final estimate: PPL = 5.0 +/- 0.1"
    )
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=ppl_fake) as run:
        tools.calculate_perplexity("/some/model.gguf")
    ppl_cmd = run.call_args[0][0]

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"fake-logits" * 1000)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run) as run:
        tools.save_base_logits("/base/model.gguf", str(data_file), str(out_logits))
    base_logits_cmd = run.call_args[0][0]

    kl_fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=_KL_STDOUT, stderr=""
    )
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=kl_fake) as run:
        tools.calculate_kl_divergence(
            "/quant/model.gguf", str(out_logits), str(data_file)
        )
    kl_cmd = run.call_args[0][0]

    def _batch_kv(cmd):
        return {
            cmd[i]: cmd[i + 1]
            for i in range(len(cmd))
            if cmd[i] in ("--batch-size", "--ubatch-size")
        }

    assert _batch_kv(ppl_cmd), "expected explicit batch flags on the PPL path"
    assert _batch_kv(ppl_cmd) == _batch_kv(base_logits_cmd)
    assert _batch_kv(ppl_cmd) == _batch_kv(kl_cmd)


def test_calculate_kl_divergence_cmd_unchanged_when_unset(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    base_logits = tmp_path / "base.kld"
    base_logits.write_bytes(b"fake-logits")

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_KL_STDOUT, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.calculate_kl_divergence("/quant/model.gguf", str(base_logits), str(corpus))
    cmd = run.call_args[0][0]
    assert "-ngl" not in cmd
    assert "-t" not in cmd


def test_calculate_kl_divergence_cmd_includes_flags_when_set(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    base_logits = tmp_path / "base.kld"
    base_logits.write_bytes(b"fake-logits")

    tools = _bare_tools(ngl=99, threads=16)
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_KL_STDOUT, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.calculate_kl_divergence("/quant/model.gguf", str(base_logits), str(corpus))
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("-ngl") + 1] == "99"
    assert cmd[cmd.index("-t") + 1] == "16"


# --- Live smoke test (real binaries + real tiny model) ----------------------

_LLAMA_BENCH_BIN = "/home/lucas/ROCmFPX/build-strix-rocmfp4/bin/llama-bench"
_LLAMA_PERPLEXITY_BIN = "/home/lucas/ROCmFPX/build-strix-rocmfp4/bin/llama-perplexity"
_TINY_MODEL = (
    "/server/ai/models/source/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf"
)

_MISSING = [
    p for p in (_LLAMA_BENCH_BIN, _LLAMA_PERPLEXITY_BIN, _TINY_MODEL) if not Path(p).is_file()
]

# llama-perplexity needs at least ctx_size*chunks TOKENS in the corpus or it
# exits 0 but silently fails (see _MIN_LOGITS_FILE_BYTES in llamacpp.py); at
# ctx_size=128/chunks=2 below that's 256 tokens minimum, so this repeats a
# short passage rather than relying on one being long enough by luck.
_TINY_CORPUS = ("""The quick brown fox jumps over the lazy dog. In the heart of the old city,
narrow cobblestone streets wind between weathered buildings while merchants call
out from small shops selling spices and handcrafted goods. Scientists have long
studied the behavior of complex systems, from the turbulent flow of fluids to the
intricate dance of planets around distant stars.
""" * 6)


@pytest.mark.skipif(
    bool(_MISSING),
    reason=f"ROCmFPX bin/tiny model not available: {_MISSING}",
)
def test_bench_and_kl_self_consistency_live(tmp_path):
    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.perplexity_tool = _LLAMA_PERPLEXITY_BIN
    tools.bench_tool = _LLAMA_BENCH_BIN
    tools.ctx_size = 128
    tools.ngl = None
    tools.threads = None

    # --- speed measurement ---
    bench_result = tools.bench(_TINY_MODEL, n_prompt=8, n_gen=8, reps=1, timeout=120)
    assert bench_result is not None
    assert bench_result["pp_ts"] > 0
    assert bench_result["tg_ts"] > 0

    # --- KL self-consistency: q8_0 vs its own saved logits ---
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_TINY_CORPUS)
    base_logits = tmp_path / "base_logits.kld"

    saved_ppl = tools.save_base_logits(
        _TINY_MODEL, str(corpus), str(base_logits), ctx_size=128, chunks=2, timeout=120,
    )
    assert saved_ppl is not None and saved_ppl > 0
    assert base_logits.is_file()

    kl_result = tools.calculate_kl_divergence(
        _TINY_MODEL, str(base_logits), str(corpus), ctx_size=128, chunks=2, timeout=120,
    )
    assert kl_result is not None
    # Empirically the self-consistency mean lands at ~-0.000019 (a small,
    # deterministic saved-logits precision artifact -- see module docstring)
    # rather than exactly 0, so assert magnitude rather than sign.
    assert abs(kl_result["mean_kl"]) < 1e-3

    # Fusion parity (Features 1+2, live): the KL pass's own "Mean PPL(Q)"
    # must match a standalone/save-base-logits perplexity pass over the same
    # model/corpus/chunks -- the exact invariant run_measured_search's
    # baseline+candidate fusion relies on to replace two llama-perplexity
    # invocations with one.
    assert kl_result.get("ppl") is not None
    assert kl_result["ppl"] == pytest.approx(saved_ppl, abs=0.01)


@pytest.mark.gpu
@pytest.mark.skipif(
    bool(_MISSING),
    reason=f"ROCmFPX bin/tiny model not available: {_MISSING}",
)
def test_calculate_perplexity_ngl_actually_offloads_to_gpu(tmp_path):
    """Manually verified (see module docstring above): passing -ngl 99 to
    this build assigns layers to ROCm0/Vulkan0 devices, confirmed via
    `llama-perplexity -v` device-assignment + memory-breakdown lines. This
    just pins that the flag threads through end-to-end and the run still
    succeeds/parses on the real GPU-offload build, not just CPU-only mocks.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_TINY_CORPUS)

    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.perplexity_tool = _LLAMA_PERPLEXITY_BIN
    tools.bench_tool = _LLAMA_BENCH_BIN
    tools.ctx_size = 128
    tools.ngl = 99
    tools.threads = None
    tools.data_file = None

    ppl = tools.calculate_perplexity(_TINY_MODEL, data_file=str(corpus), ctx_size=128)
    assert ppl is not None
    assert ppl > 0


def test_resolve_data_file_falls_back_to_bundled_corpus(tmp_path, capsys):
    """A llama.cpp dir with no wikitext nearby must fall back to the bundled
    calibration corpus (with a warning) instead of returning None -- an
    otherwise-valid measured search died on this when llamacpp_path pointed
    at a ROCmFPX build dir (2026-07-04)."""
    from magicquant.imatrix import DEFAULT_CORPUS_PATH

    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.llamacpp_path = str(tmp_path / "fork-build")
    tools.data_file = None

    resolved = tools._resolve_data_file(None)
    assert resolved == str(DEFAULT_CORPUS_PATH.resolve())
    assert "falling back to the bundled calibration corpus" in capsys.readouterr().out


def test_resolve_data_file_still_prefers_wikitext(tmp_path):
    wiki = tmp_path / "wikitext-2-raw" / "wiki.test.raw"
    wiki.parent.mkdir(parents=True)
    wiki.write_text("real corpus")

    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.llamacpp_path = str(tmp_path)
    tools.data_file = None
    assert tools._resolve_data_file(None) == str(wiki.resolve())


def test_ppl_chunks_env_caps_all_measurement_passes(tmp_path, monkeypatch):
    """MAGICQUANT_PPL_CHUNKS must cap perplexity AND both KL passes so every
    measurement in a run uses the same corpus slice (a full-corpus pass on a
    27B is ~55 min on this box; a measured search needs ~20 passes)."""
    monkeypatch.setenv("MAGICQUANT_PPL_CHUNKS", "150")
    monkeypatch.setattr(LlamaCppTools, "_find_llamacpp", lambda self: str(tmp_path))
    monkeypatch.setattr(LlamaCppTools, "_find_quantize_tool", lambda self: "/bin/true")
    monkeypatch.setattr(LlamaCppTools, "_find_perplexity_tool", lambda self: "/bin/true")
    tools = LlamaCppTools()
    assert tools.ppl_chunks == 150

    corpus = tmp_path / "c.txt"; corpus.write_text("hi")
    model = tmp_path / "m.gguf"; model.write_bytes(b"g")
    captured = {}

    def fake_run(cmd, timeout):
        captured["cmd"] = cmd
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(tools, "_run_perplexity_subprocess", fake_run)
    tools.data_file = str(corpus)

    tools.calculate_perplexity(str(model), verbose=False)
    assert captured["cmd"][captured["cmd"].index("--chunks") + 1] == "150"

    tools.save_base_logits(str(model), str(corpus), str(tmp_path / "o.kld"))
    assert captured["cmd"][captured["cmd"].index("--chunks") + 1] == "150"

    tools.calculate_kl_divergence(str(model), str(tmp_path / "o.kld"), str(corpus))
    assert captured["cmd"][captured["cmd"].index("--chunks") + 1] == "150"


def test_ppl_chunks_unset_keeps_historical_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("MAGICQUANT_PPL_CHUNKS", raising=False)
    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.llamacpp_path = str(tmp_path)
    tools.perplexity_tool = "/bin/true"
    tools.ctx_size = 512
    tools.data_file = None
    corpus = tmp_path / "c.txt"; corpus.write_text("hi")
    tools.data_file = str(corpus)
    captured = {}

    def fake_run(cmd, timeout):
        captured["cmd"] = cmd
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(tools, "_run_perplexity_subprocess", fake_run)
    tools.calculate_perplexity(str(tmp_path / "m.gguf"), verbose=False)
    assert "--chunks" not in captured["cmd"]

    tools.calculate_kl_divergence(str(tmp_path / "m.gguf"), "b.kld", str(corpus))
    assert captured["cmd"][captured["cmd"].index("--chunks") + 1] == "-1"


def test_bench_defaults_are_3reps_128gen():
    import subprocess
    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_BENCH_JSON, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.bench("/m.gguf")  # no explicit reps/n_gen
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("-r") + 1] == "3"
    assert cmd[cmd.index("-n") + 1] == "128"


def test_bench_env_overrides_reps_and_ngen(monkeypatch):
    import subprocess
    monkeypatch.setenv("MAGICQUANT_BENCH_REPS", "5")
    monkeypatch.setenv("MAGICQUANT_BENCH_NGEN", "256")
    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_BENCH_JSON, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.bench("/m.gguf")
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("-r") + 1] == "5"
    assert cmd[cmd.index("-n") + 1] == "256"


def test_bench_explicit_arg_beats_env(monkeypatch):
    import subprocess
    monkeypatch.setenv("MAGICQUANT_BENCH_REPS", "5")
    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_BENCH_JSON, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake) as run:
        tools.bench("/m.gguf", reps=2)
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("-r") + 1] == "2"


# ── Measurement-timeout fix (2026-08 field report): flat 2h _SUBPROCESS_
# ── TIMEOUT didn't scale with artifact size, so a healthy (not hung),
# ── bandwidth-bound pass over a 37.8GB candidate hit the cap on BOTH the
# ── KL and PPL-fallback legs -- 4h burned for zero measurements, with no
# ── record distinguishing "measured and lost" from "never attempted". ──


def test_bench_default_timeout_unchanged_at_300():
    """bench() keeps its own flat _BENCH_TIMEOUT -- explicitly NOT wired
    to _measure_timeout, per the fix's scope (bench measures throughput
    over a short prompt/gen, not a full corpus pass, so it never needed
    the size-aware scaling)."""
    import inspect
    sig = inspect.signature(LlamaCppTools.bench)
    assert sig.parameters["timeout"].default == llamacpp_mod._BENCH_TIMEOUT == 300


def test_subprocess_timeout_env_override_at_import(monkeypatch):
    """_SUBPROCESS_TIMEOUT is read ONCE at module-import time
    (``_env_int("MAGICQUANT_SUBPROCESS_TIMEOUT") or 7200``) -- there is no
    per-instance re-read to exercise the way MAGICQUANT_PPL_CHUNKS gets
    re-read fresh in __init__ (see test_ppl_chunks_env_caps_all_
    measurement_passes above). Spawn a subprocess with the env var set and
    import fresh rather than importlib.reload()-ing this module in-process,
    which would mutate the SAME module dict every other already-imported
    test file's LlamaCppTools methods read their globals from."""
    import subprocess
    import sys

    env = dict(os.environ)
    env["MAGICQUANT_SUBPROCESS_TIMEOUT"] = "3600"
    out = subprocess.run(
        [
            sys.executable, "-c",
            "from magicquant.utils.llamacpp import _SUBPROCESS_TIMEOUT; "
            "print(_SUBPROCESS_TIMEOUT)",
        ],
        env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    assert out.stdout.strip() == "3600"


def test_subprocess_timeout_env_unset_defaults_to_7200():
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("MAGICQUANT_SUBPROCESS_TIMEOUT", None)
    out = subprocess.run(
        [
            sys.executable, "-c",
            "from magicquant.utils.llamacpp import _SUBPROCESS_TIMEOUT; "
            "print(_SUBPROCESS_TIMEOUT)",
        ],
        env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    assert out.stdout.strip() == "7200"


def test_measure_timeout_small_file_returns_base(tmp_path):
    small = tmp_path / "small.gguf"
    small.write_bytes(b"x" * 1024)  # 1 KiB -- far below the bandwidth floor
    tools = _bare_tools()
    assert tools._measure_timeout(str(small)) == llamacpp_mod._SUBPROCESS_TIMEOUT


def test_measure_timeout_huge_file_scales(monkeypatch, tmp_path):
    """37.8GB is the field report's actual candidate size -- confirms the
    worked example (~2.6h for the plain-PPL leg) arithmetically."""
    huge_path = tmp_path / "huge.gguf"
    huge_path.write_bytes(b"")  # existence only; size is faked below
    huge_bytes = 37_800_000_000
    monkeypatch.setattr(os.path, "getsize", lambda p: huge_bytes)

    tools = _bare_tools()
    expected = huge_bytes // llamacpp_mod._MIN_MEASURE_BANDWIDTH
    assert expected > llamacpp_mod._SUBPROCESS_TIMEOUT  # scaling actually kicks in
    result = tools._measure_timeout(str(huge_path))
    assert result == expected
    assert 2.5 * 3600 < result < 2.7 * 3600  # ~2.6h, matching the field report


def test_measure_timeout_kl_doubles_plain_ppl(monkeypatch, tmp_path):
    huge_path = tmp_path / "huge.gguf"
    huge_path.write_bytes(b"")
    monkeypatch.setattr(os.path, "getsize", lambda p: 37_800_000_000)

    tools = _bare_tools()
    ppl_timeout = tools._measure_timeout(str(huge_path), kl=False)
    kl_timeout = tools._measure_timeout(str(huge_path), kl=True)
    assert kl_timeout == ppl_timeout * 2


def test_measure_timeout_missing_file_returns_base():
    tools = _bare_tools()
    assert tools._measure_timeout("/does/not/exist.gguf") == llamacpp_mod._SUBPROCESS_TIMEOUT


def test_measure_timeout_always_finite(monkeypatch, tmp_path):
    """Hang protection must never be disabled, only sized correctly (this
    box has an OOM-livelock history) -- even a pathological fake size must
    still produce a finite int, never inf/nan."""
    import math
    monkeypatch.setattr(os.path, "getsize", lambda p: 10 ** 15)
    tools = _bare_tools()
    result = tools._measure_timeout(str(tmp_path / "whatever.gguf"), kl=True)
    assert isinstance(result, int) and math.isfinite(result)


def test_run_subprocess_or_none_records_timeout_failure():
    import subprocess

    tools = _bare_tools()

    def _raise_timeout(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_raise_timeout):
        result = tools._run_subprocess_or_none(["fake"], 10, "Test op")
    assert result is None
    assert tools._last_subprocess_failure == {"kind": "timeout", "label": "Test op"}


def test_run_subprocess_or_none_records_error_failure():
    import subprocess

    tools = _bare_tools()

    def _raise_called_process_error(cmd, timeout):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_raise_called_process_error):
        result = tools._run_subprocess_or_none(["fake"], 10, "Test op")
    assert result is None
    assert tools._last_subprocess_failure == {"kind": "error", "label": "Test op"}


def test_run_subprocess_or_none_clears_failure_on_success():
    import subprocess

    tools = _bare_tools()
    tools._last_subprocess_failure = {"kind": "timeout", "label": "stale"}  # leftover from a prior call
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake):
        result = tools._run_subprocess_or_none(["fake"], 10, "Test op")
    assert result is fake
    assert tools._last_subprocess_failure is None


def test_run_subprocess_or_none_failure_attr_getattr_safe_on_never_called_instance():
    """A bare __new__ instance that has never run _run_subprocess_or_none
    has no _last_subprocess_failure attribute at all -- callers (the
    orchestrator) must read it via getattr(..., None), not direct access."""
    tools = LlamaCppTools.__new__(LlamaCppTools)
    assert getattr(tools, "_last_subprocess_failure", None) is None


# ── Q2 (Opus review, 2026-08-10): _last_subprocess_failure LEAK across
# ── calls -- calculate_perplexity's corpus-resolution early return (fires
# ── BEFORE _run_subprocess_or_none is ever reached) used to leave a
# ── PREVIOUS call's stale "timeout"/"error" reading in place, since
# ── nothing cleared the flag on that path. Reviewer reproduced 1 real
# ── timeout turning into 5 disclosure entries downstream in the
# ── orchestrator, each permanently blacklisting its config. ─────────────


def test_calculate_perplexity_early_return_does_not_leak_previous_timeout(tmp_path):
    """Two sequential calls on the SAME instance: the first genuinely
    times out (a real _run_perplexity_subprocess TimeoutExpired, setting
    _last_subprocess_failure for real); the second hits the
    resolve-data-file early return (corpus unresolvable for that one
    call) and must read back _last_subprocess_failure as None -- not the
    first call's stale timeout -- because _run_subprocess_or_none is
    never reached on the second call at all."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")

    tools = _bare_tools(data_file=str(corpus))

    def _raise_timeout(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_raise_timeout):
        first = tools.calculate_perplexity("/model_1.gguf")
    assert first is None
    assert tools._last_subprocess_failure == {
        "kind": "timeout", "label": "Perplexity calculation",
    }

    # Second call: force the SAME early return calculate_perplexity has at
    # `if resolved_data_file is None: return None` -- _resolve_data_file
    # returns None for this call only, so _run_perplexity_subprocess is
    # never invoked (asserted below).
    with mock.patch.object(
        tools, "_resolve_data_file", return_value=None,
    ), mock.patch.object(
        tools, "_run_perplexity_subprocess",
    ) as run_mock:
        second = tools.calculate_perplexity("/model_2.gguf")

    assert second is None
    run_mock.assert_not_called()
    assert tools._last_subprocess_failure is None, (
        "the early return must have cleared the flag, not left the "
        "first call's timeout reading in place"
    )


def test_save_base_logits_and_calculate_kl_divergence_also_clear_at_top(tmp_path):
    """Defensive clearing (Q2): even though save_base_logits/
    calculate_kl_divergence have no KNOWN early-return-before-subprocess
    path today, both now clear _last_subprocess_failure at their own top
    too -- confirms a stale reading from a prior call never survives
    into either method's own outcome."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    base_logits = tmp_path / "base.kld"
    base_logits.write_bytes(b"fake-logits")

    tools = _bare_tools()
    tools._last_subprocess_failure = {"kind": "timeout", "label": "stale from a prior call"}

    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_KL_STDOUT, stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake):
        tools.calculate_kl_divergence("/quant/model.gguf", str(base_logits), str(corpus))
    assert tools._last_subprocess_failure is None

    tools._last_subprocess_failure = {"kind": "error", "label": "stale from a prior call"}
    out_logits = tmp_path / "out.kld"

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"fake-logits" * 1000)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert tools._last_subprocess_failure is None


def test_calculate_perplexity_scales_timeout_with_file_size(monkeypatch, tmp_path):
    import subprocess

    data_file = tmp_path / "corpus.txt"
    data_file.write_text("hello world\n")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"g")
    monkeypatch.setattr(os.path, "getsize", lambda p: 37_800_000_000)

    tools = _bare_tools(data_file=str(data_file))
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="Final estimate: PPL = 5.0 +/- 0.1"
    )
    captured = {}

    def _fake_run(cmd, timeout):
        captured["timeout"] = timeout
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        tools.calculate_perplexity(str(model))
    assert captured["timeout"] == 37_800_000_000 // llamacpp_mod._MIN_MEASURE_BANDWIDTH
    assert captured["timeout"] > llamacpp_mod._SUBPROCESS_TIMEOUT


def test_save_base_logits_uses_2x_kl_timeout_by_default(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"
    base_model = tmp_path / "base.gguf"
    base_model.write_bytes(b"g")
    monkeypatch.setattr(os.path, "getsize", lambda p: 37_800_000_000)

    tools = _bare_tools()
    captured = {}

    def _fake_run(cmd, timeout):
        captured["timeout"] = timeout
        out_logits.write_bytes(b"fake-logits" * 1000)
        import subprocess
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        tools.save_base_logits(str(base_model), str(corpus), str(out_logits))
    ppl_leg = 37_800_000_000 // llamacpp_mod._MIN_MEASURE_BANDWIDTH
    assert captured["timeout"] == ppl_leg * 2


def test_calculate_kl_divergence_uses_2x_kl_timeout_by_default(monkeypatch, tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    base_logits = tmp_path / "base.kld"
    base_logits.write_bytes(b"fake-logits")
    quant_model = tmp_path / "quant.gguf"
    quant_model.write_bytes(b"g")
    monkeypatch.setattr(os.path, "getsize", lambda p: 37_800_000_000)

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_KL_STDOUT, stderr="")
    captured = {}

    def _fake_run(cmd, timeout):
        captured["timeout"] = timeout
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        tools.calculate_kl_divergence(str(quant_model), str(base_logits), str(corpus))
    ppl_leg = 37_800_000_000 // llamacpp_mod._MIN_MEASURE_BANDWIDTH
    assert captured["timeout"] == ppl_leg * 2


def test_calculate_kl_divergence_explicit_timeout_still_overrides_default(monkeypatch, tmp_path):
    """An explicit timeout= (e.g. the live smoke test's timeout=120) must
    keep winning over the new size-aware default -- backward compatible."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    base_logits = tmp_path / "base.kld"
    base_logits.write_bytes(b"fake-logits")
    monkeypatch.setattr(os.path, "getsize", lambda p: 37_800_000_000)

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=_KL_STDOUT, stderr="")
    captured = {}

    def _fake_run(cmd, timeout):
        captured["timeout"] = timeout
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        tools.calculate_kl_divergence(
            "/quant/model.gguf", str(base_logits), str(corpus), timeout=120,
        )
    assert captured["timeout"] == 120
