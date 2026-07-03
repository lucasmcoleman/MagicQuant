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
from pathlib import Path
from unittest import mock

import pytest

from magicquant.utils.llamacpp import (
    LlamaCppTools,
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
    assert parsed == {"pp_ts": pytest.approx(1113.103876), "tg_ts": pytest.approx(172.806123)}


def test_parse_bench_json_ignores_surrounding_banner_noise():
    parsed = _parse_bench_json(_BENCH_JSON_WITH_BANNER_NOISE)
    assert parsed == {"pp_ts": pytest.approx(1113.103876), "tg_ts": pytest.approx(172.806123)}


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


def test_parse_kl_output_returns_none_when_absent():
    assert _parse_kl_output("nothing relevant here") is None


# --- LlamaCppTools method tests (mocked subprocess) -------------------------


def _bare_tools(**attrs) -> LlamaCppTools:
    """Construct a LlamaCppTools bypassing __init__ (no binary discovery)."""
    tools = LlamaCppTools.__new__(LlamaCppTools)
    tools.perplexity_tool = "/bin/true"
    tools.bench_tool = "/bin/true"
    tools.ctx_size = 512
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
    assert result == {"pp_ts": pytest.approx(1113.103876), "tg_ts": pytest.approx(172.806123)}
    # Sanity: bench invoked llama-bench with -o json and the given -p/-n/-r.
    cmd = run.call_args[0][0]
    assert cmd[0] == "/bin/true"
    assert "-o" in cmd and "json" in cmd
    assert "8" in cmd  # -p 8 / -n 8


def test_save_base_logits_true_when_file_written(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="saved", stderr="")

    def _fake_run(cmd, timeout):
        # Simulate llama-perplexity actually writing a real (not stub-sized)
        # logits file -- save_base_logits rejects anything <= a few KB as a
        # failed-but-exit-0 stub (see _MIN_LOGITS_FILE_BYTES).
        out_logits.write_bytes(b"fake-logits" * 1000)
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        ok = tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert ok is True
    assert out_logits.is_file()


def test_save_base_logits_false_when_file_is_stub_sized(tmp_path):
    """A too-short corpus makes llama-perplexity exit 0 but write only a
    ~12-byte header stub -- save_base_logits must treat that as failure, not
    just check the file exists (regression: this previously returned True,
    then calculate_kl_divergence silently failed to find 'Mean KLD')."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def _fake_run(cmd, timeout):
        out_logits.write_bytes(b"_logits_\x80\x00\x00\x00")  # the real 12-byte stub
        return fake

    with mock.patch.object(tools, "_run_perplexity_subprocess", side_effect=_fake_run):
        ok = tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert ok is False


def test_save_base_logits_false_when_file_missing(tmp_path):
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\n")
    out_logits = tmp_path / "base.kld"  # never created

    tools = _bare_tools()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with mock.patch.object(tools, "_run_perplexity_subprocess", return_value=fake):
        ok = tools.save_base_logits("/base/model.gguf", str(corpus), str(out_logits))
    assert ok is False


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

    # --- speed measurement ---
    bench_result = tools.bench(_TINY_MODEL, n_prompt=8, n_gen=8, reps=1, timeout=120)
    assert bench_result is not None
    assert bench_result["pp_ts"] > 0
    assert bench_result["tg_ts"] > 0

    # --- KL self-consistency: q8_0 vs its own saved logits ---
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_TINY_CORPUS)
    base_logits = tmp_path / "base_logits.kld"

    saved_ok = tools.save_base_logits(
        _TINY_MODEL, str(corpus), str(base_logits), ctx_size=128, chunks=2, timeout=120,
    )
    assert saved_ok is True
    assert base_logits.is_file()

    kl_result = tools.calculate_kl_divergence(
        _TINY_MODEL, str(base_logits), str(corpus), ctx_size=128, chunks=2, timeout=120,
    )
    assert kl_result is not None
    # Empirically the self-consistency mean lands at ~-0.000019 (a small,
    # deterministic saved-logits precision artifact -- see module docstring)
    # rather than exactly 0, so assert magnitude rather than sign.
    assert abs(kl_result["mean_kl"]) < 1e-3
