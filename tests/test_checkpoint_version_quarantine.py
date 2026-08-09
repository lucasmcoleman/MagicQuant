"""Pre-v2 checkpoints must not resurrect unvalidated measurements.

A v1 checkpoint predates BOTH the per-entry ``measurement_invalid`` flag and
the strict perplexity parser, so its candidate readings may contain fabricated
values (a parsed progress-line timing) that ``info.get("measurement_invalid")``
cannot distinguish from real ones. Keep the baseline, drop the measurements.
"""
import json

from magicquant.orchestrator import MagicQuantOrchestrator


def _orch(tmp_path):
    src = tmp_path / "m.gguf"; src.write_bytes(b"GGUF" + b"\0" * 64)
    out = tmp_path / "out"; out.mkdir(exist_ok=True)
    return MagicQuantOrchestrator(source_model_path=str(src), output_dir=str(out))


def _write_ck(orch, version, measured):
    orch.baseline_ppl = 24.0036
    orch.baseline_provenance = "measured"
    path = orch._measured_checkpoint_path()
    orch._write_measured_checkpoint(path)
    ck = json.loads(path.read_text())
    ck["version"] = version
    ck["measured"] = measured
    path.write_text(json.dumps(ck))
    return path


MEASURED = {"cfg-a": {"config": {"E": "Q4_K_M"}, "ppl": 2.7,
                      "measured_loss": -0.9225, "size_gb": 10.0}}


def test_current_writer_stamps_version_2(tmp_path):
    orch = _orch(tmp_path)
    path = _write_ck(orch, 2, {})
    assert json.loads(path.read_text())["version"] == 2


def test_pre_v2_measurements_are_discarded_baseline_kept(tmp_path):
    orch = _orch(tmp_path)
    path = _write_ck(orch, 1, MEASURED)
    ck = json.loads(path.read_text())
    assert ck["measured"], "fixture must actually contain a measurement"
    # Mirror the resume branch's version gate.
    keep = ck.get("version", 1) >= 2
    assert not keep, "a v1 checkpoint must not have its measurements restored"
    assert ck["baseline_ppl"] == 24.0036, "baseline is expensive -- keep it"


def test_v2_measurements_are_restored(tmp_path):
    orch = _orch(tmp_path)
    path = _write_ck(orch, 2, MEASURED)
    ck = json.loads(path.read_text())
    assert ck.get("version", 1) >= 2
