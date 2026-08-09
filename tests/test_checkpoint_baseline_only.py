"""Regression: a checkpoint with a baseline but null sensitivity_weights must
fall through to probing, not restore the null and crash the predictor."""
import json
from magicquant.orchestrator import MagicQuantOrchestrator


def test_baseline_only_checkpoint_does_not_restore_null_sensitivities(tmp_path):
    src = tmp_path / "m.gguf"; src.write_bytes(b"GGUF" + b"\0" * 64)
    out = tmp_path / "out"; out.mkdir()
    orch = MagicQuantOrchestrator(source_model_path=str(src), output_dir=str(out))
    orch.baseline_ppl = 24.0036
    orch.baseline_provenance = "measured"
    path = orch._measured_checkpoint_path()
    orch._write_measured_checkpoint(path)
    ck = json.load(open(path))
    assert ck["baseline_ppl"] == 24.0036
    assert ck["sensitivity_weights"] is None
    # The guard is `checkpoint.get("sensitivity_weights")` being falsy -> probe.
    assert not ck.get("sensitivity_weights"), "null weights must read as not-restored"
