"""GitHub Actions workflow files must be valid YAML with the expected shape.

This exists because a single unquoted step name containing ": " --

    - name: Install (with dev extras: pytest + gguf + ruff)

-- made .github/workflows/ci.yml unparseable, and GitHub's response to an
unparseable workflow is to complete the run in 0s with "This run likely failed
because of a workflow file issue". That reads like flaky infrastructure, so it
went unnoticed through 61 consecutive failed runs across roughly two months,
during which the project had NO CI coverage at all.

The guard has to live in the test suite rather than in CI itself: a workflow
that cannot parse also cannot run a job that would have validated it. These
tests run locally and in any environment that runs pytest.
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _workflow_files():
    if not WORKFLOW_DIR.is_dir():
        return []
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def test_workflow_directory_is_present():
    assert _workflow_files(), f"no workflow files found under {WORKFLOW_DIR}"


@pytest.mark.parametrize("wf", _workflow_files(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(wf):
    """The failure this whole module exists for."""
    try:
        yaml.safe_load(wf.read_text())
    except yaml.YAMLError as e:
        pytest.fail(
            f"{wf.name} is not valid YAML, so GitHub will refuse to run it and "
            f"report only a generic 'workflow file issue':\n{e}"
        )


@pytest.mark.parametrize("wf", _workflow_files(), ids=lambda p: p.name)
def test_workflow_has_jobs_with_steps(wf):
    """Parseable but empty is still broken, just less obviously."""
    doc = yaml.safe_load(wf.read_text())
    assert isinstance(doc, dict), f"{wf.name} did not parse to a mapping"

    # 'on' is a YAML 1.1 boolean, so safe_load turns the key into True.
    # Accept either form rather than pinning one loader's behaviour.
    assert "on" in doc or True in doc, f"{wf.name} has no trigger ('on') block"

    jobs = doc.get("jobs")
    assert jobs, f"{wf.name} defines no jobs"
    for job_name, job in jobs.items():
        assert job.get("runs-on"), f"{wf.name}:{job_name} has no runs-on"
        if "uses" in job:            # reusable workflow call, no inline steps
            continue
        steps = job.get("steps")
        assert steps, f"{wf.name}:{job_name} has no steps"
        for i, step in enumerate(steps):
            assert "uses" in step or "run" in step, (
                f"{wf.name}:{job_name} step {i} has neither 'uses' nor 'run'"
            )
