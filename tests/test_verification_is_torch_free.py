"""The verification job promises to be stdlib-only. Prove it.

`.github/workflows/tests.yml` splits `verify-headline-numbers` out from
the test job on purpose, and says why in its own comment: those scripts
are stdlib-only, *so the job cannot go red for reasons unrelated to the
numbers* — no torch wheel, no pip index, no transformers pin. A badge
that fails for an unrelated reason is worse than no badge on a repo whose
whole claim is that the numbers hold up.

Nothing enforced that. Two steps were added to the job whose script
reached the pinned scorer through `arcttt.text_ttt`, which imports torch
at module scope, and the badge went red on a machine with no torch
installed — while the full test suite passed, and while every one of
those scripts ran clean locally, because locally torch is installed.

So this runs each of them in a subprocess with torch made unimportable,
which is the only way to test the promise the workflow makes.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"

# A stub package directory that shadows torch with a module that raises on
# import, so any transitive `import torch` fails exactly as it would on a
# runner that never installed it.
BLOCKER = '''raise ImportError(
    "torch is deliberately unavailable: this script is invoked by the "
    "stdlib-only verification job and must not need it")
'''


def _torch_free_run(script: str, args: list[str], tmp_path: pathlib.Path):
    stub = tmp_path / "torchstub"
    stub.mkdir(exist_ok=True)
    (stub / "torch.py").write_text(BLOCKER, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "PYTHONPATH": f"{stub}:{REPO / 'src'}"}
    return subprocess.run([sys.executable, str(REPO / script), *args],
                          capture_output=True, text=True, cwd=REPO,
                          env=env, timeout=300)


def _job_steps() -> list[str]:
    """Every `python3 scripts/...` command in the verification job.

    Parsed from the workflow rather than listed here, so a step added to
    CI is covered by this test the day it is added — which is precisely
    what did not happen.
    """
    if not WORKFLOW.exists():
        return []
    text = WORKFLOW.read_text(encoding="utf-8")
    marker = "verify-headline-numbers:"
    if marker not in text:
        return []
    block = text[text.index(marker):]
    steps = []
    for line in block.split("\n"):
        line = line.strip()
        if line.startswith("run:") and "scripts/" in line:
            steps.append(line[len("run:"):].strip())
    return steps


def test_the_workflow_still_has_a_stdlib_only_job() -> None:
    """If this job is ever removed the test below passes vacuously."""
    if not WORKFLOW.exists():
        pytest.skip("no workflow in this cut")
    assert _job_steps(), (
        "the verify-headline-numbers job has no script steps — either it "
        "was removed or this parser stopped matching, and in both cases "
        "the torch-free guarantee is no longer being tested")


@pytest.mark.skipif(not WORKFLOW.exists(), reason="no workflow in this cut")
@pytest.mark.parametrize("command", _job_steps())
def test_each_verification_step_runs_without_torch(
        command: str, tmp_path: pathlib.Path) -> None:
    parts = command.split()
    assert parts[0] in ("python3", "python"), command
    script, args = parts[1], parts[2:]
    result = _torch_free_run(script, args, tmp_path)
    assert "No module named 'torch'" not in result.stderr, (
        f"{script} needs torch, but the job that runs it installs none.\n"
        "Reach the pinned scorer through arcttt.scoring, which is "
        "torch-free by design, rather than through arcttt.text_ttt, which "
        f"imports torch at module scope.\n\n{result.stderr[-1200:]}")
    assert "deliberately unavailable" not in result.stderr, (
        f"{script} imported torch transitively.\n{result.stderr[-1200:]}")
    assert result.returncode == 0, (
        f"{script} failed without torch for another reason:\n"
        f"{result.stderr[-1200:]}")
