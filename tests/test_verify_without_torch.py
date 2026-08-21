"""The verification path must run on a machine with nothing installed.

Both commands printed in outreach — and in the README's "verify the
headline" section — are claimed to be stdlib-only. That claim was false
for `verify_from_primary.py`: it imported the scorer from `text_ttt`,
which imports torch at module scope, so on any box without PyTorch the
command died with a bare ModuleNotFoundError. A hostile reader running
the two commands we dared them to run would have hit a stack trace on
the one carrying the primary-evidence claim.

These tests make the claim enforceable: they simulate a machine with no
torch (a stub that raises exactly as a missing module does, first on the
path) and require the verification scripts to complete anyway.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STUB = "raise ModuleNotFoundError(\"No module named 'torch'\")\n"


@pytest.fixture(scope="module")
def no_torch_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """An environment where `import torch` fails as if never installed."""
    shim = tmp_path_factory.mktemp("no-torch")
    (shim / "torch.py").write_text(STUB)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shim), str(REPO / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _run(script: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=600,
    )


def test_the_stub_really_blocks_torch(no_torch_env: dict[str, str]) -> None:
    """Guard the guard: if the shim stopped working these tests would pass
    vacuously on any machine that happens to have torch installed."""
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True, text=True, env=no_torch_env, cwd=REPO)
    assert probe.returncode != 0
    assert "No module named 'torch'" in probe.stderr


def test_verify_from_primary_runs_without_torch(no_torch_env: dict[str, str]) -> None:
    artifact = "experiments/novel_schema_f_0.5b_k30_seed1_docadapted_2026-08-19.json"
    result = _run("verify_from_primary.py", artifact, env=no_torch_env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "PRIMARY-VERIFIED" in result.stdout
    assert "torch" not in result.stderr


def test_verify_verdict_runs_without_torch(no_torch_env: dict[str, str]) -> None:
    result = _run("verify_verdict.py", env=no_torch_env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "MATCHES" in result.stdout


def test_addendum_e_summary_runs_without_torch(no_torch_env: dict[str, str]) -> None:
    result = _run("addendum_e_summary.py", env=no_torch_env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "PASS" in result.stdout


def test_read_addendum_d_runs_without_torch(no_torch_env: dict[str, str]) -> None:
    result = _run("read_addendum_d.py", env=no_torch_env)
    assert result.returncode == 0, result.stderr[-2000:]


def test_scoring_module_itself_is_torch_free(no_torch_env: dict[str, str]) -> None:
    """The scorer is the piece the verification path depends on; keep it
    importable on its own so a future edit cannot quietly re-couple it."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "from arcttt.scoring import score_text_output;"
         "s = score_text_output('{\"a\": \"1,000\"}', '{\"a\": 1000}');"
         "print(s.valid_json, s.micro_f1)"],
        capture_output=True, text=True, env=no_torch_env, cwd=REPO)
    assert probe.returncode == 0, probe.stderr[-2000:]
    # numeric normalization still applies: "1,000" == 1000
    assert probe.stdout.strip() == "True 1.0"


def test_text_ttt_still_re_exports_the_scorer() -> None:
    """Backwards compatibility: kernels and older callers import the scorer
    from text_ttt. That must keep working (with torch present)."""
    pytest.importorskip("torch")
    from arcttt.scoring import score_text_output as direct
    from arcttt.text_ttt import score_text_output as reexported
    assert direct is reexported
