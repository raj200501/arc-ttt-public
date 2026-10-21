"""The one-line adoption paths: the GitHub Action and the pre-commit hook.

Roadmap step 4 is an outside team running `fencecheck scan` in its CI.
Two files make that one line for them: `action.yml` at the repository
root and `.pre-commit-hooks.yaml` beside it, whose entry is
`tools/fencecheck_hook.py`. Neither can be exercised on a real runner
from here, so this file pins what can be checked offline: the metadata
parses and declares what the README documents, every path they name
exists, the command they run is one the tool's CLI accepts, the hook
reports exactly what the tool reports, and the exit status both
wrappers branch on is 1 on a fail-open site and 0 on a clean file.

Standard library plus pytest. YAML is parsed with PyYAML when it is
importable and with a tiny line-based reader when it is not, so this
file never adds a dependency to the suite.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "fencecheck.py"
HOOK = REPO / "tools" / "fencecheck_hook.py"
ACTION = REPO / "action.yml"
HOOKS = REPO / ".pre-commit-hooks.yaml"
README = REPO / "tools" / "README.md"
CENSUS = REPO / "experiments" / "census_extension_live_2026-08-31.json"
SLUG = "raj200501/arc-ttt-public"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (TOOL, HOOK, ACTION, HOOKS)),
    reason="the adoption files are not in this cut")

FAIL_OPEN = '''import json


def grade(completion):
    try:
        obj = json.loads(completion)
    except ValueError:
        return 0.0
    return obj["answer"]
'''

CLEAN = '''def grade(completion):
    return completion.strip() == "yes"
'''


# ------------------------------------------------------------- readers
def _yaml(path: Path):
    """Parsed YAML, or None when PyYAML is not importable."""
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _unquote(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _line_input_defaults(text: str) -> dict[str, str]:
    """`inputs:` name -> default, read line by line (the fallback)."""
    defaults: dict[str, str] = {}
    inside, current = False, None
    for line in text.splitlines():
        if re.match(r"^\S", line):
            inside = line.startswith("inputs:")
            continue
        if not inside:
            continue
        name = re.match(r"^  ([\w-]+):\s*$", line)
        if name:
            current = name.group(1)
            continue
        default = re.match(r"^    default:(.*)$", line)
        if default and current:
            defaults[current] = _unquote(default.group(1))
    return defaults


def _line_run_step(text: str) -> tuple[dict[str, str], str]:
    """(env, script) of the step that runs the tool, read line by line."""
    lines = text.splitlines()
    run_at = next(i for i, line in enumerate(lines)
                  if re.match(r"^\s+run: \|\s*$", line))
    run_indent = len(lines[run_at]) - len(lines[run_at].lstrip())
    body = []
    for line in lines[run_at + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= run_indent:
            break
        body.append(line)
    width = min(len(b) - len(b.lstrip()) for b in body if b.strip())
    script = "\n".join(b[width:] for b in body).strip() + "\n"
    env_at = max(i for i in range(run_at)
                 if re.match(r"^\s+env:\s*$", lines[i]))
    env = {}
    for line in lines[env_at + 1:run_at]:
        pair = re.match(r"^\s+([A-Z_]+):\s*(.+)$", line)
        if pair:
            env[pair.group(1)] = _unquote(pair.group(2))
    return env, script


def _hook() -> dict:
    data = _yaml(HOOKS)
    if data is not None:
        assert isinstance(data, list) and len(data) == 1, data
        return data[0]
    text = HOOKS.read_text(encoding="utf-8")
    hook = {}
    for key in ("id", "name", "entry", "language"):
        found = re.search(rf"^(?:-|\s)\s*{key}:\s*(.+)$", text, re.MULTILINE)
        assert found, f"{key} missing from {HOOKS.name}"
        hook[key] = _unquote(found.group(1))
    types = re.search(r"^\s+types:\s*\[(.*)\]\s*$", text, re.MULTILINE)
    if types:
        hook["types"] = [t.strip() for t in types.group(1).split(",")]
    return hook


def _run(*args: str, cwd: Path | None = None):
    return subprocess.run([sys.executable, str(TOOL), *args], check=False,
                          capture_output=True, text=True,
                          cwd=str(cwd or REPO), timeout=120)


def _hook_run(*args: str, cwd: Path | None = None):
    return subprocess.run([sys.executable, str(HOOK), *args], check=False,
                          capture_output=True, text=True,
                          cwd=str(cwd or REPO), timeout=120)


def _ci_section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Run it in your CI")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else len(text)]


# --------------------------------------------------------------- action
def test_the_action_declares_the_documented_inputs() -> None:
    data = _yaml(ACTION)
    text = ACTION.read_text(encoding="utf-8")
    if data is not None:
        assert data["name"] and data["description"]
        assert data["runs"]["using"] == "composite"
        defaults = {k: str(v.get("default"))
                    for k, v in data["inputs"].items()}
    else:
        assert re.search(r"^name:\s*\S", text, re.MULTILINE)
        assert re.search(r"^description:", text, re.MULTILINE)
        assert re.search(r"^runs:\n  using: composite$", text, re.MULTILINE)
        defaults = _line_input_defaults(text)
    assert defaults == {"path": ".", "args": "", "python-version": "3.11"}


def test_the_action_runs_the_tool_from_its_own_checkout() -> None:
    env, script = _line_run_step(ACTION.read_text(encoding="utf-8"))
    data = _yaml(ACTION)
    if data is not None:
        steps = [s for s in data["runs"]["steps"] if "run" in s]
        assert len(steps) == 1
        assert steps[0]["env"] == env
        assert steps[0]["run"].strip() == script.strip()
        assert steps[0]["shell"] == "bash"
    tool = env["FENCECHECK_TOOL"]
    prefix = "${{ github.action_path }}/"
    assert tool.startswith(prefix), tool
    assert (REPO / tool[len(prefix):]) == TOOL
    assert 'python "$FENCECHECK_TOOL" scan ' in script


def test_the_action_installs_nothing_but_the_interpreter() -> None:
    """The README says the action pins no Python package. The only other
    thing it may use is actions/setup-python."""
    text = ACTION.read_text(encoding="utf-8")
    assert "pip install" not in text and "requirements" not in text
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", text, re.MULTILINE)
    assert uses and all(u.startswith("actions/setup-python@") for u in uses)


def test_action_inputs_never_reach_the_shell_as_text() -> None:
    """An input expanded inside `run:` is pasted into the script before
    bash parses it, so a `$(...)` in it would execute. Inputs go through
    env only."""
    _, script = _line_run_step(ACTION.read_text(encoding="utf-8"))
    assert "${{" not in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_action_step_end_to_end(tmp_path: Path) -> None:
    """The run step exactly as written, under the shell GitHub uses for
    `shell: bash`, with `${{ }}` expressions substituted the way the
    runner substitutes them. Only actions/setup-python is not run: a
    `python` on PATH stands in for it."""
    env_spec, script = _line_run_step(ACTION.read_text(encoding="utf-8"))
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)
    entry = tmp_path / "step.sh"
    entry.write_text(script, encoding="utf-8")
    work = tmp_path / "workspace"
    (work / "my evals").mkdir(parents=True)
    (work / "my evals" / "grade.py").write_text(FAIL_OPEN, encoding="utf-8")
    (work / "clean.py").write_text(CLEAN, encoding="utf-8")

    def step(path: str, args: str = ""):
        context = {"github.action_path": str(REPO),
                   "inputs.path": path, "inputs.args": args}
        env = dict(os.environ, PATH=f"{shim_dir}{os.pathsep}{os.environ['PATH']}")
        for key, value in env_spec.items():
            env[key] = re.sub(r"\$\{\{\s*(.+?)\s*\}\}",
                              lambda m: context[m.group(1)], value)
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(entry)],
            check=False, capture_output=True, text=True, cwd=str(work), env=env,
            timeout=120)

    assert step(".").returncode == 1
    assert step("clean.py").returncode == 0
    report = step("my evals", "--json")   # a space stays inside one path
    assert report.returncode == 1, report.stderr
    assert len(json.loads(report.stdout)["findings"]) == 1
    hostile = step("$(touch PWNED)")
    assert hostile.returncode == 2
    assert not (work / "PWNED").exists()


# ------------------------------------------------------------ pre-commit
def test_the_hook_manifest_fits_the_pre_commit_schema() -> None:
    """Required keys per https://pre-commit.com/#new-hooks: id, name,
    entry, language. `script` (renamed `unsupported_script` in
    pre-commit 4.4.0, old name still accepted) runs entry as a path in
    this repository with nothing installed. `python` would pip-install
    this repository, and its core dependency is transformers."""
    hook = _hook()
    for key in ("id", "name", "entry", "language"):
        assert isinstance(hook.get(key), str) and hook[key], key
    assert hook["id"] == "fencecheck"
    assert hook["language"] in ("script", "unsupported_script")
    assert hook.get("types") == ["python"]
    assert hook.get("pass_filenames", True) is True


def test_the_hook_entry_is_an_executable_script_in_this_repo() -> None:
    """pre-commit resolves the first word of a script entry against the
    hook repository and executes it directly, so the file needs a
    shebang and the executable bit. Without the bit pre-commit reports
    'is not executable' and fails every commit for the wrong reason."""
    assert shlex.split(_hook()["entry"]) == ["tools/fencecheck_hook.py"]
    assert HOOK.read_text(encoding="utf-8").startswith(
        "#!/usr/bin/env python3\n")
    if os.name != "nt":
        assert os.access(HOOK, os.X_OK), "tools/fencecheck_hook.py lost +x"


def test_the_documented_command_is_in_the_cli() -> None:
    """The action runs `fencecheck.py scan [args] <path>`; the hook script
    loads fencecheck.py from its own directory and calls its scan."""
    result = _run("scan", "--help")
    assert result.returncode == 0
    assert re.search(r"scan \[-h\] \[--json\] path", result.stdout), result.stdout
    assert HOOK.parent == TOOL.parent
    assert 'with_name("fencecheck.py")' in HOOK.read_text(encoding="utf-8")


def test_the_hook_invocation_shape(tmp_path: Path) -> None:
    """The command line pre-commit builds: entry, then staged files.
    Two staged files with one name must print two locations."""
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "utils.py").write_text(FAIL_OPEN, encoding="utf-8")
    (tmp_path / "clean.py").write_text(CLEAN, encoding="utf-8")
    staged = ["clean.py", str(Path("a/utils.py")), str(Path("b/utils.py"))]
    result = _hook_run(*staged, cwd=tmp_path)
    assert result.returncode == 1, result.stdout
    assert "2 site(s) in 3 file(s)" in result.stdout
    for name in staged[1:]:
        assert f"  {name}:6  json.loads" in result.stdout, result.stdout
    assert _hook_run("clean.py", cwd=tmp_path).returncode == 0
    assert _hook_run("clean.py", "missing.py", cwd=tmp_path).returncode == 2
    assert _hook_run(cwd=tmp_path).returncode == 2


def test_the_hook_reports_exactly_what_the_tool_reports(
        tmp_path: Path) -> None:
    """The hook is a loop around the tool's own scan, not a second
    scanner: the same sites, line for line."""
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "grade.py").write_text(FAIL_OPEN, encoding="utf-8")
        (tmp_path / sub / "clean.py").write_text(CLEAN, encoding="utf-8")
    tool = json.loads(_run("scan", "--json", str(tmp_path)).stdout)
    expected = sorted((str(tmp_path / f["file"]), f["line"], f["call"])
                      for f in tool["findings"])
    hook = _hook_run(*(str(p) for p in sorted(tmp_path.rglob("*.py"))))
    got = sorted(re.findall(r"^  (.+):(\d+)  (\S+)$", hook.stdout, re.MULTILINE))
    assert [(f, int(n), c) for f, n, c in got] == expected
    assert len(expected) == 2


# ------------------------------------------------------------ exit codes
def test_scan_exit_codes_on_a_fail_open_site_and_a_clean_file(
        tmp_path: Path) -> None:
    """The contract both wrappers branch on, from the tool and from the
    hook: 1 on a fail-open site, 0 on a clean file, 2 on a missing path."""
    (tmp_path / "grade.py").write_text(FAIL_OPEN, encoding="utf-8")
    (tmp_path / "clean.py").write_text(CLEAN, encoding="utf-8")
    for target, code in ((tmp_path, 1), (tmp_path / "grade.py", 1),
                         (tmp_path / "clean.py", 0),
                         (tmp_path / "missing.py", 2)):
        assert _run("scan", str(target)).returncode == code, target
        assert _hook_run(str(target)).returncode == code, target


# ---------------------------------------------------------------- README
def test_the_readme_snippets_match_the_action_and_the_hook() -> None:
    section = _ci_section()
    blocks = re.findall(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert len(blocks) == 2, "expected the workflow and the pre-commit config"
    workflow, precommit = blocks

    uses = re.search(rf"uses:\s*{re.escape(SLUG)}@(\S+)", workflow)
    assert uses, "the workflow snippet does not use this repository"
    try:
        import yaml
    except ImportError:
        used_inputs = set(re.findall(r"^\s{10}([\w-]+):", workflow, re.MULTILINE))
    else:
        steps = yaml.safe_load(workflow)["jobs"]["fencecheck"]["steps"]
        ours = [s for s in steps if s.get("uses", "").startswith(SLUG + "@")]
        assert len(ours) == 1
        used_inputs = set(ours[0].get("with", {}))
        config = yaml.safe_load(precommit)
        assert config["repos"][0]["hooks"] == [{"id": _hook()["id"]}]
    action = _yaml(ACTION)
    declared = (set(action["inputs"]) if action is not None
                else set(_line_input_defaults(ACTION.read_text("utf-8"))))
    assert used_inputs and used_inputs <= declared, used_inputs

    assert re.search(rf"repo:\s*https://github\.com/{re.escape(SLUG)}\s*$",
                     precommit, re.MULTILINE)
    ids = re.findall(r"-\s*id:\s*(\S+)", precommit)
    assert ids == [_hook()["id"]]

    assert "python tools/fencecheck.py scan" in section
    for name in ("path", "args", "python-version"):
        assert f"`{name}`" in section, name


def test_the_readme_precision_numbers_are_the_artifacts() -> None:
    """The limit the README quotes is read from the census artifact, not
    remembered."""
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    assert census["raw_findings"]["total"] == 64
    assert len(census["confirmed"]) == 1
    assert census["rejected_count"] == 63
    section = " ".join(_ci_section().split())
    assert "**64** candidate sites" in section
    assert "confirmed **1**" in section
    assert "The other 63 were rejected" in section
    assert CENSUS.name in section
