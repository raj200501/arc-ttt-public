"""fencecheck as a PyPI distribution: tools/pyproject.toml, checked offline.

`pip install fencecheck` installs tools/fencecheck.py and
tools/fencecheck_hook.py as two console scripts. This file pins what can
go wrong in that packaging, without building anything: the metadata
parses, each console script
names a real function that can be called with no arguments, the version
setuptools will stamp is the version `fencecheck --version` prints, the
Python floor is the one the README documents, the licence file is the
repository's, every number in the PyPI README is one tools/README.md
prints, and the release workflow can only be started by hand.

Building is left to the release workflow and to a person: a build here
would download setuptools and write into the tree.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
PYPROJECT = TOOLS / "pyproject.toml"
WORKFLOW = REPO / ".github" / "workflows" / "release-fencecheck.yml"
PUBLIC = "https://github.com/raj200501/arc-ttt-public"

pytestmark = pytest.mark.skipif(not PYPROJECT.exists(),
                                reason="the packaging files are not in this cut")


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")   # Python 3.11+
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _load(module: str):
    """Import tools/<module>.py by path, under a private name."""
    spec = importlib.util.spec_from_file_location(
        f"_packaging_{module}", TOOLS / f"{module}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)*", text))


# ------------------------------------------------------------- metadata
def test_the_distribution_is_fencecheck_and_depends_on_nothing() -> None:
    """The root pyproject.toml depends on transformers. This one must not
    depend on anything: the pitch is one standard-library file."""
    data = _pyproject()
    project = data["project"]
    assert project["name"] == "fencecheck"
    assert project.get("dependencies", []) == []
    assert "optional-dependencies" not in project
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"


def test_every_console_script_is_a_real_function() -> None:
    """A console script whose target is missing installs cleanly and
    fails on first run, on the user's machine. Each one must name a
    module the wheel ships and a function callable with no arguments,
    which is how the generated wrapper calls it."""
    data = _pyproject()
    scripts = data["project"]["scripts"]
    assert set(scripts) == {"fencecheck", "fencecheck-hook"}
    shipped = data["tool"]["setuptools"]["py-modules"]
    for name, target in scripts.items():
        module, _, attr = target.partition(":")
        assert attr, f"{name}: {target!r} names no function"
        assert module in shipped, f"{name}: {module} is not in py-modules"
        assert (TOOLS / f"{module}.py").is_file(), f"{name}: no {module}.py"
        func = getattr(_load(module), attr, None)
        assert inspect.isfunction(func), f"{name}: {target} is not a function"
        inspect.signature(func).bind()   # raises if it needs arguments
    assert sorted(shipped) == sorted(
        t.partition(":")[0] for t in scripts.values()), (
        "py-modules ships a module no console script uses")


def test_the_hook_finds_the_tool_beside_it_once_installed() -> None:
    """The wheel puts both modules side by side in site-packages, and the
    hook loads fencecheck.py from its own directory. That only works
    while both stay top-level modules of the same distribution."""
    hook = _load("fencecheck_hook")
    assert hook.TOOL == (TOOLS / "fencecheck.py").resolve()
    assert hook.TOOL.parent == Path(hook.__file__).resolve().parent


# -------------------------------------------------------------- version
def test_the_packaged_version_is_the_one_the_tool_prints() -> None:
    """setuptools reads `__version__` from fencecheck.py without
    importing it, so it must stay a literal string assignment, and it
    must be what `fencecheck --version` prints."""
    data = _pyproject()
    assert "version" not in data["project"], "a static version would drift"
    assert "version" in data["project"]["dynamic"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "fencecheck.__version__"}

    tree = ast.parse((TOOLS / "fencecheck.py").read_text(encoding="utf-8"))
    literal = [node.value.value for node in tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "__version__"
                       for t in node.targets)
               and isinstance(node.value, ast.Constant)
               and isinstance(node.value.value, str)]
    assert len(literal) == 1, "__version__ is not one literal string"
    version = literal[0]
    assert re.fullmatch(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?",
                        version), f"{version!r} is not a plain PEP 440 version"
    assert _load("fencecheck").__version__ == version

    printed = subprocess.run(
        [sys.executable, str(TOOLS / "fencecheck.py"), "--version"],
        capture_output=True, text=True, timeout=120, check=False)
    assert printed.returncode == 0, printed.stderr
    assert printed.stdout.strip() == f"fencecheck {version}"


def test_the_python_floor_is_the_documented_one() -> None:
    """tools/README.md and the tool's docstring say Python 3.9+. The
    metadata must say the same, and no classifier may claim a version
    below it. 3.8 is out because ast.unparse does not exist there."""
    data = _pyproject()
    assert data["project"]["requires-python"] == ">=3.9"
    assert "Python 3.9+" in (TOOLS / "README.md").read_text(encoding="utf-8")
    assert "Python 3.9+" in (TOOLS / "fencecheck.py").read_text(encoding="utf-8")
    claimed = [tuple(int(p) for p in m.group(1).split("."))
               for c in data["project"]["classifiers"]
               for m in [re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", c)]
               if m]
    assert claimed and min(claimed) == (3, 9)


# -------------------------------------------------- licence and README
def test_the_licence_is_the_repository_licence() -> None:
    """PEP 639 licence files must sit beside pyproject.toml, so tools/
    carries a copy of the root LICENSE. The copy must not drift."""
    data = _pyproject()
    assert data["project"]["license"] == "MIT"
    assert data["project"]["license-files"] == ["LICENSE"]
    assert (TOOLS / "LICENSE").read_bytes() == (REPO / "LICENSE").read_bytes()
    assert (REPO / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_the_pypi_readme_keeps_every_number_and_link_honest() -> None:
    """The PyPI page is a trimmed copy of tools/README.md. It may drop a
    number; it may not print one tools/README.md does not. Its links are
    absolute, so each must name a file this tree has, and the public
    export carries; and the PyPI page is public, so it must not point at
    the private strategy folder."""
    data = _pyproject()
    readme = data["project"]["readme"]
    assert readme["content-type"] == "text/markdown"
    text = (TOOLS / readme["file"]).read_text(encoding="utf-8")
    source = (TOOLS / "README.md").read_text(encoding="utf-8")

    extra = sorted(_numbers(text) - _numbers(source))
    assert not extra, f"numbers on the PyPI page that tools/README.md lacks: {extra}"

    urls = list(data["project"]["urls"].values())
    assert urls and all(u == PUBLIC or u.startswith(PUBLIC + "/") for u in urls)
    links = re.findall(rf"{re.escape(PUBLIC)}/(?:blob|tree)/main/([^)\s#]+)",
                       text + "\n" + "\n".join(urls))
    assert links
    missing = sorted({p for p in links if not (REPO / p).exists()})
    assert not missing, f"links to paths this tree does not have: {missing}"
    assert "docs/strategy" not in text


# ------------------------------------------------------ release workflow
@pytest.mark.skipif(not WORKFLOW.exists(), reason="no release workflow in this cut")
def test_the_release_workflow_only_runs_by_hand() -> None:
    """A release must never fire on push, tag or schedule. Upload goes
    through PyPI trusted publishing: no stored token, and the publish
    action pinned to a commit. Whether the upload step is switched on is
    the owner's call, so it is deliberately not pinned here."""
    text = WORKFLOW.read_text(encoding="utf-8")
    block = re.search(r"^on:\n((?:[ \t]+.*\n|[ \t]*\n)*)", text, re.MULTILINE)
    assert block, "no `on:` block"
    assert re.findall(r"^  ([\w-]+):", block.group(1), re.MULTILINE) == [
        "workflow_dispatch"]
    assert "secrets." not in text and "password" not in text
    assert re.search(r"^\s+id-token:\s*write\s*$", text, re.MULTILINE)
    assert re.search(r"uses:\s*pypa/gh-action-pypi-publish@[0-9a-f]{40}\b", text)
    assert "python -m build --outdir dist tools/" in text
