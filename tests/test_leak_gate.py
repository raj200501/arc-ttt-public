"""The leak gate refuses strategy terms in a tree, keeps the CORD
provenance exemption narrow, ignores git internals, and runs on every
incremental public push, not only on fresh exports.

The gate script is private-only (it names the terms it refuses), so in
the public tree this file skips. Planted terms are built by joining
pieces, so this file does not trip the gate it tests."""
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "leak_gate.sh"
pytestmark = pytest.mark.skipif(not GATE.exists(), reason="private-only gate script")


def _gate(tree):
    return subprocess.run(["bash", str(GATE), str(tree)], capture_output=True, text=True)


def _tree(tmp_path, text, name="README.md"):
    tree = tmp_path / "pub"
    tree.mkdir(exist_ok=True)
    (tree / name).write_text(text)
    return tree


def test_clean_tree_passes(tmp_path):
    assert _gate(_tree(tmp_path, "A plain results page.\n")).returncode == 0


@pytest.mark.parametrize("term", ["inve" + "stor", "fund" + "raising", "DECK" + ".md", "per" + "sona"])
def test_a_planted_term_fails(tmp_path, term):
    result = _gate(_tree(tmp_path, f"line one\nthe {term} line\n", name="notes.json"))
    assert result.returncode == 1 and term in result.stderr


def test_the_case_sensitive_pass_and_its_provenance_exemption(tmp_path):
    assert _gate(_tree(tmp_path, "a raw output: TAHU BAKSO Grg Ayam " + "V" + "C 1\n")).returncode == 0
    assert _gate(_tree(tmp_path, "the " + "V" + "C" + " meeting went well\n")).returncode == 1
    assert _gate(_tree(tmp_path, "lowercase vc is not the token\n")).returncode == 0


def test_git_internals_are_not_scanned(tmp_path):
    tree = _tree(tmp_path, "clean\n")
    (tree / ".git").mkdir()
    (tree / ".git" / "COMMIT_EDITMSG.txt").write_text("inve" + "stor\n")
    assert _gate(tree).returncode == 0


def test_both_push_paths_run_the_gate():
    sync = (REPO / "scripts" / "check_public_sync.sh").read_text()
    export = (REPO / "scripts" / "export_public.sh").read_text()
    assert '"$SRC/scripts/leak_gate.sh" "$PUB"' in sync
    assert sync.index("leak_gate.sh\" \"$PUB\"") < sync.index("# 1) library + scripts byte-identical.")
    assert '"$SRC/scripts/leak_gate.sh" "$TARGET" || exit 1' in export
    assert "grep -v '^scripts/leak_gate\\.sh$'" in export
    assert "GATE_RE=" not in export, "the regex must live in one place"
