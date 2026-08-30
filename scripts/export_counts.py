#!/usr/bin/env python3
"""Measure the PUBLIC EXPORT, because it is a different tree.

    python3 scripts/export_counts.py --tree /tmp/pubcut

`export_public.sh` deliberately drops files -- the application-gate test
and its artifact, the strategy documents -- so the exported tree collects
FEWER tests than the source tree. Two counts, two trees, one noun.

Nothing measured that, and `sync_test_counts.py` rewrites every
documented test count to the SOURCE tree's number. So when the source
suite grew from 305 to 326 to 333, the fixer walked into ten sentences
that describe the public repository and rewrote each one to the source
count. The public repo collects 312. Every one of those ten sentences
became false, in our favour, automatically, in documents whose entire
premise is that a stranger can check them -- and one did: an outside
reviewer cloned the public repo, ran `pytest --collect-only`, and read
back a number 21 higher than the truth.

The fix has two halves and both are needed. `sync_test_counts.py` now
REFUSES to rewrite a count in a sentence naming the export. This is the
other half: the export's real counts, measured from the export and
banked, so copy about the public repository has an artifact to be
reconciled against instead of a number somebody remembered.

The banked record carries the tree's git SHA and the measurement date,
because unlike the source count this one cannot be re-derived on demand
from the working tree -- if the export has not been rebuilt, the honest
thing to report is a dated measurement of a named commit, not a fresh
number that was never taken.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "public_export_counts.json"


def collected(tree: pathlib.Path) -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=tree, timeout=900)
    match = re.search(r"(\d+)\s+tests? collected", result.stdout)
    if not match:
        raise SystemExit(
            "could not read a collection count from the export at "
            f"{tree}:\n{result.stdout[-2000:]}")
    return int(match.group(1))


def head_sha(tree: pathlib.Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree,
                            capture_output=True, text=True)
    return result.stdout.strip() or "(not a git tree)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default="/tmp/pubcut",
                        help="a built public export (see export_public.sh)")
    parser.add_argument("--date", required=True,
                        help="ISO date of this measurement; required "
                             "because the number is only meaningful with "
                             "one, and guessing it from the clock would "
                             "make a stale rebuild look fresh")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    tree = pathlib.Path(args.tree)
    if not (tree / "tests").is_dir():
        raise SystemExit(
            f"no export at {tree}. Build one with scripts/export_public.sh "
            "first — this script must measure the exported tree itself, "
            "not infer its counts from the source tree, because inferring "
            "them from the source tree is the defect it exists to fix.")

    tests = collected(tree)
    artifacts = len(list((tree / "experiments").glob("*.json")))
    source_tests = collected(REPO)

    record = {
        "what": "Test and artifact counts measured in a BUILT PUBLIC "
                "EXPORT, which is a different tree from the source and "
                "carries different numbers.",
        "public_export_tests": tests,
        "public_export_artifacts": artifacts,
        "source_tree_tests_at_same_moment": source_tests,
        "difference": source_tests - tests,
        "why_they_differ": (
            "export_public.sh excludes the application-gate test and its "
            "artifact along with the strategy documents. The gap is the "
            "excluded files and nothing else; if it ever differs from the "
            "count of files that script drops, something else is being "
            "lost on the way out and that is worth knowing."),
        "measured_from": str(tree),
        "export_head": head_sha(tree),
        "measured_on": args.date,
        "how_to_check": (
            "git clone https://github.com/raj200501/arc-ttt-public && "
            "cd arc-ttt-public && python3 -m pytest tests/ -q "
            "--collect-only | tail -1"),
        "why_this_is_banked": (
            "Ten sentences across seven documents stated the public "
            "repository's test count as the SOURCE tree's number, because "
            "sync_test_counts.py rewrote them automatically and had no "
            "notion that some sentences describe a different tree. Every "
            "one was wrong in our favour. An outside reviewer found it by "
            "cloning the public repo. Copy about the export now has an "
            "artifact to reconcile against."),
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"public export: {tests} tests, {artifacts} artifacts "
          f"(source tree: {source_tests} tests, {source_tests - tests} more)")
    print(f"banked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
