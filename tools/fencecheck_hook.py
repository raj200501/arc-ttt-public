#!/usr/bin/env python3
"""pre-commit entry for fencecheck: one `scan` over every staged file.

`fencecheck.py scan` takes one path, and pre-commit passes one argument
per staged file. This runs the same scan over each of them and reports
them together, with the same exit status: 1 when a site is found, 2 when
a path does not exist, 0 when every file is clean.

It loads fencecheck.py from this directory and changes nothing in it.
That keeps the scanner byte-identical, which matters because
scripts/heldout_census.py refuses a scanner whose sha256 differs from
the one its frame records. Standard library only.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

TOOL = pathlib.Path(__file__).resolve().with_name("fencecheck.py")


def _fencecheck():
    spec = importlib.util.spec_from_file_location("_fencecheck", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    given = list(sys.argv[1:] if argv is None else argv)
    if not given:
        print("usage: fencecheck_hook.py PATH [PATH ...]", file=sys.stderr)
        return 2
    fc = _fencecheck()
    findings, files = [], 0
    for name in given:
        root = pathlib.Path(name)
        if not root.exists():
            print(f"fencecheck: no such path: {root}", file=sys.stderr)
            return 2
        found, scanned = fc.scan_path(root)
        for item in found:
            # scan_path names a single-file root by its bare name, and two
            # staged files called utils.py must not print one location.
            item["file"] = str(root if root.is_file() else root / item["file"])
        findings.extend(found)
        files += scanned
    if not findings:
        print(f"fencecheck: {files} file(s) scanned, nothing exposed.")
        return 0
    print(f"fencecheck: {len(findings)} site(s) in {files} file(s) parse "
          "model output as JSON with no fence handling AND turn the "
          "failure into a zero or a silent skip.\n")
    for item in findings:
        print(f"  {item['file']}:{item['line']}  {item['call']}")
        print(f"      {item['code']}")
    print("\nEach of these scores a correct-but-fenced answer as a failure. "
          "That is only a defect if model output reaches it. Check the "
          "line and decide; mark a deliberate one `# fencecheck: ignore`.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
