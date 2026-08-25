#!/usr/bin/env python3
"""fencecheck — is your evaluation scoring a correct answer as zero?

When a language model returns a JSON object wrapped in a markdown code
fence, code that calls ``json.loads`` on it gets a parse failure. If that
failure becomes a score, a correct answer becomes a zero.

That is not a small effect. On a 30-document extraction task, a
Qwen2.5-3B produced a correct object for every document and scored
**0.0000 with 30 of 30 unparseable**. Removing the fence and changing
nothing else: **0.8958, zero invalid**.

Two commands, no installation, Python 3.9+, standard library only:

    python3 fencecheck.py scan  <path-to-your-repo>
    python3 fencecheck.py score <predictions.jsonl>

``scan`` reads your code and reports every place that parses model output
as JSON without handling a fence *and* turns the failure into a zero or a
silent skip. ``score`` reads your saved model outputs and tells you how
many are fenced-but-valid, and what your numbers would be without the
fence.

Exit status is 1 when something is found, so this drops into CI.

WHAT IT DOES NOT SAY. A finding is not a bug report. A library entitled
to clean input is entitled to assume it; parsing without fence handling
is only a defect when the failure becomes a score. Every finding prints
its file and line so you can decide in ten seconds.

MIT licensed. Copy it into your repo if that is easier than depending on
it.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

__version__ = "0.1.0"

PARSE_CALLS = {"loads", "decode", "literal_eval", "safe_load", "load"}
PARSE_MODULES = {"json", "ast", "yaml", "json5", "demjson3", "dirtyjson",
                 "simplejson", "ujson", "orjson"}

MODEL_WORDS = re.compile(
    r"\b(completion|response|generation|generated|model_output|output_text"
    r"|llm|predict|prediction|answer|assistant|choices)\b", re.I)

FENCE_MARKERS = re.compile(
    r"```|`{3}|removeprefix\s*\(|strip\s*\(\s*[\"']`|extract_json|"
    r"json_repair|dirtyjson|demjson|json5|repair_json|"
    r"_extract_(?:json|code)|strip_(?:code_)?(?:fence|block|markdown)",
    re.I)

ZERO_LITERALS = {0, 0.0, False, None}

# A parse whose argument is a file read is reading config or an artifact,
# not a model's answer. Without this, any module that both mentions a
# model and loads a JSON file is flagged, and precision is what makes a
# census number defensible. Added before any package was inspected, so
# no result influenced it; recorded as a dated amendment in the
# preregistration.
FILE_READ = re.compile(
    r"\b(?:read_text|read_bytes|\.read\s*\(|open\s*\(|Path\s*\(|"
    r"pathlib\.|\bf\.read\b|fh\.read|handle\.read|infile|fp\.read)")


def _argument_is_a_file_read(node, segment: str) -> bool:
    """True when this parse is consuming a file rather than model output."""
    if not getattr(node, "args", None):
        return False
    first = node.args[0]
    try:
        text = ast.unparse(first)
    except Exception:                                    # noqa: BLE001
        text = segment
    return bool(FILE_READ.search(text))


# An opt-out, because every linter needs one and because this tool flagged
# ITSELF on first run: `_parses` below deliberately asks "does this parse
# exactly as written?", so strictness is its whole job. Put
# `# fencecheck: ignore` on the parse line or anywhere in the enclosing
# function. A tool that cannot be told "yes, on purpose" gets deleted
# after the first false positive.
IGNORE_MARKER = re.compile(r"#\s*fencecheck:\s*ignore", re.I)

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
             ".tox", ".mypy_cache", ".pytest_cache", "build", "dist",
             "site-packages"}


# ---------------------------------------------------------------- scan
class _Visitor(ast.NodeVisitor):
    def __init__(self, source: str, path: str) -> None:
        self.lines = source.split("\n")
        self.source = source
        self.path = path
        self.sites: list[dict] = []
        self._tries: list[ast.Try] = []
        self._funcs: list[ast.AST] = []

    @staticmethod
    def _parse_call(node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in PARSE_CALLS:
            base = func.value
            name = (base.id if isinstance(base, ast.Name)
                    else getattr(base, "attr", None))
            if name in PARSE_MODULES:
                return f"{name}.{func.attr}"
            if func.attr == "decode" and name and "decoder" in name.lower():
                return f"{name}.decode"
        return None

    def _segment(self, node: ast.AST) -> str:
        start = max(0, getattr(node, "lineno", 1) - 1)
        end = min(len(self.lines), getattr(node, "end_lineno", start + 1))
        return "\n".join(self.lines[start:end])

    @staticmethod
    def _zeroish(body: list) -> bool:
        for stmt in body:
            if isinstance(stmt, (ast.Pass, ast.Continue)):
                return True
            if isinstance(stmt, ast.Return):
                value = stmt.value
                if value is None:
                    return True
                if isinstance(value, ast.Constant) and \
                        value.value in ZERO_LITERALS:
                    return True
                if isinstance(value, (ast.Dict, ast.List)) and \
                        not getattr(value, "elts", None) and \
                        not getattr(value, "keys", None):
                    return True
                if isinstance(value, ast.Tuple) and value.elts and all(
                        isinstance(e, ast.Constant) and e.value in ZERO_LITERALS
                        for e in value.elts):
                    return True
            if isinstance(stmt, ast.Assign) and isinstance(
                    stmt.value, ast.Constant) and \
                    stmt.value.value in ZERO_LITERALS:
                return True
        return False

    def visit_FunctionDef(self, node):          # noqa: N802
        self._funcs.append(node)
        self.generic_visit(node)
        self._funcs.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Try(self, node):                  # noqa: N802
        self._tries.append(node)
        for child in node.body:
            self.visit(child)
        self._tries.pop()
        for handler in node.handlers:
            for child in handler.body:
                self.visit(child)
        for child in node.orelse + node.finalbody:
            self.visit(child)

    def visit_Call(self, node):                 # noqa: N802
        name = self._parse_call(node)
        if name:
            enclosing = (self._segment(self._funcs[-1]) if self._funcs
                         else self.source)
            fence_aware = bool(FENCE_MARKERS.search(enclosing))
            zeroish = any(self._zeroish(h.body)
                          for t in self._tries for h in t.handlers)
            reads_a_file = _argument_is_a_file_read(node, self._segment(node))
            line = self.lines[node.lineno - 1] if node.lineno <= len(
                self.lines) else ""
            suppressed = bool(IGNORE_MARKER.search(line)
                              or IGNORE_MARKER.search(enclosing))
            if (not fence_aware) and zeroish and not reads_a_file \
                    and not suppressed:
                self.sites.append({
                    "file": self.path,
                    "line": node.lineno,
                    "call": name,
                    "code": self._segment(node).strip().split("\n")[0][:100],
                })
        self.generic_visit(node)


def scan_source(source: str, path: str) -> list[dict]:
    """Findings for one file. Empty list if the file is not in scope."""
    if not MODEL_WORDS.search(source):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _Visitor(source, path)
    visitor.visit(tree)
    return visitor.sites


def scan_path(root: pathlib.Path) -> tuple[list[dict], int]:
    findings, files = [], 0
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for path in paths:
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files += 1
        try:
            shown = path.relative_to(root)
        except ValueError:
            shown = path
        findings.extend(scan_source(source, str(shown)))
    return findings, files


# --------------------------------------------------------------- score
def strip_fence(text: str) -> tuple[str, bool]:
    """Remove one markdown fence. Returns (text, was_fenced)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped, False
    body = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped[3:]
    if "```" in body:
        body = body.rsplit("```", 1)[0]
    return body.strip(), True


def _candidate_strings(record: object):
    """Any string in the record that might be the model's raw output."""
    if isinstance(record, str):
        yield record
        return
    if isinstance(record, dict):
        preferred = ("prediction", "output", "completion", "response",
                     "generation", "text", "raw", "answer", "content")
        for key in preferred:
            value = record.get(key)
            if isinstance(value, str):
                yield value
                return
        for value in record.values():
            if isinstance(value, str):
                yield value


def score_file(path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    records: list[object] = []
    stripped = raw.strip()
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        records = loaded if isinstance(loaded, list) else [loaded]
    else:
        for line in stripped.split("\n"):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(line)

    total = fenced = valid_asis = valid_stripped = recovered = 0
    examples: list[str] = []
    for record in records:
        for text in _candidate_strings(record):
            total += 1
            cleaned, was_fenced = strip_fence(text)
            fenced += was_fenced
            asis_ok = _parses(text)
            stripped_ok = _parses(cleaned)
            valid_asis += asis_ok
            valid_stripped += stripped_ok
            if stripped_ok and not asis_ok:
                recovered += 1
                if len(examples) < 3:
                    examples.append(text.strip()[:120])
            break
    return {
        "outputs": total,
        "fenced": fenced,
        "parse_as_written": valid_asis,
        "parse_after_stripping": valid_stripped,
        "recovered_by_stripping": recovered,
        "examples": examples,
    }


def _parses(text: str) -> bool:
    # fencecheck: ignore -- asking "does this parse exactly as written?"
    # is this function's entire job, so strictness here is deliberate.
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fencecheck",
        description="Find evaluation code that scores a fenced JSON answer "
                    "as zero.")
    parser.add_argument("--version", action="version",
                        version=f"fencecheck {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="read code and report exposed sites")
    scan.add_argument("path")
    scan.add_argument("--json", action="store_true", help="machine-readable")

    score = sub.add_parser(
        "score", help="read saved model outputs and report the fence tax")
    score.add_argument("path")
    score.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    target = pathlib.Path(args.path).expanduser()
    if not target.exists():
        print(f"fencecheck: no such path: {target}", file=sys.stderr)
        return 2

    if args.command == "scan":
        findings, files = scan_path(target)
        if args.json:
            print(json.dumps({"findings": findings,
                              "files_scanned": files}, indent=2))
        elif not findings:
            print(f"fencecheck: {files} file(s) scanned, nothing exposed.")
            print("Every site that parses model output either handles a "
                  "fence or lets the failure reach the caller.")
        else:
            print(f"fencecheck: {len(findings)} site(s) in {files} file(s) "
                  "parse model output as JSON with no fence handling AND "
                  "turn the failure into a zero or a silent skip.\n")
            for item in findings:
                print(f"  {item['file']}:{item['line']}  {item['call']}")
                print(f"      {item['code']}")
            print("\nEach of these scores a correct-but-fenced answer as a "
                  "failure. That is only a defect if model output reaches "
                  "it — check the line and decide.")
        return 1 if findings else 0

    report = score_file(target)
    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if report["recovered_by_stripping"] else 0

    total = report["outputs"]
    if not total:
        print("fencecheck: no model outputs found in that file.")
        print("Expected JSONL or a JSON list, with the raw model text under "
              "one of: prediction, output, completion, response, generation, "
              "text, raw, answer, content.")
        return 2
    print(f"fencecheck: {total} output(s)")
    print(f"  fenced                  {report['fenced']}")
    print(f"  parse as written        {report['parse_as_written']}")
    print(f"  parse after stripping   {report['parse_after_stripping']}")
    recovered = report["recovered_by_stripping"]
    if not recovered:
        print("\nNothing is being lost to a fence in this file.")
        return 0
    print(f"\n  {recovered} of {total} outputs are VALID JSON that your "
          "scorer would reject.")
    print(f"  That is {recovered / total:.0%} of this file scoring zero for "
          "formatting rather than for content.")
    for example in report["examples"]:
        print(f"      {example}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
