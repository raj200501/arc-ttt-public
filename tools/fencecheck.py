#!/usr/bin/env python3
"""fencecheck — is your evaluation scoring a correct answer as zero?

When a language model returns a JSON object wrapped in a markdown code
fence, code that calls ``json.loads`` on it gets a parse failure. If that
failure becomes a score, a correct answer becomes a zero.

That is not a small effect. On a 30-document extraction task, a
Qwen2.5-3B produced a well-formed JSON object for every document and
scored **0.0000 with 30 of 30 unparseable**. Removing the fence and
changing nothing else: **0.8958 mean micro-F1, zero invalid**. Note the
precise claim: every output was well-formed, not every output was
exactly right — a 0.8958 mean is not thirty perfect extractions, and
saying "correct object" for both is the kind of slide this tool exists
to catch in other people's numbers.

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

# `model_text`, `sampled`, `raw_output` joined after a reviewer wrote a
# synthetic fail-open grader using `model_text` and the scan came back
# clean — a conservative miss, consistent with the precision-first
# design, and still a miss worth closing when the fix is three tokens.
# `sampled` is how openai/evals names the completion in its basic
# elsuites, so its absence here meant the tool found json_match.py only
# through the enclosing function's vocabulary, not the variable's.
MODEL_WORDS = re.compile(
    r"\b(completion|response|generation|generated|model_output|output_text"
    r"|model_text|raw_output|sampled"
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
        # When `root` IS the file, `relative_to` returns `.` and every
        # finding is reported against a path that names nothing. Scanning
        # a single file is the first thing anyone tries on a tool they
        # just downloaded, so this was the first output a reviewer saw.
        if path == root:
            shown = path.name
        else:
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


_OUTPUT_KEYS = ("prediction", "output", "completion", "response",
                "generation", "text", "raw", "answer", "content")
# Strings that are never the model's output, so the any-string fallback
# must not score them. A reviewer fed `score` a predictions file whose
# `prediction` field held the PARSED object (a dict) and got a clean
# bill: the tool skipped the dict, fell through to "any string", and
# scored the document ids. A fail-open in the fail-open detector, again.
_METADATA_KEYS = frozenset({"id", "doc_id", "document_id", "task_id",
                            "name", "model", "date", "seed", "split",
                            "arm", "mode", "config", "run", "sha256"})


def _already_parsed(record: object) -> bool:
    """True when the record carries an OUTPUT key whose value is not
    text -- a dict, list or null: an already-parsed object, not model
    text. There is nothing to classify in such a record, and nothing
    beside it is the model's output either."""
    if not isinstance(record, dict):
        return False
    present = [k for k in _OUTPUT_KEYS if k in record]
    return bool(present) and not any(
        isinstance(record[k], str) for k in present)


def _candidate_strings(record: object):
    """Any string in the record that might be the model's raw output."""
    if isinstance(record, str):
        yield record
        return
    if isinstance(record, dict):
        present = [k for k in _OUTPUT_KEYS if k in record]
        if present:
            for key in present:
                if isinstance(record[key], str):
                    yield record[key]
                    return
            return  # output keys present, none of them text: parsed
        for key, value in record.items():
            if isinstance(value, str) and key not in _METADATA_KEYS:
                yield value


def _whole_json(text: str):
    # fencecheck: ignore -- asking whether a whole FILE is one JSON
    # document, to refuse it; not classifying model output.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def score_file(path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    records: list[object] = []
    stripped = raw.strip()
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        records = loaded if isinstance(loaded, list) else [loaded]
    elif stripped.startswith("{") and (whole := _whole_json(stripped)) is not None:
        # The file is ONE JSON document. That is a legitimate input only
        # if it is itself an output record (it carries one of the keys
        # this tool reads model text from). Anything else -- an
        # experiment artifact, a config, a summary -- used to fall
        # through to the line scorer, score its own pretty-printed
        # LINES, find no fences in them, and print a clean bill with
        # exit 0. A reviewer fed this tool a banked artifact and got
        # "Nothing is being lost to a fence in this file" for a file
        # containing fenced outputs three levels deep. A fail-open in
        # the fail-open detector; it now refuses loudly instead.
        preferred = _OUTPUT_KEYS
        if isinstance(whole, dict) and any(
                isinstance(whole.get(k), str) for k in preferred):
            records = [whole]
        else:
            return {"refused": (
                "this file is a single JSON document, not a file of model "
                "outputs. score reads JSONL (one output record per line), "
                "a JSON list of records, or a single record carrying one "
                "of: " + ", ".join(preferred) + ". Point it at the "
                "predictions file, not the experiment artifact.")}
    else:
        parsed_lines = unparsed_lines = 0
        for line in stripped.split("\n"):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
                parsed_lines += 1
            except json.JSONDecodeError:
                records.append(line)
                unparsed_lines += 1
        if parsed_lines and unparsed_lines:
            return {"refused": (
                f"{parsed_lines} line(s) parse as JSON and "
                f"{unparsed_lines} do not -- that is neither JSONL nor "
                "plain-text completions, and scoring the readable subset "
                "would report a rate about a file it did not read. Fix "
                "or split the file.")}

    total = fenced = valid_asis = valid_stripped = recovered = 0
    already_parsed = 0
    examples: list[str] = []
    for record in records:
        if _already_parsed(record):
            already_parsed += 1
            continue
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
        "already_parsed_records": already_parsed,
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
    if report.get("refused"):
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("fencecheck: REFUSING to score -- " + report["refused"])
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if report["recovered_by_stripping"] else 0

    total = report["outputs"]
    if not total and report.get("already_parsed_records"):
        print(f"fencecheck: REFUSING to score -- all "
              f"{report['already_parsed_records']} record(s) carry an "
              "output field that is already a parsed object (or null), "
              "not model text. There is no fence to find in a parse "
              "tree; point score at the file that stores the RAW text.")
        return 2
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
