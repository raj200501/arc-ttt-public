#!/usr/bin/env python3
"""How much published code turns a fenced JSON answer into a zero?

Preregistered in `docs/research/FENCE_CENSUS_PREREGISTRATION.md`, which
was committed before any package here was downloaded. The frame, the
classifier, the definition of EXPOSED and the accuracy check on the
classifier itself are all frozen there. This is the executable form of
that document and it is committed before it is run.

    PYTHONPATH=src python3 scripts/fence_census.py --download
    PYTHONPATH=src python3 scripts/fence_census.py            # reuse cache

The frozen definition, restated here because a reader of the code should
not have to fetch another file to know what is being counted:

    A package is EXPOSED if it has at least one call site that parses
    model text as JSON with NO fence handling (not F) AND converts the
    resulting failure into a zero, a False, an empty object or a silent
    skip (is Z).

The conjunction is deliberate and narrow. Parsing without fence handling
is not a defect; a library entitled to clean input is entitled to assume
it. Scoring the failure is what silently turns a correct answer into a
zero.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import random
import re
import subprocess
import sys
import tarfile
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path("/tmp/fence_census")

# ---------------------------------------------------------------------
# THE SEED LIST. Written down before any inspection, per the frame's
# rule 1. Three categories: evaluation and benchmark harnesses; agent and
# application frameworks that parse structured model output; libraries
# whose stated job is turning model text into JSON.
#
# Being on this list is not an accusation. It is the universe.
# ---------------------------------------------------------------------
SEED_PACKAGES = (
    # (a) evaluation and benchmark harnesses
    "lm-eval", "evaluate", "ragas", "deepeval", "opencompass",
    "promptbench", "inspect-ai", "langsmith", "trulens-eval", "uptrain",
    "phoenix-evals", "arize-phoenix-evals", "giskard", "evalplus",
    "bigcode-eval", "agentbench", "helm", "crfm-helm", "lighteval",
    "openai-evals", "evals", "pytest-evals", "continuous-eval",
    "llm-guard", "guardrails-ai", "nemoguardrails",
    # (b) agent and application frameworks with output parsers
    "langchain", "langchain-core", "llama-index", "llama-index-core",
    "haystack-ai", "autogen-core", "pyautogen", "crewai", "semantic-kernel",
    "griffe-pydantic", "dspy", "dspy-ai", "guidance", "outlines",
    "instructor", "marvin", "mirascope", "litellm", "smolagents",
    # (c) libraries whose job is model-text to JSON
    "json-repair", "dirtyjson", "demjson3", "json5", "jsonformer",
    "lmformatenforcer", "pydantic-ai", "fuzzy-json", "partial-json-parser",
)

# Frozen PyPI search queries for frame expansion (rule 2). Every returned
# name is recorded whether or not it is kept.
SEARCH_QUERIES = ("llm evaluation", "llm eval harness", "llm output parser",
                  "json extraction llm", "benchmark language model")

# ---- inclusion test (frame rule 3) ----------------------------------
PARSE_CALLS = {"loads", "decode", "literal_eval", "safe_load", "load"}
PARSE_MODULES = {"json", "ast", "yaml", "json5", "demjson3", "dirtyjson",
                 "simplejson", "ujson", "orjson"}
MODEL_WORDS = re.compile(
    r"\b(completion|response|generation|generated|model_output|output_text"
    r"|llm|predict|prediction|answer|assistant|choices)\b", re.I)

# ---- F: fence-aware --------------------------------------------------
FENCE_MARKERS = re.compile(
    r"```|`{3}|removeprefix\s*\(|strip\s*\(\s*[\"']`|extract_json|"
    r"json_repair|dirtyjson|demjson|json5|repair_json|"
    r"_extract_(?:json|code)|strip_(?:code_)?(?:fence|block|markdown)",
    re.I)

# ---- Z: failure becomes a zero or a silent drop -----------------------
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



class SiteVisitor(ast.NodeVisitor):
    """Find JSON-parse call sites and classify their failure handling."""

    def __init__(self, source: str, path: str) -> None:
        self.source = source
        self.lines = source.split("\n")
        self.path = path
        self.sites: list[dict] = []
        self._handlers: list[ast.Try] = []
        self._funcs: list[ast.AST] = []

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _is_parse_call(node: ast.Call) -> str | None:
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

    def _enclosing_source(self) -> str:
        if not self._funcs:
            return self.source
        return self._segment(self._funcs[-1])

    @staticmethod
    def _body_is_zeroish(body: list[ast.stmt]) -> bool:
        for stmt in body:
            if isinstance(stmt, (ast.Pass, ast.Continue)):
                return True
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return True
                if isinstance(stmt.value, ast.Constant) and \
                        stmt.value.value in ZERO_LITERALS:
                    return True
                if isinstance(stmt.value, (ast.Dict, ast.List)) and \
                        not getattr(stmt.value, "elts", None) and \
                        not getattr(stmt.value, "keys", None):
                    return True
                if isinstance(stmt.value, ast.Tuple):
                    if all(isinstance(e, ast.Constant)
                           and e.value in ZERO_LITERALS
                           for e in stmt.value.elts):
                        return True
            if isinstance(stmt, ast.Assign) and isinstance(
                    stmt.value, ast.Constant) and \
                    stmt.value.value in ZERO_LITERALS:
                return True
        return False

    # -- traversal -----------------------------------------------------
    def visit_FunctionDef(self, node):          # noqa: N802
        self._funcs.append(node)
        self.generic_visit(node)
        self._funcs.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Try(self, node):                  # noqa: N802
        self._handlers.append(node)
        for child in node.body:
            self.visit(child)
        self._handlers.pop()
        for handler in node.handlers:
            for child in handler.body:
                self.visit(child)
        for child in node.orelse + node.finalbody:
            self.visit(child)

    def visit_Call(self, node):                 # noqa: N802
        name = self._is_parse_call(node)
        if name:
            enclosing = self._enclosing_source()
            fence_aware = bool(FENCE_MARKERS.search(enclosing))
            zeroish = False
            handled = False
            for guard in self._handlers:
                for handler in guard.handlers:
                    handled = True
                    if self._body_is_zeroish(handler.body):
                        zeroish = True
            reads_a_file = _argument_is_a_file_read(node, self._segment(node))
            self.sites.append({
                "file": self.path,
                "line": node.lineno,
                "call": name,
                "fence_aware": fence_aware,
                "handled": handled,
                "zero_on_failure": zeroish,
                "reads_a_file": reads_a_file,
                "exposed": (not fence_aware) and zeroish and not reads_a_file,
                "evidence": self._segment(node).strip()[:160],
            })
        self.generic_visit(node)


def analyse_source(source: str, path: str) -> list[dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = SiteVisitor(source, path)
    visitor.visit(tree)
    return visitor.sites


def python_files(archive: pathlib.Path):
    """Yield (name, source) for every .py file in an sdist or wheel."""
    try:
        if archive.suffix == ".whl" or archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.filename.endswith(".py") and \
                            info.file_size < 2_000_000:
                        yield info.filename, zf.read(info).decode(
                            "utf-8", "replace")
            return
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.name.endswith(".py") and member.size < 2_000_000:
                    handle = tf.extractfile(member)
                    if handle is None:
                        continue
                    yield member.name, handle.read().decode("utf-8", "replace")
    except (tarfile.TarError, zipfile.BadZipFile, EOFError):
        return


def download(name: str, out: pathlib.Path) -> pathlib.Path | None:
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*"))
    if existing:
        return existing[0]
    for extra in (["--no-binary", ":all:"], []):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps",
             *extra, "-d", str(out), name],
            capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            files = sorted(out.glob("*"))
            if files:
                return files[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=20260825,
                        help="fixed so the adjudication sample is replayable")
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "fence_census.json"))
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []

    for name in SEED_PACKAGES:
        target = CACHE / name
        archive = (download(name, target) if args.download
                   else (sorted(target.glob("*")) or [None])[0])
        if archive is None:
            excluded.append({"package": name,
                             "reason": "could not be downloaded from PyPI"})
            continue

        sites, scanned = [], 0
        for path, source in python_files(archive):
            if not MODEL_WORDS.search(source):
                continue
            scanned += 1
            sites.extend(analyse_source(source, path))

        if not sites:
            excluded.append({
                "package": name,
                "version": archive.name,
                "reason": ("no JSON-parse call site inside a module that "
                           "references model output — outside the universe "
                           "by the frame's inclusion test"),
                "modules_referencing_model_output": scanned,
            })
            continue

        exposed = [s for s in sites if s["exposed"]]
        rows.append({
            "package": name,
            "artifact": archive.name,
            "modules_referencing_model_output": scanned,
            "parse_sites": len(sites),
            "sites_not_fence_aware": sum(1 for s in sites
                                         if not s["fence_aware"]),
            "sites_zero_on_failure": sum(1 for s in sites
                                         if s["zero_on_failure"]),
            "exposed": bool(exposed),
            "exposed_sites": exposed[:12],
            "all_sites": sites[:200],
        })
        print(f"{name:26s} sites={len(sites):4d}  exposed="
              f"{'YES' if exposed else 'no ':3s}  ({len(exposed)} site(s))",
              flush=True)

    included = len(rows)
    exposed_count = sum(1 for r in rows if r["exposed"])

    # ---- the frozen adjudication sample -----------------------------
    rng = random.Random(args.seed)
    pool_exposed = [(r["package"], s) for r in rows for s in r["all_sites"]
                    if s["exposed"]]
    pool_clean = [(r["package"], s) for r in rows for s in r["all_sites"]
                  if not s["exposed"]]
    sample = (rng.sample(pool_exposed, min(5, len(pool_exposed)))
              + rng.sample(pool_clean, min(5, len(pool_clean))))

    record = {
        "what": "Published Python packages that score or parse language-"
                "model output, classified for whether a fenced JSON answer "
                "becomes a zero.",
        "preregistered": "docs/research/FENCE_CENSUS_PREREGISTRATION.md, "
                         "committed before any package was downloaded.",
        "definition_of_exposed": (
            "At least one call site that parses model text as JSON with NO "
            "fence handling AND converts the failure into a zero, False, an "
            "empty container or a silent skip. The conjunction is "
            "deliberate: parsing without fence handling is not by itself a "
            "defect."),
        "frame": {
            "universe": "PyPI packages whose documented purpose includes "
                        "evaluating, scoring, parsing or validating language-"
                        "model output",
            "seed_packages": list(SEED_PACKAGES),
            "search_queries_frozen": list(SEARCH_QUERIES),
            "inclusion_test": "at least one JSON-parse call site inside a "
                              "module that references model output",
        },
        "counts": {
            "seed_listed": len(SEED_PACKAGES),
            "included": included,
            "excluded": len(excluded),
            "exposed": exposed_count,
        },
        "headline": (
            f"Of {included} published Python packages that score or parse "
            f"language-model output, {exposed_count} have at least one path "
            f"that reads a model's JSON without handling a markdown code "
            f"fence and converts the resulting parse failure into a zero or "
            f"a silent drop."),
        "what_this_does_not_claim": (
            "It does not claim any package is buggy — a library entitled to "
            "clean input is entitled to assume it. It does not claim any "
            "published benchmark number is wrong; this measures code, not "
            "papers, and that step is not taken. It does not measure how "
            "often models actually fence, which is reported separately and "
            "is not multiplied by this."),
        "adjudication_sample": [
            {"package": pkg, "site": site, "hand_verdict": "PENDING"}
            for pkg, site in sample],
        "adjudication_note": (
            "A mechanical classifier over unfamiliar code makes mistakes and "
            "they are not symmetric. These ten sites are read by hand and "
            "the agreement rate is published beside the count. Below 80% "
            "agreement, no count publishes at all."),
        "packages": rows,
        "excluded": excluded,
        "snapshot_note": "Latest release as of the run date, pinned per "
                         "package. A package fixed after that date is still "
                         "counted as exposed at the version inspected, and "
                         "the fix may already exist. This is a snapshot, not "
                         "an accusation.",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"\n{exposed_count}/{included} included packages EXPOSED "
          f"({len(excluded)} excluded and listed)")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
