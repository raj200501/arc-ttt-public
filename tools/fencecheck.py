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
import io
import json
import pathlib
import re
import sys
import tokenize

__version__ = "0.1.0"

PARSE_CALLS = {"loads", "decode", "literal_eval", "safe_load", "load"}
PARSE_MODULES = {"json", "ast", "yaml", "json5", "demjson3", "dirtyjson",
                 "simplejson", "ujson", "orjson"}
# Methods that turn JSON text into a typed object: pydantic v2
# (`Model.model_validate_json`, `TypeAdapter(...).validate_json`) and v1
# (`Model.parse_raw`). A fenced answer fails them exactly as it fails
# json.loads.
PARSE_METHODS = {"model_validate_json", "validate_json", "parse_raw"}

# ---- whose text is this? -------------------------------------------------
# A finding needs evidence that the parsed text is a MODEL's output, read
# off the argument itself and the values it was built from: its variable
# names, the keys and attributes it was pulled out of, the calls that
# produced it, and (one hop) what in-module callers pass for a parameter.
# Module-wide vocabulary alone is not evidence. The census found that a
# module which merely mentions a model also parses tracing attributes,
# SSE frames, RPC messages, config and cache files, and every one of those
# came back as a finding (1 confirmed of 64 in the live census).
#
# Identifiers are split into words (snake_case and camelCase), so
# `raw_completion`, `modelResponse` and `completions` all read as what they
# are; the old `\bcompletion\b` test missed all three.

# Words that name a language model's output.
_STRONG = frozenset({
    "completion", "completions", "generation", "generations", "generated",
    "sampled", "llm", "llms", "assistant", "choices", "prediction",
    "predictions", "predicted"})
# Word pairs that do the same: model_output, raw_completion, output_text.
_STRONG_PAIRS = {
    "model": frozenset({"output", "outputs", "text", "response", "responses",
                        "answer", "answers", "reply", "completion",
                        "content"}),
    "raw": frozenset({"output", "outputs", "completion", "completions",
                      "generation"}),
    "output": frozenset({"text"}),
}
# Calls whose result is a model's output.
_MODEL_CALLS = frozenset({
    "generate", "agenerate", "complete", "acomplete", "completion",
    "completions", "acompletion", "chat", "achat", "invoke", "ainvoke",
    "predict", "apredict", "llm", "judge"})
# Words that usually name model output in evaluation code, but also name
# HTTP responses, process output and protocol messages. Enough on their
# own only in a module that is about models or evaluation, and never
# enough when the value is traced back to a carrier (below).
_WEAK = frozenset({
    "response", "responses", "output", "outputs", "answer", "answers",
    "reply", "replies", "message", "messages", "msg", "msgs", "pred",
    "preds", "verdict", "judgment", "judgement"})
# Words that say the text is somebody else's, wherever they appear on the
# path: a gold answer, configuration, a command line, an environment
# variable, a schema, a telemetry record, an error message, a tool call's
# structured arguments.
_GOLD = frozenset({
    "gold", "golden", "ideal", "expected", "reference", "references", "ref",
    "refs", "truth", "groundtruth", "gt", "label", "labels", "correct"})
_NOT_MODEL = _GOLD | frozenset({
    "prompt", "prompts", "question", "questions",
    "config", "configs", "cfg", "conf", "settings", "setting", "env",
    "environ", "getenv", "argv", "args", "kwargs", "arguments", "argument",
    "cli", "params", "param", "options", "opts", "secret", "secrets",
    "credential", "credentials", "password", "schema", "schemas",
    "span", "spans", "otel", "opentelemetry", "telemetry", "tracing",
    "tracer", "trace", "traces", "attributes", "attribute", "attrs",
    "metadata", "header", "headers", "error", "errors", "err", "exc",
    "exception", "exceptions", "traceback", "tool", "tools"})
# Carriers: the text came out of a file, a socket, an HTTP exchange, a
# process, a stream frame, a database row, a cache, or a parse that
# already succeeded. A carrier ends the trace as "not model output"
# unless a key or attribute between the carrier and the parse names model
# output (`resp.json()["choices"][0]["message"]["content"]`).
_CARRIER = frozenset({
    "file", "files", "path", "paths", "fp", "fh", "fd", "f", "handle",
    "read", "readline", "readlines", "stdin", "stdout", "stderr",
    "subprocess", "proc", "popen", "requests", "request", "req", "httpx",
    "aiohttp", "urllib", "urlopen", "http", "url", "urls", "socket", "sock",
    "recv", "websocket", "websockets", "ws", "line", "lines", "chunk",
    "chunks", "frame", "frames", "body", "payload", "sse", "row", "rows",
    "db", "database", "sql", "cursor", "store", "redis", "cache", "cached",
    "plaintext", "decrypt", "cookie", "cookies", "dump", "dumps",
    "serialize", "serialized", "partial", "delta", "__parsed__"})
# A traced variable is judged by what it was assigned from, except when its
# head noun is one of these, which a local name does not take by accident
# (`response_stdout`, `row_json`, `completion_path`).
_NAME_CARRIER = frozenset({
    "stdout", "stderr", "stdin", "socket", "sse", "frame", "frames", "chunk",
    "chunks", "line", "lines", "header", "headers", "cookie", "cookies",
    "url", "urls", "path", "paths", "file", "files", "fp", "fh", "fd", "row",
    "rows", "cache", "cached", "request", "req"})
# A dataset record: a weak key read out of one (`sample["input_output"]`,
# `doc["answer"]`) is a dataset field. A strong key still counts
# (`sample["completion"]`).
_RECORD = frozenset({"sample", "samples", "doc", "docs", "example",
                     "examples", "instance", "instances", "problem",
                     "problems", "datum"})
# Message words name model text only when its content is read:
# `msg.content`, `result["message"]["content"]`.
_MESSAGE_WORDS = frozenset({"message", "messages", "msg", "msgs"})
# A module whose path names one of these is not "about models or
# evaluation" for weak evidence (`caching/response_cache.py`,
# `tracing/otel/...`).
_NON_EVAL_PATH = frozenset({
    "cache", "cached", "caching", "tracing", "tracer", "trace", "traces",
    "otel", "opentelemetry", "telemetry", "span", "spans", "instrumentation",
    "exporter", "exporters", "logging", "config", "configs", "settings",
    "secret", "secrets", "credential", "credentials", "cli", "transport",
    "rpc", "server", "gateway", "proxy", "socket", "websocket", "websockets",
    "sse", "http", "httpx", "requests", "db", "database", "sql", "redis",
    "store"})
# Attributes only an HTTP response has: a variable read this way in its
# function is an HTTP response, whatever it is called.
_HTTP_ATTRS = frozenset({"status_code", "raise_for_status", "iter_lines",
                         "iter_content", "aiter_lines", "elapsed", "cookies"})
# A value that went through something named for fence handling.
# `__span__` marks text cut out by a brace/bracket slice or regex first.
_FENCE_WORDS = frozenset({"fence", "fences", "fenced", "unfence",
                          "markdown", "codeblock", "backtick", "backticks",
                          "__span__"})
# A module is "about models or evaluation" if it says so somewhere. Only
# consulted for _WEAK evidence.
_CONTEXT = _STRONG | frozenset({
    "response", "responses", "answer", "answers", "output", "outputs",
    "reply", "replies", "predict", "score",
    "scores", "scorer", "scorers", "scoring", "grade", "grader", "graders",
    "grading", "judge", "judges", "metric", "metrics", "eval", "evals",
    "evaluate", "evaluation", "evaluator", "benchmark", "benchmarks",
    "accuracy", "openai", "anthropic", "langchain", "transformers", "vllm",
    "ollama", "gemini", "claude", "gpt"})

# Kept for callers that imported it: the old module-level vocabulary.
MODEL_WORDS = re.compile(
    r"(?<![A-Za-z0-9])(completions?|responses?|generations?|generated"
    r"|model_outputs?|output_text|model_text|raw_outputs?|sampled"
    r"|llms?|predict|predictions?|answers?|assistant|choices)(?![A-Za-z0-9])",
    re.I)

# Fence handling by name: a helper, a repair library, or a variable called
# for the fence. Three backticks on their own are judged separately, below.
FENCE_MARKERS = re.compile(
    r"removeprefix\s*\(|extract_json|"
    r"json_repair|dirtyjson|demjson|json5|repair_json|"
    r"_extract_(?:json|code)|strip_(?:code_)?(?:fence|block|markdown)|"
    r"fence(?!check)|markdown|code_?block",      # not this tool's own marker
    re.I)
# A backtick literal handles a ```json fence only if it names the json tag,
# matches any tag (`\w`, `[a-z]`, `[^\n]*\n`), or the code drops the whole
# opening line. `"```python"` alone, `.replace("```", "")` and
# `.strip("`")` all leave a ```json fence (or its "json" tag) in the text:
# LiveBench's tablejoin scorer handled ```python only and scored a
# ```json-fenced correct answer 0.0 for ten months.
_FENCE_LITERAL = re.compile(r"```|`\{3\}")
_ANY_TAG = re.compile(
    r"(?:```|`\{3\})(?:[^`]{0,6}?(?:json|\\w|\\S|\[\^|\[a-z|\[A-Z)"
    r"|\.\*\??\\n|\[\^\\n\]\*)", re.I)
_LINE_DROP = re.compile(
    r"(?:split|rsplit|partition|index|find)\s*\(\s*[\"']\\n|splitlines\s*\(")
_BACKTICK_STRIP = re.compile(r"strip\s*\(\s*[\"']`")
_TAG_STRIP = re.compile(
    r"(?:removeprefix|strip|replace|startswith)\s*\(\s*[\"']json", re.I)


def _fence_aware(text: str, node=None) -> bool:
    """Does this code handle a fence around a JSON answer?"""
    if FENCE_MARKERS.search(text):
        return True
    if node is not None:
        literals = [n.value for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and _FENCE_LITERAL.search(n.value)]
    else:
        literals = [text] if _FENCE_LITERAL.search(text) else []
    if not literals:
        return bool(_BACKTICK_STRIP.search(text) and _TAG_STRIP.search(text))
    return bool(any(_ANY_TAG.search(s) for s in literals)
                or _LINE_DROP.search(text) or _TAG_STRIP.search(text))

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

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _words(text: str) -> list[str]:
    """`raw_completion` -> [raw, completion]; `modelResponse` -> [model,
    response]; `APIResponse` -> [api, response]."""
    out = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        if chunk:
            out.extend(w.lower() for w in _CAMEL.split(chunk) if w)
    return out


def _is_test_path(rel: pathlib.PurePath) -> bool:
    """Test files are not shipped behaviour and cannot mis-score anything
    (the census's second named cause of false positives)."""
    parts = rel.parts
    if any(p in ("tests", "test", "testing") for p in parts[:-1]):
        return True
    name = parts[-1] if parts else ""
    return (name.startswith("test_") or name.endswith("_test.py")
            or name == "conftest.py")


# ------------------------------------------------- handler semantics
# Only the handler that actually catches the parse error decides whether
# the failure is swallowed. The old rule asked whether ANY handler of ANY
# enclosing try looked like a zero, so `except json.JSONDecodeError:
# raise ParserError(...)` inside `try: ... except KeyError: return None`
# was reported as fail-open.
_PARSE_ERRORS = {"ValueError", "JSONDecodeError", "DecodeError",
                 "SyntaxError", "YAMLError", "ScannerError", "ParserError",
                 "ValidationError"}
_PARSE_ERROR_HINT = re.compile(r"json|decod|pars|valid|syntax|yaml", re.I)
_BROAD = {"Exception", "BaseException"}
_RERAISE = object()
_SCOREISH = re.compile(
    r"score|value|passed|pass|correct|accuracy|reward|success|valid|match"
    r"|result|grade|verdict|label", re.I)
_APPENDERS = {"append", "add", "insert", "put", "put_nowait", "extend",
              "appendleft"}


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _handler_names(handler: ast.ExceptHandler) -> list[str]:
    kinds = handler.type
    elts = kinds.elts if isinstance(kinds, ast.Tuple) else [kinds]
    return [_dotted(e).rsplit(".", 1)[-1] for e in elts if e is not None]


def _catches(handler: ast.ExceptHandler, pending, bases=None) -> bool:
    """Does this handler catch the parse error (pending None) or the class
    `pending` that a handler further in raised? `bases` maps classes
    defined in the module to their base-class names, so `class
    FormatError(ValueError)` is caught by `except ValueError`."""
    if handler.type is None:
        return True
    names = _handler_names(handler)
    if any(n in _BROAD for n in names):
        return True
    if pending is not None:
        family = {pending} | _ancestors(pending, bases or {})
        return any(n in family for n in names)
    return any(n in _PARSE_ERRORS or _PARSE_ERROR_HINT.search(n)
               for n in names)


def _ancestors(name: str, bases: dict) -> set:
    out, todo = set(), [name]
    while todo:
        for parent in bases.get(todo.pop(), ()):
            if parent not in out:
                out.add(parent)
                todo.append(parent)
    return out


def _raises(handler: ast.ExceptHandler):
    """None if the handler does not raise unconditionally; _RERAISE if it
    re-raises the same error; else the name of the class it raises."""
    for stmt in handler.body:
        if isinstance(stmt, ast.Raise):
            exc = stmt.exc
            if exc is None or (isinstance(exc, ast.Name)
                               and exc.id == handler.name):
                return _RERAISE
            target = exc.func if isinstance(exc, ast.Call) else exc
            return _dotted(target).rsplit(".", 1)[-1] or "?"
    return None


def _falsy(node) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in ZERO_LITERALS or node.value in ("", b"")
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        return _dotted(node.func) in ("dict", "list", "set", "tuple", "str")
    return False


def _zero_number(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is not None and \
        node.value in (0, 0.0, False) and not isinstance(node.value, str)


def _score_call(node) -> bool:
    """`Score(value=0)`, `Score(score=0)`, `Score(value=INCORRECT)`,
    `record_match(False, ...)`: a call that records a zero, a False, a
    negative verdict or a failed match."""
    if not isinstance(node, ast.Call):
        return False
    if any(kw.arg and _SCOREISH.search(kw.arg) and (
            _zero_number(kw.value) or _sentinel(kw.value))
           for kw in node.keywords):
        return True
    return bool(_SCOREISH.search(_dotted(node.func))) and any(
        _zero_number(a) or _sentinel(a) for a in node.args)


# A string that records a failed, negative or neutral verdict: judge
# helpers return "no", ("incorrect", 0, "") or "tie" when the reply does not
# parse, and scorers return Score(value=INCORRECT).
_NEGATIVE_SENTINELS = {"no", "false", "incorrect", "wrong", "fail", "failed",
                       "failure", "invalid", "tie", "error", "unknown"}


def _sentinel(node) -> bool:
    if isinstance(node, ast.Name) and node.id.isupper():
        return node.id.lower() in _NEGATIVE_SENTINELS
    return isinstance(node, ast.Constant) and isinstance(node.value, str) \
        and node.value.strip().lower() in _NEGATIVE_SENTINELS


def _zero_value(value) -> bool:
    """A value that stands for "this scored nothing"."""
    if value is None or _falsy(value) or _score_call(value) or \
            _sentinel(value):
        return True
    if isinstance(value, (ast.Tuple, ast.List)) and any(
            _falsy(e) or _sentinel(e) for e in value.elts):
        return True
    return isinstance(value, ast.Dict) and any(
        v is not None and (_zero_number(v) or _sentinel(v))
        for v in value.values)


def _zeroish(body: list) -> bool:
    for stmt in body:
        if isinstance(stmt, (ast.Pass, ast.Continue)):
            return True
        if isinstance(stmt, ast.Return) and _zero_value(stmt.value):
            return True
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and \
                stmt.value is not None and _zero_value(stmt.value):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr in _APPENDERS \
                    and any(_falsy(a) for a in call.args):
                return True
            if _score_call(call):
                return True
        if isinstance(stmt, ast.If) and (_zeroish(stmt.body)
                                         or _zeroish(stmt.orelse)):
            return True
    return False


_ESCAPES = "escapes"


def _handling(tries: list, pending=None, bases=None):
    """Walk outward from the innermost try; the first handler that
    catches the error decides. A handler that raises something else
    passes the decision outward, to a handler that catches THAT.

    Returns (handler, try) for the deciding handler, or (_ESCAPES,
    pending) when the error leaves the function: pending is None for the
    parse error itself, else the name of the class it was re-raised as."""
    for t in reversed(tries):
        handler = next((h for h in t.handlers
                        if _catches(h, pending, bases)), None)
        if handler is None:
            continue
        raised = _raises(handler)
        if raised is None:
            return handler, t
        if raised is not _RERAISE:
            pending = raised
    return _ESCAPES, pending


def _only_passes(body: list) -> bool:
    """`except ...: pass`, possibly beside a log call: the failure falls
    through to whatever the code does next."""
    return any(isinstance(s, ast.Pass) for s in body) and all(
        isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(
            s.value, (ast.Call, ast.Constant))) for s in body)


# ------------------------------------------------- provenance
_MODEL, _WEAK_ONLY, _OTHER, _HANDLED, _UNKNOWN = (
    "model", "weak", "other", "fence-handled", "unknown")
_RANK = {_HANDLED: 4, _MODEL: 3, _WEAK_ONLY: 2, _OTHER: 1, _UNKNOWN: 0}
_TRANSPARENT_FUNCS = {"str", "repr", "bytes", "StringIO", "BytesIO", "dedent",
                      "reversed", "sorted", "list", "tuple", "iter", "next",
                      "cast"}
_TRANSPARENT_METHODS = {"strip", "lstrip", "rstrip", "decode", "encode",
                        "lower", "upper", "replace", "removeprefix",
                        "removesuffix", "group", "groups", "pop",
                        "splitlines", "split", "copy", "format"}
_KEY_METHODS = {"get", "pop", "setdefault"}
_REGEX_FUNCS = {"search", "match", "fullmatch", "findall", "finditer", "sub"}


def _classify_spine(tokens: list) -> tuple[str, str]:
    """tokens: (words, kind) from the parse argument outward. kind is
    `name` (a variable that was traced further), `leaf` (a variable with
    nothing further to trace), `nav` (a key or attribute the value was
    pulled out of), `call` (a call that produced it) or `fname` (the
    function whose parameter it is). Returns (class, evidence word)."""
    for words, _kind in tokens:
        hit = next((w for w in words if w in _FENCE_WORDS), None)
        if hit:
            # A {...} span cut out by a regex is found INSIDE a fence, so a
            # fenced answer takes that path whatever the fallback is.
            return _HANDLED, hit
    # A function's name can say what data it handles (`load_config`,
    # `_parse_int_attribute`), but a gold or verdict word in it says what
    # it computes (`is_correct`, `match_expected`), not where its
    # parameter came from.
    for words, kind in tokens:
        banned = _NOT_MODEL - _GOLD if kind == "fname" else _NOT_MODEL
        hit = next((w for w in words if w in banned), None)
        if hit:
            return _OTHER, hit
    weak = weak_nav = None
    content_read = False
    for words, kind in tokens:
        strong = _STRONG | _MODEL_CALLS if kind == "call" else _STRONG
        head = len(words) - 1
        for i in range(head, -1, -1):               # head word first
            w = words[i]
            if w in strong:
                return _MODEL, w
            if i > 0 and w in _STRONG_PAIRS.get(words[i - 1], ()):
                return _MODEL, f"{words[i - 1]}_{w}"
            # A traced variable's own name does not decide, what it was
            # assigned from does, unless its head noun is a transport word.
            # `open` counts only as the called head word, so `OpenAI()` is
            # not a file.
            if kind in ("name", "leaf") and i == head and w in _RECORD \
                    and weak_nav:
                return _OTHER, w
            if kind == "name":
                carrier = i == head and w in _NAME_CARRIER
            else:
                carrier = kind != "fname" and (
                    w in _CARRIER or (kind == "call" and i == head
                                      and w == "open"))
            if carrier:
                if weak_nav:
                    return _WEAK_ONLY, weak_nav
                return _OTHER, w.strip("_")
            # Function names are generic (`read_messages`,
            # `handle_response`): only a strong word in one counts.
            if w in _WEAK and kind != "fname" and (
                    w not in _MESSAGE_WORDS or content_read):
                weak = weak or w
                if kind == "nav":
                    weak_nav = weak_nav or w
        if kind == "nav" and {"content", "text"} & set(words):
            content_read = True
    return (_WEAK_ONLY, weak) if weak else (_UNKNOWN, "")


def _span_pattern(pattern: str) -> bool:
    """A regex that cuts out a {...} or [...] span, or a fenced block: a
    fence cannot reach a parse of what it returns."""
    if "```" in pattern or "`{3}" in pattern:
        return True
    if re.search(r"\\[{\[]", pattern) or re.search(r"\[[^\]]*[{\[]", pattern):
        return True
    # an unescaped `{` that opens a JSON span: `{.*}`, `{[^}]*}`, `{(...)}`,
    # not LaTeX `\text{` and not a `{2,3}` quantifier
    return bool(re.search(r"(?<![\\A-Za-z])\{(?:\.|\[\^|\()", pattern))


def _brace_constants(node) -> bool:
    return any(isinstance(n, ast.Constant) and n.value in ("{", "}", "[", "]")
               for n in ast.walk(node))


class _Flow:
    """Intra-module data flow, just deep enough to say whose text a parse
    argument holds: local assignments, loop and `with` targets, in-module
    callers of a parameter (one hop), and in-module helpers' returns."""

    MAX_SPINES = 48
    MAX_DEPTH = 10

    def __init__(self, tree: ast.AST, aliases: dict, lines: list) -> None:
        self.aliases = aliases
        self.lines = lines
        self.parent: dict = {}
        self.scopes: dict = {}
        self.functions: dict = {}
        self.calls: dict = {}
        self.bases: dict = {}
        self.via: dict = {}
        self.site = None
        self.http: set = set()
        self.returns: dict = {}
        self._scope: dict = {}
        scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        for node in ast.walk(tree):            # parents before children
            inner = node if isinstance(node, scopes) else \
                self._scope.get(node)
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
                self._scope[child] = inner
        self._index(tree)

    # ---- indexing
    def _scope_of(self, node):
        """The function (or module) whose body holds `node`."""
        return self._scope.get(node)

    def _bind(self, scope, name, binding) -> None:
        self.scopes.setdefault(scope, {}).setdefault(name, []).append(binding)

    def _bind_target(self, scope, target, kind, expr, path=()) -> None:
        if isinstance(target, ast.Name):
            self._bind(scope, target.id, (kind, expr, path))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(target.elts):
                self._bind_target(scope, elt, kind, expr, path + (i,))
        elif isinstance(target, ast.Starred):
            self._bind_target(scope, target.value, kind, expr, path)

    def _index(self, tree) -> None:
        for node in ast.walk(tree):
            scope = self._scope_of(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, []).append(node)
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs:
                    self._bind(node, arg.arg, ("param", node, ()))
            elif isinstance(node, ast.ClassDef):
                self.bases[node.name] = [_dotted(b).rsplit(".", 1)[-1]
                                         for b in node.bases]
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    self._bind_target(scope, target, "value", node.value)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and \
                    node.value is not None:
                self._bind_target(scope, node.target, "value", node.value)
            elif isinstance(node, ast.NamedExpr):
                self._bind_target(scope, node.target, "value", node.value)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                self._bind_target(scope, node.target, "iter", node.iter)
            elif isinstance(node, ast.comprehension):
                self._bind_target(scope, node.target, "iter", node.iter)
            elif isinstance(node, ast.withitem) and node.optional_vars:
                self._bind_target(scope, node.optional_vars, "value",
                                  node.context_expr)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self._bind(scope, node.name, ("except", None, ()))
            elif isinstance(node, ast.Return) and node.value is not None:
                self.returns.setdefault(scope, []).append(node.value)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None)
                if name:
                    self.calls.setdefault(name, []).append(node)
            if isinstance(node, ast.Attribute) and node.attr in _HTTP_ATTRS \
                    and isinstance(node.value, ast.Name):
                self.http.add((id(scope), node.value.id))

    def call_sites(self, func) -> list:
        """In-module calls of `func`: `f(...)` for a function, and
        `self.f(...)` / `cls.f(...)` / `Class.f(...)` for a method, so an
        unrelated `client.parse(...)` is not mistaken for a caller."""
        owner = self.parent.get(func)
        out = []
        for call in self.calls.get(func.name, []):
            f = call.func
            if isinstance(owner, ast.ClassDef):
                if isinstance(f, ast.Attribute) and isinstance(
                        f.value, ast.Name) and f.value.id in (
                        "self", "cls", owner.name):
                    out.append(call)
            elif isinstance(f, ast.Name):
                out.append(call)
        return out

    def text_of(self, node) -> str:
        if isinstance(node, ast.Module) or node is None:
            return "\n".join(self.lines)
        start = max(0, node.lineno - 1)
        return "\n".join(self.lines[start:node.end_lineno])

    def tries_at(self, node) -> list:
        """The try statements, inside the same function, whose BODY holds
        `node`, outermost first."""
        tries, child, cur = [], node, self.parent.get(node)
        while cur is not None and not isinstance(
                cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module,
                      ast.Lambda, ast.ClassDef)):
            if isinstance(cur, ast.Try) or type(cur).__name__ == "TryStar":
                if any(child is s for s in cur.body):
                    tries.insert(0, cur)
            child, cur = cur, self.parent.get(cur)
        return tries

    def following(self, stmt) -> list:
        """Statements after `stmt` in the block that holds it."""
        parent = self.parent.get(stmt)
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if isinstance(block, list):
                for i, s in enumerate(block):
                    if s is stmt:
                        return block[i + 1:]
        return []

    def following_up(self, stmt) -> list:
        """Everything that runs after `stmt` falls through: the rest of its
        block, then the rest of each enclosing block, up to the function."""
        out, cur = [], stmt
        while cur is not None and not isinstance(
                cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module,
                      ast.ClassDef, ast.Lambda)):
            out.extend(self.following(cur))
            cur = self.parent.get(cur)
            if isinstance(cur, ast.ExceptHandler):
                cur = self.parent.get(cur)
        return out

    # ---- tracing
    def classify(self, expr, at, via=None) -> tuple[str, str]:
        """via: {id(function): call} limits one function's callers to the
        call that matters (the one that turns the failure into a zero)."""
        self.via = via or {}
        self.site = at
        spines: list = []
        self._walk(expr, self._scope_of(at), [], spines, 0, set())
        best, evidence = _UNKNOWN, ""
        for tokens in spines:
            cls, why = _classify_spine(tokens)
            if _RANK[cls] > _RANK[best]:
                best, evidence = cls, why
        return best, evidence.strip("_")

    def _emit(self, spines, tokens) -> None:
        if len(spines) < self.MAX_SPINES:
            spines.append(tokens)

    def _walk(self, expr, scope, tokens, spines, depth, seen) -> None:
        if depth > self.MAX_DEPTH or len(spines) >= self.MAX_SPINES:
            self._emit(spines, tokens)
            return
        nxt = depth + 1
        if isinstance(expr, ast.Name):
            if expr.id in ("self", "cls"):
                self._emit(spines, tokens)
                return
            self._name(expr.id, scope, tokens, spines, nxt, seen)
        elif isinstance(expr, ast.Attribute):
            self._walk(expr.value, scope,
                       tokens + [(_words(expr.attr), "nav")], spines, nxt, seen)
        elif isinstance(expr, ast.Subscript):
            key = expr.slice
            if isinstance(key, ast.Slice) and self._brace_slice(key, scope):
                # text[text.find("{"):text.rfind("}") + 1]
                return self._emit(spines, tokens + [(["__span__"], "call")])
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                tokens = tokens + [(_words(key.value), "nav")]
            self._walk(expr.value, scope, tokens, spines, nxt, seen)
        elif isinstance(expr, ast.Call):
            self._call(expr, scope, tokens, spines, nxt, seen)
        elif isinstance(expr, (ast.Await, ast.Starred, ast.NamedExpr)):
            self._walk(expr.value, scope, tokens, spines, nxt, seen)
        elif isinstance(expr, ast.IfExp):
            self._walk(expr.body, scope, tokens, spines, nxt, seen)
            self._walk(expr.orelse, scope, tokens, spines, nxt, seen)
        elif isinstance(expr, ast.BoolOp):
            for value in expr.values:
                self._walk(value, scope, tokens, spines, nxt, seen)
        elif isinstance(expr, ast.BinOp):
            self._walk(expr.left, scope, tokens, spines, nxt, seen)
            self._walk(expr.right, scope, tokens, spines, nxt, seen)
        elif isinstance(expr, (ast.List, ast.Set)) and expr.elts:
            # `matches = [s]` ... `matches[-1]`: an element is the value
            for elt in expr.elts[:4]:
                self._walk(elt, scope, tokens, spines, nxt, seen)
        else:
            self._emit(spines, tokens)

    def _brace_slice(self, key, scope) -> bool:
        if _brace_constants(key):
            return True
        for n in ast.walk(key):              # start = text.find("{")
            if isinstance(n, ast.Name):
                _, bindings = self._lookup(n.id, scope)
                if any(kind == "value" and _brace_constants(expr)
                       for kind, expr, _ in bindings):
                    return True
        return False

    def _lookup(self, name, scope):
        cur = scope
        while cur is not None:
            found = self.scopes.get(cur, {}).get(name)
            if found:
                return cur, found
            cur = self._scope_of(cur) if not isinstance(cur, ast.Module) \
                else None
        return None, []

    def _name(self, name, scope, tokens, spines, depth, seen) -> None:
        owner, bindings = self._lookup(name, scope)
        words = _words(name)
        if (id(scope), name) in self.http or (id(owner), name) in self.http:
            self._emit(spines, tokens + [(words, "name"), (["http"], "leaf")])
            return
        key = (id(owner), name)
        if not bindings or key in seen:
            self._emit(spines, tokens + [(words, "leaf")])
            return
        seen = seen | {key}
        traced = tokens + [(words, "name")]
        for kind, expr, path in bindings:
            if kind in ("value", "iter") and self.site is not None and any(
                    n is self.site for n in ast.walk(expr)):
                continue        # `x = json.loads(x)`: the site's own result
            if kind == "except":
                self._emit(spines, traced + [(["exception"], "leaf")])
            elif kind == "param":
                self._param(expr, name, tokens, words, spines, depth, seen)
            elif kind == "iter":
                self._element(expr, path, owner, traced, spines, depth, seen)
            elif path and isinstance(expr, (ast.Tuple, ast.List)):
                if path[0] < len(expr.elts):
                    self._walk(expr.elts[path[0]], owner, traced, spines,
                               depth, seen)
            elif path and isinstance(expr, (ast.Call, ast.Await)):
                # `review, model = get_eval(...)`: element 0 of what the
                # helper returns
                call = expr.value if isinstance(expr, ast.Await) else expr
                if isinstance(call, ast.Call):
                    self._call(call, owner, traced, spines, depth, seen,
                               elt=path[0])
                else:
                    self._walk(expr, owner, traced, spines, depth, seen)
            else:
                self._walk(expr, owner, traced, spines, depth, seen)

    def _element(self, it, path, scope, tokens, spines, depth, seen) -> None:
        """An element of `it`, at tuple position `path` in the target."""
        if isinstance(it, ast.Call) and path:
            fname = _dotted(it.func).rsplit(".", 1)[-1]
            if fname == "zip" and path[0] < len(it.args):
                return self._walk(it.args[path[0]], scope, tokens, spines,
                                  depth, seen)
            if fname == "enumerate" and it.args:
                if path[0] == 0:
                    return self._emit(spines, tokens + [(["index"], "name")])
                return self._walk(it.args[0], scope, tokens, spines, depth,
                                  seen)
            if fname == "items" and isinstance(it.func, ast.Attribute):
                if path[0] == 0:
                    return self._emit(spines, tokens + [(["key"], "name")])
                return self._walk(it.func.value, scope, tokens, spines,
                                  depth, seen)
        self._walk(it, scope, tokens, spines, depth, seen)

    def _param(self, func, name, tokens, words, spines, depth, seen) -> None:
        """A parameter: its own name and its function's name are evidence
        (`def parse_completion(text)`), then what in-module callers pass.
        The function name counts as a noun only: the parameter of
        `generate(prompt)` is its input, not its output. A caller whose
        own function handles a fence hands over handled text."""
        a = func.args
        positional = [x.arg for x in a.posonlyargs + a.args]
        is_method = isinstance(self.parent.get(func), ast.ClassDef)
        if id(func) in self.via:
            sites = [self.via.pop(id(func))]
        else:
            sites = self.call_sites(func)
        fname = (_words(func.name), "fname")
        traced = tokens + [(words, "name"), fname]
        before = len(spines)
        for call in sites:
            caller = self._scope_of(call)
            if caller is func:
                continue                   # recursion adds nothing new
            arg = None
            for kw in call.keywords:
                if kw.arg == name:
                    arg = kw.value
            if arg is None and name in positional:
                idx = positional.index(name)
                if is_method and isinstance(call.func, ast.Attribute) and \
                        positional and positional[0] in ("self", "cls"):
                    idx -= 1
                if 0 <= idx < len(call.args) and not any(
                        isinstance(x, ast.Starred) for x in call.args[:idx + 1]):
                    arg = call.args[idx]
            if arg is None:
                continue
            if not isinstance(caller, ast.Module) and \
                    _fence_aware(self.text_of(caller), caller):
                self._emit(spines, traced + [(["fence"], "call")])
                continue
            self._walk(arg, caller, traced, spines, depth, seen)
        if len(spines) == before:
            self._emit(spines, tokens + [(words, "leaf"), fname])

    def _regex_receiver_pattern(self, recv, scope, plain=False):
        """`PATTERN.search(text)` where PATTERN = re.compile(r"..."), or
        (plain) `re.findall(pattern, text)` where pattern = r"..."."""
        if isinstance(recv, ast.Name):
            _, bindings = self._lookup(recv.id, scope)
            for kind, expr, _ in bindings:
                if plain and kind == "value" and isinstance(
                        expr, ast.Constant) and isinstance(expr.value, str):
                    return expr.value
                if kind == "value" and isinstance(expr, ast.Call) and \
                        _dotted(expr.func).rsplit(".", 1)[-1] == "compile" \
                        and expr.args and isinstance(expr.args[0], ast.Constant) \
                        and isinstance(expr.args[0].value, str):
                    return expr.args[0].value
        return None

    def _call(self, call, scope, tokens, spines, depth, seen,
              elt=None) -> None:
        func = call.func
        if _parse_call_name(call, self.aliases):
            return self._emit(spines, tokens + [(["__parsed__"], "call")])
        if isinstance(func, ast.Attribute):
            method = func.attr
            recv = func.value
            if method in _REGEX_FUNCS:
                if _dotted(recv) in ("re", "regex"):
                    pattern = call.args[0] if call.args else None
                    pos = 2 if method == "sub" else 1
                    target = call.args[pos] if len(call.args) > pos else None
                    if isinstance(pattern, ast.Name):   # pattern = r"\[..."
                        pattern = self._regex_receiver_pattern(
                            pattern, scope, plain=True)
                    elif isinstance(pattern, ast.Constant):
                        pattern = pattern.value
                    else:
                        pattern = None
                else:
                    pattern = self._regex_receiver_pattern(recv, scope)
                    target = call.args[0] if call.args else None
                # An anchored brace regex (`match`, `fullmatch`, a leading
                # `^`) needs the text to START with `{`, so a fenced reply
                # fails to match; only an unanchored search cuts the
                # object out from around a fence.
                anchored = method in ("match", "fullmatch") or (
                    isinstance(pattern, str)
                    and pattern.lstrip().startswith(("^", "\\A")))
                if isinstance(pattern, str) and _span_pattern(pattern) and (
                        "```" in pattern or "`{3}" in pattern
                        or not anchored):
                    return self._emit(spines,
                                      tokens + [(["__span__"], "call")])
                if target is not None and (pattern is not None or _dotted(
                        recv) in ("re", "regex")):
                    # `re.findall("%s(.*)%s" % (...), s)`: the pattern
                    # is built at run time, the text searched is still s.
                    return self._walk(target, scope, tokens, spines, depth,
                                      seen)
            if method == "join" and call.args:
                return self._walk(call.args[0], scope, tokens, spines, depth,
                                  seen)
            if method in _KEY_METHODS and call.args and isinstance(
                    call.args[0], ast.Constant) and isinstance(
                    call.args[0].value, str):
                tokens = tokens + [(_words(call.args[0].value), "nav")]
            elif method in ("StringIO", "BytesIO") and call.args:
                return self._walk(call.args[0], scope, tokens, spines, depth,
                                  seen)
            elif method not in _TRANSPARENT_METHODS and \
                    method not in _KEY_METHODS:
                tokens = tokens + [(_words(method), "call")]
            return self._walk(recv, scope, tokens, spines, depth, seen)
        if isinstance(func, ast.Name):
            fname = func.id
            if fname == "getattr" and len(call.args) >= 2:
                attr = call.args[1]
                if isinstance(attr, ast.Constant) and isinstance(attr.value,
                                                                 str):
                    tokens = tokens + [(_words(attr.value), "nav")]
                return self._walk(call.args[0], scope, tokens, spines, depth,
                                  seen)
            if fname in _TRANSPARENT_FUNCS:
                pos = 1 if fname == "cast" else 0
                if len(call.args) > pos:
                    return self._walk(call.args[pos], scope, tokens, spines,
                                      depth, seen)
                return self._emit(spines, tokens)
            words = _words(fname)
            tokens = tokens + [(words, "call")]
            if set(words) & (_STRONG | _MODEL_CALLS):
                # `completion_fn(prompt)`: the call made the text; what it
                # was given is the prompt, not the answer.
                return self._emit(spines, tokens)
            helpers = [f for f in self.functions.get(fname, [])
                       if ("ret", id(f)) not in seen]
            if helpers:
                for helper in helpers:
                    for value in self.returns.get(helper, [])[:6]:
                        if elt is not None:
                            if not isinstance(value, (ast.Tuple, ast.List)) \
                                    or elt >= len(value.elts):
                                continue
                            value = value.elts[elt]
                        self._walk(value, helper, tokens, spines, depth,
                                   seen | {("ret", id(helper))})
                return
            if call.args:
                return self._walk(call.args[0], scope, tokens, spines, depth,
                                  seen)
        self._emit(spines, tokens)


def _imports(tree) -> dict:
    """alias -> what it names: `import json as js` gives js -> json;
    `from json import loads` gives loads -> json.loads."""
    aliases: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def _parse_call_name(node: ast.Call, aliases: dict) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in PARSE_METHODS:
            base = _dotted(func.value) or "<obj>"
            return f"{base.rsplit('.', 1)[-1]}.{func.attr}"
        if func.attr in PARSE_CALLS:
            base = func.value
            name = (base.id if isinstance(base, ast.Name)
                    else getattr(base, "attr", None))
            module = aliases.get(name, name) if name else None
            if module in PARSE_MODULES:
                return f"{module}.{func.attr}"
            if func.attr == "decode" and name and "decoder" in name.lower():
                return f"{name}.decode"
    elif isinstance(func, ast.Name):
        target = aliases.get(func.id, "")
        module, _, attr = target.rpartition(".")
        if module in PARSE_MODULES and attr in PARSE_CALLS:
            return f"{module}.{attr}"
    return None


_RETRY_WORDS = re.compile(
    r"pars|extract|repair|search|findall|match|load|decode", re.I)
_LOG_CALL = re.compile(
    r"(^|\.)(print|log|debug|info|warning|warn|error|exception|critical)$"
    r"|logger|logging", re.I)


# ---------------------------------------------------------------- scan
class _Visitor(ast.NodeVisitor):
    def __init__(self, source: str, path: str, tree: ast.AST) -> None:
        self.lines = source.split("\n")
        self.source = source
        self.path = path
        self.tree = tree
        self.sites: list[dict] = []
        self.suppressed: list[dict] = []
        self._marker_lines = None
        self._tries: list[list[ast.Try]] = [[]]
        self._funcs: list[ast.AST] = []
        self.aliases = _imports(tree)
        words = set(_words(source))
        place = set(_words(path))
        self.context = (bool(words & _CONTEXT) or any(
            words & seconds for first, seconds in _STRONG_PAIRS.items()
            if first in words)) and not place & _NON_EVAL_PATH
        self._flow = None

    @property
    def flow(self) -> _Flow:
        if self._flow is None:
            self._flow = _Flow(self.tree, self.aliases, self.lines)
        return self._flow

    def _parse_call(self, node: ast.Call) -> str | None:
        return _parse_call_name(node, self.aliases)

    def _segment(self, node: ast.AST) -> str:
        start = max(0, getattr(node, "lineno", 1) - 1)
        end = min(len(self.lines), getattr(node, "end_lineno", start + 1))
        return "\n".join(self.lines[start:end])

    # Kept as a method for callers of the old visitor.
    _zeroish = staticmethod(_zeroish)

    def visit_FunctionDef(self, node):          # noqa: N802
        self._funcs.append(node)
        self._tries.append([])                  # a nested def is not
        self.generic_visit(node)                # protected by the outer try
        self._tries.pop()
        self._funcs.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Try(self, node):                  # noqa: N802
        self._tries[-1].append(node)
        for child in node.body:
            self.visit(child)
        self._tries[-1].pop()
        for handler in node.handlers:
            for child in handler.body:
                self.visit(child)
        for child in node.orelse + node.finalbody:
            self.visit(child)

    visit_TryStar = visit_Try                   # noqa: N815

    def visit_Call(self, node):                 # noqa: N802
        name = self._parse_call(node)
        if name:
            self._consider(node, name)
        self.generic_visit(node)

    def _falls_back(self, call: ast.Call, try_node: ast.Try) -> bool:
        """`except ...: pass` is a silent skip only if nothing after it
        recovers. It is a fallback when the parse overwrote its own input
        (`value = json.loads(value)`: the raw text is kept), or when the
        code after the try tries another parser or extraction, or returns
        the raw input."""
        arg = call.args[0] if call.args else None
        arg_text = ast.unparse(arg) if arg is not None else None
        root = arg
        while isinstance(root, (ast.Attribute, ast.Subscript, ast.Call)):
            root = root.func.value if isinstance(root, ast.Call) and \
                isinstance(root.func, ast.Attribute) else (
                root.value if not isinstance(root, ast.Call) else None)
        root = root.id if isinstance(root, ast.Name) else None
        targets = set()
        for stmt in try_node.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign) and node.value is call:
                    names = {ast.unparse(t) for t in node.targets}
                    if arg_text in names:
                        return True          # `x = json.loads(x)` keeps x
                    targets |= names
        # A default set before the try survives the failure:
        # `ans_list = [ans_str]` then `try: ans_list = literal_eval(...)`.
        func = self.flow._scope_of(try_node)
        if targets and not isinstance(func, ast.Module):
            for node in ast.walk(func):
                if isinstance(node, ast.Assign) and \
                        node.lineno < try_node.lineno and \
                        {ast.unparse(t) for t in node.targets} & targets \
                        and not _zero_value(node.value):
                    return True
        for stmt in self.flow.following_up(try_node):
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and (
                        self._parse_call(node)
                        or _RETRY_WORDS.search(_dotted(node.func))):
                    return True
                if isinstance(node, ast.Return) and node.value is not None \
                        and arg is not None and self._uses(node.value, arg):
                    return True
            if root and self._keeps_using(stmt, root):
                return True
        return False

    @staticmethod
    def _root_name(expr):
        while isinstance(expr, (ast.Attribute, ast.Subscript, ast.Call)):
            if isinstance(expr, ast.Call):
                if not isinstance(expr.func, ast.Attribute):
                    return None
                expr = expr.func.value
            else:
                expr = expr.value
        return expr.id if isinstance(expr, ast.Name) else None

    def _inspects(self, handler: ast.ExceptHandler, arg) -> bool:
        """The handler looks at the raw text again (`if "yes" in
        llm_response`, `re.search(..., response)`): a second, cruder
        reader, not a zero."""
        root = self._root_name(arg)
        if not root:
            return False
        for stmt in handler.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and _RETRY_WORDS.search(
                        _dotted(node.func)) and any(
                        isinstance(n, ast.Name) and n.id == root
                        for a in node.args for n in ast.walk(a)):
                    return True
                test = getattr(node, "test", None) if isinstance(
                    node, (ast.If, ast.IfExp, ast.While)) else (
                    node if isinstance(node, ast.Compare) else None)
                if test is not None and any(
                        isinstance(n, ast.Name) and n.id == root
                        for n in ast.walk(test)):
                    return True
        return False

    def _gold_target(self, call: ast.Call) -> bool:
        """`gt_bbox = literal_eval(...)`: the code names what it parsed."""
        parent = self.flow.parent.get(call)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) \
                else [parent.target]
            return any(set(_words(ast.unparse(t))) & _GOLD for t in targets)
        return False

    def _retries(self, handler: ast.ExceptHandler) -> bool:
        """The handler tries another parser (`except JSONDecodeError:
        return ast.literal_eval(text)`): this site is not where the zero
        is made; the last parser is, and it is judged on its own."""
        return any(isinstance(n, ast.Call) and self._parse_call(n)
                   for stmt in handler.body for n in ast.walk(stmt))

    @staticmethod
    def _keeps_using(stmt, root: str) -> bool:
        """The code after the failure goes on working on the raw input
        (other than to log it)."""
        logged = set()
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and _LOG_CALL.search(
                    _dotted(node.func)):
                logged |= {id(n) for n in ast.walk(node)}
        return any(isinstance(n, ast.Name) and n.id == root
                   and isinstance(n.ctx, ast.Load) and id(n) not in logged
                   for n in ast.walk(stmt))

    @staticmethod
    def _uses(expr, arg) -> bool:
        """Does `expr` (a return value) hand back the raw input `arg`,
        as is or converted: `return value`, `return str(value)`?"""
        if isinstance(arg, ast.Name):
            return any(isinstance(n, ast.Name) and n.id == arg.id
                       for n in ast.walk(expr))
        return ast.unparse(arg) in ast.unparse(expr)

    def _record(self, node, name, evidence, **extra) -> None:
        site = {
            "file": self.path,
            "line": node.lineno,
            "call": name,
            "code": self._segment(node).strip().split("\n")[0][:100],
            "evidence": evidence,
        }
        site.update(extra)
        func = self._funcs[-1] if self._funcs else None
        caller_line = [extra["zeroed_at"]] if "zeroed_at" in extra else []
        if self._suppressed(node, func, caller_line):
            self.suppressed.append(site)   # counted, never silent
        else:
            self.sites.append(site)

    def _flags(self, kind: str) -> bool:
        return kind == _MODEL or (kind == _WEAK_ONLY and self.context)

    @property
    def marker_lines(self) -> set:
        """Lines holding a `# fencecheck: ignore` COMMENT. A marker inside
        a string or docstring does not count."""
        if self._marker_lines is None:
            found = set()
            if not IGNORE_MARKER.search(self.source):
                self._marker_lines = found
                return found
            try:
                tokens = tokenize.generate_tokens(
                    io.StringIO(self.source).readline)
                for tok in tokens:
                    if tok.type == tokenize.COMMENT and \
                            IGNORE_MARKER.search(tok.string):
                        found.add(tok.start[0])
            except (tokenize.TokenError, SyntaxError, IndentationError):
                found = {i + 1 for i, text in enumerate(self.lines)
                         if IGNORE_MARKER.search(text)}
            self._marker_lines = found
        return self._marker_lines

    def _suppressed(self, node, func, extra_lines=()) -> bool:
        """A marker on the parse's lines, on an extra line (the caller that
        makes the zero), or anywhere in the enclosing function."""
        marks = self.marker_lines
        if not marks:
            return False
        lines = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        lines |= set(extra_lines)
        if func is not None:
            lines |= set(range(func.lineno, (func.end_lineno or func.lineno)
                               + 1))
        return bool(lines & marks)

    def _consider(self, node: ast.Call, name: str) -> None:
        # A fence handled anywhere around the parse: this function, or a
        # function that encloses a nested helper.
        if any(_fence_aware(self._segment(f), f) for f in self._funcs) \
                or (not self._funcs and _fence_aware(self.source, self.tree)):
            return
        if _argument_is_a_file_read(node, self._segment(node)):
            return
        arg = node.args[0] if node.args else (
            node.keywords[0].value if node.keywords else None)
        if arg is None:
            return
        bases = self.flow.bases
        handler, where = _handling(self._tries[-1], None, bases)
        if handler is _ESCAPES:
            if self._funcs:
                self._via_caller(node, name, arg, self._funcs[-1], where)
            return
        if not _zeroish(handler.body) or self._retries(handler) or \
                self._inspects(handler, arg):
            return
        if _only_passes(handler.body) and self._falls_back(node, where):
            return
        if self._gold_target(node):
            return
        kind, evidence = self.flow.classify(arg, node)
        if self._flags(kind):
            self._record(node, name, evidence)

    def _via_caller(self, node, name, arg, func, pending) -> None:
        """One hop: the parse error (or what it was re-raised as) leaves
        `func`; an in-module caller catches it and makes a zero."""
        flow = self.flow
        for call in flow.call_sites(func):
            caller = flow._scope_of(call)
            text = flow.text_of(caller)
            if _fence_aware(text, caller):
                continue
            tries = flow.tries_at(call)
            handler, where = _handling(tries, pending, flow.bases)
            if handler is _ESCAPES or not _zeroish(handler.body) or \
                    self._retries(handler):
                continue
            if _only_passes(handler.body) and self._falls_back(call, where):
                continue
            kind, evidence = flow.classify(arg, node, via={id(func): call})
            if self._flags(kind):
                self._record(node, name, evidence, zeroed_at=call.lineno)
                return


def scan_source(source: str, path: str,
                suppressed: list | None = None,
                unanalysed: list | None = None) -> list[dict]:
    """Findings for one file. Empty list if the file is not in scope.
    Findings silenced by `# fencecheck: ignore` are appended to
    `suppressed` when a list is passed, so they can be counted. A file
    nested too deeply for the parser or the visitor (RecursionError) is
    appended to `unanalysed` instead of crashing the whole scan."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    except RecursionError:
        if unanalysed is not None:
            unanalysed.append(path)
        return []
    visitor = _Visitor(source, path, tree)
    try:
        visitor.visit(tree)
    except RecursionError:
        if unanalysed is not None:
            unanalysed.append(path)
        return []
    if suppressed is not None:
        suppressed.extend(visitor.suppressed)
    return visitor.sites


def scan_path(root: pathlib.Path,
              suppressed: list | None = None,
              unanalysed: list | None = None) -> tuple[list[dict], int]:
    findings, files = [], 0
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for path in paths:
        # Only directories BELOW the scan root are skipped: a repository
        # that itself sits under a directory named build/ or venv/ used to
        # scan zero files and exit 0.
        below = path.relative_to(root).parts[:-1] if path != root else ()
        if any(part in SKIP_DIRS for part in below):
            continue
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
            # A file named on the command line is always read; inside a
            # directory, tests are skipped.
            if _is_test_path(pathlib.PurePath(shown)):
                continue
        try:
            # utf-8-sig: a byte-order mark otherwise makes ast.parse refuse a
            # valid file, which was then counted as scanned and silently empty.
            source = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        files += 1
        findings.extend(scan_source(source, str(shown), suppressed, unanalysed))
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


_KEY_PATTERN = re.compile(r'(?:"[^"\n]{1,80}"|\'[^\'\n]{1,80}\')\s*:')
_FENCE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _bare_object_in_prose(text: str):
    """The first {...} span that parses strictly to an object AND whose
    surrounding text holds no other brace and no "key": pattern -- one
    complete object with prose around it (Addendum U's
    `exact_object_present` rule). None otherwise."""
    for m in re.finditer(r"\{", text):
        start = m.start()
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            span = text[start:end]
            # fencecheck: ignore -- probing a candidate span for the
            # opt-in wider scope; the fail-closed parse is the point.
            try:
                value = json.loads(span)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            rest = text[:start] + text[end:]
            if not rest.strip():
                return None  # the text IS the object; nothing to widen
            if "{" in rest or "}" in rest or _KEY_PATTERN.search(rest):
                return None
            return span
    return None


def extract_json(text: str, scope: str = "leading") -> tuple[str, str]:
    """Return (body, kind). scope="leading" is exactly strip_fence -- one
    leading fence, nothing else; every frozen reading uses it. scope="any"
    (Addendum W, opt-in) widens, in this order: a leading fence; else the
    first fenced block anywhere in the text; else a bare object inside
    prose; else the text unchanged. kind is one of "none",
    "leading_fence", "fence_after_prose", "bare_object_in_prose"."""
    if scope not in ("leading", "any"):
        raise ValueError(f"scope must be 'leading' or 'any', not {scope!r}")
    body, was_fenced = strip_fence(text)
    if was_fenced:
        return body, "leading_fence"
    if scope == "leading":
        return body, "none"
    block = _FENCE_BLOCK.search(text)
    if block:
        return block.group(1).strip(), "fence_after_prose"
    bare = _bare_object_in_prose(text)
    if bare is not None:
        return bare, "bare_object_in_prose"
    return body, "none"


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


def score_file(path: pathlib.Path, scope: str = "leading") -> dict:
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
    valid_any = 0
    any_kinds: dict[str, int] = {}
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
            if scope == "any":
                body, kind = extract_json(text, "any")
                any_ok = _parses(body)
                valid_any += any_ok
                if any_ok and not stripped_ok:
                    any_kinds[kind] = any_kinds.get(kind, 0) + 1
            break
    result = {
        "outputs": total,
        "fenced": fenced,
        "parse_as_written": valid_asis,
        "parse_after_stripping": valid_stripped,
        "recovered_by_stripping": recovered,
        "already_parsed_records": already_parsed,
        "examples": examples,
    }
    if scope == "any":
        # Addendum W: the opt-in wider scope, reported BESIDE the
        # leading-fence numbers, never instead of them.
        result["parse_after_any_scope"] = valid_any
        result["recovered_by_any_scope_beyond_stripping"] = any_kinds
    return result


def _parses(text: str) -> bool:
    # fencecheck: ignore -- asking "does this parse exactly as written?"
    # is this function's entire job, so strictness here is deliberate.
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


# ------------------------------------------------- chat templates that move
# A chat template that reads the clock renders a DIFFERENT PROMPT EVERY DAY.
# Nothing warns you. Outputs banked on Tuesday and outputs banked on Friday
# were not produced from the same input, so they cannot be compared to each
# other and neither can be reproduced later.
#
# This is not hypothetical and it is not rare. It was found in this
# repository's own banked work on 2026-09-16:
# `ibm-granite/granite-3.1-2b-instruct` builds its own system message
# containing `strftime_now('%B %d, %Y')`, and two cells of one experiment --
# compared against each other in a published result -- had been produced five
# days apart. The comparison was reported as byte-identical arms without
# anyone knowing the arms had been sent different prompts.
CLOCK_CALLS = (
    "strftime_now", "datetime.now", "datetime.today", "date.today",
    "utcnow", "time.time", "now()", "today()",
)
_TEMPLATE_FILES = ("tokenizer_config.json", "chat_template.json",
                   "chat_template.jinja", "chat_template.j2")


def _template_sources(path: pathlib.Path):
    """Yield (file, key, template_text) for every chat template under path.

    Accepts a tokenizer_config.json, a bare .jinja template, or a directory
    to walk. Kept to the standard library on purpose: this has to run in a
    repository that has no model libraries installed.
    """
    files = ([path] if path.is_file()
             else sorted(f for f in path.rglob("*") if f.name in _TEMPLATE_FILES))
    for file in files:
        try:
            raw = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if file.suffix == ".json":
            try:
                # fencecheck: ignore -- a tokenizer config read from disk, not
                # model output; an unparseable file is skipped, never scored.
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            template = data.get("chat_template")
            if isinstance(template, str):
                yield file, "chat_template", template
            elif isinstance(template, list):
                # some checkpoints ship several named templates
                for entry in template:
                    if isinstance(entry, dict) and isinstance(entry.get("template"), str):
                        yield file, f"chat_template[{entry.get('name', '?')}]", entry["template"]
        else:
            yield file, file.name, raw


def scan_chat_templates(path: pathlib.Path) -> list:
    """Every chat template whose rendered prompt depends on the clock."""
    findings = []
    for file, key, template in _template_sources(path):
        hits = sorted({call for call in CLOCK_CALLS if call in template})
        if not hits:
            continue
        where = min(template.find(h) for h in hits if h in template)
        line = template.count("\n", 0, where) + 1
        findings.append({
            "file": str(file),
            "template": key,
            "calls": hits,
            "line_in_template": line,
            "excerpt": " ".join(template[max(0, where - 90):where + 60].split()),
        })
    return findings


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
    score.add_argument("--scope", choices=("leading", "any"), default="leading",
                       help="leading (default): one leading fence, the scope every "
                            "published reading uses; any: also a fence after prose "
                            "or a bare object inside prose, reported beside it")

    template = sub.add_parser(
        "template", help="report chat templates that render a different "
                         "prompt every day")
    template.add_argument("path")
    template.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    target = pathlib.Path(args.path).expanduser()
    if not target.exists():
        print(f"fencecheck: no such path: {target}", file=sys.stderr)
        return 2

    if args.command == "scan":
        suppressed: list[dict] = []
        unanalysed: list[str] = []
        findings, files = scan_path(target, suppressed, unanalysed)
        if unanalysed:
            print(f"fencecheck: {len(unanalysed)} file(s) nested too deeply to "
                  "analyse were skipped, not cleared: " + ", ".join(unanalysed),
                  file=sys.stderr)
        if args.json:
            print(json.dumps({"findings": findings,
                              "files_scanned": files,
                              "suppressed": suppressed,
                              "unanalysed": unanalysed}, indent=2))
            return 1 if findings else 0
        if suppressed:
            # A suppression CI cannot count is a suppression nobody audits.
            print(f"fencecheck: {len(suppressed)} site(s) silenced by "
                  "`# fencecheck: ignore` (not counted below):")
            for item in suppressed:
                print(f"  {item['file']}:{item['line']}  {item['call']}")
            print()
        if not findings:
            print(f"fencecheck: {files} file(s) scanned, nothing exposed.")
            print("No site was found where model output is parsed with no "
                  "fence handling and the failure becomes a zero, in its "
                  "own function or in a caller one call up in the same file.")
            print("scan does not follow calls across files or further up: "
                  "a zero made there is not reported (tools/README.md).")
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

    if args.command == "template":
        findings = scan_chat_templates(target)
        if args.json:
            print(json.dumps({"findings": findings}, indent=2))
        elif not findings:
            print("fencecheck: no chat template here reads the clock.")
            print("Prompts rendered from these templates depend on the "
                  "messages and nothing else.")
        else:
            print(f"fencecheck: {len(findings)} chat template(s) render a "
                  "DIFFERENT PROMPT EVERY DAY.\n")
            for item in findings:
                print(f"  {item['file']}  ({item['template']}, line "
                      f"{item['line_in_template']})")
                print(f"      calls: {', '.join(item['calls'])}")
                print(f"      {item['excerpt']}")
            print("\nOutputs you banked on different days were not produced "
                  "from the same prompt, so they cannot be compared with each "
                  "other and cannot be reproduced later. Pass the system "
                  "message explicitly, pinned to a fixed value, and record it "
                  "beside the outputs.")
        return 1 if findings else 0

    report = score_file(target, scope=args.scope)
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
    if not report["parse_after_stripping"]:
        # Nothing parses even after stripping: a clean bill would be a
        # statement about a file that holds no JSON output at all (a
        # reviewer fed plain prose and got "nothing is being lost",
        # 2026-09-02). Say what was seen and refuse the verdict.
        print(f"fencecheck: REFUSING a verdict -- none of the {total} "
              "output(s) parse as JSON, fenced or not. Either this file "
              "is not model JSON output, or the outputs are broken in a "
              "way no fence explains; a fence-loss number about it would "
              "be meaningless.")
        return 2
    print(f"fencecheck: {total} output(s)")
    print(f"  fenced                  {report['fenced']}")
    print(f"  parse as written        {report['parse_as_written']}")
    print(f"  parse after stripping   {report['parse_after_stripping']}")
    if "parse_after_any_scope" in report:
        # Addendum W: the opt-in wider scope, printed BESIDE the default
        # numbers (a reviewer ran --scope any without --json and saw
        # nothing of it, 2026-09-08).
        extra = report["recovered_by_any_scope_beyond_stripping"]
        print(f"  parse after any scope   {report['parse_after_any_scope']}"
              + (f"   (+{sum(extra.values())}: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(extra.items())) + ")"
                 if extra else "   (no further recoveries)"))
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
