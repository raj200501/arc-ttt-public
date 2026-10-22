"""Regression tests for the scanner review of 2026-10-22.

The review's `surviving` list holds the defects it reproduced in
`tools/fencecheck.py` after commits 7f9068a and 452527b. Each test here pins one of them with the
smallest case that shows it, and each failed against the scanner as it
stood at 452527b. The held-out census (Addendum AB) freezes this file's
sha256 afterwards, so these are the cases that have to stay right.

The cases are synthetic, written from the review's descriptions. They are
a development set, like tests/test_fencecheck_probes.py: they say the
listed shapes behave as documented, and nothing about precision or recall
on code nobody has looked at.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "fencecheck.py"
HOOK = REPO / "tools" / "fencecheck_hook.py"
README = REPO / "tools" / "README.md"

pytestmark = pytest.mark.skipif(not TOOL.exists(),
                                reason="the tool is not in this cut")


def _load():
    spec = importlib.util.spec_from_file_location("_fencecheck_review", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fc = _load()


def _scan(src: str, name: str = "m.py") -> list:
    return fc.scan_source(textwrap.dedent(src).lstrip("\n"), name)


def _lines(src: str, name: str = "m.py") -> list:
    return [f["line"] for f in _scan(src, name)]


def _run(*args: str, cwd=None, env=None, timeout=60):
    return subprocess.run([sys.executable, str(TOOL), *args], check=False,
                          capture_output=True, text=True, cwd=cwd, env=env,
                          timeout=timeout)


# The baseline every case below varies: model output parsed with no fence
# handling, and the failure scored 0.
BASE = '''
import json

def score(completion):
    {extra}
    try:
        return json.loads(completion)
    except ValueError:
        return 0
'''


def test_the_baseline_is_a_finding():
    assert _lines(BASE.format(extra="pass")) == [6]


# ------------------------------------------- fence words that handle nothing
@pytest.mark.parametrize("extra", [
    "# TODO: strip markdown fences before parsing",
    'logging.warning("model answered in markdown; scoring 0")',
    '"""Fenced answers are not supported."""',
    '"""Strict: a fenced answer scores zero."""',
    '"""Example reply: ```json {"a": 1} ``` scores 0."""',
    'print("got ```json, scoring it 0")',
    'note = "extract_json and strip_code_block are not applied here"',
])
def test_a_fence_word_that_handles_nothing_does_not_exempt(extra):
    """A comment, docstring, log line or string that MENTIONS a fence used
    to exempt the whole function: the text of the function was searched,
    not its code."""
    assert _lines(BASE.format(extra=extra)) == [6], extra


def test_a_parameter_named_for_markdown_does_not_exempt():
    assert _lines('''
        import json

        def score(completion, render_markdown=False):
            if render_markdown:
                print(completion)
            try:
                return json.loads(completion)
            except ValueError:
                return 0
    ''') == [7]


def test_a_comment_in_the_caller_does_not_exempt_one_call_up():
    assert [(f["line"], f.get("zeroed_at")) for f in _scan('''
        import json

        def parse(completion):
            return json.loads(completion)

        def score(completion):
            # markdown answers are not handled
            try:
                return parse(completion)
            except ValueError:
                return 0
    ''')] == [(4, 9)]


@pytest.mark.parametrize("handling", [
    "completion = strip_markdown_fences(completion)",
    "completion = json_repair.repair_json(completion)",
    'completion = completion.removeprefix("```json").removesuffix("```")',
    'completion = FENCE.sub("", completion)',
])
def test_fence_handling_in_code_still_exempts(handling):
    assert _lines(BASE.format(extra=handling)) == [], handling


# ---------------------------------------------------------- removeprefix(
@pytest.mark.parametrize("strip,reported", [
    ('completion.strip().removeprefix("```python").removesuffix("```")', True),
    ('completion.removeprefix("Answer: ")', True),
    ('completion.removeprefix("```").removesuffix("```")', True),
    ('completion.removeprefix("```json").removesuffix("```")', False),
])
def test_removeprefix_counts_only_for_what_it_removes(strip, reported):
    """`removeprefix(` anywhere used to count as fence handling, so
    removing ```python only, or "Answer: ", was treated as handled."""
    found = _lines(BASE.format(extra=f"completion = {strip}"))
    assert bool(found) is reported, (strip, found)


# --------------------------------------------- a fence helper with a plain name
HELPERS = {
    "replace": (
        "def clean(text):\n"
        "    return text.replace(\"```json\", \"\").replace(\"```\", \"\").strip()\n"),
    "drop_line": (
        "def clean(text):\n"
        "    text = text.strip()\n"
        "    if text.startswith(\"```\"):\n"
        "        text = text.split(\"\\n\", 1)[1].rsplit(\"```\", 1)[0]\n"
        "    return text\n"),
}


@pytest.mark.parametrize("helper", sorted(HELPERS))
def test_a_fence_strip_in_a_plainly_named_helper_is_handling(helper):
    """The same strip written inline was handling; moved into a helper
    with a neutral name, it was not."""
    src = ("import json\n\n" + HELPERS[helper] + "\n"
           "def score(completion):\n"
           "    try:\n"
           "        return json.loads(clean(completion))\n"
           "    except ValueError:\n"
           "        return 0\n")
    assert fc.scan_source(src, "m.py") == [], src


def test_a_fence_strip_in_a_method_is_handling():
    assert _lines('''
        import json

        class Scorer:
            def _clean_completion(self, text):
                return text.replace("```json", "").replace("```", "")

            def score(self, completion):
                try:
                    return json.loads(self._clean_completion(completion))
                except ValueError:
                    return 0
    ''') == []


def test_a_model_wrapper_that_mentions_a_fence_in_its_prompt_is_not_handling():
    """Only a helper that calls no model counts as a cleaning helper: a
    wrapper whose prompt asks for ```json returns the raw reply."""
    assert _lines('''
        import json

        def ask(llm, prompt):
            full = prompt + "Answer in a ```json block."
            return llm.invoke(full).content

        def score(llm, question):
            try:
                return json.loads(ask(llm, question))
            except ValueError:
                return 0
    ''') == [9]


def test_a_plain_helper_that_does_not_strip_a_fence_is_not_handling():
    assert _lines('''
        import json

        def clean(text):
            return text.strip()

        def score(completion):
            try:
                return json.loads(clean(completion))
            except ValueError:
                return 0
    ''') == [8]


# ------------------------------------- a model-call word inside another name
@pytest.mark.parametrize("producer", [
    "tokenize.generate_tokens(fh.readline)",
    "generate_report(data)",
    "store.get_chat_state(key)",
])
def test_a_model_call_word_inside_another_name_is_not_a_model_call(producer):
    """pydoc.py:400 in the standard library was reported because the text
    came from `tokenize.generate_tokens`."""
    assert _lines(f'''
        import ast

        def load(data, key, fh, store):
            text = {producer}
            try:
                return ast.literal_eval(text)
            except ValueError:
                return None
    ''') == [], producer


@pytest.mark.parametrize("producer", [
    "model.generate_content(prompt).text",
    "llm.invoke(prompt).content",
    "client.generate(prompt)",
    "batch_generate(prompt)",
    "self.ask_judge(prompt)",
])
def test_a_model_call_is_still_a_model_call(producer):
    assert _lines(f'''
        import ast

        def load(self, model, llm, client, prompt):
            text = {producer}
            try:
                return ast.literal_eval(text)
            except ValueError:
                return None
    ''') == [6], producer


# ---------------------------------------------- the module-context gate
@pytest.mark.parametrize("word", ["output", "outputs", "response", "responses",
                                  "answer", "answers", "reply", "replies"])
def test_a_weak_word_does_not_vouch_for_its_own_module(word):
    """`def to_dict(output)` in a plain utility was flagged on the strength
    of the one word being judged: the weak words were also on the list of
    words that make a module about models."""
    src = f'''
        import json

        def to_dict({word}):
            try:
                return json.loads({word})
            except ValueError:
                return {{}}
    '''
    assert _lines(src, "util.py") == [], word
    assert _lines('"""Model evaluation helpers."""\n'
                  + textwrap.dedent(src).lstrip("\n"), "util.py") == [6], word
    assert _lines(src, "metrics.py") == [5], word      # the file's own name


def test_a_strong_compound_name_still_gives_context():
    assert _lines('''
        import json

        def to_dict(output, model_output_cache):
            try:
                return json.loads(output)
            except ValueError:
                return {}
    ''', "util.py") == [5]


# --------------------------------------------- one call up, the right class
TWO_CLASSES = '''
import json
import re

class A:
    def _parse(self, text):
        return json.loads(text)

    def score(self, completion):
        return self._parse(completion)

class {b}:
    def _parse(self, text):
        m = re.search(r"\\{{.*\\}}", text, re.S)
        return json.loads(m.group(0) if m else text)

    def score(self, completion):
        try:
            data = self._parse(completion)
        except ValueError:
            return 0
        return data
'''


def test_one_call_up_does_not_cross_into_another_class():
    """B.score's `self._parse(...)` runs B._parse, never A._parse; A's loud
    parse used to be reported with B's zero as `zeroed_at`."""
    assert fc.scan_source(TWO_CLASSES.format(b="B"), "m.py") == []


def test_one_call_up_follows_a_subclass_that_inherits_the_method():
    src = '''
import json

class A:
    def _parse(self, text):
        return json.loads(text)

class B(A):
    def score(self, completion):
        try:
            return self._parse(completion)
        except ValueError:
            return 0
'''
    assert [(f["line"], f.get("zeroed_at")) for f in
            fc.scan_source(src, "m.py")] == [(6, 11)]


def test_one_call_up_follows_a_template_method_in_the_base_class():
    src = '''
import json

class Base:
    def score(self, completion):
        try:
            return self._parse(completion)
        except ValueError:
            return 0

class Strict(Base):
    def _parse(self, text):
        return json.loads(text)
'''
    assert [(f["line"], f.get("zeroed_at")) for f in
            fc.scan_source(src, "m.py")] == [(13, 7)]


# ------------------------------------------ a sentinel the code recovers from
@pytest.mark.parametrize("after", [
    "if data is None:\n        data = ast.literal_eval(completion)",
    "if data is None:\n        return completion",
    "if not data:\n        data = completion",
])
def test_a_sentinel_followed_by_recovery_is_not_a_zero(after):
    src = '''
import ast
import json

def score(completion):
    try:
        data = json.loads(completion)
    except ValueError:
        data = None
    {after}
    return data
'''.format(after=after)
    assert fc.scan_source(src, "m.py") == [], after


@pytest.mark.parametrize("after", [
    'print("unparseable")',
    'return {"score": 0, "raw": completion}',
])
def test_a_sentinel_with_no_recovery_is_still_a_zero(after):
    src = '''
import json

def score(completion, gold):
    try:
        data = json.loads(completion)
    except ValueError:
        data = None
    if data is None:
        {after}
    return float(data == gold)
'''.format(after=after)
    assert [f["line"] for f in fc.scan_source(src, "m.py")] == [6], after


# --------------------------------------------- process output by bare name
@pytest.mark.parametrize("imports,call", [
    ("from subprocess import check_output", "check_output(cmd, text=True)"),
    ("import subprocess as sp", "sp.check_output(cmd, text=True)"),
])
def test_process_output_is_a_carrier_however_it_was_imported(imports, call):
    assert _lines(f'''
        import json
        {imports}

        def score_run(cmd):
            output = {call}
            try:
                return json.loads(output)
            except ValueError:
                return 0
    ''') == [], imports


# ------------------------------------------------------------- speed
def _generated_module() -> str:
    """4,000 lines built from every shape the review timed: many
    module-level parses (15 s at 1,000 lines), many guarded parses in one
    function, silent skips, one helper called from hundreds of places,
    re-assigned names that dead-end (24 s at 60 lines), and a class whose
    methods call one another."""
    out = ["import json", "", "", "def h():", "    pass", "", "",
           "def grade(completion):", "    try:",
           "        return json.loads(completion)", "    except ValueError:",
           "        return 0", ""]
    out += [f"m{i} = json.loads('{{\"a\": {i}}}')" for i in range(600)]
    out += ["", "", "def score_all(completions):"]
    for i in range(300):
        out += ["    try:", f"        p{i} = json.loads(completions[{i}])",
                "    except ValueError:", f"        p{i} = None"]
    out += ["", "", "def skim(outputs):", "    kept = []"]
    for i in range(200):
        out += ["    try:", f"        kept.append(json.loads(outputs[{i}]))",
                "    except ValueError:", "        pass"]
    out += ["    return kept", "", "", "def run(responses):"]
    out += [f"    s{i} = grade(responses[{i}])" for i in range(600)]
    out += ["", "", "def chain(x):", "    v8 = h()"]
    for i in range(7, -1, -1):
        out += [f"    v{i} = v{i + 1}"] * 6
    out += ["    try:", "        obj = json.loads(v0)", "    except ValueError:",
            "        return 0", "    return obj", "", "", "class Judge:",
            "    def _parse(self, text):", "        return json.loads(text)", ""]
    for i in range(120):
        out += [f"    def score_{i}(self, completion):", "        try:",
                "            return self._parse(completion)",
                "        except ValueError:", "            return 0.0"]
    while len(out) < 4000:
        out.append(f"pad_{len(out)} = {len(out)}")
    assert len(out) == 4000
    return "\n".join(out) + "\n"


def test_a_4000_line_generated_module_scans_in_under_5_seconds():
    """The scan was quadratic (a fence check re-walked the enclosing
    function for every parse and every caller) and exponential (a trace's
    cap counted what it emitted, not what it visited). This module took
    70 s of CPU at 452527b."""
    src = _generated_module()
    started = time.process_time()
    found = fc.scan_source(src, "generated.py")
    took = time.process_time() - started
    assert took < 5, f"{took:.1f} s"
    # every guarded parse in score_all, grade itself, the last silent skip
    # (nothing after it recovers) and Judge._parse via its callers
    assert len(found) == 303, len(found)


@pytest.mark.parametrize("b", [6, 12])
def test_reassigned_names_that_dead_end_do_not_explode(b):
    lines = ["import json", "", "def h():", "    pass", "", "def score(x):",
             "    v8 = h()"]
    for i in range(7, -1, -1):
        lines += [f"    v{i} = v{i + 1}"] * b
    lines += ["    try:", "        obj = json.loads(v0)", "    except ValueError:",
              "        return 0", "    return obj"]
    started = time.process_time()
    fc.scan_source("\n".join(lines) + "\n", "dag.py")
    assert time.process_time() - started < 2


# ------------------------------------------------- files that are not read
def test_a_file_this_python_cannot_parse_is_named_not_cleared(tmp_path, capsys):
    """A file the running interpreter cannot parse was counted as scanned
    and clean. Python 3.11 cannot parse 3.12 syntax, and 3.11 was the
    Action's default."""
    (tmp_path / "py2.py").write_text('print "hello"\n')
    (tmp_path / "ok.py").write_text("x = 1\n")
    if sys.version_info < (3, 12):
        (tmp_path / "pep695.py").write_text("type Score = float\n")
    unparsed: list = []
    findings, files = fc.scan_path(tmp_path, [], [], unparsed)
    names = sorted(u["file"] for u in unparsed)
    expected = ["py2.py"] + (["pep695.py"] if sys.version_info < (3, 12) else [])
    assert names == sorted(expected) and files == len(expected) + 1
    assert all(u["error"].startswith("SyntaxError") for u in unparsed)
    # Nothing found, but not every file analysed: no clean bill, exit 2
    # (it was 0, so the Action and the hook passed the file; check of
    # 9b41c99).
    assert fc.main(["scan", "--json", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert sorted(u["file"] for u in report["unparsed"]) == sorted(expected)
    assert report["files_scanned"] == len(expected) + 1
    assert report["files_analysed"] == 1
    assert "py2.py" in captured.err and "not analysed" in captured.err
    assert fc.main(["scan", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "not analysed" in captured.out and "py2.py" in captured.err
    assert "nothing exposed." not in captured.out
    assert "no verdict" in captured.out
    # a finding still exits 1, with the unparsed file named beside it
    (tmp_path / "grade.py").write_text(textwrap.dedent(BASE.format(extra="pass")))
    assert fc.main(["scan", "--json", str(tmp_path)]) == 1
    capsys.readouterr()
    # and with every file analysed, nothing found is still 0
    for name in ["py2.py", "grade.py"] + (["pep695.py"] if sys.version_info < (3, 12) else []):
        (tmp_path / name).unlink()
    assert fc.main(["scan", "--json", str(tmp_path)]) == 0


def test_the_action_runs_a_python_that_parses_current_syntax():
    text = (REPO / "action.yml").read_text(encoding="utf-8")
    assert 'default: "3.13"' in text


def test_a_coding_cookie_is_honoured(tmp_path):
    body = ("# -*- coding: latin-1 -*-\n"
            "import json\n"
            "caf\xe9 = 1\n"
            "def score(completion):\n"
            "    try:\n"
            "        return json.loads(completion)\n"
            "    except ValueError:\n"
            "        return 0\n")
    (tmp_path / "latin.py").write_bytes(body.encode("latin-1"))
    unparsed: list = []
    findings, files = fc.scan_path(tmp_path, [], [], unparsed)
    assert [f["line"] for f in findings] == [6] and unparsed == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_a_named_pipe_ending_in_py_does_not_hang_the_scan(tmp_path):
    (tmp_path / "grade.py").write_text(textwrap.dedent(BASE.format(extra="pass")))
    os.mkfifo(tmp_path / "pipe.py")
    result = _run("scan", "--json", str(tmp_path), timeout=30)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["files_scanned"] == 1 and len(report["findings"]) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file names")
def test_a_file_name_that_is_not_utf8_does_not_crash_the_report(tmp_path):
    name = os.fsdecode(b"sc\xffore.py")
    (tmp_path / name).write_text(textwrap.dedent(BASE.format(extra="pass")))
    env = dict(os.environ, PYTHONIOENCODING="utf-8:strict")
    result = _run("scan", str(tmp_path), env=env)
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    assert "\\udcff" in result.stdout or "\\xff" in result.stdout


# ------------------------------------------------------------------ score
@pytest.mark.parametrize("content", [
    '["```json\\n{\\"a\\": 1}\\n```"]\n["{\\"a\\": 1}"]\n',     # JSONL of lists
    "[1, 2]\n[3, 4]\n",
    '[INFO] the answer is\n```json\n{"a": 1}\n```\n',              # text
    '[{"output": "x"},\n {"output": "y"\n',                       # broken list
])
def test_score_never_crashes_on_a_file_that_starts_with_a_bracket(
        tmp_path, content):
    path = tmp_path / "p.jsonl"
    path.write_text(content, encoding="utf-8")
    for extra in ([], ["--json"]):
        result = _run("score", *extra, str(path))
        assert result.returncode in (0, 1, 2), result.stderr
        assert "Traceback" not in result.stderr, result.stderr


def test_score_refuses_a_directory_with_exit_2(tmp_path):
    result = _run("score", str(tmp_path))
    assert result.returncode == 2 and "Traceback" not in result.stderr


@pytest.mark.parametrize("content,why", [
    ("the quick brown fox\njumps over the lazy dog\n", "parse"),
    ('{"id": 1, "prediction": {"a": 1}}\n{"id": 2, "prediction": null}\n',
     "already a parsed object"),
    ('{"id": "a"}\n{"id": "b"}\n', "no model outputs"),
])
def test_score_json_refuses_what_text_mode_refuses(tmp_path, content, why):
    path = tmp_path / "p.jsonl"
    path.write_text(content, encoding="utf-8")
    assert _run("score", str(path)).returncode == 2
    result = _run("score", "--json", str(path))
    assert result.returncode == 2, result.stdout
    assert why in json.loads(result.stdout)["refused"]


def test_the_wider_scope_is_linear_in_the_output(tmp_path):
    """`--scope any` tried every later `}` for every `{`: a 20 KB code
    output took 1.4 s."""
    text = "Here is the code:\n" + "int f(int x) { return x; }\n" * 1600
    started = time.process_time()
    assert fc.extract_json(text, "any")[1] == "none"
    assert time.process_time() - started < 1
    assert fc.extract_json('Sure! {"a": {"b": 1}} ok', "any") == (
        '{"a": {"b": 1}}', "bare_object_in_prose")


# --------------------------------------------------------------- template
def test_template_reads_every_jinja_file_in_a_directory(tmp_path):
    ckpt = tmp_path / "ckpt"
    (ckpt / "additional_chat_templates").mkdir(parents=True)
    (ckpt / "tokenizer_config.json").write_text('{"model_max_length": 8}')
    (ckpt / "chat_template.jinja").write_text("{{ messages }}")
    (ckpt / "additional_chat_templates" / "tool_use.jinja").write_text(
        "{{ strftime_now('%Y-%m-%d') }}{{ messages }}")
    found = fc.scan_chat_templates(ckpt)
    assert [pathlib.Path(f["file"]).name for f in found] == ["tool_use.jinja"]


# ------------------------------------------------------- the hook, the docs
def test_the_hook_names_a_file_it_could_not_parse(tmp_path):
    (tmp_path / "py2.py").write_text('print "hello"\n')
    result = subprocess.run([sys.executable, str(HOOK), "py2.py"],
                            capture_output=True, text=True, cwd=tmp_path,
                            timeout=60, check=False)
    assert result.returncode == 2
    assert "py2.py" in result.stderr and "not analysed" in result.stderr
    assert "nothing exposed." not in result.stdout


def _not_catch_section() -> str:
    text = " ".join(README.read_text(encoding="utf-8").split())
    start = text.index("It will not catch:")
    return text[start:text.index("How many real sites it misses", start)]


def test_the_readme_lists_every_directory_scan_skips():
    section = _not_catch_section()
    for name in sorted(fc.SKIP_DIRS) + ["tests", "test", "testing"]:
        assert f"`{name}/`" in section, name


def test_the_readme_says_unparsed_files_are_named():
    section = _not_catch_section()
    assert "`unparsed`" in section and "3.13" in section


def test_the_readme_describes_zeroed_at_as_the_call():
    text = " ".join(README.read_text(encoding="utf-8").split())
    assert "zero made at line" not in text
    assert "(`scoring.py:38`, caught by the call at line 131)" in text


def test_the_readme_says_a_strong_name_outweighs_a_file():
    """`completion = fh.read()` is reported: a file of saved completions is
    model output. The README used to say any parse of a file was exempt."""
    assert _lines('''
        import json

        def rescore(path):
            with open(path) as fh:
                completion = fh.read()
            try:
                return json.loads(completion)
            except ValueError:
                return 0
    ''') == [7]
    text = " ".join(README.read_text(encoding="utf-8").split())
    assert "unless a name or key on the way names model output" in text


# ===================================================================
# The check of 9b41c99 (2026-10-22): two checkers broke the fixed scanner
# again. Each case below failed against 9b41c99's scanner (sha256
# 6b8cccef...), except the controls, which pass on both. Synthetic, like
# the rest of this file: a development set.
# ===================================================================

# ------------------------- a fence that is only mentioned handles nothing
@pytest.mark.parametrize("extra", [
    # prompts that ask for a fence, then call the model
    'prompt = f"Rate it. Return your answer in a ```json code block."\n'
    '    completion = client.complete(prompt)',
    'messages = [{"role": "system", "content": "Reply in ```json ... ```"}]\n'
    '    completion = client.chat(messages)',
    'prompt = "Answer with a bare JSON object. Do not use ``` fences.\\n"\n'
    '    n_lines = len(completion.splitlines())',
    # error text that is not raised or logged
    'reason = {"score": 0, "error": "a ```json fence is not stripped"}',
    'assert completion, "empty reply; a ```json fence?"',
    'sys.stderr.write("answer was not JSON (a ```json fence?)\\n")',
    'click.echo("got ```json, scoring 0")',
    'errors.append("model wrapped its answer in ```json; scored 0")',
    # a test whose branch scores the reply, or replaces it with a sentinel
    'if completion.strip().startswith("```json"):\n        return 0',
    'if re.search(r"```json", completion):\n        return 0',
    'if completion.startswith("```"):\n        completion = "INVALID"',
    # calls that render markdown, or only contain the word
    'print(pd.DataFrame([1]).to_markdown())',
    'st.markdown("## Result")',
    'write_markdown_report(completion)',
    'html = markdown(completion)',
    'markdown_rows.append("| x |")',
    'code_blocks.extend(re.findall(r"```python\\n(.*?)```", completion, re.S))',
    'completion = client.generate(apply_defence(completion))',
])
def test_a_fence_that_is_only_mentioned_does_not_exempt(extra):
    """9b41c99 still exempted the whole function for any ```json literal
    outside `raise` and log calls, and for any call whose name or
    receiver contained `markdown`, `fence` or `code_block`: prompts,
    returned error text, a detect-and-zero test, `st.markdown`,
    `df.to_markdown()`, a `markdown_rows` list, `apply_defence`."""
    src = textwrap.dedent(BASE.format(extra=extra)).lstrip("\n").replace(
        "def score(completion):",
        "def score(completion, client=None, errors=(), markdown_rows=(),\n"
        "          code_blocks=()):")
    site = 1 + src.split("\n").index("        return json.loads(completion)")
    assert [f["line"] for f in fc.scan_source(src, "m.py")] == [site], extra


@pytest.mark.parametrize("handling", [
    'completion = completion.split("```json")[1].split("```")[0]',
    'completion = completion.replace("```json", "").replace("```", "")',
    'm = re.search(r"```(?:json)?\\s*(.*?)\\s*```", completion, re.S)\n'
    '    completion = m.group(1) if m else completion',
    'if completion.startswith("```json"):\n'
    '        completion = completion[7:]\n'
    '    if completion.endswith("```"):\n'
    '        completion = completion[:-3]',
    'lines = [l for l in completion.splitlines() if not l.startswith("```")]\n'
    '    completion = "\\n".join(lines)',
    'completion = completion.replace(FENCE_TAG, "").strip("`")',
    'for prefix in ("```json", "```"):\n'
    '        if completion.startswith(prefix):\n'
    '            completion = completion[len(prefix):]',
    'm = FENCED.search(completion)\n'
    '    completion = m.group(1) if m else completion',
    'blocks = re.findall(r"%s(.*)%s" % ("```json", "```"), completion, re.S)\n'
    '    completion = blocks[0] if blocks else completion',
    'completion = fences.strip(completion)',
    'for f in FENCES:\n        completion = completion.replace(f, "")',
    'if completion.lstrip().startswith("```"):\n'
    '        completion = tidy(completion)\n'
    '    completion = completion.split("\\n", 1)[-1]',
    'if m := re.search(r"```json(.*?)```", completion, re.S):\n'
    '        completion = m.group(1)',
])
def test_fence_handling_in_code_still_exempts_when_it_cuts(handling):
    """The controls: code that removes, splits on or matches the fence
    still exempts, inline, through a module constant or a loop over
    prefixes, through a compiled pattern, and through a module named for
    fences."""
    src = ('import re\nfrom . import fences\n'
           'FENCE_TAG = "```json"\n'
           'FENCES = [FENCE_TAG, "```"]\n'
           'FENCED = re.compile(r"```json\\s*(.*?)```", re.S)\n'
           + textwrap.dedent(BASE.format(extra=handling)).lstrip("\n"))
    assert fc.scan_source(src, "m.py") == [], handling


# ------------------------------------- a fence helper passed by reference
@pytest.mark.parametrize("src", [
    '''
    import json
    from .utils import extract_json

    def score_all(df):
        out = []
        for completion in df["completion"].apply(extract_json):
            try:
                out.append(json.loads(completion)["score"])
            except ValueError:
                out.append(0)
        return out
    ''',
    '''
    import json
    from .utils import strip_code_fence

    def score(completion, postprocess=strip_code_fence):
        completion = postprocess(completion)
        try:
            return json.loads(completion)
        except ValueError:
            return 0
    ''',
    '''
    import json

    class Scorer:
        def _strip_fence(self, text):
            return text.strip().removeprefix("```json").removesuffix("```")

        def score(self, completion):
            for step in [self._strip_fence, str.strip]:
                completion = step(completion)
            try:
                return json.loads(completion)
            except ValueError:
                return 0
    ''',
])
def test_a_fence_helper_passed_by_reference_is_handling(src):
    assert _lines(src) == [], src


def test_a_variable_named_for_a_fence_is_not_a_helper():
    assert _lines('''
        import json

        def score(completion, strip_fences=True):
            if strip_fences:
                print("not implemented")
            try:
                return json.loads(completion)
            except ValueError:
                return 0
    ''') == [7]


# ------------------------------ a plain cleaning helper, whatever it calls
@pytest.mark.parametrize("param,body", [
    ("completion", 'return completion.strip().removeprefix("```json")'
                   '.removesuffix("```")'),
    ("prediction", 'return prediction.split("\\n", 1)[1].rsplit("```", 1)[0]'
                   ' if prediction.startswith("```") else prediction'),
    ("llm_output", 'return llm_output.replace("```json", "").replace("```", "")'),
    ("resp", 'return resp.choices[0].message.content.strip()'
             '.removeprefix("```json").removesuffix("```")'),
])
def test_a_cleaning_helper_is_handling_whatever_its_parameter_is_called(
        param, body):
    """`completion.strip()` inside the helper was read as a model call,
    because `completion` is a strong word, and so was
    `resp.choices[0]...strip()`, so the helper did not count."""
    src = (f"import json\n\ndef clean({param}):\n    {body}\n\n"
           "def score(completion):\n"
           "    text = clean(completion)\n"
           "    try:\n"
           "        return json.loads(text)\n"
           "    except ValueError:\n"
           "        return 0\n")
    assert fc.scan_source(src, "m.py") == [], param


@pytest.mark.parametrize("call", [
    'client.messages.create(model="m", max_tokens=9, messages=['
    '{"role": "user", "content": question + "\\nReply in a ```json block."}])'
    '.content[0].text',
    'client.responses.create(model="m", input=question + '
    '"\\nReturn your answer in a ```json code block.").output_text',
])
def test_a_provider_sdk_wrapper_is_not_a_cleaning_helper(call):
    """A wrapper whose prompt asks for ```json and that calls the model
    through `client.messages.create` or `client.responses.create` was
    taken for a fence-cleaning helper: its prompt literal made it
    'fence aware' and those calls were not seen as model calls."""
    src = (f"import json\n\ndef ask(client, question):\n    return {call}\n\n"
           "def score(client, question, gold):\n"
           "    try:\n"
           "        answer = json.loads(ask(client, question))\n"
           "    except json.JSONDecodeError:\n"
           "        return 0\n"
           "    return int(answer == gold)\n")
    assert [f["line"] for f in fc.scan_source(src, "grader.py")] == [8]


# ------------------------------------------ recovery: the failure branch only
RECOVERY = '''
import json

def score(completion, gold, stats, metrics):
    try:
        data = json.loads(completion)
    except ValueError:
        data = None
{after}
'''


@pytest.mark.parametrize("after", [
    "    if data is None:\n        log_parse_failure(completion)\n        return 0",
    "    if data is None:\n        record_parse_error(completion)\n        return 0",
    "    if data is None:\n        save_unparseable(completion)\n        return 0",
    "    if data is None:\n        stats.count_parse_errors(completion)\n        return 0",
    "    if data is None:\n        upload_failed_sample(completion)\n        return 0",
    "    if data is None:\n        metrics.mark_parse_failed(completion)\n        return 0",
    "    if data is None:\n        data = {\"score\": 0, \"raw\": completion}\n    return data",
    "    if data is None:\n        data = Score(value=0, explanation=completion)\n    return data",
    "    if data is not None:\n        return {\"correct\": data == gold, \"completion\": completion}\n"
    "    return {\"correct\": False, \"completion\": completion}",
    "    if data:\n        return compare(data, gold, completion)\n    return 0",
])
def test_a_zero_with_the_raw_text_beside_it_is_still_a_zero(after):
    """9b41c99 took any call naming pars/match/load in the `if` branch,
    any re-assignment that mentions the raw text, and either branch of the
    `if` as recovery from the None."""
    assert [f["line"] for f in fc.scan_source(
        RECOVERY.format(after=after), "m.py")] == [6], after


@pytest.mark.parametrize("after", [
    "    if data is None:\n        data = parse_loose(completion)\n    return data",
    "    if data is None:\n        m = re.search(r'\\{.*\\}', completion)\n"
    "        return m.group(0) if m else completion\n    return data",
    "    if data is not None:\n        return data\n    else:\n        return completion",
])
def test_a_real_recovery_is_still_not_a_zero(after):
    assert fc.scan_source(RECOVERY.format(after=after), "m.py") == [], after


def test_many_sentinels_in_one_function_are_linear():
    """Each `try ... except: data = None` re-walked every later statement
    of the function: 1,000 of them took 6.4 s."""
    out = ["import json", "", "def score_all(completions):", "    total = 0"]
    for i in range(1000):
        out += ["    try:", f"        data = json.loads(completions[{i}])",
                "    except ValueError:", "        data = None",
                "    if data is None:", "        total += 1"]
    started = time.process_time()
    found = fc.scan_source("\n".join(out) + "\n", "m.py")
    assert time.process_time() - started < 3
    assert len(found) == 1000


# ------------------------------------------------------------- reading files
@pytest.mark.parametrize("cookie", ["idna", "undefined"])
def test_a_coding_cookie_that_cannot_decode_does_not_end_the_scan(
        tmp_path, cookie):
    (tmp_path / "a_cookie.py").write_bytes(
        f"# -*- coding: {cookie} -*-\nimport json\n".encode())
    (tmp_path / "z_site.py").write_text(
        textwrap.dedent(BASE.format(extra="pass")).lstrip("\n"))
    result = _run("scan", "--json", str(tmp_path))
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    report = json.loads(result.stdout)
    assert [(f["file"], f["line"]) for f in report["findings"]] == [("z_site.py", 6)]


def test_parsing_warnings_do_not_reach_stderr():
    """Python 3.12+ printed `<unknown>:2: SyntaxWarning: invalid escape
    sequence` for every such string, in among the not-analysed list."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fc.scan_source('import json, re\nP = re.compile("\\d+")\n'
                       'x = json.loads("1")\n', "m.py")
    assert caught == []


# ------------------------------------------- the module-context gate, again
@pytest.mark.parametrize("src", [
    "import ast\n\ndef to_obj(output):\n    try:\n"
    "        return ast.literal_eval(output)\n    except ValueError:\n"
    "        return None\n",
    "import json\n\ndef to_dict(output, expr):\n    value = eval(expr)\n"
    "    try:\n        return json.loads(output)\n    except ValueError:\n"
    "        return {}\n",
])
def test_the_parse_call_does_not_vouch_for_its_own_module(src):
    """`literal_eval` split into `literal` and `eval`, and `eval` made the
    module 'about evaluation'; so did the `eval(` builtin."""
    assert fc.scan_source(src, "util.py") == []
    assert fc.scan_source('"""Model evaluation helpers."""\n' + src,
                          "util.py") != []


@pytest.mark.parametrize("provider", ["mistralai", "cohere", "groq"])
def test_a_listed_provider_gives_context(provider):
    src = (f"import json\nimport {provider}\n\ndef parse(response):\n"
           "    try:\n        return json.loads(response)\n"
           "    except ValueError:\n        return None\n")
    assert [f["line"] for f in fc.scan_source(src, "client.py")] == [6]


def test_the_readme_lists_every_provider_name():
    text = README.read_text(encoding="utf-8")
    for name in sorted(fc._PROVIDERS):
        assert f"`{name}`" in text, name


def test_the_hook_and_a_file_scan_agree_with_a_scan_of_the_repository(
        tmp_path):
    """The directory names that say a module is server or cache code
    were read relative to the scan root, which for one file is the file
    itself: the hook reported a file a scan of the repository cleared."""
    body = textwrap.dedent('''
        import json

        def grade(response):
            try:
                return json.loads(response)["score"]
            except ValueError:
                return 0
    ''').lstrip("\n")
    for folder, expect in (("server", 0), ("scoring", 1)):
        (tmp_path / "proj" / folder).mkdir(parents=True)
        (tmp_path / "proj" / folder / "grader.py").write_text(body)
        rel = f"proj/{folder}/grader.py"
        codes = [_run("scan", "--json", "proj", cwd=tmp_path).returncode,
                 _run("scan", "--json", rel, cwd=tmp_path).returncode,
                 subprocess.run([sys.executable, str(HOOK), rel], cwd=tmp_path,
                                capture_output=True, text=True, timeout=60,
                                check=False).returncode]
        assert codes == [expect] * 3, (folder, codes)


# ------------------------------------------------------- model-call verbs
@pytest.mark.parametrize("producer", [
    "generate_json(prompt)", "generate_samples(prompt)",
    "generate_structured_output(prompt)", "generate_and_parse(prompt)",
    "chat_json(prompt)", "chat_complete(prompt)", "complete_json(prompt)",
    "complete_chat(prompt)", "openai_generate(prompt)", "predict_json(prompt)",
    "client.invoke_model(prompt)", "judge.judge_pair(prompt, prompt)",
    "agent.chat_turn(prompt)", "self.client.generate_json(prompt)",
    "self.consult_judge(prompt)", "self.invoke_judge(prompt)",
    "self.request_judge(prompt)", "self.send_to_judge(prompt)",
])
def test_a_model_call_by_another_name_is_still_a_model_call(producer):
    """9b41c99 accepted a verb only before a closed list of words, so
    `generate_json`, `invoke_model` and `consult_judge` were not model
    calls; the scanner before it caught them all."""
    assert _lines(f'''
        import ast

        def load(self, client, judge, agent, prompt):
            text = {producer}
            try:
                return ast.literal_eval(text)
            except ValueError:
                return None
    ''') == [6], producer


@pytest.mark.parametrize("producer", [
    "complete_task(key)", "chat_template(data)", "apply_chat_template(data)",
    "get_judge(key)", "build_judge(key)", "generate_uuid(key)",
])
def test_a_model_word_that_does_not_run_a_model_is_not_a_model_call(producer):
    assert _lines(f'''
        import ast

        def load(data, key):
            text = {producer}
            try:
                return ast.literal_eval(text)
            except ValueError:
                return None
    ''') == [], producer


# ------------------------------------------------------------ method calls
def test_one_call_up_follows_a_sibling_mixin():
    src = '''
import json

class ParseMixin:
    def _parse(self, completion):
        return json.loads(completion)

class ScoreMixin:
    def score(self, completion):
        try:
            return self._parse(completion)
        except ValueError:
            return 0

class Grader(ParseMixin, ScoreMixin):
    pass
'''
    assert [(f["line"], f.get("zeroed_at")) for f in
            fc.scan_source(src, "m.py")] == [(6, 11)]


def test_one_call_up_stops_at_an_override_in_between():
    src = '''
import json
import re

class A:
    def _parse(self, text):
        return json.loads(text)

class B(A):
    def _parse(self, text):
        m = re.search(r"\\{.*\\}", text, re.S)
        return json.loads(m.group(0) if m else text)

class C(B):
    def score(self, completion):
        try:
            return self._parse(completion)
        except ValueError:
            return 0
'''
    assert fc.scan_source(src, "m.py") == []


def test_two_classes_that_share_a_name_are_not_one_class():
    src = '''
import json
import re

def make_a():
    class Judge:
        def _parse(self, text):
            return json.loads(text)

        def score(self, completion):
            return self._parse(completion)
    return Judge

def make_b():
    class Judge:
        def _parse(self, text):
            return json.loads(re.search(r"\\{.*\\}", text, re.S).group(0))

        def score(self, completion):
            try:
                return self._parse(completion)
            except ValueError:
                return 0
    return Judge
'''
    assert fc.scan_source(src, "m.py") == []


def test_a_long_chain_of_classes_is_linear():
    """Resolving `self._parse()` by method resolution order, once per
    class in a 1,000-class chain, must not walk the chain once per class."""
    out = ["import json", "", "class C0:", "    def _parse(self, completion):",
           "        return json.loads(completion)", ""]
    for i in range(1, 1000):
        out += [f"class C{i}(C{i - 1}):", f"    def score{i}(self, completion):",
                "        try:", "            return self._parse(completion)",
                "        except ValueError:", "            return 0", ""]
    started = time.process_time()
    found = fc.scan_source("\n".join(out) + "\n", "m.py")
    assert time.process_time() - started < 3
    assert [(f["line"], f["zeroed_at"]) for f in found] == [(5, 10)]


# ------------------------------------------------- a trace cut short
def _chain(last: str) -> str:
    lines = ["import json", "", "def h():", "    pass", "", "def score(path):",
             "    v8 = h()"]
    for i in range(7, -1, -1):
        lines += [f"    v{i} = v{i + 1}"] * 2
    lines += ["    with open(path) as fh:", "        v0 = fh.read()",
              f"    {last} = v0", "    try:", f"        return json.loads({last})",
              "    except ValueError:", "        return 0"]
    return "\n".join(lines) + "\n"


def test_a_trace_cut_short_is_not_weak_evidence():
    """The step budget stopped the trace before the file read that clears
    it, and the weak name `output` alone made a finding."""
    assert fc.scan_source(_chain("output"), "scorer.py") == []
    # a strong name nearer the parse still decides
    assert [f["evidence"] for f in
            fc.scan_source(_chain("completion"), "scorer.py")] == ["completion"]


# ------------------------------------------------------------------ score
def test_the_wider_scope_is_linear_on_unclosed_nesting():
    for text in ('note: ' + '{"a":[' * 4000, 'note: ' + '{"a":' * 4000):
        started = time.process_time()
        assert fc.extract_json(text, "any")[1] == "none"
        assert time.process_time() - started < 0.5
    assert fc.extract_json('x {"a": 1} y {"b": 2}', "any")[1] == "none"
    assert fc.extract_json('{bad} then {"a": 1}', "any")[1] == "none"


def test_score_survives_a_deeply_nested_output(tmp_path):
    """Python 3.11 raises RecursionError on 1,200 `[`, and `score` ended
    with a traceback."""
    assert fc._parses("[" * 3000) is False
    path = tmp_path / "p.jsonl"
    rows = [{"completion": '```json\n{"a": 1}\n```'}, {"completion": '{"a": 1}'},
            {"completion": "[" * 3000}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    deep = tmp_path / "deep.jsonl"
    deep.write_text("[" * 100000 + "\n" + '{"completion": "x"}\n')
    for target in (path, deep):
        for extra in ([], ["--json"], ["--scope", "any"]):
            result = _run("score", *extra, str(target))
            assert "Traceback" not in result.stderr, (target.name, extra)
            assert result.returncode in (0, 1, 2)
    assert _run("score", "--json", str(path)).returncode == 1
