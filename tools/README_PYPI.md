# fencecheck

**Is your evaluation scoring a correct answer as zero?**

When a language model returns a JSON object wrapped in a markdown code
fence, code that calls `json.loads` on it gets a parse failure. If that
failure becomes a score, a correct answer becomes a zero.

Here is what that costs, measured:

| Qwen2.5-3B-Instruct, 30 held-out documents, field list only | micro-F1 | invalid |
|---|---|---|
| scored as the model emitted it | **0.0000** | 30 / 30 |
| identical outputs, one fence removed | **0.8958** | 0 / 30 |

Same model, same prompt, same documents, same scorer. The only difference
is three backticks. Banked at
[`experiments/fence_rescore.json`](https://github.com/raj200501/arc-ttt-public/blob/main/experiments/fence_rescore.json)
in the source repository.

## Run it

Python 3.9+. Standard library only. The package installs one file,
`fencecheck.py`, as the `fencecheck` command, plus the pre-commit wrapper
`fencecheck-hook`.

```
pip install fencecheck
fencecheck scan     path/to/your/repo
fencecheck score    path/to/predictions.jsonl
fencecheck template path/to/checkpoint/
```

You can skip the install: copy
[`tools/fencecheck.py`](https://github.com/raj200501/arc-ttt-public/blob/main/tools/fencecheck.py)
anywhere and run `python3 fencecheck.py scan path/to/your/repo`.

Try `score` before you have a file of your own. This makes three outputs
(one fenced, one bare, one fenced after prose), and `score` reports
1 of 3 as valid JSON a strict scorer rejects, with exit status 1:

~~~
printf '%s\n' '{"prediction": "```json\n{\"a\": 1}\n```"}' \
  '{"prediction": "{\"a\": 1}"}' \
  '{"prediction": "Sure:\n```json\n{\"a\": 1}\n```"}' > /tmp/p.jsonl
fencecheck score /tmp/p.jsonl
fencecheck score --scope any /tmp/p.jsonl   # also credits the third
~~~

Try `scan` on a real finding: the `evals` package, from the pinned
release. The download is 44 MB. The full archive unpacks to about
0.7 GB, so this extracts only the `.py` files (about 3 MB):

```
curl -sSLO https://files.pythonhosted.org/packages/source/e/evals/evals-3.0.1.post1.tar.gz
python3 -c "import tarfile; t = tarfile.open('evals-3.0.1.post1.tar.gz'); t.extractall(members=[m for m in t if m.name.endswith('.py')])"
fencecheck scan evals-3.0.1.post1    # 7 site(s) in 328 file(s), monitor.py:72 among them
```

`scan` reads your code and reports every place that parses model output
as JSON **without handling a fence** *and* **turns the failure into a
zero or a silent skip**. `score` reads your saved model outputs and tells
you how many are valid JSON that your scorer would reject.

Exit status is `1` when something is found, so it drops into CI:

```
- run: pip install fencecheck && fencecheck scan src/
```

Pin the version if the gate must not change under you. Add `--json` to
either command for machine-readable output.

## Run it in your CI without installing anything

One `uses:` line in a GitHub workflow, after the checkout:

```yaml
name: fencecheck
on: [push, pull_request]
jobs:
  fencecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: raj200501/arc-ttt-public@main
        with:
          path: .   # optional; narrow it to your scoring code
```

The action is
[`action.yml`](https://github.com/raj200501/arc-ttt-public/blob/main/action.yml)
at the root of the source repository. It runs
`python tools/fencecheck.py scan <path>` from its own copy of that
repository, so you copy no file. It installs nothing from PyPI and pins
no Python package, because the tool is standard library only. Its one
dependency is `actions/setup-python@v5`, which provides the interpreter.
The job fails when the scan finds a site, through the tool's own exit
status. Inputs:

- `path`: one file or directory to scan, relative to your checkout.
  Default `"."`. Add a step per path to scan several.
- `args`: extra arguments for `scan`, for example `--json`. Default `""`.
- `python-version`: passed to `actions/setup-python`. Default `"3.11"`.

`@main` follows the source repository's main branch. Pin a full commit
SHA if the gate must not change under you.

As a [pre-commit](https://pre-commit.com/) hook, in
`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/raj200501/arc-ttt-public
    rev: main
    hooks:
      - id: fencecheck
```

Then run `pre-commit autoupdate` once. pre-commit warns that `main` is a
moving reference. The hook runs
[`fencecheck_hook.py`](https://github.com/raj200501/arc-ttt-public/blob/main/tools/fencecheck_hook.py),
the same file this package installs as `fencecheck-hook`. It applies the
same `scan` to every staged Python file and reports them together, and a
finding fails the commit. It uses pre-commit's `script` language: no
virtualenv and nothing installed. The manifest is
[`.pre-commit-hooks.yaml`](https://github.com/raj200501/arc-ttt-public/blob/main/.pre-commit-hooks.yaml).

### What it will and will not catch

Expect most findings on a large codebase to be false alarms. The least
flattering measurement here: over seven actively maintained eval
packages, `scan` surfaced **64** candidate sites and hand adjudication
confirmed **1**. The other 63 were rejected, most of them telemetry and
tracing code, file and config reads, or protocol transport
([`experiments/census_extension_live_2026-08-31.json`](https://github.com/raj200501/arc-ttt-public/blob/main/experiments/census_extension_live_2026-08-31.json)).
The detection rules changed on 2026-10-21, after that run. A finding now
needs evidence that the parsed text is a model's output (see below). The
changes were tuned on development sets only: synthetic probes, packages
already scanned, and repositories already read by hand. That tuning is
not evidence that precision or recall improved. The held-out census
(Addendum AB, `scripts/heldout_census.py`) is the measurement. So point
`path` at your scoring code rather than the whole repository, read each
finding, and mark the deliberate ones with `# fencecheck: ignore`.

It will not catch:

- a file that does not end in `.py`, or a test file (`tests/`,
  `test_*.py`, `*_test.py`, `conftest.py`) when it scans a directory;
- a parse whose input it cannot trace to model output inside the same
  file, for example a parameter called `text` that no caller in the file
  fills with model output;
- a zero made more than one call up, or in another file;
- partial fence handling it cannot tell apart from full handling, for
  example a regex for the `json` tag that misses `JSON`;
- a wrong score with no exception at all, such as a validity check
  followed by a string-against-dict comparison;
- a parse whose failure reaches the top of its caller too, which is loud
  and left out on purpose.
- a file nested too deeply for Python's parser to analyse (a very long
  expression, for example); it is skipped, never cleared, and named on
  stderr and under `unanalysed` in `--json`.

How many real sites it misses on code nobody has looked at has not been
measured.

## What a finding means, and what it does not

**A finding is not a bug report.** Parsing without fence handling is not
by itself a defect — a library entitled to clean input is entitled to
assume it. The defect is parsing without fence handling *and then
converting the failure into a score*, because that is the combination
that silently turns a correct answer into a zero.

So `scan` reports only the conjunction, and it prints the file and line
for every finding so you can decide in ten seconds. It deliberately does
**not** flag:

- a parse that lets the failure out of its own function;
- a parse whose input is a config file or a saved artifact rather than
  model output;
- anything in a module that never mentions model output at all.

**What that first exemption misses.** `scan` reads one function at a
time. If a helper parses strictly and raises its own exception, and the
caller catches that exception and scores 0, the zero is one call away
and `scan` does not report it. The source repository's own scorer has
exactly that shape: `parse_json_object` in `src/arcttt/scoring.py` raises
`TextTaskFormatError`, and `score_text_output` catches it and returns
`micro_f1=0.0`. There it is deliberate (fail-closed, the frozen rule the
gates were scored under), and `scan` reports nothing on that file. The
same logic written as one function is reported. Read any parse helper
that raises, and its callers, by hand.

## Why this happens, and to whom

Instruction-tuned chat models fence heavily. Models fine-tuned on a
strict output format do not. So a scorer that rejects fenced output does
not fail at random — it is systematically harsher on chat models than on
format-trained ones, which is exactly the axis most "small open model vs.
frontier model" comparisons are measured along.

Whether that bias shows up in published work is a separate question that
this tool does not answer. It answers a narrower one: *is it happening in
your repo, right now.*

## Example

```
$ fencecheck score predictions.jsonl
fencecheck: 30 output(s)
  fenced                  30
  parse as written        0
  parse after stripping   30

  30 of 30 outputs are VALID JSON that your scorer would reject.
  That is 100% of this file scoring zero for formatting rather than for content.
```

`score` accepts JSONL or a JSON list, and looks for the raw model text
under any of `prediction`, `output`, `completion`, `response`,
`generation`, `text`, `raw`, `answer`, `content` — or a bare string per
line.

## Silencing a deliberate one

Put `# fencecheck: ignore` on the parse line, or anywhere in the
enclosing function:

```python
def parses(text):
    # fencecheck: ignore -- strictness is the point here
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True
```

This tool flagged itself on its first run, on exactly that function. A
tool you cannot tell "yes, on purpose" gets deleted after the first false
positive.

## Is your prompt the same one you used last week?

```
fencecheck template path/to/checkpoint/    # or a whole HF cache
```

Some chat templates build their own system message and put the **current
date** in it. Nothing warns you. Outputs you banked on Tuesday and
outputs you banked on Friday were not produced from the same prompt, so
they cannot be compared with each other and neither can be reproduced
later.

`template` reads `tokenizer_config.json` or a bare `.jinja` template —
standard library only, no model libraries needed — and reports every
template that calls the clock, with the line and the surrounding text.
Exit status is `1` when something is found, so it drops into CI beside
the other two.

This is not hypothetical. It was found in **the source repository's own
banked work** on 2026-09-16: `ibm-granite/granite-3.1-2b-instruct` renders
`strftime_now('%B %d, %Y')` into its system message, and two cells of one
experiment — compared against each other in a published result — had been
produced five days apart. The published sentence said the two arms were
byte-identical; nobody knew the arms had been sent different prompts. The
row is in
[`CORRECTIONS.md`](https://github.com/raj200501/arc-ttt-public/blob/main/CORRECTIONS.md),
with dated errata beside the frozen protocols. Of the seven checkpoints
that repository has run, exactly one is affected — which is why a checker
is worth more than a habit of remembering.

The fix is the same one Addendum X uses: pass the system message
explicitly, pinned to a fixed value, and record it beside the outputs.

## A wider scope, opt-in

`score --scope any` also credits a fence that comes after prose and a
bare object inside prose (one complete object, nothing JSON-like around
it), and reports those counts beside the default numbers. The default
stays one leading fence, because that is the scope every published
number in the source repository was measured with; the wider scope is a
policy choice you make. On that repository's own 2,130 banked outputs it
credits 3 more (Addendum W).

## Not in this package

The source repository's
[`tools/`](https://github.com/raj200501/arc-ttt-public/tree/main/tools)
directory also holds `fence_corpus.py` and `jsongreedy.py`. They are not
installed by `pip install fencecheck`. `jsongreedy.py` needs torch and
transformers.

## Licence

MIT. Copy the single file into your repo if that is easier than depending
on it.
