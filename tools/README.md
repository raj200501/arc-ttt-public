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
[`experiments/fence_rescore.json`](../experiments/fence_rescore.json).

## Run it

No installation. Python 3.9+. Standard library only. Run it from the
repository root, or copy `tools/fencecheck.py` anywhere and drop the
`tools/` prefix.

```
python3 tools/fencecheck.py scan     path/to/your/repo
python3 tools/fencecheck.py score    path/to/predictions.jsonl
python3 tools/fencecheck.py template path/to/checkpoint/
```

Once published, `pip install fencecheck` will install this same file
as a `fencecheck` command (`fencecheck scan path/to/your/repo`), plus
the pre-commit wrapper below as `fencecheck-hook`. It is not on PyPI
yet. Until it is, copy the file.

Try `score` before you have a file of your own. This makes three outputs
(one fenced, one bare, one fenced after prose), and `score` reports
1 of 3 as valid JSON a strict scorer rejects, with exit status 1:

~~~
printf '%s\n' '{"prediction": "```json\n{\"a\": 1}\n```"}' \
  '{"prediction": "{\"a\": 1}"}' \
  '{"prediction": "Sure:\n```json\n{\"a\": 1}\n```"}' > /tmp/p.jsonl
python3 tools/fencecheck.py score /tmp/p.jsonl
python3 tools/fencecheck.py score --scope any /tmp/p.jsonl   # also credits the third
~~~

Try `scan` on a real finding: the README's `evals` example, from the
pinned release. The download is 44 MB. The full archive unpacks to
about 0.7 GB, so this extracts only the `.py` files (about 3 MB):

```
curl -sSLO https://files.pythonhosted.org/packages/source/e/evals/evals-3.0.1.post1.tar.gz
python3 -c "import tarfile; t = tarfile.open('evals-3.0.1.post1.tar.gz'); t.extractall(members=[m for m in t if m.name.endswith('.py')])"
python3 tools/fencecheck.py scan evals-3.0.1.post1    # 7 site(s) in 328 file(s), monitor.py:72 among them
```

`scan` reads your code and reports every place that parses model output
as JSON **without handling a fence** *and* **turns the failure into a
zero or a silent skip**. `score` reads your saved model outputs and tells
you how many are valid JSON that your scorer would reject.

Exit status is `1` when something is found, so it drops into CI:

```
- run: python3 tools/fencecheck.py scan src/
```

Add `--json` to either command for machine-readable output.

## Run it in your CI

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

The action is [`action.yml`](../action.yml) at the root of this
repository. It runs `python tools/fencecheck.py scan <path>` from its own
copy of this repository, so you copy no file. It installs nothing from
PyPI and pins no Python package, because the tool is standard library
only. Its one dependency is `actions/setup-python@v5`, which provides the
interpreter. The job fails when the scan finds a site, through the
tool's own exit status. Inputs:

- `path`: one file or directory to scan, relative to your checkout.
  Default `"."`. Add a step per path to scan several.
- `args`: extra arguments for `scan`, for example `--json`. Default `""`.
- `python-version`: passed to `actions/setup-python`. Default `"3.11"`.

`@main` follows this repository's main branch. Pin a full commit SHA if
the gate must not change under you.

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
moving reference. This repository has no release tags, so autoupdate
replaces `main` with the current commit SHA. The hook runs
[`fencecheck_hook.py`](fencecheck_hook.py), which applies the same
`scan` to every staged Python file and reports them together, and a
finding fails the commit. It uses pre-commit's `script` language: no
virtualenv and nothing installed. The manifest is
[`.pre-commit-hooks.yaml`](../.pre-commit-hooks.yaml).

### What it will and will not catch

Expect false alarms on a large codebase. The least flattering
measurement here: over seven actively maintained eval packages, the
scanner as it stood on 2026-08-31 surfaced **64** candidate sites and
hand adjudication confirmed **1**. The other 63 were rejected, most of
them telemetry and tracing code, file and config reads, or protocol
transport
([`experiments/census_extension_live_2026-08-31.json`](../experiments/census_extension_live_2026-08-31.json)).

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

- a parse that lets the failure out of its own function, unless a
  caller in the same file catches it and makes a zero; then the parse
  line is reported, with the caller's line as `zeroed_at` in `--json`;
- a parse whose input traces to a file, an HTTP or socket exchange, a
  process, a stream frame, a database row, a cache, configuration, the
  environment, the command line, a telemetry record, an error message, a
  gold answer or a prompt;
- a parse with no evidence of model output. Evidence is a model call
  (`generate`, `complete`, `invoke`, a judge), or a name or key such as
  `completion`, `choices`, `prediction` or `model_output`, or, in a
  module about models or evaluation, a name such as `response`, `output`,
  `answer` or `reply`. `--json` gives the word under `evidence`;
- a parse after fence handling: a fence helper, a regex that cuts a
  `{...}` span out of the text, or a strip that removes the backticks and
  the `json` tag. Removing ```` ```python ```` only, or the backticks
  only, is not handling: a ```` ```json ```` fence still gets through;
- a failure that falls back to the raw text or to another parser.

A zero is a literal `0`, `False`, `None` or empty value, a skipped item,
a `Score(value=0)` or `Score(value=INCORRECT)` style object, or a
verdict string such as `"no"` or `"tie"`.

**What that first exemption still misses.** `scan` follows a parse
failure one call up, inside the same file. If a helper parses strictly
and raises, and a caller in the same file catches that exception and
scores 0, the parse line is reported. This repository's own scorer has
that shape: `parse_json_object` in `src/arcttt/scoring.py` raises
`TextTaskFormatError`, and `score_text_output` catches it and returns
`micro_f1=0.0`. There it is deliberate (fail-closed, the frozen rule the
gates were scored under), and `scan` now reports it
(`scoring.py:38`, zero made at line 131). A zero made two calls up, or
by a caller in another file, is still not reported. Read any parse
helper that raises, and its callers in other files, by hand.

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
$ python3 tools/fencecheck.py score predictions.jsonl
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

Put a `# fencecheck: ignore` comment on the parse line, or anywhere in
the enclosing function. Only a comment counts; the same text inside a
string or docstring does not. A silenced site is still counted: `scan`
lists silenced sites before its verdict, and `--json` reports them under
`suppressed`, so a CI job can audit them. The marker covers the whole
function, including parses added to it later.

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
python3 fencecheck.py template path/to/checkpoint/    # or a whole HF cache
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

This is not hypothetical. It was found in **this repository's own banked
work** on 2026-09-16: `ibm-granite/granite-3.1-2b-instruct` renders
`strftime_now('%B %d, %Y')` into its system message, and two cells of one
experiment — compared against each other in a published result — had been
produced five days apart. The published sentence said the two arms were
byte-identical; nobody knew the arms had been sent different prompts. The
row is in [`CORRECTIONS.md`](../CORRECTIONS.md), with dated errata beside
the frozen protocols. Of the seven checkpoints this repository has run,
exactly one is affected — which is why a checker is worth more than a
habit of remembering.

The fix is the same one Addendum X uses: pass the system message
explicitly, pinned to a fixed value, and record it beside the outputs.

## A wider scope, opt-in

`score --scope any` also credits a fence that comes after prose and a
bare object inside prose (one complete object, nothing JSON-like around
it), and reports those counts beside the default numbers. The default
stays one leading fence, because that is the scope every published
number here was measured with; the wider scope is a policy choice you
make. On this repository's own 2,130 banked outputs it credits 3 more
(Addendum W).

## Measure what a parser does to saved outputs

`score` tells you how many of your outputs a strict parse rejects.
Two sibling scripts go one step further and run real parsers over
real outputs, the way Addendum U did on this repository's own banked
model text:

```
python3 tools/fence_corpus.py                       # every banked raw output -> one JSONL, SHA-banked manifest
PYTHONPATH=src python3 scripts/parser_robustness.py           # strict json.loads, autoevals, langchain, json_repair
PYTHONPATH=src python3 scripts/parser_robustness.py --panel ext   # instructor, smolagents, llama-index helpers
```

The builder is standard library only. The runner is not: it imports
each parser it measures. The banked readings record the versions they
ran under (`versions` in each artifact): autoevals 0.3.0, langchain-core
1.6.2 and json-repair 0.63.4 for the first panel
(`experiments/parser_robustness_2026-09-05.json`); instructor 1.16.0,
smolagents 1.26.0 and llama-index-core 0.14.24 for the second
(`experiments/parser_robustness_ext_2026-09-05.json`). Install those in
a fresh virtual environment before running it.

The builder refuses artifacts whose "prediction" is already a parsed
object, refuses a registry that contradicts an artifact's own labels,
and names absent artifacts rather than dropping them. The runner
refuses a corpus that does not match a fresh rebuild. A parser is one
function `text -> object | None`; add yours to the panel and the frozen
readings apply to it unchanged. Per-record statuses are banked, so
every rate in the artifact can be recomputed by a stranger.

## The other side: stop emitting invalid JSON in the first place

`jsongreedy.py` is the JSON-constrained greedy decoder this repository
measured with, as one file for any Hugging Face causal LM: at each step
the top-k candidate tokens are tried in logit order and the first that
keeps the output a valid JSON prefix is emitted; if none does, the
top-1 token goes out anyway and the fallback is counted, so it never
stalls and never invents text. Schema-blind by design.

This one is not standard library only: it needs torch and transformers.

```
python3 tools/jsongreedy.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..."
```

```python
from jsongreedy import generate
text, info = generate(model, tokenizer, [{"role": "user", "content": prompt}])
```

What it does and does not do is measured, not asserted, and the second
measurement went against it. Ladder II rung E7 (Qwen2.5-3B, both arms):
every invalid output removed, consistent with the constraint firing on
E6's syntax faults (11 of 12 removals carry a constrained step; one
vanished without one). Addendum V (five families, schema-only prompts,
fifty receipts each): **no family cleared the bar** — the remaining
invalid outputs are truncations at the token cap, which a syntax
constraint cannot close; the validator fired 0 times in four families
and, on Falcon3, on exactly its seven non-truncation faults, fixing six
(14 → 8, one short of the bar). Read both rows in
[`VERDICT.md`](../VERDICT.md) before using it; it addresses one fault
class — syntax — and nothing else. A valid object is not a correct
one. The decoding core is pinned byte-identical to
`src/arcttt/constrained_json.py`, the module every banked run used.

**Both of those rows compared this decoder against `model.generate`,
which varies the decoding path as well as the constraint — so they say
what the constraint was *credited with*, not what it *did*.** Addendum X
was preregistered to separate them: the same function with the
constraint toggled by one boolean, against `generate` with the
checkpoint's own generation defaults neutralised, on the same fifty
documents per family. Its bars, and the prediction that arms C and P are
byte-identical wherever the validator never fired, were frozen before
the arm ran — `docs/research/ADDENDUM_X_PROTOCOL.md`. The `enforce` flag that makes
the toggle possible defaults to the constrained behaviour, so nothing
already banked changes and the pinned core stays what it was.

Addendum X then withheld itself: its determinism gate found Phi-3-mini
not reproducing its banked cells, so nothing was read about the decoder
at all. Addendum Y measured that failure and found the decoder
reproduces byte-for-byte in one process on every family and dtype tested
(0 of 10 on every count), and that the one document that flaked decodes
differently at four threads and at one — reduction order at bfloat16
precision, not a random decoder. If you bank outputs from this decoder in
bfloat16, pin the thread count and record it beside them; the environment
record every cell here now carries is the minimum.

## Licence

MIT. Copy the single file into your repo if that is easier than depending
on it.
