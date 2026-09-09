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

No installation. Python 3.9+. Standard library only.

```
python3 fencecheck.py scan  path/to/your/repo
python3 fencecheck.py score path/to/predictions.jsonl
```

`scan` reads your code and reports every place that parses model output
as JSON **without handling a fence** *and* **turns the failure into a
zero or a silent skip**. `score` reads your saved model outputs and tells
you how many are valid JSON that your scorer would reject.

Exit status is `1` when something is found, so it drops into CI:

```
- run: python3 fencecheck.py scan src/
```

Add `--json` to either command for machine-readable output.

## What a finding means, and what it does not

**A finding is not a bug report.** Parsing without fence handling is not
by itself a defect — a library entitled to clean input is entitled to
assume it. The defect is parsing without fence handling *and then
converting the failure into a score*, because that is the combination
that silently turns a correct answer into a zero.

So `scan` reports only the conjunction, and it prints the file and line
for every finding so you can decide in ten seconds. It deliberately does
**not** flag:

- a parse that lets the failure reach the caller (you will see it);
- a parse whose input is a config file or a saved artifact rather than
  model output;
- anything in a module that never mentions model output at all.

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
$ python3 fencecheck.py score predictions.jsonl
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

## Licence

MIT. Copy the single file into your repo if that is easier than depending
on it.
