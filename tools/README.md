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

## Licence

MIT. Copy the single file into your repo if that is easier than depending
on it.
