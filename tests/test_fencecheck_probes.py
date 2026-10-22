"""Labelled probes for `fencecheck scan`: a development set, not evidence.

Each probe is a small module with one parse site marked `# SITE`. A
`defect` probe parses model output with no fence handling AND turns the
failure into a zero, a None, an empty value or a skipped item; the scan
must report that exact line. A `clean` probe must produce no finding at
all: a config or file read, an HTTP or RPC payload, telemetry, a gold
answer, a test file, an input that was already stripped, or a parse
whose failure is loud.

These probes were written before the scanner was changed and were not
edited afterwards to make it pass. They are a DEVELOPMENT set. Numbers
measured on them say nothing about precision or recall on code nobody
has looked at; only a held-out census can say that.

Probes the scanner still gets wrong are marked `xfail(strict=True)`
with the reason, so a miss stays visible and a fix shows up as a
failure to update the mark.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "fencecheck.py"

pytestmark = pytest.mark.skipif(not TOOL.exists(),
                                reason="the tool is not in this cut")


def _load():
    spec = importlib.util.spec_from_file_location("_fencecheck_probes", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _p(pid, label, path, why, src):
    return {"id": pid, "label": label, "path": path, "why": why,
            "src": textwrap.dedent(src).lstrip("\n")}


PROBES = [
    # ------------------------------------------------------------ defects
    _p("D01", "defect", "grader.py", "plural `completions`, zip loop, append(0.0)", '''
        import json

        def score_batch(completions, gold):
            scores = []
            for completion, target in zip(completions, gold):
                try:
                    pred = json.loads(completion)  # SITE
                except json.JSONDecodeError:
                    scores.append(0.0)
                    continue
                scores.append(float(pred == target))
            return scores
    '''),
    _p("D02", "defect", "parse.py", "plural `responses`, append(None)", '''
        import json

        def parse_responses(responses):
            parsed = []
            for r in responses:
                try:
                    parsed.append(json.loads(r))  # SITE
                except ValueError:
                    parsed.append(None)
            return parsed
    '''),
    _p("D03", "defect", "metrics.py", "plural `outputs`, orjson, continue", '''
        import orjson

        def exact_match(outputs, references):
            hits = 0
            for out, ref in zip(outputs, references):
                try:
                    obj = orjson.loads(out)  # SITE
                except orjson.JSONDecodeError:
                    continue
                hits += obj == ref
            return hits / len(outputs)
    '''),
    _p("D04", "defect", "lists.py", "plural `generations`, ast.literal_eval, append([])", '''
        import ast

        def extract_lists(generations):
            out = []
            for gen in generations:
                try:
                    out.append(ast.literal_eval(gen))  # SITE
                except (ValueError, SyntaxError):
                    out.append([])
            return out
    '''),
    _p("D05", "defect", "check.py", "`answer` parameter, return False", '''
        import json

        def check_answer(answer, expected_keys):
            try:
                obj = json.loads(answer)  # SITE
            except Exception:
                return False
            return all(k in obj for k in expected_keys)
    '''),
    _p("D06", "defect", "chat.py", "`msg.content` of an assistant message", '''
        import json

        def last_assistant_json(messages):
            for msg in reversed(messages):
                if msg.role == "assistant":
                    try:
                        return json.loads(msg.content)  # SITE
                    except json.JSONDecodeError:
                        return None
            return None
    '''),
    _p("D07", "defect", "judge.py", "`choices[0].message.content`, return {'score': 0}", '''
        import json
        from openai import OpenAI

        client = OpenAI()

        def judge(prompt):
            resp = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            try:
                verdict = json.loads(resp.choices[0].message.content)  # SITE
            except json.JSONDecodeError:
                return {"score": 0}
            return verdict
    '''),
    _p("D08", "defect", "gemini_grade.py", "`result.text` from generate_content, parsed = {}", '''
        import json

        def grade(model, item):
            result = model.generate_content(item["prompt"])
            try:
                parsed = json.loads(result.text)  # SITE
            except ValueError:
                parsed = {}
            return float(parsed.get("label") == item["label"])
    '''),
    _p("D09", "defect", "verdict.py", "pydantic model_validate_json on `completion`", '''
        from pydantic import BaseModel, ValidationError

        class Verdict(BaseModel):
            correct: bool

        def is_correct(completion: str) -> bool:
            try:
                return Verdict.model_validate_json(completion).correct  # SITE
            except ValidationError:
                return False
    '''),
    _p("D10", "defect", "loader.py", "json.load of io.StringIO(output_text)", '''
        import io
        import json

        def load_prediction(output_text):
            try:
                return json.load(io.StringIO(output_text))  # SITE
            except json.JSONDecodeError:
                return None
    '''),
    _p("D11", "defect", "blocks.py", "text block inside result['message']['content']", '''
        import json

        def extract(result):
            blocks = result["message"]["content"]
            for block in blocks:
                if block.get("type") == "text":
                    try:
                        return json.loads(block["text"])  # SITE
                    except json.JSONDecodeError:
                        continue
            return None
    '''),
    _p("D12", "defect", "classify.py", "`reply.content` from llm.invoke", '''
        import json

        def classify(llm, text):
            reply = llm.invoke(f"Classify: {text}. Answer in JSON.")
            try:
                return json.loads(reply.content)["label"]  # SITE
            except (json.JSONDecodeError, KeyError):
                return None
    '''),
    _p("D13", "defect", "score.py", "snake_case compound `raw_completion`", '''
        import json

        def score(sample, raw_completion):
            try:
                pred = json.loads(raw_completion)  # SITE
            except ValueError:
                return 0
            return int(pred == sample["target"])
    '''),
    _p("D14", "defect", "camel.py", "camelCase `modelResponse`, return {}", '''
        import json

        def parseModelResponse(modelResponse):
            try:
                return json.loads(modelResponse)  # SITE
            except Exception:
                return {}
    '''),
    _p("D15", "defect", "ajudge.py", "awaited acompletion, choices[...] via a local", '''
        import json

        class Judge:
            def __init__(self, client):
                self.client = client

            async def ajudge(self, messages):
                response = await self.client.acompletion(model="m", messages=messages)
                text = response.choices[0].message.content
                try:
                    return json.loads(text)["score"]  # SITE
                except (json.JSONDecodeError, KeyError, TypeError):
                    return 0.0
    '''),
    _p("D16", "defect", "helpers.py", "neutral helper `_safe_parse(text)` called with predictions", '''
        import json

        def _safe_parse(text):
            try:
                return json.loads(text)  # SITE
            except json.JSONDecodeError:
                return None

        def evaluate(predictions, references):
            parsed = [_safe_parse(p) for p in predictions]
            return sum(a == b for a, b in zip(parsed, references)) / len(references)
    '''),
    _p("D17", "defect", "monitor.py", "nested validator called with completion_fn output", '''
        import json

        def run_monitor(completion_fn, prompt, attempts=3):
            def is_valid(text):
                try:
                    json.loads(text)  # SITE
                    return True
                except json.JSONDecodeError:
                    return False

            for _ in range(attempts):
                out = completion_fn(prompt)
                if is_valid(out):
                    return json.loads(out)
            return {"suspicious": 0}
    '''),
    _p("D18", "defect", "plan.py", "yaml.safe_load on `generation`", '''
        import yaml

        def parse_plan(generation):
            try:
                plan = yaml.safe_load(generation)  # SITE
            except yaml.YAMLError:
                return []
            return plan.get("steps", [])
    '''),
    _p("D19", "defect", "bare_loads.py", "`from json import loads`", '''
        from json import loads

        def grade_response(response, gold):
            try:
                return float(loads(response) == gold)  # SITE
            except ValueError:
                return 0.0
    '''),
    _p("D20", "defect", "alias.py", "`import json as js`", '''
        import json as js

        def score_generation(generation):
            try:
                data = js.loads(generation)  # SITE
            except js.JSONDecodeError:
                data = None
            return 1.0 if data and data.get("ok") else 0.0
    '''),
    _p("D21", "defect", "scorer.py", "handler returns Score(value=0, ...)", '''
        import json

        class Score:
            def __init__(self, value, explanation=""):
                self.value = value
                self.explanation = explanation

        def json_scorer(state, target):
            completion = state.output.completion
            try:
                answer = json.loads(completion)  # SITE
            except json.JSONDecodeError as exc:
                return Score(value=0, explanation=f"unparseable: {exc}")
            return Score(value=int(answer == json.loads(target.text)))
    '''),
    _p("D22", "defect", "rescore.py", "saved records, `rec['prediction']`", '''
        import json

        def rescore(records):
            """records: saved model outputs, one dict per document."""
            total = 0.0
            for rec in records:
                try:
                    pred = json.loads(rec["prediction"])  # SITE
                except json.JSONDecodeError:
                    pred = None
                total += float(pred == rec["gold"])
            return total / max(len(records), 1)
    '''),

    # ------------------------------------------------------- true negatives
    _p("N01", "clean", "settings.py", "config file through a file handle", '''
        import json

        def load_settings(path):
            """Generation settings for the completion endpoint."""
            try:
                with open(path) as fh:
                    return json.load(fh)  # SITE
            except (OSError, json.JSONDecodeError):
                return {}
    '''),
    _p("N02", "clean", "disk_cache.py", "cache file through pathlib read_text", '''
        import json
        import pathlib

        def read_cache(cache_dir, key):
            """Cached completions live on disk as JSON."""
            p = pathlib.Path(cache_dir) / f"{key}.json"
            try:
                return json.loads(p.read_text())  # SITE
            except (FileNotFoundError, json.JSONDecodeError):
                return None
    '''),
    _p("N03", "clean", "leaderboard.py", "requests response body", '''
        import json
        import requests

        def fetch_leaderboard(url):
            response = requests.get(url, timeout=10)
            try:
                return json.loads(response.text)  # SITE
            except ValueError:
                return []
    '''),
    _p("N04", "clean", "models_api.py", "httpx response content", '''
        import json
        import httpx

        async def get_model_list(base_url):
            """List the models that can serve a completion."""
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{base_url}/v1/models")
            try:
                payload = json.loads(r.content)  # SITE
            except json.JSONDecodeError:
                return None
            return [m["id"] for m in payload["data"]]
    '''),
    _p("N05", "clean", "sse.py", "server-sent event lines (transport framing)", '''
        import json

        def iter_stream_events(response):
            """Yield parsed server-sent events from a streaming completion."""
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)  # SITE
                except json.JSONDecodeError:
                    continue
    '''),
    _p("N06", "clean", "rpc.py", "JSON-RPC frames read from a stream", '''
        import json
        import sys

        def read_rpc_messages(stream=sys.stdin):
            """JSON-RPC transport carrying model responses to the sandbox."""
            for raw in stream:
                try:
                    message = json.loads(raw)  # SITE
                except json.JSONDecodeError:
                    continue
                yield message
    '''),
    _p("N07", "clean", "trace_view.py", "telemetry span attribute, display only", '''
        import json

        def span_output(span):
            """Pull the recorded LLM output off a tracing span for display."""
            value = span.attributes.get("llm.output")
            try:
                return json.loads(value)  # SITE
            except (TypeError, json.JSONDecodeError):
                return None
    '''),
    _p("N08", "clean", "gold.py", "gold answer from the dataset sample", '''
        import json

        def compare(sample, completion):
            try:
                ideal = json.loads(sample["ideal"])  # SITE
            except json.JSONDecodeError:
                return 0
            return int(ideal == completion)
    '''),
    _p("N09", "clean", "tests/test_grader.py", "a test file, not shipped behaviour", '''
        import json

        def test_grader_handles_bad_completion():
            completion = "```json\\n{}\\n```"
            try:
                json.loads(completion)  # SITE
            except json.JSONDecodeError:
                pass
    '''),
    _p("N10", "clean", "stripped.py", "fence stripped by a helper before the parse", '''
        import json

        def strip_code_fence(text):
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\\n", 1)[1].rsplit("```", 1)[0]
            return text

        def parse_completion(completion):
            cleaned = strip_code_fence(completion)
            try:
                return json.loads(cleaned)  # SITE
            except json.JSONDecodeError:
                return None
    '''),
    _p("N11", "clean", "regex_fence.py", "fence removed by a module-level regex", '''
        import json
        import re

        FENCE = re.compile(r"```(?:json)?\\s*(.*?)```", re.S)

        def parse_answer(answer):
            m = FENCE.search(answer)
            body = m.group(1) if m else answer
            try:
                return json.loads(body)  # SITE
            except ValueError:
                return None
    '''),
    _p("N12", "clean", "loud.py", "no try: the failure reaches the caller", '''
        import json

        def grade(completion):
            obj = json.loads(completion)  # SITE
            return obj["answer"]
    '''),
    _p("N13", "clean", "reraise.py", "inner handler raises; outer handler is KeyError only", '''
        import json

        class GraderError(Exception):
            pass

        def grade(completion):
            try:
                try:
                    obj = json.loads(completion)  # SITE
                except json.JSONDecodeError as exc:
                    raise GraderError(f"unparseable completion: {exc}") from exc
                return float(obj["correct"])
            except KeyError:
                return 0.0
    '''),
    _p("N14", "clean", "lookup.py", "handler cannot catch a decode error", '''
        import json

        def lookup(completion, table):
            try:
                key = json.loads(completion)["id"]  # SITE
                return table[key]
            except KeyError:
                return None
    '''),
    _p("N15", "clean", "cli.py", "command-line argument", '''
        import argparse
        import json

        def main():
            parser = argparse.ArgumentParser(description="Run the completion benchmark")
            parser.add_argument("--extra", default="{}")
            args = parser.parse_args()
            try:
                extra = json.loads(args.extra)  # SITE
            except json.JSONDecodeError:
                extra = {}
            return extra
    '''),
    _p("N16", "clean", "env_kwargs.py", "environment variable", '''
        import json
        import os

        def model_kwargs():
            """Extra kwargs for the completion call, from the environment."""
            raw = os.environ.get("COMPLETION_KWARGS", "{}")
            try:
                return json.loads(raw)  # SITE
            except ValueError:
                return {}
    '''),
    _p("N17", "clean", "errors.py", "payload inside an exception message", '''
        import json
        import re

        def error_detail(exc):
            """Pull the provider's error payload out of a completion API exception."""
            match = re.search(r"\\{.*\\}", str(exc))
            if not match:
                return None
            try:
                return json.loads(match.group(0))  # SITE
            except json.JSONDecodeError:
                return None
    '''),
    _p("N18", "clean", "runs_db.py", "database rows", '''
        import json
        import sqlite3

        def load_runs(db_path):
            """Load stored evaluation runs (model responses and scores)."""
            conn = sqlite3.connect(db_path)
            runs = []
            for (row_json,) in conn.execute("SELECT payload FROM runs"):
                try:
                    runs.append(json.loads(row_json))  # SITE
                except json.JSONDecodeError:
                    continue
            return runs
    '''),
    _p("N19", "clean", "env_record.py", "subprocess stdout", '''
        import json
        import subprocess

        def pip_list_json():
            """Record the environment beside each generation run."""
            out = subprocess.run(["pip", "list", "--format=json"],
                                 capture_output=True, text=True).stdout
            try:
                return json.loads(out)  # SITE
            except json.JSONDecodeError:
                return []
    '''),
    _p("N20", "clean", "ws_relay.py", "websocket protocol messages", '''
        import json

        async def consume(ws, on_event):
            """Relay streaming response events from the inference server."""
            async for message in ws:
                try:
                    event = json.loads(message)  # SITE
                except json.JSONDecodeError:
                    continue
                on_event(event)
    '''),
    _p("N21", "clean", "snapshot.py", "round-trip of our own object through json.dumps", '''
        import json

        def snapshot(result):
            """Deep-copy an evaluation result (model responses included) via JSON."""
            try:
                return json.loads(json.dumps(result, default=str))  # SITE
            except (TypeError, ValueError):
                return None
    '''),
    _p("N22", "clean", "server.py", "HTTP request body in a handler", '''
        import json

        def handle_submit(request):
            """Accept a model response submission for scoring."""
            try:
                payload = json.loads(request.body)  # SITE
            except json.JSONDecodeError:
                return None
            return {"status": 200, "id": payload.get("id")}
    '''),
    _p("N23", "clean", "nested_value.py", "a field of an object already parsed loudly", '''
        import json

        def tool_result(completion):
            outer = json.loads(completion)
            try:
                return json.loads(outer["value"])  # SITE
            except (TypeError, json.JSONDecodeError):
                return None
    '''),
]

# Probes the current scanner gets wrong, with the reason. Kept visible
# rather than deleted: a development set that only holds passes says
# nothing.
KNOWN_MISSES: dict[str, str] = {
    "D11": "the only evidence is a `message` key and the module names no "
           "model or evaluation word, so weak evidence is not enough",
}


def site_line(probe) -> int:
    for number, line in enumerate(probe["src"].split("\n"), start=1):
        if "# SITE" in line:
            return number
    raise AssertionError(f"{probe['id']} has no # SITE marker")


def scan_probe(fc, probe, root: pathlib.Path):
    target = root / probe["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(probe["src"], encoding="utf-8")
    findings, _ = fc.scan_path(root)
    return findings


def test_the_probe_set_is_labelled_and_large_enough():
    ids = [p["id"] for p in PROBES]
    assert len(ids) == len(set(ids))
    assert len(PROBES) >= 25
    assert {p["label"] for p in PROBES} == {"defect", "clean"}
    for probe in PROBES:
        site_line(probe)


def _cases():
    for probe in PROBES:
        marks = []
        if probe["id"] in KNOWN_MISSES:
            marks.append(pytest.mark.xfail(strict=True,
                                           reason=KNOWN_MISSES[probe["id"]]))
        yield pytest.param(probe, id=probe["id"], marks=marks)


@pytest.mark.parametrize("probe", list(_cases()))
def test_probe(probe, tmp_path):
    fc = _load()
    findings = scan_probe(fc, probe, tmp_path)
    if probe["label"] == "defect":
        lines = [f["line"] for f in findings]
        assert site_line(probe) in lines, (probe["why"], findings)
    else:
        assert findings == [], (probe["why"], findings)


# ------------------------------------------------------------------------
# Rules added with the probes, each pinned by the smallest case that needs
# it. These are regression tests, not part of the frozen probe counts.
# ------------------------------------------------------------------------
def _scan(fc, src, name="m.py"):
    return fc.scan_source(textwrap.dedent(src).lstrip("\n"), name)


def test_helper_raises_and_a_caller_in_the_same_file_scores_zero():
    """A stranger walkthrough: this repository's own scorer shape
    (src/arcttt/scoring.py) scanned clean, because the parse raises and the
    zero is made one call up."""
    fc = _load()
    found = _scan(fc, '''
        import json

        class FormatError(ValueError):
            pass

        def parse_json_object(text):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise FormatError(str(error)) from None
            return value

        def score_model_output(prediction_text, gold):
            try:
                predicted = parse_json_object(prediction_text)
            except FormatError:
                return 0.0
            return 1.0 if predicted == gold else 0.0
    ''')
    assert [(f["line"], f.get("zeroed_at")) for f in found] == [(8, 15)]


def test_a_caller_that_lets_the_error_out_is_not_a_finding():
    fc = _load()
    assert _scan(fc, '''
        import json

        def parse_json_object(text):
            return json.loads(text)

        def score_model_output(prediction_text, gold):
            return float(parse_json_object(prediction_text) == gold)
    ''') == []


def test_a_gold_caller_does_not_make_the_helper_a_finding():
    """The helper is also called on the gold text; only the call that
    makes the zero is traced."""
    fc = _load()
    found = _scan(fc, '''
        import json

        def parse(text):
            return json.loads(text)

        def score(completion, gold_text):
            gold = parse(gold_text)
            try:
                predicted = parse(completion)
            except ValueError:
                return 0.0
            return float(predicted == gold)
    ''')
    assert [f["line"] for f in found] == [4]


@pytest.mark.parametrize("zero", [
    "Score(value=0.0)", "Score(value=INCORRECT)", "Score(name='x', score=0)",
    "Score(0)", "{'score': 0}", "('incorrect', 0, '')", "'no'", "['no', 0]",
])
def test_scorer_style_zeros(zero):
    """The design-partner dry run: inspect- and autoevals-style scorers
    return a Score object, which used to read as 'not a zero'."""
    fc = _load()
    found = _scan(fc, f'''
        import json

        def judge_score(output):
            try:
                verdict = json.loads(output)
            except json.JSONDecodeError:
                return {zero}
            return verdict
    ''')
    assert [f["line"] for f in found] == [5], zero


def test_a_verdict_sentinel_assigned_in_the_handler():
    fc = _load()
    found = _scan(fc, '''
        import json

        def pairwise(judge, a, b):
            reply = judge.ask_judge(a, b)
            try:
                winner = json.loads(reply)["winner"]
            except (json.JSONDecodeError, KeyError):
                winner = "tie"
            return {"A": 1.0, "B": 0.0}.get(winner, 0.5)
    ''')
    assert [f["line"] for f in found] == [6]


def test_partial_fence_handling_is_not_fence_handling():
    """LiveBench tablejoin, 2024-06-14 to 2025-04-18: only ```python was
    extracted, the parse was a by-name `literal_eval`, failure returned {}
    and the caller scored 0.0. A ```json-fenced correct answer scored 0."""
    fc = _load()
    partial = '''
        import re
        from ast import literal_eval

        def clean_llm_output(s):
            matches = re.findall("%s(.*)%s" % ("```python", "```"), s)
            if len(matches) == 0:
                matches = [s]
            try:
                match_d = literal_eval(matches[-1])
            except:
                return {}
            return match_d

        def joinmap_process_results(_, ground_truth, llm):
            llm_clean = clean_llm_output(llm)
            if len(llm_clean) == 0:
                return 0.0
            return float(llm_clean == ground_truth)
    '''
    assert [f["line"] for f in _scan(fc, partial)] == [9]
    handled = partial.replace('if len(matches) == 0:\n',
                              'if len(matches) == 0:\n'
                              '                matches = re.findall('
                              '"```json(.*?)```", s)\n'
                              '            if len(matches) == 0:\n', 1)
    assert _scan(fc, handled) == []


@pytest.mark.parametrize("cleaning,reported", [
    ('text.strip().strip("`")', True),            # leaves "json\\n"
    ('text.replace("```", "")', True),            # leaves "json\\n"
    ('text.strip().strip("```").strip("json")', False),
    ('text.replace("```json", "").replace("```", "")', False),
    ('re.sub(r"```\\\\w*", "", text)', False),
    ('text.split("\\n", 1)[-1].rsplit("```", 1)[0]', False),
])
def test_which_fence_strips_count(cleaning, reported):
    fc = _load()
    found = _scan(fc, f'''
        import json
        import re

        def score(completion):
            text = completion
            cleaned = {cleaning}
            try:
                return float(json.loads(cleaned)["ok"])
            except (json.JSONDecodeError, KeyError):
                return 0.0
    ''')
    assert bool(found) is reported, (cleaning, found)


def test_an_anchored_brace_regex_does_not_tolerate_a_fence():
    fc = _load()
    src = '''
        import json
        import re

        OBJECT = re.compile(r"^\\\\s*(\\\\{.*\\\\})\\\\s*$", re.S)

        def extract_record(output):
            m = OBJECT.match(output)
            if not m:
                return {}
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return {}
    '''
    assert [f["line"] for f in _scan(fc, src)] == [11]
    unanchored = src.replace("OBJECT.match", "OBJECT.search").replace(
        'r"^', 'r"')
    assert _scan(fc, unanchored) == []


def test_an_http_response_is_known_by_how_it_is_used():
    fc = _load()
    assert _scan(fc, '''
        import json

        def _is_object(string):
            try:
                return isinstance(json.loads(string), dict)
            except Exception:
                return False

        def verify(response):
            if response.status_code != 200 and _is_object(response.text):
                raise RuntimeError(response.text)
    ''') == []


def test_a_fallback_after_pass_is_not_a_zero():
    fc = _load()
    assert _scan(fc, '''
        import json

        def normalize(output):
            try:
                return json.dumps(json.loads(output))
            except json.JSONDecodeError:
                pass
            return str(output)
    ''') == []


def test_tests_are_skipped_in_a_directory_but_not_when_named(tmp_path):
    fc = _load()
    src = textwrap.dedent('''
        import json

        def check(completion):
            try:
                return json.loads(completion)
            except ValueError:
                return None
    ''')
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_grader.py").write_text(src)
    (tmp_path / "test_top.py").write_text(src)
    findings, files = fc.scan_path(tmp_path)
    assert (findings, files) == ([], 0)
    named, files = fc.scan_path(tmp_path / "tests" / "test_grader.py")
    assert files == 1 and len(named) == 1


def test_suppression_is_a_comment_and_is_counted(tmp_path, capsys):
    """The dry run: a marker inside a docstring suppressed, and nothing
    reported how many sites were silenced."""
    fc = _load()
    body = '''
        import json

        def check(completion):
            {doc}
            try:
                return json.loads(completion){mark}
            except ValueError:
                return None
    '''
    in_doc = body.format(doc='"""# fencecheck: ignore"""', mark="")
    assert len(_scan(fc, in_doc)) == 1
    marked = textwrap.dedent(body.format(
        doc='"""Strict on purpose."""',
        mark="  # fencecheck: ignore -- strict on purpose"))
    suppressed = []
    assert fc.scan_source(marked, "m.py", suppressed) == []
    assert len(suppressed) == 1
    (tmp_path / "m.py").write_text(marked)
    assert fc.main(["scan", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 site(s) silenced" in out
    assert fc.main(["scan", "--json", str(tmp_path)]) == 0
    assert len(json.loads(capsys.readouterr().out)["suppressed"]) == 1
