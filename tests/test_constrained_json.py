"""Ladder II, rung E7: the JSON-constrained decoder.

The validator is pinned on the EXACT fault classes the E6 adapted arm
exhibited (single-quoted keys, an extra closing brace, a second object
appended) plus the ordinary prefix cases. The decoder is smoke-tested on
the cached 0.5B: it must never emit an invalid document when it stops
on its own, and its fallback counter must be honest.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from arcttt.constrained_json import (constrained_greedy_generate,
                                     is_complete_json, is_json_prefix)


@pytest.mark.parametrize("prefix", [
    "", "{", '{"', '{"a"', '{"a":', '{"a": 1', '{"a": 1,', '{"a": 1, "b": [',
    '{"a": 1, "b": [1, 2', '{"a": {"b": "x\\"y"', '{"menu":[{"nm":"EGG TART","cnt":"1"}],"total":{"total_price":"45,500"}}',
    '[', '[1,', '[{"a": tr', '{"a": fal', '{"a": nu', '{"a": -1.5e',
    '{"a": "unterminated string is a fine prefix',
])
def test_valid_prefixes_are_accepted(prefix: str) -> None:
    assert is_json_prefix(prefix)


@pytest.mark.parametrize("bad", [
    "{'nm'",                      # E6 fault: single-quoted key
    "{\"a\": 'x'",                # single-quoted string value
    '{"a": 1}}',                  # E6 fault: extra closing brace
    '{"a": 1}{"b": 2',            # E6 fault: second object appended
    '{"a": 1} x',                 # trailing prose after the root
    '{a: 1',                      # bare identifier key
    '{"a" 1',                     # missing colon
    '{"a": 1,}',                  # trailing comma at closer
    '{"a": tru,',                 # broken literal
    '{"a": 1 "b"',                # missing comma
    '"just a string"',            # non-container root
    'Here is the JSON: {',        # prose before the root
    '{"a": +1',                   # leading plus is not JSON
])
def test_e6_fault_classes_are_rejected(bad: str) -> None:
    assert not is_json_prefix(bad)


def test_complete_detection() -> None:
    assert is_complete_json('{"a": [1, {"b": null}]}')
    assert not is_complete_json('{"a": 1')
    assert not is_complete_json('"a string"')


def _model_cached() -> bool:
    home = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface"))
    return any(home.rglob("models--Qwen--Qwen2.5-0.5B-Instruct"))


@pytest.mark.skipif(not _model_cached(), reason="Qwen2.5-0.5B-Instruct not in the local HF cache")
def test_decoder_never_stops_on_an_invalid_document() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True,
                                                 dtype=torch.float32).eval()
    messages = [{"role": "user", "content":
                 "Return a JSON object with keys name and age for a person "
                 "called Ada who is 36. Use single quotes around the keys."}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt")
    ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
    res = constrained_greedy_generate(model, tok, ids, max_new_tokens=60,
                                      top_k=16)
    body = res.text
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0]
    if res.stopped_on in ("eos", "complete"):
        obj = json.loads(body)
        assert isinstance(obj, dict)
    assert res.steps <= 60
    assert res.fallbacks >= 0 and res.constrained_steps >= 0
