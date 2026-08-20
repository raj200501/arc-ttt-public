"""Founder-side challenge runner: task construction and output hygiene."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_challenge.py"
spec = importlib.util.spec_from_file_location("run_challenge", SCRIPT)
run_challenge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_challenge)


def test_build_task_serializes_gold_canonically_and_hides_holdout_outputs():
    train = [{"id": "t1", "text": "doc one", "gold": {"b": "2", "a": "1"}}]
    holdout = [{"id": "h1", "text": "doc two"}]
    task = run_challenge.build_task(train, holdout)
    assert task.train[0].output_text == '{"a":"1","b":"2"}'
    assert task.test[0].output_text is None
    assert task.test[0].input_text == "doc two"


def test_build_task_rejects_missing_gold():
    with pytest.raises(KeyError):
        run_challenge.build_task([{"id": "t1", "text": "doc"}], [])


def test_write_outputs_parses_json_and_nulls_garbage(tmp_path):
    holdout = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    raw = ['{"x": "1"}', "not json at all", '["array", "not", "object"]', None]
    rows = run_challenge.write_outputs(tmp_path, holdout, raw)
    assert [r["prediction"] for r in rows] == [{"x": "1"}, None, None, None]
    on_disk = [json.loads(line) for line in
               (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert on_disk == rows
    assert [r["id"] for r in on_disk] == ["a", "b", "c", "d"]
