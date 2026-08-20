"""The challenger kit: split determinism, gold hygiene, protocol scoring."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_challenge.py"


def _docs(n: int = 12) -> list[dict]:
    return [{"id": f"doc-{i}", "text": f"body {i}", "gold": {"field": f"v{i}"}}
            for i in range(n)]


def _write_docs(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *argv],
                          capture_output=True, text=True)


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_split_partitions_ids_and_withholds_gold(tmp_path):
    docs = tmp_path / "docs.jsonl"
    _write_docs(docs, _docs())
    out = tmp_path / "pkg"
    result = _run("split", "--docs", str(docs), "--train-k", "5",
                  "--seed", "7", "--out-dir", str(out))
    assert result.returncode == 0, result.stderr
    train = _read_jsonl(out / "train.jsonl")
    holdout = _read_jsonl(out / "holdout.jsonl")
    gold = _read_jsonl(out / "gold_holdout.jsonl")
    assert len(train) == 5 and len(holdout) == 7 and len(gold) == 7
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in holdout})
    assert {r["id"] for r in holdout} == {r["id"] for r in gold}
    assert all("gold" not in r for r in holdout), "gold leaked into holdout"
    assert all("gold" in r for r in train)
    terms = (out / "TERMS.md").read_text()
    assert "seed 7" in terms and "5 labeled training pairs" in terms
    sha = [ln for ln in result.stdout.splitlines() if "sha256" in ln]
    assert sha, "gold sha256 not printed"


def test_split_is_deterministic(tmp_path):
    docs = tmp_path / "docs.jsonl"
    _write_docs(docs, _docs())
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        assert _run("split", "--docs", str(docs), "--train-k", "4",
                    "--seed", "3", "--out-dir", str(out)).returncode == 0
    for name in ("train.jsonl", "holdout.jsonl", "gold_holdout.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_split_rejects_duplicate_ids_and_missing_fields(tmp_path):
    docs = tmp_path / "docs.jsonl"
    rows = _docs(5)
    rows[3]["id"] = rows[1]["id"]
    _write_docs(docs, rows)
    result = _run("split", "--docs", str(docs), "--train-k", "2",
                  "--out-dir", str(tmp_path / "x"))
    assert result.returncode != 0 and "duplicate id" in result.stderr

    _write_docs(docs, [{"id": "a", "text": "t"}] * 4)
    result = _run("split", "--docs", str(docs), "--train-k", "2",
                  "--out-dir", str(tmp_path / "y"))
    assert result.returncode != 0 and "gold" in result.stderr


def test_score_matches_protocol_aggregation(tmp_path):
    pytest.importorskip("torch")
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    gold.write_text(
        json.dumps({"id": "a", "gold": {"x": "1", "y": "2"}}) + "\n"
        + json.dumps({"id": "b", "gold": {"x": "1"}}) + "\n"
        + json.dumps({"id": "c", "gold": {"x": "1"}}) + "\n")
    # a: exact (1.0); b: missing prediction (0); c: wrong value (0.0 overlap)
    pred.write_text(
        json.dumps({"id": "a", "prediction": {"x": "1", "y": "2"}}) + "\n"
        + json.dumps({"id": "c", "prediction": {"x": "999"}}) + "\n"
        + json.dumps({"id": "zzz", "prediction": {"x": "1"}}) + "\n")
    result = _run("score", "--pred", str(pred), "--gold", str(gold))
    assert result.returncode == 0, result.stderr
    assert "over 3 holdout docs: 0.3333" in result.stdout
    assert "ids not in holdout" in result.stdout
