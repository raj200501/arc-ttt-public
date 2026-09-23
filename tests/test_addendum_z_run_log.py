"""The Addendum Z run log makes a silent redraw detectable, and it never
touches the live run's files. Pinned on synthetic journals only."""
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


log = _load("addendum_z_run_log")


def _rows(n, tweak=None):
    rows = [{"index": i, "prediction": f"{{\"a\": {i}}}", "micro_f1_raw": 0.5 + i / 100,
             "completions": ["x", "y"]} for i in range(n)]
    if tweak is not None:
        rows[tweak]["prediction"] = "redrawn"
    return rows


def _setup(tmp_path, monkeypatch, rows):
    work, exp = tmp_path / "work", tmp_path / "exp"
    work.mkdir()
    exp.mkdir()
    stem = log.z.ckpt_stem(1, "adapted")
    (work / f"{stem}_meta.json").write_text(json.dumps(
        {"restarts": 1, "started_utc": "t", "adapter_sha256": "a" * 64}))
    # journal lines in decode order, with a torn last line mid-write
    body = "".join(json.dumps(r) + "\n" for r in rows) + '{"index": 99, "predic'
    (work / f"{stem}_docs.jsonl").write_text(body)
    monkeypatch.setattr(log, "OUT", tmp_path / "run_log.json")
    monkeypatch.setattr(log, "REPO", tmp_path)
    return work, exp, stem, body


def test_the_snapshot_reads_without_touching_a_live_journal(tmp_path, monkeypatch):
    work, exp, stem, body = _setup(tmp_path, monkeypatch, _rows(3))
    state = log.arm_state(1, "adapted", work)
    assert (work / f"{stem}_docs.jsonl").read_text() == body  # the torn line is left alone
    assert state["documents_decoded"] == 3 and state["prefix_documents"] == 3
    assert state["adapter_sha256"] == "a" * 64 and state["restarts"] == 1


def test_the_prefix_hash_is_recomputable_from_a_landed_artifact_and_catches_a_redraw(tmp_path, monkeypatch):
    work, exp, stem, _ = _setup(tmp_path, monkeypatch, _rows(3))
    log.record(work)
    landed = {"adapter": {"sha256": "a" * 64}, "predictions": list(reversed(_rows(5)))}
    name = log.z.artifact_name(1, "adapted")
    (exp / name).write_text(json.dumps(landed))
    problems = [p for p in log.verify(exp) if name in p]
    assert problems == []  # the landed arm extends what was logged
    landed["predictions"] = _rows(5, tweak=1)
    (exp / name).write_text(json.dumps(landed))
    assert any("first 3 documents differ" in p for p in log.verify(exp))
    landed = {"adapter": {"sha256": "b" * 64}, "predictions": _rows(5)}
    (exp / name).write_text(json.dumps(landed))
    assert any("adapter redrawn" in p for p in log.verify(exp))


def test_only_a_gapless_prefix_is_hashed(tmp_path, monkeypatch):
    rows = _rows(4)
    del rows[2]
    work, exp, stem, _ = _setup(tmp_path, monkeypatch, rows)
    state = log.arm_state(1, "adapted", work)
    assert state["documents_decoded"] == 3 and state["prefix_documents"] == 2


def test_the_log_records_no_score_and_no_text():
    path = REPO / "experiments" / "addendum_z_run_log.json"
    if not path.exists():
        return
    blob = path.read_text()
    for forbidden in ('"micro_f1', '"prediction"', '"mean_micro_f1"', '"completions"'):
        assert forbidden not in blob, forbidden
