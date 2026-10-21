"""The held-out census reader: Wilson interval, two-plus-tiebreak
adjudication, withholding until complete, bar words from the LOWER bound.
Synthetic only; no download, no scan of any real package."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("hc", REPO / "scripts" / "heldout_census.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


def test_wilson_matches_known_values():
    lo, hi = hc.wilson(1, 64)
    assert round(lo, 4) == 0.0028 and round(hi, 4) == 0.0833
    assert hc.wilson(0, 0) == (0.0, 1.0)
    lo, hi = hc.wilson(10, 10)
    assert round(hi, 9) == 1.0 and round(lo, 4) == 0.7225


def test_final_verdict_needs_two_and_a_tiebreak():
    assert hc.final_verdict([{"role": "A", "verdict": "CONFIRMED"}]) is None
    assert hc.final_verdict([{"role": "A", "verdict": "REJECTED"},
                             {"role": "B", "verdict": "REJECTED"}]) == "REJECTED"
    split = [{"role": "A", "verdict": "CONFIRMED"}, {"role": "B", "verdict": "REJECTED"}]
    assert hc.final_verdict(split) is None
    assert hc.final_verdict(split + [{"role": "C", "verdict": "CONFIRMED"}]) == "CONFIRMED"


def test_bar_words_use_the_lower_bound():
    bars = [{"min_precision_lower_bound": 0.0, "words": "a lint rule"},
            {"min_precision_lower_bound": 0.5, "words": "keep it in CI"}]
    assert hc.bar_words(0.49, bars) == "a lint rule"
    assert hc.bar_words(0.5, bars) == "keep it in CI"
    assert hc.bar_words(0.1, [{"min_precision_lower_bound": 0.2, "words": "x"}]) == "no bar cleared"


def _frame(tmp_path):
    frame = tmp_path / "heldout_census_frame.json"
    frame.write_text(json.dumps({"packages": {"p1": "1.0", "p2": "2.0"}, "scanner_sha256": "x",
                                 "prefer": "wheel",
                                 "bars": [{"min_precision_lower_bound": 0.0, "words": "a lint rule"},
                                          {"min_precision_lower_bound": 0.3, "words": "keep it in CI"}]}))
    scan = {"scanner_sha256": "x", "packages": {"p1": {"scanned": True}, "p2": {"scanned": True}},
            "findings": [{"id": "p1#000"}, {"id": "p1#001"}, {"id": "p2#000"}]}
    (tmp_path / "heldout_census_scan.json").write_text(json.dumps(scan))
    return frame


def test_read_withholds_until_every_finding_is_adjudicated(tmp_path, capsys):
    frame = _frame(tmp_path)
    votes = [{"id": "p1#000", "role": r, "verdict": "CONFIRMED"} for r in "AB"]
    (tmp_path / "heldout_census_adjudications.json").write_text(json.dumps({"verdicts": votes}))
    assert hc.read(frame) == 2
    assert "WITHHELD" in capsys.readouterr().out
    assert not (tmp_path / "heldout_census_reading.json").exists()


def test_read_computes_precision_hit_rate_and_disagreements(tmp_path):
    frame = _frame(tmp_path)
    votes = [{"id": "p1#000", "role": r, "verdict": "CONFIRMED"} for r in "AB"]
    votes += [{"id": "p1#001", "role": "A", "verdict": "CONFIRMED"},
              {"id": "p1#001", "role": "B", "verdict": "REJECTED"},
              {"id": "p1#001", "role": "C", "verdict": "REJECTED"}]
    votes += [{"id": "p2#000", "role": r, "verdict": "REJECTED"} for r in "AB"]
    (tmp_path / "heldout_census_adjudications.json").write_text(json.dumps({"verdicts": votes}))
    assert hc.read(frame) == 0
    r = json.loads((tmp_path / "heldout_census_reading.json").read_text())
    assert r["raw_findings"] == 3 and r["confirmed"] == 1 and r["precision"] == 0.3333
    assert r["packages_with_a_confirmed_site"] == ["p1"] and r["package_hit_rate"] == 0.5
    assert r["adjudicator_disagreements"] == 1
    assert r["reading"] == "a lint rule"  # lower bound of 1/3 is well under 0.3


def test_read_refuses_an_unknown_verdict(tmp_path):
    frame = _frame(tmp_path)
    (tmp_path / "heldout_census_adjudications.json").write_text(json.dumps(
        {"verdicts": [{"id": "p1#000", "role": "A", "verdict": "MAYBE"}]}))
    with pytest.raises(SystemExit):
        hc.read(frame)


def test_unpack_refuses_path_traversal(tmp_path):
    import zipfile
    bad = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.py", "x")
    with pytest.raises(SystemExit):
        hc._unpack(bad, tmp_path / "out")
