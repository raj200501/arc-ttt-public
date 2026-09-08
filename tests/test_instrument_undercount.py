"""Addendum W pins: the W2 and W3 readings at their frozen boundaries, and
the count arithmetic that replaced a float comparison. No corpus needed."""
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location(
    "instrument_undercount", REPO / "scripts" / "instrument_undercount.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def _cell(n, leading, any_letter, fences_only):
    return {"n": n, "fenced_leading": leading, "credited_any_kind_not_none": any_letter,
            "fenced_any_fences_only": fences_only}


def test_w2_boundary_is_a_tenth_of_the_rejected():
    assert w.read_w2(14, 143).startswith("W2 UNDER A TENTH")   # 0.0979
    assert w.read_w2(15, 143).startswith("W2 AT SIZE")         # 0.1049
    assert w.read_w2(0, 0).startswith("W2 UNDER A TENTH")


def test_w3_fires_at_exactly_five_of_a_hundred_by_count_not_float():
    # 0.97 - 0.92 in floats is 0.04999999999999993; the count test fires
    assert w.read_w3({"c": _cell(100, 92, 97, 96)}).startswith("W3 MOVES")
    assert w.read_w3({"c": _cell(100, 92, 96, 96)}).startswith("W3 STABLE")
    assert w.read_w3({"c": _cell(80, 0, 4, 4)}).startswith("W3 MOVES")     # 4/80 = 0.05
    assert w.read_w3({"c": _cell(80, 0, 3, 3)}).startswith("W3 STABLE")


def test_w3_reads_the_letter_and_reports_fences_only_beside_it():
    r = w.read_w3({"cells/x_schema.json": _cell(100, 92, 97, 96)})
    assert "92 -> 97 by the letter" in r and "96 counting fences only" in r
