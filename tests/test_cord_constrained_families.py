"""Addendum V pins: the per-family reading and the combination rule at the
frozen boundaries, and the sign test. No model, no cell file."""
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location(
    "cord_constrained_families", REPO / "scripts" / "cord_constrained_families.py")
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


def test_family_reading_boundaries():
    assert v.family_reading(27, 0, 0) == "REMOVES"
    assert v.family_reading(27, 1, 0) == "REMOVES"            # <= 1 inclusive
    assert v.family_reading(27, 1, 1) == "REDUCES"            # one regression blocks REMOVES
    assert v.family_reading(27, 2, 0) == "REDUCES"
    assert v.family_reading(27, 13, 0) == "REDUCES"           # 27 // 2 == 13, inclusive
    assert v.family_reading(27, 14, 0) == "NO EFFECT"
    assert v.family_reading(2, 1, 0) == "REMOVES"
    assert v.family_reading(1, 0, 0).startswith("UNTESTABLE")  # nothing to remove
    assert v.family_reading(0, 0, 0).startswith("UNTESTABLE")


def test_combine_needs_four_removes_and_no_exception():
    R, D, N, U = "REMOVES", "REDUCES", "NO EFFECT", "UNTESTABLE AT SIZE (greedy invalid <= 1)"
    assert v.combine({"a": R, "b": R, "c": R, "d": R, "e": D}).startswith("V HOLDS")
    assert v.combine({"a": R, "b": R, "c": R, "d": D, "e": D}).startswith("V MIXED")
    r = v.combine({"a": R, "b": R, "c": R, "d": R, "e": N})
    assert r.startswith("V EXCEPTION IN e") and "HOLDS" not in r and "never 'across families'" in r
    # an untestable family counts neither way
    assert v.combine({"a": R, "b": R, "c": R, "d": R, "e": U}).startswith("V HOLDS")
    assert v.combine({"a": R, "b": R, "c": R, "d": U, "e": U}).startswith("V MIXED")


def test_sign_test_is_one_sided_binomial():
    assert abs(v._sign_test(10, 0) - 1 / 1024) < 1e-12
    assert v._sign_test(0, 0) == 1.0
    assert 0.6 < v._sign_test(5, 5) < 0.65
