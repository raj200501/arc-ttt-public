"""The Addendum S/T readings are arithmetic on frozen thresholds. These
pins hold the boundaries exactly where the frozen protocol text puts
them, so the runner can never drift toward the flattering side."""
import importlib.util
import pathlib

_P = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "cord_fence_tax.py"
_spec = importlib.util.spec_from_file_location("cord_fence_tax", _P)
cft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cft)


def test_family_reading_boundaries_are_the_frozen_ones():
    # (a): f_schema >= 0.50 AND f_kshot <= 0.10 -- inclusive both ends
    assert cft.family_reading(0.50, 0.10).startswith("(a)")
    assert cft.family_reading(0.99, 0.00).startswith("(a)")
    # (b): schema clears, k-shot does not
    assert cft.family_reading(0.50, 0.1001).startswith("(b)")
    assert cft.family_reading(0.90, 0.50).startswith("(b)")
    # (c): schema below 0.50 regardless of k-shot
    assert cft.family_reading(0.4999, 0.00).startswith("(c)")
    assert cft.family_reading(0.00, 0.00).startswith("(c)")


def test_combine_families_holds_needs_three_a_and_no_c():
    a, b, c = "(a) REPLICATES", "(b) PARTIAL", "(c) DOES NOT REPLICATE"
    assert cft.combine_families({"w": a, "x": a, "y": a, "z": b}).startswith("HOLDS")
    assert cft.combine_families({"w": a, "x": a, "y": a, "z": a}).startswith("HOLDS")
    # two (a) is MIXED, never HOLDS
    assert cft.combine_families({"w": a, "x": a, "y": b, "z": b}).startswith("MIXED")
    # a single (c) overrides three (a): named exception, never HOLDS
    r = cft.combine_families({"w": a, "x": a, "y": a, "z": c})
    assert r.startswith("(c) IN z")
    assert "HOLDS" not in r
    assert "across model families" in r  # the phrase is named as forbidden
    assert r.index("never") < r.index("across model families")


def test_combine_sizes_unchanged_for_addendum_s():
    a, b, c = "(a) REPLICATES", "(b) PARTIAL", "(c) DOES NOT REPLICATE"
    assert cft.combine_sizes({"0.5b": a, "1.5b": a}).startswith("(a)")
    assert cft.combine_sizes({"0.5b": a, "1.5b": b}).startswith("MIXED")
    assert cft.combine_sizes({"0.5b": c, "1.5b": a}).startswith("(c) AT 0.5b")


def test_the_t_finding_text_never_says_across_families_without_holds():
    a, b = "(a) REPLICATES", "(b) PARTIAL"
    r = cft.combine_families({"w": a, "x": a, "y": b, "z": b})
    assert "HOLDS" not in r and "across organisations" not in r
