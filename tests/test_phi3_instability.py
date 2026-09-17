"""Addendum Y pins: the frozen readings at their boundaries, the frozen
process order, and the protocol's freeze markers. No model, no cell."""
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location("phi3_instability", REPO / "scripts" / "phi3_instability.py")
y = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(y)


def test_pair_reading_priority():
    assert y.pair_reading(0, 0, 0) == "REPRODUCES"
    assert y.pair_reading(0, 1, 0).startswith("STATE-DEPENDENT")
    assert y.pair_reading(0, 0, 1).startswith("STATE-DEPENDENT")
    assert y.pair_reading(1, 0, 0).startswith("IMMEDIATE-REPEAT UNSTABLE")
    assert y.pair_reading(1, 5, 5).startswith("IMMEDIATE-REPEAT UNSTABLE")  # Y1 wins


def test_dtype_reading_is_decided_by_the_qwen_pairs_only():
    R, S = "REPRODUCES", "STATE-DEPENDENT (Y2=1, Y3=0)"
    assert y.dtype_reading(S, R).startswith("BFLOAT16 IS SUFFICIENT")
    assert y.dtype_reading(R, R).startswith("BFLOAT16 IS NOT SUFFICIENT")
    # a float32 control that fails overrides everything and costs Addendum X a sentence
    assert y.dtype_reading(R, S).startswith("NOT A DTYPE STORY")
    assert y.dtype_reading(S, S).startswith("NOT A DTYPE STORY")


def test_thread_reading():
    assert y.thread_reading(0) == "THREAD-INSENSITIVE ON THIS SAMPLE"
    assert y.thread_reading(1) == "THREAD-SENSITIVE"


def test_the_process_order_is_frozen():
    docs = [{"id": f"cord-{i:03d}", "text": ""} for i in range(3)]
    plan = y._plan(docs)
    assert plan[:6] == [("cord-000", "a1"), ("cord-000", "a2"), ("cord-001", "a1"),
                        ("cord-001", "a2"), ("cord-002", "a1"), ("cord-002", "a2")]
    assert [p for _, p in plan[6:]] == ["b"] * 3 + ["g1"] * 3 + ["g2"] * 3
    assert y.N_DOCS == 10 and y.THREAD_DOCS == ("cord-004", "cord-000")


def test_protocol_is_frozen_and_names_its_cells():
    text = (REPO / "docs" / "research" / "ADDENDUM_Y_PROTOCOL.md").read_text()
    assert "Frozen 2026-09-17" in text
    for needle in ("phi3_instability_2026-09-17.json", "REPRODUCES", "STATE-DEPENDENT",
                   "BFLOAT16 IS SUFFICIENT", "NOT A DTYPE STORY", "THREAD-SENSITIVE"):
        assert needle in text, needle
