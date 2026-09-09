"""tools/jsongreedy.py is the standalone copy of the decoder every banked
run used. Its decoding core must stay byte-identical to
src/arcttt/constrained_json.py, and its convenience API must be importable
without torch or transformers."""
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
LIB = REPO / "src" / "arcttt" / "constrained_json.py"
TOOL = REPO / "tools" / "jsongreedy.py"


def _core(text: str) -> str:
    start = text.index("from __future__ import annotations")
    end = text.index("# ---------------------------------------------------------------- convenience") \
        if "# ---------------------------------------------------------------- convenience" in text else len(text)
    return text[start:end].rstrip("\n")


def test_tool_core_is_byte_identical_to_the_library():
    assert _core(TOOL.read_text()) == _core(LIB.read_text())


def test_tool_imports_without_torch_and_exposes_the_api():
    import sys
    spec = importlib.util.spec_from_file_location("jsongreedy", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jsongreedy"] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    assert callable(mod.generate) and callable(mod.constrained_greedy_generate)
    assert mod.is_json_prefix('{"a": [1, 2') and not mod.is_json_prefix("{'a': 1}")
    assert mod.is_complete_json('{"a": 1}') and not mod.is_complete_json('{"a": 1')
