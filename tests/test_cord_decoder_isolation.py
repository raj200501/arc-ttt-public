"""Addendum X pins.

Two things have to be true for this addendum to mean anything, and both
are checked here rather than asserted in the protocol:

1. `enforce=False` is genuinely the SAME loop with the constraint off --
   top-1 always taken, EOS never suppressed, the validator never called --
   while `enforce=True` behaves exactly as it did before the flag existed.
   Proven on a scripted stub model, so it runs in milliseconds and does
   not depend on a checkpoint being cached.
2. The frozen readings are applied by arithmetic at their boundaries,
   including the branch that costs us (V's attribution narrowing).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "cord_decoder_isolation", REPO / "scripts" / "cord_decoder_isolation.py")
x = importlib.util.module_from_spec(_spec)


def _load_module():
    if not hasattr(x, "x1_reading"):
        _spec.loader.exec_module(x)
    return x


# --------------------------------------------------------------------------
# 1. the toggle, on a scripted stub -- no checkpoint, no network
# --------------------------------------------------------------------------

VOCAB = ["{", '"a"', ":", "1", "}", "xx", "<eos>"] + [f"z{i}" for i in range(13)]
EOS_ID = 6
# per step, the token ids in descending logit order (only the head matters)
SCRIPT = [[0, 5], [5, 1], [2, 5], [3, 5], [4, 5], [EOS_ID, 5]]


class _StubTokenizer:
    eos_token_id = EOS_ID

    def decode(self, ids, skip_special_tokens=True):
        return "".join(VOCAB[i] for i in ids
                       if not (skip_special_tokens and i == EOS_ID))


class _Out:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = None


class _StubModel:
    """Returns the scripted preference order for each successive call."""

    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        import torch
        pref = SCRIPT[min(self.calls, len(SCRIPT) - 1)]
        self.calls += 1
        logits = torch.full((1, 1, len(VOCAB)), -10.0)
        for rank, tok in enumerate(pref):
            logits[0, 0, tok] = 10.0 - rank
        return _Out(logits)


def _run(enforce: bool):
    import torch
    from arcttt.constrained_json import constrained_greedy_generate
    return constrained_greedy_generate(
        _StubModel(), _StubTokenizer(), torch.zeros((1, 1), dtype=torch.long),
        max_new_tokens=16, top_k=4, enforce=enforce)


def test_enforce_true_overrides_the_top_token_and_stops_on_a_complete_document():
    pytest.importorskip("torch")
    res = _run(enforce=True)
    assert res.text == '{"a":1}'
    assert res.constrained_steps == 1     # step 1: top-1 'xx' rejected, rank 1 taken
    assert res.fallbacks == 0
    assert res.stopped_on == "complete"
    assert res.token_ids == [0, 1, 2, 3, 4]


def test_enforce_false_is_plain_top_1_and_never_suppresses_eos():
    pytest.importorskip("torch")
    res = _run(enforce=False)
    assert res.text == "{xx:1}"           # the top-1 path, invalid JSON, emitted anyway
    assert res.constrained_steps == 0     # the validator is never consulted
    assert res.fallbacks == 0
    assert res.stopped_on == "eos"        # EOS accepted on an INCOMPLETE document
    assert res.token_ids == [0, 5, 2, 3, 4]


def test_the_two_arms_differ_only_because_of_the_constraint():
    """The whole addendum rests on this: one loop, one boolean."""
    pytest.importorskip("torch")
    on, off = _run(enforce=True), _run(enforce=False)
    assert on.text != off.text
    assert on.token_ids[0] == off.token_ids[0]        # identical until the constraint fires
    assert on.token_ids[1] != off.token_ids[1]


def test_a_validator_that_accepts_everything_is_not_the_same_as_enforce_false():
    """A tempting shortcut -- validator=lambda _: True -- leaves the EOS
    completeness gate in place, so it is NOT the constraint turned off.
    Pinned so nobody 'simplifies' the flag away later."""
    pytest.importorskip("torch")
    import torch
    from arcttt.constrained_json import constrained_greedy_generate
    permissive = constrained_greedy_generate(
        _StubModel(), _StubTokenizer(), torch.zeros((1, 1), dtype=torch.long),
        max_new_tokens=16, top_k=4, validator=lambda _s: True)
    off = _run(enforce=False)
    # The shortcut still refuses EOS on an incomplete document, so it falls
    # through to the next candidate and keeps emitting to the cap.
    assert off.stopped_on == "eos" and off.text == "{xx:1}"
    assert permissive.stopped_on == "max_new_tokens"
    assert permissive.text.startswith("{xx:1}") and permissive.text != off.text
    assert permissive.constrained_steps > 0     # the EOS skips are counted as constraint
    assert off.constrained_steps == 0


# --------------------------------------------------------------------------
# 2. the frozen readings, at their boundaries
# --------------------------------------------------------------------------

def test_x1_reading_boundaries():
    m = _load_module()
    N = 50
    assert m.x1_reading(8, 8, 0, N, N) == "CONSTRAINT INERT"        # identical wins first
    assert m.x1_reading(8, 9, 0, 49, N).startswith("CONSTRAINT HURTS")
    assert m.x1_reading(8, 4, 3, 49, N).startswith("CONSTRAINT HURTS")   # R >= 3 regardless
    assert m.x1_reading(8, 4, 0, 49, N) == "CONSTRAINT REMOVES"     # 8 // 2 == 4, inclusive
    assert m.x1_reading(9, 4, 0, 49, N) == "CONSTRAINT REMOVES"     # 9 // 2 == 4
    assert m.x1_reading(9, 5, 0, 49, N) == "CONSTRAINT HELPS"
    assert m.x1_reading(8, 5, 0, 49, N) == "CONSTRAINT HELPS"
    assert m.x1_reading(8, 5, 2, 49, N).startswith("CONSTRAINT MIXED")
    assert m.x1_reading(8, 8, 0, 49, N) == "CONSTRAINT NEUTRAL"
    assert m.x1_reading(0, 0, 0, N, N) == "CONSTRAINT INERT"
    # Nothing to remove must never read as REMOVES: 0 <= 0 // 2 is true, and
    # that would have credited the constraint on a family it did nothing to.
    assert m.x1_reading(0, 0, 0, 49, N).startswith("UNTESTABLE AT SIZE")
    assert m.x1_reading(1, 0, 0, 49, N).startswith("UNTESTABLE AT SIZE")
    assert m.x1_reading(2, 1, 0, 49, N) == "CONSTRAINT REMOVES"   # first testable size
    # but breaking a valid output is a finding even with nothing to remove
    assert m.x1_reading(0, 1, 0, 49, N).startswith("CONSTRAINT HURTS")
    assert m.x1_reading(1, 1, 3, 49, N).startswith("CONSTRAINT HURTS")


def test_x1_combine_needs_four_removes_and_treats_inert_as_no_success():
    m = _load_module()
    R, I, H = "CONSTRAINT REMOVES", "CONSTRAINT INERT", "CONSTRAINT HURTS (…)"
    assert m.x1_combine({k: R for k in "abcd"} | {"e": I}).startswith("X1 HOLDS")
    assert m.x1_combine({k: R for k in "abc"} | {"d": I, "e": I}).startswith("X1 MIXED")
    # inert is never a success, even when every family is inert
    assert m.x1_combine({k: I for k in "abcde"}).startswith("X1 MIXED")
    # untestable families are counted neither way, and never toward the 4
    U = "UNTESTABLE AT SIZE (plain invalid <= 1)"
    assert m.x1_combine({k: U for k in "abcde"}).startswith("X1 MIXED")
    assert m.x1_combine({k: R for k in "abc"} | {"d": U, "e": U}).startswith("X1 MIXED")
    assert m.x1_combine({k: R for k in "abcd"} | {"e": U}).startswith("X1 HOLDS")
    r = m.x1_combine({k: R for k in "abcd"} | {"e": H})
    assert r.startswith("X1 EXCEPTION IN e") and "HOLDS" not in r
    assert "never 'across families'" in r


def test_x2_reading_and_prefix_divergence():
    m = _load_module()
    assert m.x2_reading(0) == "PATH REPRODUCES"
    assert m.x2_reading(1).startswith("PATH DIVERGES")
    # ours stops as soon as the document parses, generate runs on: shorter
    # is agreement, and only where ours did NOT stop on eos.
    assert m._prefix_divergence([1, 2, 3], [1, 2, 3, 4]) is None
    assert m._prefix_divergence([1, 2, 3], [1, 9, 3]) == 1
    # generate stopping first is never expected and is a divergence
    assert m._prefix_divergence([1, 2, 3, 4], [1, 2]) == 2
    # the hole the stop reason closes: ours took EOS, so generate saw the
    # same logits and must stop in the same place. Without this the prefix
    # rule alone scores an empty plain arm as agreement -- our direction.
    assert m._prefix_divergence([], [1], "eos") == 0
    assert m._prefix_divergence([], [1], "complete") is None
    assert m._prefix_divergence([1, 2], [1, 2, 3], "eos") == 2
    assert m._prefix_divergence([1, 2], [1, 2], "eos") is None
    assert m._text_prefix_divergence("abc", "abcd") is None
    assert m._text_prefix_divergence("abc", "abx") == 2
    assert m._text_prefix_divergence("abc", "abcd", "eos") == 3


def test_x3_puts_our_own_published_sentence_at_risk():
    m = _load_module()
    assert m.x3_reading(5, 5).startswith("V'S ATTRIBUTION HOLDS")
    assert m.x3_reading(4, 5).startswith("V'S ATTRIBUTION IS TOO STRONG")
    assert "residual 1" in m.x3_reading(4, 5)
    assert m.x3_reading(0, 5).startswith("V'S ATTRIBUTION IS TOO STRONG")


def test_arm_g_scope_is_decided_by_the_config_not_by_results():
    m = _load_module()
    qwen = {"do_sample": True, "repetition_penalty": 1.1, "temperature": 0.7,
            "top_p": 0.8, "top_k": 20}
    assert m.needs_generate_arm(qwen) is True
    assert m.modifiers_of(qwen)["repetition_penalty"] == 1.1
    assert m.needs_generate_arm({}) is False
    # a bare top_k cannot change greedy output, so it must not trigger arm G
    assert m.needs_generate_arm({"top_k": 20}) is False
    assert m.modifiers_of({"top_k": 20}) == {"top_k": 20}
    # neutral values are not modifiers
    assert m.needs_generate_arm({"repetition_penalty": 1.0, "temperature": 1.0}) is False
    # An unset key reads as None in GenerationConfig.to_dict(). Reading that
    # naively made every family look like it carried a modifier and would
    # have run four arms the protocol says not to run.
    assert m.modifiers_of({"repetition_penalty": None, "do_sample": None}) == {}
    assert m.needs_generate_arm({"repetition_penalty": None}) is False
    # a library default supplied as the baseline is not a modifier either
    assert m.needs_generate_arm({"temperature": 0.7}, {"temperature": 0.7}) is False
    assert m.needs_generate_arm({"temperature": 0.7}, {"temperature": 1.0}) is True


def test_the_protocol_is_frozen_before_the_cells_exist():
    """The protocol file must name the cells it will produce, so a reader
    can tell preregistration from description-after-the-fact."""
    text = (REPO / "docs" / "research" / "ADDENDUM_X_PROTOCOL.md").read_text()
    assert "Frozen 2026-09-16" in text
    for needle in ("cord_decoder_isolation_2026-09-16.json",
                   "generation_configs_2026-09-16.json",
                   "CONSTRAINT INERT", "PATH REPRODUCES",
                   "V'S ATTRIBUTION IS TOO STRONG"):
        assert needle in text, needle


# ---------------------------------------------------------------------------
# The gate is terminal. It fired for real on 2026-09-16 (phi3-mini), so the
# behaviour it forced is pinned here: a failed gate withholds the whole
# addendum, is checked BEFORE the missing-cell check so no further hours are
# spent decoding cells that cannot be read, and banks an artifact saying so.
# ---------------------------------------------------------------------------

def test_a_failed_gate_withholds_and_is_banked():
    import json as _json
    m = _load_module()
    artifact = REPO / "experiments" / "cord_decoder_isolation_2026-09-16.json"
    if not artifact.exists():
        pytest.skip("the addendum's reading has not been run in this tree")
    banked = _json.loads(artifact.read_text())
    gates = banked.get("determinism_gate", {})
    failed = [f for f, g in gates.items() if not g["passed"]]
    if failed:
        assert banked["reading"] == "WITHHELD BY THE DETERMINISM GATE"
        assert banked["failed_families"] == sorted(failed)
        # none of the three contrasts may be reported alongside a failed gate
        for key in ("x1_finding", "x2_per_family", "x3", "rows"):
            assert key not in banked, f"{key} published despite a failed gate"
        assert "cells_not_run" in banked and "why_the_remaining_cells_were_not_run" in banked
    else:
        assert banked["reading"] != "WITHHELD BY THE DETERMINISM GATE"


def test_both_gate_runs_are_kept_when_they_disagree():
    """The gate was run twice on phi3-mini and the runs disagree. Re-running a
    gate until it passes is the failure this page exists to prevent, so both
    runs stay banked and both must fail."""
    import json as _json
    cells = REPO / "experiments" / "cord_decoder_isolation_cells"
    second, first = cells / "phi3-mini_determinism.json", cells / "phi3-mini_determinism_run1.json"
    if not second.exists():
        pytest.skip("the phi3 gate has not been run in this tree")
    if not first.exists():
        pytest.skip("only one gate run exists")
    a, b = _json.loads(first.read_text()), _json.loads(second.read_text())
    assert not a["passed"] and not b["passed"], "a re-run that passes must not replace one that failed"
    assert b.get("first_run") == first.name
    assert "the_two_runs_disagree" in b
