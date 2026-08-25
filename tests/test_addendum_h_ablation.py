"""The difficulty ablation must change ONLY the key names.

If `mapping="mnemonic"` altered the documents, the values, the distractors
or the shuffles, it would not be an ablation -- it would be a different
corpus, and any delta difference would be uninterpretable. These tests are
the reason the H result can be read as being about the mapping.

They also pin that the default path is untouched, because every banked
artifact in `experiments/` depends on it being byte-identical.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arcttt.novel_schema import make_schema, make_task  # noqa: E402


def test_documents_are_byte_identical_across_the_two_mappings():
    a, _ = make_task(seed=1, n_train=10, n_test=20, task_id="h")
    b, _ = make_task(seed=1, n_train=10, n_test=20, task_id="h",
                     mapping="mnemonic")
    assert [p.input_text for p in a.train] == [p.input_text for p in b.train]
    assert [p.input_text for p in a.test] == [p.input_text for p in b.test]


def test_only_the_keys_differ_and_the_values_do_not():
    a, _ = make_task(seed=3, n_train=10, n_test=20, task_id="h")
    b, _ = make_task(seed=3, n_train=10, n_test=20, task_id="h",
                     mapping="mnemonic")
    for pa, pb in zip(a.test, b.test):
        ja, jb = json.loads(pa.output_text), json.loads(pb.output_text)
        assert sorted(ja) == sorted(jb), "group names must be unchanged"
        for group in ja:
            assert sorted(ja[group].values()) == sorted(jb[group].values()), \
                "the VALUES must be identical; only the keys may move"
        assert set(ja.keys()) == set(jb.keys())
    # And the keys really are different -- otherwise the ablation is a no-op.
    assert a.test[0].output_text != b.test[0].output_text


def test_mnemonic_keys_are_exactly_the_document_labels():
    schema = make_schema(seed=7, mapping="mnemonic")
    for field in schema.fields:
        assert field.json_path[-1] == field.doc_label


def test_arbitrary_keys_never_equal_their_document_label():
    schema = make_schema(seed=7)
    for field in schema.fields:
        assert field.json_path[-1] != field.doc_label


def test_the_default_path_is_unchanged_by_the_ablation():
    """Every banked artifact depends on this. Same seed, same schema text."""
    assert (make_schema(seed=2).describe()
            == make_schema(seed=2, mapping="arbitrary").describe())
    for seed in (1, 2, 3, 203, 209):
        task, schema = make_task(seed=seed, n_train=10, n_test=20)
        again, again_schema = make_task(seed=seed, n_train=10, n_test=20,
                                        mapping="arbitrary")
        assert schema.describe() == again_schema.describe()
        assert [p.output_text for p in task.test] == [
            p.output_text for p in again.test]


def test_unknown_mapping_is_rejected_rather_than_silently_defaulted():
    try:
        make_schema(seed=1, mapping="semantic")
    except ValueError as error:
        assert "semantic" in str(error)
    else:
        raise AssertionError("an unknown mapping must raise")


def test_distractor_ablation_removes_distractor_lines():
    with_decoys, _ = make_task(seed=1, n_train=2, n_test=1, task_id="h")
    without, schema = make_task(seed=1, n_train=2, n_test=1, task_id="h",
                                n_distractors=0)
    assert schema.distractor_labels == ()
    assert (len(without.test[0].input_text.splitlines())
            < len(with_decoys.test[0].input_text.splitlines()))


def test_the_ablation_changes_no_confounding_property():
    """Same output length, same ordering, no key collisions.

    A mapping change that shifted token counts, key order or produced
    duplicate keys would confound the H result with something other than
    the mapping -- and the whole value of H is that only the mapping moves.
    """
    a, _ = make_task(seed=2, n_train=10, n_test=20, task_id="h")
    b, _ = make_task(seed=2, n_train=10, n_test=20, task_id="h",
                     mapping="mnemonic")

    # Identical serialized length: no token-budget confound.
    assert (sum(len(p.output_text) for p in a.test)
            == sum(len(p.output_text) for p in b.test))

    for pa, pb in zip(a.test, b.test):
        ja, jb = json.loads(pa.output_text), json.loads(pb.output_text)
        assert list(ja) == list(jb), "group order must not move"
        for ga, gb in zip(ja.values(), jb.values()):
            assert len(ga) == len(gb)
            assert len(set(gb)) == len(gb), "mnemonic keys must not collide"
