"""Guards on the novel-schema generator.

The generator's job is to build a task that is HARD for the right reason
and SOLVABLE in principle. Every property that makes the eventual
adapted-vs-prompted comparison meaningful is pinned here, because a subtle
generator bug would produce a null result that looks like a finding.
"""

from __future__ import annotations

import json

from arcttt.novel_schema import make_record, make_schema, make_task


def test_every_target_value_appears_verbatim_in_the_document() -> None:
    """A perfect extractor must be able to score exactly 1.0.

    If any target value were absent from the document, both arms would be
    capped below 1.0 and the comparison would be measuring hallucination
    rather than schema learning.
    """

    schema = make_schema(seed=7)
    for record_seed in range(20):
        text, target = make_record(schema, record_seed)
        for group in target.values():
            for value in group.values():
                assert value in text


def test_document_labels_never_equal_their_json_keys() -> None:
    """The label -> key mapping must carry no surface similarity.

    This is the property the whole experiment turns on: if the document
    said "zelbat" and the key were also "zelbat", the task would collapse
    to copying and a prompted arm would solve it from one example.
    """

    for seed in range(30):
        schema = make_schema(seed=seed)
        for field in schema.fields:
            assert field.doc_label != field.json_path[-1]
            assert field.doc_label not in field.json_path


def test_distractor_lines_are_present_and_excluded_from_the_target() -> None:
    """Knowing what to ignore is schema knowledge and must be testable."""

    schema = make_schema(seed=3, n_distractors=4)
    text, target = make_record(schema, 11)
    flat_keys = {key for group in target.values() for key in group}
    for label in schema.distractor_labels:
        assert f"{label}:" in text          # the line is really there
        assert label not in flat_keys        # and it is really excluded
    assert len(schema.distractor_labels) == 4


def test_generation_is_deterministic_under_seed() -> None:
    """Same seed, same corpus — otherwise arms are not comparable."""

    a_text, a_target = make_record(make_schema(seed=5), 42)
    b_text, b_target = make_record(make_schema(seed=5), 42)
    assert a_text == b_text and a_target == b_target
    c_text, _ = make_record(make_schema(seed=6), 42)
    assert c_text != a_text  # and different tenants really differ


def test_field_order_is_shuffled_so_position_cannot_stand_in_for_schema() -> None:
    """If line order were fixed, position would leak the mapping."""

    schema = make_schema(seed=9)
    orders = set()
    for record_seed in range(25):
        text, _ = make_record(schema, record_seed)
        orders.add(tuple(line.split(":")[0] for line in text.splitlines()))
    assert len(orders) > 1


def test_train_and_test_documents_are_disjoint_but_share_the_schema() -> None:
    """The deployment shape: unseen documents, seen schema."""

    task, schema = make_task(seed=2, n_train=8, n_test=5)
    train_texts = {pair.input_text for pair in task.train}
    test_texts = {pair.input_text for pair in task.test}
    assert len(task.train) == 8 and len(task.test) == 5
    assert not (train_texts & test_texts)

    expected_keys = {field.json_path[0] for field in schema.fields}
    for pair in task.train + task.test:
        assert set(json.loads(pair.output_text)) == expected_keys


def test_targets_are_nested_not_flat() -> None:
    """Flat key-value output would be close to copying."""

    task, _ = make_task(seed=4, n_train=2, n_test=1, n_groups=2)
    target = json.loads(task.train[0].output_text)
    assert len(target) == 2
    assert all(isinstance(value, dict) and value for value in target.values())


def test_two_tenants_do_not_share_labels_keys_or_documents() -> None:
    """Cross-tenant leakage would let one task's learning serve another."""

    a = make_schema(seed=101)
    b = make_schema(seed=202)
    a_labels = {f.doc_label for f in a.fields}
    b_labels = {f.doc_label for f in b.fields}
    assert not (a_labels & b_labels)


def test_labels_keys_and_distractors_never_collide_within_a_schema() -> None:
    """A label doubling as a key or distractor would make the task ambiguous."""

    for seed in range(25):
        schema = make_schema(seed=seed, n_fields=8, n_distractors=4)
        labels = [f.doc_label for f in schema.fields]
        keys = [f.json_path[-1] for f in schema.fields]
        groups = [f.json_path[0] for f in schema.fields]
        names = labels + keys + list(schema.distractor_labels) + groups
        assert len(set(names)) == len(set(labels)) + len(set(keys)) + len(
            set(schema.distractor_labels)
        ) + len(set(groups))


def test_canonical_json_matches_the_projects_definition_without_importing_it() -> None:
    """Drift guard on the deliberately-duplicated canonicaliser.

    ``novel_schema`` re-implements ``text_ttt.json_canonical`` so the corpus
    generator does not depend on torch. That duplication is only safe while
    the two agree, so this reads text_ttt's SOURCE (importing it would pull
    in torch and defeat the purpose) and asserts the call is identical. If
    canonicalization changes there, this fails and forces the update here.
    """

    from pathlib import Path

    from arcttt.novel_schema import _json_canonical

    source = (
        Path(__file__).resolve().parent.parent / "src" / "arcttt" / "text_ttt.py"
    ).read_text()
    expected_call = (
        'json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)'
    )
    assert expected_call in source, "text_ttt.json_canonical changed — update novel_schema"

    # and the local one really behaves that way (key order, spacing, unicode)
    assert _json_canonical({"b": 1, "a": {"d": "x", "c": "é"}}) == '{"a":{"c":"é","d":"x"},"b":1}'
