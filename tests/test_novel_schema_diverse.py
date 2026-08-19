"""Addendum E: diverse-geometry tenant generation."""

import pytest

from arcttt.novel_schema import make_schema, make_task


def test_fixed_mode_unchanged_by_the_new_parameter():
    assert make_schema(1) == make_schema(1, geometry="fixed")


def test_diverse_shapes_actually_vary_across_seeds():
    shapes = set()
    for seed in range(101, 111):
        s = make_schema(seed, geometry="diverse")
        groups = {f.json_path[0] for f in s.fields}
        shapes.add((len(groups), len(s.fields), len(s.distractor_labels)))
        assert 2 <= len(groups) <= 4
        assert 6 <= len(s.fields) <= 12
    assert len(shapes) > 3  # not vocabulary re-rolls of one shape


def test_diverse_is_deterministic_and_groups_nonempty():
    a = make_schema(101, geometry="diverse")
    b = make_schema(101, geometry="diverse")
    assert a == b
    groups = {f.json_path[0] for f in a.fields}
    for g in groups:
        assert any(f.json_path[0] == g for f in a.fields)


def test_make_task_diverse_end_to_end():
    task, schema = make_task(seed=101, n_train=3, n_test=2,
                             task_id="e-test", geometry="diverse")
    assert len(task.train) == 3 and len(task.test) == 2
    task.validate()


def test_unknown_geometry_rejected():
    with pytest.raises(ValueError):
        make_schema(1, geometry="banana")
