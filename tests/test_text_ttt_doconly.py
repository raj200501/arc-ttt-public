"""Addendum D: document-only prompt construction (include_demos=False)."""

from arcttt.text_ttt import text_task_to_messages
from arcttt.novel_schema import make_task


def test_doconly_prompt_carries_only_the_document():
    task, _ = make_task(seed=1, n_train=5, n_test=2, task_id="d-test")
    with_demos = text_task_to_messages(task, 1)
    doc_only = text_task_to_messages(task, 1, include_demos=False)
    assert len(with_demos) == 2 * len(task.train) + 1
    assert len(doc_only) == 1
    assert doc_only[0].role == "user"
    assert doc_only[0].content == task.test[1].input_text
    # default unchanged (backward compatibility with every existing caller)
    assert with_demos == text_task_to_messages(task, 1, include_demos=True)


def test_docmode_training_examples_are_single_pair_sequences():
    from arcttt.text_ttt import text_docmode_training_examples
    task, _ = make_task(seed=3, n_train=4, n_test=1, task_id="f-test")
    examples = text_docmode_training_examples(task)
    assert len(examples) == 4
    for ex, pair in zip(examples, task.train):
        assert len(ex) == 2
        assert ex[0].role == "user" and ex[0].content == pair.input_text
        assert ex[1].role == "assistant" and ex[1].content == pair.output_text
