"""Text-mode TTT unit tests: loaders, LOO corpus, scorers, tiny-model loop.

Offline throughout — the end-to-end test uses the same tiny in-test model
pattern as test_model_loop.py (no downloads, CPU only).
"""

from __future__ import annotations

import json

import pytest

from arcttt.serialize import ChatTurn, ttt_training_examples
from arcttt.tasks import Pair, Task
from arcttt.text_task import (
    TextPair,
    TextTask,
    TextTaskFormatError,
    from_cord_gt,
    load_text_task,
    load_text_tasks_jsonl,
)
from arcttt.text_ttt import (
    TextScore,
    field_micro_f1,
    normalize_value,
    parse_json_object,
    score_text_output,
    text_task_to_messages,
    text_ttt_training_examples,
)


def make_text_task(train: int = 3) -> TextTask:
    words = ("aa", "bb", "cc", "dd", "ee")[:train]
    return TextTask(
        task_id="tiny-text",
        train=tuple(
            TextPair(input_text=f"item {word}", output_text=f'{{"nm": "{word}"}}')
            for word in words
        ),
        test=(TextPair(input_text="item zz", output_text='{"nm": "zz"}'),),
    )


# -- loader ------------------------------------------------------------------


def test_json_loader_roundtrip(tmp_path) -> None:
    payload = {
        "train": [{"input": "in a", "output": "out a"}, {"input": "in b", "output": "out b"}],
        "test": [{"input": "in c"}],
    }
    path = tmp_path / "receipts.json"
    path.write_text(json.dumps(payload))
    task = load_text_task(path)
    assert task.task_id == "receipts"  # stem, matching tasks.load_task
    assert task.train[1] == TextPair("in b", "out b")
    assert task.test[0].output_text is None

    payload["task_id"] = "explicit-id"
    path.write_text(json.dumps(payload))
    assert load_text_task(path).task_id == "explicit-id"


def test_json_loader_failure_modes(tmp_path) -> None:
    path = tmp_path / "bad.json"

    def expect_failure(payload: object) -> None:
        path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
        with pytest.raises(TextTaskFormatError):
            load_text_task(path)

    expect_failure("{not json")
    expect_failure({"train": []})  # missing test key
    expect_failure({"train": [], "test": [{"input": "x"}]})  # empty train
    expect_failure({"train": [{"input": "a", "output": "b"}], "test": []})  # empty test
    expect_failure({"train": [{"input": "a"}], "test": [{"input": "x"}]})  # no train output
    expect_failure({"train": [{"output": "b"}], "test": [{"input": "x"}]})  # no input
    expect_failure({"train": [{"input": "", "output": "b"}], "test": [{"input": "x"}]})
    expect_failure({"train": [{"input": "a", "output": "  "}], "test": [{"input": "x"}]})
    expect_failure({"train": [{"input": 3, "output": "b"}], "test": [{"input": "x"}]})
    expect_failure({"train": {"input": "a"}, "test": [{"input": "x"}]})  # not a list


def test_jsonl_loader_and_failure_modes(tmp_path) -> None:
    good = {
        "task_id": "one",
        "train": [{"input": "a", "output": "b"}],
        "test": [{"input": "c", "output": "d"}],
    }
    other = dict(good, task_id="two")
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(good) + "\n\n" + json.dumps(other) + "\n")
    tasks = load_text_tasks_jsonl(path)
    assert sorted(tasks) == ["one", "two"]
    assert tasks["one"].test[0].output_text == "d"

    path.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(TextTaskFormatError, match="duplicate task_id"):
        load_text_tasks_jsonl(path)
    path.write_text(json.dumps(dict(good) | {"task_id": None}) + "\n")
    with pytest.raises(TextTaskFormatError, match="task_id"):
        load_text_tasks_jsonl(path)
    path.write_text("{broken\n")
    with pytest.raises(TextTaskFormatError, match="invalid JSON"):
        load_text_tasks_jsonl(path)
    path.write_text("\n")
    with pytest.raises(TextTaskFormatError, match="no tasks"):
        load_text_tasks_jsonl(path)


def test_validate_is_fail_closed_on_hand_built_tasks() -> None:
    with pytest.raises(TextTaskFormatError):
        TextTask("t", train=(), test=(TextPair("a", None),)).validate()
    with pytest.raises(TextTaskFormatError):
        TextTask("t", train=(TextPair("a", None),), test=(TextPair("b", None),)).validate()
    with pytest.raises(TextTaskFormatError):
        TextTask(
            "t", train=(TextPair("a", "b"),), test=(TextPair(" ", None),)
        ).validate()
    make_text_task().validate()  # the happy path validates cleanly




def test_text_task_to_messages_shape() -> None:
    task = make_text_task(train=2)
    turns = text_task_to_messages(task, test_index=0)
    assert turns == (
        ChatTurn("user", "item aa"),
        ChatTurn("assistant", '{"nm": "aa"}'),
        ChatTurn("user", "item bb"),
        ChatTurn("assistant", '{"nm": "bb"}'),
        ChatTurn("user", "item zz"),
    )
    with pytest.raises(TextTaskFormatError, match="out of range"):
        text_task_to_messages(task, test_index=1)


def test_loo_corpus_structure_and_determinism() -> None:
    task = make_text_task(train=4)
    examples = text_ttt_training_examples(task, shuffle_seed=0)
    assert len(examples) == 4  # one example per held-out demonstration
    for held_out, turns in enumerate(examples):
        assert len(turns) == 8  # 3 context pairs + the supervised pair
        assert [turn.role for turn in turns] == ["user", "assistant"] * 4
        assert turns[-2] == ChatTurn("user", task.train[held_out].input_text)
        assert turns[-1] == ChatTurn("assistant", task.train[held_out].output_text)
    assert examples == text_ttt_training_examples(task, shuffle_seed=0)

    unshuffled = text_ttt_training_examples(task)
    assert unshuffled == text_ttt_training_examples(task)
    context_inputs = [turn.content for turn in unshuffled[0][:-2] if turn.role == "user"]
    assert context_inputs == ["item bb", "item cc", "item dd"]  # train order kept
    assert any(
        text_ttt_training_examples(task, shuffle_seed=seed) != unshuffled
        for seed in range(10)
    ), "shuffle seeds must be able to permute the context order"


def test_loo_corpus_mirrors_grid_path_semantics() -> None:
    # Same seed, same pair count: the text builder must order contexts exactly
    # like serialize.ttt_training_examples. 1x1 grids serialize to single
    # digits, so the two corpora are directly comparable turn by turn.
    grid_task = Task(
        task_id="mirror",
        train=tuple(Pair(input=((i,),), output=((9 - i,),)) for i in range(5)),
        test=(Pair(input=((0,),), output=None),),
    )
    text_task = TextTask(
        task_id="mirror",
        train=tuple(
            TextPair(input_text=str(i), output_text=str(9 - i)) for i in range(5)
        ),
        test=(TextPair(input_text="0", output_text=None),),
    )
    for seed in (None, 0, 7, 123):
        assert text_ttt_training_examples(text_task, seed) == ttt_training_examples(
            grid_task, seed
        )


def test_loo_corpus_rejects_empty_task() -> None:
    task = TextTask("empty", train=(), test=(TextPair("x", None),))
    with pytest.raises(TextTaskFormatError, match="no usable TTT examples"):
        text_ttt_training_examples(task)


# -- scoring -----------------------------------------------------------------


def test_normalize_value_cases() -> None:
    assert normalize_value("12,000") == "12000"
    assert normalize_value("12000") == "12000"
    assert normalize_value(12000) == "12000"
    assert normalize_value(12000.0) == "12000"
    assert normalize_value("5.50") == "5.5"
    assert normalize_value(" Latte  Co ") == "latte co"
    assert normalize_value("2 x") == "2 x"  # not purely numeric: name fold only
    assert normalize_value(True) == "true"
    assert normalize_value(None) == "null"
    with pytest.raises(TextTaskFormatError):
        normalize_value(object())


def test_parse_json_object_fail_closed() -> None:
    assert parse_json_object(' {"a": 1} ') == {"a": 1}
    for bad in ("not json", "[1, 2]", '"scalar"', "42", '{"a": 1} trailing'):
        with pytest.raises(TextTaskFormatError):
            parse_json_object(bad)


def test_field_micro_f1_hand_computed() -> None:
    gold = {
        "menu": [{"nm": "Latte"}, {"nm": "Mocha"}],
        "total": {"total_price": "12,000"},
    }
    # Reordered list, case/format drift: all 3 leaves still match -> F1 = 1.
    perfect = {
        "total": {"total_price": "12000"},
        "menu": [{"nm": "mocha"}, {"nm": "LATTE"}],
    }
    assert field_micro_f1(perfect, gold) == 1.0
    # One dropped item and one wrong price: overlap 1 of pred=2 / gold=3
    # leaves -> F1 = 2*1 / (2+3) = 0.4.
    partial = {"menu": [{"nm": "latte"}], "total": {"total_price": "13,000"}}
    assert field_micro_f1(partial, gold) == pytest.approx(0.4)
    # Same value under the wrong path scores nothing.
    assert field_micro_f1({"sub_total": {"total_price": "12,000"}}, gold) == pytest.approx(
        2 * 0 / (1 + 3)
    )
    assert field_micro_f1({}, gold) == 0.0
    assert field_micro_f1({}, {}) == 1.0
    # Duplicate leaves count as a multiset: predicting one "aa" of two.
    assert field_micro_f1(
        {"menu": [{"nm": "aa"}]}, {"menu": [{"nm": "aa"}, {"nm": "aa"}]}
    ) == pytest.approx(2 / 3)


def test_score_text_output_metrics_and_invalid_json() -> None:
    gold_text = '{"total": {"total_price": "12,000"}, "menu": [{"nm": "Latte"}]}'
    same = score_text_output(
        '{\n  "menu": [{"nm": "Latte"}],\n  "total": {"total_price": "12,000"}\n}',
        gold_text,
    )
    assert same == TextScore(valid_json=True, exact_match=True, micro_f1=1.0)
    close = score_text_output(
        '{"menu": [{"nm": "latte"}], "total": {"total_price": "12000"}}', gold_text
    )
    assert close.valid_json and not close.exact_match  # canonical strings differ
    assert close.micro_f1 == 1.0  # ...but every normalized field matches
    invalid = score_text_output('{"menu": [', gold_text)
    assert invalid == TextScore(valid_json=False, exact_match=False, micro_f1=0.0)
    with pytest.raises(TextTaskFormatError):
        score_text_output("{}", "gold is not json")  # malformed gold is a harness bug


# -- tiny-model end-to-end ---------------------------------------------------

pytest.importorskip("transformers")

import torch  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from arcttt.model import TTTConfig  # noqa: E402
from arcttt.text_ttt import TextPredictor  # noqa: E402

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' }}{{ message['content'] }}{{ '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def tiny_text_tokenizer() -> PreTrainedTokenizerFast:
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789{}":,. \n'
    vocab = {ch: i for i, ch in enumerate(chars)}
    for special in ("<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"):
        vocab[special] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<|pad|>",
        eos_token="<|end|>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def tiny_model(vocab_size: int) -> Qwen2ForCausalLM:
    torch.manual_seed(11)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config)


def test_text_predictor_refuses_dfs_config() -> None:
    tokenizer = tiny_text_tokenizer()
    model = tiny_model(len(tokenizer))
    with pytest.raises(ValueError, match="DFS"):
        TextPredictor(
            model, tokenizer, TTTConfig(use_dfs=True), torch.device("cpu")
        )


def test_text_ttt_end_to_end_on_tiny_model() -> None:
    tokenizer = tiny_text_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = TextPredictor(
        model,
        tokenizer,
        TTTConfig(
            lora_rank=4, lora_alpha=8, epochs=6, learning_rate=5e-3, max_new_tokens=16
        ),
        torch.device("cpu"),
    )
    task = make_text_task(train=3)
    gold = task.test[0].output_text
    assert gold is not None

    before = predictor.log_probabilities_text(task, 0, [gold])[0]
    predictor.adapt_text(task, shuffle_seeds=(0, 1))
    assert any("lora" in name for name, _ in predictor.model.named_parameters())
    trainable = [p for p in predictor.model.parameters() if p.requires_grad]
    assert trainable, "LoRA adaptation must leave trainable parameters"

    after = predictor.log_probabilities_text(task, 0, [gold])[0]
    assert after > before, f"TTT should raise target log-probability ({before} -> {after})"
    assert after < 0.0  # log-probability stays a log-probability

    # Generation executes end to end; a tiny random-ish model rarely emits
    # valid JSON, so only the plumbing contract is asserted (the scorer's
    # behavior on both outcomes is covered above).
    texts = predictor.predict_text(task, test_index=0, samples=2)
    assert isinstance(texts, list)
    assert all(isinstance(text, str) and text for text in texts)
    for text in texts:
        score = score_text_output(text, gold)
        assert isinstance(score, TextScore)


def test_from_cord_gt_renders_ocr_lines_and_canonical_target() -> None:

    def cord_row(items, total, extra_key=False):
        gt = {
            "menu": [{"nm": nm, "cnt": cnt, "price": price}
                     for nm, cnt, price in items],
            "total": {"total_price": total},
        }
        if extra_key:
            gt["unreleased_class"] = {"x": "1"}
        return {
            "gt_parse": gt,
            "valid_line": [
                {"words": [{"text": nm}, {"text": price}]}
                for nm, _, price in items
            ] + [{"words": [{"text": "TOTAL"}, {"text": total}]}],
        }

    row_a = cord_row([("LATTE", "1", "4,500")], "4,500", extra_key=True)
    row_b = cord_row([("MOCHA", "2", "9,000")], "9,000")
    task = from_cord_gt([row_a], [row_b], task_id="cord-test")

    assert task.train[0].input_text == "LATTE 4,500\nTOTAL 4,500"
    # unreleased classes are dropped; canonical JSON is sorted + compact
    assert "unreleased_class" not in task.train[0].output_text
    assert '"menu":[{"cnt":"1","nm":"LATTE","price":"4,500"}]' in task.train[0].output_text
    assert task.test[0].output_text is not None
    assert "9,000" in task.test[0].output_text

    import pytest as _pytest

    from arcttt.text_task import TextTaskFormatError

    with _pytest.raises(TextTaskFormatError):
        from_cord_gt([{"gt_parse": {"menu": []}, "valid_line": []}], [row_b])


def test_vote_text_candidates_pools_by_canonical_form() -> None:
    from arcttt.text_ttt import select_text_attempts, vote_text_candidates

    # Three completions: two are the SAME JSON in different formatting (must
    # pool), one is different; one unparseable near-miss pools alone.
    same_a = '{"total": "4,500", "menu": []}'
    same_b = '{"menu":[],"total":"4,500"}'
    other = '{"total": "9,000"}'
    broken = '{"total": '
    candidates = vote_text_candidates(
        [same_a, same_b, other, broken], [-0.5, -0.1, -0.05, -0.01]
    )
    by_count = {c.found_count: c for c in candidates}
    assert len(candidates) == 3
    assert by_count[2].text == same_b  # representative = highest-lp member
    assert by_count[2].mean_log_probability == pytest.approx(-0.3)
    # count dominates: the pooled pair (count 2) outranks the higher-lp
    # singletons, exactly like the grid path's count + exp(mean lp) rule
    assert select_text_attempts(candidates, attempts=1) == (same_b,)


def test_select_text_attempts_probability_breaks_count_ties() -> None:
    from arcttt.text_ttt import TextCandidate, select_text_attempts

    low = TextCandidate(text="a", key="a", found_count=1, mean_log_probability=-3.0)
    high = TextCandidate(text="b", key="b", found_count=1, mean_log_probability=-0.1)
    assert select_text_attempts([low, high], attempts=2) == ("b", "a")
    # ...but never outweighs one extra find
    found_twice = TextCandidate(
        text="c", key="c", found_count=2, mean_log_probability=-9.0
    )
    assert select_text_attempts([low, high, found_twice], attempts=1) == ("c",)


def test_predict_text_voted_selects_majority_completion() -> None:
    from arcttt.text_task import TextPair, TextTask
    from arcttt.text_ttt import predict_text_voted

    class FakePredictor:
        def predict_text(self, task: object, index: int, samples: int,
                         include_demos: bool = True) -> list[str]:
            assert samples == 5
            return ['{"a": 1}', '{"a":1}', '{"a": 2}']

        def log_probabilities_text(
            self, task: object, index: int, outputs: list[str],
            include_demos: bool = True,
        ) -> list[float]:
            # the odd one out is the model's single favorite; the pool wins
            return [-2.0 if "1" in text else -0.01 for text in outputs]

    task = TextTask(
        task_id="t",
        train=(TextPair(input_text="x", output_text='{"a": 1}'),),
        test=(TextPair(input_text="y", output_text=None),),
    )
    selected = predict_text_voted(FakePredictor(), task, 0, samples=5)  # type: ignore[arg-type]
    assert selected == '{"a": 1}'


def test_template_ids_normalizes_batchencoding_and_tensor() -> None:
    """transformers 5.x returns BatchEncoding from apply_chat_template;
    4.x returns the tensor. Both must normalize (cord-scale kernel
    incident 2026-08-11 — the 5.x object raises an EMPTY AttributeError
    on .shape, the same pinned-image API-drift class as v7's cache bug)."""

    import torch

    from arcttt.model import _template_ids

    ids = torch.tensor([[1, 2, 3]])
    assert _template_ids(ids) is ids

    class FakeBatchEncoding:
        # mirrors tokenization_utils_base: attribute access hits data[item]
        def __init__(self, input_ids: torch.Tensor) -> None:
            self.data = {"input_ids": input_ids}

        def __getattr__(self, item: str) -> object:
            try:
                return self.data[item]
            except KeyError:
                raise AttributeError from None

    wrapped = FakeBatchEncoding(ids)
    assert _template_ids(wrapped) is ids
    with pytest.raises(AttributeError):
        _ = wrapped.shape  # the exact failure mode the probe defends against
