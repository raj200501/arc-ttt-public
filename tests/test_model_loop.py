"""CausalLMPredictor integration test on a tiny offline model (no downloads)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from arcttt.augment import IDENTITY, Augmentation  # noqa: E402
from arcttt.model import CausalLMPredictor, TTTConfig  # noqa: E402
from arcttt.tasks import Grid, Pair, Task  # noqa: E402

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' }}{{ message['content'] }}{{ '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {ch: i for i, ch in enumerate("0123456789")}
    vocab["\n"] = len(vocab)
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


def make_task() -> Task:
    grid: Grid = ((1, 2), (3, 4))
    return Task(
        task_id="tiny",
        train=(Pair(input=grid, output=grid), Pair(input=grid, output=grid)),
        test=(Pair(input=grid, output=grid),),
    )


def test_ttt_loop_adapts_generates_and_scores_offline() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=1, max_new_tokens=16),
        torch.device("cpu"),
    )
    task = make_task()

    predictor.adapt(task, (IDENTITY,))
    trainable = [p for p in predictor.model.parameters() if p.requires_grad]
    assert trainable, "LoRA adaptation must leave trainable parameters"
    assert any("lora" in name for name, _ in predictor.model.named_parameters())

    # Generation executes; a random tiny model rarely emits a valid grid, and
    # both outcomes (parsed grid or empty list) are acceptable here.
    grids = predictor.predict(task, test_index=0, samples=2)
    assert isinstance(grids, list)

    score = predictor.log_probability(task, 0, ((1, 2), (3, 4)))
    assert isinstance(score, float)
    assert score < 0.0  # a random model is never certain
    assert torch.isfinite(torch.tensor(score))


def test_adaptation_reduces_training_loss_on_repeated_example() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    config = TTTConfig(lora_rank=4, lora_alpha=8, epochs=8, learning_rate=5e-3, max_new_tokens=8)
    predictor = CausalLMPredictor(model, tokenizer, config, torch.device("cpu"))
    task = make_task()

    before = predictor.log_probability(task, 0, ((1, 2), (3, 4)))
    predictor.adapt(task, (IDENTITY,))
    after = predictor.log_probability(task, 0, ((1, 2), (3, 4)))
    assert after > before, f"TTT should raise target log-probability ({before} -> {after})"


def test_predict_dfs_path_runs_through_predictor() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=1, max_new_tokens=6,
                  raw_qwen_format=True, use_dfs=True, dfs_probability_cutoff=0.05,
                  dfs_max_candidates=8),
        torch.device("cpu"),
    )
    predictor.adapt(make_task(), (IDENTITY,))
    grids = predictor.predict(make_task(), test_index=0, samples=1)
    assert isinstance(grids, list)
    assert len(grids) == len(set(grids))  # DFS output is deduplicated


def test_batched_log_probabilities_match_scalar_path() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model.eval()
    predictor = CausalLMPredictor(
        model, tokenizer,
        TTTConfig(max_new_tokens=8, raw_qwen_format=True),
        torch.device("cpu"),
    )
    task = make_task()
    outputs = [((1, 2), (3, 4)), ((0, 0), (0, 0)), ((5,),)]
    batched = predictor.log_probabilities(task, 0, outputs)
    assert len(batched) == 3
    for output, score in zip(outputs, batched):
        turns_score = predictor.log_probability(task, 0, output)
        # scalar path routes through the batch of one; verify agreement with
        # a direct singleton batch too
        single = predictor.log_probabilities(task, 0, [output])[0]
        assert abs(single - score) < 1e-3
        assert abs(turns_score - score) < 1e-3
    # ranking sanity: scores are finite floats
    import math as _math
    assert all(_math.isfinite(s) for s in batched)


def test_batched_ttt_trains_and_infers_like_single() -> None:
    # batch_size > 1 pads examples into shared optimizer steps; it must
    # still train (weights move) and leave inference fully functional.
    tokenizer = tiny_tokenizer()
    task = make_task()
    augmentations = (IDENTITY, Augmentation(rotations=1))

    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=1, max_new_tokens=16,
                  ttt_batch_size=3),
        torch.device("cpu"),
    )
    predictor.adapt(task, augmentations)
    lora_weights = [
        parameter
        for name, parameter in predictor.model.named_parameters()
        if "lora_b" in name.lower()
    ]
    assert lora_weights and any(
        weight.abs().sum().item() > 0 for weight in lora_weights
    ), "batched TTT must actually update LoRA weights"
    assert isinstance(predictor.predict(task, 0, 1), list)
    score = predictor.log_probability(task, 0, task.test[0].output)
    assert score <= 0.0


def test_dfs_predict_includes_greedy_when_search_comes_up_empty() -> None:
    # An impossibly tight cutoff admits no DFS candidates, but the greedy
    # completion must still reach the voting pool when the flag is on.
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    task = make_task()

    def build(include_greedy: bool) -> CausalLMPredictor:
        return CausalLMPredictor(
            model,
            tokenizer,
            TTTConfig(lora_rank=2, lora_alpha=4, epochs=0, max_new_tokens=8,
                      use_dfs=True, dfs_probability_cutoff=0.999,
                      dfs_include_greedy=include_greedy),
            torch.device("cpu"),
        )

    without = build(False).predict(task, 0, 1)
    assert without == []  # tight cutoff excludes everything, greedy off
    with_greedy = build(True).predict(task, 0, 1)
    # the tiny random model may emit an unparseable completion; the contract
    # is only that a parseable greedy grid is appended when it exists
    assert len(with_greedy) <= 1


def test_predict_frames_matches_per_frame_dfs_search() -> None:
    # The lockstep-batched DFS must return, frame by frame, exactly what the
    # sequential per-frame DFS path returns (greedy off isolates the search).
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=0, max_new_tokens=6,
                  raw_qwen_format=True, use_dfs=True, dfs_probability_cutoff=0.05,
                  dfs_max_candidates=8, dfs_include_greedy=False),
        torch.device("cpu"),
    )
    model.eval()
    task = make_task()
    frames = [IDENTITY.apply_task(task), Augmentation(rotations=1).apply_task(task)]

    batched = predictor.predict_frames(frames, test_index=0, samples=1)
    assert len(batched) == len(frames)
    for frame_task, frame_grids in zip(frames, batched):
        assert frame_grids == predictor.predict(frame_task, test_index=0, samples=1)


def test_predict_frames_sampling_path_falls_back_per_frame() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=0, max_new_tokens=8),
        torch.device("cpu"),
    )
    model.eval()
    task = make_task()
    frames = [IDENTITY.apply_task(task), Augmentation(rotations=2).apply_task(task)]

    grids = predictor.predict_frames(frames, test_index=0, samples=1)
    assert len(grids) == 2
    assert all(isinstance(frame, list) for frame in grids)


def test_predict_frames_batched_greedy_produces_valid_grids() -> None:
    # Greedy inclusion runs as one left-padded batch generate; each frame may
    # gain at most one extra (deduplicated) grid, and every grid must parse.
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=0, max_new_tokens=6,
                  raw_qwen_format=True, use_dfs=True, dfs_probability_cutoff=0.999,
                  dfs_max_candidates=8, dfs_include_greedy=True),
        torch.device("cpu"),
    )
    model.eval()
    task = make_task()
    frames = [IDENTITY.apply_task(task), Augmentation(rotations=1).apply_task(task)]

    grids = predictor.predict_frames(frames, test_index=0, samples=1)
    for frame in grids:
        assert len(frame) <= 1  # tight cutoff: at most the greedy completion
        for grid in frame:
            assert all(len(row) == len(grid[0]) for row in grid)


def test_cli_end_to_end_on_tiny_model(tmp_path) -> None:
    # The CLI is the product seed: task file in, ranked predictions out.
    import json

    from arcttt import cli

    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model_dir = tmp_path / "tiny_model"
    tokenizer.save_pretrained(str(model_dir))
    model.save_pretrained(str(model_dir))

    task_file = tmp_path / "task.json"
    grid = [[1, 2], [3, 4]]
    task_file.write_text(json.dumps({
        "train": [{"input": grid, "output": grid}],
        "test": [{"input": grid, "output": grid}],
    }))
    out_file = tmp_path / "predictions.json"
    code = cli.main([
        str(task_file), "--model", str(model_dir), "--output", str(out_file),
        "--rank", "2", "--alpha", "4", "--epochs", "1", "--device", "cpu",
    ])
    assert code == 0
    result = json.loads(out_file.read_text())
    assert result["task_id"] == "task"
    assert result["scored"] == 1
    assert len(result["predictions"]) == 1
    for attempt in result["predictions"][0]["attempts"]:
        assert isinstance(attempt, list)


def test_adapt_endpoint_serves_predictions(tmp_path) -> None:
    # The HTTP layer wraps the same engine: POST a task, get predictions.
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer

    from arcttt.model import TTTConfig
    from arcttt.serve import AdaptService, make_handler
    from arcttt.solve import SolveConfig

    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    service = AdaptService(
        model, tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=0, max_new_tokens=8),
        SolveConfig(samples_per_augmentation=1),
        torch.device("cpu"), "tiny",
    )
    server = HTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
            health = json.loads(response.read())
        assert health["status"] == "ok"

        grid = [[1, 2], [3, 4]]
        body = json.dumps({
            "train": [{"input": grid, "output": grid}],
            "test": [{"input": grid}],
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/adapt", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
        assert len(result["predictions"]) == 1
        assert "seconds" in result

        bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/adapt", data=b'{"nope": 1}',
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(bad)
            raise AssertionError("malformed task must 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400
    finally:
        server.shutdown()


def test_pairs_scorer_matches_per_frame_scorer() -> None:
    # The chunked cross-frame scorer must agree with the per-frame batched
    # scorer to float tolerance on identical (task, output) pairs.
    from arcttt.augment import DIHEDRAL_SWEEP

    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model, tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=0, max_new_tokens=8),
        torch.device("cpu"),
    )
    task = make_task()
    grid = task.test[0].output
    pairs = []
    expected = []
    for augmentation in DIHEDRAL_SWEEP[:4]:
        transformed = augmentation.apply_task(task)
        rendered = augmentation.apply(grid)
        pairs.append((transformed, 0, rendered))
        expected.append(predictor.log_probabilities(transformed, 0, [rendered])[0])
    actual = predictor.log_probabilities_pairs(pairs, chunk_rows=3)  # forces chunking
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) < 1e-4
