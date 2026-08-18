"""Constrained DFS decoder tests on a tiny offline model."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from arcttt import decode
from arcttt.decode import build_grid_vocab, constrained_dfs, constrained_dfs_multi  # noqa: E402


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {ch: i for i, ch in enumerate("0123456789")}
    vocab["\n"] = len(vocab)
    for special in ("<|im_end|>", "<|pad|>"):
        vocab[special] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<|pad|>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<|pad|>", eos_token="<|im_end|>"
    )


def tiny_model(vocab_size: int) -> Qwen2ForCausalLM:
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return Qwen2ForCausalLM(config)


def test_build_grid_vocab_finds_digits_newline_stop() -> None:
    tokenizer = tiny_tokenizer()
    vocab = build_grid_vocab(tokenizer)
    assert len(vocab.digit_ids) == 10
    assert vocab.newline_ids
    assert tokenizer.eos_token_id in vocab.stop_ids
    # allowed set is digits + newline + stop, all distinct
    assert len(set(vocab.allowed())) == len(vocab.allowed())


def test_constrained_dfs_returns_scored_grids_within_cutoff() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model.eval()
    vocab = build_grid_vocab(tokenizer)
    prompt = tokenizer("12\n", return_tensors="pt", add_special_tokens=False).input_ids

    results = constrained_dfs(
        model,
        prompt,
        vocab,
        tokenizer,
        max_score=-math.log(0.05),  # generous cutoff so the tiny model yields some grids
        max_new_tokens=6,
        max_candidates=8,
    )
    # every returned item is a valid grid with a finite ascending score
    assert isinstance(results, list)
    scores = [score for _, score in results]
    assert scores == sorted(scores)
    for grid, score in results:
        assert all(len(row) == len(grid[0]) for row in grid)
        assert math.isfinite(score)


def test_constrained_dfs_respects_candidate_cap_and_tight_cutoff() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model.eval()
    vocab = build_grid_vocab(tokenizer)
    prompt = tokenizer("3\n", return_tensors="pt", add_special_tokens=False).input_ids

    capped = constrained_dfs(
        model, prompt, vocab, tokenizer, max_score=-math.log(0.02),
        max_new_tokens=5, max_candidates=3,
    )
    assert len(capped) <= 3

    # An impossibly tight cutoff admits nothing (no single token has prob > 0.999).
    empty = constrained_dfs(
        model, prompt, vocab, tokenizer, max_score=-math.log(0.999),
        max_new_tokens=5, max_candidates=8,
    )
    assert empty == []


def oracle_enumerate(model, tokenizer, prompt_ids, vocab, max_score, max_new_tokens):
    """Reference enumeration with full forward passes and no KV cache at all."""

    from arcttt.decode import _decode_grid

    allowed = vocab.allowed()
    stop = set(vocab.stop_ids)
    results = []

    def recurse(tokens: list[int], score: float) -> None:
        ids = prompt_ids
        if tokens:
            ids = torch.cat([prompt_ids, torch.tensor([tokens], dtype=torch.long)], dim=1)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[:, -1].float()
        log_probs = torch.log_softmax(logits[0], dim=-1)
        for token in allowed:
            total = score - float(log_probs[token].item())
            if total >= max_score:
                continue
            if token in stop:
                grid = _decode_grid(tokenizer, tokens)
                if grid is not None:
                    results.append((grid, total))
            elif len(tokens) + 1 < max_new_tokens:
                recurse([*tokens, token], total)

    recurse([], 0.0)
    return sorted(results, key=lambda pair: pair[1])


def test_constrained_dfs_matches_cache_free_oracle_exactly() -> None:
    # The DFS shares one KV cache across branches (crop-on-backtrack); any
    # cache aliasing between sibling beams shows up as score or set drift
    # against this cache-free reference enumeration.
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model.eval()
    vocab = build_grid_vocab(tokenizer)
    prompt = tokenizer("12\n", return_tensors="pt", add_special_tokens=False).input_ids

    # ~uniform tiny model: each step costs ~log(13) nats, so admitting the
    # 2-3 step completions (digits + stop) needs a cutoff past ~2*2.56.
    max_score = -math.log(0.001)
    expected = oracle_enumerate(model, tokenizer, prompt, vocab, max_score, 6)
    actual = constrained_dfs(
        model, prompt, vocab, tokenizer,
        max_score=max_score, max_new_tokens=6, max_candidates=10_000,
    )
    assert len(expected) > 3  # the comparison must actually exercise forks
    assert len(actual) == len(expected)
    for (grid_a, score_a), (grid_e, score_e) in zip(actual, expected):
        assert grid_a == grid_e
        assert score_a == pytest.approx(score_e, abs=1e-4)


def multi_fixture():
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model.eval()
    vocab = build_grid_vocab(tokenizer)
    texts = ("12\n", "3\n", "0505\n")  # deliberately different prompt lengths
    prompts = [
        tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        for text in texts
    ]
    return tokenizer, model, vocab, prompts


def test_constrained_dfs_multi_matches_cache_free_oracle_exactly() -> None:
    # The batched search shares ONE cache across frames that backtrack to
    # different depths at different times; any cross-row leakage, stale-column
    # attention, or RoPE position drift shows up against this cache-free
    # per-prompt reference enumeration.
    tokenizer, model, vocab, prompts = multi_fixture()
    max_score = -math.log(0.001)

    actual = constrained_dfs_multi(
        model, prompts, vocab, tokenizer,
        max_score=max_score, max_new_tokens=6, max_candidates=10_000,
    )
    assert len(actual) == len(prompts)
    for prompt, frame_actual in zip(prompts, actual):
        expected = oracle_enumerate(model, tokenizer, prompt, vocab, max_score, 6)
        assert len(expected) > 3  # the comparison must actually exercise forks
        assert len(frame_actual) == len(expected)
        for (grid_a, score_a), (grid_e, score_e) in zip(frame_actual, expected):
            assert grid_a == grid_e
            assert score_a == pytest.approx(score_e, abs=1e-4)


def test_constrained_dfs_multi_mixed_prompt_lengths_match_per_frame() -> None:
    # Right-padded priming + per-row logical lengths: frames whose prompts
    # differ in length must each reproduce their own single-frame search.
    tokenizer, model, vocab, prompts = multi_fixture()
    assert len({ids.shape[1] for ids in prompts}) > 1
    max_score = -math.log(0.02)

    batched = constrained_dfs_multi(
        model, prompts, vocab, tokenizer,
        max_score=max_score, max_new_tokens=5, max_candidates=8,
    )
    for prompt, frame_batched in zip(prompts, batched):
        sequential = constrained_dfs(
            model, prompt, vocab, tokenizer,
            max_score=max_score, max_new_tokens=5, max_candidates=8,
        )
        assert len(frame_batched) == len(sequential)
        for (grid_b, score_b), (grid_s, score_s) in zip(frame_batched, sequential):
            assert grid_b == grid_s
            assert score_b == pytest.approx(score_s, abs=1e-4)


def test_constrained_dfs_multi_compaction_preserves_results(monkeypatch) -> None:
    # A tiny slack forces the stale-column gather to run many times mid-search;
    # results must not move (compaction only relabels physical columns).
    tokenizer, model, vocab, prompts = multi_fixture()
    max_score = -math.log(0.001)

    relaxed = constrained_dfs_multi(
        model, prompts, vocab, tokenizer,
        max_score=max_score, max_new_tokens=6, max_candidates=10_000,
    )
    monkeypatch.setattr(decode, "_COMPACT_SLACK", 2)
    compacted = constrained_dfs_multi(
        model, prompts, vocab, tokenizer,
        max_score=max_score, max_new_tokens=6, max_candidates=10_000,
    )
    for frame_relaxed, frame_compacted in zip(relaxed, compacted):
        assert len(frame_relaxed) == len(frame_compacted)
        for (grid_r, score_r), (grid_c, score_c) in zip(frame_relaxed, frame_compacted):
            assert grid_r == grid_c
            assert score_r == pytest.approx(score_c, abs=1e-4)


def test_constrained_dfs_multi_is_deterministic_and_caps_candidates() -> None:
    tokenizer, model, vocab, prompts = multi_fixture()
    kwargs = {"max_score": -math.log(0.01), "max_new_tokens": 5, "max_candidates": 3}

    first = constrained_dfs_multi(model, prompts, vocab, tokenizer, **kwargs)
    second = constrained_dfs_multi(model, prompts, vocab, tokenizer, **kwargs)
    assert first == second  # bitwise-identical scores, same grids, same order
    assert all(len(frame) <= 3 for frame in first)

    assert constrained_dfs_multi(model, [], vocab, tokenizer, **kwargs) == []
    empty = constrained_dfs_multi(
        model, prompts, vocab, tokenizer,
        max_score=-math.log(0.999), max_new_tokens=5, max_candidates=8,
    )
    assert empty == [[], [], []]


class _LayeredOnlyCache:
    """Mimics transformers >= 5: layered API, no to_legacy_cache, and generic
    iteration yields raw tensors — the exact shape that cost v7 98 tasks."""

    class _Layer:
        def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
            self.keys = keys
            self.values = values

    def __init__(self) -> None:
        self.layers: list[_LayeredOnlyCache._Layer] = []

    def update(self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int):
        if layer_idx == len(self.layers):
            self.layers.append(self._Layer(keys, values))
        else:
            layer = self.layers[layer_idx]
            layer.keys = torch.cat([layer.keys, keys], dim=2)
            layer.values = torch.cat([layer.values, values], dim=2)
        return self.layers[layer_idx].keys, self.layers[layer_idx].values

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys  # raw tensors, like Cache.__iter__ on v5
            yield layer.values


def _filled(cache, batch: int = 3, layers: int = 2, seq: int = 5):
    torch.manual_seed(7)
    for index in range(layers):
        cache.update(torch.randn(batch, 2, seq, 4), torch.randn(batch, 2, seq, 4), index)
    return cache


def test_cache_layers_never_iterates_the_cache_object() -> None:
    cache = _filled(_LayeredOnlyCache())
    with pytest.raises(ValueError):  # proves the fake reproduces the v7 failure
        for _key, _value in cache:
            pass
    pairs = decode._cache_layers(cache)
    assert len(pairs) == 2
    assert all(key.shape == value.shape == (3, 2, 5, 4) for key, value in pairs)


def test_cache_layers_handles_dynamic_and_legacy_formats() -> None:
    from transformers import DynamicCache

    dynamic = _filled(DynamicCache())
    legacy = tuple(
        (key.clone(), value.clone()) for key, value in decode._cache_layers(dynamic)
    )
    for pairs in (decode._cache_layers(dynamic), decode._cache_layers(legacy)):
        assert len(pairs) == 2
        for (key, value), (want_key, want_value) in zip(pairs, legacy):
            assert torch.equal(key, want_key)
            assert torch.equal(value, want_value)


@pytest.mark.parametrize("factory", ["dynamic", "layered", "legacy"])
def test_gather_cache_selects_rows_and_columns_across_apis(factory: str) -> None:
    from transformers import DynamicCache

    if factory == "dynamic":
        cache = _filled(DynamicCache())
    elif factory == "layered":
        cache = _filled(_LayeredOnlyCache())
    else:
        cache = tuple(
            (key, value) for key, value in decode._cache_layers(_filled(DynamicCache()))
        )
    reference = [
        (key.clone(), value.clone()) for key, value in decode._cache_layers(cache)
    ]
    row_index = torch.tensor([2, 0])
    column_index = torch.tensor([[0, 2, 4], [1, 2, 3]])

    gathered = decode._gather_cache(cache, row_index, column_index)

    assert type(gathered) is type(cache)
    for (key, value), (want_key, want_value) in zip(
        decode._cache_layers(gathered), reference
    ):
        assert key.shape == value.shape == (2, 2, 3, 4)
        for out_row, src_row in enumerate([2, 0]):
            for out_col, src_col in enumerate(column_index[out_row].tolist()):
                assert torch.equal(key[out_row, :, out_col], want_key[src_row, :, src_col])
                assert torch.equal(
                    value[out_row, :, out_col], want_value[src_row, :, src_col]
                )


def test_crop_cache_without_crop_method_uses_layer_probes() -> None:
    cache = _filled(_LayeredOnlyCache())
    cropped = decode._crop_cache(cache, 3)
    pairs = decode._cache_layers(cropped)
    assert all(key.shape[2] == 3 and value.shape[2] == 3 for key, value in pairs)


def test_dfs_stop_reason_distinguishes_deadline_from_exhausted() -> None:
    """The three DFS stop reasons are the only signal that separates a
    time-limited search from a bound-limited one — which is the difference
    between 'buy more search time' and 'widen the NLL bound' as the next
    lever. Without it the only feedback is the next day's leaderboard score."""

    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model.eval()
    vocab = build_grid_vocab(tokenizer)
    prompt = tokenizer("12\n", return_tensors="pt", add_special_tokens=False).input_ids

    # generous bound + generous cap + no deadline -> the tree is fully searched
    stats: list[tuple[str, int]] = []
    with torch.no_grad():
        constrained_dfs_multi(
            model, [prompt], vocab, tokenizer,
            max_score=9.0, max_new_tokens=8, max_candidates=1000, stats_out=stats,
        )
    assert len(stats) == 1
    assert stats[0][0] == "exhausted", stats[0]

    # same search, tiny candidate cap -> the cap is what stopped it
    stats_cap: list[tuple[str, int]] = []
    with torch.no_grad():
        constrained_dfs_multi(
            model, [prompt], vocab, tokenizer,
            max_score=9.0, max_new_tokens=8, max_candidates=2, stats_out=stats_cap,
        )
    assert stats_cap[0][0] == "candidate_cap", stats_cap[0]
    assert stats_cap[0][1] >= 2

    # same search, already-expired deadline -> time is what stopped it
    stats_dl: list[tuple[str, int]] = []
    with torch.no_grad():
        constrained_dfs_multi(
            model, [prompt], vocab, tokenizer,
            max_score=9.0, max_new_tokens=8, max_candidates=1000,
            deadline=0.0, stats_out=stats_dl,
        )
    assert stats_dl[0][0] == "deadline", stats_dl[0]


def test_stats_out_does_not_change_search_results() -> None:
    """Telemetry must be pure observation."""

    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    model.eval()
    vocab = build_grid_vocab(tokenizer)
    prompt = tokenizer("12\n", return_tensors="pt", add_special_tokens=False).input_ids
    kwargs = dict(max_score=8.0, max_new_tokens=10, max_candidates=64)
    with torch.no_grad():
        without = constrained_dfs_multi(model, [prompt], vocab, tokenizer, **kwargs)
        collected: list[tuple[str, int]] = []
        with_stats = constrained_dfs_multi(
            model, [prompt], vocab, tokenizer, stats_out=collected, **kwargs
        )
    assert without == with_stats
    assert collected and collected[0][1] == len(with_stats[0])
