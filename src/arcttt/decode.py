"""Constrained depth-first decoding over the ARC grid vocabulary.

Clean-room implementation of the search idea described in the NVARC 2025
paper: because a serialized ARC grid uses only digit tokens, a row separator,
and the end token, the model's next-token distribution can be restricted to
that small vocabulary and every completion whose cumulative negative
log-probability stays under a cutoff can be enumerated by depth-first search.
This yields many scored candidate grids per forward pass family, which the
voting layer then ranks. No code from the NVARC repository is used; the
implementation below is independent and expressed against the HuggingFace
KV-cache API.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from arcttt.serialize import text_to_grid
from arcttt.tasks import Grid, TaskFormatError


@dataclass(frozen=True)
class GridVocab:
    """Token ids for the characters that can appear in a serialized grid."""

    digit_ids: tuple[int, ...]  # ids for "0".."9", in digit order
    newline_ids: tuple[int, ...]  # ids that decode to a row separator
    stop_ids: tuple[int, ...]  # ids that end a grid (eos / im_end)

    def allowed(self) -> tuple[int, ...]:
        return self.digit_ids + self.newline_ids + self.stop_ids


def build_grid_vocab(tokenizer: Any) -> GridVocab:
    """Discover the grid-relevant token ids for a tokenizer, robustly.

    Digits and the newline may tokenize as standalone ids; when a tokenizer
    merges them into larger pieces this returns the single-character ids it
    can find and raises if a digit is unrepresentable, so failures are loud.
    """

    def safe_encode(text: str) -> list[int]:
        # Cut/WordLevel tokenizers without an UNK token raise on out-of-vocab
        # input instead of returning ids; treat that as "not representable".
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return []

    digit_ids = []
    for digit in "0123456789":
        ids = safe_encode(digit)
        if len(ids) != 1:
            raise TaskFormatError(f"digit {digit!r} does not map to a single token")
        digit_ids.append(ids[0])

    newline_ids = set()
    ids = safe_encode("\n")
    if len(ids) == 1:
        newline_ids.add(ids[0])
    # "Ċ" is the GPT2-style byte-level token NAME for "\n", not text —
    # it must be looked up with convert_tokens_to_ids, never encode().
    try:
        candidate = tokenizer.convert_tokens_to_ids("Ċ")
    except Exception:
        candidate = None
    if (
        isinstance(candidate, int)
        and candidate >= 0
        and candidate != getattr(tokenizer, "unk_token_id", None)
        and tokenizer.decode([candidate]) == "\n"
    ):
        newline_ids.add(candidate)
    if not newline_ids:
        raise TaskFormatError("no single-token newline found for grid decoding")

    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for special in ("<|im_end|>", "<|endoftext|>"):
        ids = safe_encode(special)
        if len(ids) == 1:
            stop_ids.add(ids[0])
    if not stop_ids:
        raise TaskFormatError("no stop token found for grid decoding")

    return GridVocab(
        digit_ids=tuple(digit_ids),
        newline_ids=tuple(sorted(newline_ids)),
        stop_ids=tuple(sorted(stop_ids)),
    )


def _cache_layers(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (key, value) tensors of a KV cache, across transformers APIs.

    Never iterates the cache object itself: what ``__iter__`` yields varies
    by build — the Kaggle image's transformers (a 5.x build) yielded raw 4-D
    tensors whose 2-way unpacking "iterates" the batch dimension, and current
    5.x (e.g. 5.15.0) yields (keys, values, sliding_window) 3-tuples — and
    either way ``key, value`` unpacking raises ``ValueError: too many values
    to unpack``; the v7 kernel lost 98 tasks to exactly that.
    Explicit attribute probes
    cover the layered API (>= 4.54, including 5.x), the key_cache/value_cache
    lists (4.36-4.53), and the legacy tuple-of-pairs format.
    """

    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer[0], layer[1]) for layer in cache]


def _rebuild_like(cache: Any, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Build a fresh cache of ``cache``'s concrete type from (key, value) pairs.

    ``update()`` is the one mutation entry point every DynamicCache version
    shares; ``from_legacy_cache`` no longer exists on transformers >= 5.
    """

    if hasattr(cache, "layers") or hasattr(cache, "key_cache"):
        rebuilt = type(cache)()
        for index, (key, value) in enumerate(layers):
            rebuilt.update(key, value, index)
        return rebuilt
    return tuple(layers)


def _crop_cache(cache: Any, length: int) -> Any:
    """Truncate a KV cache to ``length`` positions, whatever its concrete type."""

    if hasattr(cache, "crop"):
        excess = cache.get_seq_length() - length
        if excess > 0:
            # negative = remove that many tokens (non-deprecated on 4.x and
            # 5.x; a positive "absolute target" is legacy and flips meaning
            # to tokens_to_remove once transformers drops the legacy branch).
            cache.crop(-excess)
        return cache
    return _rebuild_like(
        cache,
        [(key[:, :, :length], value[:, :, :length]) for key, value in _cache_layers(cache)],
    )


def constrained_dfs(
    model: Any,
    prompt_ids: torch.Tensor,
    vocab: GridVocab,
    tokenizer: Any,
    *,
    max_score: float = -math.log(0.2),
    max_new_tokens: int = 992,
    max_candidates: int = 64,
    deadline: float | None = None,
) -> list[tuple[Grid, float]]:
    """Enumerate grid completions under a cumulative-NLL cutoff.

    Returns (grid, score) pairs sorted by ascending score (lower is better).
    The search keeps ONE shared KV cache and backtracks by cropping it — HF
    caches mutate in place on every forward, so per-beam cache references
    would alias each other and corrupt sibling branches. The invariant is
    that when a frame (tokens, pending expansions) is on top of the stack,
    the cache holds exactly prompt + tokens. Bounded by ``max_candidates``
    and, if given, a ``deadline`` on the monotonic clock
    (``time.monotonic()``).
    """

    device = model.device
    allowed = vocab.allowed()
    allowed_tensor = torch.tensor(allowed, device=device)
    stop = set(vocab.stop_ids)

    with torch.no_grad():
        # logits_to_keep=1: only the last position's logits are ever read,
        # and full-prompt full-vocab logits are the dominant prime tensor.
        primed = model(
            input_ids=prompt_ids.to(device),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    cache = primed.past_key_values
    prompt_length = prompt_ids.shape[1]

    def expansions_under_cutoff(
        logits: torch.Tensor, score: float
    ) -> list[tuple[float, int]]:
        log_probs = torch.log_softmax(logits[0], dim=-1)
        # one device sync for all allowed tokens, not one .item() per token
        steps = (-log_probs[allowed_tensor]).tolist()
        found = [
            (score + step, token)
            for step, token in zip(steps, allowed)
            if score + step < max_score
        ]
        found.sort(reverse=True)  # best (lowest NLL) last, popped first
        return found

    completed: list[tuple[Grid, float]] = []
    # frame: (tokens to reach this node, expansions not yet tried)
    stack: list[tuple[list[int], list[tuple[float, int]]]] = [
        ([], expansions_under_cutoff(primed.logits[:, -1].float(), 0.0))
    ]
    while stack and len(completed) < max_candidates:
        if deadline is not None and time.monotonic() > deadline:
            break
        tokens, expansions = stack[-1]
        if not expansions:
            stack.pop()
            if stack:  # restore the parent frame's cache state
                cache = _crop_cache(cache, prompt_length + len(tokens) - 1)
            continue
        total, token = expansions.pop()
        if token in stop:
            grid = _decode_grid(tokenizer, tokens)
            if grid is not None:
                completed.append((grid, total))
            continue
        if len(tokens) + 1 >= max_new_tokens:
            continue
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[token]], device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        cache = out.past_key_values
        stack.append(
            ([*tokens, token], expansions_under_cutoff(out.logits[:, -1].float(), total))
        )

    completed.sort(key=lambda pair: pair[1])
    return completed[:max_candidates]


_COMPACT_SLACK = 64  # stale cache columns tolerated before a compaction gather


@dataclass
class _FrameSearch:
    """DFS state for one prompt of the lockstep multi-frame search."""

    prompt_length: int
    # frame: (tokens to reach this node, expansions not yet tried)
    stack: list[tuple[list[int], list[tuple[float, int]]]]
    slots: list[int]  # physical cache column of each token on the current path
    completed: list[tuple[Grid, float]]
    done: bool = False
    # Why this frame stopped: "exhausted" (whole tree under the NLL bound was
    # searched), "candidate_cap" (hit max_candidates), or "deadline" (ran out
    # of wall-clock). This is the ONLY signal that separates a bound-limited
    # search from a time-limited one, and it decides whether the next lever is
    # a wider bound or a bigger time budget. Observation only - setting it
    # never changes what the search does.
    stop_reason: str = "exhausted"


def _expansions_under_cutoff(
    logits_row: torch.Tensor,
    score: float,
    allowed: tuple[int, ...],
    allowed_tensor: torch.Tensor,
    max_score: float,
) -> list[tuple[float, int]]:
    log_probs = torch.log_softmax(logits_row, dim=-1)
    # one device sync for all allowed tokens, not one .item() per token
    steps = (-log_probs[allowed_tensor]).tolist()
    found = [
        (score + step, token)
        for step, token in zip(steps, allowed)
        if score + step < max_score
    ]
    found.sort(reverse=True)  # best (lowest NLL) last, popped first
    return found


def _gather_cache(cache: Any, row_index: torch.Tensor, column_index: torch.Tensor) -> Any:
    """Select rows and per-row columns from a KV cache, returning a fresh cache."""

    layers = []
    for key, value in _cache_layers(cache):
        key = key.index_select(0, row_index.to(key.device))
        value = value.index_select(0, row_index.to(value.device))
        index = column_index.to(key.device)[:, None, :, None].expand(
            -1, key.shape[1], -1, key.shape[3]
        )
        layers.append((torch.gather(key, 2, index), torch.gather(value, 2, index)))
    return _rebuild_like(cache, layers)


def constrained_dfs_multi(
    model: Any,
    prompts: list[torch.Tensor],
    vocab: GridVocab,
    tokenizer: Any,
    *,
    max_score: float = -math.log(0.2),
    max_new_tokens: int = 992,
    max_candidates: int = 64,
    deadline: float | None = None,
    stats_out: list[tuple[str, int]] | None = None,
) -> list[list[tuple[Grid, float]]]:
    """Run ``constrained_dfs`` for several prompts in lockstep batched forwards.

    If ``stats_out`` is given, one ``(stop_reason, candidates_found)`` pair per
    frame is appended: why each frame's search ended and how much it found.
    Pure observation - it cannot change the search.

    Returns one (grid, score) list per prompt, each equal to what the
    single-frame search returns. Every step is ONE batched forward over the
    live frames instead of one forward per frame, which is the win when the
    augmentation frames of a task are searched together.

    Frames backtrack to different depths at different times, so the shared
    batched cache cannot be cropped physically the way ``constrained_dfs``
    crops its single cache. Cropping is logical instead: each row tracks the
    live cache columns holding its prompt plus its current DFS path, every
    batched forward appends one physical column, and the attention mask
    hides dead columns. RoPE positions are written per row from the row's
    logical length, never from the physical column, so attention sees
    exactly the tokens and positions the single-frame search would. Stale
    columns are reclaimed by an occasional gather (compaction) — never by
    copying caches per fork.
    """

    if not prompts:
        return []
    device = model.device
    allowed = vocab.allowed()
    allowed_tensor = torch.tensor(allowed, device=device)
    stop = set(vocab.stop_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # Prime all frames in one right-padded batch: row positions run 0..width-1,
    # so real tokens get their natural RoPE positions and pad columns are born dead.
    prompt_lengths = [int(ids.shape[1]) for ids in prompts]
    width = max(prompt_lengths)
    batch = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    live = torch.zeros((len(prompts), width), dtype=torch.bool)
    for row, ids in enumerate(prompts):
        batch[row, : prompt_lengths[row]] = ids[0].cpu()
        live[row, : prompt_lengths[row]] = True
    live = live.to(device)
    # Only one position per row is ever read from the prime logits; keeping
    # every position would materialize [rows, width, vocab] (tens of GiB at
    # the 8-frame / 8192-token operating point). logits_to_keep with a 1-D
    # index tensor keeps just the needed positions (transformers >= 4.49).
    keep = sorted({length - 1 for length in prompt_lengths})
    keep_pos = {position: index for index, position in enumerate(keep)}
    with torch.no_grad():
        primed = model(
            input_ids=batch.to(device),
            attention_mask=live.long(),
            use_cache=True,
            return_dict=True,
            logits_to_keep=torch.tensor(keep, device=device),
        )
    cache = primed.past_key_values
    physical_length = width

    states = []
    for row, length in enumerate(prompt_lengths):
        root = _expansions_under_cutoff(
            primed.logits[row, keep_pos[length - 1]].float(),
            0.0,
            allowed,
            allowed_tensor,
            max_score,
        )
        states.append(
            _FrameSearch(prompt_length=length, stack=[([], root)], slots=[], completed=[])
        )

    def advance(state: _FrameSearch, row: int) -> tuple[int, float] | None:
        """Do stack work until the frame needs a forward (or finishes)."""

        while True:
            if not state.stack or len(state.completed) >= max_candidates:
                state.stop_reason = (
                    "candidate_cap" if len(state.completed) >= max_candidates else "exhausted"
                )
                state.done = True
                return None
            if deadline is not None and time.monotonic() > deadline:
                state.stop_reason = "deadline"
                state.done = True
                return None
            tokens, expansions = state.stack[-1]
            if not expansions:
                state.stack.pop()
                if state.stack:  # logical crop: kill the popped token's column
                    live[row, state.slots.pop()] = False
                continue
            total, token = expansions.pop()
            if token in stop:
                grid = _decode_grid(tokenizer, tokens)
                if grid is not None:
                    state.completed.append((grid, total))
                continue
            if len(tokens) + 1 >= max_new_tokens:
                continue
            return token, total

    frames = list(range(len(prompts)))  # frame index of each batch row
    while True:
        pending: dict[int, tuple[int, float]] = {}
        for row, frame in enumerate(frames):
            if states[frame].done:
                continue
            step = advance(states[frame], row)
            if step is not None:
                pending[frame] = step
        if not pending:
            break

        needed = [states[frame].prompt_length + len(states[frame].slots) for frame in pending]
        if len(pending) < len(frames) or physical_length - max(needed) > _COMPACT_SLACK:
            # Reclaim stale columns and drop finished rows: gather each live
            # row's live columns into a fresh contiguous cache.
            keep = [frame for frame in frames if frame in pending]
            new_width = max(needed)
            column_index = torch.zeros((len(keep), new_width), dtype=torch.long)
            live = torch.zeros((len(keep), new_width), dtype=torch.bool, device=device)
            for row, frame in enumerate(keep):
                state = states[frame]
                columns = list(range(state.prompt_length)) + state.slots
                column_index[row, : len(columns)] = torch.tensor(columns, dtype=torch.long)
                live[row, : len(columns)] = True
                state.slots = list(range(state.prompt_length, len(columns)))
            row_index = torch.tensor([frames.index(frame) for frame in keep])
            cache = _gather_cache(cache, row_index, column_index)
            frames = keep
            physical_length = new_width

        step_ids = torch.full((len(frames), 1), pad_id, dtype=torch.long)
        positions = torch.zeros((len(frames), 1), dtype=torch.long)
        for row, frame in enumerate(frames):
            token, _ = pending[frame]
            step_ids[row, 0] = token
            positions[row, 0] = states[frame].prompt_length + len(states[frame].slots)
        mask = torch.cat([live, torch.ones((len(frames), 1), dtype=torch.bool, device=device)], 1)
        with torch.no_grad():
            out = model(
                input_ids=step_ids.to(device),
                position_ids=positions.to(device),
                attention_mask=mask.long(),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        cache = out.past_key_values
        live = mask
        logits = out.logits[:, -1].float()
        for row, frame in enumerate(frames):
            state = states[frame]
            token, total = pending[frame]
            state.slots.append(physical_length)
            state.stack.append(
                (
                    [*state.stack[-1][0], token],
                    _expansions_under_cutoff(
                        logits[row], total, allowed, allowed_tensor, max_score
                    ),
                )
            )
        physical_length += 1

    results = []
    for state in states:
        state.completed.sort(key=lambda pair: pair[1])
        results.append(state.completed[:max_candidates])
        if stats_out is not None:
            stats_out.append((state.stop_reason, len(state.completed)))
    return results


def _decode_grid(tokenizer: Any, tokens: list[int]) -> Grid | None:
    if not tokens:
        return None
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        return text_to_grid(text)
    except TaskFormatError:
        return None
