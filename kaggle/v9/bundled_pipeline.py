from __future__ import annotations
# arc-ttt bundled single-file pipeline (built by kaggle/build_bundle.py)


# === arcttt/tasks.py ===
"""ARC task data model: loading, validation, and exact-match scoring.

Grids are tuples of tuples of ints (colors 0-9), immutable and hashable so
augmentation dedup and vote counting stay trivial and bug-resistant.
"""



import json
from dataclasses import dataclass
from pathlib import Path

Grid = tuple[tuple[int, ...], ...]

MAX_SIDE = 30
COLORS = 10


class TaskFormatError(ValueError):
    """Raised when a task file violates the ARC schema."""


def to_grid(rows: object) -> Grid:
    if not isinstance(rows, list) or not rows:
        raise TaskFormatError("grid must be a non-empty list of rows")
    width = None
    out: list[tuple[int, ...]] = []
    for row in rows:
        if not isinstance(row, list) or not row:
            raise TaskFormatError("grid rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise TaskFormatError("grid rows must share one width")
        for cell in row:
            if not isinstance(cell, int) or not 0 <= cell < COLORS:
                raise TaskFormatError("grid cells must be ints in [0, 10)")
        out.append(tuple(row))
    if len(out) > MAX_SIDE or (width or 0) > MAX_SIDE:
        raise TaskFormatError(f"grid exceeds {MAX_SIDE}x{MAX_SIDE}")
    return tuple(out)


def grid_to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid | None  # None for hidden test outputs


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]

    def validate(self) -> None:
        if not self.train or not self.test:
            raise TaskFormatError(f"{self.task_id}: needs train and test pairs")
        for pair in self.train:
            if pair.output is None:
                raise TaskFormatError(f"{self.task_id}: train pairs need outputs")


def load_task(path: str | Path) -> Task:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "train" not in raw or "test" not in raw:
        raise TaskFormatError(f"{path.name}: missing train/test keys")

    def pairs(items: object, split: str) -> tuple[Pair, ...]:
        if not isinstance(items, list):
            raise TaskFormatError(f"{path.name}: {split} must be a list")
        built = []
        for item in items:
            if not isinstance(item, dict) or "input" not in item:
                raise TaskFormatError(f"{path.name}: {split} pair needs input")
            output = item.get("output")
            built.append(
                Pair(
                    input=to_grid(item["input"]),
                    output=to_grid(output) if output is not None else None,
                )
            )
        return tuple(built)

    task = Task(task_id=path.stem, train=pairs(raw["train"], "train"), test=pairs(raw["test"], "test"))
    task.validate()
    return task


def load_directory(directory: str | Path) -> dict[str, Task]:
    directory = Path(directory)
    tasks = {}
    for path in sorted(directory.glob("*.json")):
        task = load_task(path)
        tasks[task.task_id] = task
    if not tasks:
        raise TaskFormatError(f"no task files under {directory}")
    return tasks


def score_attempts(attempts: list[Grid], solution: Grid, max_attempts: int = 2) -> bool:
    """ARC scoring: correct if ANY of the first `max_attempts` attempts matches exactly."""

    return any(attempt == solution for attempt in attempts[:max_attempts])


# === arcttt/augment.py ===
"""Invertible grid augmentations for test-time training.

The dihedral group D4 (rotations + reflections) and color permutations are the
standard TTT augmentation family for ARC: apply a transform to every grid of a
task, adapt/predict in the transformed frame, then invert the transform on the
prediction before voting. Every augmentation here therefore carries an exact
inverse, and round-trip identity is unit-tested.
"""



from dataclasses import dataclass



def rotate90(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""

    return tuple(tuple(row[i] for row in reversed(grid)) for i in range(len(grid[0])))


def flip_horizontal(grid: Grid) -> Grid:
    """Mirror left-right."""

    return tuple(tuple(reversed(row)) for row in grid)


def apply_dihedral(grid: Grid, rotations: int, flip: bool) -> Grid:
    out = grid
    for _ in range(rotations % 4):
        out = rotate90(out)
    if flip:
        out = flip_horizontal(out)
    return out


def invert_dihedral(grid: Grid, rotations: int, flip: bool) -> Grid:
    out = grid
    if flip:
        out = flip_horizontal(out)
    for _ in range((-rotations) % 4):
        out = rotate90(out)
    return out


def apply_palette(grid: Grid, palette: tuple[int, ...]) -> Grid:
    return tuple(tuple(palette[cell] for cell in row) for row in grid)


def invert_palette(palette: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(palette)
    for source, target in enumerate(palette):
        inverse[target] = source
    return tuple(inverse)


@dataclass(frozen=True)
class Augmentation:
    """One invertible task transform: dihedral element + color permutation."""

    rotations: int = 0
    flip: bool = False
    palette: tuple[int, ...] = tuple(range(COLORS))

    def validate(self) -> None:
        if sorted(self.palette) != list(range(COLORS)):
            raise ValueError("palette must be a permutation of the colors")
        if not 0 <= self.rotations < 4:
            raise ValueError("rotations must be in [0, 4)")

    def apply(self, grid: Grid) -> Grid:
        return apply_palette(apply_dihedral(grid, self.rotations, self.flip), self.palette)

    def invert(self, grid: Grid) -> Grid:
        return invert_dihedral(
            apply_palette(grid, invert_palette(self.palette)), self.rotations, self.flip
        )

    def apply_task(self, task: Task) -> Task:
        def convert(pair: Pair) -> Pair:
            return Pair(
                input=self.apply(pair.input),
                output=self.apply(pair.output) if pair.output is not None else None,
            )

        return Task(
            task_id=task.task_id,
            train=tuple(convert(pair) for pair in task.train),
            test=tuple(convert(pair) for pair in task.test),
        )


IDENTITY = Augmentation()

#: The eight dihedral elements with identity palette — the standard TTT sweep.
DIHEDRAL_SWEEP: tuple[Augmentation, ...] = tuple(
    Augmentation(rotations=rotations, flip=flip)
    for flip in (False, True)
    for rotations in range(4)
)


def expanded_sweep(
    seed: int, palettes_per_element: int, fix_background: bool = False
) -> tuple[Augmentation, ...]:
    """Dihedral sweep crossed with seeded color permutations, for TTT data.

    Each of the eight dihedral elements appears once with the identity palette
    and ``palettes_per_element`` more times with distinct seeded permutations,
    so the identity-frame demonstrations are always in the training mix.
    """

    augmentations: list[Augmentation] = []
    for index, element in enumerate(DIHEDRAL_SWEEP):
        augmentations.append(element)
        for palette in deterministic_palettes(
            seed + index, palettes_per_element, fix_background=fix_background
        ):
            augmentations.append(
                Augmentation(
                    rotations=element.rotations, flip=element.flip, palette=palette
                )
            )
    return tuple(augmentations)


def deterministic_palettes(
    seed: int, count: int, fix_background: bool = True
) -> tuple[tuple[int, ...], ...]:
    """Seeded color permutations; champion-style full permutations when unfixed."""

    import random

    rng = random.Random(seed)
    palettes = []
    for _ in range(count):
        if fix_background:
            rest = list(range(1, COLORS))
            rng.shuffle(rest)
            palettes.append((0, *rest))
        else:
            everything = list(range(COLORS))
            rng.shuffle(everything)
            palettes.append(tuple(everything))
    return tuple(palettes)


# === arcttt/serialize.py ===
"""Grid and task serialization for language-model training and inference.

Clean-room implementation of the digit-serialization scheme described in the
NVARC 2025 paper (grids as digit rows inside chat turns): each grid row is a
line of digits, demonstration pairs become user/assistant turns, and the test
input is the final user turn awaiting an assistant completion. No code from
the NVARC repository (which carries no license) is used.
"""



from dataclasses import dataclass



def grid_to_text(grid: Grid) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def text_to_grid(text: str) -> Grid:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if not all(ch.isdigit() for ch in line):
            raise TaskFormatError(f"non-digit character in grid text: {line[:20]!r}")
        rows.append([int(ch) for ch in line])
    return to_grid(rows)


@dataclass(frozen=True)
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


def task_to_messages(task: Task, test_index: int = 0) -> tuple[ChatTurn, ...]:
    """Serialize demonstrations plus one test input as chat turns.

    The model is expected to complete the final assistant turn with the
    serialized output grid.
    """

    if not 0 <= test_index < len(task.test):
        raise TaskFormatError(f"{task.task_id}: test index {test_index} out of range")
    turns: list[ChatTurn] = []
    for pair in task.train:
        if pair.output is None:
            raise TaskFormatError(f"{task.task_id}: train pair missing output")
        turns.append(ChatTurn("user", grid_to_text(pair.input)))
        turns.append(ChatTurn("assistant", grid_to_text(pair.output)))
    turns.append(ChatTurn("user", grid_to_text(task.test[test_index].input)))
    return tuple(turns)


def ttt_training_examples(
    task: Task, shuffle_seed: int | None = None
) -> tuple[tuple[ChatTurn, ...], ...]:
    """Leave-one-out demonstration examples for per-task adaptation.

    For each demonstration pair, the remaining pairs act as context and the
    held-out pair supplies the supervised completion — the standard per-task
    TTT corpus construction, multiplied later by augmentations. A shuffle seed
    permutes the context-pair order deterministically, so different
    augmentations of the same task present the demonstrations differently.
    """

    import random

    order_rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    examples = []
    for held_out in range(len(task.train)):
        context = [
            index
            for index, pair in enumerate(task.train)
            if index != held_out and pair.output is not None
        ]
        if order_rng is not None:
            order_rng.shuffle(context)
        turns: list[ChatTurn] = []
        for index in context:
            pair = task.train[index]
            if pair.output is None:  # excluded above; narrows the type
                continue
            turns.append(ChatTurn("user", grid_to_text(pair.input)))
            turns.append(ChatTurn("assistant", grid_to_text(pair.output)))
        target = task.train[held_out]
        if target.output is None:
            continue
        turns.append(ChatTurn("user", grid_to_text(target.input)))
        turns.append(ChatTurn("assistant", grid_to_text(target.output)))
        examples.append(tuple(turns))
    if not examples:
        raise TaskFormatError(f"{task.task_id}: no usable TTT examples")
    return tuple(examples)


# === arcttt/vote.py ===
"""Candidate pooling, augmentation-inverse mapping, and selection scoring.

Clean-room implementation of the selection scheme described in the NVARC 2025
paper: candidates found by search are re-scored under a fixed set of
augmentations, and the final ranking combines how often a candidate was found
with the geometric mean of its probabilities across augmentations
(equivalently, the mean log-probability). Two attempts per test input are
submitted, so selection returns an ordered list.
"""



import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass



@dataclass(frozen=True)
class Candidate:
    """A predicted output grid in the ORIGINAL (un-augmented) frame."""

    grid: Grid
    found_count: int
    mean_log_probability: float


def pool_predictions(
    predictions: Iterable[tuple[Augmentation, Grid]],
) -> dict[Grid, int]:
    """Invert each prediction back to the original frame and count occurrences."""

    counts: Counter[Grid] = Counter()
    for augmentation, grid in predictions:
        counts[augmentation.invert(grid)] += 1
    return dict(counts)


def rescore_candidates(
    counts: dict[Grid, int],
    augmentations: tuple[Augmentation, ...],
    log_probability: Callable[[Grid, Augmentation], float],
) -> tuple[Candidate, ...]:
    """Score every pooled candidate under the same augmentation set.

    ``log_probability(grid, augmentation)`` must return the model's
    log-probability of the candidate rendered in that augmentation's frame.
    Using one identical augmentation set for every candidate keeps scores
    comparable (a property the NVARC paper calls out explicitly).
    """

    if not augmentations:
        raise ValueError("rescoring needs at least one augmentation")
    candidates = []
    for grid, found in counts.items():
        scores = [log_probability(grid, augmentation) for augmentation in augmentations]
        candidates.append(
            Candidate(
                grid=grid,
                found_count=found,
                mean_log_probability=sum(scores) / len(scores),
            )
        )
    return tuple(candidates)


def select_attempts(candidates: Iterable[Candidate], attempts: int = 2) -> tuple[Grid, ...]:
    """Rank by (found_count + normalized probability score), descending.

    The combined score follows the paper's shape: occurrence count plus the
    geometric-mean probability term. exp(mean log p) lies in (0, 1], so it
    breaks ties between equal counts without ever outweighing one extra find.
    """

    ranked = sorted(
        candidates,
        key=lambda c: c.found_count + math.exp(c.mean_log_probability),
        reverse=True,
    )
    return tuple(candidate.grid for candidate in ranked[:attempts])


# === arcttt/lora.py ===
"""Minimal LoRA for per-task adaptation, dependency-free beyond torch.

The Kaggle competition image does not ship ``peft`` and submission reruns are
offline, so per-task adaptation must not depend on it. This module injects
low-rank trainable deltas into the linear layers of a frozen model and can
remove them again, which is all the TTT loop needs. It is a small, tested
reimplementation of the standard LoRA update (rank-decomposed A@B added to a
frozen weight), not a general adapter framework.
"""



import math
from typing import Any

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Wraps a frozen linear layer with a trainable low-rank delta."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, use_rslora: bool) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        # scaling: rslora uses alpha/sqrt(rank), classic uses alpha/rank
        self.scaling = alpha / (math.sqrt(rank) if use_rslora else rank)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.base(x)
        delta = torch.nn.functional.linear(x, self.lora_a)
        delta = torch.nn.functional.linear(delta, self.lora_b)
        return out + self.scaling * delta


def _named_linears(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Linear]]:
    found = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            found.append((path, child))
        else:
            found.extend(_named_linears(child, path))
    return found


def _set_submodule(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    *,
    use_rslora: bool = True,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> list[str]:
    """Replace every eligible ``nn.Linear`` with a LoRA-wrapped version.

    Returns the paths that were wrapped. All non-LoRA parameters are frozen.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped = []
    for path, linear in _named_linears(model):
        if any(skip in path for skip in skip_names):
            continue
        _set_submodule(model, path, LoRALinear(linear, rank, alpha, use_rslora))
        wrapped.append(path)
    if not wrapped:
        raise RuntimeError("no linear layers found to wrap with LoRA")
    return wrapped


def remove_lora(model: nn.Module) -> None:
    """Restore the original frozen linear layers, dropping all LoRA deltas."""

    for path, module in list(_named_linears_including_lora(model)):
        if isinstance(module, LoRALinear):
            _set_submodule(model, path, module.base)


def _named_linears_including_lora(
    module: nn.Module, prefix: str = ""
) -> list[tuple[str, nn.Module]]:
    found: list[tuple[str, nn.Module]] = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRALinear):
            found.append((path, child))
        else:
            found.extend(_named_linears_including_lora(child, path))
    return found


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]


def trainable_parameter_count(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# === arcttt/decode.py ===
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



import math
import time
from dataclasses import dataclass
from typing import Any

import torch



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
    for candidate in ("\n", "Ċ"):
        ids = safe_encode(candidate)
        if len(ids) == 1:
            newline_ids.add(ids[0])
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

    Never iterates the cache object itself: on transformers >= 5 generic
    iteration yields raw tensors, and unpacking a 4-D tensor "iterates" its
    batch dimension — the v7 kernel lost 98 tasks to exactly that
    (``ValueError: too many values to unpack``). Explicit attribute probes
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
        cache.crop(length)
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
    and, if given, a wall-clock ``deadline``.
    """

    device = model.device
    allowed = vocab.allowed()
    allowed_tensor = torch.tensor(allowed, device=device)
    stop = set(vocab.stop_ids)

    with torch.no_grad():
        primed = model(input_ids=prompt_ids.to(device), use_cache=True, return_dict=True)
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
        if deadline is not None and time.time() > deadline:
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
) -> list[list[tuple[Grid, float]]]:
    """Run ``constrained_dfs`` for several prompts in lockstep batched forwards.

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
    with torch.no_grad():
        primed = model(
            input_ids=batch.to(device),
            attention_mask=live.long(),
            use_cache=True,
            return_dict=True,
        )
    cache = primed.past_key_values
    physical_length = width

    states = []
    for row, length in enumerate(prompt_lengths):
        root = _expansions_under_cutoff(
            primed.logits[row, length - 1].float(), 0.0, allowed, allowed_tensor, max_score
        )
        states.append(
            _FrameSearch(prompt_length=length, stack=[([], root)], slots=[], completed=[])
        )

    def advance(state: _FrameSearch, row: int) -> tuple[int, float] | None:
        """Do stack work until the frame needs a forward (or finishes)."""

        while True:
            if not state.stack or len(state.completed) >= max_candidates:
                state.done = True
                return None
            if deadline is not None and time.time() > deadline:
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
    return results


def _decode_grid(tokenizer: Any, tokens: list[int]) -> Grid | None:
    if not tokens:
        return None
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        return text_to_grid(text)
    except TaskFormatError:
        return None


# === arcttt/model.py ===
"""Per-task test-time training and generation.

The solver is model-agnostic behind ``Predictor``; the real implementation
wraps a causal LM with a fresh LoRA adapter per task (the NVARC shape:
adapt on leave-one-out augmented demonstrations, then generate the test
output in each augmentation's frame). Everything here is CPU-runnable with a
tiny model so the pipeline is testable offline; scale comes from swapping the
base model and device, not from changing this code.
"""



from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch



class Predictor(Protocol):
    """Adapts to one task, then predicts and scores in augmented frames."""

    def adapt(self, task: Task, augmentations: Sequence[Augmentation]) -> None: ...

    def predict(self, task: Task, test_index: int, samples: int) -> list[Grid]: ...

    def log_probability(self, task: Task, test_index: int, output: Grid) -> float: ...


@dataclass(frozen=True)
class TTTConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 5e-5
    epochs: int = 1
    max_new_tokens: int = 992  # 30 rows * (30 digits + newline) + margin
    max_sequence_tokens: int = 8192
    temperature: float = 0.7
    raw_qwen_format: bool = False  # champion-style <|im_start|> framing, no system turn
    gradient_checkpointing: bool = False
    use_dfs: bool = False  # constrained DFS decoding instead of sampled generation
    dfs_probability_cutoff: float = 0.2  # keep completions with per-step prob >= this
    dfs_max_candidates: int = 32
    shuffle_examples: bool = False  # permute demonstration order per augmentation
    ttt_batch_size: int = 1  # examples per optimizer step (padded batch)
    dfs_time_budget_seconds: float | None = None  # wall-clock cap per predict()
    dfs_include_greedy: bool = True  # always add the greedy completion; the
    # cumulative-NLL cutoff can exclude even the argmax path, and greedy
    # guarantees one candidate per augmentation frame for the voting pool.


def turns_to_raw_qwen(turns: Sequence[ChatTurn], add_generation_prompt: bool) -> str:
    """Champion-format serialization: no system turn, no inter-turn newlines."""

    text = "".join(
        f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>" for turn in turns
    )
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def turns_to_chat(turns: Sequence[ChatTurn]) -> list[dict[str, str]]:
    return [{"role": turn.role, "content": turn.content} for turn in turns]


class CausalLMPredictor:
    """LoRA-per-task TTT over a HuggingFace causal LM. Device-agnostic."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: TTTConfig,
        device: torch.device,
    ) -> None:
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model: Any = model
        self._grid_vocab: Any = None

    # -- adaptation ---------------------------------------------------------

    def adapt(self, task: Task, augmentations: Sequence[Augmentation]) -> None:
        examples: list[tuple[ChatTurn, ...]] = []
        for augmentation_index, augmentation in enumerate(augmentations):
            transformed = augmentation.apply_task(task)
            shuffle_seed = (
                augmentation_index if self.config.shuffle_examples else None
            )
            examples.extend(ttt_training_examples(transformed, shuffle_seed))
        self.adapt_on_examples(examples)

    def adapt_on_examples(self, examples: Sequence[Sequence[ChatTurn]]) -> None:
        """Inject a fresh LoRA adapter and train it on supervised chat examples.

        Task-agnostic core of ``adapt``: each example is a chat-turn sequence
        whose final assistant turn is the supervised completion (grid or text
        alike — the text-mode TTT path calls this directly)."""


        remove_lora(self.base_model)  # drop any adapter from a prior task
        inject_lora(
            self.base_model,
            self.config.lora_rank,
            self.config.lora_alpha,
            use_rslora=True,
        )
        self.model = self.base_model
        if self.config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            # Frozen embeddings mean the checkpointed segment has no input
            # requiring grad; this hook makes the embedding output require it.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
        self.model.train()
        optimizer = torch.optim.AdamW(
            lora_parameters(self.model), lr=self.config.learning_rate
        )
        encoded = [
            batch
            for turns in examples
            if (batch := self._encode(turns, supervise_final=True)) is not None
        ]
        batch_size = max(1, self.config.ttt_batch_size)
        if batch_size > 1:
            # Sort by length so padded batches stay dense (batch_size 1 keeps
            # the original example order and exact legacy behavior).
            encoded.sort(key=lambda pair: pair[0].shape[1])
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        for _ in range(self.config.epochs):
            for start in range(0, len(encoded), batch_size):
                chunk = encoded[start : start + batch_size]
                if len(chunk) == 1:
                    input_ids, labels = chunk[0]
                    attention_mask = torch.ones_like(input_ids)
                else:
                    width = max(ids.shape[1] for ids, _ in chunk)
                    input_ids = torch.full(
                        (len(chunk), width), pad_id, dtype=torch.long
                    )
                    labels = torch.full((len(chunk), width), -100, dtype=torch.long)
                    attention_mask = torch.zeros(
                        (len(chunk), width), dtype=torch.long
                    )
                    for row, (ids, label_row) in enumerate(chunk):
                        length = ids.shape[1]
                        input_ids[row, :length] = ids[0].cpu()
                        labels[row, :length] = label_row[0].cpu()
                        attention_mask[row, :length] = 1
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                optimizer.zero_grad()
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                loss.backward()
                optimizer.step()
        if self.config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_disable"
        ):
            self.model.gradient_checkpointing_disable()  # cached generation next
        self.model.eval()

    # -- inference ----------------------------------------------------------

    def predict(self, task: Task, test_index: int, samples: int) -> list[Grid]:
        turns = task_to_messages(task, test_index)
        prompt = self._prompt_ids(turns)
        if prompt is None:
            return []
        if self.config.use_dfs:
            return self._predict_dfs(prompt)
        grids: list[Grid] = []
        for text in self._sample_texts(prompt, samples):
            try:
                grids.append(text_to_grid(text))
            except TaskFormatError:
                continue
        return grids

    def _sample_texts(self, prompt: torch.Tensor, samples: int) -> list[str]:
        """Decoded completions for one prompt: greedy first, then samples."""

        texts: list[str] = []
        attention_mask = torch.ones_like(prompt)
        with torch.no_grad():
            for sample in range(samples):
                generated = self.model.generate(
                    input_ids=prompt,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=sample > 0,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )
                texts.append(
                    self.tokenizer.decode(
                        generated[0][prompt.shape[1] :], skip_special_tokens=True
                    )
                )
        return texts

    def _predict_dfs(self, prompt: torch.Tensor) -> list[Grid]:
        import math
        import time


        if self._grid_vocab is None:
            self._grid_vocab = build_grid_vocab(self.tokenizer)
        budget = self.config.dfs_time_budget_seconds
        results = constrained_dfs(
            self.model,
            prompt,
            self._grid_vocab,
            self.tokenizer,
            max_score=-math.log(self.config.dfs_probability_cutoff),
            max_new_tokens=self.config.max_new_tokens,
            max_candidates=self.config.dfs_max_candidates,
            deadline=time.time() + budget if budget is not None else None,
        )
        seen: set[Grid] = set()
        ordered: list[Grid] = []
        for grid, _score in results:
            if grid not in seen:
                seen.add(grid)
                ordered.append(grid)
        if self.config.dfs_include_greedy:
            attention_mask = torch.ones_like(prompt)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=prompt,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                generated[0][prompt.shape[1] :], skip_special_tokens=True
            )
            try:
                greedy = text_to_grid(text)
            except TaskFormatError:
                greedy = None
            if greedy is not None and greedy not in seen:
                ordered.append(greedy)
        return ordered

    def predict_frames(
        self, tasks: Sequence[Task], test_index: int, samples: int
    ) -> list[list[Grid]]:
        """Predict for several augmentation frames of one task at once.

        With DFS decoding the frames' searches run in lockstep batched
        forwards (``constrained_dfs_multi``) instead of one sequential
        search per frame; the sampling path falls back to per-frame
        ``predict``. Results per frame match the per-frame path.
        """

        if not self.config.use_dfs:
            return [self.predict(task, test_index, samples) for task in tasks]
        import math
        import time


        prompts = [
            self._prompt_ids(task_to_messages(task, test_index)) for task in tasks
        ]
        live = [(i, p) for i, p in enumerate(prompts) if p is not None]
        per_frame: list[list[Grid]] = [[] for _ in tasks]
        if not live:
            return per_frame
        if self._grid_vocab is None:
            self._grid_vocab = build_grid_vocab(self.tokenizer)
        budget = self.config.dfs_time_budget_seconds
        results = constrained_dfs_multi(
            self.model,
            [prompt for _, prompt in live],
            self._grid_vocab,
            self.tokenizer,
            max_score=-math.log(self.config.dfs_probability_cutoff),
            max_new_tokens=self.config.max_new_tokens,
            max_candidates=self.config.dfs_max_candidates,
            deadline=time.time() + budget if budget is not None else None,
        )
        greedy: list[Grid | None] = [None] * len(live)
        if self.config.dfs_include_greedy:
            greedy = self._greedy_grids([prompt for _, prompt in live])
        for (index, _), frame_results, greedy_grid in zip(live, results, greedy):
            seen: set[Grid] = set()
            ordered: list[Grid] = []
            for grid, _score in frame_results:
                if grid not in seen:
                    seen.add(grid)
                    ordered.append(grid)
            if greedy_grid is not None and greedy_grid not in seen:
                ordered.append(greedy_grid)
            per_frame[index] = ordered
        return per_frame

    def _greedy_grids(self, prompts: Sequence[torch.Tensor]) -> list[Grid | None]:
        """Greedy completion per prompt via one left-padded batch generate."""

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        width = max(prompt.shape[1] for prompt in prompts)
        ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            length = prompt.shape[1]
            ids[row, width - length :] = prompt[0].cpu()
            mask[row, width - length :] = 1
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        grids: list[Grid | None] = []
        for row in range(len(prompts)):
            text = self.tokenizer.decode(generated[row][width:], skip_special_tokens=True)
            try:
                grids.append(text_to_grid(text))
            except TaskFormatError:
                grids.append(None)
        return grids

    def log_probability(self, task: Task, test_index: int, output: Grid) -> float:
        return self.log_probabilities(task, test_index, [output])[0]

    def log_probabilities_pairs(
        self,
        pairs: Sequence[tuple[Task, int, Grid]],
        chunk_rows: int = 12,
    ) -> list[float]:
        """Score heterogeneous (task, test_index, output) pairs, chunked.

        One padded batch forward per chunk instead of one call per
        augmentation frame; chunking bounds activation memory (the cut
        16-token vocabulary keeps the logits tensor negligible)."""

        encoded: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for task, test_index, output in pairs:
            turns = task_to_messages(task, test_index) + (
                ChatTurn("assistant", grid_to_text(output)),
            )
            encoded.append(self._encode(turns, supervise_final=True))
        scores = [float("-inf")] * len(pairs)
        live = [(i, e) for i, e in enumerate(encoded) if e is not None]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        for start in range(0, len(live), chunk_rows):
            chunk = live[start : start + chunk_rows]
            width = max(e[0].shape[1] for _, e in chunk)
            ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            labels = torch.full((len(chunk), width), -100, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, (_, (input_ids, label_row)) in enumerate(chunk):
                length = input_ids.shape[1]
                ids[row, width - length :] = input_ids[0].cpu()
                labels[row, width - length :] = label_row[0].cpu()
                mask[row, width - length :] = 1
            with torch.no_grad():
                # use_cache=False: scoring needs logits only; the config
                # default (True) materializes a multi-GB throwaway KV cache
                # per chunk — the dominant OOM at the rescore stage on T4.
                logits = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    use_cache=False,
                ).logits.float()
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = labels[:, 1:].to(self.device)
            supervised = targets != -100
            gathered = log_probs.gather(
                -1, targets.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            for row, (index, _) in enumerate(chunk):
                positions = supervised[row]
                count = int(positions.sum().item())
                if count:
                    scores[index] = (
                        float(gathered[row][positions].sum().item()) / count
                    )
        return scores

    def log_probabilities(
        self, task: Task, test_index: int, outputs: Sequence[Grid]
    ) -> list[float]:
        """Score many candidate outputs in one left-padded batch forward."""

        sequences = [
            task_to_messages(task, test_index)
            + (ChatTurn("assistant", grid_to_text(output)),)
            for output in outputs
        ]
        return self.score_turn_sequences(sequences)

    def score_turn_sequences(self, sequences: Sequence[Sequence[ChatTurn]]) -> list[float]:
        """Mean supervised-token log-probability per chat sequence, one batch.

        Task-agnostic core of ``log_probabilities``: each sequence must end
        with the assistant turn being scored (grid or text alike)."""

        encoded = [self._encode(turns, supervise_final=True) for turns in sequences]
        live = [(i, e) for i, e in enumerate(encoded) if e is not None]
        scores = [float("-inf")] * len(sequences)
        if not live:
            return scores
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        width = max(e[0].shape[1] for _, e in live)
        ids = torch.full((len(live), width), pad_id, dtype=torch.long)
        labels = torch.full((len(live), width), -100, dtype=torch.long)
        mask = torch.zeros((len(live), width), dtype=torch.long)
        for row, (_, (input_ids, label_row)) in enumerate(live):
            length = input_ids.shape[1]
            ids[row, width - length :] = input_ids[0].cpu()
            labels[row, width - length :] = label_row[0].cpu()
            mask[row, width - length :] = 1
        with torch.no_grad():
            # use_cache=False for the same reason as the chunked scorer above.
            logits = self.model(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                use_cache=False,
            ).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = labels[:, 1:].to(self.device)
        supervised = targets != -100
        safe_targets = targets.clamp(min=0)
        gathered = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        for row, (index, _) in enumerate(live):
            positions = supervised[row]
            count = int(positions.sum().item())
            if count == 0:
                continue
            scores[index] = float(gathered[row][positions].sum().item()) / count
        return scores

    # -- encoding -----------------------------------------------------------

    def _prompt_ids(self, turns: Sequence[ChatTurn]) -> torch.Tensor | None:
        if self.config.raw_qwen_format:
            ids = self.tokenizer(
                turns_to_raw_qwen(turns, add_generation_prompt=True),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
        else:
            ids = self.tokenizer.apply_chat_template(
                turns_to_chat(turns),
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if ids.shape[1] > self.config.max_sequence_tokens:
            return None
        return cast(torch.Tensor, ids.to(self.device))

    def _encode(
        self, turns: Sequence[ChatTurn], supervise_final: bool
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Token ids plus labels masking everything except the final assistant turn."""

        if turns[-1].role != "assistant":
            raise TaskFormatError("supervised encoding needs a final assistant turn")
        if self.config.raw_qwen_format:
            full = self.tokenizer(
                turns_to_raw_qwen(turns, add_generation_prompt=False),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            prefix = self.tokenizer(
                turns_to_raw_qwen(turns[:-1], add_generation_prompt=True),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
        else:
            full = self.tokenizer.apply_chat_template(
                turns_to_chat(turns), add_generation_prompt=False, return_tensors="pt"
            )
            prefix = self.tokenizer.apply_chat_template(
                turns_to_chat(turns[:-1]), add_generation_prompt=True, return_tensors="pt"
            )
        if full.shape[1] > self.config.max_sequence_tokens:
            return None
        labels = full.clone()
        boundary = min(prefix.shape[1], full.shape[1])
        labels[0, :boundary] = -100
        if not supervise_final:
            labels[:, :] = -100
        return full.to(self.device), labels.to(self.device)


# === arcttt/solve.py ===
"""End-to-end solver: tasks -> per-test attempts -> submission.json."""



import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path



@dataclass(frozen=True)
class SolveConfig:
    augmentations: tuple[Augmentation, ...] = DIHEDRAL_SWEEP
    samples_per_augmentation: int = 2
    rescore_augmentations: tuple[Augmentation, ...] = DIHEDRAL_SWEEP
    attempts: int = 2
    # TTT can train on a larger augmentation set (e.g. dihedral x color
    # permutations) than the frames predictions are generated in.
    ttt_augmentations: tuple[Augmentation, ...] | None = None


@dataclass
class SolveResult:
    attempts: dict[str, list[dict[str, list[list[int]]]]] = field(default_factory=dict)
    solved: int = 0
    scored: int = 0

    @property
    def accuracy(self) -> float:
        return self.solved / self.scored if self.scored else 0.0


def solve_task(task: Task, predictor: Predictor, config: SolveConfig) -> list[list[Grid]]:
    """Return ranked attempts for every test input of one task."""

    predictor.adapt(task, config.ttt_augmentations or config.augmentations)
    per_test: list[list[Grid]] = []
    for test_index in range(len(task.test)):
        predictions: list[tuple[Augmentation, Grid]] = []
        frame_predictor = getattr(predictor, "predict_frames", None)
        if callable(frame_predictor):
            # One lockstep-batched pass over all augmentation frames.
            transformed_tasks = [
                augmentation.apply_task(task) for augmentation in config.augmentations
            ]
            frame_grids = frame_predictor(
                transformed_tasks, test_index, config.samples_per_augmentation
            )
            for augmentation, grids in zip(config.augmentations, frame_grids):
                for grid in grids:
                    predictions.append((augmentation, grid))
        else:
            for augmentation in config.augmentations:
                transformed = augmentation.apply_task(task)
                for grid in predictor.predict(
                    transformed, test_index, config.samples_per_augmentation
                ):
                    predictions.append((augmentation, grid))
        counts = pool_predictions(predictions)
        if not counts:
            per_test.append([])
            continue

        pair_scorer = getattr(predictor, "log_probabilities_pairs", None)
        batch_scorer = getattr(predictor, "log_probabilities", None)
        if callable(pair_scorer):
            # All (augmentation, candidate) pairs scored in chunked batch
            # forwards — no per-frame call overhead.
            grids = list(counts)
            pairs = []
            for augmentation in config.rescore_augmentations:
                transformed_task = augmentation.apply_task(task)
                for grid in grids:
                    pairs.append(
                        (transformed_task, test_index, augmentation.apply(grid))
                    )
            flat_scores = pair_scorer(pairs)
            totals = {grid: 0.0 for grid in grids}
            for pair_index, score in enumerate(flat_scores):
                totals[grids[pair_index % len(grids)]] += score

            candidates = tuple(
                Candidate(
                    grid=grid,
                    found_count=counts[grid],
                    mean_log_probability=totals[grid] / len(config.rescore_augmentations),
                )
                for grid in grids
            )
        elif callable(batch_scorer):
            # One padded batch forward per augmentation scores every candidate.
            grids = list(counts)
            totals = {grid: 0.0 for grid in grids}
            for augmentation in config.rescore_augmentations:
                transformed_task = augmentation.apply_task(task)
                rendered = [augmentation.apply(grid) for grid in grids]
                for grid, score in zip(
                    grids, batch_scorer(transformed_task, test_index, rendered)
                ):
                    totals[grid] += score

            candidates = tuple(
                Candidate(
                    grid=grid,
                    found_count=counts[grid],
                    mean_log_probability=totals[grid] / len(config.rescore_augmentations),
                )
                for grid in grids
            )
        else:

            def log_probability(grid: Grid, augmentation: Augmentation) -> float:
                transformed_task = augmentation.apply_task(task)
                return predictor.log_probability(
                    transformed_task, test_index, augmentation.apply(grid)
                )

            candidates = rescore_candidates(
                counts, config.rescore_augmentations, log_probability
            )
        per_test.append(list(select_attempts(candidates, config.attempts)))
    return per_test


def solve_tasks(
    tasks: Sequence[Task], predictor: Predictor, config: SolveConfig
) -> SolveResult:
    result = SolveResult()
    fallback: Grid = ((0,),)
    for task in tasks:
        ranked = solve_task(task, predictor, config)
        entries = []
        for test_index, attempts in enumerate(ranked):
            first = attempts[0] if attempts else fallback
            second = attempts[1] if len(attempts) > 1 else first
            entries.append(
                {"attempt_1": grid_to_lists(first), "attempt_2": grid_to_lists(second)}
            )
            solution = task.test[test_index].output
            if solution is not None:
                result.scored += 1
                if score_attempts([first, second], solution):
                    result.solved += 1
        result.attempts[task.task_id] = entries
    return result


def write_submission(result: SolveResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(result.attempts), encoding="utf-8")
    return path


# === entry: entry_v9.py ===
"""Kaggle kernel entry v9: v8 pipeline with the DFS recall bound widened.

v8 scored 1.67 (150 real predictions, ~2.7% per-attempt) — solver quality,
not coverage, is the binding constraint (kaggle_v8_scored_2026-08-10.json).
The t4-champ-diag artifact splits the misses: in 4 of 9 scored tests the
TRUE solution outscores the returned top candidate under teacher forcing
(e.g. lp_true -0.099 vs lp_top -0.352) yet is absent from the candidate
set — a pure search-recall failure, not a ranking failure. Cause: the DFS
prunes on TOTAL completion NLL at -ln(cutoff); v8's cutoff 0.1 bounds the
tree at 2.30 nats, while the diag's recall-miss truths total 2.5-3.8 nats.
v9 changes exactly two knobs, both aimed at that class:
  dfs_probability_cutoff 0.1 -> 0.02 (total-NLL bound 2.30 -> 3.91 nats,
    admitting every recall-miss truth observed in the diagnostic), and
  dfs_max_candidates 32 -> 64 (room for the wider tree's completions in
    the vote/rescore).
Everything else is v8 verbatim: 4 search frames + 8 rescore frames, rank
256 rslora, OOM ladder, pacing guards, 60 s DFS budget (the wider tree
fills time the old bound left idle; the cap and deadline math are
unchanged). Preregistered read: if the recall class converts, per-attempt
hit rate rises well above v8's 2.7%; if it does not move, the bound was
not binding and the next lever is TTT sharpening (epochs), not search.
"""



import glob
import json as _json
import multiprocessing as _mp
import os as _os
import time as _time

import torch as _torch


WALL_BUDGET_SECONDS = 11.0 * 3600
PER_WORKER_MARGIN = 20 * 60  # stop starting tasks this close to the deadline

# Four maximally spread D4 elements: both parities, all rotation offsets.
AUG_QUARTET: tuple[Augmentation, ...] = (
    Augmentation(rotations=0, flip=False),
    Augmentation(rotations=1, flip=False),
    Augmentation(rotations=2, flip=True),
    Augmentation(rotations=3, flip=True),
)


def _make_configs() -> tuple[TTTConfig, SolveConfig]:
    # rank/alpha: champion pairing r=256/alpha=32 rslora; the T4 sweep
    # measured all ranks at near-identical wall-clock and memory, so
    # champion parity costs nothing. OOM ladder covers the tail — and at
    # 4 lockstep frames the peak that drove v7's 52 level-0 OOMs halves.
    ttt = TTTConfig(
        lora_rank=256,
        lora_alpha=32,
        epochs=1,
        max_new_tokens=992,
        max_sequence_tokens=2560,
        raw_qwen_format=True,
        gradient_checkpointing=True,
        use_dfs=True,
        dfs_probability_cutoff=0.02,
        dfs_max_candidates=64,
        shuffle_examples=True,
        dfs_time_budget_seconds=60.0,
        ttt_batch_size=1,
    )
    solve = SolveConfig(
        augmentations=AUG_QUARTET,
        samples_per_augmentation=1,
        rescore_augmentations=DIHEDRAL_SWEEP,
    )
    return ttt, solve


def _find_model_dir() -> str:
    for config_path in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        directory = _os.path.dirname(config_path)
        if glob.glob(_os.path.join(directory, "*.safetensors")):
            return directory
    raise RuntimeError("no model directory with config.json + safetensors found")


def _load_challenges(path: str) -> list[Task]:
    raw = _json.loads(open(path).read())
    tasks = []
    for task_id, payload in sorted(raw.items()):
        train = tuple(
            Pair(input=to_grid(p["input"]), output=to_grid(p["output"]))
            for p in payload["train"]
        )
        test = tuple(Pair(input=to_grid(p["input"]), output=None) for p in payload["test"])
        tasks.append(Task(task_id=task_id, train=train, test=test))
    return tasks


def _task_cells(task: Task) -> int:
    total = 0
    for pair in list(task.train) + list(task.test):
        total += len(pair.input) * len(pair.input[0])
        if pair.output is not None:
            total += len(pair.output) * len(pair.output[0])
    return total


def _write_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    open(tmp, "w").write(_json.dumps(payload))
    _os.replace(tmp, path)


def _bf16_compute_works(device: "_torch.device") -> bool:
    """Functional probe: is_bf16_supported() lies in both directions across
    torch versions (False on T4 where emulated bf16 works; potentially True
    where kernels are missing). Running an actual matmul is ground truth."""

    try:
        a = _torch.randn(8, 8, dtype=_torch.bfloat16, device=device)
        result = float((a @ a).float().sum().item())
        return result == result  # reached => kernels exist (NaN-safe tautology)
    except Exception:
        return False


def _worker(
    rank: int,
    model_dir: str,
    tasks: list[Task],
    deadline: float,
    big_task_cells: int,
) -> None:
    import dataclasses
    import traceback

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _torch.device(f"cuda:{rank}")
    # The kernel's own run produces the scored file (no separate rerun), so
    # the pipeline must actually execute HERE. machine_shape pins a T4; this
    # probe is the belt-and-suspenders: degrade to fp16 inference-only
    # (TTT off — fp16 training without a loss scaler is unsafe) rather than
    # dying, and say so loudly.
    bf16 = _bf16_compute_works(device)
    dtype = _torch.bfloat16 if bf16 else _torch.float16
    if not bf16:
        print(
            f"[w{rank}] WARNING: bf16 compute unavailable on this GPU — "
            "fp16 inference-only mode (TTT disabled). If this appears on a "
            "submitted run, the accelerator pin failed.",
            flush=True,
        )
    print(
        f"[w{rank}] {_torch.cuda.get_device_name(rank)} | bf16={bf16} "
        f"| dtype={dtype}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)
    ttt, solve = _make_configs()
    if not bf16:  # fp16 training without a loss scaler is unsafe; skip TTT
        ttt = dataclasses.replace(ttt, epochs=0)
    first_error = True
    first_alloc_error = True
    part_path = f"submission_part_{rank}.json"
    part: dict[str, list[dict[str, list[list[int]]]]] = {}
    for index, task in enumerate(tasks):
        if _time.time() > deadline - PER_WORKER_MARGIN:
            print(f"[w{rank}] deadline margin reached at task {index}", flush=True)
            break
        started = _time.monotonic()
        try:
            # OOM degrade-and-retry ladder: a memory failure costs a retry
            # at a lighter config, not the whole task. Level 1 halves the
            # sequence cap (drops the biggest training examples); level 2
            # additionally skips TTT entirely (inference-only).
            ladder = (
                (ttt, solve),
                (dataclasses.replace(ttt, max_sequence_tokens=1280), solve),
                (
                    dataclasses.replace(
                        ttt, max_sequence_tokens=1280, epochs=0
                    ),
                    solve,
                ),
            )
            # Tail cap + pacing: v7 measured 793-1104 s on the largest
            # tasks against a 311 s/task budget. Tasks in the top size
            # quartile start at level 1 (their level-0 attempt was the
            # usual OOM-then-retry anyway); when the remaining wall-clock
            # per remaining task falls below the thresholds, every task
            # degrades a level so the tail gets attempted at all.
            start_level = 1 if _task_cells(task) >= big_task_cells else 0
            remaining = deadline - PER_WORKER_MARGIN - _time.time()
            per_task = remaining / max(1, len(tasks) - index)
            if per_task < 150:
                start_level = 2
            elif per_task < 240:
                start_level = max(start_level, 1)
            if index == 0 and start_level > 0 and _task_cells(task) < big_task_cells:
                # the guard binding on the FIRST task means the whole run is
                # time-starved (e.g. single-GPU fallback) — say so once, loudly
                print(
                    f"[w{rank}] pacing guard active from task 0: "
                    f"{per_task:.0f}s/task budget -> start level {start_level}",
                    flush=True,
                )
            ranked = None
            for level, (ttt_try, solve_try) in enumerate(ladder):
                if level < start_level:
                    continue
                try:
                    predictor = CausalLMPredictor(model, tokenizer, ttt_try, device)
                    ranked = solve_task(task, predictor, solve_try)
                    break
                except _torch.OutOfMemoryError:
                    _torch.cuda.empty_cache()
                    print(
                        f"[w{rank}] {task.task_id}: OOM at level {level}",
                        flush=True,
                    )
                except RuntimeError as error:
                    # cuBLAS/cuDNN allocation failures surface as plain
                    # RuntimeError, not OutOfMemoryError — same remedy.
                    # "CUBLAS" also matches non-alloc CUBLAS faults, so the
                    # first full message is logged per worker: a postmortem
                    # must be able to tell alloc pressure from a real bug.
                    message = str(error)
                    if (
                        "out of memory" not in message
                        and "CUBLAS" not in message
                        and "CUDNN" not in message
                    ):
                        raise
                    _torch.cuda.empty_cache()
                    if first_alloc_error:
                        first_alloc_error = False
                        print(f"[w{rank}] first alloc message: {message[:300]}", flush=True)
                    print(
                        f"[w{rank}] {task.task_id}: alloc RuntimeError at "
                        f"level {level}",
                        flush=True,
                    )
            if ranked is None:
                continue  # every ladder level OOMed; fallback rows stand
            entries = []
            for attempts in ranked:
                if attempts:
                    first = attempts[0]
                    second = attempts[1] if len(attempts) > 1 else first
                    entries.append(
                        {
                            "attempt_1": grid_to_lists(first),
                            "attempt_2": grid_to_lists(second),
                        }
                    )
                else:
                    entries.append(None)  # keep the parent's fallback for this test
            part[task.task_id] = entries
        except Exception as error:
            print(f"[w{rank}] {task.task_id}: {type(error).__name__}", flush=True)
            if first_error:  # one full traceback per worker, not 60
                first_error = False
                traceback.print_exc()
        _write_atomic(part_path, part)
        if (index + 1) % 5 == 0:
            print(
                f"[w{rank}] {index + 1}/{len(tasks)} "
                f"({_time.monotonic() - started:.0f}s last)",
                flush=True,
            )


def main() -> None:
    started = _time.time()
    deadline = started + WALL_BUDGET_SECONDS
    model_dir = _find_model_dir()
    print("model dir:", model_dir, flush=True)
    # _rebuild_like reconstructs KV caches as full-attention layers; that is
    # only exact when the model uses no sliding-window layers. Qwen3 dense
    # defaults to none — surface the actual config so the assumption is a
    # printed fact, not a guess (see review finding on cache rebuild).
    # Exception-proof: this is a script kernel, so an uncaught exception in
    # main() means "Notebook Threw Exception" and no score even after the
    # fallback file exists — a diagnostic print must never be able to do that.
    try:
        model_config = _json.loads(open(_os.path.join(model_dir, "config.json")).read())
        sliding = model_config.get("use_sliding_window", False)
        layer_types = set(model_config.get("layer_types") or [])
        print(
            f"config: use_sliding_window={sliding} layer_types={sorted(layer_types)}",
            flush=True,
        )
        if sliding or (layer_types - {"full_attention"}):
            print(
                "WARNING: sliding-window attention configured — cache rebuild "
                "in the batched search assumes full-attention layers; results "
                "from compacted searches may be degraded on this model.",
                flush=True,
            )
    except Exception as error:
        print(f"config probe failed (non-fatal): {type(error).__name__}", flush=True)
    matches = glob.glob("/kaggle/input/**/*test_challenges.json", recursive=True)
    if not matches:
        matches = glob.glob("/kaggle/input/**/*evaluation_challenges.json", recursive=True)
    if not matches:
        listing = "\n".join(sorted(glob.glob("/kaggle/input/**", recursive=True))[:80])
        raise RuntimeError("no challenges file under /kaggle/input; tree:\n" + listing)
    print("challenges:", matches[0], flush=True)
    tasks = _load_challenges(matches[0])
    tasks.sort(key=_task_cells)  # smallest first: budget -> most real predictions

    fallback: Grid = ((0,),)
    submission: dict[str, list[dict[str, list[list[int]]]]] = {
        task.task_id: [
            {"attempt_1": grid_to_lists(fallback), "attempt_2": grid_to_lists(fallback)}
            for _ in task.test
        ]
        for task in tasks
    }
    _write_atomic("submission.json", submission)

    sizes = sorted(_task_cells(task) for task in tasks)
    big_task_cells = sizes[(3 * len(sizes)) // 4]  # top-quartile threshold

    workers = max(1, _torch.cuda.device_count())
    print(
        f"{len(tasks)} tasks | {workers} gpu workers | "
        f"big-task threshold {big_task_cells} cells",
        flush=True,
    )
    if workers == 1:
        _worker(0, model_dir, tasks, deadline, big_task_cells)
    else:
        context = _mp.get_context("spawn")
        procs = [
            context.Process(
                target=_worker,
                args=(rank, model_dir, tasks[rank::workers], deadline, big_task_cells),
            )
            for rank in range(workers)
        ]
        for proc in procs:
            proc.start()
        # A worker hung mid-task must not block past the platform's 12h kill,
        # or the merged submission never gets written at all.
        for proc in procs:
            proc.join(timeout=max(60.0, deadline + 10 * 60 - _time.time()))
        for proc in procs:
            if proc.is_alive():
                print(f"terminating straggler worker {proc.pid}", flush=True)
                proc.terminate()
                proc.join(timeout=60)

    for part_path in glob.glob("submission_part_*.json"):
        try:
            part = _json.loads(open(part_path).read())
        except Exception as error:
            print(f"unreadable {part_path}: {type(error).__name__}", flush=True)
            continue
        for task_id, entries in part.items():
            for test_index, entry in enumerate(entries):
                if entry is not None and test_index < len(submission[task_id]):
                    submission[task_id][test_index] = entry
    _write_atomic("submission.json", submission)
    real = sum(
        1
        for entries in submission.values()
        for entry in entries
        if entry["attempt_1"] != grid_to_lists(fallback)
    )
    print(
        f"submission.json complete | {real} real predictions | "
        f"{(_time.time() - started) / 60:.0f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
