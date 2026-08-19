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
                state.stop_reason = (
                    "candidate_cap" if len(state.completed) >= max_candidates else "exhausted"
                )
                state.done = True
                return None
            if deadline is not None and time.time() > deadline:
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




def _template_ids(rendered: Any) -> torch.Tensor:
    """Normalize ``apply_chat_template(..., return_tensors="pt")`` output.

    transformers 4.x returns the id tensor directly; 5.x returns a
    BatchEncoding whose bare attribute access raises an empty
    ``AttributeError`` (the cord-scale kernel incident, 2026-08-11 — same
    pinned-image API-drift class as the v7 cache incident). Probe the shape,
    not the version string.
    """

    ids = rendered.input_ids if hasattr(rendered, "input_ids") else rendered
    return cast(torch.Tensor, ids)


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
    chunked_loss_tokens: int = 0  # 0 = HF labels-path loss (legacy, exact).
    # N > 0 = compute the SAME shifted fp32 cross-entropy in N-token slices
    # of the sequence with a two-phase backward, so the seq x vocab logits
    # tensor never materializes whole. Mathematically identical (the loss is
    # per-token additive; mean = sum / count either way); exists because a
    # 7.5k-token training sequence's full logits over a 152k vocab OOM a T4
    # (observed: Addendum B k=30 adapted arms, 2026-08-15). Guarded by
    # tests/test_chunked_loss.py, which pins loss AND gradients against the
    # labels path.


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

    # Search telemetry, CLASS-level on purpose: the kernel builds a fresh
    # predictor per task (and per OOM-ladder level), so instance counters
    # would reset ~240 times and measure nothing. These accumulate across the
    # whole worker process and are dumped once at the end of the run.
    dfs_stop_reasons: dict[str, int] = {}
    dfs_candidates_found: int = 0
    dfs_frames_searched: int = 0

    @classmethod
    def dfs_telemetry(cls) -> dict[str, Any]:
        """Snapshot of why searches stopped and how much they found."""

        frames = cls.dfs_frames_searched
        return {
            "frames_searched": frames,
            "stop_reasons": dict(cls.dfs_stop_reasons),
            "candidates_found_total": cls.dfs_candidates_found,
            "candidates_per_frame_mean": (
                round(cls.dfs_candidates_found / frames, 2) if frames else 0.0
            ),
        }

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
        # v8 forensics closed the case on HF's gradient_checkpointing_enable
        # in this image: flags read True everywhere, yet ~430 MB/layer of
        # activations stay resident (11.3 GB at 5.4k tokens) - a no-op at
        # layer level. When the chunked path (and therefore long-sequence
        # training) is in play, checkpoint each decoder layer OURSELVES with
        # torch.utils.checkpoint, preserving HF's forward orchestration.
        # Wrappers are restored in the finally below so eval and generation
        # see the original forwards.
        wrapped_layers: list[tuple[object, object]] = []
        if self.config.chunked_loss_tokens > 0 and self.config.gradient_checkpointing:
            layers = getattr(getattr(self.model, "model", None), "layers", None)
            if layers is not None:
                from torch.utils.checkpoint import checkpoint as _torch_checkpoint

                def _wrap(fwd):
                    def wrapped(*args, **kwargs):
                        return _torch_checkpoint(
                            fwd, *args, use_reentrant=False, **kwargs
                        )

                    return wrapped

                for layer in layers:
                    wrapped_layers.append((layer, layer.forward))
                    layer.forward = _wrap(layer.forward)
                print(
                    f"manually checkpointed {len(wrapped_layers)} decoder layers",
                    flush=True,
                )
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
                if self.config.chunked_loss_tokens > 0:
                    self._chunked_loss_backward(input_ids, attention_mask, labels)
                else:
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    ).loss
                    loss.backward()
                optimizer.step()
        for layer, original_forward in wrapped_layers:
            layer.forward = original_forward
        if self.config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_disable"
        ):
            self.model.gradient_checkpointing_disable()  # cached generation next
        self.model.eval()

    def _chunked_loss_backward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """The HF causal-LM loss and its gradients, without whole-seq logits.

        Reproduces exactly what ``forward(labels=...)`` computes - shift by
        one, cross-entropy in fp32, mean over non-ignored tokens - but applies
        the lm_head to ``chunked_loss_tokens``-sized slices of the hidden
        states, backpropagating each slice's SUM loss scaled by the global
        token count. Per-token CE is additive, so slice sums divided by the
        one global count equal the whole-sequence mean, and the accumulated
        gradients are equal too (autograd sums across backward calls).

        Two-phase backward: slices push gradients into a detached copy of the
        hidden states (freeing each slice's logits before the next), then one
        trunk backward carries the accumulated hidden-state gradient through
        the LoRA parameters. The lm_head is frozen under LoRA, so skipping
        its weight gradient changes nothing.
        """

        trunk = getattr(self.model, "model", None)
        lm_head = getattr(self.model, "lm_head", None)
        if trunk is None or lm_head is None:  # architecture without the split
            loss = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            ).loss
            loss.backward()
            return
        # gradient_checkpointing_enable() is called on the wrapper; whether
        # the flag reaches the trunk that we call DIRECTLY is version-
        # dependent (11 GB of held activations at 5.4k tokens says it did
        # not, 2026-08-15). Assert-and-repair, loudly, once.
        if not getattr(self, "_ckpt_reported", False):
            self._ckpt_reported = True
            engaged = bool(getattr(trunk, "gradient_checkpointing", False))
            print(f"trunk gradient_checkpointing={engaged}", flush=True)
            if self.config.gradient_checkpointing and not engaged:
                trunk.gradient_checkpointing = True
                print("trunk gradient_checkpointing FORCED on", flush=True)
        # torch-level activation offload: the scoring image's transformers
        # ignores BOTH its own gradient_checkpointing_enable and a per-layer
        # torch.utils.checkpoint monkeypatch (verified: saved-activation
        # bytes match the no-checkpoint profile exactly, while the identical
        # mechanisms verifiably work in transformers 5.15 locally). save_on_cpu
        # is beneath the framework - every tensor autograd saves for backward
        # lives in host RAM instead of GPU memory, whatever HF does or does
        # not do. Gradient-identical (pinned in the local probe); costs one
        # PCIe round-trip per training example, noise next to generation.
        if input_ids.is_cuda:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                hidden = trunk(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
        else:
            hidden = trunk(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        detached = hidden.detach().requires_grad_(True)
        shifted_labels = labels[:, 1:]
        total = int((shifted_labels != -100).sum().item())
        if total == 0:
            return
        width = hidden.shape[1]
        step = self.config.chunked_loss_tokens
        for start in range(0, width - 1, step):
            end = min(width - 1, start + step)
            logits = lm_head(detached[:, start:end])
            slice_loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                shifted_labels[:, start:end].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            (slice_loss / total).backward()
        assert detached.grad is not None
        hidden.backward(detached.grad)

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
        frame_stats: list[tuple[str, int]] = []
        results = constrained_dfs_multi(
            self.model,
            [prompt for _, prompt in live],
            self._grid_vocab,
            self.tokenizer,
            max_score=-math.log(self.config.dfs_probability_cutoff),
            max_new_tokens=self.config.max_new_tokens,
            max_candidates=self.config.dfs_max_candidates,
            deadline=time.time() + budget if budget is not None else None,
            stats_out=frame_stats,
        )
        # Running tally of WHY searches stop, across the whole run. A run
        # dominated by "deadline" is time-limited (buy search time); one
        # dominated by "exhausted" is bound-limited (widen the NLL bound);
        # "candidate_cap" means raise max_candidates. Without this the only
        # signal is the next day's leaderboard score - one bit per day.
        cls = type(self)  # class-level tally: `self.x += 1` would shadow it
        for reason, found in frame_stats:
            cls.dfs_stop_reasons[reason] = cls.dfs_stop_reasons.get(reason, 0) + 1
            cls.dfs_candidates_found += found
            cls.dfs_frames_searched += 1
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
            ids = _template_ids(
                self.tokenizer.apply_chat_template(
                    turns_to_chat(turns),
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
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
            full = _template_ids(
                self.tokenizer.apply_chat_template(
                    turns_to_chat(turns), add_generation_prompt=False, return_tensors="pt"
                )
            )
            prefix = _template_ids(
                self.tokenizer.apply_chat_template(
                    turns_to_chat(turns[:-1]),
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
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


# === arcttt/text_task.py ===
"""Text task data model for the enterprise text-mode TTT path.

The text counterpart of ``tasks.py`` (ENTERPRISE_EVAL_SPEC.md section 2.2):
a task is a fixed transform demonstrated by (input_text, output_text) pairs —
e.g. post-OCR receipt text in, canonical ``gt_parse`` JSON out. Pairs are
frozen dataclasses over plain strings; validation is fail-closed so malformed
files raise ``TextTaskFormatError`` instead of silently degrading a corpus.
"""



import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path



class TextTaskFormatError(TaskFormatError):
    """Raised when a text task file violates the text-task schema."""


def to_text(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TextTaskFormatError(f"{context}: must be a string")
    if not value.strip():
        raise TextTaskFormatError(f"{context}: must be a non-empty string")
    return value


@dataclass(frozen=True)
class TextPair:
    input_text: str
    output_text: str | None  # None for hidden test outputs


@dataclass(frozen=True)
class TextTask:
    task_id: str
    train: tuple[TextPair, ...]
    test: tuple[TextPair, ...]

    def validate(self) -> None:
        if not self.task_id or not self.task_id.strip():
            raise TextTaskFormatError("text task needs a non-empty task_id")
        if not self.train or not self.test:
            raise TextTaskFormatError(f"{self.task_id}: needs train and test pairs")
        for pair in self.train:
            if pair.output_text is None:
                raise TextTaskFormatError(f"{self.task_id}: train pairs need outputs")
        for split, pairs in (("train", self.train), ("test", self.test)):
            for index, pair in enumerate(pairs):
                to_text(pair.input_text, f"{self.task_id}: {split}[{index}].input_text")
                if pair.output_text is not None:
                    to_text(
                        pair.output_text, f"{self.task_id}: {split}[{index}].output_text"
                    )


def _pairs(items: object, context: str, split: str) -> tuple[TextPair, ...]:
    if not isinstance(items, list):
        raise TextTaskFormatError(f"{context}: {split} must be a list")
    built = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or "input" not in item:
            raise TextTaskFormatError(f"{context}: {split}[{index}] needs an input")
        output = item.get("output")
        built.append(
            TextPair(
                input_text=to_text(item["input"], f"{context}: {split}[{index}].input"),
                output_text=(
                    to_text(output, f"{context}: {split}[{index}].output")
                    if output is not None
                    else None
                ),
            )
        )
    return tuple(built)


def _task_from_mapping(raw: object, context: str, task_id: str | None = None) -> TextTask:
    if not isinstance(raw, dict) or "train" not in raw or "test" not in raw:
        raise TextTaskFormatError(f"{context}: missing train/test keys")
    if task_id is None:
        task_id_value = raw.get("task_id")
        if task_id_value is None:
            raise TextTaskFormatError(f"{context}: missing task_id")
        task_id = to_text(task_id_value, f"{context}: task_id")
    task = TextTask(
        task_id=task_id,
        train=_pairs(raw["train"], context, "train"),
        test=_pairs(raw["test"], context, "test"),
    )
    task.validate()
    return task


def load_text_task(path: str | Path) -> TextTask:
    """Load one task from a JSON file: ``{"train": [...], "test": [...]}``.

    Pairs are ``{"input": str, "output": str}`` (test outputs optional — the
    same shape as ARC task files, with strings where grids were). ``task_id``
    defaults to the file stem, matching ``tasks.load_task``.
    """

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TextTaskFormatError(f"{path.name}: invalid JSON ({error})") from None
    if isinstance(raw, dict) and "task_id" in raw:
        return _task_from_mapping(raw, path.name)
    return _task_from_mapping(raw, path.name, task_id=path.stem)


def load_text_tasks_jsonl(path: str | Path) -> dict[str, TextTask]:
    """Load many tasks from JSONL: one ``{"task_id", "train", "test"}`` per line."""

    path = Path(path)
    tasks: dict[str, TextTask] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        context = f"{path.name}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise TextTaskFormatError(f"{context}: invalid JSON ({error})") from None
        task = _task_from_mapping(raw, context)
        if task.task_id in tasks:
            raise TextTaskFormatError(f"{context}: duplicate task_id {task.task_id!r}")
        tasks[task.task_id] = task
    if not tasks:
        raise TextTaskFormatError(f"{path.name}: no tasks in file")
    return tasks


def from_cord_gt(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    task_id: str = "cord-v2",
) -> TextTask:
    """Adapter from CORD rows to a ``TextTask``. STUB — dataset unit implements.

    Mapping to be implemented (ENTERPRISE_EVAL_SPEC.md sections 1.2 and 2.2,
    row "CORD data prep"), over rows of the HF dataset
    ``naver-clova-ix/cord-v2`` at a pinned revision. Each row's
    ``ground_truth`` field is a JSON string carrying ``gt_parse`` (the target
    parse) and ``valid_line`` (per-line OCR groups with ``words[*].text`` and
    quads); the image is ignored — this is the text-only post-OCR variant.

    - ``input_text``: OCR text reconstructed from annotations, no OCR engine:
      for each ``valid_line`` entry in the dataset's given order, join its
      ``words[*].text`` with single spaces; join lines with ``"\\n"``. No
      layout/coordinate information is encoded in v1.
    - ``output_text``: the canonical JSON serialization
      (``text_ttt.json_canonical``: sorted keys, compact separators) of
      ``gt_parse`` restricted to the released superclasses — ``menu``,
      ``void_menu``, ``sub_total``, ``void_total``, ``total`` (the spec's 30
      semantic classes in 5 superclasses; on-disk key spellings follow the
      dataset, e.g. ``sub_total``). Leaf values stay verbatim strings —
      numeric normalization happens only at scoring time
      (``text_ttt.normalize_value``), never in stored targets. Repeated
      groups (multi-item ``menu``) stay lists in ``gt_parse`` order.
    - ``train_rows`` become train pairs (outputs required), ``test_rows``
      become test pairs (outputs kept — CORD eval outputs are not hidden).
    - Out of scope for this adapter (dataset unit responsibilities): k-shot
      sampling with seeds, dataset revision pinning, and SHA-256 hashes of
      the rendered texts for the experiment artifact.

    """


    superclasses = ("menu", "void_menu", "sub_total", "void_total", "total")

    def render(row: Mapping[str, object]) -> tuple[str, str]:
        lines = []
        valid_line = row.get("valid_line")
        if not isinstance(valid_line, list) or not valid_line:
            raise TextTaskFormatError("CORD row missing valid_line OCR groups")
        for group in valid_line:
            words = group.get("words") if isinstance(group, Mapping) else None
            if not isinstance(words, list):
                raise TextTaskFormatError("CORD valid_line entry missing words")
            line = " ".join(
                str(word.get("text", ""))
                for word in words
                if isinstance(word, Mapping)
            ).strip()
            if line:
                lines.append(line)
        gt_parse = row.get("gt_parse")
        if not isinstance(gt_parse, Mapping):
            raise TextTaskFormatError("CORD row missing gt_parse")
        target = {
            key: gt_parse[key] for key in superclasses if key in gt_parse
        }
        if not target:
            raise TextTaskFormatError("CORD gt_parse has no released superclass")
        return "\n".join(lines), json_canonical(target)

    train_pairs = []
    for row in train_rows:
        input_text, output_text = render(row)
        train_pairs.append(TextPair(input_text=input_text, output_text=output_text))
    test_pairs = []
    for row in test_rows:
        input_text, output_text = render(row)
        test_pairs.append(TextPair(input_text=input_text, output_text=output_text))
    task = TextTask(
        task_id=task_id, train=tuple(train_pairs), test=tuple(test_pairs)
    )
    task.validate()
    return task


# === arcttt/text_ttt.py ===
"""Text-mode TTT: leave-one-out corpora, generation, and JSON-field scoring.

The text counterpart of ``serialize.ttt_training_examples`` plus the metric
helpers of ENTERPRISE_EVAL_SPEC.md section 3.2. The corpus builder emits
``ChatTurn`` sequences that feed ``CausalLMPredictor``'s existing chat
encoding path unchanged; ``TextPredictor`` below is a thin subclass adding
text-typed entry points only — training, generation, and rescoring all run
through the shared cores (``adapt_on_examples``, ``_sample_texts``,
``score_turn_sequences``).

Deliberately NOT imported here: ``decode`` (constrained DFS assumes the
16-token grid vocabulary) and ``augment`` (dihedral/palette transforms are
meaningless on text). The text-mode analog of augmentation is the
demonstration-order shuffle, driven by explicit ``shuffle_seeds``.
"""



import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import torch


# -- corpus construction -----------------------------------------------------


def text_task_to_messages(
    task: TextTask, test_index: int = 0, include_demos: bool = True
) -> tuple[ChatTurn, ...]:
    """Serialize demonstrations plus one test input as chat turns.

    The model is expected to complete the final assistant turn with the
    output text (for CORD: the canonical ``gt_parse`` JSON).
    ``include_demos=False`` is the document-only serving configuration
    (Addendum D): the prompt carries ONLY the test document; any task
    knowledge must live in the adapter weights.
    """

    if not 0 <= test_index < len(task.test):
        raise TextTaskFormatError(f"{task.task_id}: test index {test_index} out of range")
    turns: list[ChatTurn] = []
    if include_demos:
        for pair in task.train:
            if pair.output_text is None:
                raise TextTaskFormatError(f"{task.task_id}: train pair missing output")
            turns.append(ChatTurn("user", pair.input_text))
            turns.append(ChatTurn("assistant", pair.output_text))
    turns.append(ChatTurn("user", task.test[test_index].input_text))
    return tuple(turns)


def text_docmode_training_examples(
    task: TextTask,
) -> tuple[tuple[ChatTurn, ...], ...]:
    """Document-only training sequences (Addendum F).

    One example per train pair: (user: document) -> (assistant: target
    JSON), with NO leave-one-out context. Trains the adapter on exactly
    the serving configuration the payload economics describe — the
    corrective to the D.6 finding that LOO-context adapters produce
    prose, not JSON, on bare documents.
    """

    examples = []
    for pair in task.train:
        if pair.output_text is None:
            raise TextTaskFormatError(f"{task.task_id}: train pair missing output")
        examples.append((
            ChatTurn("user", pair.input_text),
            ChatTurn("assistant", pair.output_text),
        ))
    return tuple(examples)


def text_ttt_training_examples(
    task: TextTask, shuffle_seed: int | None = None
) -> tuple[tuple[ChatTurn, ...], ...]:
    """Leave-one-out demonstration examples for per-task adaptation.

    Mirrors ``serialize.ttt_training_examples`` exactly: for each
    demonstration pair the remaining pairs act as context and the held-out
    pair supplies the supervised completion; a shuffle seed permutes the
    context order deterministically (one RNG across all held-out picks, as in
    the grid path).
    """

    order_rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    examples = []
    for held_out in range(len(task.train)):
        context = [
            index
            for index, pair in enumerate(task.train)
            if index != held_out and pair.output_text is not None
        ]
        if order_rng is not None:
            order_rng.shuffle(context)
        turns: list[ChatTurn] = []
        for index in context:
            pair = task.train[index]
            if pair.output_text is None:  # excluded above; narrows the type
                continue
            turns.append(ChatTurn("user", pair.input_text))
            turns.append(ChatTurn("assistant", pair.output_text))
        target = task.train[held_out]
        if target.output_text is None:
            continue
        turns.append(ChatTurn("user", target.input_text))
        turns.append(ChatTurn("assistant", target.output_text))
        examples.append(tuple(turns))
    if not examples:
        raise TextTaskFormatError(f"{task.task_id}: no usable TTT examples")
    return tuple(examples)


# -- prediction --------------------------------------------------------------


class TextPredictor(CausalLMPredictor):
    """``CausalLMPredictor`` reused for text tasks (greedy + sampled only).

    No new model machinery: text-typed wrappers over the inherited cores.
    Grid-vocabulary DFS decoding is refused up front — a JSON-grammar decoder
    is a possible v2 (spec section 2.2), not a config flag away.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: TTTConfig,
        device: torch.device,
    ) -> None:
        if config.use_dfs:
            raise ValueError(
                "TextPredictor does not support DFS decoding; it assumes the "
                "16-token grid vocabulary (set use_dfs=False)"
            )
        super().__init__(model, tokenizer, config, device)

    def adapt_text(
        self, task: TextTask, shuffle_seeds: Sequence[int | None] = (None,)
    ) -> None:
        """Fresh LoRA adapter trained on the LOO corpus, once per shuffle seed."""

        examples: list[tuple[ChatTurn, ...]] = []
        for seed in shuffle_seeds:
            examples.extend(text_ttt_training_examples(task, seed))
        self.adapt_on_examples(examples)

    def predict_text(
        self, task: TextTask, test_index: int, samples: int,
        include_demos: bool = True,
    ) -> list[str]:
        """Raw completion texts (greedy first, then samples); empty ones dropped.

        Parsing/validation is the scorer's job (``score_text_output``), so the
        voting layer can still count near-miss candidates by canonical form.
        """

        prompt = self._prompt_ids(text_task_to_messages(task, test_index, include_demos))
        if prompt is None:
            return []
        return [text.strip() for text in self._sample_texts(prompt, samples) if text.strip()]

    def log_probabilities_text(
        self, task: TextTask, test_index: int, outputs: Sequence[str],
        include_demos: bool = True,
    ) -> list[float]:
        """Mean supervised-token log-probability per candidate output text."""

        base = text_task_to_messages(task, test_index, include_demos)
        return self.score_turn_sequences(
            [base + (ChatTurn("assistant", output),) for output in outputs]
        )


# -- vote/rescore (spec section 2.2; Addendum A scaled run) -------------------


@dataclass(frozen=True)
class TextCandidate:
    """A pooled completion: representative text + count + mean candidate lp."""

    text: str  # highest-lp member of the pool (emitted verbatim if selected)
    key: str  # canonical-JSON key, or the raw stripped text if unparseable
    found_count: int
    mean_log_probability: float


def _candidate_key(text: str) -> str:
    """Canonical-JSON pooling key; unparseable texts pool by raw form.

    Mirrors ``vote.pool_predictions``'s invert-then-count: two completions
    that differ only in key order/whitespace are the same answer. A candidate
    that does not parse still participates (near-miss counting per
    ``predict_text``'s contract) but only ever matches itself.
    """

    try:
        return json_canonical(parse_json_object(text))
    except TextTaskFormatError:
        return text


def vote_text_candidates(
    texts: Sequence[str], log_probabilities: Sequence[float]
) -> tuple[TextCandidate, ...]:
    """Pool completions by canonical key; count + mean-lp per pool.

    Pure so the selection arithmetic is unit-testable without a model. The
    representative text of a pool is its highest-lp member (pool members are
    canonically equal but may differ in formatting; the emitted text should
    be the one the model believed most).
    """

    if len(texts) != len(log_probabilities):
        raise ValueError("texts and log_probabilities must align")
    pools: dict[str, list[tuple[str, float]]] = {}
    for text, lp in zip(texts, log_probabilities):
        pools.setdefault(_candidate_key(text), []).append((text, lp))
    candidates = []
    for key, members in pools.items():
        best_text = max(members, key=lambda member: member[1])[0]
        mean_lp = sum(lp for _, lp in members) / len(members)
        candidates.append(
            TextCandidate(
                text=best_text,
                key=key,
                found_count=len(members),
                mean_log_probability=mean_lp,
            )
        )
    return tuple(candidates)


def select_text_attempts(
    candidates: Iterable[TextCandidate], attempts: int = 1
) -> tuple[str, ...]:
    """Rank by (found_count + exp(mean lp)) — ``vote.select_attempts``'s shape.

    exp(mean lp) lies in (0, 1], so it breaks ties between equal counts
    without outweighing one extra find, exactly as in the grid path.
    """

    ranked = sorted(
        candidates,
        key=lambda c: c.found_count + math.exp(c.mean_log_probability),
        reverse=True,
    )
    return tuple(candidate.text for candidate in ranked[:attempts])


def predict_text_voted(
    predictor: TextPredictor, task: TextTask, test_index: int, samples: int = 5,
    include_demos: bool = True,
) -> str | None:
    """The Addendum-A decode: greedy + sampled pool -> vote/rescore -> top-1.

    ``samples`` counts the WHOLE pool (1 greedy + samples-1 sampled), matching
    ``predict_text``'s contract. Candidate lp is the model's mean
    supervised-token log-probability of each distinct completion
    (``log_probabilities_text``), computed once per distinct text.
    """

    texts = predictor.predict_text(task, test_index, samples, include_demos)
    if not texts:
        return None
    distinct = list(dict.fromkeys(texts))  # lp once per distinct text
    # One candidate per forward: a batched call materializes logits of shape
    # [n_candidates, prompt+completion, vocab] — ~14 GB float32 at 5×4.6k
    # tokens on a 151k vocab, which OOM-kills CPU containers. Chunking is
    # math-identical (per-sequence mean supervised-token lp).
    lps = [
        predictor.log_probabilities_text(task, test_index, [text], include_demos)[0]
        for text in distinct
    ]
    lp_by_text = dict(zip(distinct, lps))
    candidates = vote_text_candidates(texts, [lp_by_text[text] for text in texts])
    selected = select_text_attempts(candidates, attempts=1)
    return selected[0] if selected else None


# -- scoring (spec section 3.2) ----------------------------------------------

_NUMERIC = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def json_canonical(value: object) -> str:
    """Canonical JSON: sorted keys, compact separators, unicode kept."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_json_object(text: str) -> dict[str, object]:
    """Fail-closed parse: the entire text must be exactly one JSON object."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise TextTaskFormatError(f"output is not valid JSON: {error}") from None
    if not isinstance(value, dict):
        raise TextTaskFormatError("output JSON must be an object")
    return {str(key): item for key, item in value.items()}


def _canonical_number(number: Decimal) -> str:
    normalized = number.normalize()
    if normalized == normalized.to_integral_value():
        return str(normalized.quantize(Decimal(1)))  # 1.2E+4 -> 12000
    return format(normalized, "f")


def normalize_value(value: object) -> str:
    """Leaf normalization per spec section 3.2.

    Numeric normalization for prices/counts ("12,000", "12000", 12000.0 all
    compare equal); whitespace-collapse + casefold for names. Booleans and
    null map to their JSON spellings.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return _canonical_number(Decimal(str(value)))
    if isinstance(value, str):
        folded = " ".join(value.split()).casefold()
        if _NUMERIC.fullmatch(folded):
            try:
                return _canonical_number(Decimal(folded.replace(",", "")))
            except InvalidOperation:  # pragma: no cover - regex precludes this
                return folded
        return folded
    raise TextTaskFormatError(f"unsupported JSON leaf type: {type(value).__name__}")


def field_pairs(value: Mapping[str, object]) -> Counter[tuple[str, str]]:
    """Multiset of (field-path, normalized value) leaves of a JSON object.

    Paths are dot-joined key chains; list indices are dropped so repeated
    groups (CORD's multi-item ``menu``) compare as unordered multisets rather
    than by position.
    """

    pairs: Counter[tuple[str, str]] = Counter()

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)
        else:
            pairs[(path, normalize_value(node))] += 1

    walk(dict(value), "")
    return pairs


def field_micro_f1(predicted: Mapping[str, object], gold: Mapping[str, object]) -> float:
    """Micro-F1 over (field-path, normalized value) pairs (primary metric)."""

    predicted_pairs = field_pairs(predicted)
    gold_pairs = field_pairs(gold)
    if not predicted_pairs and not gold_pairs:
        return 1.0
    overlap = sum((predicted_pairs & gold_pairs).values())
    denominator = sum(predicted_pairs.values()) + sum(gold_pairs.values())
    return 2.0 * overlap / denominator if denominator else 0.0


@dataclass(frozen=True)
class TextScore:
    valid_json: bool
    exact_match: bool  # canonicalized-JSON equality (secondary metric)
    micro_f1: float  # field-level micro-F1 (primary metric)


def score_text_output(predicted_text: str, gold_text: str) -> TextScore:
    """Score one model completion against one gold output, fail-closed.

    The gold text must parse as a JSON object (a malformed reference is a
    harness bug and raises). A malformed prediction scores zero and is
    flagged, feeding the invalid-JSON rate (gate G-E1).
    """

    gold = parse_json_object(gold_text)
    try:
        predicted = parse_json_object(predicted_text)
    except TextTaskFormatError:
        return TextScore(valid_json=False, exact_match=False, micro_f1=0.0)
    return TextScore(
        valid_json=True,
        exact_match=json_canonical(predicted) == json_canonical(gold),
        micro_f1=field_micro_f1(predicted, gold),
    )


# === arcttt/novel_schema.py ===
"""Synthetic novel-schema extraction tasks: the fair test of the wedge.

WHY THIS EXISTS
---------------
G-E2 measured per-request adaptation against a k-shot baseline on CORD and
found no benefit at 0.5B or 4B. That is a real negative, but it was
collected in the regime least favourable to the hypothesis the product
rests on. CORD is a public dataset that is almost certainly in the base
model's pretraining, so the prompted arm already knows what a receipt is,
what fields it has, and what they are called. Adaptation cannot add
knowledge the model already has; the measurement mostly says "prompting
already knows this", which is close to a tautology.

The regime the product actually claims is the opposite one: a schema the
model has NEVER seen, which no amount of general pretraining supplies, and
enough examples that a weight update can encode it. This module builds that
regime synthetically, so the claim can be tested rather than asserted.

WHAT MAKES A SCHEMA "NOVEL" HERE
--------------------------------
Three properties, each chosen because it defeats a way the prompted arm
could win without learning anything:

1. Pseudoword labels and keys. Document labels and JSON keys are generated
   consonant-vowel nonsense ("vokrin", "zelbat"). A model cannot fall back
   on knowing that "TOTAL" means the total, because no such word appears.
2. Arbitrary label -> key mapping. The document says ``vokrin:`` and the
   target JSON calls it ``zelbat``. The mapping is fixed per tenant and
   carries no surface similarity, so it must be LEARNED, not guessed. This
   is the single most important property: it is the part that in-context
   examples convey poorly and weight updates should convey well.
3. Distractor lines. Some document lines use labels outside the schema and
   must be omitted from the output entirely. Knowing what to IGNORE is
   schema knowledge, and it is where a prompted arm with few examples
   tends to over-extract.

FAIRNESS INVARIANTS (deliberate, and load-bearing)
--------------------------------------------------
- Every target value appears VERBATIM in the document, so a perfect
  extractor scores exactly 1.0. If the task were ambiguous both arms would
  saturate low and the comparison would measure nothing.
- The schema is fixed across every example of a tenant. A task where each
  example had a different schema would be unlearnable by either arm and
  would produce a null for the wrong reason.
- Documents shuffle field order per record, so position cannot substitute
  for the label -> key mapping.
- Generation is fully seeded and deterministic: same seed, same corpus.

This module deliberately produces the SAME (input_text, output_text) shape
as ``from_cord_gt``, so the existing scorer, adapter and paired-arm harness
apply unchanged. Nothing here touches the frozen G-E2 preregistration; this
is a separate gate (ENTERPRISE_EVAL_SPEC Addendum B).
"""



import json
import random
from dataclasses import dataclass

_CONSONANTS = "bdfgklmnprstvz"
_VOWELS = "aeiou"


def _json_canonical(value: object) -> str:
    """Canonical JSON, byte-identical to ``text_ttt.json_canonical``.

    Deliberately duplicated rather than imported: ``text_ttt`` imports torch
    at module scope, and a corpus generator that cannot run without a GPU
    stack is a generator that cannot be unit-tested, inspected on a laptop,
    or used to sanity-check a schema before committing compute to it. The
    duplication is four keyword arguments; the coupling it removes is an
    entire deep-learning dependency.

    ``test_novel_schema.py`` pins this against the definition in
    ``text_ttt.py`` by reading its SOURCE (not importing it), so if
    canonicalization ever changes there, the guard fails here and forces a
    matching update rather than silently emitting targets in a different
    convention from every other task in the project.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _pseudoword(rng: random.Random, syllables: int = 3) -> str:
    """A consonant-vowel nonsense word.

    Three syllables gives ~14^3 * 5^3 distinct forms, which is enough that
    collisions inside one schema are rare and are rejected by the caller
    anyway. Restricted to letters so the token stream stays ordinary; the
    point is novelty of MEANING, not exotic bytes that would confound the
    comparison with tokenizer effects.
    """

    return "".join(
        rng.choice(_CONSONANTS) + rng.choice(_VOWELS) for _ in range(syllables)
    )


def _unique_pseudowords(rng: random.Random, count: int) -> list[str]:
    seen: list[str] = []
    used: set[str] = set()
    while len(seen) < count:
        word = _pseudoword(rng)
        if word not in used:
            used.add(word)
            seen.append(word)
    return seen


@dataclass(frozen=True)
class FieldSpec:
    """One extractable field: what the document calls it, where it lands."""

    doc_label: str
    json_path: tuple[str, ...]
    value_kind: str


@dataclass(frozen=True)
class NovelSchema:
    """A tenant's fixed, invented extraction schema."""

    tenant_id: str
    fields: tuple[FieldSpec, ...]
    distractor_labels: tuple[str, ...]

    def describe(self) -> str:
        """Human-readable dump, for artifacts — never shown to the model.

        Kept out of the prompt on purpose: handing the model the mapping
        would test instruction-following, not learning, and would make both
        arms trivially correct.
        """

        rows = [
            f"  {f.doc_label} -> {'.'.join(f.json_path)} ({f.value_kind})"
            for f in self.fields
        ]
        return f"tenant {self.tenant_id}\n" + "\n".join(rows)


def make_schema(
    seed: int,
    n_fields: int = 8,
    n_groups: int = 2,
    n_distractors: int = 4,
    geometry: str = "fixed",
) -> NovelSchema:
    """Build one tenant schema with nesting and distractors.

    ``geometry="fixed"`` (default) keeps the historical deterministic
    shape — every tenant has the same group/kind layout, differing only
    in vocabulary. This is the geometry the Addendum B gate ran on and
    it MUST stay byte-identical for artifact reproducibility.
    ``geometry="diverse"`` (Addendum E) derives the shape itself from
    the seed: group count, field count, per-field value kinds and group
    assignments all vary per tenant, answering the B.9.5 limitation
    that fixed-mode tenants are vocabulary re-rolls of one shape.

    Fields are distributed across ``n_groups`` invented top-level objects
    so the target is nested rather than flat — flat key-value extraction is
    close to copying, and would let both arms score well without holding a
    schema in mind.
    """

    rng = random.Random(seed)
    if geometry == "diverse":
        # Shape drawn from the seed BEFORE the vocabulary pool so fixed
        # mode's pool consumption (and thus its schemas) is untouched.
        n_groups = rng.randint(2, 4)
        n_fields = rng.randint(max(6, n_groups), 12)
        n_distractors = rng.randint(3, 7)
    elif geometry == "diverse-compact":
        # E-r1: shape-varying but bounded so k=30 LOO sequences fit the
        # frozen 8192-token budget (E.4 made "diverse" unmeasurable at
        # its largest draws).
        n_groups = rng.randint(2, 3)
        n_fields = rng.randint(max(6, n_groups), 9)
        n_distractors = rng.randint(3, 5)
    elif geometry != "fixed":
        raise ValueError(f"unknown geometry: {geometry!r}")
    if n_fields < n_groups:
        raise ValueError("n_fields must be at least n_groups")
    # One pool, so a label can never coincide with a key or a distractor.
    pool = _unique_pseudowords(rng, n_fields * 2 + n_groups + n_distractors + 1)
    tenant_id = pool.pop()
    group_names = [pool.pop() for _ in range(n_groups)]
    if geometry in ("diverse", "diverse-compact"):
        # every group non-empty, remainder assigned at random
        assignment = list(range(n_groups)) + [
            rng.randrange(n_groups) for _ in range(n_fields - n_groups)
        ]
        rng.shuffle(assignment)
        kinds = [rng.choice(("amount", "code", "date", "name"))
                 for _ in range(n_fields)]
    fields: list[FieldSpec] = []
    for index in range(n_fields):
        doc_label = pool.pop()
        json_key = pool.pop()  # deliberately unrelated to doc_label
        if geometry in ("diverse", "diverse-compact"):
            group = group_names[assignment[index]]
            kind = kinds[index]
        else:
            group = group_names[index % n_groups]
            kind = ("amount", "code", "date", "name")[index % 4]
        fields.append(
            FieldSpec(
                doc_label=doc_label,
                json_path=(group, json_key),
                value_kind=kind,
            )
        )
    distractors = tuple(pool.pop() for _ in range(n_distractors))
    return NovelSchema(
        tenant_id=tenant_id,
        fields=tuple(fields),
        distractor_labels=distractors,
    )


def _value(rng: random.Random, kind: str) -> str:
    if kind == "amount":
        return f"{rng.randint(1, 9999)}.{rng.randint(0, 99):02d}"
    if kind == "code":
        letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))
        return f"{letters}-{rng.randint(1000, 9999)}"
    if kind == "date":
        return f"{rng.randint(2019, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return _pseudoword(rng, 2).capitalize() + " " + _pseudoword(rng, 2).capitalize()


def make_record(schema: NovelSchema, seed: int) -> tuple[str, dict]:
    """One (document text, target object) pair for a schema.

    Field order is shuffled per record and distractor lines are interleaved,
    so neither arm can succeed by memorising line positions.
    """

    rng = random.Random(seed)
    lines: list[tuple[str, str]] = []
    target: dict[str, dict[str, str]] = {}
    for field in schema.fields:
        value = _value(rng, field.value_kind)
        lines.append((field.doc_label, value))
        group, key = field.json_path
        target.setdefault(group, {})[key] = value
    for label in schema.distractor_labels:
        # Distractors carry realistic-looking values so they cannot be
        # filtered on surface form alone - only on schema membership.
        lines.append((label, _value(rng, ("amount", "code", "date")[rng.randint(0, 2)])))
    rng.shuffle(lines)
    text = "\n".join(f"{label}: {value}" for label, value in lines)
    return text, target


def make_task(
    seed: int,
    n_train: int,
    n_test: int,
    n_fields: int = 8,
    n_groups: int = 2,
    n_distractors: int = 4,
    task_id: str | None = None,

    geometry: str = "fixed",
):
    """A ``TextTask`` over one invented tenant schema.

    Train and test records share the schema and are drawn from disjoint
    record seeds, so the test set is unseen documents of a seen schema —
    exactly the deployment shape the product describes.
    """


    schema = make_schema(
        seed, n_fields=n_fields, n_groups=n_groups,
        n_distractors=n_distractors, geometry=geometry
    )
    total = n_train + n_test
    # Offset record seeds by the schema seed so two tenants never share
    # documents, and keep train/test slices disjoint by construction.
    record_seeds = [seed * 100_000 + i for i in range(total)]
    pairs = []
    for record_seed in record_seeds:
        text, target = make_record(schema, record_seed)
        pairs.append(TextPair(input_text=text, output_text=_json_canonical(target)))
    task = TextTask(
        task_id=task_id or f"novel-{schema.tenant_id}",
        train=tuple(pairs[:n_train]),
        test=tuple(pairs[n_train:]),
    )
    task.validate()
    return task, schema


# === entry: entry_novel_schema_er_s204k.py ===
"""Kaggle kernel entry: Addendum B novel-schema gate, 0.5B CPU.

The one experiment that can still rescue the quality thesis, run exactly as
frozen in ENTERPRISE_EVAL_SPEC.md Addendum B (2026-08-12T19:40Z, before any
record existed):

- Corpus: synthetic novel-schema tenants (novel_schema.make_task) — the ONLY
  variable changed from Addendum A. Model, LoRA config, 1 epoch, decode,
  scorer, pairing and seed discipline are Addendum A's frozen values.
- Decision point: k=30. k=10 is a comparability point next to Addendum A's
  k=10 numbers and may NOT be promoted if it alone comes out positive.
- eval_n=60 per seed (180 paired records over seeds {1,2,3}), the power fix:
  MDE ~4 F1 at CORD-observed spread, below the +5 bar for the first time.
- Validity gates BEFORE the delta is read: k-shot mean < 0.15 -> FLOOR
  (task too hard for the rung; delta uninformative); > 0.95 -> CEILING
  (no headroom). Both are stamped into every artifact so the summary can
  refuse to interpret an invalid rung without re-deriving anything.

No dataset file is needed for the corpus — generation is seeded and
deterministic in-kernel (schema seed -> tenant, record seeds disjoint
between train and eval by construction). The attached dataset is used only
to seed RESUME artifacts from prior sessions, same skip-if-exists protocol
as the CORD builds; novel-schema documents are ~12 short lines, so arms are
cheaper per record than CORD arms despite eval_n=60.

Each seed is a DIFFERENT tenant (schema seed = arm seed): the gate then
averages over three novel schemas rather than three draws of one schema,
so a pass cannot be a quirk of one lucky vocabulary.
"""



import os as _os_early

# Before torch import: lets the CUDA caching allocator grow segments instead
# of hunting for one contiguous block - the v5/v6 OOMs died asking for
# 3.3 GB contiguous while 2.3 GB free + 0.8 GB reserved-unallocated existed.
_os_early.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob as _glob
import json as _json
import os as _os
import random as _random
import time as _time

import torch as _torch


RUNG = "0.5b"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EVAL_N = 60  # Addendum B power fix; CORD's 20 could not resolve the bar
POOL_SAMPLES = 5  # frozen: 1 greedy + 4 sampled (T=0.7)
DATE = "2026-08-19"  # Addendum E freeze date
FLOOR = 0.15
CEILING = 0.95
_ARM_IDENTITY = {"rung", "k", "seed", "arm"}
WALL_BUDGET_SECONDS = 11.0 * 3600
MARGIN_SECONDS = 25 * 60

# Decision arms (k=30) first, comparability arms (k=10) after — a cancelled
# session should die holding the gate, not the garnish.
# Arm-scoped shard (B.7-r5): ONE k=30 arm per kernel so every arm
# completes well inside Kaggle's 12h CPU cap and saves at natural
# session end (v11 was cancelled ~9h in with zero k=30 arms saved;
# a pair may not fit one session). Races the pair shards under the
# gate is decidable at the slowest shard, not the end of a sequential
# chain. Duplicate-arm policy (decided before banking): first terminal
# kernel's artifact banks; the duplicate is preserved as a free
# same-environment reproducibility datum.
ARM_ORDER = [(30, 204, "kshot")]


def _write_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    open(tmp, "w").write(_json.dumps(payload, indent=2))
    _os.replace(tmp, path)


def _seed_resume_artifacts() -> int:
    """Copy prior KERNEL novel-schema artifacts from attached inputs.

    Same environment-homogeneity filter as the CORD builds: only artifacts
    carrying a "device" field (kernel-produced) and no "error" field seed a
    resume. There are no local novel-schema arms today, but the filter is
    what KEEPS that true tomorrow.
    """

    import shutil

    seeded = 0
    for path in _glob.glob("/kaggle/input/**/novel_schema_*.json", recursive=True):
        name = _os.path.basename(path)
        if _os.path.exists(name):
            continue
        try:
            record = _json.loads(open(path).read())
        except Exception:
            continue
        if not _ARM_IDENTITY.issubset(record):
            continue
        if "device" not in record or "error" in record:
            print(f"not seeding (will re-run here): {name}", flush=True)
            continue
        shutil.copy(path, name)
        seeded += 1
    return seeded


def _seed_resume_checkpoints() -> int:
    """Copy B.7-r6 checkpoint files (novel_ckpt_*) from attached inputs.

    Named outside every novel_schema_* glob on purpose: the banker, the
    artifact seeding filter and the r4 purge must never see them.
    """

    import shutil

    seeded = 0
    for path in _glob.glob("/kaggle/input/**/novel_ckpt_*", recursive=True):
        name = _os.path.basename(path)
        if not _os.path.exists(name):
            shutil.copy(path, name)
            seeded += 1
    return seeded


def main() -> None:
    started = _time.time()
    deadline = started + WALL_BUDGET_SECONDS
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Fail FAST on a wrong-architecture draw. Kaggle's batch pool holds
        # P100s (cc 6.0), and this image's torch ships no sm_60 kernel
        # images - every op raises cudaErrorNoKernelImageForDevice, the
        # per-arm handler swallows it, and the session ends COMPLETE with
        # zero arms (observed 2026-08-15, novel-schema v2; same class as
        # ARC incident v6). Two minutes of loud failure beats an hour of
        # silent nothing; the push side pins --accelerator NvidiaTeslaT4,
        # and this probe is the backstop if the pin is ever ignored.
        major, minor = _torch.cuda.get_device_capability()
        name = _torch.cuda.get_device_name()
        if major < 7:
            print(f"WRONG GPU: {name} (cc {major}.{minor}) has no kernel "
                  "images in this torch build; exiting for re-push on T4",
                  flush=True)
            raise SystemExit(1)
        print(f"gpu ok: {name} (cc {major}.{minor})", flush=True)
    seeded = _seed_resume_artifacts()
    ckpts = _seed_resume_checkpoints()
    if ckpts:
        print(f"seeded {ckpts} checkpoint files", flush=True)
    if device.type == "cpu":
        # B.7-r4: the k=30 pairs run on CPU/fp32 - the proven path. Purge
        # any seeded k=30 artifact from the GPU attempts so both sides of
        # every pair are produced here.
        purged = 0
        for name in list(_glob.glob("novel_schema_*_k30_*.json")):
            try:
                record = _json.loads(open(name).read())
            except Exception:
                continue
            if not (
                record.get("device") == "cpu"
                and record.get("dtype") == "torch.float32"
            ):
                _os.remove(name)
                purged += 1
        if purged:
            print(f"purged {purged} non-cpu/fp32 k30 artifacts", flush=True)
    if device.type == "cuda":
        # Purge seeded k=30 arms that were NOT produced under fp16: their
        # pairs must be dtype-homogeneous with the fp16 adapted arms this
        # run produces. k=10 pairs stay bf16/bf16 (both sides done) and are
        # untouched.
        purged = 0
        for name in list(_glob.glob("novel_schema_*_k30_*.json")):
            try:
                record = _json.loads(open(name).read())
            except Exception:
                continue
            if record.get("dtype") != "torch.float16":
                _os.remove(name)
                purged += 1
        if purged:
            print(f"purged {purged} non-fp16 k30 artifacts for pair homogeneity", flush=True)
    print(f"device {device} | {seeded} artifacts resumed", flush=True)
    print(f"rung {RUNG} -> {MODEL_ID}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _torch.bfloat16 if device.type == "cuda" else _torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if device.type == "cuda":
        # B.7-r3: fp16, not bf16. v5-v7 forensics: SDPA's memory-efficient
        # kernel is sm80+ for bf16, so on this sm75 T4 every path (eager,
        # sdpa-math) materializes the T^2 attention per layer and the k=30
        # trunk backward dies at the same 3.3 GB fp32 softmax buffer -
        # with chunked loss AND verified gradient checkpointing. fp16 IS
        # mem-efficient-eligible on sm75: attention memory goes linear in T.
        # Pair homogeneity is preserved by re-running the k=30 KSHOT arms in
        # fp16 too (the purge below); dtype is stamped into every artifact.
        dtype = _torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        print(f"attention: sdpa | dtype: {dtype}", flush=True)
    except (TypeError, ValueError) as error:
        print(f"sdpa unavailable ({error}); falling back to default attention", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)

    done = skipped = 0
    for k, seed, arm in ARM_ORDER:
        if _time.time() > deadline - MARGIN_SECONDS:
            print(f"deadline margin reached; stopping before k{k} seed{seed} {arm}", flush=True)
            break
        out_path = f"novel_schema_e_{RUNG}_k{k}_seed{seed}_{arm}_{DATE}.json"
        if _os.path.exists(out_path):
            skipped += 1
            print(f"skip (exists): {out_path}", flush=True)
            continue
        arm_started = _time.monotonic()
        if device.type == "cuda":
            _torch.cuda.empty_cache()
            _torch.cuda.reset_peak_memory_stats()
        # seed -> tenant AND draws: paired arms at (k, seed) share the exact
        # corpus; different seeds are different invented schemas.
        task, schema = make_task(
            seed=seed,
            n_train=k,
            n_test=EVAL_N,
            task_id=f"novel-e-{RUNG}-k{k}-seed{seed}",
            geometry="diverse-compact",
        )
        config = TTTConfig(
            lora_rank=16,
            lora_alpha=32,
            epochs=1 if arm == "adapted" else 0,
            max_new_tokens=512,
            max_sequence_tokens=8192 if k == 30 else 4096,
            # B.7-r2: identical math, sliced logits — the full seq x vocab
            # logits tensor OOMed the T4 on every k=30 adapted arm (all
            # three recorded as error artifacts, 2026-08-15). Gradient
            # equivalence to the labels path is pinned by
            # tests/test_chunked_loss.py before this could ship.
            chunked_loss_tokens=512,
            gradient_checkpointing=True,
            shuffle_examples=True,
        )
        try:
            predictor = TextPredictor(model, tokenizer, config, device)
            # B.7-r6: cancellation-proofing. The adapter is saved once after
            # training and every scored doc is journaled; a relaunch seeded
            # with these files resumes instead of restarting. Same frozen
            # computation - the adapter weights are restored bit-identically
            # and completed docs are not re-decoded.
            ckpt_stem = f"novel_ckpt_e_{RUNG}_k{k}_seed{seed}_{arm}"
            adapter_path = f"{ckpt_stem}_adapter.pt"
            docs_path = f"{ckpt_stem}_docs.jsonl"
            resumed_adapter = False
            if arm == "adapted" and _os.path.exists(adapter_path):
                saved = _torch.load(adapter_path, map_location="cpu")
                remove_lora(model)
                inject_lora(model, config.lora_rank, config.lora_alpha, use_rslora=True)
                lora_state = {n: p for n, p in model.named_parameters() if "lora_" in n}
                if set(saved) != set(lora_state):
                    raise SystemExit(f"adapter checkpoint mismatch: {adapter_path}")
                with _torch.no_grad():
                    for n, p in lora_state.items():
                        p.copy_(saved[n].to(p.device, p.dtype))
                model.train()  # parity with post-adapt state
                adapt_seconds = _json.loads(open(f"{ckpt_stem}_meta.json").read())["adapt_seconds"]
                resumed_adapter = True
                print(f"resumed adapter: {adapter_path}", flush=True)
            else:
                adapt_started = _time.monotonic()
                predictor.adapt_text(task, shuffle_seeds=(seed,))
                adapt_seconds = _time.monotonic() - adapt_started
                if arm == "adapted":
                    _torch.save(
                        {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
                        adapter_path,
                    )
                    _write_atomic(f"{ckpt_stem}_meta.json", {"adapt_seconds": adapt_seconds})

            results = []
            exact = 0
            f1_sum = 0.0
            scored = 0
            invalid = 0
            no_completion = 0
            done_indices = set()
            if _os.path.exists(docs_path):
                for line in open(docs_path):
                    line = line.strip()
                    if not line:
                        continue
                    row = _json.loads(line)
                    done_indices.add(row["index"])
                    if "error" in row:
                        no_completion += 1
                        results.append({"index": row["index"], "error": row["error"]})
                        continue
                    scored += 1
                    exact += int(row["exact_match"])
                    invalid += int(not row["valid_json"])
                    f1_sum += row["micro_f1_raw"]
                    results.append(
                        {
                            "index": row["index"],
                            "valid_json": row["valid_json"],
                            "exact_match": row["exact_match"],
                            "micro_f1": round(row["micro_f1_raw"], 4),
                            "prediction": row.get("prediction"),
                        }
                    )
                print(f"resumed {len(done_indices)} docs from journal", flush=True)
            docs_log = open(docs_path, "a")
            for index in range(len(task.test)):
                if index in done_indices:
                    continue
                gold = task.test[index].output_text
                assert gold is not None
                selected = predict_text_voted(predictor, task, index, samples=POOL_SAMPLES)
                if selected is None:
                    no_completion += 1
                    results.append({"index": index, "error": "no completion"})
                    docs_log.write(_json.dumps({"index": index, "error": "no completion"}) + "\n")
                    docs_log.flush()
                    _os.fsync(docs_log.fileno())
                    continue
                score = score_text_output(selected, gold)
                scored += 1
                exact += int(score.exact_match)
                invalid += int(not score.valid_json)
                f1_sum += score.micro_f1
                results.append(
                    {
                        "index": index,
                        "valid_json": score.valid_json,
                        "exact_match": score.exact_match,
                        "micro_f1": round(score.micro_f1, 4),
                        "prediction": selected,
                    }
                )
                docs_log.write(_json.dumps(
                    {
                        "index": index,
                        "valid_json": score.valid_json,
                        "exact_match": score.exact_match,
                        "micro_f1_raw": score.micro_f1,
                        "prediction": selected,
                    }
                ) + "\n")
                docs_log.flush()
                _os.fsync(docs_log.fileno())
            docs_log.close()
            results.sort(key=lambda row: row["index"])
            mean_f1 = round(f1_sum / scored, 4) if scored else 0.0
            validity = "ok"
            if arm == "kshot":  # Addendum B B.5: judged on the BASELINE arm
                if mean_f1 < FLOOR:
                    validity = "floor"
                elif mean_f1 > CEILING:
                    validity = "ceiling"
            report = {
                "spec": "ENTERPRISE_EVAL_SPEC.md Addendum B (frozen 2026-08-12T19:40Z)",
                "dataset": "synthetic novel-schema tenants (novel_schema.py), no external data",
                "tenant": schema.tenant_id,
                "schema": schema.describe(),  # artifact-only; never in any prompt
                "rung": RUNG,
                "model": MODEL_ID,
                "arm": arm,
                "k": k,
                "eval_n": EVAL_N,
                "resumed": bool(resumed_adapter or done_indices),
                "seed": seed,
                "gate_role": "DECISION" if k == 30 else "comparability-only (may not be promoted)",
                "validity": validity,
                "decode": "vote/rescore ON: 1 greedy + 4 sampled (T=0.7), "
                "canonical-JSON pooling, count+likelihood top-1",
                "config": {
                    "rank": 16,
                    "alpha": 32,
                    "epochs": config.epochs,
                    "max_new_tokens": 512,
                    "max_seq": config.max_sequence_tokens,
                },
                "device": str(device),
                "dtype": str(dtype),
                "adapt_seconds": round(adapt_seconds, 1),
                "exact_match": exact,
                "scored": scored,
                "invalid_json": invalid,
                "no_completion": no_completion,
                "mean_micro_f1": mean_f1,
                "results": results,
            }
            _write_atomic(out_path, report)
            done += 1
            print(
                _json.dumps(
                    {
                        "artifact": out_path,
                        "mean_micro_f1": mean_f1,
                        "validity": validity,
                        "invalid_json": invalid,
                        "wall_seconds": round(_time.monotonic() - arm_started, 1),
                    }
                ),
                flush=True,
            )
        except _torch.OutOfMemoryError:
            import traceback

            # The site matters: v4 proved the LOSS was not the (only) hog,
            # so a bare "OOM" line hides the next bug. Print where and how
            # much before recording.
            traceback.print_exc()
            if device.type == "cuda":
                print(
                    f"cuda mem: allocated {_torch.cuda.memory_allocated()/1e9:.2f} GB, "
                    f"peak {_torch.cuda.max_memory_allocated()/1e9:.2f} GB, "
                    f"reserved {_torch.cuda.memory_reserved()/1e9:.2f} GB",
                    flush=True,
                )
                _torch.cuda.empty_cache()
            print(f"OOM: k{k} seed{seed} {arm} — recorded, not imputed", flush=True)
            _write_atomic(
                out_path,
                {
                    "rung": RUNG,
                    "arm": arm,
                    "k": k,
                    "seed": seed,
                    "device": str(device),
                    "error": "oom",
                    "note": "arm OOMed at frozen config; no number imputed",
                },
            )
        except Exception as error:
            import traceback

            print(f"ERROR k{k} seed{seed} {arm}: {type(error).__name__}: {error}", flush=True)
            if done == 0 and skipped <= 3:
                traceback.print_exc()
    del model
    print(f"novel-schema {RUNG}: {done} arms run, {skipped} skipped | "
          f"{(_time.time()-started)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
