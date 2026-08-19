"""Candidate pooling, augmentation-inverse mapping, and selection scoring.

Clean-room implementation of the selection scheme described in the NVARC 2025
paper: candidates found by search are re-scored under a fixed set of
augmentations, and the final ranking combines how often a candidate was found
with the geometric mean of its probabilities across augmentations
(equivalently, the mean log-probability). Two attempts per test input are
submitted, so selection returns an ordered list.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from arcttt.augment import Augmentation
from arcttt.tasks import Grid


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
    """Rank lexicographically by (found_count, mean log-probability), descending.

    Order-equivalent to the paper's additive shape (count + exp(mean log p))
    under exact arithmetic — exp is monotonic and bounded by 1, so the count
    strictly dominates and the probability term only breaks ties among equal
    counts — but the tuple key avoids float64 absorption: exp(mean log p)
    underflows (or is absorbed by the integer count) for very negative mean
    log-probabilities, which would make genuinely different candidates
    compare exactly equal.
    """

    ranked = sorted(
        candidates,
        key=lambda c: (c.found_count, c.mean_log_probability),
        reverse=True,
    )
    return tuple(candidate.grid for candidate in ranked[:attempts])
