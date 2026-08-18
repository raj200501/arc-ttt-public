"""Invertible grid augmentations for test-time training.

The dihedral group D4 (rotations + reflections) and color permutations are the
standard TTT augmentation family for ARC: apply a transform to every grid of a
task, adapt/predict in the transformed frame, then invert the transform on the
prediction before voting. Every augmentation here therefore carries an exact
inverse, and round-trip identity is unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from arcttt.tasks import COLORS, Grid, Pair, Task


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
