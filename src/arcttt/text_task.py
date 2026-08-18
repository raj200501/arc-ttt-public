"""Text task data model for the enterprise text-mode TTT path.

The text counterpart of ``tasks.py`` (ENTERPRISE_EVAL_SPEC.md section 2.2):
a task is a fixed transform demonstrated by (input_text, output_text) pairs —
e.g. post-OCR receipt text in, canonical ``gt_parse`` JSON out. Pairs are
frozen dataclasses over plain strings; validation is fail-closed so malformed
files raise ``TextTaskFormatError`` instead of silently degrading a corpus.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from arcttt.tasks import TaskFormatError


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

    from arcttt.text_ttt import json_canonical

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
