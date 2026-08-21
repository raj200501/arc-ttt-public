"""Scoring for text-mode extraction — spec section 3.2, torch-free.

Split out of ``text_ttt`` deliberately: the verification path (re-scoring
stored predictions against regenerated gold) must run on a clean machine
with nothing installed. ``text_ttt`` imports torch at module scope for the
adaptation engine, so importing the scorer from there dragged a ~250MB
dependency into a check that is pure arithmetic over JSON — and made
``scripts/verify_from_primary.py`` die with ModuleNotFoundError on any box
without PyTorch, while the README promised it needed none.

``text_ttt`` re-exports every name here, so existing callers are unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from arcttt.text_task import TextTaskFormatError

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
