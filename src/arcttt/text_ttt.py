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

from __future__ import annotations

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

from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.serialize import ChatTurn
from arcttt.text_task import TextTask, TextTaskFormatError

# -- corpus construction -----------------------------------------------------


def text_task_to_messages(task: TextTask, test_index: int = 0) -> tuple[ChatTurn, ...]:
    """Serialize demonstrations plus one test input as chat turns.

    The model is expected to complete the final assistant turn with the
    output text (for CORD: the canonical ``gt_parse`` JSON).
    """

    if not 0 <= test_index < len(task.test):
        raise TextTaskFormatError(f"{task.task_id}: test index {test_index} out of range")
    turns: list[ChatTurn] = []
    for pair in task.train:
        if pair.output_text is None:
            raise TextTaskFormatError(f"{task.task_id}: train pair missing output")
        turns.append(ChatTurn("user", pair.input_text))
        turns.append(ChatTurn("assistant", pair.output_text))
    turns.append(ChatTurn("user", task.test[test_index].input_text))
    return tuple(turns)


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

    def predict_text(self, task: TextTask, test_index: int, samples: int) -> list[str]:
        """Raw completion texts (greedy first, then samples); empty ones dropped.

        Parsing/validation is the scorer's job (``score_text_output``), so the
        voting layer can still count near-miss candidates by canonical form.
        """

        prompt = self._prompt_ids(text_task_to_messages(task, test_index))
        if prompt is None:
            return []
        return [text.strip() for text in self._sample_texts(prompt, samples) if text.strip()]

    def log_probabilities_text(
        self, task: TextTask, test_index: int, outputs: Sequence[str]
    ) -> list[float]:
        """Mean supervised-token log-probability per candidate output text."""

        base = text_task_to_messages(task, test_index)
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
    predictor: TextPredictor, task: TextTask, test_index: int, samples: int = 5
) -> str | None:
    """The Addendum-A decode: greedy + sampled pool -> vote/rescore -> top-1.

    ``samples`` counts the WHOLE pool (1 greedy + samples-1 sampled), matching
    ``predict_text``'s contract. Candidate lp is the model's mean
    supervised-token log-probability of each distinct completion
    (``log_probabilities_text``), computed once per distinct text.
    """

    texts = predictor.predict_text(task, test_index, samples)
    if not texts:
        return None
    distinct = list(dict.fromkeys(texts))  # lp once per distinct text
    # One candidate per forward: a batched call materializes logits of shape
    # [n_candidates, prompt+completion, vocab] — ~14 GB float32 at 5×4.6k
    # tokens on a 151k vocab, which OOM-kills CPU containers. Chunking is
    # math-identical (per-sequence mean supervised-token lp).
    lps = [
        predictor.log_probabilities_text(task, test_index, [text])[0]
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
