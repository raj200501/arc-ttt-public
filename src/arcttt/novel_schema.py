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

from __future__ import annotations

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
) -> NovelSchema:
    """Build one tenant schema with nesting and distractors.

    Fields are distributed across ``n_groups`` invented top-level objects
    so the target is nested rather than flat — flat key-value extraction is
    close to copying, and would let both arms score well without holding a
    schema in mind.
    """

    if n_fields < n_groups:
        raise ValueError("n_fields must be at least n_groups")
    rng = random.Random(seed)
    # One pool, so a label can never coincide with a key or a distractor.
    pool = _unique_pseudowords(rng, n_fields * 2 + n_groups + n_distractors + 1)
    tenant_id = pool.pop()
    group_names = [pool.pop() for _ in range(n_groups)]
    fields: list[FieldSpec] = []
    for index in range(n_fields):
        doc_label = pool.pop()
        json_key = pool.pop()  # deliberately unrelated to doc_label
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
):
    """A ``TextTask`` over one invented tenant schema.

    Train and test records share the schema and are drawn from disjoint
    record seeds, so the test set is unseen documents of a seen schema —
    exactly the deployment shape the product describes.
    """

    from arcttt.text_task import TextPair, TextTask

    schema = make_schema(
        seed, n_fields=n_fields, n_groups=n_groups, n_distractors=n_distractors
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
