# Addendum U-ext — three more shipped helpers on the same corpus (preregistration)

**Frozen 2026-09-04, after Addendum U was banked and corrected, before
any of these three functions ran on the corpus.** U's row says *six
parsers, not twenty-one* and names the census packages it did not
attempt. Three of them turned out to be installable on this box, and
each ships a JSON helper that a team reaching for "just handle the
fence" would call. Same corpus (`experiments/fence_corpus_2026-09-04.jsonl`,
same SHA, rebuilt only if the Phi-3 cells have landed — the manifest
says which), same reference labeler, same quantities, same U2
thresholds. Nothing in Addendum U is re-read; this is a second panel.

## The panel — shipped helpers, called as shipped

Every helper is wrapped as `text -> object | None` with the same two
rules U froze: an exception is `None`; a non-object return is `None`
(and, as in U, the raw non-object returns are banked beside the
readings so the rule hides nothing).

| key | what is called | shipped behaviour, read from source before the run |
|---|---|---|
| `instructor_extract_json_from_codeblock` | `json.loads(instructor.utils.extract_json_from_codeblock(text))` | returns the LAST balanced `{…}`/`[…]` span in the text (documented as a prompt-injection defence); the string is then parsed strictly. No repair. |
| `smolagents_parse_json_blob` | `smolagents.utils.parse_json_blob(text)[0]` | slices from the first `{` to the last `}` and parses with `strict=False`; raises `ValueError` on no braces or invalid JSON. No repair. |
| `llama_index_parse_json_markdown` | `llama_index.core.output_parsers.utils.parse_json_markdown(text)` | strips a ```` ```json ```` opener if present, marshals, tries `json.loads`, then FALLS BACK TO `yaml.safe_load` — which turns `{"a": 2 * 3}` into `{"a": "2 * 3"}`: the expression becomes a string value. Raises `OutputParserException` when both fail; returns `None` on empty input. |

Versions pinned in the artifact from `importlib.metadata` (instructor
1.16.0, smolagents 1.26.0, llama-index-core 0.14.24 at freeze time).

## Quantities and readings

Per parser: `lost`, `diverged`, `fabricated`, `hazard`, per slice and
overall, exactly as U defines them; the substance decomposition of
`fabricated` (the corrected four-category version); the raw
non-object returns.

**Reading, per parser — U2 verbatim:** MATERIAL if `hazard ≥ 0.05` on
any slice with n ≥ 30; HARMLESS ON THIS CORPUS if overall
`hazard < 0.01` and no slice ≥ 0.05; else PRESENT, stated at size.

**One prediction, written before the run so it can be wrong in
public:** the YAML fallback in llama-index is expected to produce
objects on malformed schema-only outputs that hold expressions
(`fabricated`, with string-valued fields), and the last-span rule in
instructor is expected to lose nothing the reference parses (`lost` =
0) because a reference record is exactly one object. If either fails,
the failure is published beside the prediction.

No U1 or U3 here: both are readings on `strict`, which does not change.

## What this cannot show

Same limits as U: two corpora, ≤ 4B checkpoints, CPU, no hosted APIs.
Three helpers from three packages, the ones this box could install;
the fourteen remaining census packages are still named as not
attempted. A helper's `fabricated` object may be right or wrong —
correctness is not adjudicated here either.

Artifact: `experiments/parser_robustness_ext_2026-09-04.json`. Runner:
`PYTHONPATH=src python3 scripts/parser_robustness.py --panel ext`.

## Erratum — 2026-09-04, after the run

The substance check's `exact_object_present` category is satisfied
trivially by a helper that returns a SUBSTRING, so after this panel ran
it was split by brace depth: `exact_object_present` now means the
returned object sits at the top level of the text (trailing or leading
prose around a complete object — the reference's undercount, recovered),
and `nested_fragment_returned` means the returned object is a sub-object
inside a larger object the model wrote that does not parse — a fragment
handed back as the answer. Added after the data, disclosed here; the
frozen U2 readings do not depend on it. Base-panel counts under the
split are unchanged (json_repair 5, LangChain 3, all top-level).
