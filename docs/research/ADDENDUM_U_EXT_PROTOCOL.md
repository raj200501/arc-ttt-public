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

## Errata — 2026-09-04, after the run, two rounds of adversarial review

1. The substance check's `exact_object_present` category is satisfied
   trivially by a helper that returns a SUBSTRING. After this panel ran
   it was split — first by a brace-depth heuristic (published for under
   an hour: 81 fragments, 18 recovered), which a review showed misfiles
   a receipt's `total` block as top-level when the model's own braces
   close early. The rule now looks at the text OUTSIDE the span:
   `exact_object_present` = no brace outside it (prose, fence or code
   sample around one complete object — recovered);
   `one_of_several_objects` = after peeling every other complete object,
   nothing JSON-like remains; `nested_fragment_returned` = JSON keys
   remain outside the complete objects, so the span is a piece of a
   larger object that does not parse; `stray_brace_around_object` =
   braces but no keys. Under this rule instructor is 93 fragments /
   3 recovered / 2 stray / 1 several, which matches an independent hand count
   of all 99. Base-panel counts: json_repair 2 recovered, LangChain 2.
   Added after the data, disclosed here and in the artifact's own
   `substance_check_post_hoc.why`; the frozen U2 readings do not read it.
2. "Turned into objects with string-valued expressions" over-described
   llama-index's 27 YAML acceptances; the artifact now banks the count
   of manufactured objects carrying an expression-valued string
   (15 of 33) and the row says "accepted".
3. The row's "(it did not)" for the instructor prediction read as the
   prediction failing; it lost none (0/1,823). Reworded.
4. smolagents parses with `json.loads(strict=False)`, not a strict
   parse; the row now says so.
