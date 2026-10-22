#!/usr/bin/env python3
"""Addendum Z: print the page text its frozen reading licenses. Writes nothing.

The day `experiments/novel_schema_rerun_2026-09-23.json` lands, every page
that quotes +46.5 has to change the way the protocol froze it, and only
that way: the primary-verifiable number replaces +46.5 whichever way it
moved (banked number kept as history); a non-GO withdraws the headline;
DIFFERENT EXCLUSIONS moves nothing until the discrepancy is filed. This
script holds the replacement text for every such place, written before
any arm existed, and fills it from the reading. It prints; a person
applies. It refuses, and prints no page text, when:

- the reading file (or either summary it rests on) is absent;
- the reading does not carry the runner's protocol, spec and date
  (the same identity `novel_schema_rerun._six` demands of an arm);
- the protocol's bytes no longer hash to what its OpenTimestamps proof
  commits to (`decoder_isolation_read.anchored_digest`, the repo's one
  proof reader);
- a file that decides what the run computes changed since the freeze
  commit fa94857 (`addendum_z_run_log.code_pins`). One exception, and
  only at its exact digests: scripts/run_addendum_z.sh, whose liveness
  check changed in 534e7fd after the freeze, is accepted only when its
  sha256 at fa94857 is
  1c96957f74d958e9b36355b278c6f7622d4c95ae94df4f9b7d68d12fd91b8d67
  and its sha256 now is
  64211d3dc9b99a3a2d3c6fe3a890253f8c64dad61cbadf855755a227f91af0fb.
  Any other change to it refuses;
- the run log shows a logged document or adapter was redrawn
  (`addendum_z_run_log.verify`);
- the stranger's verifier does not pass on the landed arms
  (`verify_z_from_primary.py`: every score, the gate and Z1 to Z4
  re-derived from the stored outputs);
- the reading disagrees with the summaries it was computed from, or Z2
  does not re-derive (`novel_schema_rerun.agreement_reading`);
- any placeholder would stay unfilled, or a template breaks the
  non-flattering rule (checked on every run, not only in tests).

Landing-day runbook, in this order:

    python3 scripts/publish_z_reading.py --check-anchors   # FIRST: every anchor on the live pages
    python3 scripts/publish_z_reading.py                   # then print the page text
    python3 scripts/publish_z_reading.py --site verdict-z-row
    python3 scripts/publish_z_reading.py --templates WITHDRAWN   # the staged text, unfilled
    python3 scripts/publish_z_reading.py --list

`--check-anchors` is an operator command, not a CI test. It reads the
live pages, needs no reading, and exits 1 if any anchor is missing or
ambiguous, if a rewrite would land inside a dated `## Update` section,
or if a tracked page carries a +46.5 that no site rewrites, no banner
covers and no rule leaves alone. The CI tests run on frozen excerpts
instead (public pages under tests/fixtures/z_publish/, internal pages
beside the internal site file, which the public export drops), so prose
edits elsewhere do not break them. After re-anchoring a template,
refresh the excerpts with `--freeze-fixtures`.

Exit 0: printed. Exit 2: refused. Exit 3: printed, but some anchors were
not found because a page changed after this was staged; place that text
by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import addendum_z_run_log as runlog  # noqa: E402
import novel_schema_rerun as z  # noqa: E402

READING_NAME = f"novel_schema_rerun_{z.DATE}.json"
NEW_SUMMARY_NAME = f"novel_schema_summary_{z.DATE}.json"
BANKED_SUMMARY_NAME = f"novel_schema_summary_{z.BANKED_DATE}.json"
PROTOCOL_PATH = REPO / "docs" / "research" / "ADDENDUM_Z_PROTOCOL.md"
PROTOCOL_STAMP = "2026-09-23T0219Z"
BANKED_MEAN = 0.4648  # the protocol's m0, as the reader rounds it
BANKED_PTS = "+46.5"

# Changed since the freeze commit and known not to touch what the run
# computes: the launcher's liveness check (commit 534e7fd, 2026-09-23
# 03:14Z) now also checks the pid's command line, because pids restart
# after a container reboot. Pinned to the exact before/after digests, so
# any further change to the launcher refuses.
KNOWN_CHANGES = {
    "scripts/run_addendum_z.sh": (
        "1c96957f74d958e9b36355b278c6f7622d4c95ae94df4f9b7d68d12fd91b8d67",
        "64211d3dc9b99a3a2d3c6fe3a890253f8c64dad61cbadf855755a227f91af0fb",
    ),
}

OUTCOMES = ("AGREES", "MOVED", "WITHDRAWN", "NOT_COMPARABLE")
EOL = "<end of line>"

# The protocol's own sentences, quoted verbatim (tests pin each one
# against docs/research/ADDENDUM_Z_PROTOCOL.md, whitespace-normalized).
P_DISAGREE = "the two runs disagree, the disagreement is the result"
P_FIRST = "no page keeps the banked GO on the strength of its being first"
P_LARGER = ("A larger number is not promoted on the strength of being larger; a smaller one is not softened on the strength of being second.")
P_NOTHING = "nothing moves on any page"
P_CAP = ("the cap is a property of the corpus and the tokenizer, not of the box, so a difference is a generator or tokenizer change and is read before anything else")
P_LIMIT = "Six arms, one rung, one box."
P_FAIL = "a failure publishes as written"
PROTOCOL_QUOTES = (P_DISAGREE, P_FIRST, P_LARGER, P_NOTHING, P_CAP, P_LIMIT, P_FAIL)

# Z3's last clause, as `novel_schema_rerun.spread_reading` writes it.
B93_CLAUSES = (
    "it is larger, so it is the figure quoted from now on",
    "it is not larger, so B.9.3's figure stands and this one sits beside it",
)

PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
FLATTERING = re.compile(
    r"\b(confirm(s|ed|ing)?|validat\w*|prove[sn]?|robust\w*|strong(er|est|ly)?|"
    r"independently|replicated|reproduce[sd]?|impressive\w*|remarkabl\w*|decisive\w*|"
    r"landmark|vindicat\w*|held up|holds up|survive[sd]?)\b",
    re.IGNORECASE,
)
STALE = ("re-run pending", "not yet landed", "no number exists yet", "no result is known",
         "it has no result")


class Refused(Exception):
    pass


# -- templates ---------------------------------------------------------------
#
# Each site: where it is (start/end anchors copied from the page as it
# stood when this was staged), and the text for each outcome. A value of
# None means "this outcome leaves the place unchanged". A dict value
# overrides the anchors for that outcome. Placeholders are {UPPER_CASE};
# a template that names a value the reading does not license (for
# example {NEW_PTS} when the headline is withdrawn) refuses.

ARTIFACTS = ("`experiments/novel_schema_0.5b_k30_seed{1,2,3}_{adapted,kshot}_2026-09-23.json` (the arms, PRIMARY), `experiments/novel_schema_summary_2026-09-23.json`, `experiments/novel_schema_rerun_2026-09-23.json`")
Z_TITLE = "| **Addendum Z: the Addendum B gate, run again with every prediction stored** | "
Z_CHECK = (" | " + ARTIFACTS + " | `python3 scripts/verify_z_from_primary.py` (re-derives every "
           "score, the gate and Z1 to Z4 from the stored outputs, and checks the run log: no logged document or adapter was redrawn); `python3 scripts/publish_z_reading.py` (re-prints this row from the reading) |")
Z_READER = ("The unchanged reader (`scripts/novel_schema_summary.py`, `--date 2026-09-23`) on the six re-run arms: \"{Z1_DETAIL}\" ")
Z_Z3 = ("**Z3 {Z3_WORD}:** the largest per-arm difference between the two runs is {Z3_MAX} micro-F1 ({Z3_MAX_AT}), against a prediction of at most 0.03. It is an upper bound on run-to-run plus environment spread, which this design cannot separate. Against B.9.3's single-pair 0.0098, {Z3_B93_CLAUSE} (dated erratum beside B.9.3, spec P16). ")
Z_Z4 = ("**Z4 {Z4_WORD}:** over-cap documents per arm, re-run vs banked: {Z4_TABLE}. Sampling attrition (all five completions empty) is counted separately: {EMPTY_POOL_TOTAL} documents across the six arms. ")
Z_TAIL = ("**Predictions written before the run:** {PRED_LINE}. Process restarts per arm: {RESTARTS}. **What it cannot show:** this is a replication by the same operator on a second CPU box. It makes a stranger's replication possible; it is not one. " + P_LIMIT)

GATE_STATS = ("| Sign test | {WINS}W / {LOSSES}L / {TIES}T over {N_SCORED} scored paired receipts of {N_DESIGNED} designed ({P_STR}; banked run: 156W / 0L / 2T over 158 of 180) | same | same |\n| CIs | receipt-level [{CI_LO}, {CI_HI}], the unchanged reader's interval on the re-run. Banked run: receipt-level [42.9, 49.4], cluster-level (3 seeds) [23.1, 69.9]. The re-run's cluster-level interval is not in the reading file and is not quoted here. | same | same |\n| Baseline validity (not a floor) | kshot arms {KSHOT_RANGE} on the re-run (banked: 0.4333–0.6208), inside preregistered [0.15, 0.95] window | per-arm `novel_schema_0.5b_k30_seed*_kshot_2026-09-23.json` | same |\n| Scoping of the gate's delta (the re-run's, and the banked +46.5) | adaptation ADDED ON TOP of the same model's 30-shot prompt, at 0.5B | spec B.9.1 | read the spec |")

README_LEAD = ("Four commands check the arithmetic against the raw records, with\nnothing installed. The scripts print fractions; the prose quotes F1\npoints (+0.4035 is +40.4):\n\n")
COVERAGE_HEAD = "\"The +46.5 headline is the least verifiable number here, and \""
COVERAGE_PRICE = ("The fix was to re-run those arms with predictions stored; the Addendum Z protocol's estimate is ~35-57 wall-hours on four cores, quoted there as 40-60, not the nine CPU-hours first written (CORRECTIONS.md, 2026-09-23 and 2026-10-21). ")
COVERAGE_GO = ("The banked +46.5 headline was the least verifiable number here. `novel_schema_summary_2026-08-12.json` is AGGREGATE and its ARMS are ARITHMETIC: per-receipt scores, no predictions, so the step from model output to score cannot be re-checked by anyone, including us, and never will be. " + COVERAGE_PRICE
               + "Addendum Z did it and landed {LANDED_DATE}: its six arms store every "
                 "prediction, "
               "and the number quoted for the gate is now its {NEW_PTS} ")


def _pystr(prose: str, indent: int = 12, width: int = 66) -> str:
    """Prose as the Python source of a parenthesized string literal run, ending '),'."""
    pieces, cur = [], ""
    for word in re.split(r"(?<= )", prose):
        if cur and len(cur) + len(word) > width:
            pieces.append(cur)
            cur = word
        else:
            cur += word
    pieces.append(cur)
    quoted = ['"' + p.replace("\\", "\\\\").replace('"', '\\"') + '"' for p in pieces]
    return ("\n" + " " * indent).join(quoted) + "),"


def _banner(outcome: str) -> str | None:
    return {
        "AGREES": ("**Addendum Z landed {LANDED_DATE}.** Every +46.5 in this file is the banked 2026-08-12 figure, kept as history. The re-run with every model output stored read GO at {NEW_PTS} (Z2 AGREES); quote {NEW_PTS}, from `experiments/novel_schema_rerun_2026-09-23.json`. This file is a dated record and is not otherwise edited."),
        "MOVED": ("**Addendum Z landed {LANDED_DATE}.** Every +46.5 in this file is the banked 2026-08-12 figure, kept as history. The re-run with every model output stored read GO at {NEW_PTS}, and the gate mean {MOVE_PHRASE}, more than the frozen 0.05 tolerance (Z2 GO BUT MOVED). Quote {NEW_PTS} with that movement beside it, from `experiments/novel_schema_rerun_2026-09-23.json`. This file is a dated record and is not otherwise edited."),
        "WITHDRAWN": ("**Addendum Z landed {LANDED_DATE}: the +46.5 headline quoted in this file "
                      "is withdrawn.** The re-run with every model output stored read {Z1_WORD} "
                      "where the "
                      "banked run read GO (Z2 DISAGREES). By the frozen protocol " + P_DISAGREE + ". "
                      "Do not quote the withdrawn +46.5 from this file as a result (`CORRECTIONS.md` {LANDED_DATE}). This file is a dated record and is not otherwise edited."),
        "NOT_COMPARABLE": ("**Addendum Z landed {LANDED_DATE} NOT COMPARABLE.** The re-run scored a different document set from the banked run (Z4 DIFFERENT EXCLUSIONS), so every +46.5 in this file stands unconfirmed until the discrepancy filed in `CORRECTIONS.md` is resolved. The re-run's own gate word was {Z1_WORD}. This file is a dated record and is not otherwise edited."),
    }[outcome]


def _site(id, path, start, end=None, kind="replace", optional=False, note="", **texts):
    names = {"agrees": "AGREES", "moved": "MOVED", "withdrawn": "WITHDRAWN", "nc": "NOT_COMPARABLE"}
    table = {}
    go = texts.pop("go", None)
    for key, outcome in names.items():
        table[outcome] = texts.get(key, go if outcome in ("AGREES", "MOVED") else None)
    return {"id": id, "path": path, "start": start, "end": end if end is not None else start,
            "kind": kind, "optional": optional, "note": note, "text": table}


SITES = [
    # ---------------------------------------------------------------- VERDICT
    _site(
        "verdict-headline-row", "VERDICT.md", "| Novel-schema k=30 gate (Addendum B) |", EOL,
        agrees=("| Novel-schema k=30 gate (Addendum B, re-run as Addendum Z) | GO: {NEW_PTS} mean F1 over seeds {1,2,3} ({SEED_DELTAS_PTS}) vs +5 bar, from the Addendum Z re-run with every model output stored (landed {LANDED_DATE}; Z2 AGREES: the gate mean {MOVE_PHRASE}). History: the banked 2026-08-12 run read +46.5 (+36.0/+49.0/+54.4) and discarded its model outputs. | `experiments/novel_schema_summary_2026-09-23.json`, `experiments/novel_schema_rerun_2026-09-23.json`; banked: `experiments/novel_schema_summary_2026-08-12.json` | `python3 scripts/verify_z_from_primary.py` (re-derives every score and the reading from the stored outputs); banked run: `python3 scripts/verify_verdict.py` |"),
        moved=("| Novel-schema k=30 gate (Addendum B, re-run as Addendum Z) | GO: {NEW_PTS} mean F1 over seeds {1,2,3} ({SEED_DELTAS_PTS}) vs +5 bar, from the Addendum Z re-run with every model output stored (landed {LANDED_DATE}). **Moved: Z2 GO BUT MOVED, the gate mean {MOVE_PHRASE}, more than the frozen 0.05 tolerance.** History: the banked 2026-08-12 run read +46.5 (+36.0/+49.0/+54.4) and discarded its model outputs. | `experiments/novel_schema_summary_2026-09-23.json`, `experiments/novel_schema_rerun_2026-09-23.json`; banked: `experiments/novel_schema_summary_2026-08-12.json` | `python3 scripts/verify_z_from_primary.py` (re-derives every score and the reading from the stored outputs); banked run: `python3 scripts/verify_verdict.py` |"),
        withdrawn=("| Novel-schema k=30 gate (Addendum B) | **WITHDRAWN {LANDED_DATE}.** The "
                   "Addendum Z re-run of this gate, with every model output stored, read "
                   "{Z1_WORD} where the banked run read GO "
                   "(Z2 DISAGREES). By the frozen protocol " + P_DISAGREE + ", and " + P_FIRST + ". History "
                   "only, withdrawn: the banked 2026-08-12 run read GO at +46.5 (+36.0/+49.0/+54.4) and discarded its model outputs. | `experiments/novel_schema_rerun_2026-09-23.json`, `experiments/novel_schema_summary_2026-09-23.json`; banked: `experiments/novel_schema_summary_2026-08-12.json` | `CORRECTIONS.md` {LANDED_DATE} |"),
        nc=("| Novel-schema k=30 gate (Addendum B) | GO: +46.5 mean F1 over seeds {1,2,3} "
            "(+36.0/+49.0/+54.4) vs +5 bar (banked 2026-08-12, unconfirmed: Addendum Z landed "
            "{LANDED_DATE} and read DIFFERENT EXCLUSIONS, so the re-run scored a different "
            "document set and Z2 and Z3 are NOT COMPARABLE. By "
            "the frozen protocol " + P_NOTHING + " until the discrepancy is filed in "
                                                 "`CORRECTIONS.md`. The "
            "re-run's own gate word was {Z1_WORD}; `docs/research/ADDENDUM_Z_PROTOCOL.md`) | `experiments/novel_schema_summary_2026-08-12.json`; `experiments/novel_schema_rerun_2026-09-23.json` | `python3 scripts/verify_verdict.py` |"),
    ),
    _site(
        "verdict-headline-stats", "VERDICT.md",
        "| Sign test | 156W / 0L / 2T over 158 scored paired receipts of 180 designed | same | same |",
        "| Scoping of +46.5 | adaptation ADDED ON TOP of the same model's 30-shot prompt, at 0.5B | spec B.9.1 | read the spec |",
        go=GATE_STATS,
        withdrawn=("| Sign test (withdrawn headline, banked run) | 156W / 0L / 2T over 158 scored paired receipts of 180 designed, history only | banked | `python3 scripts/verify_verdict.py` |\n| CIs (withdrawn headline, banked run) | receipt-level [42.9, 49.4]; cluster-level (3 seeds) [23.1, 69.9], history only | banked | same |\n| Baseline validity (not a floor) | re-run kshot arms {KSHOT_RANGE} against the preregistered [0.15, 0.95] window (banked: 0.4333–0.6208) | per-arm `novel_schema_0.5b_k30_seed*_kshot_2026-09-23.json` | the reader's `validity_trips` |\n| Scoping of the withdrawn +46.5 | adaptation ADDED ON TOP of the same model's 30-shot prompt, at 0.5B | spec B.9.1 | read the spec |"),
    ),
    _site(
        "verdict-format-restriction", "VERDICT.md",
        "a no-op: +46.5 → **+46.5**, +40.4 → **+40.4**, +24.0 → **+24.5**.",
        go=("a no-op: +46.5 → **+46.5** on gate 1's banked arms (history; the Addendum Z arms were not put through this restriction), +40.4 → **+40.4**, +24.0 → **+24.5**."),
        withdrawn=("a no-op: +46.5 → **+46.5** on gate 1's banked arms (that headline is withdrawn), +40.4 → **+40.4**, +24.0 → **+24.5**."),
    ),
    _site(
        "verdict-difficulty-row", "VERDICT.md", "arbitrary-mapping synthetic **+46.5/+24.0/+40.4**",
        agrees="arbitrary-mapping synthetic **+46.5/+24.0/+40.4** (gate 1 as banked; its re-run reads {NEW_PTS})",
        moved=("arbitrary-mapping synthetic **+46.5/+24.0/+40.4** (gate 1 as banked; its re-run reads {NEW_PTS}, Z2 GO BUT MOVED)"),
        withdrawn="arbitrary-mapping synthetic **+46.5/+24.0/+40.4** (gate 1's +46.5 withdrawn {LANDED_DATE})",
    ),
    _site(
        "verdict-z-row", "VERDICT.md",
        "| **Addendum Z — the +46.5 gate, run again with every prediction stored** |", EOL,
        agrees=(Z_TITLE + "**LANDED {LANDED_DATE}: Z1 GO, Z2 AGREES, Z3 {Z3_WORD}, Z4 SAME "
                          "EXCLUSIONS.** "
                + Z_READER + "**Z2 AGREES:** the gate mean is {MEAN_DELTA}; it {MOVE_PHRASE}, "
                             "inside the "
                "frozen tolerance of 0.05. By the frozen citation rule, {NEW_PTS} replaces the "
                "banked +46.5 on every unfrozen page from today, and the banked number stays "
                "beside it as history. "
                + Z_Z3 + Z_Z4 + Z_TAIL + Z_CHECK),
        moved=(Z_TITLE + "**LANDED {LANDED_DATE}: Z1 GO, Z2 GO BUT MOVED, Z3 {Z3_WORD}, Z4 SAME "
                         "EXCLUSIONS.** "
               + Z_READER + "**Z2 GO BUT MOVED: the gate mean is {MEAN_DELTA}; it {MOVE_PHRASE}, "
                            "more than "
               "the frozen tolerance of 0.05.** By the frozen citation rule, {NEW_PTS} replaces "
               "the banked +46.5 on every unfrozen page from today, with the movement beside it "
               "at full size, and the "
               "banked number stays as history. " + P_LARGER + " " + Z_Z3 + Z_Z4 + Z_TAIL + Z_CHECK),
        withdrawn=(Z_TITLE + "**LANDED {LANDED_DATE}: Z1 {Z1_WORD}, Z2 DISAGREES, Z3 {Z3_WORD}, "
                             "Z4 SAME "
                   "EXCLUSIONS. The +46.5 headline is withdrawn.** " + Z_READER + "{RERUN_MEAN_SENTENCE} "
                   "**Z2 DISAGREES:** {Z2_DETAIL}. By the frozen protocol " + P_DISAGREE + ", "
                   "`CORRECTIONS.md` carries the row, and " + P_FIRST + ". " + Z_Z3 + Z_Z4 + Z_TAIL + Z_CHECK),
        nc=(Z_TITLE + "**LANDED {LANDED_DATE}: Z4 DIFFERENT EXCLUSIONS, so Z2 and Z3 are NOT "
                      "COMPARABLE.** "
            "The two runs scored different document sets. Over-cap documents per arm, re-run vs "
            "banked: "
            "{Z4_TABLE}. By the frozen protocol, " + P_CAP + ". The citation rule is suspended, "
            + P_NOTHING + ", and the discrepancy goes to `CORRECTIONS.md` first. Until it is "
                          "resolved, "
            "+46.5 stays on this page unconfirmed. The unchanged reader's word on the re-run's own document set was {Z1_WORD}; it is not compared with the banked GO. Sampling attrition (all five completions empty), counted separately: {EMPTY_POOL_TOTAL} documents. **Predictions written before the run:** {PRED_LINE}. Process restarts per arm: {RESTARTS}. | " + ARTIFACTS
            + " | `CORRECTIONS.md` {LANDED_DATE} |"),
    ),
    _site(
        "verdict-coverage-row", "VERDICT.md",
        "**And the number this page cites most is the worst one on that scale.**",
        "launched after its anchor.",
        agrees=("**And the number this page cited most until {LANDED_DATE} was the worst one on "
                "that scale.** "
            "The banked +46.5 headline's summary is AGGREGATE and its arms are ARITHMETIC: their model outputs were discarded at generation time, so the step from output to score cannot be re-checked **by anyone, including us**, at any level. Re-running those arms with predictions stored was the fix; this row first priced it at *about nine CPU-hours*; the Addendum Z protocol computes ~35–57 wall-hours on four cores and quotes 40–60 (CORRECTIONS 2026-09-23 and 2026-10-21), wrong by ~5× if the old estimate is read as nine wall-hours of a session and by ~20× if read as literal CPU-hours. The re-run, Addendum Z, was launched after its anchor and landed {LANDED_DATE} (row above). Its six arms store every prediction, and the number this page quotes for the gate is now its {NEW_PTS} (Z2 AGREES)."),
        moved=("**And the number this page cited most until {LANDED_DATE} was the worst one on "
               "that scale.** "
            "The banked +46.5 headline's summary is AGGREGATE and its arms are ARITHMETIC: their model outputs were discarded at generation time, so the step from output to score cannot be re-checked **by anyone, including us**, at any level. Re-running those arms with predictions stored was the fix; this row first priced it at *about nine CPU-hours*; the Addendum Z protocol computes ~35–57 wall-hours on four cores and quotes 40–60 (CORRECTIONS 2026-09-23 and 2026-10-21), wrong by ~5× if the old estimate is read as nine wall-hours of a session and by ~20× if read as literal CPU-hours. The re-run, Addendum Z, was launched after its anchor and landed {LANDED_DATE} (row above). Its six arms store every prediction, and the number this page quotes for the gate is now its {NEW_PTS}, and the gate mean {MOVE_PHRASE} (Z2 GO BUT MOVED)."),
        withdrawn=("**And the number this page cited most until {LANDED_DATE} was the worst one on that scale.** The withdrawn +46.5 headline's summary is AGGREGATE and its arms are ARITHMETIC: their model outputs were discarded at generation time, so the step from output to score cannot be re-checked **by anyone, including us**, at any level. Re-running those arms with predictions stored was the fix; this row first priced it at *about nine CPU-hours*; the Addendum Z protocol computes ~35–57 wall-hours on four cores and quotes 40–60 (CORRECTIONS 2026-09-23 and 2026-10-21), wrong by ~5× if the old estimate is read as nine wall-hours of a session and by ~20× if read as literal CPU-hours. The re-run, Addendum Z, was launched after its anchor and landed {LANDED_DATE} (row above): it read {Z1_WORD}, and the headline is withdrawn."),
        nc={"start": "and the re-run is preregistered as Addendum Z (PENDING row above), launched after its anchor.",
            "text": ("and the re-run, Addendum Z, launched after its anchor, landed {LANDED_DATE} NOT COMPARABLE (row above): it scored a different document set, and " + P_NOTHING
                     + " until the discrepancy is filed.")},
    ),
    _site(
        "verdict-coverage-gone", "VERDICT.md", "The +46.5 arms are gone for good.",
        go="The banked +46.5 arms are gone for good.",
        withdrawn="The withdrawn +46.5 arms are gone for good.",
    ),
    _site(
        "verdict-headroom-row", "VERDICT.md", "synthetic baseline 0.43–0.62 → +46.5;",
        agrees="synthetic baseline 0.43–0.62 → +46.5 (banked; its Addendum Z re-run reads {NEW_PTS});",
        moved="synthetic baseline 0.43–0.62 → +46.5 (banked; its Addendum Z re-run reads {NEW_PTS}, Z2 GO BUT MOVED);",
        withdrawn="synthetic baseline 0.43–0.62 → +46.5 (withdrawn {LANDED_DATE});",
    ),
    _site(
        "verdict-primary-paragraph", "VERDICT.md",
        "**Which rows are primary-verifiable, and which are not.**",
        "preregistered as Addendum Z and not yet landed.",
        agrees=("**Which rows are primary-verifiable, and which are not.** Addendum E/F onward "
                "store raw "
            "predictions and can be re-scored against gold regenerated from the deterministic generator. **Addendum B's banked arms do not**: they record per-receipt scores and no predictions, so the step from model output to score behind the banked +46.5 cannot be re-checked, only the aggregation of scores already recorded. `verify_from_primary.py` says so rather than passing them silently. Re-running those arms with predictions stored was the fix, priced here until 2026-09-23 as ~9 CPU-hours for the prompted arms, which was wrong (CORRECTIONS 2026-09-23; by the arithmetic of the `ADDENDUM_Z_PROTOCOL.md` cost paragraph, the three prompted arms alone are ~14–24 wall-hours on four cores: 158 scored documents at its 5–9 minutes each, a per-document figure that is prose only, not banked). Addendum Z did it and landed {LANDED_DATE}; the number quoted for the gate is now its {NEW_PTS} (Z2 AGREES), and the banked figure is history."),
        moved=("**Which rows are primary-verifiable, and which are not.** Addendum E/F onward "
               "store raw "
            "predictions and can be re-scored against gold regenerated from the deterministic generator. **Addendum B's banked arms do not**: they record per-receipt scores and no predictions, so the step from model output to score behind the banked +46.5 cannot be re-checked, only the aggregation of scores already recorded. `verify_from_primary.py` says so rather than passing them silently. Re-running those arms with predictions stored was the fix, priced here until 2026-09-23 as ~9 CPU-hours for the prompted arms, which was wrong (CORRECTIONS 2026-09-23; by the arithmetic of the `ADDENDUM_Z_PROTOCOL.md` cost paragraph, the three prompted arms alone are ~14–24 wall-hours on four cores: 158 scored documents at its 5–9 minutes each, a per-document figure that is prose only, not banked). Addendum Z did it and landed {LANDED_DATE}; the number quoted for the gate is now its {NEW_PTS}, and the gate mean {MOVE_PHRASE} (Z2 GO BUT MOVED); the banked figure is history."),
        withdrawn=("**Which rows are primary-verifiable, and which are not.** Addendum E/F onward store raw predictions and can be re-scored against gold regenerated from the deterministic generator. **Addendum B's banked arms do not**: they record per-receipt scores and no predictions, so the step from model output to score behind the withdrawn +46.5 cannot be re-checked, only the aggregation of scores already recorded. Re-running those arms with predictions stored was the fix, priced here until 2026-09-23 as ~9 CPU-hours for the prompted arms, which was wrong (CORRECTIONS 2026-09-23; by the arithmetic of the `ADDENDUM_Z_PROTOCOL.md` cost paragraph, the three prompted arms alone are ~14–24 wall-hours on four cores: 158 scored documents at its 5–9 minutes each, a per-document figure that is prose only, not banked). Addendum Z did it and landed {LANDED_DATE}: it read {Z1_WORD}, and the headline is withdrawn."),
        nc={"start": "preregistered as Addendum Z and not yet landed.",
            "text": ("preregistered as Addendum Z, which landed {LANDED_DATE} NOT COMPARABLE: it "
                     "scored a "
                     "different document set (Z4 DIFFERENT EXCLUSIONS), and " + P_NOTHING + " until the "
                     "discrepancy is filed in `CORRECTIONS.md`.")},
    ),
    # ---------------------------------------------------------------- ROADMAP
    _site(
        "roadmap-z", "ROADMAP.md", "- **Addendum Z (preregistered and anchored 2026-09-23T0219Z;",
        "the nine this repository had estimated (CORRECTIONS).",
        agrees=("- **Addendum Z (landed {LANDED_DATE}): GO, and Z2 AGREES with the banked run.** The Addendum B gate ran again on this box with every prediction stored, under a protocol whose anchor was attested before any document of the run was decoded. Z1 GO; Z2 AGREES (the gate mean {MOVE_PHRASE}); Z3 {Z3_WORD} (largest per-arm difference {Z3_MAX}); Z4 SAME EXCLUSIONS. By the frozen citation rule, {NEW_PTS} replaces the banked +46.5 on every unfrozen page, with the banked number beside it as history. Owed the same day: dated errata at the frozen sites (spec P16), `scripts/verify_verdict.py` by a second date, and the coverage map regenerated with its \"is not done\" sentence and the test that pins it retired."),
        moved=("- **Addendum Z (landed {LANDED_DATE}): GO, but the gate mean moved (Z2 GO BUT MOVED).** The Addendum B gate ran again on this box with every prediction stored, under a protocol whose anchor was attested before any document of the run was decoded. Z1 GO; Z2 GO BUT MOVED (the gate mean {MOVE_PHRASE}, more than the frozen 0.05 tolerance); Z3 {Z3_WORD} (largest per-arm difference {Z3_MAX}); Z4 SAME EXCLUSIONS. By the frozen citation rule, {NEW_PTS} replaces the banked +46.5 on every unfrozen page, with the movement beside it at full size and the banked number as history. Owed the same day: dated errata at the frozen sites (spec P16), `scripts/verify_verdict.py` by a second date, and the coverage map regenerated with its \"is not done\" sentence and the test that pins it retired."),
        withdrawn=("- **Addendum Z (landed {LANDED_DATE}): {Z1_WORD}. The +46.5 headline is "
                   "withdrawn.** The Addendum B gate ran again on this box with every prediction "
                   "stored, under a protocol whose anchor was attested before any document of the "
                   "run was decoded, and the unchanged reader read {Z1_WORD} where the banked "
                   "run read GO (Z2 DISAGREES). By the frozen protocol " + P_DISAGREE + ", and "
                   "`CORRECTIONS.md` carries the row. Z3 {Z3_WORD}; Z4 SAME EXCLUSIONS. Owed the same day: dated errata at every frozen site that carries the withdrawn +46.5 (spec P16), and the coverage map regenerated."),
        nc=("- **Addendum Z (landed {LANDED_DATE}): DIFFERENT EXCLUSIONS, so Z2 and Z3 are NOT "
            "COMPARABLE.** The re-run scored a different document set from the banked run. By the "
            "frozen protocol, "
            + P_CAP + ". The citation rule is suspended, " + P_NOTHING + ", and the discrepancy goes to "
            "`CORRECTIONS.md` first. The re-run's own gate word was {Z1_WORD}. Next: find the cause and file it; until then the banked +46.5 stands unconfirmed."),
    ),
    # --------------------------------------------------------------- EVIDENCE
    _site(
        "evidence-intro", "EVIDENCE.md",
        "Gate 1, the +46.5\nheadline, is the exception:", "because it is the headline.",
        agrees=("Gate 1 is the exception: its 2026-08-12 freeze predates the public repository "
                "and rests on git "
            "history we control. Its re-run, Addendum Z, kept the bar and the rule, and its protocol's anchor was attested before any document of the run was decoded. The number quoted for gate 1 is now that re-run's {NEW_PTS} (Z2 AGREES; the banked +46.5 is history). Stated here rather than only in the ledger at the bottom, because it is the headline."),
        moved=("Gate 1 is the exception: its 2026-08-12 freeze predates the public repository and "
               "rests on git "
            "history we control. Its re-run, Addendum Z, kept the bar and the rule, and its protocol's anchor was attested before any document of the run was decoded. The number quoted for gate 1 is now that re-run's {NEW_PTS}; the gate mean {MOVE_PHRASE} (Z2 GO BUT MOVED), and the banked +46.5 is history. Stated here rather than only in the ledger at the bottom, because it is the headline."),
        withdrawn=("Gate 1, the withdrawn +46.5 headline, was the exception: its 2026-08-12 freeze predates the public repository and rests on git history we control. It was withdrawn on {LANDED_DATE}, when its re-run, Addendum Z, read {Z1_WORD} (`CORRECTIONS.md`)."),
    ),
    _site(
        "evidence-gate1-row", "EVIDENCE.md",
        "| 1 | Does adaptation beat the **same model's own 30-shot prompt** on schemas it has never seen? |", EOL,
        agrees=("| 1 | Does adaptation beat the **same model's own 30-shot prompt** on schemas it has never seen? | **GO** | **{NEW_PTS} F1** seed-mean over 3 tenant schemas ({SEED_DELTAS_PTS}) vs a +5.0 bar frozen beforehand, from the Addendum Z re-run with every model output stored (landed {LANDED_DATE}; Z2 AGREES: the gate mean {MOVE_PHRASE}). Sign test {WINS}W/{LOSSES}L/{TIES}T over **{N_SCORED} scored pairs of {N_DESIGNED} designed**; {N_OVER_CAP} were excluded because the 30-shot prompt exceeded the frozen 8192-token budget, disclosed symmetrically, and {EMPTY_POOL_TOTAL} documents were lost to empty sampling pools across the six arms. History: the banked 2026-08-12 run read +46.5 (+36.0/+49.0/+54.4), 156W/0L/2T over 158 of 180, and discarded its model outputs. |"),
        moved=("| 1 | Does adaptation beat the **same model's own 30-shot prompt** on schemas it has never seen? | **GO** | **{NEW_PTS} F1** seed-mean over 3 tenant schemas ({SEED_DELTAS_PTS}) vs a +5.0 bar frozen beforehand, from the Addendum Z re-run with every model output stored (landed {LANDED_DATE}). **The gate mean {MOVE_PHRASE}, more than the frozen 0.05 tolerance (Z2 GO BUT MOVED).** Sign test {WINS}W/{LOSSES}L/{TIES}T over **{N_SCORED} scored pairs of {N_DESIGNED} designed**; {N_OVER_CAP} were excluded because the 30-shot prompt exceeded the frozen 8192-token budget, disclosed symmetrically, and {EMPTY_POOL_TOTAL} documents were lost to empty sampling pools across the six arms. History: the banked 2026-08-12 run read +46.5 (+36.0/+49.0/+54.4), 156W/0L/2T over 158 of 180, and discarded its model outputs. |"),
        withdrawn=("| 1 | Does adaptation beat the **same model's own 30-shot prompt** on schemas "
                   "it has never seen? | **WITHDRAWN** | The Addendum Z re-run, with every model "
                   "output stored, read **{Z1_WORD}** where the banked run read GO (landed "
                   "{LANDED_DATE}; Z2 DISAGREES). By the "
                   "frozen protocol " + P_DISAGREE + ". History only, withdrawn: the banked "
                                                     "2026-08-12 run "
                   "read GO at +46.5 F1 (+36.0/+49.0/+54.4) and discarded its model outputs. See `CORRECTIONS.md` {LANDED_DATE}. |"),
        nc=("| 1 | Does adaptation beat the **same model's own 30-shot prompt** on schemas it has "
            "never seen? | **GO** | **+46.5 F1** seed-mean over 3 tenant schemas "
            "(+36.0/+49.0/+54.4) vs a +5.0 bar frozen beforehand. Sign test 156W/0L/2T over **158 "
            "scored pairs of 180 designed**; 22 were excluded because the 30-shot prompt exceeded "
            "the frozen 8192-token budget, disclosed symmetrically. (Banked 2026-08-12, "
            "unconfirmed: Addendum Z landed {LANDED_DATE} and read DIFFERENT EXCLUSIONS, so the "
            "re-run scored a different document set and Z2 and Z3 are NOT COMPARABLE; by the "
            "frozen "
            "protocol " + P_NOTHING + " until the discrepancy is filed in `CORRECTIONS.md`. The "
                                      "re-run's own "
            "gate word was {Z1_WORD}.) |"),
    ),
    _site(
        "evidence-baseline-sentence", "EVIDENCE.md",
        "On the synthetic novel-schema corpus the same model's 30-shot prompt scores 0.4333–0.6208, which is what makes +46.5 possible.",
        go=("On the synthetic novel-schema corpus the same model's 30-shot prompt scores {KSHOT_RANGE} on the Addendum Z re-run (0.4333–0.6208 in the banked run behind +46.5), which is what leaves room for the re-run's delta."),
        withdrawn=("On the synthetic novel-schema corpus the same model's 30-shot prompt scored 0.4333–0.6208 in the banked run, the baseline under the withdrawn +46.5 headline."),
    ),
    _site(
        "evidence-format-table", "EVIDENCE.md", "| 1 (novel schema, k=30) | **0 of 158** | +46.5 | **+46.5** |",
        go=("| 1 (novel schema, k=30), banked 2026-08-12 arms (history; not re-measured on the Addendum Z arms) | **0 of 158** | +46.5 | **+46.5** |"),
        withdrawn="| 1 (novel schema, k=30), banked arms; headline withdrawn {LANDED_DATE} | **0 of 158** | +46.5 | **+46.5** |",
    ),
    _site(
        "evidence-least-verifiable", "EVIDENCE.md",
        "- **The headline is the least verifiable number on this page, and that is worth saying plainly.**",
        "and it is preregistered as Addendum Z and not yet landed.",
        agrees=("- **Gate 1's banked arms are the least verifiable on this page; its re-run's "
                "arms store every "
            "output.** The banked arms recorded a per-receipt `micro_f1` and **no predictions**, so `verify_from_primary.py` cannot re-score the banked +46.5 against regenerated gold, and never will: that number is history. Addendum Z re-ran the gate with every model output stored, under a protocol whose anchor was attested before any document of the run was decoded, and landed {LANDED_DATE}. Its six arms store every pooled completion and the selected text per document, and the number quoted for gate 1 is now its {NEW_PTS} (Z2 AGREES). It is a replication by the same operator on a second CPU box, not by a stranger; it makes a stranger's replication possible, and none has happened. The row-1 freeze itself still rests on private git history."),
        moved=("- **Gate 1's banked arms are the least verifiable on this page; its re-run's arms "
               "store every "
            "output.** The banked arms recorded a per-receipt `micro_f1` and **no predictions**, so `verify_from_primary.py` cannot re-score the banked +46.5 against regenerated gold, and never will: that number is history. Addendum Z re-ran the gate with every model output stored, under a protocol whose anchor was attested before any document of the run was decoded, and landed {LANDED_DATE}. Its six arms store every pooled completion and the selected text per document, and the number quoted for gate 1 is now its {NEW_PTS}; the gate mean {MOVE_PHRASE} (Z2 GO BUT MOVED). It is a replication by the same operator on a second CPU box, not by a stranger; it makes a stranger's replication possible, and none has happened. The row-1 freeze itself still rests on private git history."),
        withdrawn=("- **The headline was the least verifiable number on this page, and it is "
                   "withdrawn.** Gate 1's banked arms recorded a per-receipt `micro_f1` and **no "
                   "predictions**, so the withdrawn +46.5 could never be re-scored against "
                   "regenerated gold. Addendum Z re-ran the gate with every model output stored, "
                   "under a protocol whose anchor was attested before any document of the run was "
                   "decoded, and the unchanged reader read {Z1_WORD} (landed {LANDED_DATE}; Z2 "
                   "DISAGREES). By that protocol "
                   + P_DISAGREE + " (`CORRECTIONS.md` {LANDED_DATE})."),
        nc={"start": "and it is preregistered as Addendum Z and not yet landed.",
            "text": ("and Addendum Z, which re-ran it with every output stored, landed "
                     "{LANDED_DATE} NOT COMPARABLE: it scored a different document set (Z4 "
                     "DIFFERENT EXCLUSIONS), so "
                     + P_NOTHING + " until the discrepancy is filed in `CORRECTIONS.md`.")},
    ),
    # ----------------------------------------------------------------- README
    _site(
        "readme-verify-commands", "README.md",
        "Three commands check the arithmetic against the raw records, with",
        "# gate 1: prints GO, seed mean +46.5",
        go=(README_LEAD +
            "    python3 scripts/verify_verdict.py                                          # gate 1, banked 2026-08-12 run: prints GO, seed mean +46.5 (history)\n    python3 scripts/verify_z_from_primary.py                                   # gate 1, Addendum Z re-run: re-derives every score and the reading from stored outputs"),
        withdrawn=(README_LEAD +
                   "    python3 scripts/verify_verdict.py                                          # gate 1, banked run, headline withdrawn: prints GO, seed mean +46.5\n    python3 scripts/verify_z_from_primary.py                                   # gate 1, Addendum Z re-run: re-derives every score and the {Z1_WORD} reading from stored outputs"),
        nc=(README_LEAD +
            "    python3 scripts/verify_verdict.py                                          # gate 1: prints GO, seed mean +46.5 (banked; its re-run is NOT COMPARABLE)\n    python3 scripts/verify_z_from_primary.py                                   # gate 1, Addendum Z re-run: re-derives every score from stored outputs"),
    ),
    _site(
        "readme-status", "README.md",
        "**Status (2026-08-17): the preregistered k=30 adaptation gate decided GO —",
        "and we publish our negatives.",
        agrees=("**Status ({LANDED_DATE}): the preregistered k=30 adaptation gate, re-run as Addendum Z with every model output stored, reads GO: mean {NEW_PTS} micro-F1 over 30-shot prompting across three novel-schema seeds ({SEED_DELTAS_PTS} vs a +5 bar; receipt-level sign test {WINS}W/{LOSSES}L/{TIES}T over {N_SCORED} scored of {N_DESIGNED} designed ({P_STR}); receipt-level 95% CI [{CI_LO}, {CI_HI}]; `experiments/novel_schema_rerun_2026-09-23.json`).** Z2 AGREES with the banked 2026-08-12 run (the gate mean {MOVE_PHRASE}); the banked +46.5 stays as history, and that run's model outputs were discarded (`docs/research/ADDENDUM_Z_PROTOCOL.md`). Stated per the spec's claim rule, always beside the CORD negative: on CORD receipts, a domain the base model already knows, the same adaptation recipe FAILED its preregistered gates at all three scales tested (Addendum A: −7.3 / −11.5 / −4.5 F1 on Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and Qwen3-4B-Instruct-2507). Adaptation buys novelty, not general quality, and we publish our negatives."),
        moved=("**Status ({LANDED_DATE}): the preregistered k=30 adaptation gate, re-run as Addendum Z with every model output stored, reads GO: mean {NEW_PTS} micro-F1 over 30-shot prompting across three novel-schema seeds ({SEED_DELTAS_PTS} vs a +5 bar; receipt-level sign test {WINS}W/{LOSSES}L/{TIES}T over {N_SCORED} scored of {N_DESIGNED} designed ({P_STR}); receipt-level 95% CI [{CI_LO}, {CI_HI}]; `experiments/novel_schema_rerun_2026-09-23.json`).** **It moved: the gate mean {MOVE_PHRASE}, more than the frozen 0.05 tolerance (Z2 GO BUT MOVED).** The banked +46.5 stays as history, and that run's model outputs were discarded (`docs/research/ADDENDUM_Z_PROTOCOL.md`). Stated per the spec's claim rule, always beside the CORD negative: on CORD receipts, a domain the base model already knows, the same adaptation recipe FAILED its preregistered gates at all three scales tested (Addendum A: −7.3 / −11.5 / −4.5 F1 on Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and Qwen3-4B-Instruct-2507). Adaptation buys novelty, not general quality, and we publish our negatives."),
        withdrawn=("**Status ({LANDED_DATE}): the k=30 adaptation headline is withdrawn.** Addendum Z re-ran the preregistered gate with every model output stored, and the unchanged reader read **{Z1_WORD}** where the banked 2026-08-12 run read GO (Z2 DISAGREES; `experiments/novel_schema_rerun_2026-09-23.json`). By the frozen protocol " + P_DISAGREE
                   + " (`docs/research/ADDENDUM_Z_PROTOCOL.md`; `CORRECTIONS.md` {LANDED_DATE}). "
                     "History "
                   "only, withdrawn: the banked run read +46.5 micro-F1 (+36.0 / +49.0 / +54.4) and discarded its model outputs. Stated per the spec's claim rule, beside it: on CORD receipts, a domain the base model already knows, the same adaptation recipe FAILED its preregistered gates at all three scales tested (Addendum A: −7.3 / −11.5 / −4.5 F1 on Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and Qwen3-4B-Instruct-2507). The sentence that adaptation buys novelty rested partly on this gate and is withdrawn with it; gates 4 and 5 are separate measurements and were not re-run."),
        nc={"start": "**Re-run pending:** Addendum Z, preregistered and",
            "end": "withdraws it (`docs/research/ADDENDUM_Z_PROTOCOL.md`).",
            "text": ("**Re-run landed {LANDED_DATE}, NOT COMPARABLE:** Addendum Z re-ran this "
                     "gate with every model output stored and read DIFFERENT EXCLUSIONS, so it "
                     "scored a different document set "
                     "from the banked run. By its frozen protocol " + P_NOTHING + " until the discrepancy is "
                     "filed in `CORRECTIONS.md`, and +46.5 stands unconfirmed. The re-run's own gate word was {Z1_WORD} (`docs/research/ADDENDUM_Z_PROTOCOL.md`).")},
    ),
    _site(
        "readme-format-restriction", "README.md", "nothing: +46.5 → +46.5, +40.4 → +40.4, +24.0 → +24.5.",
        go=("nothing: +46.5 → +46.5 on gate 1's banked arms (history; the Addendum Z arms were not put through this restriction), +40.4 → +40.4, +24.0 → +24.5."),
        withdrawn="nothing: +46.5 → +46.5 on gate 1's banked arms (that headline is withdrawn), +40.4 → +40.4, +24.0 → +24.5.",
    ),
    # ------------------------------------------------------------ CORRECTIONS
    _site(
        "corrections-row", "CORRECTIONS.md",
        "## Numbers corrected\n\n| Date | Was | Is | Cause |\n|---|---|---|---|\n", kind="insert_after",
        agrees=("| {LANDED_DATE} | Gate 1 headline **+46.5** (Addendum B, banked 2026-08-12; model outputs discarded, so ARITHMETIC-verifiable at best) | **{NEW_PTS}** (Addendum Z re-run with every model output stored; Z1 GO, Z2 AGREES: the gate mean {MOVE_PHRASE}) | The frozen citation rule (`docs/research/ADDENDUM_Z_PROTOCOL.md`): from the day the reading lands the primary-verifiable number is quoted, whichever way it moved, with the banked number kept beside it as history. `experiments/novel_schema_rerun_2026-09-23.json` |\n"),
        moved=("| {LANDED_DATE} | Gate 1 headline **+46.5** (Addendum B, banked 2026-08-12; model "
               "outputs discarded, so ARITHMETIC-verifiable at best) | **{NEW_PTS}** (Addendum Z "
               "re-run with every model output stored; Z1 GO, **Z2 GO BUT MOVED: the gate mean "
               "{MOVE_PHRASE}, more than the frozen 0.05 tolerance**) | The frozen citation rule "
               "(`docs/research/ADDENDUM_Z_PROTOCOL.md`): the primary-verifiable number is "
               "quoted, with the movement beside it at full size and the banked "
               "number kept as history. " + P_LARGER + " `experiments/novel_schema_rerun_2026-09-23.json` |\n"),
        withdrawn={"start": "## Claims withdrawn\n\n| Date | Claim as published | What happened | Where |\n|---|---|---|---|\n",
                   "text": ("| {LANDED_DATE} | The k=30 novel-schema gate headline: **GO at "
                            "+46.5** micro-F1 (Addendum B, banked 2026-08-12, model outputs "
                            "discarded), withdrawn | **Withdrawn.** Addendum Z re-ran the gate "
                            "with every model output stored, under a protocol whose anchor was "
                            "attested before any document of the run was decoded, and the "
                            "unchanged reader read **{Z1_WORD}** where the banked run read GO (Z2 "
                            "DISAGREES). The protocol froze this branch before the run: "
                            + P_DISAGREE + ", and " + P_FIRST + ". The reader's words: "
                                                                "\"{Z1_DETAIL}\" | "
                            "`experiments/novel_schema_rerun_2026-09-23.json`; `VERDICT.md`, `EVIDENCE.md`, `README.md`, `ROADMAP.md`, `paper/DRAFT.md`, `docs/strategy/` |\n")},
        nc={"start": "## Defects found in our own verification path\n\n| Date | Defect | Fix |\n|---|---|---|\n",
            "text": ("| {LANDED_DATE} | **Addendum Z read DIFFERENT EXCLUSIONS.** The re-run's "
                     "over-cap documents "
                     "are not the banked run's: {Z4_TABLE}. By the frozen protocol, " + P_CAP + ". The two runs "
                     "scored different document sets, Z2 and Z3 are NOT COMPARABLE, and the citation rule is suspended. The re-run's own gate word was {Z1_WORD}. | **Open.** " + P_NOTHING[0].upper()
                     + P_NOTHING[1:] + " until the cause is found and recorded here; the pages "
                                       "carry +46.5 marked "
                     "unconfirmed and NOT COMPARABLE. `experiments/novel_schema_rerun_2026-09-23.json` |\n")},
    ),
    # ------------------------------------------------- frozen spec: P16 erratum
    _site(
        "spec-erratum-p16", "docs/research/ENTERPRISE_EVAL_SPEC.md",
        "outbound draft had said the file matched its anchors.\n", kind="insert_after",
        note="appended after P15, outside the anchored first 1,303 lines; frozen text keeps its bytes",
        agrees=("\n**P16: Addendum Z landed ({LANDED_DATE}); read every +46.5 in frozen text as the banked figure, and {NEW_PTS} as the one quoted.** Addendum Z (`docs/research/ADDENDUM_Z_PROTOCOL.md`) re-ran the Addendum B k=30 gate with every model output stored. Z1 GO; Z2 AGREES: the gate mean is {MEAN_DELTA} and {MOVE_PHRASE}; Z4 SAME EXCLUSIONS (`experiments/novel_schema_rerun_2026-09-23.json`). Frozen text keeps its bytes, and each +46.5 in it now reads as the banked 2026-08-12 figure, kept as history: the B.7-r6 board note (line 823), Addendum C's preamble (line 828), B.9.1 (line 900), B.9.7 (line 961), and the tables in G.1 and H.1. F.4's \"−46.5 vs adapted-with-prompt\" (line 1153) is a different quantity and is not affected. **Beside B.9.3:** Z3 reads {Z3_WORD}. The largest per-arm difference between the two runs is {Z3_MAX} micro-F1 ({Z3_MAX_AT}). It publishes here as an upper bound on run-to-run spread for these arms; it is run-to-run plus environment, which Z's design cannot separate. Against B.9.3's single-pair 0.0098, {Z3_B93_CLAUSE}.\n"),
        moved=("\n**P16: Addendum Z landed ({LANDED_DATE}); read every +46.5 in frozen text as the banked figure, and {NEW_PTS} as the one quoted, with its movement.** Addendum Z (`docs/research/ADDENDUM_Z_PROTOCOL.md`) re-ran the Addendum B k=30 gate with every model output stored. Z1 GO; Z2 GO BUT MOVED: the gate mean is {MEAN_DELTA} and {MOVE_PHRASE}, more than the frozen 0.05 tolerance; Z4 SAME EXCLUSIONS (`experiments/novel_schema_rerun_2026-09-23.json`). Frozen text keeps its bytes, and each +46.5 in it now reads as the banked 2026-08-12 figure, kept as history: the B.7-r6 board note (line 823), Addendum C's preamble (line 828), B.9.1 (line 900), B.9.7 (line 961), and the tables in G.1 and H.1. F.4's \"−46.5 vs adapted-with-prompt\" (line 1153) is a different quantity and is not affected. **Beside B.9.3:** Z3 reads {Z3_WORD}. The largest per-arm difference between the two runs is {Z3_MAX} micro-F1 ({Z3_MAX_AT}). It publishes here as an upper bound on run-to-run spread for these arms; it is run-to-run plus environment, which Z's design cannot separate. Against B.9.3's single-pair 0.0098, {Z3_B93_CLAUSE}.\n"),
        withdrawn=("\n**P16: the +46.5 headline is withdrawn ({LANDED_DATE}).** Addendum Z (`docs/research/ADDENDUM_Z_PROTOCOL.md`) re-ran the Addendum B k=30 gate with every model output stored, and the unchanged reader read {Z1_WORD} where the banked run read GO (Z2 DISAGREES; `experiments/novel_schema_rerun_2026-09-23.json`). Frozen text keeps its bytes; read each +46.5 in it as the withdrawn banked figure: the B.7-r6 board note (line 823), Addendum C's preamble (line 828), B.9.1 (line 900), B.9.7 (line 961), and the tables in G.1 and H.1. F.4's \"−46.5 vs adapted-with-prompt\" (line 1153) is a different quantity and is not affected. **Beside B.9.3:** Z3 reads {Z3_WORD}. The largest per-arm difference between the two runs is {Z3_MAX} micro-F1 ({Z3_MAX_AT}). It publishes here as an upper bound on run-to-run spread for these arms; it is run-to-run plus environment, which Z's design cannot separate. Against B.9.3's single-pair 0.0098, {Z3_B93_CLAUSE}.\n"),
        nc=("\n**P16: Addendum Z landed NOT COMPARABLE ({LANDED_DATE}).** Z4 read DIFFERENT EXCLUSIONS: the re-run's over-cap documents are not the banked run's ({Z4_TABLE}), so the two runs scored different document sets. Z2 and Z3 are NOT COMPARABLE: no erratum is made beside B.9.3 from Z3, and the frozen +46.5 sites are unchanged until the discrepancy filed in `CORRECTIONS.md` is resolved.\n"),
    ),
    # ------------------------------------------- coverage map (code, same day)
    _site(
        "coverage-uncomfortable-one", "scripts/verification_coverage.py", COVERAGE_HEAD, "\"fix is not done.\"),",
        note="regenerate experiments/verification_coverage.json after applying",
        agrees=_pystr(COVERAGE_GO + "(Z2 AGREES). The banked 46.5 is history."),
        moved=_pystr(COVERAGE_GO + "(Z2 GO BUT MOVED: the gate mean {MOVE_PHRASE}). The banked 46.5 is history."),
        withdrawn=_pystr(
            "The withdrawn +46.5 headline was the least verifiable number here. "
            "`novel_schema_summary_2026-08-12.json` is AGGREGATE and its ARMS are ARITHMETIC: "
            "per-receipt scores, no predictions, so the step from model output to score cannot be "
            "re-checked by anyone, including us. " + COVERAGE_PRICE + "Addendum Z did it and landed "
            "{LANDED_DATE}: it read {Z1_WORD}, and the 46.5 headline is withdrawn."),
        nc={"start": "\"2026-10-21). The re-run is preregistered as Addendum Z and \"",
            "end": "\"fix is not done.\"),",
            "text": _pystr("2026-10-21). The re-run, Addendum Z, banked its six arms and landed {LANDED_DATE} NOT COMPARABLE: it scored a different document set, so nothing moves and the fix is not done.")},
    ),
    _site(
        "coverage-regenerable-contrast", "scripts/verification_coverage.py",
        "\"command in its VERDICT row. Contrast the +46.5 arms, whose \"",
        go="\"command in its VERDICT row. Contrast the banked +46.5 arms, whose \"",
        withdrawn="\"command in its VERDICT row. Contrast the withdrawn +46.5 arms, whose \"",
    ),
    _site(
        "coverage-docstring", "scripts/verification_coverage.py",
        "+46.5 headline, which is the least verifiable number on the page.",
        go="banked +46.5 headline, the least verifiable number on the page until Addendum Z landed.",
        withdrawn="withdrawn +46.5 headline, the least verifiable number the page ever carried.",
    ),
    _site(
        "coverage-test-not-done", "tests/test_verification_coverage.py",
        "    assert \"not done\" in text, (", "\"the outstanding fix must be named as outstanding\")",
        go=("    assert \"not done\" not in text and \"addendum z\" in text, (\n        \"Addendum Z landed: the retired phrase stays retired and the re-run is named\")"),
        withdrawn=("    assert \"not done\" not in text and \"withdrawn\" in text, (\n        \"Addendum Z withdrew the headline: the map must say so\")"),
    ),
    # ------------------------------------------------------------------ paper
    _site(
        "paper-6.10-epilogue", "paper/DRAFT.md",
        "Epilogue to the incident, for completeness (2026-08-17):", "at all three scales tested (Addendum A).",
        note="first and last sentence identical before and after the paper-currency pass",
        agrees=("Epilogue to the incident, for completeness (2026-08-17, updated {LANDED_DATE}): the gate this infrastructure ultimately served, on that CPU/fp32 path with a checkpoint/resume layer added when an external canceller repeatedly killed 13-hour runs, decided GO on its preregistered terms on 2026-08-17. Those banked arms read a mean paired delta of +46.5 micro-F1 at k=30 over seeds {1,2,3} against a +5 bar frozen before any data (receipt-level sign test 156W/0L/2T over 158 scored receipts of 180 designed; `novel_schema_summary_2026-08-12.json`), and discarded their model outputs. Addendum Z (`docs/research/ADDENDUM_Z_PROTOCOL.md`) re-ran the gate on a second CPU box with every output stored, under a protocol whose anchor was attested before any document of the run was decoded. The unchanged reader read GO at {NEW_PTS} ({SEED_DELTAS_PTS}; receipt-level sign test {WINS}W/{LOSSES}L/{TIES}T over {N_SCORED} scored of {N_DESIGNED} designed), and the gate mean {MOVE_PHRASE}, within the frozen 0.05 tolerance (Z2 AGREES). {NEW_PTS} is the number this paper quotes; +46.5 is the banked history. The largest per-arm difference between the two runs was {Z3_MAX} micro-F1, an upper bound on run-to-run plus environment spread that the design cannot separate. A re-run by the same operator is not a stranger's replication; it makes one possible. The full novel-schema study is its own artifact (ENTERPRISE_EVAL_SPEC.md Addendum B and the novel_schema_* records in experiments/), reported here only because this incident's repair, the chunked loss plus the CPU/fp32 migration, is what made it measurable. Per that spec's claim rule, the positive travels with its negative: the same adaptation recipe FAILED its preregistered gates on CORD at all three scales tested (Addendum A)."),
        moved=("Epilogue to the incident, for completeness (2026-08-17, updated {LANDED_DATE}): "
               "the gate this infrastructure ultimately served, on that CPU/fp32 path with a "
               "checkpoint/resume layer added when an external canceller repeatedly killed "
               "13-hour runs, decided GO on its preregistered terms on 2026-08-17. Those banked "
               "arms read a mean paired delta of +46.5 micro-F1 at k=30 over seeds {1,2,3} "
               "against a +5 bar frozen before any data (receipt-level sign test 156W/0L/2T over "
               "158 scored receipts of 180 designed; `novel_schema_summary_2026-08-12.json`), and "
               "discarded their model outputs. Addendum Z "
               "(`docs/research/ADDENDUM_Z_PROTOCOL.md`) re-ran the gate on a second CPU box with "
               "every output stored, under a protocol whose anchor was attested before any "
               "document of the run was decoded. The unchanged reader read GO at {NEW_PTS} "
               "({SEED_DELTAS_PTS}; receipt-level sign test {WINS}W/{LOSSES}L/{TIES}T over "
               "{N_SCORED} scored of {N_DESIGNED} designed), and the gate mean {MOVE_PHRASE}, "
               "more than the frozen 0.05 tolerance (Z2 GO BUT MOVED). {NEW_PTS} is the number "
               "this paper quotes, with that "
               "movement beside it; +46.5 is the banked history. " + P_LARGER + " The largest per-arm "
               "difference between the two runs was {Z3_MAX} micro-F1, an upper bound on run-to-run plus environment spread that the design cannot separate. A re-run by the same operator is not a stranger's replication; it makes one possible. The full novel-schema study is its own artifact (ENTERPRISE_EVAL_SPEC.md Addendum B and the novel_schema_* records in experiments/), reported here only because this incident's repair, the chunked loss plus the CPU/fp32 migration, is what made it measurable. Per that spec's claim rule, the positive travels with its negative: the same adaptation recipe FAILED its preregistered gates on CORD at all three scales tested (Addendum A)."),
        withdrawn=("Epilogue to the incident (2026-08-17, updated {LANDED_DATE}): the gate this infrastructure ultimately served, on that CPU/fp32 path with a checkpoint/resume layer added when an external canceller repeatedly killed 13-hour runs, was first banked as GO on arms that discarded their model outputs. That headline, +46.5 micro-F1, is withdrawn. Addendum Z (`docs/research/ADDENDUM_Z_PROTOCOL.md`) re-ran the gate on a second CPU box with every output stored, under a protocol whose anchor was attested before any document of the run was decoded, and the unchanged reader read {Z1_WORD} where the banked run read GO (Z2 DISAGREES). By that protocol " + P_DISAGREE
                   + ". This incident's repair, the chunked loss plus the CPU/fp32 migration, is "
                     "what made both "
                   "runs measurable. The same adaptation recipe also FAILED its preregistered gates on CORD at all three scales tested (Addendum A)."),
        nc={"start": ("travels with its negative: the same adaptation recipe FAILED its\npreregistered gates on CORD at all three scales tested (Addendum A)."),
            "text": ("travels with its negative: the same adaptation recipe FAILED its\npreregistered gates on CORD at all three scales tested (Addendum A). Addendum Z, which re-ran this gate with every output stored, landed {LANDED_DATE} NOT COMPARABLE: it scored a different set of documents (Z4 DIFFERENT EXCLUSIONS), so by its frozen protocol nothing here moves until the discrepancy is filed, and +46.5 stands unconfirmed. Its own gate word was {Z1_WORD}.")},
    ),
    _site(
        "paper-6.10-fact-scope", "paper/DRAFT.md",
        "(1) Scope. Both arms carried the full 30-demonstration prompt, so +46.5\nis adaptation added on top of the same 0.5B model's 30-shot prompt.",
        note="added by the paper-currency pass (in source HEAD 38a4f28)",
        agrees=("(1) Scope. Both arms carried the full 30-demonstration prompt, so the re-run's {NEW_PTS} (and the banked +46.5) is adaptation added on top of the same 0.5B model's 30-shot prompt."),
        moved=("(1) Scope. Both arms carried the full 30-demonstration prompt, so the re-run's {NEW_PTS} (the gate mean {MOVE_PHRASE}) and the banked +46.5 are both adaptation added on top of the same 0.5B model's 30-shot prompt."),
        withdrawn=("(1) Scope. Both arms carried the full 30-demonstration prompt, so the withdrawn +46.5 measured adaptation added on top of the same 0.5B model's 30-shot prompt."),
    ),
    _site(
        "paper-6.10-fact-verifiability", "paper/DRAFT.md",
        "Addendum Z re-runs the gate with every output stored; it has no result\nyet (section 6.12).",
        note="added by the paper-currency pass (in source HEAD 38a4f28)",
        agrees="Addendum Z re-ran the gate with every output stored and read GO at {NEW_PTS} (Z2 AGREES; section 6.12).",
        moved=("Addendum Z re-ran the gate with every output stored and read GO at {NEW_PTS}; the gate mean {MOVE_PHRASE} (Z2 GO BUT MOVED; section 6.12)."),
        withdrawn=("Addendum Z re-ran the gate with every output stored and read {Z1_WORD}; the headline is withdrawn (section 6.12)."),
        nc=("Addendum Z re-ran the gate with every output stored and landed NOT COMPARABLE: it scored a different document set (section 6.12)."),
    ),
    _site(
        "paper-6.10-fact-not-withdrawn", "paper/DRAFT.md",
        "The gate is not withdrawn. It measures a real\ndelta over a weak baseline,",
        note="added by the paper-currency pass (in source HEAD 38a4f28)",
        withdrawn=("The gate is withdrawn ({LANDED_DATE}): its re-run read {Z1_WORD}. Even before that, the delta was measured over a weak baseline,"),
        nc=("The gate is not withdrawn, and its re-run is NOT COMPARABLE (section 6.12). It measures a real delta over a weak baseline,"),
    ),
    _site(
        "paper-6.12-no-result", "paper/DRAFT.md",
        "No Z\nresult exists, and this paper quotes none. Even a completed Z\nwould be a replication by the same operator on a second box.",
        note="added by the paper-currency pass (in source HEAD 38a4f28)",
        go=("The reading landed on {LANDED_DATE}: Z1 GO, Z2 {Z2_WORD}, Z3 {Z3_WORD}, Z4 SAME EXCLUSIONS (`experiments/novel_schema_rerun_2026-09-23.json`). It is a replication by the same operator on a second box."),
        withdrawn=("The reading landed on {LANDED_DATE}: Z1 {Z1_WORD}, Z2 DISAGREES, Z3 {Z3_WORD}, Z4 SAME EXCLUSIONS, and the +46.5 headline is withdrawn (`experiments/novel_schema_rerun_2026-09-23.json`). It is a replication by the same operator on a second box."),
        nc=("The reading landed on {LANDED_DATE}: Z4 DIFFERENT EXCLUSIONS, so Z2 and Z3 are NOT COMPARABLE, and by the protocol nothing moves until the discrepancy is filed. Even a comparable Z would be a replication by the same operator on a second box."),
    ),
    # ------------------------------------------------ other public pages
    _site(
        "challenges-format-table", "CHALLENGES.md", "| 1 | +46.5 | **+46.5** |",
        go="| 1 (banked 2026-08-12 arms; history) | +46.5 | **+46.5** |",
        withdrawn="| 1 (banked arms; headline withdrawn {LANDED_DATE}) | +46.5 | **+46.5** |",
    ),
    _site(
        "demo-measured-version", "demo/ENDPOINT_DEMO.md",
        "(VERDICT.md, +46.5 mean F1 over three tenant schemas, 156W/0L/2T)\nis the measured version of this delta.",
        agrees=("(VERDICT.md: {NEW_PTS} mean F1 over three tenant schemas on the Addendum Z re-run, {WINS}W/{LOSSES}L/{TIES}T; banked +46.5 as history) is the measured version of this delta."),
        moved=("(VERDICT.md: {NEW_PTS} mean F1 over three tenant schemas on the Addendum Z re-run, {WINS}W/{LOSSES}L/{TIES}T; the gate mean {MOVE_PHRASE}; banked +46.5 as history) is the measured version of this delta."),
        withdrawn=("(VERDICT.md) was the measured version of this delta; its +46.5 headline was withdrawn on {LANDED_DATE} when its Addendum Z re-run read {Z1_WORD}."),
        nc=("(VERDICT.md, +46.5 mean F1 over three tenant schemas, 156W/0L/2T; its Addendum Z re-run is NOT COMPARABLE, so the number stands unconfirmed) is the measured version of this delta."),
    ),
]


# Sites on internal pages (strategy documents, session notes) live in a data
# file beside those pages, so the public export, which drops that directory,
# carries no reference to them. Same shape as the sites above; dated
# records listed under "banners" get the banner below, not a rewrite.
INTERNAL_SITES = REPO / "docs" / "strategy" / "Z_PUBLISH_SITES.json"


def _internal_sites(path: pathlib.Path = INTERNAL_SITES) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for site in data["sites"]:
        site = dict(site, optional=False, internal=True)
        site["end"] = EOL if site["end"] == "<end of line>" else site["end"]
        site["text"] = {o: site["text"].get(o) for o in OUTCOMES}
        out.append(site)
    for banner in data["banners"]:
        html = banner["path"].endswith(".html")
        texts = {}
        for o in OUTCOMES:
            if html:
                plain = _banner(o).replace("**", "").replace("`", "")
                texts[o] = '<section class="slide" id="z-landed"><p><strong>' + plain + "</strong></p></section>\n"
            else:
                texts[o] = "> " + _banner(o) + "\n\n"
        start = banner.get("start", "")
        out.append({"id": "banner:" + pathlib.Path(banner["path"]).name, "path": banner["path"], "start": start,
                    "end": start, "kind": banner["kind"], "optional": False, "internal": True,
                    "note": banner.get("note", "dated record: banner only"), "text": texts})
    return out


# CORRECTIONS.md first: the protocol sends a withdrawal or a discrepancy there
# before any page moves.
SITES.sort(key=lambda site: site["id"] != "corrections-row")
ALL_SITES = SITES + _internal_sites()
INTERNAL_PATHS = {site["path"] for site in ALL_SITES if site.get("internal")}


# -- the non-flattering rule, applied to the templates -----------------------


def _units(text: str) -> list[str]:
    """Sentences: table rows are their own unit, prose is joined then split."""
    units, para = [], []

    def flush():
        if para:
            units.extend(re.split(r"(?<=[.!?])\s+", " ".join(para)))
            para.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("|"):
            flush()
            units.extend(re.split(r"(?<=[.!?])\s+", stripped))
        else:
            para.append(stripped)
    flush()
    return [u for u in units if u]


def template_problems(site_id: str, outcome: str, template: str) -> list[str]:
    """What the frozen citation rule forbids, checked on the unfilled text."""
    problems = []
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", template))
    low = flat.lower()
    for phrase in STALE:
        if phrase in low:
            problems.append(f"{site_id}/{outcome}: stale pending phrase {phrase!r}")
    match = FLATTERING.search(flat)
    if match:
        problems.append(f"{site_id}/{outcome}: flattering word {match.group(0)!r}")
    names = set(PLACEHOLDER.findall(template))
    if outcome == "WITHDRAWN":
        for name in ("NEW_PTS", "MOVE_PHRASE"):
            if name in names:
                problems.append(f"{site_id}/{outcome}: a withdrawn headline cannot quote {{{name}}}")
        for unit in _units(template):
            if BANKED_PTS in unit and "withdraw" not in unit.lower():
                problems.append(f"{site_id}/{outcome}: +46.5 without 'withdrawn' in: {unit[:120]!r}")
    elif outcome in ("AGREES", "MOVED"):
        for unit in _units(template):
            if BANKED_PTS in unit and not ("banked" in unit.lower() or "history" in unit.lower()):
                problems.append(f"{site_id}/{outcome}: +46.5 not marked banked/history in: {unit[:120]!r}")
        if outcome == "MOVED" and "NEW_PTS" in names and not ("MOVE_PHRASE" in names or "GO BUT MOVED" in flat):
            problems.append(f"{site_id}/{outcome}: the new number without its movement beside it")
    elif outcome == "NOT_COMPARABLE":
        for name in ("NEW_PTS", "MOVE_PHRASE", "MEAN_DELTA"):
            if name in names:
                problems.append(f"{site_id}/{outcome}: nothing moves, so no {{{name}}}")
        if BANKED_PTS in flat and not ("NOT COMPARABLE" in flat or "unconfirmed" in low):
            problems.append(f"{site_id}/{outcome}: +46.5 without NOT COMPARABLE/unconfirmed")
    return problems


def all_template_problems() -> list[str]:
    problems = []
    for site in ALL_SITES:
        for outcome in OUTCOMES:
            text = _text_of(site, outcome)
            if text is not None:
                problems += template_problems(site["id"], outcome, text)
    return problems


def _text_of(site: dict, outcome: str) -> str | None:
    value = site["text"][outcome]
    return value["text"] if isinstance(value, dict) else value


def _anchors_of(site: dict, outcome: str) -> tuple[str, str]:
    value = site["text"][outcome]
    if isinstance(value, dict):
        return value["start"], value.get("end", value["start"])
    return site["start"], site["end"]


# -- checks before anything is printed ----------------------------------------


def pts(x: float) -> str:
    """F1 points, one decimal, signed: tests/test_addendum_z_citation.py's headline_string."""
    return f"{x * 100:+.1f}"


def _load(path: pathlib.Path, what: str) -> dict:
    if not path.exists():
        raise Refused(f"{what} does not exist: {path}")
    return json.loads(path.read_text())


def check_identity(reading: dict) -> None:
    # the identity novel_schema_rerun._six demands of every Z arm
    for key, want in (("protocol", z.PROTOCOL), ("spec", z.SPEC), ("date", z.DATE)):
        if reading.get(key) != want:
            raise Refused(f"reading {key} is {reading.get(key)!r}, not the runner's {want!r}")


def check_protocol_anchor() -> str:
    try:
        from decoder_isolation_read import anchored_digest  # the repo's one .ots reader
    except Exception as error:  # noqa: BLE001 - fail closed
        raise Refused(f"cannot load the proof reader: {error}") from error
    proof = PROTOCOL_PATH.with_name(f"{PROTOCOL_PATH.name}.{PROTOCOL_STAMP}.ots")
    if not proof.exists():
        raise Refused(f"protocol proof missing: {proof.name}")
    digest = anchored_digest(proof)
    if digest is None:
        raise Refused(f"{proof.name} is not an attested sha256 file proof")
    current = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(current).hexdigest() == digest:
        return digest
    # the spec's rule (tests/test_spec_matches_anchor.py): changed only by
    # appending after a committed snapshot that the proof commits to
    snap = PROTOCOL_PATH.with_name(f"snapshots_{PROTOCOL_PATH.stem}_{PROTOCOL_STAMP}.md")
    if snap.exists() and hashlib.sha256(snap.read_bytes()).hexdigest() == digest \
            and current.startswith(snap.read_bytes()):
        return digest
    raise Refused("ADDENDUM_Z_PROTOCOL.md no longer hashes to what its proof commits to")


def check_code_pins(pins: dict | None = None) -> list[str]:
    pins = runlog.code_pins() if pins is None else pins
    if set(pins) != set(runlog.PINNED):
        raise Refused("code pins do not cover the run log's PINNED files")
    known = []
    for rel, pin in pins.items():
        if pin["unchanged_since_freeze"]:
            continue
        if KNOWN_CHANGES.get(rel) == (pin["at_freeze_commit"], pin["now"]):
            known.append(rel)
            continue
        raise Refused(f"{rel} changed since the freeze commit {runlog.FREEZE_COMMIT} "
                      f"({str(pin['at_freeze_commit'])[:12]} -> {str(pin['now'])[:12]})")
    return known


def check_run_log(directory: pathlib.Path) -> None:
    if not runlog.OUT.exists():
        raise Refused(f"{runlog.OUT.name} is missing: a redraw would be undetectable")
    problems = runlog.verify(directory)
    if problems:
        raise Refused("run log: " + "; ".join(problems))


def check_primary(directory: pathlib.Path, banked_dir: pathlib.Path) -> None:
    """The stranger's check, run as a stranger would: it must pass on the landed arms."""
    import contextlib
    import io

    try:
        import verify_z_from_primary  # stdlib; re-derives scores, gate and Z1 to Z4 from outputs
    except Exception as error:  # noqa: BLE001 - fail closed
        raise Refused(f"cannot load verify_z_from_primary.py: {error}") from error
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        code = verify_z_from_primary.main(["--dir", str(directory), "--banked-dir", str(banked_dir)])
    if code != 0:
        tail = " | ".join(line for line in captured.getvalue().splitlines()[-6:] if line.strip())
        raise Refused(f"verify_z_from_primary.py exited {code}: {tail}")


def check_consistency(reading: dict, new_summary: dict, banked_summary: dict) -> str:
    """The reading against the summaries it was computed from; returns the outcome."""
    z1, z2, z3, z4 = (reading[k] for k in ("Z1_gate", "Z2_agreement", "Z3_spread", "Z4_attrition"))
    head = reading["headline"]
    word = z1["reading"]
    if word not in ("GO", "PIVOT", "UNINFORMATIVE"):
        raise Refused(f"Z1 reads {word!r}; the reader writes nothing on UNDECIDABLE")
    if new_summary.get("VERDICT") != word or new_summary.get("gate_k30") != z1["gate_k30"] \
            or new_summary.get("verdict_detail") != z1["detail"] or new_summary.get("date") != z.DATE:
        raise Refused(f"Z1 does not match {NEW_SUMMARY_NAME}")
    if banked_summary.get("VERDICT") != "GO" or banked_summary["gate_k30"].get("mean_delta") != BANKED_MEAN:
        raise Refused(f"{BANKED_SUMMARY_NAME} is not the banked GO at {BANKED_MEAN}")
    if head.get("banked") != BANKED_MEAN or pts(head["banked"]) != BANKED_PTS:
        raise Refused("the reading's banked headline is not +0.4648")
    if head.get("primary_verifiable") != z1["gate_k30"].get("mean_delta"):
        raise Refused("headline.primary_verifiable is not the re-run's gate mean")
    if z4["reading"] not in ("SAME EXCLUSIONS", "DIFFERENT EXCLUSIONS"):
        raise Refused(f"Z4 reads {z4['reading']!r}")
    same = all(row["same"] for row in z4["per_arm"])
    if same != (z4["reading"] == "SAME EXCLUSIONS"):
        raise Refused("Z4's word does not match its own per-arm rows")
    comparable = z4["reading"] == "SAME EXCLUSIONS"
    expected_quote = z1["gate_k30"].get("mean_delta") if (word == "GO" and comparable) else None
    if head.get("quote") != expected_quote:
        raise Refused("headline.quote does not follow the frozen rule")
    if not comparable:
        if z2.get("reading") != "NOT COMPARABLE" or z3.get("reading") != "NOT COMPARABLE":
            raise Refused("DIFFERENT EXCLUSIONS but Z2/Z3 are not NOT COMPARABLE")
        return "NOT_COMPARABLE"
    rederived = z.agreement_reading(new_summary, banked_summary)
    if rederived != z2:
        raise Refused("Z2 does not re-derive from the two summaries by novel_schema_rerun.agreement_reading")
    wants = "WITHIN PREDICTED SPREAD" if z3["max_abs_diff"] <= z3["predicted_max"] else "SPREAD LARGER THAN PREDICTED"
    if z3["reading"] != wants or z3["replaces_b93"] != (z3["max_abs_diff"] > z3["b93_spread"]):
        raise Refused("Z3's words do not match its own numbers")
    if z3["max_abs_diff"] != max(row["abs_diff"] for row in z3["per_arm"]):
        raise Refused("Z3's maximum is not the largest per-arm difference")
    pairs = {("GO", "AGREES"): "AGREES", ("GO", "GO BUT MOVED"): "MOVED",
             ("PIVOT", "DISAGREES"): "WITHDRAWN", ("UNINFORMATIVE", "DISAGREES"): "WITHDRAWN"}
    outcome = pairs.get((word, z2["reading"]))
    if outcome is None:
        raise Refused(f"Z1 {word} with Z2 {z2['reading']} is not a combination the runner emits")
    return outcome


# -- the values ---------------------------------------------------------------


def values(reading: dict, new_summary: dict) -> dict:
    """Every placeholder's value, or None where the reading does not license it."""
    z1, z2, z3, z4 = (reading[k] for k in ("Z1_gate", "Z2_agreement", "Z3_spread", "Z4_attrition"))
    gate = z1["gate_k30"]
    rl = gate.get("receipt_level") or {}
    st = rl.get("sign_test") or {}
    head = reading["headline"]
    arms = reading["arms"]
    v: dict = {name: None for name in (
        "NEW_PTS", "MOVE_PHRASE", "Z3_MAX", "Z3_MAX_AT", "Z3_B93_CLAUSE", "P_STR", "CI_LO", "CI_HI",
        "RERUN_MEAN_SENTENCE")}
    v["LANDED_DATE"] = new_summary.get("decided")
    v["Z1_WORD"] = z1["reading"]
    v["Z1_DETAIL"] = z1["detail"]
    v["Z2_WORD"] = z2["reading"]
    v["Z2_DETAIL"] = z2.get("detail")
    v["Z3_WORD"] = z3["reading"]
    v["Z4_WORD"] = z4["reading"]
    mean = head.get("primary_verifiable")
    v["MEAN_DELTA"] = f"{mean:+.4f}" if mean is not None else None
    if head.get("quote") is not None:
        v["NEW_PTS"] = pts(head["quote"])
    if "moved_by" in z2:
        new, old = float(z2["new_mean_delta"]), float(z2["banked_mean_delta"])
        direction = " up" if new > old else (" down" if new < old else "")
        v["MOVE_PHRASE"] = (f"moved {z2['moved_by']:.4f} ({z2['moved_by'] * 100:.1f} points){direction} "
                            f"from the banked {old:+.4f}")
    if "max_abs_diff" in z3:
        v["Z3_MAX"] = f"{z3['max_abs_diff']:.4f}"
        v["Z3_MAX_AT"] = f"seed {z3['max_at']['seed']} {z3['max_at']['arm']}"
        clause = z3["detail"].rsplit(": ", 1)[-1]
        if clause not in B93_CLAUSES:
            raise Refused(f"Z3's B.9.3 clause is not one the runner writes: {clause!r}")
        v["Z3_B93_CLAUSE"] = clause
    deltas = gate.get("seed_deltas")
    v["SEED_DELTAS_PTS"] = "/".join(pts(d) for d in deltas) if deltas else None
    if rl.get("ci95"):
        v["CI_LO"], v["CI_HI"] = (f"{x * 100:.1f}" for x in rl["ci95"])
    v["WINS"], v["LOSSES"], v["TIES"] = st.get("wins"), st.get("losses"), st.get("ties")
    v["N_SCORED"] = rl.get("n")
    v["N_DESIGNED"] = sum(row["eval_n"] for row in gate.get("attrition", [])) or None
    p = st.get("p_value")
    if p is not None:  # the reader's own formatting (novel_schema_summary.verdict, B.9.4)
        v["P_STR"] = "p<1e-15" if p < 1e-15 else f"p={p:.4g}"
    kshot = [arms[z.artifact_name(s, "kshot")]["mean_micro_f1"] for s in z.SEEDS]
    v["KSHOT_RANGE"] = f"{min(kshot):.4f}–{max(kshot):.4f}"
    v["Z4_TABLE"] = "; ".join(
        f"seed {row['seed']} {row['arm']} {len(row['over_cap_new'])} re-run, {len(row['over_cap_banked'])} banked"
        + ("" if row["same"] else " (different documents)") for row in z4["per_arm"])
    v["EMPTY_POOL_TOTAL"] = sum(len(row["empty_pool_new"]) for row in z4["per_arm"])
    v["N_OVER_CAP"] = sum(len(row["over_cap_new"]) for row in z4["per_arm"] if row["arm"] == "adapted")
    v["RESTARTS"] = "; ".join(
        f"seed {s} {a} {arms[z.artifact_name(s, a)].get('restarts', 'not recorded')}" for s, a in z.ARM_ORDER)
    pre = reading["preregistered"]

    def observed(item):
        o = item["observed"]
        return "not comparable" if o is None else (f"{o:.4f}" if isinstance(o, float) else str(o))

    v["PRED_LINE"] = "; ".join(
        f"{k} predicted {pre[k]['predicted']}, observed {observed(pre[k])} "
        f"({'held' if pre[k]['holds'] else 'did not hold'})" for k in ("Z1", "Z2", "Z3", "Z4"))
    if z1["reading"] == "PIVOT" and mean is not None:
        v["RERUN_MEAN_SENTENCE"] = f"The re-run's gate mean is {pts(mean)} micro-F1 points ({mean:+.4f})."
    elif z1["reading"] == "UNINFORMATIVE":
        v["RERUN_MEAN_SENTENCE"] = ("The re-run's gate mean is not interpreted: a validity gate tripped (spec B.5).")
    return v


def fill(template: str, vals: dict, where: str) -> str:
    def sub(match):
        name = match.group(1)
        if name not in vals:
            raise Refused(f"{where}: unknown placeholder {{{name}}}")
        if vals[name] is None:
            raise Refused(f"{where}: {{{name}}} has no value in this reading")
        return str(vals[name])

    out = PLACEHOLDER.sub(sub, template)
    left = PLACEHOLDER.findall(out)
    if left:
        raise Refused(f"{where}: placeholder left unfilled: {left}")
    return out


# -- locating and printing ----------------------------------------------------

# Where pages are read from. The landing-day run reads the live tree. The CI
# tests switch FIXTURE_MODE on and read frozen excerpts instead, so an
# unrelated prose edit cannot break them: public pages under
# tests/fixtures/z_publish/, internal pages beside the internal site file
# (the public export drops that directory, so no internal text is exported).
# `--check-anchors` is what tells the operator that the live pages still
# hold every anchor.
FIXTURE_MODE = False
FIXTURES = REPO / "tests" / "fixtures" / "z_publish"
INTERNAL_FIXTURES = INTERNAL_SITES.parent / "z_publish_fixtures"
FIXTURE_SUFFIX = ".fixture"


def page_path(rel: str, fixtures: bool | None = None) -> pathlib.Path:
    if FIXTURE_MODE if fixtures is None else fixtures:
        base = INTERNAL_FIXTURES if rel in INTERNAL_PATHS else FIXTURES
        return base / (rel + FIXTURE_SUFFIX)
    return REPO / rel


def _pattern(anchor: str) -> re.Pattern:
    """The anchor with any run of inner whitespace matching any run, so a re-wrapped
    paragraph still matches; leading and trailing whitespace stay literal."""
    body = anchor.strip()
    lead = anchor[: len(anchor) - len(anchor.lstrip())]
    tail = anchor[len(anchor.rstrip()):]
    inner = r"\s+".join(re.escape(token) for token in body.split())
    return re.compile(re.escape(lead) + inner + re.escape(tail))


def find_all(text: str, anchor: str) -> list[re.Match]:
    return list(_pattern(anchor).finditer(text))


def locate(text: str, kind: str, start: str, end: str) -> tuple[int, int] | None:
    if kind == "prepend":
        return 0, 0
    if kind == "append":
        return len(text), len(text)
    found = find_all(text, start)
    if len(found) != 1:
        return None  # missing, or ambiguous: either way a person places it
    i, j0 = found[0].span()
    if kind == "insert_after":
        return j0, j0
    if kind == "insert_before":
        return i, i
    if end == EOL:
        j = text.find("\n", i)
        return i, (len(text) if j < 0 else j)
    tail = _pattern(end).search(text, i)
    return (i, tail.end()) if tail else None


def render(outcome: str, vals: dict, only: str | None = None) -> list[dict]:
    """Every site's located span and filled text; raises Refused before returning anything partial."""
    out = []
    for site in ALL_SITES:
        if only and site["id"] != only:
            continue
        template = _text_of(site, outcome)
        start, end = _anchors_of(site, outcome)
        path = page_path(site["path"])
        page = path.read_text(encoding="utf-8") if path.exists() else ""
        span = locate(page, site["kind"], start, end) if page else None
        entry = {"site": site, "path": site["path"], "kind": site["kind"], "span": span,
                 "current": page[span[0]:span[1]] if span else None,
                 "line": page[:span[0]].count("\n") + 1 if span else None,
                 "new": None if template is None else fill(template, vals, site["id"])}
        out.append(entry)
    return out


def apply(page: str, entry: dict) -> str:
    """What the page would read after one printed change (used by the tests; never on disk)."""
    i, j = entry["span"]
    return page[:i] + entry["new"] + page[j:]


def print_report(outcome: str, entries: list[dict], header: list[str]) -> int:
    drifted = 0
    print("\n".join(header))
    for e in entries:
        site = e["site"]
        tag = f"=== [{site['id']}] {e['path']}"
        if e["new"] is None:
            print(f"{tag}: unchanged for {outcome}")
            continue
        if e["span"] is None:
            drifted += 1
            print(f"{tag}: ANCHOR NOT FOUND (the page changed after this was staged). Place this text by hand:")
            print(e["new"])
            continue
        print(f"{tag}:{e['line']}  {e['kind']}" + (f"  ({site['note']})" if site["note"] else ""))
        if e["kind"] == "replace":
            print("--- current")
            print(e["current"])
            print("+++ new")
        print(e["new"])
    return 3 if drifted else 0


# -- the operator's anchor check (not a CI test) --------------------------------

# Tracked files that carry +46.5 and are deliberately not rewritten on landing.
LEAVE_ALONE = {
    "CORRECTIONS.md": "the ledger keeps its history; the landing row is a site",
    "docs/research/ENTERPRISE_EVAL_SPEC.md": "frozen spec; erratum P16 names each line",
    "docs/research/ADDENDUM_Z_PROTOCOL.md": "the frozen protocol",
    "scripts/novel_schema_rerun.py": "the pinned runner's docstring",
    "scripts/verify_z_from_primary.py": "the stranger's verifier's docstring",
    "scripts/run_addendum_h_cell.py": "Addendum H's runner docstring",
    "scripts/publish_z_reading.py": "this tool",
    "tests/test_publish_z_reading.py": "this tool's tests",
    "tests/test_addendum_z_citation.py": "the citation rule's own test",
    "tests/test_verification_coverage.py": "docstring and the map's '46.5' assertion; line 69 is a site",
}
LEAVE_ALONE_PREFIXES = ("docs/research/snapshots_",)
LEAVE_ALONE_LINES = (("paper/DRAFT.md", "its number replaces +46.5"),)  # states the rule itself
SCAN_SKIP = ("experiments/", "work/", "figures/", "tests/fixtures/",
             "docs/strategy/field_evidence_2026-10-21/")
SCAN_SUFFIXES = (".md", ".py", ".html", ".txt", ".sh")
HIT = re.compile(r"(?<![\d.])46\.5(?!\d)")


def _tracked(repo: pathlib.Path) -> list[str]:
    import subprocess

    try:
        listed = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True,
                                check=True).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        listed = [str(path.relative_to(repo)) for path in repo.rglob("*") if path.is_file()]
    return sorted(rel for rel in listed if rel.endswith(SCAN_SUFFIXES) and not rel.startswith(SCAN_SKIP))


def check_anchors(scan: list[str] | None = None) -> list[str]:
    """Every anchor found exactly once; no rewrite inside a dated section; every +46.5
    on a tracked page either rewritten, bannered, dated, or deliberately left alone."""
    from sync_test_counts import dated_spans  # the syncers' own definition of a dated record

    problems: list[str] = []
    covered: dict[str, list[tuple[int, int]]] = {}
    for site in ALL_SITES:
        path = page_path(site["path"])
        if not path.exists():
            problems.append(f"[{site['id']}] {site['path']}: page missing")
            continue
        text = path.read_text(encoding="utf-8")
        dated = dated_spans(text)
        seen = set()
        for outcome in OUTCOMES:
            if _text_of(site, outcome) is None:
                continue
            start, end = _anchors_of(site, outcome)
            if (start, end) in seen:
                continue
            seen.add((start, end))
            span = locate(text, site["kind"], start, end)
            if span is None:
                count = len(find_all(text, start))
                why = "not found" if count == 0 else f"found {count} times"
                problems.append(f"[{site['id']}] {site['path']} ({outcome}): anchor {why}: {start[:70]!r}")
                continue
            if site["kind"] == "replace":
                covered.setdefault(site["path"], []).append(span)
                if any(a <= span[0] < b for a, b in dated):
                    problems.append(f"[{site['id']}] {site['path']}: rewrites text inside a dated section")
    banners = {site["path"] for site in ALL_SITES if site["kind"] in ("prepend", "insert_before")}
    for rel in ((_tracked(REPO) if not FIXTURE_MODE else []) if scan is None else scan):
        if rel in LEAVE_ALONE or rel.startswith(LEAVE_ALONE_PREFIXES) or rel in banners:
            continue
        path = page_path(rel)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        dated = dated_spans(text)
        for match in HIT.finditer(text):
            at = match.start()
            line_start = text.rfind("\n", 0, at) + 1
            line = text[line_start:text.find("\n", at) if "\n" in text[at:] else len(text)]
            if any(a <= at < b for a, b in covered.get(rel, [])):
                continue
            if any(a <= at < b for a, b in dated):
                continue
            if any(rel == p and needle in line for p, needle in LEAVE_ALONE_LINES):
                continue
            problems.append(f"{rel}:{text[:at].count(chr(10)) + 1}: +46.5 that no site rewrites: {line.strip()[:90]!r}")
    return problems


def freeze_fixtures() -> list[pathlib.Path]:
    """Write the excerpts the CI tests run on: every line an anchor touches, nothing else."""
    written = []
    for rel in sorted({site["path"] for site in ALL_SITES}):
        text = page_path(rel, fixtures=False).read_text(encoding="utf-8")
        lines = text.split("\n")
        ranges = []
        for site in (s for s in ALL_SITES if s["path"] == rel):
            for outcome in OUTCOMES:
                if _text_of(site, outcome) is None:
                    continue
                start, end = _anchors_of(site, outcome)
                if site["kind"] in ("prepend", "append"):
                    at = 0 if site["kind"] == "prepend" else len(lines) - 1
                    ranges.append((at, at))
                    continue
                matches = find_all(text, start)
                if len(matches) != 1:
                    raise SystemExit(f"[{site['id']}] {rel}: anchor not found once; fix it before freezing")
                span = locate(text, "replace" if site["kind"] == "replace" else "insert_before", start, end)
                a = text[:matches[0].start()].count("\n")
                b = text[:max(span[1], matches[0].end()) - 1].count("\n") if span else a
                ranges.append((a, b))
        ranges.sort()
        merged = []
        for a, b in ranges:
            if merged and a <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        excerpt = "\n[...]\n".join("\n".join(lines[a:b + 1]) for a, b in merged)
        out = page_path(rel, fixtures=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(excerpt + ("\n" if not excerpt.endswith("\n") else ""), encoding="utf-8")
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(REPO / "experiments"),
                    help="where the six Z arms, their summary and the reading live")
    ap.add_argument("--banked-dir", default=str(REPO / "experiments"),
                    help="where the 2026-08-12 arms and summary live")
    ap.add_argument("--site", help="print one site only")
    ap.add_argument("--list", action="store_true", help="list the sites and stop")
    ap.add_argument("--templates", choices=OUTCOMES, help="print the staged text for one outcome, unfilled")
    ap.add_argument("--check-anchors", action="store_true",
                    help="operator check, not a CI test: every anchor on the live pages, exit 1 on any miss")
    ap.add_argument("--freeze-fixtures", action="store_true",
                    help="rewrite the page excerpts the CI tests run on, from the live pages")
    args = ap.parse_args(argv)

    problems = all_template_problems()
    if problems:
        print("REFUSED: a template breaks the frozen citation rule:", file=sys.stderr)
        print("\n".join("  " + p for p in problems), file=sys.stderr)
        return 2
    if args.list:
        for site in ALL_SITES:
            print(f"{site['id']:34} {site['path']}")
        return 0
    if args.check_anchors:
        missing = check_anchors()
        for line in missing:
            print(line)
        print(f"{len(missing)} problem(s); {len(ALL_SITES)} sites checked against the live pages"
              if missing else f"every anchor of {len(ALL_SITES)} sites found once on the live pages; "
              "every +46.5 is rewritten, bannered, dated or deliberately left alone")
        return 1 if missing else 0
    if args.freeze_fixtures:
        for path in freeze_fixtures():
            print(path)
        return 0
    if args.templates:
        for site in ALL_SITES:
            if args.site and site["id"] != args.site:
                continue
            text = _text_of(site, args.templates)
            print(f"=== [{site['id']}] {site['path']} ({site['kind']})")
            print("unchanged" if text is None else text)
        return 0

    directory, banked_dir = pathlib.Path(args.dir), pathlib.Path(args.banked_dir)
    try:
        # the reading first: until it exists nothing else is looked at
        reading = _load(directory / READING_NAME, "the reading")
        new_summary = _load(directory / NEW_SUMMARY_NAME, "the re-run's summary")
        banked_summary = _load(banked_dir / BANKED_SUMMARY_NAME, "the banked summary")
        check_identity(reading)
        digest = check_protocol_anchor()
        known = check_code_pins()
        check_run_log(directory)
        check_primary(directory, banked_dir)
        outcome = check_consistency(reading, new_summary, banked_summary)
        vals = values(reading, new_summary)
        entries = render(outcome, vals, only=args.site)
    except Refused as why:
        print(f"REFUSED: {why}. Nothing is printed for the pages.", file=sys.stderr)
        return 2
    header = [
        "Addendum Z publication text. Printed, not written: apply by hand, in the order below.",
        f"reading: {directory / READING_NAME}",
        f"outcome: {outcome} (Z1 {vals['Z1_WORD']}, Z2 {vals['Z2_WORD']}, Z3 {vals['Z3_WORD']}, Z4 {vals['Z4_WORD']})",
        f"checks: identity ok; protocol bytes match proof {PROTOCOL_STAMP} ({digest[:12]}); code pins unchanged "
        f"since {runlog.FREEZE_COMMIT}" + (f" except the known {', '.join(known)}" if known else "")
        + "; run log matches; verify_z_from_primary.py passes; summaries agree"
        + ("; Z2 re-derived" if outcome != "NOT_COMPARABLE" else ""),
        ("after applying: rerun tests/test_addendum_z_citation.py, regenerate experiments/verification_coverage.json, and run scripts/sync_test_counts.py"),
        "",
    ]
    return print_report(outcome, entries, header)


if __name__ == "__main__":
    raise SystemExit(main())
