"""Addendum K's mechanism. Pin the behaviour, and pin the safety property.

Grounding is the only component in this repository that REWRITES a
model's answer, so the property that matters most is not how much it
recovers -- it is that it never damages an answer that was already right.
Every bar in this file was set on the 20 TRAIN documents; the 30 holdout
documents are scored once, by ``scripts/addendum_k_grounding.py``, and
nothing here is allowed to consult them.

Five build errors are pinned here because each was found by the train
split rather than by review, and each would have shipped:

1. Grounding hunted for a closer-looking span and moved CORRECT answers
   off the ones they were already on (-0.13 micro-F1 per document).
2. After a contiguous-span guard, -0.01 remained, entirely from
   documents whose layout is broken -- a value can be built from document
   words without occurring in the document as one span.
3. Dropping punctuation from the compare-form made ``9761.5`` and
   ``976.15`` identical, so a misplaced decimal read as already grounded.
4. `candidate_spans` offers a span raw and punctuation-trimmed; they tie
   in compare-form, so the winner was arbitrary and grounding returned
   ``Fresno.`` and ``"Bismarck",``, which the pinned scorer counts wrong.
5. `_prefer_model_characters` was gated on equal whole-string length, so
   one extra word switched it off for every other word in the value.
"""

from __future__ import annotations

import json
import pathlib

from arcttt.grounding import (
    candidate_spans,
    dehyphenate,
    ground,
    infer_copy_fields,
    is_document_grounded,
    repair_ocr,
    repair_ocr_token,
    repair_words_from_document,
    snap_to_document,
)
from arcttt.scoring import field_micro_f1

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "experiments" / "blind_rehearsal_2026-08-20_raw"


def _train() -> list[dict]:
    return [json.loads(line)
            for line in (RAW / "train.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------- OCR repair

def test_repair_reads_digits_as_letters_inside_words() -> None:
    assert repair_ocr_token("Va11ey") == "Valley"
    assert repair_ocr_token("R0nde") == "Ronde"
    assert repair_ocr_token("Keyst0ne") == "Keystone"


def test_repair_reads_letters_as_digits_inside_numbers() -> None:
    assert repair_ocr_token("12,1OO") == "12,100"
    assert repair_ocr_token("944.1O") == "944.10"


def test_case_comes_from_the_token_not_from_a_table() -> None:
    """`1` is `I` in an upper-case word and `l` in a lower-case one."""
    assert repair_ocr_token("STRA1GHT") == "STRAIGHT"
    assert repair_ocr_token("Va11ey") == "Valley"


def test_an_even_split_is_left_alone_without_context() -> None:
    """`C0.` is one letter and one digit; alone it cannot be classified."""
    assert repair_ocr_token("C0.") == "C0."
    assert repair_ocr_token("C0.", tie_break="word") == "Co."
    # In a sentence the surrounding text supplies the class.
    assert repair_ocr("Grande R0nde Lumber C0.") == "Grande Ronde Lumber Co."


def test_a_token_with_no_mixing_is_untouched() -> None:
    assert repair_ocr_token("Valley") == "Valley"
    assert repair_ocr_token("12100") == "12100"


def test_dehyphenate_rejoins_a_word_broken_across_lines() -> None:
    assert dehyphenate("Pinnacle Elec-\ntronics") == "Pinnacle Electronics"


# ------------------------------------------------------------------- spans

def test_candidate_spans_offers_the_bare_number() -> None:
    """The document writes `$1,342.75`; the tenant's gold keeps neither
    the symbol nor the comma, so the bare form must be a candidate."""
    spans = candidate_spans("Total Charges: $1,342.75\nWeight: 19,800 lbs")
    assert "1,342.75" in spans
    assert "19,800" in spans


def test_candidate_spans_splits_a_line_holding_two_pairs() -> None:
    spans = candidate_spans("Origin: Erie        Destination: Youngstown")
    assert "Erie" in spans and "Youngstown" in spans


# -------------------------------------------------- the safety property

def test_a_value_the_model_got_right_is_never_touched() -> None:
    document = "Shipper: Harlan Fastener Works\nOrigin: Toledo"
    value, snapped = snap_to_document("Harlan Fastener Works", document)
    assert (value, snapped) == ("Harlan Fastener Works", False)


def test_a_value_split_across_a_misaligned_scan_is_still_grounded() -> None:
    """`h-3308`: the consignee's name is printed across two rows.

    No contiguous span contains it, but every word of it is a document
    word, so the model has not left the document and grounding must not
    truncate the name to the longest span it can find.
    """
    document = ("Consignee   Rustbelt Salvage &\n"
                "Shipper     Reclaim LLC\n"
                "            Erie Forge Holdings Inc.\n")
    assert is_document_grounded("Rustbelt Salvage & Reclaim LLC", document)
    value, snapped = snap_to_document("Rustbelt Salvage & Reclaim LLC",
                                      document)
    assert (value, snapped) == ("Rustbelt Salvage & Reclaim LLC", False)


def test_grounding_never_lowers_a_perfect_prediction_on_train() -> None:
    """The bar that matters. Feed gold in, gold must come out.

    This is measured with the PINNED SCORER rather than string equality,
    because "did not change the score" is the claim, not "did not change
    the bytes".
    """
    train = _train()
    pairs = [(row["text"], row["gold"]) for row in train]
    copy_fields = infer_copy_fields(pairs)
    for document, gold in pairs:
        grounded, _ = ground(dict(gold), document, copy_fields)
        assert field_micro_f1(grounded, gold) == field_micro_f1(dict(gold),
                                                                gold)


# ------------------------------------------------------------- what it fixes

def test_confabulation_is_snapped_back_to_the_document() -> None:
    document = "Shipper: Calloway Textile Mills\nOrigin: Macon"
    value, snapped = snap_to_document("Allowce Textile Mills", document)
    assert value == "Calloway Textile Mills" and snapped


def test_an_unasked_expansion_is_snapped_back() -> None:
    document = "Consignee: PacCoast Surgical Distribution\nOrigin: Tacoma"
    value, snapped = snap_to_document("Pacific Coast Surgical Distribution",
                                      document)
    assert value == "PacCoast Surgical Distribution" and snapped


def test_the_model_settles_punctuation_the_document_only_decorates() -> None:
    """`Fresno.` and `Fresno` tie in compare-form. The model wrote one."""
    document = "Destination: Fresno.\nClass: 60"
    value, _ = snap_to_document("Fersno", document)
    assert value == "Fresno"


def test_the_model_settles_characters_the_scanner_made_ambiguous() -> None:
    """`1` is `l` in `Valley` and `i` in `Philadelphia`; only the model,
    which has read English, can tell those apart -- and it must be able
    to even when it has also added a word."""
    document = "Shipper: Ph1lade1phia Casters & Whee1s\nClass: 65"
    value, _ = snap_to_document("Pacific Philadelphia Casters & Wheels",
                                document)
    assert value == "Philadelphia Casters & Wheels"


def test_a_snap_never_discards_a_word_the_document_contains() -> None:
    """The truncation guard, and the word-level fallback behind it.

    `Recalim` is not a document word and is repaired; `Rustbelt`,
    `Salvage` and `LLC` are, and survive -- even though the longest
    contiguous span in this broken layout contains none of the last two.
    """
    document = ("Consignee   Rustbelt Salvage &\n"
                "Shipper     Reclaim LLC\n")
    value, changed = snap_to_document("Rustbelt Salvage & Recalim LLC",
                                      document)
    assert value == "Rustbelt Salvage & Reclaim LLC" and changed


def test_word_repair_leaves_a_word_with_no_plausible_source_alone() -> None:
    document = "Consignee: Reclaim LLC\n"
    value, changed = repair_words_from_document("Zzzyx", document)
    assert (value, changed) == ("Zzzyx", False)


def test_a_snap_that_has_to_reach_is_refused() -> None:
    """Below threshold the model's answer is kept: rewriting a correct
    answer is worse than leaving a wrong one."""
    document = "Origin: Toledo\nDestination: Fort Wayne\n"
    value, snapped = snap_to_document("Zzzyxwvut", document)
    assert (value, snapped) == ("Zzzyxwvut", False)


def test_a_misplaced_decimal_is_not_read_as_already_grounded() -> None:
    """`9761.5` and `976.15` differ only in where the point sits, which is
    the most expensive error available on an invoice."""
    document = "Total Charges: $976.15\n"
    assert not is_document_grounded("9761.5", document)
    value, snapped = snap_to_document("9761.5", document)
    assert value == "976.15" and snapped


# --------------------------------------------------------- copy-field infer

def test_copy_fields_are_measured_from_the_tenants_own_train_pairs() -> None:
    """Transform-type fields must NOT be snapped: this corpus reformats
    dates, and snapping `2026-07-04` back to `07-04-26` would undo it."""
    copy_fields = infer_copy_fields(
        [(row["text"], row["gold"]) for row in _train()])
    assert "ship_date" not in copy_fields
    for key in ("shipper_name", "consignee_name", "origin_city",
                "destination_city"):
        assert key in copy_fields


def test_ground_reports_what_it_changed() -> None:
    document = "Shipper: Calloway Textile Mills\n"
    grounded, changes = ground({"shipper_name": "Allowce Textile Mills"},
                               document, ["shipper_name"])
    assert grounded["shipper_name"] == "Calloway Textile Mills"
    assert "shipper_name" in changes and "snapped" in changes["shipper_name"]


def test_ground_passes_nulls_through() -> None:
    grounded, changes = ground({"freight_class": None}, "Class: 60", [])
    assert grounded == {"freight_class": None} and changes == {}


# ------------------------------------------------ found on the HOLDOUT
#
# These three defects were found by inspecting what grounding damaged on
# the 30 holdout documents, AFTER Addendum K's one scoring. They are
# pinned here so they cannot come back; the repaired recipe's score on
# those same 30 documents is holdout-informed and is cited nowhere.
#
# All three share a cause worth naming: the 20 TRAIN documents contain
# none of these layouts, so the recipe cleared a 0.0000 damage bar that
# proved nothing about them.

def test_a_value_is_reachable_past_an_equals_delimiter() -> None:
    """`h-3312`: a torn form reconstructed as `key=value;` prose.

    Splitting on `:` and column runs alone left `Tallahassee` reachable
    only as `destination=Ta11ahassee`, and grounding returned that whole.
    """
    document = ("RECONSTRUCTION (per carrier system, verified):\n"
                "shipper=Gu1fstream P1astics Extrusi0n; "
                "origin=Pensac01a; destination=Ta11ahassee;\n")
    assert is_document_grounded("Tallahassee", document)
    value, snapped = snap_to_document("Tallahassee", document)
    assert (value, snapped) == ("Tallahassee", False)


def test_a_value_is_reachable_past_a_pipe_delimiter() -> None:
    """`m-2208`: a one-line `key=value|key=value` system export."""
    document = ("id=88213|shpr=Calloway Textile Mills|"
                "cnee=Fabric Row Wholesale|orig=Greenville\n")
    value, snapped = snap_to_document("Calloway Textile Mills", document)
    assert (value, snapped) == ("Calloway Textile Mills", False)


def test_a_number_never_snaps_to_a_different_number() -> None:
    """`h-3307`/`h-3305`/`h-3304`: fuzzy matching is unsafe on quantities.

    Each of these cleared a 0.6 similarity bar and turned a CORRECT
    answer into a wrong one. A name's wrong character is a misreading of
    the right word; a number's is a different number.
    """
    assert snap_to_document("60", "Wt; 2,64O 1bs     C1ass; 6O\n")[0] == "60"
    assert snap_to_document("8050", "WEIGHT: 8,O5O LB5\n")[0] == "8050"
    assert snap_to_document("23000", "Weight 2,000 lbs net\n")[0] == "23000"


def test_a_transposed_or_misplaced_digit_is_still_repaired() -> None:
    """The numeric guard is multiset equality, not sequence equality:
    moving a digit is the error the document can fix, inventing one is
    not."""
    assert snap_to_document("9761.5", "Total Charges: $976.15\n")[0] == "976.15"
    assert snap_to_document("1152.60", "Total: $1,512.60\n")[0] == "1,512.60"


def test_an_ocr_tie_is_settled_by_the_token_before_its_neighbours() -> None:
    """`8,O5O` carries a thousands comma and `6O` has only digit-shaped
    letters; both are numbers however letter-heavy the page around them
    is. `C0.` has neither signal and still defers to its context."""
    assert repair_ocr_token("8,O5O") == "8,050"
    assert repair_ocr_token("6O") == "60"
    assert repair_ocr_token("C0.") == "C0."
    assert repair_ocr("Grande R0nde Lumber C0.") == "Grande Ronde Lumber Co."
