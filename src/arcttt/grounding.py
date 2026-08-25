"""Document grounding: make a small model's output come from the document.

**Read this first: Addendum K, which this module was built for, returned
reading (c) and this module did not close the gap.** Grounded, the
adapted 0.5B scores 0.8833 on the 30 held-out waybills -- the same number
it scored ungrounded -- and an oracle bound that discards every
regression reaches only 0.8917, against a 0.90 bar. The hypothesis that
motivated it ("two deterministic mechanisms should close most of the
gap") is refuted by its own experiment, and the error taxonomy it was
built from was wrong: only 4 of the 28 field errors are the modes below.
The dominant mode is **field assignment** -- the model emits text that IS
in the document, under the wrong key -- which span snapping cannot fix by
construction. See `VERDICT.md`'s Addendum K row and CORRECTIONS.md.

The module is kept because it is a working serving component with its
numbers attached, and because the three modes it does fix are real:

  * confabulation      -- ``Calloway Textile Mills`` came back as
                          ``Allowce Textile Mills``; ``Grande Ronde
                          Lumber Co.`` as ``Grand River Lumber Co.``
  * un-asked rewriting -- ``PacCoast Surgical Distribution`` expanded to
                          ``Pacific Coast Surgical Distribution``
  * OCR passthrough    -- ``Va11ey Truss & Frame`` copied verbatim where
                          the gold corrects it to ``Valley Truss & Frame``

Each is a small model inventing or preserving characters it should have
copied or repaired, and each is deterministic given the document. Two
mechanisms:

**Span snapping.** For fields whose values are COPIED out of the document
rather than transformed, the output is constrained to a span that
actually occurs in it. A model cannot emit ``Allowce`` if ``Allowce`` is
not in the source.

**OCR repair.** This corpus's damaged documents substitute ``0`` for
``O`` and ``1`` for ``l``/``I`` inside words, and digits for letters in
numbers. The repair is by CHARACTER CLASS -- a token that is mostly
letters gets its digits read as letters, a token that is mostly digits
gets its letters read as digits -- so it needs no dictionary and no
knowledge of the tenant.

Both use the document and the model's own output. **Neither reads gold**,
so both are inference-time components of the serving stack rather than a
scoring adjustment. A hosted model could adopt the same two mechanisms;
nothing here is available only to us.

Which fields are copy-type is a property of a tenant's schema, and
:func:`infer_copy_fields` measures it from that tenant's TRAINING pairs.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence

# Digit/letter confusions this corpus's scanner produces. Case is chosen
# from the token, not baked in, so `R0nde` -> `Ronde` and `STRA1GHT` ->
# `STRAIGHT` both come out right.
_LETTER_FOR_DIGIT = {"0": "o", "1": "l", "5": "s", "8": "b"}
_DIGIT_FOR_LETTER = {"O": "0", "o": "0", "l": "1", "I": "1", "i": "1",
                     "S": "5", "s": "5", "B": "8", "b": "8"}
# `1` reads as `I` in an upper-case word and `l` in a lower-case one.
_UPPER_OVERRIDE = {"1": "I"}
# How similar two words must be before one is treated as the other's
# misreading rather than a different word entirely.
_SAME_WORD = 0.6


def _words(text: str) -> list[str]:
    """Split on whitespace AND the delimiters forms use around values.

    `destination=Ta11ahassee` is a label and a value, not a word, and
    treating it as one is what let grounding return it whole.
    """
    return [word for word in re.split(r"[\s=;|]+", text) if word]


def _looks_numeric(token: str) -> bool:
    """Does the token carry punctuation only a number carries?

    The characters flanking the separator are DIGIT-SHAPED rather than
    strictly digits, because the scanner damage is exactly what is being
    detected: in `8,O5O` the thousands comma is flanked by `8` and `O`,
    and requiring two real digits missed it and let the token read as a
    word. A terminal `.` -- `C0.` -- is not a separator and does not
    count.
    """
    digitish = "0-9" + "".join(_DIGIT_FOR_LETTER)
    return "$" in token or bool(
        re.search(rf"[{digitish}][.,][{digitish}]", token))


def _digits_of(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def _digits_only(text: str) -> bool:
    """Is this a quantity rather than a name -- digits and separators?"""
    return bool(_digits_of(text)) and not re.search(r"[A-Za-z]", str(text))


def _same_digits(span: str, value: str) -> bool:
    """Same digits, in any order -- a transposition, not a new number.

    Sequence equality would be the safer-looking test and it is too
    strict: the commonest numeric error a small model makes is swapping
    two adjacent digits, and the document is exactly the authority that
    fixes it. Multiset equality repairs `1152.60` -> `1512.60` and a
    misplaced decimal `9761.5` -> `976.15`, while still refusing every
    case that damaged the holdout -- `60` -> `0`, `8050` -> `8,50`,
    `23000` -> `2000`, `1088.60` -> `1,888.6` -- because each of those
    invents or loses a digit rather than moving one.
    """
    return sorted(_digits_of(span)) == sorted(_digits_of(value))


def _only_letters_are_confusable(token: str) -> bool:
    """Are the letters all scanner artefacts while the digits are not?"""
    letters = [ch for ch in token if ch.isalpha()]
    digits = [ch for ch in token if ch.isdigit()]
    return (bool(letters) and bool(digits)
            and all(ch in _DIGIT_FOR_LETTER for ch in letters)
            and not all(ch in _LETTER_FOR_DIGIT for ch in digits))


def _token_is_upper(token: str) -> bool:
    """Is this token upper-case, judged from letters after the first?

    The leading character is excluded because `Co.` and `Ronde` are
    title-case words whose first letter says nothing about the rest. With
    it excluded, `STRA1GHT` reads upper, `Va11ey` and `R0nde` read lower,
    and a one-letter stem like `C0.` falls through to lower, which is what
    title case wants.
    """
    rest = [ch for ch in token[1:] if ch.isalpha()]
    return bool(rest) and all(ch.isupper() for ch in rest)


def repair_ocr_token(token: str, tie_break: str | None = None) -> str:
    """Repair digit/letter confusion in one token, by character class.

    A token with no mixing is returned unchanged. A token whose letters
    and digits are evenly split -- `C0.` is the case that matters, one
    letter and one digit -- cannot be classified on its own, so it is
    resolved by ``tie_break``: the class the SURROUNDING text is written
    in. In `Grande R0nde Lumber C0.` the string is overwhelmingly
    letters, so `C0.` is a word and becomes `Co.`; standing alone with no
    context it is left untouched rather than guessed at.
    """
    letters = sum(ch.isalpha() for ch in token)
    digits = sum(ch.isdigit() for ch in token)
    if letters == 0 or digits == 0:
        return token
    if letters == digits:
        # Before reaching for context, ask the token itself. Two signals
        # settle a tie from inside it, and the holdout produced both:
        #   * numeric punctuation -- `8,O5O` carries a thousands comma,
        #     which no word does;
        #   * confusable asymmetry -- in `6O` every letter is one the
        #     scanner produces for a digit while `6` is not one it
        #     produces for a letter, so the token can only be a number.
        # `C0.` has neither (its `.` is terminal, and `C` is not a digit
        # confusion), so it still falls through to the surrounding text.
        if _looks_numeric(token) or _only_letters_are_confusable(token):
            letters, digits = 1, 2
        elif tie_break is None:
            return token
        else:
            letters, digits = ((2, 1) if tie_break == "word" else (1, 2))
    if letters > digits:
        upper = _token_is_upper(token)
        out = []
        for ch in token:
            if ch.isdigit() and ch in _LETTER_FOR_DIGIT:
                repl = _LETTER_FOR_DIGIT[ch]
                if upper:
                    repl = _UPPER_OVERRIDE.get(ch, repl).upper()
                out.append(repl)
            else:
                out.append(ch)
        return "".join(out)
    return "".join(_DIGIT_FOR_LETTER.get(ch, ch) if ch.isalpha() else ch
                   for ch in token)


def repair_ocr(text: str) -> str:
    """Apply :func:`repair_ocr_token` to every whitespace-separated token.

    The string's own overall character class breaks per-token ties, so a
    name reads as a name and an amount reads as an amount without either
    needing a dictionary.
    """
    text = str(text)
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    tie_break = ("word" if letters > digits
                 else "number" if digits > letters else None)
    return " ".join(repair_ocr_token(tok, tie_break)
                    for tok in text.split(" "))


def dehyphenate(document: str) -> str:
    """Rejoin words a fax or scanner broke across a line.

    ``Pinnacle Elec-\ntronics Assembly Inc.`` is one company name, not
    two fragments, and no amount of model capacity makes it one if the
    text it is given still has the break in it. This is a property of
    scanned documents, not of this corpus.
    """
    return re.sub(r"-\s*\n\s*", "", document)


def candidate_spans(document: str, max_words: int = 8) -> list[str]:
    """Spans a copied value could plausibly be.

    Every split here is load-bearing, and each was added because the
    TRAIN split showed a real document that needed it:

    * line-broken words are rejoined first (``h-3302``, a fax copy);
    * a line can hold more than one ``Label: Value`` pair, so lines are
      broken on runs of two or more spaces before labels are stripped;
    * values also appear in prose with no label at all -- ``From
      Allentown to Scranton`` (``m-2202``) -- so word n-grams up to
      ``max_words`` are candidates too;
    * both the raw span and a punctuation-trimmed copy are offered, so a
      snap can return ``Inc.`` rather than silently dropping a period the
      tenant's gold keeps.
    """
    seen: dict[str, None] = {}

    def offer(text: str) -> None:
        text = text.strip()
        if not text or set(text) <= set("-=_ "):
            return
        seen.setdefault(text, None)
        trimmed = text.strip(" .;,:\"'()[]")
        if trimmed and trimmed != text:
            seen.setdefault(trimmed, None)
        # A copied amount is written "$1,342.75" and a copied weight
        # "19,800 lbs"; the tenant's gold keeps neither the symbol nor the
        # unit. Offer the bare number too, or a snap turns a correct
        # value into a wrong one.
        bare = re.sub(r"^[^\d+-]*", "", trimmed)
        bare = re.sub(r"[^\d]*$", "", bare)
        if bare and bare != trimmed:
            seen.setdefault(bare, None)

    for raw_line in dehyphenate(document).splitlines():
        line = raw_line.strip()
        if not line or set(line) <= set("-=_ "):
            continue
        offer(line)
        for chunk in re.split(r"\s{2,}|[;|]", line):
            offer(chunk)
            # `:` is not the only thing a form puts between a label and
            # its value. The holdout carries `key=value|key=value` system
            # exports and `key=value;` prose reconstructions, and a
            # smudged carbon copy whose period keys read as `;`. Splitting
            # on `:` and column runs alone left `Tallahassee` reachable
            # only as `destination=Ta11ahassee`, and grounding returned
            # that. None of these three layouts occurs in the 20 TRAIN
            # documents, which is why the train split gave the tokenizer
            # a clean bill of health it had not earned.
            if ":" in chunk or "=" in chunk:
                value = re.split(r"[:=]", chunk, maxsplit=1)[1]
                offer(value)
                for part in value.split(","):
                    offer(part)
        words = _words(line)
        for start in range(len(words)):
            for size in range(1, max_words + 1):
                if start + size > len(words):
                    break
                offer(" ".join(words[start:start + size]))
    return list(seen)


def _scored(text: str) -> str:
    """The pinned scorer's own normalisation of a leaf value.

    Used wherever this module has to decide whether two strings would be
    judged equal, so that "copy-type" means "emitting this span would
    score correct" rather than "looks close enough to us".
    """
    from arcttt.scoring import normalize_value
    from arcttt.text_task import TextTaskFormatError

    try:
        return normalize_value(str(text))
    except TextTaskFormatError:
        return str(text).strip().casefold()


def _norm(text: str) -> str:
    """Compare-form: OCR-repaired, lower-cased, punctuation dropped.

    A decimal point BETWEEN DIGITS survives, because it is the one piece
    of punctuation that changes a value rather than decorating it.
    Dropping it made ``9761.5`` and ``976.15`` compare identical, so a
    misplaced decimal -- the single most costly error available on an
    invoice -- read as already grounded and was left uncorrected.
    """
    lowered = repair_ocr(str(text)).lower()
    kept = re.sub(r"(?<=\d)\.(?=\d)", "\0", lowered)
    return re.sub(r"[^a-z0-9\0]+", "", kept).replace("\0", ".")


def _document_word_forms(document: str) -> dict[str, str]:
    """Compare-form -> the document's own spelling, for every word in it.

    Line breaks are flattened first so that a value spanning two rows of a
    misaligned scan is still made of document words.
    """
    flat = repair_ocr(dehyphenate(document).replace("\n", " "))
    forms: dict[str, str] = {}
    for token in _words(flat):
        key = _norm(token)
        if key:
            forms.setdefault(key, token)
    return forms


def _document_words(document: str) -> set[str]:
    """Every word the document contains, OCR-repaired and normalised."""
    return set(_document_word_forms(document))


def is_document_grounded(value: str, document: str) -> bool:
    """Is every word of ``value`` already present in the document?

    This is the test for whether the model has LEFT the document, and it
    is deliberately weaker than "equals one contiguous span". Scanners
    misalign columns: ``h-3308`` prints ``Rustbelt Salvage &`` on the
    consignee row and ``Reclaim LLC`` on the row below it, and the
    document says in as many words that the two were swapped. A model
    that reassembles ``Rustbelt Salvage & Reclaim LLC`` out of that has
    done the hard part correctly; snapping it back to the longest
    contiguous span would undo the win and hand back a truncated name.

    So the rule is: a value built entirely out of words the document
    contains has not been invented, and grounding leaves it alone. A
    value containing a word the document does not have -- ``Allowce``,
    ``Pacific`` -- has been invented or rewritten, and that is the only
    case span snapping is for.
    """
    available = _document_words(document)
    parts = [_norm(token) for token in _words(str(value))]
    parts = [p for p in parts if p]
    return bool(parts) and all(part in available for part in parts)


def repair_words_from_document(value: str, document: str,
                               threshold: float = 0.6) -> tuple[str, bool]:
    """Ground a value word by word when no whole span will serve.

    Span snapping needs the value to exist contiguously in the document.
    On a misaligned scan it does not: ``h-3308`` prints the consignee's
    name across two rows, so the longest contiguous span is the truncated
    ``Rustbelt Salvage &`` and snapping to it LOSES a correct name.

    This is the fallback. Words the document already contains are left
    exactly as the model wrote them; only a word the document does not
    have is replaced, and only by the closest word the document does
    have, and only if that is closer than ``threshold``. So
    ``Recalim`` becomes ``Reclaim`` while ``Rustbelt``, ``Salvage`` and
    ``LLC`` are untouched -- and a word with no plausible source in the
    document is left alone rather than guessed at.
    """
    forms = _document_word_forms(document)
    out: list[str] = []
    changed = False
    for word in str(value).split():
        key = _norm(word)
        if not key or key in forms:
            out.append(word)
            continue
        best, best_score = None, 0.0
        for candidate_key, candidate in forms.items():
            score = difflib.SequenceMatcher(None, key, candidate_key).ratio()
            if score > best_score:
                best, best_score = candidate, score
        if best is not None and best_score >= threshold:
            out.append(_prefer_model_characters(best, word))
            changed = True
        else:
            out.append(word)
    grounded = " ".join(out)
    return grounded, changed and _scored(grounded) != _scored(value)


def snap_to_document(value: str, document: str,
                     threshold: float = 0.6) -> tuple[str, bool]:
    """Return the document span closest to ``value``.

    Comparison is on an OCR-repaired, punctuation-stripped form, so
    ``Va11ey Truss & Frame`` in the document matches a prediction of
    ``Valley Truss and Frame``. The span is returned in its REPAIRED
    form, which is what the corpus's gold uses.

    Returns ``(value, snapped)``. Below ``threshold`` the original is
    kept: a snap that has to reach is more likely to be wrong than the
    model was, and silently rewriting a correct answer is worse than
    leaving a wrong one.
    """
    target = _norm(value)
    if not target:
        return value, False
    # Grounding must only ever intervene where the model has LEFT the
    # document, never where it has not. Without a guard here, grounding
    # hunts for a closer-looking span and moves correct answers off them:
    # on the train split that cost -0.13 micro-F1 per document against
    # perfect input, and the residue after a contiguous-span guard was
    # still -0.01, entirely from documents whose layout was broken.
    if is_document_grounded(value, document):
        return value, False
    spans = candidate_spans(document)
    scored_spans = {_scored(s) for s in spans}
    if _scored(value) in scored_spans:
        return value, False
    # Ranked on the compare-form similarity FIRST, and ties broken on raw
    # similarity to what the model actually wrote. `candidate_spans`
    # offers a span both as it appears and with its outer punctuation
    # trimmed, and the two are identical in compare-form -- so without the
    # second key the winner is arbitrary and grounding returns `Fresno.`,
    # `"Bismarck",` or `-> Jackson`, each of which the pinned scorer
    # counts wrong. The model wrote `Fresno`, and it also wrote `Inc.`
    # where the period belongs; letting its own spelling break the tie
    # gets both right, which no rule over punctuation can. This is the
    # same division of labour as `_prefer_model_characters`: the document
    # decides WHICH span, the model decides how it is written.
    raw_value = str(value)
    best, best_key = None, (0.0, 0.0)
    for span in spans:
        key = (difflib.SequenceMatcher(None, target, _norm(span)).ratio(),
               difflib.SequenceMatcher(None, raw_value, span).ratio())
        if key > best_key:
            best, best_key = span, key
    if best is None or best_key[0] < threshold:
        return repair_words_from_document(value, document, threshold)
    # A snap must never DISCARD a word of the model's answer that the
    # document contains. Such a word is grounded evidence, and a span that
    # omits it is a truncation, not a correction -- which is exactly how
    # `Rustbelt Salvage & Reclaim LLC` became `Rustbelt Salvage &`. When
    # the best span would drop one, fall back to word-level repair, which
    # can ground a value the layout never puts in one place.
    available = _document_words(document)
    model_grounded = {_norm(word) for word in _words(raw_value)} & available
    span_words = {_norm(word) for word in _words(repair_ocr(best))}
    if model_grounded - span_words:
        return repair_words_from_document(value, document, threshold)
    repaired = _prefer_model_characters(repair_ocr(best), raw_value)
    # A NUMBER is never approximately right. Fuzzy matching is safe for a
    # name, where a wrong character is a misreading of the right word, and
    # unsafe for a quantity, where it is a different quantity: on the
    # holdout it turned `60` into `0`, `23000` into `2000` and `8050` into
    # `8,50`, each of which cleared a 0.6 similarity bar comfortably. So a
    # purely numeric value snaps only to a span with the SAME DIGITS --
    # the model read the digits, the document decides where the separators
    # go. That still repairs a misplaced decimal (`9761.5` -> `976.15`),
    # which is the expensive error, and refuses everything else.
    if _digits_only(raw_value) and not _same_digits(repaired, raw_value):
        return value, False
    return repaired, _scored(repaired) != _scored(value)


def _prefer_model_characters(span: str, model_value: str) -> str:
    """Let the model settle characters the scanner made ambiguous.

    `1` is `l` in `Va11ey` and `i` in `Ph1ladelphia`, and no rule over
    character classes can tell them apart -- but the model already wrote
    `Philadelphia`, because it has read English. So where the repaired
    span and the model's own answer are the same length and differ only
    at positions the scanner is known to confuse, the model's character
    wins. The document decides WHICH span; the model decides HOW it is
    spelled. Neither alone gets `Philadelphia` and `Valley` both right.
    """
    span_words = span.split()
    model_words = model_value.split()
    if len(span_words) == 1 and len(model_words) == 1:
        if len(span) != len(model_value):
            return span
        ambiguous = set("0O1lIi5S8B")
        return "".join(
            model_ch if (span_ch != model_ch and span_ch in ambiguous
                         and model_ch in ambiguous) else span_ch
            for span_ch, model_ch in zip(span, model_value))
    # Multi-word: align the two word sequences, then apply the rule above
    # word by word. Requiring equal WHOLE-STRING length switched the
    # mechanism off whenever the model added or dropped a single token --
    # precisely when it has most to offer. `Pacific Philadelphia Casters &
    # Wheels` snapped to the scanned `Ph1lade1phia Casters & Whee1s` came
    # back as `Phlladelphia`, because one extra word disqualified every
    # other word from being consulted.
    span_keys = [_norm(word) for word in span_words]
    model_keys = [_norm(word) for word in model_words]
    matcher = difflib.SequenceMatcher(None, span_keys, model_keys)
    out = list(span_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                out[i1 + offset] = _prefer_model_characters(
                    span_words[i1 + offset], model_words[j1 + offset])
        elif tag == "replace":
            # An unequal replace block is the case that matters, not an
            # edge case: a model that ADDS a word puts the word it also
            # misread into a 1-against-2 block. Pair each span word with
            # its most similar word in the block rather than giving up on
            # the block, and only when the two are close enough to be the
            # same word at all.
            for i in range(i1, i2):
                best, best_score = None, 0.0
                for j in range(j1, j2):
                    score = difflib.SequenceMatcher(
                        None, span_keys[i], model_keys[j]).ratio()
                    if score > best_score:
                        best, best_score = model_words[j], score
                if best is not None and best_score >= _SAME_WORD:
                    out[i] = _prefer_model_characters(span_words[i], best)
    return " ".join(out)


def infer_copy_fields(train_pairs: Sequence[tuple[str, Mapping[str, str]]],
                      min_rate: float = 0.8) -> list[str]:
    """Which fields are COPIED from the document, per this tenant.

    Measured from the tenant's own training pairs: the share of gold
    values that occur verbatim in their document, after OCR repair. A
    field at or above ``min_rate`` is copy-type and may be snapped;
    everything else is transform-type (dates get reformatted, weights get
    unit-converted, amounts get punctuation-stripped) and is left alone,
    because snapping a transformed value to its source span would undo
    the transformation.

    This is why the mechanism is not corpus-specific: the schema tells
    you which of its own fields are copies.
    """
    hits: dict[str, list[int]] = {}
    for document, gold in train_pairs:
        # A value is COPIED only if some whole candidate span, emitted
        # verbatim, would SCORE as correct -- judged by the pinned
        # scorer's own normalisation, not by a looser one of our own.
        # Two train-split failures forced this. A substring test made
        # "19800" match inside "19,800 lbs"; and a punctuation-stripping
        # comparison of our own made "$1,342.75" look identical to
        # "1342.75" when the scorer does not agree. Both would have
        # snapped correct values into wrong ones on the holdout.
        spans = {_scored(s) for s in candidate_spans(document)}
        for key, value in gold.items():
            counts = hits.setdefault(key, [0, 0])
            counts[1] += 1
            counts[0] += int(_scored(value) in spans)
    return sorted(key for key, (hit, total) in hits.items()
                  if total and hit / total >= min_rate)


def ground(prediction: Mapping[str, object], document: str,
           copy_fields: Sequence[str],
           threshold: float = 0.6) -> tuple[dict[str, object], dict[str, str]]:
    """Ground one prediction against its document.

    Every value is OCR-repaired; copy-type values are additionally
    snapped to a document span. Returns the grounded object and a record
    of what changed, so a run can report how much of its score came from
    the model and how much from grounding.
    """
    grounded: dict[str, object] = {}
    changes: dict[str, str] = {}
    for key, value in prediction.items():
        if value is None:
            grounded[key] = value
            continue
        text = str(value)
        repaired = repair_ocr(text)
        if key in copy_fields:
            snapped, did = snap_to_document(repaired, document, threshold)
            if did:
                changes[key] = f"{text!r} -> {snapped!r} (snapped)"
            elif repaired != text:
                changes[key] = f"{text!r} -> {repaired!r} (ocr)"
            grounded[key] = snapped
        else:
            if repaired != text:
                changes[key] = f"{text!r} -> {repaired!r} (ocr)"
            grounded[key] = repaired
    return grounded, changes
