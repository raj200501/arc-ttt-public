#!/usr/bin/env python3
"""Addendum U runner: what shipped parsers do to real model output.

    PYTHONPATH=src python3 scripts/parser_robustness.py          # run + bank
    PYTHONPATH=src python3 scripts/parser_robustness.py --dry    # print, bank nothing

Preregistration: docs/research/ADDENDUM_U_PROTOCOL.md (frozen 2026-09-04
before this file existed). Every parser is the SHIPPED callable where
one exists; the reference labeler is the SHIPPED tools/fencecheck.py
strip_fence plus the fail-closed parse_json_object. Readings are applied
by arithmetic from the frozen thresholds in `read_u1`, `read_u2`,
`read_u3` -- never re-derived from the shape of the data.

The runner refuses to run on a corpus whose SHA-256 does not match a
fresh rebuild (a stale corpus would let a reading be taken on outputs
the manifest does not describe).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import pathlib
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

DEFAULT_TAG = "2026-09-04"


def artifact_paths(tag: str):
    """(corpus, manifest, base-out, ext-out) for a dated corpus. The first
    reading (2026-09-04, four families) is never overwritten by a later
    tag; a later tag is the second, dated reading published beside it."""
    exp = REPO / "experiments"
    return (exp / f"fence_corpus_{tag}.jsonl", exp / f"fence_corpus_{tag}.manifest.json",
            exp / f"parser_robustness_{tag}.json", exp / f"parser_robustness_ext_{tag}.json")


CORPUS, MANIFEST, OUT, OUT_EXT = artifact_paths(DEFAULT_TAG)

LENIENT = ("langchain_parse_json_markdown", "json_repair")
# Addendum U-ext (docs/research/ADDENDUM_U_EXT_PROTOCOL.md, frozen after U
# was banked and corrected): three more shipped helpers, same corpus, same
# quantities, U2 verbatim; no U1/U3 because `strict` is not in this panel.
LENIENT_EXT = ("instructor_extract_json_from_codeblock", "smolagents_parse_json_blob",
               "llama_index_parse_json_markdown")


# --------------------------------------------------------------------------
# reference labeler -- the shipped instrument
# --------------------------------------------------------------------------

def _fencecheck():
    spec = importlib.util.spec_from_file_location("fencecheck", REPO / "tools" / "fencecheck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(text: str, fc, parse_json_object, TextTaskFormatError):
    body, fenced = fc.strip_fence(text)
    try:
        obj = parse_json_object(body)
    except TextTaskFormatError:
        obj = None
    return obj, fenced


# --------------------------------------------------------------------------
# parsers under test -- each is text -> dict | None
# --------------------------------------------------------------------------

def _as_object(value):
    return value if isinstance(value, dict) else None


def parser_strict(text: str):
    """evals/elsuite/basic/json_match.py:80 semantics, verbatim:
    `try: sampled_json = json.loads(sampled) except ValueError: sampled_json = None`.
    A non-object value is None here because the reference requires an
    object and a list/scalar cannot equal it."""
    try:
        # fencecheck: ignore -- this IS the fail-open parse under test
        value = json.loads(text)
    except ValueError:
        return None
    return _as_object(value)


def make_parsers() -> dict:
    parsers = {"strict": parser_strict}
    versions = {"python_json": "stdlib", "evals (site cited, not called)": _dist_version("evals")}

    import autoevals
    from autoevals import ValidJSON
    _vj = ValidJSON()

    def parser_autoevals_validjson(text: str):
        # autoevals/json.py:162-163 -- JSONDiff parses a string operand
        # only when ValidJSON says it is valid; otherwise the string is
        # compared as text. Reproduced through the shipped ValidJSON.
        if _vj.valid_json(text) == 1:
            # fencecheck: ignore -- gated by the shipped validator above
            return _as_object(json.loads(text))
        return None
    parsers["autoevals_validjson"] = parser_autoevals_validjson
    versions["autoevals"] = _dist_version("autoevals")

    from langchain_core.output_parsers.json import parse_json_markdown

    def parser_langchain(text: str):
        try:
            return _as_object(parse_json_markdown(text))
        except Exception:  # the shipped function raises json.JSONDecodeError on unparseable text
            return None
    parsers["langchain_parse_json_markdown"] = parser_langchain
    versions["langchain_core"] = _dist_version("langchain-core")

    import json_repair

    def parser_json_repair(text: str):
        try:
            return _as_object(json_repair.loads(text))
        except Exception:
            return None
    parsers["json_repair"] = parser_json_repair
    versions["json_repair"] = _dist_version("json-repair")
    return parsers, versions


def make_parsers_ext() -> tuple[dict, dict, dict]:
    """The U-ext panel: (parsers, versions, raw callables) -- raw callables
    return the shipped value untouched so non-object returns can be banked."""
    from instructor.utils import extract_json_from_codeblock
    from smolagents.utils import parse_json_blob
    from llama_index.core.output_parsers.utils import parse_json_markdown as li_parse

    def raw_instructor(text: str):
        # instructor/utils.py: the LAST balanced {...}/[...] span, then strict json
        # fencecheck: ignore -- the shipped helper's own parse step, under test
        return json.loads(extract_json_from_codeblock(text))

    def raw_smolagents(text: str):
        # smolagents/utils.py: first "{" to last "}", json.loads(strict=False)
        return parse_json_blob(text)[0]

    def raw_llama_index(text: str):
        # llama_index/core/output_parsers/utils.py: ```json opener stripped,
        # json.loads, then yaml.safe_load as the fallback
        return li_parse(text)

    raw = {"instructor_extract_json_from_codeblock": raw_instructor,
           "smolagents_parse_json_blob": raw_smolagents,
           "llama_index_parse_json_markdown": raw_llama_index}

    def wrap(fn):
        def parser(text: str):
            try:
                return _as_object(fn(text))
            except Exception:
                return None
        return parser

    parsers = {name: wrap(fn) for name, fn in raw.items()}
    versions = {"instructor": _dist_version("instructor"), "smolagents": _dist_version("smolagents"),
                "llama_index_core": _dist_version("llama-index-core")}
    return parsers, versions, raw


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def decompose_fabricated(text: str, obj, fenced: bool) -> str:
    """Substance check on a `fabricated` status. Added 2026-09-04 AFTER the
    first run showed the frozen U2 reading firing MATERIAL; corrected twice
    the same day after adversarial reviews (see the errata in
    ADDENDUM_U_PROTOCOL.md and ADDENDUM_U_EXT_PROTOCOL.md). Categories,
    checked by arithmetic, in this order:
      exact_object_present      -- a substring parses strictly to exactly the
                                   returned object AND the text outside that
                                   span holds no brace at all: prose (or a
                                   fence, or a code sample) around one complete
                                   object -- the reference's documented
                                   undercount; recovered, not invented;
      one_of_several_objects    -- the span is exact, and once every other
                                   complete top-level object is peeled out of
                                   the text nothing JSON-like remains: the
                                   model wrote several objects; the helper
                                   chose one;
      nested_fragment_returned  -- the span is exact, and JSON keys (`"k":` or
                                   `'k':`) remain outside every complete
                                   object: the returned object is a piece of a
                                   larger object the model wrote that does
                                   not parse;
      stray_brace_around_object -- the span is exact, and the text outside it
                                   holds braces but no keys and no object;
      fence_after_prose         -- no exact span; a fence exists but the text
                                   does not START with one;
      leading_fenced_malformed  -- no exact span; the reference stripped a
                                   leading fence and the body still failed the
                                   fail-closed parse: the parser closed or
                                   rewrote malformed text;
      unfenced_malformed        -- no exact span, no fence: the parser closed or
                                   rewrote malformed text.
    Only the first category is recovery. Repairs are not adjudicated for
    correctness. The frozen readings are untouched by this section."""
    span = _exact_span(text, obj)
    if span is not None:
        start, end = span
        rest = text[:start] + text[end:]
        if "{" not in rest and "}" not in rest:
            return "exact_object_present"
        # peel every other complete top-level object out of the rest; what
        # remains decides whether the model wrote several objects (nothing
        # JSON-like remains) or one larger object this span is a piece of
        remaining = rest
        while True:
            other = _exact_span(remaining, None)
            if other is None:
                break
            remaining = remaining[:other[0]] + remaining[other[1]:]
        if "{" not in remaining and "}" not in remaining and not _KEY_RE.search(remaining):
            return "one_of_several_objects"
        if _KEY_RE.search(remaining) or _KEY_RE.search(rest):
            return "nested_fragment_returned"
        return "stray_brace_around_object"
    if fenced:
        return "leading_fenced_malformed"
    if "```" in text:
        return "fence_after_prose"
    return "unfenced_malformed"


import re as _re
_KEY_RE = _re.compile(r'(?:"[^"\n]{1,80}"|\'[^\'\n]{1,80}\')\s*:')
_EXPR_RE = _re.compile(r"^\s*[-\d.,]+\s*[*+/-]\s*[-\d.,]+")


def _exact_span(text: str, obj):
    """(start, end) of the first substring that parses strictly to `obj`;
    with obj=None, of ANY substring that parses strictly to a dict."""
    for m in _re.finditer(r"\{", text):
        start = m.start()
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            try:
                # fencecheck: ignore -- substring probe for the substance
                # check, not a scoring path
                val = json.loads(text[start:end])
            except ValueError:
                continue
            if (obj is None and isinstance(val, dict)) or (obj is not None and val == obj):
                return start, end
    return None


def fragment_kind(obj: dict, vocab: dict) -> str:
    """Which piece of a CORD receipt a fragment is, by its key set."""
    keys = set(obj)
    for name in ("menu", "sub_total", "total"):
        if keys and keys <= set(vocab[name]):
            return name
    return "other"


def string_valued_expressions(obj) -> int:
    """Leaf string values that look like arithmetic (`2 * 13000`)."""
    n = 0
    stack = [obj]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
        elif isinstance(v, str) and _EXPR_RE.match(v):
            n += 1
    return n


def jsondiff_scorer():
    from autoevals import JSONDiff
    _jd = JSONDiff()

    def score(text: str, expected: dict) -> float:
        return float(_jd.eval(output=text, expected=expected).score)
    return score


# --------------------------------------------------------------------------
# statuses and rates
# --------------------------------------------------------------------------

def status(ref, got) -> str:
    if ref is not None:
        if got is None:
            return "lost"
        return "ok" if got == ref else "diverged"
    return "agree_none" if got is None else "fabricated"


def rates(statuses: list[str]) -> dict:
    n = len(statuses)
    n_ref = sum(s in ("ok", "lost", "diverged") for s in statuses)
    n_noref = n - n_ref
    lost = statuses.count("lost"); div = statuses.count("diverged"); fab = statuses.count("fabricated")
    return {
        "n": n, "n_ref": n_ref, "n_noref": n_noref,
        "lost": lost, "diverged": div, "fabricated": fab, "ok": statuses.count("ok"),
        "lost_rate": round(lost / n_ref, 4) if n_ref else None,
        "diverged_rate": round(div / n_ref, 4) if n_ref else None,
        "fabricated_rate": round(fab / n_noref, 4) if n_noref else None,
        "hazard_rate": round((div + fab) / n, 4) if n else None,
    }


def slice_key(r: dict) -> str:
    return "|".join([r["family"], r["size"], "adapted" if r["adapted"] else "prompted",
                     r["corpus"], r["regime"], f"k={r['k']}", r["decoder"]])


# --------------------------------------------------------------------------
# frozen readings -- arithmetic only
# --------------------------------------------------------------------------

def family_strict_reading(lost_rate: float) -> str:
    if lost_rate >= 0.50:
        return "LOSES"
    if lost_rate < 0.10:
        return "DOES NOT LOSE"
    return "PARTIAL"


def read_u1(per_family: dict) -> str:
    """per_family: family -> LOSES | PARTIAL | DOES NOT LOSE (strict, schema-only)."""
    n_loses = sum(v == "LOSES" for v in per_family.values())
    exceptions = sorted(f for f, v in per_family.items() if v == "DOES NOT LOSE")
    n_fam = len(per_family)
    if exceptions:
        headline = ("" if n_loses >= 3 else
                    f" LOSES fired in only {n_loses} of {n_fam} families, so there is NO combined "
                    f"headline in either form: the rates publish and nothing is said about "
                    f"families as a class.")
        return (f"U1 EXCEPTION IN {', '.join(exceptions)}: named at full size; the sentence "
                f"becomes 'on N of the {n_fam} families tested' -- never 'across families'.{headline} "
                f"Per-family: {json.dumps(per_family)}")
    if n_loses >= 3:
        return (f"U1 HOLDS: fail-open parsing loses the majority of schema-only outputs in "
                f"{n_loses} of {n_fam} families and no family is an exception. "
                f"Per-family: {json.dumps(per_family)}")
    return (f"U1 MIXED: LOSES in {n_loses} of {n_fam} families, no exception -- all rates "
            f"publish, no headline. Per-family: {json.dumps(per_family)}")


def read_u2(overall_hazard: float, slice_hazards: dict) -> str:
    """slice_hazards: slice -> (hazard_rate, n) for one lenient parser."""
    material = sorted((s, h) for s, (h, n) in slice_hazards.items() if n >= 30 and h is not None and h >= 0.05)
    if material:
        worst = max(material, key=lambda x: x[1])
        return (f"U2 MATERIAL: hazard >= 0.05 on {len(material)} slice(s) with n >= 30; worst "
                f"{worst[0]} at {worst[1]:.4f}. Lenient repair changes or invents content on real outputs.")
    any_slice = any(h is not None and h >= 0.05 for h, n in slice_hazards.values())
    if overall_hazard < 0.01 and not any_slice:
        return (f"U2 HARMLESS ON THIS CORPUS: overall hazard {overall_hazard:.4f} < 0.01 and no "
                f"slice reaches 0.05. Published as the finding.")
    return (f"U2 PRESENT: overall hazard {overall_hazard:.4f}, no slice with n >= 30 "
            f"reaches 0.05 -- stated at size, no headline.")


def read_u3(kshot_lost_rate, n_ref: int, residual: dict) -> str:
    if kshot_lost_rate is None:
        return "U3 NOT READABLE: no k=20 record the reference parses."
    if kshot_lost_rate < 0.05:
        return (f"U3 SAME PHENOMENON: strict loses {kshot_lost_rate:.4f} of k=20 ref outputs "
                f"(n_ref={n_ref}) -- where the fence goes, the parser loss goes.")
    return (f"U3 RESIDUAL: strict loses {kshot_lost_rate:.4f} of k=20 ref outputs (n_ref={n_ref}); "
            f"decomposed by cause: {json.dumps(residual)}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(dry: bool, panel: str = "base", tag: str = DEFAULT_TAG) -> int:
    import fence_corpus
    lenient = LENIENT if panel == "base" else LENIENT_EXT
    CORPUS, MANIFEST, OUT, OUT_EXT = artifact_paths(tag)
    out_path = OUT if panel == "base" else OUT_EXT
    if not CORPUS.exists() or not MANIFEST.exists():
        print(f"REFUSED: no corpus tagged {tag} -- run tools/fence_corpus.py --tag {tag} first.")
        return 2
    manifest = json.loads(MANIFEST.read_text())
    listed = {a["artifact"] for a in manifest["artifacts_present"]}
    records_fresh, _ = fence_corpus.build(only=listed)
    fresh_sha = hashlib.sha256(fence_corpus.serialize(records_fresh).encode("utf-8")).hexdigest()
    if hashlib.sha256(CORPUS.read_bytes()).hexdigest() != fresh_sha:
        print("REFUSED: the banked corpus does not match a rebuild from its own manifest -- "
              "an artifact it lists has changed; rebuild and re-bank deliberately.")
        return 2
    records = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    fc = _fencecheck()
    from arcttt.scoring import parse_json_object
    from arcttt.text_task import TextTaskFormatError
    if panel == "base":
        parsers, versions = make_parsers()
    else:
        parsers, versions, raw_ext = make_parsers_ext()
    jd_score = jsondiff_scorer()

    per_record = []
    by_parser_slice: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    by_parser_regime: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    jd_fenced, jd_unfenced = [], []
    residual_causes: dict[str, int] = defaultdict(int)
    fab_kinds: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    non_dict: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    frag_kinds: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    expr_counts: dict[str, int] = defaultdict(int)
    import cord_fence_tax as _cft
    _cord_vocab = _cft.VOCAB
    non_dict_status: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    n_fenced = 0
    if panel == "base":
        import json_repair as _jr
        from langchain_core.output_parsers.json import parse_json_markdown as _lc
        _raw_lenient = {"json_repair": lambda t: _jr.loads(t),
                        "langchain_parse_json_markdown": lambda t: _lc(t)}
    else:
        _raw_lenient = raw_ext
    for r in records:
        ref, fenced = reference(r["text"], fc, parse_json_object, TextTaskFormatError)
        n_fenced += fenced
        row = {"id": r["id"], "fenced": fenced, "ref": ref is not None, "status": {}}
        sk = slice_key(r)
        for name, fn in parsers.items():
            got = fn(r["text"])
            st = status(ref, got)
            row["status"][name] = st
            if st == "fabricated" and name in lenient:
                cat = decompose_fabricated(r["text"], got, fenced)
                row.setdefault("fabricated_kind", {})[name] = cat
                fab_kinds[name][cat] += 1
                if cat == "nested_fragment_returned" and r["corpus"] == "cord":
                    frag_kinds[name][fragment_kind(got, _cord_vocab)] += 1
                if string_valued_expressions(got):
                    expr_counts[name] += 1
            by_parser_slice[name][sk].append(st)
            by_parser_regime[name][r["regime"]].append(st)
            if name == "strict" and st == "lost" and r["regime"] == "kshot" and r["k"] == 20:
                # by cause: a lost ref record is fenced (the fence is the cause) or,
                # if unfenced, the raw text is invalid JSON for another reason
                residual_causes["fenced" if fenced else "unfenced_other"] += 1
        # What the shipped lenient callables return when it is NOT an object
        # (the protocol preregistered non-dict -> None; a shipped consumer
        # would receive the value). Banked beside the readings, and a second
        # status is banked as if those returns counted as fabricated.
        for name, raw in _raw_lenient.items():
            try:
                val = raw(r["text"])
            except Exception:
                val = None
            kind = ("dict" if isinstance(val, dict) else "list" if isinstance(val, list)
                    else "empty_string" if val == "" else "none" if val is None
                    else type(val).__name__)
            if kind not in ("dict", "none"):
                non_dict[name][kind] += 1
            st_counted = row["status"][name]
            if st_counted == "agree_none" and kind == "list":
                st_counted = "fabricated"  # a list of objects reaches the consumer
            non_dict_status[name][sk].append(st_counted)
        if ref is not None and panel == "base":
            (jd_fenced if fenced else jd_unfenced).append(jd_score(r["text"], ref))
        per_record.append(row)

    # tables
    slices = {name: {sk: rates(sts) for sk, sts in sorted(d.items())} for name, d in by_parser_slice.items()}
    regimes = {name: {rg: rates(sts) for rg, sts in sorted(d.items())} for name, d in by_parser_regime.items()}
    overall = {name: rates([s for d in by_parser_slice[name].values() for s in d]) for name in parsers}

    # U1: strict, schema-only, per family (base panel only)
    u1_per_family_rates, u1, u3 = {}, None, None
    if panel == "base":
        fam_schema: dict[str, list[str]] = defaultdict(list)
        for r, row in zip(records, per_record):
            if r["regime"] == "schema":
                fam_schema[r["family"]].append(row["status"]["strict"])
        u1_per_family_rates = {f: rates(s) for f, s in sorted(fam_schema.items())}
        u1_per_family = {f: family_strict_reading(v["lost_rate"]) for f, v in u1_per_family_rates.items()
                         if v["lost_rate"] is not None}
        u1 = read_u1(u1_per_family)

    # U2: each lenient parser in the panel
    u2 = {}
    for name in lenient:
        sh = {sk: (v["hazard_rate"], v["n"]) for sk, v in slices[name].items()}
        u2[name] = read_u2(overall[name]["hazard_rate"], sh)

    # U3: strict on k=20 pooled (base panel only)
    if panel == "base":
        k20 = [row["status"]["strict"] for r, row in zip(records, per_record)
               if r["regime"] == "kshot" and r["k"] == 20]
        k20_rates = rates(k20)
        u3 = read_u3(k20_rates["lost_rate"], k20_rates["n_ref"], dict(residual_causes))

    record = {
        "what": ("Addendum U: what shipped parsers do to every raw model output this "
                 "project has banked. Reference = the shipped fencecheck strip_fence + "
                 "fail-closed parse_json_object. Readings applied by arithmetic from "
                 "docs/research/ADDENDUM_U_PROTOCOL.md." if panel == "base" else
                 "Addendum U-ext: three more shipped JSON helpers (instructor, smolagents, "
                 "llama-index) on the same corpus, same reference, U2 thresholds verbatim; "
                 "docs/research/ADDENDUM_U_EXT_PROTOCOL.md."),
        "panel": panel,
        "preregistration": ("docs/research/ADDENDUM_U_PROTOCOL.md (committed 7e0cf36 before the runner was written)"
                            if panel == "base" else
                            "docs/research/ADDENDUM_U_EXT_PROTOCOL.md (committed aba5089 before these helpers ran)"),
        "corpus": {"path": CORPUS.name, "tag": tag, "sha256": fresh_sha, "n_records": len(records),
                   "n_fenced_by_reference": n_fenced,
                   "families_present": manifest["families_present"],
                   "artifacts_absent": manifest["artifacts_absent"]},
        "parsers": ({
            "strict": "json.loads(text); except ValueError -> None; non-object -> None "
                      "(evals/elsuite/basic/json_match.py:80 semantics, verbatim)",
            "autoevals_validjson": "autoevals.ValidJSON().valid_json(text) == 1 gate, then json.loads "
                                   "(autoevals/json.py:162-163, the gate JSONDiff uses)",
            "langchain_parse_json_markdown": "langchain_core.output_parsers.json.parse_json_markdown; exception -> None",
            "json_repair": "json_repair.loads; non-object -> None",
            "fencecheck (reference)": "tools/fencecheck.strip_fence + arcttt.scoring.parse_json_object; zero loss by construction, not ranked",
        } if panel == "base" else {
            "instructor_extract_json_from_codeblock": "json.loads(instructor.utils.extract_json_from_codeblock(text)); exception -> None; non-object -> None",
            "smolagents_parse_json_blob": "smolagents.utils.parse_json_blob(text)[0]; exception -> None; non-object -> None",
            "llama_index_parse_json_markdown": "llama_index.core.output_parsers.utils.parse_json_markdown(text) (json, then yaml fallback); exception -> None; non-object -> None",
        }),
        "versions": versions,
        "overall": overall,
        "by_regime": regimes,
        "by_slice": slices,
        "u1_strict_schema_by_family": u1_per_family_rates,
        "autoevals_jsondiff_score": {
            "expected": "the reference object",
            "fenced_ref_records": {"n": len(jd_fenced), "mean": round(sum(jd_fenced) / len(jd_fenced), 4) if jd_fenced else None},
            "unfenced_ref_records": {"n": len(jd_unfenced), "mean": round(sum(jd_unfenced) / len(jd_unfenced), 4) if jd_unfenced else None},
        },
        "readings": {"U1": u1, "U2": u2, "U3": u3},
        "u2_substance_added_2026-09-04": {
            "why": "added after the first run, before the row was written, and CORRECTED the "
                   "same day: the first version labelled leading-fenced-but-malformed bodies "
                   "as 'a fence after prose' (the reference's undercount). They are not: the "
                   "reference stripped that fence and the body failed the fail-closed parse "
                   "because it was truncated or held expressions. Only `exact_object_present` "
                   "is recovery; the rest are repairs whose correctness is not adjudicated. "
                   "The frozen readings above are untouched.",
            "fabricated_by_kind": {name: dict(kinds) for name, kinds in fab_kinds.items()},
            "nested_fragment_piece_of_cord_receipt": {name: dict(k) for name, k in frag_kinds.items()},
            "fabricated_objects_with_string_valued_expressions": dict(expr_counts),
        },
        "lenient_non_object_returns_added_2026-09-04": {
            "why": "the protocol preregistered non-dict -> None for the lenient parsers; the "
                   "shipped callables return lists (several objects found) and empty strings "
                   "(nothing found) that a shipped consumer would receive. Counts banked, and "
                   "the U2 slice table recomputed as if every list return counted as an "
                   "object (an empty string is a failure any consumer would notice), so the "
                   "reader can see what the preregistered rule hid.",
            "non_object_returns": {name: dict(k) for name, k in non_dict.items()},
            "overall_if_counted": {name: rates([s for d in non_dict_status[name].values() for s in d])
                                   for name in lenient},
            "slices_at_or_above_0.05_if_counted": {
                name: sorted(sk for sk, sts in non_dict_status[name].items()
                             if len(sts) >= 30 and rates(sts)["hazard_rate"] >= 0.05)
                for name in lenient},
        },
        "per_record": per_record,
    }
    print(f"corpus {len(records)} records, {n_fenced} fenced by the reference, families {manifest['families_present']}")
    for name, v in overall.items():
        print(f"  {name:32s} lost {v['lost']:4d}/{v['n_ref']:<4d} ({v['lost_rate']})  diverged {v['diverged']:3d}  "
              f"fabricated {v['fabricated']:3d}/{v['n_noref']:<4d}  hazard {v['hazard_rate']}")
    if panel == "base":
        print("  JSONDiff mean score, fenced ref:", record["autoevals_jsondiff_score"]["fenced_ref_records"],
              "unfenced ref:", record["autoevals_jsondiff_score"]["unfenced_ref_records"])
        print("\n" + u1)
    else:
        record.pop("autoevals_jsondiff_score"); record.pop("u1_strict_schema_by_family")
        record["readings"] = {"U2": u2}
        # keep the post-hoc disclosure text in the ext artifact too
        record["substance_check_post_hoc"] = record.pop("u2_substance_added_2026-09-04")
    [print(v) for v in u2.values()]
    if u3:
        print(u3)
    if dry:
        print("\n(dry: nothing banked)")
        return 0
    out_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"\nbanked: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--panel", choices=("base", "ext"), default="base",
                    help="base = Addendum U's panel; ext = Addendum U-ext's three helpers")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="date tag of the corpus to read (default: the first, four-family corpus)")
    a = ap.parse_args()
    return run(a.dry, a.panel, a.tag)


if __name__ == "__main__":
    sys.exit(main())
