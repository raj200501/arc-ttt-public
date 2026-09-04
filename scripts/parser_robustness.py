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

CORPUS = REPO / "experiments" / "fence_corpus_2026-09-04.jsonl"
MANIFEST = REPO / "experiments" / "fence_corpus_2026-09-04.manifest.json"
OUT = REPO / "experiments" / "parser_robustness_2026-09-04.json"

LENIENT = ("langchain_parse_json_markdown", "json_repair")


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
    versions = {"python_json": "stdlib"}

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
        except Exception:  # the shipped function raises OutputParserException/JSONDecodeError
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


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def decompose_fabricated(text: str, obj) -> str:
    """Substance check on a `fabricated` status, added 2026-09-04 AFTER the
    first run showed the frozen U2 reading firing MATERIAL: the reference
    under-credits by Addendum R's documented undercount (a fence after
    prose is not stripped; an object inside prose is not found), so a
    lenient parser that returns the object the model actually wrote is
    recovering, not inventing. Categories, checked by arithmetic:
      exact_object_present -- a substring of the text parses strictly to
                              exactly the returned object (the reference
                              missed it, the parser did not invent it);
      fence_elsewhere      -- a fence exists but not at the start (the
                              reference's leading-fence scope missed it);
      repaired_or_invented -- neither: the parser changed malformed text
                              into an object. NOT adjudicated for
                              correctness here.
    The frozen readings are untouched by this section."""
    import re
    for m in re.finditer(r"\{", text):
        start = m.start()
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            try:
                # fencecheck: ignore -- substring probe for the substance
                # check, not a scoring path
                if json.loads(text[start:end]) == obj:
                    return "exact_object_present"
            except ValueError:
                pass
    if "```" in text:
        return "fence_elsewhere"
    return "repaired_or_invented"


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
        return (f"U1 EXCEPTION IN {', '.join(exceptions)}: named at full size; the sentence "
                f"becomes 'on N of the {n_fam} families tested' -- never 'across families'. "
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
    if overall_hazard < 0.01:
        return (f"U2 HARMLESS ON THIS CORPUS: overall hazard {overall_hazard:.4f} < 0.01 and no "
                f"slice with n >= 30 reaches 0.05. Published as the finding.")
    return (f"U2 PRESENT: overall hazard {overall_hazard:.4f} (>= 0.01), no slice with n >= 30 "
            f"reaches 0.05 -- stated at size, no headline.")


def read_u3(kshot_lost_rate: float, n_ref: int, residual: dict) -> str:
    if kshot_lost_rate < 0.05:
        return (f"U3 SAME PHENOMENON: strict loses {kshot_lost_rate:.4f} of k=20 ref outputs "
                f"(n_ref={n_ref}) -- where the fence goes, the parser loss goes.")
    return (f"U3 RESIDUAL: strict loses {kshot_lost_rate:.4f} of k=20 ref outputs (n_ref={n_ref}); "
            f"decomposed by cause: {json.dumps(residual)}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(dry: bool) -> int:
    import fence_corpus
    records_fresh, manifest_fresh = fence_corpus.build()
    fresh_sha = hashlib.sha256(fence_corpus.serialize(records_fresh).encode("utf-8")).hexdigest()
    if not CORPUS.exists() or hashlib.sha256(CORPUS.read_bytes()).hexdigest() != fresh_sha:
        print("REFUSED: the banked corpus does not match a fresh rebuild -- run "
              "tools/fence_corpus.py first so the reading is taken on the manifest it names.")
        return 2
    manifest = json.loads(MANIFEST.read_text())
    records = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    fc = _fencecheck()
    from arcttt.scoring import parse_json_object
    from arcttt.text_task import TextTaskFormatError
    parsers, versions = make_parsers()
    jd_score = jsondiff_scorer()

    per_record = []
    by_parser_slice: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    by_parser_regime: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    jd_fenced, jd_unfenced = [], []
    residual_causes: dict[str, int] = defaultdict(int)
    fab_kinds: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_fenced = 0
    for r in records:
        ref, fenced = reference(r["text"], fc, parse_json_object, TextTaskFormatError)
        n_fenced += fenced
        row = {"id": r["id"], "fenced": fenced, "ref": ref is not None, "status": {}}
        sk = slice_key(r)
        for name, fn in parsers.items():
            got = fn(r["text"])
            st = status(ref, got)
            row["status"][name] = st
            if st == "fabricated" and name in LENIENT:
                cat = decompose_fabricated(r["text"], got)
                row.setdefault("fabricated_kind", {})[name] = cat
                fab_kinds[name][cat] += 1
            by_parser_slice[name][sk].append(st)
            by_parser_regime[name][r["regime"]].append(st)
            if name == "strict" and st == "lost" and r["regime"] == "kshot" and r["k"] == 20:
                residual_causes["fenced" if fenced else "unfenced_invalid_or_other"] += 1
        if ref is not None:
            (jd_fenced if fenced else jd_unfenced).append(jd_score(r["text"], ref))
        per_record.append(row)

    # tables
    slices = {name: {sk: rates(sts) for sk, sts in sorted(d.items())} for name, d in by_parser_slice.items()}
    regimes = {name: {rg: rates(sts) for rg, sts in sorted(d.items())} for name, d in by_parser_regime.items()}
    overall = {name: rates([s for d in by_parser_slice[name].values() for s in d]) for name in parsers}

    # U1: strict, schema-only, per family
    fam_schema: dict[str, list[str]] = defaultdict(list)
    for r, row in zip(records, per_record):
        if r["regime"] == "schema":
            fam_schema[r["family"]].append(row["status"]["strict"])
    u1_per_family_rates = {f: rates(s) for f, s in sorted(fam_schema.items())}
    u1_per_family = {f: family_strict_reading(v["lost_rate"]) for f, v in u1_per_family_rates.items()
                     if v["lost_rate"] is not None}
    u1 = read_u1(u1_per_family)

    # U2: each lenient parser
    u2 = {}
    for name in LENIENT:
        sh = {sk: (v["hazard_rate"], v["n"]) for sk, v in slices[name].items()}
        u2[name] = read_u2(overall[name]["hazard_rate"], sh)

    # U3: strict on k=20 pooled
    k20 = [row["status"]["strict"] for r, row in zip(records, per_record)
           if r["regime"] == "kshot" and r["k"] == 20]
    k20_rates = rates(k20)
    u3 = read_u3(k20_rates["lost_rate"] if k20_rates["lost_rate"] is not None else 0.0,
                 k20_rates["n_ref"], dict(residual_causes))

    record = {
        "what": "Addendum U: what shipped parsers do to every raw model output this "
                "project has banked. Reference = the shipped fencecheck strip_fence + "
                "fail-closed parse_json_object. Readings applied by arithmetic from "
                "docs/research/ADDENDUM_U_PROTOCOL.md.",
        "preregistration": "docs/research/ADDENDUM_U_PROTOCOL.md (frozen 2026-09-04 before the runner existed)",
        "corpus": {"path": CORPUS.name, "sha256": fresh_sha, "n_records": len(records),
                   "n_fenced_by_reference": n_fenced,
                   "families_present": manifest["families_present"],
                   "artifacts_absent": manifest["artifacts_absent"]},
        "parsers": {
            "strict": "json.loads(text); except ValueError -> None; non-object -> None "
                      "(evals/elsuite/basic/json_match.py:80 semantics, verbatim)",
            "autoevals_validjson": "autoevals.ValidJSON().valid_json(text) == 1 gate, then json.loads "
                                   "(autoevals/json.py:162-163, the gate JSONDiff uses)",
            "langchain_parse_json_markdown": "langchain_core.output_parsers.json.parse_json_markdown; exception -> None",
            "json_repair": "json_repair.loads; non-object -> None",
            "fencecheck (reference)": "tools/fencecheck.strip_fence + arcttt.scoring.parse_json_object; zero loss by construction, not ranked",
        },
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
            "why": "added after the first run, before the row was written: the frozen U2 "
                   "reading fires on `fabricated`, which conflates the reference's "
                   "documented undercount (fence after prose; object inside prose) with "
                   "repair of malformed text. The frozen readings above are untouched; "
                   "this decomposition is published beside them and the non-flattering "
                   "reading governs the row.",
            "fabricated_by_kind": {name: dict(kinds) for name, kinds in fab_kinds.items()},
        },
        "per_record": per_record,
    }
    print(f"corpus {len(records)} records, {n_fenced} fenced by the reference, families {manifest['families_present']}")
    for name, v in overall.items():
        print(f"  {name:32s} lost {v['lost']:4d}/{v['n_ref']:<4d} ({v['lost_rate']})  diverged {v['diverged']:3d}  "
              f"fabricated {v['fabricated']:3d}/{v['n_noref']:<4d}  hazard {v['hazard_rate']}")
    print("  JSONDiff mean score, fenced ref:", record["autoevals_jsondiff_score"]["fenced_ref_records"],
          "unfenced ref:", record["autoevals_jsondiff_score"]["unfenced_ref_records"])
    print("\n" + u1); [print(v) for v in u2.values()]; print(u3)
    if dry:
        print("\n(dry: nothing banked)")
        return 0
    OUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"\nbanked: {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    return run(a.dry)


if __name__ == "__main__":
    sys.exit(main())
