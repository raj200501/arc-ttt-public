#!/usr/bin/env python3
"""Addendum W: the instrument's undercount, measured on the five-family corpus.

    PYTHONPATH=src python3 scripts/instrument_undercount.py

Preregistration: docs/research/ADDENDUM_W_PROTOCOL.md (frozen 2026-09-08,
commit 2fa6cd6, before the wider scope ran on any banked output). For
every record of `fence_corpus_2026-09-05.jsonl` the reference object is
taken under the shipped tool's two scopes -- `leading` (one leading
fence, the scope every frozen reading uses) and `any` (Addendum W's
opt-in widening) -- through the same fail-closed parse. Banked: newly
credited records by kind and slice, divergences (must be zero), the
recovered CORD score per newly credited record, and the S/T schema-only
fence rates under both scopes. Readings applied by arithmetic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

CORPUS = REPO / "experiments" / "fence_corpus_2026-09-05.jsonl"
MANIFEST = REPO / "experiments" / "fence_corpus_2026-09-05.manifest.json"
OUT = REPO / "experiments" / "instrument_undercount_2026-09-08.json"
S_ARTIFACT = REPO / "experiments" / "cord_fence_tax_2026-08-25.json"
T_ARTIFACT = REPO / "experiments" / "cord_fence_tax_families_2026-09-03.json"


def _fencecheck():
    spec = importlib.util.spec_from_file_location("fencecheck", REPO / "tools" / "fencecheck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_w2(newly: int, n_noref_leading: int) -> str:
    share = newly / n_noref_leading if n_noref_leading else 0.0
    if share < 0.10:
        return (f"W2 UNDER A TENTH: {newly} of the {n_noref_leading} outputs the leading scope "
                f"rejected ({share:.4f}) are credited by the wider scope, on this corpus.")
    return (f"W2 AT SIZE: {newly} of {n_noref_leading} ({share:.4f}) -- the S and T rows gain a "
            f"dated note carrying the any-scope rates beside the frozen ones.")


def read_w3(moves: dict) -> str:
    """The frozen letter: a cell's rate under `any` counts every record whose
    kind is not `none` (a bare object in prose included), and a move of
    >= 0.05 fires. Tested on COUNTS (20 x delta >= n), never on floats: the
    first run compared 0.97 - 0.92 in floating point and printed STABLE for
    a move that is exactly 0.05 (protocol erratum 1). The fences-only move
    (leading + fence-after-prose) is banked beside it (erratum 2)."""
    big = {k: v for k, v in moves.items()
           if 20 * abs(v["credited_any_kind_not_none"] - v["fenced_leading"]) >= v["n"]}
    if big:
        detail = "; ".join(f"{k.split('/')[-1]}: {v['fenced_leading']} -> {v['credited_any_kind_not_none']} "
                           f"by the letter (kind != none), {v['fenced_any_fences_only']} counting fences only"
                           for k, v in sorted(big.items()))
        return ("W3 MOVES by the letter: >= 0.05 of the cell under the wider scope in "
                + ", ".join(sorted(k.split("/")[-1] for k in big)) + f" ({detail}) -- the T reading is "
                "re-checked with the any-scope rate in `t_exception_recheck_with_any_rates`; the frozen "
                "reading is not re-taken.")
    return "W3 STABLE: no S/T schema-only cell moves by >= 0.05 under the wider scope, by the letter or counting fences only."


def main() -> int:
    from arcttt.scoring import field_micro_f1, parse_json_object
    from arcttt.text_task import TextTaskFormatError
    import cord_fence_tax as cft
    fc = _fencecheck()
    manifest = json.loads(MANIFEST.read_text())
    sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if sha != manifest["corpus_sha256"]:
        print("REFUSED: corpus does not match its manifest"); return 2
    records = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in
            (cft.SPLIT_DIR / "gold.jsonl").read_text().splitlines() if l.strip()}
    for r in cft._read_jsonl(cft.SPLIT_DIR / "train.jsonl"):
        gold[r["id"]] = r["gold"]

    def obj(body):
        try:
            return parse_json_object(body)
        except TextTaskFormatError:
            return None

    newly, diverged, per_kind, per_slice = [], [], defaultdict(int), defaultdict(int)
    n_noref_leading = 0
    fenced_any: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        body_l, fenced_l = fc.strip_fence(r["text"])
        body_a, kind = fc.extract_json(r["text"], "any")
        o_l, o_a = obj(body_l), obj(body_a)
        sk = "|".join([r["family"], r["size"], "adapted" if r["adapted"] else "prompted",
                       r["corpus"], r["regime"], f"k={r['k']}", r["decoder"]])
        if o_l is None:
            n_noref_leading += 1
        if o_l is not None and o_a is not None and o_l != o_a:
            diverged.append(r["id"])
        if o_l is None and o_a is not None:
            score = field_micro_f1(o_a, gold[r["doc_id"]]) if r["corpus"] == "cord" else None
            newly.append({"id": r["id"], "kind": kind, "slice": sk,
                          "recovered_score": round(score, 4) if score is not None else None})
            per_kind[kind] += 1; per_slice[sk] += 1
        # fence rate under the wider scope for the S/T schema cells
        if r["artifact"].startswith("cord_fence_tax"):
            fenced_any[r["artifact"]]["n"] += 1
            fenced_any[r["artifact"]]["leading"] += int(fenced_l)
            fenced_any[r["artifact"]]["any_kind_not_none"] += int(kind != "none")
            fenced_any[r["artifact"]]["any_fences_only"] += int(kind in ("leading_fence", "fence_after_prose"))
            fenced_any[r["artifact"]][f"kind:{kind}"] += 1

    # The protocol's letter says "kind != none counts as fenced"; a bare
    # object in prose is not a fence, so W3 is read BY THE LETTER (the
    # non-flattering choice: it fires where fences-only would not) and the
    # fences-only rate is banked beside it (protocol erratum 2).
    moves = {a: {"n": v["n"], "fenced_leading": v["leading"],
                 "fenced_any_fences_only": v["any_fences_only"],
                 "credited_any_kind_not_none": v["any_kind_not_none"],
                 "leading": round(v["leading"] / v["n"], 4),
                 "any": round(v["any_fences_only"] / v["n"], 4),
                 "any_letter_kind_not_none": round(v["any_kind_not_none"] / v["n"], 4),
                 "by_kind": {k.split(":", 1)[1]: c for k, c in v.items() if k.startswith("kind:")}}
             for a, v in sorted(fenced_any.items())}
    # T exception re-check with the any-scope rates (published beside, never re-taken)
    t_recheck = {}
    t = json.loads(T_ARTIFACT.read_text())
    for row in t["rows"]:
        if row["regime"] != "schema":
            continue
        fam = [k for k, v in cft.FAMILIES.items() if v[0] == row["model"]][0]
        art = f"cord_fence_tax_families_cells/{fam}_schema.json"
        f_any = moves[art]["any_letter_kind_not_none"]  # the letter's rate, the wider one
        kshot_leading = [x for x in t["rows"] if x["model"] == row["model"] and x["regime"] == "kshot"][0]["fence_rate"]
        kshot_any = moves[f"cord_fence_tax_families_cells/{fam}_kshot.json"]["any_letter_kind_not_none"]
        t_recheck[fam] = {"schema_leading_rate": row["fence_rate"], "schema_any_rate": f_any,
                          "kshot_leading_rate": kshot_leading, "kshot_any_rate": kshot_any,
                          "frozen_reading": t["reading_per_model"][fam],
                          "reading_with_any_rates": cft.family_reading(f_any, kshot_any)}

    w1 = ("W1 ZERO DIVERGENCES, as the extraction order guarantees (a leading fence returns before "
          "the scope is consulted; a whole-object text has nothing outside it) -- a regression check, "
          "not an empirical finding."
          if not diverged else f"W1 DIVERGENCE IN {len(diverged)} records -- a bug in the extension; fixed before reading: {diverged[:10]}")
    w2 = read_w2(len(newly), n_noref_leading)
    w3 = read_w3(moves)
    scores = [x["recovered_score"] for x in newly if x["recovered_score"] is not None]
    record = {
        "what": "Addendum W: the shipped instrument's two scopes on every banked raw output "
                "(fence_corpus_2026-09-05). leading = one leading fence (every frozen reading); "
                "any = also a fence after prose or a bare object inside prose (opt-in). Readings "
                "applied by arithmetic from docs/research/ADDENDUM_W_PROTOCOL.md.",
        "preregistration": "docs/research/ADDENDUM_W_PROTOCOL.md (frozen 2026-09-08, commit 2fa6cd6)",
        "prediction_written_before_the_run": "newly credited between 5 and 15 of 2,130, mostly bare_object_in_prose; zero divergences; no S/T fence rate moves by >= 0.05",
        "corpus": {"path": CORPUS.name, "sha256": sha, "n_records": len(records)},
        "n_noref_leading": n_noref_leading,
        "newly_credited": {"n": len(newly), "by_kind": dict(per_kind), "by_slice": dict(sorted(per_slice.items())),
                           "records": newly,
                           "recovered_cord_score_mean": round(sum(scores) / len(scores), 4) if scores else None,
                           "recovered_cord_records": len(scores)},
        "diverged": diverged,
        "schema_cell_fence_rates_both_scopes": moves,
        "t_exception_recheck_with_any_rates": t_recheck,
        "readings": {"W1": w1, "W2": w2, "W3": w3},
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"corpus {len(records)}; leading rejects {n_noref_leading}; newly credited {len(newly)} {dict(per_kind)}; "
          f"recovered CORD score mean {record['newly_credited']['recovered_cord_score_mean']} over {len(scores)}")
    for a, v in moves.items():
        print(f"  {a:55s} leading {v['leading']:.4f} any {v['any']:.4f}")
    print(w1); print(w2); print(w3); print(f"banked: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
