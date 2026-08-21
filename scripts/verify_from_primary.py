#!/usr/bin/env python3
"""Verify prediction-carrying artifacts from PRIMARY evidence.

Where verify_verdict.py audits the arithmetic (re-aggregating stored
per-receipt scores), this script audits the EVIDENCE for any artifact
that stores raw predictions (Addendum E onward): it regenerates the
gold labels from the deterministic corpus generator (seed -> schema ->
documents, no stored data trusted), re-scores every stored prediction
against the regenerated gold with the real scorer, and compares the
recomputed per-receipt F1 to the stored one. The trust boundary moves
from "the recorded scores" to "the generator seed + the scorer code" —
both of which are in this repository and pinned by the OpenTimestamps
proof on the spec.

Requires the package (run from repo root):

    PYTHONPATH=src python3 scripts/verify_from_primary.py experiments/novel_schema_e_*.json

Artifacts without stored predictions (Addendum B/D, which predate
primary-evidence storage) are reported as such, not silently passed.
"""

import json
import pathlib
import re
import sys


def main(paths: list[str]) -> int:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    from arcttt.novel_schema import make_task
    from arcttt.scoring import score_text_output

    failures = 0
    for raw in paths:
        path = pathlib.Path(raw)
        record = json.loads(path.read_text())
        rows = [r for r in record["results"] if "micro_f1" in r]
        with_pred = [r for r in rows if r.get("prediction")]
        if not with_pred:
            print(f"{path.name}: NO stored predictions (pre-E artifact) — "
                  "primary verification not possible; use verify_verdict.py "
                  "for the arithmetic audit and kaggle/ to re-run the arm.")
            continue
        seed = record["seed"] if "seed" in record else int(
            re.search(r"seed(\d+)_", path.name).group(1))
        # Resolve geometry by matching the artifact's stored schema text
        # against each regenerated schema — never by filename guessing
        # (Addendum E ran "diverse", E-r2 "diverse-compact"; a filename
        # guess scored 60 false mismatches on the first E-r2 artifact).
        # A failed match across all geometries is itself an integrity
        # failure: the artifact's schema is not reproducible from its
        # seed. Artifacts without a stored schema fall back to "fixed".
        stored_schema = record.get("schema")
        task = None
        for geometry in ("fixed", "diverse", "diverse-compact"):
            cand, schema = make_task(seed=seed, n_train=record["k"],
                                     n_test=record["eval_n"],
                                     task_id="verify", geometry=geometry)
            if stored_schema is None or schema.describe() == stored_schema:
                task = cand
                break
        if task is None:
            print(f"{path.name}: stored schema does not match the "
                  "regenerated schema under ANY geometry -> SCHEMA "
                  "MISMATCH (integrity failure)")
            failures += 1
            continue
        mismatches = 0
        for r in with_pred:
            gold = task.test[r["index"]].output_text
            score = score_text_output(r["prediction"], gold)
            if abs(score.micro_f1 - r["micro_f1"]) > 5e-4:
                mismatches += 1
        status = "PRIMARY-VERIFIED" if mismatches == 0 else f"{mismatches} MISMATCHES"
        print(f"{path.name}: {len(with_pred)} predictions re-scored against "
              f"regenerated gold -> {status}")
        failures += mismatches
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
