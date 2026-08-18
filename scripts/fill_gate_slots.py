#!/usr/bin/env python3
"""Fill [[GATE-FILL]] slots across raise docs from the k=30 gate summary.

Verdict-day mechanization (VERDICT_DAY_RUNBOOK.md): once the summary shows
3/3 pairs, one run propagates the numbers everywhere. Refuses on partial
pairs unless --force (which exists for dry-run rehearsal only).

Usage: fill_gate_slots.py [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "experiments" / "novel_schema_summary_2026-08-12.json"
DOCS = [
    ROOT / "docs" / "strategy" / "YC_APP_DRAFT.md",
    ROOT / "docs" / "strategy" / "OUTREACH_KIT.md",
    ROOT / "docs" / "strategy" / "PILOT_ONE_PAGER.md",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="fill from a partial summary (rehearsal only)")
    args = ap.parse_args()

    summary = json.loads(SUMMARY.read_text())
    gate = summary["gate_k30"]
    if gate["pairs_complete"] != 3 and not args.force:
        print(f"REFUSED: only {gate['pairs_complete']}/3 pairs complete "
              "(--force is for rehearsal only)")
        return 2

    receipts = gate["receipt_level"]
    lo, hi = receipts["ci95"]
    delta_pp = f"{gate['mean_delta'] * 100:+.1f}"
    mean_ci = (f"{delta_pp} micro-F1 points "
               f"(receipt-level 95% CI [{lo * 100:.1f}, {hi * 100:.1f}], "
               f"n={receipts['n']}, "
               f"sign test {receipts['sign_test']['wins']}W/"
               f"{receipts['sign_test']['losses']}L)")
    verdict_line = (f"Preregistered k=30 verdict ({summary['date']} spec, "
                    f"decided {summary.get('decided','2026-08-17')}): "
                    f"**{summary['VERDICT']}** — {summary['verdict_detail']}")
    artifact_link = ("experiments/novel_schema_summary_2026-08-12.json "
                     "(plus per-arm artifacts, same directory)")

    replacements = {
        "[[GATE-FILL delta]]": delta_pp,
        "[[GATE-FILL mean, CI]]": mean_ci,
        "[[GATE-FILL: verdict line when it lands]]": verdict_line,
        "[[GATE-FILL artifact link]]": artifact_link,
    }

    total = 0
    for doc in DOCS:
        text = doc.read_text()
        hits = {k: text.count(k) for k in replacements if k in text}
        for k, v in replacements.items():
            text = text.replace(k, v)
        n = sum(hits.values())
        total += n
        if args.dry_run:
            print(f"{doc.name}: would fill {n} slot(s): {hits}")
        elif n:
            doc.write_text(text)
            print(f"{doc.name}: filled {n} slot(s)")
    print(f"{'DRY-RUN — ' if args.dry_run else ''}{total} slots total")
    if not args.dry_run and total:
        print("Now run the runbook verify: grep -rn 'GATE-FILL' docs/ "
              "— only meta/header mentions may remain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
