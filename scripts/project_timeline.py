#!/usr/bin/env python3
"""How much was built, over how long, derived rather than remembered.

Outbound copy has started citing the shape of the work -- how many days,
how many experiments, how many published negatives -- and every one of
those was a figure someone counted by hand once and then repeated. That
is precisely the pattern that put a stale `83/83 tests` next to a public
repository link for six drift cycles.

So the timeline is derived from the tree on every run: first and last
commit from git, banked artifacts from `experiments/`, tests from the
collected suite, corrections from the ledger. Nothing here is typed in.

**A caveat travels with it, in the artifact, because these numbers are
the most inflatable ones this project has.** A commit count measures
typing, not progress. An artifact count measures runs, not insight. They
are reported because copy quotes them and quoted figures must reconcile
-- not because they are evidence of quality, which is what `VERDICT.md`
is for.

    PYTHONPATH=src python3 scripts/project_timeline.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, timeout=120).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "project_timeline.json"))
    args = parser.parse_args()

    first = _git("log", "--reverse", "--format=%ad", "--date=short"
                 ).split("\n")[0]
    last = _git("log", "-1", "--format=%ad", "--date=short")
    commits = int(_git("rev-list", "--count", "HEAD") or 0)
    if not first or not last or not commits:
        raise SystemExit(
            "git produced no history. A timeline that silently reports zero "
            "days of work would be quoted as though it were measured.")
    start = dt.date.fromisoformat(first)
    end = dt.date.fromisoformat(last)
    days = (end - start).days + 1

    artifacts = len(list((REPO / "experiments").glob("*.json")))

    corrections = None
    ledger = REPO / "experiments" / "self_correction_ledger.json"
    if ledger.exists():
        corrections = json.loads(ledger.read_text(encoding="utf-8"))[
            "total_dated_corrections"]

    record = {
        "what": "The shape of the work, derived from the tree rather than "
                "remembered. Every figure here is recomputed on each run.",
        "first_commit": first,
        "latest_commit": last,
        "elapsed_days_inclusive": days,
        "commits": commits,
        "commits_per_day": round(commits / days, 1),
        "banked_artifacts": artifacts,
        "dated_corrections": corrections,
        "solo": True,
        "what_these_numbers_are_not": (
            "A commit count measures typing. An artifact count measures "
            "runs. Neither measures whether any of it was right, and both "
            "are trivially inflatable by anyone who wants to inflate them "
            "-- which is the reason to state that here rather than to hope "
            "a reader supplies the caveat. They are derived and banked "
            "because outbound copy quotes them and a quoted figure has to "
            "reconcile to something recomputable. The question of whether "
            "the work was any good is answered by VERDICT.md, where six of "
            "the results go against this project, and by "
            "verification_coverage.json, which reports what fraction of it "
            "a stranger can actually re-derive."),
        "how_to_recompute": "PYTHONPATH=src python3 scripts/project_timeline.py",
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"{first} -> {last}  ({days} days)")
    print(f"{commits} commits  ({record['commits_per_day']}/day), "
          f"{artifacts} banked artifacts, {corrections} dated corrections")
    print(f"\nbanked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
