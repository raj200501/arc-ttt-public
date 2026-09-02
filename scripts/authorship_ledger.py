#!/usr/bin/env python3
"""Bank WHO authored the commits of the tree this runs in — with the
tree's own identity, so the number can never be quoted about the wrong
repository again.

    python3 scripts/authorship_ledger.py
    python3 scripts/authorship_ledger.py --out experiments/authorship_ledger_2026-09-01.json

Why this exists (2026-09-01). The outbound copy said "run git shortlog
on the public repository: four in five commits by Claude". The public
repository is a fresh-history EXPORT whose export commits were, until
that day, authored under the founder's name — so the command returned
the inverse. Then the fix pointed readers at the export script "shipped
in this tree", and the export script is deliberately NOT shipped (it
carries the leak-gate vocabulary). Two wrong referents in a row, on the
one disclosure a reviewer is most likely to check.

So the ledger is now a banked artifact that names its own referent:
the HEAD SHA, the remote, the commit count, and the per-author split of
the tree it was run in. Run in the source tree it records the source
tree; run in the public export it records the export — and says so.
A reader of the public tree can verify the public tree's numbers and
can see that the source-tree ledger was banked at a named SHA; they
cannot re-derive the source-tree numbers without the source tree, and
this script does not pretend otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

REPO = pathlib.Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=True).stdout


def ledger() -> dict:
    head = _git("rev-parse", "HEAD").strip()
    try:
        remote = _git("remote", "get-url", "origin").strip()
    except subprocess.CalledProcessError:
        remote = None
    total = int(_git("rev-list", "--count", "HEAD").strip())
    by_author: dict[str, int] = {}
    for line in _git("shortlog", "-sne", "HEAD").splitlines():
        line = line.strip()
        if not line:
            continue
        count, ident = line.split("\t", 1)
        # Fold the founder's several e-mail identities into one line and
        # the agent org's into one, so the ratio is about people.
        key = ("Claude <noreply@anthropic.com>" if "noreply@anthropic.com" in ident
               and ident.startswith("Claude")
               else "Raj Kashikar" if "Raj Kashikar" in ident
               else ident)
        by_author[key] = by_author.get(key, 0) + int(count)
    agent = by_author.get("Claude <noreply@anthropic.com>", 0)
    human = by_author.get("Raj Kashikar", 0)
    is_export = (REPO / "scripts" / "export_public.sh").exists() is False
    return {
        "what": "Per-author commit counts of THIS tree, with the tree's "
                "identity, so the figure cannot be restated about a "
                "different repository.",
        "referent": ("public export (fresh-history; export commits are "
                     "squashes and their author is the export event, not "
                     "per-file authorship)" if is_export
                     else "source tree (full history)"),
        "head": head,
        "remote": remote,
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "total_commits": total,
        "by_author": by_author,
        "agent_commits": agent,
        "human_commits": human,
        "agent_share": round(agent / total, 4) if total else None,
        "how_to_reproduce": "git shortlog -sne HEAD in the tree named by "
                            "`head` and `remote`; any other tree is a "
                            "different referent.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="bank the ledger as JSON (default: print only)")
    args = parser.parse_args()
    record = ledger()
    print(f"{record['referent']}\nHEAD {record['head'][:12]}  "
          f"{record['total_commits']} commits  agent {record['agent_commits']}  "
          f"human {record['human_commits']}  agent share "
          f"{record['agent_share']}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                          encoding="utf-8")
        print(f"banked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
