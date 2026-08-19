"""Verdict-day banking: pull-dir -> experiments/, superseding safely.

One command instead of five judgment calls at 4am:

    python3 scripts/bank_novel_schema.py --pull-dir <kernel output> \
        --experiments experiments --date 2026-08-12

Rules, in order, per incoming artifact:

1. Non-arm files and arms without a "device" stamp are ignored (same
   provenance rule as every other gate in this project).
2. New file, no existing: banked.
3. Existing identical (same device+dtype, same mean): skipped silently.
4. Existing arm is an ERROR record and incoming is scoreable: superseded —
   the error record moves to experiments/superseded_novel/ (nothing is
   deleted) and the scoreable arm takes the canonical name.
5. Existing arm is scoreable but its (device, dtype) differs from the
   incoming arm AND from its own pair partner's: the stale side of a
   dtype migration (B.7-r3). Superseded, same move-aside rule.
6. Existing scoreable arm with the SAME environment but different numbers:
   refused loudly — that is a reproducibility problem, not a banking
   decision, and a human reads it before anything moves.

Then it runs novel_schema_summary.py and prints the verdict. The
superseded/ directory is outside the summary's non-recursive glob, so
preserved copies can never re-enter the computation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def classify(incoming: dict, existing: dict, partner: dict | None = None) -> str:
    if "error" in existing and "error" not in incoming:
        return "supersede-error"
    if "error" in incoming:
        return "skip"  # never replace anything with an error record
    same_env = existing.get("device") == incoming.get("device") and existing.get(
        "dtype"
    ) == incoming.get("dtype")
    if same_env:
        if existing.get("mean_micro_f1") == incoming.get("mean_micro_f1"):
            return "skip"
        return "conflict"
    # Rule 5's second condition: an existing arm that still matches its
    # banked pair partner's environment is NOT the stale side of a dtype
    # migration — refuse loudly for a human instead of moving it aside.
    if (
        partner is not None
        and "error" not in partner
        and existing.get("device") == partner.get("device")
        and existing.get("dtype") == partner.get("dtype")
    ):
        return "conflict"
    return "supersede-env"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-dir", required=True)
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    pull = Path(args.pull_dir)
    exp = Path(args.experiments)
    superseded = exp / "superseded_novel"

    conflicts: list[str] = []
    actions: list[str] = []
    for path in sorted(pull.glob(f"novel_schema_*_{args.date}.json")):
        record = json.loads(path.read_text())
        if not {"rung", "k", "seed", "arm"}.issubset(record):
            continue
        if "device" not in record:
            continue
        dest = exp / path.name
        if not dest.exists():
            shutil.copy(path, dest)
            actions.append(f"banked NEW      {path.name}")
            continue
        existing = json.loads(dest.read_text())
        partner = None
        for candidate in exp.glob(f"novel_schema_*_{args.date}.json"):
            if candidate == dest:
                continue
            other = json.loads(candidate.read_text())
            if (
                other.get("rung") == record.get("rung")
                and other.get("k") == record.get("k")
                and other.get("seed") == record.get("seed")
                and other.get("arm") != record.get("arm")
            ):
                partner = other
                break
        outcome = classify(record, existing, partner)
        if outcome == "skip":
            continue
        if outcome == "conflict":
            conflicts.append(
                f"{path.name}: same env, different numbers "
                f"({existing.get('mean_micro_f1')} vs {record.get('mean_micro_f1')})"
            )
            continue
        superseded.mkdir(exist_ok=True)
        # Never overwrite a previously preserved copy ("nothing is deleted"):
        # pick a collision-free destination for repeat supersedes.
        aside = superseded / path.name
        n = 1
        while aside.exists():
            aside = superseded / f"{path.stem}.superseded{n}{path.suffix}"
            n += 1
        shutil.move(str(dest), aside)
        shutil.copy(path, dest)
        actions.append(f"banked {outcome:14} {path.name}")

    for line in actions:
        print(line)
    if conflicts:
        print("\nREFUSED — same-environment number conflicts need a human:")
        for line in conflicts:
            print("  " + line)
        return 2

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "novel_schema_summary.py"),
            "--dir",
            str(exp),
            "--date",
            args.date,
        ]
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
