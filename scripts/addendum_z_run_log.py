#!/usr/bin/env python3
"""Addendum Z's run, logged so that a silent redraw is detectable.

Sampling and the LoRA initialisation are unseeded (the protocol says so),
the arms are skip-if-exists, and the checkpoint directory `work/z/` is not
committed. So nothing in the run itself stops an operator from deleting a
half-finished arm and drawing it again until it looks better. A simulated
diligence reviewer said exactly that (review round 9, 2026-09-23). This
closes it without publishing a single score early:

- every snapshot records, per arm, the adapter's sha256 and a **prefix
  hash** over the documents decoded so far — sha256 of the canonical JSON
  of each journal row, in document order — plus the row count, the
  restart count and the start clock. No score, no text.
- the snapshots are banked in `experiments/addendum_z_run_log.json`,
  committed and OpenTimestamps-anchored at every push.
- when the arms land, `--verify` recomputes every banked prefix hash from
  the committed arm artifacts (their `predictions` list is the journal,
  sorted by document index, which is the order the runner decodes in), and
  checks every banked adapter digest against the artifact's. A redraw of
  any document or adapter that was logged before it was redrawn fails.

It also pins the code: the sha256 of every file that decides what the run
computes, at the protocol's freeze commit and as it stands now.

    PYTHONPATH=src python3 scripts/addendum_z_run_log.py            # add a snapshot
    PYTHONPATH=src python3 scripts/addendum_z_run_log.py --verify   # after landing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import novel_schema_rerun as z  # noqa: E402

OUT = REPO / "experiments" / "addendum_z_run_log.json"
FREEZE_COMMIT = "fa94857"
PINNED = (
    "scripts/novel_schema_rerun.py",
    "scripts/novel_schema_summary.py",
    "src/arcttt/text_ttt.py",
    "src/arcttt/model.py",
    "src/arcttt/lora.py",
    "src/arcttt/novel_schema.py",
    "src/arcttt/scoring.py",
    "src/arcttt/text_task.py",
    "src/arcttt/serialize.py",
    "scripts/run_addendum_z.sh",
)


def row_digest(rows: list[dict]) -> str:
    """sha256 over the canonical JSON of each row, one per line, in order."""
    blob = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_pins() -> dict:
    pins = {}
    for rel in PINNED:
        now = (REPO / rel).read_bytes() if (REPO / rel).exists() else None
        try:
            then = subprocess.run(["git", "show", f"{FREEZE_COMMIT}:{rel}"], cwd=REPO,
                                  capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            then = None
        pins[rel] = {
            "at_freeze_commit": _sha(then) if then is not None else None,
            "now": _sha(now) if now is not None else None,
            "unchanged_since_freeze": then is not None and now is not None and then == now,
        }
    return pins


def _read_journal_readonly(path: pathlib.Path) -> list[dict]:
    """Parse a live journal WITHOUT touching it. The runner's own loader
    repairs a torn last line by rewriting the file, which is right at the
    runner's start and would corrupt a journal the runner is appending to
    at this moment; this reader skips an unparseable last line instead."""
    rows, seen = [], set()
    if not path.exists():
        return rows
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    for position, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if position == len(lines) - 1:
                break
            raise
        if row["index"] not in seen:
            seen.add(row["index"])
            rows.append(row)
    return rows


def arm_state(seed: int, arm: str, work: pathlib.Path) -> dict:
    stem = z.ckpt_stem(seed, arm)
    meta_path = work / f"{stem}_meta.json"
    docs_path = work / f"{stem}_docs.jsonl"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows = sorted(_read_journal_readonly(docs_path), key=lambda r: r["index"])
    # only the leading run of consecutive indices is a stable prefix
    prefix = []
    for want, row in enumerate(rows):
        if row["index"] != want:
            break
        prefix.append(row)
    return {
        "arm": z.artifact_name(seed, arm),
        "started_utc": meta.get("started_utc"),
        "restarts": meta.get("restarts"),
        "adapter_sha256": meta.get("adapter_sha256"),
        "documents_decoded": len(rows),
        "prefix_documents": len(prefix),
        "prefix_sha256": row_digest(prefix) if prefix else None,
        "banked": (REPO / "experiments" / z.artifact_name(seed, arm)).exists(),
    }


def snapshot(work: pathlib.Path) -> dict:
    return {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arms": [arm_state(s, a, work) for s, a in z.ARM_ORDER],
    }


def load_log() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {
        "what": ("Addendum Z's run, logged so that a silent redraw is detectable: per "
                 "arm, the adapter digest and a prefix hash over the documents decoded "
                 "so far. No score and no text is recorded here; the protocol forbids "
                 "quoting arm-level numbers before the reading."),
        "protocol": z.PROTOCOL,
        "how_the_prefix_hash_is_made": (
            "sha256 of '\\n'.join(json.dumps(row, sort_keys=True)) over the journal rows "
            "with indices 0..n-1, in index order. The landed arm's `predictions` list is "
            "those rows sorted by index, so the hash is recomputable from the committed "
            "artifact."),
        "why": ("sampling and the LoRA-A initialisation are unseeded, arms are "
                "skip-if-exists, and work/z is not committed: without this log a "
                "half-finished arm could be deleted and redrawn with nothing to show "
                "for it (simulated review round 9, 2026-09-23)."),
        "verify": "PYTHONPATH=src python3 scripts/addendum_z_run_log.py --verify",
        "code_pins": code_pins(),
        "snapshots": [],
    }


def record(work: pathlib.Path) -> dict:
    log = load_log()
    log["code_pins"] = code_pins()
    log["snapshots"].append(snapshot(work))
    OUT.write_text(json.dumps(log, indent=1) + "\n")
    return log


def verify(experiments: pathlib.Path = REPO / "experiments") -> list[str]:
    """Every banked prefix hash and adapter digest against the landed arms."""
    log = json.loads(OUT.read_text())
    problems = []
    for snap in log["snapshots"]:
        for state in snap["arms"]:
            path = experiments / state["arm"]
            if not path.exists():
                problems.append(f"{snap['utc']} {state['arm']}: arm not landed")
                continue
            landed = json.loads(path.read_text())
            if state["adapter_sha256"] and landed["adapter"]["sha256"] != state["adapter_sha256"]:
                problems.append(f"{snap['utc']} {state['arm']}: adapter redrawn "
                                f"({state['adapter_sha256'][:12]} logged, "
                                f"{landed['adapter']['sha256'][:12]} landed)")
            n = state["prefix_documents"]
            if n:
                rows = sorted(landed["predictions"], key=lambda r: r["index"])[:n]
                if row_digest(rows) != state["prefix_sha256"]:
                    problems.append(f"{snap['utc']} {state['arm']}: the first {n} documents "
                                    "differ from what was logged")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--work", default=str(z.DEFAULT_WORK))
    args = ap.parse_args(argv)
    if args.verify:
        problems = verify()
        print("\n".join(problems) if problems else "every logged prefix and adapter matches the landed arms")
        return 1 if problems else 0
    log = record(pathlib.Path(args.work))
    last = log["snapshots"][-1]
    for state in last["arms"]:
        print(f"{state['arm']}: restarts {state['restarts']}, decoded {state['documents_decoded']}, "
              f"adapter {'yes' if state['adapter_sha256'] else 'no'}, banked {state['banked']}")
    changed = [k for k, v in log["code_pins"].items() if not v["unchanged_since_freeze"]]
    print("code changed since the freeze commit:", changed or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
